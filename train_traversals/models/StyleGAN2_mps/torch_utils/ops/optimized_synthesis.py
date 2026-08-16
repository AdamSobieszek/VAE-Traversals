"""Isolated, opt-in StyleGAN2 synthesis optimization experiments.

The benchmark harness deep-copies the generator before preparing and executing
``OptimizedSynthesis``.  The original ``synthesis.b{resolution}`` registration,
state-dict keys, defaults, and checkpoint behavior therefore remain unchanged.
"""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import bias_act, conv2d_resample, fma, upfirdn2d
from .fused_epilogue import fused_epilogue_composite
from .fused_up_modconv import compose_weight_with_fir_expanded
from .upfirdn2d_native import _filter_2d


_PREFIX = "_b64opt_"


def _suffix_for_dtype(dtype: torch.dtype) -> str:
    if dtype == torch.float16:
        return "fp16"
    if dtype == torch.bfloat16:
        return "bf16"
    return "fp32"


def _set_buffer(module: nn.Module, name: str, value: torch.Tensor) -> None:
    if hasattr(module, name):
        delattr(module, name)
    module.register_buffer(name, value.detach().contiguous(), persistent=False)


def _prepare_layer(layer: nn.Module, *, prepare_fir: bool) -> None:
    """Precompute frozen shared weights, energies, and optional FIR kernels."""
    weight32 = layer.weight.detach().float()
    _, in_channels, kh, kw = weight32.shape
    scale = (
        weight32.abs()
        .amax(dim=(1, 2, 3), keepdim=True)
        .clamp_min(1e-8)
        .reciprocal()
        * float(1.0 / (in_channels * kh * kw) ** 0.5)
    )
    weight16_norm32 = weight32 * scale
    energy32 = weight32.square().sum(dim=(2, 3))

    _set_buffer(layer, f"{_PREFIX}weight_fp32", weight32)
    _set_buffer(layer, f"{_PREFIX}weight_fp16", weight16_norm32.half())
    _set_buffer(layer, f"{_PREFIX}weight_bf16", weight32.bfloat16())
    _set_buffer(layer, f"{_PREFIX}energy_fp32", energy32)
    _set_buffer(layer, f"{_PREFIX}energy_fp16", weight16_norm32.square().sum(dim=(2, 3)))
    _set_buffer(layer, f"{_PREFIX}energy_bf16", energy32)

    if not prepare_fir or int(layer.up) != 2:
        return

    fir, _ = _filter_2d(
        layer.resample_filter,
        device=weight32.device,
        dtype=torch.float32,
        gain=4.0,
        force_dense=True,
    )
    if fir.ndim == 1:
        fir = fir[:, None] * fir[None, :]

    for suffix, weight in (
        ("fp32", weight32),
        ("fp16", weight16_norm32.half()),
        ("bf16", weight32.bfloat16()),
    ):
        fir_t = fir.to(dtype=weight.dtype)
        composed = compose_weight_with_fir_expanded(weight, fir_t)
        _set_buffer(
            layer,
            f"{_PREFIX}expanded_{suffix}",
            composed.transpose(0, 1),
        )
        phase_order = ((0, 0), (0, 1), (1, 0), (1, 1))
        phases = [
            composed[:, :, oy::2, ox::2].flip((2, 3))
            for oy, ox in phase_order
        ]
        _set_buffer(layer, f"{_PREFIX}polyphase_{suffix}", torch.cat(phases))


def prepare_optimized_synthesis(synthesis: nn.Module, cfg: Any) -> None:
    """Prepare an isolated candidate synthesis network in place."""
    wants_shared = bool(cfg.shared_modconv or cfg.fir_compose)
    if not wants_shared:
        return
    for resolution in synthesis.block_resolutions:
        block = getattr(synthesis, f"b{resolution}")
        if hasattr(block, "conv0"):
            _prepare_layer(block.conv0, prepare_fir=bool(cfg.fir_compose))
        _prepare_layer(block.conv1, prepare_fir=False)


def _noise(layer: nn.Module, x: torch.Tensor, noise_mode: str) -> torch.Tensor | None:
    if not layer.use_noise or noise_mode == "none":
        return None
    if noise_mode == "random":
        return (
            torch.randn(
                [x.shape[0], 1, layer.resolution, layer.resolution],
                device=x.device,
            )
            * layer.noise_strength
        )
    if noise_mode == "const":
        return layer.noise_const * layer.noise_strength
    raise ValueError(f"Unsupported noise mode {noise_mode!r}")


def _prepared_up2(
    layer: nn.Module,
    x: torch.Tensor,
    *,
    mode: str,
) -> torch.Tensor:
    suffix = _suffix_for_dtype(x.dtype)
    if mode == "expanded":
        weight_t = getattr(layer, f"{_PREFIX}expanded_{suffix}")
        y = F.conv_transpose2d(x, weight_t, stride=2)
        out_h, out_w = x.shape[2] * 2, x.shape[3] * 2
        # Exact common StyleGAN case: 3x3 learned kernel, 4x4 FIR, pad [1]*4.
        return y[:, :, 2 : 2 + out_h, 2 : 2 + out_w]

    weight = getattr(layer, f"{_PREFIX}polyphase_{suffix}")
    out_channels = int(layer.out_channels)
    y = F.conv2d(x, weight, padding=1)
    n, _, h, w = y.shape
    y = (
        y.reshape(n, 4, out_channels, h, w)
        .permute(0, 2, 1, 3, 4)
        .reshape(n, 4 * out_channels, h, w)
    )
    return F.pixel_shuffle(y, 2)


def _shared_layer(
    layer: nn.Module,
    x: torch.Tensor,
    w: torch.Tensor,
    *,
    cfg: Any,
    noise_mode: str,
    gain: float,
) -> torch.Tensor:
    styles = layer.affine(w)
    # FP16 needs inf-norm pre-scaling to avoid overflow. BF16 has FP32 range,
    # so the unnormalized weights are used directly.
    if x.dtype == torch.float16:
        styles = styles * (
            styles.detach().abs().amax(dim=1, keepdim=True).clamp_min(1e-8).reciprocal()
        )
    suffix = _suffix_for_dtype(x.dtype)
    weight = getattr(layer, f"{_PREFIX}weight_{suffix}")
    energy = getattr(layer, f"{_PREFIX}energy_{suffix}")

    x = x * styles.to(x.dtype).reshape(x.shape[0], -1, 1, 1)
    if bool(cfg.fir_compose) and int(layer.up) == 2:
        x = _prepared_up2(layer, x, mode=str(cfg.fir_compose_mode))
    else:
        x = conv2d_resample.conv2d_resample(
            x=x,
            w=weight,
            f=layer.resample_filter,
            up=int(layer.up),
            padding=int(layer.padding),
            flip_weight=(int(layer.up) == 1),
        )

    dcoef = (
        styles.float().square() @ energy.float().transpose(0, 1) + 1e-8
    ).rsqrt().to(x.dtype)
    dcoef = dcoef.reshape(x.shape[0], -1, 1, 1)
    noise = _noise(layer, x, noise_mode)
    act_gain = float(layer.act_gain) * float(gain)
    act_clamp = (
        float(layer.conv_clamp) * float(gain)
        if layer.conv_clamp is not None
        else None
    )

    if bool(cfg.fused_epilogue):
        return fused_epilogue_composite(
            x,
            layer.bias.to(x.dtype),
            noise=noise,
            dcoef=dcoef,
            act=layer.activation,
            gain=act_gain,
            clamp=act_clamp,
        )

    if noise is not None:
        x = fma.fma(x, dcoef, noise.to(x.dtype))
    else:
        x = x * dcoef
    return bias_act.bias_act(
        x,
        layer.bias.to(x.dtype),
        act=layer.activation,
        gain=act_gain,
        clamp=act_clamp,
    )


def _layer(
    layer: nn.Module,
    x: torch.Tensor,
    w: torch.Tensor,
    *,
    cfg: Any,
    noise_mode: str,
    gain: float = 1.0,
) -> torch.Tensor:
    if bool(cfg.shared_modconv or cfg.fir_compose):
        return _shared_layer(
            layer,
            x,
            w,
            cfg=cfg,
            noise_mode=noise_mode,
            gain=gain,
        )
    # Epilogue-only experiments are intentionally deferred until the shared
    # path is accepted; use the untouched layer otherwise.
    return layer(x, w, noise_mode=noise_mode, gain=gain)


class OptimizedSynthesis(nn.Module):
    """Experimental synthesis wrapper with no changes to production classes."""

    def __init__(self, synthesis: nn.Module, cfg: Any):
        super().__init__()
        self.synthesis = synthesis
        self.cfg = cfg
        self.resolutions = tuple(int(r) for r in synthesis.block_resolutions)
        prepare_optimized_synthesis(synthesis, cfg)

    def forward(
        self,
        ws: torch.Tensor,
        noise_mode: str = "const",
        force_fp32: bool = False,
    ) -> torch.Tensor:
        ws = ws.to(torch.float32)
        x = None
        img = None
        w_base = 0

        for resolution in self.resolutions:
            block = getattr(self.synthesis, f"b{resolution}")
            low_dtype = getattr(self.cfg, "low_precision_dtype", torch.float16)
            dtype = (
                low_dtype
                if (block.use_fp16 or bool(self.cfg.force_fp16_all_blocks))
                and not force_fp32
                else torch.float32
            )
            memory_format = (
                torch.channels_last
                if block.channels_last and not force_fp32
                else torch.contiguous_format
            )
            if block.in_channels == 0:
                x = block.const.to(dtype=dtype, memory_format=memory_format)
                x = x.unsqueeze(0).repeat([ws.shape[0], 1, 1, 1])
            else:
                x = x.to(dtype=dtype, memory_format=memory_format)

            w_idx = w_base
            if block.in_channels == 0:
                x = _layer(
                    block.conv1,
                    x,
                    ws[:, w_idx],
                    cfg=self.cfg,
                    noise_mode=noise_mode,
                )
                w_idx += 1
            elif block.architecture == "resnet":
                y = block.skip(x, gain=float(0.5**0.5))
                x = _layer(
                    block.conv0,
                    x,
                    ws[:, w_idx],
                    cfg=self.cfg,
                    noise_mode=noise_mode,
                )
                w_idx += 1
                x = _layer(
                    block.conv1,
                    x,
                    ws[:, w_idx],
                    cfg=self.cfg,
                    noise_mode=noise_mode,
                    gain=float(0.5**0.5),
                )
                w_idx += 1
                x = y.add_(x)
            else:
                x = _layer(
                    block.conv0,
                    x,
                    ws[:, w_idx],
                    cfg=self.cfg,
                    noise_mode=noise_mode,
                )
                w_idx += 1
                x = _layer(
                    block.conv1,
                    x,
                    ws[:, w_idx],
                    cfg=self.cfg,
                    noise_mode=noise_mode,
                )
                w_idx += 1

            if img is not None:
                img = upfirdn2d.upsample2d(img, block.resample_filter)
            if block.is_last or block.architecture == "skip":
                y = block.torgb(x, ws[:, w_idx], fused_modconv=True)
                y = y.to(
                    dtype=torch.float32,
                    memory_format=torch.contiguous_format,
                )
                img = img.add_(y) if img is not None else y

            # ToRGB overlaps the next block's first style in StyleGAN2.
            w_base += int(block.num_conv)

        return img


def build_optimized_synthesis(
    synthesis: nn.Module,
    cfg: Any,
    *,
    copy_module: bool = True,
) -> OptimizedSynthesis:
    """Build an opt-in wrapper without mutating the source module by default."""
    candidate = copy.deepcopy(synthesis) if copy_module else synthesis
    optimized = OptimizedSynthesis(candidate, cfg)
    optimized.eval()
    for parameter in optimized.parameters():
        parameter.requires_grad_(False)
    return optimized
