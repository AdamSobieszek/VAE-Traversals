"""Prepared, compile-friendly StyleGAN op modules for FP16 inference.

These bake filters/gains into CUDA buffers so Inductor/CUDA Graphs do not see
CPU scalar tensors from Python floats or float32 filter staging.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .upfirdn2d_native import (
    _filter_2d,
    _pad_or_crop,
    _parse_padding,
    _parse_scaling,
)


IntOrPair = Union[int, Sequence[int]]
Padding = Union[int, Sequence[int]]


class PreparedFIR(nn.Module):
    """Static FIR resampling with precomputed depthwise weights."""

    def __init__(
        self,
        f: Optional[torch.Tensor],
        *,
        channels: int,
        up: IntOrPair = 1,
        down: IntOrPair = 1,
        padding: Padding = 0,
        flip_filter: bool = False,
        gain: float = 1.0,
        dtype: torch.dtype = torch.float16,
        device: torch.device | str = "cuda",
        variant: str = "auto",
        channels_last: bool = False,
    ) -> None:
        super().__init__()
        device = torch.device(device)
        upx, upy = _parse_scaling(up)
        downx, downy = _parse_scaling(down)
        padx0, padx1, pady0, pady1 = _parse_padding(padding)
        self.upx, self.upy = upx, upy
        self.downx, self.downy = downx, downy
        self.padx0, self.padx1, self.pady0, self.pady1 = padx0, padx1, pady0, pady1
        self.channels = int(channels)
        self.channels_last = bool(channels_last)

        if variant == "auto":
            # Prefer polyphase for up=2 with the common 4-tap StyleGAN filter.
            kernel_probe, _ = _filter_2d(
                f, device=device, dtype=torch.float32, gain=gain, force_dense=True
            )
            if (
                upx == upy == 2
                and kernel_probe.ndim == 2
                and kernel_probe.shape[0] in (4, 6)
                and kernel_probe.shape[0] == kernel_probe.shape[1]
            ):
                variant = "polyphase"
            elif upx == upy == 1:
                variant = "strided"
            else:
                variant = "transpose"
        self.variant = variant

        kernel, _ = _filter_2d(
            f, device=device, dtype=dtype, gain=gain, force_dense=True
        )
        if kernel.ndim == 1:
            kernel = kernel[:, None] * kernel[None, :]

        if variant == "polyphase":
            # Build polyphase weights once (same as _polyphase_up2).
            conv_kernel = kernel.flip((0, 1)) if flip_filter else kernel
            phase_order = ((1, 1), (1, 0), (0, 1), (0, 0))
            phases = [conv_kernel[py::2, px::2].contiguous() for py, px in phase_order]
            weight = torch.stack(phases, dim=0)[:, None].repeat(channels, 1, 1, 1)
            self.register_buffer("weight", weight.contiguous())
            pos_x0, pos_x1 = max(padx0, 0), max(padx1, 0)
            pos_y0, pos_y1 = max(pady0, 0), max(pady1, 0)
            self.pad_left = pos_x0 // 2
            self.pad_right = (pos_x1 + 1) // 2
            self.pad_top = pos_y0 // 2
            self.pad_bottom = (pos_y1 + 1) // 2
            self.start_x = 1 - (pos_x0 % 2)
            self.start_y = 1 - (pos_y0 % 2)
            self.pos_x0, self.pos_x1, self.pos_y0, self.pos_y1 = (
                pos_x0,
                pos_x1,
                pos_y0,
                pos_y1,
            )
            self.neg_x0, self.neg_x1 = max(-padx0, 0), max(-padx1, 0)
            self.neg_y0, self.neg_y1 = max(-pady0, 0), max(-pady1, 0)
            self.filter_h = int(kernel.shape[0])
            self.filter_w = int(kernel.shape[1])
        elif variant == "strided":
            conv_kernel = kernel if flip_filter else kernel.flip((0, 1))
            weight = conv_kernel[None, None].expand(
                channels, 1, conv_kernel.shape[0], conv_kernel.shape[1]
            ).contiguous()
            self.register_buffer("weight", weight)
        elif variant == "transpose":
            transpose_kernel = kernel.flip((0, 1)) if flip_filter else kernel
            weight = transpose_kernel[None, None].expand(
                channels,
                1,
                transpose_kernel.shape[0],
                transpose_kernel.shape[1],
            ).contiguous()
            self.register_buffer("weight", weight)
            self.filter_h = int(kernel.shape[0])
            self.filter_w = int(kernel.shape[1])
        else:
            raise ValueError(f"Unknown FIR variant {variant!r}")

        if channels_last and self.weight.ndim == 4:
            # Depthwise grouped conv weights stay NCHW; activation layout is separate.
            pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)

        if self.variant == "strided":
            x = _pad_or_crop(x, self.padx0, self.padx1, self.pady0, self.pady1)
            return F.conv2d(
                x, self.weight, stride=(self.downy, self.downx), groups=self.channels
            )

        if self.variant == "polyphase":
            y = F.conv2d(
                F.pad(
                    x,
                    (self.pad_left, self.pad_right, self.pad_top, self.pad_bottom),
                ),
                self.weight,
                groups=self.channels,
            )
            y = F.pixel_shuffle(y, 2)
            output_h = x.shape[2] * 2 + self.pos_y0 + self.pos_y1 - self.filter_h + 1
            output_w = x.shape[3] * 2 + self.pos_x0 + self.pos_x1 - self.filter_w + 1
            crop_y0 = self.start_y + self.neg_y0
            crop_x0 = self.start_x + self.neg_x0
            final_h = output_h - self.neg_y0 - self.neg_y1
            final_w = output_w - self.neg_x0 - self.neg_x1
            y = y[
                :,
                :,
                crop_y0 : crop_y0 + final_h,
                crop_x0 : crop_x0 + final_w,
            ]
            if self.downx > 1 or self.downy > 1:
                y = y[:, :, :: self.downy, :: self.downx]
            return y

        # transpose
        input_h, input_w = x.shape[2], x.shape[3]
        y = F.conv_transpose2d(
            x, self.weight, stride=(self.upy, self.upx), groups=self.channels
        )
        output_h = input_h * self.upy + self.pady0 + self.pady1 - self.filter_h + 1
        output_w = input_w * self.upx + self.padx0 + self.padx1 - self.filter_w + 1
        start_y = self.filter_h - 1 - self.pady0
        start_x = self.filter_w - 1 - self.padx0
        pad_left, pad_top = max(-start_x, 0), max(-start_y, 0)
        start_x += pad_left
        start_y += pad_top
        pad_right = max(start_x + output_w - (y.shape[3] + pad_left), 0)
        pad_bottom = max(start_y + output_h - (y.shape[2] + pad_top), 0)
        if pad_left or pad_right or pad_top or pad_bottom:
            y = F.pad(y, (pad_left, pad_right, pad_top, pad_bottom))
        return y[
            :,
            :,
            start_y : start_y + output_h : self.downy,
            start_x : start_x + output_w : self.downx,
        ]


class PreparedBiasAct(nn.Module):
    """Frozen inference bias+activation with device-resident constants."""

    def __init__(
        self,
        bias: torch.Tensor,
        *,
        act: str = "lrelu",
        gain: float = 1.0,
        clamp: Optional[float] = 256.0,
        alpha: float = 0.2,
        dtype: torch.dtype = torch.float16,
        device: torch.device | str = "cuda",
        homogeneous: bool = True,
        disable_clamp: bool = False,
        channels_last: bool = False,
    ) -> None:
        super().__init__()
        if act not in ("linear", "relu", "lrelu"):
            raise ValueError(f"PreparedBiasAct supports linear/relu/lrelu, got {act}")
        device = torch.device(device)
        self.act = act
        self.homogeneous = bool(homogeneous)
        self.channels_last = bool(channels_last)
        gain = float(gain)
        bias = bias.detach().to(device=device, dtype=dtype)
        # Keep alpha/gain/clamp as 0-dim CUDA tensors so reduce-overhead
        # CUDA Graphs do not see CPU scalar arguments.
        self.register_buffer(
            "alpha", torch.tensor(float(alpha), device=device, dtype=dtype)
        )
        if homogeneous:
            scaled = bias * gain
            self.register_buffer("scaled_bias", scaled)
            self.register_buffer(
                "gain", torch.tensor(gain, device=device, dtype=dtype)
            )
        else:
            self.register_buffer("bias", bias)
            self.register_buffer(
                "gain", torch.tensor(gain, device=device, dtype=dtype)
            )
        if disable_clamp or clamp is None:
            self.register_buffer("clamp", torch.empty(0, device=device, dtype=dtype))
            self.use_clamp = False
        else:
            self.register_buffer(
                "clamp", torch.tensor(float(clamp), device=device, dtype=dtype)
            )
            self.use_clamp = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.channels_last and x.ndim == 4:
            x = x.contiguous(memory_format=torch.channels_last)
        if self.homogeneous:
            bias_view = self.scaled_bias.reshape(
                [self.scaled_bias.shape[0] if i == 1 else 1 for i in range(x.ndim)]
            )
            y = torch.addcmul(bias_view, x, self.gain)
        else:
            bias_view = self.bias.reshape(
                [self.bias.shape[0] if i == 1 else 1 for i in range(x.ndim)]
            )
            y = x + bias_view
            if self.act == "linear":
                pass
            elif self.act == "relu":
                y = F.relu(y)
            else:
                # negative_slope must be a Python float for F.leaky_relu
                y = F.leaky_relu(y, negative_slope=float(self.alpha.item()))
            y = y * self.gain
            if self.use_clamp:
                y = torch.minimum(torch.maximum(y, -self.clamp), self.clamp)
            return y

        if self.act == "relu":
            y = F.relu(y)
        elif self.act == "lrelu":
            # Where available, use prelu-style path with a device tensor slope.
            y = torch.where(y >= 0, y, y * self.alpha)
        if self.use_clamp:
            y = torch.minimum(torch.maximum(y, -self.clamp), self.clamp)
        return y


def prepare_stylegan_fir(
    f: Optional[torch.Tensor],
    *,
    channels: int,
    up: IntOrPair = 1,
    down: IntOrPair = 1,
    padding: Padding = 0,
    gain: float = 1.0,
    flip_filter: bool = False,
    dtype: torch.dtype = torch.float16,
    device: torch.device | str = "cuda",
    channels_last: bool = False,
) -> PreparedFIR:
    return PreparedFIR(
        f,
        channels=channels,
        up=up,
        down=down,
        padding=padding,
        flip_filter=flip_filter,
        gain=gain,
        dtype=dtype,
        device=device,
        variant="auto",
        channels_last=channels_last,
    )
