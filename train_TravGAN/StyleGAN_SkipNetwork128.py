"""Distill the StyleGAN2 pointwise_style32 generator at 128 or 256 resolution.

Run from train_traversals with Python from the active environment::

    python -m models.StyleGAN_SkipNetwork128 --device cuda --amp \
        --out-dir experiments/sg128_pointwise_style32

Add --resolution 256 to initialize a b256 model from the full teacher.
The default source is the trained sg128_pointwise_style32.pt. Full frames,
constant teacher/student noise, balanced normal/all/first8/odd style sampling,
and the winning loss recipe are retained. --resume restores a checkpoint from
this trainer; add --reset-steps for a fresh optimizer and training schedule.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import os.path as osp
from dataclasses import asdict, dataclass, fields
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.amp import GradScaler, autocast
from PIL import Image

from StyleGAN2_mps.early_output_model import (
    FORMAT_NAME, GENERATOR_CONFIGS, Generator, _load_full_source_generator,
    load_generator_checkpoint,
)

WEIGHTS_DIR = osp.join(osp.dirname(__file__), "pretrained", "generators", "StyleGAN2")
DEFAULT_GAN_WEIGHTS = osp.join(WEIGHTS_DIR, "stylegan2-ffhq-config-f.pt")
DEFAULT_EARLY_OUTPUT_WEIGHTS = osp.join(WEIGHTS_DIR, "sg128_pointwise_style32.pt")
TRAINING_FORMAT = "stylegan128_pointwise_style32_training_v1"
Z_DIM = W_DIM = 512
EVAL_GROUPS = ("normal", "all", "first8", "odd")


@dataclass
class TrainConfig:
    resolution: int = 128
    gan_weights: str = DEFAULT_GAN_WEIGHTS
    early_output_weights: Optional[str] = DEFAULT_EARLY_OUTPUT_WEIGHTS
    out_dir: str = "experiments/sg128_pointwise_style32"
    device: str = "auto"
    batch_size: int = 12
    max_steps: int = 8000
    lr: float = 1e-4
    synthesis_lr: float = 1e-4
    synthesis_reference_decay: float = 0.001
    weight_decay: float = 2e-4
    head_only_steps: int = 500
    finetune_min_resolution: int = 32
    early_synthesis_lr_scale: float = 0.25
    warmup_steps: int = 200
    grad_clip: float = 1.0
    truncation_psi: float = 1.0
    noise_mode: str = "const"
    w_augment_percent: float = 0.5
    l1_weight: float = 0.75
    mse_weight: float = 0.1
    edge_weight: float = 0.25
    lpips_weight: float = 1.0
    saliency: bool = True
    saliency_uniform_mix: float = 0.5
    val_count: int = 64
    eval_batch_size: int = 4
    val_every: int = 600
    plot_every: int = 600
    plot_count: int = 5
    log_every: int = 200
    ckpt_every: int = 800
    seed: int = 1
    amp: bool = False
    resume: Optional[str] = None
    reset_steps: bool = False

    def __post_init__(self):
        if self.resolution == 256:
            if self.early_output_weights == DEFAULT_EARLY_OUTPUT_WEIGHTS:
                self.early_output_weights = None  # A new head, initialized from the teacher.
            if self.out_dir == "experiments/sg128_pointwise_style32":
                self.out_dir = "experiments/sg256_pointwise_style32"


class StyleGANTeacher(nn.Module):
    def __init__(self, cfg, device):
        super().__init__()
        self.G = _load_full_source_generator(cfg.gan_weights).to(device)
        self.truncation_psi, self.noise_mode = cfg.truncation_psi, cfg.noise_mode
        self.resolution = cfg.resolution

    def map_z(self, z):
        return self.G.mapping(z, None, truncation_psi=self.truncation_psi)

    @torch.inference_mode()
    def target_from_ws(self, ws):
        image = self.G.synthesis(ws.float(), noise_mode=self.noise_mode,
                                fused_modconv=False if ws.device.type == "mps" else None)
        return F.interpolate(image.float(), size=(self.resolution, self.resolution), mode="area")


@torch.no_grad()
def initialize_from_teacher(teacher, resolution, device):
    """Copy the prefix and initialize RGB as the final block's average-W toRGB."""
    model = Generator(img_resolution=resolution).to(device)
    source = teacher.G.state_dict()
    model.load_state_dict({name: source[name] if not name.startswith("synthesis.decoder.") else value
                           for name, value in model.state_dict().items()}, strict=True)
    head = model.synthesis.decoder
    torgb = getattr(model.synthesis, f"b{resolution}").torgb
    style = torgb.affine(model.mapping.w_avg[None]).flatten() * torgb.weight_gain
    head.linear.weight.zero_()
    head.linear.weight[:, :, 1, 1].copy_(torgb.weight[:, :, 0, 0] * style[None])
    head.linear.bias.copy_(torgb.bias)
    for layer in (head.mix_out, head.spatial_out, head.style[-1]):
        layer.weight.zero_()
        layer.bias.zero_()
    return model


class AdamWWithReferenceDecay(torch.optim.AdamW):
    """AdamW-style decoupled weight decay toward pretrained parameters.

    Ordinary AdamW shrinks parameters toward zero.  For a pretrained synthesis
    prefix, the useful prior is its matching full-StyleGAN value instead.  A
    group with ``reference_weight_decay`` and zero ordinary ``weight_decay`` is
    therefore updated as::

        p <- p + lr * decay * (reference - p)  # decoupled source decay
        p <- AdamW_update(p, grad, lr)          # full, unattenuated task update

    This matches AdamW's LR-scaled decay dynamics, except that the fixed point
    is the pretrained tensor instead of zero.  Applying it before AdamW keeps it
    outside Adam's moment estimates and avoids attenuating the current task
    update.  References live in the already-loaded frozen teacher and are
    intentionally excluded from optimizer checkpoints.
    """

    def __init__(
        self,
        params,
        *,
        reference_parameters: Dict[int, torch.Tensor],
        **kwargs,
    ):
        self.reference_parameters = {
            parameter_id: reference.detach()
            for parameter_id, reference in reference_parameters.items()
        }
        super().__init__(params, **kwargs)

    def step(self, closure=None):
        # Evaluate an optional closure at the pre-decay parameters, matching the
        # point at which ordinary externally-computed gradients were obtained.
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Apply the source anchor before AdamW so it neither enters the moments
        # nor attenuates the current task update.  Synthesis groups have ordinary
        # (zero-centered) AdamW weight decay explicitly disabled below.
        with torch.no_grad():
            for group in self.param_groups:
                decay = float(group.get("reference_weight_decay", 0.0))
                if decay == 0.0:
                    continue
                amount = float(group["lr"]) * decay
                if not 0.0 <= amount <= 1.0:
                    raise ValueError(
                        "reference_weight_decay * lr must be in [0, 1], "
                        f"got {amount} for group {group.get('group_name')!r}"
                    )
                buckets = {}
                for parameter in group["params"]:
                    # Match AdamW semantics: parameters without a gradient are
                    # not changed merely because they belong to an optimizer.
                    if parameter.grad is None:
                        continue
                    reference = self.reference_parameters.get(id(parameter))
                    if reference is None:
                        raise RuntimeError(
                            "Reference-decayed optimizer parameter has no teacher value"
                        )
                    bucket = buckets.setdefault((parameter.device, parameter.dtype), ([], []))
                    bucket[0].append(parameter)
                    bucket[1].append(reference)
                for parameters, references in buckets.values():
                    if parameters[0].device.type in ("cuda", "cpu"):
                        torch._foreach_lerp_(parameters, references, amount)
                    else:
                        for parameter, reference in zip(parameters, references):
                            parameter.lerp_(reference, amount)

        adam_loss = super().step(closure=None)
        return loss if closure is not None else adam_loss


def cosine_lr(
    step: int,
    total: int,
    base_lr: float,
    warmup: int,
    min_lr_ratio: float = 0.1,
) -> float:
    """Warmup, then cosine-anneal from ``base_lr`` down to ``min_lr_ratio * base_lr``.

    ``min_lr_ratio=0.1`` keeps a 10x-smaller floor instead of decaying to zero.
    """
    if step <= warmup:
        return base_lr * step / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine)


def weighted_spatial_mean(values: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """FP32 products and per-image normalization, including under autocast."""
    values = values.float()
    weight = weight.to(device=values.device, dtype=torch.float32).expand_as(values)
    axes = tuple(range(1, values.ndim))
    return ((values * weight).sum(axes) / weight.sum(axes).clamp_min(1e-12)).mean()


def _weighted_or_mean(values: torch.Tensor, weight: Optional[torch.Tensor]) -> torch.Tensor:
    return weighted_spatial_mean(values, weight) if weight is not None else values.mean()


def _depthwise_3x3(image: torch.Tensor, coefficients: Tuple[float, ...]) -> torch.Tensor:
    """Apply one fixed 3x3 filter independently to every image channel."""
    if len(coefficients) != 9:
        raise ValueError("A 3x3 filter requires exactly nine coefficients")
    kernel = image.new_tensor(coefficients).reshape(1, 1, 3, 3)
    kernel = kernel.expand(image.shape[1], 1, 3, 3)
    padded = F.pad(image, (1, 1, 1, 1), mode="reflect")
    return F.conv2d(padded, kernel, groups=image.shape[1])


def _charbonnier_map(error: torch.Tensor, epsilon: float = 1e-3) -> torch.Tensor:
    """Robust differentiable absolute error with an exact zero minimum."""
    return torch.sqrt(error.square() + epsilon * epsilon) - epsilon


def _fine_detail_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weight: Optional[torch.Tensor],
) -> torch.Tensor:
    # Subpixel residuals and sparse highlight responses are especially easy to
    # quantize away in FP16.  Keep this small fixed-filter objective in FP32
    # even when the heavier LPIPS path is autocast.
    try:
        detail_autocast = autocast(prediction.device.type, enabled=False)
    except TypeError:  # pragma: no cover - older CUDA-only torch.autocast
        detail_autocast = autocast(enabled=False)
    with detail_autocast:
        return _fine_detail_loss_fp32(
            prediction.float(),
            target.float(),
            weight,
        )


def _fine_detail_loss_fp32(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weight: Optional[torch.Tensor],
) -> torch.Tensor:
    """Focal high-frequency loss for wrinkles, eyelashes, and tiny highlights.

    A 3x3 Gaussian residual isolates roughly one-to-two-pixel detail without
    adding a coarse pyramid term.  Scharr responses retain directional wrinkle
    and eyelash structure.  Detached focal weights emphasize locations that
    contain real target detail *and* remain difficult for the student, avoiding
    the spatial dilution of the old uniform first-difference loss.
    """
    gaussian = (
        1 / 16, 2 / 16, 1 / 16,
        2 / 16, 4 / 16, 2 / 16,
        1 / 16, 2 / 16, 1 / 16,
    )
    scharr_x = (
        3 / 32, 0.0, -3 / 32,
        10 / 32, 0.0, -10 / 32,
        3 / 32, 0.0, -3 / 32,
    )
    scharr_y = (
        3 / 32, 10 / 32, 3 / 32,
        0.0, 0.0, 0.0,
        -3 / 32, -10 / 32, -3 / 32,
    )

    prediction_high = prediction - _depthwise_3x3(prediction, gaussian)
    target_high = target - _depthwise_3x3(target, gaussian)
    # Apply orientation filters to the high-pass bands, not the RGB images, so
    # smooth shading and already-correct coarse contours receive no pressure.
    prediction_x = _depthwise_3x3(prediction_high, scharr_x)
    target_x = _depthwise_3x3(target_high, scharr_x)
    prediction_y = _depthwise_3x3(prediction_high, scharr_y)
    target_y = _depthwise_3x3(target_high, scharr_y)

    high_error = _charbonnier_map(prediction_high - target_high)
    directional_error = 0.5 * (
        _charbonnier_map(prediction_x - target_x)
        + _charbonnier_map(prediction_y - target_y)
    )
    detail_error = 0.6 * high_error + 0.4 * directional_error

    with torch.no_grad():
        target_strength = (
            target_high.abs()
            + 0.5 * torch.sqrt(target_x.square() + target_y.square() + 1e-6)
        ).mean(dim=1, keepdim=True)
        remaining_error = detail_error.mean(dim=1, keepdim=True)
        target_scale = target_strength.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-4)
        error_scale = remaining_error.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-4)
        focal = torch.sqrt(
            (target_strength / target_scale).clamp(max=6.0)
            * (remaining_error / error_scale).clamp(max=6.0)
        )
        # Preserve a uniform floor while giving sparse hard details up to 4x
        # their ordinary spatial influence.
        focal_weight = 1.0 + focal.clamp(max=3.0)
        if weight is not None:
            focal_weight = focal_weight * weight.to(
                device=focal_weight.device, dtype=focal_weight.dtype
            )
    return _weighted_or_mean(detail_error, focal_weight)


def image_reconstruction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    lpips_fn: Optional[nn.Module],
    *,
    l1_weight: float,
    mse_weight: float,
    edge_weight: float,
    lpips_weight: float,
    saliency: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """LPIPS plus pixel and focal-detail losses in native ``[-1, 1]`` space."""
    prediction = prediction.float()
    target = target.to(device=prediction.device, dtype=torch.float32)
    difference = prediction - target
    l1 = _weighted_or_mean(difference.abs(), saliency)
    mse = _weighted_or_mean(difference.square(), saliency)
    edge = (_fine_detail_loss(prediction, target, saliency)
            if edge_weight > 0 else prediction.new_zeros(()))
    perceptual = prediction.new_zeros(())
    if lpips_weight > 0.0:
        if lpips_fn is None:
            raise RuntimeError("LPIPS weight is nonzero but no LPIPS network was provided")
        with torch.autocast(prediction.device.type, enabled=False):
            perceptual_map = lpips_fn(prediction.clamp(-1, 1), target.clamp(-1, 1)).float()
        if saliency is not None:
            perceptual_weight = saliency
            if perceptual_weight.shape[-2:] != perceptual_map.shape[-2:]:
                perceptual_weight = F.interpolate(
                    perceptual_weight,
                    size=perceptual_map.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            perceptual = weighted_spatial_mean(perceptual_map, perceptual_weight)
        else:
            perceptual = perceptual_map.mean()
    loss = (
        l1_weight * l1
        + mse_weight * mse
        + edge_weight * edge
        + lpips_weight * perceptual
    )
    return loss, {
        "loss": loss.detach(),
        "l1": l1.detach(),
        "mse": mse.detach(),
        "edge": edge.detach(),
        "lpips": perceptual.detach(),
    }


def build_lpips(device: torch.device, required: bool, spatial: bool) -> Optional[nn.Module]:
    if not required:
        return None
    try:
        import lpips  # type: ignore
    except ImportError as error:
        raise RuntimeError(
            "LPIPS is required for validation. Install it with `pip install lpips`."
        ) from error
    network = lpips.LPIPS(net="vgg", spatial=spatial).to(device).eval()
    for parameter in network.parameters():
        parameter.requires_grad_(False)
    return network


_FFHQ_TZONE = (
    (0.37, 0.48, 0.110, 0.070),  # left eye
    (0.63, 0.48, 0.110, 0.070),  # right eye
    (0.50, 0.60, 0.080, 0.100),  # nose
    (0.50, 0.70, 0.140, 0.070),  # mouth
)
_FFHQ_FACE_CENTER = (0.50, 0.55)
_FFHQ_FACE_SIGMA = (0.36, 0.42)
_SALIENCY_LAYOUT = "ffhq-sg2-v2"
_SALIENCY_CACHE: Dict[Tuple[int, int, str, float], torch.Tensor] = {}


def _mesh_norm(height: int, width: int, device=None, dtype=torch.float32):
    ys = torch.linspace(0.5 / height, 1.0 - 0.5 / height, height, device=device, dtype=dtype)
    xs = torch.linspace(0.5 / width, 1.0 - 0.5 / width, width, device=device, dtype=dtype)
    return torch.meshgrid(ys, xs, indexing="ij")


def _gaussian2d(yy, xx, cx, cy, sx, sy):
    return torch.exp(-0.5 * (((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2))


def tzone_support_mask(height: int, width: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """Boolean ``[1, 1, H, W]`` support of the eyes / nose / mouth ellipses."""
    yy, xx = _mesh_norm(height, width, device=device, dtype=dtype)
    support = torch.zeros(height, width, device=device, dtype=torch.bool)
    for cx, cy, rx, ry in _FFHQ_TZONE:
        support |= ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0
    return support.view(1, 1, height, width)


def face_saliency_mask(
    height: int,
    width: int,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    inner_mass: float = 0.5,
) -> torch.Tensor:
    """
    Spatial weights for an aligned StyleGAN face.

    Half of the total mass (``inner_mass``) sits on the small T-zone around
    the eyes, nose, and mouth. The remaining mass follows a face-centered
    falloff that decays toward the background.
    """
    if not 0.0 <= inner_mass <= 1.0:
        raise ValueError("inner_mass must be in [0, 1]")
    key = (int(height), int(width), _SALIENCY_LAYOUT, float(inner_mass))
    cached = _SALIENCY_CACHE.get(key)
    if cached is None:
        yy, xx = _mesh_norm(height, width, device="cpu", dtype=torch.float32)
        support = tzone_support_mask(height, width, device="cpu", dtype=torch.float32)[0, 0]
        inner = torch.zeros(height, width, dtype=torch.float32)
        for cx, cy, rx, ry in _FFHQ_TZONE:
            inner = torch.maximum(inner, _gaussian2d(yy, xx, cx, cy, rx, ry))
        inner = inner * support.float()
        face = _gaussian2d(yy, xx, _FFHQ_FACE_CENTER[0], _FFHQ_FACE_CENTER[1], *_FFHQ_FACE_SIGMA)
        outer = face * (~support).float() + 1e-3
        inner = inner / inner.sum().clamp_min(1e-8)
        outer = outer / outer.sum().clamp_min(1e-8)
        cached = (float(inner_mass) * inner + (1.0 - float(inner_mass)) * outer)
        cached = cached.view(1, 1, height, width)
        _SALIENCY_CACHE[key] = cached
    return cached.to(device=device if device is not None else cached.device, dtype=dtype)


@torch.inference_mode()
def _sample_training_ws(extractor, cfg, generator, device):
    # CPU RNG is private and plot/validation independent. Draw
    # displacement vectors for all examples to keep its consumption stable.
    z = torch.randn(cfg.batch_size, Z_DIM, generator=generator).to(device)
    dz = torch.randn(cfg.batch_size, Z_DIM, generator=generator).to(device)
    ws = extractor.map_z(z)
    displacement = extractor.G.mapping(dz, None, truncation_psi=1.0)
    # Largest-remainder allocation with a rotating randomized remainder.
    augmented = int(round(cfg.batch_size * cfg.w_augment_percent))
    labels = torch.zeros(cfg.batch_size, dtype=torch.long)
    cycle = torch.randperm(3, generator=generator) + 1
    labels[:augmented] = cycle.repeat(math.ceil(augmented / 3))[:augmented]
    labels = labels[torch.randperm(cfg.batch_size, generator=generator)]
    indices = torch.arange(ws.shape[1], device=device)
    masks = torch.stack((torch.zeros_like(indices), torch.ones_like(indices),
                         (indices < 8).long(), (indices % 2).long()))
    ws = ws + displacement * masks[labels.to(device), :, None]
    return ws, labels


def _forward(model, ws, noise_mode):
    # Native StyleGAN blocks manage their own mixed precision. An enclosing
    # autocast would otherwise override the prefix's explicit FP32 policy.
    with torch.autocast(ws.device.type, enabled=False):
        return model.synthesis(ws[:, :model.num_ws], noise_mode=noise_mode,
                               fused_modconv=False if ws.device.type == "mps" else None)


@torch.inference_mode()
def make_validation_bank(teacher, cfg, device):
    rng = torch.Generator().manual_seed(cfg.seed + 12345)
    z = torch.randn(cfg.val_count, Z_DIM, generator=rng)
    dz = torch.randn(cfg.val_count, Z_DIM, generator=rng)
    all_ws, targets = [], []
    for start in range(0, cfg.val_count, cfg.eval_batch_size):
        base = teacher.map_z(z[start:start + cfg.eval_batch_size].to(device))
        delta = teacher.G.mapping(dz[start:start + cfg.eval_batch_size].to(device), None)
        ws, delta = base.repeat_interleave(4, 0), delta.repeat_interleave(4, 0)
        indices = torch.arange(ws.shape[1], device=device)
        masks = torch.stack((torch.zeros_like(indices), torch.ones_like(indices),
                             (indices < 8).long(), (indices % 2).long()))
        ws = ws + delta * masks.repeat(base.shape[0], 1)[:, :, None]
        for offset in range(0, len(ws), cfg.eval_batch_size):
            current = ws[offset:offset + cfg.eval_batch_size]
            all_ws.append(current.cpu())
            targets.append(teacher.target_from_ws(current).cpu())
    return dict(ws=torch.cat(all_ws), targets=torch.cat(targets))


@torch.no_grad()
def evaluate(model, bank, lpips_fn, cfg, device):
    model.eval()
    sums = torch.zeros(4, 3, device=device)
    counts = torch.zeros(4, device=device)
    for start in range(0, len(bank["ws"]), cfg.eval_batch_size):
        ws = bank["ws"][start:start + cfg.eval_batch_size].to(device)
        target = bank["targets"][start:start + cfg.eval_batch_size].to(device)
        prediction = _forward(model, ws, cfg.noise_mode).float()
        error = prediction - target
        with torch.autocast(device.type, enabled=False):
            perceptual = lpips_fn(prediction.clamp(-1, 1), target.clamp(-1, 1)).float()
        values = torch.stack((error.abs().flatten(1).mean(1),
                              error.square().flatten(1).mean(1),
                              perceptual.flatten(1).mean(1)), dim=1)
        labels = torch.arange(start, start + len(ws), device=device) % 4
        sums.index_add_(0, labels, values)
        counts.index_add_(0, labels, torch.ones(len(ws), device=device))
    values = (sums / counts[:, None]).cpu().tolist()
    return {name: dict(zip(("l1", "mse", "lpips"), row))
            for name, row in zip(EVAL_GROUPS, values)}


def evaluation_score(metrics, initial, augment_percent):
    weights = (1 - augment_percent, *(augment_percent / 3 for _ in range(3)))
    return sum(weight * 0.5 * sum(metrics[name][key] / max(initial[name][key], 1e-8)
                                 for key in ("l1", "lpips"))
               for name, weight in zip(EVAL_GROUPS, weights))


@torch.no_grad()
def save_panels(model, bank, cfg, device, prefix):
    model.eval()
    for label, name in enumerate(EVAL_GROUPS):
        indices = torch.arange(label, len(bank["ws"]), 4)[:cfg.plot_count]
        target = bank["targets"][indices]
        prediction = torch.cat([_forward(model, bank["ws"][i:i+1].to(device), cfg.noise_mode).cpu()
                                for i in indices.tolist()])
        rows = (((target + 1) * .5).clamp(0, 1), ((prediction + 1) * .5).clamp(0, 1),
                ((prediction - target).abs() * 2).clamp(0, 1))
        canvas = torch.cat([torch.cat(list(row), dim=2) for row in rows], dim=1)
        Image.fromarray((canvas.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)).save(f"{prefix}_{name}.png")


def _optimizer_groups(model, teacher, cfg):
    model.requires_grad_(False)
    references = dict(teacher.G.synthesis.named_parameters())
    groups, reference_map = [], {}
    decoder_params = list(model.synthesis.decoder.parameters())
    groups.append(dict(params=decoder_params, base_lr=cfg.lr, group_name="decoder", weight_decay=cfg.weight_decay))
    for resolution in model.synthesis.block_resolutions:
        if resolution < cfg.finetune_min_resolution:
            continue
        named = [(name, p) for name, p in model.synthesis.named_parameters()
                 if name.startswith(f"b{resolution}.") and ".torgb." not in name]
        scale = cfg.early_synthesis_lr_scale if resolution < 64 else 1.0
        groups.append(dict(params=[p for _, p in named], base_lr=float(cfg.synthesis_lr) * scale,
                           group_name=f"b{resolution}", weight_decay=0.0,
                           reference_weight_decay=float(cfg.synthesis_reference_decay)))
        for name, parameter in named:
            reference = references.get(name)
            if reference is None or reference.shape != parameter.shape:
                raise ValueError(f"Missing source reference: {name}")
            reference_map[id(parameter)] = reference
    for group in groups:
        for parameter in group["params"]:
            parameter.requires_grad_(True)
    return groups, reference_map


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_save(value: dict, path: str) -> None:
    torch.save(value, path + ".tmp")
    os.replace(path + ".tmp", path)


def save_generator(model, path):
    _atomic_save(dict(format=FORMAT_NAME, config=GENERATOR_CONFIGS[model.img_resolution], generator=model.state_dict()), path)


def _validate_config(cfg):
    if cfg.resolution not in GENERATOR_CONFIGS:
        raise ValueError("Resolution must be 128 or 256")
    for field in fields(cfg):
        value = getattr(cfg, field.name)
        if isinstance(value, (int, float)) and not math.isfinite(value):
            raise ValueError(f"{field.name} must be finite")
    if min(cfg.batch_size, cfg.max_steps, cfg.val_count, cfg.eval_batch_size,
           cfg.plot_count, cfg.log_every, cfg.ckpt_every) < 1:
        raise ValueError("Batch, step, evaluation and logging counts must be positive")
    if not 0 <= cfg.head_only_steps < cfg.max_steps or min(cfg.warmup_steps, cfg.val_every, cfg.plot_every) < 0:
        raise ValueError("Invalid head-only, warmup, validation or plotting interval")
    if min(cfg.lr, cfg.synthesis_lr, cfg.grad_clip, cfg.early_synthesis_lr_scale) <= 0:
        raise ValueError("Learning rates, early LR scale and clipping must be positive")
    if min(cfg.weight_decay, cfg.synthesis_reference_decay) < 0:
        raise ValueError("Decay coefficients must be nonnegative")
    if cfg.synthesis_lr * max(1., cfg.early_synthesis_lr_scale) * cfg.synthesis_reference_decay > 1:
        raise ValueError("Reference decay times synthesis LR must not exceed 1")
    if cfg.finetune_min_resolution not in (4, 8, 16, 32, 64, 128, 256) or cfg.finetune_min_resolution > cfg.resolution:
        raise ValueError("Invalid minimum synthesis resolution")
    if not 0 <= cfg.w_augment_percent <= 1 or not 0 <= cfg.saliency_uniform_mix <= 1:
        raise ValueError("Augmentation and saliency fractions must be in [0, 1]")
    weights = (cfg.l1_weight, cfg.mse_weight, cfg.edge_weight, cfg.lpips_weight)
    if min(weights) < 0 or sum(weights) == 0:
        raise ValueError("Require nonnegative losses with at least one nonzero weight")
    if cfg.noise_mode not in ("const", "none"):
        raise ValueError("Teacher/student noise must be matched: choose const or none")


def train(cfg: TrainConfig):
    _validate_config(cfg)
    device = select_device() if cfg.device == "auto" else torch.device(cfg.device)
    set_seed(cfg.seed, device)
    source_path = cfg.resume or cfg.early_output_weights
    model = load_generator_checkpoint(source_path, device) if (source_path and os.path.exists(source_path)) else None
    teacher = StyleGANTeacher(cfg, device)
    if model is None:
        model = initialize_from_teacher(teacher, cfg.resolution, device)
    if model.img_resolution != cfg.resolution:
        raise ValueError("Checkpoint resolution does not match --resolution")
    # A frozen mapping must match the one used to create training Ws; const
    # noise buffers must also match before comparing the two synthesis paths.
    for name, value in model.mapping.state_dict().items():
        torch.testing.assert_close(value, teacher.G.mapping.state_dict()[name], rtol=1e-6, atol=1e-7)
    teacher_buffers = dict(teacher.G.synthesis.named_buffers())
    for name, value in model.synthesis.named_buffers():
        if name.endswith("noise_const") and not torch.equal(value, teacher_buffers[name]):
            raise ValueError(f"Student/teacher constant noise mismatch: {name}")
    teacher_hash = _file_sha256(cfg.gan_weights)
    continuing = bool(cfg.resume and not cfg.reset_steps)
    state = torch.load(cfg.resume, map_location="cpu", weights_only=False) if continuing else {}
    if continuing:
        if state.get("format") != TRAINING_FORMAT.replace("128", str(cfg.resolution)):
            raise ValueError("Use --reset-steps to start a fresh run from the uploaded model")
        ignored = {"resume", "reset_steps", "out_dir", "log_every", "plot_every", "ckpt_every", "device"}
        changed = [key for key, value in asdict(cfg).items()
                   if key not in ignored and state["config"].get(key, 128 if key == "resolution" else None) != value]
        if changed or state["teacher_sha256"] != teacher_hash:
            raise ValueError(f"Continuation configuration changed ({changed}); use --reset-steps")
    os.makedirs(cfg.out_dir, exist_ok=True)
    checkpoint_path = osp.join(cfg.out_dir, "checkpoint.pt")
    if not continuing and osp.exists(checkpoint_path):
        raise ValueError("Output contains a checkpoint; resume it or choose a fresh directory")
    with open(osp.join(cfg.out_dir, "args.json"), "w") as handle:
        json.dump(asdict(cfg), handle, indent=2)
    plots = osp.join(cfg.out_dir, "plots")
    os.makedirs(plots, exist_ok=True)
    groups, references = _optimizer_groups(model, teacher, cfg)
    optimizer = AdamWWithReferenceDecay(groups, reference_parameters=references, lr=cfg.lr, betas=(.9, .999))
    scaler = GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")
    if device.type == "cuda" and not cfg.amp:
        for resolution in model.synthesis.block_resolutions:
            getattr(model.synthesis, f"b{resolution}").use_fp16 = False
    lpips_fn = build_lpips(device, required=True, spatial=True)
    private_rng = torch.Generator().manual_seed(cfg.seed + 777)
    step, skipped = 0, 0
    if continuing:
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        private_rng.set_state(state["sampling_rng"])
        torch.set_rng_state(state["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        step, skipped = state["step"], state["skipped_updates"]
    if step >= cfg.max_steps:
        raise ValueError("Checkpoint already reached max-steps")
    print(f"Finetuning {cfg.resolution} pointwise_style32 on {device} from {source_path or 'full teacher'}", flush=True)
    bank = make_validation_bank(teacher, cfg, device)
    if continuing:
        initial, best_score = state["initial_validation"], state["best_score"]
        best_step, best_state = state["best_step"], state["best_generator"]
    else:
        initial = evaluate(model, bank, lpips_fn, cfg, device)
        best_score, best_step = 1., 0
        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        save_generator(model, osp.join(cfg.out_dir, "best_generator.pt"))
        for filename in ("train_metrics.jsonl", "val_metrics.jsonl"):
            open(osp.join(cfg.out_dir, filename), "w").close()
        with open(osp.join(cfg.out_dir, "initial_validation.json"), "w") as handle:
            json.dump(initial, handle, indent=2)
    face = face_saliency_mask(cfg.resolution, cfg.resolution, device=device)
    mask = ((1 - cfg.saliency_uniform_mix) * face + cfg.saliency_uniform_mix / cfg.resolution**2) if cfg.saliency else None
    trainable = [p for group in groups for p in group["params"]]
    while step < cfg.max_steps:
        update = step + 1
        model.train()
        model.mapping.eval()
        for group in optimizer.param_groups:
            prefix = group["group_name"] != "decoder"
            active = not prefix or update > cfg.head_only_steps
            local_step = update - cfg.head_only_steps if prefix else update
            total = cfg.max_steps - cfg.head_only_steps if prefix else cfg.max_steps
            group["lr"] = cosine_lr(local_step, total, group["base_lr"], cfg.warmup_steps) if active else 0.
            for parameter in group["params"]:
                parameter.requires_grad_(active)
        generated_ws, _ = _sample_training_ws(teacher, cfg, private_rng, device)
        generated_target = teacher.target_from_ws(generated_ws)
        # Clone outside inference_mode to allow saving inputs for backward.
        ws, target = generated_ws.clone(), generated_target.clone()
        optimizer.zero_grad(set_to_none=True)
        prediction = _forward(model, ws, cfg.noise_mode)
        loss, stats = image_reconstruction_loss(prediction, target, lpips_fn,
            l1_weight=cfg.l1_weight, mse_weight=cfg.mse_weight, edge_weight=cfg.edge_weight,
            lpips_weight=cfg.lpips_weight, saliency=mask)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
        if not scaler.is_enabled() and not torch.isfinite(grad_norm):
            raise FloatingPointError("Non-finite unscaled gradient")
        previous_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() < previous_scale:
            skipped += 1
            if skipped > 100:
                raise FloatingPointError("More than 100 skipped AMP updates")
            continue
        step = update
        if step == 1 or step % cfg.log_every == 0:
            record = dict(step=step, grad_norm=float(grad_norm), skipped=skipped,
                          learning_rates={g["group_name"]: g["lr"] for g in optimizer.param_groups},
                          **{key: float(value) for key, value in stats.items()})
            with open(osp.join(cfg.out_dir, "train_metrics.jsonl"), "a") as handle:
                handle.write(json.dumps(record) + "\n")
            print(f"step {step}/{cfg.max_steps} loss={float(loss.detach()):.5f} grad={float(grad_norm):.3f}", flush=True)
        if (cfg.val_every and step % cfg.val_every == 0) or step == cfg.max_steps:
            metrics = evaluate(model, bank, lpips_fn, cfg, device)
            score = evaluation_score(metrics, initial, cfg.w_augment_percent)
            with open(osp.join(cfg.out_dir, "val_metrics.jsonl"), "a") as handle:
                handle.write(json.dumps(dict(step=step, score=score, groups=metrics)) + "\n")
            print(f"VAL step={step} score={score:.5f} (initial=1)", flush=True)
            if score < best_score:
                best_score, best_step = score, step
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                save_generator(model, osp.join(cfg.out_dir, "best_generator.pt"))
        if cfg.plot_every and step % cfg.plot_every == 0:
            save_panels(model, bank, cfg, device, osp.join(plots, f"step_{step:06d}"))
        if step % cfg.ckpt_every == 0 or step == cfg.max_steps:
            checkpoint = dict(format=TRAINING_FORMAT.replace("128", str(cfg.resolution)), step=step, generator=model.state_dict(),
                generator_config=GENERATOR_CONFIGS[cfg.resolution], config=asdict(cfg), optimizer=optimizer.state_dict(),
                scaler=scaler.state_dict(), teacher_sha256=teacher_hash, sampling_rng=private_rng.get_state(),
                torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                best_generator=best_state, best_score=best_score, best_step=best_step,
                initial_validation=initial, skipped_updates=skipped)
            _atomic_save(checkpoint, checkpoint_path)
    save_generator(model, osp.join(cfg.out_dir, "last_generator.pt"))
    model.load_state_dict(best_state)
    save_generator(model, osp.join(cfg.out_dir, "best_generator.pt"))
    if cfg.plot_every:
        save_panels(model, bank, cfg, device, osp.join(plots, "best"))
    print(f"Completed: {cfg.out_dir}; best step={best_step}, validation score={best_score:.5f}", flush=True)
    return model


def build_argparser():
    parser = argparse.ArgumentParser(description="Finetune the fixed pointwise_style32 generator")
    defaults = TrainConfig()
    for field in fields(defaults):
        value = getattr(defaults, field.name)
        options = dict(default=value)
        if isinstance(value, bool):
            options["action"] = argparse.BooleanOptionalAction
        else:
            options["type"] = type(value) if value is not None else str
        if field.name == "device":
            options["choices"] = ("auto", "cuda", "mps", "cpu")
        elif field.name == "resolution":
            options["choices"] = (128, 256)
        elif field.name == "noise_mode":
            options["choices"] = ("const", "none")
        parser.add_argument("--" + field.name.replace("_", "-"), **options)
    return parser


if __name__ == "__main__":
    train(TrainConfig(**vars(build_argparser().parse_args())))
