"""Train an early RGB output for StyleGAN2 ``b128`` activations.

The frozen generator supplies paired samples:

* input: its ``b128`` feature map, ``[B, C, 128, 128]``;
* target: its final 1024 image area-downsampled to ``[B, 3, 128, 128]``.

``EarlyOutputDecoder`` starts from StyleGAN's pretrained ``b128.torgb``
evaluated at the mapping network's average W.  A tiny reparameterizable head
learns a spatially linear correction and a low-rank nonlinear color correction.
At export, toRGB and the linear branch become one ordinary 3x3 convolution.
There is no PCA, latent bottleneck, posterior, KL term, or normalization tower.
Pass ``--train-baseline`` to fine-tune the folded toRGB branch at half the
learning rate used by the residual head.

The default ``decoder`` training mode retains that feature/target training
path.  ``--training-mode finetuning`` instead loads an already-integrated
early-output generator, freezes its mapping network, and jointly fine-tunes
its copied StyleGAN synthesis prefix and fused decoder against W/target pairs
made by the frozen full-resolution teacher::

    /opt/anaconda3/envs/manip311/bin/python -m models.StyleGAN_SkipNetwork128 \
        --training-mode finetuning \
        --early-output-weights models/pretrained/generators/StyleGAN2/\
stylegan2-ffhq-config-f-early-output-128.pt \
        --gan-weights models/pretrained/generators/StyleGAN2/\
stylegan2-ffhq-config-f.pt \
        --out-dir experiments/stylegan_early_output_128_finetuned

Run from ``train_traversals`` with, for example::

    /opt/anaconda3/envs/manip311/bin/python -m models.StyleGAN_SkipNetwork128 \
        --gan-weights /path/to/stylegan2-ffhq-1024x1024.pkl \
        --out-dir experiments/stylegan_early_output_128_fast_v2
"""

from __future__ import annotations

import argparse
import copy
import io
import json
import math
import os
import os.path as osp
import pickle
import sys
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

try:
    from torch.amp import GradScaler, autocast
except ImportError:  # pragma: no cover - compatibility with older PyTorch
    from torch.cuda.amp import GradScaler, autocast

from models.stylegan_feature_codec import face_saliency_mask, weighted_spatial_mean
from models.StyleGAN2_mps.early_output_model import (
    FORMAT_NAME as EARLY_OUTPUT_FORMAT_NAME,
    Generator as IntegratedEarlyOutputGenerator,
    load_generator_checkpoint,
)


# ---------------------------------------------------------------------------
# StyleGAN loading and paired feature/target generation
# ---------------------------------------------------------------------------

_TRAIN_TRAVERSALS_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
DEFAULT_GAN_WEIGHTS = osp.join(
    _TRAIN_TRAVERSALS_ROOT,
    "models",
    "pretrained",
    "generators",
    "StyleGAN2",
    "stylegan2-ffhq-config-f.pt",
)
DEFAULT_EARLY_OUTPUT_WEIGHTS = osp.join(
    _TRAIN_TRAVERSALS_ROOT,
    "models",
    "pretrained",
    "generators",
    "StyleGAN2",
    "stylegan2-ffhq-config-f-early-output-128.pt",
)
STOP_RESOLUTION = 128
GAN_RESOLUTION = 1024
Z_DIM = 512
W_DIM = 512
TRAINING_MODES = ("decoder", "finetuning")
W_AUGMENTATION_TYPES = 3
DECODER_LOSS_WEIGHTS = {
    "l1_weight": 1.0,
    "mse_weight": 0.25,
    "edge_weight": 0.1,
    "lpips_weight": 0.5,
}
FINETUNING_LOSS_WEIGHTS = {
    # LPIPS is the main fine-detail objective once the pretrained prefix can
    # adapt.  L1 remains a strong color/structure anchor; MSE and the gradient
    # focal detail term are deliberately balanced against losses already
    # covered by L1 and VGG features.
    "l1_weight": 0.75,
    "mse_weight": 0.1,
    "edge_weight": 0.25,
    "lpips_weight": 1.0,
}


def _convert_rosinality_stylegan2_state_dict(
    state: dict, latent_avg: Optional[torch.Tensor] = None
) -> dict:
    """Convert the bundled Rosinality checkpoint to the local NVLabs layout."""
    converted = {}

    def copy_tensor(source: str, target: str, transform=None) -> None:
        if source not in state:
            raise KeyError(f"Missing Rosinality StyleGAN2 key: {source!r}")
        value = state[source]
        converted[target] = transform(value) if transform is not None else value

    for index in range(8):
        copy_tensor(f"style.{index + 1}.weight", f"mapping.fc{index}.weight")
        copy_tensor(f"style.{index + 1}.bias", f"mapping.fc{index}.bias")
    if latent_avg is not None:
        converted["mapping.w_avg"] = latent_avg.reshape(-1)

    copy_tensor("input.input", "synthesis.b4.const", lambda value: value.squeeze(0))

    def copy_conv(source: str, target: str, noise_index: int) -> None:
        copy_tensor(
            f"{source}.conv.weight",
            f"{target}.weight",
            lambda value: value.squeeze(0),
        )
        copy_tensor(f"{source}.conv.modulation.weight", f"{target}.affine.weight")
        copy_tensor(f"{source}.conv.modulation.bias", f"{target}.affine.bias")
        copy_tensor(
            f"{source}.noise.weight",
            f"{target}.noise_strength",
            lambda value: value.squeeze(),
        )
        copy_tensor(f"{source}.activate.bias", f"{target}.bias")
        copy_tensor(
            f"noises.noise_{noise_index}",
            f"{target}.noise_const",
            lambda value: value.squeeze(0).squeeze(0),
        )

    def copy_torgb(source: str, target: str) -> None:
        copy_tensor(
            f"{source}.conv.weight",
            f"{target}.weight",
            lambda value: value.squeeze(0),
        )
        copy_tensor(f"{source}.conv.modulation.weight", f"{target}.affine.weight")
        copy_tensor(f"{source}.conv.modulation.bias", f"{target}.affine.bias")
        copy_tensor(f"{source}.bias", f"{target}.bias", lambda value: value.reshape(-1))

    copy_conv("conv1", "synthesis.b4.conv1", noise_index=0)
    copy_torgb("to_rgb1", "synthesis.b4.torgb")
    for block_index, resolution in enumerate((8, 16, 32, 64, 128, 256, 512, 1024)):
        copy_conv(
            f"convs.{2 * block_index}",
            f"synthesis.b{resolution}.conv0",
            noise_index=2 * block_index + 1,
        )
        copy_conv(
            f"convs.{2 * block_index + 1}",
            f"synthesis.b{resolution}.conv1",
            noise_index=2 * block_index + 2,
        )
        copy_torgb(f"to_rgbs.{block_index}", f"synthesis.b{resolution}.torgb")

    return converted


def _load_stylegan_state_dict(gan_weights: str) -> dict:
    if not osp.isfile(gan_weights):
        raise FileNotFoundError(
            f"StyleGAN2 checkpoint not found: {gan_weights!r}. "
            "Pass --gan-weights to an FFHQ 1024 .pkl or .pt file."
        )
    ext = osp.splitext(gan_weights)[1].lower()
    if ext == ".pkl":
        mps_root = osp.join(osp.dirname(__file__), "StyleGAN2_mps")
        if mps_root not in sys.path:
            sys.path.insert(0, mps_root)
        with open(gan_weights, "rb") as handle:
            data = pickle.load(handle)
        if not isinstance(data, dict) or "G_ema" not in data:
            raise KeyError(
                f"Expected NVIDIA pickle with 'G_ema', got keys="
                f"{list(data) if isinstance(data, dict) else type(data)}"
            )
        return data["G_ema"].state_dict()

    checkpoint = torch.load(gan_weights, map_location="cpu")
    if isinstance(checkpoint, dict):
        for key in ("g_ema", "G_ema", "state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                state = checkpoint[key]
                if "style.1.weight" in state:
                    return _convert_rosinality_stylegan2_state_dict(
                        state, latent_avg=checkpoint.get("latent_avg")
                    )
                return state
            if key in checkpoint and hasattr(checkpoint[key], "state_dict"):
                return checkpoint[key].state_dict()
        if all(isinstance(key, str) for key in checkpoint):
            return checkpoint
    raise ValueError(f"Unrecognized StyleGAN checkpoint format: {gan_weights!r}")


def downsample_stylegan_output(image: torch.Tensor) -> torch.Tensor:
    """Area-downsample a StyleGAN 1024 output to the early-output resolution."""
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError(f"Expected BCHW RGB image, got {tuple(image.shape)}")
    if image.shape[-2:] == (STOP_RESOLUTION, STOP_RESOLUTION):
        return image.float()
    return F.interpolate(
        image.float(),
        size=(STOP_RESOLUTION, STOP_RESOLUTION),
        mode="area",
    )


def apply_w_augmentation(
    ws: torch.Tensor,
    displacements: torch.Tensor,
    augmentation_types: torch.Tensor,
) -> torch.Tensor:
    """Add mapped W displacements using one of three layer-wise patterns."""
    if ws.ndim != 3 or ws.shape[-1] != W_DIM:
        raise ValueError(f"Expected W latents shaped [B, L, {W_DIM}], got {tuple(ws.shape)}")
    if displacements.shape != ws.shape:
        raise ValueError(
            "W displacement shape must match the input latents: "
            f"{tuple(displacements.shape)} != {tuple(ws.shape)}"
        )
    if augmentation_types.shape != (ws.shape[0],):
        raise ValueError(
            "augmentation_types must contain one value per W latent, got "
            f"{tuple(augmentation_types.shape)}"
        )
    if augmentation_types.device != ws.device:
        augmentation_types = augmentation_types.to(ws.device)
    if bool(((augmentation_types < 0) | (augmentation_types >= W_AUGMENTATION_TYPES)).any()):
        raise ValueError(
            f"augmentation_types values must be in [0, {W_AUGMENTATION_TYPES - 1}]"
        )

    layer_indices = torch.arange(ws.shape[1], device=ws.device)
    layer_masks = torch.stack(
        (
            torch.ones_like(layer_indices, dtype=torch.bool),
            layer_indices < 8,
            layer_indices.remainder(2) == 1,
        )
    )
    selected_masks = layer_masks.index_select(0, augmentation_types.to(torch.long))
    return ws + displacements * selected_masks.unsqueeze(-1).to(displacements.dtype)


class StyleGANFeatureExtractor(nn.Module):
    """Frozen full StyleGAN used only to make supervised early-output pairs."""

    def __init__(
        self,
        gan_weights: Optional[str] = DEFAULT_GAN_WEIGHTS,
        noise_mode: str = "const",
        truncation_psi: float = 1.0,
        device: Optional[torch.device] = None,
        load_weights: bool = True,
    ):
        super().__init__()
        from models.StyleGAN2_mps.model import Generator as StyleGAN2Generator

        self.G = StyleGAN2Generator(Z_DIM, 0, W_DIM, GAN_RESOLUTION, 3)
        if load_weights:
            if gan_weights is None:
                raise ValueError("gan_weights is required when load_weights=True")
            state = _load_stylegan_state_dict(gan_weights)
            missing, unexpected = self.G.load_state_dict(state, strict=False)
            missing = [
                key
                for key in missing
                if not key.startswith("synthesis.blocks.")
                and not key.endswith("resample_filter")
            ]
            if missing:
                print(f"  \\__Warning: missing StyleGAN keys: {len(missing)}")
            if unexpected:
                print(f"  \\__Warning: unexpected StyleGAN keys: {len(unexpected)}")

        self.noise_mode = noise_mode
        self.truncation_psi = float(truncation_psi)
        self.z_dim = Z_DIM
        stop_block = getattr(self.G.synthesis, f"b{STOP_RESOLUTION}")
        self.feature_channels = int(stop_block.conv1.out_channels)
        self.feature_resolution = STOP_RESOLUTION
        self.block_resolutions = list(self.G.synthesis.block_resolutions)

        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()
        if device is not None:
            self.to(device)

    def train(self, mode: bool = True):
        return super().train(False)

    def map_z(self, z: torch.Tensor, random_truncation: bool = False) -> torch.Tensor:
        if random_truncation:
            truncation_multiplier = (1.0 + float(np.random.randn()) * 0.1)*0.95
        else:
            truncation_multiplier = 1.0

        return self.G.mapping(z, None, truncation_psi=self.truncation_psi * truncation_multiplier)

    @torch.inference_mode()
    def augment_ws(self, ws: torch.Tensor, percent: float) -> torch.Tensor:
        """Randomly apply style-mixing-like displacement patterns per sample."""
        if not 0.0 <= percent <= 1.0:
            raise ValueError(f"W augmentation percent must be in [0, 1], got {percent}")
        if percent == 0.0 or ws.shape[0] == 0:
            return ws

        augment_mask = torch.rand(ws.shape[0], device=ws.device) < percent
        selected_indices = augment_mask.nonzero(as_tuple=False).flatten()
        if selected_indices.numel() == 0:
            return ws

        augmentation_types = torch.randint(
            W_AUGMENTATION_TYPES,
            (selected_indices.numel(),),
            device=ws.device,
        )
        augmented_ws = ws.clone()
        augmented_ws[selected_indices] = self.augment_ws_by_type(
            ws.index_select(0, selected_indices),
            augmentation_types,
        )
        return augmented_ws

    @torch.inference_mode()
    def augment_ws_by_type(
        self,
        ws: torch.Tensor,
        augmentation_types: torch.Tensor,
    ) -> torch.Tensor:
        """Sample untruncated displacements and apply the requested patterns."""
        displacement_z = torch.randn(
            ws.shape[0],
            self.z_dim,
            device=ws.device,
        )
        # Displacements always use the untruncated mapping distribution,
        # independently of the truncation used to sample the base W latents.
        displacements = self.G.mapping(
            displacement_z,
            None,
            truncation_psi=1.0,
        )
        return apply_w_augmentation(
            ws,
            displacements,
            augmentation_types,
        )

    @torch.inference_mode()
    def pairs_from_ws(self, ws: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(b128 features, 8x-downsampled final output)``."""
        ws = ws.to(torch.float32)
        x = img = features = None
        w_index = 0
        fused_modconv = False if ws.device.type == "mps" else None
        for resolution in self.block_resolutions:
            block = getattr(self.G.synthesis, f"b{resolution}")
            cur_ws = ws.narrow(1, w_index, block.num_conv + block.num_torgb)
            x, img = block(
                x,
                img,
                cur_ws,
                noise_mode=self.noise_mode,
                fused_modconv=fused_modconv,
            )
            w_index += block.num_conv
            if resolution == STOP_RESOLUTION:
                features = x.float()

        if features is None or img is None:
            raise RuntimeError("StyleGAN synthesis did not produce b128 features and RGB")
        target = downsample_stylegan_output(img)
        return features.detach(), target.detach()

    @torch.inference_mode()
    def target_from_ws(self, ws: torch.Tensor) -> torch.Tensor:
        """Synthesize and downsample only the final teacher output for W inputs."""
        ws = ws.to(torch.float32)
        x = img = None
        w_index = 0
        fused_modconv = False if ws.device.type == "mps" else None
        for resolution in self.block_resolutions:
            block = getattr(self.G.synthesis, f"b{resolution}")
            cur_ws = ws.narrow(1, w_index, block.num_conv + block.num_torgb)
            x, img = block(
                x,
                img,
                cur_ws,
                noise_mode=self.noise_mode,
                fused_modconv=fused_modconv,
            )
            w_index += block.num_conv
        if img is None:
            raise RuntimeError("StyleGAN synthesis did not produce an RGB target")
        return downsample_stylegan_output(img).detach()

    @torch.inference_mode()
    def pairs_from_z(self, z: torch.Tensor, random_truncation: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.pairs_from_ws(self.map_z(z, random_truncation))

    @torch.inference_mode()
    def sample_pairs(
        self,
        batch_size: int,
        device: torch.device,
        w_augment_percent: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z = torch.randn(batch_size, self.z_dim, device=device)
        ws = self.map_z(z, random_truncation=True)
        ws = self.augment_ws(ws, w_augment_percent)
        return self.pairs_from_ws(ws)

    @torch.inference_mode()
    def sample_ws_targets(
        self,
        batch_size: int,
        device: torch.device,
        w_augment_percent: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return teacher-mapped W inputs and their downsampled final RGB targets."""
        z = torch.randn(batch_size, self.z_dim, device=device)
        ws = self.map_z(z, random_truncation=True)
        ws = self.augment_ws(ws, w_augment_percent)
        return ws.detach(), self.target_from_ws(ws)

    def make_torgb_baseline(self, *, trainable: bool = False) -> nn.Module:
        block = getattr(self.G.synthesis, f"b{STOP_RESOLUTION}")
        return FrozenToRGBBaseline(
            block.torgb, self.G.mapping.w_avg, trainable=trainable
        )


# Kept for callers that used the old extractor name.
AbridgedStyleGANFeatureExtractor = StyleGANFeatureExtractor


# ---------------------------------------------------------------------------
# Folded toRGB baseline plus trainable residual decoder
# ---------------------------------------------------------------------------

@torch.no_grad()
def _fold_average_w_torgb(
    to_rgb: nn.Module, average_w: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Turn fixed-style StyleGAN toRGB into an ordinary 1x1 convolution."""
    average_w = average_w.detach().to(
        device=to_rgb.weight.device,
        dtype=to_rgb.weight.dtype,
    ).reshape(1, -1)
    style = to_rgb.affine(average_w).reshape(-1) * float(to_rgb.weight_gain)
    weight = to_rgb.weight.detach() * style.reshape(1, -1, 1, 1)
    bias = to_rgb.bias.detach().reshape(-1)
    return weight.contiguous(), bias.contiguous()


class FrozenToRGBBaseline(nn.Module):
    """Average-W toRGB folded into a standard 1x1 convolution.

    By default the folded weights remain frozen.  ``trainable=True`` exposes
    them to the optimizer as the optional half-learning-rate baseline group.
    """

    def __init__(
        self,
        to_rgb: nn.Module,
        average_w: torch.Tensor,
        *,
        trainable: bool = False,
    ):
        super().__init__()
        weight, bias = _fold_average_w_torgb(to_rgb, average_w)
        self.in_channels = int(weight.shape[1])
        self.trainable = bool(trainable)
        if self.trainable:
            self.weight = nn.Parameter(weight)
            self.bias = nn.Parameter(bias)
        else:
            self.register_buffer("weight", weight)
            self.register_buffer("bias", bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.conv2d(features.float(), self.weight, self.bias)


class FastRGBResidualRefiner(nn.Module):
    """Latency-oriented residual head with only dense, compiler-friendly ops.

    The spatial branch is linear so it can be fused with the fixed toRGB at
    export.  The pointwise bottleneck supplies inexpensive nonlinear channel
    mixing without normalization or memory-bound depthwise convolutions.
    """

    def __init__(self, in_channels: int, hidden_channels: int = 16):
        super().__init__()
        if hidden_channels < 1:
            raise ValueError("hidden_channels must be positive")
        self.spatial = nn.Conv2d(in_channels, 3, kernel_size=3, padding=1)
        self.mix_in = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.activation = nn.SiLU(inplace=True)
        self.mix_out = nn.Conv2d(hidden_channels, 3, kernel_size=1)
        nn.init.zeros_(self.spatial.weight)
        nn.init.zeros_(self.spatial.bias)
        nn.init.zeros_(self.mix_out.weight)
        nn.init.zeros_(self.mix_out.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = features.float()
        nonlinear = self.mix_out(self.activation(self.mix_in(features)))
        return self.spatial(features) + nonlinear


class FusedFastEarlyOutputDecoder(nn.Module):
    """Deployment form: fused toRGB/spatial conv plus a pointwise MLP."""

    def __init__(self, in_channels: int, hidden_channels: int = 16):
        super().__init__()
        self.in_channels = int(in_channels)
        self.linear = nn.Conv2d(in_channels, 3, kernel_size=3, padding=1)
        self.mix_in = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.activation = nn.SiLU(inplace=True)
        self.mix_out = nn.Conv2d(hidden_channels, 3, kernel_size=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = features.float()
        return self.linear(features) + self.mix_out(
            self.activation(self.mix_in(features))
        )


class EarlyOutputDecoder(nn.Module):
    """Fast train-time decoder initialized as folded toRGB plus zero residual."""

    def __init__(
        self,
        baseline: FrozenToRGBBaseline,
        hidden_channels: int = 16,
    ):
        super().__init__()
        self.baseline = baseline
        self.refiner = FastRGBResidualRefiner(
            baseline.in_channels, hidden_channels=hidden_channels
        )
        self.residual_gain = nn.Parameter(torch.ones(()))
        self.in_channels = baseline.in_channels

    def train(self, mode: bool = True):
        super().train(mode)
        if not self.baseline.trainable:
            self.baseline.eval()
        return self

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        baseline = self.baseline(features)
        residual = self.refiner(features)
        prediction = baseline + self.residual_gain * residual
        return {
            "prediction": prediction,
            "baseline": baseline,
            "residual": residual,
        }

    @torch.no_grad()
    def to_deploy(self) -> FusedFastEarlyOutputDecoder:
        """Exactly fold the baseline and spatial branches for deployment."""
        hidden_channels = int(self.refiner.mix_in.out_channels)
        deployed = FusedFastEarlyOutputDecoder(self.in_channels, hidden_channels).to(
            device=self.refiner.spatial.weight.device,
            dtype=self.refiner.spatial.weight.dtype,
        )
        gain = self.residual_gain.detach().to(self.refiner.spatial.weight)
        linear_weight = gain * self.refiner.spatial.weight.detach()
        linear_bias = gain * self.refiner.spatial.bias.detach()
        center = linear_weight.shape[-1] // 2
        linear_weight[:, :, center, center].add_(
            self.baseline.weight.detach()[:, :, 0, 0]
        )
        linear_bias.add_(self.baseline.bias.detach())
        deployed.linear.weight.copy_(linear_weight)
        deployed.linear.bias.copy_(linear_bias)
        deployed.mix_in.load_state_dict(self.refiner.mix_in.state_dict())
        deployed.mix_out.weight.copy_(gain * self.refiner.mix_out.weight.detach())
        deployed.mix_out.bias.copy_(gain * self.refiner.mix_out.bias.detach())
        deployed.eval()
        return deployed

    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def deploy_macs(self, resolution: int = STOP_RESOLUTION) -> int:
        hidden = int(self.refiner.mix_in.out_channels)
        per_pixel = (
            self.in_channels * 3 * 3 * 3
            + self.in_channels * hidden
            + hidden * 3
        )
        return int(resolution * resolution * per_pixel)


# ---------------------------------------------------------------------------
# Losses, metrics, visualization, and schedules
# ---------------------------------------------------------------------------

class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.995):
        self.decay = float(decay)
        self.ema = copy.deepcopy(model)
        for parameter in self.ema.parameters():
            parameter.requires_grad_(False)
        self.ema.eval()
        self._ema_parameters = tuple(self.ema.parameters())

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        amount = 1.0 - self.decay
        for ema_parameter, parameter in zip(self._ema_parameters, model.parameters()):
            ema_parameter.lerp_(parameter.detach(), amount)


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


def random_crop_pair(*tensors: torch.Tensor, crop: int) -> Tuple[torch.Tensor, ...]:
    if not tensors:
        raise ValueError("At least one tensor is required")
    height, width = tensors[0].shape[-2:]
    if any(tensor.shape[-2:] != (height, width) for tensor in tensors):
        raise ValueError("All tensors must have matching spatial dimensions")
    if crop >= height or crop >= width:
        return tensors
    top = int(torch.randint(0, height - crop + 1, ()).item())
    left = int(torch.randint(0, width - crop + 1, ()).item())
    return tuple(
        tensor[..., top : top + crop, left : left + crop] for tensor in tensors
    )


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
    target = target.to(device=prediction.device, dtype=prediction.dtype)
    difference = prediction - target
    l1 = _weighted_or_mean(difference.abs(), saliency)
    mse = _weighted_or_mean(difference.square(), saliency)
    edge = _fine_detail_loss(prediction, target, saliency)
    perceptual = prediction.new_zeros(())
    if lpips_weight > 0.0:
        if lpips_fn is None:
            raise RuntimeError("LPIPS weight is nonzero but no LPIPS network was provided")
        perceptual_map = lpips_fn(prediction.clamp(-1, 1), target.clamp(-1, 1))
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
            "LPIPS is part of the early-output objective. Install it with `pip install lpips` "
            "or explicitly set --lpips-weight 0."
        ) from error
    network = lpips.LPIPS(net="vgg", spatial=spatial).to(device).eval()
    for parameter in network.parameters():
        parameter.requires_grad_(False)
    return network


def _rgb_for_display(image: torch.Tensor) -> torch.Tensor:
    return ((image.detach().float().cpu() + 1.0) * 0.5).clamp(0.0, 1.0)


def _forward_training_model(
    model: nn.Module,
    model_inputs: torch.Tensor,
    *,
    training_mode: str,
    noise_mode: str,
) -> Dict[str, torch.Tensor]:
    """Normalize decoder-only and integrated-generator forward outputs."""
    if training_mode == "decoder":
        output = model(model_inputs)
        if not isinstance(output, dict) or "prediction" not in output:
            raise RuntimeError("Decoder training model returned an unexpected output")
        return output
    if training_mode != "finetuning":
        raise ValueError(f"Unknown training mode: {training_mode!r}")
    if not isinstance(model, IntegratedEarlyOutputGenerator):
        raise TypeError("Finetuning requires an integrated early-output Generator")

    # Teacher mapping has the full generator's W-slot count.  Prefix slots are
    # laid out identically, so the integrated b128 synthesis consumes its
    # leading slots directly; its own frozen mapping is intentionally bypassed.
    prefix_ws = model_inputs[:, : model.num_ws]
    # NVLabs StyleGAN layers manage precision themselves (the b128 block is
    # FP16 on CUDA while earlier prefix blocks remain FP32) and assert those
    # explicit dtypes.  An enclosing torch autocast context would override the
    # FP32 blocks and trip SynthesisBlock's dtype assertion.  Disable only the
    # outer autocast here; GradScaler still scales the backward pass, the
    # StyleGAN-native FP16 policy remains active, and the loss/LPIPS operations
    # resume autocasting after this context exits.
    try:
        synthesis_autocast = autocast(prefix_ws.device.type, enabled=False)
    except TypeError:  # pragma: no cover - older CUDA-only torch.autocast
        synthesis_autocast = autocast(enabled=False)
    with synthesis_autocast:
        components = model.synthesis(
            prefix_ws,
            return_components=True,
            noise_mode=noise_mode,
            fused_modconv=False if prefix_ws.device.type == "mps" else None,
        )
    if not isinstance(components, dict) or "prediction" not in components:
        raise RuntimeError("Integrated early-output synthesis returned no prediction")
    return components


@torch.no_grad()
def save_reconstruction_grid(
    path: str,
    model: nn.Module,
    model_inputs: torch.Tensor,
    targets: torch.Tensor,
    count: int,
    *,
    training_mode: str = "decoder",
    noise_mode: str = "const",
) -> None:
    was_training = model.training
    model.eval()
    count = min(int(count), model_inputs.shape[0])
    output = _forward_training_model(
        model,
        model_inputs[:count],
        training_mode=training_mode,
        noise_mode=noise_mode,
    )
    target_disp = _rgb_for_display(targets[:count])
    prediction_disp = _rgb_for_display(output["prediction"])
    if "baseline" in output:
        rows = (
            target_disp,
            _rgb_for_display(output["baseline"]),
            prediction_disp,
            (prediction_disp - target_disp).abs(),
        )
    else:
        rows = (
            target_disp,
            prediction_disp,
            (prediction_disp - target_disp).abs(),
        )
    height, width = targets.shape[-2:]
    canvas = torch.zeros(3, len(rows) * height, count * width)
    for row_index, row in enumerate(rows):
        for column_index in range(count):
            canvas[
                :,
                row_index * height : (row_index + 1) * height,
                column_index * width : (column_index + 1) * width,
            ] = row[column_index]
    array = (canvas.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    os.makedirs(osp.dirname(path) or ".", exist_ok=True)
    Image.fromarray(array, mode="RGB").save(path, format="JPEG", quality=92)
    model.train(was_training)
    if training_mode == "finetuning":
        model.mapping.eval()


@torch.no_grad()
def save_w_augmentation_grid(
    path: str,
    model: nn.Module,
    extractor: StyleGANFeatureExtractor,
    *,
    training_mode: str,
    noise_mode: str,
    seed: int,
) -> None:
    """Save three seeded identities under each of the three W augmentations."""
    was_training = model.training
    model.eval()

    # Use a private CPU generator so these base identities stay fixed without
    # resetting the live RNG used for the displacement latents.
    base_generator = torch.Generator(device="cpu")
    base_generator.manual_seed(seed)
    base_z = torch.randn(3, extractor.z_dim, generator=base_generator).to(
        next(model.parameters()).device
    )
    base_ws = extractor.map_z(base_z)
    expanded_ws = base_ws.repeat_interleave(W_AUGMENTATION_TYPES, dim=0)
    augmentation_types = torch.arange(
        W_AUGMENTATION_TYPES,
        device=expanded_ws.device,
    ).repeat(base_ws.shape[0])
    augmented_ws = extractor.augment_ws_by_type(
        expanded_ws,
        augmentation_types,
    )

    target_columns: List[torch.Tensor] = []
    prediction_columns: List[torch.Tensor] = []
    difference_columns: List[torch.Tensor] = []
    # Synthesize and reconstruct one column at a time to avoid retaining nine
    # full-resolution StyleGAN outputs or prefix activation graphs at once.
    for index in range(augmented_ws.shape[0]):
        ws = augmented_ws[index : index + 1]
        if training_mode == "finetuning":
            model_inputs = ws
            target = extractor.target_from_ws(ws)
        else:
            model_inputs, target = extractor.pairs_from_ws(ws)
        output = _forward_training_model(
            model,
            model_inputs,
            training_mode=training_mode,
            noise_mode=noise_mode,
        )
        target_display = _rgb_for_display(target)[0]
        prediction_display = _rgb_for_display(output["prediction"])[0]
        target_columns.append(target_display)
        prediction_columns.append(prediction_display)
        difference_columns.append((prediction_display - target_display).abs())

    rows = (target_columns, prediction_columns, difference_columns)
    height = width = STOP_RESOLUTION
    canvas = torch.zeros(
        3,
        len(rows) * height,
        augmented_ws.shape[0] * width,
    )
    for row_index, row in enumerate(rows):
        for column_index, image in enumerate(row):
            canvas[
                :,
                row_index * height : (row_index + 1) * height,
                column_index * width : (column_index + 1) * width,
            ] = image
    array = (canvas.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    os.makedirs(osp.dirname(path) or ".", exist_ok=True)
    Image.fromarray(array, mode="RGB").save(path, format="JPEG", quality=92)

    model.train(was_training)
    if training_mode == "finetuning":
        model.mapping.eval()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    model_inputs: torch.Tensor,
    targets: torch.Tensor,
    lpips_fn: Optional[nn.Module],
    cfg: "TrainConfig",
    saliency: Optional[torch.Tensor],
) -> Dict[str, float]:
    was_training = model.training
    model.eval()
    sums: Dict[str, float] = {}
    for index in range(model_inputs.shape[0]):
        output = _forward_training_model(
            model,
            model_inputs[index : index + 1],
            training_mode=cfg.training_mode,
            noise_mode=cfg.noise_mode,
        )
        loss, stats = image_reconstruction_loss(
            output["prediction"],
            targets[index : index + 1],
            lpips_fn,
            l1_weight=cfg.l1_weight,
            mse_weight=cfg.mse_weight,
            edge_weight=cfg.edge_weight,
            lpips_weight=cfg.lpips_weight,
            saliency=saliency,
        )
        values = {
            **stats,
            "loss": loss.detach(),
        }
        if "baseline" in output:
            values["baseline_mse"] = _weighted_or_mean(
                (output["baseline"] - targets[index : index + 1]).square(),
                saliency,
            ).detach()
        for key, value in values.items():
            sums[key] = sums.get(key, 0.0) + float(value)
    metrics = {key: value / model_inputs.shape[0] for key, value in sums.items()}
    metrics["psnr"] = 10.0 * math.log10(4.0 / max(metrics["mse"], 1e-12))
    if "baseline_mse" in metrics:
        metrics["relative_mse_improvement"] = (
            metrics["baseline_mse"] - metrics["mse"]
        ) / max(metrics["baseline_mse"], 1e-12)
    model.train(was_training)
    if cfg.training_mode == "finetuning":
        model.mapping.eval()
    return metrics


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    training_mode: str = "decoder"
    gan_weights: str = DEFAULT_GAN_WEIGHTS
    early_output_weights: str = DEFAULT_EARLY_OUTPUT_WEIGHTS
    out_dir: str = "experiments/stylegan_early_output_128_fast_v2"
    batch_size: int = 2
    max_steps: int = 10_000
    lr: float = 2e-4
    synthesis_lr: Optional[float] = None
    weight_decay: float = 2e-4
    synthesis_reference_decay: Optional[float] = None
    hidden_channels: int = 16
    train_baseline: bool = False
    truncation_psi: float = 1.0
    w_augment_percent: float = 0.5
    noise_mode: str = "const"
    l1_weight: Optional[float] = None
    mse_weight: Optional[float] = None
    edge_weight: Optional[float] = None
    lpips_weight: Optional[float] = None
    crop_size: int = 128
    crops_per_sample: int = 4
    fullres_every: int = 100
    saliency: bool = True
    warmup_steps: int = 500
    ema_decay: float = 0.995
    grad_clip: float = 1.0
    log_every: int = 25
    val_every: int = 200
    plot_every: int = 200
    plot_count: int = 4
    ckpt_every: int = 500
    seed: int = 1
    resume: Optional[str] = None
    reset_steps: bool = False
    amp: bool = False
    skip_gan_load: bool = False

    def __post_init__(self) -> None:
        defaults = (
            FINETUNING_LOSS_WEIGHTS
            if self.training_mode == "finetuning"
            else DECODER_LOSS_WEIGHTS
        )
        for name, value in defaults.items():
            if getattr(self, name) is None:
                setattr(self, name, value)
        if self.synthesis_lr is None:
            self.synthesis_lr = 0.1 * self.lr
        if self.synthesis_reference_decay is None:
            # AdamW-style coefficient: the optimizer multiplies it by the
            # synthesis group's current scheduled LR on every step.
            self.synthesis_reference_decay = (
                0.1 if self.training_mode == "finetuning" else self.weight_decay
            )


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


@torch.inference_mode()
def make_validation_pairs(
    extractor: StyleGANFeatureExtractor,
    count: int,
    device: torch.device,
    seed: int,
    skip_gan_load: bool,
    training_mode: str = "decoder",
) -> Tuple[torch.Tensor, torch.Tensor]:
    if skip_gan_load:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        features = torch.randn(
            count,
            extractor.feature_channels,
            STOP_RESOLUTION,
            STOP_RESOLUTION,
            generator=generator,
        ).to(device)
        return features, torch.tanh(features[:, :3]).detach()

    # Generate one validation sample at a time so the temporary 1024 tensor
    # does not scale with plot_count.
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    all_features: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []
    for _ in range(count):
        z = torch.randn(1, Z_DIM, generator=generator).to(device)
        if training_mode == "finetuning":
            ws = extractor.map_z(z)
            model_inputs = ws.detach()
            target = extractor.target_from_ws(ws)
        else:
            model_inputs, target = extractor.pairs_from_z(z)
        all_features.append(model_inputs)
        all_targets.append(target)
    return torch.cat(all_features), torch.cat(all_targets)


def _integrated_generator_config(model: IntegratedEarlyOutputGenerator) -> dict:
    return {
        "z_dim": model.z_dim,
        "c_dim": model.c_dim,
        "w_dim": model.w_dim,
        "img_resolution": model.img_resolution,
        "img_channels": model.img_channels,
        "decoder_hidden_channels": int(model.synthesis.decoder.mix_in.out_channels),
    }


def _validate_mapping_match(
    student: IntegratedEarlyOutputGenerator,
    teacher: StyleGANFeatureExtractor,
) -> None:
    """Ensure frozen inference mapping matches the mapping producing train Ws."""
    student_state = student.mapping.state_dict()
    teacher_state = teacher.G.mapping.state_dict()
    if student_state.keys() != teacher_state.keys():
        raise ValueError("Teacher and early-output mapping networks have different states")
    mismatches = []
    for name, student_value in student_state.items():
        teacher_value = teacher_state[name]
        if not torch.allclose(student_value, teacher_value, rtol=1e-6, atol=1e-7):
            mismatches.append(name)
    if mismatches:
        raise ValueError(
            "The integrated generator mapping does not match the full StyleGAN "
            f"teacher ({len(mismatches)} mismatches; first: {mismatches[:5]}). "
            "Because mapping is frozen, use checkpoints derived from the same source."
        )


def _finetuning_parameter_groups(
    model: IntegratedEarlyOutputGenerator,
    teacher: StyleGANFeatureExtractor,
    cfg: TrainConfig,
) -> Tuple[List[dict], List[nn.Parameter], Dict[int, torch.Tensor]]:
    """Unfreeze used prefix/decoder parameters while keeping mapping and toRGBs fixed."""
    model.requires_grad_(False)
    model.mapping.eval()

    decoder = model.synthesis.decoder
    residual_parameters = list(decoder.mix_in.parameters()) + list(
        decoder.mix_out.parameters()
    )
    linear_parameters = list(decoder.linear.parameters())
    synthesis_named_parameters = [
        (name, parameter)
        for name, parameter in model.synthesis.named_parameters()
        if not name.startswith("decoder.") and ".torgb." not in name
    ]
    synthesis_parameters = [parameter for _, parameter in synthesis_named_parameters]
    teacher_parameters = dict(teacher.G.synthesis.named_parameters())
    reference_parameters: Dict[int, torch.Tensor] = {}
    for name, parameter in synthesis_named_parameters:
        reference = teacher_parameters.get(name)
        if reference is None or reference.shape != parameter.shape:
            raise ValueError(
                f"Teacher has no matching synthesis reference for {name!r} "
                f"with shape {tuple(parameter.shape)}"
            )
        reference_parameters[id(parameter)] = reference.detach()
    for parameter in residual_parameters + linear_parameters + synthesis_parameters:
        parameter.requires_grad_(True)

    parameter_groups = [
        {
            "params": residual_parameters,
            "lr": cfg.lr,
            "base_lr": cfg.lr,
            "lr_scale": 1.0,
            "warmup_steps": cfg.warmup_steps,
            "group_name": "decoder_residual",
        },
        {
            "params": linear_parameters,
            "lr": 0.5 * cfg.lr,
            "base_lr": 0.5 * cfg.lr,
            "lr_scale": 0.5,
            "warmup_steps": 0,
            "group_name": "decoder_linear",
        },
        {
            "params": synthesis_parameters,
            "lr": float(cfg.synthesis_lr),
            "base_lr": float(cfg.synthesis_lr),
            "warmup_steps": 0,
            "group_name": "synthesis_prefix",
            # Disable zero-centered AdamW decay for the pretrained prefix and
            # use decoupled decay toward the full teacher weights instead.
            "weight_decay": 0.0,
            "reference_weight_decay": float(cfg.synthesis_reference_decay),
        },
    ]
    trainable_parameters = (
        residual_parameters + linear_parameters + synthesis_parameters
    )
    if not all(group["params"] for group in parameter_groups):
        raise RuntimeError("A finetuning optimizer parameter group is unexpectedly empty")
    return parameter_groups, trainable_parameters, reference_parameters


def _save_integrated_generator(
    path: str,
    model: IntegratedEarlyOutputGenerator,
    cfg: TrainConfig,
    *,
    step: int,
    metrics: Optional[Dict[str, float]] = None,
) -> None:
    """Save a finetuned EMA state in the public combined-checkpoint format."""
    checkpoint = {
        "format": EARLY_OUTPUT_FORMAT_NAME,
        "generator": model.state_dict(),
        "config": _integrated_generator_config(model),
        "source": {
            "finetuned_from": osp.basename(cfg.early_output_weights),
            "teacher_weights": osp.basename(cfg.gan_weights),
            "step": int(step),
            "synthesis_reference_decay": float(cfg.synthesis_reference_decay),
            "synthesis_reference_decay_semantics": "adamw_lr_scaled_pre_update_v1",
            "loss_weights": {
                name: float(getattr(cfg, name))
                for name in FINETUNING_LOSS_WEIGHTS
            },
        },
    }
    if metrics is not None:
        checkpoint["metrics"] = metrics
    torch.save(checkpoint, path)


def export_finetuning_checkpoint(
    checkpoint_path: str,
    output_path: str,
    *,
    verify: bool = True,
) -> str:
    """Export a training checkpoint's last, non-EMA combined generator."""
    if not osp.isfile(checkpoint_path):
        raise FileNotFoundError(f"Training checkpoint not found: {checkpoint_path!r}")
    try:
        training_checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
    except TypeError:  # pragma: no cover - older PyTorch
        training_checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(training_checkpoint, dict):
        raise ValueError("Training checkpoint must be a dictionary")
    if training_checkpoint.get("format") == EARLY_OUTPUT_FORMAT_NAME:
        raise ValueError(
            "Input is already a public combined generator checkpoint; pass the "
            "finetuning run's checkpoint.pt instead"
        )
    architecture = training_checkpoint.get("architecture")
    if architecture != "stylegan_early_output_128_finetuning_v1":
        raise ValueError(
            "Expected a stylegan_early_output_128_finetuning_v1 checkpoint, got "
            f"{architecture!r}"
        )

    generator_state = training_checkpoint.get("generator")
    generator_config = training_checkpoint.get("generator_config")
    if not isinstance(generator_state, dict):
        raise ValueError(
            "Finetuning checkpoint has no non-EMA 'generator' state dictionary"
        )
    if not isinstance(generator_config, dict):
        raise ValueError("Finetuning checkpoint has no 'generator_config' dictionary")

    training_config = training_checkpoint.get("config", {})
    if not isinstance(training_config, dict):
        training_config = {}
    source = {
        "exported_from": osp.basename(checkpoint_path),
        "state": "generator",
        "step": int(training_checkpoint.get("step", 0)),
    }
    for output_key, config_key in (
        ("finetuned_from", "early_output_weights"),
        ("teacher_weights", "gan_weights"),
    ):
        value = training_config.get(config_key)
        if value:
            source[output_key] = osp.basename(str(value))
    reference_decay = training_config.get("synthesis_reference_decay")
    if reference_decay is not None:
        source["synthesis_reference_decay"] = float(reference_decay)
    decay_semantics = training_checkpoint.get(
        "synthesis_reference_decay_semantics"
    )
    if decay_semantics is not None:
        source["synthesis_reference_decay_semantics"] = str(decay_semantics)

    public_checkpoint = {
        "format": EARLY_OUTPUT_FORMAT_NAME,
        "generator": dict(generator_state),
        "config": dict(generator_config),
        "source": source,
    }
    os.makedirs(osp.dirname(osp.abspath(output_path)), exist_ok=True)
    torch.save(public_checkpoint, output_path)
    if verify:
        # Strict construction/loading is the compatibility contract used by
        # early_output_model.load_generator_checkpoint().
        reloaded = load_generator_checkpoint(output_path, device="cpu")
        del reloaded
    return output_path


def _validate_config(cfg: TrainConfig) -> None:
    if cfg.training_mode not in TRAINING_MODES:
        raise ValueError(
            f"training_mode must be one of {TRAINING_MODES}, got {cfg.training_mode!r}"
        )
    if cfg.batch_size < 1 or cfg.max_steps < 1:
        raise ValueError("batch_size and max_steps must be positive")
    if cfg.lr <= 0.0 or float(cfg.synthesis_lr) <= 0.0:
        raise ValueError("lr and synthesis_lr must be positive")
    if cfg.weight_decay < 0.0 or not math.isfinite(cfg.weight_decay):
        raise ValueError("weight_decay must be finite and nonnegative")
    if (
        float(cfg.synthesis_reference_decay) < 0.0
        or not math.isfinite(float(cfg.synthesis_reference_decay))
    ):
        raise ValueError(
            "synthesis_reference_decay must be finite and nonnegative"
        )
    if float(cfg.synthesis_lr) * float(cfg.synthesis_reference_decay) > 1.0:
        raise ValueError(
            "synthesis_lr * synthesis_reference_decay must not exceed 1"
        )
    if cfg.log_every < 1 or cfg.ckpt_every < 1:
        raise ValueError("log_every and ckpt_every must be positive")
    if cfg.val_every < 0 or cfg.plot_every < 0 or cfg.plot_count < 1:
        raise ValueError("val_every/plot_every must be nonnegative and plot_count positive")
    if cfg.crop_size < 32 or cfg.crop_size > STOP_RESOLUTION:
        raise ValueError(f"crop_size must be in [32, {STOP_RESOLUTION}]")
    if cfg.crops_per_sample < 1:
        raise ValueError("crops_per_sample must be at least 1")
    if cfg.fullres_every < 0:
        raise ValueError("fullres_every must be nonnegative")
    if cfg.hidden_channels < 1:
        raise ValueError("hidden_channels must be positive")
    if not 0.0 <= cfg.w_augment_percent <= 1.0:
        raise ValueError("w_augment_percent must be in [0, 1]")
    if not 0.0 <= cfg.ema_decay < 1.0:
        raise ValueError("ema_decay must be in [0, 1)")
    if cfg.training_mode == "finetuning":
        if cfg.skip_gan_load:
            raise ValueError("finetuning requires the full StyleGAN teacher checkpoint")
        if not cfg.early_output_weights:
            raise ValueError("finetuning requires --early-output-weights")
        if cfg.noise_mode == "random":
            raise ValueError(
                "finetuning requires --noise-mode const or none so teacher and "
                "student prefix noise remain matched"
            )
    for name in ("l1_weight", "mse_weight", "edge_weight", "lpips_weight"):
        value = getattr(cfg, name)
        if value < 0.0 or not math.isfinite(value):
            raise ValueError(f"{name} must be finite and nonnegative")
    if cfg.l1_weight + cfg.mse_weight + cfg.edge_weight + cfg.lpips_weight == 0.0:
        raise ValueError("At least one reconstruction loss weight must be nonzero")


def train(cfg: TrainConfig) -> nn.Module:
    _validate_config(cfg)
    device = select_device()
    set_seed(cfg.seed, device)
    use_amp = bool(cfg.amp and device.type == "cuda")

    os.makedirs(cfg.out_dir, exist_ok=True)
    plots_dir = osp.join(cfg.out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    with open(osp.join(cfg.out_dir, "args.json"), "w") as handle:
        json.dump(asdict(cfg), handle, indent=2)

    print(f"#. Device: {device}")
    print(f"#. Output: {cfg.out_dir}")
    print(f"#. Training mode: {cfg.training_mode}")
    print(
        f"#. W augmentation: {cfg.w_augment_percent:.1%} of training inputs, "
        f"uniformly across {W_AUGMENTATION_TYPES} layer patterns"
    )
    print("#. Building frozen full StyleGAN pair generator...")
    extractor = StyleGANFeatureExtractor(
        gan_weights=None if cfg.skip_gan_load else cfg.gan_weights,
        noise_mode=cfg.noise_mode,
        truncation_psi=cfg.truncation_psi,
        device=device,
        load_weights=not cfg.skip_gan_load,
    )
    reference_parameters: Dict[int, torch.Tensor] = {}
    if cfg.training_mode == "decoder":
        print(
            f"  \\__Input: [{extractor.feature_channels}, {STOP_RESOLUTION}, "
            f"{STOP_RESOLUTION}]; target: [3, {STOP_RESOLUTION}, "
            f"{STOP_RESOLUTION}] from area-downsampled 1024 RGB"
        )
        model: nn.Module = EarlyOutputDecoder(
            baseline=extractor.make_torgb_baseline(trainable=cfg.train_baseline),
            hidden_channels=cfg.hidden_channels,
        ).to(device)
        print(f"#. Trainable decoder parameters: {model.num_parameters():,}")
        print(
            f"  \\__Estimated deployed decoder MACs: "
            f"{model.deploy_macs() / 1e9:.3f}G per image"
        )
        baseline_parameters = [
            parameter
            for parameter in model.baseline.parameters()
            if parameter.requires_grad
        ]
        baseline_parameter_ids = {id(parameter) for parameter in baseline_parameters}
        refiner_parameters = [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad and id(parameter) not in baseline_parameter_ids
        ]
        parameter_groups = [
            {
                "params": refiner_parameters,
                "lr": cfg.lr,
                "base_lr": cfg.lr,
                "lr_scale": 1.0,
                "warmup_steps": cfg.warmup_steps,
                "group_name": "decoder_refiner",
            }
        ]
        if baseline_parameters:
            parameter_groups.append(
                {
                    "params": baseline_parameters,
                    "lr": 0.5 * cfg.lr,
                    "base_lr": 0.5 * cfg.lr,
                    "lr_scale": 0.5,
                    "warmup_steps": 0,
                    "group_name": "decoder_baseline",
                }
            )
            print(
                "  \\__Trainable folded toRGB baseline LR: 0.5x, "
                "cosine with no warmup"
            )
        trainable_parameters = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
    else:
        print(f"#. Loading integrated early-output model: {cfg.early_output_weights}")
        model = load_generator_checkpoint(cfg.early_output_weights, device=device)
        if model.img_resolution != STOP_RESOLUTION:
            raise ValueError(
                f"Finetuning requires a b{STOP_RESOLUTION} combined model; "
                f"checkpoint produces b{model.img_resolution}"
            )
        _validate_mapping_match(model, extractor)
        (
            parameter_groups,
            trainable_parameters,
            reference_parameters,
        ) = _finetuning_parameter_groups(model, extractor, cfg)
        synthesis_count = sum(
            parameter.numel() for parameter in parameter_groups[2]["params"]
        )
        decoder_count = sum(
            parameter.numel()
            for group in parameter_groups[:2]
            for parameter in group["params"]
        )
        print(
            f"  \\__Teacher W input: [B, {extractor.G.num_ws}, {W_DIM}]; "
            f"student prefix consumes first {model.num_ws} slots"
        )
        print(
            f"  \\__Trainable prefix: {synthesis_count:,} params at "
            f"{float(cfg.synthesis_lr):.2e}; decoder: {decoder_count:,} params "
            f"at {cfg.lr:.2e} (linear branch 0.5x)"
        )
        print(
            "  \\__Synthesis reference-decay coefficient: "
            f"{float(cfg.synthesis_reference_decay):.2e}; peak per-step pull "
            f"= {float(cfg.synthesis_lr) * float(cfg.synthesis_reference_decay):.2e} "
            "(scaled by scheduled LR; zero-centered decay disabled)"
        )
        print("  \\__Frozen mapping verified equal to the teacher mapping")
    optimizer_kwargs = {
        "lr": cfg.lr,
        "weight_decay": cfg.weight_decay,
        "betas": (0.9, 0.999),
    }
    if cfg.training_mode == "finetuning":
        optimizer = AdamWWithReferenceDecay(
            parameter_groups,
            reference_parameters=reference_parameters,
            **optimizer_kwargs,
        )
    else:
        optimizer = torch.optim.AdamW(parameter_groups, **optimizer_kwargs)
    try:
        scaler = GradScaler("cuda", enabled=use_amp)
    except TypeError:  # pragma: no cover - older PyTorch
        scaler = GradScaler(enabled=use_amp)
    ema = ModelEMA(model, decay=cfg.ema_decay)
    lpips_fn = build_lpips(
        device,
        required=cfg.lpips_weight > 0.0,
        spatial=cfg.saliency,
    )

    start_step = 0
    best_loss = float("inf")
    if cfg.resume:
        print(f"#. Resuming early-output checkpoint {cfg.resume}")
        try:
            checkpoint = torch.load(
                cfg.resume, map_location=device, weights_only=False
            )
        except TypeError:  # pragma: no cover - older PyTorch
            checkpoint = torch.load(cfg.resume, map_location=device)
        expected_architecture = (
            "stylegan_early_output_128_v2"
            if cfg.training_mode == "decoder"
            else "stylegan_early_output_128_finetuning_v1"
        )
        if checkpoint.get("architecture") != expected_architecture:
            raise ValueError(
                f"Resume checkpoint architecture {checkpoint.get('architecture')!r} "
                f"does not match {expected_architecture!r}."
            )
        if cfg.training_mode == "decoder":
            saved_train_baseline = bool(
                checkpoint.get("config", {}).get("train_baseline", False)
            )
            if saved_train_baseline != cfg.train_baseline:
                raise ValueError(
                    "Resume checkpoint train_baseline setting does not match this run"
                )
            model_state = checkpoint["decoder"]
        else:
            model_state = checkpoint["generator"]
        model.load_state_dict(model_state)
        ema.ema.load_state_dict(checkpoint.get("ema", model_state))
        best_loss = float(checkpoint.get("best_loss", best_loss))
        if not cfg.reset_steps:
            if "optimizer" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer"])
                if cfg.training_mode == "finetuning":
                    # Normalize optimizer-group fields across earlier reference
                    # decay implementations and resume with the requested
                    # AdamW-style coefficient.
                    synthesis_group = optimizer.param_groups[2]
                    synthesis_group["weight_decay"] = 0.0
                    synthesis_group.pop("reference_decay", None)
                    synthesis_group["reference_weight_decay"] = float(
                        cfg.synthesis_reference_decay
                    )
                    saved_semantics = checkpoint.get(
                        "synthesis_reference_decay_semantics"
                    )
                    if saved_semantics != "adamw_lr_scaled_pre_update_v1":
                        print(
                            "  \\__Migrated optimizer reference decay to "
                            "AdamW-style LR scaling applied before the task update"
                        )
            start_step = int(checkpoint.get("step", 0))

    validation_count = max(cfg.plot_count, 1)
    print(f"#. Generating {validation_count} held-out pair(s)...")
    val_features, val_targets = make_validation_pairs(
        extractor,
        validation_count,
        device,
        cfg.seed + 12345,
        cfg.skip_gan_load,
        cfg.training_mode,
    )
    # Inference tensors cannot be saved for backward by the finetuned prefix.
    # Validation never backpropagates, but ordinary tensors also keep behavior
    # consistent across PyTorch backends.
    val_features = val_features.detach().clone()
    val_targets = val_targets.detach().clone()
    saliency_full = (
        face_saliency_mask(
            STOP_RESOLUTION, STOP_RESOLUTION, device=device, dtype=torch.float32
        )
        if cfg.saliency
        else None
    )

    if cfg.training_mode == "decoder":
        with torch.no_grad():
            initial = _forward_training_model(
                model,
                val_features[:1],
                training_mode=cfg.training_mode,
                noise_mode=cfg.noise_mode,
            )
            initial_delta = (
                initial["prediction"] - initial["baseline"]
            ).abs().max()
        print(f"  \\__Initial residual max: {float(initial_delta):.2e} (should be 0)")

    model.train()
    if cfg.training_mode == "finetuning":
        model.mapping.eval()
    cached_features: Optional[torch.Tensor] = None
    cached_targets: Optional[torch.Tensor] = None
    running: Dict[str, torch.Tensor] = {}
    running_updates = 0
    started_at = time.time()
    last_log = time.perf_counter()

    for step in range(start_step + 1, cfg.max_steps + 1):
        for group in optimizer.param_groups:
            base_lr = group.get("base_lr")
            if base_lr is None:
                # Backward compatibility for optimizer states saved before
                # explicit absolute group learning rates were introduced.
                base_lr = cfg.lr * float(group.get("lr_scale", 1.0))
            group["lr"] = cosine_lr(
                step,
                cfg.max_steps,
                float(base_lr),
                int(group.get("warmup_steps", cfg.warmup_steps)),
            )
        learning_rate = float(optimizer.param_groups[0]["lr"])

        refresh = cached_features is None or (step - 1) % cfg.crops_per_sample == 0
        if refresh:
            with torch.inference_mode():
                if cfg.skip_gan_load:
                    generated_features = torch.randn(
                        cfg.batch_size,
                        extractor.feature_channels,
                        STOP_RESOLUTION,
                        STOP_RESOLUTION,
                        device=device,
                    )
                    generated_targets = torch.tanh(generated_features[:, :3])
                else:
                    if cfg.training_mode == "finetuning":
                        generated_features, generated_targets = (
                            extractor.sample_ws_targets(
                                cfg.batch_size,
                                device,
                                w_augment_percent=cfg.w_augment_percent,
                            )
                        )
                    else:
                        generated_features, generated_targets = extractor.sample_pairs(
                            cfg.batch_size,
                            device,
                            w_augment_percent=cfg.w_augment_percent,
                        )
            # ``detach`` preserves an inference tensor's special status. Clone
            # outside inference_mode so the trainable decoder can save its
            # input for backward while the generator remains fully detached.
            cached_features = generated_features.detach().clone()
            cached_targets = generated_targets.detach().clone()

        assert cached_features is not None and cached_targets is not None
        use_full_resolution = (
            cfg.fullres_every > 0 and step % cfg.fullres_every == 0
        )
        if cfg.training_mode == "finetuning":
            # Prefix synthesis needs complete W inputs and full spatial frames.
            # Cropping is applied to its RGB prediction and the target below.
            features = cached_features
            targets = cached_targets
            saliency = saliency_full
        elif use_full_resolution or cfg.crop_size == STOP_RESOLUTION:
            features = cached_features
            targets = cached_targets
            saliency = saliency_full
        elif saliency_full is None:
            features, targets = random_crop_pair(
                cached_features, cached_targets, crop=cfg.crop_size
            )
            saliency = None
        else:
            features, targets, saliency = random_crop_pair(
                cached_features,
                cached_targets,
                saliency_full,
                crop=cfg.crop_size,
            )

        optimizer.zero_grad(set_to_none=True)
        try:
            amp_context = autocast("cuda", enabled=use_amp)
        except TypeError:  # pragma: no cover - older PyTorch
            amp_context = autocast(enabled=use_amp)
        with amp_context:
            output = _forward_training_model(
                model,
                features,
                training_mode=cfg.training_mode,
                noise_mode=cfg.noise_mode,
            )
            prediction = output["prediction"]
            loss_targets = targets
            loss_saliency = saliency
            if (
                cfg.training_mode == "finetuning"
                and not use_full_resolution
                and cfg.crop_size < STOP_RESOLUTION
            ):
                if saliency is None:
                    prediction, loss_targets = random_crop_pair(
                        prediction, targets, crop=cfg.crop_size
                    )
                else:
                    prediction, loss_targets, loss_saliency = random_crop_pair(
                        prediction,
                        targets,
                        saliency,
                        crop=cfg.crop_size,
                    )
            loss, stats = image_reconstruction_loss(
                prediction,
                loss_targets,
                lpips_fn,
                l1_weight=cfg.l1_weight,
                mse_weight=cfg.mse_weight,
                edge_weight=cfg.edge_weight,
                lpips_weight=cfg.lpips_weight,
                saliency=loss_saliency,
            )
            if "baseline" in output:
                with torch.no_grad():
                    baseline = output["baseline"]
                    if baseline.shape[-2:] != loss_targets.shape[-2:]:
                        # Decoder mode crops inputs before forward, so this is
                        # defensive rather than part of the normal path.
                        baseline = F.interpolate(
                            baseline,
                            size=loss_targets.shape[-2:],
                            mode="area",
                        )
                stats["baseline_mse"] = _weighted_or_mean(
                    (baseline - loss_targets).square(), loss_saliency
                )

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_parameters, cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_parameters, cfg.grad_clip)
            optimizer.step()
        ema.update(model)

        for key, value in stats.items():
            running[key] = running.get(key, torch.zeros_like(value)) + value
        running_updates += 1

        if step % cfg.log_every == 0 or step == 1:
            keys = ["loss", "l1", "mse", "edge", "lpips"]
            if cfg.training_mode == "decoder":
                keys.append("baseline_mse")
            zero = torch.zeros((), device=device)
            values = torch.stack(
                [running.get(key, zero) / running_updates for key in keys]
            ).cpu().tolist()
            average = dict(zip(keys, values))
            now = time.perf_counter()
            rate = running_updates / max(now - last_log, 1e-8)
            last_log = now
            elapsed = time.time() - started_at
            message = (
                f"step {step:06d}/{cfg.max_steps}  loss={average['loss']:.4f}  "
                f"l1={average['l1']:.4f}  mse={average['mse']:.5f}  "
                f"lpips={average['lpips']:.4f}  detail={average['edge']:.4f}  "
            )
            if cfg.training_mode == "decoder":
                relative = (
                    average["baseline_mse"] - average["mse"]
                ) / max(average["baseline_mse"], 1e-12)
                message += f"ΔtoRGB={relative:+.2%}  "
            elif len(optimizer.param_groups) > 2:
                message += f"synth_lr={optimizer.param_groups[2]['lr']:.2e}  "
            message += (
                f"full={int(use_full_resolution)}  "
                f"lr={learning_rate:.2e}  rate={rate:.2f} step/s  t={elapsed:.0f}s"
            )
            print(message)
            running = {}
            running_updates = 0

        if cfg.val_every > 0 and (step % cfg.val_every == 0 or step == 1):
            metrics = evaluate(
                ema.ema,
                val_features,
                val_targets,
                lpips_fn,
                cfg,
                saliency_full,
            )
            validation_message = (
                f"  \\__VAL ema  loss={metrics['loss']:.4f}  "
                f"mse={metrics['mse']:.5f}  psnr={metrics['psnr']:.2f}dB  "
                f"lpips={metrics['lpips']:.4f}"
            )
            if "relative_mse_improvement" in metrics:
                validation_message += (
                    f"  ΔtoRGB={metrics['relative_mse_improvement']:+.2%}"
                )
            print(validation_message)
            with open(osp.join(cfg.out_dir, "val_metrics.jsonl"), "a") as handle:
                handle.write(json.dumps({"step": step, **metrics}) + "\n")
            if metrics["loss"] < best_loss:
                best_loss = metrics["loss"]
                best_path = osp.join(cfg.out_dir, "best_ema.pt")
                if cfg.training_mode == "decoder":
                    torch.save(
                        {
                            "architecture": "stylegan_early_output_128_v2",
                            "step": step,
                            "decoder": ema.ema.state_dict(),
                            "metrics": metrics,
                            "config": asdict(cfg),
                        },
                        best_path,
                    )
                else:
                    _save_integrated_generator(
                        best_path,
                        ema.ema,
                        cfg,
                        step=step,
                        metrics=metrics,
                    )

        if cfg.plot_every > 0 and (step % cfg.plot_every == 0 or step == 1):
            plot_path = osp.join(plots_dir, f"early_output_step_{step:06d}.jpg")
            save_reconstruction_grid(
                plot_path,
                model,
                val_features,
                val_targets,
                cfg.plot_count,
                training_mode=cfg.training_mode,
                noise_mode=cfg.noise_mode,
            )
            plot_rows = (
                "target / toRGB / refined / |refined-target|"
                if cfg.training_mode == "decoder"
                else "target / finetuned / |finetuned-target|"
            )
            print(f"  \\__Plot {plot_path} (rows: {plot_rows})")
            augmentation_plot_path = osp.join(
                plots_dir,
                f"w_augmentation_step_{step:06d}.jpg",
            )
            save_w_augmentation_grid(
                augmentation_plot_path,
                model,
                extractor,
                training_mode=cfg.training_mode,
                noise_mode=cfg.noise_mode,
                seed=cfg.seed + 23456,
            )
            print(
                f"  \\__W augmentation plot {augmentation_plot_path} "
                "(3 seeded identities x all/first-8/odd; rows: "
                "target / reconstruction / |reconstruction-target|)"
            )

        if step % cfg.ckpt_every == 0 or step == cfg.max_steps:
            checkpoint_path = osp.join(cfg.out_dir, "checkpoint.pt")
            checkpoint = {
                "architecture": (
                    "stylegan_early_output_128_v2"
                    if cfg.training_mode == "decoder"
                    else "stylegan_early_output_128_finetuning_v1"
                ),
                "step": step,
                "ema": ema.ema.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": asdict(cfg),
                "best_loss": best_loss,
            }
            if cfg.training_mode == "decoder":
                checkpoint["decoder"] = model.state_dict()
            else:
                checkpoint["generator"] = model.state_dict()
                checkpoint["generator_config"] = _integrated_generator_config(model)
                checkpoint[
                    "synthesis_reference_decay_semantics"
                ] = "adamw_lr_scaled_pre_update_v1"
            torch.save(checkpoint, checkpoint_path)
            if cfg.training_mode == "decoder":
                torch.save(
                    ema.ema.state_dict(),
                    osp.join(cfg.out_dir, "early_output_decoder_ema.pt"),
                )
            else:
                _save_integrated_generator(
                    osp.join(cfg.out_dir, "early_output_generator_ema.pt"),
                    ema.ema,
                    cfg,
                    step=step,
                )
            print(f"  \\__Saved {checkpoint_path}")

    if cfg.training_mode == "finetuning":
        final_checkpoint_path = osp.join(cfg.out_dir, "checkpoint.pt")
        final_generator_path = osp.join(
            cfg.out_dir, "early_output_generator_last.pt"
        )
        export_finetuning_checkpoint(
            final_checkpoint_path,
            final_generator_path,
            verify=True,
        )
        print(
            "  \\__Exported final non-EMA combined generator: "
            f"{final_generator_path}"
        )

    print(f"#. Training finished. Best EMA validation loss: {best_loss:.5f}")
    return model


# ---------------------------------------------------------------------------
# Focused smoke checks and CLI
# ---------------------------------------------------------------------------

def smoke_test_decoder() -> None:
    from models.StyleGAN2_mps.model import ToRGBLayer

    torch.manual_seed(1)
    channels = 16
    to_rgb = ToRGBLayer(channels, 3, w_dim=8)
    baseline = FrozenToRGBBaseline(to_rgb, torch.zeros(8))
    model = EarlyOutputDecoder(
        baseline,
        hidden_channels=8,
    )
    features = torch.randn(2, channels, 32, 32)
    target = torch.tanh(features[:, :3])
    output = model(features)
    assert output["prediction"].shape == target.shape
    assert torch.equal(output["prediction"], output["baseline"])

    full_baseline = model.baseline(features)
    crop_features, crop_target = random_crop_pair(features, target, crop=24)
    # Verify with explicit shared coordinates as random_crop_pair deliberately
    # samples them internally.
    assert torch.allclose(
        model.baseline(features[..., 4:28, 3:27]),
        full_baseline[..., 4:28, 3:27],
        atol=1e-6,
    )
    loss, stats = image_reconstruction_loss(
        output["prediction"],
        target,
        None,
        l1_weight=1.0,
        mse_weight=0.25,
        edge_weight=0.1,
        lpips_weight=0.0,
    )
    loss.backward()
    assert any(
        parameter.grad is not None and bool(parameter.grad.abs().sum() > 0)
        for parameter in model.refiner.parameters()
    )
    assert all(parameter.grad is None for parameter in model.baseline.parameters())

    model.eval()
    deployed = model.to_deploy()
    with torch.no_grad():
        deployed_output = deployed(features)
        reference_output = model(features)["prediction"]
    assert torch.allclose(deployed_output, reference_output, atol=1e-6, rtol=1e-6)

    buffer = io.BytesIO()
    torch.save(
        {
            "architecture": "stylegan_early_output_128_v2",
            "decoder": model.state_dict(),
            "stats": stats,
        },
        buffer,
    )
    buffer.seek(0)
    try:
        restored = torch.load(buffer, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - older PyTorch
        restored = torch.load(buffer, map_location="cpu")
    assert restored["architecture"] == "stylegan_early_output_128_v2"
    assert crop_features.shape == (2, channels, 24, 24)
    assert crop_target.shape == (2, 3, 24, 24)
    print(
        f"[smoke] decoder OK  input={tuple(features.shape)}  "
        f"output={tuple(output['prediction'].shape)}  params={model.num_parameters():,}"
    )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a residual early RGB output from StyleGAN b128 features, or "
            "fine-tune an integrated StyleGAN prefix/decoder"
        )
    )
    parser.add_argument(
        "--training-mode",
        choices=TRAINING_MODES,
        default="decoder",
        help="decoder preserves the original decoder-only path; finetuning trains the integrated prefix and decoder",
    )
    parser.add_argument("--gan-weights", default=DEFAULT_GAN_WEIGHTS)
    parser.add_argument(
        "--early-output-weights",
        default=DEFAULT_EARLY_OUTPUT_WEIGHTS,
        help="pretrained combined generator used to initialize finetuning",
    )
    parser.add_argument(
        "--out-dir", default="experiments/stylegan_early_output_128_fast_v2"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=80_000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--synthesis-lr",
        type=float,
        default=None,
        help="copied StyleGAN prefix LR in finetuning mode (default: 0.1x --lr)",
    )
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument(
        "--synthesis-reference-decay",
        type=float,
        default=None,
        help=(
            "AdamW-style decoupled decay coefficient for synthesis-prefix "
            "weights toward the frozen teacher; the per-step pull is this "
            "value times the current synthesis LR (finetuning default: 0.1). "
            "Ordinary zero-centered weight decay is disabled"
        ),
    )
    parser.add_argument("--hidden-channels", type=int, default=16)
    parser.add_argument(
        "--frozen-torgb",
        action="store_true",
        help="fine-tune folded StyleGAN toRGB weights at half the refiner LR",
    )
    parser.add_argument("--truncation-psi", type=float, default=1.0)
    parser.add_argument(
        "--w-augment-percent",
        type=float,
        default=0.5,
        help=(
            "probability per training input of adding an untruncated W "
            "displacement using one of three uniformly sampled layer patterns"
        ),
    )
    parser.add_argument(
        "--noise-mode",
        default=None,
        choices=("const", "random", "none"),
        help="defaults to random for decoder mode and const for finetuning",
    )
    parser.add_argument(
        "--l1-weight",
        type=float,
        default=None,
        help="mode default: decoder 1.0, finetuning 0.75",
    )
    parser.add_argument(
        "--mse-weight",
        type=float,
        default=None,
        help="mode default: decoder 0.25, finetuning 0.1",
    )
    parser.add_argument(
        "--edge-weight",
        type=float,
        default=None,
        help=(
            "weight of the focal 1-2px high-frequency detail loss; mode "
            "default: decoder 0.1, finetuning 0.25"
        ),
    )
    parser.add_argument(
        "--lpips-weight",
        type=float,
        default=None,
        help="mode default: decoder 0.5, finetuning 1.0",
    )
    parser.add_argument("--crop-size", type=int, default=96)
    parser.add_argument(
        "--crops-per-sample",
        type=int,
        default=4,
        help="optimizer steps that reuse one expensive StyleGAN pair batch",
    )
    parser.add_argument(
        "--fullres-every",
        type=int,
        default=10,
        help="train on full 128 frames every N steps (0 disables)",
    )
    parser.add_argument(
        "--saliency",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="weight LPIPS/pixel/detail losses toward aligned FFHQ face detail",
    )
    parser.add_argument("--warmup-steps", type=int, default=10_000)
    parser.add_argument("--ema-decay", type=float, default=0.9)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--val-every", type=int, default=400)
    parser.add_argument("--plot-every", type=int, default=400)
    parser.add_argument("--plot-count", type=int, default=5)
    parser.add_argument("--ckpt-every", type=int, default=800)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--resume")
    parser.add_argument("--reset-steps", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--skip-gan-load",
        action="store_true",
        help="use a random StyleGAN and synthetic RGB targets for pipeline checks",
    )
    parser.add_argument("--smoke", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    args.train_baseline = not args.frozen_torgb
    if args.noise_mode is None:
        args.noise_mode = (
            "const" if args.training_mode == "finetuning" else "random"
        )
    if args.smoke:
        smoke_test_decoder()
        return
    cfg = TrainConfig(
        training_mode=args.training_mode,
        gan_weights=args.gan_weights,
        early_output_weights=args.early_output_weights,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        lr=args.lr,
        synthesis_lr=args.synthesis_lr,
        weight_decay=args.weight_decay,
        synthesis_reference_decay=args.synthesis_reference_decay,
        hidden_channels=args.hidden_channels,
        train_baseline=args.train_baseline,
        truncation_psi=args.truncation_psi,
        w_augment_percent=args.w_augment_percent,
        noise_mode=args.noise_mode,
        l1_weight=args.l1_weight,
        mse_weight=args.mse_weight,
        edge_weight=args.edge_weight,
        lpips_weight=args.lpips_weight,
        crop_size=args.crop_size,
        crops_per_sample=args.crops_per_sample,
        fullres_every=args.fullres_every,
        saliency=args.saliency,
        warmup_steps=args.warmup_steps,
        ema_decay=args.ema_decay,
        grad_clip=args.grad_clip,
        log_every=args.log_every,
        val_every=args.val_every,
        plot_every=args.plot_every,
        plot_count=args.plot_count,
        ckpt_every=args.ckpt_every,
        seed=args.seed,
        resume=args.resume,
        reset_steps=args.reset_steps,
        amp=args.amp,
        skip_gan_load=args.skip_gan_load,
    )
    train(cfg)


if __name__ == "__main__":
    main()
