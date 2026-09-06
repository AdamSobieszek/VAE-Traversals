"""Train an early RGB output for StyleGAN2 ``b256`` activations.

The frozen generator supplies paired samples:

* input: its ``b256`` feature map, ``[B, C, 256, 256]``;
* target: its final 1024 image area-downsampled to ``[B, 3, 256, 256]``.

``EarlyOutputDecoder`` starts from StyleGAN's pretrained ``b256.torgb``
evaluated at the mapping network's average W.  A tiny reparameterizable head
learns a spatially linear correction and a low-rank nonlinear color correction.
At export, toRGB and the linear branch become one ordinary 3x3 convolution.
There is no PCA, latent bottleneck, posterior, KL term, or normalization tower.
Pass ``--train-baseline`` to fine-tune the folded toRGB branch at half the
learning rate used by the residual head.

Run from ``train_traversals`` with, for example::

    /opt/anaconda3/envs/manip311/bin/python -m models.StyleGAN_SkipNetwork \
        --gan-weights /path/to/stylegan2-ffhq-1024x1024.pkl \
        --out-dir experiments/stylegan_early_output_fast_v2
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
STOP_RESOLUTION = 256
GAN_RESOLUTION = 1024
Z_DIM = 512
W_DIM = 512


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
        block256 = getattr(self.G.synthesis, f"b{STOP_RESOLUTION}")
        self.feature_channels = int(block256.conv1.out_channels)
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
            truncation_multiplier = 1.0 + float(np.random.randn()) * 0.2
        else:
            truncation_multiplier = 1.0

        return self.G.mapping(z, None, truncation_psi=self.truncation_psi * truncation_multiplier)

    @torch.inference_mode()
    def pairs_from_ws(self, ws: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(b256 features, 4x-downsampled final output)``."""
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
            raise RuntimeError("StyleGAN synthesis did not produce b256 features and RGB")
        target = downsample_stylegan_output(img)
        return features.detach(), target.detach()

    @torch.inference_mode()
    def pairs_from_z(self, z: torch.Tensor, random_truncation: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.pairs_from_ws(self.map_z(z, random_truncation))

    @torch.inference_mode()
    def sample_pairs(
        self, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z = torch.randn(batch_size, self.z_dim, device=device)
        return self.pairs_from_z(z, random_truncation=True)

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


def cosine_lr(step: int, total: int, base_lr: float, warmup: int) -> float:
    if step <= warmup:
        return base_lr * step / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


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


def _gradient_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weight: Optional[torch.Tensor],
) -> torch.Tensor:
    pred_x = prediction[..., :, 1:] - prediction[..., :, :-1]
    target_x = target[..., :, 1:] - target[..., :, :-1]
    pred_y = prediction[..., 1:, :] - prediction[..., :-1, :]
    target_y = target[..., 1:, :] - target[..., :-1, :]
    if weight is None:
        return F.l1_loss(pred_x, target_x) + F.l1_loss(pred_y, target_y)
    weight_x = 0.5 * (weight[..., :, 1:] + weight[..., :, :-1])
    weight_y = 0.5 * (weight[..., 1:, :] + weight[..., :-1, :])
    return weighted_spatial_mean((pred_x - target_x).abs(), weight_x) + weighted_spatial_mean(
        (pred_y - target_y).abs(), weight_y
    )


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
    """LPIPS plus pixel and edge losses in StyleGAN's native ``[-1, 1]`` space."""
    target = target.to(device=prediction.device, dtype=prediction.dtype)
    difference = prediction - target
    l1 = _weighted_or_mean(difference.abs(), saliency)
    mse = _weighted_or_mean(difference.square(), saliency)
    edge = _gradient_loss(prediction, target, saliency)
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


@torch.no_grad()
def save_reconstruction_grid(
    path: str,
    model: nn.Module,
    features: torch.Tensor,
    targets: torch.Tensor,
    count: int,
) -> None:
    was_training = model.training
    model.eval()
    count = min(int(count), features.shape[0])
    output = model(features[:count])
    target_disp = _rgb_for_display(targets[:count])
    baseline_disp = _rgb_for_display(output["baseline"])
    prediction_disp = _rgb_for_display(output["prediction"])
    rows = (
        target_disp,
        baseline_disp,
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


@torch.no_grad()
def evaluate(
    model: nn.Module,
    features: torch.Tensor,
    targets: torch.Tensor,
    lpips_fn: Optional[nn.Module],
    cfg: "TrainConfig",
    saliency: Optional[torch.Tensor],
) -> Dict[str, float]:
    was_training = model.training
    model.eval()
    sums: Dict[str, float] = {}
    for index in range(features.shape[0]):
        output = model(features[index : index + 1])
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
        baseline_mse = _weighted_or_mean(
            (output["baseline"] - targets[index : index + 1]).square(), saliency
        )
        values = {
            **stats,
            "loss": loss.detach(),
            "baseline_mse": baseline_mse.detach(),
        }
        for key, value in values.items():
            sums[key] = sums.get(key, 0.0) + float(value)
    metrics = {key: value / features.shape[0] for key, value in sums.items()}
    metrics["psnr"] = 10.0 * math.log10(4.0 / max(metrics["mse"], 1e-12))
    metrics["relative_mse_improvement"] = (
        metrics["baseline_mse"] - metrics["mse"]
    ) / max(metrics["baseline_mse"], 1e-12)
    model.train(was_training)
    return metrics


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    gan_weights: str = DEFAULT_GAN_WEIGHTS
    out_dir: str = "experiments/stylegan_early_output_fast_v2"
    batch_size: int = 2
    max_steps: int = 10_000
    lr: float = 2e-4
    weight_decay: float = 2e-4
    hidden_channels: int = 16
    train_baseline: bool = False
    truncation_psi: float = 1.0
    noise_mode: str = "const"
    l1_weight: float = 1.0
    mse_weight: float = 0.25
    edge_weight: float = 0.1
    lpips_weight: float = 0.5
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
        features, target = extractor.pairs_from_z(z)
        all_features.append(features)
        all_targets.append(target)
    return torch.cat(all_features), torch.cat(all_targets)


def _validate_config(cfg: TrainConfig) -> None:
    if cfg.batch_size < 1 or cfg.max_steps < 1:
        raise ValueError("batch_size and max_steps must be positive")
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
    if not 0.0 <= cfg.ema_decay < 1.0:
        raise ValueError("ema_decay must be in [0, 1)")
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
    print("#. Building frozen full StyleGAN pair generator...")
    extractor = StyleGANFeatureExtractor(
        gan_weights=None if cfg.skip_gan_load else cfg.gan_weights,
        noise_mode=cfg.noise_mode,
        truncation_psi=cfg.truncation_psi,
        device=device,
        load_weights=not cfg.skip_gan_load,
    )
    print(
        f"  \\__Input: [{extractor.feature_channels}, 256, 256]; "
        "target: [3, 256, 256] from area-downsampled 1024 RGB"
    )

    model = EarlyOutputDecoder(
        baseline=extractor.make_torgb_baseline(trainable=cfg.train_baseline),
        hidden_channels=cfg.hidden_channels,
    ).to(device)
    print(f"#. Trainable decoder parameters: {model.num_parameters():,}")
    if hasattr(model, "deploy_macs"):
        print(
            f"  \\__Estimated deployed decoder MACs: "
            f"{model.deploy_macs() / 1e9:.3f}G per image"
        )
    baseline_parameters = [
        parameter for parameter in model.baseline.parameters() if parameter.requires_grad
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
            "lr_scale": 1.0,
            "warmup_steps": cfg.warmup_steps,
        }
    ]
    if baseline_parameters:
        parameter_groups.append(
            {
                "params": baseline_parameters,
                "lr": 0.5 * cfg.lr,
                "lr_scale": 0.5,
                "warmup_steps": 0,
            }
        )
        print("  \\__Trainable folded toRGB baseline LR: 0.5x, cosine with no warmup")
    optimizer = torch.optim.AdamW(
        parameter_groups,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.999),
    )
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
        checkpoint = torch.load(cfg.resume, map_location=device)
        expected_architecture = "stylegan_early_output_v2"
        if checkpoint.get("architecture") != expected_architecture:
            raise ValueError(
                f"Resume checkpoint architecture {checkpoint.get('architecture')!r} "
                f"does not match {expected_architecture!r}."
            )
        saved_train_baseline = bool(
            checkpoint.get("config", {}).get("train_baseline", False)
        )
        if saved_train_baseline != cfg.train_baseline:
            raise ValueError(
                "Resume checkpoint train_baseline setting does not match this run"
            )
        model.load_state_dict(checkpoint["decoder"])
        ema.ema.load_state_dict(checkpoint.get("ema", checkpoint["decoder"]))
        best_loss = float(checkpoint.get("best_loss", best_loss))
        if not cfg.reset_steps:
            if "optimizer" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer"])
            start_step = int(checkpoint.get("step", 0))

    validation_count = max(cfg.plot_count, 1)
    print(f"#. Generating {validation_count} held-out pair(s)...")
    val_features, val_targets = make_validation_pairs(
        extractor,
        validation_count,
        device,
        cfg.seed + 12345,
        cfg.skip_gan_load,
    )
    saliency_full = (
        face_saliency_mask(
            STOP_RESOLUTION, STOP_RESOLUTION, device=device, dtype=torch.float32
        )
        if cfg.saliency
        else None
    )

    with torch.no_grad():
        initial = model(val_features[:1])
        initial_delta = (initial["prediction"] - initial["baseline"]).abs().max()
    print(f"  \\__Initial residual max: {float(initial_delta):.2e} (should be 0)")

    model.train()
    cached_features: Optional[torch.Tensor] = None
    cached_targets: Optional[torch.Tensor] = None
    running: Dict[str, torch.Tensor] = {}
    running_updates = 0
    started_at = time.time()
    last_log = time.perf_counter()

    for step in range(start_step + 1, cfg.max_steps + 1):
        for group in optimizer.param_groups:
            group["lr"] = cosine_lr(
                step,
                cfg.max_steps,
                cfg.lr * float(group.get("lr_scale", 1.0)),
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
                    generated_features, generated_targets = extractor.sample_pairs(
                        cfg.batch_size, device
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
        if use_full_resolution or cfg.crop_size == STOP_RESOLUTION:
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
            output = model(features)
            loss, stats = image_reconstruction_loss(
                output["prediction"],
                targets,
                lpips_fn,
                l1_weight=cfg.l1_weight,
                mse_weight=cfg.mse_weight,
                edge_weight=cfg.edge_weight,
                lpips_weight=cfg.lpips_weight,
                saliency=saliency,
            )
            with torch.no_grad():
                stats["baseline_mse"] = _weighted_or_mean(
                    (output["baseline"] - targets).square(), saliency
                )

        trainable_parameters = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
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
            keys = ("loss", "l1", "mse", "edge", "lpips", "baseline_mse")
            zero = torch.zeros((), device=device)
            values = torch.stack(
                [running.get(key, zero) / running_updates for key in keys]
            ).cpu().tolist()
            average = dict(zip(keys, values))
            relative = (average["baseline_mse"] - average["mse"]) / max(
                average["baseline_mse"], 1e-12
            )
            now = time.perf_counter()
            rate = running_updates / max(now - last_log, 1e-8)
            last_log = now
            elapsed = time.time() - started_at
            print(
                f"step {step:06d}/{cfg.max_steps}  loss={average['loss']:.4f}  "
                f"l1={average['l1']:.4f}  mse={average['mse']:.5f}  "
                f"lpips={average['lpips']:.4f}  edge={average['edge']:.4f}  "
                f"ΔtoRGB={relative:+.2%}  full={int(use_full_resolution)}  "
                f"lr={learning_rate:.2e}  rate={rate:.2f} step/s  t={elapsed:.0f}s"
            )
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
            print(
                f"  \\__VAL ema  loss={metrics['loss']:.4f}  "
                f"mse={metrics['mse']:.5f}  psnr={metrics['psnr']:.2f}dB  "
                f"lpips={metrics['lpips']:.4f}  "
                f"ΔtoRGB={metrics['relative_mse_improvement']:+.2%}"
            )
            with open(osp.join(cfg.out_dir, "val_metrics.jsonl"), "a") as handle:
                handle.write(json.dumps({"step": step, **metrics}) + "\n")
            if metrics["loss"] < best_loss:
                best_loss = metrics["loss"]
                torch.save(
                    {
                        "architecture": "stylegan_early_output_v2",
                        "step": step,
                        "decoder": ema.ema.state_dict(),
                        "metrics": metrics,
                        "config": asdict(cfg),
                    },
                    osp.join(cfg.out_dir, "best_ema.pt"),
                )

        if cfg.plot_every > 0 and (step % cfg.plot_every == 0 or step == 1):
            plot_path = osp.join(plots_dir, f"early_output_step_{step:06d}.jpg")
            save_reconstruction_grid(
                plot_path,
                model,
                val_features,
                val_targets,
                cfg.plot_count,
            )
            print(
                f"  \\__Plot {plot_path} "
                "(rows: target / toRGB / refined / |refined-target|)"
            )

        if step % cfg.ckpt_every == 0 or step == cfg.max_steps:
            checkpoint_path = osp.join(cfg.out_dir, "checkpoint.pt")
            torch.save(
                {
                    "architecture": "stylegan_early_output_v2",
                    "step": step,
                    "decoder": model.state_dict(),
                    "ema": ema.ema.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": asdict(cfg),
                    "best_loss": best_loss,
                },
                checkpoint_path,
            )
            torch.save(
                ema.ema.state_dict(),
                osp.join(cfg.out_dir, "early_output_decoder_ema.pt"),
            )
            print(f"  \\__Saved {checkpoint_path}")

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
            "architecture": "stylegan_early_output_v2",
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
    assert restored["architecture"] == "stylegan_early_output_v2"
    assert crop_features.shape == (2, channels, 24, 24)
    assert crop_target.shape == (2, 3, 24, 24)
    print(
        f"[smoke] decoder OK  input={tuple(features.shape)}  "
        f"output={tuple(output['prediction'].shape)}  params={model.num_parameters():,}"
    )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a residual early RGB output from StyleGAN b256 features"
    )
    parser.add_argument("--gan-weights", default=DEFAULT_GAN_WEIGHTS)
    parser.add_argument("--out-dir", default="experiments/stylegan_early_output_fast_v2")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=30_000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--hidden-channels", type=int, default=16)
    parser.add_argument(
        "--train-baseline",
        action="store_true",
        help="fine-tune folded StyleGAN toRGB weights at half the refiner LR",
    )
    parser.add_argument("--truncation-psi", type=float, default=1.0)
    parser.add_argument(
        "--noise-mode", default="random", choices=("const", "random", "none")
    )
    parser.add_argument("--l1-weight", type=float, default=1.0)
    parser.add_argument("--mse-weight", type=float, default=0.25)
    parser.add_argument("--edge-weight", type=float, default=0.1)
    parser.add_argument("--lpips-weight", type=float, default=0.5)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument(
        "--crops-per-sample",
        type=int,
        default=4,
        help="optimizer steps that reuse one expensive StyleGAN pair batch",
    )
    parser.add_argument(
        "--fullres-every",
        type=int,
        default=100,
        help="train on full 256 frames every N steps (0 disables)",
    )
    parser.add_argument(
        "--saliency",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="weight LPIPS/pixel/edge losses toward aligned FFHQ face detail",
    )
    parser.add_argument("--warmup-steps", type=int, default=10_000)
    parser.add_argument("--ema-decay", type=float, default=0.99)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--val-every", type=int, default=400)
    parser.add_argument("--plot-every", type=int, default=400)
    parser.add_argument("--plot-count", type=int, default=4)
    parser.add_argument("--ckpt-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
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
    if args.smoke:
        smoke_test_decoder()
        return
    cfg = TrainConfig(
        gan_weights=args.gan_weights,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden_channels=args.hidden_channels,
        train_baseline=args.train_baseline,
        truncation_psi=args.truncation_psi,
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
