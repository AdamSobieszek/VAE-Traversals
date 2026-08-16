"""
StyleGAN2 feature codec trainer (PCA-residual VAE).

Phase 1 trains a PCA-anchored nonlinear codec on frozen StyleGAN ``b256``
features. Phase 2 optionally fine-tunes the decoder through frozen
``b512``/``b1024`` for final-image fidelity.

Usage:
    /opt/anaconda3/envs/manip311/bin/python -m models.StyleGAN_SDVAE \\
        --out-dir experiments/sdvae_pca_residual \\
        --batch-size 2 --max-steps 10000 --plot-every 100
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import os.path as osp
import pickle
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

try:
    from torch.amp import GradScaler, autocast
except ImportError:  # pragma: no cover
    from torch.cuda.amp import GradScaler, autocast

from models.stylegan_feature_codec import (
    FEATURE_CHANNELS,
    FEATURE_RESOLUTION,
    LATENT_CHANNELS,
    FrozenRank4PCA,
    PCAResidualFeatureCodec,
    StreamingPCACalibrator,
    eval_metrics,
    face_saliency_mask,
    feature_codec_loss,
    weighted_spatial_mean,
)

# ---------------------------------------------------------------------------
# Defaults
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
PREFIX_NUM_WS = 14
FULL_NUM_WS = 18


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
        copy_tensor(f"{source}.conv.weight", f"{target}.weight", lambda value: value.squeeze(0))
        copy_tensor(f"{source}.conv.modulation.weight", f"{target}.affine.weight")
        copy_tensor(f"{source}.conv.modulation.bias", f"{target}.affine.bias")
        copy_tensor(f"{source}.noise.weight", f"{target}.noise_strength", lambda value: value.squeeze())
        copy_tensor(f"{source}.activate.bias", f"{target}.bias")
        copy_tensor(
            f"noises.noise_{noise_index}",
            f"{target}.noise_const",
            lambda value: value.squeeze(0).squeeze(0),
        )

    def copy_torgb(source: str, target: str) -> None:
        copy_tensor(f"{source}.conv.weight", f"{target}.weight", lambda value: value.squeeze(0))
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
        with open(gan_weights, "rb") as f:
            data = pickle.load(f)
        if not isinstance(data, dict) or "G_ema" not in data:
            raise KeyError(
                f"Expected NVIDIA pickle with 'G_ema', got keys="
                f"{list(data) if isinstance(data, dict) else type(data)}"
            )
        return data["G_ema"].state_dict()

    ckpt = torch.load(gan_weights, map_location="cpu")
    if isinstance(ckpt, dict):
        for key in ("g_ema", "G_ema", "state_dict"):
            if key in ckpt and isinstance(ckpt[key], dict):
                state = ckpt[key]
                if "style.1.weight" in state:
                    return _convert_rosinality_stylegan2_state_dict(
                        state, latent_avg=ckpt.get("latent_avg")
                    )
                return state
            if key in ckpt and hasattr(ckpt[key], "state_dict"):
                return ckpt[key].state_dict()
        if all(isinstance(k, str) for k in ckpt.keys()):
            return ckpt
    raise ValueError(f"Unrecognized StyleGAN checkpoint format: {gan_weights!r}")


# ---------------------------------------------------------------------------
# StyleGAN feature extractor + optional high-res tail
# ---------------------------------------------------------------------------

class StyleGANFeatureExtractor(nn.Module):
    """
    Frozen StyleGAN2 generator for feature extraction and optional tail synthesis.

    When ``keep_tail=False`` (Phase 1), high-res blocks are dropped to save memory.
    When ``keep_tail=True`` (Phase 2), retains ``b512``/``b1024`` and full 18 W slots.
    """

    def __init__(
        self,
        gan_weights: Optional[str] = DEFAULT_GAN_WEIGHTS,
        stop_resolution: int = STOP_RESOLUTION,
        noise_mode: str = "const",
        truncation_psi: float = 1.0,
        device: Optional[torch.device] = None,
        load_weights: bool = True,
        keep_tail: bool = False,
    ):
        super().__init__()
        from models.StyleGAN2_mps.model import Generator as StyleGAN2Generator

        self.stop_resolution = int(stop_resolution)
        self.noise_mode = noise_mode
        self.truncation_psi = float(truncation_psi)
        self.keep_tail = bool(keep_tail)
        self.z_dim = Z_DIM
        self.w_dim = W_DIM

        self.G = StyleGAN2Generator(Z_DIM, 0, W_DIM, GAN_RESOLUTION, 3)
        if load_weights:
            state = _load_stylegan_state_dict(gan_weights)
            missing, unexpected = self.G.load_state_dict(state, strict=False)
            # ``blocks`` aliases the named ``b4`` ... ``b1024`` modules, while
            # resampling filters are fixed architecture constants. Neither is
            # present in the bundled Rosinality checkpoint.
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

        syn = self.G.synthesis
        if not self.keep_tail:
            kept = [r for r in syn.block_resolutions if r <= self.stop_resolution]
            for res in list(syn.block_resolutions):
                if res > self.stop_resolution:
                    delattr(syn, f"b{res}")
            syn.block_resolutions = kept
            num_ws = 0
            for res in syn.block_resolutions:
                block = getattr(syn, f"b{res}")
                num_ws += block.num_conv
                if res == self.stop_resolution:
                    num_ws += block.num_torgb
            syn.num_ws = num_ws
            self.G.num_ws = num_ws
            self.G.mapping.num_ws = num_ws
        else:
            # Full generator: mapping keeps 18 W slots.
            self.G.num_ws = syn.num_ws
            self.G.mapping.num_ws = syn.num_ws

        self.feature_channels = min(32768 // self.stop_resolution, 512)
        self.feature_resolution = self.stop_resolution
        self.prefix_resolutions = [r for r in self.G.synthesis.block_resolutions if r <= self.stop_resolution]
        self.tail_resolutions = [r for r in self.G.synthesis.block_resolutions if r > self.stop_resolution]

        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()
        if device is not None:
            self.to(device)

    def train(self, mode: bool = True):
        return super().train(False)

    def map_z(self, z: torch.Tensor) -> torch.Tensor:
        return self.G.mapping(z, None, truncation_psi=self.truncation_psi)

    def _run_blocks(
        self,
        resolutions: List[int],
        ws: torch.Tensor,
        x=None,
        img=None,
        start_w_idx: int = 0,
        compute_rgb: bool = True,
    ):
        syn = self.G.synthesis
        w_idx = start_w_idx
        # Grouped "fused" modulated convolution is substantially slower on
        # MPS; the unfused path uses regular batched convolutions.
        fused_modconv = False if ws.device.type == "mps" else None
        for res in resolutions:
            block = getattr(syn, f"b{res}")
            cur_ws = ws.narrow(1, w_idx, block.num_conv + block.num_torgb)
            x, img = block(
                x,
                img,
                cur_ws,
                noise_mode=self.noise_mode,
                fused_modconv=fused_modconv,
                skip_torgb=not compute_rgb,
            )
            w_idx += block.num_conv
        return x, img, w_idx

    @torch.inference_mode()
    def features_from_ws(self, ws: torch.Tensor) -> torch.Tensor:
        x, _img, _ = self._run_blocks(
            self.prefix_resolutions, ws.to(torch.float32), compute_rgb=False
        )
        assert x is not None
        return x

    @torch.inference_mode()
    def features_and_skip_from_ws(self, ws: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(x256, img256, ws)`` through the 256 stage."""
        ws = ws.to(torch.float32)
        x, img, _ = self._run_blocks(self.prefix_resolutions, ws)
        assert x is not None and img is not None
        return x, img, ws

    def synthesize_from_features(
        self,
        x256: torch.Tensor,
        img256: torch.Tensor,
        ws: torch.Tensor,
    ) -> torch.Tensor:
        """
        Continue frozen synthesis from reconstructed ``x256`` using GT skip RGB.

        Gradients flow into ``x256`` (and ``img256`` if requires_grad). Tail
        parameters stay frozen. Requires ``keep_tail=True``.
        """
        if not self.keep_tail or not self.tail_resolutions:
            raise RuntimeError("Tail synthesis requires keep_tail=True")
        # After prefix, w_idx == 13 (sum of num_conv through b256).
        x, img, _ = self._run_blocks(
            self.tail_resolutions,
            ws.to(torch.float32),
            x=x256,
            img=img256,
            start_w_idx=PREFIX_NUM_WS - 1,  # overlapping shared slot at index 13
        )
        return img

    @torch.inference_mode()
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.features_from_ws(self.map_z(z))

    def sample_features(self, batch_size: int, device: torch.device) -> torch.Tensor:
        z = torch.randn(batch_size, self.z_dim, device=device)
        return self.forward(z)

    def sample_pack(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        z = torch.randn(batch_size, self.z_dim, device=device)
        ws = self.map_z(z)
        x, img, ws = self.features_and_skip_from_ws(ws)
        return {"z": z, "ws": ws, "x": x, "img256": img}


# Backwards-compatible alias
AbridgedStyleGANFeatureExtractor = StyleGANFeatureExtractor


# ---------------------------------------------------------------------------
# Visualization PCA (rank-3 RGB projection only)
# ---------------------------------------------------------------------------

class OnlineChannelPCA(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 3, momentum: float = 0.05, max_samples: int = 8192):
        super().__init__()
        assert out_channels <= in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.momentum = float(momentum)
        self.max_samples = int(max_samples)
        self.register_buffer("mean", torch.zeros(in_channels))
        self.register_buffer("cov", torch.eye(in_channels))
        self.register_buffer("components", torch.eye(in_channels, out_channels).clone())
        self.register_buffer("n_updates", torch.zeros((), dtype=torch.long))
        self.register_buffer("initialized", torch.zeros((), dtype=torch.bool))

    @torch.no_grad()
    def update(self, maps: torch.Tensor) -> None:
        flat = maps.detach().float().permute(0, 2, 3, 1).reshape(-1, self.in_channels)
        n = flat.shape[0]
        if n > self.max_samples:
            idx = torch.randint(0, n, (self.max_samples,), device=flat.device)
            flat = flat[idx]
            n = flat.shape[0]
        batch_mean = flat.mean(dim=0)
        if not bool(self.initialized.item()):
            self.mean.copy_(batch_mean)
            centered = flat - self.mean
            self.cov.copy_((centered.T @ centered) / max(n - 1, 1))
            self.initialized.fill_(True)
        else:
            m = self.momentum
            self.mean.mul_(1.0 - m).add_(batch_mean, alpha=m)
            centered = flat - self.mean
            batch_cov = (centered.T @ centered) / max(n - 1, 1)
            self.cov.mul_(1.0 - m).add_(batch_cov, alpha=m)
        cov_cpu = (0.5 * (self.cov + self.cov.T)).detach().float().cpu()
        _evals, evecs = torch.linalg.eigh(cov_cpu)
        comps = evecs[:, -self.out_channels :].flip(1)
        for k in range(self.out_channels):
            v = comps[:, k]
            i = int(torch.argmax(v.abs()).item())
            if v[i] < 0:
                comps[:, k] = -v
        self.components.copy_(comps.to(device=self.components.device, dtype=self.components.dtype))
        self.n_updates.add_(1)

    @torch.no_grad()
    def project(self, maps: torch.Tensor) -> torch.Tensor:
        b, c, h, w = maps.shape
        flat = maps.detach().float().permute(0, 2, 3, 1).reshape(-1, c)
        proj = (flat - self.mean.to(flat)) @ self.components.to(flat)
        return proj.reshape(b, h, w, self.out_channels).permute(0, 3, 1, 2).contiguous()

    @torch.no_grad()
    def project_to_rgb(self, maps: torch.Tensor, ref: Optional[torch.Tensor] = None) -> torch.Tensor:
        rgb = self.project(maps)
        stats = self.project(ref) if ref is not None else rgb
        flat = stats.reshape(-1)
        lo = torch.quantile(flat, 0.02)
        hi = torch.quantile(flat, 0.98)
        return ((rgb - lo) / (hi - lo + 1e-8)).clamp(0.0, 1.0)


def _stylegan_preview(images: torch.Tensor, size: int) -> torch.Tensor:
    """Downsample StyleGAN RGB in ``[-1, 1]`` to an ``[0, 1]`` preview."""
    preview = F.interpolate(images.float(), size=(size, size), mode="area")
    return ((preview + 1.0) * 0.5).clamp(0.0, 1.0)


def make_pca_viz_grid(
    originals: torch.Tensor,
    latents: torch.Tensor,
    recons: torch.Tensor,
    feature_pca: OnlineChannelPCA,
    latent_pca: OnlineChannelPCA,
    n: int = 4,
    gan_gt: Optional[torch.Tensor] = None,
    gan_from_recon: Optional[torch.Tensor] = None,
) -> Image.Image:
    """
    Feature-space rows plus optional StyleGAN image rows:
      row0 original features, row1 latent, row2 nonlinear recon,
      row3 cached 1024 StyleGAN (downsampled),
      row4 StyleGAN continued from reconstructed activations.
    """
    n = min(n, originals.shape[0], latents.shape[0], recons.shape[0])
    x, z, r = originals[:n], latents[:n], recons[:n]
    x_rgb = feature_pca.project(x)
    r_rgb = feature_pca.project(r)
    z_rgb = latent_pca.project(z)

    # Project each map only once. All feature-space rows use the original's
    # color range so their colors remain directly comparable.
    x_lo = torch.quantile(x_rgb, 0.02)
    x_hi = torch.quantile(x_rgb, 0.98)
    z_lo = torch.quantile(z_rgb, 0.02)
    z_hi = torch.quantile(z_rgb, 0.98)

    def normalize(v: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
        return ((v - lo) / (hi - lo + 1e-8)).clamp(0.0, 1.0)

    rows_rgb = [
        normalize(x_rgb, x_lo, x_hi),
        normalize(z_rgb, z_lo, z_hi),
        normalize(r_rgb, x_lo, x_hi),
    ]
    preview_size = int(x.shape[-1])
    if gan_gt is not None:
        rows_rgb.append(_stylegan_preview(gan_gt[:n], preview_size).to(device=x.device))
    if gan_from_recon is not None:
        rows_rgb.append(_stylegan_preview(gan_from_recon[:n], preview_size).to(device=x.device))

    tiles = torch.cat(rows_rgb, dim=0)
    _, _, h, w = tiles.shape
    rows, cols = len(rows_rgb), n
    canvas = torch.zeros(3, rows * h, cols * w, dtype=tiles.dtype)
    for i in range(rows * cols):
        rr, cc = divmod(i, cols)
        canvas[:, rr * h:(rr + 1) * h, cc * w:(cc + 1) * w] = tiles[i]
    arr = (canvas.permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


# ---------------------------------------------------------------------------
# EMA / schedules / crops
# ---------------------------------------------------------------------------

class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.ema = copy.deepcopy(model)
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.ema.eval()
        self._ema_params = tuple(self.ema.parameters())

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        one_minus_decay = 1.0 - self.decay
        for ema_param, model_param in zip(self._ema_params, model.parameters()):
            ema_param.lerp_(model_param.detach(), one_minus_decay)


def cosine_lr(step: int, total: int, base_lr: float, warmup: int) -> float:
    if step <= warmup:
        return base_lr * step / max(warmup, 1)
    t = (step - warmup) / max(total - warmup, 1)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(t, 1.0)))


def kl_weight_schedule(step: int, warmup_steps: int, max_kl: float) -> float:
    if step <= warmup_steps:
        return 0.0
    t = (step - warmup_steps) / max(warmup_steps, 1)
    return max_kl * min(1.0, t)


def save_saliency_mask_png(
    path: str,
    mask: torch.Tensor,
    scale: int = 2,
    overlay_rgb: Optional[torch.Tensor] = None,
) -> None:
    """Write a magma heatmap of the spatial saliency weights."""
    plane = mask.detach().float().cpu().reshape(mask.shape[-2], mask.shape[-1])
    # Log scale keeps the face falloff visible; linear max-norm hides it
    # because the compact T-zone carries half the mass.
    plane = torch.log1p(plane / plane.max().clamp_min(1e-8) * 24.0)
    plane = plane / plane.max().clamp_min(1e-8)
    lut = torch.tensor(
        [
            (0.001, 0.000, 0.014),
            (0.040, 0.028, 0.142),
            (0.163, 0.042, 0.247),
            (0.327, 0.058, 0.327),
            (0.492, 0.077, 0.333),
            (0.646, 0.116, 0.293),
            (0.775, 0.204, 0.228),
            (0.880, 0.317, 0.162),
            (0.954, 0.456, 0.110),
            (0.987, 0.618, 0.140),
            (0.987, 0.787, 0.282),
            (0.988, 0.998, 0.645),
        ],
        dtype=torch.float32,
    )
    idx = plane * (lut.shape[0] - 1)
    lo = idx.floor().long().clamp(0, lut.shape[0] - 2)
    t = (idx - lo.float()).unsqueeze(-1)
    rgb = (1.0 - t) * lut[lo] + t * lut[lo + 1]
    arr = (rgb.clamp(0, 1).numpy() * 255.0).astype(np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    if scale > 1:
        img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    os.makedirs(osp.dirname(path) or ".", exist_ok=True)
    img.save(path)
    if overlay_rgb is None:
        return
    faces = overlay_rgb.detach().float().cpu()
    if faces.ndim == 3:
        faces = faces.unsqueeze(0)
    preview = _stylegan_preview(faces, plane.shape[-1])
    heat = torch.from_numpy(arr).float().div_(255.0).permute(2, 0, 1)
    if heat.shape[-2:] != preview.shape[-2:]:
        heat = F.interpolate(heat.unsqueeze(0), size=preview.shape[-2:], mode="bilinear", align_corners=False)[0]
    tiles = []
    for i in range(preview.shape[0]):
        mix = (0.5 * preview[i] + 0.55 * heat).clamp(0.0, 1.0)
        tiles.append(mix)
    strip = torch.cat(tiles, dim=2)
    over = (strip.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    root, ext = osp.splitext(path)
    Image.fromarray(over, mode="RGB").save(f"{root}_overlay{ext or '.png'}")


def random_crop_pair(*tensors: torch.Tensor, crop: int) -> Tuple[torch.Tensor, ...]:
    _, _, h, w = tensors[0].shape
    if crop >= h or crop >= w:
        return tensors
    top = torch.randint(0, h - crop + 1, (1,)).item()
    left = torch.randint(0, w - crop + 1, (1,)).item()
    return tuple(t[:, :, top:top + crop, left:left + crop] for t in tensors)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    gan_weights: str = DEFAULT_GAN_WEIGHTS
    out_dir: str = "experiments/sdvae_pca_residual"
    batch_size: int = 2
    max_steps: int = 10_000
    lr: float = 1.5e-4
    weight_decay: float = 1e-4
    hidden_channels: int = 64
    num_blocks: int = 5
    use_context: bool = True
    use_grn: bool = False
    drop_path: float = 0.05
    truncation_psi: float = 1.0
    noise_mode: str = "const"
    log_every: int = 50
    ckpt_every: int = 500
    plot_every: int = 100
    plot_count: int = 4
    val_every: int = 200
    grad_clip: float = 1.0
    seed: int = 0
    resume: Optional[str] = None
    reset_steps: bool = False
    amp: bool = False
    skip_gan_load: bool = False
    # PCA calibration
    pca_calib_batches: int = 64
    pca_samples_per_batch: int = 4096
    pca_path: Optional[str] = None
    # Training efficiency
    crop_size: int = 128
    crops_per_feature: int = 4
    fullres_every: int = 100
    warmup_steps: int = 500
    det_warmup_steps: int = 2000
    max_kl_weight: float = 1e-3
    free_bits: float = 0.0
    ema_decay: float = 0.999
    # Phase 2
    phase: str = "feature"  # feature | finetune
    lpips_weight: float = 0.5
    rgb_weight: float = 0.1
    freeze_encoder: bool = True
    saliency: bool = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def calibrate_pca(
    extractor: StyleGANFeatureExtractor,
    device: torch.device,
    cfg: TrainConfig,
) -> FrozenRank4PCA:
    print(f"#. Calibrating rank-4 PCA ({cfg.pca_calib_batches} batches)...")
    calib = StreamingPCACalibrator(FEATURE_CHANNELS)
    for i in range(cfg.pca_calib_batches):
        if cfg.skip_gan_load:
            feats = torch.randn(
                cfg.batch_size, FEATURE_CHANNELS, FEATURE_RESOLUTION, FEATURE_RESOLUTION, device=device
            )
        else:
            feats = extractor.sample_features(cfg.batch_size, device)
        calib.update(feats, max_samples=cfg.pca_samples_per_batch)
        if (i + 1) % 16 == 0 or i == 0:
            print(f"  \\__calib batch {i+1}/{cfg.pca_calib_batches}, n={calib.n}")
    pca = calib.finalize(LATENT_CHANNELS)
    print(f"  \\__PCA fitted on {int(pca.n_samples.item())} vectors; "
          f"eigs={[round(float(v), 4) for v in pca.eigenvalues.tolist()]}")
    return pca


@torch.no_grad()
def save_recon_plot(
    path: str,
    codec: PCAResidualFeatureCodec,
    extractor: StyleGANFeatureExtractor,
    feature_pca: OnlineChannelPCA,
    latent_pca: OnlineChannelPCA,
    device: torch.device,
    n: int = 4,
    skip_gan_load: bool = False,
    ema: Optional[ModelEMA] = None,
    features: Optional[torch.Tensor] = None,
    gan_ws: Optional[torch.Tensor] = None,
    gan_img256: Optional[torch.Tensor] = None,
    gan_img1024: Optional[torch.Tensor] = None,
) -> None:
    model = ema.ema if ema is not None else codec
    was_training = codec.training
    model.eval()
    if features is not None:
        feats = features[:n]
    elif skip_gan_load:
        feats = torch.randn(n, FEATURE_CHANNELS, FEATURE_RESOLUTION, FEATURE_RESOLUTION, device=device)
    else:
        feats = extractor.sample_features(n, device).detach().clone()
    out = model(feats, sample_posterior=False, deterministic=True)
    recon = out["recon"]
    z = out["z"]
    feature_pca.update(feats)
    latent_pca.update(z)

    gan_gt = gan_from_recon = None
    can_continue = (
        extractor.keep_tail
        and gan_ws is not None
        and gan_img256 is not None
    )
    if can_continue:
        gan_from_recon = extractor.synthesize_from_features(
            recon.float(), gan_img256[:n].detach(), gan_ws[:n].detach()
        )
        gan_gt = gan_img1024[:n] if gan_img1024 is not None else None
        if gan_gt is None:
            gan_gt = extractor.synthesize_from_features(
                feats.float(), gan_img256[:n].detach(), gan_ws[:n].detach()
            )

    img = make_pca_viz_grid(
        feats, z, recon, feature_pca, latent_pca, n=n,
        gan_gt=gan_gt, gan_from_recon=gan_from_recon,
    )
    os.makedirs(osp.dirname(path) or ".", exist_ok=True)
    img.save(path, format="JPEG", quality=92)
    if was_training:
        codec.train()


def try_lpips(device: torch.device, spatial: bool = False):
    try:
        import lpips  # type: ignore
        return lpips.LPIPS(net="vgg", spatial=spatial).to(device).eval()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(cfg: TrainConfig) -> PCAResidualFeatureCodec:
    device = select_device()
    set_seed(cfg.seed, device)
    use_amp = bool(cfg.amp and device.type == "cuda")
    phase2 = cfg.phase == "finetune"

    os.makedirs(cfg.out_dir, exist_ok=True)
    plots_dir = osp.join(cfg.out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    with open(osp.join(cfg.out_dir, "args.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    print(f"#. Device: {device}")
    print(f"#. Output: {cfg.out_dir}")
    print(f"#. Phase: {cfg.phase}")

    print("#. Building StyleGAN feature extractor...")
    keep_tail = phase2 or (cfg.plot_every > 0 and not cfg.skip_gan_load)
    extractor = StyleGANFeatureExtractor(
        gan_weights=None if cfg.skip_gan_load else cfg.gan_weights,
        stop_resolution=STOP_RESOLUTION,
        noise_mode=cfg.noise_mode,
        truncation_psi=cfg.truncation_psi,
        device=device,
        load_weights=not cfg.skip_gan_load,
        keep_tail=keep_tail,
    )
    print(
        f"  \\__Features: [{extractor.feature_channels}, "
        f"{extractor.feature_resolution}, {extractor.feature_resolution}], "
        f"num_ws={extractor.G.num_ws}, keep_tail={keep_tail}"
    )

    # PCA: load or calibrate
    pca_path = cfg.pca_path or osp.join(cfg.out_dir, "rank4_pca.pt")
    if cfg.pca_path and osp.isfile(cfg.pca_path):
        print(f"#. Loading PCA from {cfg.pca_path}")
        pca = FrozenRank4PCA(FEATURE_CHANNELS, LATENT_CHANNELS).to(device)
        pca.load_compact(torch.load(cfg.pca_path, map_location=device))
    elif osp.isfile(pca_path) and cfg.resume:
        print(f"#. Loading PCA from {pca_path}")
        pca = FrozenRank4PCA(FEATURE_CHANNELS, LATENT_CHANNELS).to(device)
        pca.load_compact(torch.load(pca_path, map_location=device))
    else:
        pca = calibrate_pca(extractor, device, cfg).to(device)
        torch.save(pca.state_dict_compact(), pca_path)
        print(f"  \\__Saved {pca_path}")

    # Held-out PCA baseline metric, plus cached full-res StyleGAN previews.
    print("#. Computing held-out PCA baseline...")
    val_ws = val_img256 = val_img1024 = None
    n_val = max(cfg.plot_count, 4)
    with torch.no_grad():
        if cfg.skip_gan_load:
            val_feats = torch.randn(
                n_val, FEATURE_CHANNELS, FEATURE_RESOLUTION, FEATURE_RESOLUTION, device=device
            )
        else:
            g = torch.Generator(device="cpu")
            g.manual_seed(cfg.seed + 12345)
            z_val = torch.randn(n_val, Z_DIM, generator=g).to(device)
            if extractor.keep_tail:
                val_ws = extractor.map_z(z_val)
                val_feats, val_img256, val_ws = extractor.features_and_skip_from_ws(val_ws)
                print("  \\__Caching 1024 StyleGAN previews for plots...")
                val_img1024 = extractor.synthesize_from_features(val_feats, val_img256, val_ws)
            else:
                val_feats = extractor(z_val)
        pca_base = pca.reconstruction_mse(val_feats)
        print(f"  \\__Held-out PCA MSE: {float(pca_base):.6f}")

    print("#. Building PCAResidualFeatureCodec...")
    codec = PCAResidualFeatureCodec(
        pca=pca,
        hidden_channels=cfg.hidden_channels,
        num_blocks=cfg.num_blocks,
        use_context=cfg.use_context,
        use_grn=cfg.use_grn,
        drop_path=cfg.drop_path,
        residual_gate_init=1.0,
    ).to(device)
    print(f"  \\__Trainable parameters: {codec.num_parameters():,}")

    # Zero-init residual head sanity: equals PCA
    with torch.no_grad():
        out0 = codec(val_feats[:1], sample_posterior=False, deterministic=True)
        diff0 = F.mse_loss(out0["recon"], pca.reconstruct(val_feats[:1]))
        print(f"  \\__Zero-init residual vs PCA MSE: {float(diff0):.2e} (should be ~0)")

    if phase2 and cfg.freeze_encoder:
        for p in codec.encoder.parameters():
            p.requires_grad_(False)
        codec.enc_gate.requires_grad_(False)
        print("  \\__Encoder frozen for Phase-2 fine-tune")

    params = [p for p in codec.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.999))
    try:
        scaler = GradScaler("cuda", enabled=use_amp)
    except TypeError:
        scaler = GradScaler(enabled=use_amp)

    ema = ModelEMA(codec, decay=cfg.ema_decay)
    feature_pca = OnlineChannelPCA(FEATURE_CHANNELS, 3).to(device)
    latent_pca = OnlineChannelPCA(LATENT_CHANNELS, 3).to(device)
    lpips_fn = try_lpips(device, spatial=bool(cfg.saliency)) if phase2 else None
    if phase2 and lpips_fn is None:
        print("  \\__Warning: lpips not installed; Phase-2 uses RGB loss only")

    start_step = 0
    if cfg.resume:
        print(f"#. Resuming from {cfg.resume}")
        ckpt = torch.load(cfg.resume, map_location=device)
        codec.load_state_dict(ckpt["codec"], strict=False)
        if "ema" in ckpt:
            ema.ema.load_state_dict(ckpt["ema"], strict=False)
        if "feature_pca" in ckpt:
            feature_pca.load_state_dict(ckpt["feature_pca"])
        if "latent_pca" in ckpt:
            latent_pca.load_state_dict(ckpt["latent_pca"])
        if cfg.reset_steps:
            print(
                f"  \\__Reset steps (checkpoint was step {int(ckpt.get('step', 0))}); "
                "LR schedule and optimizer start fresh"
            )
        else:
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
            start_step = int(ckpt.get("step", 0))

    codec.train()
    t0 = time.time()
    last_log_time = time.perf_counter()
    running: Dict[str, torch.Tensor] = {}
    running_updates = 0
    best_nmse = float("inf")
    cached_feats: Optional[torch.Tensor] = None
    cached_pack: Dict[str, Optional[torch.Tensor]] = {
        "x": None, "img256": None, "ws": None
    }
    saliency_full = (
        face_saliency_mask(FEATURE_RESOLUTION, FEATURE_RESOLUTION, device=device)
        if cfg.saliency
        else None
    )
    if saliency_full is not None:
        mask_path = osp.join(cfg.out_dir, "saliency_mask.png")
        save_saliency_mask_png(mask_path, saliency_full, overlay_rgb=val_img1024)
        print(f"  \\__Face saliency mask enabled (T-zone >= 50% of spatial loss mass)")
        print(f"  \\__Saved {mask_path}")

    for step in range(start_step + 1, cfg.max_steps + 1):
        # LR schedule
        lr = cosine_lr(step, cfg.max_steps, cfg.lr, cfg.warmup_steps)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        deterministic = phase2 or (step <= cfg.det_warmup_steps)
        kl_w = 0.0 if deterministic else kl_weight_schedule(
            step, cfg.det_warmup_steps, cfg.max_kl_weight
        )

        # One loop step is one optimizer update. Reuse each expensive frozen
        # StyleGAN feature batch for several independently sampled crops.
        reuse_steps = 1 if phase2 else max(cfg.crops_per_feature, 1)
        refresh_features = cached_feats is None or (step - 1) % reuse_steps == 0
        if refresh_features:
            with torch.inference_mode():
                if cfg.skip_gan_load:
                    new_feats = torch.randn(
                        cfg.batch_size,
                        FEATURE_CHANNELS,
                        FEATURE_RESOLUTION,
                        FEATURE_RESOLUTION,
                        device=device,
                    )
                    cached_pack = {"x": new_feats, "img256": None, "ws": None}
                elif phase2:
                    cached_pack = extractor.sample_pack(cfg.batch_size, device)
                    new_feats = cached_pack["x"]
                else:
                    new_feats = extractor.sample_features(cfg.batch_size, device)
                    cached_pack = {"x": new_feats, "img256": None, "ws": None}
            assert new_feats is not None
            cached_feats = new_feats.detach().clone()
        feats = cached_feats

        use_full = phase2 or (
            cfg.fullres_every > 0 and step % cfg.fullres_every == 0
        )
        if use_full:
            x_in = feats
            saliency_in = saliency_full
        elif saliency_full is not None:
            x_in, saliency_in = random_crop_pair(feats, saliency_full, crop=cfg.crop_size)
        else:
            (x_in,) = random_crop_pair(feats, crop=cfg.crop_size)
            saliency_in = None

        optimizer.zero_grad(set_to_none=True)
        try:
            amp_ctx = autocast("cuda", enabled=use_amp)
        except TypeError:
            amp_ctx = autocast(enabled=use_amp)

        with amp_ctx:
            out = codec(
                x_in,
                sample_posterior=not deterministic,
                deterministic=deterministic,
            )
            recon = out["recon"]
            posterior = out["posterior"]
            loss, stats = feature_codec_loss(
                recon, x_in, posterior, codec.pca,
                z=out["z"], z_pca=out["z_pca"], mean_res=out["mean_res"],
                kl_weight=kl_w, free_bits=cfg.free_bits,
                deterministic=deterministic,
                return_stats_tensors=True,
                saliency=saliency_in if saliency_in is not None else False,
            )

            if (
                phase2
                and cached_pack["ws"] is not None
                and cached_pack["img256"] is not None
            ):
                # Phase 2 is deterministic and full-resolution, so reuse the
                # codec result above instead of running the codec twice.
                img_pred = extractor.synthesize_from_features(
                    recon,
                    cached_pack["img256"].detach(),
                    cached_pack["ws"].detach(),
                )
                with torch.inference_mode():
                    img_gt = extractor.synthesize_from_features(
                        feats, cached_pack["img256"], cached_pack["ws"]
                    )
                img_gt = img_gt.detach()
                img_w = None
                if saliency_full is not None:
                    img_w = F.interpolate(
                        saliency_full, size=img_pred.shape[-2:], mode="bilinear", align_corners=False
                    )
                    rgb = weighted_spatial_mean((img_pred - img_gt).abs(), img_w)
                else:
                    rgb = F.l1_loss(img_pred, img_gt)
                loss = loss + cfg.rgb_weight * rgb
                stats["rgb"] = rgb.detach()
                if lpips_fn is not None:
                    # LPIPS expects [-1,1]. Spatial maps are weighted with
                    # the same 1024-upsampled FFHQ saliency mask as RGB L1.
                    perc_map = lpips_fn(img_pred.clamp(-1, 1), img_gt.clamp(-1, 1))
                    if img_w is not None:
                        if perc_map.shape[-2:] != img_w.shape[-2:]:
                            perc_w = F.interpolate(
                                img_w, size=perc_map.shape[-2:], mode="bilinear", align_corners=False
                            )
                        else:
                            perc_w = img_w
                        perc = weighted_spatial_mean(perc_map, perc_w)
                    else:
                        perc = perc_map.mean()
                    loss = loss + cfg.lpips_weight * perc
                    stats["lpips"] = perc.detach()

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            optimizer.step()

        ema.update(codec)
        for k, v in stats.items():
            assert isinstance(v, torch.Tensor)
            running[k] = running.get(k, torch.zeros_like(v)) + v
        running_updates += 1

        if step % cfg.log_every == 0 or step == 1:
            keys = ("loss", "mse", "nmse", "rel_improve", "pca_mse", "kl_per_latent")
            zero = torch.zeros((), device=device)
            avg_values = torch.stack([
                running.get(k, zero) / max(running_updates, 1) for k in keys
            ]).cpu().tolist()
            avg = dict(zip(keys, avg_values))
            logged_updates = running_updates
            running = {}
            running_updates = 0
            now = time.perf_counter()
            interval = now - last_log_time
            last_log_time = now
            steps_per_second = logged_updates / max(interval, 1e-8)
            elapsed = time.time() - t0
            print(
                f"step {step:06d}/{cfg.max_steps}  "
                f"loss={avg['loss']:.4f}  mse={avg['mse']:.5f}  nmse={avg['nmse']:.4f}  "
                f"Δpca={avg['rel_improve']:+.3%}  kl/lat={avg['kl_per_latent']:.5f}  "
                f"det={int(deterministic)}  lr={lr:.2e}  "
                f"rate={steps_per_second:.2f} step/s  t={elapsed:.0f}s"
            )

        if cfg.val_every > 0 and (step % cfg.val_every == 0 or step == 1):
            ema.ema.eval()
            with torch.no_grad():
                vf = val_feats
                vout = ema.ema(vf, sample_posterior=False, deterministic=True)
                vm = eval_metrics(vout["recon"], vf, codec.pca, vout["posterior"])
            print(
                f"  \\__VAL ema  mse={vm['mse']:.5f}  pca_mse={vm['pca_mse']:.5f}  "
                f"nmse={vm['nmse']:.4f}  Δpca={vm['rel_improve']:+.3%}  cos={vm['cosine']:.4f}"
            )
            with open(osp.join(cfg.out_dir, "val_metrics.jsonl"), "a") as f:
                f.write(json.dumps({"step": step, **vm}) + "\n")
            if vm["nmse"] < best_nmse:
                best_nmse = vm["nmse"]
                torch.save(
                    {"step": step, "codec": ema.ema.state_dict(), "metrics": vm, "config": asdict(cfg)},
                    osp.join(cfg.out_dir, "best_ema.pt"),
                )

        if cfg.plot_every > 0 and (step % cfg.plot_every == 0 or step == 1):
            plot_path = osp.join(plots_dir, f"recon_step_{step:06d}.jpg")
            save_recon_plot(
                plot_path, codec, extractor, feature_pca, latent_pca, device,
                n=cfg.plot_count, skip_gan_load=cfg.skip_gan_load,
                features=val_feats,
                gan_ws=val_ws, gan_img256=val_img256, gan_img1024=val_img1024,
            )
            print(f"  \\__Plot {plot_path}")

        if step % cfg.ckpt_every == 0 or step == cfg.max_steps:
            ckpt_path = osp.join(cfg.out_dir, "checkpoint.pt")
            torch.save({
                "step": step,
                "codec": codec.state_dict(),
                "ema": ema.ema.state_dict(),
                "optimizer": optimizer.state_dict(),
                "pca": codec.pca.state_dict_compact(),
                "feature_pca": feature_pca.state_dict(),
                "latent_pca": latent_pca.state_dict(),
                "config": asdict(cfg),
                "best_nmse": best_nmse,
            }, ckpt_path)
            torch.save(ema.ema.state_dict(), osp.join(cfg.out_dir, "feature_codec_ema.pt"))
            print(f"  \\__Saved {ckpt_path}")

    print("#. Training finished.")
    if best_nmse < 1.0:
        print(f"#. Best EMA NMSE={best_nmse:.4f} (<1 means beat PCA)")
    else:
        print(f"#. Best EMA NMSE={best_nmse:.4f} (did not beat PCA yet)")
    return codec


# ---------------------------------------------------------------------------
# Smoke / unit checks
# ---------------------------------------------------------------------------

def smoke_test_pca_and_codec() -> None:
    device = torch.device("cpu")
    # Synthetic features with a clear rank-4 subspace + noise
    B, C, H, W = 2, FEATURE_CHANNELS, 64, 64
    basis = torch.randn(C, LATENT_CHANNELS)
    basis, _ = torch.linalg.qr(basis)
    codes = torch.randn(B, LATENT_CHANNELS, H, W)
    x = torch.einsum("ck,bkhw->bchw", basis, codes) + 0.01 * torch.randn(B, C, H, W)

    calib = StreamingPCACalibrator(C)
    calib.update(x, max_samples=8192)
    for _ in range(8):
        codes = torch.randn(B, LATENT_CHANNELS, H, W)
        xx = torch.einsum("ck,bkhw->bchw", basis, codes) + 0.01 * torch.randn(B, C, H, W)
        calib.update(xx, max_samples=4096)
    pca = calib.finalize(LATENT_CHANNELS)

    with torch.no_grad():
        recon = pca.reconstruct(x)
        mse = F.mse_loss(recon, x)
        assert mse < 0.05, f"PCA recon MSE too high: {mse}"
        z = pca.encode(x)
        assert z.shape[1] == LATENT_CHANNELS

    codec = PCAResidualFeatureCodec(
        pca, hidden_channels=32, num_blocks=2, use_context=False, use_grn=False, drop_path=0.0
    )
    with torch.no_grad():
        out = codec(x, sample_posterior=False, deterministic=True)
        diff = F.mse_loss(out["recon"], pca.reconstruct(x))
        assert float(diff) < 1e-6, f"Zero-init residual should equal PCA, got {diff}"
    print(f"[smoke] PCA OK  mse={float(mse):.5f}  zero-init Δ={float(diff):.2e}  params={codec.num_parameters():,}")

    # One train step
    opt = torch.optim.AdamW([p for p in codec.parameters() if p.requires_grad], lr=1e-3)
    codec.train()
    out = codec(x, sample_posterior=False, deterministic=True)
    loss, stats = feature_codec_loss(
        out["recon"], x, out["posterior"], pca,
        z=out["z"], z_pca=out["z_pca"], mean_res=out["mean_res"],
        kl_weight=0.0, deterministic=True,
    )
    loss.backward()
    opt.step()
    print(f"[smoke] Train step OK  loss={stats['loss']:.4f}  nmse={stats['nmse']:.4f}")


def smoke_test_extractor_layout() -> None:
    extractor = StyleGANFeatureExtractor(load_weights=False, keep_tail=False, device=torch.device("cpu"))
    assert extractor.G.num_ws == 14
    assert list(extractor.G.synthesis.block_resolutions) == [4, 8, 16, 32, 64, 128, 256]
    z = torch.randn(1, Z_DIM)
    feats = extractor(z)
    assert feats.shape == (1, FEATURE_CHANNELS, FEATURE_RESOLUTION, FEATURE_RESOLUTION)
    print(f"[smoke] Extractor OK  num_ws={extractor.G.num_ws}  feats={tuple(feats.shape)}")

    # Tail packing check with keep_tail (random init, no weights)
    full = StyleGANFeatureExtractor(load_weights=False, keep_tail=True, device=torch.device("cpu"))
    assert full.G.num_ws == 18
    assert full.tail_resolutions == [512, 1024]
    ws = full.map_z(torch.randn(1, Z_DIM))
    x, img, ws = full.features_and_skip_from_ws(ws)
    assert x.shape == (1, 128, 256, 256)
    assert img.shape == (1, 3, 256, 256)
    # Tail from GT features should run
    img1024 = full.synthesize_from_features(x, img, ws)
    assert img1024.shape == (1, 3, 1024, 1024)
    print(f"[smoke] Tail OK  ws={tuple(ws.shape)}  img1024={tuple(img1024.shape)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train PCA-residual StyleGAN feature codec")
    p.add_argument("--gan-weights", type=str, default=DEFAULT_GAN_WEIGHTS)
    p.add_argument("--out-dir", type=str, default="experiments/sdvae_pca_residual")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=10_000)
    p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--weight-decay", type=float, default=2e-4)
    p.add_argument("--hidden-channels", type=int, default=64)
    p.add_argument("--num-blocks", type=int, default=5)
    p.add_argument("--use-context", action="store_true", default=True)
    p.add_argument("--no-context", action="store_true")
    p.add_argument("--use-grn", action="store_true")
    p.add_argument("--drop-path", type=float, default=0.05)
    p.add_argument("--truncation-psi", type=float, default=1.0)
    p.add_argument("--noise-mode", type=str, default="const", choices=("const", "random", "none"))
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--ckpt-every", type=int, default=500)
    p.add_argument("--plot-every", type=int, default=100)
    p.add_argument("--val-every", type=int, default=200)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument(
        "--reset-steps",
        action="store_true",
        help="when resuming, restart the step counter, cosine LR, and optimizer "
        "(keeps model/EMA/PCA weights). Use for finetune or a fresh schedule.",
    )
    p.add_argument("--amp", action="store_true")
    p.add_argument("--skip-gan-load", action="store_true")
    p.add_argument("--pca-calib-batches", type=int, default=64)
    p.add_argument("--pca-path", type=str, default=None)
    p.add_argument("--crop-size", type=int, default=128)
    p.add_argument(
        "--crops-per-feature",
        type=int,
        default=2,
        help="optimizer steps that reuse each generated StyleGAN feature batch",
    )
    p.add_argument("--fullres-every", type=int, default=100)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--det-warmup-steps", type=int, default=2000)
    p.add_argument("--max-kl-weight", type=float, default=1e-4)
    p.add_argument("--free-bits", type=float, default=0.0)
    p.add_argument("--ema-decay", type=float, default=0.995)
    p.add_argument("--phase", type=str, default="feature", choices=("feature", "finetune"))
    p.add_argument("--lpips-weight", type=float, default=0.5)
    p.add_argument("--rgb-weight", type=float, default=0.1)
    p.add_argument("--no-freeze-encoder", action="store_true")
    p.add_argument(
        "--saliency",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="weight spatial reconstruction losses with an FFHQ face/T-zone saliency mask",
    )
    p.add_argument("--smoke", action="store_true")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    if args.smoke:
        smoke_test_pca_and_codec()
        smoke_test_extractor_layout()
        print("[smoke] All checks passed.")
        return

    cfg = TrainConfig(
        gan_weights=args.gan_weights,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden_channels=args.hidden_channels,
        num_blocks=args.num_blocks,
        use_context=not args.no_context,
        use_grn=args.use_grn,
        drop_path=args.drop_path,
        truncation_psi=args.truncation_psi,
        noise_mode=args.noise_mode,
        log_every=args.log_every,
        ckpt_every=args.ckpt_every,
        plot_every=args.plot_every,
        val_every=args.val_every,
        grad_clip=args.grad_clip,
        seed=args.seed,
        resume=args.resume,
        reset_steps=args.reset_steps,
        amp=args.amp,
        skip_gan_load=args.skip_gan_load,
        pca_calib_batches=args.pca_calib_batches,
        pca_path=args.pca_path,
        crop_size=args.crop_size,
        crops_per_feature=args.crops_per_feature,
        fullres_every=args.fullres_every,
        warmup_steps=args.warmup_steps,
        det_warmup_steps=args.det_warmup_steps,
        max_kl_weight=args.max_kl_weight,
        free_bits=args.free_bits,
        ema_decay=args.ema_decay,
        phase=args.phase,
        lpips_weight=args.lpips_weight,
        rgb_weight=args.rgb_weight,
        freeze_encoder=not args.no_freeze_encoder,
        saliency=args.saliency,
    )
    train(cfg)


if __name__ == "__main__":
    main()
