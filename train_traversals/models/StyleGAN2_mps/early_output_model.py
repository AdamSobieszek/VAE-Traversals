# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.
"""StyleGAN2 b128/b256 prefix with a pointwise_style32 RGB head.

Load the trained model (public export or training checkpoint)::

    G = load_generator_checkpoint("sg128_pointwise_style32.pt", device="cuda")
    image = G(torch.randn(1, 512, device="cuda"), noise_mode="const")

For optimized inference, use this module's build_optimized_early_output_synthesis.
The shared optimized_synthesis.py requires no modifications.
"""
from __future__ import annotations

import copy
import os.path as osp
import pickle
import sys
from contextlib import nullcontext
from typing import Dict, Mapping, Optional

import torch
from torch import nn
from torch.nn import functional as F

from .model import Generator as FullStyleGAN2Generator, MappingNetwork, SynthesisBlock
from .torch_utils import misc

FORMAT_NAME = "stylegan2_early_output_generator_v2"
Z_DIM = W_DIM = 512
SOURCE_RESOLUTION = 1024
OUTPUT_RESOLUTION = 128
IMG_CHANNELS = 3
PREFIX_NUM_WS = 12
# (feature channels, prefix W count, first conditioning W slot).
RESOLUTION_SPECS = {128: (256, 12, 9), 256: (128, 14, 11)}
GENERATOR_CONFIG = dict(z_dim=512, c_dim=0, w_dim=512, img_resolution=128,
                        img_channels=3, decoder_hidden_channels=16,
                        decoder_architecture="pointwise_style", decoder_spatial_channels=32)
GENERATOR_CONFIGS = {r: dict(GENERATOR_CONFIG, img_resolution=r) for r in RESOLUTION_SPECS}


class PointwiseStyleDecoder(nn.Module):
    """Fused linear RGB + width-16 residual + width-32 W-conditioned correction.

    Keep trained tensor names (spatial_in/out) so the winning checkpoint loads
    directly. The conditioned branch contains only 1x1 convolutions.
    """
    def __init__(self, img_resolution=OUTPUT_RESOLUTION):
        super().__init__()
        channels, _, self.style_start = RESOLUTION_SPECS[img_resolution]
        self.linear = nn.Conv2d(channels, 3, 3, padding=1)
        self.mix_in = nn.Conv2d(channels, 16, 1)
        self.mix_out = nn.Conv2d(16, 3, 1)
        self.spatial_in = nn.Conv2d(channels, 32, 1)
        self.spatial_out = nn.Conv2d(32, 3, 1)
        self.style = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, 64),
                                   nn.SiLU(), nn.Linear(64, 64))

    def forward(self, features: torch.Tensor, ws: torch.Tensor) -> torch.Tensor:
        features = features.float()
        output = self.linear(features) + self.mix_out(F.silu(self.mix_in(features)))
        hidden = F.silu(self.spatial_in(features))
        scale, bias = self.style(ws[:, self.style_start:self.style_start + 2].flatten(1).float()).chunk(2, dim=1)
        hidden = hidden * (1 + scale[:, :, None, None]) + bias[:, :, None, None]
        return output + self.spatial_out(F.silu(hidden))


class SynthesisNetwork(nn.Module):
    """Original 1024 generator's convolution path through b128 or b256."""
    def __init__(self, img_resolution=OUTPUT_RESOLUTION):
        super().__init__()
        self.w_dim, self.num_ws = W_DIM, RESOLUTION_SPECS[img_resolution][1]
        self.img_resolution, self.img_channels = img_resolution, IMG_CHANNELS
        self.block_resolutions = [r for r in (4, 8, 16, 32, 64, 128, 256) if r <= img_resolution]
        channels = {r: min(32768 // r, 512) for r in self.block_resolutions}
        for resolution in self.block_resolutions:
            # Retain source toRGB tensors for strict checkpoint loading; they
            # are skipped during forward and frozen during finetuning.
            block = SynthesisBlock(
                channels[resolution // 2] if resolution > 4 else 0,
                channels[resolution], w_dim=W_DIM, resolution=resolution,
                img_channels=3, is_last=resolution == img_resolution,
                use_fp16=resolution >= 128,
            )
            setattr(self, f"b{resolution}", block)
        self.decoder = PointwiseStyleDecoder(img_resolution)

    def forward(self, ws: torch.Tensor, **block_kwargs) -> torch.Tensor:
        misc.assert_shape(ws, [None, self.num_ws, self.w_dim])
        ws = ws.float()
        features = None
        index = 0
        for resolution in self.block_resolutions:
            block = getattr(self, f"b{resolution}")
            current = ws.narrow(1, index, block.num_conv + block.num_torgb)
            features, _ = block(features, None, current, skip_torgb=True, **block_kwargs)
            index += block.num_conv
        return self.decoder(features, ws)


class Generator(nn.Module):
    """Pointwise_style32 generator; select img_resolution=128 (default) or 256."""
    def __init__(self, img_resolution=OUTPUT_RESOLUTION, **config):
        super().__init__()
        expected = GENERATOR_CONFIGS.get(img_resolution)
        if expected is None or any(key not in expected or value != expected[key]
               for key, value in config.items()):
            raise ValueError("Only 128/256 pointwise_style32 configurations are supported")
        self.z_dim, self.c_dim, self.w_dim = Z_DIM, 0, W_DIM
        self.img_resolution, self.img_channels = img_resolution, IMG_CHANNELS
        self.num_ws = RESOLUTION_SPECS[img_resolution][1]
        self.mapping = MappingNetwork(Z_DIM, 0, W_DIM, self.num_ws)
        self.synthesis = SynthesisNetwork(img_resolution)

    @property
    def device(self):
        return next(self.parameters()).device

    def get_latent(self, z, truncation_psi=1.0):
        return self.mapping(z, None, truncation_psi=truncation_psi)[:, 0]

    def mean_latent(self, count, truncation_psi=1.0):
        z = torch.randn(count, self.z_dim, device=self.device)
        return self.mapping(z, None, truncation_psi=truncation_psi).mean(0, keepdim=True)

    def forward(self, z, c=None, truncation_psi=1.0, truncation_cutoff=None,
                update_emas=False, **synthesis_kwargs):
        ws = self.mapping(z, c, truncation_psi=truncation_psi,
                          truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        return self.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)


class OptimizedPointwiseStyleSynthesis(nn.Module):
    """Compose the unmodified optimized prefix with this file's W-aware head."""
    def __init__(self, synthesis, cfg, copy_module=True):
        super().__init__()
        from .torch_utils.ops.optimized_synthesis import build_optimized_early_output_synthesis
        synthesis = copy.deepcopy(synthesis) if copy_module else synthesis
        self.decoder = synthesis.decoder
        # A private module registry lets the generic optimized code return
        # features through Identity, without replacing the caller's decoder.
        prefix = copy.copy(synthesis)
        prefix._modules = dict(synthesis._modules)
        prefix.decoder = nn.Identity()
        self.prefix = build_optimized_early_output_synthesis(prefix, cfg, copy_module=False)
        self.cfg = cfg

    def forward(self, ws, noise_mode="const", force_fp32=False):
        features = self.prefix(ws, noise_mode=noise_mode, force_fp32=force_fp32)
        context = (torch.autocast("cuda", dtype=self.cfg.low_precision_dtype)
                   if features.device.type == "cuda" and not force_fp32 else nullcontext())
        with context:
            return self.decoder(features, ws)


def build_optimized_early_output_synthesis(synthesis, cfg, *, copy_module=True):
    return OptimizedPointwiseStyleSynthesis(synthesis, cfg, copy_module).eval().requires_grad_(False)


def load_generator_checkpoint(path: str, device="cpu") -> Generator:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint.get("generator_config") or checkpoint.get("config", {})
    if config != GENERATOR_CONFIGS.get(config.get("img_resolution")) or "generator" not in checkpoint:
        raise ValueError("Expected a pointwise_style32 generator export or training checkpoint")
    model = Generator(**config)
    model.load_state_dict(checkpoint["generator"], strict=True)
    return model.eval().requires_grad_(False).to(device)


# The full 1024 teacher is needed only for continued distillation. These
# readers handle the repo's Rosinality weights and NVIDIA source checkpoints.


def _convert_rosinality_stylegan2_state_dict(
    state: Mapping[str, torch.Tensor], latent_avg: Optional[torch.Tensor] = None
) -> Dict[str, torch.Tensor]:
    converted: Dict[str, torch.Tensor] = {}

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
            f"{source}.conv.weight", f"{target}.weight", lambda value: value.squeeze(0)
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
            f"{source}.conv.weight", f"{target}.weight", lambda value: value.squeeze(0)
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


def _torch_load(path: str, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # pragma: no cover - older PyTorch
        return torch.load(path, map_location=map_location)


def load_ffhq_state_dict(path: str) -> Dict[str, torch.Tensor]:
    if not osp.isfile(path):
        raise FileNotFoundError(f"FFHQ StyleGAN checkpoint not found: {path!r}")
    if osp.splitext(path)[1].lower() == ".pkl":
        module_root = osp.dirname(__file__)
        if module_root not in sys.path:
            sys.path.insert(0, module_root)
        with open(path, "rb") as handle:
            checkpoint = pickle.load(handle)
        if not isinstance(checkpoint, dict) or "G_ema" not in checkpoint:
            raise KeyError("NVIDIA pickle must contain a G_ema generator")
        return checkpoint["G_ema"].state_dict()

    checkpoint = _torch_load(path, map_location="cpu")
    if isinstance(checkpoint, dict):
        for key in ("g_ema", "G_ema", "state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, Mapping):
                if "style.1.weight" in value:
                    return _convert_rosinality_stylegan2_state_dict(
                        value, latent_avg=checkpoint.get("latent_avg")
                    )
                return dict(value)
            if hasattr(value, "state_dict"):
                return value.state_dict()
        if "style.1.weight" in checkpoint:
            return _convert_rosinality_stylegan2_state_dict(
                checkpoint, latent_avg=checkpoint.get("latent_avg")
            )
        if checkpoint and all(isinstance(key, str) for key in checkpoint):
            return checkpoint
    raise ValueError(f"Unrecognized StyleGAN checkpoint format: {path!r}")


def _load_full_source_generator(path: str) -> FullStyleGAN2Generator:
    source = FullStyleGAN2Generator(Z_DIM, 0, W_DIM, SOURCE_RESOLUTION, IMG_CHANNELS)
    missing, unexpected = source.load_state_dict(load_ffhq_state_dict(path), strict=False)
    substantive_missing = [
        key
        for key in missing
        if not key.startswith("synthesis.blocks.")
        and not key.endswith("resample_filter")
    ]
    if substantive_missing:
        raise RuntimeError(
            f"FFHQ checkpoint is missing {len(substantive_missing)} required keys; "
            f"first: {substantive_missing[:5]}"
        )
    if unexpected:
        raise RuntimeError(
            f"FFHQ checkpoint has {len(unexpected)} unexpected keys; first: {unexpected[:5]}"
        )
    source.eval()
    for parameter in source.parameters():
        parameter.requires_grad_(False)
    return source
