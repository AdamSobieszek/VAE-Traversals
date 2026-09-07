# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.
# Modifications implement a learned early RGB output at StyleGAN2 b128/b256.

"""StyleGAN2 generator whose high-resolution tail is replaced by an RGB decoder.

The mapping network and synthesis blocks through ``b128`` or ``b256`` use the
classes and fast operators from :mod:`models.StyleGAN2_mps.model`.  RGB skip
outputs are not evaluated while producing the prefix activation.  Instead,
the captured activation is decoded by the trained early-output network
embedded in this file.

Convert an FFHQ generator and a trained decoder into one checkpoint::

    python -m models.StyleGAN2_mps.early_output_model \
        --ffhq-weights /path/to/stylegan2-ffhq-1024x1024.pkl \
        --decoder-weights experiments/stylegan_early_output_fast_v2/checkpoint.pt \
        --output experiments/stylegan_early_output_fast_v2/generator.pt

Load and run the converted generator::

    G = load_generator_checkpoint("generator.pt", device="cuda")
    image = G(torch.randn(1, G.z_dim, device="cuda"))
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import pickle
import sys
from dataclasses import asdict, dataclass
from typing import Dict, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    from .model import (
        Generator as FullStyleGAN2Generator,
        MappingNetwork,
        SynthesisBlock,
    )
    from .torch_utils import misc, persistence
except ImportError:  # pragma: no cover - permits direct execution from train_traversals
    from models.StyleGAN2_mps.model import (
        Generator as FullStyleGAN2Generator,
        MappingNetwork,
        SynthesisBlock,
    )
    from models.StyleGAN2_mps.torch_utils import misc, persistence


FORMAT_NAME = "stylegan2_early_output_generator_v2"
Z_DIM = 512
W_DIM = 512
SOURCE_RESOLUTION = 1024
OUTPUT_RESOLUTION = 256
IMG_CHANNELS = 3
SUPPORTED_OUTPUT_RESOLUTIONS = (128, 256)
PREFIX_NUM_WS_BY_RESOLUTION = {128: 12, 256: 14}
# Retained for callers that imported the original b256 constant.
PREFIX_NUM_WS = PREFIX_NUM_WS_BY_RESOLUTION[OUTPUT_RESOLUTION]


# ---------------------------------------------------------------------------
# Fused deployment decoder (state-compatible with StyleGAN_SkipNetwork.py)
# ---------------------------------------------------------------------------


@torch.no_grad()
def _fold_average_w_torgb(
    to_rgb: nn.Module, average_w: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Turn the fixed average-W toRGB into an ordinary 1x1 convolution."""
    average_w = average_w.detach().to(
        device=to_rgb.weight.device,
        dtype=to_rgb.weight.dtype,
    ).reshape(1, -1)
    style = to_rgb.affine(average_w).reshape(-1) * float(to_rgb.weight_gain)
    weight = to_rgb.weight.detach() * style.reshape(1, -1, 1, 1)
    return weight.contiguous(), to_rgb.bias.detach().reshape(-1).contiguous()


class FusedFastEarlyOutputDecoder(nn.Module):
    """Original fused head with an optional W-conditioned correction branch.

    ``pointwise_style`` omits the correction branch's 3x3 convolution. The
    original fused toRGB/spatial linear convolution remains in all three heads.
    ``spatial_*`` module names preserve existing spatial_style checkpoints.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 16,
        baseline_weight: Optional[torch.Tensor] = None,
        baseline_bias: Optional[torch.Tensor] = None,
        architecture: str = "pointwise",
        spatial_channels: int = 32,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        if architecture not in ("pointwise", "pointwise_style", "spatial_style"):
            raise ValueError(f"Unknown decoder architecture: {architecture}")
        self.architecture = architecture
        self.spatial_channels = int(spatial_channels)
        self.linear = nn.Conv2d(in_channels, IMG_CHANNELS, kernel_size=3, padding=1)
        self.mix_in = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.activation = nn.SiLU(inplace=True)
        self.mix_out = nn.Conv2d(hidden_channels, IMG_CHANNELS, kernel_size=1)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        nn.init.zeros_(self.mix_out.weight)
        nn.init.zeros_(self.mix_out.bias)
        if architecture != "pointwise":
            self.spatial_in = nn.Conv2d(in_channels, spatial_channels, 1)
            self.spatial_conv = (
                nn.Conv2d(spatial_channels, spatial_channels, 3, padding=1)
                if architecture == "spatial_style" else nn.Identity()
            )
            self.spatial_out = nn.Conv2d(spatial_channels, IMG_CHANNELS, 1)
            nn.init.zeros_(self.spatial_out.weight)
            nn.init.zeros_(self.spatial_out.bias)
            # W9/W10 encode the repeated odd/even tail styles for the supported
            # augmentations. LayerNorm has no running state or batch dependence.
            self.style = nn.Sequential(
                nn.LayerNorm(2 * W_DIM), nn.Linear(2 * W_DIM, 64), nn.SiLU(),
                nn.Linear(64, 2 * spatial_channels),
            )
            nn.init.zeros_(self.style[-1].weight)
            nn.init.zeros_(self.style[-1].bias)
        if baseline_weight is not None:
            center = self.linear.weight.shape[-1] // 2
            with torch.no_grad():
                self.linear.weight[:, :, center, center].copy_(
                    baseline_weight[:, :, 0, 0]
                )
                self.linear.bias.copy_(baseline_bias)

    def forward(self, features: torch.Tensor, ws: Optional[torch.Tensor] = None) -> torch.Tensor:
        features = features.float()
        output = self.linear(features) + self.mix_out(
            self.activation(self.mix_in(features))
        )
        if self.architecture != "pointwise":
            hidden = torch.nn.functional.silu(self.spatial_in(features))
            if ws is None or ws.shape[1] < 11:
                raise ValueError("Style-conditioned decoder requires W slots 9 and 10")
            scale, bias = self.style(ws[:, 9:11].flatten(1).float()).chunk(2, dim=1)
            hidden = hidden * (1 + scale[:, :, None, None]) + bias[:, :, None, None]
            output = output + self.spatial_out(
                torch.nn.functional.silu(self.spatial_conv(hidden))
            )
        return output


# ---------------------------------------------------------------------------
# StyleGAN prefix synthesis and integrated Generator
# ---------------------------------------------------------------------------

@persistence.persistent_class
class SynthesisNetwork(nn.Module):
    """StyleGAN2 prefix synthesis followed by ``EarlyOutputDecoder``."""

    def __init__(
        self,
        w_dim: int,
        img_resolution: int = OUTPUT_RESOLUTION,
        img_channels: int = IMG_CHANNELS,
        channel_base: int = 32768,
        channel_max: int = 512,
        source_img_resolution: int = SOURCE_RESOLUTION,
        source_num_fp16_res: int = 4,
        decoder_hidden_channels: int = 16,
        decoder_architecture: str = "pointwise",
        decoder_spatial_channels: int = 32,
        average_w: Optional[torch.Tensor] = None,
        **block_kwargs,
    ):
        super().__init__()
        if img_resolution not in SUPPORTED_OUTPUT_RESOLUTIONS:
            raise ValueError(
                "Early-output synthesis resolution must be one of "
                f"{SUPPORTED_OUTPUT_RESOLUTIONS}, got {img_resolution}"
            )
        if img_channels != IMG_CHANNELS:
            raise ValueError(f"Early-output synthesis must have {IMG_CHANNELS} image channels")
        self.w_dim = int(w_dim)
        self.img_resolution = int(img_resolution)
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = int(img_channels)
        self.block_resolutions = [
            2**index for index in range(2, self.img_resolution_log2 + 1)
        ]
        channels = {
            resolution: min(channel_base // resolution, channel_max)
            for resolution in self.block_resolutions
        }

        # Match the precision assignment of the original 1024 generator.  A
        # standalone 256 calculation would otherwise start FP16 too early.
        source_log2 = int(np.log2(source_img_resolution))
        fp16_resolution = max(2 ** (source_log2 + 1 - source_num_fp16_res), 8)
        self.num_ws = 0
        for resolution in self.block_resolutions:
            in_channels = channels[resolution // 2] if resolution > 4 else 0
            out_channels = channels[resolution]
            block = SynthesisBlock(
                in_channels,
                out_channels,
                w_dim=w_dim,
                resolution=resolution,
                img_channels=img_channels,
                is_last=resolution == img_resolution,
                use_fp16=resolution >= fp16_resolution,
                **block_kwargs,
            )
            self.num_ws += block.num_conv
            if resolution == img_resolution:
                self.num_ws += block.num_torgb
            setattr(self, f"b{resolution}", block)

        expected_num_ws = PREFIX_NUM_WS_BY_RESOLUTION[self.img_resolution]
        if self.num_ws != expected_num_ws:
            raise RuntimeError(
                f"Expected {expected_num_ws} prefix W slots for b{self.img_resolution}, "
                f"got {self.num_ws}"
            )
        if average_w is None:
            average_w = torch.zeros(w_dim)
        to_rgb = getattr(self, f"b{self.img_resolution}").torgb
        baseline_weight, baseline_bias = _fold_average_w_torgb(to_rgb, average_w)
        self.decoder = FusedFastEarlyOutputDecoder(
            to_rgb.in_channels,
            hidden_channels=decoder_hidden_channels,
            baseline_weight=baseline_weight,
            baseline_bias=baseline_bias,
            architecture=decoder_architecture,
            spatial_channels=decoder_spatial_channels,
        )

    def forward(
        self,
        ws: torch.Tensor,
        return_components: bool = False,
        **block_kwargs,
    ):
        with misc.record_function("split_ws"):
            misc.assert_shape(ws, [None, self.num_ws, self.w_dim])
            ws = ws.to(torch.float32)
            block_ws = []
            w_index = 0
            for resolution in self.block_resolutions:
                block = getattr(self, f"b{resolution}")
                block_ws.append(
                    ws.narrow(1, w_index, block.num_conv + block.num_torgb)
                )
                w_index += block.num_conv

        features = None
        for resolution, current_ws in zip(self.block_resolutions, block_ws):
            block = getattr(self, f"b{resolution}")
            features, _ = block(
                features,
                None,
                current_ws,
                skip_torgb=True,
                **block_kwargs,
            )
        if features is None:
            raise RuntimeError("StyleGAN prefix produced no features")
        output = self.decoder(features.float(), ws)
        if return_components:
            return {"prediction": output, "features": features}
        return output

    def extra_repr(self):
        return (
            f"w_dim={self.w_dim}, num_ws={self.num_ws}, "
            f"img_resolution={self.img_resolution}, img_channels={self.img_channels}"
        )


@persistence.persistent_class
class Generator(nn.Module):
    """Drop-in StyleGAN2 generator producing learned 128 or 256 RGB."""

    def __init__(
        self,
        z_dim: int = Z_DIM,
        c_dim: int = 0,
        w_dim: int = W_DIM,
        img_resolution: int = OUTPUT_RESOLUTION,
        img_channels: int = IMG_CHANNELS,
        mapping_kwargs: Optional[dict] = None,
        **synthesis_kwargs,
    ):
        super().__init__()
        self.z_dim = int(z_dim)
        self.c_dim = int(c_dim)
        self.w_dim = int(w_dim)
        self.img_resolution = int(img_resolution)
        self.img_channels = int(img_channels)
        mapping_kwargs = {} if mapping_kwargs is None else dict(mapping_kwargs)

        # Mapping is built first so its average W initializes the embedded
        # fixed-style baseline. Converted decoder state subsequently restores
        # the exact training-time value.
        if self.img_resolution not in SUPPORTED_OUTPUT_RESOLUTIONS:
            raise ValueError(
                "Early-output generator resolution must be one of "
                f"{SUPPORTED_OUTPUT_RESOLUTIONS}, got {self.img_resolution}"
            )
        self.num_ws = PREFIX_NUM_WS_BY_RESOLUTION[self.img_resolution]
        self.mapping = MappingNetwork(
            z_dim=z_dim,
            c_dim=c_dim,
            w_dim=w_dim,
            num_ws=self.num_ws,
            **mapping_kwargs,
        )
        self.synthesis = SynthesisNetwork(
            w_dim=w_dim,
            img_resolution=img_resolution,
            img_channels=img_channels,
            average_w=self.mapping.w_avg,
            **synthesis_kwargs,
        )
        if self.synthesis.num_ws != self.num_ws:
            raise RuntimeError("Mapping and synthesis disagree on num_ws")

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def get_latent(self, z: torch.Tensor, truncation_psi: float = 1.0) -> torch.Tensor:
        return self.mapping(z, None, truncation_psi=truncation_psi)[:, 0]

    def mean_latent(self, count: int, truncation_psi: float = 1.0) -> torch.Tensor:
        latent_in = torch.randn(count, self.z_dim, device=self.device)
        return self.mapping(
            latent_in, None, truncation_psi=truncation_psi
        ).mean(0, keepdim=True)

    def forward(
        self,
        z: torch.Tensor,
        c: Optional[torch.Tensor] = None,
        truncation_psi: float = 1.0,
        truncation_cutoff: Optional[int] = None,
        update_emas: bool = False,
        **synthesis_kwargs,
    ):
        ws = self.mapping(
            z,
            c,
            truncation_psi=truncation_psi,
            truncation_cutoff=truncation_cutoff,
            update_emas=update_emas,
        )
        return self.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)

    @classmethod
    def from_checkpoint(
        cls, path: str, device: torch.device | str = "cpu"
    ) -> "Generator":
        return load_generator_checkpoint(path, device=device)


# ---------------------------------------------------------------------------
# Source checkpoint loading and conversion
# ---------------------------------------------------------------------------

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


def _strip_module_prefix(state: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if state and all(key.startswith("module.") for key in state):
        return {key[len("module.") :]: value for key, value in state.items()}
    return dict(state)


@dataclass
class DecoderConfig:
    hidden_channels: int = 16
    train_baseline: bool = False
    output_resolution: int = OUTPUT_RESOLUTION


def _infer_decoder_config(
    state: Mapping[str, torch.Tensor], saved_config: Optional[Mapping] = None,
    saved_architecture: Optional[str] = None,
) -> DecoderConfig:
    saved_config = {} if saved_config is None else saved_config
    mix_in = state.get("refiner.mix_in.weight")
    if mix_in is None:
        raise ValueError(
            "Decoder checkpoint is not the fast v2 architecture; legacy decoders "
            "are no longer supported"
        )
    input_channels = int(mix_in.shape[1])
    resolution_by_channels = {
        min(32768 // resolution, 512): resolution
        for resolution in SUPPORTED_OUTPUT_RESOLUTIONS
    }
    state_resolution = resolution_by_channels.get(input_channels)
    if state_resolution is None:
        raise ValueError(
            f"Unsupported decoder input width {input_channels}; expected one of "
            f"{sorted(resolution_by_channels)} for b128/b256"
        )

    declared_resolution = saved_config.get(
        "output_resolution", saved_config.get("stop_resolution")
    )
    architecture_resolutions = {
        "stylegan_early_output_v2": 256,
        "stylegan_early_output_128_v2": 128,
    }
    architecture_resolution = architecture_resolutions.get(saved_architecture)
    for label, resolution in (
        ("saved config", declared_resolution),
        ("checkpoint architecture", architecture_resolution),
    ):
        if resolution is not None and int(resolution) != state_resolution:
            raise ValueError(
                f"Decoder {label} declares b{resolution}, but its {input_channels} "
                f"input channels identify b{state_resolution}"
            )

    return DecoderConfig(
        hidden_channels=int(saved_config.get("hidden_channels", mix_in.shape[0])),
        train_baseline=bool(saved_config.get("train_baseline", False)),
        output_resolution=state_resolution,
    )


def load_decoder_checkpoint(
    path: str, state_choice: str = "auto"
) -> Tuple[Dict[str, torch.Tensor], DecoderConfig, str]:
    """Load a training checkpoint, best EMA checkpoint, or raw decoder state."""
    if state_choice not in ("auto", "ema", "model"):
        raise ValueError("state_choice must be auto, ema, or model")
    if not osp.isfile(path):
        raise FileNotFoundError(f"Decoder checkpoint not found: {path!r}")
    checkpoint = _torch_load(path, map_location="cpu")
    saved_config: Mapping = {}
    selected = "raw"
    if isinstance(checkpoint, Mapping):
        saved_config = checkpoint.get("config", {})
        if state_choice in ("auto", "ema") and isinstance(checkpoint.get("ema"), Mapping):
            state = checkpoint["ema"]
            selected = "ema"
        elif isinstance(checkpoint.get("decoder"), Mapping):
            state = checkpoint["decoder"]
            selected = "decoder"
        elif state_choice == "ema":
            raise KeyError("Requested EMA state, but checkpoint has no 'ema' entry")
        elif checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
            state = checkpoint
        else:
            raise ValueError("Could not locate a decoder state dictionary in checkpoint")
    else:
        raise ValueError("Decoder checkpoint must contain a state dictionary")
    state = _strip_module_prefix(state)
    saved_architecture = checkpoint.get("architecture")
    config = _infer_decoder_config(state, saved_config, saved_architecture)
    return state, config, selected


def _validate_decoder_source_match(
    decoder_state: Mapping[str, torch.Tensor],
    decoder_config: DecoderConfig,
    source: FullStyleGAN2Generator,
    allow_mismatch: bool,
) -> None:
    mismatches = []
    if decoder_config.train_baseline:
        print("  \\__Baseline source equality check skipped: baseline was fine-tuned")
    else:
        expected_weight, expected_bias = _fold_average_w_torgb(
            getattr(
                source.synthesis, f"b{decoder_config.output_resolution}"
            ).torgb,
            source.mapping.w_avg,
        )
        for key, expected in (
            ("baseline.weight", expected_weight),
            ("baseline.bias", expected_bias),
        ):
            actual = decoder_state.get(key)
            if actual is None or not torch.allclose(
                actual.detach().cpu(), expected.detach().cpu(), rtol=1e-5, atol=1e-6
            ):
                mismatches.append(key)
    if mismatches and not allow_mismatch:
        raise ValueError(
            "Decoder baseline does not match the supplied FFHQ generator "
            f"({len(mismatches)} mismatches; first: {mismatches[:5]}). "
            "Use the same FFHQ weights used for training, or pass "
            "--allow-baseline-mismatch deliberately."
        )
    if mismatches:
        print(f"  \\__Warning: accepted {len(mismatches)} decoder/source mismatches")


def convert_checkpoints(
    ffhq_weights: str,
    decoder_weights: str,
    output_path: str,
    *,
    state_choice: str = "auto",
    allow_baseline_mismatch: bool = False,
    verify: bool = True,
) -> dict:
    """Merge frozen StyleGAN prefix and trained decoder into one checkpoint."""
    print(f"#. Loading FFHQ generator: {ffhq_weights}")
    source = _load_full_source_generator(ffhq_weights)
    print(f"#. Loading decoder: {decoder_weights}")
    decoder_state, decoder_config, selected_state = load_decoder_checkpoint(
        decoder_weights, state_choice=state_choice
    )
    print(f"  \\__Selected decoder state: {selected_state}")
    print(f"  \\__Decoder config: {json.dumps(asdict(decoder_config), sort_keys=True)}")
    _validate_decoder_source_match(
        decoder_state,
        decoder_config,
        source,
        allow_mismatch=allow_baseline_mismatch,
    )

    generator = Generator(
        z_dim=source.z_dim,
        c_dim=source.c_dim,
        w_dim=source.w_dim,
        img_resolution=decoder_config.output_resolution,
        img_channels=source.img_channels,
        decoder_hidden_channels=decoder_config.hidden_channels,
    )
    generator.mapping.load_state_dict(source.mapping.state_dict(), strict=True)
    for resolution in generator.synthesis.block_resolutions:
        target_block = getattr(generator.synthesis, f"b{resolution}")
        source_block = getattr(source.synthesis, f"b{resolution}")
        target_block.load_state_dict(source_block.state_dict(), strict=True)
    from models.StyleGAN_SkipNetwork import (
        EarlyOutputDecoder as TrainingEarlyOutputDecoder,
        FrozenToRGBBaseline as TrainingFrozenToRGBBaseline,
    )

    training_decoder = TrainingEarlyOutputDecoder(
        TrainingFrozenToRGBBaseline(
            getattr(
                source.synthesis, f"b{decoder_config.output_resolution}"
            ).torgb,
            source.mapping.w_avg,
            trainable=decoder_config.train_baseline,
        ),
        hidden_channels=decoder_config.hidden_channels,
    )
    training_decoder.load_state_dict(decoder_state, strict=True)
    deployed_decoder = training_decoder.eval().to_deploy()
    generator.synthesis.decoder.load_state_dict(
        deployed_decoder.state_dict(), strict=True
    )
    generator.eval()

    config = {
        "z_dim": generator.z_dim,
        "c_dim": generator.c_dim,
        "w_dim": generator.w_dim,
        "img_resolution": generator.img_resolution,
        "img_channels": generator.img_channels,
        "decoder_hidden_channels": decoder_config.hidden_channels,
    }
    checkpoint = {
        "format": FORMAT_NAME,
        "generator": generator.state_dict(),
        "config": config,
        "source": {
            "ffhq_weights": osp.basename(ffhq_weights),
            "decoder_weights": osp.basename(decoder_weights),
            "decoder_state": selected_state,
            "train_baseline": decoder_config.train_baseline,
            "output_resolution": decoder_config.output_resolution,
        },
    }

    output_directory = osp.dirname(osp.abspath(output_path))
    os.makedirs(output_directory, exist_ok=True)
    torch.save(checkpoint, output_path)
    print(f"#. Saved integrated generator: {output_path}")

    if verify:
        print("#. Verifying converted generator...")
        with torch.inference_mode():
            z = torch.randn(1, generator.z_dim)
            image = generator(z, noise_mode="const", force_fp32=True)
        expected_shape = (
            1,
            IMG_CHANNELS,
            decoder_config.output_resolution,
            decoder_config.output_resolution,
        )
        if image.shape != expected_shape:
            raise RuntimeError(f"Converted generator produced unexpected shape {tuple(image.shape)}")
        if not torch.isfinite(image).all():
            raise RuntimeError("Converted generator produced non-finite values")
        reloaded = load_generator_checkpoint(output_path, device="cpu")
        with torch.inference_mode():
            reloaded_image = reloaded(z, noise_mode="const", force_fp32=True)
        if not torch.equal(image, reloaded_image):
            max_difference = float((image - reloaded_image).abs().max())
            raise RuntimeError(
                f"Reloaded generator changed output (max difference {max_difference:.3e})"
            )
        print(f"  \\__Verified output shape {tuple(image.shape)} and exact reload")
    return checkpoint


def load_generator_checkpoint(
    path: str, device: torch.device | str = "cpu"
) -> Generator:
    checkpoint = _torch_load(path, map_location="cpu")
    checkpoint_format = checkpoint.get("format") if isinstance(checkpoint, Mapping) else None
    if checkpoint_format != FORMAT_NAME:
        raise ValueError(f"Not a {FORMAT_NAME} checkpoint: {path!r}")
    config = dict(checkpoint["config"])
    generator = Generator(**config)
    generator.load_state_dict(checkpoint["generator"], strict=True)
    generator.eval().requires_grad_(False)
    return generator.to(device)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge FFHQ StyleGAN2 and an early-output decoder checkpoint"
    )
    parser.add_argument("--ffhq-weights", default=None)
    parser.add_argument("--decoder-weights", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--decoder-state",
        choices=("auto", "ema", "model"),
        default="auto",
        help="auto prefers EMA, then decoder; model selects the decoder entry",
    )
    parser.add_argument(
        "--allow-baseline-mismatch",
        action="store_true",
        help="allow decoder toRGB/average-W from a different FFHQ checkpoint",
    )
    parser.add_argument(
        "--verify",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="run a CPU forward and exact save/reload comparison",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    _TRAVERSALS_MODELS_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))

    if args.ffhq_weights is None:
        args.ffhq_weights = osp.join(
            _TRAVERSALS_MODELS_DIR,
            "pretrained",
            "generators",
            "StyleGAN2",
            "stylegan2-ffhq-config-f.pt",
        )
    if args.output is None:
        _, decoder_config, _ = load_decoder_checkpoint(
            args.decoder_weights, state_choice=args.decoder_state
        )
        resolution_suffix = (
            "" if decoder_config.output_resolution == OUTPUT_RESOLUTION else "-128"
        )
        args.output = osp.join(
            _TRAVERSALS_MODELS_DIR,
            "pretrained",
            "generators",
            "StyleGAN2",
            f"stylegan2-ffhq-config-f-early-output{resolution_suffix}.pt",
        )
    convert_checkpoints(
        args.ffhq_weights,
        args.decoder_weights,
        args.output,
        state_choice=args.decoder_state,
        allow_baseline_mismatch=args.allow_baseline_mismatch,
        verify=args.verify,
    )


if __name__ == "__main__":
    main()
