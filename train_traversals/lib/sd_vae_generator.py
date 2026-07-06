from pathlib import Path
import sys

import torch
from omegaconf import OmegaConf
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train_eqvae.ldm.models.autoencoder import AutoencoderKL


class SDVAEGenerator(nn.Module):
    """Frozen SD-VAE decoder with the generator API expected by TrainerPotential."""

    def __init__(
        self,
        config_path,
        ckpt_path,
        latent_channels=4,
        latent_size=32,
        scaling_factor=1.0,
    ):
        super().__init__()
        self.latent_channels = int(latent_channels)
        self.latent_size = int(latent_size)
        self.scaling_factor = float(scaling_factor)
        self.dim_z = self.latent_channels * self.latent_size * self.latent_size
        self.latent_size_flat = self.dim_z
        self.shift_in_w_space = False
        self.uses_vae_latent_shape = True

        config = OmegaConf.load(str(config_path))
        model_params = OmegaConf.to_container(config.model.params, resolve=True)
        # This wrapper only uses the decoder; avoid instantiating LPIPS/discriminator loss.
        model_params["lossconfig"] = {"target": "torch.nn.Identity"}
        self.vae = AutoencoderKL(**model_params)
        checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
        state_dict = {key: value for key, value in state_dict.items() if not key.startswith("loss.")}
        missing, unexpected = self.vae.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"SD-VAE missing keys: {len(missing)}")
        if unexpected:
            print(f"SD-VAE unexpected keys: {len(unexpected)}")

        self.vae.eval()
        self.requires_grad_(False)

    def forward(self, z):
        if z.ndim == 2:
            z = z.reshape(z.shape[0], self.latent_channels, self.latent_size, self.latent_size)
        elif z.ndim != 4:
            raise ValueError(f"Expected latent shape [B,{self.dim_z}] or [B,C,H,W], got {tuple(z.shape)}")

        z = z.to(dtype=next(self.vae.parameters()).dtype)
        if self.scaling_factor != 1.0:
            z = z / self.scaling_factor
        return self.vae.decode(z)


def load_sd_vae_generator(config_path, ckpt_path, scaling_factor=1.0):
    config_path = Path(config_path).expanduser().resolve()
    ckpt_path = Path(ckpt_path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"SD-VAE config not found: {config_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"SD-VAE checkpoint not found: {ckpt_path}")
    return SDVAEGenerator(config_path=config_path, ckpt_path=ckpt_path, scaling_factor=scaling_factor)
