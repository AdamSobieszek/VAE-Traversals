import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from models.BigGAN import BigGAN, utils
from models.ProgGAN.model import Generator as ProgGANGenerator
from models.SNGAN.sn_gen_resnet import SN_RES_GEN_CONFIGS, make_resnet_generator
from models.SNGAN.distribution import NormalDistribution

REPO_ROOT = Path(__file__).resolve().parents[2]
GAT_ROOT = REPO_ROOT / "GAT"


def _dtype_from_mixed_precision(mixed_precision):
    if mixed_precision in (None, "", "no", False):
        return torch.float32
    if mixed_precision == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported mixed precision mode: {mixed_precision!r}")



########################################################################################################################
##                                                                                                                    ##
##                                                     [ SNGAN ]                                                      ##
##                                                                                                                    ##
########################################################################################################################
class SNGANWrapper(nn.Module):
    def __init__(self, G):
        super(SNGANWrapper, self).__init__()
        self.G = G.model
        self.dim_z = G.distribution.dim
        self.shift_in_w_space = False

    def forward(self, z, shift=None):
        return self.G(z if shift is None else z + shift)


def build_sngan(pretrained_gan_weights, gan_type):
    # SNGAN configuration for MNIST and AnimeFaces datasets
    SNGAN_CONFIG = {
        'SNGAN_MNIST': {
            'image_channels': 1,
            'latent_dim': 128,
            'model': 'sn_resnet32',
            'img_size': 32
        },
        'SNGAN_AnimeFaces': {
            'image_channels': 3,
            'latent_dim': 128,
            'model': 'sn_resnet64',
            'img_size': 64
        }
    }

    # Build SNGAN generator (for the given dataset)
    G = make_resnet_generator(resnet_gen_config=SN_RES_GEN_CONFIGS[SNGAN_CONFIG[gan_type]['model']],
                              img_size=SNGAN_CONFIG[gan_type]['img_size'],
                              channels=SNGAN_CONFIG[gan_type]['image_channels'],
                              distribution=NormalDistribution(SNGAN_CONFIG[gan_type]['latent_dim']))

    # Load pre-trained weights
    G.load_state_dict(torch.load(pretrained_gan_weights, map_location=torch.device('cpu')), strict=False)

    return SNGANWrapper(G)


########################################################################################################################
##                                                                                                                    ##
##                                                   [ BigGAN ]                                                       ##
##                                                                                                                    ##
########################################################################################################################
class BigGANWrapper(nn.Module):
    def __init__(self, G, target_classes=(239, )):
        super(BigGANWrapper, self).__init__()
        self.G = G
        self.target_classes = nn.Parameter(data=torch.tensor(target_classes, dtype=torch.int64),
                                           requires_grad=False)
        self.dim_z = self.G.dim_z
        self.shift_in_w_space = False

    def mixed_classes(self, batch_size):
        if len(self.target_classes.data.shape) == 0:
            return self.target_classes.repeat(batch_size).cuda()
        else:
            return torch.from_numpy(np.random.choice(self.target_classes.cpu(), [batch_size])).cuda()

    def forward(self, z, shift=None):
        target_classes = self.mixed_classes(z.shape[0]).to(z.device)
        return self.G(z if shift is None else z + shift, self.G.shared(target_classes))


def build_biggan(pretrained_gan_weights, target_classes):
    # Get BigGAN configuration
    with open('models/BigGAN/generator_config.json') as f:
        config = json.load(f)

    # Build BigGAN generator for the given configuration
    config['resolution'] = utils.imsize_dict[config['dataset']]
    config['n_classes'] = utils.nclass_dict[config['dataset']]
    config['G_activation'] = utils.activation_dict[config['G_nl']]
    config['D_activation'] = utils.activation_dict[config['D_nl']]
    config['skip_init'] = True
    config['no_optim'] = True
    G = BigGAN.Generator(**config)

    # Load pre-trained weights
    G.load_state_dict(torch.load(pretrained_gan_weights, map_location=torch.device('cpu')), strict=True)

    return BigGANWrapper(G, target_classes)


########################################################################################################################
##                                                                                                                    ##
##                                                    [ ProgGAN ]                                                     ##
##                                                                                                                    ##
########################################################################################################################
class ProgGANWrapper(nn.Module):
    def __init__(self, G):
        super(ProgGANWrapper, self).__init__()
        self.G = G
        self.dim_z = 512
        self.shift_in_w_space = False

    @staticmethod
    def _reshape(z):
        return z.reshape(z.size()[0], z.size()[1], 1, 1)

    def forward(self, z, shift=None):
        return self.G(self._reshape(z) if shift is None else self._reshape(z + shift))


def build_proggan(pretrained_gan_weights):
    # Build ProgGAN generator model
    G = ProgGANGenerator()
    # Load pre-trained generator model
    G.load_state_dict(torch.load(pretrained_gan_weights, map_location='cpu'))

    return ProgGANWrapper(G)


########################################################################################################################
##                                                                                                                    ##
##                                                     [ GAT ]                                                        ##
##                                                                                                                    ##
########################################################################################################################
def _normalize_gat_model_name(name):
    return name.replace("SiT-", "GAT-", 1) if name.startswith("SiT-") else name


def _get_gat_checkpoint_state(checkpoint, weight_key):
    for key in (weight_key, "ema", "generator", "model"):
        state_dict = checkpoint.get(key) if isinstance(checkpoint, dict) else None
        if isinstance(state_dict, dict):
            return state_dict, key
    raise RuntimeError("GAT checkpoint does not contain generator weights.")


def _read_gat_checkpoint_settings(checkpoint):
    settings = {}
    if not isinstance(checkpoint, dict):
        return settings
    for key in ("args", "config"):
        ckpt_cfg = checkpoint.get(key)
        if ckpt_cfg is None:
            continue
        for name in ("model", "resolution", "num_classes", "fused_attn", "qk_norm"):
            if hasattr(ckpt_cfg, name):
                settings[name] = getattr(ckpt_cfg, name)
        if settings:
            break
    return settings


class GATWrapper(nn.Module):
    def __init__(
        self,
        G,
        resolution=256,
        target_classes=(239,),
        truncation_psi=0.3,
        vae_variant="ema",
        projector_embed_dims=(768, 1024),
        vae=None,
        mixed_precision="no",
    ):
        super(GATWrapper, self).__init__()
        self.G = G
        self.resolution = int(resolution)
        self.truncation_psi = float(truncation_psi)
        self.vae_variant = vae_variant
        self.projector_embed_dims = tuple(int(d) for d in projector_embed_dims)
        self.target_classes = nn.Parameter(
            data=torch.tensor(target_classes, dtype=torch.int64),
            requires_grad=False,
        )

        self.latent_channels = int(G.in_channels)
        self.latent_size = self.resolution // 8
        self.dim_z = int(G.latent_size)
        self.shift_in_w_space = False
        self.uses_vae_latent_shape = False

        self.vae = vae
        self.set_mixed_precision(mixed_precision)
        self.register_buffer(
            "latents_scale",
            torch.tensor([0.18215] * self.latent_channels).view(1, self.latent_channels, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "latents_bias",
            torch.zeros(1, self.latent_channels, 1, 1),
            persistent=False,
        )

    def set_mixed_precision(self, mixed_precision="no"):
        self.mixed_precision = mixed_precision or "no"
        self.compute_dtype = _dtype_from_mixed_precision(self.mixed_precision)
        self.G.to(dtype=self.compute_dtype)

    def mixed_classes(self, batch_size):
        if len(self.target_classes.data.shape) == 0:
            return self.target_classes.repeat(batch_size)
        return torch.from_numpy(np.random.choice(self.target_classes.cpu(), [batch_size]))

    def _ensure_vae(self, device):
        if self.vae is None:
            from diffusers.models import AutoencoderKL

            self.vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{self.vae_variant}")
        if next(self.vae.parameters()).device != device:
            self.vae = self.vae.to(device)
        self.vae.eval()

    def forward(self, z, shift=None):
        z = z if shift is None else z + shift
        z_gat = z.to(dtype=self.compute_dtype)
        batch_size = z.shape[0]
        x = torch.randn(
            batch_size,
            self.latent_channels,
            self.latent_size,
            self.latent_size,
            device=z.device,
            dtype=self.compute_dtype,
        )
        y = self.mixed_classes(batch_size).to(device=z.device)
        return self.G(x=x, y=y, z=z_gat, truncation_psi=self.truncation_psi)

    def decode_with_vae(self, latents):
        if latents.ndim == 2:
            latents = latents.reshape(
                latents.shape[0],
                self.latent_channels,
                self.latent_size,
                self.latent_size,
            )
        self._ensure_vae(latents.device)
        latents = latents.to(dtype=torch.float32)
        scale = self.latents_scale.to(device=latents.device, dtype=latents.dtype)
        bias = self.latents_bias.to(device=latents.device, dtype=latents.dtype)
        with torch.amp.autocast(device_type=latents.device.type, enabled=False):
            return self.vae.decode((latents - bias) / scale).sample


def build_gat(
    pretrained_gan_weights,
    target_classes=(239,),
    model_name=None,
    resolution=None,
    num_classes=None,
    weight_key="ema",
    vae_variant="ema",
    truncation_psi=0.3,
    projector_embed_dims=(768, 1024),
    fused_attn=True,
    qk_norm=True,
    legacy=False,
    encoder_depth=8,
    load_vae=False,
    mixed_precision="no",
):
    gat_root = str(GAT_ROOT)
    if gat_root not in sys.path:
        sys.path.insert(0, gat_root)
    from models.generator import GAT_models
    from utils import load_legacy_checkpoints

    checkpoint = torch.load(pretrained_gan_weights, map_location=torch.device("cpu"), weights_only=False)
    ckpt_settings = _read_gat_checkpoint_settings(checkpoint)

    model_name = _normalize_gat_model_name(model_name or ckpt_settings.get("model", "GAT-XL/2"))
    resolution = int(resolution or ckpt_settings.get("resolution", 256))
    num_classes = int(num_classes or ckpt_settings.get("num_classes", 1000))
    fused_attn = bool(ckpt_settings.get("fused_attn", fused_attn))
    qk_norm = bool(ckpt_settings.get("qk_norm", qk_norm))

    state_dict, _ = _get_gat_checkpoint_state(checkpoint, weight_key)
    if legacy:
        state_dict = load_legacy_checkpoints(state_dict, encoder_depth=encoder_depth)

    latent_spatial_size = resolution // 8
    G = GAT_models[model_name](
        input_size=latent_spatial_size,
        num_classes=num_classes,
        z_dims=[int(z_dim) for z_dim in projector_embed_dims],
        fused_attn=fused_attn,
        qk_norm=qk_norm,
    )
    G.load_state_dict(state_dict, strict=True)
    G.eval()

    vae = None
    if load_vae:
        from diffusers.models import AutoencoderKL

        vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{vae_variant}")
        vae.eval()

    return GATWrapper(
        G,
        resolution=resolution,
        target_classes=target_classes,
        truncation_psi=truncation_psi,
        vae_variant=vae_variant,
        projector_embed_dims=projector_embed_dims,
        vae=vae,
        mixed_precision=mixed_precision,
    )
