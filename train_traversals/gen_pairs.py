"""
Generate paired images for VP metric using a trained experiment directory.

Example:
    python gen_pairs.py --exp /path/to/exp_dir 
"""

import argparse
import json
import os
import os.path as osp
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib import *  # brings in GAN_WEIGHTS, GAN_RESOLUTIONS, TraversalPDE, etc.
from models.gan_load import (
    build_biggan, build_proggan, build_sngan, build_gat,
)
from lib.aux import choose_device
# ------------------
# Helpers
# ------------------

class ModelArgs:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def sample_z(batch_size, dim_z, device, truncation=None):
    """Sample latent z with optional truncation."""
    if truncation is None or truncation == 1.0:
        return torch.randn(batch_size, dim_z, device=device)
    else:
        from scipy.stats import truncnorm
        z_np = truncnorm.rvs(-truncation, truncation, size=(batch_size, dim_z))
        return torch.from_numpy(z_np).to(device=device, dtype=torch.float32)


def resolve_gat_checkpoint(exp_args, script_dir):
    gan_cfg = GAN_WEIGHTS["GAT"]
    resolution = int(getattr(exp_args, "resolution", None) or GAN_RESOLUTIONS["GAT"])
    if getattr(exp_args, "gat_ckpt", ""):
        ckpt_path = Path(exp_args.gat_ckpt).expanduser()
        if not ckpt_path.is_absolute():
            ckpt_path = (script_dir / ckpt_path).resolve()
    else:
        ckpt_path = (script_dir / gan_cfg["weights"][resolution]).resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"GAT checkpoint not found: {ckpt_path}. "
            "Run scripts/setup_gat_inference.sh or pass --gat-ckpt during training."
        )
    return ckpt_path, resolution, gan_cfg


def build_gan(exp_args, device, script_dir):
    gan_type = exp_args.__dict__['gan_type']
    # BigGAN
    if gan_type == 'BigGAN':
        G = build_biggan(
            pretrained_gan_weights=GAN_WEIGHTS[gan_type]['weights'][GAN_RESOLUTIONS[gan_type]],
            target_classes=exp_args.__dict__.get('biggan_target_classes', None)
        )
    # ProgGAN
    elif gan_type == 'ProgGAN':
        G = build_proggan(
            pretrained_gan_weights=GAN_WEIGHTS[gan_type]['weights'][GAN_RESOLUTIONS[gan_type]]
        )
    # GAT
    elif gan_type == 'GAT':
        ckpt_path, resolution, gan_cfg = resolve_gat_checkpoint(exp_args, script_dir)
        G = build_gat(
            str(ckpt_path),
            target_classes=tuple(exp_args.__dict__.get('target_classes', [239])),
            model_name=exp_args.__dict__.get('gat_model', '') or gan_cfg.get('model'),
            resolution=resolution,
            vae_variant=exp_args.__dict__.get('vae_variant', None) or gan_cfg.get('vae_variant', 'ema'),
            truncation_psi=float(exp_args.__dict__.get('truncation_psi', 0.8)),
            mixed_precision=exp_args.__dict__.get('mixed_precision', 'bf16'),
            load_vae=True,
        )
    # StyleGAN2
    elif gan_type == 'StyleGAN2':
        raise RuntimeError("StyleGAN2 pair generation is not available in the current gan_load.py.")
    # SNGAN family
    else:
        G = build_sngan(
            pretrained_gan_weights=GAN_WEIGHTS[gan_type]['weights'][GAN_RESOLUTIONS[gan_type]],
            gan_type=gan_type
        )

    G = G.to(device).eval()
    return G


@torch.no_grad()
def generate_rgb(generator, z):
    output = generator(z)
    base = generator.module if hasattr(generator, "module") else generator
    decode = getattr(base, "decode_with_vae", None)
    if decode is not None:
        output = decode(output)
    return output

def load_support_sets(exp_models_dir, device):
    """
    Load TraversalPDE from checkpoint. We’ll:
      1) Read args.json to get K, T.
      2) Instantiate TraversalPDE(K, T, D) after we create G (so we know dim_z).
      3) Load weights from checkpoint dict (robust to a few key layouts).
    """
    # Read args.json
    args_json_file = osp.join(osp.dirname(exp_models_dir), 'args.json')
    if not osp.isfile(args_json_file):
        raise FileNotFoundError(f"File not found: {args_json_file}")
    a = ModelArgs(**json.load(open(args_json_file)))

    # Choose checkpoint
    ckpt_path = osp.join(exp_models_dir, 'checkpoint.pt')
    if not osp.isfile(ckpt_path):
        # fall back to final support_sets.pt, then last support_sets-*.pt
        cands = []
        if osp.isfile(osp.join(exp_models_dir, 'support_sets.pt')):
            cands.append('support_sets.pt')
        cands.extend(sorted([f for f in os.listdir(exp_models_dir) if f.startswith('support_sets-')]))
        if not cands:
            raise FileNotFoundError(f"No checkpoint found in {exp_models_dir}")
        ckpt_path = osp.join(exp_models_dir, cands[-1])

    ckpt = torch.load(ckpt_path, map_location=device)

    return a, ckpt, ckpt_path

def robust_load_waves(S: nn.Module, ckpt):
    """
    Try a few common layouts to load TraversalPDE weights from checkpoint.
    """
    def looks_like_support_state(state_dict):
        return any(
            k == 'c'
            or k.startswith('F.')
            or k.startswith('PSI')
            or k == 'PSI_SET'
            for k in state_dict.keys()
        )

    sd = None
    if isinstance(ckpt, dict):
        if 'support_sets' in ckpt and isinstance(ckpt['support_sets'], dict):
            sd = ckpt['support_sets']
        elif 'state_dict' in ckpt and isinstance(ckpt['state_dict'], dict):
            # if state_dict looks like TraversalPDE already
            if looks_like_support_state(ckpt['state_dict']):
                sd = ckpt['state_dict']
        elif all(isinstance(k, str) for k in ckpt.keys()):
            # checkpoint is the state_dict itself
            if looks_like_support_state(ckpt):
                sd = ckpt
    if sd is None:
        raise RuntimeError("Could not find TraversalPDE weights in checkpoint. Expected keys like 'support_sets', 'F.*', or 'c'.")
    S.load_state_dict(sd, strict=True)

# ------------------
# Main
# ------------------

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Generate paired images for VP metric')
    p.add_argument('--exp', type=str, required=True, help="experiment dir (created by train.py)")
    p.add_argument('--shift-steps', type=int, default=16, help="# shifts per direction (unused for PDE rollout)")
    p.add_argument('--eps', type=float, default=0.2, help="shift magnitude (unused for PDE rollout)")
    p.add_argument('--shift-leap', type=float, default=1.0, help="frame stride for saving (unused here)")
    p.add_argument('--batch-size', type=int, default=2, help="generator batch size")
    p.add_argument('--img-size', type=int, default=256, help="saved image size (resized)")
    p.add_argument('--img-quality', type=int, default=75, help="JPEG quality")
    p.add_argument('--gif', action='store_true', help="(unused)")
    p.add_argument('--gif-size', type=int, default=256)
    p.add_argument('--gif-fps', type=int, default=30)
    p.add_argument('--n-samples', type=int, default=10000, help="number of samples to generate")
    p.add_argument('--only-potential', type=bool, default=True, help="only generate potential pairs")
    args = p.parse_args()   

    device = choose_device()

    # Resolve paths
    if not osp.isdir(args.exp):
        raise NotADirectoryError(f"Invalid experiment directory: {args.exp}")
    models_dir = osp.join(args.exp, 'models')
    if not osp.isdir(models_dir):
        raise NotADirectoryError(f"Invalid models directory: {models_dir}")

    # Load args + checkpoint metadata
    a, ckpt, ckpt_path = load_support_sets(models_dir, device)

    # Build generator (aligned with train.py)
    script_dir = Path(__file__).resolve().parent
    G = build_gan(a, device=device, script_dir=script_dir).eval()

    # Instantiate TraversalPDE with D = G.dim_z (same as train.py), then load weights
    S = TraversalPDE(
        num_support_sets=a.__dict__['num_support_sets'],
        num_support_timesteps=a.__dict__['num_support_timesteps'],
        support_vectors_dim=G.dim_z,
        only_potential=a.__dict__.get('only_potential', True)
    ).to(device).eval()
    robust_load_waves(S, ckpt)

    # Output directory
    out_dir = osp.join(args.exp, 'vp_pairs')
    os.makedirs(out_dir, exist_ok=True)

    # Pair generation config
    n_samples = args.n_samples
    B = int(args.batch_size)
    K = int(S.num_support_sets)

    # PDE rollout length: match training (half_range = T // 2)
    half_range = S.num_support_timesteps // 2

    # Truncation (if present in args.json)
    z_trunc = a.__dict__.get('z_truncation', None)

    all_labels = []
    pair_idx = 0  # global pair counter (across all batches)

    # Generate until we have exactly n_samples pairs (each pair corresponds to one support set)
    while pair_idx < n_samples:
        remaining = n_samples - pair_idx
        # Each base latent yields K pairs (one per support set).
        cur_B = min(B, int(np.ceil(remaining / float(K))))

        print(f'Generating image pairs batch with base batch-size={cur_B}, remaining={remaining} ...')

        # Sample batch z on device, with truncation if specified
        z0 = sample_z(cur_B*K, G.dim_z, device=device, truncation=z_trunc)

        # Optionally move to W space for StyleGAN2
        if a.__dict__.get('shift_in_w_space', True if a.__dict__['gan_type'] == 'StyleGAN2' else False) and hasattr(G, 'get_w'):
            with torch.no_grad():
                z_cur = G.get_w(z0.reshape(cur_B*K, G.dim_z)).reshape(cur_B, K, G.dim_z)
        else:
            z_cur = z0.reshape(cur_B, K, G.dim_z)
        z0_batch = z_cur.reshape(cur_B * K, G.dim_z)

        # Rollout by PDE: latent_{t+1} = latent_t + ∇_z u(latent_t, t)
        with torch.no_grad():
            for step in range(half_range-1):
                t_b = torch.full((cur_B, 1), float(step), device=device, dtype=z_cur.dtype)
                z_cur, dz = S.inference(z_cur, t_b, dt=args.shift_leap*2/max(1,half_range-1))  # returns (z_curr, delta_z) with K support sets batched
                z_cur = z_cur + dz

        # One-hot labels for VP: [cur_B*K, K]
        label = torch.eye(K, device=device).repeat(cur_B, 1)
        z_cur = z_cur.reshape(cur_B * K, G.dim_z)

        # How many of the cur_B*K pairs do we actually need from this batch?
        batch_pairs = cur_B * K
        take = min(batch_pairs, remaining)

        all_labels.append(label[:take].cpu().numpy())

        # Generate images for the selected pairs
        with torch.no_grad():
            img1 = generate_rgb(G, z0_batch[:take])
            img2 = generate_rgb(G, z_cur[:take])

        # Resize to requested output size
        if args.img_size is not None:
            img1 = F.interpolate(img1, size=(args.img_size, args.img_size), mode="bilinear", align_corners=False)
            img2 = F.interpolate(img2, size=(args.img_size, args.img_size), mode="bilinear", align_corners=False)

        # Save pairs as JPEG
        # Convert from [-1,1] RGB to uint8 BGR for cv2
        img1 = img1.clamp(-1, 1)
        img2 = img2.clamp(-1, 1)
        for im1, im2 in zip(img1, img2):
            a1 = im1.detach().cpu().numpy().transpose(1, 2, 0)  # HWC, RGB
            a2 = im2.detach().cpu().numpy().transpose(1, 2, 0)
            pair = np.concatenate([a1, a2], axis=1)
            pair = ((pair + 1.0) * 127.5).round().astype(np.uint8)
            pair = pair[:, :, ::-1]  # RGB -> BGR
            cv2.imwrite(
                osp.join(out_dir, f'pair_{pair_idx:06d}.jpg'),
                pair,
                [int(cv2.IMWRITE_JPEG_QUALITY), int(args.img_quality)]
            )
            pair_idx += 1
            if pair_idx >= n_samples:
                break

    labels = np.concatenate(all_labels, axis=0)
    # Be safe and trim to exactly n_samples in case of any rounding
    labels = labels[:n_samples]
    np.save(osp.join(out_dir, 'labels.npy'), labels)
    print(f"Done. Saved {n_samples} pairs and labels.npy to {out_dir}")
