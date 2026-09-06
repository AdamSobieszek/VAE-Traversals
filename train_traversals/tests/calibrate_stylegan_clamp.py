"""Sample StyleGAN activations and report pre-clamp magnitudes.

Example:
    PYTHONPATH=. python tests/calibrate_stylegan_clamp.py --resolution 256 --batches 4 --batch-size 4
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from lib.config import GAN_WEIGHTS
from models.gan_load import build_stylegan2mps
from models.StyleGAN2_mps.torch_utils.ops import bias_act as bias_act_mod


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, default=256, choices=(256, 1024))
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(args.device)
    wrapper = build_stylegan2mps(
        GAN_WEIGHTS["StyleGAN2"]["weights"][args.resolution],
        resolution=args.resolution,
        shift_in_w_space=True,
    )
    G = wrapper.G.to(device).eval()

    stats: dict[str, float] = {}
    original = bias_act_mod.bias_act

    def hooked(x, b=None, dim=1, act="linear", alpha=None, gain=None, clamp=None, impl="cuda"):
        y = original(x, b=b, dim=dim, act=act, alpha=alpha, gain=gain, clamp=None, impl="ref")
        key = f"act={act},shape={tuple(x.shape)},clamp={clamp}"
        mag = float(y.detach().float().abs().amax())
        stats[key] = max(stats.get(key, 0.0), mag)
        return original(x, b=b, dim=dim, act=act, alpha=alpha, gain=gain, clamp=clamp, impl=impl)

    bias_act_mod.bias_act = hooked
    g = torch.Generator(device="cpu")
    g.manual_seed(args.seed)
    with torch.inference_mode():
        for _ in range(args.batches):
            z = torch.randn(args.batch_size, G.z_dim, generator=g, device="cpu")
            ws = G.mapping(z.to(device), None)
            _ = G.synthesis(ws, noise_mode="const")
    bias_act_mod.bias_act = original

    overall = max(stats.values()) if stats else 0.0
    print(f"configured samples={args.batches * args.batch_size}")
    print(f"largest observed pre-clamp magnitude: {overall}")
    for key, value in sorted(stats.items(), key=lambda kv: -kv[1])[:20]:
        print(f"{value:10.4f}  {key}")


if __name__ == "__main__":
    main()
