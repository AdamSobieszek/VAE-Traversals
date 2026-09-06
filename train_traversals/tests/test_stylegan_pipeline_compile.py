"""Smoke the StyleGAN wrapper under TrainerPotential's BF16 autocast + compile path."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.config import GAN_WEIGHTS
from models.gan_load import build_stylegan2mps


def _require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for StyleGAN pipeline compile tests")


def _weights(resolution: int = 256) -> str:
    path = REPO_ROOT / GAN_WEIGHTS["StyleGAN2"]["weights"][resolution]
    if not path.is_file():
        pytest.skip(f"missing StyleGAN2 weights: {path}")
    return str(path)


def _build(device, **kwargs):
    G = build_stylegan2mps(
        _weights(),
        resolution=256,
        shift_in_w_space=True,
        noise_mode="const",
        **kwargs,
    ).to(device).eval()
    G.requires_grad_(False)
    G.set_mixed_precision(kwargs.get("mixed_precision", "no"))
    return G


def _make_w(G, batch: int, device: torch.device, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    z = torch.randn(batch, G.dim_z, generator=generator).to(device)
    with torch.no_grad():
        return G.get_w(z).detach()


def _trainer_like_step(G, w, *, autocast_dtype):
    """Match TrainerPotential: no-grad img0/img1, grad through img2, optional BF16 autocast."""
    device_type = w.device.type
    amp = (
        torch.amp.autocast(device_type=device_type, dtype=autocast_dtype, enabled=True)
        if autocast_dtype is not None
        else torch.amp.autocast(device_type=device_type, enabled=False)
    )
    w0 = w.detach()
    w2 = w.detach().requires_grad_(True)
    with torch.no_grad(), amp:
        img0 = G(w0)
        img1 = G(w0)
    with amp:
        img2 = G(w2)
    loss = img2.square().mean()
    loss.backward()
    return img0, img1, img2, w2.grad


def _stats(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    delta = (actual.float() - expected.float()).abs()
    return {
        "max_abs": float(delta.max().item()),
        "mean_abs": float(delta.mean().item()),
        "finite": bool(torch.isfinite(actual).all().item()),
    }


@pytest.mark.cuda
def test_stylegan_wrapper_bf16_and_compile():
    _require_cuda()
    device = torch.device("cuda")
    w = None
    results = {}

    control = _build(device, compile=False, use_optimized=False, mixed_precision="no")
    w = _make_w(control, batch=4, device=device)
    _, _, img2, grad = _trainer_like_step(control, w, autocast_dtype=None)
    assert torch.isfinite(img2).all()
    assert torch.isfinite(grad).all()
    results["fp32_eager"] = {
        "img_finite": True,
        "grad_finite": True,
        "img_abs_mean": float(img2.float().abs().mean().item()),
    }

    control.set_mixed_precision("bf16")
    _, _, img2_b, grad_b = _trainer_like_step(control, w, autocast_dtype=torch.bfloat16)
    results["bf16_isolated"] = {
        **_stats(img2_b, img2),
        "grad": _stats(grad_b, grad),
    }
    assert results["bf16_isolated"]["finite"]
    assert results["bf16_isolated"]["grad"]["finite"]
    # Isolation should keep StyleGAN on its native FP16 path.
    assert results["bf16_isolated"]["max_abs"] < 1e-4

    control.isolate_amp = False
    try:
        _, _, img2_l, grad_l = _trainer_like_step(control, w, autocast_dtype=torch.bfloat16)
        results["bf16_leaked_autocast"] = {
            **_stats(img2_l, img2),
            "grad": _stats(grad_l, grad),
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        message = str(exc).splitlines()[0] if str(exc) else repr(exc)
        results["bf16_leaked_autocast"] = {
            "finite": False,
            "error": f"{type(exc).__name__}: {message[:240]}",
        }
    control.isolate_amp = True

    control.compile_enabled = False
    control.use_optimized = True
    control.set_mixed_precision("bf16")
    _, _, img2_e, grad_e = _trainer_like_step(control, w, autocast_dtype=torch.bfloat16)
    results["bf16_eager_optimized"] = {
        **_stats(img2_e, img2),
        "grad": _stats(grad_e, grad),
        "low_precision": control._optimized.cfg.low_precision,
    }
    assert control._optimized.cfg.low_precision == "bf16"
    assert results["bf16_eager_optimized"]["finite"]
    assert results["bf16_eager_optimized"]["grad"]["finite"]

    control.compile_enabled = True
    control.prepare_runtime(w)
    assert control._optimized.cfg.low_precision == "bf16"
    _, _, img2_c, grad_c = _trainer_like_step(control, w, autocast_dtype=torch.bfloat16)
    results["bf16_compile_optimized"] = {
        **_stats(img2_c, img2),
        "grad": _stats(grad_c, grad),
        "low_precision": control._optimized.cfg.low_precision,
    }
    assert results["bf16_compile_optimized"]["finite"]
    assert results["bf16_compile_optimized"]["grad"]["finite"]
    # BF16 has a 7-bit mantissa, so images move more than the FP16 path.
    assert results["bf16_compile_optimized"]["max_abs"] < 0.25
    cosine = torch.nn.functional.cosine_similarity(
        grad_c.float().flatten(), grad.float().flatten(), dim=0
    )
    results["bf16_compile_optimized"]["grad"]["cosine"] = float(cosine.item())
    assert float(cosine.item()) > 0.98

    out = REPO_ROOT / "tests/results/b64_optimize/pipeline_bf16_compile.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="tests/results/b64_optimize/pipeline_bf16_compile.json")
    args = parser.parse_args()
    os.environ.setdefault("PYTHONPATH", str(REPO_ROOT))
    test_stylegan_wrapper_bf16_and_compile()
