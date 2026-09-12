"""MPS-only microbenchmarks for the Traversals training hot path.

This is intentionally a script rather than a pytest: it measures synchronized
wall-clock time and exits instead of silently falling back when MPS is absent.
"""

from __future__ import annotations

import argparse
import gc
import statistics
import sys
import time
from pathlib import Path

import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.TraversalPDE import StackedLinear, TraversalPDE
from lib.utils import sample_z
from lib.recognizer import AntisymmetricRecognizer, Recognizer
from lib.trainer import TraversalTrainer
from models.gan_load import build_sngan


DEVICE = torch.device("mps")


def sync() -> None:
    torch.mps.synchronize()


def timed(name, fn, *, warmup=1, repeats=3):
    for _ in range(warmup):
        fn()
        sync()
    samples = []
    for _ in range(repeats):
        gc.collect()
        sync()
        start = time.perf_counter()
        fn()
        sync()
        samples.append(1e3 * (time.perf_counter() - start))
    print(f"{name:36s} median={statistics.median(samples):9.2f} ms  samples={samples}")


def make_trainer(k: int) -> TraversalTrainer:
    trainer = TraversalTrainer.__new__(TraversalTrainer)
    trainer.params = argparse.Namespace(
        lambda_cls=1.0,
        lambda_pde=1.0,
        detach_img1_for_cls=False,
        z_truncation=1.0,
        mixed_precision="no",
        num_traversal_sets=k,
    )
    trainer.device = DEVICE
    trainer.use_cuda = False
    trainer.use_mps = True
    trainer.amp_enabled = False
    trainer.amp_dtype = None
    trainer.mixed_precision = "no"
    trainer.cross_entropy = nn.CrossEntropyLoss()
    return trainer


def main() -> None:
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is required; refusing to run on CPU")

    torch.manual_seed(0)
    batch, k, dim, steps = 6, 32, 128, 20
    weights = REPO_ROOT / "models/pretrained/generators/SNGAN_AnimeFaces/generator.pt"
    generator = build_sngan(str(weights), "SNGAN_AnimeFaces").to(DEVICE).eval().requires_grad_(False)
    recognizer = Recognizer("LeNet", k, channels=3).to(DEVICE).train()
    antisymmetric_recognizer = AntisymmetricRecognizer("LeNet", k, channels=3).to(DEVICE).train()
    traversal = TraversalPDE(
        num_traversal_sets=k,
        num_traversal_timesteps=steps,
        traversal_vectors_dim=dim,
        n_hidden=32,
        lambdas={"BB": 0.25, "signed_g2orth": 1.0},
    ).to(DEVICE).train()
    trainer = make_trainer(k)

    def sample_z_legacy():
        z0 = torch.randn(dim, device=DEVICE)
        radius = z0.norm()
        basis = [z0 / (radius + 1e-8)]
        for _ in range(1, min(batch, dim)):
            vector = torch.randn(dim, device=DEVICE)
            for previous in basis:
                vector = vector - (vector @ previous) * previous
            vector_norm = vector.norm()
            vector = vector / vector_norm if vector_norm >= 1e-8 else torch.zeros_like(vector)
            basis.append(vector)
        return torch.stack(basis).mul(radius)

    timed("sample_z legacy Gram-Schmidt", sample_z_legacy, repeats=5)
    timed("sample_z current reduced QR", lambda: sample_z(batch, generator, trainer.params, DEVICE), repeats=5)

    z_b = torch.randn(batch, dim, device=DEVICE)
    z_bk = z_b[:, None].expand(batch, k, dim).reshape(batch * k, dim)
    timed("SNGAN synth B", lambda: generator(z_b), repeats=3)
    timed("SNGAN synth B*K", lambda: generator(z_bk), repeats=3)
    timed("SNGAN img0+img1 current", lambda: (generator(z_bk), generator(z_bk)), repeats=3)
    timed("SNGAN img0 shared+img1", lambda: (generator(z_b), generator(z_bk)), repeats=3)
    timed(
        "SNGAN concat shared+img1",
        lambda: generator(torch.cat((z_b, z_bk), dim=0)),
        repeats=3,
    )

    images = generator(torch.randn(batch * k, dim, device=DEVICE)).detach()
    a = images
    b = images.roll(1, 0)

    def antisym_sequential():
        recognizer.zero_grad(set_to_none=True)
        b_grad = b.detach().requires_grad_(True)
        pos = recognizer(2 * a - b_grad, b_grad)[0]
        neg = recognizer(b_grad, 2 * a - b_grad)[0]
        (pos - neg).float().square().mean().backward()

    def antisym_batched():
        recognizer.zero_grad(set_to_none=True)
        b_grad = b.detach().requires_grad_(True)
        reflected = 2 * a - b_grad
        x1 = torch.cat((reflected, b_grad), dim=0)
        x2 = torch.cat((b_grad, reflected), dim=0)
        both = recognizer(x1, x2)[0]
        pos, neg = both.chunk(2)
        (pos - neg).float().square().mean().backward()

    timed("recognizer antisym sequential", antisym_sequential, repeats=3)
    timed("recognizer antisym batched", antisym_batched, repeats=3)

    def antisym_structural():
        antisymmetric_recognizer.zero_grad(set_to_none=True)
        b_grad = b.detach().requires_grad_(True)
        logits = antisymmetric_recognizer.antisymmetric_logits(a, b_grad)
        logits.float().square().mean().backward()

    timed("recognizer antisym structural", antisym_structural, repeats=3)

    stacked = StackedLinear(k, dim, 32).to(DEVICE)
    x = torch.randn(batch, k, dim, device=DEVICE)
    timed("StackedLinear einsum forward", lambda: stacked(x), repeats=10)
    timed(
        "StackedLinear matmul forward",
        lambda: torch.matmul(x.movedim(0, 1), stacked.weight.mT).movedim(0, 1) + stacked.bias,
        repeats=10,
    )

    def pde_step():
        traversal.zero_grad(set_to_none=True)
        z = torch.randn(batch, dim, device=DEVICE)
        t = torch.full((batch, 1), 8, device=DEVICE, dtype=torch.long)
        dt = torch.full((batch, 1), 0.2, device=DEVICE)
        out = traversal(z, t, dt)
        out[3].backward()

    timed("TraversalPDE forward+backward", pde_step, warmup=0, repeats=2)

    def full_loss():
        traversal.zero_grad(set_to_none=True)
        recognizer.zero_grad(set_to_none=True)
        z = torch.randn(batch, dim, device=DEVICE)
        t = torch.full((batch, 1), 8, device=DEVICE, dtype=torch.long)
        dt = torch.full((batch, 1), 0.2, device=DEVICE)
        trainer.loss_allK(traversal, generator, recognizer, z, t, dt, 1)

    def full_loss_unshared():
        generator.share_initial_output = False
        try:
            full_loss()
        finally:
            generator.share_initial_output = True

    def full_loss_structural():
        traversal.zero_grad(set_to_none=True)
        antisymmetric_recognizer.zero_grad(set_to_none=True)
        z = torch.randn(batch, dim, device=DEVICE)
        t = torch.full((batch, 1), 8, device=DEVICE, dtype=torch.long)
        dt = torch.full((batch, 1), 0.2, device=DEVICE)
        trainer.loss_allK(traversal, generator, antisymmetric_recognizer, z, t, dt, 1)

    timed("loss_allK unshared img0", full_loss_unshared, warmup=0, repeats=2)
    timed("loss_allK shared img0", full_loss, warmup=0, repeats=2)
    timed("loss_allK structural recognizer", full_loss_structural, warmup=0, repeats=2)

    generator.to(memory_format=torch.channels_last)
    recognizer.to(memory_format=torch.channels_last)
    timed("SNGAN channels_last synth B*K", lambda: generator(z_bk), repeats=3)
    timed("loss_allK channels_last", full_loss, warmup=0, repeats=2)
    print(f"MPS current allocated: {torch.mps.current_allocated_memory() / 2**20:.1f} MiB")
    print(f"MPS driver allocated:  {torch.mps.driver_allocated_memory() / 2**20:.1f} MiB")


if __name__ == "__main__":
    main()
