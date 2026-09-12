"""Chunked StyleGAN/recognizer backward should not see packed B×K in one call."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.trainer import (
    TraversalTrainer,
    choose_generator_chunk_size,
    estimate_stylegan_backward_gib,
)


def test_choose_generator_chunk_size():
    assert choose_generator_chunk_size(200) == 200
    assert choose_generator_chunk_size(200, resolution=1024) == 200
    assert choose_generator_chunk_size(200, resolution=256, device_memory_gib=95) == 200
    assert choose_generator_chunk_size(200, resolution=1024, device_memory_gib=95) == 25
    assert choose_generator_chunk_size(200, explicit=8, resolution=1024, device_memory_gib=95) == 8
    assert choose_generator_chunk_size(7, resolution=256, device_memory_gib=95) == 7
    assert estimate_stylegan_backward_gib(100, 256) == 17.25
    packed_1024 = estimate_stylegan_backward_gib(200, 1024)
    assert packed_1024 > 95
    assert estimate_stylegan_backward_gib(25, 1024) < 95


class _FakeTraversal(nn.Module):
    def __init__(self, k: int):
        super().__init__()
        self.k = k
        self.scale = nn.Parameter(torch.tensor(0.05))

    def forward(self, z, t_index, dt=None, direction=None):
        b, dim = z.shape
        latent1 = z.unsqueeze(1).expand(b, self.k, dim).contiguous() + self.scale
        latent2 = latent1 + 0.1
        pde = self.scale.square()
        preds = torch.zeros(b, self.k, device=z.device)
        return preds, latent1, latent2, pde, dt


class _FakeGenerator(nn.Module):
    dim_z = 8
    uses_vae_latent_shape = False
    img_resolution = 8

    def __init__(self):
        super().__init__()
        self.batches: list[int] = []

    def forward(self, z):
        self.batches.append(int(z.shape[0]))
        loc = z.sum(dim=-1).reshape(-1, 1, 1, 1)
        return loc.expand(-1, 3, 8, 8)


class _FakeRecognizer(nn.Module):
    def __init__(self, k: int):
        super().__init__()
        self.fc = nn.Linear(3 * 8 * 8 * 2, k)

    def forward(self, a, b):
        x = torch.cat([a.flatten(1), b.flatten(1)], dim=1)
        return self.fc(x), None


def _cpu_trainer(chunk_size: int) -> TraversalTrainer:
    trainer = TraversalTrainer.__new__(TraversalTrainer)
    trainer.params = argparse.Namespace(
        lambda_cls=1.0,
        lambda_pde=1.0,
        generator_chunk_size=chunk_size,
        gan_type="StyleGAN2",
        stylegan2_resolution=256,
        detach_img1_for_cls=False,
        mixed_precision="no",
    )
    trainer.device = torch.device("cpu")
    trainer.use_cuda = False
    trainer.use_mps = False
    trainer.amp_enabled = False
    trainer.amp_dtype = None
    trainer.mixed_precision = "no"
    trainer.cross_entropy = nn.CrossEntropyLoss()
    return trainer


def test_loss_allk_chunks_generator_batch():
    k = 6
    chunk = 2
    trainer = _cpu_trainer(chunk)
    generator = _FakeGenerator()
    recognizer = _FakeRecognizer(k)
    traversal = _FakeTraversal(k)
    z = torch.randn(1, generator.dim_z)
    t_index = torch.tensor([[2]])
    dt = torch.ones(1, 1)

    loss_dict, logits, logits0, targets, *_ = trainer.loss_allK(
        traversal, generator, recognizer, z, t_index, dt, acc_denominator=1
    )

    assert generator.batches == [2] * 9
    assert logits.shape == (k, k)
    assert logits0.shape == (k, k)
    assert targets.tolist() == list(range(k))
    assert traversal.scale.grad is not None
    assert torch.isfinite(torch.tensor(loss_dict["classification_loss"]))
