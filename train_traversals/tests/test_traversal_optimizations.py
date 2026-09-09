from __future__ import annotations

import argparse

import pytest
import torch
from torch import nn

from lib.aux import TrainingStatTracker, sample_z, tb_hists
from lib.recognizer import AntisymmetricRecognizer, Recognizer
from lib.trainer_potential_nue import TrainerPotential


pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="these optimization tests are MPS-only"
)
DEVICE = torch.device("mps")


class _LatentSpec:
    dim_z = 16
    shift_in_w_space = False


def test_sample_z_is_equal_norm_and_orthogonal():
    z = sample_z(6, _LatentSpec(), argparse.Namespace(z_truncation=None), DEVICE)
    gram = z @ z.mT
    diagonal = gram.diagonal()
    off_diagonal = gram - torch.diag_embed(diagonal)
    assert torch.allclose(diagonal, diagonal[:1].expand_as(diagonal), rtol=1e-5, atol=1e-5)
    assert off_diagonal.abs().max().item() < 1e-4


def test_legacy_antisymmetric_helper_matches_explicit_evaluation():
    recognizer = Recognizer("LeNet", 4, channels=3).to(DEVICE).eval()
    center = torch.randn(3, 3, 64, 64, device=DEVICE)
    endpoint = torch.randn_like(center)
    reflected = 2 * center - endpoint
    expected = recognizer(reflected, endpoint)[0] - recognizer(endpoint, reflected)[0]
    actual = recognizer.antisymmetric_logits(center, endpoint)
    assert torch.equal(actual, expected)


def test_structural_recognizer_is_antisymmetric():
    recognizer = AntisymmetricRecognizer("LeNet", 4, channels=3).to(DEVICE).train()
    x1 = torch.randn(3, 3, 64, 64, device=DEVICE)
    x2 = torch.randn_like(x1)
    forward = recognizer(x1, x2)[0]
    reverse = recognizer(x2, x1)[0]
    assert torch.allclose(forward, -reverse, rtol=1e-5, atol=1e-5)


def test_vectorized_per_k_stats_match_legacy_updates():
    batch, k = 5, 7
    preds = torch.randint(k, (batch, k), device=DEVICE).cpu().numpy()
    grad_norms = torch.rand(k, device=DEVICE).cpu().numpy()
    legacy = TrainingStatTracker(ema_decay=0.9)
    current = TrainingStatTracker(ema_decay=0.9)
    legacy.init_per_k(k)
    current.init_per_k(k)

    for index in range(k):
        legacy.update_per_k_after_micro(
            true_k=index,
            preds=preds[:, index],
            batch_size=batch,
            grad_norm_selected_mlp=grad_norms[index],
        )
    current.update_all_k_after_micro(preds=preds, grad_norms=grad_norms)

    assert (legacy.confusion == current.confusion).all()
    assert (legacy.per_k_select_counts == current.per_k_select_counts).all()
    assert pytest.approx(legacy.per_k_ema_acc) == current.per_k_ema_acc
    assert pytest.approx(legacy.per_k_ema_grad) == current.per_k_ema_grad


def test_histogram_logging_filters_nonfinite_values_after_one_transfer():
    class Writer:
        def __init__(self):
            self.records = []

        def add_histogram(self, name, values, step):
            assert values.device.type == "cpu"
            assert torch.isfinite(values).all()
            self.records.append((name, step))

    writer = Writer()
    logits = torch.randn(8, 4, device=DEVICE)
    potentials = torch.randn(2, 4, 3, device=DEVICE)
    potentials[:, 2] = torch.nan
    tb_hists(writer, 7, logits_det=logits, potential_preds_det=potentials, K=4)
    assert writer.records == [
        ("train/logits", 7),
        ("potential_distribution/0", 7),
        ("potential_distribution/1", 7),
        ("potential_distribution/3", 7),
    ]


class _FakeTraversal(nn.Module):
    def __init__(self, k: int):
        super().__init__()
        self.k = k
        self.scale = nn.Parameter(torch.tensor(0.05))

    def forward(self, z, t_index, dt=None, direction=None):
        batch, dim = z.shape
        latent1 = z[:, None].expand(batch, self.k, dim).contiguous() + self.scale
        latent2 = latent1 + 0.1
        predictions = torch.zeros(batch, self.k, 2, device=z.device)
        return predictions, latent1, latent2, self.scale.square(), dt


class _FakeGenerator(nn.Module):
    dim_z = 8
    uses_vae_latent_shape = False
    share_initial_output = True

    def __init__(self):
        super().__init__()
        self.batches = []

    def forward(self, z):
        self.batches.append(int(z.shape[0]))
        return z.sum(-1).view(-1, 1, 1, 1).expand(-1, 3, 8, 8)


class _FakeRecognizer(nn.Module):
    def __init__(self, k: int):
        super().__init__()
        self.fc = nn.Linear(3 * 8 * 8 * 2, k)

    def forward(self, a, b):
        return self.fc(torch.cat((a.flatten(1), b.flatten(1)), dim=1)), None


def test_loss_allk_synthesizes_shared_initial_image_once():
    batch, k = 2, 4
    trainer = TrainerPotential.__new__(TrainerPotential)
    trainer.params = argparse.Namespace(
        lambda_cls=1.0,
        lambda_pde=1.0,
        detach_img1_for_cls=False,
        mixed_precision="no",
    )
    trainer.device = DEVICE
    trainer.use_cuda = False
    trainer.use_mps = True
    trainer.amp_enabled = False
    trainer.amp_dtype = None
    trainer.cross_entropy = nn.CrossEntropyLoss()

    traversal = _FakeTraversal(k).to(DEVICE)
    generator = _FakeGenerator().to(DEVICE)
    recognizer = _FakeRecognizer(k).to(DEVICE)
    z = torch.randn(batch, generator.dim_z, device=DEVICE)
    t_index = torch.ones(batch, 1, device=DEVICE, dtype=torch.long)
    dt = torch.ones(batch, 1, device=DEVICE)

    _, logits, logits0, targets, *_ = trainer.loss_allK(
        traversal, generator, recognizer, z, t_index, dt, acc_denominator=1
    )

    assert generator.batches == [batch, batch * k, batch * k]
    assert logits.shape == logits0.shape == (batch * k, k)
    assert targets.tolist() == list(range(k)) * batch
    assert traversal.scale.grad is not None
