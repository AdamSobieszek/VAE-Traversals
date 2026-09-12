"""One tiny SNGAN loss backward per architecture, without optimizer updates."""
import argparse
from pathlib import Path

import pytest
import torch
from torch import nn

from lib.TraversalPDE import TraversalPDE
from lib.recognizer import Recognizer
from lib.trainer import TraversalTrainer
from models.gan_load import build_sngan


WEIGHTS = Path(__file__).resolve().parents[1] / "models/pretrained/generators/SNGAN_AnimeFaces/generator.pt"
pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available() or not WEIGHTS.is_file(),
                                reason="MPS and local SNGAN AnimeFaces weights required")


@pytest.fixture(scope="module")
def generator():
    g = build_sngan(str(WEIGHTS), "SNGAN_AnimeFaces").to("mps").eval().requires_grad_(False)
    yield g
    del g
    torch.mps.empty_cache()


@pytest.mark.parametrize("architecture", ["euler", "potential_jet", "midpoint"])
@pytest.mark.parametrize("recognizer_type", ["LeNet", "CAT"])
def test_sngan_recognizer_backward(architecture, recognizer_type, generator):
    torch.manual_seed(43)
    net = TraversalPDE(2, 3, 128, n_hidden=32, architecture=architecture, noise_aperture=0,
                        lambdas={"BB": .25, "signed_g2orth": 1.}).to("mps").train()
    recognizer = Recognizer(recognizer_type, 2, channels=3).to("mps").train()
    trainer = TraversalTrainer.__new__(TraversalTrainer)
    # Isolate recognizer supervision: PDE gradients must not make this pass.
    trainer.params = argparse.Namespace(lambda_cls=1., lambda_pde=0., bidirectional=True,
                                        mixed_precision="no", generator_recompute_chunk=0)
    trainer.device = torch.device("mps")
    trainer.use_cuda, trainer.use_mps = False, True
    trainer.amp_enabled, trainer.amp_dtype = False, None
    trainer.K, trainer.cross_entropy = 2, nn.CrossEntropyLoss()
    z = torch.randn(1, 128, device="mps")
    batches = []
    hook = generator.register_forward_pre_hook(lambda _, args: batches.append(args[0].shape[0]))
    try:
        metrics, *_ = trainer.loss_allK(net, generator, recognizer, z,
                                        torch.ones(1, 1, device="mps", dtype=torch.long),
                                        torch.full((1, 1), 2 / 9, device="mps"), 1)
    finally:
        hook.remove()
    assert batches == [1, 2, 2]
    norms = {}
    for name, p in (("linear", net.F.dir_linear.weight), ("nonlinear", net.F.fc1.weight)):
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm() > 0
        norms[name] = p.grad.norm().item()
    assert all(p.grad is None for p in generator.parameters())
    print(recognizer_type, architecture, "classification_loss", metrics["classification_loss"], "gradient_norms", norms)
