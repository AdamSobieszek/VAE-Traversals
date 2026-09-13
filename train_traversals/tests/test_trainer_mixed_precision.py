"""Exercise the generator/recognizer precision boundary in the real loss path."""
from argparse import Namespace
import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from lib.TraversalPDE import TraversalPDE
from lib.recognizer import Recognizer
from lib.trainer import TraversalTrainer
from models.gan_load import SNGANWrapper


@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize("precision", ["no", "bf16"])
def test_bidirectional_jet_generator_recognizer_precision(tmp_path, device, precision):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    torch.manual_seed(43)
    params = Namespace(mixed_precision=precision, tensorboard=False,
                       lambda_cls=1., lambda_pde=0., bidirectional=True,
                       generator_recompute_chunk=0)
    trainer = TraversalTrainer(params, exp_dir=str(tmp_path), device=device)
    trainer.K = 2
    generator = SNGANWrapper(
        SimpleNamespace(model=nn.Sequential(nn.Linear(4, 3 * 32 * 32),
                                            nn.Unflatten(1, (3, 32, 32))),
                        distribution=SimpleNamespace(dim=4)),
        mixed_precision=precision,
    ).to(device).eval().requires_grad_(False)
    recognizer = Recognizer("LeNet", 2).to(device).train()
    net = TraversalPDE(2, 3, 4, n_hidden=32, architecture="potential_jet",
                       noise_aperture=0, lambdas={"BB": .25, "signed_g2orth": 1.}).to(device)
    z = torch.randn(2, 4, device=device)
    metrics, *_ = trainer.loss_allK(
        net, generator, recognizer, z,
        torch.ones(2, 1, device=device, dtype=torch.long),
        torch.full((2, 1), 2 / 9, device=device), 1,
    )
    assert math.isfinite(metrics["classification_loss"])
    for parameter in (net.F.dir_linear.weight, net.F.fc1.weight,
                      recognizer.feature_extractor[0].weight):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.norm() > 0
    assert all(p.grad is None for p in generator.parameters())
    # Potential derivatives must stay FP32 even inside an outer AMP context.
    with trainer.gan_recognizer_context(), trainer.fp32_context():
        assert (z @ z.T).dtype == torch.float32
