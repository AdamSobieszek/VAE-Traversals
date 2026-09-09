"""Numerical regression tests: accelerator tensors and derivatives only on MPS."""
import pytest
import torch
from torch import nn

from lib.TraversalPDE import StackedSemanticPotential, TraversalPDE, gaussian_cone_noise
from lib.pde_ops import PDEState

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS required")


@pytest.fixture(autouse=True)
def mps_default():
    previous = torch.get_default_device()
    torch.set_default_device("mps")
    torch.manual_seed(43)
    yield
    torch.set_default_device(previous)


@pytest.mark.parametrize("final", [nn.Identity, nn.Tanh])
@pytest.mark.parametrize("activation", [lambda: nn.Softplus(beta=.7, threshold=1.), nn.Tanh])
def test_explicit_chain_rule_and_fallback(final, activation):
    potential = StackedSemanticPotential(4, 16, n_hidden=8, activation=activation(),
                                         final_activation=final()).eval()
    x = (torch.randn(3, 4, 16) * 8).requires_grad_()
    expected_value = potential(x)
    expected = torch.autograd.grad(expected_value.sum(), x, create_graph=True)[0]
    value, actual = potential.value_and_grad(x)
    torch.testing.assert_close(value, expected_value)
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-5)
    probe = torch.randn_like(x)
    for wrt in (x, potential.fc1.weight, potential.c, potential.dir_linear.weight):
        a = torch.autograd.grad((actual * probe).sum(), wrt, retain_graph=True)[0]
        b = torch.autograd.grad((expected * probe).sum(), wrt, retain_graph=True)[0]
        torch.testing.assert_close(a, b, atol=2e-6, rtol=1e-4)


@pytest.mark.parametrize("value_first", [False, True])
def test_pde_state_cache_matches_generic_autograd(value_first):
    potential = StackedSemanticPotential(4, 16, n_hidden=8).eval()
    x = torch.randn(3, 4, 16)
    old = PDEState(potential, x, explicit_input_grad=False)
    new = PDEState(potential, x)
    if value_first:
        new.f()
    torch.testing.assert_close(new.f_grad(), old.f_grad(), atol=1e-7, rtol=1e-5)
    torch.testing.assert_close(new.Xf(), old.Xf(), atol=1e-6, rtol=1e-5)
    assert new.f_grad() is new.f_grad()


@pytest.mark.parametrize("scale", [0., 1e-9, 1.])
def test_gaussian_cone_preserves_kernel_and_stop_gradient(scale):
    delta = (torch.randn(3, 4, 16) * scale).requires_grad_()
    gaussian = torch.randn_like(delta)
    untouched = gaussian.clone()
    actual = gaussian_cone_noise(delta, gaussian=gaussian)
    with torch.no_grad():
        norm2 = delta.square().sum(-1, keepdim=True)
        orth = gaussian - delta * ((gaussian * delta).sum(-1, keepdim=True) / norm2.clamp_min(1e-12))
        expected = orth / orth.norm(dim=-1, keepdim=True).clamp_min(1e-12) * (norm2.sqrt() / 5.)
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-5)
    assert torch.equal(gaussian, untouched)
    assert not actual.requires_grad
    assert torch.isfinite(actual).all()


@pytest.mark.parametrize("detach", [False, True])
def test_endpoint_gather_preserves_step_selection_and_gradients(detach):
    model = TraversalPDE(4, 5, 16, n_hidden=8, detach_between_steps=detach,
                         lambdas={"BB": .25, "signed_g2orth": 1.}).eval()
    z = torch.randn(3, 16)
    dt = torch.full((3, 1), .1)
    target = torch.tensor([-3, 1, 20])
    offset = torch.randn(16)
    result = model(z, target, dt, w_avg=offset)
    current = (z - offset)[:, None].expand(3, 4, 16).contiguous()
    left, right = torch.zeros_like(current), torch.zeros_like(current)
    for i in range(4):
        _, following, _, _ = model._per_step(current, dt=dt)
        mask = (target.clamp(0, 3) == i)[:, None, None]
        left = torch.where(mask, current, left)
        right = torch.where(mask, following, right)
        current = following
    torch.testing.assert_close(result[1], left + offset)
    torch.testing.assert_close(result[2], right + offset)
    a = torch.autograd.grad(result[2].square().mean(), model.F.fc1.weight)[0]
    b = torch.autograd.grad((right + offset).square().mean(), model.F.fc1.weight)[0]
    torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)


def test_eval_inference_omits_losses_but_retains_derivatives():
    model = TraversalPDE(4, 5, 16, n_hidden=8, lambdas={"BB": .25}).eval()
    z = torch.randn(3, 16)
    dt = torch.full((3, 1), .1)
    current = z[:, None].expand(3, 4, 16).contiguous()
    _, expected, _, _ = model._per_step(current, dt=dt)

    class ForbiddenLoss:
        def __call__(self, state):
            raise AssertionError("Inference must not evaluate unused PDE losses")

    model.losses = [ForbiddenLoss()]
    origin, displacement = model.inference(z, dt=dt)
    torch.testing.assert_close(origin + displacement, expected)
    a = torch.autograd.grad(displacement.square().mean(), model.F.fc1.weight)[0]
    b = torch.autograd.grad((expected - current).square().mean(), model.F.fc1.weight)[0]
    torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)


def test_image_worker_errors_surface_and_writer_closes():
    from lib.aux import ImageLogger

    class Writer:
        log_dir = ""
        closed = False

        def add_image(self, *args):
            raise ValueError("encoding failed")

        def close(self):
            self.closed = True

    writer = Writer()
    logger = ImageLogger(writer)
    images = torch.rand(1, 2, 3, 8, 8)
    logger.log_triplet("images", images, images, images, 1)
    with pytest.raises(ValueError, match="encoding failed"):
        logger.close()
    assert writer.closed
