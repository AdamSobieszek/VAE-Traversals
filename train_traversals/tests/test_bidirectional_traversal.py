"""Small deterministic MPS checks; no training and no large generators."""
import copy

import pytest
import torch

from lib.TraversalPDE import TraversalPDE, DirectionalSemanticPotential, _base_velocity
from lib.pde_ops import broadcast_bk

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS required")


@pytest.fixture(autouse=True)
def mps_default():
    previous = torch.get_default_device()
    torch.set_default_device("mps")
    torch.manual_seed(127)
    yield
    torch.set_default_device(previous)


def model(mode, **kwargs):
    return TraversalPDE(2, 4, 4, n_hidden=8, architecture=mode,
                        noise_aperture=0, midpoint_atol=1e-6, midpoint_rtol=1e-6,
                        **kwargs).eval()


@pytest.mark.parametrize("mode", ["euler", "potential_jet", "midpoint"])
def test_shared_parameters_and_constant_sign_rollout(mode):
    net = model(mode)
    baseline = model("euler")
    assert {k: v.shape for k, v in net.state_dict().items()} == {
        k: v.shape for k, v in baseline.state_dict().items()}
    assert sum(p.numel() for p in net.parameters()) == sum(p.numel() for p in baseline.parameters())
    # Make F affine: reversal must be exact for arbitrarily large translations.
    with torch.no_grad():
        net.F.c.zero_()
    z = torch.randn(2, 4)
    sign = torch.tensor([[[1.], [-1.]], [[-1.], [1.]]])
    h = torch.tensor([[.3], [.5]])
    pred, first, last, _, _ = net(z, torch.tensor([1, 2]), h, direction=sign)
    _, delta = net.inference(z, dt=h, direction=sign)
    torch.testing.assert_close(first, z[:, None] + delta * torch.tensor([2., 3.])[:, None, None] - delta)
    torch.testing.assert_close(last, first + delta)
    assert pred.shape == (2, 2, 4)


@pytest.mark.parametrize("eps", [0., 1e-8, .01])
def test_midpoint_retraces_curved_path_and_is_a_scalar_potential(eps):
    net = model("midpoint", eps_norm2=eps)
    x = torch.randn(2, 2, 4).requires_grad_()
    signs = torch.tensor([[[1.], [-1.]], [[-1.], [1.]]])
    st, y, _, _ = net._per_step(x, dt=.2, direction=signs)
    value, gradient = st.f(), st.f_grad()
    exact_gradient = torch.autograd.grad(value.sum(), x, retain_graph=True)[0]
    torch.testing.assert_close(gradient, exact_gradient, atol=2e-6, rtol=2e-5)
    _, reverse = net.inference(y, dt=.2, direction=-signs)
    # This random 4-D field has a slowly contracting head: residual errors
    # are amplified by the inverse-map conditioning, not bounded by atol alone.
    torch.testing.assert_close(y + reverse, x, atol=5e-5, rtol=1e-5)
    midpoint = (x + y) / 2
    torch.testing.assert_close(y - x, signs * .2 * _base_velocity(net.F, midpoint, eps),
                               atol=3e-6, rtol=3e-6)


def test_implicit_gradient_matches_differentiated_solver():
    net = model("midpoint")
    reference = copy.deepcopy(net)
    x = torch.randn(2, 2, 4).requires_grad_()
    xr = x.detach().clone().requires_grad_()
    h = torch.full((2, 2, 1), .15, requires_grad=True)
    hr = h.detach().clone().requires_grad_()
    signs = torch.tensor([[[1.], [-1.]], [[-1.], [1.]]])
    _, actual = net.inference(x, dt=h, direction=signs)
    m = xr
    for _ in range(30):
        m = xr + signs * hr / 2 * _base_velocity(reference.F, m, 1e-8)
    expected = signs * hr * _base_velocity(reference.F, m, 1e-8)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    probe = torch.randn_like(actual)
    actual_grads = torch.autograd.grad((actual * probe).sum(), (x, h, *net.F.parameters()), allow_unused=True)
    expected_grads = torch.autograd.grad((expected * probe).sum(), (xr, hr, *reference.F.parameters()), allow_unused=True)
    for a, b in zip(actual_grads, expected_grads):
        if a is None or b is None:
            assert a is b
        else:
            torch.testing.assert_close(a, b, atol=2e-5, rtol=3e-4)


@pytest.mark.parametrize("mode", ["potential_jet", "midpoint"])
def test_recognizer_cotangent_updates_both_shared_branches(mode):
    net = model(mode, lambdas={"BB": .25, "signed_g2orth": 1.})
    # Keep this multi-step derivative test on a contracting branch. Separate
    # tests check the raw initialization and explicit solver failure behavior.
    with torch.no_grad():
        net.F.dir_linear.weight.mul_(3)
    net.train()
    for direction in (1, -1):
        net.zero_grad(set_to_none=True)
        _, _, endpoint, penalty, _ = net(torch.randn(2, 4), torch.tensor([0, 2]), .1, direction=direction)
        # A surrogate endpoint cotangent tests the same potential backward
        # boundary used by TraversalTrainer, without loading a generator.
        loss = (endpoint * torch.randn_like(endpoint)).sum() + penalty
        loss.backward()
        for parameter in (net.F.dir_linear.weight, net.F.fc1.weight, net.F.fc3.weight):
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
            assert parameter.grad.norm() > 0


def test_jet_is_a_potential_and_improves_small_step_reversal():
    net = model("potential_jet")
    euler = model("euler")
    euler.load_state_dict(net.state_dict())
    x = torch.randn(2, 2, 4).requires_grad_()
    st, _, _, _ = net._per_step(x, dt=.1, direction=-1)
    direct = torch.autograd.grad(st.f().sum(), x, retain_graph=True)[0]
    torch.testing.assert_close(direct, st.f_grad(), atol=1e-7, rtol=1e-5)

    def error(n, dt):
        _, f = n.inference(x, dt=dt)
        _, r = n.inference(x + f, dt=dt, direction=-1)
        return (f + r).norm().item()

    jet_error, euler_error = error(net, .1), error(euler, .1)
    assert jet_error < euler_error * .2
    assert error(net, .05) < jet_error * .3


def test_shape_contract_and_signed_dt_compatibility():
    x = torch.randn(3, 2, 4)
    for value in (1., torch.ones(3), torch.ones(3, 1), torch.ones(1, 2), torch.ones(3, 2, 1)):
        assert broadcast_bk(value, x, "test").shape == (3, 2, 1)
    net = model("midpoint")
    _, a = net.inference(x, dt=-.1)
    _, b = net.inference(x, dt=.1, direction=-1)
    torch.testing.assert_close(a, b)
    with pytest.raises(ValueError, match="direction"):
        net.inference(x, direction=0)


def test_solver_failure_is_explicit_and_higher_order_losses_rejected():
    net = model("midpoint", midpoint_max_iter=1)
    with pytest.raises(RuntimeError, match="did not converge"):
        net.inference(torch.randn(2, 4), dt=.2)
    with pytest.raises(ValueError, match="supports"):
        model("midpoint", lambdas={"fconvex": 1.})


@pytest.mark.parametrize("mode", ["euler", "potential_jet", "midpoint"])
def test_random_signs_do_not_hide_duplicate_semantic_directions(mode):
    net = model(mode, lambdas={"signed_g2orth": 1.})
    with torch.no_grad():
        net.F.c.zero_()
        net.F.dir_linear.weight[1].copy_(net.F.dir_linear.weight[0])
    x = torch.randn(2, 2, 4)
    _, _, canonical, _ = net._per_step(x, dt=.1, direction=1)
    _, _, mixed, _ = net._per_step(x, dt=.1, direction=torch.tensor([[1., -1.]]))
    torch.testing.assert_close(mixed, canonical)
    assert mixed.min() > .9
