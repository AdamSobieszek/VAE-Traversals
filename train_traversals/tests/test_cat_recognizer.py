"""Small contract and gradient tests; no training or pretrained encoders."""
import argparse
import copy

import pytest
import torch
from torch.nn import functional as F

from lib.config import RECONSTRUCTOR_TYPES
from lib.recognizer import Recognizer, add_recognizer_arguments, recognizer_options


@pytest.fixture(scope='module', autouse=True)
def limited_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def tiny(**options):
    return Recognizer('CAT', 3, hidden_size=48, depth=2, num_heads=3, **options)


@pytest.mark.parametrize('kind', ['LeNet', 'ResNet'])
def test_legacy_recognizers(kind):
    r = Recognizer(kind, 3).train()
    x, y = torch.randn(2, 3, 64, 64), torch.randn(2, 3, 64, 64)
    z, aux = r(x, y)
    assert z.shape == (2, 3) and aux is None
    F.cross_entropy(z, torch.tensor([0, 2])).backward()
    head = r.path_indices if kind == 'ResNet' else r.path_indices[-1]
    assert head.weight.grad is not None
    r.eval()
    torch.testing.assert_close(r(x, y, antisymmetric=True)[0], r.antisymmetric_logits(x, y))


def test_default_constructor_and_count():
    r = Recognizer('CAT', 32)
    assert 'CAT' in RECONSTRUCTOR_TYPES
    assert sum(p.numel() for p in r.parameters()) == 1_806_752
    assert not r.pair_encoder.gradient_checkpointing
    assert len(r.pair_encoder.blocks) == 4
    assert r.pair_encoder.num_scales == 2
    z, aux = r(torch.randn(1, 3, 32, 32), torch.randn(1, 3, 32, 32))
    assert z.shape == (1, 32) and aux is None


def test_configuration():
    parser = argparse.ArgumentParser()
    parser.add_argument('--recognizer-type', default='CAT')
    add_recognizer_arguments(parser)
    args = parser.parse_args(['--recognizer-hidden-size', '48', '--recognizer-depth', '2',
                              '--recognizer-num-heads', '3', '--recognizer-num-scales', '3',
                              '--recognizer-patch-size', '2', '--recognizer-checkpointing',
                              '--recognizer-unfused-attention'])
    r = Recognizer(args.recognizer_type, 3, **recognizer_options(args))
    assert r.pair_encoder.num_scales == 3
    assert r.pair_encoder.gradient_checkpointing
    assert not r.pair_encoder.blocks[0].attn.fused_attn
    args.recognizer_type = 'LeNet'
    assert recognizer_options(args) == {}


@pytest.mark.parametrize('num_scales', [1, 2, 3, 4])
@pytest.mark.parametrize('channels,resolution,pool_size', [(1, 32, 1), (3, 64, 2), (4, 128, 4)])
def test_pyramid_shapes_pooling_and_input_channels(num_scales, channels, resolution, pool_size):
    r = tiny(channels=channels, num_scales=num_scales, pool_size=pool_size)
    x, y = torch.randn(2, channels, resolution, resolution), torch.randn(2, channels, resolution, resolution)
    z, aux = r(x, y)
    assert z.shape == (2, 3) and aux is None
    paired = torch.cat([x, y], dim=1)
    if pool_size > 1:
        paired = F.avg_pool2d(paired, pool_size)
    pyramid = r.pair_encoder.make_pyramid(paired)
    assert [p.shape[-1] for p in pyramid] == [32 // 2**i for i in range(num_scales)]
    expected, per_scale = r.pair_encoder.forward_pyramid(pyramid, return_per_scale=True)
    torch.testing.assert_close(z, expected)
    torch.testing.assert_close(z, per_scale.mean(1))


@pytest.mark.parametrize('whitened', [False, True])
def test_antisymmetry_and_all_k_batching(whitened):
    torch.manual_seed(7)
    r = tiny().train()
    x, y = torch.randn(2, 3, 32, 32), torch.randn(6, 3, 32, 32)
    x_full = x[:, None].expand(2, 3, 3, 32, 32).flatten(0, 1)
    method = r.whitened_antisymmetric_logits if whitened else r.antisymmetric_logits
    actual = method(x, y)
    expected = (r(2*x_full-y, y)[0] - r(2*y-x_full, x_full)[0]
                if whitened else r(x_full, y)[0] - r(y, x_full)[0])
    assert actual.shape == (6, 3)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(actual, method(x_full, y), atol=0, rtol=0)
    torch.testing.assert_close(actual, -method(y, x_full), atol=0, rtol=0)
    torch.testing.assert_close(method(x, x), torch.zeros(2, 3), atol=0, rtol=0)
    labels = torch.arange(3).repeat(2)
    torch.testing.assert_close(F.cross_entropy(actual, labels),
                               F.cross_entropy(actual, F.one_hot(labels, 3).float()))
    if not whitened:
        torch.testing.assert_close(r(x, y, antisymmetric=True)[0], actual)


def test_scales_cannot_exchange_information():
    r = tiny(num_scales=3).pair_encoder
    paired = torch.randn(1, 6, 32, 32)
    pyramid = r.make_pyramid(paired)
    _, before = r.forward_pyramid(pyramid, return_per_scale=True)
    pyramid[1] = torch.randn_like(pyramid[1])
    _, after = r.forward_pyramid(pyramid, return_per_scale=True)
    torch.testing.assert_close(before[:, [0, 2]], after[:, [0, 2]], atol=0, rtol=0)
    assert not torch.allclose(before[:, 1], after[:, 1])


@pytest.mark.parametrize('fused_attn', [False, True])
@pytest.mark.parametrize('gradient_checkpointing', [False, True])
@pytest.mark.parametrize('whitened', [False, True])
def test_input_and_parameter_gradients(fused_attn, gradient_checkpointing, whitened):
    r = tiny(fused_attn=fused_attn, gradient_checkpointing=gradient_checkpointing)
    x = torch.randn(2, 3, 32, 32, requires_grad=True)
    y = torch.randn(6, 3, 32, 32, requires_grad=True)
    z = r.whitened_antisymmetric_logits(x, y) if whitened else r.antisymmetric_logits(x, y)
    F.cross_entropy(z, torch.arange(3).repeat(2)).backward()
    for grad in (x.grad, y.grad):
        assert grad is not None and torch.isfinite(grad).all() and grad.norm() > 0
    grads = [p.grad for p in r.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert sum(g.square().sum() for g in grads) > 0
    assert r.pair_encoder.scale_embed.grad.norm(dim=1).min() > 0
    assert r.pair_encoder.cls_tokens.grad.flatten(1).norm(dim=1).min() > 0


def test_checkpoint_parity_and_second_derivative():
    torch.manual_seed(8)
    r = tiny(fused_attn=False)
    checked = copy.deepcopy(r)
    checked.pair_encoder.gradient_checkpointing = True
    x, y = torch.randn(1, 3, 32, 32), torch.randn(1, 3, 32, 32)
    results = []
    for model in (r, checked):
        a, b = x.clone().requires_grad_(), y.clone().requires_grad_()
        z = model.whitened_antisymmetric_logits(a, b)
        grad = torch.autograd.grad(z.square().sum(), b, create_graph=True)[0]
        second = torch.autograd.grad(grad.square().sum(), a)[0]
        assert torch.isfinite(second).all() and second.norm() > 0
        results.append((z, grad, second))
    for plain, ckpt in zip(*results):
        torch.testing.assert_close(plain, ckpt)


def test_state_dict_survives_resolution_changes():
    r = tiny()
    keys = set(r.state_dict())
    for size in (32, 64, 32):
        x = torch.randn(1, 3, size, size)
        r(x, x)
        assert set(r.state_dict()) == keys
    other = tiny()
    other.load_state_dict(r.state_dict(), strict=True)
    torch.testing.assert_close(r(x, x)[0], other(x, x)[0])


def test_inference_cache_remains_safe_for_training():
    r = tiny()
    x = torch.randn(1, 3, 32, 32)
    y = torch.randn_like(x, requires_grad=True)
    with torch.inference_mode():
        r(x, y)
    r(x, y)[0].square().sum().backward()
    assert y.grad is not None and torch.isfinite(y.grad).all() and y.grad.norm() > 0


@pytest.mark.parametrize('options', [dict(hidden_size=50), dict(num_scales=0), dict(depth=0),
                                    dict(patch_size=0), dict(mlp_ratio=0)])
def test_invalid_architecture(options):
    with pytest.raises(ValueError):
        Recognizer('CAT', 3, **options)


def test_invalid_resolution_is_not_silently_cropped():
    r = tiny(num_scales=4)
    with pytest.raises(ValueError, match='divisible'):
        r(torch.randn(1, 3, 48, 48), torch.randn(1, 3, 48, 48))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='MPS required')
@pytest.mark.parametrize('checkpointing,precision', [(False, None), (True, None),
                                                    (False, torch.float16), (False, torch.bfloat16)])
def test_default_mps_backward(checkpointing, precision):
    torch.manual_seed(9)
    r = Recognizer('CAT', 3, gradient_checkpointing=checkpointing).to('mps')
    x = torch.randn(1, 3, 32, 32, device='mps', requires_grad=True)
    y = torch.randn(3, 3, 32, 32, device='mps', requires_grad=True)
    with torch.autocast('mps', enabled=precision is not None, dtype=precision or torch.float16):
        z = r.whitened_antisymmetric_logits(x, y)
        loss = F.cross_entropy(z.float(), torch.arange(3, device='mps'))
    loss.backward()
    for grad in (x.grad, y.grad, r.pair_encoder.patch_proj.weight.grad):
        assert grad is not None and torch.isfinite(grad).all() and grad.norm() > 0
    torch.mps.synchronize()
    print('MPS', checkpointing, precision, 'loss', loss.item(), 'input grad norms',
          x.grad.norm().item(), y.grad.norm().item(), 'live tensor bytes', torch.mps.current_allocated_memory(),
          'driver bytes', torch.mps.driver_allocated_memory())
