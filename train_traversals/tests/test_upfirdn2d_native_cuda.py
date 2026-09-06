"""Additional CUDA / variant coverage for native upfirdn2d."""

from __future__ import annotations

import pytest
import torch

from models.StyleGAN2_mps.torch_utils.ops.upfirdn2d import (
    _upfirdn2d_ref,
    setup_filter,
    upfirdn2d,
)
from models.StyleGAN2_mps.torch_utils.ops.upfirdn2d_native import (
    clear_weight_cache,
    upfirdn2d_native,
)


@pytest.mark.parametrize("weight_mode", ("expand", "repeat", "contiguous", "cached"))
@pytest.mark.parametrize(
    "up,down,padding,gain",
    (
        (1, 1, (2, 1, 2, 1), 1.0),
        (1, 2, (1, 1, 1, 1), 1.0),
        (2, 1, (2, 1, 2, 1), 4.0),
    ),
)
def test_weight_modes_match_reference(weight_mode, up, down, padding, gain):
    clear_weight_cache()
    f = setup_filter([1, 3, 3, 1])
    x = torch.randn(2, 5, 16, 17, dtype=torch.float64)
    expected = _upfirdn2d_ref(x, f, up=up, down=down, padding=padding, gain=gain)
    actual = upfirdn2d_native(
        x, f, up=up, down=down, padding=padding, gain=gain, weight_mode=weight_mode
    )
    assert torch.allclose(actual, expected, rtol=1e-7, atol=1e-8)


@pytest.mark.parametrize("padding", ((2, 1, 2, 1), (1, 1, 1, 1), (0, 0, 0, 0), (-1, 2, 0, 3)))
@pytest.mark.parametrize("flip_filter", (False, True))
def test_polyphase_up2_matches_reference(padding, flip_filter):
    f = setup_filter([1, 3, 3, 1])
    x = torch.randn(2, 3, 9, 11, dtype=torch.float64)
    expected = _upfirdn2d_ref(
        x, f, up=2, padding=padding, gain=4.0, flip_filter=flip_filter
    )
    actual = upfirdn2d_native(
        x,
        f,
        up=2,
        padding=padding,
        gain=4.0,
        flip_filter=flip_filter,
        impl_variant="polyphase",
    )
    assert actual.shape == expected.shape
    assert torch.allclose(actual, expected, rtol=1e-7, atol=1e-8)


def test_separable_mode_matches_reference():
    f = setup_filter([1, 2, 1], separable=True)
    x = torch.randn(2, 4, 16, 16, dtype=torch.float64)
    expected = _upfirdn2d_ref(x, f, down=2, padding=(1, 1, 1, 1), gain=1.0)
    dense = upfirdn2d_native(
        x, f, down=2, padding=(1, 1, 1, 1), gain=1.0, separable_mode="dense"
    )
    separable = upfirdn2d_native(
        x, f, down=2, padding=(1, 1, 1, 1), gain=1.0, separable_mode="separable"
    )
    assert torch.allclose(dense, expected, rtol=1e-6, atol=1e-7)
    assert torch.allclose(separable, expected, rtol=1e-10, atol=1e-10)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", (torch.float32, torch.float16))
def test_native_matches_legacy_cuda(dtype):
    device = torch.device("cuda")
    f = setup_filter([1, 3, 3, 1], device=device)
    x = torch.randn(2, 8, 32, 32, device=device, dtype=dtype)
    expected = upfirdn2d(x, f, up=2, padding=(2, 1, 2, 1), gain=4, impl="cuda")
    actual = upfirdn2d_native(x, f, up=2, padding=(2, 1, 2, 1), gain=4)
    tol = 3e-3 if dtype == torch.float16 else 2e-5
    assert torch.allclose(actual, expected, rtol=tol, atol=tol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_impl_native_dispatch():
    device = torch.device("cuda")
    f = setup_filter([1, 3, 3, 1], device=device)
    x = torch.randn(1, 4, 16, 16, device=device)
    ref = upfirdn2d(x, f, up=1, padding=(1, 1, 1, 1), gain=1, impl="ref")
    native = upfirdn2d(x, f, up=1, padding=(1, 1, 1, 1), gain=1, impl="native")
    assert torch.allclose(native, ref, rtol=2e-5, atol=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_channels_last_native_cuda():
    device = torch.device("cuda")
    f = setup_filter([1, 3, 3, 1], device=device)
    x = torch.randn(2, 16, 32, 32, device=device).contiguous(
        memory_format=torch.channels_last
    )
    expected = _upfirdn2d_ref(x.contiguous(), f, down=2, padding=(1, 1, 1, 1))
    actual = upfirdn2d_native(x, f, down=2, padding=(1, 1, 1, 1))
    assert torch.allclose(actual.contiguous(), expected, rtol=2e-5, atol=2e-5)
