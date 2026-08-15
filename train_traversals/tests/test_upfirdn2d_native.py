"""Correctness tests for the compile-friendly native upfirdn2d prototype."""

from __future__ import annotations

import pytest
import torch

from models.StyleGAN2_mps.torch_utils.ops.upfirdn2d import (
    _upfirdn2d_ref,
    setup_filter,
)
from models.StyleGAN2_mps.torch_utils.ops.upfirdn2d_native import (
    upfirdn2d_native,
)


CASES = (
    # up, down, padding, flip_filter, gain
    (1, 1, 0, False, 1.0),
    (1, 1, (-1, 2, 0, 3), False, 1.7),
    (1, 2, (1, 1, 1, 1), False, 1.0),
    (1, (2, 3), (2, 0, 1, -1), True, 0.75),
    (2, 1, (2, 1, 2, 1), False, 4.0),
    (2, 2, (3, 0, 1, 2), True, 1.5),
    ((2, 3), 1, (4, -1, 3, -2), False, 1.7),
)


@pytest.mark.parametrize("filter_kind", ("none", "dense", "separable"))
@pytest.mark.parametrize("up,down,padding,flip_filter,gain", CASES)
def test_native_matches_reference(
    filter_kind,
    up,
    down,
    padding,
    flip_filter,
    gain,
):
    if filter_kind == "none":
        f = None
    elif filter_kind == "dense":
        f = setup_filter([1, 3, 3, 1])
    else:
        f = setup_filter([1, 2, 1], separable=True)

    x = torch.randn(2, 3, 9, 11, dtype=torch.float64)
    expected = _upfirdn2d_ref(
        x,
        f,
        up=up,
        down=down,
        padding=padding,
        flip_filter=flip_filter,
        gain=gain,
    )
    actual = upfirdn2d_native(
        x,
        f,
        up=up,
        down=down,
        padding=padding,
        flip_filter=flip_filter,
        gain=gain,
    )

    assert actual.shape == expected.shape
    # A 1D separable filter is evaluated as one 2D convolution in the native
    # path rather than two passes, so accumulation order can differ slightly.
    assert torch.allclose(actual, expected, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("use_strided_down", (False, True))
def test_native_downsample_modes_are_exact(use_strided_down):
    x = torch.randn(2, 5, 16, 17)
    f = setup_filter([1, 3, 3, 1])
    expected = _upfirdn2d_ref(
        x, f, down=2, padding=(1, 1, 1, 1), gain=0.5
    )
    actual = upfirdn2d_native(
        x,
        f,
        down=2,
        padding=(1, 1, 1, 1),
        gain=0.5,
        use_strided_down=use_strided_down,
    )
    assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_native_input_gradient_matches_reference():
    f = setup_filter([1, 3, 3, 1])
    x_ref = torch.randn(1, 2, 8, 9, dtype=torch.float64, requires_grad=True)
    x_native = x_ref.detach().clone().requires_grad_(True)

    ref_loss = _upfirdn2d_ref(
        x_ref, f, up=2, down=1, padding=(2, 1, 2, 1), gain=4
    ).square().mean()
    native_loss = upfirdn2d_native(
        x_native, f, up=2, down=1, padding=(2, 1, 2, 1), gain=4
    ).square().mean()
    ref_loss.backward()
    native_loss.backward()

    assert torch.allclose(native_loss, ref_loss, rtol=1e-10, atol=1e-10)
    assert torch.allclose(x_native.grad, x_ref.grad, rtol=1e-10, atol=1e-10)
