"""Additional CUDA / compile coverage for native bias_act."""

from __future__ import annotations

import math

import pytest
import torch

from models.StyleGAN2_mps.torch_utils.ops.bias_act import _bias_act_ref, bias_act
from models.StyleGAN2_mps.torch_utils.ops.bias_act_native import (
    bias_act_native,
    bias_act_native_homogeneous_inference,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", (torch.float32, torch.float16))
@pytest.mark.parametrize("layout", ("contiguous", "channels_last"))
def test_native_matches_legacy_cuda(dtype, layout):
    device = torch.device("cuda")
    x = torch.randn(2, 32, 64, 64, device=device, dtype=dtype)
    if layout == "channels_last":
        x = x.contiguous(memory_format=torch.channels_last)
    b = torch.randn(32, device=device, dtype=dtype)
    expected = bias_act(
        x, b, act="lrelu", gain=math.sqrt(2), clamp=256, impl="cuda"
    )
    actual = bias_act_native(x, b, act="lrelu", gain=math.sqrt(2), clamp=256)
    tol = 3e-3 if dtype == torch.float16 else 2e-5
    assert torch.allclose(actual, expected, rtol=tol, atol=tol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_impl_native_dispatch():
    device = torch.device("cuda")
    x = torch.randn(2, 16, 8, 8, device=device)
    b = torch.randn(16, device=device)
    ref = bias_act(x, b, act="lrelu", gain=math.sqrt(2), clamp=256, impl="ref")
    native = bias_act(x, b, act="lrelu", gain=math.sqrt(2), clamp=256, impl="native")
    assert torch.allclose(native, ref, rtol=2e-5, atol=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_compiled_native_fullgraph():
    device = torch.device("cuda")
    x = torch.randn(2, 64, 32, 32, device=device)
    b = torch.randn(64, device=device)

    def fn(value):
        return bias_act_native(value, b, act="lrelu", gain=math.sqrt(2), clamp=256)

    compiled = torch.compile(fn, fullgraph=True, mode="default")
    with torch.inference_mode():
        y = compiled(x)
    expected = _bias_act_ref(x, b, act="lrelu", gain=math.sqrt(2), clamp=256)
    assert torch.allclose(y, expected, rtol=2e-5, atol=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_homogeneous_cached_bias_matches_reference():
    device = torch.device("cuda")
    x = torch.randn(2, 128, 64, 64, device=device)
    b = torch.randn(128, device=device)
    gain = math.sqrt(2)
    expected = _bias_act_ref(x, b, act="lrelu", gain=gain, clamp=256)
    with torch.inference_mode():
        actual = bias_act_native_homogeneous_inference(
            x, b * gain, act="lrelu", gain=gain, clamp=256
        )
    assert torch.allclose(actual, expected, rtol=2e-5, atol=2e-5)
