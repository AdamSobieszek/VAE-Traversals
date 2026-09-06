"""Correctness for prepared FP16 FIR / bias_act modules."""

from __future__ import annotations

import math

import pytest
import torch

from models.StyleGAN2_mps.torch_utils.ops.bias_act_native import bias_act_native
from models.StyleGAN2_mps.torch_utils.ops.prepared_ops import PreparedBiasAct, PreparedFIR
from models.StyleGAN2_mps.torch_utils.ops.upfirdn2d import setup_filter
from models.StyleGAN2_mps.torch_utils.ops.upfirdn2d_native import upfirdn2d_native


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


@pytest.mark.parametrize("up,down,padding,gain", [(2, 1, 1, 4.0), (1, 1, 1, 1.0), (1, 2, 1, 1.0)])
@pytest.mark.parametrize("dtype", (torch.float16, torch.float32))
def test_prepared_fir_matches_native(up, down, padding, gain, dtype):
    device = torch.device("cuda")
    f = setup_filter([1, 3, 3, 1]).to(device)
    x = torch.randn(2, 16, 32, 32, device=device, dtype=dtype)
    prep = PreparedFIR(
        f,
        channels=16,
        up=up,
        down=down,
        padding=padding,
        gain=gain,
        dtype=dtype,
        device=device,
    ).eval()
    with torch.inference_mode():
        y_prep = prep(x)
        y_nat = upfirdn2d_native(x, f, up=up, down=down, padding=padding, gain=gain)
    tol = 3e-3 if dtype == torch.float16 else 2e-5
    assert torch.allclose(y_prep.float(), y_nat.float(), atol=tol, rtol=tol)


@pytest.mark.parametrize("act,clamp", [("lrelu", 256.0), ("linear", 256.0), ("relu", None)])
def test_prepared_bias_act_matches_native(act, clamp):
    device = torch.device("cuda")
    dtype = torch.float16
    x = torch.randn(2, 64, 32, 32, device=device, dtype=dtype)
    b = torch.randn(64, device=device, dtype=dtype)
    gain = math.sqrt(2.0) if act != "linear" else 1.0
    prep = PreparedBiasAct(
        b, act=act, gain=gain, clamp=clamp, dtype=dtype, device=device
    ).eval()
    with torch.inference_mode():
        y_prep = prep(x)
        y_nat = bias_act_native(x, b, act=act, gain=gain, clamp=clamp)
    # Homogeneous reorder vs reference path.
    assert torch.allclose(y_prep.float(), y_nat.float(), atol=5e-2, rtol=5e-2)


def test_prepared_fir_compiles():
    device = torch.device("cuda")
    f = setup_filter([1, 3, 3, 1]).to(device)
    x = torch.randn(1, 8, 16, 16, device=device, dtype=torch.float16)
    prep = PreparedFIR(
        f, channels=8, up=2, padding=1, gain=4.0, dtype=torch.float16, device=device
    ).eval()
    compiled = torch.compile(prep, fullgraph=True, mode="default")
    with torch.inference_mode():
        y = compiled(x)
    assert y.shape == (1, 8, 31, 31)
