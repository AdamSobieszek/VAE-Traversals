"""Unit tests for B64 compile-optimization ops (shared modconv, FIR compose, epilogue)."""

from __future__ import annotations

import os
import sys

import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.StyleGAN2_mps.torch_utils.ops import upfirdn2d
from models.StyleGAN2_mps.torch_utils.ops.conv2d_resample import conv2d_resample
from models.StyleGAN2_mps.torch_utils.ops.fused_epilogue import (
    fused_epilogue_composite,
    fused_epilogue,
    triton_epilogue_available,
)
from models.StyleGAN2_mps.torch_utils.ops.fused_up_modconv import (
    PreparedFusedUpConv,
    expanded_up2_modconv,
    polyphase_up2_modconv,
    synthesis_up2_fir_padding,
)
from models.StyleGAN2_mps.torch_utils.ops.native_backend import use_native_ops
from models.StyleGAN2_mps.torch_utils.ops.shared_modconv import modulated_conv2d_shared
from models.StyleGAN2_mps.torch_utils.ops.upfirdn2d_native import _filter_2d
from models.StyleGAN2_mps.model import modulated_conv2d


def _require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_shared_modconv_matches_unfused(dtype):
    _require_cuda()
    device = torch.device("cuda")
    B, I, O, H, k = 4, 8, 16, 16, 3
    torch.manual_seed(0)
    x = torch.randn(B, I, H, H, device=device, dtype=dtype)
    w = torch.randn(O, I, k, k, device=device, dtype=dtype)
    styles = torch.randn(B, I, device=device, dtype=dtype) * 0.5 + 1.0
    fir = upfirdn2d.setup_filter([1, 3, 3, 1]).to(device)

    with use_native_ops():
        ref = modulated_conv2d(
            x, w, styles, up=1, padding=1, resample_filter=fir, fused_modconv=False
        )
        got = modulated_conv2d_shared(
            x, w, styles, up=1, padding=1, resample_filter=fir, demodulate=True
        )
    max_err = (ref.float() - got.float()).abs().max().item()
    tol = 5e-3 if dtype == torch.float16 else 1e-5
    assert max_err < tol, max_err


def test_shared_modconv_ws_grad():
    _require_cuda()
    device = torch.device("cuda")
    B, I, O, H = 2, 4, 8, 8
    x = torch.randn(B, I, H, H, device=device, requires_grad=False)
    w = torch.randn(O, I, 3, 3, device=device)
    styles = torch.randn(B, I, device=device, requires_grad=True)
    y = modulated_conv2d_shared(x, w, styles, up=1, padding=1, demodulate=True)
    loss = y.square().mean()
    (g,) = torch.autograd.grad(loss, styles)
    assert g.shape == styles.shape
    assert torch.isfinite(g).all()


@pytest.mark.parametrize("mode", ["polyphase", "expanded"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_fir_compose_up2_vs_conv2d_resample(mode, dtype):
    _require_cuda()
    device = torch.device("cuda")
    torch.manual_seed(1)
    B, I, O, H = 2, 4, 8, 8
    x = torch.randn(B, I, H, H, device=device, dtype=dtype)
    weight = torch.randn(O, I, 3, 3, device=device, dtype=dtype)
    fir = upfirdn2d.setup_filter([1, 3, 3, 1]).to(device)
    fir_pad = synthesis_up2_fir_padding()
    assert fir_pad == (1, 1, 1, 1)

    with use_native_ops():
        ref = conv2d_resample(x, weight, f=fir, up=2, padding=1, flip_weight=False)
        fir_dense, _ = _filter_2d(
            fir, device=device, dtype=torch.float32, gain=4.0, force_dense=True
        )
        fir_d = fir_dense.to(dtype)
        if mode == "polyphase":
            got = polyphase_up2_modconv(x, weight, fir_d)
        else:
            got = expanded_up2_modconv(x, weight, fir_d, fir_padding=fir_pad)
    assert got.shape == ref.shape, (got.shape, ref.shape)
    max_err = (ref.float() - got.float()).abs().max().item()
    tol = 5e-2 if dtype == torch.float16 else 5e-4
    assert max_err < tol, max_err


def test_fir_compose_impulse_alignment():
    _require_cuda()
    device = torch.device("cuda")
    B, I, O, H = 1, 1, 1, 4
    x = torch.zeros(B, I, H, H, device=device)
    x[0, 0, 1, 1] = 1.0
    weight = torch.zeros(O, I, 3, 3, device=device)
    weight[0, 0, 1, 1] = 1.0
    fir = upfirdn2d.setup_filter([1, 3, 3, 1]).to(device)
    fir_dense, _ = _filter_2d(
        fir, device=device, dtype=torch.float32, gain=4.0, force_dense=True
    )
    with use_native_ops():
        ref = conv2d_resample(x, weight, f=fir, up=2, padding=1, flip_weight=False)
        got = polyphase_up2_modconv(x, weight, fir_dense)
    assert got.shape == ref.shape
    assert (ref - got).abs().max().item() < 1e-5


@pytest.mark.parametrize("mode", ["polyphase", "expanded"])
def test_prepared_fir_compose_matches_reference_and_gradients(mode):
    _require_cuda()
    device = torch.device("cuda")
    torch.manual_seed(2)
    batch, in_channels, out_channels, size = 2, 3, 5, 8
    x_ref = torch.randn(
        batch, in_channels, size, size, device=device, requires_grad=True
    )
    x_got = x_ref.detach().clone().requires_grad_(True)
    styles_ref = torch.randn(
        batch, in_channels, device=device, requires_grad=True
    )
    styles_got = styles_ref.detach().clone().requires_grad_(True)
    weight = torch.randn(out_channels, in_channels, 3, 3, device=device)
    fir = upfirdn2d.setup_filter([1, 3, 3, 1]).to(device)
    prepared = PreparedFusedUpConv(weight, fir, mode=mode).to(device)

    with use_native_ops():
        ref = conv2d_resample(
            x_ref * styles_ref.reshape(batch, in_channels, 1, 1),
            weight,
            f=fir,
            up=2,
            padding=1,
            flip_weight=False,
        )
        got = prepared(
            x_got * styles_got.reshape(batch, in_channels, 1, 1)
        )
    grad_ref = torch.autograd.grad(ref.square().mean(), (x_ref, styles_ref))
    grad_got = torch.autograd.grad(got.square().mean(), (x_got, styles_got))

    torch.testing.assert_close(got, ref, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(grad_got[0], grad_ref[0], rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(grad_got[1], grad_ref[1], rtol=2e-5, atol=2e-5)


def test_fused_epilogue_matches_bias_act_chain():
    _require_cuda()
    device = torch.device("cuda")
    from models.StyleGAN2_mps.torch_utils.ops import bias_act

    B, C, H = 2, 8, 16
    x = torch.randn(B, C, H, H, device=device, dtype=torch.float16)
    bias = torch.randn(C, device=device, dtype=torch.float16)
    noise = torch.randn(B, 1, H, H, device=device, dtype=torch.float16) * 0.1
    dcoef = torch.rand(B, C, device=device, dtype=torch.float16) * 0.5 + 0.5
    gain = 2**0.5
    clamp = 256.0

    y_comp = fused_epilogue_composite(
        x, bias, noise=noise, dcoef=dcoef, act="lrelu", alpha=0.2, gain=gain, clamp=clamp
    )
    y = x * dcoef.reshape(B, C, 1, 1) + noise + bias.reshape(1, C, 1, 1)
    with use_native_ops():
        y_ref = bias_act.bias_act(y, None, act="lrelu", gain=gain, clamp=clamp)
    max_err = (y_comp.float() - y_ref.float()).abs().max().item()
    assert max_err < 2e-3, max_err


def test_fused_epilogue_grad_to_dcoef():
    _require_cuda()
    device = torch.device("cuda")
    x = torch.randn(2, 4, 8, 8, device=device, requires_grad=True)
    bias = torch.randn(4, device=device)
    dcoef = torch.randn(2, 4, device=device, requires_grad=True)
    y = fused_epilogue(x, bias, dcoef=dcoef, act="lrelu", gain=1.0, clamp=None)
    loss = y.square().mean()
    gx, gd = torch.autograd.grad(loss, (x, dcoef))
    assert torch.isfinite(gx).all() and torch.isfinite(gd).all()


def test_triton_epilogue_optional():
    _require_cuda()
    if not triton_epilogue_available():
        pytest.skip("triton epilogue unavailable")
    device = torch.device("cuda")
    x = torch.randn(2, 8, 16, 16, device=device, dtype=torch.float16).contiguous()
    bias = torch.randn(8, device=device, dtype=torch.float16)
    y = fused_epilogue(
        x, bias, act="lrelu", gain=1.414, clamp=256.0, use_triton=True
    )
    y_ref = fused_epilogue_composite(x, bias, act="lrelu", gain=1.414, clamp=256.0)
    max_err = (y.float() - y_ref.float()).abs().max().item()
    assert max_err < 5e-2, max_err


def test_synthesis_up2_padding_constant():
    assert synthesis_up2_fir_padding() == (1, 1, 1, 1)
