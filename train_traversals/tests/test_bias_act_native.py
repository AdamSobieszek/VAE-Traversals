"""Correctness tests for native PyTorch bias_act prototypes."""

from __future__ import annotations

import pytest
import torch

from models.StyleGAN2_mps.torch_utils.ops.bias_act import _bias_act_ref
from models.StyleGAN2_mps.torch_utils.ops.bias_act_native import (
    bias_act_native,
    bias_act_native_homogeneous_inference,
    bias_act_native_inference,
)


ACTIVATIONS = (
    "linear",
    "relu",
    "lrelu",
    "tanh",
    "sigmoid",
    "elu",
    "selu",
    "softplus",
    "swish",
)


@pytest.mark.parametrize("act", ACTIVATIONS)
@pytest.mark.parametrize("with_bias", (False, True))
def test_native_outputs_match_reference(act, with_bias):
    x = torch.randn(2, 3, 7, 5)
    b = torch.randn(3) if with_bias else None
    kwargs = {"act": act, "gain": 0.75, "clamp": 1.3}
    if act == "lrelu":
        kwargs["alpha"] = 0.3

    expected = _bias_act_ref(x, b, **kwargs)
    actual = bias_act_native(x, b, **kwargs)
    with torch.inference_mode():
        inference = bias_act_native_inference(x, b, **kwargs)

    assert torch.allclose(actual, expected, rtol=1e-6, atol=1e-7)
    assert torch.allclose(inference, expected, rtol=1e-6, atol=1e-7)


def test_nonstandard_bias_dimension():
    x = torch.randn(2, 3, 4, 5)
    b = torch.randn(4)
    expected = _bias_act_ref(
        x, b, dim=2, act="lrelu", gain=1.25, clamp=2.0
    )
    actual = bias_act_native(
        x, b, dim=2, act="lrelu", gain=1.25, clamp=2.0
    )
    with torch.inference_mode():
        inference = bias_act_native_inference(
            x, b, dim=2, act="lrelu", gain=1.25, clamp=2.0
        )
    assert torch.allclose(actual, expected)
    assert torch.allclose(inference, expected)


@pytest.mark.parametrize("act", ACTIVATIONS)
def test_gradient_safe_native_matches_reference(act):
    x_ref = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)
    b_ref = torch.randn(3, dtype=torch.float64, requires_grad=True)
    x_native = x_ref.detach().clone().requires_grad_(True)
    b_native = b_ref.detach().clone().requires_grad_(True)

    kwargs = {"act": act, "gain": 0.8, "clamp": 1.7}
    ref_loss = _bias_act_ref(x_ref, b_ref, **kwargs).square().mean()
    native_loss = bias_act_native(x_native, b_native, **kwargs).square().mean()
    ref_loss.backward()
    native_loss.backward()

    assert torch.allclose(native_loss, ref_loss, rtol=1e-10, atol=1e-10)
    assert torch.allclose(x_native.grad, x_ref.grad, rtol=1e-9, atol=1e-10)
    assert torch.allclose(b_native.grad, b_ref.grad, rtol=1e-9, atol=1e-10)


def test_inference_path_does_not_mutate_input():
    x = torch.randn(2, 3, 4, 5)
    original = x.clone()
    with torch.inference_mode():
        bias_act_native_inference(
            x, None, act="lrelu", gain=2.0, clamp=1.0
        )
    assert torch.equal(x, original)


def test_inference_path_rejects_autograd_inputs():
    x = torch.randn(2, 3, requires_grad=True)
    with pytest.raises(RuntimeError, match="inference-only"):
        bias_act_native_inference(x, act="lrelu")


@pytest.mark.parametrize("act", ("linear", "relu", "lrelu"))
def test_homogeneous_inference_matches_reference(act):
    x = torch.randn(2, 3, 7, 5)
    b = torch.randn(3)
    gain = 1.7
    expected = _bias_act_ref(
        x, b, act=act, gain=gain, clamp=2.5
    )
    with torch.inference_mode():
        actual = bias_act_native_homogeneous_inference(
            x,
            b * gain,
            act=act,
            gain=gain,
            clamp=2.5,
        )
    assert torch.allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_homogeneous_inference_rejects_nonhomogeneous_activation():
    x = torch.randn(2, 3)
    with pytest.raises(ValueError, match="not supported"):
        bias_act_native_homogeneous_inference(x, act="swish")
