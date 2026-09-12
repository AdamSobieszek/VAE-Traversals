"""Compile-friendly native PyTorch alternatives to StyleGAN's bias_act op.

This is an isolated prototype and is not wired into the generator.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F


_DEFAULT_ALPHA = {
    "linear": 0.0,
    "relu": 0.0,
    "lrelu": 0.2,
    "tanh": 0.0,
    "sigmoid": 0.0,
    "elu": 0.0,
    "selu": 0.0,
    "softplus": 0.0,
    "swish": 0.0,
}

_DEFAULT_GAIN = {
    "linear": 1.0,
    "relu": math.sqrt(2.0),
    "lrelu": math.sqrt(2.0),
    "tanh": 1.0,
    "sigmoid": 1.0,
    "elu": 1.0,
    "selu": 1.0,
    "softplus": 1.0,
    "swish": math.sqrt(2.0),
}


def _resolve_parameters(
    act: str,
    alpha: Optional[float],
    gain: Optional[float],
    clamp: Optional[float],
) -> tuple[float, float, Optional[float]]:
    if act not in _DEFAULT_GAIN:
        raise ValueError(f"Unsupported activation {act!r}")
    alpha = float(_DEFAULT_ALPHA[act] if alpha is None else alpha)
    gain = float(_DEFAULT_GAIN[act] if gain is None else gain)
    if clamp is not None:
        clamp = float(clamp)
        if clamp < 0:
            raise ValueError(f"Clamp must be non-negative, got {clamp}")
    return alpha, gain, clamp


def _bias_view(x: torch.Tensor, b: torch.Tensor, dim: int) -> torch.Tensor:
    if b.ndim != 1:
        raise ValueError(f"Bias must be one-dimensional, got shape {tuple(b.shape)}")
    if not 0 <= dim < x.ndim:
        raise ValueError(f"Bias dimension {dim} is invalid for {x.ndim}D input")
    if b.shape[0] != x.shape[dim]:
        raise ValueError(
            f"Bias length {b.shape[0]} does not match input dimension {x.shape[dim]}"
        )
    return b.reshape([b.shape[0] if axis == dim else 1 for axis in range(x.ndim)])


def _activate(x: torch.Tensor, act: str, alpha: float) -> torch.Tensor:
    if act == "linear":
        return x
    if act == "relu":
        return F.relu(x)
    if act == "lrelu":
        return F.leaky_relu(x, negative_slope=alpha)
    if act == "tanh":
        return torch.tanh(x)
    if act == "sigmoid":
        return torch.sigmoid(x)
    if act == "elu":
        return F.elu(x)
    if act == "selu":
        return F.selu(x)
    if act == "softplus":
        return F.softplus(x)
    if act == "swish":
        return torch.sigmoid(x) * x
    raise AssertionError(f"Unreachable activation {act!r}")


def _activate_inplace(x: torch.Tensor, act: str, alpha: float) -> torch.Tensor:
    if act == "linear":
        return x
    if act == "relu":
        return F.relu_(x)
    if act == "lrelu":
        return F.leaky_relu_(x, negative_slope=alpha)
    if act == "tanh":
        return x.tanh_()
    if act == "sigmoid":
        return x.sigmoid_()
    if act == "elu":
        return F.elu_(x)
    if act == "selu":
        return F.selu_(x)
    # PyTorch has no in-place softplus, and in-place swish would overwrite a
    # value needed by its multiplication. These paths still avoid later
    # gain/clamp allocations.
    if act == "softplus":
        return F.softplus(x)
    if act == "swish":
        return torch.sigmoid(x) * x
    raise AssertionError(f"Unreachable activation {act!r}")


def bias_act_native(
    x: torch.Tensor,
    b: Optional[torch.Tensor] = None,
    dim: int = 1,
    act: str = "linear",
    alpha: Optional[float] = None,
    gain: Optional[float] = None,
    clamp: Optional[float] = None,
) -> torch.Tensor:
    """Gradient-safe pure-PyTorch equivalent of StyleGAN's reference op."""
    alpha, gain, clamp = _resolve_parameters(act, alpha, gain, clamp)
    if b is not None:
        x = x + _bias_view(x, b, dim)
    x = _activate(x, act, alpha)
    if gain != 1.0:
        x = x * gain
    if clamp is not None:
        x = x.clamp(-clamp, clamp)
    return x


def bias_act_native_inference(
    x: torch.Tensor,
    b: Optional[torch.Tensor] = None,
    dim: int = 1,
    act: str = "linear",
    alpha: Optional[float] = None,
    gain: Optional[float] = None,
    clamp: Optional[float] = None,
) -> torch.Tensor:
    """Allocation-reduced inference-only equivalent using in-place epilogues.

    The input tensor is never mutated: adding bias creates a new tensor, and a
    clone is made when no bias is supplied but an in-place operation is needed.
    This function intentionally rejects autograd-tracked inputs because later
    in-place gain/clamp operations invalidate activation backward state.
    """
    if torch.is_grad_enabled() and (
        x.requires_grad or (b is not None and b.requires_grad)
    ):
        raise RuntimeError(
            "bias_act_native_inference is inference-only; use bias_act_native "
            "when gradients are required"
        )

    alpha, gain, clamp = _resolve_parameters(act, alpha, gain, clamp)
    has_epilogue = act != "linear" or gain != 1.0 or clamp is not None
    if b is not None:
        x = x + _bias_view(x, b, dim)
    elif has_epilogue:
        x = x.clone()

    x = _activate_inplace(x, act, alpha)
    if gain != 1.0:
        x.mul_(gain)
    if clamp is not None:
        x.clamp_(-clamp, clamp)
    return x


def bias_act_native_homogeneous_inference(
    x: torch.Tensor,
    scaled_bias: Optional[torch.Tensor] = None,
    dim: int = 1,
    act: str = "linear",
    alpha: Optional[float] = None,
    gain: Optional[float] = None,
    clamp: Optional[float] = None,
) -> torch.Tensor:
    """Faster inference path for positive-homogeneous StyleGAN activations.

    ``scaled_bias`` must equal ``bias * gain``. For linear, ReLU, and leaky
    ReLU with positive gain,

    ``gain * act(x + bias) == act(gain * x + gain * bias)``.

    Supplying the pre-scaled frozen bias lets us combine bias addition and
    activation gain as ``act(x * gain + scaled_bias)``, removing a full
    activation-sized multiplication versus ``gain * act(x + bias)``. Floating-
    point operation order differs from the reference, so outputs are
    numerically equivalent rather than bit-identical.
    """
    if torch.is_grad_enabled() and (
        x.requires_grad
        or (scaled_bias is not None and scaled_bias.requires_grad)
    ):
        raise RuntimeError(
            "bias_act_native_homogeneous_inference is inference-only"
        )
    if act not in ("linear", "relu", "lrelu"):
        raise ValueError(
            f"Activation {act!r} is not supported by the homogeneous path"
        )

    alpha, gain, clamp = _resolve_parameters(act, alpha, gain, clamp)
    if gain <= 0:
        raise ValueError(f"Homogeneous path requires positive gain, got {gain}")

    if scaled_bias is not None:
        # Prefer mul+add over torch.add(..., alpha=) — Python float alphas
        # frequently become CPU scalar tensors and skip CUDA Graph capture.
        x = x * gain + _bias_view(x, scaled_bias, dim)
    elif gain != 1.0:
        x = x * gain
    elif act != "linear" or clamp is not None:
        x = x.clone()

    x = _activate_inplace(x, act, alpha)
    if clamp is not None:
        x.clamp_(-clamp, clamp)
    return x


def bias_act_native_compile_friendly(
    x: torch.Tensor,
    b: Optional[torch.Tensor] = None,
    dim: int = 1,
    act: str = "linear",
    alpha: Optional[float] = None,
    gain: Optional[float] = None,
    clamp: Optional[float] = None,
    *,
    scaled_bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Out-of-place bias+act intended for Inductor / CUDA Graphs.

    Avoids in-place kernels and ``torch.add(..., alpha=python_float)``. Prefer
    ``scaled_bias = bias * gain`` for the homogeneous StyleGAN path.
    """
    alpha, gain, clamp = _resolve_parameters(act, alpha, gain, clamp)
    if scaled_bias is not None:
        x = x * gain + _bias_view(x, scaled_bias, dim)
        x = _activate(x, act, alpha)
    elif b is not None:
        x = x + _bias_view(x, b, dim)
        x = _activate(x, act, alpha)
        if gain != 1.0:
            x = x * gain
    else:
        x = _activate(x, act, alpha)
        if gain != 1.0:
            x = x * gain
    if clamp is not None:
        x = x.clamp(-clamp, clamp)
    return x