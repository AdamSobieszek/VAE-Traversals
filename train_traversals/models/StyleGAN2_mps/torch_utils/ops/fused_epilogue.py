"""Compiler-visible post-convolution epilogue for StyleGAN synthesis layers.

Inductor-friendly composite:

  y = clamp(gain * lrelu(x * dcoef + noise + bias, alpha), ±clamp)

When ``dcoef`` is None (fused_modconv baked demod into weights), this reduces to
noise + bias + lrelu + gain + clamp.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

def fused_epilogue_composite(
    x: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    *,
    noise: Optional[torch.Tensor] = None,
    dcoef: Optional[torch.Tensor] = None,
    act: str = "lrelu",
    alpha: float = 0.2,
    gain: float = 1.0,
    clamp: Optional[float] = None,
) -> torch.Tensor:
    """Out-of-place epilogue intended for Inductor fusion."""
    if dcoef is not None:
        x = x * dcoef.to(dtype=x.dtype).reshape(dcoef.shape[0], dcoef.shape[1], 1, 1)
    if noise is not None:
        x = x + noise.to(dtype=x.dtype)
    if bias is not None:
        x = x + bias.to(dtype=x.dtype).reshape(1, -1, 1, 1)
    if act == "lrelu":
        x = F.leaky_relu(x, negative_slope=float(alpha))
    elif act == "relu":
        x = F.relu(x)
    if gain != 1.0:
        x = x * float(gain)
    if clamp is not None:
        x = x.clamp(-float(clamp), float(clamp))
    return x


def fused_epilogue(
    x: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    *,
    noise: Optional[torch.Tensor] = None,
    dcoef: Optional[torch.Tensor] = None,
    act: str = "lrelu",
    alpha: float = 0.2,
    gain: float = 1.0,
    clamp: Optional[float] = None,
    use_triton: bool = False,
) -> torch.Tensor:
    """Run the compiler-visible composite.

    Explicit Triton was rejected at B64: Inductor already emits a fused
    pointwise kernel, while the custom kernel lacked per-sample demodulation
    and autograd support.  ``use_triton`` remains accepted for API
    compatibility but intentionally has no effect.
    """
    _ = use_triton
    return fused_epilogue_composite(
        x, bias, noise=noise, dcoef=dcoef, act=act, alpha=alpha, gain=gain, clamp=clamp
    )


def triton_epilogue_available() -> bool:
    return False
