"""Shared-weight modulated convolution (inference / ws-gradient friendly).

Avoids materializing per-sample ``[B,O,I,k,k]`` weights and ``groups=B``
convolution. Instead:

  x' = x * styles
  y  = conv2d_resample(x', weight)          # shared learned kernel
  y  = y * dcoefs + noise                   # dcoefs from styles @ weight_energy

``weight_energy = weight.square().sum((2,3))`` can be precomputed for frozen
weights. Demodulation still depends on styles, so gradients flow to ``ws``.
"""

from __future__ import annotations

from typing import Optional

import torch

from . import conv2d_resample
from . import fma


def _weight_energy(weight: torch.Tensor) -> torch.Tensor:
    """Return ``[O, I]`` sum of squares over spatial dims."""
    return weight.square().sum(dim=(2, 3))


def demod_coefs_from_styles(
    styles: torch.Tensor,
    weight_energy: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute StyleGAN demodulation coefficients without expanding weights.

    ``dcoefs[b,o] = (sum_i styles[b,i]^2 * weight_energy[o,i] + eps)^(-1/2)``
    """
    # styles.square(): [B, I], weight_energy.T: [I, O] -> [B, O]
    return (styles.square() @ weight_energy.transpose(0, 1) + eps).rsqrt()


def modulated_conv2d_shared(
    x: torch.Tensor,
    weight: torch.Tensor,
    styles: torch.Tensor,
    noise: Optional[torch.Tensor] = None,
    up: int = 1,
    down: int = 1,
    padding: int = 0,
    resample_filter: Optional[torch.Tensor] = None,
    demodulate: bool = True,
    flip_weight: bool = True,
    weight_energy: Optional[torch.Tensor] = None,
    return_dcoefs: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Shared-weight modulated conv (unfused StyleGAN path, compile-friendly).

    If ``return_dcoefs=True``, skip applying demod/noise and return
    ``(y, dcoefs)`` so a fused epilogue can absorb them.
    """
    batch_size = x.shape[0]
    out_channels, in_channels, kh, kw = weight.shape

    # FP16 overflow guard (same intent as fused path).
    if x.dtype == torch.float16 and demodulate:
        w_scale = weight.detach().abs().amax(dim=[1, 2, 3], keepdim=True).clamp_min(1e-4)
        weight = weight * ((1.0 / (in_channels * kh * kw) ** 0.5) * w_scale.reciprocal())
        s_scale = styles.detach().abs().amax(dim=1, keepdim=True).clamp_min(1e-4)
        styles = styles * s_scale.reciprocal()
        weight_energy = None  # stale after rescale

    x = x * styles.to(x.dtype).reshape(batch_size, -1, 1, 1)
    x = conv2d_resample.conv2d_resample(
        x=x,
        w=weight.to(x.dtype),
        f=resample_filter,
        up=up,
        down=down,
        padding=padding,
        flip_weight=flip_weight,
    )
    dcoefs: Optional[torch.Tensor] = None
    if demodulate:
        if weight_energy is None:
            weight_energy = _weight_energy(weight.float())
        dcoefs = demod_coefs_from_styles(styles.float(), weight_energy.float()).to(x.dtype)
        dcoefs = dcoefs.reshape(batch_size, -1, 1, 1)
        if return_dcoefs:
            return x, dcoefs
        if noise is not None:
            x = fma.fma(x, dcoefs, noise.to(x.dtype))
        else:
            x = x * dcoefs
    elif noise is not None:
        x = x + noise.to(x.dtype)
    if return_dcoefs:
        return x, dcoefs
    return x


class SharedModConvCache(torch.nn.Module):
    """Precomputes ``weight_energy`` for a frozen SynthesisLayer weight."""

    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.register_buffer("weight_energy", _weight_energy(weight.detach().float()))

    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        styles: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        return modulated_conv2d_shared(
            x, weight, styles, weight_energy=self.weight_energy, **kwargs
        )
