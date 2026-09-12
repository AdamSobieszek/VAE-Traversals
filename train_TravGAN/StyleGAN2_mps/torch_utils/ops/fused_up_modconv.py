"""Compose StyleGAN fixed FIR with learned up-convolution (up=2).

Two experimental routes, both intended to replace:

  conv_transpose2d(stride=2) → upfirdn2d(FIR, gain=4, pad=[1,1,1,1])

with a single activation-side kernel launch.

1. ``polyphase``: low-res 4-phase conv from the composed 6×6 kernel + pixel_shuffle
2. ``expanded``: single 6×6 stride-2 transpose convolution + crop
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .upfirdn2d_native import _filter_2d


def synthesis_up2_fir_padding(
    *,
    kernel_size: int = 3,
    filter_size: int = 4,
    user_padding: int = 1,
    up: int = 2,
) -> Tuple[int, int, int, int]:
    """Return FIR padding after conv2d_resample's up + transpose adjustments.

    For StyleGAN synthesis ``conv0`` (k=3, f=4, pad=1, up=2) this yields
    ``(1, 1, 1, 1)``.
    """
    fw = fh = filter_size
    kh = kw = kernel_size
    px0 = py0 = user_padding
    px1 = py1 = user_padding
    # up-adjust
    px0 += (fw + up - 1) // 2
    px1 += (fw - up) // 2
    py0 += (fh + up - 1) // 2
    py1 += (fh - up) // 2
    # transpose-adjust
    px0 -= kw - 1
    px1 -= kw - up
    py0 -= kh - 1
    py1 -= kh - up
    pxt = max(min(-px0, -px1), 0)
    pyt = max(min(-py0, -py1), 0)
    return px0 + pxt, px1 + pxt, py0 + pyt, py1 + pyt


def _fir_kernel_2d(
    f: Optional[torch.Tensor],
    *,
    device: torch.device,
    dtype: torch.dtype,
    gain: float,
) -> torch.Tensor:
    kernel, _ = _filter_2d(f, device=device, dtype=dtype, gain=gain, force_dense=True)
    if kernel.ndim == 1:
        kernel = kernel[:, None] * kernel[None, :]
    return kernel


def compose_weight_with_fir_expanded(
    weight: torch.Tensor,
    fir: torch.Tensor,
) -> torch.Tensor:
    """Spatially convolve learned ``[O,I,kh,kw]`` with FIR → expanded ``[O,I,eh,ew]``.

    Matches StyleGAN ``conv2d_resample(..., up=2, flip_weight=False)`` where the
    transpose path uses the unflipped weight in correlation form.
    """
    o, i, kh, kw = weight.shape
    fh, fw = fir.shape
    w_flat = weight.reshape(1, o * i, kh, kw)
    fir_w = fir.to(dtype=weight.dtype, device=weight.device)[None, None].expand(
        o * i, 1, fh, fw
    )
    composed = F.conv2d(w_flat, fir_w, padding=(fh - 1, fw - 1), groups=o * i)
    return composed.reshape(o, i, composed.shape[2], composed.shape[3])


def expanded_up2_modconv(
    x: torch.Tensor,
    weight: torch.Tensor,
    fir: torch.Tensor,
    *,
    fir_padding: Tuple[int, int, int, int] = (1, 1, 1, 1),
    crop_start: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    """Single expanded-kernel stride-2 transpose conv + crop."""
    composed = compose_weight_with_fir_expanded(weight, fir)
    w_t = composed.transpose(0, 1).contiguous()
    y = F.conv_transpose2d(x, w_t, stride=2, padding=0)
    padx0, padx1, pady0, pady1 = fir_padding
    fh, fw = int(fir.shape[0]), int(fir.shape[1])
    if crop_start is None:
        # Empirically matches StyleGAN pad=[1,1,1,1] after 3x3∘4x4 composition.
        start_y = fh - 1 - pady0  # 2 for common case
        start_x = fw - 1 - padx0
    else:
        start_y, start_x = crop_start
    out_h = x.shape[2] * 2
    out_w = x.shape[3] * 2
    start_y = max(int(start_y), 0)
    start_x = max(int(start_x), 0)
    y = y[:, :, start_y : start_y + out_h, start_x : start_x + out_w]
    if y.shape[2] != out_h or y.shape[3] != out_w:
        y = y[:, :, :out_h, :out_w]
        if y.shape[2] < out_h or y.shape[3] < out_w:
            y = F.pad(y, (0, out_w - y.shape[3], 0, out_h - y.shape[2]))
    return y


def polyphase_up2_modconv(
    x: torch.Tensor,
    weight: torch.Tensor,
    fir: torch.Tensor,
) -> torch.Tensor:
    """Fused up=2 via composed-kernel polyphase + pixel_shuffle.

    Builds the same 6×6 composed kernel as ``expanded_up2_modconv``, splits it
    into four stride-2 phases (pixel_shuffle order), flips each phase to match
    ``conv_transpose2d`` correlation, runs one low-resolution conv, then
    ``pixel_shuffle(2)``.
    """
    composed = compose_weight_with_fir_expanded(weight, fir)  # [O,I,6,6]
    o = composed.shape[0]
    # pixel_shuffle channel order: (oy,ox) = (0,0),(0,1),(1,0),(1,1)
    phase_order = ((0, 0), (0, 1), (1, 0), (1, 1))
    phases = [
        composed[:, :, oy::2, ox::2].flip([2, 3]).contiguous()
        for oy, ox in phase_order
    ]
    stacked = torch.cat(phases, dim=0)  # [4O, I, ~3, ~3]
    pad_y = (phases[0].shape[2] - 1) // 2
    pad_x = (phases[0].shape[3] - 1) // 2
    y = F.conv2d(x, stacked, padding=(pad_y, pad_x))
    n, _, h, w_sp = y.shape
    y = y.reshape(n, 4, o, h, w_sp).permute(0, 2, 1, 3, 4).reshape(n, 4 * o, h, w_sp)
    return F.pixel_shuffle(y, 2)


class PreparedFusedUpConv(nn.Module):
    """Static (non-modulated) fused up=2 conv for resnet skip layers."""

    def __init__(
        self,
        weight: torch.Tensor,
        fir: torch.Tensor,
        *,
        mode: str = "polyphase",
        gain: float = 4.0,
    ):
        super().__init__()
        self.mode = mode
        fir_k = _fir_kernel_2d(
            fir, device=weight.device, dtype=torch.float32, gain=gain
        )
        self.register_buffer("fir", fir_k)
        self.fir_padding = synthesis_up2_fir_padding()
        composed = compose_weight_with_fir_expanded(weight.detach(), fir_k)
        if mode == "expanded":
            self.register_buffer("weight_t", composed.transpose(0, 1).contiguous())
        else:
            phase_order = ((0, 0), (0, 1), (1, 0), (1, 1))
            phases = [
                composed[:, :, oy::2, ox::2].flip([2, 3]).contiguous()
                for oy, ox in phase_order
            ]
            self.register_buffer("weight", torch.cat(phases, dim=0).contiguous())
            self.pad = ((phases[0].shape[2] - 1) // 2, (phases[0].shape[3] - 1) // 2)
            self.out_channels = int(composed.shape[0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "expanded":
            y = F.conv_transpose2d(
                x, self.weight_t.to(dtype=x.dtype), stride=2
            )
            out_h, out_w = x.shape[2] * 2, x.shape[3] * 2
            return y[:, :, 2 : 2 + out_h, 2 : 2 + out_w]
        y = F.conv2d(x, self.weight.to(dtype=x.dtype), padding=self.pad)
        n, _, h, w_sp = y.shape
        o = self.out_channels
        y = y.reshape(n, 4, o, h, w_sp).permute(0, 2, 1, 3, 4).reshape(n, 4 * o, h, w_sp)
        return F.pixel_shuffle(y, 2)
