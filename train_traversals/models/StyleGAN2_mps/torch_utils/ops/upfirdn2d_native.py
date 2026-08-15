"""Compile-friendly native PyTorch implementation of StyleGAN's upfirdn2d.

This module is an isolated prototype. It does not replace the legacy CUDA op
or change the generator's current execution path.

The legacy reference implementation explicitly inserts zeros before filtering
when ``up > 1``. The native implementation expresses the same operation as a
depthwise transposed convolution. For ``up == 1``, filtering and decimation can
be combined into one strided depthwise convolution.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F


IntOrPair = Union[int, Sequence[int]]
Padding = Union[int, Sequence[int]]


def _parse_scaling(scaling: IntOrPair) -> Tuple[int, int]:
    if isinstance(scaling, int):
        scaling = (scaling, scaling)
    if len(scaling) != 2:
        raise ValueError(f"Scaling must contain two values, got {scaling}")
    sx, sy = int(scaling[0]), int(scaling[1])
    if sx < 1 or sy < 1:
        raise ValueError(f"Scaling values must be positive, got {(sx, sy)}")
    return sx, sy


def _parse_padding(padding: Padding) -> Tuple[int, int, int, int]:
    if isinstance(padding, int):
        return padding, padding, padding, padding
    if len(padding) == 2:
        padx, pady = int(padding[0]), int(padding[1])
        return padx, padx, pady, pady
    if len(padding) != 4:
        raise ValueError(f"Padding must contain two or four values, got {padding}")
    return tuple(int(v) for v in padding)  # type: ignore[return-value]


def _filter_2d(
    f: Optional[torch.Tensor],
    *,
    device: torch.device,
    dtype: torch.dtype,
    gain: float,
) -> torch.Tensor:
    if f is None:
        f = torch.ones((1, 1), device=device, dtype=torch.float32)
    else:
        if f.ndim not in (1, 2):
            raise ValueError(f"Filter must be one- or two-dimensional, got {f.ndim}D")
        f = f.to(device=device)
    if f.ndim == 1:
        # Match the reference's two separable passes, each receiving sqrt(gain).
        f = f * float(gain) ** 0.5
        f = f[:, None] * f[None, :]
    else:
        f = f * float(gain)
    return f.to(dtype=dtype)


def _pad_or_crop(
    x: torch.Tensor,
    padx0: int,
    padx1: int,
    pady0: int,
    pady1: int,
) -> torch.Tensor:
    positive = (max(padx0, 0), max(padx1, 0), max(pady0, 0), max(pady1, 0))
    if any(positive):
        x = F.pad(x, positive)
    crop_left, crop_right = max(-padx0, 0), max(-padx1, 0)
    crop_top, crop_bottom = max(-pady0, 0), max(-pady1, 0)
    return x[
        :,
        :,
        crop_top : x.shape[2] - crop_bottom,
        crop_left : x.shape[3] - crop_right,
    ]


def _expanded_depthwise_weight(kernel: torch.Tensor, channels: int) -> torch.Tensor:
    # ``expand`` avoids constructing C physical copies in eager mode. Backends
    # remain free to materialize their preferred convolution weight layout.
    return kernel[None, None].expand(channels, 1, kernel.shape[0], kernel.shape[1])


def upfirdn2d_native(
    x: torch.Tensor,
    f: Optional[torch.Tensor],
    up: IntOrPair = 1,
    down: IntOrPair = 1,
    padding: Padding = 0,
    flip_filter: bool = False,
    gain: float = 1.0,
    *,
    use_strided_down: Optional[bool] = None,
) -> torch.Tensor:
    """Apply exact StyleGAN-compatible upsample, FIR, and downsample operations.

    Args mirror :func:`upfirdn2d`. ``use_strided_down`` controls whether an
    ``up == 1`` decimation is folded into ``conv2d(stride=down)``. ``None``
    selects the strided path. The slice path remains available because backend
    and shape-specific convolution heuristics can occasionally favor it.
    """
    if x.ndim != 4:
        raise ValueError(f"Expected NCHW input, got shape {tuple(x.shape)}")

    upx, upy = _parse_scaling(up)
    downx, downy = _parse_scaling(down)
    padx0, padx1, pady0, pady1 = _parse_padding(padding)
    kernel = _filter_2d(f, device=x.device, dtype=x.dtype, gain=gain)
    filter_h, filter_w = int(kernel.shape[0]), int(kernel.shape[1])
    channels = int(x.shape[1])

    if upx == 1 and upy == 1:
        # F.conv2d performs correlation; StyleGAN's default is convolution.
        conv_kernel = kernel if flip_filter else kernel.flip((0, 1))
        weight = _expanded_depthwise_weight(conv_kernel, channels)
        x = _pad_or_crop(x, padx0, padx1, pady0, pady1)

        if use_strided_down is None:
            use_strided_down = True
        stride = (downy, downx) if use_strided_down else (1, 1)
        x = F.conv2d(x, weight, stride=stride, groups=channels)
        if not use_strided_down and (downx > 1 or downy > 1):
            x = x[:, :, ::downy, ::downx]
        return x

    # A transposed convolution exactly represents zero insertion followed by
    # FIR filtering, without allocating the sparse upsampled tensor.
    input_h, input_w = x.shape[2], x.shape[3]
    transpose_kernel = kernel.flip((0, 1)) if flip_filter else kernel
    weight = _expanded_depthwise_weight(transpose_kernel, channels)
    x = F.conv_transpose2d(
        x,
        weight,
        stride=(upy, upx),
        groups=channels,
    )

    output_h = input_h * upy + pady0 + pady1 - filter_h + 1
    output_w = input_w * upx + padx0 + padx1 - filter_w + 1
    start_y = filter_h - 1 - pady0
    start_x = filter_w - 1 - padx0

    pad_left, pad_top = max(-start_x, 0), max(-start_y, 0)
    start_x += pad_left
    start_y += pad_top
    pad_right = max(start_x + output_w - (x.shape[3] + pad_left), 0)
    pad_bottom = max(start_y + output_h - (x.shape[2] + pad_top), 0)
    if pad_left or pad_right or pad_top or pad_bottom:
        x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))

    return x[
        :,
        :,
        start_y : start_y + output_h : downy,
        start_x : start_x + output_w : downx,
    ]
