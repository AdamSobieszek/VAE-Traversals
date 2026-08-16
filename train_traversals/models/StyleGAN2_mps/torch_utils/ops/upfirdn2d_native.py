"""Compile-friendly native PyTorch implementation of StyleGAN's upfirdn2d.

This module is an isolated prototype by default. Production code can opt in
through ``upfirdn2d(..., impl="native")`` once wired.

The legacy reference implementation explicitly inserts zeros before filtering
when ``up > 1``. The native implementation expresses the same operation as a
depthwise transposed convolution. For ``up == 1``, filtering and decimation can
be combined into one strided depthwise convolution.

Optional CUDA-oriented variants:
- weight construction: expand / repeat / contiguous / cached contiguous
- separable two-pass execution for 1D filters
- exact polyphase upsampling for ``up == 2`` with square dense filters
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F


IntOrPair = Union[int, Sequence[int]]
Padding = Union[int, Sequence[int]]
WeightMode = str  # expand | repeat | contiguous | cached
SeparableMode = str  # auto | dense | separable

_WEIGHT_CACHE: Dict[tuple, torch.Tensor] = {}


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
    force_dense: bool = True,
) -> Tuple[torch.Tensor, bool]:
    """Return (kernel, was_separable_input).

    When ``force_dense`` is True, one-dimensional filters are expanded to a 2D
    outer-product kernel (matches the previous native prototype).
    """
    was_separable = False
    if f is None:
        f = torch.ones((1, 1), device=device, dtype=torch.float32)
    else:
        if f.ndim not in (1, 2):
            raise ValueError(f"Filter must be one- or two-dimensional, got {f.ndim}D")
        f = f.to(device=device)
    if f.ndim == 1:
        was_separable = True
        if force_dense:
            f = f * float(gain) ** 0.5
            f = f[:, None] * f[None, :]
            return f.to(dtype=dtype), was_separable
        f = f * float(gain) ** 0.5
        return f.to(dtype=dtype), was_separable
    f = f * float(gain)
    return f.to(dtype=dtype), was_separable


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
    if not (crop_left or crop_right or crop_top or crop_bottom):
        return x
    # Positive start + length avoids end-relative sympy diagnostics.
    out_h = x.shape[2] - crop_top - crop_bottom
    out_w = x.shape[3] - crop_left - crop_right
    return x[:, :, crop_top : crop_top + out_h, crop_left : crop_left + out_w]


def _depthwise_weight(
    kernel: torch.Tensor,
    channels: int,
    *,
    weight_mode: WeightMode = "expand",
) -> torch.Tensor:
    if kernel.ndim != 2:
        raise ValueError(f"Expected 2D kernel for depthwise weight, got {kernel.shape}")
    base = kernel[None, None]
    if weight_mode == "expand":
        return base.expand(channels, 1, kernel.shape[0], kernel.shape[1])
    if weight_mode == "repeat":
        return base.repeat(channels, 1, 1, 1)
    if weight_mode == "contiguous":
        return base.expand(channels, 1, kernel.shape[0], kernel.shape[1]).contiguous()
    if weight_mode == "cached":
        # Content-addressed: data_ptr keys are unsafe for ephemeral tensors.
        kernel_cpu = kernel.detach().to(dtype=torch.float32, device="cpu").contiguous()
        key = (
            tuple(kernel_cpu.shape),
            hash(kernel_cpu.numpy().tobytes()),
            str(kernel.device),
            str(kernel.dtype),
            channels,
            "dw2d",
        )
        weight = _WEIGHT_CACHE.get(key)
        if (
            weight is None
            or weight.shape[0] != channels
            or weight.device != kernel.device
            or weight.dtype != kernel.dtype
        ):
            weight = base.expand(channels, 1, kernel.shape[0], kernel.shape[1]).contiguous()
            _WEIGHT_CACHE[key] = weight
        return weight
    raise ValueError(f"Unknown weight_mode {weight_mode!r}")


def _separable_weights(
    taps: torch.Tensor,
    channels: int,
    *,
    flip_filter: bool,
    weight_mode: WeightMode,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if taps.ndim != 1:
        raise ValueError("Separable path expects 1D taps")
    fx = taps if flip_filter else taps.flip(0)
    fy = fx
    wx = fx.view(1, 1, 1, -1)
    wy = fy.view(1, 1, -1, 1)
    if weight_mode == "expand":
        return wx.expand(channels, 1, 1, fx.numel()), wy.expand(channels, 1, fy.numel(), 1)
    if weight_mode == "repeat":
        return wx.repeat(channels, 1, 1, 1), wy.repeat(channels, 1, 1, 1)
    if weight_mode in ("contiguous", "cached"):
        return (
            wx.expand(channels, 1, 1, fx.numel()).contiguous(),
            wy.expand(channels, 1, fy.numel(), 1).contiguous(),
        )
    raise ValueError(f"Unknown weight_mode {weight_mode!r}")


def _conv_separable_up1(
    x: torch.Tensor,
    taps: torch.Tensor,
    *,
    channels: int,
    downx: int,
    downy: int,
    flip_filter: bool,
    use_strided_down: bool,
    weight_mode: WeightMode,
) -> torch.Tensor:
    wx, wy = _separable_weights(
        taps, channels, flip_filter=flip_filter, weight_mode=weight_mode
    )
    # Horizontal then vertical, matching the legacy reference order.
    if use_strided_down:
        x = F.conv2d(x, wx, stride=(1, downx), groups=channels)
        x = F.conv2d(x, wy, stride=(downy, 1), groups=channels)
        return x
    x = F.conv2d(x, wx, groups=channels)
    x = F.conv2d(x, wy, groups=channels)
    if downx > 1 or downy > 1:
        x = x[:, :, ::downy, ::downx]
    return x


def _polyphase_up2(
    x: torch.Tensor,
    kernel: torch.Tensor,
    *,
    padx0: int,
    padx1: int,
    pady0: int,
    pady1: int,
    downx: int,
    downy: int,
    flip_filter: bool,
    weight_mode: WeightMode,
) -> torch.Tensor:
    """Exact polyphase upsampling for square dense FIR with ``up == 2``.

    Splits the FIR into four phase kernels, convolves at the low resolution,
    then interleaves with ``pixel_shuffle``. Negative StyleGAN padding is
    applied as a final high-resolution crop.
    """
    if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
        raise ValueError("Polyphase path requires a square dense filter")
    if kernel.shape[0] % 2 != 0:
        raise ValueError("Polyphase path currently requires even filter size")
    if kernel.shape[0] not in (4, 6):
        raise ValueError(
            "Polyphase path is validated for 4x4 and 6x6 filters; "
            f"got {tuple(kernel.shape)}"
        )

    channels = int(x.shape[1])
    filter_h = int(kernel.shape[0])
    filter_w = int(kernel.shape[1])
    # Orientation matches the default native transposed-convolution path.
    conv_kernel = kernel.flip((0, 1)) if flip_filter else kernel

    # Channel order is the pixel_shuffle layout that matches StyleGAN's FIR
    # origin for the common 4-tap filter. Empirically verified across paddings.
    phase_order = ((1, 1), (1, 0), (0, 1), (0, 0))
    phases = [conv_kernel[py::2, px::2].contiguous() for py, px in phase_order]
    weight = torch.stack(phases, dim=0)[:, None, :, :]
    weight = weight.repeat(channels, 1, 1, 1)
    if weight_mode in ("contiguous", "cached"):
        weight = weight.contiguous()

    pos_x0, pos_x1 = max(padx0, 0), max(padx1, 0)
    pos_y0, pos_y1 = max(pady0, 0), max(pady1, 0)
    neg_x0, neg_x1 = max(-padx0, 0), max(-padx1, 0)
    neg_y0, neg_y1 = max(-pady0, 0), max(-pady1, 0)

    pad_left = pos_x0 // 2
    pad_right = (pos_x1 + 1) // 2
    pad_top = pos_y0 // 2
    pad_bottom = (pos_y1 + 1) // 2
    start_x = 1 - (pos_x0 % 2)
    start_y = 1 - (pos_y0 % 2)

    y = F.conv2d(
        F.pad(x, (pad_left, pad_right, pad_top, pad_bottom)),
        weight,
        groups=channels,
    )
    y = F.pixel_shuffle(y, 2)

    output_h = x.shape[2] * 2 + pos_y0 + pos_y1 - filter_h + 1
    output_w = x.shape[3] * 2 + pos_x0 + pos_x1 - filter_w + 1
    # Prefer positive start+length slices (avoid end-relative `:-k` forms that
    # trigger Inductor pow_by_natural diagnostics on symbolic shapes).
    crop_y0 = start_y + neg_y0
    crop_x0 = start_x + neg_x0
    final_h = output_h - neg_y0 - neg_y1
    final_w = output_w - neg_x0 - neg_x1
    y = y[:, :, crop_y0 : crop_y0 + final_h, crop_x0 : crop_x0 + final_w]
    if downx > 1 or downy > 1:
        y = y[:, :, ::downy, ::downx]
    return y


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
    weight_mode: WeightMode = "expand",
    separable_mode: SeparableMode = "auto",
    impl_variant: str = "default",
) -> torch.Tensor:
    """Apply exact StyleGAN-compatible upsample, FIR, and downsample operations.

    Args mirror :func:`upfirdn2d`. Extra keyword-only options select CUDA
    candidate variants without changing the default algorithm:

    - ``weight_mode``: ``expand`` (default), ``repeat``, ``contiguous``, ``cached``
    - ``separable_mode``: ``auto`` keeps dense outer-product for 1D filters;
      ``separable`` forces two-pass 1D convolution when ``up == 1``;
      ``dense`` always builds a 2D kernel
    - ``impl_variant``: ``default`` or ``polyphase`` (``up == (2,2)`` only)
    """
    if x.ndim != 4:
        raise ValueError(f"Expected NCHW input, got shape {tuple(x.shape)}")

    upx, upy = _parse_scaling(up)
    downx, downy = _parse_scaling(down)
    padx0, padx1, pady0, pady1 = _parse_padding(padding)
    channels = int(x.shape[1])

    want_separable = separable_mode == "separable"
    force_dense = separable_mode != "separable"
    kernel, was_separable = _filter_2d(
        f, device=x.device, dtype=x.dtype, gain=gain, force_dense=force_dense
    )

    if impl_variant == "polyphase":
        if not (upx == 2 and upy == 2):
            raise ValueError("polyphase variant currently supports only up=2")
        if kernel.ndim != 2:
            kernel, _ = _filter_2d(
                f, device=x.device, dtype=x.dtype, gain=gain, force_dense=True
            )
        return _polyphase_up2(
            x,
            kernel,
            padx0=padx0,
            padx1=padx1,
            pady0=pady0,
            pady1=pady1,
            downx=downx,
            downy=downy,
            flip_filter=flip_filter,
            weight_mode=weight_mode,
        )

    # FP16/BF16: prefer exact polyphase for the common StyleGAN up=2 square FIR.
    if (
        impl_variant == "default"
        and upx == 2
        and upy == 2
        and x.dtype in (torch.float16, torch.bfloat16)
        and kernel.ndim == 2
        and int(kernel.shape[0]) in (4, 6)
        and int(kernel.shape[0]) == int(kernel.shape[1])
    ):
        return _polyphase_up2(
            x,
            kernel,
            padx0=padx0,
            padx1=padx1,
            pady0=pady0,
            pady1=pady1,
            downx=downx,
            downy=downy,
            flip_filter=flip_filter,
            weight_mode="contiguous",
        )

    if upx == 1 and upy == 1:
        if use_strided_down is None:
            use_strided_down = True
        x = _pad_or_crop(x, padx0, padx1, pady0, pady1)

        if want_separable and (was_separable or (f is not None and f.ndim == 1)):
            taps = kernel if kernel.ndim == 1 else None
            if taps is None:
                # Caller asked for separable but supplied a dense filter; fall back.
                pass
            else:
                return _conv_separable_up1(
                    x,
                    taps,
                    channels=channels,
                    downx=downx,
                    downy=downy,
                    flip_filter=flip_filter,
                    use_strided_down=use_strided_down,
                    weight_mode=weight_mode,
                )

        if kernel.ndim == 1:
            # Dense-ize for the single-convolution path.
            kernel = kernel[:, None] * kernel[None, :]
        conv_kernel = kernel if flip_filter else kernel.flip((0, 1))
        weight = _depthwise_weight(conv_kernel, channels, weight_mode=weight_mode)
        stride = (downy, downx) if use_strided_down else (1, 1)
        x = F.conv2d(x, weight, stride=stride, groups=channels)
        if not use_strided_down and (downx > 1 or downy > 1):
            x = x[:, :, ::downy, ::downx]
        return x

    # Transposed-convolution upsample path.
    if kernel.ndim == 1:
        kernel = kernel[:, None] * kernel[None, :]
    filter_h, filter_w = int(kernel.shape[0]), int(kernel.shape[1])
    input_h, input_w = x.shape[2], x.shape[3]
    transpose_kernel = kernel.flip((0, 1)) if flip_filter else kernel
    weight = _depthwise_weight(transpose_kernel, channels, weight_mode=weight_mode)
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


def clear_weight_cache() -> None:
    _WEIGHT_CACHE.clear()
