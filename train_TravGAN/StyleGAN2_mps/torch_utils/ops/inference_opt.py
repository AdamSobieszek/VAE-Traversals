"""Configuration and checkpoint cleanup for opt-in synthesis optimization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn


_SCALAR_ATTRS = (
    "weight_gain",
    "bias_gain",
    "act_gain",
    "w_avg_beta",
)


def normalize_scalar_attrs(module: nn.Module) -> int:
    """Cast NumPy/tensor scalar attributes to compiler-friendly Python floats."""
    fixed = 0
    for child in module.modules():
        for name in _SCALAR_ATTRS:
            if not hasattr(child, name):
                continue
            value = getattr(child, name)
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                setattr(child, name, float(value.detach().cpu().item()))
                fixed += 1
            elif type(value).__module__.startswith("numpy"):
                setattr(child, name, float(value))
                fixed += 1
            elif not isinstance(value, (float, int, type(None), bool)):
                try:
                    setattr(child, name, float(value))
                    fixed += 1
                except (TypeError, ValueError):
                    pass
    return fixed


@dataclass
class InferenceOptConfig:
    """Single opt-in configuration for the isolated synthesis wrapper."""

    shared_modconv: bool = False
    fir_compose: bool = False
    fir_compose_mode: str = "polyphase"  # polyphase | expanded
    fused_epilogue: bool = False
    force_fp16_all_blocks: bool = False
    fused_modconv: Optional[bool] = None
    low_precision: str = "fp16"  # fp16 | bf16
    notes: list[str] = field(default_factory=list)

    @property
    def low_precision_dtype(self) -> torch.dtype:
        if self.low_precision == "bf16":
            return torch.bfloat16
        if self.low_precision in ("fp16", "float16"):
            return torch.float16
        raise ValueError(f"Unsupported low precision {self.low_precision!r}")


def b64_compile_config(*, low_precision: str = "fp16") -> InferenceOptConfig:
    """Return the validated B64 compile policy."""
    return InferenceOptConfig(
        shared_modconv=True,
        fir_compose=True,
        fir_compose_mode="polyphase",
        low_precision=low_precision,
    )
