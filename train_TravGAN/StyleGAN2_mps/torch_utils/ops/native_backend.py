"""Opt-in backend overrides for StyleGAN custom ops."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator, Optional

import torch

_UPFIRDN_IMPL: ContextVar[Optional[str]] = ContextVar("upfirdn2d_impl", default=None)
_BIAS_ACT_IMPL: ContextVar[Optional[str]] = ContextVar("bias_act_impl", default=None)
# Dynamo cannot trace ContextVar.get().  These globals are only consulted while
# Dynamo is tracing, with values scoped by use_native_ops(); eager execution
# continues to use the thread-/task-local ContextVars above.
_COMPILE_UPFIRDN_IMPL: Optional[str] = None
_COMPILE_BIAS_ACT_IMPL: Optional[str] = None


def get_upfirdn_impl(explicit: str) -> str:
    forced = (
        _COMPILE_UPFIRDN_IMPL
        if torch.compiler.is_compiling()
        else _UPFIRDN_IMPL.get()
    )
    return explicit if forced is None else forced


def get_bias_act_impl(explicit: str) -> str:
    forced = (
        _COMPILE_BIAS_ACT_IMPL
        if torch.compiler.is_compiling()
        else _BIAS_ACT_IMPL.get()
    )
    return explicit if forced is None else forced


@contextmanager
def use_native_ops(
    *,
    upfirdn: str = "native",
    bias_act: str = "native",
) -> Iterator[None]:
    """Temporarily force op backends for generator benchmarks.

    Does not change the default ``impl='cuda'`` when the context is inactive.
    """
    global _COMPILE_UPFIRDN_IMPL, _COMPILE_BIAS_ACT_IMPL
    previous_compile_u = _COMPILE_UPFIRDN_IMPL
    previous_compile_b = _COMPILE_BIAS_ACT_IMPL
    _COMPILE_UPFIRDN_IMPL = upfirdn
    _COMPILE_BIAS_ACT_IMPL = bias_act
    token_u = _UPFIRDN_IMPL.set(upfirdn)
    token_b = _BIAS_ACT_IMPL.set(bias_act)
    try:
        yield
    finally:
        _BIAS_ACT_IMPL.reset(token_b)
        _UPFIRDN_IMPL.reset(token_u)
        _COMPILE_BIAS_ACT_IMPL = previous_compile_b
        _COMPILE_UPFIRDN_IMPL = previous_compile_u
