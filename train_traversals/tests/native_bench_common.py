"""Shared helpers for native StyleGAN op CUDA benchmarks."""

from __future__ import annotations

import json
import os
import statistics
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional, Sequence

import torch


DTYPE_MAP = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


@dataclass
class TimedResult:
    name: str
    median_ms: float
    mean_ms: float
    stdev_ms: float
    samples_ms: list[float]
    max_abs_error: float
    mean_abs_error: float
    rel_error: float
    peak_allocated_mib: float
    peak_reserved_mib: float
    compile_ms: Optional[float] = None
    layout_in: str = "unknown"
    layout_out: str = "unknown"
    notes: list[str] = field(default_factory=list)
    skipped: bool = False
    skip_reason: str = ""


def parse_dtypes(values: Sequence[str]) -> list[torch.dtype]:
    return [DTYPE_MAP[v.lower()] for v in values]


def select_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def configure_cuda_backends(
    *,
    cudnn_benchmark: bool = True,
    allow_tf32: bool = False,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"cudnn_benchmark": False, "allow_tf32": False}
    torch.backends.cudnn.benchmark = bool(cudnn_benchmark)
    torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
    torch.backends.cudnn.allow_tf32 = bool(allow_tf32)
    return {
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32,
    }


def memory_format_name(t: torch.Tensor) -> str:
    if t.ndim != 4:
        return "n/a"
    if t.is_contiguous(memory_format=torch.channels_last):
        return "channels_last"
    if t.is_contiguous():
        return "contiguous"
    return "strided"


def apply_layout(x: torch.Tensor, layout: str) -> torch.Tensor:
    if layout == "channels_last":
        if x.ndim != 4:
            raise ValueError("channels_last requires NCHW tensors")
        return x.contiguous(memory_format=torch.channels_last)
    return x.contiguous()


def empty_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    elif device.type == "mps":
        torch.mps.empty_cache()


def peak_memory_mib(device: torch.device) -> tuple[float, float]:
    if device.type != "cuda":
        return 0.0, 0.0
    allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
    return allocated, reserved


def error_stats(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float, float]:
    diff = (actual.float() - expected.float()).abs()
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())
    denom = float(expected.float().abs().mean().clamp_min(1e-12).item())
    return max_abs, mean_abs, max_abs / denom


def tolerance_for(dtype: torch.dtype) -> tuple[float, float]:
    if dtype in (torch.float16, torch.bfloat16):
        return 3e-3, 3e-3
    return 2e-5, 2e-5


def compile_callable(
    fn: Callable,
    example_args: tuple,
    *,
    mode: str = "default",
    fullgraph: bool = True,
    device: torch.device,
    cache_size_limit: int = 64,
) -> tuple[Callable, float, list[str]]:
    notes: list[str] = []
    torch._dynamo.config.cache_size_limit = cache_size_limit
    torch._dynamo.config.recompile_limit = cache_size_limit
    compiled = torch.compile(fn, fullgraph=fullgraph, mode=mode)
    synchronize(device)
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = compiled(*example_args)
    synchronize(device)
    compile_ms = (time.perf_counter() - t0) * 1000.0
    notes.append(f"fullgraph={fullgraph}")
    notes.append(f"compile_mode={mode}")
    if device.type == "cuda" and mode in ("reduce-overhead", "max-autotune"):
        notes.append("cuda_graphs_candidate=True")
    _ = out
    return compiled, compile_ms, notes


def probe_profiler_kernels(
    fn: Callable,
    args: tuple,
    device: torch.device,
    *,
    rows: int = 8,
) -> list[str]:
    if device.type != "cuda":
        return []
    notes: list[str] = []
    try:
        with torch.inference_mode():
            for _ in range(2):
                fn(*args)
            synchronize(device)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
                fn(*args)
                synchronize(device)
        events = [
            e
            for e in prof.key_averages()
            if e.device_type == torch.profiler.DeviceType.CUDA and e.self_device_time_total > 0
        ]
        events.sort(key=lambda e: e.self_device_time_total, reverse=True)
        for event in events[:rows]:
            notes.append(
                f"kernel:{event.key}:{event.self_device_time_total / 1000.0:.3f}ms"
            )
        joined = " ".join(e.key.lower() for e in events[:rows])
        if "cuda_graph" in joined or "graphexec" in joined:
            notes.append("cuda_graph_seen=True")
        else:
            notes.append("cuda_graph_seen=False")
    except Exception as exc:  # profiler is best-effort metadata
        notes.append(f"profiler_error={type(exc).__name__}:{exc}")
    return notes


def measure_implementations(
    implementations: Sequence[tuple[str, Callable]],
    args: tuple,
    device: torch.device,
    warmup: int,
    repeats: int,
    rounds: int = 6,
) -> dict[str, list[float]]:
    with torch.inference_mode():
        for _, fn in implementations:
            for _ in range(warmup):
                fn(*args)
        synchronize(device)
        samples = {name: [] for name, _ in implementations}
        for round_idx in range(rounds):
            ordered = (
                list(implementations)
                if round_idx % 2 == 0
                else list(reversed(implementations))
            )
            for name, fn in ordered:
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                start = time.perf_counter()
                for _ in range(repeats):
                    fn(*args)
                synchronize(device)
                samples[name].append((time.perf_counter() - start) * 1000.0 / repeats)
    return samples


def summarize_timing(
    name: str,
    samples: list[float],
    *,
    actual: Optional[torch.Tensor] = None,
    expected: Optional[torch.Tensor] = None,
    peak_allocated_mib: float = 0.0,
    peak_reserved_mib: float = 0.0,
    compile_ms: Optional[float] = None,
    layout_in: str = "unknown",
    layout_out: str = "unknown",
    notes: Optional[list[str]] = None,
) -> TimedResult:
    if actual is not None and expected is not None:
        max_abs, mean_abs, rel = error_stats(actual, expected)
    else:
        max_abs = mean_abs = rel = float("nan")
    return TimedResult(
        name=name,
        median_ms=statistics.median(samples),
        mean_ms=statistics.mean(samples),
        stdev_ms=statistics.pstdev(samples) if len(samples) > 1 else 0.0,
        samples_ms=list(samples),
        max_abs_error=max_abs,
        mean_abs_error=mean_abs,
        rel_error=rel,
        peak_allocated_mib=peak_allocated_mib,
        peak_reserved_mib=peak_reserved_mib,
        compile_ms=compile_ms,
        layout_in=layout_in,
        layout_out=layout_out,
        notes=list(notes or []),
    )


def write_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, TimedResult):
        return asdict(obj)
    if isinstance(obj, torch.dtype):
        return str(obj).removeprefix("torch.")
    if isinstance(obj, torch.device):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)!r} is not JSON serializable")


def env_metadata() -> dict[str, Any]:
    meta = {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        meta.update(
            {
                "cuda_runtime": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(),
                "gpu": torch.cuda.get_device_name(0),
                "capability": list(torch.cuda.get_device_capability(0)),
                "bf16": torch.cuda.is_bf16_supported(),
            }
        )
        try:
            import triton

            meta["triton"] = triton.__version__
        except Exception:
            meta["triton"] = None
    return meta


def format_result_line(
    case_name: str,
    results: Sequence[TimedResult],
    *,
    baseline_name: str = "reference",
    extra: str = "",
) -> str:
    baseline = next((r.median_ms for r in results if r.name == baseline_name and not r.skipped), None)
    parts = []
    for result in results:
        if result.skipped:
            parts.append(f"{result.name}=SKIP({result.skip_reason})")
            continue
        speedup = (baseline / result.median_ms) if baseline else float("nan")
        piece = f"{result.name}={result.median_ms:.3f}ms ({speedup:.2f}x)"
        if result.compile_ms is not None:
            piece += f" compile={result.compile_ms:.1f}ms"
        piece += f" max_abs={result.max_abs_error:.3e}"
        parts.append(piece)
    prefix = f"{case_name:28s}"
    if extra:
        prefix += f" {extra}"
    return prefix + "  " + "  ".join(parts)
