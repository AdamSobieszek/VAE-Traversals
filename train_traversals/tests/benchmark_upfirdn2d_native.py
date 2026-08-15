"""Benchmark native PyTorch upfirdn2d against StyleGAN's reference path.

Examples:
    python tests/benchmark_upfirdn2d_native.py --device mps
    python tests/benchmark_upfirdn2d_native.py --device cuda --compile

The CUDA run additionally includes the legacy custom CUDA extension. Compilation
time is excluded from measurements.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Callable, Sequence

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.StyleGAN2_mps.torch_utils.ops import upfirdn2d as legacy
from models.StyleGAN2_mps.torch_utils.ops.upfirdn2d_native import (
    upfirdn2d_native,
)


@dataclass(frozen=True)
class Case:
    name: str
    shape: tuple[int, int, int, int]
    up: int | tuple[int, int]
    down: int | tuple[int, int]
    padding: tuple[int, int, int, int]
    gain: float


CASES = (
    Case("rgb_up2_256", (4, 3, 256, 256), 2, 1, (2, 1, 2, 1), 4.0),
    Case("feature_up2_128", (2, 128, 128, 128), 2, 1, (2, 1, 2, 1), 4.0),
    # Typical FIR immediately after a 3x3 stride-2 transposed convolution.
    Case("fir_after_upconv_256", (2, 128, 257, 257), 1, 1, (1, 1, 1, 1), 4.0),
    Case("filter_same_256", (2, 128, 256, 256), 1, 1, (2, 1, 2, 1), 1.0),
    Case("down2_256", (2, 128, 256, 256), 1, 2, (1, 1, 1, 1), 1.0),
)


def select_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def measure_implementations(
    implementations: Sequence[tuple[str, Callable[[torch.Tensor], torch.Tensor]]],
    x: torch.Tensor,
    device: torch.device,
    warmup: int,
    repeats: int,
    rounds: int = 6,
) -> list[tuple[str, float]]:
    """Time implementations in alternating order to reduce thermal bias."""
    with torch.inference_mode():
        for _, fn in implementations:
            for _ in range(warmup):
                fn(x)
        synchronize(device)
        samples: dict[str, list[float]] = {
            name: [] for name, _ in implementations
        }
        for round_idx in range(rounds):
            ordered = (
                implementations
                if round_idx % 2 == 0
                else tuple(reversed(implementations))
            )
            for name, fn in ordered:
                start = time.perf_counter()
                for _ in range(repeats):
                    fn(x)
                synchronize(device)
                samples[name].append(
                    (time.perf_counter() - start) * 1000.0 / repeats
                )
    return [(name, statistics.median(samples[name])) for name, _ in implementations]


def parse_dtypes(values: Sequence[str]) -> list[torch.dtype]:
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    return [mapping[v.lower()] for v in values]


def benchmark(args: argparse.Namespace) -> None:
    device = select_device(args.device)
    dtypes = parse_dtypes(args.dtype)
    print(f"PyTorch {torch.__version__}  device={device}")
    print(
        "implementation notes: native uses conv_transpose2d for upsampling; "
        "filtering and downsampling use strided conv2d"
    )
    compile_enabled = bool(args.compile)

    for dtype in dtypes:
        print(f"\n[{str(dtype).removeprefix('torch.')}]")
        for case in CASES:
            try:
                x = torch.randn(case.shape, device=device, dtype=dtype)
            except (RuntimeError, TypeError) as exc:
                print(f"{case.name:24s} skipped: {exc}")
                continue
            f = legacy.setup_filter([1, 3, 3, 1], device=device)

            reference = lambda value: legacy._upfirdn2d_ref(
                value,
                f,
                up=case.up,
                down=case.down,
                padding=case.padding,
                gain=case.gain,
            )
            native = lambda value: upfirdn2d_native(
                value,
                f,
                up=case.up,
                down=case.down,
                padding=case.padding,
                gain=case.gain,
            )
            implementations: list[tuple[str, Callable[[torch.Tensor], torch.Tensor]]] = [
                ("reference", reference),
                ("native", native),
            ]
            if case.down != 1 and case.up == 1:
                implementations.extend([
                    (
                        "native-stride",
                        lambda value: upfirdn2d_native(
                            value,
                            f,
                            up=case.up,
                            down=case.down,
                            padding=case.padding,
                            gain=case.gain,
                            use_strided_down=True,
                        ),
                    ),
                    (
                        "native-slice",
                        lambda value: upfirdn2d_native(
                            value,
                            f,
                            up=case.up,
                            down=case.down,
                            padding=case.padding,
                            gain=case.gain,
                            use_strided_down=False,
                        ),
                    ),
                ])
            if device.type == "cuda":
                implementations.append((
                    "legacy-cuda",
                    lambda value: legacy.upfirdn2d(
                        value,
                        f,
                        up=case.up,
                        down=case.down,
                        padding=case.padding,
                        gain=case.gain,
                        impl="cuda",
                    ),
                ))
            if compile_enabled:
                try:
                    compiled = torch.compile(
                        native,
                        fullgraph=True,
                        mode=args.compile_mode,
                    )
                    # Trigger compilation before correctness/timing.
                    compiled(x)
                    synchronize(device)
                    implementations.append(("native-compiled", compiled))
                except Exception as exc:  # backend availability varies by PyTorch build
                    lines = [line.strip() for line in str(exc).splitlines() if line.strip()]
                    message = " | ".join(lines[:2]) if lines else type(exc).__name__
                    print(f"{case.name:24s} compile unavailable: {message}")
                    compile_enabled = False

            expected = reference(x)
            synchronize(device)
            valid_implementations = []
            for name, fn in implementations:
                try:
                    actual = fn(x)
                    synchronize(device)
                    tolerance = 3e-3 if dtype in (torch.float16, torch.bfloat16) else 2e-5
                    if not torch.allclose(actual, expected, rtol=tolerance, atol=tolerance):
                        error = float((actual.float() - expected.float()).abs().max())
                        print(f"{case.name:24s} {name:16s} mismatch max_abs={error:.3e}")
                        continue
                    valid_implementations.append((name, fn))
                except (RuntimeError, TypeError) as exc:
                    print(f"{case.name:24s} {name:16s} skipped: {exc}")

            timings = measure_implementations(
                valid_implementations,
                x,
                device,
                args.warmup,
                args.repeats,
            )
            baseline = dict(timings).get("reference")
            rendered = []
            for name, elapsed in timings:
                speedup = baseline / elapsed if baseline is not None else float("nan")
                rendered.append(f"{name}={elapsed:.3f}ms ({speedup:.2f}x)")
            print(f"{case.name:24s} " + "  ".join(rendered))

            del x, expected
            if device.type == "mps":
                torch.mps.empty_cache()
            elif device.type == "cuda":
                torch.cuda.empty_cache()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument("--dtype", nargs="+", default=("float32", "float16"))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--compile-mode",
        default="default",
        choices=("default", "reduce-overhead", "max-autotune"),
    )
    return parser


if __name__ == "__main__":
    benchmark(build_parser().parse_args())
