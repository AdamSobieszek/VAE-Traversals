"""Benchmark native PyTorch bias_act alternatives on StyleGAN-like shapes."""

from __future__ import annotations

import argparse
import math
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

from models.StyleGAN2_mps.torch_utils.ops import bias_act as legacy
from models.StyleGAN2_mps.torch_utils.ops.bias_act_native import (
    bias_act_native,
    bias_act_native_homogeneous_inference,
    bias_act_native_inference,
)


@dataclass(frozen=True)
class Case:
    name: str
    shape: tuple[int, ...]
    act: str
    gain: float
    clamp: float | None


CASES = (
    Case("mapping_b4_c512", (4, 512), "lrelu", math.sqrt(2.0), None),
    Case("synth_b2_c512_64", (2, 512, 64, 64), "lrelu", math.sqrt(2.0), 256.0),
    Case("synth_b2_c256_128", (2, 256, 128, 128), "lrelu", math.sqrt(2.0), 256.0),
    Case("synth_b2_c128_256", (2, 128, 256, 256), "lrelu", math.sqrt(2.0), 256.0),
    Case("synth_b1_c64_512", (1, 64, 512, 512), "lrelu", math.sqrt(2.0), 256.0),
    Case("torgb_b2_c3_256", (2, 3, 256, 256), "linear", 1.0, 256.0),
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


def parse_dtypes(values: Sequence[str]) -> list[torch.dtype]:
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    return [mapping[value.lower()] for value in values]


def measure_implementations(
    implementations: Sequence[tuple[str, Callable[[torch.Tensor], torch.Tensor]]],
    x: torch.Tensor,
    device: torch.device,
    warmup: int,
    repeats: int,
    rounds: int = 6,
) -> list[tuple[str, float]]:
    with torch.inference_mode():
        for _, fn in implementations:
            for _ in range(warmup):
                fn(x)
        synchronize(device)
        samples = {name: [] for name, _ in implementations}
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


def benchmark(args: argparse.Namespace) -> None:
    device = select_device(args.device)
    dtypes = parse_dtypes(args.dtype)
    print(f"PyTorch {torch.__version__}  device={device}")
    print("native-inference preserves exact clamp semantics; no-clamp is conditional")
    compile_enabled = bool(args.compile)

    for dtype in dtypes:
        print(f"\n[{str(dtype).removeprefix('torch.')}]")
        for case in CASES:
            try:
                x = torch.randn(case.shape, device=device, dtype=dtype)
                b = torch.randn(case.shape[1], device=device, dtype=dtype)
            except (RuntimeError, TypeError) as exc:
                print(f"{case.name:24s} skipped: {exc}")
                continue

            scaled_bias = b * case.gain
            reference = lambda value: legacy._bias_act_ref(
                value,
                b,
                act=case.act,
                gain=case.gain,
                clamp=case.clamp,
            )
            native_safe = lambda value: bias_act_native(
                value,
                b,
                act=case.act,
                gain=case.gain,
                clamp=case.clamp,
            )
            native_inference = lambda value: bias_act_native_inference(
                value,
                b,
                act=case.act,
                gain=case.gain,
                clamp=case.clamp,
            )
            implementations: list[tuple[str, Callable[[torch.Tensor], torch.Tensor]]] = [
                ("reference", reference),
                ("native-safe", native_safe),
                ("native-inference", native_inference),
                (
                    "native-homogeneous",
                    lambda value: bias_act_native_homogeneous_inference(
                        value,
                        scaled_bias,
                        act=case.act,
                        gain=case.gain,
                        clamp=case.clamp,
                    ),
                ),
            ]
            if case.clamp is not None:
                implementations.append((
                    "native-no-clamp",
                    lambda value: bias_act_native_inference(
                        value,
                        b,
                        act=case.act,
                        gain=case.gain,
                        clamp=None,
                    ),
                ))
                implementations.append((
                    "homogeneous-no-clamp",
                    lambda value: bias_act_native_homogeneous_inference(
                        value,
                        scaled_bias,
                        act=case.act,
                        gain=case.gain,
                        clamp=None,
                    ),
                ))
            if device.type == "cuda":
                implementations.append((
                    "legacy-cuda",
                    lambda value: legacy.bias_act(
                        value,
                        b,
                        act=case.act,
                        gain=case.gain,
                        clamp=case.clamp,
                        impl="cuda",
                    ),
                ))
            if compile_enabled:
                try:
                    compiled = torch.compile(
                        native_safe,
                        fullgraph=True,
                        mode=args.compile_mode,
                    )
                    compiled(x)
                    synchronize(device)
                    implementations.append(("native-compiled", compiled))
                except Exception as exc:
                    lines = [line.strip() for line in str(exc).splitlines() if line.strip()]
                    message = " | ".join(lines[:2]) if lines else type(exc).__name__
                    print(f"{case.name:24s} compile unavailable: {message}")
                    compile_enabled = False

            expected = reference(x)
            synchronize(device)
            valid = []
            for name, fn in implementations:
                try:
                    actual = fn(x)
                    synchronize(device)
                    tolerance = 3e-3 if dtype in (torch.float16, torch.bfloat16) else 2e-5
                    if not torch.allclose(actual, expected, rtol=tolerance, atol=tolerance):
                        error = float((actual.float() - expected.float()).abs().max())
                        print(f"{case.name:24s} {name:18s} mismatch max_abs={error:.3e}")
                        continue
                    valid.append((name, fn))
                except (RuntimeError, TypeError) as exc:
                    print(f"{case.name:24s} {name:18s} skipped: {exc}")

            timings = measure_implementations(
                valid, x, device, args.warmup, args.repeats
            )
            baseline = dict(timings).get("reference")
            rendered = []
            for name, elapsed in timings:
                speedup = baseline / elapsed if baseline is not None else float("nan")
                rendered.append(f"{name}={elapsed:.3f}ms ({speedup:.2f}x)")
            preclamp_max = float(
                bias_act_native(
                    x, b, act=case.act, gain=case.gain, clamp=None
                ).abs().amax()
            )
            print(
                f"{case.name:24s} preclamp_max={preclamp_max:.2f}  "
                + "  ".join(rendered)
            )

            del x, b, expected
            if device.type == "mps":
                torch.mps.empty_cache()
            elif device.type == "cuda":
                torch.cuda.empty_cache()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument("--dtype", nargs="+", default=("float32", "float16"))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--compile-mode",
        default="default",
        choices=("default", "reduce-overhead", "max-autotune"),
    )
    return parser


if __name__ == "__main__":
    benchmark(build_parser().parse_args())
