"""Benchmark native PyTorch upfirdn2d against StyleGAN reference and CUDA.

Examples:
    PYTHONPATH=. python tests/benchmark_upfirdn2d_native.py --device cuda
    PYTHONPATH=. python tests/benchmark_upfirdn2d_native.py --device cuda --compile --compile-mode reduce-overhead
    PYTHONPATH=. python tests/benchmark_upfirdn2d_native.py --device cuda --suite matrix --json-out tests/results/upfirdn_matrix.json
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Callable, Optional

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.StyleGAN2_mps.torch_utils.ops import upfirdn2d as legacy
from models.StyleGAN2_mps.torch_utils.ops.upfirdn2d_native import upfirdn2d_native
from tests.native_bench_common import (
    TimedResult,
    apply_layout,
    compile_callable,
    configure_cuda_backends,
    empty_cache,
    env_metadata,
    format_result_line,
    measure_implementations,
    memory_format_name,
    parse_dtypes,
    peak_memory_mib,
    probe_profiler_kernels,
    select_device,
    summarize_timing,
    synchronize,
    tolerance_for,
    write_json,
)


@dataclass(frozen=True)
class Case:
    name: str
    shape: tuple[int, int, int, int]
    up: int | tuple[int, int]
    down: int | tuple[int, int]
    padding: tuple[int, int, int, int]
    gain: float
    filter_taps: Optional[list[float]] = None
    separable: Optional[bool] = None
    flip_filter: bool = False


SMOKE_CASES = (
    Case("rgb_up2_256", (4, 3, 256, 256), 2, 1, (2, 1, 2, 1), 4.0),
    Case("feature_up2_128", (2, 128, 128, 128), 2, 1, (2, 1, 2, 1), 4.0),
    Case("fir_after_upconv_256", (2, 128, 257, 257), 1, 1, (1, 1, 1, 1), 4.0),
    Case("filter_same_256", (2, 128, 256, 256), 1, 1, (2, 1, 2, 1), 1.0),
    Case("down2_256", (2, 128, 256, 256), 1, 2, (1, 1, 1, 1), 1.0),
)


def stylegan_channel(resolution: int) -> int:
    return min(32768 // resolution, 512)


def build_matrix_cases() -> list[Case]:
    cases: list[Case] = list(SMOKE_CASES)
    # Actual StyleGAN config-F stages with representative modes.
    for res in (4, 8, 16, 32, 64, 128, 256, 512, 1024):
        ch = stylegan_channel(res)
        for batch in (1, 2, 4, 8):
            if batch * ch * res * res * 4 > 1_500_000_000:
                continue
            cases.append(
                Case(
                    f"filter_b{batch}_c{ch}_{res}",
                    (batch, ch, res, res),
                    1,
                    1,
                    (2, 1, 2, 1),
                    1.0,
                )
            )
            if res <= 512:
                cases.append(
                    Case(
                        f"rgb_up2_b{batch}_{res}",
                        (batch, 3, res, res),
                        2,
                        1,
                        (2, 1, 2, 1),
                        4.0,
                    )
                )
            if res >= 8:
                cases.append(
                    Case(
                        f"down2_b{batch}_c{ch}_{res}",
                        (batch, ch, res, res),
                        1,
                        2,
                        (1, 1, 1, 1),
                        1.0,
                    )
                )
            if res <= 256:
                cases.append(
                    Case(
                        f"fir_after_up_b{batch}_c{ch}_{res}",
                        (batch, ch, res + 1, res + 1),
                        1,
                        1,
                        (1, 1, 1, 1),
                        4.0,
                    )
                )
    # Filter-size and separable comparisons at a representative shape.
    for taps, sep in (
        (None, None),
        ([1.0, 1.0], True),
        ([1.0, 2.0, 1.0], True),
        ([1.0, 3.0, 3.0, 1.0], False),
        ([1.0, 2.0, 3.0, 3.0, 2.0, 1.0], True),
        ([1.0, 1.0, 3.0, 3.0, 1.0, 1.0, 1.0, 1.0], True),
    ):
        name = "identity" if taps is None else f"taps{len(taps)}_{'sep' if sep else 'dense'}"
        cases.append(
            Case(
                f"filt_{name}_128",
                (2, 128, 128, 128),
                1,
                1,
                (2, 1, 2, 1),
                1.0,
                filter_taps=taps,
                separable=sep,
            )
        )
        cases.append(
            Case(
                f"up2_{name}_64",
                (2, 64, 64, 64),
                2,
                1,
                (2, 1, 2, 1),
                4.0,
                filter_taps=taps,
                separable=sep,
            )
        )
    cases.append(
        Case("neg_pad_up2", (2, 32, 32, 32), 2, 1, (-1, 2, 0, 3), 4.0, flip_filter=True)
    )
    cases.append(
        Case("up2_down2", (2, 64, 64, 64), 2, 2, (3, 0, 1, 2), 1.5)
    )
    return cases


def make_filter(case: Case, device: torch.device) -> Optional[torch.Tensor]:
    if case.filter_taps is None and case.name.startswith(("filt_identity", "up2_identity")):
        return None
    if case.filter_taps is None:
        return legacy.setup_filter([1, 3, 3, 1], device=device)
    if case.separable is None:
        return legacy.setup_filter(case.filter_taps, device=device)
    return legacy.setup_filter(case.filter_taps, device=device, separable=case.separable)


def make_native_fn(
    f: Optional[torch.Tensor],
    case: Case,
    *,
    use_strided_down: Optional[bool] = None,
    weight_mode: str = "expand",
    separable_mode: str = "auto",
    impl_variant: str = "default",
) -> Callable[[torch.Tensor], torch.Tensor]:
    def fn(value: torch.Tensor) -> torch.Tensor:
        return upfirdn2d_native(
            value,
            f,
            up=case.up,
            down=case.down,
            padding=case.padding,
            flip_filter=case.flip_filter,
            gain=case.gain,
            use_strided_down=use_strided_down,
            weight_mode=weight_mode,
            separable_mode=separable_mode,
            impl_variant=impl_variant,
        )

    fn.__name__ = f"native_{impl_variant}_{weight_mode}_{separable_mode}"
    return fn


def estimate_bytes(shape: tuple[int, ...], dtype: torch.dtype) -> int:
    return int(torch.tensor([], dtype=dtype).element_size() * int(torch.tensor(shape).prod()))


def benchmark(args: argparse.Namespace) -> None:
    device = select_device(args.device)
    dtypes = parse_dtypes(args.dtype)
    backend = configure_cuda_backends(
        cudnn_benchmark=args.cudnn_benchmark,
        allow_tf32=args.allow_tf32,
    )
    cases = list(SMOKE_CASES) if args.suite == "smoke" else build_matrix_cases()
    payload = {
        "env": env_metadata(),
        "backend": backend,
        "args": vars(args),
        "results": [],
    }
    print(f"PyTorch {torch.__version__}  device={device}  suite={args.suite}")
    print(f"backend={backend}")

    for dtype in dtypes:
        print(f"\n[{str(dtype).removeprefix('torch.')}] layout={args.layout}")
        for case in cases:
            approx = estimate_bytes(case.shape, dtype) * 8
            if approx > args.memory_guard_bytes:
                print(f"{case.name:28s} skipped: memory guard ({approx} bytes)")
                continue
            try:
                x = torch.randn(case.shape, device=device, dtype=dtype)
                x = apply_layout(x, args.layout)
            except (RuntimeError, TypeError) as exc:
                print(f"{case.name:28s} skipped: {exc}")
                continue
            f = make_filter(case, device)
            skipped_pre: Optional[TimedResult] = None

            def reference(value: torch.Tensor, _f=f, _case=case) -> torch.Tensor:
                return legacy._upfirdn2d_ref(
                    value,
                    _f,
                    up=_case.up,
                    down=_case.down,
                    padding=_case.padding,
                    flip_filter=_case.flip_filter,
                    gain=_case.gain,
                )

            implementations: list[tuple[str, Callable]] = [
                ("reference", reference),
                ("native", make_native_fn(f, case)),
            ]
            if case.up == 1 and case.down != 1:
                implementations.append(
                    ("native-stride", make_native_fn(f, case, use_strided_down=True))
                )
                implementations.append(
                    ("native-slice", make_native_fn(f, case, use_strided_down=False))
                )
            if args.variants:
                for weight_mode in ("repeat", "contiguous", "cached"):
                    implementations.append(
                        (
                            f"native-{weight_mode}",
                            make_native_fn(f, case, weight_mode=weight_mode),
                        )
                    )
                if case.up == 1 and f is not None and f.ndim == 1:
                    implementations.append(
                        (
                            "native-separable",
                            make_native_fn(f, case, separable_mode="separable"),
                        )
                    )
                if case.up == 2 or case.up == (2, 2):
                    implementations.append(
                        (
                            "native-polyphase",
                            make_native_fn(f, case, impl_variant="polyphase"),
                        )
                    )
            if device.type == "cuda" and dtype is not torch.bfloat16:

                def legacy_cuda(value: torch.Tensor, _f=f, _case=case) -> torch.Tensor:
                    return legacy.upfirdn2d(
                        value,
                        _f,
                        up=_case.up,
                        down=_case.down,
                        padding=_case.padding,
                        flip_filter=_case.flip_filter,
                        gain=_case.gain,
                        impl="cuda",
                    )

                implementations.append(("legacy-cuda", legacy_cuda))
            elif device.type == "cuda" and dtype is torch.bfloat16:
                skipped_pre = TimedResult(
                    name="legacy-cuda",
                    median_ms=float("nan"),
                    mean_ms=float("nan"),
                    stdev_ms=float("nan"),
                    samples_ms=[],
                    max_abs_error=float("nan"),
                    mean_abs_error=float("nan"),
                    rel_error=float("nan"),
                    peak_allocated_mib=0.0,
                    peak_reserved_mib=0.0,
                    skipped=True,
                    skip_reason="legacy CUDA lacks BF16",
                )

            compile_meta: dict[str, tuple[float, list[str]]] = {}
            if args.compile:
                native_fn = make_native_fn(f, case)
                try:
                    compiled, compile_ms, notes = compile_callable(
                        native_fn,
                        (x,),
                        mode=args.compile_mode,
                        device=device,
                    )
                    implementations.append(("native-compiled", compiled))
                    compile_meta["native-compiled"] = (compile_ms, notes)
                    if args.profile_kernels:
                        notes.extend(probe_profiler_kernels(compiled, (x,), device))
                except Exception as exc:
                    lines = [line.strip() for line in str(exc).splitlines() if line.strip()]
                    message = " | ".join(lines[:2]) if lines else type(exc).__name__
                    print(f"{case.name:28s} compile unavailable: {message}")

            expected = reference(x)
            synchronize(device)
            valid: list[tuple[str, Callable]] = []
            outputs: dict[str, torch.Tensor] = {}
            skipped: list[TimedResult] = []
            if skipped_pre is not None:
                skipped.append(skipped_pre)
            rtol, atol = tolerance_for(dtype)
            for name, fn in implementations:
                try:
                    empty_cache(device)
                    actual = fn(x)
                    synchronize(device)
                    max_abs, mean_abs, rel = (
                        float((actual.float() - expected.float()).abs().max()),
                        float((actual.float() - expected.float()).abs().mean()),
                        float(
                            (actual.float() - expected.float()).abs().max()
                            / expected.float().abs().mean().clamp_min(1e-12)
                        ),
                    )
                    if not torch.allclose(actual, expected, rtol=rtol, atol=atol):
                        print(
                            f"{case.name:28s} {name:18s} mismatch max_abs={max_abs:.3e}"
                        )
                        skipped.append(
                            TimedResult(
                                name=name,
                                median_ms=float("nan"),
                                mean_ms=float("nan"),
                                stdev_ms=float("nan"),
                                samples_ms=[],
                                max_abs_error=max_abs,
                                mean_abs_error=mean_abs,
                                rel_error=rel,
                                peak_allocated_mib=0.0,
                                peak_reserved_mib=0.0,
                                skipped=True,
                                skip_reason=f"mismatch max_abs={max_abs:.3e}",
                            )
                        )
                        continue
                    valid.append((name, fn))
                    outputs[name] = actual.detach().clone()
                except Exception as exc:
                    skipped.append(
                        TimedResult(
                            name=name,
                            median_ms=float("nan"),
                            mean_ms=float("nan"),
                            stdev_ms=float("nan"),
                            samples_ms=[],
                            max_abs_error=float("nan"),
                            mean_abs_error=float("nan"),
                            rel_error=float("nan"),
                            peak_allocated_mib=0.0,
                            peak_reserved_mib=0.0,
                            skipped=True,
                            skip_reason=str(exc).splitlines()[0][:120],
                        )
                    )

            samples = measure_implementations(
                valid, (x,), device, args.warmup, args.repeats, rounds=args.rounds
            )
            results: list[TimedResult] = []
            for name, _ in valid:
                empty_cache(device)
                with torch.inference_mode():
                    _ = dict(valid)[name](x)
                synchronize(device)
                peak_a, peak_r = peak_memory_mib(device)
                compile_ms, notes = compile_meta.get(name, (None, []))
                results.append(
                    summarize_timing(
                        name,
                        samples[name],
                        actual=outputs[name],
                        expected=expected,
                        peak_allocated_mib=peak_a,
                        peak_reserved_mib=peak_r,
                        compile_ms=compile_ms,
                        layout_in=memory_format_name(x),
                        layout_out=memory_format_name(outputs[name]),
                        notes=notes,
                    )
                )
            results.extend(skipped)
            print(format_result_line(case.name, results))
            payload["results"].append(
                {
                    "case": case.name,
                    "dtype": str(dtype).removeprefix("torch."),
                    "layout": args.layout,
                    "shape": list(case.shape),
                    "up": case.up,
                    "down": case.down,
                    "padding": list(case.padding),
                    "gain": case.gain,
                    "flip_filter": case.flip_filter,
                    "implementations": results,
                }
            )
            del x, expected
            empty_cache(device)

    if args.json_out:
        write_json(args.json_out, payload)
        print(f"wrote {args.json_out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument("--dtype", nargs="+", default=("float32", "float16"))
    parser.add_argument("--layout", default="contiguous", choices=("contiguous", "channels_last"))
    parser.add_argument("--suite", default="smoke", choices=("smoke", "matrix"))
    parser.add_argument("--variants", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--compile-mode",
        default="default",
        choices=("default", "reduce-overhead", "max-autotune"),
    )
    parser.add_argument("--cudnn-benchmark", action="store_true", default=True)
    parser.add_argument("--no-cudnn-benchmark", action="store_false", dest="cudnn_benchmark")
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--profile-kernels", action="store_true")
    parser.add_argument("--memory-guard-bytes", type=int, default=2_000_000_000)
    parser.add_argument("--json-out", default="")
    return parser


if __name__ == "__main__":
    benchmark(build_parser().parse_args())
