"""Benchmark native PyTorch bias_act alternatives on StyleGAN-like shapes."""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from typing import Callable, Optional

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
    shape: tuple[int, ...]
    act: str
    gain: float
    clamp: float | None


SMOKE_CASES = (
    Case("mapping_b4_c512", (4, 512), "lrelu", math.sqrt(2.0), None),
    Case("synth_b2_c512_64", (2, 512, 64, 64), "lrelu", math.sqrt(2.0), 256.0),
    Case("synth_b2_c256_128", (2, 256, 128, 128), "lrelu", math.sqrt(2.0), 256.0),
    Case("synth_b2_c128_256", (2, 128, 256, 256), "lrelu", math.sqrt(2.0), 256.0),
    Case("synth_b1_c64_512", (1, 64, 512, 512), "lrelu", math.sqrt(2.0), 256.0),
    Case("torgb_b2_c3_256", (2, 3, 256, 256), "linear", 1.0, 256.0),
)


def stylegan_channel(resolution: int) -> int:
    return min(32768 // resolution, 512)


def build_matrix_cases() -> list[Case]:
    cases: list[Case] = list(SMOKE_CASES)
    for res in (4, 8, 16, 32, 64, 128, 256, 512, 1024):
        ch = stylegan_channel(res)
        for batch in (1, 2, 4, 8):
            if batch * ch * res * res * 2 > 1_500_000_000:
                continue
            cases.append(
                Case(
                    f"synth_b{batch}_c{ch}_{res}",
                    (batch, ch, res, res),
                    "lrelu",
                    math.sqrt(2.0),
                    256.0,
                )
            )
            if ch >= 3:
                cases.append(
                    Case(
                        f"torgb_b{batch}_{res}",
                        (batch, 3, res, res),
                        "linear",
                        1.0,
                        256.0,
                    )
                )
    for batch in (1, 2, 4, 8):
        cases.append(
            Case(f"mapping_b{batch}_c512", (batch, 512), "lrelu", math.sqrt(2.0), None)
        )
    return cases


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
    print("native-inference preserves exact clamp semantics; no-clamp is conditional")

    for dtype in dtypes:
        print(f"\n[{str(dtype).removeprefix('torch.')}] layout={args.layout}")
        for case in cases:
            try:
                x = torch.randn(case.shape, device=device, dtype=dtype)
                if len(case.shape) == 4:
                    x = apply_layout(x, args.layout)
                b = torch.randn(case.shape[1], device=device, dtype=dtype)
            except (RuntimeError, TypeError, ValueError) as exc:
                print(f"{case.name:28s} skipped: {exc}")
                continue

            scaled_bias = b * case.gain
            # Dynamic scaled bias recomputes every call (launch cost probe).
            def homogeneous_dynamic(value: torch.Tensor, _b=b, _case=case) -> torch.Tensor:
                return bias_act_native_homogeneous_inference(
                    value,
                    _b * _case.gain,
                    act=_case.act,
                    gain=_case.gain,
                    clamp=_case.clamp,
                )

            def reference(value: torch.Tensor, _b=b, _case=case) -> torch.Tensor:
                return legacy._bias_act_ref(
                    value, _b, act=_case.act, gain=_case.gain, clamp=_case.clamp
                )

            def native_safe(value: torch.Tensor, _b=b, _case=case) -> torch.Tensor:
                return bias_act_native(
                    value, _b, act=_case.act, gain=_case.gain, clamp=_case.clamp
                )

            native_safe.__name__ = "bias_act_native_safe"

            implementations: list[tuple[str, Callable]] = [
                ("reference", reference),
                ("native-safe", native_safe),
                (
                    "native-inference",
                    lambda value, _b=b, _case=case: bias_act_native_inference(
                        value, _b, act=_case.act, gain=_case.gain, clamp=_case.clamp
                    ),
                ),
                (
                    "native-homogeneous",
                    lambda value, _sb=scaled_bias, _case=case: bias_act_native_homogeneous_inference(
                        value,
                        _sb,
                        act=_case.act,
                        gain=_case.gain,
                        clamp=_case.clamp,
                    ),
                ),
                ("homogeneous-dynamic-bias", homogeneous_dynamic),
            ]
            if case.clamp is not None:
                implementations.extend(
                    [
                        (
                            "native-no-clamp",
                            lambda value, _b=b, _case=case: bias_act_native_inference(
                                value, _b, act=_case.act, gain=_case.gain, clamp=None
                            ),
                        ),
                        (
                            "homogeneous-no-clamp",
                            lambda value, _sb=scaled_bias, _case=case: bias_act_native_homogeneous_inference(
                                value,
                                _sb,
                                act=_case.act,
                                gain=_case.gain,
                                clamp=None,
                            ),
                        ),
                    ]
                )
            skipped_pre: Optional[TimedResult] = None
            if device.type == "cuda" and dtype is not torch.bfloat16:

                def legacy_cuda(value: torch.Tensor, _b=b, _case=case) -> torch.Tensor:
                    return legacy.bias_act(
                        value,
                        _b,
                        act=_case.act,
                        gain=_case.gain,
                        clamp=_case.clamp,
                        impl="cuda",
                    )

                implementations.append(("legacy-cuda", legacy_cuda))
            elif device.type == "cuda":
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
                for label, fn in (
                    ("native-compiled", native_safe),
                    (
                        "homogeneous-compiled",
                        lambda value, _sb=scaled_bias, _case=case: bias_act_native_homogeneous_inference(
                            value,
                            _sb,
                            act=_case.act,
                            gain=_case.gain,
                            clamp=_case.clamp,
                        ),
                    ),
                ):
                    try:
                        compiled, compile_ms, notes = compile_callable(
                            fn, (x,), mode=args.compile_mode, device=device
                        )
                        implementations.append((label, compiled))
                        compile_meta[label] = (compile_ms, notes)
                        if args.profile_kernels:
                            notes.extend(probe_profiler_kernels(compiled, (x,), device))
                    except Exception as exc:
                        lines = [
                            line.strip() for line in str(exc).splitlines() if line.strip()
                        ]
                        message = " | ".join(lines[:2]) if lines else type(exc).__name__
                        print(f"{case.name:28s} {label} unavailable: {message}")

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
                    if name.endswith("no-clamp"):
                        # Conditional experiment: compare against unclamped reference.
                        ref_cmp = bias_act_native(
                            x, b, act=case.act, gain=case.gain, clamp=None
                        )
                    else:
                        ref_cmp = expected
                    max_abs = float((actual.float() - ref_cmp.float()).abs().max())
                    if not torch.allclose(actual, ref_cmp, rtol=rtol, atol=atol):
                        print(
                            f"{case.name:28s} {name:22s} mismatch max_abs={max_abs:.3e}"
                        )
                        skipped.append(
                            TimedResult(
                                name=name,
                                median_ms=float("nan"),
                                mean_ms=float("nan"),
                                stdev_ms=float("nan"),
                                samples_ms=[],
                                max_abs_error=max_abs,
                                mean_abs_error=float("nan"),
                                rel_error=float("nan"),
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
                ref_cmp = (
                    bias_act_native(x, b, act=case.act, gain=case.gain, clamp=None)
                    if name.endswith("no-clamp")
                    else expected
                )
                results.append(
                    summarize_timing(
                        name,
                        samples[name],
                        actual=outputs[name],
                        expected=ref_cmp,
                        peak_allocated_mib=peak_a,
                        peak_reserved_mib=peak_r,
                        compile_ms=compile_ms,
                        layout_in=memory_format_name(x) if x.ndim == 4 else "n/a",
                        layout_out=memory_format_name(outputs[name])
                        if outputs[name].ndim == 4
                        else "n/a",
                        notes=notes,
                    )
                )
            results.extend(skipped)
            preclamp_max = float(
                bias_act_native(x, b, act=case.act, gain=case.gain, clamp=None)
                .abs()
                .amax()
            )
            print(
                format_result_line(
                    case.name, results, extra=f"preclamp_max={preclamp_max:.2f}"
                )
            )
            payload["results"].append(
                {
                    "case": case.name,
                    "dtype": str(dtype).removeprefix("torch."),
                    "layout": args.layout,
                    "shape": list(case.shape),
                    "act": case.act,
                    "gain": case.gain,
                    "clamp": case.clamp,
                    "preclamp_max": preclamp_max,
                    "implementations": results,
                }
            )
            del x, b, expected
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
    parser.add_argument("--json-out", default="")
    return parser


if __name__ == "__main__":
    benchmark(build_parser().parse_args())
