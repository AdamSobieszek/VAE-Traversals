"""Benchmark the integrated StyleGAN2 b128 early-output generator.

This benchmark intentionally loads only the converted 128x128 generator.  It
times synthesis from one shared batch of W tensors, so mapping-network cost and
different random inputs cannot skew the CPU/MPS comparison.

Default precision policy:

* CPU: FP32 and FP16 (BF16 is available as an override);
* MPS: FP32 and FP16.

Every eager and compiled result is compared with the same backend's eager FP32
output before its timing is reported.  Unsupported device, dtype, or compiler
combinations are recorded as skipped instead of aborting the other cases.

The default compiler policy uses ``aot_eager`` on CPU and Inductor on MPS.
Per-device overrides keep the comparison usable when compiler support differs
between backends in a particular PyTorch nightly.

Run from ``train_traversals``::

    /opt/anaconda3/envs/manip311/bin/python \
        tests/benchmark_early_output_128.py
"""

from __future__ import annotations

import argparse
import gc
import os
import platform
import statistics
import sys
import time
import traceback
from collections.abc import Callable, Sequence
from typing import Any

import torch

TRAIN_TRAVERSALS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if TRAIN_TRAVERSALS_ROOT not in sys.path:
    sys.path.insert(0, TRAIN_TRAVERSALS_ROOT)
TESTS_ROOT = os.path.dirname(os.path.abspath(__file__))
if TESTS_ROOT not in sys.path:
    sys.path.insert(0, TESTS_ROOT)

from lib.config import GAN_WEIGHTS
from models.StyleGAN2_mps.early_output_model import load_generator_checkpoint
from models.StyleGAN2_mps.torch_utils.ops.inference_opt import (
    InferenceOptConfig,
    normalize_scalar_attrs,
)
from models.StyleGAN2_mps.torch_utils.ops.native_backend import use_native_ops
from models.StyleGAN2_mps.torch_utils.ops.optimized_synthesis import (
    build_optimized_early_output_synthesis,
)
from native_bench_common import error_stats, synchronize, write_json


BATCH_SIZE = 8
OUTPUT_RESOLUTION = 128
DEFAULT_WEIGHT_KEY = "early_output_128"
DEFAULT_OUTPUT = "tests/results/early_output_128_b8_cpu_mps.json"

PRECISION_DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}
DEFAULT_MPS_PRECISIONS = ("fp32", "fp16")


def default_weights_path() -> str:
    relative = GAN_WEIGHTS["StyleGAN2"]["weights"][DEFAULT_WEIGHT_KEY]
    return os.path.join(TRAIN_TRAVERSALS_ROOT, relative)


def device_available(device_name: str) -> tuple[bool, str]:
    if device_name == "cpu":
        return True, ""
    if device_name == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_built():
            return False, "this PyTorch build has no MPS support"
        if not torch.backends.mps.is_available():
            return False, "MPS is not available on this machine"
        return True, ""
    return False, f"unsupported benchmark device {device_name!r}"


def make_ws(generator, batch: int, seed: int) -> torch.Tensor:
    cpu_rng = torch.Generator(device="cpu")
    cpu_rng.manual_seed(seed)
    z = torch.randn(batch, generator.z_dim, generator=cpu_rng)
    with torch.inference_mode():
        ws = generator.mapping(z, None)
    return ws.detach().clone()


def load_small_generator(weights: str, device: torch.device):
    generator = load_generator_checkpoint(weights, device=device)
    if generator.img_resolution != OUTPUT_RESOLUTION:
        raise ValueError(
            f"Expected the integrated {OUTPUT_RESOLUTION}x{OUTPUT_RESOLUTION} "
            f"checkpoint, got {generator.img_resolution}x{generator.img_resolution}"
        )
    if generator.num_ws != 12:
        raise ValueError(f"Expected the b128 prefix to use 12 W slots, got {generator.num_ws}")
    if generator.synthesis.block_resolutions[-1] != OUTPUT_RESOLUTION:
        raise ValueError("Checkpoint synthesis extends beyond b128")
    normalize_scalar_attrs(generator)
    generator.eval().requires_grad_(False)
    return generator


def timing_samples(
    fn: Callable[[], torch.Tensor],
    *,
    device: torch.device,
    warmup: int,
    repeats: int,
    rounds: int,
) -> list[float]:
    samples: list[float] = []
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        synchronize(device)
        for _ in range(rounds):
            start = time.perf_counter()
            for _ in range(repeats):
                fn()
            synchronize(device)
            samples.append((time.perf_counter() - start) * 1000.0 / repeats)
    return samples


def summarize(samples: Sequence[float], batch: int) -> dict[str, Any]:
    median_ms = statistics.median(samples)
    return {
        "median_ms": median_ms,
        "mean_ms": statistics.mean(samples),
        "stdev_ms": statistics.pstdev(samples) if len(samples) > 1 else 0.0,
        "samples_ms": list(samples),
        "ms_per_image": median_ms / batch,
        "images_per_second": batch * 1000.0 / median_ms,
    }


def correctness(actual: torch.Tensor, expected: torch.Tensor, precision: str) -> dict[str, Any]:
    max_abs, mean_abs, relative = error_stats(actual, expected)
    tolerance = 3e-3 if precision == "fp16" else 8e-3 if precision == "bf16" else 2e-5
    return {
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "max_over_mean_abs": relative,
        "tolerance": tolerance,
        "allclose": bool(
            torch.allclose(
                actual.float(),
                expected.float(),
                rtol=tolerance,
                atol=tolerance,
            )
        ),
    }


def compile_forward(
    fn: Callable[[torch.Tensor], torch.Tensor],
    ws: torch.Tensor,
    *,
    device: torch.device,
    backend: str,
    mode: str,
    fullgraph: bool,
) -> tuple[Callable[[torch.Tensor], torch.Tensor], torch.Tensor, float]:
    if hasattr(torch, "_dynamo"):
        torch._dynamo.reset()
        torch._dynamo.config.cache_size_limit = 128
        if hasattr(torch._dynamo.config, "recompile_limit"):
            torch._dynamo.config.recompile_limit = 128
    compiled = torch.compile(
        fn,
        backend=backend,
        mode=mode,
        fullgraph=fullgraph,
    )
    synchronize(device)
    start = time.perf_counter()
    with torch.inference_mode(), use_native_ops(upfirdn="native", bias_act="native"):
        output = compiled(ws)
    synchronize(device)
    compile_ms = (time.perf_counter() - start) * 1000.0
    return compiled, output, compile_ms


def run_precision_case(
    generator,
    ws: torch.Tensor,
    reference: torch.Tensor,
    *,
    device: torch.device,
    precision: str,
    compiler_backend: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    low_precision = "bf16" if precision == "bf16" else "fp16"
    force_fp32 = precision == "fp32"
    result: dict[str, Any] = {
        "device": device.type,
        "precision": precision,
        "compute_dtype": str(PRECISION_DTYPES[precision]).removeprefix("torch."),
        "compiler_backend": compiler_backend,
        "batch": args.batch,
        "output_resolution": OUTPUT_RESOLUTION,
        "status": "pending",
    }

    try:
        # The CUDA B64 policy composes upsampling FIR filters into polyphase
        # convolution weights. MPS has a 65,536-channel convolution limit, so
        # this keeps shared modulation but uses ordinary backend resampling.
        optimization_config = InferenceOptConfig(
            shared_modconv=True,
            fir_compose=False,
            fused_modconv=False,
            low_precision=low_precision,
        )
        optimized = build_optimized_early_output_synthesis(
            generator.synthesis,
            optimization_config,
            copy_module=True,
        ).to(device)

        def forward(current_ws: torch.Tensor) -> torch.Tensor:
            return optimized(
                current_ws,
                noise_mode="const",
                force_fp32=force_fp32,
            )

        with torch.inference_mode(), use_native_ops(
            upfirdn="native", bias_act="native"
        ):
            eager_output = forward(ws)
        synchronize(device)
        result["eager_correctness_vs_fp32"] = correctness(
            eager_output, reference, precision
        )
        with use_native_ops(upfirdn="native", bias_act="native"):
            eager_samples = timing_samples(
                lambda: forward(ws),
                device=device,
                warmup=args.warmup,
                repeats=args.repeats,
                rounds=args.rounds,
            )
        result["eager"] = summarize(eager_samples, args.batch)

        compiled, compiled_output, compile_ms = compile_forward(
            forward,
            ws,
            device=device,
            backend=compiler_backend,
            mode=args.compile_mode,
            fullgraph=args.fullgraph,
        )
        result["compile_ms"] = compile_ms
        result["compiled_correctness_vs_fp32"] = correctness(
            compiled_output, reference, precision
        )
        result["compiled_correctness_vs_eager"] = correctness(
            compiled_output, eager_output, precision
        )
        with use_native_ops(upfirdn="native", bias_act="native"):
            compiled_samples = timing_samples(
                lambda: compiled(ws),
                device=device,
                warmup=args.warmup,
                repeats=args.repeats,
                rounds=args.rounds,
            )
        result["compiled"] = summarize(compiled_samples, args.batch)
        result["compile_speedup_over_eager"] = (
            result["eager"]["median_ms"] / result["compiled"]["median_ms"]
        )
        result["status"] = "ok"
    except Exception as exc:  # Keep the remaining backend/precision cases useful.
        result["status"] = "skipped"
        result["skip_reason"] = f"{type(exc).__name__}: {str(exc)[:1000]}"
        result["traceback"] = traceback.format_exc(limit=12)
    return result


def compare_backends(
    cases: Sequence[dict[str, Any]], cpu_low_precision: str
) -> dict[str, Any]:
    successful = {
        (case["device"], case["precision"]): case
        for case in cases
        if case.get("status") == "ok"
    }
    comparisons: dict[str, Any] = {}

    def add(name: str, cpu_key: tuple[str, str], mps_key: tuple[str, str]) -> None:
        cpu_case = successful.get(cpu_key)
        mps_case = successful.get(mps_key)
        if cpu_case is None or mps_case is None:
            comparisons[name] = {"status": "unavailable"}
            return
        cpu_ms = cpu_case["compiled"]["median_ms"]
        mps_ms = mps_case["compiled"]["median_ms"]
        comparisons[name] = {
            "status": "ok",
            "cpu_ms": cpu_ms,
            "mps_ms": mps_ms,
            "mps_speedup_over_cpu": cpu_ms / mps_ms,
        }

    add("compiled_fp32", ("cpu", "fp32"), ("mps", "fp32"))
    add(
        "compiled_low_precision",
        ("cpu", cpu_low_precision),
        ("mps", "fp16"),
    )
    return comparisons


def environment_metadata() -> dict[str, Any]:
    return {
        "torch": torch.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_threads": torch.get_num_threads(),
        "mps_built": bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_built()
        ),
        "mps_available": bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.batch != BATCH_SIZE:
        raise ValueError(f"This comparison intentionally uses batch {BATCH_SIZE}")
    if args.cpu_threads > 0:
        torch.set_num_threads(args.cpu_threads)
    if not os.path.isfile(args.weights):
        raise FileNotFoundError(f"Early-output checkpoint not found: {args.weights}")

    cpu_generator = load_small_generator(args.weights, torch.device("cpu"))
    ws_cpu = make_ws(cpu_generator, args.batch, args.seed)
    del cpu_generator
    gc.collect()

    cases: list[dict[str, Any]] = []
    device_status: dict[str, Any] = {}
    for device_name in args.devices:
        available, reason = device_available(device_name)
        device_status[device_name] = {"available": available, "reason": reason}
        if not available:
            continue
        device = torch.device(device_name)
        generator = load_small_generator(args.weights, device)
        ws = ws_cpu.to(device)
        with torch.inference_mode(), use_native_ops(
            upfirdn="native", bias_act="native"
        ):
            reference = generator.synthesis(
                ws,
                noise_mode="const",
                force_fp32=True,
            )
        synchronize(device)

        precisions = (
            tuple(args.cpu_precisions or ("fp32", args.cpu_low_precision))
            if device_name == "cpu"
            else tuple(args.mps_precisions or DEFAULT_MPS_PRECISIONS)
        )
        for precision in precisions:
            print(
                f"#. Benchmarking {device_name.upper()} {precision.upper()}...",
                flush=True,
            )
            case = run_precision_case(
                generator,
                ws,
                reference,
                device=device,
                precision=precision,
                compiler_backend=(
                    args.cpu_backend or args.backend
                    if device_name == "cpu"
                    else args.mps_backend or args.backend
                ),
                args=args,
            )
            cases.append(case)
            if case["status"] == "skipped":
                print(f"  \\__Skipped: {case['skip_reason']}", flush=True)

        del reference, ws, generator
        gc.collect()
        if device_name == "mps":
            torch.mps.empty_cache()

    return {
        "benchmark": "stylegan2_integrated_early_output_128_b8",
        "weights": os.path.abspath(args.weights),
        "environment": environment_metadata(),
        "arguments": vars(args),
        "device_status": device_status,
        "cases": cases,
        "comparisons": compare_backends(cases, args.cpu_low_precision),
    }


def print_report(payload: dict[str, Any]) -> None:
    print("\nCompiled synthesis results (batch 8)")
    print(f"{'device':<8} {'dtype':<7} {'median ms':>11} {'ms/image':>11} {'images/s':>11} {'compile ms':>12}")
    for case in payload["cases"]:
        if case["status"] != "ok":
            print(f"{case['device']:<8} {case['precision']:<7} {'SKIPPED':>11}")
            continue
        row = case["compiled"]
        print(
            f"{case['device']:<8} {case['precision']:<7} "
            f"{row['median_ms']:>11.3f} {row['ms_per_image']:>11.3f} "
            f"{row['images_per_second']:>11.2f} {case['compile_ms']:>12.1f}"
        )

    print("\nMPS speedup over CPU")
    for name, comparison in payload["comparisons"].items():
        if comparison["status"] != "ok":
            print(f"  {name}: unavailable")
        else:
            print(
                f"  {name}: {comparison['mps_speedup_over_cpu']:.2f}x "
                f"(CPU {comparison['cpu_ms']:.3f} ms, "
                f"MPS {comparison['mps_ms']:.3f} ms)"
            )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark compiled batch-8 inference for the integrated StyleGAN2 "
            "128x128 early-output generator on CPU and MPS"
        )
    )
    parser.add_argument("--weights", default=default_weights_path())
    parser.add_argument("--devices", nargs="+", choices=("cpu", "mps"), default=("cpu", "mps"))
    parser.add_argument("--batch", type=int, default=BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument(
        "--backend",
        default="aot_eager",
        help="torch.compile backend (use inductor when supported by the build)",
    )
    parser.add_argument(
        "--cpu-backend",
        default="aot_eager",
        help="override --backend for CPU (default: aot_eager)",
    )
    parser.add_argument(
        "--mps-backend",
        default="inductor",
        help="override --backend for MPS (default: inductor)",
    )
    parser.add_argument("--compile-mode", default="default")
    parser.add_argument("--fullgraph", action="store_true")
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=0,
        help="CPU intra-op threads; 0 keeps the PyTorch default",
    )
    parser.add_argument(
        "--cpu-low-precision",
        choices=("fp16", "bf16"),
        default="fp16",
        help="CPU comparison dtype; BF16 can be preferable on supported CPUs",
    )
    parser.add_argument(
        "--cpu-precisions",
        nargs="+",
        choices=tuple(PRECISION_DTYPES),
        help="explicit CPU precision cases (overrides --cpu-low-precision)",
    )
    parser.add_argument(
        "--mps-precisions",
        nargs="+",
        choices=tuple(PRECISION_DTYPES),
        help="explicit MPS precision cases",
    )
    parser.add_argument("--out", default=DEFAULT_OUTPUT)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    args.weights = os.path.abspath(args.weights)
    payload = run(args)
    output_path = (
        args.out
        if os.path.isabs(args.out)
        else os.path.join(TRAIN_TRAVERSALS_ROOT, args.out)
    )
    write_json(output_path, payload)
    print_report(payload)
    print(f"\nwrote {output_path}")


if __name__ == "__main__":
    main()
