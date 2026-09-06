"""CUDA benchmark: compiled full StyleGAN2 vs compiled early-output generator.

This is based on ``benchmark_b64_safe.py``, but compares two model topologies
instead of comparing FP32 with an optimized candidate.  For each requested
precision it builds and compiles:

* the original FFHQ synthesis through 1024;
* the early-output synthesis through b256 plus its trained RGB decoder.

Both paths use the same prepared shared-modconv/polyphase-FIR implementation,
the same latent samples, constant noise, and the original model's CUDA-native
operators.  FP16 and BF16 are run as separate compiler graphs.  The benchmark
reports latency, module tensor footprint, output bytes, absolute CUDA peaks,
and incremental peak allocation above each call's resident baseline.

Example on a large NVIDIA GPU::

    cd train_traversals
    PYTHONPATH=. python tests/benchmark_early_output_cuda.py \
        --ffhq-weights /path/to/stylegan2-ffhq-1024x1024.pkl \
        --early-weights experiments/stylegan_early_output/early_output_generator.pt \
        --batch 4 --precisions fp16 bf16

Use ``--smoke`` first to compile and execute one sample per precision.
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import statistics
import sys
import time
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if TEST_ROOT not in sys.path:
    sys.path.insert(0, TEST_ROOT)

from models.StyleGAN2_mps.early_output_model import (  # noqa: E402
    OUTPUT_RESOLUTION,
    _load_full_source_generator,
    load_generator_checkpoint,
)
from models.StyleGAN2_mps.torch_utils.ops.inference_opt import (  # noqa: E402
    b64_compile_config,
    normalize_scalar_attrs,
)
from models.StyleGAN2_mps.torch_utils.ops.native_backend import (  # noqa: E402
    use_native_ops,
)
from models.StyleGAN2_mps.torch_utils.ops.optimized_synthesis import (  # noqa: E402
    build_optimized_early_output_synthesis,
    build_optimized_synthesis,
)
from native_bench_common import (  # noqa: E402
    configure_cuda_backends,
    empty_cache,
    env_metadata,
    error_stats,
    probe_profiler_kernels,
    synchronize,
    write_json,
)


MIB = 1024**2


def module_footprint(module: nn.Module) -> dict[str, float | int]:
    parameter_count = sum(parameter.numel() for parameter in module.parameters())
    parameter_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in module.parameters()
    )
    buffer_bytes = sum(
        buffer.numel() * buffer.element_size() for buffer in module.buffers()
    )
    state_bytes = sum(
        tensor.numel() * tensor.element_size()
        for tensor in module.state_dict().values()
    )
    return {
        "parameter_count": parameter_count,
        "parameter_mib": parameter_bytes / MIB,
        "buffer_mib": buffer_bytes / MIB,
        "module_tensor_mib": (parameter_bytes + buffer_bytes) / MIB,
        "persistent_state_mib": state_bytes / MIB,
    }


def make_latents(
    full_generator: nn.Module,
    early_generator: nn.Module,
    batch: int,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    random = torch.Generator(device="cpu")
    random.manual_seed(seed)
    z = torch.randn(batch, full_generator.z_dim, generator=random).to(device)
    with torch.inference_mode():
        full_ws = full_generator.mapping(z, None)
        early_ws = early_generator.mapping(z, None)
    if not torch.equal(full_ws[:, : early_ws.shape[1]], early_ws):
        difference = float(
            (full_ws[:, : early_ws.shape[1]] - early_ws).abs().max().cpu()
        )
        raise ValueError(
            "Full and early mapping networks differ; use an integrated checkpoint "
            f"converted from the supplied FFHQ weights (max W difference {difference:.3e})"
        )
    return full_ws.detach().clone(), early_ws.detach().clone()


def compiler_counters() -> dict[str, dict[str, int]]:
    counters = getattr(torch._dynamo.utils, "counters", {})
    result: dict[str, dict[str, int]] = {}
    for group in ("frames", "stats", "graph_break", "inductor", "aot_autograd"):
        values = counters.get(group, {})
        if values:
            result[group] = {str(key): int(value) for key, value in values.items()}
    return result


def compile_forward(
    function: Callable[[torch.Tensor], torch.Tensor],
    example: torch.Tensor,
    *,
    mode: str,
    fullgraph: bool,
    device: torch.device,
) -> tuple[Callable[[torch.Tensor], torch.Tensor], float, dict[str, Any]]:
    torch._dynamo.utils.counters.clear()
    torch._dynamo.config.cache_size_limit = 128
    torch._dynamo.config.recompile_limit = 128
    compiled = torch.compile(function, mode=mode, fullgraph=fullgraph)
    synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode(), use_native_ops(
        upfirdn="native", bias_act="native"
    ):
        compiled(example)
    synchronize(device)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return compiled, elapsed_ms, compiler_counters()


def call_compiled(
    function: Callable[[torch.Tensor], torch.Tensor], ws: torch.Tensor
) -> torch.Tensor:
    with use_native_ops(upfirdn="native", bias_act="native"):
        return function(ws)


def measure_compiled(
    implementations: list[tuple[str, Callable[[], torch.Tensor]]],
    *,
    device: torch.device,
    batch: int,
    warmup: int,
    repeats: int,
    rounds: int,
) -> dict[str, dict[str, float]]:
    samples = {name: [] for name, _ in implementations}
    baseline_allocations = {name: [] for name, _ in implementations}
    absolute_peaks = {name: [] for name, _ in implementations}
    incremental_peaks = {name: [] for name, _ in implementations}
    reserved_peaks = {name: [] for name, _ in implementations}

    with torch.inference_mode():
        for _, function in implementations:
            for _ in range(warmup):
                function()
        synchronize(device)

        for round_index in range(rounds):
            ordered = (
                implementations
                if round_index % 2 == 0
                else list(reversed(implementations))
            )
            for name, function in ordered:
                empty_cache(device)
                resident = torch.cuda.memory_allocated(device)
                torch.cuda.reset_peak_memory_stats(device)
                started = time.perf_counter()
                for _ in range(repeats):
                    function()
                synchronize(device)
                elapsed_ms = (time.perf_counter() - started) * 1000.0 / repeats
                peak = torch.cuda.max_memory_allocated(device)
                reserved = torch.cuda.max_memory_reserved(device)
                samples[name].append(elapsed_ms)
                baseline_allocations[name].append(resident / MIB)
                absolute_peaks[name].append(peak / MIB)
                incremental_peaks[name].append(max(0, peak - resident) / MIB)
                reserved_peaks[name].append(reserved / MIB)

    return {
        name: {
            "median_ms": statistics.median(values),
            "mean_ms": statistics.mean(values),
            "stdev_ms": statistics.stdev(values) if len(values) > 1 else 0.0,
            "ms_per_image": statistics.median(values) / batch,
            "resident_allocated_mib": statistics.median(
                baseline_allocations[name]
            ),
            "peak_allocated_mib": max(absolute_peaks[name]),
            "incremental_peak_mib": max(incremental_peaks[name]),
            "peak_reserved_mib": max(reserved_peaks[name]),
        }
        for name, values in samples.items()
    }


def reconstruction_stats(
    early_output: torch.Tensor, full_output: torch.Tensor
) -> dict[str, float]:
    target = F.interpolate(
        full_output.float(),
        size=(OUTPUT_RESOLUTION, OUTPUT_RESOLUTION),
        mode="area",
    )
    difference = early_output.float() - target
    mse = float(difference.square().mean().item())
    mae = float(difference.abs().mean().item())
    return {
        "mse": mse,
        "mae": mae,
        "psnr_db_range_minus1_1": 10.0 * math.log10(4.0 / max(mse, 1e-12)),
        "target_min": float(target.min().item()),
        "target_max": float(target.max().item()),
        "prediction_min": float(early_output.min().item()),
        "prediction_max": float(early_output.max().item()),
    }


def theoretical_memory(
    batch: int,
    b512_channels: int,
    b1024_channels: int,
    decoder_hidden_channels: int,
) -> dict[str, Any]:
    low_bytes = 2
    fp32_bytes = 4
    b512_features = batch * b512_channels * 512 * 512 * low_bytes
    b1024_features = batch * b1024_channels * 1024 * 1024 * low_bytes
    full_output = batch * 3 * 1024 * 1024 * fp32_bytes
    early_output = batch * 3 * 256 * 256 * fp32_bytes
    decoder_hidden = (
        batch * decoder_hidden_channels * 256 * 256 * low_bytes
    )
    decoder_expanded = (
        batch * 2 * decoder_hidden_channels * 256 * 256 * low_bytes
    )
    return {
        "removed_b512_feature_mib": b512_features / MIB,
        "removed_b1024_feature_mib": b1024_features / MIB,
        "removed_tail_feature_workload_mib": (
            b512_features + b1024_features
        )
        / MIB,
        "full_output_mib": full_output / MIB,
        "early_output_mib": early_output / MIB,
        "output_tensor_saving_mib": (full_output - early_output) / MIB,
        "representative_decoder_hidden_mib": decoder_hidden / MIB,
        "representative_decoder_expanded_mib": decoder_expanded / MIB,
        "note": (
            "Feature workload values sum tensors produced by removed blocks; "
            "they are not all simultaneously live. Measured incremental peak is authoritative."
        ),
    }


def decoder_hidden_channels(decoder: nn.Module) -> int:
    return int(decoder.mix_in.out_channels)


def run_precision(
    precision: str,
    full_generator: nn.Module,
    early_generator: nn.Module,
    full_ws: torch.Tensor,
    early_ws: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        return {"skipped": True, "reason": "CUDA device does not support BF16"}

    cfg = b64_compile_config(low_precision=precision)
    print(f"#. [{precision}] preparing original optimized synthesis...")
    full_optimized = build_optimized_synthesis(full_generator.synthesis, cfg).to(device)
    print(f"#. [{precision}] preparing early optimized synthesis...")
    early_optimized = build_optimized_early_output_synthesis(
        early_generator.synthesis,
        cfg,
        copy_module=True,
    ).to(device)
    normalize_scalar_attrs(full_optimized)
    normalize_scalar_attrs(early_optimized)

    def full_raw(ws: torch.Tensor) -> torch.Tensor:
        return full_optimized(ws, noise_mode="const", force_fp32=False)

    def early_raw(ws: torch.Tensor) -> torch.Tensor:
        return early_optimized(ws, noise_mode="const", force_fp32=False)

    with torch.inference_mode(), use_native_ops(
        upfirdn="native", bias_act="native"
    ):
        full_eager = full_raw(full_ws)
        early_eager = early_raw(early_ws)

    print(f"#. [{precision}] compiling original 1024 graph...")
    full_compiled, full_compile_ms, full_counters = compile_forward(
        full_raw,
        full_ws,
        mode=args.compile_mode,
        fullgraph=args.fullgraph,
        device=device,
    )
    print(f"#. [{precision}] compiling early-output graph...")
    early_compiled, early_compile_ms, early_counters = compile_forward(
        early_raw,
        early_ws,
        mode=args.compile_mode,
        fullgraph=args.fullgraph,
        device=device,
    )

    with torch.inference_mode():
        full_compiled_output = call_compiled(full_compiled, full_ws)
        early_compiled_output = call_compiled(early_compiled, early_ws)
        full_repeat = call_compiled(full_compiled, full_ws)
        early_repeat = call_compiled(early_compiled, early_ws)

    full_compile_error = dict(
        zip(
            ("max_abs", "mean_abs", "max_over_mean_abs"),
            error_stats(full_compiled_output, full_eager),
        )
    )
    early_compile_error = dict(
        zip(
            ("max_abs", "mean_abs", "max_over_mean_abs"),
            error_stats(early_compiled_output, early_eager),
        )
    )
    full_repeat_error = dict(
        zip(
            ("max_abs", "mean_abs", "max_over_mean_abs"),
            error_stats(full_repeat, full_compiled_output),
        )
    )
    early_repeat_error = dict(
        zip(
            ("max_abs", "mean_abs", "max_over_mean_abs"),
            error_stats(early_repeat, early_compiled_output),
        )
    )

    implementations = [
        ("original", lambda: call_compiled(full_compiled, full_ws)),
        ("early_output", lambda: call_compiled(early_compiled, early_ws)),
    ]
    timing = measure_compiled(
        implementations,
        device=device,
        batch=args.batch,
        warmup=args.warmup,
        repeats=args.repeats,
        rounds=args.rounds,
    )
    original = timing["original"]
    early = timing["early_output"]
    comparison = {
        "speedup_x": original["median_ms"] / max(early["median_ms"], 1e-12),
        "latency_saved_ms": original["median_ms"] - early["median_ms"],
        "incremental_peak_saved_mib": (
            original["incremental_peak_mib"] - early["incremental_peak_mib"]
        ),
        "incremental_peak_saved_percent": 100.0
        * (
            original["incremental_peak_mib"] - early["incremental_peak_mib"]
        )
        / max(original["incremental_peak_mib"], 1e-12),
        "optimized_module_tensor_saved_mib": (
            module_footprint(full_optimized)["module_tensor_mib"]
            - module_footprint(early_optimized)["module_tensor_mib"]
        ),
        "output_tensor_saved_mib": (
            full_compiled_output.numel() * full_compiled_output.element_size()
            - early_compiled_output.numel() * early_compiled_output.element_size()
        )
        / MIB,
    }

    profiler = {}
    if args.profile:
        profiler = {
            "original": probe_profiler_kernels(
                call_compiled, (full_compiled, full_ws), device
            ),
            "early_output": probe_profiler_kernels(
                call_compiled, (early_compiled, early_ws), device
            ),
        }

    result = {
        "skipped": False,
        "config": cfg.__dict__,
        "compile": {
            "original": {
                "compile_ms": full_compile_ms,
                "counters": full_counters,
            },
            "early_output": {
                "compile_ms": early_compile_ms,
                "counters": early_counters,
            },
        },
        "correctness": {
            "original_compiled_vs_eager": full_compile_error,
            "early_compiled_vs_eager": early_compile_error,
            "original_repeatability": full_repeat_error,
            "early_repeatability": early_repeat_error,
            "early_vs_area_downsampled_original": reconstruction_stats(
                early_compiled_output, full_compiled_output
            ),
        },
        "output": {
            "original_shape": list(full_compiled_output.shape),
            "early_shape": list(early_compiled_output.shape),
            "original_mib": full_compiled_output.numel()
            * full_compiled_output.element_size()
            / MIB,
            "early_mib": early_compiled_output.numel()
            * early_compiled_output.element_size()
            / MIB,
        },
        "optimized_module_footprint": {
            "original": module_footprint(full_optimized),
            "early_output": module_footprint(early_optimized),
        },
        "timing_and_memory": timing,
        "comparison": comparison,
        "profiler": profiler,
    }
    print(
        f"  {precision}: original={original['median_ms']:.2f}ms, "
        f"early={early['median_ms']:.2f}ms, speedup={comparison['speedup_x']:.2f}x, "
        f"incremental peak saved={comparison['incremental_peak_saved_mib']:.1f}MiB"
    )

    # Drop precision-specific compiler graphs before preparing the next dtype.
    del full_compiled, early_compiled, full_optimized, early_optimized
    del full_eager, early_eager, full_compiled_output, early_compiled_output
    torch._dynamo.reset()
    gc.collect()
    empty_cache(device)
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires an NVIDIA CUDA GPU")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("--device must select a CUDA device")
    if args.batch < 1 or args.repeats < 1 or args.rounds < 1 or args.warmup < 0:
        raise ValueError("batch/repeats/rounds must be positive and warmup nonnegative")
    if args.smoke:
        args.batch = 1
        args.warmup = 0
        args.repeats = 1
        args.rounds = 1

    configure_cuda_backends(cudnn_benchmark=True, allow_tf32=args.tf32)
    print(f"#. Loading original FFHQ generator: {args.ffhq_weights}")
    full_generator = _load_full_source_generator(args.ffhq_weights).to(device)
    print(f"#. Loading integrated early generator: {args.early_weights}")
    early_generator = load_generator_checkpoint(args.early_weights, device=device)
    normalize_scalar_attrs(full_generator)
    normalize_scalar_attrs(early_generator)
    full_ws, early_ws = make_latents(
        full_generator, early_generator, args.batch, device, args.seed
    )

    payload: dict[str, Any] = {
        "env": env_metadata(),
        "args": vars(args),
        "base_model_footprint": {
            "original": module_footprint(full_generator),
            "early_output": module_footprint(early_generator),
        },
        "theoretical_memory": theoretical_memory(
            args.batch,
            b512_channels=int(full_generator.synthesis.b512.conv1.out_channels),
            b1024_channels=int(full_generator.synthesis.b1024.conv1.out_channels),
            decoder_hidden_channels=decoder_hidden_channels(
                early_generator.synthesis.decoder
            ),
        ),
        "precisions": {},
    }
    full_base = payload["base_model_footprint"]["original"]
    early_base = payload["base_model_footprint"]["early_output"]
    payload["base_model_comparison"] = {
        "module_tensor_saved_mib": full_base["module_tensor_mib"]
        - early_base["module_tensor_mib"],
        "persistent_state_saved_mib": full_base["persistent_state_mib"]
        - early_base["persistent_state_mib"],
        "parameter_count_change": early_base["parameter_count"]
        - full_base["parameter_count"],
    }

    for precision in args.precisions:
        payload["precisions"][precision] = run_precision(
            precision,
            full_generator,
            early_generator,
            full_ws,
            early_ws,
            args,
            device,
        )
    return payload


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compile and compare full StyleGAN2 with its early-output replacement"
    )
    parser.add_argument("--ffhq-weights", required=True)
    parser.add_argument("--early-weights", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--precisions",
        nargs="+",
        choices=("fp16", "bf16"),
        default=("fp16", "bf16"),
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--compile-mode", default="default")
    parser.add_argument("--fullgraph", action="store_true")
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--out",
        default="tests/results/early_output_cuda.json",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    payload = run(args)
    output_path = (
        args.out
        if os.path.isabs(args.out)
        else os.path.join(REPO_ROOT, args.out)
    )
    write_json(output_path, payload)
    print(f"#. Wrote {output_path}")


if __name__ == "__main__":
    main()
