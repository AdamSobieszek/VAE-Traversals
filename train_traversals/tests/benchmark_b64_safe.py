"""Decision-grade B64 StyleGAN2 optimization benchmark.

Key invariants:
* ``model.py`` and the control generator are never mutated.
* Legacy FP32, native FP16 control, and candidates use identical weights,
  noise buffers, and ``ws``.
* Forward compilation happens under inference mode; backward compilation is
  separate and includes the first AOTAutograd backward.
* Candidate correctness against the untouched native FP16 control is checked
  before timing.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
import warnings
from collections.abc import Callable
from typing import Any

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from lib.config import GAN_WEIGHTS
from models.gan_load import build_stylegan2mps
from models.StyleGAN2_mps.torch_utils.ops.inference_opt import (
    InferenceOptConfig,
    normalize_scalar_attrs,
)
from models.StyleGAN2_mps.torch_utils.ops.native_backend import use_native_ops
from models.StyleGAN2_mps.torch_utils.ops.optimized_synthesis import (
    build_optimized_synthesis,
)
from tests.native_bench_common import (
    configure_cuda_backends,
    empty_cache,
    env_metadata,
    error_stats,
    probe_profiler_kernels,
    select_device,
    synchronize,
    write_json,
)


def load_generator(resolution: int, device: torch.device):
    weights = GAN_WEIGHTS["StyleGAN2"]["weights"][resolution]
    G = build_stylegan2mps(
        weights, resolution=resolution, shift_in_w_space=True
    ).G.to(device).eval()
    for parameter in G.parameters():
        parameter.requires_grad_(False)
    normalized = normalize_scalar_attrs(G)
    return G, normalized


def make_ws(G, batch: int, device: torch.device, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    z = torch.randn(batch, G.z_dim, generator=generator).to(device)
    with torch.inference_mode():
        ws = G.mapping(z, None)
    return ws.detach().clone()


def _compiler_counters() -> dict[str, Any]:
    counters = getattr(torch._dynamo.utils, "counters", {})
    result: dict[str, Any] = {}
    for group in ("frames", "stats", "graph_break", "inductor", "aot_autograd"):
        values = counters.get(group, {})
        if values:
            result[group] = {str(k): int(v) for k, v in values.items()}
    return result


def compile_forward(
    fn: Callable[[torch.Tensor], torch.Tensor],
    ws: torch.Tensor,
    *,
    mode: str,
    fullgraph: bool,
    device: torch.device,
) -> tuple[Callable[[torch.Tensor], torch.Tensor], float, dict[str, Any]]:
    torch._dynamo.utils.counters.clear()
    torch._dynamo.config.cache_size_limit = 128
    torch._dynamo.config.recompile_limit = 128
    # PyTorch's default shape specialization is materially faster than
    # dynamic=False for this fixed B64 graph on cuDNN/Inductor.
    compiled = torch.compile(fn, mode=mode, fullgraph=fullgraph)
    synchronize(device)
    start = time.perf_counter()
    with torch.inference_mode():
        compiled(ws)
    synchronize(device)
    compile_ms = (time.perf_counter() - start) * 1000.0
    return compiled, compile_ms, _compiler_counters()


def compile_backward(
    fn: Callable[[torch.Tensor], torch.Tensor],
    ws: torch.Tensor,
    *,
    mode: str,
    device: torch.device,
) -> tuple[Callable[[torch.Tensor], torch.Tensor], float, dict[str, Any]]:
    torch._dynamo.utils.counters.clear()
    compiled = torch.compile(fn, mode=mode, fullgraph=False)
    sample = ws.detach().clone().requires_grad_(True)
    synchronize(device)
    start = time.perf_counter()
    image = compiled(sample)
    loss = image.float().square().mean()
    torch.autograd.grad(loss, sample)
    synchronize(device)
    compile_ms = (time.perf_counter() - start) * 1000.0
    return compiled, compile_ms, _compiler_counters()


def measure_many(
    implementations: list[tuple[str, Callable[[], torch.Tensor]]],
    *,
    device: torch.device,
    warmup: int,
    repeats: int,
    rounds: int,
    grad: bool,
) -> dict[str, dict[str, float]]:
    context = torch.enable_grad if grad else torch.inference_mode
    samples: dict[str, list[float]] = {name: [] for name, _ in implementations}
    peaks: dict[str, list[float]] = {name: [] for name, _ in implementations}
    with context():
        for _, fn in implementations:
            for _ in range(warmup):
                fn()
        synchronize(device)
        for round_index in range(rounds):
            ordered = (
                implementations
                if round_index % 2 == 0
                else list(reversed(implementations))
            )
            for name, fn in ordered:
                empty_cache(device)
                start = time.perf_counter()
                for _ in range(repeats):
                    fn()
                synchronize(device)
                samples[name].append(
                    (time.perf_counter() - start) * 1000.0 / repeats
                )
                peaks[name].append(
                    torch.cuda.max_memory_allocated(device) / (1024**2)
                )
    return {
        name: {
            "median_ms": statistics.median(values),
            "mean_ms": statistics.mean(values),
            "stdev_ms": statistics.stdev(values) if len(values) > 1 else 0.0,
            "peak_alloc_mib": max(peaks[name]),
            "ms_per_image": statistics.median(values) / 64.0,
        }
        for name, values in samples.items()
    }


def gradient_of(
    fn: Callable[[torch.Tensor], torch.Tensor],
    ws: torch.Tensor,
) -> torch.Tensor:
    value = ws.detach().clone().requires_grad_(True)
    image = fn(value)
    (gradient,) = torch.autograd.grad(image.float().square().mean(), value)
    return gradient


def call_compiled_native(
    fn: Callable[[torch.Tensor], torch.Tensor],
    ws: torch.Tensor,
) -> torch.Tensor:
    """Keep lazy Dynamo subgraph compilation on the native backend."""
    with use_native_ops(upfirdn="native", bias_act="native"):
        return fn(ws)


def gradient_stats(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    max_abs, mean_abs, relative = error_stats(actual, expected)
    cosine = torch.nn.functional.cosine_similarity(
        actual.float().flatten(), expected.float().flatten(), dim=0
    )
    l2_relative = (
        (actual.float() - expected.float()).norm()
        / expected.float().norm().clamp_min(1e-12)
    )
    return {
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "max_over_mean_abs": relative,
        "relative_l2": float(l2_relative.item()),
        "cosine": float(cosine.item()),
    }


def explain(fn: Callable[[torch.Tensor], torch.Tensor], ws: torch.Tensor) -> dict[str, Any]:
    try:
        result = torch._dynamo.explain(fn)(ws)
        return {
            "graph_count": int(result.graph_count),
            "graph_break_count": int(result.graph_break_count),
            "break_reasons": [str(v)[:300] for v in result.break_reasons[:20]],
            "ops_per_graph": [len(v) for v in result.ops_per_graph],
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {str(exc)[:400]}"}


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.batch != 64:
        raise ValueError("This harness intentionally supports batch 64 only")
    if args.fused_epilogue and not (args.shared_modconv or args.fir_compose):
        raise ValueError("Epilogue experiments currently require shared modconv")

    device = select_device(args.device)
    if not args.tf32:
        # Intentional accuracy policy, not a compiler failure.
        warnings.filterwarnings(
            "ignore",
            message=r"TensorFloat32 tensor cores.*not enabled.*",
            category=UserWarning,
            module=r"torch\._inductor\.compile_fx",
        )
    configure_cuda_backends(cudnn_benchmark=True, allow_tf32=args.tf32)
    G, normalized_scalars = load_generator(args.resolution, device)
    ws = make_ws(G, args.batch, device, args.seed)

    cfg = InferenceOptConfig(
        shared_modconv=args.shared_modconv,
        fir_compose=args.fir_compose,
        fir_compose_mode=args.fir_compose_mode,
        fused_epilogue=args.fused_epilogue,
        force_fp16_all_blocks=args.force_fp16_all,
        fused_modconv=False if args.unfused_modconv else None,
        low_precision=args.low_precision,
    )
    has_candidate = any(
        (
            args.shared_modconv,
            args.fir_compose,
            args.fused_epilogue,
            args.force_fp16_all,
            args.unfused_modconv,
        )
    )

    candidate_module = None
    if has_candidate and not args.unfused_modconv:
        candidate_module = build_optimized_synthesis(G.synthesis, cfg).to(device)

    def legacy(w: torch.Tensor) -> torch.Tensor:
        with use_native_ops(upfirdn="cuda", bias_act="cuda"):
            return G.synthesis(w, noise_mode="const", force_fp32=True)

    def reference(w: torch.Tensor) -> torch.Tensor:
        with use_native_ops(upfirdn="ref", bias_act="ref"):
            return G.synthesis(w, noise_mode="const", force_fp32=True)

    def control_raw(w: torch.Tensor) -> torch.Tensor:
        return G.synthesis(w, noise_mode="const", force_fp32=False)

    def control(w: torch.Tensor) -> torch.Tensor:
        with use_native_ops(upfirdn="native", bias_act="native"):
            return control_raw(w)

    def candidate_raw(w: torch.Tensor) -> torch.Tensor:
        if args.unfused_modconv:
            return G.synthesis(
                w,
                noise_mode="const",
                force_fp32=False,
                fused_modconv=False,
            )
        if candidate_module is None:
            return G.synthesis(w, noise_mode="const", force_fp32=False)
        return candidate_module(w, noise_mode="const", force_fp32=False)

    def candidate(w: torch.Tensor) -> torch.Tensor:
        with use_native_ops(upfirdn="native", bias_act="native"):
            return candidate_raw(w)

    payload: dict[str, Any] = {
        "env": env_metadata(),
        "args": vars(args),
        "normalized_checkpoint_scalars": normalized_scalars,
        "control_immutable": True,
        "model_registration_unchanged": True,
        "opt": cfg.__dict__,
        "correctness": {},
        "compile": {},
        "forward": {},
        "backward": {},
    }

    # Correctness is a hard gate before compilation/timing.
    with torch.inference_mode():
        legacy_out = legacy(ws).float()
        control_out = control(ws).float()
        control_repeat = control(ws).float()
        candidate_out = candidate(ws).float()
        candidate_repeat = candidate(ws).float()
        payload["correctness"]["control_vs_legacy_fp32"] = dict(
            zip(
                ("max_abs", "mean_abs", "max_over_mean_abs"),
                error_stats(control_out, legacy_out),
            )
        )
        payload["correctness"]["candidate_vs_control_fp16"] = dict(
            zip(
                ("max_abs", "mean_abs", "max_over_mean_abs"),
                error_stats(candidate_out, control_out),
            )
        )
        payload["correctness"]["candidate_vs_legacy_fp32"] = dict(
            zip(
                ("max_abs", "mean_abs", "max_over_mean_abs"),
                error_stats(candidate_out, legacy_out),
            )
        )
        payload["correctness"]["control_repeatability"] = dict(
            zip(
                ("max_abs", "mean_abs", "max_over_mean_abs"),
                error_stats(control_repeat, control_out),
            )
        )
        payload["correctness"]["candidate_repeatability"] = dict(
            zip(
                ("max_abs", "mean_abs", "max_over_mean_abs"),
                error_stats(candidate_repeat, candidate_out),
            )
        )
        if not args.skip_ref:
            payload["correctness"]["reference_vs_legacy_fp32"] = dict(
                zip(
                    ("max_abs", "mean_abs", "max_over_mean_abs"),
                    error_stats(reference(ws).float(), legacy_out),
                )
            )

    control_compiled = None
    candidate_compiled = None
    if not args.skip_compile:
        with use_native_ops(upfirdn="native", bias_act="native"):
            control_compiled, compile_ms, counters = compile_forward(
                control_raw,
                ws,
                mode=args.compile_mode,
                fullgraph=args.fullgraph,
                device=device,
            )
        payload["compile"]["control_forward"] = {
            "compile_ms": compile_ms,
            "counters": counters,
            "profiler": probe_profiler_kernels(
                call_compiled_native, (control_compiled, ws), device
            ),
        }
        if has_candidate:
            with use_native_ops(upfirdn="native", bias_act="native"):
                candidate_compiled, compile_ms, counters = compile_forward(
                    candidate_raw,
                    ws,
                    mode=args.compile_mode,
                    fullgraph=args.fullgraph,
                    device=device,
                )
            payload["compile"]["candidate_forward"] = {
                "compile_ms": compile_ms,
                "counters": counters,
                "profiler": probe_profiler_kernels(
                    call_compiled_native, (candidate_compiled, ws), device
                ),
            }
            with torch.inference_mode():
                compiled_out = call_compiled_native(
                    candidate_compiled, ws
                ).float()
                compiled_repeat = call_compiled_native(
                    candidate_compiled, ws
                ).float()
            payload["correctness"]["candidate_compile_vs_eager"] = dict(
                zip(
                    ("max_abs", "mean_abs", "max_over_mean_abs"),
                    error_stats(compiled_out, candidate_out),
                )
            )
            payload["correctness"]["candidate_compile_vs_legacy_fp32"] = dict(
                zip(
                    ("max_abs", "mean_abs", "max_over_mean_abs"),
                    error_stats(compiled_out, legacy_out),
                )
            )
            payload["correctness"]["candidate_compile_repeatability"] = dict(
                zip(
                    ("max_abs", "mean_abs", "max_over_mean_abs"),
                    error_stats(compiled_repeat, compiled_out),
                )
            )

    if args.explain:
        with use_native_ops(upfirdn="native", bias_act="native"):
            payload["compile"]["control_explain"] = explain(control_raw, ws)
        if has_candidate:
            with use_native_ops(upfirdn="native", bias_act="native"):
                payload["compile"]["candidate_explain"] = explain(candidate_raw, ws)

    forward_impls: list[tuple[str, Callable[[], torch.Tensor]]] = [
        ("fp32-legacy", lambda: legacy(ws)),
        ("fp16-control-eager", lambda: control(ws)),
    ]
    if not args.skip_ref:
        forward_impls.append(("fp32-reference", lambda: reference(ws)))
    if control_compiled is not None:
        forward_impls.append(
            (
                "fp16-control-compile",
                lambda: call_compiled_native(control_compiled, ws),
            )
        )
    if has_candidate:
        forward_impls.append(("fp16-candidate-eager", lambda: candidate(ws)))
    if candidate_compiled is not None:
        forward_impls.append(
            (
                "fp16-candidate-compile",
                lambda: call_compiled_native(candidate_compiled, ws),
            )
        )
    payload["forward"] = measure_many(
        forward_impls,
        device=device,
        warmup=args.warmup,
        repeats=args.repeats,
        rounds=args.rounds,
        grad=False,
    )

    if not args.skip_backward:
        with torch.enable_grad():
            legacy_grad = gradient_of(legacy, ws)
            control_grad = gradient_of(control, ws)
            candidate_grad = gradient_of(candidate, ws)
        payload["correctness"]["control_grad_vs_legacy_fp32"] = gradient_stats(
            control_grad, legacy_grad
        )
        payload["correctness"]["candidate_grad_vs_control_fp16"] = gradient_stats(
            candidate_grad, control_grad
        )

        control_compiled_grad = None
        candidate_compiled_grad = None
        if not args.skip_compile:
            with use_native_ops(upfirdn="native", bias_act="native"):
                control_compiled_grad, compile_ms, counters = compile_backward(
                    control_raw, ws, mode=args.compile_mode, device=device
                )
            payload["compile"]["control_backward"] = {
                "compile_ms": compile_ms,
                "counters": counters,
            }
            control_compiled_gradient = gradient_of(
                lambda value: call_compiled_native(
                    control_compiled_grad, value
                ),
                ws,
            )
            payload["correctness"][
                "control_compile_grad_vs_eager"
            ] = gradient_stats(control_compiled_gradient, control_grad)
            if has_candidate:
                with use_native_ops(upfirdn="native", bias_act="native"):
                    candidate_compiled_grad, compile_ms, counters = compile_backward(
                        candidate_raw, ws, mode=args.compile_mode, device=device
                    )
                payload["compile"]["candidate_backward"] = {
                    "compile_ms": compile_ms,
                    "counters": counters,
                }
                candidate_compiled_gradient = gradient_of(
                    lambda value: call_compiled_native(
                        candidate_compiled_grad, value
                    ),
                    ws,
                )
                payload["correctness"][
                    "candidate_compile_grad_vs_eager"
                ] = gradient_stats(candidate_compiled_gradient, candidate_grad)

        def backward_run(fn: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
            return gradient_of(fn, ws)

        def compiled_backward_run(
            fn: Callable[[torch.Tensor], torch.Tensor],
        ) -> torch.Tensor:
            return gradient_of(
                lambda value: call_compiled_native(fn, value),
                ws,
            )

        backward_impls: list[tuple[str, Callable[[], torch.Tensor]]] = [
            ("fp32-legacy", lambda: backward_run(legacy)),
            ("fp16-control-eager", lambda: backward_run(control)),
        ]
        if control_compiled_grad is not None:
            backward_impls.append(
                (
                    "fp16-control-compile",
                    lambda: compiled_backward_run(control_compiled_grad),
                )
            )
        if has_candidate:
            backward_impls.append(
                ("fp16-candidate-eager", lambda: backward_run(candidate))
            )
        if candidate_compiled_grad is not None:
            backward_impls.append(
                (
                    "fp16-candidate-compile",
                    lambda: compiled_backward_run(candidate_compiled_grad),
                )
            )
        payload["backward"] = measure_many(
            backward_impls,
            device=device,
            warmup=max(0, args.warmup // 2),
            repeats=max(1, args.repeats // 2),
            rounds=max(1, args.rounds),
            grad=True,
        )

    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--compile-mode", default="default")
    parser.add_argument("--fullgraph", action="store_true")
    parser.add_argument("--skip-compile", action="store_true")
    parser.add_argument("--skip-ref", action="store_true")
    parser.add_argument("--skip-backward", action="store_true")
    parser.add_argument("--explain", action="store_true")
    parser.add_argument("--shared-modconv", action="store_true")
    parser.add_argument("--fir-compose", action="store_true")
    parser.add_argument(
        "--fir-compose-mode",
        default="polyphase",
        choices=("polyphase", "expanded"),
    )
    parser.add_argument("--fused-epilogue", action="store_true")
    parser.add_argument("--force-fp16-all", action="store_true")
    parser.add_argument("--unfused-modconv", action="store_true")
    parser.add_argument(
        "--low-precision",
        default="fp16",
        choices=("fp16", "bf16"),
        help="compute dtype for high-resolution optimized blocks",
    )
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument(
        "--out",
        default="tests/results/b64_optimize/safe_results.json",
    )
    args = parser.parse_args()
    payload = run(args)
    out_path = (
        args.out
        if os.path.isabs(args.out)
        else os.path.join(REPO_ROOT, args.out)
    )
    write_json(out_path, payload)
    print(f"wrote {out_path}")
    for section in ("forward", "backward"):
        if payload.get(section):
            print(section)
            for name, row in payload[section].items():
                print(f"  {name:28s} {row['median_ms']:.2f} ms")


if __name__ == "__main__":
    main()
