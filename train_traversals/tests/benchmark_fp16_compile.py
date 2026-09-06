"""Benchmark native FP16 ± torch.compile vs FP32 baselines.

Legacy / custom-CUDA are timed in float32 only (they are not forced into
FP16 or compile). The candidate stack is native (or prepared) + FP16 +
optional Inductor.

Examples:
    PYTHONPATH=. python tests/benchmark_fp16_compile.py --smoke
    PYTHONPATH=. python tests/benchmark_fp16_compile.py --e2e --resolution 256 --batch 2
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from typing import Any, Callable, Optional

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.StyleGAN2_mps.torch_utils.ops import bias_act as bias_act_mod
from models.StyleGAN2_mps.torch_utils.ops import upfirdn2d as upfirdn_mod
from models.StyleGAN2_mps.torch_utils.ops.bias_act_native import (
    bias_act_native,
    bias_act_native_compile_friendly,
    bias_act_native_homogeneous_inference,
)
from models.StyleGAN2_mps.torch_utils.ops.native_backend import use_native_ops
from models.StyleGAN2_mps.torch_utils.ops.prepared_ops import PreparedBiasAct, PreparedFIR
from models.StyleGAN2_mps.torch_utils.ops.upfirdn2d_native import upfirdn2d_native
from tests.native_bench_common import (
    compile_callable,
    configure_cuda_backends,
    empty_cache,
    env_metadata,
    error_stats,
    measure_implementations,
    peak_memory_mib,
    probe_profiler_kernels,
    select_device,
    summarize_timing,
    synchronize,
    write_json,
)

SQRT2 = math.sqrt(2.0)


def _stylegan_filter(device: torch.device) -> torch.Tensor:
    return upfirdn_mod.setup_filter([1, 3, 3, 1]).to(device)


def upfirdn_cases(device: torch.device) -> list[dict[str, Any]]:
    f = _stylegan_filter(device)
    return [
        {
            "name": "rgb_up2_256",
            "shape": (2, 3, 128, 128),
            "kwargs": dict(up=2, padding=1, gain=4.0),
            "f": f,
        },
        {
            "name": "feature_up2_128",
            "shape": (2, 512, 64, 64),
            "kwargs": dict(up=2, padding=1, gain=4.0),
            "f": f,
        },
        {
            "name": "fir_after_upconv_256",
            "shape": (2, 256, 256, 256),
            "kwargs": dict(up=1, padding=1, gain=1.0),
            "f": f,
        },
        {
            "name": "filter_same_256",
            "shape": (2, 256, 256, 256),
            "kwargs": dict(up=1, down=1, padding=1, gain=1.0),
            "f": f,
        },
        {
            "name": "down2_256",
            "shape": (2, 256, 256, 256),
            "kwargs": dict(down=2, padding=1, gain=1.0),
            "f": f,
        },
    ]


def bias_act_cases(device: torch.device) -> list[dict[str, Any]]:
    return [
        {
            "name": "mapping_512",
            "shape": (8, 512),
            "dim": 1,
            "act": "lrelu",
            "gain": SQRT2,
            "clamp": None,
        },
        {
            "name": "synth_64x64_512",
            "shape": (2, 512, 64, 64),
            "dim": 1,
            "act": "lrelu",
            "gain": SQRT2,
            "clamp": 256.0,
        },
        {
            "name": "synth_128x128_256",
            "shape": (2, 256, 128, 128),
            "dim": 1,
            "act": "lrelu",
            "gain": SQRT2,
            "clamp": 256.0,
        },
        {
            "name": "synth_256x256_128",
            "shape": (2, 128, 256, 256),
            "dim": 1,
            "act": "lrelu",
            "gain": SQRT2,
            "clamp": 256.0,
        },
        {
            "name": "torgb_256",
            "shape": (2, 3, 256, 256),
            "dim": 1,
            "act": "linear",
            "gain": 1.0,
            "clamp": 256.0,
        },
    ]


def speedup(baseline_ms: Optional[float], candidate_ms: Optional[float]) -> Optional[float]:
    if baseline_ms is None or candidate_ms is None or candidate_ms <= 0:
        return None
    return baseline_ms / candidate_ms


def run_upfirdn(
    device: torch.device,
    *,
    warmup: int,
    repeats: int,
    compile_mode: str,
    channels_last: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in upfirdn_cases(device):
        empty_cache(device)
        f = case["f"]
        kwargs = case["kwargs"]
        g = torch.Generator(device="cpu")
        g.manual_seed(0)
        x32 = torch.randn(case["shape"], generator=g, dtype=torch.float32).to(device)
        if channels_last:
            x32 = x32.contiguous(memory_format=torch.channels_last)
        x16 = x32.to(dtype=torch.float16)

        ref_fn = lambda: upfirdn_mod.upfirdn2d(x32, f, impl="ref", **kwargs)  # noqa: E731
        native32_fn = lambda: upfirdn2d_native(x32, f, **kwargs)  # noqa: E731
        legacy32_fn = lambda: upfirdn_mod.upfirdn2d(x32, f, impl="cuda", **kwargs)  # noqa: E731
        native16_fn = lambda: upfirdn2d_native(x16, f, **kwargs)  # noqa: E731

        prep = PreparedFIR(
            f,
            channels=case["shape"][1],
            dtype=torch.float16,
            device=device,
            channels_last=channels_last,
            **kwargs,
        ).eval()
        prep_fn = lambda: prep(x16)  # noqa: E731

        implementations: list[tuple[str, Callable]] = [
            ("fp32-ref", ref_fn),
            ("fp32-native", native32_fn),
            ("fp32-legacy", legacy32_fn),
            ("fp16-native-eager", native16_fn),
            ("fp16-prepared-eager", prep_fn),
        ]

        compile_meta: dict[str, tuple[Optional[float], list[str]]] = {}

        def _native_call(t, _f=f, _kw=dict(kwargs)):
            return upfirdn2d_native(t, _f, **_kw)

        for label, fn, args in (
            ("fp16-native-compile", _native_call, (x16,)),
            ("fp16-prepared-compile", prep, (x16,)),
        ):
            try:
                compiled, compile_ms, notes = compile_callable(
                    fn, args, mode=compile_mode, device=device, fullgraph=True
                )
                implementations.append((label, lambda c=compiled, a=args: c(*a)))
                notes.extend(
                    probe_profiler_kernels(implementations[-1][1], (), device)
                )
                compile_meta[label] = (compile_ms, notes)
            except Exception as exc:  # noqa: BLE001
                compile_meta[label] = (
                    None,
                    [f"compile_error={type(exc).__name__}:{str(exc).splitlines()[0][:180]}"],
                )

        samples = measure_implementations(
            implementations, (), device, warmup=warmup, repeats=repeats
        )
        with torch.inference_mode():
            ref_out = legacy32_fn().float()
            cand_out = prep(x16).float()
            max_abs, mean_abs, rel = error_stats(cand_out, ref_out)

        row: dict[str, Any] = {
            "case": case["name"],
            "shape": list(case["shape"]),
            "kwargs": {k: v for k, v in kwargs.items()},
            "max_abs_vs_legacy_fp32": max_abs,
            "mean_abs_vs_legacy_fp32": mean_abs,
            "rel_vs_legacy_fp32": rel,
            "timings_ms": {},
            "speedups_vs_fp32": {},
            "compile": {},
        }
        medians: dict[str, float] = {}
        for name, vals in samples.items():
            timing = summarize_timing(name, vals)
            medians[name] = timing.median_ms
            peak_a, peak_r = peak_memory_mib(device)
            row["timings_ms"][name] = {
                "median_ms": timing.median_ms,
                "mean_ms": timing.mean_ms,
                "stdev_ms": timing.stdev_ms,
                "peak_alloc_mib": peak_a,
                "peak_reserved_mib": peak_r,
            }
            c_ms, notes = compile_meta.get(name, (None, []))
            if c_ms is not None or notes:
                row["compile"][name] = {"compile_ms": c_ms, "notes": notes}

        for base in ("fp32-ref", "fp32-native", "fp32-legacy"):
            row["speedups_vs_fp32"][base] = {
                cand: speedup(medians.get(base), medians.get(cand))
                for cand in (
                    "fp16-native-eager",
                    "fp16-prepared-eager",
                    "fp16-native-compile",
                    "fp16-prepared-compile",
                )
                if cand in medians
            }

        best_cand = min(
            (
                n
                for n in medians
                if n.startswith("fp16-")
            ),
            key=lambda n: medians[n],
            default=None,
        )
        row["best_fp16"] = best_cand
        row["best_fp16_ms"] = medians.get(best_cand) if best_cand else None
        row["vs_legacy_fp32"] = speedup(medians.get("fp32-legacy"), row["best_fp16_ms"])
        rows.append(row)
        print(
            f"upfirdn {case['name']:22s} legacy32={medians.get('fp32-legacy', float('nan')):.3f}ms "
            f"best_fp16={best_cand} {row['best_fp16_ms']:.3f}ms "
            f"vs_legacy={row['vs_legacy_fp32']:.2f}x"
            if row["best_fp16_ms"] is not None
            else f"upfirdn {case['name']} failed"
        )
    return rows


def run_bias_act(
    device: torch.device,
    *,
    warmup: int,
    repeats: int,
    compile_mode: str,
    channels_last: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in bias_act_cases(device):
        empty_cache(device)
        g = torch.Generator(device="cpu")
        g.manual_seed(1)
        x32 = torch.randn(case["shape"], generator=g, dtype=torch.float32).to(device)
        if channels_last and x32.ndim == 4:
            x32 = x32.contiguous(memory_format=torch.channels_last)
        x16 = x32.to(dtype=torch.float16)
        b32 = torch.randn(case["shape"][case["dim"]], device=device, dtype=torch.float32)
        b16 = b32.to(dtype=torch.float16)
        gain = case["gain"]
        clamp = case["clamp"]
        act = case["act"]
        dim = case["dim"]
        scaled16 = b16 * gain

        ref_fn = lambda: bias_act_mod.bias_act(  # noqa: E731
            x32, b32, dim=dim, act=act, gain=gain, clamp=clamp, impl="ref"
        )
        native32_fn = lambda: bias_act_native(  # noqa: E731
            x32, b32, dim=dim, act=act, gain=gain, clamp=clamp
        )
        legacy32_fn = lambda: bias_act_mod.bias_act(  # noqa: E731
            x32, b32, dim=dim, act=act, gain=gain, clamp=clamp, impl="cuda"
        )
        eager16_fn = lambda: bias_act_native_homogeneous_inference(  # noqa: E731
            x16, scaled16, dim=dim, act=act, gain=gain, clamp=clamp
        )
        prep = PreparedBiasAct(
            b16,
            act=act,
            gain=gain,
            clamp=clamp,
            dtype=torch.float16,
            device=device,
            homogeneous=True,
            channels_last=channels_last and x16.ndim == 4,
        ).eval()
        prep_fn = lambda: prep(x16)  # noqa: E731

        implementations: list[tuple[str, Callable]] = [
            ("fp32-ref", ref_fn),
            ("fp32-native", native32_fn),
            ("fp32-legacy", legacy32_fn),
            ("fp16-hom-eager", eager16_fn),
            ("fp16-prepared-eager", prep_fn),
        ]
        compile_meta: dict[str, tuple[float, list[str]]] = {}

        def _compile_call(t, _sb=scaled16, _dim=dim, _act=act, _gain=gain, _clamp=clamp):
            return bias_act_native_compile_friendly(
                t, dim=_dim, act=_act, gain=_gain, clamp=_clamp, scaled_bias=_sb
            )

        for label, fn, args in (
            ("fp16-compile-friendly", _compile_call, (x16,)),
            ("fp16-prepared-compile", prep, (x16,)),
        ):
            try:
                compiled, compile_ms, notes = compile_callable(
                    fn, args, mode=compile_mode, device=device, fullgraph=True
                )
                implementations.append((label, lambda c=compiled, a=args: c(*a)))
                notes.extend(
                    probe_profiler_kernels(implementations[-1][1], (), device)
                )
                compile_meta[label] = (compile_ms, notes)
            except Exception as exc:  # noqa: BLE001
                compile_meta[label] = (
                    None,
                    [f"compile_error={type(exc).__name__}:{str(exc).splitlines()[0][:180]}"],
                )

        samples = measure_implementations(
            implementations, (), device, warmup=warmup, repeats=repeats
        )
        with torch.inference_mode():
            ref_out = legacy32_fn().float()
            cand_out = prep(x16).float()
            max_abs, mean_abs, rel = error_stats(cand_out, ref_out)

        medians: dict[str, float] = {}
        row: dict[str, Any] = {
            "case": case["name"],
            "shape": list(case["shape"]),
            "max_abs_vs_legacy_fp32": max_abs,
            "timings_ms": {},
            "speedups_vs_fp32": {},
            "compile": {},
        }
        for name, vals in samples.items():
            timing = summarize_timing(name, vals)
            medians[name] = timing.median_ms
            row["timings_ms"][name] = {
                "median_ms": timing.median_ms,
                "mean_ms": timing.mean_ms,
                "stdev_ms": timing.stdev_ms,
            }
            c_ms, notes = compile_meta.get(name, (None, []))
            if c_ms is not None or notes:
                row["compile"][name] = {"compile_ms": c_ms, "notes": notes}

        for base in ("fp32-ref", "fp32-native", "fp32-legacy"):
            row["speedups_vs_fp32"][base] = {
                cand: speedup(medians.get(base), medians.get(cand))
                for cand in (
                    "fp16-hom-eager",
                    "fp16-prepared-eager",
                    "fp16-compile-friendly",
                    "fp16-prepared-compile",
                )
                if cand in medians
            }
        best_cand = min(
            (n for n in medians if n.startswith("fp16-")),
            key=lambda n: medians[n],
            default=None,
        )
        row["best_fp16"] = best_cand
        row["best_fp16_ms"] = medians.get(best_cand) if best_cand else None
        row["vs_legacy_fp32"] = speedup(medians.get("fp32-legacy"), row["best_fp16_ms"])
        rows.append(row)
        print(
            f"bias_act {case['name']:22s} legacy32={medians.get('fp32-legacy', float('nan')):.3f}ms "
            f"best_fp16={best_cand} {row['best_fp16_ms']:.3f}ms "
            f"vs_legacy={row['vs_legacy_fp32']:.2f}x"
        )
    return rows


def run_e2e(
    device: torch.device,
    *,
    resolution: int,
    batch: int,
    warmup: int,
    repeats: int,
    compile_mode: str,
    try_compile: bool,
) -> dict[str, Any]:
    from lib.config import GAN_WEIGHTS
    from models.gan_load import build_stylegan2mps

    weights = GAN_WEIGHTS["StyleGAN2"]["weights"][resolution]
    wrapper = build_stylegan2mps(weights, resolution=resolution, shift_in_w_space=True)
    G = wrapper.G.to(device).eval()
    for p in G.parameters():
        p.requires_grad_(False)

    g = torch.Generator(device="cpu")
    g.manual_seed(0)
    z = torch.randn(batch, G.z_dim, generator=g).to(device)
    with torch.inference_mode():
        ws = G.mapping(z, None)

    def legacy_fp32():
        with use_native_ops(upfirdn="cuda", bias_act="cuda"):
            return G.synthesis(ws, noise_mode="const", force_fp32=True)

    def native_fp32():
        with use_native_ops(upfirdn="native", bias_act="native"):
            return G.synthesis(ws, noise_mode="const", force_fp32=True)

    def native_fp16():
        with use_native_ops(upfirdn="native", bias_act="native"):
            return G.synthesis(ws, noise_mode="const", force_fp32=False)

    implementations: list[tuple[str, Callable]] = [
        ("fp32-legacy", legacy_fp32),
        ("fp32-native", native_fp32),
        ("fp16-native-eager", native_fp16),
    ]
    compile_info: dict[str, Any] = {}
    # reduce-overhead CUDA Graphs often fail on the full generator; default
    # mode still fuses pointwise regions without empty-graph warnings.
    e2e_mode = compile_mode if compile_mode != "reduce-overhead" else "default"
    if try_compile:
        try:
            compiled, compile_ms, notes = compile_callable(
                native_fp16,
                (),
                mode=e2e_mode,
                device=device,
                fullgraph=False,
            )
            implementations.append(("fp16-native-compile", compiled))
            notes.extend(probe_profiler_kernels(compiled, (), device))
            compile_info = {"compile_ms": compile_ms, "notes": notes, "mode": e2e_mode}
        except Exception as exc:  # noqa: BLE001
            compile_info = {
                "error": f"{type(exc).__name__}: {str(exc).splitlines()[0][:300]}"
            }

    samples = measure_implementations(
        implementations, (), device, warmup=warmup, repeats=repeats
    )
    medians = {
        n: summarize_timing(n, v).median_ms for n, v in samples.items()
    }
    with torch.inference_mode():
        ref = legacy_fp32().float()
        cand = native_fp16().float()
        max_abs, mean_abs, rel = error_stats(cand, ref)

    best = min(
        (n for n in medians if n.startswith("fp16-")),
        key=lambda n: medians[n],
        default=None,
    )
    out = {
        "resolution": resolution,
        "batch": batch,
        "timings_ms": {
            n: {
                "median_ms": summarize_timing(n, v).median_ms,
                "mean_ms": summarize_timing(n, v).mean_ms,
                "stdev_ms": summarize_timing(n, v).stdev_ms,
            }
            for n, v in samples.items()
        },
        "max_abs_vs_legacy_fp32": max_abs,
        "mean_abs_vs_legacy_fp32": mean_abs,
        "rel_vs_legacy_fp32": rel,
        "best_fp16": best,
        "vs_legacy_fp32": speedup(medians.get("fp32-legacy"), medians.get(best) if best else None),
        "vs_native_fp32": speedup(medians.get("fp32-native"), medians.get(best) if best else None),
        "per_image_legacy_ms": (
            medians["fp32-legacy"] / batch if "fp32-legacy" in medians else None
        ),
        "per_image_best_fp16_ms": (
            medians[best] / batch if best and best in medians else None
        ),
        "compile": compile_info,
    }
    print(
        f"e2e {resolution}x B{batch}: legacy32={medians.get('fp32-legacy', float('nan')):.2f}ms "
        f"native32={medians.get('fp32-native', float('nan')):.2f}ms "
        f"best_fp16={best} {medians.get(best, float('nan')):.2f}ms "
        f"vs_legacy={out['vs_legacy_fp32']}"
    )
    return out

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=60)
    parser.add_argument("--compile-mode", default="reduce-overhead")
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="ops only")
    parser.add_argument("--e2e", action="store_true")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--skip-compile-e2e", action="store_true")
    parser.add_argument(
        "--out",
        default="tests/results/fp16_compile/results.json",
    )
    args = parser.parse_args()

    device = select_device(args.device)
    configure_cuda_backends(cudnn_benchmark=True, allow_tf32=False)
    os.makedirs(os.path.dirname(os.path.join(REPO_ROOT, args.out)), exist_ok=True)

    payload: dict[str, Any] = {
        "env": env_metadata(),
        "compile_mode": args.compile_mode,
        "channels_last": args.channels_last,
        "note": "FP16/compile candidates are native/prepared only; legacy is FP32 baseline",
    }

    if args.smoke or not args.e2e:
        payload["upfirdn"] = run_upfirdn(
            device,
            warmup=args.warmup,
            repeats=args.repeats,
            compile_mode=args.compile_mode,
            channels_last=args.channels_last,
        )
        payload["bias_act"] = run_bias_act(
            device,
            warmup=args.warmup,
            repeats=args.repeats,
            compile_mode=args.compile_mode,
            channels_last=args.channels_last,
        )
    if args.e2e:
        payload["e2e"] = run_e2e(
            device,
            resolution=args.resolution,
            batch=args.batch,
            warmup=max(5, args.warmup // 2),
            repeats=max(20, args.repeats // 2),
            compile_mode=args.compile_mode,
            try_compile=not args.skip_compile_e2e,
        )

    out_path = args.out if os.path.isabs(args.out) else os.path.join(REPO_ROOT, args.out)
    write_json(out_path, payload)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
