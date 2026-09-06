"""End-to-end StyleGAN2 native-op benchmarks (prefix and full synthesis).

Uses context-managed backend overrides so production defaults stay unchanged.

Examples:
    PYTHONPATH=. python tests/benchmark_stylegan2_native.py --resolution 256 --batch 2
    PYTHONPATH=. python tests/benchmark_stylegan2_native.py --resolution 1024 --batch 1 --compile
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Optional

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from lib.config import GAN_WEIGHTS
from models.gan_load import build_stylegan2mps
from models.StyleGAN2_mps.torch_utils.ops.native_backend import use_native_ops
from tests.native_bench_common import (
    configure_cuda_backends,
    empty_cache,
    env_metadata,
    error_stats,
    measure_implementations,
    peak_memory_mib,
    select_device,
    summarize_timing,
    synchronize,
    write_json,
)


def load_generator(resolution: int, device: torch.device):
    weights = GAN_WEIGHTS["StyleGAN2"]["weights"][resolution]
    wrapper = build_stylegan2mps(weights, resolution=resolution, shift_in_w_space=True)
    G = wrapper.G.to(device).eval()
    for param in G.parameters():
        param.requires_grad_(False)
    return G


@torch.inference_mode()
def synthesize_full(G, ws, *, noise_mode: str = "const", force_fp32: bool = False):
    return G.synthesis(ws, noise_mode=noise_mode, force_fp32=force_fp32)


@torch.inference_mode()
def synthesize_prefix(
    G,
    ws,
    *,
    max_resolution: int = 256,
    skip_torgb: bool = True,
    noise_mode: str = "const",
    force_fp32: bool = False,
):
    """Run synthesis blocks through ``max_resolution``, optionally skipping ToRGB."""
    synth = G.synthesis
    block_ws = []
    w_idx = 0
    ws = ws.to(torch.float32)
    for res in synth.block_resolutions:
        block = getattr(synth, f"b{res}")
        block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
        w_idx += block.num_conv
        if res >= max_resolution:
            break

    x = img = None
    for res, cur_ws in zip(synth.block_resolutions, block_ws):
        block = getattr(synth, f"b{res}")
        x, img = block(
            x,
            img,
            cur_ws,
            noise_mode=noise_mode,
            force_fp32=force_fp32,
            skip_torgb=skip_torgb,
        )
        if res >= max_resolution:
            break
    return x if skip_torgb else img


def make_ws(G, batch: int, device: torch.device, seed: int = 0) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    z = torch.randn(batch, G.z_dim, generator=g)
    with torch.inference_mode():
        ws = G.mapping(z.to(device), None)
    return ws


def benchmark(args: argparse.Namespace) -> None:
    device = select_device(args.device)
    backend = configure_cuda_backends(
        cudnn_benchmark=args.cudnn_benchmark, allow_tf32=args.allow_tf32
    )
    G = load_generator(args.resolution, device)
    ws = make_ws(G, args.batch, device, seed=args.seed)
    force_fp32 = args.dtype == "float32"
    payload = {
        "env": env_metadata(),
        "backend": backend,
        "args": vars(args),
        "results": [],
    }
    print(
        f"PyTorch {torch.__version__} device={device} res={args.resolution} "
        f"batch={args.batch} dtype={args.dtype}"
    )

    modes = [
        ("legacy-default", None),
        ("native", ("native", "native")),
        ("ref", ("ref", "ref")),
    ]

    def run_once(backend_pair: Optional[tuple[str, str]]):
        if backend_pair is None:
            if args.prefix:
                return synthesize_prefix(
                    G,
                    ws,
                    max_resolution=args.prefix_resolution,
                    skip_torgb=args.skip_torgb,
                    force_fp32=force_fp32,
                )
            return synthesize_full(G, ws, force_fp32=force_fp32)
        with use_native_ops(upfirdn=backend_pair[0], bias_act=backend_pair[1]):
            if args.prefix:
                return synthesize_prefix(
                    G,
                    ws,
                    max_resolution=args.prefix_resolution,
                    skip_torgb=args.skip_torgb,
                    force_fp32=force_fp32,
                )
            return synthesize_full(G, ws, force_fp32=force_fp32)

    # Correctness versus legacy default.
    empty_cache(device)
    baseline = run_once(None)
    synchronize(device)
    implementations = []
    outputs = {}
    for name, pair in modes:
        fn = (lambda _pair=pair: (lambda: run_once(_pair)))()
        # Bind properly
        def make_fn(p):
            return lambda: run_once(p)

        implementations.append((name, make_fn(pair)))

    compile_ms = None
    if args.compile:
        def native_step():
            return run_once(("native", "native"))

        try:
            compiled = torch.compile(native_step, fullgraph=False, mode=args.compile_mode)
            synchronize(device)
            t0 = time.perf_counter()
            with torch.inference_mode():
                _ = compiled()
            synchronize(device)
            compile_ms = (time.perf_counter() - t0) * 1000.0
            implementations.append(("native-compiled", compiled))
        except Exception as exc:
            print(f"native-compiled unavailable: {type(exc).__name__}: {str(exc).splitlines()[0][:200]}")
            payload["compile_error"] = str(exc).splitlines()[0][:500]

    valid = []
    for name, fn in implementations:
        try:
            empty_cache(device)
            out = fn()
            synchronize(device)
            max_abs, mean_abs, rel = error_stats(out, baseline)
            print(
                f"{name:18s} shape={tuple(out.shape)} dtype={out.dtype} "
                f"max_abs={max_abs:.3e} mean_abs={mean_abs:.3e} rel={rel:.3e}"
            )
            if max_abs > (3e-2 if args.dtype == "float16" else 1e-3):
                print(f"{name:18s} WARNING large error vs legacy-default")
            valid.append((name, fn))
            outputs[name] = out
        except Exception as exc:
            print(f"{name:18s} skipped: {exc}")

    # Timing: callables take no args.
    samples = {}
    with torch.inference_mode():
        for name, fn in valid:
            for _ in range(args.warmup):
                fn()
        synchronize(device)
        samples = {name: [] for name, _ in valid}
        for round_idx in range(args.rounds):
            ordered = valid if round_idx % 2 == 0 else list(reversed(valid))
            for name, fn in ordered:
                empty_cache(device)
                start = time.perf_counter()
                for _ in range(args.repeats):
                    fn()
                synchronize(device)
                samples[name].append(
                    (time.perf_counter() - start) * 1000.0 / args.repeats
                )

    baseline_ms = None
    results = []
    for name, _ in valid:
        empty_cache(device)
        _ = dict(valid)[name]()
        synchronize(device)
        peak_a, peak_r = peak_memory_mib(device)
        result = summarize_timing(
            name,
            samples[name],
            actual=outputs[name],
            expected=baseline,
            peak_allocated_mib=peak_a,
            peak_reserved_mib=peak_r,
            compile_ms=compile_ms if name == "native-compiled" else None,
        )
        results.append(result)
        if name == "legacy-default":
            baseline_ms = result.median_ms
        speedup = (baseline_ms / result.median_ms) if baseline_ms else float("nan")
        print(
            f"{name:18s} {result.median_ms:.3f}ms ({speedup:.3f}x) "
            f"peak_alloc={result.peak_allocated_mib:.1f}MiB"
        )

    payload["results"] = results
    if args.json_out:
        write_json(args.json_out, payload)
        print(f"wrote {args.json_out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resolution", type=int, default=256, choices=(256, 1024))
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--dtype", default="float32", choices=("float32", "float16"))
    parser.add_argument("--prefix", action="store_true")
    parser.add_argument("--prefix-resolution", type=int, default=256)
    parser.add_argument("--skip-torgb", action="store_true", default=True)
    parser.add_argument("--no-skip-torgb", action="store_false", dest="skip_torgb")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--compile-mode",
        default="default",
        choices=("default", "reduce-overhead", "max-autotune"),
    )
    parser.add_argument("--cudnn-benchmark", action="store_true", default=True)
    parser.add_argument("--no-cudnn-benchmark", action="store_false", dest="cudnn_benchmark")
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--json-out", default="")
    return parser


if __name__ == "__main__":
    benchmark(build_parser().parse_args())
