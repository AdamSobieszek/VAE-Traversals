"""Sweep B64 StyleGAN2 compile optimization candidates and write aggregate JSON."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import Any

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO_ROOT, "tests/results/b64_optimize")


CONFIGS: list[dict[str, Any]] = [
    {"name": "baseline_native_fp16", "flags": []},
    {"name": "shared_modconv", "flags": ["--shared-modconv"]},
    {
        "name": "fir_polyphase",
        "flags": ["--shared-modconv", "--fir-compose", "--fir-compose-mode", "polyphase"],
    },
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--skip-backward", action="store_true")
    parser.add_argument("--skip-ref", action="store_true", default=True)
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    configs = CONFIGS
    if args.only:
        configs = [c for c in CONFIGS if c["name"] in args.only]

    summary: dict[str, Any] = {"runs": {}}
    for cfg in configs:
        name = cfg["name"]
        out = os.path.join(OUT_DIR, f"{name}.json")
        cmd = [
            sys.executable,
            os.path.join(REPO_ROOT, "tests/benchmark_b64_optimize.py"),
            "--out",
            out,
            "--skip-ref",
            "--warmup",
            str(args.warmup),
            "--repeats",
            str(args.repeats),
            "--rounds",
            str(args.rounds),
            *cfg["flags"],
        ]
        if args.skip_backward:
            cmd.append("--skip-backward")
        print("=" * 60)
        print("Running", name, ":", " ".join(cmd))
        print("=" * 60)
        env = os.environ.copy()
        env["PYTHONPATH"] = REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
        proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env)
        summary["runs"][name] = {"exit": proc.returncode, "out": out}
        if proc.returncode != 0:
            print(f"FAILED {name} exit={proc.returncode}")

    summary_path = os.path.join(OUT_DIR, "sweep_summary.json")
    import json

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print("wrote", summary_path)


if __name__ == "__main__":
    main()
