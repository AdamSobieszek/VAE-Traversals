#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"${PYTHON:-python}" "$ROOT/run_vp.py" \
  --run-name BigGAN \
  --data-dir /workspace/experiments/wip/BigGAN-239-LeNet-K120-D20__20260627_164520/vp_pairs \
  --result-dir /workspace/experiments/wip/BigGAN-239-LeNet-K120-D20__20260627_164520/vp_results \
  --in-channels 6 --input-mode concat --out-dim 120 \
  --lr 0.005 --batch-size 32 --epochs 300 --test-ratio 0.9 --workers 8 "$@"
