#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"${PYTHON:-python}" "$ROOT/run_vp.py" \
  --run-name StyleGAN2 \
  --data-dir /workspace/experiments/wip/StyleGAN2-256-W-ResNet-K200-D20__20260628_145750/vp_pairs \
  --result-dir /workspace/experiments/wip/StyleGAN2-256-W-ResNet-K200-D20__20260628_145750/vp_results \
  --in-channels 6 --input-mode concat --out-dim 200 \
  --lr 0.005 --batch-size 32 --epochs 300 --test-ratio 0.9 --workers 8 "$@"
