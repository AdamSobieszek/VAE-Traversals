#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"${PYTHON:-python}" "$ROOT/run_vp.py" \
  --run-name SNGAN \
  --data-dir /workspace/experiments/complete/SNGAN_AnimeFaces-LeNet-K64-D19__20251219_152949/vp_pairs \
  --result-dir /workspace/experiments/complete/SNGAN_AnimeFaces-LeNet-K64-D19__20251219_152949/vp_results \
  --in-channels 6 --input-mode concat --out-dim 64 \
  --lr 0.005 --batch-size 32 --epochs 300 --test-ratio 0.9 --workers 4 "$@"
