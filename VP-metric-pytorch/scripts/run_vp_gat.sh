#!/usr/bin/env bash
set -euo pipefail
# Required: --data-dir DATASET --result-dir RESULTS; additional VP options pass through.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"${PYTHON:-python}" "$ROOT/run_vp.py" \
  --run-name GAT \
  --in-channels 6 --input-mode concat --out-dim 120 \
  --lr 0.005 --batch-size 32 --epochs 300 --test-ratio 0.9 --workers 8 "$@"
