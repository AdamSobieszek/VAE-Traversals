#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"${PYTHON:-python}" "$ROOT/run_vp.py" \
  --run-name VAE \
  --data-dir /path/to/vae_vp_pairs \
  --result-dir /path/to/vae_vp_results \
  --in-channels 3 --input-mode diff --out-dim 30 \
  --lr 0.005 --batch-size 32 --epochs 300 --test-ratio 0.9 --workers 4 "$@"
