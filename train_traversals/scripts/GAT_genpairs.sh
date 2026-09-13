#!/usr/bin/env bash
set -euo pipefail
train_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export VP_RUNNER=run_vp_gat.sh
export PAIR_BATCH_SIZE="${PAIR_BATCH_SIZE:-1}" PAIR_IMG_SIZE="${PAIR_IMG_SIZE:-256}"
exec bash "${train_dir}/scripts/genpairs_vp.sh" "$@"
