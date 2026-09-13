#!/usr/bin/env bash
set -euo pipefail
train_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if (( $# == 0 )); then
  set -- "${train_dir}/experiments/wip/SNGAN_AnimeFaces-LeNet-K64-D20__20260912_192639"
fi
export VP_RUNNER=run_vp_sngan.sh
export PAIR_BATCH_SIZE="${PAIR_BATCH_SIZE:-8}" PAIR_IMG_SIZE="${PAIR_IMG_SIZE:-32}"
exec bash "${train_dir}/scripts/genpairs_vp.sh" "$@"
