#!/usr/bin/env bash
# Shared pipeline; invoked by the model-specific *_genpairs.sh launchers.
set -euo pipefail
train_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
vp_root="${VP_ROOT:-${train_dir}/../VP-metric-pytorch}"
vp_root="$(cd "$vp_root" && pwd)"
vp_script="${vp_root}/scripts/${VP_RUNNER:?Set VP_RUNNER in the model launcher}"
if [[ ! -f "$vp_script" ]]; then
  echo "VP runner not found: $vp_script (set VP_ROOT)" >&2
  exit 1
fi
if (( $# == 0 )); then
  echo "Usage: bash scripts/MODEL_genpairs.sh EXPERIMENT_DIR [EXPERIMENT_DIR ...]" >&2
  exit 2
fi
pair_python="${PAIR_PYTHON:-${PYTHON:-python}}"
if [[ -z "${PAIR_PYTHON:-${PYTHON:-}}" && -x /opt/anaconda3/envs/manip311/bin/python ]]; then
  pair_python=/opt/anaconda3/envs/manip311/bin/python
fi
vp_python="${VP_PYTHON:-$pair_python}"
options=(--batch-size "${PAIR_BATCH_SIZE:-1}" --img-size "${PAIR_IMG_SIZE:-256}"
         --shift-leap "${SHIFT_LEAP:-1.0}" --img-quality "${IMG_QUALITY:-85}"
         --n-samples "${N_SAMPLES:-20000}" --seed "${SEED:-123}")
if [[ "${BIDIRECTIONAL:-1}" == 1 ]]; then options+=(--bidirectional); fi
for experiment in "$@"; do
  exp="$(cd "$experiment" && pwd)"
  (
    cd "$train_dir"
    "$pair_python" lib/val_utils.py --exp "$exp" "${options[@]}"
    out_dim=$("$pair_python" -c 'import numpy as np, sys; print(np.load(sys.argv[1], mmap_mode="r").shape[1])' "$exp/vp_pairs/labels.npy")
    PYTHON="$vp_python" bash "$vp_script" \
      --data-dir "$exp/vp_pairs" --result-dir "$exp/vp_results" \
      --out-dim "$out_dim" --seed "${SEED:-123}" --epochs "${VP_EPOCHS:-300}"
  )
done
