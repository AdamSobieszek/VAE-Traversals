#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PAIR_PYTHON="${PAIR_PYTHON:-/opt/anaconda3/envs/manip311/bin/python}"
declare -a EXPERIMENTS=("/Users/adamsobieszek/PycharmProjects/VAE-Traversals/train_traversals/experiments/wip/SNGAN_AnimeFaces-LeNet-K64-D20__20260912_192639")

for exp in "${EXPERIMENTS[@]}"; do
  # Isolated outputs share the same inputs without overwriting existing VP pairs.
  comparison=$(mktemp -d "${exp}/vp_comparison.XXXXXX")
  for version in old new; do
    mkdir "${comparison}/${version}"
    ln -s "${exp}/args.json" "${comparison}/${version}/args.json"
    ln -s "${exp}/models" "${comparison}/${version}/models"
  done
  options=(--batch-size 8 --img-size 32 --shift-leap 1.0 --img-quality 85
           --n-samples 200 --bidirectional --seed 123)
  "$PAIR_PYTHON" gen_pairs.py --exp "${comparison}/old" "${options[@]}"
  "$PAIR_PYTHON" lib/val_utils.py --exp "${comparison}/new" "${options[@]}"
  # Compare every JPEG and the labels/directions arrays byte for byte; fail on differences.
  diff -rq "${comparison}/old/vp_pairs" "${comparison}/new/vp_pairs"
  echo "Identical JPEGs, labels and directions: ${comparison}"
done
