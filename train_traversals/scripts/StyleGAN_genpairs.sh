#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
train_dir="$(cd -- "${script_dir}/.." && pwd)"
exp="/workspace/VAE-Traversals/train_traversals/experiments/wip/StyleGAN2-1024-EarlyOutput-W-ResNet-K200-D20__20260816_191033"

cd "${train_dir}"

# Training used base batch size 1 with K=200 packed traversals and a 256px early output.
python gen_pairs.py \
    --exp="${exp}" \
    --eps=0.2 \
    --shift-leap=1.0 \
    --batch-size=1 \
    --img-size=256 \
    --img-quality=85 \
    --n-samples=20000 \
    --only-potential=true
