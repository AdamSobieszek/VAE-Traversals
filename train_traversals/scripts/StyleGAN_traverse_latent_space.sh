#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
train_dir="$(cd -- "${script_dir}/.." && pwd)"
exp="/workspace/VAE-Traversals/train_traversals/experiments/wip/StyleGAN2-1024-EarlyOutput-W-ResNet-K200-D20__20260816_191033"

cd "${train_dir}"

# The Python script samples 50 latents using the experiment's W-space and truncation settings.
# K=200 matches the packed generator batch used by the training configuration.
python traverse_latent_space.py \
    --exp="${exp}" \
    --shift-leap=1.0 \
    --batch-size=200 \
    --img-size=256 \
    --img-quality=85 \
    --cuda \
    --verbose
