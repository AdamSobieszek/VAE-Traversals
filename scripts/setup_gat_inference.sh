#!/usr/bin/env bash
set -euo pipefail

# Minimal SD-VAE and Python setup for GAT inference (generate.py).
# Skips ImageNet dataset downloads, FID reference stats, and training deps.
# Run from anywhere. By default it downloads assets into this repository.
#
# Requires an existing PyTorch CUDA build (e.g. conda pytorch-cu). This script
# never installs or upgrades torch/torchvision.
#
# Useful options:
#   ROOT=/path/to/VAE-Traversals bash setup_gat_inference.sh
#   SKIP_DOWNLOAD=1 bash setup_gat_inference.sh
#   DOWNLOAD_CHECKPOINT=0 bash setup_gat_inference.sh
#   VAE_VARIANT=mse bash setup_gat_inference.sh
#   RUN_INFERENCE_SMOKE=1 bash setup_gat_inference.sh

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYTHON:-python3}"

SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
DOWNLOAD_CHECKPOINT="${DOWNLOAD_CHECKPOINT:-1}"
INSTALL_INFERENCE_REQUIREMENTS="${INSTALL_INFERENCE_REQUIREMENTS:-1}"
VAE_VARIANT="${VAE_VARIANT:-ema}"
RUN_INFERENCE_SMOKE="${RUN_INFERENCE_SMOKE:-0}"
CKPT="${CKPT:-}"
SAMPLE_DIR="${SAMPLE_DIR:-$ROOT/samples_smoke}"
NUM_FID_SAMPLES="${NUM_FID_SAMPLES:-8}"
PER_PROC_BATCH_SIZE="${PER_PROC_BATCH_SIZE:-4}"
SDVAE_DIR="${SDVAE_DIR:-$ROOT/pretrained_models/sd-vae-ft-${VAE_VARIANT}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$ROOT/GAT/checkpoints}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$CHECKPOINT_DIR/GAT-XL-2.pt}"
GAT_XL2_CKPT_ID="${GAT_XL2_CKPT_ID:-1eMDfFlhFB_doTIJuLLZ9qpS1NknD0Ss2}"

if [[ "$VAE_VARIANT" != "ema" && "$VAE_VARIANT" != "mse" ]]; then
  echo "VAE_VARIANT must be 'ema' or 'mse' (generate.py --vae), got: $VAE_VARIANT" >&2
  exit 1
fi

if [[ "$INSTALL_INFERENCE_REQUIREMENTS" == "1" ]]; then
  if ! "$PYTHON" -c "import torch" >/dev/null 2>&1; then
    echo "PyTorch must already be installed (e.g. conda pytorch-cu). This script does not install torch." >&2
    exit 1
  fi

  # Install packages that do not pull torch from PyPI.
  "$PYTHON" -m pip install -r "$ROOT/GAT/requirements-inference.txt"

  # timm and diffusers declare torch as a dependency; skip deps to keep the
  # preexisting CUDA build untouched.
  "$PYTHON" -m pip install --no-deps diffusers timm

  # Runtime deps for diffusers/timm that do not install torch.
  "$PYTHON" -m pip install filelock fsspec packaging pyyaml regex requests importlib-metadata
fi

if [[ "$SKIP_DOWNLOAD" != "1" ]]; then
  # Install the Hugging Face CLI used for resumable model downloads.
  "$PYTHON" -m pip install huggingface_hub

  mkdir -p "$SDVAE_DIR"

  # Download the Stable Diffusion VAE used to decode GAT latents at inference.
  # generate.py calls AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-{ema|mse}").
  hf download "stabilityai/sd-vae-ft-${VAE_VARIANT}" --local-dir "$SDVAE_DIR"
  hf download "stabilityai/sd-vae-ft-${VAE_VARIANT}"
fi

if [[ "$DOWNLOAD_CHECKPOINT" == "1" ]]; then
  "$PYTHON" -m pip install gdown
  mkdir -p "$CHECKPOINT_DIR"

  if [[ -f "$CHECKPOINT_PATH" ]]; then
    echo "Checkpoint already present: $CHECKPOINT_PATH"
  else
    gdown "https://drive.google.com/uc?id=${GAT_XL2_CKPT_ID}" -O "$CHECKPOINT_PATH"
  fi
fi

if [[ -z "$CKPT" && -f "$CHECKPOINT_PATH" ]]; then
  CKPT="$CHECKPOINT_PATH"
fi

if [[ "$RUN_INFERENCE_SMOKE" == "1" ]]; then
  if [[ -z "$CKPT" ]]; then
    echo "RUN_INFERENCE_SMOKE=1 requires a checkpoint. Set CKPT=/path/to/checkpoint.pt or run with DOWNLOAD_CHECKPOINT=1." >&2
    exit 1
  fi
  if [[ ! -f "$CKPT" ]]; then
    echo "Checkpoint not found: $CKPT" >&2
    exit 1
  fi

  # Confirm the CUDA-enabled PyTorch wheel can see the GPU.
  "$PYTHON" - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
PY

  # Generate a handful of samples to verify checkpoint + VAE decoding.
  (
    cd "$ROOT/GAT"
    torchrun --standalone --nproc_per_node=1 generate.py \
      --ckpt="$CKPT" \
      --sample-dir="$SAMPLE_DIR" \
      --vae="$VAE_VARIANT" \
      --num-fid-samples="$NUM_FID_SAMPLES" \
      --per-proc-batch-size="$PER_PROC_BATCH_SIZE"
  )
fi
