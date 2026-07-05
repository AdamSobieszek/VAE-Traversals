#!/usr/bin/env bash
set -euo pipefail

# Reproduce the SD-VAE and ImageNet-256 asset setup used for GAT/CAT.
# Run from anywhere. By default it downloads assets into this repository.
#
# Useful options:
#   ROOT=/path/to/VAE-Traversals bash setup_gat_assets.sh
#   SKIP_DOWNLOAD=1 EXTRACT_TINY_N=32 bash setup_gat_assets.sh
#   SKIP_DOWNLOAD=1 EXTRACT_TINY_N=16 RUN_GAT_SMOKE=1 bash setup_gat_assets.sh
#   JOIN_ARCHIVES=1 EXTRACT_ARCHIVES=1 bash setup_gat_assets.sh

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PYTHON="${PYTHON:-python3}"
DATASET_DIR="${DATASET_DIR:-$ROOT/dataset}"
CHUNKS_DIR="${CHUNKS_DIR:-$DATASET_DIR/_chunks}"
SDVAE_DIR="${SDVAE_DIR:-$ROOT/pretrained_models/sd-vae-ft-mse}"

SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
INSTALL_GAT_REQUIREMENTS="${INSTALL_GAT_REQUIREMENTS:-1}"
INSTALL_CUDA128_TORCH="${INSTALL_CUDA128_TORCH:-1}"
JOIN_ARCHIVES="${JOIN_ARCHIVES:-0}"
EXTRACT_ARCHIVES="${EXTRACT_ARCHIVES:-0}"
EXTRACT_TINY_N="${EXTRACT_TINY_N:-0}"
TINY_DATASET_DIR="${TINY_DATASET_DIR:-$ROOT/dataset_tiny_gat}"
RUN_GAT_SMOKE="${RUN_GAT_SMOKE:-0}"

if [[ "$INSTALL_GAT_REQUIREMENTS" == "1" ]]; then
  # Install the dependencies used by GAT/CAT training.
  "$PYTHON" -m pip install -r "$ROOT/GAT/requirements.txt"
fi

if [[ "$INSTALL_CUDA128_TORCH" == "1" ]]; then
  # The unpinned PyPI torch package may resolve to a CUDA 13 build.
  # Use CUDA 12.8 wheels for NVIDIA driver 570.x / CUDA 12.9 VMs.
  "$PYTHON" -m pip install --force-reinstall --no-cache-dir torch torchvision \
    --index-url https://download.pytorch.org/whl/cu128
fi

if [[ "$SKIP_DOWNLOAD" != "1" ]]; then
  # Install the Hugging Face CLI used for resumable model/dataset downloads.
  "$PYTHON" -m pip install huggingface_hub

  # Create the locations expected by the local training scripts.
  mkdir -p "$SDVAE_DIR" "$CHUNKS_DIR"

  # Download the Stable Diffusion VAE used by GAT/CAT latents.
  # GAT itself calls AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse"),
  # so we cache it both in a project-local folder and in the normal HF cache.
  hf download stabilityai/sd-vae-ft-mse --local-dir "$SDVAE_DIR"
  hf download stabilityai/sd-vae-ft-mse

  # Download the ADM/OpenAI ImageNet-256 reference statistics used for FID.
  wget -c -O "$DATASET_DIR/VIRTUAL_imagenet256_labeled.npz" \
    https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz

  # Download the public preprocessed ImageNet-1k 256x256 dataset chunks.
  # The dataset was generated with REPA/EDM preprocessing and stabilityai/sd-vae-ft-mse.
  hf download Yuehao/imagenet-1k-256 \
    images.zip.chunk_001 images.zip.chunk_002 images.zip.chunk_003 \
    images.zip.chunk_004 images.zip.chunk_005 images.zip.chunk_006 \
    vae-sd.zip.chunk_001 vae-sd.zip.chunk_002 README.md \
    --repo-type dataset \
    --local-dir "$CHUNKS_DIR" \
    --max-workers 2
fi

if [[ "$JOIN_ARCHIVES" == "1" ]]; then
  # On a larger VM, join the split files into the original ZIP archives.
  # This requires roughly 275G extra beyond the chunk files.
  cat "$CHUNKS_DIR"/images.zip.chunk_* > "$DATASET_DIR/images.zip"
  cat "$CHUNKS_DIR"/vae-sd.zip.chunk_* > "$DATASET_DIR/vae-sd.zip"
fi

if [[ "$EXTRACT_ARCHIVES" == "1" ]]; then
  # Extract the full dataset into the directory layout expected by GAT/CAT:
  #   dataset/images/
  #   dataset/vae-sd/
  # This requires substantially more disk than the compressed chunks.
  mkdir -p "$DATASET_DIR/images" "$DATASET_DIR/vae-sd"
  unzip -q "$DATASET_DIR/images.zip" -d "$DATASET_DIR/images"
  unzip -q "$DATASET_DIR/vae-sd.zip" -d "$DATASET_DIR/vae-sd"
fi

if [[ "$EXTRACT_TINY_N" != "0" ]]; then
  # Extract a tiny real subset without joining the huge ZIP chunks.
  # This is useful for a smoke training run in a disk-constrained VM.
  "$PYTHON" - "$CHUNKS_DIR" "$TINY_DATASET_DIR" "$EXTRACT_TINY_N" <<'PY'
import io
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path

chunks_dir = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
limit = int(sys.argv[3])

class SplitFile(io.RawIOBase):
    def __init__(self, paths):
        self.paths = [Path(p) for p in paths]
        self.sizes = [p.stat().st_size for p in self.paths]
        self.offsets = []
        total = 0
        for size in self.sizes:
            self.offsets.append(total)
            total += size
        self.total = total
        self.pos = 0
        self.handles = [p.open("rb") for p in self.paths]

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=os.SEEK_SET):
        if whence == os.SEEK_SET:
            new_pos = offset
        elif whence == os.SEEK_CUR:
            new_pos = self.pos + offset
        elif whence == os.SEEK_END:
            new_pos = self.total + offset
        else:
            raise ValueError(f"unsupported whence: {whence}")
        self.pos = max(0, min(new_pos, self.total))
        return self.pos

    def read(self, size=-1):
        if self.pos >= self.total:
            return b""
        if size is None or size < 0:
            size = self.total - self.pos
        size = min(size, self.total - self.pos)
        chunks = []
        remaining = size
        while remaining > 0:
            idx = max(i for i, start in enumerate(self.offsets) if start <= self.pos)
            local = self.pos - self.offsets[idx]
            take = min(remaining, self.sizes[idx] - local)
            handle = self.handles[idx]
            handle.seek(local)
            data = handle.read(take)
            if not data:
                break
            chunks.append(data)
            self.pos += len(data)
            remaining -= len(data)
        return b"".join(chunks)

    def close(self):
        for handle in getattr(self, "handles", []):
            handle.close()
        super().close()

def split_zip(prefix):
    paths = sorted(chunks_dir.glob(f"{prefix}.zip.chunk_*"))
    if not paths:
        raise FileNotFoundError(f"No chunks found for {prefix}.zip in {chunks_dir}")
    return zipfile.ZipFile(SplitFile(paths))

if out_dir.exists():
    shutil.rmtree(out_dir)
(out_dir / "images").mkdir(parents=True)
(out_dir / "vae-sd").mkdir(parents=True)

with split_zip("images") as images_zip, split_zip("vae-sd") as vae_zip:
    image_names = sorted(
        name for name in images_zip.namelist()
        if not name.endswith("/") and Path(name).suffix.lower() in {".png", ".jpg", ".jpeg", ".npy"}
    )
    feature_names = sorted(
        name for name in vae_zip.namelist()
        if not name.endswith("/") and Path(name).suffix.lower() == ".npy"
    )
    selected_features = feature_names[:limit]
    selected_images = image_names[:limit]

    with vae_zip.open("dataset.json") as f:
        labels = dict(json.load(f)["labels"])
    selected_labels = [[name, labels[name.replace("\\", "/")]] for name in selected_features]

    for src_name in selected_images:
        dst = out_dir / "images" / src_name
        dst.parent.mkdir(parents=True, exist_ok=True)
        with images_zip.open(src_name) as src, dst.open("wb") as out:
            shutil.copyfileobj(src, out)

    for src_name in selected_features:
        dst = out_dir / "vae-sd" / src_name
        dst.parent.mkdir(parents=True, exist_ok=True)
        with vae_zip.open(src_name) as src, dst.open("wb") as out:
            shutil.copyfileobj(src, out)

    with (out_dir / "vae-sd" / "dataset.json").open("w") as f:
        json.dump({"labels": selected_labels}, f)

print(f"Extracted {len(selected_features)} samples to {out_dir}")
PY
fi

if [[ "$RUN_GAT_SMOKE" == "1" ]]; then
  # Verify the tiny dataset can be loaded by the existing GAT dataset class.
  "$PYTHON" - "$ROOT/GAT" "$TINY_DATASET_DIR" <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
from dataset import CustomDataset

d = CustomDataset(sys.argv[2])
image, features, label = d[0]
print("len", len(d))
print("image", tuple(image.shape), image.dtype)
print("features", tuple(features.shape), features.dtype)
print("label", label.item(), label.dtype)
PY

  # Confirm the CUDA-enabled PyTorch wheel can see the GPU.
  "$PYTHON" - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
PY

  # Run a one-step GAT smoke training job on the tiny real-data subset.
  rm -rf "$ROOT/exps_smoke/gat_s8_tiny_smoke"
  (
    cd "$ROOT/GAT"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    WANDB_MODE="${WANDB_MODE:-offline}" \
    accelerate launch --num_processes 1 --main_process_port "${GAT_SMOKE_PORT:-29511}" train.py \
      --report-to="wandb" \
      --allow-tf32 \
      --mixed-precision="no" \
      --seed=0 \
      --sampling-steps=999999 \
      --eval-steps=999999 \
      --resolution=256 \
      --model="GAT-S/8" \
      --modelD="GAT-S/8" \
      --enc-type="None" \
      --proj-coeff=0.0 \
      --output-dir="$ROOT/exps_smoke" \
      --exp-name="gat_s8_tiny_smoke" \
      --batch-size=2 \
      --data-dir="$TINY_DATASET_DIR" \
      --resume-step=0 \
      --wandb-name="GAT smoke" \
      --learning-rate=2e-4 \
      --R1_gamma=1e-1 \
      --R2_gamma=1e-1 \
      --R1_every=1 \
      --R2_every=1 \
      --num-workers=0 \
      --max-train-steps=1 \
      --epochs=1 \
      --checkpointing-steps=1000 \
      --latest-checkpointing-steps=1000
  )
fi
