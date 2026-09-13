#!/usr/bin/env bash
set -euo pipefail

# One-command train_traversals setup for a CUDA PyTorch Docker image.
# Installs Python deps, downloads pretrained generators into
# train_traversals/models/pretrained/, and checks that SNGAN training can start.
#
# Requires an existing PyTorch CUDA build. This script never installs or
# upgrades torch/torchvision.
#
# Run from anywhere.
#
# Useful options:
#   ROOT=/path/to/VAE-Traversals bash setup_train_traversals.sh
#   SKIP_DOWNLOAD=1 bash setup_train_traversals.sh
#   RUN_TRAIN_SMOKE=1 bash setup_train_traversals.sh

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYTHON:-python3}"
TRAVERSALS="$ROOT/train_traversals"

SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
INSTALL_REQUIREMENTS="${INSTALL_REQUIREMENTS:-1}"
RUN_TRAIN_SMOKE="${RUN_TRAIN_SMOKE:-0}"
SMOKE_MIXED_PRECISION="${SMOKE_MIXED_PRECISION:-no}"

torch_identity() {
  "$PYTHON" - <<'PY'
import importlib.util
import torch
print(torch.__version__)
print(torch.__file__)
spec = importlib.util.find_spec("torchvision")
if spec is not None:
    import torchvision
    print(torchvision.__version__)
    print(torchvision.__file__)
PY
}

if ! "$PYTHON" -c "import torch" >/dev/null 2>&1; then
  echo "PyTorch must already be installed (e.g. the CUDA Docker image). This script does not install torch." >&2
  exit 1
fi

TORCH_BEFORE="$(torch_identity)"

if [[ "$INSTALL_REQUIREMENTS" == "1" ]]; then
  req_no_torch="$(mktemp)"
  # lpips and timm declare torch as a dependency; install them separately.
  grep -v -E '^(lpips|timm)([[:space:]=].*)?$' "$TRAVERSALS/requirements.txt" > "$req_no_torch"
  "$PYTHON" -m pip install -r "$req_no_torch"
  rm -f "$req_no_torch"

  # Keep the preexisting CUDA build untouched.
  "$PYTHON" -m pip install --no-deps lpips timm

  # Runtime deps for timm that do not install torch.
  "$PYTHON" -m pip install huggingface_hub safetensors
fi

TORCH_AFTER="$(torch_identity)"
if [[ "$TORCH_BEFORE" != "$TORCH_AFTER" ]]; then
  echo "PyTorch or torchvision changed during setup; aborting so the image build stays intact." >&2
  echo "before:" >&2
  echo "$TORCH_BEFORE" >&2
  echo "after:" >&2
  echo "$TORCH_AFTER" >&2
  exit 1
fi

# Confirm the CUDA-enabled PyTorch wheel can see the GPU.
"$PYTHON" - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
    print("cuda", torch.version.cuda)
PY

if [[ "$SKIP_DOWNLOAD" != "1" ]]; then
  # Relative paths in download_models.py resolve under train_traversals/.
  (
    cd "$TRAVERSALS"
    "$PYTHON" download_models.py
  )
fi

required_weights=(
  "models/pretrained/generators/SNGAN_MNIST/generator.pt"
  "models/pretrained/generators/SNGAN_AnimeFaces/generator.pt"
  "models/pretrained/generators/BigGAN/G_ema.pth"
  "models/pretrained/generators/ProgGAN/100_celeb_hq_network-snapshot-010403.pth"
  "models/pretrained/generators/StyleGAN2/stylegan2-ffhq-config-f.pt"
)

missing=0
for rel in "${required_weights[@]}"; do
  if [[ ! -f "$TRAVERSALS/$rel" ]]; then
    echo "Missing pretrained weight: $TRAVERSALS/$rel" >&2
    missing=1
  fi
done
if [[ "$missing" == "1" ]]; then
  echo "Expected files from download_models.py are missing. Re-run without SKIP_DOWNLOAD=1." >&2
  exit 1
fi

# Load the SNGAN used by scripts/SNGAN_train.sh to verify weights and imports.
(
  cd "$TRAVERSALS"
  "$PYTHON" - <<'PY'
import torch
from lib.config import GAN_RESOLUTIONS, GAN_WEIGHTS
from models.gan_load import build_sngan

gan_type = "SNGAN_AnimeFaces"
weights = GAN_WEIGHTS[gan_type]["weights"][GAN_RESOLUTIONS[gan_type]]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = build_sngan(pretrained_gan_weights=weights, gan_type=gan_type).to(device).eval()
with torch.no_grad():
    images = generator(torch.randn(2, generator.dim_z, device=device))
print("sngan", gan_type, tuple(images.shape), images.dtype, device)
PY
)

if [[ "$RUN_TRAIN_SMOKE" == "1" ]]; then
  (
    cd "$TRAVERSALS"
    "$PYTHON" train.py \
      --gan-type=SNGAN_AnimeFaces \
      --recognizer-type=LeNet \
      --num-traversal-sets=4 \
      --num-traversal-timesteps=8 \
      --batch-size=2 \
      --max-iter=1 \
      --val-freq=0 \
      --log-freq=1 \
      --mixed-precision="$SMOKE_MIXED_PRECISION" \
      --new-experiment
  )
fi

echo "train_traversals setup is ready. From $TRAVERSALS run: bash scripts/SNGAN_train.sh"
