export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${script_dir}/.."

ckpt=${1:-exps/G-CAT-G-B-2_D-CAT-D-B-2_cat_b2_256/checkpoints/latest.pt}

torchrun --standalone --nproc_per_node=1 generate.py \
  --ckpt="${ckpt}" \
  --model="CAT-G-B/2" \
  --num-fid-samples=50000 \
  --per-proc-batch-size=32 \
  --truncation-psi=0.85
