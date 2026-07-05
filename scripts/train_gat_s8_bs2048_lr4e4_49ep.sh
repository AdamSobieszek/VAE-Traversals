#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_MODE="${WANDB_MODE:-offline}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${script_dir}/../GAT"

model="${MODEL:-GAT-S/8}"
modelD="${MODELD:-GAT-S/8}"
resolution="${RESOLUTION:-256}"
batch_size="${BATCH_SIZE:-2048}"
learning_rate="${LEARNING_RATE:-4e-4}"
epochs="${EPOCHS:-49}"
steps_per_epoch="${STEPS_PER_EPOCH:-625}"
max_train_steps="${MAX_TRAIN_STEPS:-$((steps_per_epoch * epochs))}"
checkpointing_steps="${CHECKPOINTING_STEPS:-625}"
latest_checkpointing_steps="${LATEST_CHECKPOINTING_STEPS:-625}"
resume_step="${RESUME_STEP:-0}"
num_workers="${NUM_WORKERS:-4}"

data_path="${DATA_PATH:-../dataset}"
expdir="${EXPDIR:-../exps}"
expname="${EXPNAME:-gat_s8_256_bs2048_lr4e4_49ep}"
wandb_name="${WANDB_NAME:-GAT S8 bs2048 lr4e-4 49ep}"

accelerate launch --num_processes 1 --main_process_port "${MAIN_PROCESS_PORT:-29501}" train.py \
  --report-to="${REPORT_TO:-wandb}" \
  --allow-tf32 \
  --mixed-precision="bf16" \
  --seed="${SEED:-0}" \
  --sampling-steps="${SAMPLING_STEPS:-999999}" \
  --eval-steps="${EVAL_STEPS:-999999}" \
  --resolution="${resolution}" \
  --model="${model}" \
  --modelD="${modelD}" \
  --enc-type="${ENC_TYPE:-None}" \
  --proj-coeff="${PROJ_COEFF:-0.0}" \
  --output-dir="${expdir}" \
  --exp-name="${expname}" \
  --batch-size="${batch_size}" \
  --data-dir="${data_path}" \
  --resume-step="${resume_step}" \
  --wandb-name="${wandb_name}" \
  --learning-rate="${learning_rate}" \
  --R1_gamma="${R1_GAMMA:-1e-1}" \
  --R2_gamma="${R2_GAMMA:-1e-1}" \
  --R1_every="${R1_EVERY:-1}" \
  --R2_every="${R2_EVERY:-1}" \
  --num-workers="${num_workers}" \
  --max-train-steps="${max_train_steps}" \
  --epochs="${epochs}" \
  --checkpointing-steps="${checkpointing_steps}" \
  --latest-checkpointing-steps="${latest_checkpointing_steps}"
