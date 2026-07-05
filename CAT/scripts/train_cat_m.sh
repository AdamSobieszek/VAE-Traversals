export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

model="CAT-G-M/2"
modelD="CAT-D-B/2"

wandb_name="CAT"
resolution=256
batch_size=512

data_path="../dataset"
expdir="../exps"

expname="cat_m2_256"
resume_step=0

model_cleaned=${model//\//-}
modelD_cleaned=${modelD//\//-}
expname="G-${model_cleaned}_D-${modelD_cleaned}_${expname}"

accelerate launch --main_process_port 29501 train.py \
  --report-to="wandb" \
  --allow-tf32 \
  --mixed-precision="bf16" \
  --seed=0 \
  --sampling-steps=1250 \
  --resolution=${resolution} \
  --model=${model} \
  --modelD=${modelD} \
  --enc-type="dinov2-vit-b" \
  --lambda-repa=1.0 \
  --lambda-cons=0.1 \
  --output-dir=${expdir} \
  --exp-name="${expname}" \
  --batch-size=${batch_size} \
  --data-dir="${data_path}" \
  --resume-step=${resume_step} \
  --wandb-name="${wandb_name}" \
  --learning-rate=2e-4 \
  --R1_gamma=1.0 \
  --R2_gamma=1.0 \
  --R1_every=1 \
  --R2_every=1 \
  --gp-eps=0.01 \
  --gp-batch-frac=0.25 \
  --ema-decay=0.999
