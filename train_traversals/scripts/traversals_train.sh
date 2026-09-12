gan_type="SD-VAE"
num_traversal_sets=4
num_traversal_timesteps=4
warmup_fraction=0.001
accumulate_grad_steps=1
recognizer_type="LeNet"
batch_size=1
val_batch_size=${batch_size}
val_freq=100
val_num_positions=32
val_seed=12345
val_dt=""  # Empty uses the timestep-based default.
val_dt_args=()
if [ -n "$val_dt" ]; then
  val_dt_args+=(--val-dt="$val_dt")
fi
max_iter=3000
tensorboard=true
new_experiment=true

# ================================


tb=""
if $tensorboard ; then
  tb="--tensorboard"
fi
new=""
if $new_experiment ; then
  new="--new-experiment"
fi

python train.py $tb \
                --gan-type=${gan_type} \
                --recognizer-type=${recognizer_type} \
                --num-traversal-sets=${num_traversal_sets} \
                --num-traversal-timesteps=${num_traversal_timesteps} \
                --batch-size=${batch_size} \
                --val-batch-size=${val_batch_size} \
                --val-freq=${val_freq} \
                --val-num-positions=${val_num_positions} \
                --val-seed=${val_seed} \
                "${val_dt_args[@]}" \
                --max-iter=${max_iter} \
                --warmup-fraction=${warmup_fraction} \
                --accumulate-grad-steps=${accumulate_grad_steps} \
                --log-freq=50 \
                --ckp-freq=1000 \
                --reset_lr \
                --reset_weight_decay \
                --reset_schedulers \
                $new