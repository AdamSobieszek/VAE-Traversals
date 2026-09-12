gan_type="SNGAN_AnimeFaces"
# Opt in with TRAVERSAL_ARCHITECTURE=midpoint BIDIRECTIONAL=true bash scripts/SNGAN_train.sh
traversal_architecture=${TRAVERSAL_ARCHITECTURE:-euler}
bidirectional=${BIDIRECTIONAL:-false}
bidirectional_args=()
if [ "$bidirectional" = true ]; then
  bidirectional_args+=(--bidirectional)
fi
num_traversal_sets=64
num_traversal_timesteps=20
warmup_fraction=0.001
accumulate_grad_steps=2
recognizer_type="CAT"
batch_size=3
val_batch_size=6
val_freq=250
val_num_positions=18
val_seed=12345
val_dt=""  # Empty uses the timestep-based default.
val_dt_args=()
if [ -n "$val_dt" ]; then
  val_dt_args+=(--val-dt="$val_dt")
fi
max_iter=10_000
tensorboard=true
new_experiment=true

tb=""
if $tensorboard ; then
  tb="--tensorboard"
fi
new=""
if $new_experiment ; then
  new="--new-experiment"
fi

python train.py $tb \
                --traversal-architecture="$traversal_architecture" \
                "${bidirectional_args[@]}" \
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
                --recognizer-lr 2e-4 \
                --traversal-set-lr 2e-4 \
                --warmup-fraction=${warmup_fraction} \
                --accumulate-grad-steps=${accumulate_grad_steps} \
                --log-freq=50 \
                --ckp-freq=200 \
                --reset_lr \
                --reset_weight_decay \
                --reset_schedulers \
                $new "$@"
