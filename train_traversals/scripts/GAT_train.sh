gan_type="GAT"
num_support_sets=64
num_support_timesteps=20
warmup_fraction=0.001
accumulate_grad_steps=2
recognizer_type="ResNet"
z_truncation=0.75
batch_size=4
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

python train_GAT.py $tb \
                --gan-type=${gan_type} \
                --recognizer-type=${recognizer_type} \
                --num-support-sets=${num_support_sets} \
                --num-support-timesteps=${num_support_timesteps} \
                --batch-size=${batch_size} \
                --max-iter=${max_iter} \
                --warmup-fraction=${warmup_fraction} \
                --accumulate-grad-steps=${accumulate_grad_steps} \
                --z-truncation=${z_truncation} \
                --log-freq=50 \
                --ckp-freq=100 \
                --reset_lr \
                --reset_weight_decay \
                --reset_schedulers \
                $new