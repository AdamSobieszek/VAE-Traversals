gan_type="SD-VAE"
num_support_sets=4
num_support_timesteps=4
warmup_fraction=0.001
accumulate_grad_steps=1
recognizer_type="LeNet"
batch_size=1
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
                --num-support-sets=${num_support_sets} \
                --num-support-timesteps=${num_support_timesteps} \
                --batch-size=${batch_size} \
                --max-iter=${max_iter} \
                --warmup-fraction=${warmup_fraction} \
                --accumulate-grad-steps=${accumulate_grad_steps} \
                --log-freq=50 \
                --ckp-freq=1000 \
                --reset_lr \
                --reset_weight_decay \
                --reset_schedulers \
                $new