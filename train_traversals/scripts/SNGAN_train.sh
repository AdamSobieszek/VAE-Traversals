gan_type="SNGAN_AnimeFaces"
num_traversal_sets=64
num_traversal_timesteps=20
warmup_fraction=0.01
accumulate_grad_steps=2
recognizer_type="LeNet"
batch_size=4
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
                --gan-type=${gan_type} \
                --recognizer-type=${recognizer_type} \
                --num-traversal-sets=${num_traversal_sets} \
                --num-traversal-timesteps=${num_traversal_timesteps} \
                --batch-size=${batch_size} \
                --max-iter=${max_iter} \
                --recognizer-lr 2e-4 \
                --traversal-set-lr 2e-4 \
                --warmup-fraction=${warmup_fraction} \
                --accumulate-grad-steps=${accumulate_grad_steps} \
                --log-freq=50 \
                --ckp-freq=1000 \
                --reset_lr \
                --reset_weight_decay \
                --reset_schedulers \
                $new