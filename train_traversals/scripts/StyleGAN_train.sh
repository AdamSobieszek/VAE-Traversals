gan_type="StyleGAN2"
num_traversal_sets=200
num_traversal_timesteps=20
warmup_fraction=0.001
accumulate_grad_steps=8
recognizer_type="ResNet"
z_truncation=0.6
batch_size=1
max_iter=14_000
tensorboard=true
new_experiment=true
mixed_precision="bf16"
compile=true
# Packed generator batch is B*K. Auto-chunk only if that backward would not fit.

# ================================


tb=""
if $tensorboard ; then
  tb="--tensorboard"
fi
new=""
if $new_experiment ; then
  new="--new-experiment"
fi
compile_flag=""
if $compile ; then
  compile_flag="--compile"
fi

python train_StyleGAN.py $tb \
                --gan-type=${gan_type} \
                --recognizer-type=${recognizer_type} \
                --num-traversal-sets=${num_traversal_sets} \
                --num-traversal-timesteps=${num_traversal_timesteps} \
                --batch-size=${batch_size} \
                --max-iter=${max_iter} \
                --recognizer-lr 4e-5 \
                --traversal-set-lr 2e-4 \
                --shift-in-w-space \
                --warmup-fraction=${warmup_fraction} \
                --accumulate-grad-steps=${accumulate_grad_steps} \
                --z-truncation=${z_truncation} \
                --stylegan2-resolution=1024 \
                --early-output \
                --mixed-precision=${mixed_precision} \
                $compile_flag \
                --log-freq=20 \
                --ckp-freq=100 \
                --reset_lr \
                --reset_weight_decay \
                --reset_schedulers \
                --no-optimized-synthesis \
                $new