declare -a EXPERIMENTS=("/workspace/experiments/wip/BigGAN-239-LeNet-K120-D20__20260627_164520")
gan_type="BigGAN"


for exp in "${EXPERIMENTS[@]}"
do
  # Traverse latent space
  python gen_pairs.py --exp="${exp}" \
                --batch-size 8 \
                --img-size 256 \
                --shift-leap 1.0 \
                --img-quality 85 \
                --n-samples 20000 \
                --only-potential=true
done
