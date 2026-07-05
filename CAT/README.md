# Cross-scale Aligned Transformer GANs (CAT)

CAT is a class-conditional one-step GAN in Stable Diffusion VAE latent space. The generator emits four accumulated stage outputs at `32×32×4`; the discriminator judges a resized latent pyramid at scales `32, 16, 8, 4` with block-diagonal attention; and the generator is trained with cross-scale consistency regularization plus optional DINOv2 REPA supervision.

## Supported Models

| Role | Name | Config |
| --- | --- | --- |
| Generator (Small) | `CAT-G-S/2` | depth 12, dim 384, heads 6, outputs at layers 3/6/9/12 |
| Generator (Small benchmark) | `CAT-G-S/4` | depth 12, dim 384, heads 6, outputs at layers 3/6/9/12 |
| Generator (Small benchmark) | `CAT-G-S/8` | depth 12, dim 384, heads 6, outputs at layers 3/6/9/12 |
| Generator (Base) | `CAT-G-B/2` | depth 12, dim 768, heads 12, outputs at layers 3/6/9/12 |
| Generator (Medium) | `CAT-G-M/2` | depth 24, dim 768, heads 12, outputs at layers 6/12/18/24 |
| Generator (Huge) | `CAT-G-H/2` | depth 32, dim 1280, heads 16, outputs at layers 8/16/24/32 |
| Discriminator (Small) | `CAT-D-S/2` | depth 12, dim 384, heads 6 |
| Discriminator (Small benchmark) | `CAT-D-S/4` | depth 12, dim 384, heads 6 |
| Discriminator (Base) | `CAT-D-B/2` | depth 12, dim 768, heads 12, default for paper-fidelity runs |

All ImageNet-256 runs use SD-VAE latents (`32×32×4`), batch size 512, learning rate `2e-4` for both generator and discriminator, bf16 mixed precision, and EMA decay `0.999`.

## Installation

```bash
conda create -n cat python=3.10 -y
conda activate cat
pip install -r requirements.txt
accelerate config
```

Run training and tests from the `CAT/` directory so local imports resolve.

## Dataset

```text
DATA_DIR/
  images/
  vae-sd/
    dataset.json
  VIRTUAL_imagenet256_labeled.npz
```

Each batch provides raw RGB images (for DINOv2), SD-VAE latents, and class labels.

For 128-resolution latent-feature experiments, pass `--dit-features-dir` and `--dit-labels-dir`.

## Training

Direct launch:

```bash
cd CAT
accelerate launch train.py \
  --model CAT-G-B/2 \
  --modelD CAT-D-B/2 \
  --resolution 256 \
  --data-dir /path/to/DATA_DIR \
  --output-dir exps \
  --exp-name cat_b2_256 \
  --batch-size 512 \
  --learning-rate 2e-4 \
  --enc-type dinov2-vit-b \
  --lambda-repa 1.0 \
  --lambda-cons 0.1 \
  --mixed-precision bf16 \
  --allow-tf32
```

Convenience scripts:

```bash
bash scripts/train_cat_b.sh
bash scripts/train_cat_m.sh
bash scripts/train_cat_h.sh
```

Checkpoints are saved under `OUTPUT_DIR/EXP_NAME/checkpoints/`.

## Inference

```bash
cd CAT
torchrun --standalone --nproc_per_node=1 generate.py \
  --ckpt /path/to/checkpoints/latest.pt \
  --model CAT-G-B/2 \
  --sample-dir samples \
  --num-fid-samples 50000 \
  --per-proc-batch-size 32 \
  --truncation-psi 0.85
```

Or:

```bash
bash scripts/sample_cat.sh /path/to/checkpoints/latest.pt
```

## Evaluation

During training, live FID-5K is computed against `VIRTUAL_imagenet256_labeled.npz`. Offline sampling writes PNGs and an ADM-style `.npz` archive next to the sample folder.

## Smoke Tests

```bash
cd CAT
python test_cat_shapes.py
```

This validates generator stage shapes, pyramid construction, discriminator logits `B×4`, block-diagonal attention mask groups, consistency loss, and one D/G loss step.

## Key Implementation Files

| File | Purpose |
| --- | --- |
| `models/generator.py` | `CATGenerator`, stage outputs, `CAT_models` |
| `models/discriminator.py` | `CATDiscriminator`, multi-scale pyramid input, `CATD_models` |
| `cat_pyramid.py` | Pyramid resize helpers and consistency loss |
| `losses/CAT_loss.py` | Scale-wise adversarial, REPA, R1/R2, consistency loss |
| `train.py` | CAT-only training entrypoint |
| `generate.py` | Final-latent sampling and npz export |
