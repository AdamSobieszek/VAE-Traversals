# Variation Predictability Metric

This repository contains the independent code for VP-metric in [Learning Disentangled Representations with Latent Variation Predictability].

## Requirements

* Numpy.
* PyTorch >= 2.0

## Training

Once you have a dataset of `[(x1, x2) --> delta z]`, use `run_vp.py` to train
the classifier and print its validation score. Set `--out-dim` to the number
of directions represented by the dataset labels.

```bash
CUDA_VISIBLE_DEVICES=0 \
    python run_vp.py \
    --run-name my-experiment \
    --result-dir /path/to/result-dir \
    --data-dir /path/to/image-pair/dir \
    --in-channels 3 \
    --out-dim 30 \
    --lr 0.01 \
    --batch-size 32 \
    --epochs 200 \
    --input-mode diff \
    --test-ratio 0.9
```

Run `python run_vp.py --help` for all options. `--dry-run` prints the underlying
training command, while `--skip-score` suppresses score reporting.

### Evaluation modes

`--mode fixed` (the default) reproduces the original metric: train on the
fraction implied by `--test-ratio`, retain the maximum validation accuracy, and
average that maximum across `--n-fold` folds. Fold validation windows are
evenly shifted around one seeded permutation, making their overlap as small as
their size permits.

`--mode learning-curve` evaluates multiple training pools while holding the
number of samples seen per epoch constant. The per-epoch budget comes from the
baseline `--test-ratio` (the historical value `0.9` gives a `0.1` training
budget). For larger pools, each epoch rotates to another subset of that same
size. For example:

```bash
scripts/run_vp_biggan.sh \
    --mode learning-curve \
    --n_fold=2 \
    --train-fractions 0.1 0.2 0.4 0.8
```

Each fraction receives a fresh model in every fold. The output summary averages
the maximum validation accuracy across folds within each fraction.

### Results and checkpoints

Training writes `stats.json` in the result directory. It contains the full
configuration, train statistics for every epoch, every validation result, the
best validation result for each fold/fraction, and aggregate means and standard
deviations. The file is updated atomically during training; the old `train.log`,
`val.log`, and `best_epoch.txt` files are no longer produced.

Models are not saved by default. Add `--save-best` to retain one best model per
fold/fraction, or `--save-all-checkpoints` to retain a model at every validation
point. The historical scripts intentionally enable neither option.

The runner automatically selects CUDA, then Apple MPS, then CPU. Validation
uses the configured training batch size rather than the original unsafe
`batch_size * 50` behavior.

### Training performance

The default path preserves the model, Adam optimizer, sampling order, image
decoder, float32 arithmetic, and validation schedule. It accelerates the
surrounding pipeline in several ways:

* The first run decodes every JPEG once with the historical Pillow decoder and
  writes the exact uint8 pixels to `RESULT_DIR/images.uint8.npy`. Later epochs,
  folds, fractions, and reruns memory-map that array instead of decoding JPEGs
  again. The cache is automatically invalidated when source file metadata
  changes. Cache creation falls back to direct JPEG loading when there is not
  enough free disk space. Use `--image-cache off` to disable it,
  `--image-cache-path PATH` to place it on faster/larger storage, or
  `--rebuild-image-cache` after an unusual in-place dataset modification.
* Images remain uint8 through loading and host-to-device transfer. The original
  `ToTensor` and normalization operations are applied to the complete batch on
  the target device, reducing transfer volume by 4x without changing input
  values.
* Data-loader workers persist across epochs and prefetch batches. CUDA runs
  additionally overlap transfer and normalization of the next batch with the
  current model step. Use `--no-persistent-workers` or `--no-cuda-prefetch`
  for troubleshooting.
* Training and validation metrics accumulate on the device, avoiding three
  device synchronizations per batch. `stats.json` is persisted at validation
  points and every 10 epochs rather than rewriting the growing history after
  every epoch; change this with `--stats-write-interval`.

`--val-batch-size` can be increased independently when validation has memory
headroom. The default remains the training batch size.

The following optional accelerators can change floating-point rounding and are
therefore **off by default**:

```bash
scripts/run_vp_biggan.sh \
    --amp bfloat16 \
    --tf32 \
    --channels-last \
    --compile \
    --fused-adam \
    --val-batch-size 128
```

Hardware support varies. `--amp float16` uses gradient scaling;
`--amp bfloat16` is generally preferable on recent CUDA hardware.
MPS supports `float16` rather than `bfloat16`, while CPU autocast supports
`bfloat16` rather than `float16`.
`--compile-mode` accepts `default`, `reduce-overhead`, or `max-autotune`.

The `scripts/` directory contains the exact settings used for each recorded
model dataset. For example, run `scripts/run_vp_biggan.sh`. These scripts may be
called from any working directory; set `PYTHON` to select a particular Python
interpreter. Extra arguments are forwarded, so
`scripts/run_vp_biggan.sh --dry-run` can inspect a preset without starting
training.

## Citation
```
@inproceedings{VPdis_eccv20,
author={Xinqi Zhu and Chang Xu and Dacheng Tao},
title={Learning Disentangled Representations with Latent Variation Predictability},
booktitle={ECCV},
year={2020}
}
```
