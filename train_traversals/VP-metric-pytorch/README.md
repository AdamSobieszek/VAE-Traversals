# Variation Predictability Metric

This repository contains the independent code for VP-metric in [Learning Disentangled Representations with Latent Variation Predictability].

## Requirements

* Numpy.
* PyTorch >= 1.3.1

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
