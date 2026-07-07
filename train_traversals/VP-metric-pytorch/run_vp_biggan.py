#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run VP metric for a BigGAN model by calling `main_vp.py` with the
correct hyper-parameters, then computing the VP score via
`get_best_score.py`.

You need a folder that contains:
  - paired images:  `pair_000000.jpg`, ..., `pair_00xxxx.jpg`
  - labels:         `labels.npy`     (shape: [N, K], one-hot over K directions)
            """

from __future__ import print_function

import os
import os.path as osp
import argparse
import subprocess
import sys

from get_best_score import get_dis_score


# ------------------------------
# Configuration (edit these)
# ------------------------------

MODEL_TYPE = "BigGAN"

# Directory containing `pair_*.jpg` and `labels.npy` for this model
DATA_DIR = "/workspace/experiments/wip/BigGAN-239-LeNet-K120-D20__20260627_164520/vp_pairs"  # TODO: set this

# Where to store VP training logs/checkpoints
RESULT_DIR = "/workspace/experiments/wip/BigGAN-239-LeNet-K120-D20__20260627_164520/vp_results"  # TODO: set this

IN_CHANNELS = 6          # must match input channels expected by VarPred
INPUT_MODE = "concat"      # "diff" or "concat"
OUT_DIM = 120            # number of support sets for BigGAN
LR = 0.005
BATCH_SIZE = 32
EPOCHS = 300
TEST_RATIO = 0.9         # 10% train / 90% test  -> 10% train, 90% test
WORKERS = 8


def main():
    parser = argparse.ArgumentParser(description="Run VP metric for BigGAN")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=DATA_DIR,
        help="Directory containing pair_*.jpg and labels.npy",
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default=RESULT_DIR,
        help="Directory to save VP logs and checkpoints",
    )
    args_cli = parser.parse_args()

    data_dir = args_cli.data_dir
    result_dir = args_cli.result_dir

    if not osp.isdir(data_dir):
        raise NotADirectoryError("Data dir not found: {}".format(data_dir))
    if not osp.exists(result_dir):
        os.makedirs(result_dir)

    print("Running VP metric for model: {}".format(MODEL_TYPE))
    print("  data_dir   = {}".format(data_dir))
    print("  result_dir = {}".format(result_dir))

    # ------------------------------
    # Call main_vp.py with the right arguments
    # ------------------------------
    main_vp_path = osp.join(osp.dirname(__file__), "main_vp.py")
    cmd = [
        sys.executable,
        main_vp_path,
        "--result_dir", result_dir,
        "--data_dir", data_dir,
        "--in_channels", str(IN_CHANNELS),
        "--out_dim", str(OUT_DIM),
        "--lr", str(LR),
        "--batch_size", str(BATCH_SIZE),
        "--epochs", str(EPOCHS),
        "--input_mode", INPUT_MODE,
        "--test_ratio", str(TEST_RATIO),
        "--workers", str(WORKERS),
    ]
    print("Calling:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    # ------------------------------
    # Compute VP score (best test acc)
    # ------------------------------
    class _Args(object):
        def __init__(self, target_dir):
            self.target_dir = target_dir
            self.vp_dis_type = "best"

    vp_score = get_dis_score(_Args(target_dir=result_dir))
    print("\nVP score (best test accuracy) for {}: {:.3f}".format(MODEL_TYPE, vp_score))§


if __name__ == "__main__":
    main()


