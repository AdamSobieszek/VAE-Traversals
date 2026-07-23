#!/usr/bin/env python
"""Launch VP training and report fixed-split or learning-curve metrics."""

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train the VP classifier for a generated pair dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run-name",
        default="VP",
        help="Descriptive name used only in status and score messages.",
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Directory containing pair_*.jpg images and labels.npy.",
    )
    parser.add_argument(
        "--result-dir",
        required=True,
        help="Directory in which stats and optional checkpoints are written.",
    )
    parser.add_argument(
        "--in-channels", type=int, default=6,
        help="Number of channels presented to the VP classifier.",
    )
    parser.add_argument(
        "--input-mode", choices=("concat", "diff"), default="concat",
        help="How the two images in each pair are combined.",
    )
    parser.add_argument(
        "--out-dim", type=int, required=True,
        help="Number of direction classes in labels.npy.",
    )
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate.")
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Training batch size."
    )
    parser.add_argument(
        "--epochs", type=int, default=300, help="Number of training epochs."
    )
    parser.add_argument(
        "--test-ratio", type=float, default=0.9,
        help="Fraction of samples assigned to the test split.",
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Data-loader worker processes."
    )
    parser.add_argument("--seed", type=int, default=0, help="Split and model seed.")
    parser.add_argument(
        "--mode",
        choices=("fixed", "learning-curve", "learning_curve"),
        default="fixed",
        help="Fixed-split VP score or controlled learning-curve estimation.",
    )
    parser.add_argument(
        "--n-fold", "--n-folds", "--n_fold",
        dest="n_folds", type=int, default=1,
        help="Number of maximally nonoverlapping validation folds.",
    )
    parser.add_argument(
        "--train-fractions",
        type=float,
        nargs="+",
        default=(0.1, 0.2, 0.4, 0.8),
        help="Training fractions used by learning-curve mode.",
    )
    parser.add_argument(
        "--save-best",
        action="store_true",
        help="Save the best validation model in every fold/fraction run.",
    )
    parser.add_argument(
        "--save-all-checkpoints",
        action="store_true",
        help="Save a model checkpoint at every validation.",
    )
    parser.add_argument(
        "--skip-score",
        action="store_true",
        help="Do not read and print the best validation score after training.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the main_vp.py command without running it.",
    )
    return parser


def validate_args(parser, args):
    if args.in_channels <= 0:
        parser.error("--in-channels must be positive")
    if args.out_dim <= 0:
        parser.error("--out-dim must be positive")
    if args.lr <= 0:
        parser.error("--lr must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    if not 0 < args.test_ratio < 1:
        parser.error("--test-ratio must be between 0 and 1")
    if args.workers < 0:
        parser.error("--workers cannot be negative")
    if args.n_folds <= 0:
        parser.error("--n-fold must be positive")
    if any(not 0 < fraction < 1 for fraction in args.train_fractions):
        parser.error("--train-fractions values must be between 0 and 1")
    if args.mode != "fixed":
        baseline_fraction = 1.0 - args.test_ratio
        if any(
            fraction < baseline_fraction - 1e-12
            for fraction in args.train_fractions
        ):
            parser.error(
                "--train-fractions cannot be smaller than the baseline train "
                "fraction ({:.6g})".format(baseline_fraction)
            )


def training_command(args):
    main_vp = Path(__file__).resolve().with_name("main_vp.py")
    command = [
        sys.executable,
        os.fspath(main_vp),
        "--result_dir", args.result_dir,
        "--data_dir", args.data_dir,
        "--run_name", args.run_name,
        "--in_channels", str(args.in_channels),
        "--out_dim", str(args.out_dim),
        "--lr", str(args.lr),
        "--batch_size", str(args.batch_size),
        "--epochs", str(args.epochs),
        "--input_mode", args.input_mode,
        "--test_ratio", str(args.test_ratio),
        "--workers", str(args.workers),
        "--seed", str(args.seed),
        "--mode", args.mode.replace("-", "_"),
        "--n_folds", str(args.n_folds),
        "--train_fractions", *[str(value) for value in args.train_fractions],
    ]
    if args.save_best:
        command.append("--save_best")
    if args.save_all_checkpoints:
        command.append("--save_all_checkpoints")
    return command


def main():
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)

    command = training_command(args)
    print("Running VP metric for: {}".format(args.run_name))
    print("Command: {}".format(shlex.join(command)))

    if args.dry_run:
        return

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        parser.error("dataset directory not found: {}".format(data_dir))
    if not (data_dir / "labels.npy").is_file():
        parser.error("labels file not found: {}".format(data_dir / "labels.npy"))

    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
    subprocess.run(command, check=True)

    if not args.skip_score:
        from get_best_score import get_score_summary

        rows = get_score_summary(args.result_dir)
        print("\nVP results for {}:".format(args.run_name))
        for row in rows:
            print(
                "  train fraction {:.3g}: {:.3f} +/- {:.3f} across {} fold(s)".format(
                    row["train_fraction"],
                    row["mean_best_accuracy"],
                    row["std_best_accuracy"],
                    len(row["fold_best_accuracies"]),
                )
            )


if __name__ == "__main__":
    main()
