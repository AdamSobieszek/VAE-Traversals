"""Command-line configuration for VP metric training."""

import argparse


def init_parser():
    parser = argparse.ArgumentParser(
        description="Train and evaluate the VP classifier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--result_dir", required=True, help="Results directory.")
    parser.add_argument("--data_dir", required=True, help="Pair dataset directory.")
    parser.add_argument("--run_name", default="VP", help="Descriptive run name.")
    parser.add_argument("--in_channels", type=int, default=6)
    parser.add_argument("--out_dim", type=int, required=True)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument(
        "--input_mode", choices=("concat", "diff"), default="concat"
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.9,
        help="Test fraction in fixed mode and baseline test fraction in curve mode.",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--mode", choices=("fixed", "learning_curve"), default="fixed"
    )
    parser.add_argument("--n_folds", type=int, default=1)
    parser.add_argument(
        "--train_fractions",
        type=float,
        nargs="+",
        default=(0.1, 0.2, 0.4, 0.8),
        help="Training fractions evaluated in learning_curve mode.",
    )
    parser.add_argument(
        "--save_best",
        action="store_true",
        help="Save the best validation model for each fold/fraction run.",
    )
    parser.add_argument(
        "--save_all_checkpoints",
        action="store_true",
        help="Save a checkpoint at every validation for each fold/fraction run.",
    )
    return parser
