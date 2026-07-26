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
    parser.add_argument(
        "--val_batch_size",
        type=int,
        default=None,
        help="Validation batch size; defaults to --batch_size.",
    )
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
    parser.add_argument(
        "--prefetch_factor",
        type=int,
        default=2,
        help="Batches prefetched by each persistent data-loader worker.",
    )
    parser.add_argument(
        "--no_persistent_workers",
        action="store_true",
        help="Restart data-loader workers every epoch (normally slower).",
    )
    parser.add_argument(
        "--no_cuda_prefetch",
        action="store_true",
        help="Disable asynchronous CUDA batch transfer.",
    )
    parser.add_argument(
        "--device_preprocessing",
        action="store_true",
        help=(
            "Opt in to uint8 transfer and accelerator-side normalization. "
            "The default preserves historical worker-side CPU preprocessing."
        ),
    )
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
    parser.add_argument(
        "--downsample",
        type=int,
        default=1,
        help=(
            "Spatial downsample factor applied with non-overlapping max-pooling "
            "before caching/training (e.g. 2 means 2x2 max-pool). 1 disables it."
        ),
    )
    parser.add_argument(
        "--image_cache",
        choices=("auto", "off"),
        default="auto",
        help=(
            "Cache Pillow-decoded uint8 images in a memory-mapped .npy file. "
            "'auto' falls back to JPEGs when there is insufficient disk space. "
            "Downsampling is applied before the cache is written."
        ),
    )
    parser.add_argument(
        "--image_cache_path",
        default=None,
        help=(
            "Decoded cache path (default: RESULT_DIR/images.uint8.npy, or "
            "RESULT_DIR/images.uint8.dsN.npy when --downsample N>1)."
        ),
    )
    parser.add_argument(
        "--rebuild_image_cache",
        action="store_true",
        help="Rebuild the decoded cache after source images have changed.",
    )
    parser.add_argument(
        "--stats_write_interval",
        type=int,
        default=10,
        help="Persist in-progress epoch records at least this often.",
    )

    # These options can alter floating-point behavior, so all are off by
    # default even when they are faster on the available hardware.
    parser.add_argument(
        "--amp",
        choices=("off", "float16", "bfloat16"),
        default="off",
        help="Opt-in automatic mixed precision.",
    )
    parser.add_argument(
        "--tf32",
        action="store_true",
        help="Opt in to CUDA TensorFloat-32 convolution/matmul kernels.",
    )
    parser.add_argument(
        "--channels_last",
        action="store_true",
        help="Opt in to NHWC/channels-last convolution storage.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Opt in to torch.compile for the classifier.",
    )
    parser.add_argument(
        "--compile_mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        default="default",
    )
    parser.add_argument(
        "--fused_adam",
        action="store_true",
        help="Opt in to the CUDA fused Adam implementation.",
    )
    return parser
