#!/usr/bin/env python
"""Train VP classifiers on fixed folds or a controlled learning curve."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
import statistics

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

from model import VarPred
from pair_dataset import PairDataset, prepare_image_cache
from parser_config import init_parser
from train_val import train, validate
from utils import (
    EpochSubsetSampler,
    fraction_count,
    make_fold_split,
    worker_init_fn,
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, payload):
    """Atomically update the machine-readable run record."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def should_validate(epoch, epochs):
    interval = 4 if 16 < epoch < 50 else 10
    return (epoch + 1) % interval == 0 or epoch == epochs - 1


def fraction_label(value):
    return ("{:g}".format(value)).replace(".", "p")


def _checkpoint_state_dict(model):
    """Keep historical DataParallel keys while hiding torch.compile internals."""
    if isinstance(model, nn.DataParallel):
        if hasattr(model.module, "_orig_mod"):
            return {
                "module." + key: value
                for key, value in model.module._orig_mod.state_dict().items()
            }
        return model.state_dict()
    if hasattr(model, "_orig_mod"):
        return model._orig_mod.state_dict()
    return model.state_dict()


def checkpoint_state(model, epoch, metrics):
    return {
        "epoch": epoch + 1,
        "state_dict": _checkpoint_state_dict(model),
        "validation": metrics,
    }


def summarize(stats):
    rows = []
    fractions = sorted({run["train_fraction"] for run in stats["runs"]})
    for fraction in fractions:
        completed = [
            run for run in stats["runs"]
            if run["train_fraction"] == fraction and run.get("best_validation")
        ]
        scores = [run["best_validation"]["accuracy_top1"] for run in completed]
        if not scores:
            continue
        rows.append({
            "train_fraction": fraction,
            "test_fraction": round(1.0 - fraction, 12),
            "fold_best_accuracies": scores,
            "mean_best_accuracy": statistics.mean(scores),
            "std_best_accuracy": statistics.pstdev(scores),
        })
    stats["summary"] = {
        "metric": "maximum_validation_accuracy_top1",
        "by_train_fraction": rows,
    }


def build_loaders(
    args,
    n_data,
    targets,
    image_cache,
    train_indices,
    validation_indices,
    samples_per_epoch,
    run_seed,
    device,
):
    full_dataset = PairDataset(
        args.data_dir,
        np.arange(n_data),
        image_tmpl="pair_{:06d}.jpg",
        targets=targets,
        image_cache=image_cache,
        normalize_on_cpu=not args.device_preprocessing,
        downsample=args.downsample,
    )
    validation_dataset = PairDataset(
        args.data_dir,
        validation_indices,
        image_tmpl="pair_{:06d}.jpg",
        targets=targets,
        image_cache=image_cache,
        normalize_on_cpu=not args.device_preprocessing,
        downsample=args.downsample,
    )
    sampler = EpochSubsetSampler(train_indices, samples_per_epoch, run_seed)
    pin_memory = device.type == "cuda"
    common = {
        "num_workers": args.workers,
        "pin_memory": pin_memory,
        "worker_init_fn": worker_init_fn,
    }
    if args.workers:
        common.update({
            "persistent_workers": not args.no_persistent_workers,
            "prefetch_factor": args.prefetch_factor,
        })
    train_loader = torch.utils.data.DataLoader(
        full_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        **common,
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=args.val_batch_size or args.batch_size,
        shuffle=False,
        **common,
    )
    return train_loader, validation_loader, sampler


def _amp_settings(args, device):
    if args.amp == "off":
        return None, None
    amp_dtype = (
        torch.float16 if args.amp == "float16" else torch.bfloat16
    )
    scaler = (
        torch.amp.GradScaler(device.type)
        if amp_dtype == torch.float16 else None
    )
    return amp_dtype, scaler


def run_one(
    args,
    stats,
    stats_path,
    n_data,
    targets,
    image_cache,
    train_fraction,
    fold,
    samples_per_epoch,
    device,
):
    run_seed = args.seed + fold
    torch.manual_seed(run_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(run_seed)
    train_indices, validation_indices = make_fold_split(
        n_data, train_fraction, fold, args.n_folds, args.seed
    )
    if samples_per_epoch > len(train_indices):
        raise ValueError(
            "training fraction {} contains fewer samples than the baseline "
            "per-epoch budget".format(train_fraction)
        )

    run_record = {
        "fold": fold + 1,
        "train_fraction": train_fraction,
        "test_fraction": round(1.0 - train_fraction, 12),
        "train_pool_size": len(train_indices),
        "validation_size": len(validation_indices),
        "samples_per_epoch": samples_per_epoch,
        "seed": run_seed,
        "status": "running",
        "epochs": [],
        "validations": [],
        "best_validation": None,
    }
    stats["runs"].append(run_record)
    write_json(stats_path, stats)

    print(
        "\nTrain fraction {:.3f}, fold {}/{}: pool={}, samples/epoch={}, validation={}".format(
            train_fraction, fold + 1, args.n_folds, len(train_indices),
            samples_per_epoch, len(validation_indices)
        )
    )
    model = VarPred(
        in_channels=args.in_channels,
        out_dim=args.out_dim,
        input_mode=args.input_mode,
    ).to(device)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    if args.compile:
        model = torch.compile(model, mode=args.compile_mode)
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    optimizer_kwargs = {"lr": args.lr}
    if args.fused_adam:
        optimizer_kwargs["fused"] = True
    optimizer = torch.optim.Adam(model.parameters(), **optimizer_kwargs)
    criterion = nn.CrossEntropyLoss().to(device)
    train_loader, validation_loader, sampler = build_loaders(
        args,
        n_data,
        targets,
        image_cache,
        train_indices,
        validation_indices,
        samples_per_epoch,
        run_seed,
        device,
    )
    amp_dtype, scaler = _amp_settings(args, device)

    checkpoint_dir = (
        Path(args.result_dir)
        / "train_fraction_{}".format(fraction_label(train_fraction))
        / "fold_{:02d}".format(fold + 1)
    )
    if args.save_best or args.save_all_checkpoints:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        run_record["checkpoint_dir"] = os.fspath(checkpoint_dir)

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        train_metrics = train(
            train_loader,
            model,
            criterion,
            optimizer,
            epoch,
            device,
            amp_dtype=amp_dtype,
            scaler=scaler,
            channels_last=args.channels_last,
            cuda_prefetch=not args.no_cuda_prefetch,
        )
        run_record["epochs"].append({"epoch": epoch + 1, "train": train_metrics})

        validated = should_validate(epoch, args.epochs)
        if validated:
            validation_metrics = validate(
                validation_loader,
                model,
                criterion,
                epoch,
                device,
                amp_dtype=amp_dtype,
                channels_last=args.channels_last,
                cuda_prefetch=not args.no_cuda_prefetch,
            )
            validation_record = {"epoch": epoch + 1, **validation_metrics}
            run_record["validations"].append(validation_record)
            is_best = (
                run_record["best_validation"] is None
                or validation_record["accuracy_top1"]
                > run_record["best_validation"]["accuracy_top1"]
            )
            if is_best:
                run_record["best_validation"] = validation_record.copy()

            if args.save_all_checkpoints or (args.save_best and is_best):
                state = checkpoint_state(model, epoch, validation_record)
                if args.save_all_checkpoints:
                    torch.save(
                        state,
                        checkpoint_dir / "epoch_{:04d}.pth.tar".format(epoch + 1),
                    )
                if args.save_best and is_best:
                    torch.save(state, checkpoint_dir / "model_best.pth.tar")

        if (
            validated
            or (epoch + 1) % args.stats_write_interval == 0
            or epoch == args.epochs - 1
        ):
            write_json(stats_path, stats)

    run_record["status"] = "completed"
    run_record["completed_at"] = utc_now()
    summarize(stats)
    write_json(stats_path, stats)


def validate_configuration(parser, args, n_data):
    if args.n_folds < 1:
        parser.error("--n_folds must be positive")
    if args.n_folds > n_data:
        parser.error("--n_folds cannot exceed the number of dataset samples")
    if args.in_channels < 1 or args.out_dim < 1:
        parser.error("model dimensions must be positive")
    if args.lr <= 0 or args.batch_size < 1 or args.epochs < 1:
        parser.error("learning rate, batch size, and epochs must be positive")
    if args.val_batch_size is not None and args.val_batch_size < 1:
        parser.error("--val_batch_size must be positive")
    if args.workers < 0:
        parser.error("--workers cannot be negative")
    if args.prefetch_factor < 1:
        parser.error("--prefetch_factor must be positive")
    if args.stats_write_interval < 1:
        parser.error("--stats_write_interval must be positive")
    if args.downsample < 1:
        parser.error("--downsample must be >= 1")
    if not 0 < args.test_ratio < 1:
        parser.error("--test_ratio must be between 0 and 1")
    if n_data < 2:
        parser.error("the dataset must contain at least two samples")
    fractions = (
        [round(1.0 - args.test_ratio, 12)]
        if args.mode == "fixed"
        else list(dict.fromkeys(round(value, 12) for value in args.train_fractions))
    )
    if any(not 0 < fraction < 1 for fraction in fractions):
        parser.error("all training fractions must be between 0 and 1")
    baseline_size = n_data - fraction_count(n_data, args.test_ratio)
    if baseline_size < 1:
        parser.error("the baseline training split contains no samples")
    for fraction in fractions:
        pool_size = n_data - fraction_count(n_data, 1.0 - fraction)
        if pool_size < baseline_size:
            parser.error(
                "training fractions cannot be smaller than the baseline "
                "training fraction ({:.6g})".format(1.0 - args.test_ratio)
            )
        if pool_size >= n_data:
            parser.error("each split must leave at least one validation sample")
    return fractions, baseline_size


def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    parser = init_parser()
    args = parser.parse_args()
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    labels = np.load(Path(args.data_dir) / "labels.npy", mmap_mode="r")
    n_data = labels.shape[0]
    targets = np.argmax(np.abs(labels), axis=1).astype(np.int64)
    del labels
    fractions, baseline_size = validate_configuration(parser, args, n_data)
    device = select_device()
    if args.fused_adam and device.type != "cuda":
        parser.error("--fused_adam requires CUDA")
    if args.amp == "float16" and device.type == "cpu":
        parser.error("--amp float16 is not supported for CPU training")
    if args.amp == "bfloat16" and device.type == "mps":
        parser.error("--amp bfloat16 is not supported for MPS training")
    if args.compile and not hasattr(torch, "compile"):
        parser.error("--compile requires PyTorch 2.0 or newer")

    cudnn.benchmark = device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = args.tf32
        cudnn.allow_tf32 = args.tf32
    print("Using device: {}".format(device))

    cache_info = {
        "path": None,
        "enabled": False,
        "mode": args.image_cache,
        "downsample": args.downsample,
    }
    if args.image_cache == "auto":
        if args.image_cache_path:
            cache_path = Path(args.image_cache_path)
        elif args.downsample > 1:
            cache_path = result_dir / "images.uint8.ds{}.npy".format(
                args.downsample
            )
        else:
            cache_path = result_dir / "images.uint8.npy"
        cache_info.update(
            prepare_image_cache(
                args.data_dir,
                cache_path,
                n_data,
                workers=args.workers,
                batch_size=min(64, args.val_batch_size or args.batch_size),
                rebuild=args.rebuild_image_cache,
                downsample=args.downsample,
            )
        )
        cache_info["enabled"] = cache_info["path"] is not None
    image_cache = cache_info["path"]

    stats_path = result_dir / "stats.json"
    stats = {
        "schema_version": 1,
        "run_name": args.run_name,
        "mode": args.mode,
        "started_at": utc_now(),
        "status": "running",
        "dataset": {
            "path": os.fspath(Path(args.data_dir)),
            "samples": n_data,
            "image_cache": cache_info,
        },
        "config": {
            "in_channels": args.in_channels,
            "input_mode": args.input_mode,
            "out_dim": args.out_dim,
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "validation_batch_size": args.val_batch_size or args.batch_size,
            "epochs": args.epochs,
            "workers": args.workers,
            "prefetch_factor": args.prefetch_factor,
            "persistent_workers": (
                args.workers > 0 and not args.no_persistent_workers
            ),
            "cuda_prefetch": (
                device.type == "cuda" and not args.no_cuda_prefetch
            ),
            "device_preprocessing": args.device_preprocessing,
            "seed": args.seed,
            "device": str(device),
            "n_folds": args.n_folds,
            "baseline_test_fraction": args.test_ratio,
            "baseline_samples_per_epoch": baseline_size,
            "train_fractions": fractions,
            "save_best": args.save_best,
            "save_all_checkpoints": args.save_all_checkpoints,
            "stats_write_interval": args.stats_write_interval,
            "amp": args.amp,
            "tf32": args.tf32,
            "channels_last": args.channels_last,
            "compile": args.compile,
            "compile_mode": args.compile_mode if args.compile else None,
            "fused_adam": args.fused_adam,
        },
        "runs": [],
        "summary": None,
    }
    write_json(stats_path, stats)

    for fraction in fractions:
        samples_per_epoch = (
            n_data - fraction_count(n_data, 1.0 - fraction)
            if args.mode == "fixed" else baseline_size
        )
        for fold in range(args.n_folds):
            run_one(
                args,
                stats,
                stats_path,
                n_data,
                targets,
                image_cache,
                fraction,
                fold,
                samples_per_epoch,
                device,
            )

    stats["status"] = "completed"
    stats["completed_at"] = utc_now()
    summarize(stats)
    write_json(stats_path, stats)


if __name__ == "__main__":
    main()
