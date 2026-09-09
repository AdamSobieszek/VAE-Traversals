"""Measure sensitivity of both age raters to 128x128 image downsampling.

The strict reference path decodes every original image at full resolution and
resizes it directly to the models' 224x224 input. The low-resolution path
decodes the same original, downsamples/crops it to 128x128, and then upsamples
it to 224x224. Both paths use identical frozen models and preprocessing.
"""

import argparse
import csv
import json
import platform
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision

from differentiable_age_predictors import (
    build_fast_scorer,
    discover_images,
    rate_image_directory_fast,
    resolve_device,
)


MODEL_INFO = {
    'fairface': {
        'full_column': 'fairface_age',
        'low_column': 'fairface_age',
        'title': 'FairFace age (9 bins)',
        'bins': 9,
    },
    'celeba': {
        'full_column': 'celeba_age',
        'low_column': 'celeba_age',
        'title': 'CelebA / Talk-to-Edit age (6 bins)',
        'bins': 6,
    },
}


def _write_csv(path, rows, fieldnames):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _pair_predictions(full_results, low_results):
    full_by_name = {row['filename']: row for row in full_results}
    low_by_name = {row['filename']: row for row in low_results}
    if full_by_name.keys() != low_by_name.keys():
        missing_low = sorted(full_by_name.keys() - low_by_name.keys())
        missing_full = sorted(low_by_name.keys() - full_by_name.keys())
        raise ValueError(
            'Prediction sets differ: missing low={}, missing full={}'.format(
                missing_low[:5],
                missing_full[:5],
            )
        )

    rows = []
    for full_row in full_results:
        filename = full_row['filename']
        low_row = low_by_name[filename]
        row = {'filename': filename}
        for model, info in MODEL_INFO.items():
            full = float(full_row[info['full_column']])
            low = float(low_row[info['low_column']])
            row[model + '_full'] = full
            row[model + '_128'] = low
            row[model + '_error'] = low - full
            row[model + '_absolute_error'] = abs(low - full)
        rows.append(row)
    return rows


def _quantile_bins(full, low, num_bins=10):
    order = np.argsort(full)
    groups = np.array_split(order, num_bins)
    rows = []
    for index, group in enumerate(groups, start=1):
        if group.size == 0:
            continue
        x = full[group]
        y = low[group]
        error = y - x
        rows.append({
            'quantile_bin': index,
            'count': int(group.size),
            'full_min': float(x.min()),
            'full_max': float(x.max()),
            'full_mean': float(x.mean()),
            'low_128_mean': float(y.mean()),
            'mean_error': float(error.mean()),
            'mean_absolute_error': float(np.abs(error).mean()),
        })
    return rows


def _metrics(full, low, num_age_bins):
    error = low - full
    absolute_error = np.abs(error)
    slope, intercept = np.polyfit(full, low, 1)
    correlation = np.corrcoef(full, low)[0, 1]
    denominator = np.square(full - full.mean()).sum()
    r_squared_oracle = 1.0 - np.square(error).sum() / denominator
    mae = absolute_error.mean()
    rmse = np.sqrt(np.square(error).mean())
    return {
        'count': int(full.size),
        'mean_absolute_error': float(mae),
        'mean_absolute_error_age_bins': float(mae * num_age_bins),
        'root_mean_squared_error': float(rmse),
        'root_mean_squared_error_age_bins': float(rmse * num_age_bins),
        'mean_bias_low_minus_full': float(error.mean()),
        'mean_bias_age_bins': float(error.mean() * num_age_bins),
        'median_absolute_error': float(np.median(absolute_error)),
        'p95_absolute_error': float(np.quantile(absolute_error, 0.95)),
        'maximum_absolute_error': float(absolute_error.max()),
        'pearson_correlation': float(correlation),
        'r_squared_against_oracle': float(r_squared_oracle),
        'linear_slope': float(slope),
        'linear_intercept': float(intercept),
        'fraction_within_0.01': float((absolute_error <= 0.01).mean()),
        'fraction_within_0.05': float((absolute_error <= 0.05).mean()),
    }


def _arrays(paired_rows, model):
    full = np.asarray([row[model + '_full'] for row in paired_rows])
    low = np.asarray([row[model + '_128'] for row in paired_rows])
    return full, low


def create_scatterplot(paired_rows, metrics, binned, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.0), constrained_layout=True)
    for axis, (model, info) in zip(axes, MODEL_INFO.items()):
        full, low = _arrays(paired_rows, model)
        combined_min = min(full.min(), low.min())
        combined_max = max(full.max(), low.max())
        padding = max((combined_max - combined_min) * 0.08, 0.01)
        limits = (combined_min - padding, combined_max + padding)
        x_line = np.linspace(*limits, 200)
        model_metrics = metrics[model]

        axis.scatter(
            full,
            low,
            s=16,
            alpha=0.28,
            edgecolors='none',
            label='Individual faces',
        )
        axis.plot(
            x_line,
            x_line,
            color='black',
            linewidth=1.4,
            linestyle=':',
            label='No-change line (y = x)',
        )
        axis.plot(
            x_line,
            model_metrics['linear_slope'] * x_line + model_metrics['linear_intercept'],
            linewidth=2.0,
            label='Linear trend',
        )
        axis.plot(
            [row['full_mean'] for row in binned[model]],
            [row['low_128_mean'] for row in binned[model]],
            marker='o',
            markersize=4,
            linewidth=1.5,
            label='Decile mean trend',
        )
        axis.set(
            title=info['title'],
            xlabel='Full-resolution predicted age score (oracle)',
            ylabel='128x128 predicted age score',
            xlim=limits,
            ylim=limits,
        )
        axis.set_aspect('equal', adjustable='box')
        axis.grid(alpha=0.18)
        axis.text(
            0.03,
            0.97,
            'MAE = {:.4f} ({:.3f} bins)\nr = {:.4f}\nslope = {:.3f}'.format(
                model_metrics['mean_absolute_error'],
                model_metrics['mean_absolute_error_age_bins'],
                model_metrics['pearson_correlation'],
                model_metrics['linear_slope'],
            ),
            transform=axis.transAxes,
            va='top',
            bbox={'facecolor': 'white', 'alpha': 0.85, 'edgecolor': 'none'},
        )
        axis.legend(loc='lower right', fontsize=8)

    fig.suptitle(
        'Effect of 128x128 downsampling on predicted age',
        fontsize=15,
    )
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def create_residual_plot(paired_rows, binned, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.3), constrained_layout=True)
    for axis, (model, info) in zip(axes, MODEL_INFO.items()):
        full, low = _arrays(paired_rows, model)
        error = low - full
        axis.scatter(full, error, s=16, alpha=0.28, edgecolors='none')
        axis.axhline(0.0, color='black', linewidth=1.4, linestyle=':')
        axis.plot(
            [row['full_mean'] for row in binned[model]],
            [row['mean_error'] for row in binned[model]],
            marker='o',
            markersize=4,
            linewidth=2.0,
            label='Mean error by oracle-age decile',
        )
        axis.set(
            title=info['title'],
            xlabel='Full-resolution predicted age score (oracle)',
            ylabel='Error: 128x128 score − full-resolution score',
        )
        axis.grid(alpha=0.18)
        axis.legend(loc='best', fontsize=8)
    fig.suptitle('Where downsampling changes predicted age', fontsize=15)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _summary_text(metrics):
    lines = [
        'This experiment measures prediction consistency, not ground-truth age accuracy.',
        'The full-resolution model predictions are treated as the oracle.',
        '',
    ]
    for model, info in MODEL_INFO.items():
        values = metrics[model]
        lines.extend([
            info['title'],
            '  MAE: {:.6f} normalized score ({:.4f} age bins)'.format(
                values['mean_absolute_error'],
                values['mean_absolute_error_age_bins'],
            ),
            '  RMSE: {:.6f} normalized score ({:.4f} age bins)'.format(
                values['root_mean_squared_error'],
                values['root_mean_squared_error_age_bins'],
            ),
            '  Bias (low - full): {:.6f} ({:.4f} age bins)'.format(
                values['mean_bias_low_minus_full'],
                values['mean_bias_age_bins'],
            ),
            '  Pearson r: {:.6f}; R2 vs oracle: {:.6f}'.format(
                values['pearson_correlation'],
                values['r_squared_against_oracle'],
            ),
            '  Trend: low = {:.6f} * full + {:.6f}'.format(
                values['linear_slope'],
                values['linear_intercept'],
            ),
            '  Within 0.01 / 0.05: {:.2%} / {:.2%}'.format(
                values['fraction_within_0.01'],
                values['fraction_within_0.05'],
            ),
            '',
        ])
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Compare full-resolution and 128x128 age ratings',
    )
    parser.add_argument(
        '--input-dir',
        default='/Users/adamsobieszek/PycharmProjects/psychGAN/omi/images',
    )
    parser.add_argument(
        '--output-root',
        default=str(Path(__file__).resolve().parent / 'experiments'),
    )
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()

    image_paths = discover_images(args.input_dir, args.limit)
    device = resolve_device(args.device)
    output_dir = (
        Path(args.output_root).expanduser().resolve()
        / ('age_resolution_128_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    setup_started = time.perf_counter()
    scorer, device = build_fast_scorer(
        device=device,
        channels_last=True,
        compile_model=False,
    )
    setup_seconds = time.perf_counter() - setup_started

    full_results, full_timing = rate_image_directory_fast(
        scorer=scorer,
        device=device,
        image_paths=image_paths,
        batch_size=args.batch_size,
        workers=args.workers,
        amp=False,
        channels_last=True,
        jpeg_draft=False,
        intermediate_size=None,
    )
    low_results, low_timing = rate_image_directory_fast(
        scorer=scorer,
        device=device,
        image_paths=image_paths,
        batch_size=args.batch_size,
        workers=args.workers,
        amp=False,
        channels_last=True,
        jpeg_draft=False,
        intermediate_size=128,
    )
    paired_rows = _pair_predictions(full_results, low_results)

    metrics = {}
    binned = {}
    binned_rows = []
    for model, info in MODEL_INFO.items():
        full, low = _arrays(paired_rows, model)
        metrics[model] = _metrics(full, low, info['bins'])
        binned[model] = _quantile_bins(full, low)
        for row in binned[model]:
            binned_rows.append({'model': model, **row})

    prediction_fields = [
        'filename',
        'fairface_full',
        'fairface_128',
        'fairface_error',
        'fairface_absolute_error',
        'celeba_full',
        'celeba_128',
        'celeba_error',
        'celeba_absolute_error',
    ]
    _write_csv(output_dir / 'paired_predictions.csv', paired_rows, prediction_fields)
    _write_csv(
        output_dir / 'full_resolution_predictions.csv',
        full_results,
        ('filename', 'fairface_age', 'celeba_age'),
    )
    _write_csv(
        output_dir / 'low_resolution_128_predictions.csv',
        low_results,
        ('filename', 'fairface_age', 'celeba_age'),
    )
    _write_csv(
        output_dir / 'metrics_by_oracle_age_decile.csv',
        binned_rows,
        (
            'model',
            'quantile_bin',
            'count',
            'full_min',
            'full_max',
            'full_mean',
            'low_128_mean',
            'mean_error',
            'mean_absolute_error',
        ),
    )
    with (output_dir / 'metrics.json').open('w') as handle:
        json.dump(metrics, handle, indent=2)

    create_scatterplot(
        paired_rows,
        metrics,
        binned,
        output_dir / 'age_full_vs_128_scatter.png',
    )
    create_residual_plot(
        paired_rows,
        binned,
        output_dir / 'age_128_residuals.png',
    )
    (output_dir / 'summary.txt').write_text(_summary_text(metrics))

    metadata = {
        'created_at': datetime.now().astimezone().isoformat(),
        'input_dir': str(Path(args.input_dir).expanduser().resolve()),
        'images': len(image_paths),
        'comparison': {
            'oracle': 'full image decode -> resize/center-crop to 224x224',
            'low_resolution': (
                'full image decode -> resize/center-crop to 128x128 '
                '-> bilinear upscale to 224x224'
            ),
            'jpeg_draft': False,
            'interpolation': 'Pillow bilinear',
        },
        'device': str(device),
        'machine': platform.machine(),
        'torch_version': torch.__version__,
        'torchvision_version': torchvision.__version__,
        'batch_size': args.batch_size,
        'workers': args.workers,
        'amp': False,
        'compile': False,
        'channels_last': True,
        'setup_seconds': setup_seconds,
        'full_resolution_timing': full_timing,
        'low_resolution_timing': low_timing,
    }
    with (output_dir / 'metadata.json').open('w') as handle:
        json.dump(metadata, handle, indent=2)

    print(_summary_text(metrics))
    print('Saved resolution experiment:', output_dir)


if __name__ == '__main__':
    main()
