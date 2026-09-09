"""Compare full-resolution model age predictions with apparent-age ratings."""

import argparse
import csv
import json
import pickle
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr, trim_mean

from differentiable_age_predictors import (
    build_fast_scorer,
    discover_images,
    rate_image_directory_fast,
)


MODEL_INFO = {
    'fairface': {
        'prediction_column': 'fairface_age',
        'label': 'FairFace',
    },
    'celeba': {
        'prediction_column': 'celeba_age',
        'label': 'CelebA / Talk-to-Edit',
    },
}


class RestrictedUnpickler(pickle.Unpickler):
    """Load primitive pickle data without permitting arbitrary globals."""

    def find_class(self, module, name):
        raise pickle.UnpicklingError(
            'Refusing pickle global {}.{}'.format(module, name)
        )


def load_age_ratings(path):
    with Path(path).expanduser().open('rb') as handle:
        data = RestrictedUnpickler(handle).load()
    ratings = data['age']
    if not isinstance(ratings, dict):
        raise TypeError("Expected data['age'] to be a dictionary")
    return {
        str(filename): np.asarray(values, dtype=np.float64)
        for filename, values in ratings.items()
    }


def load_predictions(path):
    with Path(path).expanduser().open(newline='') as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fieldnames):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def correlation_metrics(model_scores, target):
    pearson = pearsonr(model_scores, target)
    spearman = spearmanr(model_scores, target)
    return {
        'pearson_r': float(pearson.statistic),
        'pearson_p': float(pearson.pvalue),
        'pearson_ci95': [
            float(value) for value in pearson.confidence_interval(0.95)
        ],
        'spearman_rho': float(spearman.statistic),
        'spearman_p': float(spearman.pvalue),
    }


def rowwise_correlation(x, y):
    x = x - x.mean(axis=1, keepdims=True)
    y = y - y.mean(axis=1, keepdims=True)
    denominator = np.sqrt(
        np.square(x).sum(axis=1) * np.square(y).sum(axis=1)
    )
    return (x * y).sum(axis=1) / denominator


def bootstrap_summary_differences(
        model_scores,
        human_mean,
        human_median,
        human_trimmed,
        repeats,
        rng,
):
    mean_minus_median = []
    mean_minus_trimmed = []
    num_images = model_scores.size
    batch_size = 250
    for start in range(0, repeats, batch_size):
        count = min(batch_size, repeats - start)
        indices = rng.integers(0, num_images, size=(count, num_images))
        scores = model_scores[indices]
        corr_mean = rowwise_correlation(scores, human_mean[indices])
        corr_median = rowwise_correlation(scores, human_median[indices])
        corr_trimmed = rowwise_correlation(scores, human_trimmed[indices])
        mean_minus_median.extend((corr_mean - corr_median).tolist())
        mean_minus_trimmed.extend((corr_mean - corr_trimmed).tolist())

    def summarize(values):
        values = np.asarray(values)
        return {
            'estimate': float(values.mean()),
            'ci95': [
                float(np.quantile(values, 0.025)),
                float(np.quantile(values, 0.975)),
            ],
            'probability_difference_above_zero': float((values > 0).mean()),
        }

    return {
        'pearson_mean_minus_median': summarize(mean_minus_median),
        'pearson_mean_minus_trimmed_mean': summarize(mean_minus_trimmed),
    }


def split_half_reliability(rating_arrays, repeats, rng):
    mean_correlations = []
    median_correlations = []
    for _ in range(repeats):
        first_means = []
        second_means = []
        first_medians = []
        second_medians = []
        for values in rating_arrays:
            shuffled = values[rng.permutation(values.size)]
            midpoint = values.size // 2
            first = shuffled[:midpoint]
            second = shuffled[midpoint:]
            first_means.append(first.mean())
            second_means.append(second.mean())
            first_medians.append(np.median(first))
            second_medians.append(np.median(second))
        mean_correlations.append(pearsonr(first_means, second_means).statistic)
        median_correlations.append(
            pearsonr(first_medians, second_medians).statistic
        )

    def corrected_summary(values):
        values = np.asarray(values)
        corrected = 2.0 * values / (1.0 + values)
        return {
            'raw_split_half_median': float(np.median(values)),
            'spearman_brown_median': float(np.median(corrected)),
            'spearman_brown_ci95': [
                float(np.quantile(corrected, 0.025)),
                float(np.quantile(corrected, 0.975)),
            ],
            'implied_observed_correlation_ceiling': float(
                np.sqrt(np.median(corrected))
            ),
        }

    return {
        'mean': corrected_summary(mean_correlations),
        'median': corrected_summary(median_correlations),
    }


def cross_validated_linear_predictions(model_scores, target, rng, folds=10):
    indices = rng.permutation(model_scores.size)
    test_folds = np.array_split(indices, folds)
    predictions = np.empty_like(target)
    for test_indices in test_folds:
        train_mask = np.ones(model_scores.size, dtype=bool)
        train_mask[test_indices] = False
        slope, intercept = np.polyfit(
            model_scores[train_mask],
            target[train_mask],
            1,
        )
        predictions[test_indices] = (
            slope * model_scores[test_indices] + intercept
        )
    return predictions


def partial_correlation_controlling_age(x, y, age):
    controls = np.column_stack((np.ones(age.size), age, np.square(age)))
    x_residual = x - controls @ np.linalg.lstsq(controls, x, rcond=None)[0]
    y_residual = y - controls @ np.linalg.lstsq(controls, y, rcond=None)[0]
    return float(pearsonr(x_residual, y_residual).statistic)


def disagreement_metrics(
        model_scores,
        human_mean,
        rater_sd,
        rater_sem,
        rng,
):
    calibrated = cross_validated_linear_predictions(
        model_scores,
        human_mean,
        rng,
    )
    residual = calibrated - human_mean
    absolute_residual = np.abs(residual)
    quartiles = np.quantile(rater_sd, [0.25, 0.5, 0.75])
    groups = np.digitize(rater_sd, quartiles)
    by_quartile = []
    for group in range(4):
        selected = groups == group
        by_quartile.append({
            'quartile': group + 1,
            'count': int(selected.sum()),
            'rater_sd_mean': float(rater_sd[selected].mean()),
            'model_mae': float(absolute_residual[selected].mean()),
        })

    return calibrated, residual, {
        'cross_validated_mae': float(absolute_residual.mean()),
        'cross_validated_rmse': float(np.sqrt(np.square(residual).mean())),
        'pearson_abs_error_vs_rater_sd': float(
            pearsonr(absolute_residual, rater_sd).statistic
        ),
        'spearman_abs_error_vs_rater_sd': float(
            spearmanr(absolute_residual, rater_sd).statistic
        ),
        'partial_correlation_controlling_mean_age_and_squared_age': (
            partial_correlation_controlling_age(
                absolute_residual,
                rater_sd,
                human_mean,
            )
        ),
        'fraction_abs_error_within_one_human_sem': float(
            (absolute_residual <= rater_sem).mean()
        ),
        'mae_by_rater_disagreement_quartile': by_quartile,
        'highest_vs_lowest_disagreement_mae_ratio': float(
            by_quartile[-1]['model_mae'] / by_quartile[0]['model_mae']
        ),
    }


def create_human_model_scatter(
        human_mean,
        model_scores,
        correlation_results,
        output_path,
):
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.7), constrained_layout=True)
    x_line = np.linspace(human_mean.min(), human_mean.max(), 200)
    for axis, (model, info) in zip(axes, MODEL_INFO.items()):
        scores = model_scores[model]
        slope, intercept = np.polyfit(human_mean, scores, 1)
        axis.scatter(
            human_mean,
            scores,
            s=16,
            alpha=0.28,
            edgecolors='none',
            label='Individual faces',
        )
        axis.plot(
            x_line,
            slope * x_line + intercept,
            linewidth=2.2,
            label='Linear trend',
        )
        axis.set(
            title=info['label'],
            xlabel='Mean apparent-age rating (normalized)',
            ylabel='Full-resolution model age score (normalized)',
        )
        axis.grid(alpha=0.18)
        axis.text(
            0.03,
            0.97,
            'Pearson r = {:.4f}\nSpearman ρ = {:.4f}'.format(
                correlation_results[model]['mean']['pearson_r'],
                correlation_results[model]['mean']['spearman_rho'],
            ),
            transform=axis.transAxes,
            va='top',
            bbox={'facecolor': 'white', 'alpha': 0.85, 'edgecolor': 'none'},
        )
        axis.legend(loc='lower right')
    fig.suptitle(
        'Human apparent-age judgments versus model predictions (n={})'.format(
            human_mean.size
        ),
        fontsize=15,
    )
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def create_summary_comparison_plot(correlation_results, output_path):
    summaries = ('mean', 'median', 'trimmed_mean_10pct')
    labels = ('Mean', 'Median', '10% trimmed mean')
    x = np.arange(len(summaries))
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0), constrained_layout=True)
    for axis, metric, title in (
            (axes[0], 'pearson_r', 'Pearson correlation'),
            (axes[1], 'spearman_rho', 'Spearman rank correlation'),
    ):
        for offset, (model, info) in zip((-width / 2, width / 2), MODEL_INFO.items()):
            values = [
                correlation_results[model][summary][metric]
                for summary in summaries
            ]
            axis.bar(x + offset, values, width, label=info['label'])
        axis.set(
            title=title,
            ylabel='Correlation with model prediction',
            xticks=x,
            xticklabels=labels,
            ylim=(0.0, 1.0),
        )
        axis.grid(axis='y', alpha=0.18)
        axis.legend()
    fig.suptitle('Changing the human summary statistic does not improve agreement')
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def create_disagreement_plot(
        rater_sd,
        absolute_residuals,
        disagreement_results,
        output_path,
):
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.5), constrained_layout=True)
    for axis, (model, info) in zip(axes, MODEL_INFO.items()):
        residual = absolute_residuals[model]
        slope, intercept = np.polyfit(rater_sd, residual, 1)
        x_line = np.linspace(rater_sd.min(), rater_sd.max(), 200)
        axis.scatter(
            rater_sd,
            residual,
            s=16,
            alpha=0.28,
            edgecolors='none',
        )
        axis.plot(x_line, slope * x_line + intercept, linewidth=2.2)
        axis.set(
            title=info['label'],
            xlabel='Within-image rater standard deviation',
            ylabel='Absolute model–mean disagreement after CV calibration',
        )
        axis.grid(alpha=0.18)
        axis.text(
            0.03,
            0.97,
            'Spearman ρ = {:.3f}\nAge-controlled r = {:.3f}'.format(
                disagreement_results[model][
                    'spearman_abs_error_vs_rater_sd'
                ],
                disagreement_results[model][
                    'partial_correlation_controlling_mean_age_and_squared_age'
                ],
            ),
            transform=axis.transAxes,
            va='top',
            bbox={'facecolor': 'white', 'alpha': 0.85, 'edgecolor': 'none'},
        )
    fig.suptitle('Are model disagreements concentrated on ambiguous faces?')
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def create_celeba_preprocessing_plot(
        human_mean,
        original_scores,
        standard_scores,
        output_path,
):
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.5), constrained_layout=True)
    x_line = np.linspace(human_mean.min(), human_mean.max(), 200)
    for axis, scores, title in (
            (axes[0], original_scores, 'Original StyleGAN-specific preprocessing'),
            (axes[1], standard_scores, 'Standard ImageNet preprocessing'),
    ):
        slope, intercept = np.polyfit(human_mean, scores, 1)
        axis.scatter(
            human_mean,
            scores,
            s=16,
            alpha=0.28,
            edgecolors='none',
        )
        axis.plot(x_line, slope * x_line + intercept, linewidth=2.2)
        axis.set(
            title=title,
            xlabel='Mean apparent-age rating (normalized)',
            ylabel='CelebA age score (normalized)',
        )
        axis.grid(alpha=0.18)
        axis.text(
            0.03,
            0.97,
            'Pearson r = {:.4f}\nSpearman ρ = {:.4f}'.format(
                pearsonr(scores, human_mean).statistic,
                spearmanr(scores, human_mean).statistic,
            ),
            transform=axis.transAxes,
            va='top',
            bbox={'facecolor': 'white', 'alpha': 0.85, 'edgecolor': 'none'},
        )
    fig.suptitle('CelebA disagreement is largely a preprocessing mismatch')
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='Analyze agreement between apparent-age raters and models',
    )
    parser.add_argument(
        '--ratings',
        default=(
            '/Users/adamsobieszek/PycharmProjects/psych_gen_app/data/'
            'dim_to_photo_to_ratings.pkl'
        ),
    )
    parser.add_argument(
        '--predictions',
        default=(
            Path(__file__).resolve().parent
            / 'experiments'
            / 'age_resolution_128_20260907_220750_735982'
            / 'full_resolution_predictions.csv'
        ),
    )
    parser.add_argument(
        '--images',
        default='/Users/adamsobieszek/PycharmProjects/psychGAN/omi/images',
    )
    parser.add_argument(
        '--output-root',
        default=Path(__file__).resolve().parent / 'experiments',
    )
    parser.add_argument('--bootstrap-repeats', type=int, default=5000)
    parser.add_argument('--split-half-repeats', type=int, default=500)
    parser.add_argument('--seed', type=int, default=20260908)
    parser.add_argument('--skip-celeba-preprocessing-diagnostic', action='store_true')
    args = parser.parse_args()

    started = time.perf_counter()
    ratings_by_filename = load_age_ratings(args.ratings)
    prediction_rows = load_predictions(args.predictions)
    matched_rows = [
        row for row in prediction_rows
        if row['filename'] in ratings_by_filename
    ]
    if len(matched_rows) != len(prediction_rows):
        raise ValueError(
            'Only {} of {} predictions have human ratings'.format(
                len(matched_rows),
                len(prediction_rows),
            )
        )

    filenames = [row['filename'] for row in matched_rows]
    rating_arrays = [ratings_by_filename[name] for name in filenames]
    human_mean = np.asarray([values.mean() for values in rating_arrays])
    human_median = np.asarray([np.median(values) for values in rating_arrays])
    human_trimmed = np.asarray([
        trim_mean(values, 0.1) for values in rating_arrays
    ])
    rater_sd = np.asarray([
        values.std(ddof=1) for values in rating_arrays
    ])
    rater_iqr = np.asarray([
        np.quantile(values, 0.75) - np.quantile(values, 0.25)
        for values in rating_arrays
    ])
    rating_counts = np.asarray([values.size for values in rating_arrays])
    rater_sem = rater_sd / np.sqrt(rating_counts)
    model_scores = {
        model: np.asarray([
            float(row[info['prediction_column']])
            for row in matched_rows
        ])
        for model, info in MODEL_INFO.items()
    }

    corrected_celeba_scores = None
    corrected_celeba_timing = None
    if not args.skip_celeba_preprocessing_diagnostic:
        paths_by_name = {
            path.name: path for path in discover_images(args.images)
        }
        diagnostic_paths = [paths_by_name[name] for name in filenames]
        standard_scorer, diagnostic_device = build_fast_scorer(
            celeba_stylegan_scaling=False,
        )
        standard_results, corrected_celeba_timing = rate_image_directory_fast(
            scorer=standard_scorer,
            device=diagnostic_device,
            image_paths=diagnostic_paths,
            batch_size=128,
            workers=0,
            amp=False,
            channels_last=True,
            jpeg_draft=False,
        )
        corrected_by_name = {
            row['filename']: row for row in standard_results
        }
        corrected_celeba_scores = np.asarray([
            corrected_by_name[name]['celeba_age'] for name in filenames
        ])

    correlation_results = {}
    bootstrap_results = {}
    disagreement_results = {}
    calibrated_scores = {}
    residuals = {}
    for index, (model, scores) in enumerate(model_scores.items()):
        correlation_results[model] = {
            'mean': correlation_metrics(scores, human_mean),
            'median': correlation_metrics(scores, human_median),
            'trimmed_mean_10pct': correlation_metrics(scores, human_trimmed),
        }
        bootstrap_results[model] = bootstrap_summary_differences(
            scores,
            human_mean,
            human_median,
            human_trimmed,
            args.bootstrap_repeats,
            np.random.default_rng(args.seed + index),
        )
        calibrated, residual, disagreement = disagreement_metrics(
            scores,
            human_mean,
            rater_sd,
            rater_sem,
            np.random.default_rng(args.seed + 100 + index),
        )
        calibrated_scores[model] = calibrated
        residuals[model] = residual
        disagreement_results[model] = disagreement

    reliability = split_half_reliability(
        rating_arrays,
        args.split_half_repeats,
        np.random.default_rng(args.seed + 200),
    )
    model_intercorrelation = correlation_metrics(
        model_scores['fairface'],
        model_scores['celeba'],
    )
    standardized_ensemble = (
        (model_scores['fairface'] - model_scores['fairface'].mean())
        / model_scores['fairface'].std()
        + (model_scores['celeba'] - model_scores['celeba'].mean())
        / model_scores['celeba'].std()
    ) / 2.0
    ensemble_correlation = correlation_metrics(
        standardized_ensemble,
        human_mean,
    )
    celeba_preprocessing_diagnostic = None
    corrected_celeba_calibrated = None
    corrected_celeba_residual = None
    if corrected_celeba_scores is not None:
        corrected_celeba_calibrated, corrected_celeba_residual, corrected_disagreement = (
            disagreement_metrics(
                corrected_celeba_scores,
                human_mean,
                rater_sd,
                rater_sem,
                np.random.default_rng(args.seed + 300),
            )
        )
        corrected_correlations = {
            'mean': correlation_metrics(corrected_celeba_scores, human_mean),
            'median': correlation_metrics(corrected_celeba_scores, human_median),
            'trimmed_mean_10pct': correlation_metrics(
                corrected_celeba_scores,
                human_trimmed,
            ),
        }
        celeba_preprocessing_diagnostic = {
            'original_stylegan_preprocessing': correlation_results['celeba'],
            'standard_imagenet_preprocessing': corrected_correlations,
            'pearson_improvement_for_human_mean': (
                corrected_correlations['mean']['pearson_r']
                - correlation_results['celeba']['mean']['pearson_r']
            ),
            'standard_preprocessing_disagreement': corrected_disagreement,
            'timing': corrected_celeba_timing,
        }

    output_dir = (
        Path(args.output_root).expanduser().resolve()
        / ('human_model_age_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    paired_rows = []
    for index, filename in enumerate(filenames):
        row = {
            'filename': filename,
            'rating_count': int(rating_counts[index]),
            'human_mean': human_mean[index],
            'human_median': human_median[index],
            'human_trimmed_mean_10pct': human_trimmed[index],
            'human_sd': rater_sd[index],
            'human_iqr': rater_iqr[index],
            'human_sem': rater_sem[index],
        }
        for model in MODEL_INFO:
            row[model + '_score'] = model_scores[model][index]
            row[model + '_cv_calibrated_human_mean'] = (
                calibrated_scores[model][index]
            )
            row[model + '_residual'] = residuals[model][index]
            row[model + '_absolute_residual'] = abs(residuals[model][index])
        if corrected_celeba_scores is not None:
            row['celeba_standard_input_score'] = corrected_celeba_scores[index]
            row['celeba_standard_input_cv_calibrated_human_mean'] = (
                corrected_celeba_calibrated[index]
            )
            row['celeba_standard_input_residual'] = (
                corrected_celeba_residual[index]
            )
            row['celeba_standard_input_absolute_residual'] = abs(
                corrected_celeba_residual[index]
            )
        paired_rows.append(row)

    fields = list(paired_rows[0])
    write_csv(output_dir / 'paired_human_model_age.csv', paired_rows, fields)
    top_rows = []
    for model in MODEL_INFO:
        top_indices = np.argsort(np.abs(residuals[model]))[-20:][::-1]
        for rank, index in enumerate(top_indices, start=1):
            top_rows.append({
                'model': model,
                'rank': rank,
                **paired_rows[index],
            })
    write_csv(
        output_dir / 'largest_disagreements.csv',
        top_rows,
        ('model', 'rank', *fields),
    )

    human_metrics = {
        'images': len(filenames),
        'ratings_total': int(rating_counts.sum()),
        'ratings_per_image': {
            'minimum': int(rating_counts.min()),
            'median': float(np.median(rating_counts)),
            'mean': float(rating_counts.mean()),
            'maximum': int(rating_counts.max()),
        },
        'within_image_sd': {
            'mean': float(rater_sd.mean()),
            'median': float(np.median(rater_sd)),
        },
        'within_image_sem': {
            'mean': float(rater_sem.mean()),
            'median': float(np.median(rater_sem)),
        },
        'mean_vs_median': correlation_metrics(human_mean, human_median),
        'split_half_reliability': reliability,
    }
    metrics = {
        'human_ratings': human_metrics,
        'model_vs_human_summary': correlation_results,
        'bootstrap_summary_comparisons': bootstrap_results,
        'model_disagreement_vs_rater_disagreement': disagreement_results,
        'model_intercorrelation': model_intercorrelation,
        'standardized_two_model_ensemble_vs_human_mean': ensemble_correlation,
        'celeba_preprocessing_diagnostic': celeba_preprocessing_diagnostic,
    }
    with (output_dir / 'metrics.json').open('w') as handle:
        json.dump(metrics, handle, indent=2)

    create_human_model_scatter(
        human_mean,
        model_scores,
        correlation_results,
        output_dir / 'human_mean_vs_models_scatter.png',
    )
    create_summary_comparison_plot(
        correlation_results,
        output_dir / 'mean_median_trimmed_correlations.png',
    )
    create_disagreement_plot(
        rater_sd,
        {model: np.abs(values) for model, values in residuals.items()},
        disagreement_results,
        output_dir / 'model_error_vs_rater_disagreement.png',
    )
    if corrected_celeba_scores is not None:
        create_celeba_preprocessing_plot(
            human_mean,
            model_scores['celeba'],
            corrected_celeba_scores,
            output_dir / 'celeba_preprocessing_comparison.png',
        )

    metadata = {
        'created_at': datetime.now().astimezone().isoformat(),
        'ratings_path': str(Path(args.ratings).expanduser().resolve()),
        'predictions_path': str(Path(args.predictions).expanduser().resolve()),
        'images_path': str(Path(args.images).expanduser().resolve()),
        'bootstrap_repeats': args.bootstrap_repeats,
        'split_half_repeats': args.split_half_repeats,
        'random_seed': args.seed,
        'celeba_preprocessing_diagnostic_run': (
            corrected_celeba_scores is not None
        ),
        'elapsed_seconds': time.perf_counter() - started,
    }
    with (output_dir / 'metadata.json').open('w') as handle:
        json.dump(metadata, handle, indent=2)

    print(json.dumps(metrics, indent=2))
    print('Saved human/model age experiment:', output_dir)


if __name__ == '__main__':
    main()
