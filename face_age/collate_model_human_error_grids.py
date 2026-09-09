"""Collate photo grids for young/old model–human age discrepancies.

Young and old faces are the bottom and top quartiles of the human mean
apparent-age rating. Discrepancy is the difference in percentile ranks so
that "relatively older" is comparable across the two model score scales.
CelebA uses the standard ImageNet-preprocessing scores only.
"""

import argparse
import base64
import csv
import io
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


IMAGE_DIR = Path('/Users/adamsobieszek/PycharmProjects/psychGAN/omi/images')
PAIRED_CSV = (
    Path(__file__).resolve().parent
    / 'experiments'
    / 'human_model_age_20260908_005342_647916'
    / 'paired_human_model_age.csv'
)

MODELS = {
    'fairface': {
        'label': 'FairFace',
        'score_column': 'fairface_score',
    },
    'celeba': {
        'label': 'CelebA (standard ImageNet input)',
        'score_column': 'celeba_standard_input_score',
    },
}

ERROR_TYPES = (
    {
        'id': 'young_model_older',
        'age_group': 'young',
        'direction': 'model_older',
        'title': 'Young faces: model relatively older than raters',
        'subtitle': (
            'Lowest human-age quartile. Largest positive model-minus-human '
            'percentile gap.'
        ),
    },
    {
        'id': 'young_raters_older',
        'age_group': 'young',
        'direction': 'raters_older',
        'title': 'Young faces: raters relatively older than model',
        'subtitle': (
            'Lowest human-age quartile. Largest negative model-minus-human '
            'percentile gap.'
        ),
    },
    {
        'id': 'old_model_older',
        'age_group': 'old',
        'direction': 'model_older',
        'title': 'Old faces: model relatively older than raters',
        'subtitle': (
            'Highest human-age quartile. Largest positive model-minus-human '
            'percentile gap.'
        ),
    },
    {
        'id': 'old_raters_older',
        'age_group': 'old',
        'direction': 'raters_older',
        'title': 'Old faces: raters relatively older than model',
        'subtitle': (
            'Highest human-age quartile. Largest negative model-minus-human '
            'percentile gap.'
        ),
    },
)


def percentile_ranks(values):
    order = np.argsort(values, kind='mergesort')
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = (np.arange(values.size) + 0.5) / values.size
    return ranks


def load_rows(path):
    with Path(path).open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    human = np.asarray([float(row['human_mean']) for row in rows])
    young_cut, old_cut = np.quantile(human, [0.25, 0.75])
    human_pct = percentile_ranks(human)
    for index, row in enumerate(rows):
        row['_human_mean'] = human[index]
        row['_human_pct'] = human_pct[index]
        row['_age_group'] = (
            'young' if human[index] <= young_cut
            else 'old' if human[index] >= old_cut
            else 'middle'
        )
        for model, info in MODELS.items():
            score = float(row[info['score_column']])
            row['_{}_score'.format(model)] = score
    for model in MODELS:
        scores = np.asarray([row['_{}_score'.format(model)] for row in rows])
        model_pct = percentile_ranks(scores)
        for index, row in enumerate(rows):
            row['_{}_pct'.format(model)] = model_pct[index]
            row['_{}_gap'.format(model)] = (
                model_pct[index] - row['_human_pct']
            )
    return rows, {
        'young_max': float(young_cut),
        'old_min': float(old_cut),
        'young_count': int(sum(row['_age_group'] == 'young' for row in rows)),
        'old_count': int(sum(row['_age_group'] == 'old' for row in rows)),
    }


def select_examples(rows, model, error, count):
    candidates = [
        row for row in rows
        if row['_age_group'] == error['age_group']
    ]
    reverse = error['direction'] == 'model_older'
    ranked = sorted(
        candidates,
        key=lambda row: row['_{}_gap'.format(model)],
        reverse=reverse,
    )
    return ranked[:count]


def draw_grid(examples, model, error, output_path):
    columns = 3
    rows_n = int(np.ceil(len(examples) / columns))
    fig, axes = plt.subplots(
        rows_n,
        columns,
        figsize=(10.2, 3.55 * rows_n),
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)
    for index, axis in enumerate(axes.ravel()):
        axis.set_axis_off()
        if index >= len(examples):
            continue
        example = examples[index]
        image = Image.open(IMAGE_DIR / example['filename']).convert('RGB')
        axis.imshow(image)
        gap = example['_{}_gap'.format(model)]
        axis.set_title(
            '{}\nH {:.2f}  M {:.2f}  Δq {:+.2f}'.format(
                example['filename'],
                example['_human_mean'],
                example['_{}_score'.format(model)],
                gap,
            ),
            fontsize=9,
            pad=6,
        )
    fig.suptitle(
        '{} — {}'.format(MODELS[model]['label'], error['title']),
        fontsize=13,
    )
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def jpeg_data_uri(path, max_width=920, quality=72):
    image = Image.open(path).convert('RGB')
    if image.width > max_width:
        height = int(round(image.height * max_width / float(image.width)))
        image = image.resize((max_width, height), Image.Resampling.BILINEAR)
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=quality, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode('ascii')
    return 'data:image/jpeg;base64,' + encoded


def write_csv(path, rows, fieldnames):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description='Build photo grids of young/old model–human age errors',
    )
    parser.add_argument('--predictions', default=str(PAIRED_CSV))
    parser.add_argument('--count', type=int, default=9)
    parser.add_argument(
        '--output-root',
        default=str(Path(__file__).resolve().parent / 'experiments'),
    )
    args = parser.parse_args()

    rows, cuts = load_rows(args.predictions)
    output_dir = (
        Path(args.output_root).expanduser().resolve()
        / ('human_model_error_grids_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    selection_rows = []
    grids = []
    for model, model_info in MODELS.items():
        for error in ERROR_TYPES:
            examples = select_examples(rows, model, error, args.count)
            stem = '{}_{}'.format(model, error['id'])
            grid_path = output_dir / (stem + '.png')
            draw_grid(examples, model, error, grid_path)
            for rank, example in enumerate(examples, start=1):
                selection_rows.append({
                    'model': model,
                    'error_type': error['id'],
                    'rank': rank,
                    'filename': example['filename'],
                    'age_group': example['_age_group'],
                    'human_mean': example['_human_mean'],
                    'human_percentile': example['_human_pct'],
                    'model_score': example['_{}_score'.format(model)],
                    'model_percentile': example['_{}_pct'.format(model)],
                    'percentile_gap_model_minus_human': example[
                        '_{}_gap'.format(model)
                    ],
                })
            grids.append({
                'id': stem,
                'model': model,
                'model_label': model_info['label'],
                'error_id': error['id'],
                'title': error['title'],
                'subtitle': error['subtitle'],
                'png_path': str(grid_path),
                'mean_gap': float(np.mean([
                    example['_{}_gap'.format(model)] for example in examples
                ])),
                'filenames': [example['filename'] for example in examples],
                'data_uri': jpeg_data_uri(grid_path),
            })

    write_csv(
        output_dir / 'selected_error_examples.csv',
        selection_rows,
        (
            'model',
            'error_type',
            'rank',
            'filename',
            'age_group',
            'human_mean',
            'human_percentile',
            'model_score',
            'model_percentile',
            'percentile_gap_model_minus_human',
        ),
    )
    metadata = {
        'created_at': datetime.now().astimezone().isoformat(),
        'source_predictions': str(Path(args.predictions).expanduser().resolve()),
        'image_dir': str(IMAGE_DIR),
        'celeba_preprocessing': 'standard ImageNet [0,1] normalization',
        'young_old_definition': (
            'young = bottom quartile of human mean apparent age; '
            'old = top quartile of human mean apparent age'
        ),
        'discrepancy': (
            'percentile(model score) - percentile(human mean), '
            'ranks computed over all 1004 faces'
        ),
        'quartile_cuts': cuts,
        'examples_per_grid': args.count,
    }
    with (output_dir / 'metadata.json').open('w') as handle:
        json.dump(metadata, handle, indent=2)
    with (output_dir / 'grid_data_uris.json').open('w') as handle:
        json.dump(grids, handle)

    print(json.dumps({**metadata, 'output_dir': str(output_dir)}, indent=2))
    print('Saved error-grid experiment:', output_dir)


if __name__ == '__main__':
    main()
