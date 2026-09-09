#!/usr/bin/env python
"""Read VP score summaries from the structured stats file."""

import argparse
import json
from pathlib import Path


def get_score_summary(target_dir):
    stats_path = Path(target_dir) / "stats.json"
    if not stats_path.is_file():
        return []
    with stats_path.open("r", encoding="utf-8") as handle:
        stats = json.load(handle)
    return (stats.get("summary") or {}).get("by_train_fraction", [])


def get_dis_score(args):
    """Return the fixed-mode score, retaining the historical public helper."""
    rows = get_score_summary(args.target_dir)
    if not rows:
        return 0.0
    return float(rows[-1]["mean_best_accuracy"])


def main():
    parser = argparse.ArgumentParser(description="Read VP metric results.")
    parser.add_argument("--target-dir", "--target_dir", required=True)
    args = parser.parse_args()
    rows = get_score_summary(args.target_dir)
    if not rows:
        print("No completed validation results found.")
        return
    for row in rows:
        print(
            "train_fraction={:.6g} mean_best_accuracy={:.3f} std={:.3f} folds={}".format(
                row["train_fraction"],
                row["mean_best_accuracy"],
                row["std_best_accuracy"],
                len(row["fold_best_accuracies"]),
            )
        )


if __name__ == "__main__":
    main()
