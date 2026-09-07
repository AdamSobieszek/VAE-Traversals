"""Reproduce the bundle analysis without loading checkpoints or executing bundled code."""
import argparse
import json
import tarfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

NAMES = ("pointwise16", "pointwise32", "spatial32", "spatial_style32")
GROUPS = ("normal", "all", "first8", "odd")
WEIGHTS = np.array([.5, 1/6, 1/6, 1/6])


def analyze(bundle, destination):
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(bundle) as archive:
        def read(name):
            return archive.extractfile("results_bundle/" + name).read().decode()
        summaries = {n: json.loads(read(f"{n}/run_summary.json")) for n in NAMES}
        rows = {n: json.loads(read(f"{n}/best_test.json"))["samples"] for n in NAMES}
        curves = {n: [json.loads(line) for line in read(f"{n}/val_metrics.jsonl").splitlines()] for n in NAMES}
    base = summaries["pointwise16"]
    for name, summary in summaries.items():
        for key in ("source_sha256", "teacher_sha256", "validation_bank", "test_bank"):
            assert summary[key] == base[key], (name, key)
    def tensor(name, metric):
        lookup = {(r["identity"], r["group"]): r[metric] for r in rows[name]}
        identities = sorted({r["identity"] for r in rows[name]})
        return np.array([[lookup[i, g] for g in GROUPS] for i in identities])
    rng = np.random.default_rng(604)
    draws = rng.integers(0, len(tensor("pointwise16", "l1")), (20000, len(tensor("pointwise16", "l1"))))
    results = {"runs": {}, "style_vs_control": {}}
    for name, s in summaries.items():
        score = sum(w * .5 * sum(s["best_test"][g][k] / base["initial_test"][g][k] for k in ("l1", "lpips"))
                    for g, w in zip(GROUPS, WEIGHTS))
        results["runs"][name] = dict(test_score=score, test_improvement_pct=100*(1-score),
            validation_score=s["best_validation_score"], best_step=s["best_step"],
            batch1_ms=s["latency"]["1"]["batch_ms_median"], batch12_ms=s["latency"]["12"]["batch_ms_median"])
    for metric in ("l1", "mse", "lpips", "face_lpips", "tzone_l1", "tzone_edge", "outside_l1"):
        a, b = tensor("pointwise16", metric), tensor("spatial_style32", metric)
        change = 100 * (1 - b.mean(0) / a.mean(0))
        boot = 100 * (1 - b[draws].mean(1) / a[draws].mean(1))
        low, high = np.percentile(boot, [2.5, 97.5], axis=0)
        results["style_vs_control"][metric] = {
            g: dict(reduction_pct=float(change[j]), paired_95pct_interval=[float(low[j]), float(high[j])],
                    improved_identities=int((b[:, j] < a[:, j]).sum())) for j, g in enumerate(GROUPS)}
    (destination / "analysis.json").write_text(json.dumps(results, indent=2))

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), layout="constrained")
    colors = ("#707984", "#4194ae", "#bd8a42", "#1e7968")
    for name, color in zip(NAMES, colors):
        axes[0].plot([0] + [r["step"] for r in curves[name]], [1] + [r["score"] for r in curves[name]],
                     label=name, color=color, lw=2 if name == "spatial_style32" else 1.3,
                     linestyle="-" if name == "spatial_style32" else "--")
    axes[0].axhline(1, color="#aaaaaa", lw=.8)
    axes[0].set(xlabel="Training updates", ylabel="Validation score (initial aug5 = 1)",
                title="Style conditioning separates from the other heads")
    axes[0].legend(frameon=False, fontsize=9)
    axes[0].grid(alpha=.15)
    x = np.arange(4)
    for offset, metric, color, label in ((-.24, "lpips", "#1e7968", "Global LPIPS"),
                                        (0, "tzone_l1", "#4194ae", "T-zone L1"),
                                        (.24, "tzone_edge", "#bd8a42", "T-zone edge")):
        data = results["style_vs_control"][metric]
        values = np.array([data[g]["reduction_pct"] for g in GROUPS])
        bounds = np.array([data[g]["paired_95pct_interval"] for g in GROUPS])
        axes[1].bar(x+offset, values, width=.22, color=color, label=label,
                    yerr=np.stack((values-bounds[:, 0], bounds[:, 1]-values)), capsize=2)
    axes[1].set(xticks=x, xticklabels=["Normal", "All slots", "First 8", "Odd slots"],
                ylabel="Error reduction vs trained pointwise16 (%)",
                title="Clear fidelity gains; much smaller edge gains")
    axes[1].legend(frameon=False, fontsize=9)
    axes[1].grid(axis="y", alpha=.15)
    fig.savefig(destination / "architecture_comparison.png", dpi=180)
    fig.savefig(destination / "architecture_comparison.pdf")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    analyze(args.bundle, args.output)
