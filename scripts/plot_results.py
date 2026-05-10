#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"

METRICS = ("recall@10", "mrr@10", "ndcg@10")
METHOD_ORDER = ("sparse", "dense", "hybrid", "router", "oracle")
METHOD_COLORS = {
    "sparse": "#6c8ebf",
    "dense": "#82b366",
    "hybrid": "#d6b656",
    "router": "#b85450",
    "oracle": "#9673a6",
}


def parse_args():
    p = argparse.ArgumentParser(description="Plot retrieval/router comparison charts.")
    p.add_argument(
        "--datasets",
        nargs="+",
        default=["scifact", "scidocs", "nfcorpus"],
        help="Datasets under results/ to include.",
    )
    p.add_argument(
        "--out-dir",
        default=str(RESULTS_DIR),
        help="Output directory for PNGs. Per-dataset PNGs go under <out>/<dataset>/.",
    )
    return p.parse_args()


def load_dataset_scores(dataset: str) -> dict[str, dict[str, float]]:
    base = RESULTS_DIR / dataset
    scores: dict[str, dict[str, float]] = {}
    for mode in ("sparse", "dense", "hybrid"):
        path = base / f"eval_{mode}.json"
        if path.exists():
            scores[mode] = json.loads(path.read_text())
    report_path = base / "router_report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text())
        if "router" in report:
            scores["router"] = report["router"]
        if "oracle" in report:
            scores["oracle"] = report["oracle"]
    return scores


def plot_per_dataset(dataset: str, scores: dict[str, dict[str, float]], out_path: Path) -> None:
    methods = [m for m in METHOD_ORDER if m in scores]
    if not methods:
        print(f"[skip] no scores for {dataset}")
        return

    x = np.arange(len(METRICS))
    width = 0.8 / len(methods)
    fig, ax = plt.subplots(figsize=(8.5, 5.0))

    for i, method in enumerate(methods):
        values = [float(scores[method].get(metric, 0.0)) for metric in METRICS]
        offset = (i - (len(methods) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            values,
            width,
            label=method.capitalize(),
            color=METHOD_COLORS.get(method, "#888888"),
            edgecolor="black",
            linewidth=0.4,
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.005,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in METRICS])
    ax.set_ylabel("Score")
    ax.set_title(f"{dataset}: retrieval method comparison")
    ax.set_ylim(0, max(0.05, max(
        float(scores[m].get(metric, 0.0))
        for m in methods for metric in METRICS
    )) * 1.18)
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved {out_path.relative_to(ROOT)}")


def best_fixed_mrr(scores: dict[str, dict[str, float]]) -> tuple[str, float]:
    candidates = [(m, float(scores[m].get("mrr@10", 0.0))) for m in ("sparse", "dense", "hybrid") if m in scores]
    if not candidates:
        return ("none", 0.0)
    return max(candidates, key=lambda kv: kv[1])


def plot_cross_dataset(per_dataset: dict[str, dict[str, dict[str, float]]], out_path: Path) -> None:
    rows = []
    for dataset, scores in per_dataset.items():
        if not scores:
            continue
        best_name, best_value = best_fixed_mrr(scores)
        router_value = float(scores.get("router", {}).get("mrr@10", 0.0))
        oracle_value = float(scores.get("oracle", {}).get("mrr@10", 0.0))
        rows.append((dataset, best_name, best_value, router_value, oracle_value))

    if not rows:
        print("[skip] no data for cross-dataset summary")
        return

    datasets = [r[0] for r in rows]
    best_values = [r[2] for r in rows]
    router_values = [r[3] for r in rows]
    oracle_values = [r[4] for r in rows]
    best_labels = [f"best fixed\n({r[1]})" for r in rows]

    x = np.arange(len(datasets))
    width = 0.26
    fig, ax = plt.subplots(figsize=(9.0, 5.2))

    ax.bar(x - width, best_values, width, label="Best fixed baseline", color=METHOD_COLORS["hybrid"], edgecolor="black", linewidth=0.4)
    ax.bar(x, router_values, width, label="Learned router", color=METHOD_COLORS["router"], edgecolor="black", linewidth=0.4)
    ax.bar(x + width, oracle_values, width, label="Oracle upper bound", color=METHOD_COLORS["oracle"], edgecolor="black", linewidth=0.4)

    for i, (b, r, o) in enumerate(zip(best_values, router_values, oracle_values)):
        for offset, value in zip((-width, 0.0, width), (b, r, o)):
            ax.text(i + offset, value + 0.006, f"{value:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{d}\n{lbl}" for d, lbl in zip(datasets, best_labels)], fontsize=9)
    ax.set_ylabel("MRR@10")
    ax.set_title("Router vs best fixed baseline vs oracle (MRR@10)")
    ax.set_ylim(0, max(oracle_values + router_values + best_values) * 1.18)
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved {out_path.relative_to(ROOT)}")


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)

    per_dataset: dict[str, dict[str, dict[str, float]]] = {}
    for dataset in args.datasets:
        scores = load_dataset_scores(dataset)
        per_dataset[dataset] = scores
        plot_per_dataset(dataset, scores, out_dir / dataset / "comparison.png")

    plot_cross_dataset(per_dataset, out_dir / "cross_dataset_summary.png")


if __name__ == "__main__":
    main()
