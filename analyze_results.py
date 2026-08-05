"""
Compares the four temporal-ablation configs from run_temporal_ablation.py
using the results pickles in results/:

  a. baseline_nontemporal        -- non-temporal features, random selection,
                                     random-shuffle bucketing.
  b. temporal_random_selection   -- temporal features, random selection,
                                     fixed sequential temporal bucketing.
  c. temporal_jitter             -- (b) + jittered bucket phase per iteration.
  d. temporal_jitter_overlap     -- (c) + 50% overlapping bucket windows.

Usage: python analyze_results.py
"""
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = "figures"

# Ordered (key, name, label) matching run_temporal_ablation.py's CONFIGS.
CONFIGS = [
    ("a", "baseline_nontemporal", "Baseline (Non-Temporal)"),
    ("b", "temporal_random_selection", "Temporal + Random Selection"),
    ("c", "temporal_jitter", "Temporal + Jitter"),
    ("d", "temporal_jitter_overlap", "Temporal + Jitter + Overlap"),
]

# Categorical slots 1-4 (blue, orange, aqua, yellow) from the validated palette.
CONFIG_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
COLOR_NORMAL = "#0ca30c"
COLOR_ANOMALOUS = "#d03b3b"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"


def compute_anomaly_scores(iterations):
    """Fig. 7's anomaly score: sum over ensemble groups/runs of the
    normalized deviation |p_i - bucket_mean| / bucket_std for each point."""
    high_risk = set(iterations[0]["high_risk_indices"])
    n_points = max(idx for it in iterations for b in it["buckets"] for idx in b) + 1
    scores = np.zeros(n_points)

    for it in iterations:
        for bucket, bucket_result in zip(it["buckets"], it["bucket_results"]):
            final_results = np.array(bucket_result["final_results"])
            n_bucket = len(bucket)
            n_runs = len(final_results) // n_bucket
            for run in range(n_runs):
                run_results = final_results[run * n_bucket:(run + 1) * n_bucket]
                mu = run_results.mean()
                sigma = run_results.std()
                if sigma == 0:
                    continue
                deviation = np.abs(run_results - mu) / sigma
                for pos, idx in enumerate(bucket):
                    scores[idx] += deviation[pos]

    y_true = np.zeros(n_points, dtype=int)
    y_true[list(high_risk)] = 1
    return scores, y_true


def threshold_metrics(scores, y_true):
    """Flag the top-N highest-scoring points as anomalous, N = true anomaly count."""
    n_anomalies = y_true.sum()
    order = np.argsort(-scores)
    y_pred = np.zeros_like(y_true)
    y_pred[order[:n_anomalies]] = 1

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(y_true)
    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}


def detection_curve(scores, y_true):
    """Fraction of dataset examined (sorted by score desc) vs. fraction of
    true anomalies captured -- Fig. 9."""
    order = np.argsort(-scores)
    sorted_true = y_true[order]
    cum_anomalies = np.cumsum(sorted_true)
    total_anomalies = y_true.sum()
    frac_dataset = np.arange(1, len(y_true) + 1) / len(y_true)
    frac_anomalies = cum_anomalies / total_anomalies
    return frac_dataset, frac_anomalies


def style_axes(ax):
    ax.set_facecolor("#fcfcfb")
    ax.grid(True, color=GRID, linewidth=0.75)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(MUTED)
    ax.tick_params(colors=MUTED)


def plot_metrics_comparison(results, out_path):
    labels = ["Recall", "Precision", "F1 Score", "Accuracy"]
    keys = ["recall", "precision", "f1", "accuracy"]
    n_configs = len(results)
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
    x = np.arange(1)
    width = 0.8 / n_configs
    offsets = (np.arange(n_configs) - (n_configs - 1) / 2) * width

    for ax, label, key in zip(axes, labels, keys):
        for (_, _, config_label, metrics, _, _), offset, color in zip(results, offsets, CONFIG_COLORS):
            ax.bar(x + offset, [metrics[key]], width, color=color, label=config_label)
        ax.set_ylim(0, 1.0)
        ax.set_xticks([])
        ax.set_title(label, color=INK, fontsize=11)
        style_axes(ax)

    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="upper center", ncol=n_configs, frameon=False,
               bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_detection_rate(results, out_path):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for (_, _, config_label, _, curve, _), color in zip(results, CONFIG_COLORS):
        ax.plot(*curve, color=color, linewidth=2, label=config_label)
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle="--", label="Random")
    ax.set_xlabel("Fraction of Dataset", color=INK)
    ax.set_ylabel("Fraction of Anomalies Detected", color=INK)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    style_axes(ax)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_score_separation(results, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharey=False)
    for ax, (_, _, config_label, _, _, (scores, y_true)) in zip(axes.flat, results):
        order = np.argsort(scores)
        sorted_scores = scores[order]
        sorted_true = y_true[order]
        colors = np.where(sorted_true == 1, COLOR_ANOMALOUS, COLOR_NORMAL)
        ax.bar(np.arange(len(sorted_scores)), sorted_scores, color=colors, width=1.0)
        ax.set_title(config_label, color=INK, fontsize=11)
        ax.set_xlabel("Data Points (Sorted)", color=INK)
        ax.set_ylabel("Sum Absolute Std. Deviation", color=INK)
        style_axes(ax)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLOR_NORMAL, label="Normal Samples"),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_ANOMALOUS, label="Anomalous Samples"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    results = []
    for key, name, label in CONFIGS:
        path = f"results/ablation_{key}_{name}.pkl"
        with open(path, "rb") as f:
            iterations = pickle.load(f)
        scores, y_true = compute_anomaly_scores(iterations)
        metrics = threshold_metrics(scores, y_true)
        curve = detection_curve(scores, y_true)
        results.append((key, name, label, metrics, curve, (scores, y_true)))

    header = f"{'Config':<30}{'Precision':>12}{'Recall':>10}{'F1':>10}{'Accuracy':>10}"
    print(header)
    for _, _, label, metrics, _, _ in results:
        print(f"{label:<30}{metrics['precision']:>12.3f}{metrics['recall']:>10.3f}"
              f"{metrics['f1']:>10.3f}{metrics['accuracy']:>10.3f}")

    print()
    for _, _, label, _, curve, _ in results:
        frac_dataset, frac_anomalies = curve
        idx_10pct = int(0.10 * len(frac_dataset)) - 1
        print(f"{label}: {frac_anomalies[idx_10pct]:.1%} of anomalies detected "
              f"within top 10% of scored points")

    os.makedirs(OUT_DIR, exist_ok=True)
    plot_metrics_comparison(results, f"{OUT_DIR}/ablation_metrics_comparison.png")
    plot_detection_rate(results, f"{OUT_DIR}/ablation_detection_rate.png")
    plot_score_separation(results, f"{OUT_DIR}/ablation_score_separation.png")
    print(f"\nSaved figures to {OUT_DIR}/")


if __name__ == "__main__":
    main()
