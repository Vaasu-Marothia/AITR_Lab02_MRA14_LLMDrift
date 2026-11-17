"""
src/visualizer.py
Publication-quality plots for drift detection study.
Generates ROC curves, detection timelines, and window-size ablation charts.
All figures saved as PNG (300 dpi) and PDF for LaTeX inclusion.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})

COLORS = {
    "ACC-MON": "#e74c3c",    # red
    "JSD-MON": "#f39c12",    # orange
    "MMD-MON": "#2980b9",    # blue
}
LINESTYLES = {
    "ACC-MON": "--",
    "JSD-MON": "-.",
    "MMD-MON": "-",
}


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ---------------------------------------------------------------------------
# Figure 1 — ROC Curves
# ---------------------------------------------------------------------------

def plot_roc_curves(
    results: Dict[str, Dict[str, np.ndarray]],
    model_name: str,
    output_dir: str = "results/figures",
) -> None:
    """
    Parameters
    ----------
    results : {monitor_name: {'y_true': array, 'y_score': array}}
    model_name  : e.g. 'GPT-2-medium'
    output_dir  : directory to save figure
    """
    _ensure_dir(output_dir)
    fig, ax = plt.subplots(figsize=(4.5, 4.0))

    for monitor_name, data in results.items():
        y_true  = data["y_true"]
        y_score = data["y_score"]
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        ax.plot(
            fpr, tpr,
            label=f"{monitor_name} (AUC = {roc_auc:.2f})",
            color=COLORS.get(monitor_name, "black"),
            linestyle=LINESTYLES.get(monitor_name, "-"),
            linewidth=2.0,
        )

    ax.plot([0, 1], [0, 1], "k:", linewidth=0.8, label="Random (AUC = 0.50)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves — {model_name}")
    ax.legend(loc="lower right")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()

    stem = model_name.replace("-", "").replace(" ", "_").lower()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(output_dir, f"roc_{stem}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved ROC figure → {output_dir}/roc_{stem}.[png|pdf]")


# ---------------------------------------------------------------------------
# Figure 2 — Detection Timeline
# ---------------------------------------------------------------------------

def plot_detection_timeline(
    stream_length: int,
    drift_onset: int,
    monitor_statistics: Dict[str, List[float]],
    monitor_thresholds: Dict[str, float],
    model_name: str,
    output_dir: str = "results/figures",
) -> None:
    """
    Plot detector statistic over the test stream with drift onset and
    detection points marked.

    Parameters
    ----------
    stream_length       : total number of samples in test stream
    drift_onset         : index where true drift begins
    monitor_statistics  : {monitor_name: [statistic_per_step]}
    monitor_thresholds  : {monitor_name: threshold_value}
    """
    _ensure_dir(output_dir)
    n_monitors = len(monitor_statistics)
    fig, axes = plt.subplots(n_monitors, 1, figsize=(7.0, 2.5 * n_monitors), sharex=True)
    if n_monitors == 1:
        axes = [axes]

    x = np.arange(stream_length)

    for ax, (name, stats) in zip(axes, monitor_statistics.items()):
        stats_arr = np.array(stats[:stream_length])
        threshold = monitor_thresholds.get(name, None)
        color = COLORS.get(name, "black")

        ax.plot(x[:len(stats_arr)], stats_arr, color=color, linewidth=1.2, label=name)
        ax.axvline(drift_onset, color="red", linestyle="--", linewidth=1.0, label="Drift onset")

        if threshold is not None:
            ax.axhline(threshold, color=color, linestyle=":", linewidth=1.0, alpha=0.7, label="Threshold")
            # Mark first detection
            detected_indices = np.where(stats_arr > threshold)[0]
            if len(detected_indices) > 0 and detected_indices[0] > 0:
                d_idx = detected_indices[detected_indices >= drift_onset]
                if len(d_idx) > 0:
                    ax.axvline(d_idx[0], color=color, linestyle="-.", linewidth=1.0, label=f"Detected @ {d_idx[0]}")

        ax.set_ylabel(name)
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))

    axes[-1].set_xlabel("Stream index (samples)")
    fig.suptitle(f"Drift Detection Timeline — {model_name}", fontsize=12)
    fig.tight_layout()

    stem = model_name.replace("-", "").replace(" ", "_").lower()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(output_dir, f"timeline_{stem}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved timeline figure → {output_dir}/timeline_{stem}.[png|pdf]")


# ---------------------------------------------------------------------------
# Figure 3 — Window Size Ablation
# ---------------------------------------------------------------------------

def plot_ablation(
    window_sizes: List[int],
    auc_roc:      List[float],
    add:          List[float],
    fpr_at_90:    List[float],
    model_name: str,
    output_dir: str = "results/figures",
) -> None:
    """
    Three-panel plot: AUC-ROC, Average Detection Delay, FPR@TPR=0.90
    vs. sliding window size.
    """
    _ensure_dir(output_dir)
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.2))

    metrics = [
        (auc_roc,   "AUC-ROC",                      "#2980b9"),
        (add,       "Avg. Detection Delay (samples)", "#e74c3c"),
        (fpr_at_90, "FPR @ TPR = 0.90",              "#27ae60"),
    ]

    for ax, (vals, ylabel, color) in zip(axes, metrics):
        ax.plot(window_sizes, vals, "o-", color=color, linewidth=2.0, markersize=6)
        ax.set_xlabel("Window size $w$")
        ax.set_ylabel(ylabel)
        ax.set_xticks(window_sizes)
        ax.grid(True, linestyle=":", alpha=0.5)
        # Highlight best window
        best_idx = int(np.argmax(vals)) if "AUC" in ylabel else int(np.argmin(vals))
        ax.axvline(window_sizes[best_idx], color="gray", linestyle="--", linewidth=0.8)

    fig.suptitle(f"Window Size Ablation — MMD-MON — {model_name}", fontsize=12)
    fig.tight_layout()

    stem = model_name.replace("-", "").replace(" ", "_").lower()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(output_dir, f"ablation_{stem}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved ablation figure → {output_dir}/ablation_{stem}.[png|pdf]")


# ---------------------------------------------------------------------------
# Figure 4 — Task Performance Bar Chart
# ---------------------------------------------------------------------------

def plot_task_performance(
    results: Dict[str, Dict[str, float]],
    output_dir: str = "results/figures",
) -> None:
    """
    Grouped bar chart of EM and F1 across conditions and models.

    Parameters
    ----------
    results : {model_name: {condition: {'em': float, 'f1': float}}}
              e.g. {'GPT-2-medium': {'Reference': {'em': 0.134, 'f1': 0.217}, ...}}
    """
    _ensure_dir(output_dir)

    models     = list(results.keys())
    conditions = list(next(iter(results.values())).keys())
    n_conditions = len(conditions)
    x = np.arange(len(models))
    width = 0.35 / n_conditions

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.5))
    cmap = plt.get_cmap("tab10")

    for ax_idx, metric in enumerate(("em", "f1")):
        ax = axes[ax_idx]
        for c_idx, condition in enumerate(conditions):
            vals = [results[m][condition][metric] * 100 for m in models]
            offsets = (c_idx - n_conditions / 2 + 0.5) * width * 2.5
            ax.bar(x + offsets, vals, width * 2.0,
                   label=condition, color=cmap(c_idx), edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15, ha="right")
        ax.set_ylabel(("Exact Match (%)" if metric == "em" else "F1 (%)"))
        ax.set_title("Task Exact Match" if metric == "em" else "Task Token F1")
        ax.legend(fontsize=8)
        ax.grid(axis="y", linestyle=":", alpha=0.5)

    fig.suptitle("Task Performance Under Controlled Domain Shift", fontsize=12)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(output_dir, f"task_performance.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved task performance figure → {output_dir}/task_performance.[png|pdf]")
