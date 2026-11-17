"""
scripts/04_evaluate_results.py
Compute all metrics, generate tables (CSV) and publication-quality figures.

Usage:
    python scripts/04_evaluate_results.py          # all tables and figures
    python scripts/04_evaluate_results.py --table 2  # only Table 2 (drift detection)
    python scripts/04_evaluate_results.py --table 3  # only Table 3 (ablation)
"""

import os
import sys
import json
import glob
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.visualizer import (
    plot_roc_curves,
    plot_detection_timeline,
    plot_ablation,
    plot_task_performance,
)

CONFIG_PATH  = "configs/experiment_config.json"
RESULTS_DIR  = "results"
FIGURES_DIR  = "results/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _fpr_at_tpr(y_true, y_score, target_tpr=0.90):
    """FPR at the operating point where TPR >= target_tpr."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    idx = np.searchsorted(tpr, target_tpr)
    if idx >= len(fpr):
        return float(fpr[-1])
    return float(fpr[idx])


def average_detection_delay(y_true, y_score, threshold, drift_onset):
    """
    Mean index of first threshold-crossing minus drift_onset, for positive runs.
    Lower is better.
    """
    flags = np.array(y_score) >= threshold
    # First crossing at or after drift onset
    post_drift = np.where((flags) & (np.arange(len(flags)) >= drift_onset))[0]
    if len(post_drift) == 0:
        return float(len(y_score))   # never detected → worst case
    return float(post_drift[0] - drift_onset)


def compute_optimal_threshold(y_score_ref):
    """95th percentile of reference-window statistic values."""
    return float(np.percentile(y_score_ref, 95))


# ---------------------------------------------------------------------------
# Load stream results
# ---------------------------------------------------------------------------

def load_stream(path: str) -> pd.DataFrame:
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def get_model_stem(model_name: str) -> str:
    return model_name.replace("/", "_").replace("-", "_")


# ---------------------------------------------------------------------------
# Table 1 — Task Performance
# ---------------------------------------------------------------------------

def compute_table1(cfg) -> pd.DataFrame:
    """EM and F1 on reference, in-dist stream, drift stream."""
    rows = []
    for model_name in [cfg["models"]["primary"], cfg["models"]["comparator"]]:
        stem = get_model_stem(model_name)
        w    = cfg["monitors"]["window_size"]
        stream_path = os.path.join(RESULTS_DIR, f"stream_{stem}_w{w}.jsonl")
        if not os.path.exists(stream_path):
            print(f"[WARN] {stream_path} not found — skipping")
            continue

        df = load_stream(stream_path)
        in_dist  = df[~df["is_drift"]]
        drifted  = df[df["is_drift"]]

        # Reference metrics from baseline JSONL
        ref_path = os.path.join(RESULTS_DIR, f"baseline_{stem}.jsonl")
        ref_df = load_stream(ref_path) if os.path.exists(ref_path) else pd.DataFrame()

        label = "GPT-2-medium" if "medium" in model_name else "DistilGPT-2"

        for cond, subset in [
            ("Reference",       ref_df),
            ("In-dist stream",  in_dist),
            ("Drift stream",    drifted),
        ]:
            if len(subset) == 0:
                continue
            rows.append({
                "Model": label,
                "Condition": cond,
                "EM (%)": f"{subset['em'].mean()*100:.1f}",
                "F1 (%)": f"{subset['f1'].mean()*100:.1f}",
                "n": len(subset),
            })

    t1 = pd.DataFrame(rows)
    t1.to_csv(os.path.join(RESULTS_DIR, "table1_task_performance.csv"), index=False)
    print("\nTable 1 — Task Performance:")
    print(t1.to_string(index=False))
    return t1


# ---------------------------------------------------------------------------
# Table 2 — Drift Detection Performance
# ---------------------------------------------------------------------------

def compute_table2(cfg) -> pd.DataFrame:
    rows = []
    model_labels = {
        cfg["models"]["primary"]:   "GPT-2-medium",
        cfg["models"]["comparator"]: "DistilGPT-2",
    }
    w = cfg["monitors"]["window_size"]

    for model_name, label in model_labels.items():
        stem = get_model_stem(model_name)
        stream_path = os.path.join(RESULTS_DIR, f"stream_{stem}_w{w}.jsonl")
        if not os.path.exists(stream_path):
            print(f"[WARN] {stream_path} not found — skipping")
            continue

        df = load_stream(stream_path)
        y_true = df["is_drift"].astype(int).values
        drift_onset = int(df[df["is_drift"]]["stream_idx"].min()) if df["is_drift"].any() else len(df)

        for monitor, stat_col in [
            ("ACC-MON", "acc_statistic"),
            ("JSD-MON", "jsd_statistic"),
            ("MMD-MON", "mmd_statistic"),
        ]:
            y_score = df[stat_col].values
            fpr_arr, tpr_arr, _ = roc_curve(y_true, y_score)
            auc_roc = auc(fpr_arr, tpr_arr)
            fpr90   = _fpr_at_tpr(y_true, y_score, 0.90)
            thresh  = compute_optimal_threshold(y_score[:drift_onset])  # reference-window stats
            add     = average_detection_delay(y_true, y_score, thresh, drift_onset)

            rows.append({
                "Monitor": monitor,
                "Model": label,
                "AUC-ROC": f"{auc_roc:.2f}",
                "Avg. Detection Delay": f"{add:.0f}",
                "FPR@TPR=0.90": f"{fpr90:.2f}",
            })

    t2 = pd.DataFrame(rows)
    t2.to_csv(os.path.join(RESULTS_DIR, "table2_drift_detection.csv"), index=False)
    print("\nTable 2 — Drift Detection Performance:")
    print(t2.to_string(index=False))
    return t2


# ---------------------------------------------------------------------------
# Table 3 — Ablation
# ---------------------------------------------------------------------------

def compute_table3(cfg) -> pd.DataFrame:
    primary = cfg["models"]["primary"]
    stem    = get_model_stem(primary)
    rows = []

    for w in cfg["ablation"]["window_sizes"]:
        stream_path = os.path.join(RESULTS_DIR, f"stream_{stem}_w{w}.jsonl")
        if not os.path.exists(stream_path):
            print(f"[WARN] {stream_path} not found — run with --ablation first")
            continue

        df = load_stream(stream_path)
        y_true      = df["is_drift"].astype(int).values
        y_score     = df["mmd_statistic"].values
        drift_onset = int(df[df["is_drift"]]["stream_idx"].min()) if df["is_drift"].any() else len(df)

        fpr_arr, tpr_arr, _ = roc_curve(y_true, y_score)
        auc_roc = auc(fpr_arr, tpr_arr)
        thresh  = compute_optimal_threshold(y_score[:drift_onset])
        add     = average_detection_delay(y_true, y_score, thresh, drift_onset)
        fpr90   = _fpr_at_tpr(y_true, y_score, 0.90)

        rows.append({
            "Window w": w,
            "AUC-ROC": f"{auc_roc:.2f}",
            "Avg. Detection Delay": f"{add:.0f}",
            "FPR@TPR=0.90": f"{fpr90:.2f}",
        })

    t3 = pd.DataFrame(rows)
    t3.to_csv(os.path.join(RESULTS_DIR, "table3_ablation.csv"), index=False)
    print("\nTable 3 — Window Size Ablation (MMD-MON, GPT-2-medium):")
    print(t3.to_string(index=False))
    return t3


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def generate_figures(cfg) -> None:
    model_labels = {
        cfg["models"]["primary"]:   "GPT-2-medium",
        cfg["models"]["comparator"]: "DistilGPT-2",
    }
    w = cfg["monitors"]["window_size"]

    for model_name, label in model_labels.items():
        stem = get_model_stem(model_name)
        stream_path = os.path.join(RESULTS_DIR, f"stream_{stem}_w{w}.jsonl")
        if not os.path.exists(stream_path):
            continue

        df = load_stream(stream_path)
        y_true = df["is_drift"].astype(int).values
        drift_onset = int(df[df["is_drift"]]["stream_idx"].min()) if df["is_drift"].any() else len(df)

        # Figure 1 — ROC Curves
        roc_data = {}
        for monitor, stat_col in [
            ("ACC-MON", "acc_statistic"),
            ("JSD-MON", "jsd_statistic"),
            ("MMD-MON", "mmd_statistic"),
        ]:
            roc_data[monitor] = {
                "y_true": y_true,
                "y_score": df[stat_col].values,
            }
        plot_roc_curves(roc_data, label, FIGURES_DIR)

        # Figure 2 — Detection Timeline
        monitor_stats = {
            "ACC-MON": df["acc_statistic"].tolist(),
            "JSD-MON": df["jsd_statistic"].tolist(),
            "MMD-MON": df["mmd_statistic"].tolist(),
        }
        thresholds = {
            m: compute_optimal_threshold(df[col].values[:drift_onset])
            for m, col in [
                ("ACC-MON", "acc_statistic"),
                ("JSD-MON", "jsd_statistic"),
                ("MMD-MON", "mmd_statistic"),
            ]
        }
        plot_detection_timeline(len(df), drift_onset, monitor_stats, thresholds, label, FIGURES_DIR)

    # Figure 3 — Ablation
    primary = cfg["models"]["primary"]
    stem    = get_model_stem(primary)
    abl_rows = {"window_sizes": [], "auc": [], "add": [], "fpr": []}
    for w_a in cfg["ablation"]["window_sizes"]:
        p = os.path.join(RESULTS_DIR, f"stream_{stem}_w{w_a}.jsonl")
        if not os.path.exists(p):
            continue
        df = load_stream(p)
        y_true  = df["is_drift"].astype(int).values
        y_score = df["mmd_statistic"].values
        d_onset = int(df[df["is_drift"]]["stream_idx"].min()) if df["is_drift"].any() else len(df)
        fpr_arr, tpr_arr, _ = roc_curve(y_true, y_score)
        thresh = compute_optimal_threshold(y_score[:d_onset])
        abl_rows["window_sizes"].append(w_a)
        abl_rows["auc"].append(auc(fpr_arr, tpr_arr))
        abl_rows["add"].append(average_detection_delay(y_true, y_score, thresh, d_onset))
        abl_rows["fpr"].append(_fpr_at_tpr(y_true, y_score, 0.90))

    if abl_rows["window_sizes"]:
        plot_ablation(
            abl_rows["window_sizes"], abl_rows["auc"], abl_rows["add"], abl_rows["fpr"],
            "GPT-2-medium", FIGURES_DIR,
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--table", type=int, choices=[1, 2, 3],
                        help="Compute only the specified table")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    if args.table == 1 or args.table is None:
        compute_table1(cfg)
    if args.table == 2 or args.table is None:
        compute_table2(cfg)
    if args.table == 3 or args.table is None:
        compute_table3(cfg)
    if args.table is None:
        generate_figures(cfg)
        print(f"\nAll figures saved → {FIGURES_DIR}/")

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
