"""
scripts/03_run_experiment.py
Main drift detection experiment.
Processes the sequential test stream (stable → drift) through all three monitors.
Saves per-step statistics for evaluation.

Usage:
    python scripts/03_run_experiment.py              # main experiment
    python scripts/03_run_experiment.py --ablation   # window-size ablation

IMPORTANT: Protocol was frozen before this script was run on the test stream.
Any legitimate protocol changes must be logged in research_decisions.md.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import GPT2Tokenizer, GPT2LMHeadModel
from sentence_transformers import SentenceTransformer

from src.data_loader import load_partitions
from src.drift_detector import AccuracyMonitor, JSDMonitor, MMDMonitor
from src.evaluator import compute_exact_match, compute_f1

CONFIG_PATH  = "configs/experiment_config.json"
PARTITION_DIR = "data/partitions"
RESULTS_DIR  = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers (shared with baseline script)
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(model_name, device):
    from transformers import GPT2Tokenizer, GPT2LMHeadModel
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    model.eval()
    return model, tokenizer


def run_qa_inference(model, tokenizer, question, max_new_tokens, device):
    prompt = f"Q: {question}\nA:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model(**inputs, output_hidden_states=False)
        first_token_logits = output.logits[0, -1, :].cpu().numpy()
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated_text = tokenizer.decode(
        gen_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()
    return {"generated_text": generated_text, "first_token_logits": first_token_logits}


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment_for_model(
    model_name: str,
    reference: pd.DataFrame,
    stream: pd.DataFrame,
    cfg: dict,
    device: torch.device,
    window_size_override: int = None,
) -> dict:
    """
    Run the full drift detection experiment for one model.

    Returns
    -------
    dict with per-step statistics and drift flags for all three monitors.
    """
    win_size = window_size_override or cfg["monitors"]["window_size"]
    print(f"\n--- {model_name} | window_size={win_size} ---")

    model, tokenizer = load_model_and_tokenizer(model_name, device)
    emb_model = SentenceTransformer(cfg["embedding_model"])

    # ---- Load reference artefacts ----
    stem = model_name.replace("/", "_").replace("-", "_")
    ref_logits_path = os.path.join(RESULTS_DIR, f"ref_logits_{stem}.npy")
    ref_embeddings_path = os.path.join(RESULTS_DIR, f"ref_embeddings_{stem}.npy")

    if not os.path.exists(ref_logits_path):
        raise FileNotFoundError(
            f"Reference logits not found: {ref_logits_path}\n"
            "Run 02_run_baseline.py first."
        )

    ref_logits     = np.load(ref_logits_path)        # (n_ref, vocab_size)
    ref_embeddings = np.load(ref_embeddings_path)     # (n_ref, 384)

    # Load reference correctness from baseline JSONL
    baseline_jsonl = os.path.join(RESULTS_DIR, f"baseline_{stem}.jsonl")
    ref_correctness = []
    with open(baseline_jsonl) as f:
        for line in f:
            rec = json.loads(line)
            ref_correctness.append(rec["em"])

    ref_accuracy = float(np.mean(ref_correctness))
    print(f"  Reference accuracy: {ref_accuracy*100:.1f}%")

    # ---- Initialise and calibrate monitors ----
    acc_mon = AccuracyMonitor(window_size=win_size)
    acc_mon.calibrate_from_reference(ref_correctness, percentile=cfg["monitors"]["acc_threshold_drop"] if isinstance(cfg["monitors"]["acc_threshold_drop"], float) else 95.0)

    jsd_mon = JSDMonitor(window_size=win_size, temperature=cfg["generation"]["temperature"])
    jsd_mon.calibrate_from_reference(list(ref_logits), percentile=cfg["monitors"]["jsd_percentile"])

    mmd_mon = MMDMonitor(window_size=win_size, kernel_sigma=cfg["monitors"]["mmd_kernel_sigma"])
    mmd_mon.calibrate_from_reference(ref_embeddings, percentile=cfg["monitors"]["mmd_percentile"])

    print(f"  Thresholds — ACC drop: {acc_mon.threshold_drop:.4f} | "
          f"JSD: {jsd_mon.threshold:.4f} | MMD: {mmd_mon.threshold:.6f}")

    # ---- Process test stream ----
    per_step = []
    for idx, row in tqdm(stream.iterrows(), total=len(stream), desc=f"Stream [{model_name}]"):
        question = row["question"]
        aliases  = row["answers"] if isinstance(row["answers"], list) else eval(str(row["answers"]))
        is_drift = bool(row["is_drift"])

        inf = run_qa_inference(model, tokenizer, question,
                               cfg["generation"]["max_new_tokens"], device)
        embedding = emb_model.encode([question])[0]

        em = compute_exact_match(inf["generated_text"], aliases)
        f1 = compute_f1(inf["generated_text"], aliases)

        acc_stat, acc_flag = acc_mon.update(bool(em))
        jsd_stat, jsd_flag = jsd_mon.update(inf["first_token_logits"])
        mmd_stat, mmd_flag = mmd_mon.update(embedding)

        per_step.append({
            "stream_idx": int(idx),
            "question": question,
            "generated_text": inf["generated_text"],
            "em": em,
            "f1": f1,
            "is_drift": is_drift,
            "acc_statistic": float(acc_stat),
            "acc_flag": bool(acc_flag),
            "jsd_statistic": float(jsd_stat),
            "jsd_flag": bool(jsd_flag),
            "mmd_statistic": float(mmd_stat),
            "mmd_flag": bool(mmd_flag),
        })

    # ---- Save results ----
    out_path = os.path.join(
        RESULTS_DIR,
        f"stream_{stem}_w{win_size}.jsonl"
    )
    with open(out_path, "w") as f:
        for rec in per_step:
            f.write(json.dumps(rec) + "\n")

    print(f"  Stream results saved → {out_path}")

    # Quick summary
    drift_df = pd.DataFrame(per_step)
    in_dist  = drift_df[~drift_df["is_drift"]]
    drifted  = drift_df[drift_df["is_drift"]]
    print(f"  In-dist EM: {in_dist['em'].mean()*100:.1f}%  |  Drift EM: {drifted['em'].mean()*100:.1f}%")

    return {
        "model": model_name,
        "window_size": win_size,
        "output_file": out_path,
        "thresholds": {
            "acc": float(acc_mon.threshold_drop),
            "jsd": float(jsd_mon.threshold),
            "mmd": float(mmd_mon.threshold),
        },
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--ablation", action="store_true",
                        help="Run window-size ablation for MMD-MON on primary model")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    reference, _, _, stream = load_partitions(PARTITION_DIR)
    drift_onset = int(stream[stream["is_drift"]].index.min())
    print(f"Drift onset at stream index: {drift_onset}")

    experiment_log = []

    if args.ablation:
        print("\n=== ABLATION: Window Size ===")
        primary = cfg["models"]["primary"]
        for w in cfg["ablation"]["window_sizes"]:
            rec = run_experiment_for_model(primary, reference, stream, cfg, device,
                                           window_size_override=w)
            experiment_log.append(rec)
    else:
        print("\n=== MAIN EXPERIMENT ===")
        for model_name in [cfg["models"]["primary"], cfg["models"]["comparator"]]:
            rec = run_experiment_for_model(model_name, reference, stream, cfg, device)
            experiment_log.append(rec)

    # Save experiment log
    log_path = os.path.join(RESULTS_DIR, "experiment_log.json")
    with open(log_path, "w") as f:
        json.dump(experiment_log, f, indent=2)
    print(f"\nExperiment log saved → {log_path}")
    print("Next: python scripts/04_evaluate_results.py")


if __name__ == "__main__":
    main()
