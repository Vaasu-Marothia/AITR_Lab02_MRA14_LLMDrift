"""
scripts/02_run_baseline.py
Evaluate GPT-2-medium and DistilGPT-2 on the reference window (in-distribution).
Outputs per-example EM/F1, model logits, and reference statistics for monitors.

Usage:
    python scripts/02_run_baseline.py

Outputs (saved to results/):
    baseline_gpt2medium.jsonl    — per-example outputs + logits
    baseline_distilgpt2.jsonl    — per-example outputs + logits
    reference_stats.json         — reference accuracy + embedding stats for calibration
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import GPT2Tokenizer, GPT2LMHeadModel
from sentence_transformers import SentenceTransformer

from src.data_loader import prepare_and_save_partitions, load_partitions
from src.evaluator import score_batch, compute_exact_match, compute_f1


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONFIG_PATH = "configs/experiment_config.json"
PARTITION_DIR = "data/partitions"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(model_name: str, device: torch.device):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    model.eval()
    return model, tokenizer


def run_qa_inference(
    model,
    tokenizer,
    question: str,
    max_new_tokens: int = 50,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """
    Run one QA inference step.

    Returns dict with keys:
        generated_text : str
        first_token_logits : np.ndarray  (vocab_size,) logits at the generation step
    """
    prompt = f"Q: {question}\nA:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model(
            **inputs,
            output_hidden_states=False,
            use_cache=True,
        )
        # Logits for the NEXT token (after "A:") — shape (1, vocab_size)
        first_token_logits = output.logits[0, -1, :].cpu().numpy()

        # Generate full answer
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_text = tokenizer.decode(
        gen_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()

    return {
        "generated_text": generated_text,
        "first_token_logits": first_token_logits,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate_model_on_reference(
    model_name: str,
    reference: pd.DataFrame,
    cfg: dict,
    device: torch.device,
) -> None:
    """Evaluate model on reference window; save per-example outputs + stats."""

    print(f"\n{'='*50}")
    print(f"Evaluating: {model_name}")
    print(f"Reference window size: {len(reference)}")
    print(f"{'='*50}")

    model, tokenizer = load_model_and_tokenizer(model_name, device)
    emb_model = SentenceTransformer(cfg["embedding_model"])

    results_per_example = []
    all_logits = []
    all_embeddings = []

    for _, row in tqdm(reference.iterrows(), total=len(reference), desc=model_name):
        question = row["question"]
        aliases  = row["answers"] if isinstance(row["answers"], list) else eval(row["answers"])

        # LLM inference
        inf = run_qa_inference(model, tokenizer, question,
                               max_new_tokens=cfg["generation"]["max_new_tokens"],
                               device=device)

        # QA metrics
        em = compute_exact_match(inf["generated_text"], aliases)
        f1 = compute_f1(inf["generated_text"], aliases)

        # Sentence embedding
        embedding = emb_model.encode([question])[0]

        results_per_example.append({
            "q_id": row["q_id"],
            "question": question,
            "generated_text": inf["generated_text"],
            "em": em,
            "f1": f1,
            "is_drift": bool(row["is_drift"]),
        })
        all_logits.append(inf["first_token_logits"].tolist())
        all_embeddings.append(embedding.tolist())

    # Save per-example JSONL
    stem = model_name.replace("/", "_").replace("-", "_")
    out_path = os.path.join(RESULTS_DIR, f"baseline_{stem}.jsonl")
    with open(out_path, "w") as f:
        for rec in results_per_example:
            f.write(json.dumps(rec) + "\n")

    # Save logits and embeddings as numpy (compressed)
    np.save(os.path.join(RESULTS_DIR, f"ref_logits_{stem}.npy"), np.array(all_logits))
    np.save(os.path.join(RESULTS_DIR, f"ref_embeddings_{stem}.npy"), np.array(all_embeddings))

    # Print summary
    em_mean = np.mean([r["em"] for r in results_per_example])
    f1_mean = np.mean([r["f1"] for r in results_per_example])
    print(f"\nReference EM : {em_mean*100:.1f}%")
    print(f"Reference F1 : {f1_mean*100:.1f}%")
    print(f"Results saved → {out_path}")

    return {
        "model": model_name,
        "reference_em":    float(em_mean),
        "reference_f1":    float(f1_mean),
        "n_reference":     len(reference),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=CONFIG_PATH)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Prepare partitions if not already done
    if not os.path.exists(os.path.join(PARTITION_DIR, "reference.csv")):
        print("Partitions not found — preparing now …")
        prepare_and_save_partitions(args.config)

    reference, _, _, _ = load_partitions(PARTITION_DIR)

    all_stats = []
    for model_name in [cfg["models"]["primary"], cfg["models"]["comparator"]]:
        stats = evaluate_model_on_reference(model_name, reference, cfg, device)
        all_stats.append(stats)

    # Save reference stats for calibration in experiment script
    ref_stats_path = os.path.join(RESULTS_DIR, "reference_stats.json")
    with open(ref_stats_path, "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"\nReference stats saved → {ref_stats_path}")
    print("\nNext: python scripts/03_run_experiment.py")


if __name__ == "__main__":
    main()
