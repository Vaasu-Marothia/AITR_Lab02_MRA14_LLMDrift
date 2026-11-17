"""
src/data_loader.py
Load TriviaQA (rc.nocontext) and partition into reference / stable-stream / drift-stream.

Drift simulation approach (frozen before result inspection):
  - "stable" questions: contain keywords suggesting geography, science, or history facts
    whose answers are unlikely to change over time.
  - "drift" questions: contain keywords suggesting person-centric, sports, or entertainment
    facts whose ground-truth answers are more temporally variable.

This is a tractable proxy for real temporal API drift as documented in
Chen et al. (2023) [arXiv:2307.09009].
"""

import os
import json
import random
import numpy as np
import pandas as pd
from datasets import load_dataset
from typing import Tuple, List


# ---------------------------------------------------------------------------
# Keyword lists (FROZEN — do not change after first run)
# ---------------------------------------------------------------------------

STABLE_KEYWORDS = [
    "capital of", "largest", "highest", "deepest", "longest", "smallest",
    "made of", "chemical", "symbol for", "founded in", "born in", "century",
    "named after", "atomic", "speed of", "distance", "how many",
    "what is the name", "what country", "what continent",
]

DRIFT_KEYWORDS = [
    "who won", "who sang", "who directed", "who played", "who scored",
    "which team won", "best known for", "whose", "who was awarded",
    "who became", "who is the", "who was the first to",
]


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def classify_question(question: str) -> str:
    """
    Classify a question as 'stable', 'drift', or 'neutral' using frozen keywords.

    Returns
    -------
    str  one of {'stable', 'drift', 'neutral'}
    """
    q = question.lower()
    if any(kw in q for kw in DRIFT_KEYWORDS):
        return "drift"
    if any(kw in q for kw in STABLE_KEYWORDS):
        return "stable"
    return "neutral"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_triviaqa(
    n_samples: int = 1500,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Load TriviaQA rc.nocontext validation split and return a DataFrame.

    Columns
    -------
    question  : str
    answers   : List[str]  (all acceptable alias strings)
    q_id      : str
    category  : str  {'stable', 'drift', 'neutral'}
    """
    set_seed(seed)
    print("Loading TriviaQA from Hugging Face datasets …")
    ds = load_dataset("trivia_qa", "rc.nocontext", split="validation")
    print(f"  Full validation set: {len(ds)} examples")

    rows = []
    for ex in ds:
        rows.append({
            "question": ex["question"],
            "answers": ex["answer"]["aliases"],
            "q_id": ex["question_id"],
        })

    df = pd.DataFrame(rows)
    df["category"] = df["question"].apply(classify_question)

    # Sample n_samples rows, preserving category distribution
    df = df.sample(min(n_samples, len(df)), random_state=seed).reset_index(drop=True)

    print(f"  Sampled: {len(df)} questions")
    print(f"  Categories: {df['category'].value_counts().to_dict()}")
    return df


# ---------------------------------------------------------------------------
# Partitioning
# ---------------------------------------------------------------------------

def partition_dataset(
    df: pd.DataFrame,
    n_reference: int = 500,
    n_stable_stream: int = 350,
    n_drift_stream: int = 350,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Partition df into reference window, stable stream, and drift stream.

    - Reference and stable stream are drawn from 'stable' + 'neutral' rows.
    - Drift stream is drawn from 'drift' rows.

    Parameters
    ----------
    df              : full TriviaQA DataFrame from load_triviaqa()
    n_reference     : size of reference window (in-distribution)
    n_stable_stream : size of in-distribution test stream segment
    n_drift_stream  : size of drifted test stream segment

    Returns
    -------
    reference, stable_stream, drift_stream  — three DataFrames
    """
    set_seed(seed)

    stable_pool = df[df["category"].isin(["stable", "neutral"])].copy()
    drift_pool  = df[df["category"] == "drift"].copy()

    n_stable_needed = n_reference + n_stable_stream
    if len(stable_pool) < n_stable_needed:
        raise ValueError(
            f"Not enough stable/neutral questions ({len(stable_pool)}) "
            f"for reference ({n_reference}) + stable stream ({n_stable_stream})."
        )
    if len(drift_pool) < n_drift_stream:
        raise ValueError(
            f"Not enough drift questions ({len(drift_pool)}) "
            f"for drift stream ({n_drift_stream})."
        )

    stable_pool = stable_pool.sample(n_stable_needed, random_state=seed).reset_index(drop=True)
    reference    = stable_pool.iloc[:n_reference].copy()
    stable_stream = stable_pool.iloc[n_reference:].copy()

    drift_stream = drift_pool.sample(n_drift_stream, random_state=seed).reset_index(drop=True)

    # Add label column
    reference["is_drift"]     = False
    stable_stream["is_drift"] = False
    drift_stream["is_drift"]  = True

    print(f"  Reference window : {len(reference)} samples")
    print(f"  Stable stream    : {len(stable_stream)} samples")
    print(f"  Drift stream     : {len(drift_stream)} samples")

    return reference, stable_stream, drift_stream


def create_drift_stream(
    stable_stream: pd.DataFrame,
    drift_stream: pd.DataFrame,
) -> pd.DataFrame:
    """
    Concatenate stable stream followed by drift stream into one sequential test stream.
    The drift onset occurs at index = len(stable_stream).

    Returns
    -------
    DataFrame with is_drift column; sequential order represents time.
    """
    stream = pd.concat(
        [stable_stream.reset_index(drop=True), drift_stream.reset_index(drop=True)],
        ignore_index=True,
    )
    return stream


# ---------------------------------------------------------------------------
# Save / load helpers
# ---------------------------------------------------------------------------

def save_partitions(
    reference: pd.DataFrame,
    stable_stream: pd.DataFrame,
    drift_stream: pd.DataFrame,
    stream: pd.DataFrame,
    output_dir: str = "data/partitions",
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    reference.to_csv(os.path.join(output_dir, "reference.csv"), index=False)
    stable_stream.to_csv(os.path.join(output_dir, "stable_stream.csv"), index=False)
    drift_stream.to_csv(os.path.join(output_dir, "drift_stream.csv"), index=False)
    stream.to_csv(os.path.join(output_dir, "test_stream.csv"), index=False)
    print(f"Partitions saved to '{output_dir}/'")


def load_partitions(
    partition_dir: str = "data/partitions",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load pre-computed partitions from disk."""
    reference     = pd.read_csv(os.path.join(partition_dir, "reference.csv"))
    stable_stream = pd.read_csv(os.path.join(partition_dir, "stable_stream.csv"))
    drift_stream  = pd.read_csv(os.path.join(partition_dir, "drift_stream.csv"))
    stream        = pd.read_csv(os.path.join(partition_dir, "test_stream.csv"))

    # Restore answers list from CSV string representation
    import ast
    for df in (reference, stable_stream, drift_stream, stream):
        df["answers"] = df["answers"].apply(ast.literal_eval)
        df["is_drift"] = df["is_drift"].astype(bool)

    return reference, stable_stream, drift_stream, stream


def prepare_and_save_partitions(config_path: str = "configs/experiment_config.json") -> None:
    """Convenience entry-point: load config, build partitions, save to disk."""
    with open(config_path) as f:
        cfg = json.load(f)

    seed = cfg["seed"]
    d    = cfg["data"]

    df = load_triviaqa(n_samples=d["n_total"], seed=seed)
    reference, stable_stream, drift_stream = partition_dataset(
        df,
        n_reference=d["n_reference"],
        n_stable_stream=d["n_stable_stream"],
        n_drift_stream=d["n_drift_stream"],
        seed=seed,
    )
    stream = create_drift_stream(stable_stream, drift_stream)
    save_partitions(reference, stable_stream, drift_stream, stream)
