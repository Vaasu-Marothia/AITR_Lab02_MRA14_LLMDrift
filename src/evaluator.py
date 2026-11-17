"""
src/evaluator.py
Exact Match and token-level F1 metrics for open-domain QA.
Follows the TriviaQA / SQuAD convention (Rajpurkar et al., 2016; Joshi et al., 2017).
"""

import re
import string
from typing import List, Union


# ---------------------------------------------------------------------------
# Text normalisation helpers
# ---------------------------------------------------------------------------

def normalize_answer(text: str) -> str:
    """Lower-case, strip punctuation, collapse whitespace, remove articles."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = " ".join(text.split())
    return text


def _get_tokens(text: str) -> List[str]:
    return normalize_answer(text).split()


# ---------------------------------------------------------------------------
# Single-example scorers
# ---------------------------------------------------------------------------

def compute_exact_match(prediction: str, gold_aliases: List[str]) -> int:
    """
    Return 1 if the normalised prediction exactly matches any gold alias.

    Parameters
    ----------
    prediction   : model-generated answer string
    gold_aliases : list of acceptable answer strings from TriviaQA

    Returns
    -------
    int  1 if correct, 0 otherwise
    """
    pred_norm = normalize_answer(prediction)
    for alias in gold_aliases:
        if pred_norm == normalize_answer(alias):
            return 1
    return 0


def compute_f1(prediction: str, gold_aliases: List[str]) -> float:
    """
    Compute max token-level F1 across all gold aliases.

    Parameters
    ----------
    prediction   : model-generated answer string
    gold_aliases : list of acceptable answer strings from TriviaQA

    Returns
    -------
    float  best F1 score in [0, 1]
    """
    pred_tokens = _get_tokens(prediction)
    if not pred_tokens:
        return 0.0

    best_f1 = 0.0
    for alias in gold_aliases:
        gold_tokens = _get_tokens(alias)
        if not gold_tokens:
            continue
        common = set(pred_tokens) & set(gold_tokens)
        if not common:
            continue
        num_common = sum(
            min(pred_tokens.count(t), gold_tokens.count(t)) for t in common
        )
        precision = num_common / len(pred_tokens)
        recall = num_common / len(gold_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        best_f1 = max(best_f1, f1)
    return best_f1


def answer_in_generation(prediction: str, gold_aliases: List[str]) -> int:
    """
    Softer EM: return 1 if any gold alias appears as a substring of the
    normalised prediction.  Used as a fallback for generative models that
    output the answer inside a longer sentence.
    """
    pred_norm = normalize_answer(prediction)
    for alias in gold_aliases:
        if normalize_answer(alias) in pred_norm:
            return 1
    return 0


# ---------------------------------------------------------------------------
# Batch scorer
# ---------------------------------------------------------------------------

def score_batch(
    predictions: List[str],
    gold_aliases_list: List[List[str]],
    soft: bool = True,
) -> dict:
    """
    Score a list of (prediction, gold_aliases) pairs.

    Parameters
    ----------
    predictions       : list of model output strings
    gold_aliases_list : list of alias lists (one per example)
    soft              : if True, use answer_in_generation for EM (lenient)

    Returns
    -------
    dict with keys 'em', 'f1', 'em_per_example', 'f1_per_example'
    """
    assert len(predictions) == len(gold_aliases_list), (
        f"Length mismatch: {len(predictions)} predictions vs "
        f"{len(gold_aliases_list)} gold lists"
    )

    em_scores, f1_scores = [], []
    for pred, aliases in zip(predictions, gold_aliases_list):
        em_fn = answer_in_generation if soft else compute_exact_match
        em_scores.append(em_fn(pred, aliases))
        f1_scores.append(compute_f1(pred, aliases))

    return {
        "em": sum(em_scores) / len(em_scores),
        "f1": sum(f1_scores) / len(f1_scores),
        "em_per_example": em_scores,
        "f1_per_example": f1_scores,
    }
