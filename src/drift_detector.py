"""
src/drift_detector.py
Three drift monitoring approaches for LLM output streams.

ACC-MON  — Accuracy-Based Monitor         (requires labels)
JSD-MON  — Jensen-Shannon Divergence      (label-free, uses output logits)
MMD-MON  — Maximum Mean Discrepancy       (label-free, uses input embeddings)

Theory references:
  JSD  : Lin (1991) IEEE Transactions on Information Theory 37(1):145-151
  MMD  : Gretton et al. (2012) JMLR 13:723-773
  Drift: Gama et al. (2014) ACM CSUR 46(4)
"""

from __future__ import annotations

import numpy as np
from collections import deque
from typing import List, Optional, Tuple


# ===========================================================================
# Utility functions
# ===========================================================================

def _softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Numerically stable softmax with temperature scaling."""
    logits = logits / temperature
    logits = logits - logits.max()
    exp = np.exp(logits)
    return exp / exp.sum()


def _jsd(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
    """Jensen-Shannon Divergence between two probability vectors. Result in [0, ln2]."""
    p = p + eps; p /= p.sum()
    q = q + eps; q /= q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    return float(0.5 * kl_pm + 0.5 * kl_qm)


def _rbf_kernel_matrix(X: np.ndarray, Y: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Compute RBF kernel k(x,y) = exp(-||x-y||^2 / (2*sigma^2)) for all pairs."""
    XX = np.sum(X ** 2, axis=1, keepdims=True)
    YY = np.sum(Y ** 2, axis=1, keepdims=True)
    dists = XX + YY.T - 2.0 * X @ Y.T
    return np.exp(-dists / (2.0 * sigma ** 2))


def _mmd_squared(X: np.ndarray, Y: np.ndarray, sigma: float = 1.0) -> float:
    """
    Unbiased MMD^2 estimator with RBF kernel.

    Parameters
    ----------
    X : (m, d) reference embeddings
    Y : (n, d) test-window embeddings
    """
    m, n = len(X), len(Y)
    kxx = _rbf_kernel_matrix(X, X, sigma)
    kyy = _rbf_kernel_matrix(Y, Y, sigma)
    kxy = _rbf_kernel_matrix(X, Y, sigma)

    # Unbiased: exclude diagonal for within-set terms
    np.fill_diagonal(kxx, 0.0)
    np.fill_diagonal(kyy, 0.0)

    term1 = kxx.sum() / (m * (m - 1)) if m > 1 else 0.0
    term2 = kyy.sum() / (n * (n - 1)) if n > 1 else 0.0
    term3 = kxy.sum() / (m * n)

    return float(term1 + term2 - 2.0 * term3)


# ===========================================================================
# Threshold calibration
# ===========================================================================

def calibrate_threshold(
    reference_statistics: List[float],
    percentile: float = 95.0,
) -> float:
    """
    Set the detection threshold at a given percentile of reference statistics.
    This calibration is performed ONCE on the reference window before
    inspecting any test-stream results.

    Parameters
    ----------
    reference_statistics : list of statistic values computed on non-overlapping
                           reference windows (e.g. 10 windows of size w drawn
                           from the 500-sample reference set)
    percentile           : float in (0, 100); default 95 → nominal FPR ≈ 5 %

    Returns
    -------
    float  threshold value
    """
    threshold = float(np.percentile(reference_statistics, percentile))
    return threshold


# ===========================================================================
# Monitor 1 — Accuracy-Based Monitor (ACC-MON)
# ===========================================================================

class AccuracyMonitor:
    """
    Sliding-window accuracy monitor.
    Flags drift when rolling accuracy drops > delta below reference accuracy.
    Requires ground-truth labels — cannot be used in a fully unsupervised setting.
    """

    def __init__(
        self,
        window_size: int = 150,
        threshold_drop: float = 0.15,
    ):
        self.window_size = window_size
        self.threshold_drop = threshold_drop
        self.reference_accuracy: Optional[float] = None
        self._window: deque = deque(maxlen=window_size)
        self.statistic_history: List[float] = []
        self.flag_history: List[bool] = []

    def set_reference_accuracy(self, accuracy: float) -> None:
        self.reference_accuracy = accuracy

    def update(self, is_correct: bool) -> Tuple[float, bool]:
        """
        Update the monitor with one new correctness signal.

        Returns
        -------
        (statistic, drift_flagged) tuple
        statistic = how much accuracy has dropped from reference (positive = drop)
        """
        self._window.append(float(is_correct))
        if len(self._window) < self.window_size:
            self.statistic_history.append(0.0)
            self.flag_history.append(False)
            return 0.0, False

        rolling_acc = float(np.mean(self._window))
        drop = (self.reference_accuracy - rolling_acc) if self.reference_accuracy is not None else 0.0
        flagged = drop > self.threshold_drop

        self.statistic_history.append(drop)
        self.flag_history.append(flagged)
        return drop, flagged

    def calibrate_from_reference(
        self,
        reference_correctness: List[int],
        percentile: float = 95.0,
    ) -> None:
        """Set reference accuracy and calibrate threshold from reference window."""
        self.reference_accuracy = float(np.mean(reference_correctness))
        # Compute rolling drops over non-overlapping reference sub-windows
        ref_arr = np.array(reference_correctness, dtype=float)
        n_windows = len(ref_arr) // self.window_size
        drops = []
        for i in range(n_windows):
            w = ref_arr[i * self.window_size : (i + 1) * self.window_size]
            drops.append(max(0.0, self.reference_accuracy - w.mean()))
        if drops:
            self.threshold_drop = calibrate_threshold(drops, percentile)


# ===========================================================================
# Monitor 2 — JSD-Based Monitor (JSD-MON)
# ===========================================================================

class JSDMonitor:
    """
    Jensen-Shannon Divergence monitor over output probability distributions.
    At each step receives a logit vector; flags drift when the averaged
    distribution over the sliding window diverges from the reference average.
    Label-free — no ground truth required.
    """

    def __init__(
        self,
        window_size: int = 150,
        threshold: float = 0.05,
        temperature: float = 1.0,
    ):
        self.window_size = window_size
        self.threshold = threshold
        self.temperature = temperature

        self.reference_distribution: Optional[np.ndarray] = None
        self._prob_window: deque = deque(maxlen=window_size)
        self.statistic_history: List[float] = []
        self.flag_history: List[bool] = []

    def set_reference_distribution(self, avg_probs: np.ndarray) -> None:
        self.reference_distribution = avg_probs.copy()

    def update(self, logits: np.ndarray) -> Tuple[float, bool]:
        """
        Parameters
        ----------
        logits : (vocab_size,) raw output logit vector from the LLM

        Returns
        -------
        (jsd_value, drift_flagged)
        """
        probs = _softmax(logits, self.temperature)
        self._prob_window.append(probs)

        if len(self._prob_window) < self.window_size:
            self.statistic_history.append(0.0)
            self.flag_history.append(False)
            return 0.0, False

        window_avg = np.mean(np.stack(list(self._prob_window)), axis=0)
        jsd_val = _jsd(self.reference_distribution, window_avg) if self.reference_distribution is not None else 0.0
        flagged = jsd_val > self.threshold

        self.statistic_history.append(jsd_val)
        self.flag_history.append(flagged)
        return jsd_val, flagged

    def calibrate_from_reference(
        self,
        reference_logits: List[np.ndarray],
        percentile: float = 95.0,
    ) -> None:
        """
        Compute reference distribution and calibrate threshold.

        Parameters
        ----------
        reference_logits : list of (vocab_size,) logit vectors from reference window
        """
        ref_probs = [_softmax(l, self.temperature) for l in reference_logits]
        self.reference_distribution = np.mean(np.stack(ref_probs), axis=0)

        # Compute JSD over non-overlapping reference sub-windows
        n_windows = len(ref_probs) // self.window_size
        jsds = []
        for i in range(n_windows):
            w_probs = ref_probs[i * self.window_size : (i + 1) * self.window_size]
            w_avg = np.mean(np.stack(w_probs), axis=0)
            jsds.append(_jsd(self.reference_distribution, w_avg))
        if jsds:
            self.threshold = calibrate_threshold(jsds, percentile)


# ===========================================================================
# Monitor 3 — MMD-Based Monitor (MMD-MON)
# ===========================================================================

class MMDMonitor:
    """
    Maximum Mean Discrepancy monitor over sentence embeddings.
    Receives a 384-dim Sentence-BERT embedding for each input prompt;
    flags drift when the MMD^2 between the reference embedding set and
    the current sliding window exceeds the calibrated threshold.
    Label-free — no ground truth or model internals required.
    """

    def __init__(
        self,
        window_size: int = 150,
        threshold: float = 0.01,
        kernel_sigma: float = 1.0,
    ):
        self.window_size = window_size
        self.threshold = threshold
        self.kernel_sigma = kernel_sigma

        self.reference_embeddings: Optional[np.ndarray] = None
        self._emb_window: deque = deque(maxlen=window_size)
        self.statistic_history: List[float] = []
        self.flag_history: List[bool] = []

    def set_reference_embeddings(self, embeddings: np.ndarray) -> None:
        """embeddings : (n_ref, d) array of reference sentence embeddings."""
        self.reference_embeddings = embeddings.copy()

    def update(self, embedding: np.ndarray) -> Tuple[float, bool]:
        """
        Parameters
        ----------
        embedding : (d,) sentence embedding for the current input prompt

        Returns
        -------
        (mmd_squared, drift_flagged)
        """
        self._emb_window.append(embedding)

        if len(self._emb_window) < self.window_size:
            self.statistic_history.append(0.0)
            self.flag_history.append(False)
            return 0.0, False

        window_arr = np.stack(list(self._emb_window))
        mmd2 = _mmd_squared(self.reference_embeddings, window_arr, self.kernel_sigma) \
               if self.reference_embeddings is not None else 0.0
        flagged = mmd2 > self.threshold

        self.statistic_history.append(mmd2)
        self.flag_history.append(flagged)
        return mmd2, flagged

    def calibrate_from_reference(
        self,
        reference_embeddings: np.ndarray,
        percentile: float = 95.0,
    ) -> None:
        """
        Parameters
        ----------
        reference_embeddings : (n_ref, d) array — the full reference window
        """
        self.reference_embeddings = reference_embeddings.copy()
        n_ref = len(reference_embeddings)
        n_windows = n_ref // self.window_size
        mmds = []
        for i in range(n_windows):
            w = reference_embeddings[i * self.window_size : (i + 1) * self.window_size]
            # Reference for calibration sub-window is remaining reference samples
            ref_rest = np.delete(
                reference_embeddings,
                range(i * self.window_size, (i + 1) * self.window_size),
                axis=0,
            )
            mmds.append(_mmd_squared(ref_rest, w, self.kernel_sigma))
        if mmds:
            self.threshold = calibrate_threshold(mmds, percentile)
