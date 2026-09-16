# securemesh_sce/inference/calibration.py
"""Calibration metrics for Bayesian attacker-type inference.

Measures how well-calibrated the defender's posterior beliefs are
relative to the true attacker types. Includes:
- Brier score
- Log loss
- Expected Calibration Error (ECE)
- Calibration curves
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple

from ..game.actions import AttackerType, N_ATTACKER_TYPES


def brier_score(
    predicted_probs: np.ndarray,
    true_type_indices: np.ndarray,
) -> float:
    """Multi-class Brier score.

    BS = (1/N) Σ_t Σ_k (p_{t,k} - y_{t,k})²

    Parameters
    ----------
    predicted_probs : np.ndarray, shape (N, K)
        Predicted probability distributions at each step.
    true_type_indices : np.ndarray, shape (N,)
        True attacker type index at each step.

    Returns
    -------
    float
        Brier score (lower is better, 0 = perfect).
    """
    N = len(true_type_indices)
    if N == 0:
        return 0.0
    K = predicted_probs.shape[1]
    # One-hot encode true types
    one_hot = np.zeros((N, K), dtype=np.float64)
    one_hot[np.arange(N), true_type_indices] = 1.0
    return float(np.mean(np.sum((predicted_probs - one_hot) ** 2, axis=1)))


def log_loss(
    predicted_probs: np.ndarray,
    true_type_indices: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """Multi-class log loss (cross-entropy).

    LL = -(1/N) Σ_t log(p_{t, y_t})

    Parameters
    ----------
    predicted_probs : np.ndarray, shape (N, K)
        Predicted probability distributions at each step.
    true_type_indices : np.ndarray, shape (N,)
        True attacker type index at each step.

    Returns
    -------
    float
        Log loss (lower is better).
    """
    N = len(true_type_indices)
    if N == 0:
        return 0.0
    clipped = np.clip(predicted_probs, eps, 1.0 - eps)
    return float(-np.mean(np.log(clipped[np.arange(N), true_type_indices])))


def expected_calibration_error(
    predicted_probs: np.ndarray,
    true_type_indices: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error (ECE).

    Bins predictions by confidence and computes the weighted average
    of |accuracy - confidence| per bin.

    Parameters
    ----------
    predicted_probs : np.ndarray, shape (N, K)
        Predicted probability distributions at each step.
    true_type_indices : np.ndarray, shape (N,)
        True attacker type index at each step.
    n_bins : int
        Number of equal-width bins.

    Returns
    -------
    float
        ECE (lower is better).
    """
    N = len(true_type_indices)
    if N == 0:
        return 0.0

    # Use the argmax class and its probability
    predicted_classes = np.argmax(predicted_probs, axis=1)
    confidences = np.max(predicted_probs, axis=1)
    accuracies = (predicted_classes == true_type_indices).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bin_edges[i]) & (confidences <= bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = accuracies[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += (mask.sum() / N) * abs(bin_acc - bin_conf)

    return float(ece)


def calibration_curve(
    predicted_probs: np.ndarray,
    true_type_indices: np.ndarray,
    n_bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute calibration curve data.

    Returns
    -------
    bin_centres : np.ndarray
        Centre of each confidence bin.
    bin_accuracies : np.ndarray
        Accuracy in each bin (NaN if bin is empty).
    bin_counts : np.ndarray
        Number of predictions in each bin.
    """
    N = len(true_type_indices)
    predicted_classes = np.argmax(predicted_probs, axis=1)
    confidences = np.max(predicted_probs, axis=1)
    accuracies = (predicted_classes == true_type_indices).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_accuracies = np.full(n_bins, np.nan)
    bin_counts = np.zeros(n_bins, dtype=np.int64)

    for i in range(n_bins):
        mask = (confidences > bin_edges[i]) & (confidences <= bin_edges[i + 1])
        count = mask.sum()
        bin_counts[i] = count
        if count > 0:
            bin_accuracies[i] = accuracies[mask].mean()

    return bin_centres, bin_accuracies, bin_counts


class CalibrationTracker:
    """Accumulates predictions and ground truths for calibration analysis.

    Usage
    -----
    tracker = CalibrationTracker()
    for step in experiment:
        tracker.record(belief_vector, true_attacker_type_idx)
    print(tracker.brier_score())
    """

    def __init__(self):
        self._predictions: List[np.ndarray] = []
        self._truths: List[int] = []

    def record(self, predicted_probs: np.ndarray, true_type_idx: int):
        """Record one step of predictions and ground truth."""
        self._predictions.append(predicted_probs.copy())
        self._truths.append(int(true_type_idx))

    def clear(self):
        self._predictions.clear()
        self._truths.clear()

    @property
    def n_samples(self) -> int:
        return len(self._truths)

    def _arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self._predictions:
            return np.empty((0, N_ATTACKER_TYPES)), np.empty(0, dtype=np.int64)
        preds = np.array(self._predictions, dtype=np.float64)
        truths = np.array(self._truths, dtype=np.int64)
        return preds, truths

    def brier_score(self) -> float:
        preds, truths = self._arrays()
        return brier_score(preds, truths)

    def log_loss(self) -> float:
        preds, truths = self._arrays()
        return log_loss(preds, truths)

    def ece(self, n_bins: int = 10) -> float:
        preds, truths = self._arrays()
        return expected_calibration_error(preds, truths, n_bins)

    def calibration_curve(self, n_bins: int = 10):
        preds, truths = self._arrays()
        return calibration_curve(preds, truths, n_bins)

    def accuracy(self) -> float:
        preds, truths = self._arrays()
        if len(truths) == 0:
            return 0.0
        return float((np.argmax(preds, axis=1) == truths).mean())

    def summary(self) -> dict:
        """Return a dict of all calibration metrics."""
        return {
            "accuracy": self.accuracy(),
            "brier_score": self.brier_score(),
            "log_loss": self.log_loss(),
            "ece": self.ece(),
            "n_samples": self.n_samples,
        }
