# tests/test_inference.py
"""Unit tests for Bayesian posterior updates and calibration metrics."""

import pytest
import numpy as np

from securemesh_sce.inference.bayesian import BayesianInference
from securemesh_sce.inference.calibration import CalibrationTracker
from securemesh_sce.game.actions import AttackerAction, AttackerType


def test_bayesian_update_normalisation():
    bayes = BayesianInference()
    assert np.isclose(bayes.belief.sum(), 1.0)

    # Observe a reconnaissance scan
    bayes.update(AttackerAction.RECON_SCAN)
    assert np.isclose(bayes.belief.sum(), 1.0)
    assert all(p >= 0.0 for p in bayes.belief)

    # Observe multiple scans
    for _ in range(5):
        bayes.update(AttackerAction.RECON_SCAN)
    assert np.isclose(bayes.belief.sum(), 1.0)
    assert isinstance(bayes.most_likely_type, AttackerType)


def test_calibration_tracker():
    tracker = CalibrationTracker()
    # Predict high confidence for class 0
    tracker.record(np.array([0.9, 0.05, 0.03, 0.02]), true_type_idx=0)
    tracker.record(np.array([0.85, 0.05, 0.05, 0.05]), true_type_idx=0)
    tracker.record(np.array([0.1, 0.8, 0.05, 0.05]), true_type_idx=1)

    summary = tracker.summary()
    assert "accuracy" in summary
    assert "brier_score" in summary
    assert "ece" in summary
    assert summary["accuracy"] == 1.0
    assert summary["brier_score"] < 0.2
    assert summary["ece"] >= 0.0
