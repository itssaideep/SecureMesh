# securemesh_testbed/agents/defender/ml_defender.py
"""Supervised ML-based defender using a Random Forest classifier.

The defender is trained on (observation, best_action) pairs collected
from a scripted "oracle" policy during a warm-up phase.  After training,
it predicts the best defensive action from the observation vector.

This serves as the middle baseline between the static rule-based
defender and the fully adaptive RL defender.
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple

from ..base_agent import BaseAgent
from ...game.actions import DefenderAction


class MLDefender(BaseAgent):
    """Random-Forest defender (scikit-learn).

    Training data is accumulated via ``record(obs, action)`` calls
    during a warm-up period.  Call ``fit()`` to train the classifier
    before switching to ``select_action`` for inference.
    """

    def __init__(self, n_actions: int | None = None, seed: int = 0):
        self.n_actions = n_actions or len(DefenderAction)
        self.seed = seed
        self._X: List[np.ndarray] = []
        self._y: List[int] = []
        self._model = None
        self._fitted = False

    # --- data collection (warm-up) ------------------------------------------
    def record(self, observation: np.ndarray, action: int):
        """Store a labelled training example."""
        self._X.append(np.asarray(observation, dtype=np.float32).flatten())
        self._y.append(int(action))

    def fit(self):
        """Train the Random Forest on collected data."""
        from sklearn.ensemble import RandomForestClassifier

        if len(self._X) < 10:
            # not enough data — fall back to NOOP
            return
        X = np.stack(self._X)
        y = np.array(self._y)
        self._model = RandomForestClassifier(
            n_estimators=50,
            max_depth=8,
            random_state=self.seed,
        )
        self._model.fit(X, y)
        self._fitted = True

    # --- BaseAgent interface -------------------------------------------------
    def select_action(self, observation) -> int:
        if not self._fitted or self._model is None:
            return DefenderAction.NOOP.value
        obs = np.asarray(observation, dtype=np.float32).reshape(1, -1)
        return int(self._model.predict(obs)[0])

    def update(self, transition: tuple):
        """Optionally collect new data for online retraining."""
        obs, act, _rew, _nobs, _done = transition
        self.record(obs, act)
