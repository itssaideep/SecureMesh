# securemesh_sce/agents/defender/ml.py
"""Defender 2: Machine Learning (Random Forest) defender.

Uses a simple Random Forest classifier trained on labelled attack data
to classify observations and select defence actions.
"""

from __future__ import annotations

import json
import numpy as np
from typing import Optional, List

from ..attacker.scripted import BaseAgent
from ...game.actions import DefenderAction, N_DEFENDER_ACTIONS


class DecisionTree:
    """Minimal NumPy-only decision tree for classification."""

    def __init__(self, max_depth: int = 5, min_samples: int = 5, seed: int = 0):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.rng = np.random.RandomState(seed)
        self._tree = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self._n_classes = int(y.max()) + 1
        self._tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        n_samples = len(y)
        if (depth >= self.max_depth or n_samples <= self.min_samples
                or len(np.unique(y)) == 1):
            # Leaf node: majority class
            counts = np.bincount(y, minlength=self._n_classes)
            return {"leaf": True, "class": int(np.argmax(counts)), "probs": counts / counts.sum()}

        # Random feature subset (like Random Forest)
        n_features = X.shape[1]
        n_try = max(1, int(np.sqrt(n_features)))
        feature_idxs = self.rng.choice(n_features, size=n_try, replace=False)

        best_feat, best_thresh, best_gain = None, None, -np.inf

        for f in feature_idxs:
            thresholds = np.unique(X[:, f])
            if len(thresholds) > 20:
                thresholds = np.percentile(X[:, f], np.linspace(10, 90, 10))
            for t in thresholds:
                left = y[X[:, f] <= t]
                right = y[X[:, f] > t]
                if len(left) == 0 or len(right) == 0:
                    continue
                gain = self._info_gain(y, left, right)
                if gain > best_gain:
                    best_gain = gain
                    best_feat = f
                    best_thresh = t

        if best_feat is None:
            counts = np.bincount(y, minlength=self._n_classes)
            return {"leaf": True, "class": int(np.argmax(counts)), "probs": counts / counts.sum()}

        left_mask = X[:, best_feat] <= best_thresh
        return {
            "leaf": False,
            "feature": int(best_feat),
            "threshold": float(best_thresh),
            "left": self._build_tree(X[left_mask], y[left_mask], depth + 1),
            "right": self._build_tree(X[~left_mask], y[~left_mask], depth + 1),
        }

    def _info_gain(self, parent, left, right):
        def entropy(arr):
            _, counts = np.unique(arr, return_counts=True)
            probs = counts / len(arr)
            return -np.sum(probs * np.log2(probs + 1e-12))

        n = len(parent)
        return entropy(parent) - (len(left) / n * entropy(left) + len(right) / n * entropy(right))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._predict_one(x, self._tree) for x in X])

    def _predict_one(self, x, node):
        if node["leaf"]:
            return node["class"]
        if x[node["feature"]] <= node["threshold"]:
            return self._predict_one(x, node["left"])
        return self._predict_one(x, node["right"])


class RandomForestDefender(BaseAgent):
    """Defender 2: Random Forest-based ML defender.

    Trained on labelled (observation, optimal_action) data from
    simulated environments. Serves as a supervised learning baseline.
    """

    def __init__(
        self,
        n_trees: int = 10,
        max_depth: int = 8,
        seed: int = 0,
    ):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.rng = np.random.RandomState(seed)
        self.trees: List[DecisionTree] = []
        self._is_trained = False

    def train(self, X: np.ndarray, y: np.ndarray):
        """Train the random forest from labelled data.

        Parameters
        ----------
        X : np.ndarray, shape (N, obs_dim)
            Observation data.
        y : np.ndarray, shape (N,)
            Action labels (indices into DefenderAction).
        """
        self.trees = []
        N = len(X)
        for i in range(self.n_trees):
            tree = DecisionTree(max_depth=self.max_depth, seed=self.rng.randint(1e6))
            # Bootstrap sample
            idxs = self.rng.choice(N, size=N, replace=True)
            tree.fit(X[idxs], y[idxs])
            self.trees.append(tree)
        self._is_trained = True

    def select_action(self, observation: np.ndarray) -> int:
        if not self._is_trained:
            # Fallback: monitor
            return list(DefenderAction).index(DefenderAction.MONITOR)

        obs = np.asarray(observation, dtype=np.float32).reshape(1, -1)
        votes = np.array([tree.predict(obs)[0] for tree in self.trees])
        return int(np.bincount(votes, minlength=N_DEFENDER_ACTIONS).argmax())
