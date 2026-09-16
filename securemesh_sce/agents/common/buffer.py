# securemesh_sce/agents/common/buffer.py
"""Trajectory and rollout buffers for RL agents.

Provides both on-policy (RolloutBuffer) for PPO and
replay-style (TrajectoryBuffer) for imitation learning.
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Dict, Optional


class RolloutBuffer:
    """On-policy rollout buffer for PPO-style algorithms.

    Stores a single episode trajectory and computes GAE-λ advantages.
    """

    def __init__(self, gamma: float = 0.99, lam: float = 0.95):
        self.gamma = gamma
        self.lam = lam
        self._obs: List[np.ndarray] = []
        self._actions: List[int] = []
        self._rewards: List[float] = []
        self._next_obs: List[np.ndarray] = []
        self._dones: List[bool] = []
        self._log_probs: List[float] = []

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        log_prob: float = 0.0,
    ):
        self._obs.append(np.asarray(obs, dtype=np.float32))
        self._actions.append(int(action))
        self._rewards.append(float(reward))
        self._next_obs.append(np.asarray(next_obs, dtype=np.float32))
        self._dones.append(bool(done))
        self._log_probs.append(float(log_prob))

    @property
    def size(self) -> int:
        return len(self._obs)

    def compute_advantages(
        self,
        values: np.ndarray,
        next_values: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute GAE-λ advantages and returns.

        Parameters
        ----------
        values : np.ndarray, shape (T,)
            Value estimates for each step.
        next_values : np.ndarray, shape (T,)
            Value estimates for next states.

        Returns
        -------
        advantages, returns : (np.ndarray, np.ndarray)
        """
        T = self.size
        rews = np.array(self._rewards, dtype=np.float32)
        dones = np.array(self._dones, dtype=np.float32)

        deltas = rews + self.gamma * next_values * (1.0 - dones) - values
        advantages = np.zeros(T, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(T)):
            gae = deltas[t] + self.gamma * self.lam * (1.0 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values
        return advantages, returns

    def get(self) -> Dict[str, np.ndarray]:
        """Return all buffer data as arrays."""
        return {
            "obs": np.stack(self._obs),
            "actions": np.array(self._actions, dtype=np.int64),
            "rewards": np.array(self._rewards, dtype=np.float32),
            "next_obs": np.stack(self._next_obs),
            "dones": np.array(self._dones, dtype=np.float32),
            "log_probs": np.array(self._log_probs, dtype=np.float32),
        }

    def clear(self):
        self._obs.clear()
        self._actions.clear()
        self._rewards.clear()
        self._next_obs.clear()
        self._dones.clear()
        self._log_probs.clear()


class TrajectoryBuffer:
    """Replay-style trajectory buffer for imitation learning (BC, GAIL).

    Stores expert demonstrations as (state, action) pairs.
    """

    def __init__(self, max_size: int = 100000):
        self.max_size = max_size
        self._obs: List[np.ndarray] = []
        self._actions: List[int] = []

    def add(self, obs: np.ndarray, action: int):
        if len(self._obs) >= self.max_size:
            # FIFO eviction
            self._obs.pop(0)
            self._actions.pop(0)
        self._obs.append(np.asarray(obs, dtype=np.float32))
        self._actions.append(int(action))

    def add_trajectory(self, obs_list: List[np.ndarray], action_list: List[int]):
        """Add a full trajectory."""
        for obs, act in zip(obs_list, action_list):
            self.add(obs, act)

    @property
    def size(self) -> int:
        return len(self._obs)

    def sample(self, batch_size: int, rng: np.random.RandomState) -> Tuple[np.ndarray, np.ndarray]:
        """Sample a random mini-batch."""
        n = self.size
        if n == 0:
            raise ValueError("Buffer is empty")
        idxs = rng.choice(n, size=min(batch_size, n), replace=False)
        obs = np.stack([self._obs[i] for i in idxs])
        acts = np.array([self._actions[i] for i in idxs], dtype=np.int64)
        return obs, acts

    def get_all(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return all stored demonstrations."""
        return np.stack(self._obs), np.array(self._actions, dtype=np.int64)

    def clear(self):
        self._obs.clear()
        self._actions.clear()
