# securemesh_sce/agents/attacker/bc.py
"""Level 1: Behavioral Cloning attacker.

Learns to imitate the scripted attacker from demonstration data.
Used to study how well an imitator can replicate known attack patterns.
"""

from __future__ import annotations

import numpy as np
from typing import List

from .scripted import BaseAgent
from ..common.mlp import NumpyMLP
from ..common.buffer import TrajectoryBuffer
from ...game.actions import N_ATTACKER_ACTIONS


class BCAttacker(BaseAgent):
    """Level 1: Behavioral Cloning attacker.

    Supervised learning from expert demonstrations.
    Policy: π(a|s) = softmax(MLP(s))
    Loss: cross-entropy between π and expert actions.
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_sizes: tuple = (64, 64),
        lr: float = 1e-3,
        batch_size: int = 64,
        seed: int = 0,
    ):
        self.obs_dim = obs_dim
        self.act_dim = N_ATTACKER_ACTIONS
        self.lr = lr
        self.batch_size = batch_size
        self.rng = np.random.RandomState(seed)

        dims = [obs_dim] + list(hidden_sizes) + [self.act_dim]
        self.net = NumpyMLP(dims, activation="tanh", seed=seed)
        self.demo_buffer = TrajectoryBuffer()

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - logits.max(axis=-1, keepdims=True)
        e = np.exp(shifted)
        return e / e.sum(axis=-1, keepdims=True)

    def select_action(self, observation: np.ndarray) -> int:
        obs = np.asarray(observation, dtype=np.float32).reshape(1, -1)
        logits, _ = self.net.forward(obs)
        probs = self._softmax(logits[0])
        return int(self.rng.choice(self.act_dim, p=probs))

    def add_demonstrations(self, obs_list: List[np.ndarray], action_list: List[int]):
        """Add expert demonstrations to the buffer."""
        self.demo_buffer.add_trajectory(obs_list, action_list)

    def train(self, n_epochs: int = 50) -> List[float]:
        """Train BC from demonstrations. Returns per-epoch losses."""
        if self.demo_buffer.size == 0:
            raise ValueError("No demonstrations added. Call add_demonstrations() first.")

        all_obs, all_acts = self.demo_buffer.get_all()
        N = len(all_obs)
        losses = []

        for epoch in range(n_epochs):
            idxs = self.rng.permutation(N)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, N, self.batch_size):
                mb = idxs[start:start + self.batch_size]
                obs_mb = all_obs[mb]
                acts_mb = all_acts[mb]
                B = len(mb)

                # Forward
                logits, hiddens = self.net.forward(obs_mb)
                probs = self._softmax(logits)
                log_probs = np.log(probs[np.arange(B), acts_mb] + 1e-8)
                loss = -log_probs.mean()
                epoch_loss += loss
                n_batches += 1

                # Gradient of cross-entropy w.r.t. logits
                d_logits = probs.copy()
                d_logits[np.arange(B), acts_mb] -= 1.0
                d_logits /= B

                # Backprop and update
                dw, db = self.net.backward(obs_mb, hiddens, d_logits)
                self.net.apply_gradients(dw, db, self.lr, ascend=False)

            losses.append(epoch_loss / max(n_batches, 1))
        return losses

    def save(self, path: str):
        self.net.save(path)

    def load(self, path: str):
        self.net = NumpyMLP.load(path)
