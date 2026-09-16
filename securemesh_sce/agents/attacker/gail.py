# securemesh_sce/agents/attacker/gail.py
"""Level 2: Generative Adversarial Imitation Learning (GAIL) attacker.

GAIL learns a policy that produces trajectories indistinguishable from
expert demonstrations, using a discriminator to provide reward signal.
"""

from __future__ import annotations

import numpy as np
from typing import List

from .scripted import BaseAgent
from ..common.mlp import NumpyMLP
from ..common.ppo_core import PPOCore
from ..common.buffer import TrajectoryBuffer
from ...game.actions import N_ATTACKER_ACTIONS


class GAILAttacker(BaseAgent):
    """Level 2: GAIL attacker.

    Uses a discriminator D(s,a) to distinguish expert vs policy trajectories.
    The policy is optimised with PPO using -log(D(s,a)) as reward.
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_sizes: tuple = (64, 64),
        disc_lr: float = 1e-3,
        pi_lr: float = 3e-4,
        vf_lr: float = 1e-3,
        batch_size: int = 64,
        seed: int = 0,
    ):
        self.obs_dim = obs_dim
        self.act_dim = N_ATTACKER_ACTIONS
        self.batch_size = batch_size
        self.rng = np.random.RandomState(seed)

        # Discriminator: D(s,a) → [0,1] (1 = expert)
        disc_input_dim = obs_dim + self.act_dim  # one-hot action concatenated
        disc_dims = [disc_input_dim] + list(hidden_sizes) + [1]
        self.discriminator = NumpyMLP(disc_dims, activation="tanh", seed=seed)
        self.disc_lr = disc_lr

        # Generator policy (PPO)
        self.ppo = PPOCore(
            obs_dim=obs_dim,
            act_dim=self.act_dim,
            hidden_sizes=hidden_sizes,
            pi_lr=pi_lr,
            vf_lr=vf_lr,
            seed=seed,
        )

        # Expert buffer
        self.expert_buffer = TrajectoryBuffer()

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))

    def _make_sa_input(self, obs: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Concatenate obs with one-hot actions."""
        B = len(obs)
        one_hot = np.zeros((B, self.act_dim), dtype=np.float32)
        one_hot[np.arange(B), actions] = 1.0
        return np.concatenate([obs, one_hot], axis=1)

    def select_action(self, observation: np.ndarray) -> int:
        return self.ppo.select_action(observation)

    def add_demonstrations(self, obs_list: List[np.ndarray], action_list: List[int]):
        self.expert_buffer.add_trajectory(obs_list, action_list)

    def discriminator_reward(self, obs: np.ndarray, action: int) -> float:
        """Compute GAIL reward: -log(1 - D(s,a))."""
        sa = self._make_sa_input(obs.reshape(1, -1), np.array([action]))
        logit, _ = self.discriminator.forward(sa)
        d = self._sigmoid(logit[0, 0])
        return float(-np.log(1.0 - d + 1e-8))

    def update_discriminator(self, policy_obs: np.ndarray, policy_acts: np.ndarray):
        """Train discriminator to distinguish expert from policy."""
        expert_obs, expert_acts = self.expert_buffer.sample(
            min(len(policy_obs), self.expert_buffer.size), self.rng
        )

        # Expert labels = 1, policy labels = 0
        expert_sa = self._make_sa_input(expert_obs, expert_acts)
        policy_sa = self._make_sa_input(policy_obs, policy_acts)

        all_sa = np.concatenate([expert_sa, policy_sa], axis=0)
        labels = np.concatenate([
            np.ones(len(expert_sa)),
            np.zeros(len(policy_sa)),
        ])

        # Forward
        logits, hiddens = self.discriminator.forward(all_sa)
        preds = self._sigmoid(logits.squeeze(-1))

        # BCE gradient: d/dz = pred - label
        d_output = (preds - labels)[:, None] / len(all_sa)

        dw, db = self.discriminator.backward(all_sa, hiddens, d_output)
        self.discriminator.apply_gradients(dw, db, self.disc_lr, ascend=False)

    def update(self, transition: tuple):
        """Store transition with GAIL reward and train when episode ends."""
        obs, action, _, next_obs, done = transition
        gail_reward = self.discriminator_reward(np.asarray(obs), action)
        self.ppo.store_transition(obs, action, gail_reward, next_obs, done)

    def save(self, path: str):
        self.ppo.save(path)

    def load(self, path: str):
        self.ppo.load(path)
