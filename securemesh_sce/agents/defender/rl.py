# securemesh_sce/agents/defender/rl.py
"""Defender 3: RL Defender without Bayesian belief.

Uses PPO to learn a defence policy observing only the physical system
state x_t (without b_t, h_t, or r_t), serving as the un-augmented RL baseline.
"""

from __future__ import annotations

import numpy as np
from typing import Optional
from ..attacker.scripted import BaseAgent
from ..common.ppo_core import PPOCore
from ...game.actions import N_DEFENDER_ACTIONS, N_ATTACKER_TYPES
from ...game.state import AttackHistory, RiskState


class RLDefender(BaseAgent):
    """Defender 3: PPO defender without Bayesian belief.

    Observes only the raw system state x_t. If given the composite
    defender observation [x_t, b_t, h_t, r_t], it automatically strips
    out belief, history, and risk features to operate strictly under
    un-augmented system observability.
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_sizes: tuple = (64, 64),
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_eps: float = 0.2,
        pi_lr: float = 3e-4,
        vf_lr: float = 1e-3,
        train_epochs: int = 4,
        batch_size: int = 32,
        entropy_coef: float = 0.02,
        seed: int = 0,
        extra_dim: int = N_ATTACKER_TYPES + len(AttackHistory.FEATURE_NAMES) + 4,
    ):
        self.extra_dim = extra_dim
        # If obs_dim is full composite dimension, system_obs_dim = obs_dim - extra_dim
        self.system_obs_dim = max(1, obs_dim - extra_dim if obs_dim > extra_dim else obs_dim)

        self.ppo = PPOCore(
            obs_dim=self.system_obs_dim,
            act_dim=N_DEFENDER_ACTIONS,
            hidden_sizes=hidden_sizes,
            gamma=gamma,
            lam=lam,
            clip_eps=clip_eps,
            pi_lr=pi_lr,
            vf_lr=vf_lr,
            train_epochs=train_epochs,
            batch_size=batch_size,
            entropy_coef=entropy_coef,
            seed=seed,
        )

    def _extract_system_obs(self, observation: np.ndarray) -> np.ndarray:
        """Strip belief, history, and risk if present."""
        obs = np.asarray(observation, dtype=np.float32).ravel()
        if len(obs) > self.extra_dim and len(obs) != self.system_obs_dim:
            return obs[:self.system_obs_dim]
        return obs[:self.system_obs_dim]

    def select_action(self, observation: np.ndarray) -> int:
        sys_obs = self._extract_system_obs(observation)
        return self.ppo.select_action(sys_obs)

    def action_probs(self, observation: np.ndarray) -> np.ndarray:
        sys_obs = self._extract_system_obs(observation)
        return self.ppo.action_probs(sys_obs)

    def update(self, transition: tuple):
        obs, action, reward, next_obs, done = transition
        sys_obs = self._extract_system_obs(obs)
        next_sys_obs = self._extract_system_obs(next_obs)
        self.ppo.store_transition(sys_obs, action, reward, next_sys_obs, done)

    def save(self, path: str):
        self.ppo.save(path)

    def load(self, path: str):
        self.ppo.load(path)
