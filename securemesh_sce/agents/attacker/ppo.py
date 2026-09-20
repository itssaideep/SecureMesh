# securemesh_sce/agents/attacker/ppo.py
"""Level 3: Adaptive PPO attacker.

A fully autonomous RL attacker that learns to exploit the system.
Uses the shared PPO core with attacker-specific action space.
"""

from __future__ import annotations

import numpy as np
from .scripted import BaseAgent
from ..common.ppo_core import PPOCore
from ...game.actions import N_ATTACKER_ACTIONS


class PPOAttacker(BaseAgent):
    """Level 3: Adaptive PPO attacker.

    Learns attack strategies purely from environment rewards.
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
        entropy_coef: float = 0.05,
        seed: int = 0,
    ):
        self.ppo = PPOCore(
            obs_dim=obs_dim,
            act_dim=N_ATTACKER_ACTIONS,
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

    def select_action(self, observation: np.ndarray) -> int:
        return self.ppo.select_action(observation)

    def action_probs(self, observation: np.ndarray) -> np.ndarray:
        return self.ppo.action_probs(observation)

    def update(self, transition: tuple):
        obs, action, reward, next_obs, done = transition
        self.ppo.store_transition(obs, action, reward, next_obs, done)

    def save(self, path: str):
        self.ppo.save(path)

    def load(self, path: str):
        self.ppo.load(path)
