# securemesh_sce/agents/defender/bayesian_rl.py
"""Defender 4: Bayesian RL defender (PPO with augmented state [x_t, b_t, h_t, r_t]).

Directly conditions the policy on the Bayesian posterior belief b_t over hidden
attacker types, enabling proactive adaptation to the inferred adversary profile.
"""

from __future__ import annotations

import numpy as np
from ..attacker.scripted import BaseAgent
from ..common.ppo_core import PPOCore
from ...game.actions import N_DEFENDER_ACTIONS


class BayesianRLDefender(BaseAgent):
    """Defender 4: Bayesian PPO defender.

    Observes the full augmented state vector s_t = [x_t, b_t, h_t, r_t]
    incorporating exact Bayesian posterior belief distributions.
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
    ):
        self.obs_dim = obs_dim
        self.ppo = PPOCore(
            obs_dim=obs_dim,
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
