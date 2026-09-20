# securemesh_sce/agents/common/ppo_core.py
"""Shared PPO implementation for both attacker and defender agents.

Uses the modular NumpyMLP. Supports:
- Clipped surrogate objective
- Generalised Advantage Estimation (GAE-λ)
- Entropy bonus for exploration
- Mini-batch training
"""

from __future__ import annotations

import json
import numpy as np
from typing import Optional

from .mlp import NumpyMLP
from .buffer import RolloutBuffer


class PPOCore:
    """NumPy-only Proximal Policy Optimisation core.

    Used by both attacker (PPOAttacker) and defender (PPODefender)
    via composition.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: tuple = (64, 64),
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_eps: float = 0.2,
        pi_lr: float = 1e-3,
        vf_lr: float = 1e-3,
        train_epochs: int = 4,
        batch_size: int = 32,
        entropy_coef: float = 0.01,
        activation: str = "tanh",
        seed: int = 0,
    ):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.rng = np.random.RandomState(seed)

        # Policy network
        pi_dims = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.policy_net = NumpyMLP(pi_dims, activation=activation, seed=seed)

        # Value network
        vf_dims = [obs_dim] + list(hidden_sizes) + [1]
        self.value_net = NumpyMLP(vf_dims, activation=activation, seed=seed + 1)

        # Hyperparameters
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.pi_lr = pi_lr
        self.vf_lr = vf_lr
        self.train_epochs = train_epochs
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef

        # Rollout buffer
        self.buffer = RolloutBuffer(gamma=gamma, lam=lam)

    # ---- softmax ----

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - logits.max(axis=-1, keepdims=True)
        e = np.exp(shifted)
        return e / e.sum(axis=-1, keepdims=True)

    # ---- forward ----

    def policy_logits(self, obs: np.ndarray) -> np.ndarray:
        """(B, obs_dim) -> (B, act_dim) logits."""
        out, _ = self.policy_net.forward(obs)
        return out

    def value(self, obs: np.ndarray) -> np.ndarray:
        """(B, obs_dim) -> (B,) scalar values."""
        out, _ = self.value_net.forward(obs)
        return out.squeeze(-1)

    # ---- action selection ----

    def select_action(self, obs: np.ndarray) -> int:
        """Sample an action from the policy."""
        obs_2d = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        logits = self.policy_logits(obs_2d)[0]
        probs = self._softmax(logits)
        return int(self.rng.choice(self.act_dim, p=probs))

    def action_probs(self, obs: np.ndarray) -> np.ndarray:
        """Return the full probability vector."""
        obs_2d = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        return self._softmax(self.policy_logits(obs_2d)[0])

    # ---- update ----

    def store_transition(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ):
        """Store a transition. Train when episode ends."""
        log_prob = float(np.log(self.action_probs(obs)[action] + 1e-8))
        self.buffer.add(obs, action, reward, next_obs, done, log_prob)
        if done:
            self._train()

    def _train(self):
        """Run PPO update on the collected episode."""
        if self.buffer.size == 0:
            return

        data = self.buffer.get()
        obs = data["obs"]
        acts = data["actions"]
        old_log_probs = data["log_probs"]
        T = len(obs)

        # Compute values and advantages
        values = self.value(obs)
        next_values = self.value(data["next_obs"])
        advantages, returns = self.buffer.compute_advantages(values, next_values)

        # Normalise advantages
        adv_std = advantages.std()
        if adv_std > 1e-8:
            advantages = (advantages - advantages.mean()) / adv_std

        # PPO epochs
        idxs = np.arange(T)
        for _ in range(self.train_epochs):
            self.rng.shuffle(idxs)
            for start in range(0, T, self.batch_size):
                mb = idxs[start:start + self.batch_size]
                self._update_policy(obs[mb], acts[mb], advantages[mb], old_log_probs[mb])
                self._update_value(obs[mb], returns[mb])

        self.buffer.clear()

    def _update_policy(self, obs, acts, adv, old_log_pi):
        """Single mini-batch policy gradient step."""
        B = len(obs)
        logits, hiddens = self.policy_net.forward(obs)
        probs = self._softmax(logits)
        log_pi = np.log(probs[np.arange(B), acts] + 1e-8)

        # Ratio and clipped surrogate
        ratio = np.exp(log_pi - old_log_pi)
        clipped = np.clip(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)

        # Gradient of log π w.r.t. logits: ∂ log π_a / ∂ logit_j = 1(j==a) - π_j
        d_logits = -probs.copy()
        d_logits[np.arange(B), acts] += 1.0

        # Use unclipped where it gives smaller surrogate
        use_unclipped = (ratio * adv <= clipped * adv).astype(np.float32)
        scale = (use_unclipped * adv)[:, None]
        d_logits = d_logits * scale / B

        # Entropy bonus gradient: ∂ H / ∂ logit_j = π_j * (log π_j + 1 - Σ π_k log π_k)
        # Simplified: add small uniform push
        entropy_grad = -probs * (np.log(probs + 1e-8) + 1.0)
        d_logits += self.entropy_coef * entropy_grad / B

        # Backprop
        dw, db = self.policy_net.backward(obs, hiddens, d_logits)
        self.policy_net.apply_gradients(dw, db, self.pi_lr, ascend=True)

    def _update_value(self, obs, returns):
        """Single mini-batch value function step."""
        B = len(obs)
        v_pred, hiddens = self.value_net.forward(obs)
        v_pred = v_pred.squeeze(-1)
        delta = (2.0 / B) * (v_pred - returns)  # d/dv MSE
        d_output = delta[:, None]  # (B, 1)

        dw, db = self.value_net.backward(obs, hiddens, d_output)
        self.value_net.apply_gradients(dw, db, self.vf_lr, ascend=False)

    # ---- persistence ----

    def save(self, path: str):
        state = {
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "policy": self.policy_net.state_dict(),
            "value": self.value_net.state_dict(),
        }
        with open(path, "w") as f:
            json.dump(state, f)

    def load(self, path: str):
        with open(path) as f:
            state = json.load(f)
        self.policy_net.load_state_dict(state["policy"])
        self.value_net.load_state_dict(state["value"])
