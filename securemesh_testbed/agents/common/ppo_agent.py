# securemesh_testbed/agents/common/ppo_agent.py
"""Lightweight NumPy-only Proximal Policy Optimisation (PPO) agent.

Design goals
------------
* Zero heavy dependencies — works on constrained hardware.
* Two-layer tanh MLP for both policy and value networks.
* Generalised Advantage Estimation (GAE-lambda).
* Clipped surrogate objective with mini-batch SGD.

The agent collects transitions via `update(transition)`.  When a
terminal transition (`done=True`) arrives the collected episode is used
for a PPO training step before the buffer is cleared.
"""

from __future__ import annotations

import json
import numpy as np
from typing import List, Tuple, Dict

from ..base_agent import BaseAgent


class PPOAgent(BaseAgent):
    """NumPy-only PPO with analytical MLP gradients."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_size: int = 64,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_eps: float = 0.2,
        pi_lr: float = 3e-4,
        vf_lr: float = 1e-3,
        train_epochs: int = 4,
        batch_size: int = 32,
        seed: int = 0,
    ):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.rng = np.random.RandomState(seed)

        # --- policy network (obs_dim -> hidden -> act_dim) ---
        scale = np.sqrt(2.0 / obs_dim)
        self.pi_w1 = self.rng.randn(obs_dim, hidden_size).astype(np.float32) * scale
        self.pi_b1 = np.zeros(hidden_size, dtype=np.float32)
        scale_h = np.sqrt(2.0 / hidden_size)
        self.pi_w2 = self.rng.randn(hidden_size, act_dim).astype(np.float32) * scale_h
        self.pi_b2 = np.zeros(act_dim, dtype=np.float32)

        # --- value network (obs_dim -> hidden -> 1) ---
        self.v_w1 = self.rng.randn(obs_dim, hidden_size).astype(np.float32) * scale
        self.v_b1 = np.zeros(hidden_size, dtype=np.float32)
        self.v_w2 = self.rng.randn(hidden_size, 1).astype(np.float32) * scale_h
        self.v_b2 = np.zeros(1, dtype=np.float32)

        # hyper-parameters
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.pi_lr = pi_lr
        self.vf_lr = vf_lr
        self.train_epochs = train_epochs
        self.batch_size = batch_size

        # trajectory buffer
        self._obs: List[np.ndarray] = []
        self._act: List[int] = []
        self._rew: List[float] = []
        self._nobs: List[np.ndarray] = []
        self._done: List[bool] = []

    # ------------------------------------------------------------------
    # Forward passes
    # ------------------------------------------------------------------
    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        shifted = logits - logits.max(axis=-1, keepdims=True)
        e = np.exp(shifted)
        return e / e.sum(axis=-1, keepdims=True)

    def _policy_logits(self, obs: np.ndarray) -> np.ndarray:
        """(B, obs_dim) -> (B, act_dim) logits."""
        h = np.tanh(obs @ self.pi_w1 + self.pi_b1)
        return h @ self.pi_w2 + self.pi_b2

    def _value(self, obs: np.ndarray) -> np.ndarray:
        """(B, obs_dim) -> (B,) scalar values."""
        h = np.tanh(obs @ self.v_w1 + self.v_b1)
        return (h @ self.v_w2 + self.v_b2).squeeze(-1)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------
    def select_action(self, observation) -> int:
        obs = np.asarray(observation, dtype=np.float32).reshape(1, -1)
        logits = self._policy_logits(obs)[0]
        probs = self._softmax(logits)
        return int(self.rng.choice(self.act_dim, p=probs))

    def action_probs(self, observation) -> np.ndarray:
        """Return the full probability vector (useful for metrics)."""
        obs = np.asarray(observation, dtype=np.float32).reshape(1, -1)
        return self._softmax(self._policy_logits(obs)[0])

    # ------------------------------------------------------------------
    # Update / training
    # ------------------------------------------------------------------
    def update(self, transition: tuple):
        obs, act, rew, next_obs, done = transition
        self._obs.append(np.asarray(obs, dtype=np.float32))
        self._act.append(int(act))
        self._rew.append(float(rew))
        self._nobs.append(np.asarray(next_obs, dtype=np.float32))
        self._done.append(bool(done))
        if done:
            self._train_on_episode()

    def _train_on_episode(self):
        T = len(self._obs)
        if T == 0:
            return
        obs = np.stack(self._obs)          # (T, D)
        acts = np.array(self._act)         # (T,)
        rews = np.array(self._rew, dtype=np.float32)
        nobs = np.stack(self._nobs)
        dones = np.array(self._done, dtype=np.float32)

        # --- compute old log-probs (before any updates) ---
        old_logits = self._policy_logits(obs)
        old_probs = self._softmax(old_logits)
        old_log_pi = np.log(old_probs[np.arange(T), acts] + 1e-8)

        # --- GAE ---
        vals = self._value(obs)
        next_vals = self._value(nobs)
        deltas = rews + self.gamma * next_vals * (1.0 - dones) - vals
        adv = np.zeros(T, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(T)):
            gae = deltas[t] + self.gamma * self.lam * (1.0 - dones[t]) * gae
            adv[t] = gae
        returns = adv + vals
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # --- PPO epochs ---
        idxs = np.arange(T)
        for _ in range(self.train_epochs):
            self.rng.shuffle(idxs)
            for start in range(0, T, self.batch_size):
                mb = idxs[start:start + self.batch_size]
                self._update_policy(obs[mb], acts[mb], adv[mb], old_log_pi[mb])
                self._update_value(obs[mb], returns[mb])

        self._clear_buffer()

    # --- policy gradient step ------------------------------------------------
    def _update_policy(self, obs, acts, adv, old_log_pi):
        B = len(obs)
        # forward
        h = np.tanh(obs @ self.pi_w1 + self.pi_b1)        # (B, H)
        logits = h @ self.pi_w2 + self.pi_b2               # (B, A)
        probs = self._softmax(logits)
        log_pi = np.log(probs[np.arange(B), acts] + 1e-8)

        # ratio and clipped surrogate
        ratio = np.exp(log_pi - old_log_pi)
        clipped = np.clip(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
        # we maximise min(ratio*adv, clipped*adv)
        # gradient of log pi w.r.t. logits: d log pi_a / d logit_j = 1(j==a) - pi_j
        d_logits = -probs.copy()                            # (B, A)
        d_logits[np.arange(B), acts] += 1.0
        # mask: use ratio*adv where it is the minimum, else clipped (no grad)
        use_unclipped = (ratio * adv <= clipped * adv).astype(np.float32)
        scale = (use_unclipped * adv)[:, None]              # (B, 1)
        d_logits *= scale / B                               # weighted

        # back-prop through layer 2
        dw2 = h.T @ d_logits
        db2 = d_logits.sum(axis=0)
        # back-prop through tanh
        dh = (d_logits @ self.pi_w2.T) * (1.0 - h ** 2)
        dw1 = obs.T @ dh
        db1 = dh.sum(axis=0)

        # gradient *ascent* (maximise objective)
        self.pi_w2 += self.pi_lr * dw2
        self.pi_b2 += self.pi_lr * db2
        self.pi_w1 += self.pi_lr * dw1
        self.pi_b1 += self.pi_lr * db1

    # --- value gradient step -------------------------------------------------
    def _update_value(self, obs, returns):
        B = len(obs)
        h = np.tanh(obs @ self.v_w1 + self.v_b1)
        v_pred = (h @ self.v_w2 + self.v_b2).squeeze(-1)
        delta = (2.0 / B) * (v_pred - returns)             # d/dv MSE

        dw2 = h.T @ delta[:, None]
        db2 = delta.sum(keepdims=True)
        dh = (delta[:, None] * self.v_w2.T) * (1.0 - h ** 2)
        dw1 = obs.T @ dh
        db1 = dh.sum(axis=0)

        self.v_w1 -= self.vf_lr * dw1
        self.v_b1 -= self.vf_lr * db1
        self.v_w2 -= self.vf_lr * dw2
        self.v_b2 -= self.vf_lr * db2.flatten()

    def _clear_buffer(self):
        self._obs.clear()
        self._act.clear()
        self._rew.clear()
        self._nobs.clear()
        self._done.clear()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str):
        data = {k: v.tolist() for k, v in self.__dict__.items()
                if isinstance(v, np.ndarray)}
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path: str):
        with open(path) as f:
            data = json.load(f)
        for k, v in data.items():
            setattr(self, k, np.array(v, dtype=np.float32))
