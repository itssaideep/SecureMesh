#!/usr/bin/env python
"""Quick RL diagnostic: Does PPO actually learn anything?

Runs PPO attacker against static defender for N episodes and tracks:
- Action entropy per episode (should DECREASE if learning)
- Mean reward per episode (should INCREASE if learning)
- Policy weight norms (should CHANGE if gradients flow)
- Action distribution (should become NON-uniform if learning)

This test runs WITHOUT hardware to isolate the RL signal.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from securemesh_sce.agents.common.ppo_core import PPOCore
from securemesh_sce.agents.common.mlp import NumpyMLP


class MiniAttackerEnv:
    """Simplified version of BayesianGameEnv for RL diagnostics only."""
    
    N_ACTIONS = 13
    OBS_DIM = 25
    
    def __init__(self, seed=42):
        self.rng = np.random.RandomState(seed)
        self.step_count = 0
        self.max_steps = 60
        self.obs = None
    
    def reset(self):
        self.step_count = 0
        self.obs = self.rng.randn(self.OBS_DIM).astype(np.float32) * 0.1
        return self.obs.copy()
    
    def step(self, action):
        self.step_count += 1
        
        reward = -0.05  # step cost
        
        detection_probs = {
            0: 0.4, 1: 0.2, 2: 0.6, 3: 0.3,
            4: 0.5, 5: 0.45,
            6: 0.7, 7: 0.4, 8: 0.35,
            9: 0.0, 10: 0.0,
            11: 0.3, 12: 0.0,
        }
        
        success_probs = {
            0: 0.0, 1: 0.0, 2: 0.15, 3: 0.10,
            4: 0.30, 5: 0.25,
            6: 0.20, 7: 0.15, 8: 0.15,
            9: 0.0, 10: 0.0,
            11: 0.10, 12: 0.0,
        }
        
        if action in (0, 1):
            reward += 10.0
        
        if self.rng.random() < success_probs.get(action, 0):
            reward += 100.0
        
        if self.rng.random() < detection_probs.get(action, 0):
            reward -= 30.0
        
        self.obs = self.obs * 0.95 + self.rng.randn(self.OBS_DIM).astype(np.float32) * 0.05
        
        done = self.step_count >= self.max_steps
        return self.obs.copy(), reward, done


def compute_entropy(probs):
    p = probs[probs > 1e-10]
    return -np.sum(p * np.log(p))


def weight_norm(mlp):
    return sum(np.linalg.norm(w) for w in mlp.weights)


def run_diagnostic(n_episodes=50, seed=42):
    env = MiniAttackerEnv(seed=seed)
    
    ppo = PPOCore(
        obs_dim=env.OBS_DIM,
        act_dim=env.N_ACTIONS,
        hidden_sizes=(64, 64),
        gamma=0.99, lam=0.95, clip_eps=0.2,
        pi_lr=2e-3, vf_lr=1e-3,
        train_epochs=4, batch_size=32,
        entropy_coef=0.01, seed=seed,
    )
    
    init_weight_norm = weight_norm(ppo.policy_net)
    
    ACTION_NAMES = ["RECON_SCAN", "RECON_FP", "AUTH_BF", "AUTH_CS", 
                    "EXPLOIT_SVC", "EXPLOIT_IOT", "MAL_DROP", "PERSIST_BD",
                    "PERSIST_C2", "EVADE_OBF", "EVADE_SLOW", "LAT_MOVE", "NOOP"]
    
    print("=" * 70)
    print("  RL DIAGNOSTIC: PPO Attacker Learning Test")
    print("=" * 70)
    print(f"  Obs dim: {env.OBS_DIM} | Act dim: {env.N_ACTIONS}")
    param_count = sum(w.size for w in ppo.policy_net.weights) + sum(b.size for b in ppo.policy_net.biases)
    print(f"  Network: MLP(64, 64) | Params: ~{param_count}")
    print(f"  Episodes: {n_episodes} | Steps/ep: {env.max_steps}")
    print(f"  Initial weight norm: {init_weight_norm:.4f}")
    print(f"  Max entropy (uniform): {np.log(env.N_ACTIONS):.4f}")
    print("=" * 70)
    print(f"{'Ep':>4} | {'Mean Reward':>12} | {'Entropy':>8} | {'W Norm':>8} | {'Top Action':>12} | {'Top Prob':>8}")
    print("-" * 70)
    
    ep_rewards = []
    ep_entropies = []
    ep_weight_norms = []
    
    for ep in range(n_episodes):
        obs = env.reset()
        total_reward = 0.0
        action_counts = np.zeros(env.N_ACTIONS)
        
        done = False
        while not done:
            action = ppo.select_action(obs)
            action_counts[action] += 1
            next_obs, reward, done = env.step(action)
            total_reward += reward
            ppo.store_transition(obs, action, reward, next_obs, done)
            obs = next_obs
        
        test_obs = env.reset()
        probs = ppo.action_probs(test_obs)
        entropy = compute_entropy(probs)
        w_norm = weight_norm(ppo.policy_net)
        top_action = np.argmax(action_counts)
        top_prob = probs[top_action]
        
        ep_rewards.append(total_reward)
        ep_entropies.append(entropy)
        ep_weight_norms.append(w_norm)
        
        if ep < 5 or ep >= n_episodes - 5 or ep % 10 == 0:
            print(f"{ep:4d} | {total_reward:12.2f} | {entropy:8.4f} | {w_norm:8.4f} | {ACTION_NAMES[top_action]:>12} | {top_prob:8.4f}")
    
    print("\n" + "=" * 70)
    print("  LEARNING DIAGNOSTICS SUMMARY")
    print("=" * 70)
    
    first_5_reward = np.mean(ep_rewards[:5])
    last_5_reward = np.mean(ep_rewards[-5:])
    first_5_entropy = np.mean(ep_entropies[:5])
    last_5_entropy = np.mean(ep_entropies[-5:])
    
    print(f"  Reward (first 5 eps):  {first_5_reward:.2f}")
    print(f"  Reward (last 5 eps):   {last_5_reward:.2f}")
    print(f"  Reward change:         {last_5_reward - first_5_reward:+.2f} ({'IMPROVING' if last_5_reward > first_5_reward else 'NOT IMPROVING'})")
    print()
    print(f"  Entropy (first 5 eps): {first_5_entropy:.4f}")
    print(f"  Entropy (last 5 eps):  {last_5_entropy:.4f}")
    print(f"  Entropy change:        {last_5_entropy - first_5_entropy:+.4f} ({'SPECIALIZING' if last_5_entropy < first_5_entropy - 0.1 else 'STILL RANDOM'})")
    print()
    print(f"  Weight norm (initial): {init_weight_norm:.4f}")
    print(f"  Weight norm (final):   {ep_weight_norms[-1]:.4f}")
    print(f"  Weight norm change:    {ep_weight_norms[-1] - init_weight_norm:+.4f} ({'GRADIENTS FLOWING' if abs(ep_weight_norms[-1] - init_weight_norm) > 0.1 else 'NO GRADIENT FLOW'})")
    print()
    
    test_obs = env.reset()
    probs = ppo.action_probs(test_obs)
    print("  Final action distribution:")
    for i, name in enumerate(ACTION_NAMES):
        bar = "#" * int(probs[i] * 50)
        print(f"    {name:>12}: {probs[i]:.4f} {bar}")
    
    is_learning = (
        last_5_reward > first_5_reward + 10
        and last_5_entropy < first_5_entropy - 0.1
        and abs(ep_weight_norms[-1] - init_weight_norm) > 0.1
    )
    
    print()
    print(f"  RL VERDICT: {'LEARNING IS WORKING' if is_learning else 'RL IS NOT LEARNING EFFECTIVELY'}")
    print("=" * 70)
    
    return is_learning


if __name__ == "__main__":
    episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    run_diagnostic(n_episodes=episodes)
