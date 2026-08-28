# securemesh_testbed/experiments/run_experiment.py
"""Main CLI entry point for running SecureMesh experiments.

Usage
-----
    python -m securemesh_testbed.experiments.run_experiment \
        --scenario known_attacks \
        --attacker scripted \
        --defender rl \
        --episodes 100 \
        --seed 42 \
        --output results/exp_001
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import numpy as np
import pandas as pd

from ..game.markov_game import SecureMeshEnv
from ..game.actions import AttackerAction, DefenderAction
from ..evaluation.logger import ExperimentLogger
from ..evaluation.metrics import compute_all
from ..evaluation.plotting import generate_all_plots
from .scenario_known import get_scenario as known_scenario
from .scenario_zeroday import get_scenario as zeroday_scenario


# -------------------------------------------------------------------------
# Agent factory helpers
# -------------------------------------------------------------------------

def _make_attacker(kind: str, obs_dim: int, seed: int):
    if kind == "random":
        from ..agents.attacker.random_attacker import RandomAttacker
        return RandomAttacker(n_actions=len(AttackerAction), seed=seed)
    elif kind == "scripted":
        from ..agents.attacker.scripted_attacker import ScriptedAttacker
        return ScriptedAttacker()
    elif kind == "rl":
        from ..agents.attacker.rl_attacker import RLAttacker
        return RLAttacker(obs_dim=obs_dim, seed=seed)
    else:
        raise ValueError(f"Unknown attacker type: {kind}")


def _make_defender(kind: str, obs_dim: int, seed: int):
    if kind == "static":
        from ..agents.defender.static_defender import StaticDefender
        return StaticDefender()
    elif kind == "ml":
        from ..agents.defender.ml_defender import MLDefender
        return MLDefender(seed=seed)
    elif kind == "rl":
        from ..agents.defender.rl_defender import RLDefender
        return RLDefender(obs_dim=obs_dim, seed=seed)
    else:
        raise ValueError(f"Unknown defender type: {kind}")


# -------------------------------------------------------------------------
# Main loop
# -------------------------------------------------------------------------

def run(scenario_name: str, attacker_kind: str, defender_kind: str,
        episodes: int, seed: int, output_dir: str):
    """Execute the experiment loop."""

    # --- build scenario ---
    if scenario_name == "known_attacks":
        scenario = known_scenario(seed)
    elif scenario_name == "zero_day":
        scenario = zeroday_scenario(seed)
    else:
        raise ValueError(f"Unknown scenario: {scenario_name}")

    env = SecureMeshEnv(scenario)

    # --- initial reset to discover obs_dim ---
    atk_obs, def_obs = env.reset(seed=seed)
    obs_dim = len(atk_obs["obs"])

    # --- build agents ---
    attacker = _make_attacker(attacker_kind, obs_dim, seed)
    defender = _make_defender(defender_kind, obs_dim, seed)

    # --- warm-up for ML defender (collect data from static policy) ---
    if defender_kind == "ml":
        from ..agents.defender.static_defender import StaticDefender
        warmup_def = StaticDefender()
        for _ in range(min(50, episodes)):
            a_obs, d_obs = env.reset(seed=seed)
            done = False
            while not done:
                a_act = attacker.select_action(a_obs["obs"])
                d_act = warmup_def.select_action(d_obs["obs"])
                defender.record(d_obs["obs"], d_act)
                (a_obs, d_obs), _, done, _ = env.step((a_act, d_act))
        defender.fit()

    # --- logging ---
    os.makedirs(output_dir, exist_ok=True)
    logger = ExperimentLogger(output_dir, experiment_name="experiment")
    logger.log_meta({
        "scenario": scenario_name,
        "attacker": attacker_kind,
        "defender": defender_kind,
        "episodes": episodes,
        "seed": seed,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })

    # --- belief tracking (RL defender only) ---
    has_belief = hasattr(defender, "update_belief")
    belief_history = []

    # --- episode loop ---
    all_episode_metrics = []
    t0 = time.time()
    for ep in range(episodes):
        atk_obs, def_obs = env.reset(seed=seed + ep)
        if has_belief:
            defender.reset_belief()

        done = False
        step_infos = []
        atk_rewards = []
        def_rewards = []
        step_idx = 0

        while not done:
            a_act = attacker.select_action(atk_obs["obs"])
            d_act = defender.select_action(def_obs["obs"])

            (next_a_obs, next_d_obs), (a_rew, d_rew), done, info = env.step(
                (a_act, d_act)
            )

            # Bayesian belief update
            if has_belief:
                defender.update_belief(a_act)
                belief_history.append(defender.belief.copy())

            # agent learning
            attacker.update((atk_obs["obs"], a_act, a_rew,
                             next_a_obs["obs"], done))
            defender.update((def_obs["obs"], d_act, d_rew,
                             next_d_obs["obs"], done))

            # logging
            step_info = {
                "attacker_action": a_act,
                "defender_action": d_act,
                "attacker_reward": a_rew,
                "defender_reward": d_rew,
                **info,
            }
            step_infos.append(step_info)
            atk_rewards.append(a_rew)
            def_rewards.append(d_rew)
            logger.log_step(ep, step_idx, step_info)

            atk_obs, def_obs = next_a_obs, next_d_obs
            step_idx += 1

        # episode metrics
        ep_metrics = compute_all(step_infos, atk_rewards, def_rewards)
        logger.log_episode(ep, ep_metrics)
        all_episode_metrics.append(ep_metrics)

        if (ep + 1) % max(1, episodes // 10) == 0:
            elapsed = time.time() - t0
            print(f"  Episode {ep+1}/{episodes}  "
                  f"atk_r={ep_metrics['attacker_cumulative_reward']:.1f}  "
                  f"def_r={ep_metrics['defender_cumulative_reward']:.1f}  "
                  f"[{elapsed:.1f}s]")

    logger.close()

    # --- summary metrics (last 10 % of episodes) ---
    tail = all_episode_metrics[max(0, len(all_episode_metrics) - episodes // 10):]
    summary = {}
    if tail:
        for key in tail[0]:
            vals = [m[key] for m in tail]
            summary[key] = float(np.mean(vals))

    # --- plots ---
    try:
        ep_csv = os.path.join(output_dir, "experiment_episodes.csv")
        if os.path.exists(ep_csv):
            df = pd.read_csv(ep_csv)
            generate_all_plots(df, output_dir, metrics=summary,
                               belief_history=belief_history if has_belief else None)
            print(f"  Plots saved to {output_dir}")
    except Exception as e:
        print(f"  Warning: could not generate plots — {e}")

    print(f"\nExperiment complete.  Results in {output_dir}/")
    return summary


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SecureMesh Testbed — Experiment Runner")
    parser.add_argument("--scenario", default="known_attacks",
                        choices=["known_attacks", "zero_day"])
    parser.add_argument("--attacker", default="scripted",
                        choices=["random", "scripted", "rl"])
    parser.add_argument("--defender", default="static",
                        choices=["static", "ml", "rl"])
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="securemesh_testbed/results/default")
    args = parser.parse_args()

    print(f"SecureMesh Testbed — {args.scenario}")
    print(f"  Attacker: {args.attacker}  |  Defender: {args.defender}")
    print(f"  Episodes: {args.episodes}  |  Seed: {args.seed}")
    print(f"  Output:   {args.output}\n")

    run(
        scenario_name=args.scenario,
        attacker_kind=args.attacker,
        defender_kind=args.defender,
        episodes=args.episodes,
        seed=args.seed,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
