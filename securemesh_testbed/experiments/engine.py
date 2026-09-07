# securemesh_testbed/experiments/engine.py
"""Security Chaos Engineering experiment engine.

Implements the full SCE lifecycle described by Jolak et al. (SCENE):
  1. Define hypothesis
  2. Design the security experiment
  3. Prepare the infrastructure (reset environment)
  4. Introduce controlled attack/failure
  5. Monitor the system (telemetry collection)
  6. Analyse the result (compute metrics)
  7. Refine the defence and record the insight

Usage
-----
    # From YAML
    python -m securemesh_testbed.experiments.engine \\
        --config experiments/scenarios/02_ssh_bruteforce.yaml

    # Programmatic
    from securemesh_testbed.experiments.engine import SCEEngine
    from securemesh_testbed.experiments.schemas import ExperimentSpec
    spec = ExperimentSpec.from_yaml("experiments/scenarios/02_ssh_bruteforce.yaml")
    engine = SCEEngine()
    record = engine.run(spec)
"""

from __future__ import annotations

import argparse
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..game.markov_game import SecureMeshEnv
from ..game.actions import AttackerAction, DefenderAction
from ..evaluation.logger import ExperimentLogger
from ..evaluation.metrics import compute_all
from ..evaluation.plotting import generate_all_plots
from .schemas import ExperimentSpec, ExperimentRecord, HypothesisSpec

# Scenario loaders
from .scenario_known import get_scenario as known_scenario
from .scenario_zeroday import get_scenario as zeroday_scenario


# -------------------------------------------------------------------------
# Agent factory (extended with new strategy attackers)
# -------------------------------------------------------------------------

def _make_attacker(kind: str, obs_dim: int, seed: int):
    """Build an attacker agent by name."""
    if kind == "random":
        from ..agents.attacker.random_attacker import RandomAttacker
        return RandomAttacker(n_actions=len(AttackerAction), seed=seed)
    elif kind == "scripted":
        from ..agents.attacker.scripted_attacker import ScriptedAttacker
        return ScriptedAttacker()
    elif kind == "aggressive":
        from ..agents.attacker.strategy_attackers import AggressiveAttacker
        return AggressiveAttacker()
    elif kind == "stealthy":
        from ..agents.attacker.strategy_attackers import StealthyAttacker
        return StealthyAttacker()
    elif kind == "recon_heavy":
        from ..agents.attacker.strategy_attackers import ReconHeavyAttacker
        return ReconHeavyAttacker()
    elif kind == "rl":
        from ..agents.attacker.rl_attacker import RLAttacker
        return RLAttacker(obs_dim=obs_dim, seed=seed)
    elif kind == "llm":
        from ..agents.attacker.llm_attacker import LLMAttacker
        return LLMAttacker(seed=seed)
    else:
        raise ValueError(f"Unknown attacker type: {kind}")


def _make_defender(kind: str, obs_dim: int, seed: int):
    """Build a defender agent by name."""
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
# SCE Experiment Engine
# -------------------------------------------------------------------------

class SCEEngine:
    """Security Chaos Engineering experiment controller.

    Manages the full SCE lifecycle: prepare → inject → monitor → analyse → record.
    """

    def __init__(self, output_base: str = "securemesh_testbed/results"):
        self.output_base = output_base

    def run(self, spec: ExperimentSpec,
            output_dir: Optional[str] = None,
            event_callback=None,
            episode_callback=None) -> ExperimentRecord:
        """Execute a single SCE experiment from its specification.

        Parameters
        ----------
        spec : ExperimentSpec
            The experiment definition (from YAML or constructed programmatically).
        output_dir : str, optional
            Override the output directory. If None, uses
            ``{output_base}/{spec.id}``.

        Returns
        -------
        ExperimentRecord
            The structured result of the experiment.
        """
        if output_dir is None:
            output_dir = os.path.join(self.output_base, spec.id)
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"  SCE EXPERIMENT: {spec.id} - {spec.name}")
        print(f"  Attacker: {spec.attacker}  |  Defender: {spec.defender}")
        print(f"  Scenario: {spec.scenario}  |  Episodes: {spec.episodes}")
        print(f"  Seed: {spec.seed}  |  Duration: {spec.duration} steps")
        if spec.hypothesis:
            print(f"  Hypothesis: {spec.hypothesis.text[:80]}...")
        print(f"{'='*60}\n")

        # --- Phase 1: Prepare infrastructure ---
        scenario = self._load_scenario(spec.scenario, spec.seed)
        env = SecureMeshEnv(scenario)
        env.max_steps = spec.duration

        atk_obs, def_obs = env.reset(seed=spec.seed)
        obs_dim = len(atk_obs["obs"])
        initial_state = env.network.snapshot()

        # --- Build agents ---
        attacker = _make_attacker(spec.attacker, obs_dim, spec.seed)
        defender = _make_defender(spec.defender, obs_dim, spec.seed)

        # --- Warm-up for ML defender ---
        if spec.defender == "ml":
            self._warmup_ml_defender(env, attacker, defender, spec)

        # --- Logging ---
        logger = ExperimentLogger(output_dir, experiment_name=spec.id)
        logger.log_meta({
            "experiment_id": spec.id,
            "name": spec.name,
            "scenario": spec.scenario,
            "attacker": spec.attacker,
            "defender": spec.defender,
            "episodes": spec.episodes,
            "duration": spec.duration,
            "seed": spec.seed,
            "target": spec.target,
            "tags": spec.tags,
            "started_at": datetime.now(timezone.utc).isoformat(),
        })

        # --- Belief tracking ---
        has_belief = hasattr(defender, "update_belief")
        belief_history = []

        # --- Phase 2-5: Inject, monitor, collect ---
        all_episode_metrics = []
        all_timeline_events = []
        t0 = time.time()

        for ep in range(spec.episodes):
            ep_events = self._run_episode(
                env, attacker, defender, spec, ep,
                logger, has_belief, belief_history,
                event_callback, episode_callback
            )
            all_timeline_events.extend(ep_events["events"])
            all_episode_metrics.append(ep_events["metrics"])

            # Progress
            if (ep + 1) % max(1, spec.episodes // 10) == 0:
                elapsed = time.time() - t0
                m = ep_events["metrics"]
                print(
                    f"  Episode {ep+1}/{spec.episodes}  "
                    f"ASR={m['attack_success_rate']:.2f}  "
                    f"DR={m['detection_rate']:.2f}  "
                    f"Resilience={m['system_resilience']:.2f}  "
                    f"[{elapsed:.1f}s]"
                )

        logger.close()
        total_time = time.time() - t0

        # --- Phase 6: Analyse ---
        summary = self._compute_summary(all_episode_metrics, spec)
        summary["total_experiment_time_s"] = total_time
        summary["mttd_steps"] = summary.get("detection_latency", 0.0)
        summary["mttr_steps"] = summary.get("recovery_time", 0.0)

        # --- Hypothesis evaluation ---
        hypothesis_result = None
        if spec.hypothesis:
            passed = spec.hypothesis.evaluate(summary)
            hypothesis_result = {
                "text": spec.hypothesis.text,
                "metric": spec.hypothesis.metric,
                "threshold": spec.hypothesis.threshold,
                "direction": spec.hypothesis.direction,
                "actual_value": summary.get(spec.hypothesis.metric, 0.0),
                "result": "PASS" if passed else "FAIL",
            }
            status = "[PASS]" if passed else "[FAIL]"
            print(f"\n  Hypothesis: {status}")
            print(f"    {spec.hypothesis.metric} = "
                  f"{summary.get(spec.hypothesis.metric, 0.0):.4f} "
                  f"({'<' if spec.hypothesis.direction == 'below' else '>'} "
                  f"{spec.hypothesis.threshold})")

        # --- Phase 7: Record ---
        final_state = env.network.snapshot()
        record = ExperimentRecord(
            experiment_id=spec.id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            system_state_initial=initial_state,
            system_state_final=final_state,
            attacker_policy=spec.attacker,
            defender_policy=spec.defender,
            target=spec.target,
            scenario=spec.scenario,
            seed=spec.seed,
            episodes=spec.episodes,
            duration=spec.duration,
            hypothesis=hypothesis_result,
            metrics=summary,
            timeline=all_timeline_events[-100:],  # last 100 events
            tags=spec.tags,
        )

        record_path = os.path.join(output_dir, f"{spec.id}.json")
        record.to_json(record_path)

        # --- Generate plots ---
        self._generate_plots(output_dir, spec, belief_history)

        print(f"\n  Results saved to {output_dir}/")
        print(f"  Experiment record: {record_path}")
        return record

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_scenario(self, name: str, seed: int):
        if name == "known_attacks":
            return known_scenario(seed)
        elif name == "zero_day":
            return zeroday_scenario(seed)
        else:
            raise ValueError(f"Unknown scenario: {name}")

    def _warmup_ml_defender(self, env, attacker, defender, spec):
        """Collect training data for the ML defender."""
        from ..agents.defender.static_defender import StaticDefender
        warmup_def = StaticDefender()
        for _ in range(min(50, spec.episodes)):
            a_obs, d_obs = env.reset(seed=spec.seed)
            done = False
            while not done:
                a_act = attacker.select_action(a_obs["obs"])
                d_act = warmup_def.select_action(d_obs["obs"])
                defender.record(d_obs["obs"], d_act)
                (a_obs, d_obs), _, done, _ = env.step((a_act, d_act))
        defender.fit()

    def _run_episode(self, env, attacker, defender, spec, episode,
                     logger, has_belief, belief_history,
                     event_callback=None, episode_callback=None):
        """Run a single episode and return events + metrics."""
        atk_obs, def_obs = env.reset(seed=spec.seed + episode)
        if has_belief:
            defender.reset_belief()

        done = False
        step_infos = []
        atk_rewards = []
        def_rewards = []
        events = []
        step_idx = 0
        ep_start = time.time()

        # Track timing for MTTD/MTTR
        attack_started_step = None
        attack_detected_step = None
        defence_started_step = None

        while not done:
            step_time = time.time()
            a_act = attacker.select_action(atk_obs["obs"])
            d_act = defender.select_action(def_obs["obs"])

            (next_a_obs, next_d_obs), (a_rew, d_rew), done, info = env.step(
                (a_act, d_act)
            )

            # Bayesian belief update
            if has_belief:
                defender.update_belief(a_act)
                belief_history.append(defender.belief.copy())

            # Agent learning
            attacker.update((atk_obs["obs"], a_act, a_rew,
                             next_a_obs["obs"], done))
            defender.update((def_obs["obs"], d_act, d_rew,
                             next_d_obs["obs"], done))

            # Timeline event
            event = {
                "episode": episode,
                "step": step_idx,
                "wall_time": time.time() - ep_start,
                "attacker_action": a_act,
                "defender_action": d_act,
                "attacker_reward": a_rew,
                "defender_reward": d_rew,
                **info,
            }

            # Track MTTD timing
            if info.get("attacker_success") and attack_started_step is None:
                attack_started_step = step_idx
                event["event_type"] = "attack_started"
            if (info.get("attacker_detected") or info.get("defender_detected")):
                if attack_detected_step is None and attack_started_step is not None:
                    attack_detected_step = step_idx
                    event["event_type"] = "attack_detected"
                    event["detection_latency"] = step_idx - attack_started_step

            events.append(event)
            step_infos.append({
                "attacker_action": a_act,
                "defender_action": d_act,
                "attacker_reward": a_rew,
                "defender_reward": d_rew,
                **info,
            })
            atk_rewards.append(a_rew)
            def_rewards.append(d_rew)
            logger.log_step(episode, step_idx, event)
            if event_callback:
                # Provide a copy and map Enums to strings if needed
                import copy
                cb_event = copy.deepcopy(event)
                if has_belief:
                    cb_event["defender_belief"] = defender.belief.tolist()
                event_callback(cb_event)

            atk_obs, def_obs = next_a_obs, next_d_obs
            step_idx += 1

        # Episode metrics
        ep_metrics = compute_all(step_infos, atk_rewards, def_rewards)
        ep_metrics["episode_duration_s"] = time.time() - ep_start
        ep_metrics["total_steps"] = step_idx
        logger.log_episode(episode, ep_metrics)
        if episode_callback:
            episode_callback(episode, ep_metrics)

        return {"events": events, "metrics": ep_metrics}

    def _compute_summary(self, all_metrics: List[Dict[str, float]],
                         spec: ExperimentSpec) -> Dict[str, float]:
        """Compute aggregate statistics over all episodes."""
        if not all_metrics:
            return {}

        summary = {}
        # Use last 20% of episodes for converged metrics
        tail_start = max(0, len(all_metrics) - len(all_metrics) // 5)
        tail = all_metrics[tail_start:]
        all_data = all_metrics

        for key in all_metrics[0]:
            vals = [m[key] for m in all_data]
            tail_vals = [m[key] for m in tail]
            summary[key] = float(np.mean(vals))
            summary[f"{key}_std"] = float(np.std(vals))
            summary[f"{key}_tail"] = float(np.mean(tail_vals))

        return summary

    def _generate_plots(self, output_dir: str, spec: ExperimentSpec,
                        belief_history: list):
        """Generate plots from the episode CSV."""
        import pandas as pd
        ep_csv = os.path.join(output_dir, f"{spec.id}_episodes.csv")
        if not os.path.exists(ep_csv):
            return
        try:
            df = pd.read_csv(ep_csv)
            generate_all_plots(
                df, output_dir,
                belief_history=belief_history if belief_history else None,
            )
            print(f"  Plots saved to {output_dir}")
        except Exception as e:
            print(f"  Warning: could not generate plots — {e}")


# -------------------------------------------------------------------------
# CLI entry point
# -------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SecureMesh-SCE — Security Chaos Engineering Engine")
    parser.add_argument("--config", required=True,
                        help="Path to YAML experiment definition")
    parser.add_argument("--output", default=None,
                        help="Override output directory")
    args = parser.parse_args()

    spec = ExperimentSpec.from_yaml(args.config)
    engine = SCEEngine()
    record = engine.run(spec, output_dir=args.output)

    print(f"\nExperiment {record.experiment_id} complete.")
    if record.hypothesis:
        print(f"  Hypothesis: {record.hypothesis['result']}")
    print(f"  Key metrics:")
    for k in ["attack_success_rate", "detection_rate", "system_resilience",
              "mttd_steps", "mttr_steps"]:
        if k in record.metrics:
            print(f"    {k}: {record.metrics[k]:.4f}")


if __name__ == "__main__":
    main()
