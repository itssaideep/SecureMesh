# securemesh_sce/experiments/engine.py
"""SCENE Experiment Engine: 8-stage Security Chaos Engineering lifecycle.

Implements the formal SCENE methodology (Jolak et al., 2026):
1. Define Hypothesis
2. Establish Steady State
3. Select Scenario & Injected Security Fault
4. Execute Joint Steps with Attacker/Defender Policies
5. Telemetry & Telemetry Anomaly Detection
6. Measure Outcome & Evaluate Hypothesis (PASS / FAIL)
7. Compute Residual Risk & Update Risk Register
8. Export Telemetry, Plots, and Structured Records
"""

from __future__ import annotations

import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from ..game.bayesian_game import BayesianGameEnv, SCEScenario
from ..game.actions import AttackerType, N_ATTACKER_ACTIONS, N_DEFENDER_ACTIONS
from ..agents.attacker import ScriptedAttacker, PPOAttacker, LLMAttacker
from ..agents.defender import StaticDefender, RLDefender, LLMDefender
from ..environment.telemetry.collector import TelemetryCollector, SteadyStateDetector
from ..evaluation.metrics import MetricsEngine, EpisodeMetrics
from ..evaluation.statistics import StatisticalAnalyzer, HypothesisResult
from ..logging.experiment_logger import ExperimentLogger


from dotenv import load_dotenv
load_dotenv()


@dataclass
class ExperimentConfig:
    id: str
    name: str
    attacker_name: str
    defender_name: str
    duration: int = 100
    episodes: int = 10
    seed: int = 42
    use_physical_hardware: bool = True
    hardware_ip: Optional[str] = None
    hypothesis: Dict[str, Any] = field(default_factory=dict)
    raw_yaml: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        exp_data = data.get("experiment", {})
        return cls(
            id=exp_data.get("id", "SCE-001"),
            name=exp_data.get("name", "Untitled Experiment"),
            attacker_name=exp_data.get("attacker", "scripted").lower(),
            defender_name=exp_data.get("defender", "static").lower(),
            duration=int(exp_data.get("duration", 100)),
            episodes=int(exp_data.get("episodes", 10)),
            seed=int(exp_data.get("seed", 42)),
            use_physical_hardware=bool(exp_data.get("use_physical_hardware", True)),
            hardware_ip=exp_data.get("hardware_ip", None),
            hypothesis=data.get("hypothesis", {}),
            raw_yaml=data,
        )


class SCENEExperimentEngine:
    """Orchestrates an end-to-end experiment lifecycle."""

    def __init__(self, config: ExperimentConfig, output_dir: str = "experiments/results"):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger = ExperimentLogger(str(self.output_dir), self.config.id)
        self.telemetry = TelemetryCollector()

    def _build_agents(
        self, env: BayesianGameEnv
    ) -> Tuple[Any, Any]:
        """Instantiate attacker and defender agents based on configuration."""
        atk_obs_dim = env._atk_obs_dim
        def_obs_dim = env._def_obs_dim

        # Attacker instantiation
        atk_type = self.config.attacker_name
        if "llm" in atk_type or "generative" in atk_type or "gemini" in atk_type or "llama" in atk_type:
            attacker = LLMAttacker(seed=self.config.seed)
        elif "ppo" in atk_type or "adaptive" in atk_type or "rl" in atk_type:
            attacker = PPOAttacker(obs_dim=atk_obs_dim, seed=self.config.seed)
        else:
            attacker = ScriptedAttacker(attacker_type=env.attacker_type)

        # Defender instantiation
        def_type = self.config.defender_name
        if "llm" in def_type or "generative" in def_type or "gemini" in def_type:
            defender = LLMDefender(seed=self.config.seed)
        elif "rl" in def_type or "ppo" in def_type or "adaptive" in def_type:
            defender = RLDefender(obs_dim=def_obs_dim, seed=self.config.seed)
        else:
            defender = StaticDefender()

        return attacker, defender

    def run(
        self,
        event_callback: Optional[Any] = None,
        episode_callback: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Execute full multi-episode experiment lifecycle."""
        # 1. Define Hypothesis
        hyp_metric = self.config.hypothesis.get("metric", "detection_rate")

        scenario = SCEScenario(
            scenario_id=self.config.id,
            name=self.config.name,
            duration=self.config.duration,
            seed=self.config.seed,
            use_physical_hardware=self.config.use_physical_hardware,
            hardware_ip=self.config.hardware_ip,
        )
        env = BayesianGameEnv(scenario=scenario)

        episode_metrics_list: List[EpisodeMetrics] = []
        observed_values: List[float] = []

        # 2. Establish Steady State
        (atk_obs, def_obs), _ = env.reset(seed=self.config.seed)
        steady_state_detector = SteadyStateDetector(window_size=5)
        for warm_step in range(5):
            sample = self.telemetry.collect(warm_step, env._network_state)
            steady_state_detector.record_sample(sample)
        steady_ok, steady_report = steady_state_detector.is_steady_state_verified()

        attacker, defender = self._build_agents(env)

        # Run episodes
        for ep in range(self.config.episodes):
            ep_seed = self.config.seed + ep
            (atk_obs, def_obs), info = env.reset(seed=ep_seed)
            if hasattr(attacker, "reset"):
                attacker.reset()
            if hasattr(defender, "reset"):
                defender.reset()

            done = False
            while not done:
                a_atk = attacker.select_action(atk_obs)
                a_def = defender.select_action(def_obs)

                (next_atk_obs, next_def_obs), (r_atk, r_def), term, trunc, step_info = env.step(
                    (a_atk, a_def)
                )
                done = term or trunc

                # Agent policy updates if learning
                if hasattr(attacker, "update"):
                    attacker.update((atk_obs, a_atk, r_atk, next_atk_obs, done))
                if hasattr(defender, "update"):
                    defender.update((def_obs, a_def, r_def, next_def_obs, done))

                # Telemetry logging
                step_record = {
                    "episode": ep,
                    **step_info,
                    "reward_attacker": r_atk,
                    "reward_defender": r_def,
                }
                self.logger.log_step(step_record)
                if event_callback:
                    try:
                        event_callback(step_record)
                    except Exception:
                        pass

                atk_obs = next_atk_obs
                def_obs = next_def_obs

            # End of episode evaluation
            ep_m = MetricsEngine.evaluate_episode(
                episode_id=ep,
                step_log=env.step_log,
                outcomes=env.episode_outcomes,
            )
            episode_metrics_list.append(ep_m)
            self.logger.log_episode_summary(ep_m.to_dict())

            if episode_callback:
                try:
                    episode_callback(ep, ep_m.to_dict())
                except Exception:
                    pass

            # Extract metric for hypothesis
            val = getattr(ep_m, hyp_metric, 0.0)
            observed_values.append(val)

        # 6. Measure Outcome & Evaluate Hypothesis
        hyp_res = StatisticalAnalyzer.evaluate_hypothesis(
            self.config.hypothesis, observed_values
        )

        mean_dr = float(np.mean([m.detection_rate for m in episode_metrics_list]))

        # 7. Export Telemetry, Metadata & Structured Summary
        experiment_summary = {
            "experiment_id": self.config.id,
            "name": self.config.name,
            "attacker": self.config.attacker_name,
            "defender": self.config.defender_name,
            "episodes": self.config.episodes,
            "duration": self.config.duration,
            "seed": self.config.seed,
            "steady_state": steady_report,
            "hypothesis_result": {
                "metric": hyp_res.metric,
                "text": hyp_res.hypothesis_text,
                "threshold": hyp_res.threshold,
                "direction": hyp_res.direction,
                "observed_mean": hyp_res.observed_mean,
                "ci_95": hyp_res.ci_95,
                "verdict": hyp_res.verdict,
                "passed": hyp_res.passed,
            },
            "aggregate_metrics": {
                "attack_success_rate": float(np.mean([m.attack_success_rate for m in episode_metrics_list])),
                "detection_rate": mean_dr,
                "precision": float(np.mean([m.precision for m in episode_metrics_list])),
                "recall": float(np.mean([m.recall for m in episode_metrics_list])),
                "f1_score": float(np.mean([m.f1_score for m in episode_metrics_list])),
                "service_availability": float(np.mean([m.mean_service_availability for m in episode_metrics_list])),
                "intervention_cost": float(np.mean([m.total_intervention_cost for m in episode_metrics_list])),
            },
        }

        self.logger.write_experiment_summary(experiment_summary)
        return experiment_summary
