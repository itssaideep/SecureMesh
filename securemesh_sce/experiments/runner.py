# securemesh_sce/experiments/runner.py
"""Single experiment execution orchestrator CLI.

Usage:
    python -m securemesh_sce.experiments.runner --config experiments/scenarios/01_baseline.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Any, Optional

from .engine import SCENEExperimentEngine, ExperimentConfig


def run_experiment(
    config_path: str,
    episodes: Optional[int] = None,
    seed: Optional[int] = None,
    output_dir: str = "experiments/results",
) -> Dict[str, Any]:
    """Execute a single scenario YAML experiment."""
    cfg = ExperimentConfig.from_yaml(config_path)
    if episodes is not None:
        cfg.episodes = episodes
    if seed is not None:
        cfg.seed = seed

    engine = SCENEExperimentEngine(config=cfg, output_dir=output_dir)
    print(f"\n=======================================================")
    print(f"Starting SCENE Experiment: {cfg.id} - {cfg.name}")
    print(f"Attacker: {cfg.attacker_name.upper()} | Defender: {cfg.defender_name.upper()}")
    print(f"Episodes: {cfg.episodes} | Steps: {cfg.duration} | Seed: {cfg.seed}")
    print(f"=======================================================")

    summary = engine.run()

    hyp = summary["hypothesis_result"]
    agg = summary["aggregate_metrics"]

    print(f"\n--- SCENE Results Summary ---")
    print(f"Hypothesis Verdict: [{hyp['verdict']}]")
    print(f"  Metric: {hyp['metric']} | Observed: {hyp['observed_mean']:.4f} | Threshold: {hyp['threshold']}")
    print(f"  95% CI: [{hyp['ci_95'][0]:.4f}, {hyp['ci_95'][1]:.4f}]")
    print(f"Aggregate Performance:")
    print(f"  Attack Success Rate:   {agg['attack_success_rate']:.4f}")
    print(f"  Detection Rate:        {agg['detection_rate']:.4f}")
    print(f"  Service Availability:  {agg['service_availability']:.4f}")
    print(f"  Mean Intervention Cost:{agg['intervention_cost']:.2f}")
    print(f"Outputs written to: {output_dir}\n")

    return summary


def main():
    parser = argparse.ArgumentParser(description="SecureMesh-SCE Scenario Runner")
    parser.add_argument("--config", type=str, required=True, help="Path to scenario YAML config")
    parser.add_argument("--episodes", type=int, default=None, help="Override number of episodes")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--output_dir", type=str, default="experiments/results", help="Results directory")

    args = parser.parse_args()
    try:
        run_experiment(args.config, args.episodes, args.seed, args.output_dir)
    except Exception as e:
        print(f"Error running experiment: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
