# securemesh_sce/experiments/runner.py
"""Single experiment execution orchestrator CLI.

Usage:
    python -m securemesh_sce.experiments.runner --config experiments/scenarios/01_baseline.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

from dotenv import load_dotenv
load_dotenv()

from .engine import SCENEExperimentEngine, ExperimentConfig


def run_experiment(
    config_path: str,
    episodes: Optional[int] = None,
    duration: Optional[int] = None,
    seed: Optional[int] = None,
    output_dir: str = "experiments/results",
    hardware: bool = True,
    hardware_ip: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute a single scenario YAML experiment in pure physical hardware mode."""
    cfg = ExperimentConfig.from_yaml(config_path)
    if episodes is not None:
        cfg.episodes = episodes
    if duration is not None:
        cfg.duration = duration
    if seed is not None:
        cfg.seed = seed
    cfg.use_physical_hardware = True
    if hardware_ip:
        cfg.hardware_ip = hardware_ip
    elif os.getenv("ESP8266_DEVICE_IP"):
        cfg.hardware_ip = os.getenv("ESP8266_DEVICE_IP")

    engine = SCENEExperimentEngine(config=cfg, output_dir=output_dir)
    print(f"\n=======================================================")
    print(f"Starting SCENE Experiment: {cfg.id} - {cfg.name}")
    print(f"Attacker: {cfg.attacker_name.upper()} | Defender: {cfg.defender_name.upper()}")
    print(f"Episodes: {cfg.episodes} | Steps: {cfg.duration} | Seed: {cfg.seed}")
    print(f"Mode: PURE PHYSICAL HARDWARE (ESP8266 @ {cfg.hardware_ip or '192.168.0.111'})")
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
    parser = argparse.ArgumentParser(description="SecureMesh-SCE Scenario Runner (Pure Physical Hardware)")
    parser.add_argument("--config", type=str, required=True, help="Path to scenario YAML config")
    parser.add_argument("--episodes", type=int, default=None, help="Override number of episodes")
    parser.add_argument("--duration", type=int, default=None, help="Override steps per episode")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--output_dir", type=str, default="experiments/results", help="Results directory")
    parser.add_argument("--hardware-ip", type=str, default=None, help="Physical IoT hardware IP address (default: from .env or 192.168.0.111)")

    args = parser.parse_args()
    try:
        run_experiment(
            args.config,
            args.episodes,
            args.duration,
            args.seed,
            args.output_dir,
            hardware=True,
            hardware_ip=args.hardware_ip,
        )
    except Exception as e:
        print(f"Error running experiment: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
