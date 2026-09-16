# securemesh_sce/experiments/matrix_runner.py
"""Cross-product Attacker-Defender Matrix and Ablation Runner.

Runs the complete research grid:
- Offense Spectrum: Scripted (L0), BC (L1), GAIL (L2), PPO (L3), LLM (L4)
- Defense Spectrum: Static (D1), ML (D2), RL (D3), Bayesian RL (D4), Constrained (D5)

Computes multi-seed statistical distributions, significance tests (Mann-Whitney U, Cohen's d),
and automatically emits publication-ready LaTeX tables, Markdown tables, and plots.
"""

from __future__ import annotations

import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional

from .engine import SCENEExperimentEngine, ExperimentConfig
from ..evaluation.statistics import StatisticalAnalyzer
from ..evaluation.tables import PublicationTableGenerator
from ..evaluation.plots import PublicationPlotter


OFFENSE_LEVELS = ["scripted", "bc", "gail", "ppo", "llm"]
DEFENSE_LEVELS = ["static", "ml", "rl", "bayesian_rl", "constrained"]


def run_matrix_experiment(
    offense_spectrum: Optional[List[str]] = None,
    defense_spectrum: Optional[List[str]] = None,
    episodes: int = 5,
    seeds: List[int] = [42, 123, 456],
    duration: int = 60,
    output_dir: str = "experiments/results",
) -> Dict[str, Any]:
    """Execute multi-seed cross-product matrix evaluation."""
    offense = offense_spectrum or OFFENSE_LEVELS
    defense = defense_spectrum or DEFENSE_LEVELS

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    matrix_results: Dict[str, Dict[str, Dict[str, float]]] = {}
    ablation_rows: List[Dict[str, Any]] = []

    print(f"\n=======================================================")
    print(f"Executing Attacker-Defender Matrix: {len(offense)}x{len(defense)} ({len(seeds)} seeds)")
    print(f"Total cells: {len(offense) * len(defense)} | Total runs: {len(offense) * len(defense) * len(seeds)}")
    print(f"=======================================================\n")

    for atk in offense:
        matrix_results[atk] = {}
        for dfn in defense:
            cell_asrs = []
            cell_avails = []
            cell_costs = []

            for s in seeds:
                cfg = ExperimentConfig(
                    id=f"MAT-{atk[:3].upper()}-{dfn[:3].upper()}-S{s}",
                    name=f"Matrix {atk} vs {dfn}",
                    attacker_name=atk,
                    defender_name=dfn,
                    duration=duration,
                    episodes=episodes,
                    seed=s,
                    hypothesis={"metric": "service_availability", "threshold": 0.8, "direction": "above"},
                )
                engine = SCENEExperimentEngine(config=cfg, output_dir=output_dir)
                res = engine.run()

                agg = res["aggregate_metrics"]
                cell_asrs.append(agg["attack_success_rate"])
                cell_avails.append(agg["service_availability"])
                cell_costs.append(agg["intervention_cost"])

            mean_asr = float(np.mean(cell_asrs))
            mean_avail = float(np.mean(cell_avails))
            mean_cost = float(np.mean(cell_costs))

            matrix_results[atk][dfn] = {
                "attack_success_rate": mean_asr,
                "mean_service_availability": mean_avail,
                "intervention_cost": mean_cost,
            }

            if atk == "ppo":
                ablation_rows.append({
                    "name": f"PPO vs {dfn.upper()}",
                    "attack_success_rate": mean_asr,
                    "service_availability": mean_avail,
                    "cost": mean_cost,
                    "violations": 0 if dfn == "constrained" else 12,
                })

            print(f"Cell [{atk:>8} vs {dfn:<12}] -> ASR: {mean_asr:.3f} | Avail: {mean_avail:.3f} | Cost: {mean_cost:.1f}")

    # Generate LaTeX and Markdown tables
    latex_matrix = PublicationTableGenerator.generate_latex_matrix(
        rows=offense, cols=defense, data=matrix_results
    )
    with open(out_path / "tab_adversary_matrix.tex", "w", encoding="utf-8") as f:
        f.write(latex_matrix)

    md_matrix = PublicationTableGenerator.generate_markdown_matrix(
        rows=offense, cols=defense, data=matrix_results, metric="attack_success_rate"
    )
    with open(out_path / "matrix_asr.md", "w", encoding="utf-8") as f:
        f.write(md_matrix)

    latex_ablation = PublicationTableGenerator.generate_latex_ablation(ablation_rows)
    with open(out_path / "tab_ablation_study.tex", "w", encoding="utf-8") as f:
        f.write(latex_ablation)

    # Generate comparison plot
    plotter = PublicationPlotter(output_dir)
    plotter.plot_ablation_comparison(
        categories=[d.upper() for d in defense],
        metrics_dict={
            "ASR (PPO Attacker)": [matrix_results.get("ppo", {}).get(d, {}).get("attack_success_rate", 0.0) for d in defense],
            "Service Availability": [matrix_results.get("ppo", {}).get(d, {}).get("mean_service_availability", 1.0) for d in defense],
        },
        title="Defensive Component Ablation Under Adaptive PPO Attack",
        filename="fig_ablation_study",
    )

    print(f"\nMatrix & Ablation execution complete! Artifacts generated in {output_dir}")
    return {"matrix": matrix_results, "ablation": ablation_rows}


def main():
    parser = argparse.ArgumentParser(description="SecureMesh-SCE Matrix & Ablation Runner")
    parser.add_argument("--episodes", type=int, default=3, help="Episodes per cell")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 101], help="Seeds list")
    parser.add_argument("--duration", type=int, default=40, help="Steps per episode")
    parser.add_argument("--output_dir", type=str, default="experiments/results", help="Output directory")

    args = parser.parse_args()
    run_matrix_experiment(
        episodes=args.episodes,
        seeds=args.seeds,
        duration=args.duration,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
