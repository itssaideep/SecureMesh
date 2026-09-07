# securemesh_testbed/experiments/matrix_runner.py
"""Full experiment matrix runner for paper-quality research results.

Runs every combination of attacker × defender × scenario × seed and
produces:
  - Per-cell summary statistics (mean ± std)
  - Cross-cell comparison tables (Markdown, CSV, LaTeX)
  - Statistical significance tests (Mann-Whitney U)
  - Effect size estimates (Cohen's d)

Usage
-----
    python -m securemesh_testbed.experiments.matrix_runner \\
        --seeds 5 \\
        --episodes 50 \\
        --output securemesh_testbed/results/matrix

    # From YAML config
    python -m securemesh_testbed.experiments.matrix_runner \\
        --config experiments/matrix_config.yaml
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .engine import SCEEngine
from .schemas import ExperimentSpec, MatrixSpec


# -------------------------------------------------------------------------
# Statistical utilities
# -------------------------------------------------------------------------

def mann_whitney_u(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Simple Mann-Whitney U test (no scipy dependency).

    Returns (U statistic, approximate p-value using normal approximation).
    """
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return 0.0, 1.0

    combined = np.concatenate([x, y])
    ranks = np.empty_like(combined, dtype=float)
    order = combined.argsort()
    ranks[order] = np.arange(1, len(combined) + 1, dtype=float)

    # Handle ties
    unique_vals = np.unique(combined)
    for val in unique_vals:
        mask = combined == val
        if mask.sum() > 1:
            ranks[mask] = ranks[mask].mean()

    u1 = ranks[:nx].sum() - nx * (nx + 1) / 2
    u2 = nx * ny - u1

    u_stat = min(u1, u2)

    # Normal approximation for p-value
    mu = nx * ny / 2
    sigma = np.sqrt(nx * ny * (nx + ny + 1) / 12)
    if sigma == 0:
        return u_stat, 1.0
    z = (u_stat - mu) / sigma
    # Two-tailed p-value approximation
    p = 2 * _norm_cdf(-abs(z))
    return float(u_stat), float(p)


def _norm_cdf(z: float) -> float:
    """Standard normal CDF approximation (Abramowitz & Stegun)."""
    import math
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Cohen's d effect size."""
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return 0.0
    pooled_std = np.sqrt(
        ((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1))
        / (nx + ny - 2)
    )
    if pooled_std == 0:
        return 0.0
    return float((np.mean(x) - np.mean(y)) / pooled_std)


# -------------------------------------------------------------------------
# Matrix Runner
# -------------------------------------------------------------------------

class MatrixRunner:
    """Run a full experiment matrix and produce comparison tables.

    Parameters
    ----------
    matrix : MatrixSpec
        Specification of the experiment grid.
    """

    def __init__(self, matrix: MatrixSpec):
        self.matrix = matrix
        self.engine = SCEEngine(output_base=matrix.output_dir)
        self.results: Dict[str, Dict[str, List[Dict[str, float]]]] = {}

    def run(self) -> pd.DataFrame:
        """Execute the full matrix and return a summary DataFrame."""
        os.makedirs(self.matrix.output_dir, exist_ok=True)

        combos = list(itertools.product(
            self.matrix.attackers,
            self.matrix.defenders,
            self.matrix.scenarios,
        ))

        total_runs = len(combos) * len(self.matrix.seeds)
        run_idx = 0
        t0 = time.time()

        print(f"\n{'='*60}")
        print(f"  EXPERIMENT MATRIX")
        print(f"  Attackers: {self.matrix.attackers}")
        print(f"  Defenders: {self.matrix.defenders}")
        print(f"  Scenarios: {self.matrix.scenarios}")
        print(f"  Seeds: {self.matrix.seeds}")
        print(f"  Total runs: {total_runs}")
        print(f"{'='*60}\n")

        for atk, dfn, scenario in combos:
            cell_key = f"{atk}_vs_{dfn}_{scenario}"
            cell_results = []

            for seed in self.matrix.seeds:
                run_idx += 1
                exp_id = f"MTX-{atk[:3].upper()}-{dfn[:3].upper()}-s{seed}"

                spec = ExperimentSpec(
                    id=exp_id,
                    name=f"{atk} vs {dfn} ({scenario})",
                    attacker=atk,
                    defender=dfn,
                    scenario=scenario,
                    duration=self.matrix.duration,
                    episodes=self.matrix.episodes_per_seed,
                    seed=seed,
                    tags=["matrix", atk, dfn, scenario],
                )

                cell_dir = os.path.join(
                    self.matrix.output_dir, cell_key, f"seed_{seed}"
                )

                print(f"\n  [{run_idx}/{total_runs}] "
                      f"{atk} vs {dfn} | {scenario} | seed={seed}")

                record = self.engine.run(spec, output_dir=cell_dir)
                cell_results.append(record.metrics)

            self.results[cell_key] = {
                "attacker": atk,
                "defender": dfn,
                "scenario": scenario,
                "runs": cell_results,
            }

        elapsed = time.time() - t0
        print(f"\n{'='*60}")
        print(f"  Matrix complete in {elapsed:.1f}s")
        print(f"{'='*60}\n")

        # Build comparison tables
        summary_df = self._build_summary_table()
        self._save_results(summary_df)
        return summary_df

    def _build_summary_table(self) -> pd.DataFrame:
        """Build a summary table with mean ± std for each cell."""
        rows = []
        key_metrics = [
            "attack_success_rate", "detection_rate", "false_positive_rate",
            "detection_latency", "system_resilience", "service_availability",
            "attacker_cumulative_reward", "defender_cumulative_reward",
        ]

        for cell_key, data in self.results.items():
            runs = data["runs"]
            if not runs:
                continue

            row = {
                "cell": cell_key,
                "attacker": data["attacker"],
                "defender": data["defender"],
                "scenario": data["scenario"],
                "n_seeds": len(runs),
            }

            for metric in key_metrics:
                vals = [r.get(metric, 0.0) for r in runs]
                row[f"{metric}_mean"] = float(np.mean(vals))
                row[f"{metric}_std"] = float(np.std(vals))
                row[f"{metric}_fmt"] = (
                    f"{np.mean(vals):.3f} ± {np.std(vals):.3f}"
                )

            rows.append(row)

        return pd.DataFrame(rows)

    def run_statistical_tests(self, baseline_attacker: str = "scripted",
                               target_attacker: str = "rl",
                               ) -> pd.DataFrame:
        """Run Mann-Whitney U tests comparing two attacker types."""
        key_metrics = [
            "attack_success_rate", "detection_rate", "system_resilience",
        ]

        rows = []
        for cell_key, data in self.results.items():
            if data["attacker"] not in (baseline_attacker, target_attacker):
                continue

        # Group by defender and scenario
        groups: Dict[str, Dict[str, List[float]]] = {}
        for cell_key, data in self.results.items():
            group_key = f"{data['defender']}_{data['scenario']}"
            if group_key not in groups:
                groups[group_key] = {}
            groups[group_key][data["attacker"]] = data["runs"]

        for group_key, attackers in groups.items():
            if baseline_attacker not in attackers or target_attacker not in attackers:
                continue

            baseline_runs = attackers[baseline_attacker]
            target_runs = attackers[target_attacker]

            for metric in key_metrics:
                baseline_vals = np.array(
                    [r.get(metric, 0.0) for r in baseline_runs]
                )
                target_vals = np.array(
                    [r.get(metric, 0.0) for r in target_runs]
                )

                u_stat, p_val = mann_whitney_u(baseline_vals, target_vals)
                d = cohens_d(target_vals, baseline_vals)

                rows.append({
                    "group": group_key,
                    "metric": metric,
                    "baseline_mean": float(np.mean(baseline_vals)),
                    "target_mean": float(np.mean(target_vals)),
                    "U_statistic": u_stat,
                    "p_value": p_val,
                    "cohens_d": d,
                    "significant": p_val < 0.05,
                })

        return pd.DataFrame(rows)

    def _save_results(self, summary_df: pd.DataFrame):
        """Save comparison tables in multiple formats."""
        out = self.matrix.output_dir

        # CSV
        csv_path = os.path.join(out, "matrix_summary.csv")
        summary_df.to_csv(csv_path, index=False)

        # Markdown
        md_path = os.path.join(out, "matrix_summary.md")
        self._save_markdown_table(summary_df, md_path)

        # LaTeX
        latex_path = os.path.join(out, "matrix_summary.tex")
        self._save_latex_table(summary_df, latex_path)

        # JSON (full results with all per-seed data)
        json_path = os.path.join(out, "matrix_results.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "matrix": {
                    "attackers": self.matrix.attackers,
                    "defenders": self.matrix.defenders,
                    "scenarios": self.matrix.scenarios,
                    "seeds": self.matrix.seeds,
                },
                "results": {k: {
                    "attacker": v["attacker"],
                    "defender": v["defender"],
                    "scenario": v["scenario"],
                    "runs": v["runs"],
                } for k, v in self.results.items()},
            }, f, indent=2, default=str)

        print(f"  Summary CSV: {csv_path}")
        print(f"  Summary Markdown: {md_path}")
        print(f"  Summary LaTeX: {latex_path}")
        print(f"  Full results JSON: {json_path}")

    def _save_markdown_table(self, df: pd.DataFrame, path: str):
        """Generate a publication-ready Markdown comparison table."""
        key_metrics = [
            "attack_success_rate", "detection_rate", "system_resilience",
        ]

        lines = ["# Experiment Matrix Results\n"]
        lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}\n")

        # Pivot: rows = attackers, columns = defenders
        for scenario in df["scenario"].unique():
            sdf = df[df["scenario"] == scenario]
            lines.append(f"\n## Scenario: {scenario}\n")

            # Build pivot table
            attackers = sorted(sdf["attacker"].unique())
            defenders = sorted(sdf["defender"].unique())

            for metric in key_metrics:
                col = f"{metric}_fmt"
                lines.append(f"\n### {metric.replace('_', ' ').title()}\n")
                header = "| Attacker | " + " | ".join(defenders) + " |"
                sep = "|---|" + "|".join(["---"] * len(defenders)) + "|"
                lines.append(header)
                lines.append(sep)

                for atk in attackers:
                    row_vals = []
                    for dfn in defenders:
                        match = sdf[
                            (sdf["attacker"] == atk) & (sdf["defender"] == dfn)
                        ]
                        if len(match) > 0:
                            row_vals.append(match.iloc[0][col])
                        else:
                            row_vals.append("—")
                    lines.append(
                        f"| {atk} | " + " | ".join(row_vals) + " |"
                    )

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def _save_latex_table(self, df: pd.DataFrame, path: str):
        """Generate a LaTeX table for paper inclusion."""
        key_metrics = [
            "attack_success_rate", "detection_rate", "system_resilience",
        ]

        lines = [
            "% Auto-generated by SecureMesh-SCE matrix runner",
            "\\begin{table}[htbp]",
            "\\centering",
            "\\caption{Experiment Matrix Results}",
            "\\label{tab:matrix_results}",
        ]

        defenders = sorted(df["defender"].unique())
        n_cols = len(defenders) + 1
        lines.append("\\begin{tabular}{l" + "c" * len(defenders) + "}")
        lines.append("\\toprule")
        lines.append(
            "Attacker & " + " & ".join(
                d.replace("_", "\\_") for d in defenders
            ) + " \\\\"
        )
        lines.append("\\midrule")

        for metric in key_metrics:
            col = f"{metric}_fmt"
            lines.append(
                f"\\multicolumn{{{n_cols}}}{{l}}"
                f"{{\\textit{{{metric.replace('_', ' ').title()}}}}}"
                " \\\\"
            )
            attackers = sorted(df["attacker"].unique())
            for atk in attackers:
                row_vals = []
                for dfn in defenders:
                    match = df[
                        (df["attacker"] == atk) & (df["defender"] == dfn)
                    ]
                    if len(match) > 0:
                        val = match.iloc[0][col]
                        row_vals.append(f"${val}$")
                    else:
                        row_vals.append("—")
                lines.append(
                    f"{atk.replace('_', ' ')} & "
                    + " & ".join(row_vals) + " \\\\"
                )
            lines.append("\\addlinespace")

        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ])

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SecureMesh-SCE — Experiment Matrix Runner")
    parser.add_argument("--config", default=None,
                        help="Path to YAML matrix configuration")
    parser.add_argument("--attackers", nargs="+",
                        default=["scripted", "aggressive", "stealthy",
                                 "recon_heavy", "rl"],
                        help="Attacker types to include")
    parser.add_argument("--defenders", nargs="+",
                        default=["static", "rl"],
                        help="Defender types to include")
    parser.add_argument("--scenarios", nargs="+",
                        default=["known_attacks"],
                        help="Scenarios to run")
    parser.add_argument("--seeds", type=int, default=3,
                        help="Number of random seeds per combination")
    parser.add_argument("--episodes", type=int, default=50,
                        help="Episodes per seed")
    parser.add_argument("--duration", type=int, default=100,
                        help="Steps per episode")
    parser.add_argument("--output", default="securemesh_testbed/results/matrix",
                        help="Output directory")
    args = parser.parse_args()

    if args.config:
        matrix = MatrixSpec.from_yaml(args.config)
    else:
        matrix = MatrixSpec(
            attackers=args.attackers,
            defenders=args.defenders,
            scenarios=args.scenarios,
            seeds=list(range(42, 42 + args.seeds)),
            episodes_per_seed=args.episodes,
            duration=args.duration,
            output_dir=args.output,
        )

    runner = MatrixRunner(matrix)
    summary = runner.run()

    # Run statistical tests if we have enough data
    if "scripted" in matrix.attackers and "rl" in matrix.attackers:
        print("\n  Running statistical tests (scripted vs rl)...")
        stats = runner.run_statistical_tests("scripted", "rl")
        if not stats.empty:
            stats_path = os.path.join(args.output, "statistical_tests.csv")
            stats.to_csv(stats_path, index=False)
            print(f"  Statistical tests: {stats_path}")
            print(stats.to_string(index=False))


if __name__ == "__main__":
    main()
