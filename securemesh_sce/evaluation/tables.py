# securemesh_sce/evaluation/tables.py
"""Publication-ready LaTeX and Markdown table generator.

Generates research-paper-ready LaTeX booktabs tables and Markdown summaries
for attacker-defender cross-product matrices, ablation comparisons, and risk registers.
"""

from __future__ import annotations

from typing import Dict, List, Any, Optional
from pathlib import Path


class PublicationTableGenerator:
    """Generates LaTeX and Markdown formatted research tables."""

    @staticmethod
    def generate_latex_matrix(
        rows: List[str],
        cols: List[str],
        data: Dict[str, Dict[str, Dict[str, float]]],
        primary_metric: str = "attack_success_rate",
        secondary_metric: str = "mean_service_availability",
    ) -> str:
        """Generate a 2-metric cross-product table in LaTeX booktabs format.

        data structure: data[row][col] -> dict of metric values
        """
        latex = [
            "\\begin{table*}[t]",
            "\\centering",
            "\\caption{Attacker-Defender Co-adaptation Matrix: Attack Success Rate (ASR) $\\downarrow$ and Service Availability $\\uparrow$ (Mean $\\pm$ Std).}",
            "\\label{tab:adversary_matrix}",
            "\\begin{tabular}{l" + "c" * len(cols) + "}",
            "\\toprule",
            "\\textbf{Attacker Level} & " + " & ".join([f"\\textbf{{{c}}}" for c in cols]) + " \\\\",
            "\\midrule",
        ]

        for r in rows:
            line_cells = [f"\\textbf{{{r}}}"]
            for c in cols:
                cell_dict = data.get(r, {}).get(c, {})
                m1 = cell_dict.get(primary_metric, 0.0)
                m2 = cell_dict.get(secondary_metric, 1.0)
                line_cells.append(f"{m1:.2f} / {m2:.2f}")
            latex.append(" & ".join(line_cells) + " \\\\")

        latex.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table*}",
        ])
        return "\n".join(latex)

    @staticmethod
    def generate_markdown_matrix(
        rows: List[str],
        cols: List[str],
        data: Dict[str, Dict[str, Dict[str, float]]],
        metric: str = "attack_success_rate",
    ) -> str:
        """Generate Markdown table for documentation and notebooks."""
        header = ["| Attacker Level | " + " | ".join(cols) + " |"]
        sep = ["|:---| " + " | ".join([":---:"] * len(cols)) + " |"]
        lines = header + sep

        for r in rows:
            cells = [f"**{r}**"]
            for c in cols:
                val = data.get(r, {}).get(c, {}).get(metric, 0.0)
                cells.append(f"{val:.3f}")
            lines.append("| " + " | ".join(cells) + " |")

        return "\n".join(lines)

    @staticmethod
    def generate_latex_ablation(
        ablation_results: List[Dict[str, Any]]
    ) -> str:
        """Generate LaTeX table for defense ablation study with significance."""
        latex = [
            "\\begin{table}[h]",
            "\\centering",
            "\\caption{Ablation Study of Defensive Components Under Adaptive PPO Attack.}",
            "\\label{tab:ablation_study}",
            "\\begin{tabular}{lcccc}",
            "\\toprule",
            "\\textbf{Configuration} & \\textbf{ASR} $\\downarrow$ & \\textbf{Availability} $\\uparrow$ & \\textbf{Cost} $\\downarrow$ & \\textbf{Violations} $\\downarrow$ \\\\",
            "\\midrule",
        ]

        for res in ablation_results:
            name = res.get("name", "Config")
            asr = res.get("attack_success_rate", 0.0)
            avail = res.get("service_availability", 1.0)
            cost = res.get("cost", 0.0)
            viol = res.get("violations", 0)
            latex.append(f"{name} & {asr:.3f} & {avail:.3f} & {cost:.1f} & {viol} \\\\")

        latex.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ])
        return "\n".join(latex)
