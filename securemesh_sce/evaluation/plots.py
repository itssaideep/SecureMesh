# securemesh_sce/evaluation/plots.py
"""Publication-ready plot generation (PNG and SVG).

Generates research-grade figures:
1. Belief convergence trajectory over steps
2. Reliability diagram (Calibration curve with ECE)
3. Attacker vs Defender ablation comparisons (ASR, Availability, Cost)
4. Security vs Availability Pareto trade-offs
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Headless rendering
import matplotlib.pyplot as plt


class PublicationPlotter:
    """Generates styled publication plots."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._set_style()

    def _set_style(self):
        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.titlesize": 13,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
        })

    def plot_belief_trajectory(
        self,
        belief_history: np.ndarray,
        true_type_idx: int,
        type_names: List[str],
        filename: str = "fig_belief_trajectory",
    ) -> str:
        """Plot posterior belief evolution b_t(theta) over episode steps."""
        fig, ax = plt.subplots(figsize=(7, 4), dpi=300)
        steps = np.arange(len(belief_history))
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

        for i, name in enumerate(type_names):
            label = f"{name} {'(Ground Truth)' if i == true_type_idx else ''}"
            lw = 2.5 if i == true_type_idx else 1.2
            ls = "-" if i == true_type_idx else "--"
            ax.plot(steps, belief_history[:, i], label=label, color=colors[i % len(colors)], linewidth=lw, linestyle=ls)

        ax.set_xlabel("Time Step (t)")
        ax.set_ylabel("Posterior Belief $b_t(\\theta)$")
        ax.set_title("Bayesian Belief Convergence Over Hidden Attacker Types")
        ax.set_ylim(-0.02, 1.05)
        ax.legend(loc="upper left", framealpha=0.9)
        fig.tight_layout()

        out_png = self.output_dir / f"{filename}.png"
        out_svg = self.output_dir / f"{filename}.svg"
        fig.savefig(out_png)
        fig.savefig(out_svg)
        plt.close(fig)
        return str(out_png)

    def plot_calibration_curve(
        self,
        brier_score: float,
        ece: float,
        bin_confs: np.ndarray,
        bin_accs: np.ndarray,
        filename: str = "fig_calibration_curve",
    ) -> str:
        """Plot reliability diagram for Bayesian posterior calibration."""
        fig, ax = plt.subplots(figsize=(5, 5), dpi=300)
        ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
        ax.plot(bin_confs, bin_accs, "s-", color="#2b5c8f", label=f"Bayesian Belief (ECE={ece:.3f})")
        ax.set_xlabel("Mean Predicted Confidence")
        ax.set_ylabel("Empirical Accuracy")
        ax.set_title(f"Reliability Diagram (Brier={brier_score:.3f})")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc="upper left")
        fig.tight_layout()

        out_png = self.output_dir / f"{filename}.png"
        fig.savefig(out_png)
        plt.close(fig)
        return str(out_png)

    def plot_ablation_comparison(
        self,
        categories: List[str],
        metrics_dict: Dict[str, List[float]],
        title: str = "Defender Policy Comparison Across Offense Spectrum",
        filename: str = "fig_ablation_comparison",
    ) -> str:
        """Bar plot comparing policies across multiple metrics."""
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
        x = np.arange(len(categories))
        width = 0.8 / len(metrics_dict)

        colors = ["#4c72b0", "#55a868", "#c44e52", "#8172b3", "#ccb974"]
        for i, (metric_name, vals) in enumerate(metrics_dict.items()):
            pos = x - 0.4 + (i + 0.5) * width
            ax.bar(pos, vals, width, label=metric_name, color=colors[i % len(colors)], alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.set_ylabel("Score / Rate")
        ax.set_title(title)
        ax.set_ylim(0.0, 1.1)
        ax.legend(loc="upper right")
        fig.tight_layout()

        out_png = self.output_dir / f"{filename}.png"
        fig.savefig(out_png)
        plt.close(fig)
        return str(out_png)
