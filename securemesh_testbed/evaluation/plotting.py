# securemesh_testbed/evaluation/plotting.py
"""Matplotlib-based plot generation for experiment results.

All functions accept pandas DataFrames (loaded from the CSV produced by
ExperimentLogger) and save publication-ready PNGs to the output
directory.
"""

from __future__ import annotations

import os
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_reward_curves(df: pd.DataFrame, output_dir: str,
                       filename: str = "reward_curves.png"):
    """Plot attacker vs defender cumulative reward over episodes."""
    fig, ax = plt.subplots(figsize=(10, 5))
    if "attacker_cumulative_reward" in df.columns:
        ax.plot(df["episode"], df["attacker_cumulative_reward"],
                label="Attacker", color="#e74c3c", linewidth=1.5)
    if "defender_cumulative_reward" in df.columns:
        ax.plot(df["episode"], df["defender_cumulative_reward"],
                label="Defender", color="#2ecc71", linewidth=1.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative Reward")
    ax.set_title("Reward Curves — Attacker vs Defender")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, filename), dpi=150)
    plt.close(fig)


def plot_detection_rate(df: pd.DataFrame, output_dir: str,
                         filename: str = "detection_rate.png"):
    """Plot detection rate (TPR) over episodes."""
    if "detection_rate" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["episode"], df["detection_rate"],
            color="#3498db", linewidth=1.5)
    ax.fill_between(df["episode"], 0, df["detection_rate"],
                     alpha=0.15, color="#3498db")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Detection Rate (TPR)")
    ax.set_title("Detection Rate Over Time")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, filename), dpi=150)
    plt.close(fig)


def plot_service_availability(df: pd.DataFrame, output_dir: str,
                               filename: str = "service_availability.png"):
    """Plot service availability over episodes."""
    if "service_availability" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(df["episode"], df["service_availability"],
           color="#9b59b6", alpha=0.7)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Service Availability")
    ax.set_title("Service Availability per Episode")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, filename), dpi=150)
    plt.close(fig)


def plot_metric_comparison(metrics: dict, output_dir: str,
                            filename: str = "metric_comparison.png"):
    """Bar chart comparing all scalar metrics for a single run."""
    fig, ax = plt.subplots(figsize=(12, 5))
    names = list(metrics.keys())
    values = [float(metrics[n]) for n in names]
    colours = plt.cm.viridis(np.linspace(0.2, 0.8, len(names)))
    ax.barh(names, values, color=colours)
    ax.set_xlabel("Value")
    ax.set_title("Metric Summary")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, filename), dpi=150)
    plt.close(fig)


def plot_belief_evolution(belief_history: list, output_dir: str,
                           filename: str = "belief_evolution.png"):
    """Plot the defender's Bayesian belief over attacker types."""
    if not belief_history:
        return
    arr = np.array(belief_history)  # (T, n_types)
    fig, ax = plt.subplots(figsize=(10, 4))
    labels = ["opportunistic", "sophisticated", "stealth"]
    colours = ["#f39c12", "#e74c3c", "#1abc9c"]
    for i, (lbl, clr) in enumerate(zip(labels, colours)):
        if i < arr.shape[1]:
            ax.plot(arr[:, i], label=lbl, color=clr, linewidth=1.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("P(attacker type)")
    ax.set_title("Defender Belief Evolution")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, filename), dpi=150)
    plt.close(fig)


def generate_all_plots(df: pd.DataFrame, output_dir: str,
                        metrics: Optional[dict] = None,
                        belief_history: Optional[list] = None):
    """Convenience wrapper that generates every available plot."""
    os.makedirs(output_dir, exist_ok=True)
    plot_reward_curves(df, output_dir)
    plot_detection_rate(df, output_dir)
    plot_service_availability(df, output_dir)
    if metrics:
        plot_metric_comparison(metrics, output_dir)
    if belief_history:
        plot_belief_evolution(belief_history, output_dir)
