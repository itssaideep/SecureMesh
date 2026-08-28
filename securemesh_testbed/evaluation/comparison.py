# securemesh_testbed/evaluation/comparison.py
"""Cross-experiment comparison tables and statistical tests.

Loads episode-level CSV files from multiple experiment runs and produces
a summary table with mean ± std for each metric across random seeds.
"""

from __future__ import annotations

import os
from typing import Dict, List

import numpy as np
import pandas as pd


def load_episode_csvs(paths: List[str]) -> pd.DataFrame:
    """Load and concatenate multiple episode CSV files."""
    frames = []
    for i, path in enumerate(paths):
        df = pd.read_csv(path)
        df["run"] = i
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def summary_table(df: pd.DataFrame,
                   metrics: List[str] | None = None) -> pd.DataFrame:
    """Compute mean ± std for each metric across runs.

    Returns a DataFrame with columns: metric, mean, std, formatted.
    """
    if metrics is None:
        metrics = [c for c in df.columns
                   if c not in ("episode", "run")]
    rows = []
    for m in metrics:
        if m not in df.columns:
            continue
        vals = df.groupby("run")[m].mean()
        rows.append({
            "metric": m,
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "formatted": f"{vals.mean():.4f} ± {vals.std():.4f}",
        })
    return pd.DataFrame(rows)


def comparison_table(experiment_dirs: Dict[str, str],
                      csv_pattern: str = "*_episodes.csv") -> pd.DataFrame:
    """Build a comparison table across named experiments.

    Parameters
    ----------
    experiment_dirs : dict
        Mapping from experiment label (e.g. "static_defender") to the
        directory containing its episode CSV(s).
    csv_pattern : str
        Glob pattern to find CSV files in each directory.

    Returns
    -------
    pd.DataFrame
        Wide-format table with one row per metric and one column per
        experiment (formatted as mean ± std).
    """
    import glob
    tables = {}
    for label, dirpath in experiment_dirs.items():
        csvs = sorted(glob.glob(os.path.join(dirpath, csv_pattern)))
        if not csvs:
            continue
        combined = load_episode_csvs(csvs)
        tables[label] = summary_table(combined)

    if not tables:
        return pd.DataFrame()

    # merge into a single wide table
    all_metrics = sorted({m for t in tables.values() for m in t["metric"]})
    rows = []
    for metric in all_metrics:
        row: Dict[str, str] = {"metric": metric}
        for label, tbl in tables.items():
            match = tbl[tbl["metric"] == metric]
            row[label] = match["formatted"].values[0] if len(match) else "—"
        rows.append(row)
    return pd.DataFrame(rows)


def save_comparison(table: pd.DataFrame, output_path: str):
    """Save the comparison table as both CSV and a Markdown file."""
    table.to_csv(output_path, index=False)
    md_path = output_path.replace(".csv", ".md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(table.to_markdown(index=False))
