# securemesh_testbed/evaluation/logger.py
"""Structured logging for experiment runs.

Writes step-level and episode-level data to JSON Lines (`.jsonl`) and
a summary CSV.  All output goes to a configurable output directory.
"""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List


class ExperimentLogger:
    """Thread-safe logger that writes JSON Lines and CSV.

    Parameters
    ----------
    output_dir : str
        Directory where log files are created.
    experiment_name : str
        Prefix for log filenames.
    """

    def __init__(self, output_dir: str, experiment_name: str = "experiment"):
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        self.experiment_name = experiment_name
        self._step_path = os.path.join(output_dir, f"{experiment_name}_steps.jsonl")
        self._episode_path = os.path.join(output_dir, f"{experiment_name}_episodes.csv")
        self._meta_path = os.path.join(output_dir, f"{experiment_name}_meta.json")
        self._step_file = open(self._step_path, "w", encoding="utf-8")
        self._episode_rows: List[Dict[str, Any]] = []
        self._current_episode = 0

    # --- step-level logging --------------------------------------------------
    def log_step(self, episode: int, step: int, info: Dict[str, Any]):
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "episode": episode,
            "step": step,
            **info,
        }
        self._step_file.write(json.dumps(record) + "\n")

    # --- episode-level logging -----------------------------------------------
    def log_episode(self, episode: int, metrics: Dict[str, float]):
        row = {"episode": episode, **metrics}
        self._episode_rows.append(row)

    # --- metadata ------------------------------------------------------------
    def log_meta(self, meta: Dict[str, Any]):
        with open(self._meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, default=str)

    # --- finalise ------------------------------------------------------------
    def close(self):
        self._step_file.close()
        if self._episode_rows:
            fieldnames = list(self._episode_rows[0].keys())
            with open(self._episode_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self._episode_rows)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
