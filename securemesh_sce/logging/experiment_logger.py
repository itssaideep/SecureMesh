# securemesh_sce/logging/experiment_logger.py
"""Structured JSONL/CSV/JSON experiment telemetry logger.

Supports SCENE-compliant reproducible experiment outputs:
1. Per-step telemetry stream (JSONL)
2. Per-episode metrics and hypothesis evaluation (CSV)
3. Experiment provenance, config, seeds, and metadata (JSON)
"""

from __future__ import annotations

import os
import json
import csv
import time
from typing import Dict, Any, List, Optional
from pathlib import Path


class ExperimentLogger:
    """Structured logger for SCENE experiments."""

    def __init__(self, output_dir: str, experiment_id: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_id = experiment_id

        self.jsonl_path = self.output_dir / f"{experiment_id}_telemetry.jsonl"
        self.csv_path = self.output_dir / f"{experiment_id}_episodes.csv"
        self.summary_json_path = self.output_dir / f"{experiment_id}_summary.json"

        self._csv_writer = None
        self._csv_file = None
        self._csv_headers_written = False

    def log_step(self, step_record: Dict[str, Any]):
        """Append a single step record to the JSONL log."""
        record = {
            "experiment_id": self.experiment_id,
            "timestamp": time.time(),
            **step_record,
        }
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def log_episode_summary(self, summary: Dict[str, Any]):
        """Append an episode summary row to CSV."""
        record = {
            "experiment_id": self.experiment_id,
            "timestamp": time.time(),
            **summary,
        }

        file_exists = self.csv_path.exists()
        fieldnames = list(record.keys())

        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists or not self._csv_headers_written:
                writer.writeheader()
                self._csv_headers_written = True
            writer.writerow(record)

    def write_experiment_summary(self, experiment_data: Dict[str, Any]):
        """Write full structured metadata and final experiment report to JSON."""
        with open(self.summary_json_path, "w", encoding="utf-8") as f:
            json.dump(experiment_data, f, indent=2, default=str)
