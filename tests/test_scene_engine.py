# tests/test_scene_engine.py
"""Unit tests for SCENE Experiment Engine lifecycle and hypothesis evaluation."""

import pytest
from pathlib import Path

from securemesh_sce.experiments.engine import SCENEExperimentEngine, ExperimentConfig
from securemesh_sce.evaluation.statistics import StatisticalAnalyzer


def test_experiment_config_parsing(tmp_path):
    yaml_content = """
experiment:
  id: "SCE-TEST"
  name: "Unit Test Scenario"
  attacker: "scripted"
  defender: "static"
  duration: 10
  episodes: 2
  seed: 123
hypothesis:
  text: "Test hypothesis text"
  metric: "detection_rate"
  threshold: 0.5
  direction: "above"
"""
    yaml_file = tmp_path / "test_scenario.yaml"
    yaml_file.write_text(yaml_content, encoding="utf-8")

    cfg = ExperimentConfig.from_yaml(yaml_file)
    assert cfg.id == "SCE-TEST"
    assert cfg.attacker_name == "scripted"
    assert cfg.defender_name == "static"
    assert cfg.duration == 10
    assert cfg.episodes == 2
    assert cfg.seed == 123


def test_scene_engine_run(tmp_path):
    cfg = ExperimentConfig(
        id="SCE-T01",
        name="Engine Smoke Test",
        attacker_name="scripted",
        defender_name="static",
        duration=15,
        episodes=2,
        seed=42,
        hypothesis={"metric": "service_availability", "threshold": 0.8, "direction": "above"},
    )
    engine = SCENEExperimentEngine(config=cfg, output_dir=str(tmp_path))
    summary = engine.run()

    assert "experiment_id" in summary
    assert "hypothesis_result" in summary
    assert "aggregate_metrics" in summary
    assert "risk_register" in summary

    # Verify generated artifact files exist
    assert (tmp_path / "SCE-T01_telemetry.jsonl").exists()
    assert (tmp_path / "SCE-T01_episodes.csv").exists()
    assert (tmp_path / "SCE-T01_summary.json").exists()


def test_statistical_hypothesis_evaluation():
    hyp_config = {
        "text": "Detection rate should exceed 60%",
        "metric": "detection_rate",
        "threshold": 0.6,
        "direction": "above",
    }
    # Passing values
    res_pass = StatisticalAnalyzer.evaluate_hypothesis(hyp_config, [0.7, 0.8, 0.75, 0.65])
    assert res_pass.passed is True
    assert res_pass.verdict == "PASS"

    # Failing values
    res_fail = StatisticalAnalyzer.evaluate_hypothesis(hyp_config, [0.3, 0.4, 0.35, 0.25])
    assert res_fail.passed is False
    assert res_fail.verdict == "FAIL"
