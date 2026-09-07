# securemesh_testbed/experiments/schemas.py
"""Pydantic models for YAML-driven SCE experiment definitions.

Each experiment specifies a target, attack strategy, defender policy,
duration, and a testable hypothesis following the SCENE methodology.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import json
import os
import yaml


# ---------------------------------------------------------------------------
# Enums for validated configuration
# ---------------------------------------------------------------------------

class AttackerType(str, Enum):
    RANDOM = "random"
    SCRIPTED = "scripted"
    AGGRESSIVE = "aggressive"
    STEALTHY = "stealthy"
    RECON_HEAVY = "recon_heavy"
    RL = "rl"
    LLM = "llm"


class DefenderType(str, Enum):
    STATIC = "static"
    ML = "ml"
    RL = "rl"


class HypothesisDirection(str, Enum):
    BELOW = "below"
    ABOVE = "above"


# ---------------------------------------------------------------------------
# Data classes (kept as dataclasses for zero-dependency compat)
# ---------------------------------------------------------------------------

@dataclass
class HypothesisSpec:
    """A testable hypothesis for an SCE experiment."""
    text: str
    metric: str = "attack_success_rate"
    threshold: float = 0.5
    direction: str = "below"  # "below" means metric should be < threshold

    def evaluate(self, metrics: Dict[str, float]) -> bool:
        """Return True if the hypothesis passes."""
        value = metrics.get(self.metric, 0.0)
        if self.direction == "below":
            return value < self.threshold
        else:
            return value > self.threshold


@dataclass
class ExperimentSpec:
    """Full specification for a single SCE experiment."""
    id: str = "EXP-001"
    name: str = "Unnamed Experiment"
    target: str = "all"
    attacker: str = "scripted"
    defender: str = "static"
    scenario: str = "known_attacks"
    duration: int = 100  # max steps per episode
    episodes: int = 50
    seed: int = 42
    expected_state: str = "service_available"
    hypothesis: Optional[HypothesisSpec] = None
    tags: List[str] = field(default_factory=list)
    description: str = ""

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentSpec":
        """Load an experiment specification from a YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        exp_data = raw.get("experiment", {})
        hyp_data = raw.get("hypothesis", None)

        hypothesis = None
        if hyp_data:
            if isinstance(hyp_data, str):
                hypothesis = HypothesisSpec(text=hyp_data)
            elif isinstance(hyp_data, dict):
                hypothesis = HypothesisSpec(**hyp_data)

        return cls(
            id=exp_data.get("id", "EXP-001"),
            name=exp_data.get("name", "Unnamed Experiment"),
            target=exp_data.get("target", "all"),
            attacker=exp_data.get("attacker", "scripted"),
            defender=exp_data.get("defender", "static"),
            scenario=exp_data.get("scenario", "known_attacks"),
            duration=exp_data.get("duration", 100),
            episodes=exp_data.get("episodes", 50),
            seed=exp_data.get("seed", 42),
            expected_state=exp_data.get("expected_state", "service_available"),
            hypothesis=hypothesis,
            tags=exp_data.get("tags", []),
            description=exp_data.get("description", ""),
        )


@dataclass
class MatrixSpec:
    """Specification for a full experiment matrix (attacker × defender)."""
    attackers: List[str] = field(default_factory=lambda: ["scripted", "rl"])
    defenders: List[str] = field(default_factory=lambda: ["static", "rl"])
    scenarios: List[str] = field(default_factory=lambda: ["known_attacks"])
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456])
    episodes_per_seed: int = 50
    duration: int = 100
    output_dir: str = "securemesh_testbed/results/matrix"

    @classmethod
    def from_yaml(cls, path: str) -> "MatrixSpec":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        data = raw.get("matrix", raw)
        return cls(
            attackers=data.get("attackers", ["scripted", "rl"]),
            defenders=data.get("defenders", ["static", "rl"]),
            scenarios=data.get("scenarios", ["known_attacks"]),
            seeds=data.get("seeds", [42, 123, 456]),
            episodes_per_seed=data.get("episodes_per_seed", 50),
            duration=data.get("duration", 100),
            output_dir=data.get("output_dir", "securemesh_testbed/results/matrix"),
        )


@dataclass
class ExperimentRecord:
    """Structured record of a completed SCE experiment.

    This is the primary output artifact — every experiment produces one of these.
    """
    experiment_id: str
    timestamp: str
    system_state_initial: Dict[str, Any] = field(default_factory=dict)
    system_state_final: Dict[str, Any] = field(default_factory=dict)
    attacker_policy: str = ""
    defender_policy: str = ""
    target: str = ""
    scenario: str = ""
    seed: int = 0
    episodes: int = 0
    duration: int = 0
    hypothesis: Optional[Dict[str, Any]] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    timeline: List[Dict[str, Any]] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    def to_json(self, path: str):
        """Serialise the experiment record to JSON."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.__dict__, f, indent=2, default=str)

    @classmethod
    def from_json(cls, path: str) -> "ExperimentRecord":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        hyp = data.pop("hypothesis", None)
        return cls(**data, hypothesis=hyp)
