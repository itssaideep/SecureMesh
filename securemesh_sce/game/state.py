# securemesh_sce/game/state.py
"""Defender state representation: s_t = [x_t, b_t, h_t, r_t].

Components
----------
x_t : system state
    Flattened network/service/IoT state vector.
b_t : belief distribution
    Full posterior P(θ_i | O_{0:t}) over attacker types.
h_t : attack history
    Exponential sliding-window summary of recent attacker behaviour.
r_t : risk state
    Current risk/resilience indicators.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List

import numpy as np

from .actions import AttackerType, N_ATTACKER_TYPES


# =====================================================================
# System state flattening
# =====================================================================

def flatten_network_state(snapshot: Dict[str, Any]) -> np.ndarray:
    """Convert a network snapshot dict into a 1-D NumPy array.

    Expected structure::

        {
            "hosts": {
                hostname: {
                    "isolated": bool,
                    "services": {
                        svc_name: {
                            "compromised": bool,
                            "active_sessions": int,
                            "port": int,
                            "vulnerability": float,
                        }
                    }
                }
            },
            "iot_devices": {
                device_id: {
                    "isolated": bool,
                    "compromised": bool,
                    "firmware_integrity": float,
                }
            }
        }
    """
    parts: List[float] = []

    # Hosts
    for hostname, hdata in sorted(snapshot.get("hosts", {}).items()):
        parts.append(1.0 if hdata.get("isolated", False) else 0.0)
        for svc_name, svc in sorted(hdata.get("services", {}).items()):
            parts.append(1.0 if svc.get("compromised", False) else 0.0)
            parts.append(float(svc.get("active_sessions", 0)))
            parts.append(float(svc.get("vulnerability", 0.0)))

    # IoT devices
    for dev_id, ddata in sorted(snapshot.get("iot_devices", {}).items()):
        parts.append(1.0 if ddata.get("isolated", False) else 0.0)
        parts.append(1.0 if ddata.get("compromised", False) else 0.0)
        parts.append(float(ddata.get("firmware_integrity", 1.0)))

    return np.array(parts, dtype=np.float32)


# =====================================================================
# Attack history summary
# =====================================================================

class AttackHistory:
    """Exponential sliding-window summary of recent attacker behaviour.

    Tracks feature counts over the last ``window`` steps with exponential
    decay, producing a fixed-size feature vector for the defender.
    """

    # Features tracked
    FEATURE_NAMES = [
        "scan_rate",
        "auth_attempts",
        "exploit_attempts",
        "malware_attempts",
        "persistence_attempts",
        "evasion_actions",
        "lateral_moves",
        "connection_rate",
        "unique_targets",
        "noop_rate",
    ]

    def __init__(self, window: int = 20, decay: float = 0.9):
        self.window = window
        self.decay = decay
        self._counts = np.zeros(len(self.FEATURE_NAMES), dtype=np.float32)
        self._step = 0

    @property
    def dim(self) -> int:
        return len(self.FEATURE_NAMES)

    def update(self, features: Dict[str, float]):
        """Update history with new observation features."""
        self._counts *= self.decay
        for i, name in enumerate(self.FEATURE_NAMES):
            self._counts[i] += features.get(name, 0.0)
        self._step += 1

    def vector(self) -> np.ndarray:
        """Return the current history feature vector."""
        return self._counts.copy()

    def reset(self):
        self._counts[:] = 0.0
        self._step = 0


# =====================================================================
# Risk state
# =====================================================================

@dataclass
class RiskState:
    """Current risk and resilience indicators.

    Attributes
    ----------
    cumulative_impact : float
        Running sum of severity-weighted attack impact (0..∞).
    service_availability : float
        Fraction of services currently operational (0..1).
    recovery_debt : float
        Accumulated recovery cost from past incidents.
    security_stage : int
        Current position in the NORMAL→RECOVER workflow.
    """
    cumulative_impact: float = 0.0
    service_availability: float = 1.0
    recovery_debt: float = 0.0
    security_stage: int = 0   # maps to SecurityStage.value

    def vector(self) -> np.ndarray:
        return np.array([
            self.cumulative_impact,
            self.service_availability,
            self.recovery_debt,
            float(self.security_stage),
        ], dtype=np.float32)

    @property
    def dim(self) -> int:
        return 4

    def reset(self):
        self.cumulative_impact = 0.0
        self.service_availability = 1.0
        self.recovery_debt = 0.0
        self.security_stage = 0


# =====================================================================
# Composite defender state
# =====================================================================

@dataclass
class DefenderState:
    """Full defender state s_t = [x_t, b_t, h_t, r_t].

    The defender policy operates on the concatenation of these components.
    """
    system_obs: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float32))
    belief: np.ndarray = field(
        default_factory=lambda: np.ones(N_ATTACKER_TYPES, dtype=np.float32) / N_ATTACKER_TYPES
    )
    history: np.ndarray = field(
        default_factory=lambda: np.zeros(len(AttackHistory.FEATURE_NAMES), dtype=np.float32)
    )
    risk: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32))

    def flat(self) -> np.ndarray:
        """Return the full concatenated observation vector."""
        return np.concatenate([self.system_obs, self.belief, self.history, self.risk])

    @property
    def dim(self) -> int:
        return len(self.system_obs) + len(self.belief) + len(self.history) + len(self.risk)
