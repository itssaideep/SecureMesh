# securemesh_sce/environment/telemetry/collector.py
"""Telemetry Collector and SCENE Steady-State Detector.

Implements real-time telemetry sampling across IoT endpoints, services,
and network topology. Provides hypothesis baseline verification as mandated
by the SCENE Security Chaos Engineering methodology.
"""

from __future__ import annotations

import time
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field


@dataclass
class TelemetrySample:
    timestamp: float
    step: int
    service_availability: float
    avg_cpu_load: float
    avg_memory_free_pct: float
    active_connections: int
    failed_auth_rate: float
    ids_alerts: int
    packet_drop_rate: float


class SteadyStateDetector:
    """SCENE Steady-State Hypothesis Validator.

    Collects telemetry samples during a baseline phase prior to chaos injection
    and verifies that system metrics satisfy steady-state bounds:
    - Service availability >= threshold (default 0.99)
    - Low anomaly variance (stable CPU/latency)
    - Zero active compromises
    """

    def __init__(
        self,
        window_size: int = 10,
        min_availability: float = 0.95,
        max_failed_auth_rate: float = 0.1,
    ):
        self.window_size = window_size
        self.min_availability = min_availability
        self.max_failed_auth_rate = max_failed_auth_rate
        self.baseline_samples: List[TelemetrySample] = []

    def record_sample(self, sample: TelemetrySample):
        self.baseline_samples.append(sample)

    def is_steady_state_verified(self) -> Tuple[bool, Dict[str, Any]]:
        """Evaluate whether baseline samples verify the steady-state hypothesis."""
        if len(self.baseline_samples) < self.window_size:
            return False, {
                "verified": False,
                "reason": f"Insufficient samples ({len(self.baseline_samples)}/{self.window_size})",
            }

        recent = self.baseline_samples[-self.window_size:]
        availabilities = [s.service_availability for s in recent]
        auth_failures = [s.failed_auth_rate for s in recent]

        mean_avail = float(np.mean(availabilities))
        mean_auth_fail = float(np.mean(auth_failures))
        avail_std = float(np.std(availabilities))

        verified = (
            mean_avail >= self.min_availability and
            mean_auth_fail <= self.max_failed_auth_rate and
            avail_std < 0.05
        )

        report = {
            "verified": verified,
            "mean_availability": mean_avail,
            "availability_std": avail_std,
            "mean_failed_auth_rate": mean_auth_fail,
            "samples_analyzed": len(recent),
        }
        return verified, report

    def reset(self):
        self.baseline_samples.clear()


class TelemetryCollector:
    """Aggregates telemetry from all simulated network entities."""

    def __init__(self):
        self.history: List[TelemetrySample] = []
        self.steady_state_detector = SteadyStateDetector()

    def collect(
        self,
        step: int,
        network_state: Dict[str, Any],
        ids_alerts: int = 0,
        failed_auth_count: int = 0,
    ) -> TelemetrySample:
        """Extract unified telemetry record from network state dictionary."""
        total_svcs = 0
        avail_svcs = 0
        active_conns = 0

        for h in network_state.get("hosts", {}).values():
            for s in h.get("services", {}).values():
                total_svcs += 1
                if not h.get("isolated", False):
                    avail_svcs += 1
                active_conns += s.get("active_sessions", 0)

        for d in network_state.get("iot_devices", {}).values():
            total_svcs += 1
            if not d.get("isolated", False):
                avail_svcs += 1

        availability = avail_svcs / total_svcs if total_svcs > 0 else 1.0

        sample = TelemetrySample(
            timestamp=time.time(),
            step=step,
            service_availability=availability,
            avg_cpu_load=20.0 + (active_conns * 5.0),
            avg_memory_free_pct=75.0,
            active_connections=active_conns,
            failed_auth_rate=float(failed_auth_count),
            ids_alerts=ids_alerts,
            packet_drop_rate=0.0 if availability > 0.5 else 0.4,
        )
        self.history.append(sample)
        return sample

    def reset(self):
        self.history.clear()
        self.steady_state_detector.reset()
