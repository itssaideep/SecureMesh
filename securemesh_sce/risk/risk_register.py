# securemesh_sce/risk/risk_register.py
"""Dynamic Risk Register supporting SCENE risk progression.

Calculates Initial Risk = Likelihood × Impact, evaluates Defensive Control Maturity (1-5),
and updates Residual Risk dynamically after chaos experiment execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional


@dataclass
class RiskEntry:
    risk_id: str
    threat_description: str
    stride_category: str
    initial_likelihood: int  # 1-5
    initial_impact: int      # 1-5
    control_name: str
    control_maturity: float  # 1.0 - 5.0
    residual_risk: float = 0.0

    def __post_init__(self):
        self.calculate_residual()

    @property
    def initial_risk(self) -> int:
        return self.initial_likelihood * self.initial_impact

    def calculate_residual(self) -> float:
        """Residual Risk = Initial Risk × (1 - (Control Maturity / 6.0))."""
        maturity_factor = min(0.9, self.control_maturity / 6.0)
        self.residual_risk = round(self.initial_risk * (1.0 - maturity_factor), 2)
        return self.residual_risk

    def update_from_experiment(self, attack_success_rate: float, detection_rate: float):
        """Update control maturity and residual risk from experimental findings."""
        if attack_success_rate > 0.4:
            self.control_maturity = max(1.0, self.control_maturity - 0.5)
        if detection_rate > 0.8:
            self.control_maturity = min(5.0, self.control_maturity + 0.3)
        self.calculate_residual()


class RiskRegister:
    """Repository of security risks evaluated across SCE experiments."""

    def __init__(self):
        self.entries: Dict[str, RiskEntry] = {}
        self._populate_defaults()

    def _populate_defaults(self):
        defaults = [
            RiskEntry("RSK-01", "Unauthorized SSH brute-force credential stuffing", "Spoofing", 4, 4, "Rate Limiting & Honeypot Redirection", 3.0),
            RiskEntry("RSK-02", "Remote command injection in web gateway", "Elevation of Privilege", 4, 5, "Service Isolation & Input Sanitization", 2.5),
            RiskEntry("RSK-03", "IoT endpoint buffer overflow & OOM crash", "Denial of Service", 5, 3, "Firmware Verification & IoT Isolation", 2.0),
            RiskEntry("RSK-04", "Lateral movement via compromised mesh nodes", "Elevation of Privilege", 3, 4, "Micro-segmentation & Dynamic IP Blocking", 3.0),
            RiskEntry("RSK-05", "Stealthy C2 beaconing & data exfiltration", "Information Disclosure", 3, 5, "Bayesian Anomaly Tracking & IDS Tuning", 2.5),
        ]
        for d in defaults:
            self.entries[d.risk_id] = d

    def update_risk(self, risk_id: str, asr: float, dr: float):
        if risk_id in self.entries:
            self.entries[risk_id].update_from_experiment(asr, dr)

    def to_markdown(self) -> str:
        """Format the risk register as a research-grade Markdown table."""
        lines = [
            "| Risk ID | Threat Description | STRIDE | Initial Risk | Control | Maturity | Residual Risk |",
            "|:---|:---|:---|:---:|:---|:---:|:---:|",
        ]
        for e in sorted(self.entries.values(), key=lambda x: x.risk_id):
            lines.append(
                f"| {e.risk_id} | {e.threat_description} | {e.stride_category} | "
                f"{e.initial_risk} (L{e.initial_likelihood}×I{e.initial_impact}) | {e.control_name} | "
                f"{e.control_maturity:.1f}/5.0 | **{e.residual_risk:.2f}** |"
            )
        return "\n".join(lines)

    def to_dict(self) -> List[Dict[str, Any]]:
        return [
            {
                "risk_id": e.risk_id,
                "threat_description": e.threat_description,
                "stride": e.stride_category,
                "initial_likelihood": e.initial_likelihood,
                "initial_impact": e.initial_impact,
                "initial_risk": e.initial_risk,
                "control": e.control_name,
                "control_maturity": e.control_maturity,
                "residual_risk": e.residual_risk,
            }
            for e in self.entries.values()
        ]
