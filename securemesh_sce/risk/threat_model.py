# securemesh_sce/risk/threat_model.py
"""TAFFAC-aligned Threat Modeling, STRIDE, and MITRE ATT&CK mapping.

Follows the TAFFAC threat assessment framework (Sood et al., 2026) to classify
adversary capabilities, access vectors, attack objectives, and defensive controls.
"""

from __future__ import annotations

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

from ..game.actions import AttackerAction, AttackerType, AttackerProfile


class STRIDECategory(Enum):
    SPOOFING = "Spoofing"
    TAMPERING = "Tampering"
    REPUDIATION = "Repudiation"
    INFO_DISCLOSURE = "Information Disclosure"
    DENIAL_OF_SERVICE = "Denial of Service"
    ELEVATION_OF_PRIVILEGE = "Elevation of Privilege"


@dataclass
class MITRETechnique:
    technique_id: str
    name: str
    tactic: str
    stride: STRIDECategory
    url: str


# Canonical MITRE ATT&CK mapping for attacker actions
ACTION_MITRE_MAPPING: Dict[AttackerAction, MITRETechnique] = {
    AttackerAction.RECON_SCAN: MITRETechnique(
        technique_id="T1595.001",
        name="Active Scanning: Scanning IP Blocks",
        tactic="Reconnaissance",
        stride=STRIDECategory.INFO_DISCLOSURE,
        url="https://attack.mitre.org/techniques/T1595/001/",
    ),
    AttackerAction.RECON_FINGERPRINT: MITRETechnique(
        technique_id="T1592",
        name="Gather Victim Host Information",
        tactic="Reconnaissance",
        stride=STRIDECategory.INFO_DISCLOSURE,
        url="https://attack.mitre.org/techniques/T1592/",
    ),
    AttackerAction.AUTH_BRUTEFORCE: MITRETechnique(
        technique_id="T1110.001",
        name="Brute Force: Password Guessing",
        tactic="Credential Access",
        stride=STRIDECategory.SPOOFING,
        url="https://attack.mitre.org/techniques/T1110/001/",
    ),
    AttackerAction.AUTH_CREDENTIAL_STUFF: MITRETechnique(
        technique_id="T1110.004",
        name="Brute Force: Credential Stuffing",
        tactic="Credential Access",
        stride=STRIDECategory.SPOOFING,
        url="https://attack.mitre.org/techniques/T1110/004/",
    ),
    AttackerAction.EXPLOIT_SERVICE: MITRETechnique(
        technique_id="T1190",
        name="Exploit Public-Facing Application",
        tactic="Initial Access",
        stride=STRIDECategory.ELEVATION_OF_PRIVILEGE,
        url="https://attack.mitre.org/techniques/T1190/",
    ),
    AttackerAction.EXPLOIT_IOT: MITRETechnique(
        technique_id="T1200",
        name="Hardware Additions / Firmware Exploit",
        tactic="Initial Access",
        stride=STRIDECategory.TAMPERING,
        url="https://attack.mitre.org/techniques/T1200/",
    ),
    AttackerAction.MALWARE_DROP: MITRETechnique(
        technique_id="T1105",
        name="Ingress Tool Transfer",
        tactic="Command and Control",
        stride=STRIDECategory.TAMPERING,
        url="https://attack.mitre.org/techniques/T1105/",
    ),
    AttackerAction.PERSIST_BACKDOOR: MITRETechnique(
        technique_id="T1543",
        name="Create or Modify System Process",
        tactic="Persistence",
        stride=STRIDECategory.ELEVATION_OF_PRIVILEGE,
        url="https://attack.mitre.org/techniques/T1543/",
    ),
    AttackerAction.PERSIST_C2: MITRETechnique(
        technique_id="T1071.001",
        name="Application Layer Protocol: Web Protocols",
        tactic="Command and Control",
        stride=STRIDECategory.INFO_DISCLOSURE,
        url="https://attack.mitre.org/techniques/T1071/001/",
    ),
    AttackerAction.EVADE_OBFUSCATE: MITRETechnique(
        technique_id="T1027",
        name="Obfuscated Files or Information",
        tactic="Defense Evasion",
        stride=STRIDECategory.REPUDIATION,
        url="https://attack.mitre.org/techniques/T1027/",
    ),
    AttackerAction.EVADE_SLOWDOWN: MITRETechnique(
        technique_id="T1029",
        name="Scheduled Transfer / Throttled Traffic",
        tactic="Command and Control",
        stride=STRIDECategory.REPUDIATION,
        url="https://attack.mitre.org/techniques/T1029/",
    ),
    AttackerAction.LATERAL_MOVE: MITRETechnique(
        technique_id="T1021.004",
        name="Remote Services: SSH / Lateral Movement",
        tactic="Lateral Movement",
        stride=STRIDECategory.ELEVATION_OF_PRIVILEGE,
        url="https://attack.mitre.org/techniques/T1021/004/",
    ),
    AttackerAction.NOOP: MITRETechnique(
        technique_id="T0000",
        name="Dormant / Inactive",
        tactic="Persistence",
        stride=STRIDECategory.REPUDIATION,
        url="",
    ),
}


class ThreatModel:
    """Manages adversary profiling under TAFFAC and threat mapping."""

    @staticmethod
    def get_mitre_mapping(action: AttackerAction) -> MITRETechnique:
        return ACTION_MITRE_MAPPING.get(
            action,
            MITRETechnique("T0000", "Unknown", "Unknown", STRIDECategory.INFO_DISCLOSURE, ""),
        )

    @staticmethod
    def create_taffac_profile(attacker_type: AttackerType) -> AttackerProfile:
        """Instantiate formal TAFFAC attribute profile for each hidden type."""
        profiles = {
            AttackerType.OPPORTUNISTIC: AttackerProfile(
                attacker_type=attacker_type,
                capability="low",
                access="none",
                system_knowledge="none",
                objective="disruption",
                attack_budget=100,
                adaptability=False,
                autonomy="scripted",
                observability="blind",
            ),
            AttackerType.STEALTH: AttackerProfile(
                attacker_type=attacker_type,
                capability="medium",
                access="none",
                system_knowledge="partial",
                objective="data_exfiltration",
                attack_budget=200,
                adaptability=False,
                autonomy="semi",
                observability="partial",
            ),
            AttackerType.ADAPTIVE: AttackerProfile(
                attacker_type=attacker_type,
                capability="high",
                access="partial",
                system_knowledge="partial",
                objective="full_compromise",
                attack_budget=300,
                adaptability=True,
                autonomy="full",
                observability="full",
            ),
            AttackerType.RESOURCE_AWARE: AttackerProfile(
                attacker_type=attacker_type,
                capability="medium",
                access="none",
                system_knowledge="full",
                objective="resource_hijack",
                attack_budget=150,
                adaptability=True,
                autonomy="semi",
                observability="partial",
            ),
        }
        return profiles[attacker_type]
