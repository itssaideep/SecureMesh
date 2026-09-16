# securemesh_sce/game/actions.py
"""Formal attacker and defender action spaces with impact tiers.

Actions are expressed as Python Enums. Each action carries metadata
(impact tier, reversibility) used by the safety-constrained action gate.
"""

from enum import Enum, auto
from dataclasses import dataclass
from typing import Dict


# =====================================================================
# Impact tiers for safety-constrained action gating
# =====================================================================

class ImpactTier(Enum):
    """Impact classification for safety-gated defender actions."""
    LOW = "low"              # Autonomous — no review needed
    MEDIUM = "medium"        # Guarded — logged, auto-approved
    HIGH = "high"            # Human approval recommended
    CRITICAL = "critical"    # Human approval required


# =====================================================================
# Attacker actions
# =====================================================================

class AttackerAction(Enum):
    """Attacker action space following the cyber kill-chain."""
    # Reconnaissance
    RECON_SCAN = auto()
    RECON_FINGERPRINT = auto()
    # Authentication
    AUTH_BRUTEFORCE = auto()
    AUTH_CREDENTIAL_STUFF = auto()
    # Exploitation
    EXPLOIT_SERVICE = auto()
    EXPLOIT_IOT = auto()
    # Post-exploitation
    MALWARE_DROP = auto()
    PERSIST_BACKDOOR = auto()
    PERSIST_C2 = auto()
    # Evasion
    EVADE_OBFUSCATE = auto()
    EVADE_SLOWDOWN = auto()
    # Lateral movement
    LATERAL_MOVE = auto()
    # Idle
    NOOP = auto()


# Impact weights per attack action (used for attack_impact metric)
ATTACK_IMPACT_WEIGHTS: Dict[AttackerAction, float] = {
    AttackerAction.RECON_SCAN:           0.05,
    AttackerAction.RECON_FINGERPRINT:    0.08,
    AttackerAction.AUTH_BRUTEFORCE:      0.30,
    AttackerAction.AUTH_CREDENTIAL_STUFF: 0.25,
    AttackerAction.EXPLOIT_SERVICE:      0.60,
    AttackerAction.EXPLOIT_IOT:          0.55,
    AttackerAction.MALWARE_DROP:         0.80,
    AttackerAction.PERSIST_BACKDOOR:     1.00,
    AttackerAction.PERSIST_C2:           0.90,
    AttackerAction.EVADE_OBFUSCATE:      0.02,
    AttackerAction.EVADE_SLOWDOWN:       0.01,
    AttackerAction.LATERAL_MOVE:         0.40,
    AttackerAction.NOOP:                 0.00,
}


# =====================================================================
# Defender actions
# =====================================================================

class DefenderAction(Enum):
    """Defender action space with impact-tier metadata."""
    # Low impact — autonomous
    MONITOR = auto()
    INCREASE_MONITORING = auto()
    ADJUST_IDS_THRESHOLD = auto()
    # Medium impact — guarded
    RATE_LIMIT = auto()
    TERMINATE_SESSION = auto()
    RESET_CREDENTIALS = auto()
    HONEYPOT_DEPLOY = auto()
    # High impact — human approval recommended
    BLOCK_IP = auto()
    ISOLATE_SERVICE = auto()
    ISOLATE_IOT = auto()
    PATCH_SERVICE = auto()
    # Critical — human approval required
    RESTORE_SERVICE = auto()
    NETWORK_SHUTDOWN = auto()
    # Idle
    NOOP = auto()


# Impact tier mapping for each defender action
DEFENDER_ACTION_TIERS: Dict[DefenderAction, ImpactTier] = {
    DefenderAction.MONITOR:              ImpactTier.LOW,
    DefenderAction.INCREASE_MONITORING:  ImpactTier.LOW,
    DefenderAction.ADJUST_IDS_THRESHOLD: ImpactTier.LOW,
    DefenderAction.RATE_LIMIT:           ImpactTier.MEDIUM,
    DefenderAction.TERMINATE_SESSION:    ImpactTier.MEDIUM,
    DefenderAction.RESET_CREDENTIALS:    ImpactTier.MEDIUM,
    DefenderAction.HONEYPOT_DEPLOY:      ImpactTier.MEDIUM,
    DefenderAction.BLOCK_IP:             ImpactTier.HIGH,
    DefenderAction.ISOLATE_SERVICE:      ImpactTier.HIGH,
    DefenderAction.ISOLATE_IOT:          ImpactTier.HIGH,
    DefenderAction.PATCH_SERVICE:        ImpactTier.HIGH,
    DefenderAction.RESTORE_SERVICE:      ImpactTier.CRITICAL,
    DefenderAction.NETWORK_SHUTDOWN:     ImpactTier.CRITICAL,
    DefenderAction.NOOP:                 ImpactTier.LOW,
}

# Whether each defender action is reversible
DEFENDER_ACTION_REVERSIBLE: Dict[DefenderAction, bool] = {
    DefenderAction.MONITOR:              True,
    DefenderAction.INCREASE_MONITORING:  True,
    DefenderAction.ADJUST_IDS_THRESHOLD: True,
    DefenderAction.RATE_LIMIT:           True,
    DefenderAction.TERMINATE_SESSION:    False,
    DefenderAction.RESET_CREDENTIALS:    False,
    DefenderAction.HONEYPOT_DEPLOY:      True,
    DefenderAction.BLOCK_IP:             True,
    DefenderAction.ISOLATE_SERVICE:      True,
    DefenderAction.ISOLATE_IOT:          True,
    DefenderAction.PATCH_SERVICE:        False,
    DefenderAction.RESTORE_SERVICE:      False,
    DefenderAction.NETWORK_SHUTDOWN:     True,
    DefenderAction.NOOP:                 True,
}


# =====================================================================
# Security workflow stages
# =====================================================================

class SecurityStage(Enum):
    """Staged security workflow for bounded autonomy."""
    NORMAL = auto()
    SUSPICIOUS = auto()
    INVESTIGATE = auto()
    CONFIRMED = auto()
    MITIGATE = auto()
    RECOVER = auto()


# =====================================================================
# Attacker type — the hidden variable θ
# =====================================================================

class AttackerType(Enum):
    """Hidden attacker type unknown to the defender.

    θ ∈ {opportunistic, stealth, adaptive, resource_aware}
    """
    OPPORTUNISTIC = "opportunistic"
    STEALTH = "stealth"
    ADAPTIVE = "adaptive"
    RESOURCE_AWARE = "resource_aware"


# =====================================================================
# Attacker attributes (TAFFAC-aligned)
# =====================================================================

@dataclass
class AttackerProfile:
    """Structured attacker attributes following TAFFAC."""
    attacker_type: AttackerType
    capability: str        # "low", "medium", "high"
    access: str            # "none", "partial", "full"
    system_knowledge: str  # "none", "partial", "full"
    objective: str         # e.g. "data_exfiltration", "disruption"
    attack_budget: int     # max steps before exhaustion
    adaptability: bool     # can modify strategy mid-episode
    autonomy: str          # "scripted", "semi", "full"
    observability: str     # "blind", "partial", "full"


# =====================================================================
# Convenience
# =====================================================================

N_ATTACKER_ACTIONS = len(AttackerAction)
N_DEFENDER_ACTIONS = len(DefenderAction)
N_ATTACKER_TYPES = len(AttackerType)
