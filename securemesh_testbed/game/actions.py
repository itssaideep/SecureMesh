# securemesh_testbed/game/actions.py
"""Definition of attacker and defender action spaces.

The actions are expressed as Python Enums so they can be enumerated
by the Gym environment and the agents.
"""

from enum import Enum, auto

class AttackerAction(Enum):
    RECON_SCAN = auto()
    RECON_FINGERPRINT = auto()
    AUTH_BRUTEFORCE = auto()
    AUTH_CREDENTIAL_STUFF = auto()
    EXPLOIT_SERVICE = auto()
    EXPLOIT_IOT = auto()
    MALWARE_DROP = auto()
    PERSIST_BACKDOOR = auto()
    PERSIST_C2 = auto()
    EVADE_OBFUSCATE = auto()
    EVADE_SLOWDOWN = auto()
    LATERAL_MOVE = auto()
    NOOP = auto()

class DefenderAction(Enum):
    BLOCK_IP = auto()
    RATE_LIMIT = auto()
    ISOLATE_SERVICE = auto()
    ISOLATE_IOT = auto()
    INCREASE_MONITORING = auto()
    ADJUST_IDS_THRESHOLD = auto()
    TERMINATE_SESSION = auto()
    PATCH_SERVICE = auto()
    RESTORE_SERVICE = auto()
    HONEYPOT_DEPLOY = auto()
    RESET_CREDENTIALS = auto()
    NOOP = auto()
