# securemesh_testbed/game/observations.py
"""Observation models for attacker and defender.

Both agents receive a partial view of the full environment state.
"""

from enum import Enum

class AttackerObservation(Enum):
    SCAN_RESULTS = "scan_results"          # list of (host, open_ports)
    AUTH_RESPONSE = "auth_response"        # success/failure flag
    SERVICE_STATUS = "service_status"      # compromised / healthy
    IDS_ALERT = "ids_alert"                # detection flag (if defender broadcasts)
    DEFENDER_ACTION = "defender_action"    # visible defender moves (e.g., block IP)
    NOOP = "noop"

class DefenderObservation(Enum):
    TRAFFIC_VOLUME = "traffic_volume"      # aggregated per segment
    ANOMALY_SCORE = "anomaly_score"        # IDS anomaly output (0..1)
    IDS_ALERT = "ids_alert"                # detection of known signatures
    SERVICE_LOGS = "service_logs"          # auth failures, exploit attempts
    IOT_HEARTBEAT = "iot_heartbeat"        # last‑seen timestamps
    ATTACKER_ACTION = "attacker_action"    # observable attacker moves (e.g., scan)
    NOOP = "noop"
