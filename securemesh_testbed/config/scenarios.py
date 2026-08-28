# securemesh_testbed/config/scenarios.py
"""Scenario configuration definitions.

Each scenario defines the initial network topology, vulnerability levels, and
whether the IDS operates in signature or zero‑day mode.
"""

from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class ServiceSpec:
    name: str
    port: int
    vulnerability: float  # 0.0 (none) .. 1.0 (fully vulnerable)
    compromised: bool = False

@dataclass
class IoTSpec:
    device_id: str
    firmware_hash: str
    vulnerable: bool = True

@dataclass
class ScenarioConfig:
    name: str
    services: List[ServiceSpec]
    iot_devices: List[IoTSpec]
    ids_signature_mode: bool = True
    seed: int = 0

# Example scenarios used by the testbed
KNOWN_ATTACKS = ScenarioConfig(
    name="known_attacks",
    services=[
        ServiceSpec(name="ssh", port=22, vulnerability=0.7),
        ServiceSpec(name="http", port=80, vulnerability=0.5),
        ServiceSpec(name="mqtt", port=1883, vulnerability=0.4),
    ],
    iot_devices=[
        IoTSpec(device_id="esp8266-01", firmware_hash="abc123"),
        IoTSpec(device_id="esp32-01", firmware_hash="def456"),
    ],
    ids_signature_mode=True,
    seed=42,
)

ZERO_DAY = ScenarioConfig(
    name="zero_day",
    services=[
        ServiceSpec(name="ssh", port=22, vulnerability=0.9),
        ServiceSpec(name="http", port=80, vulnerability=0.8),
        ServiceSpec(name="coap", port=5683, vulnerability=0.85),
    ],
    iot_devices=[
        IoTSpec(device_id="esp8266-02", firmware_hash="zzz999"),
        IoTSpec(device_id="esp32-02", firmware_hash="yyy888"),
    ],
    ids_signature_mode=False,  # zero‑day (no signatures)
    seed=123,
)
