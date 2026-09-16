# securemesh_sce/environment/iot/esp32_sim.py
"""ESP32 Rich IoT Gateway Simulator.

Simulates a dual-core 240MHz microcontroller with 520KB SRAM, hardware crypto,
signed OTA verification, and routing capabilities between mesh nodes and gateway.
"""

from __future__ import annotations

import time
import hashlib
import random
from typing import Dict, List, Any, Optional


class ESP32Simulator:
    """Simulated ESP32 IoT gateway and mesh coordinator."""

    TOTAL_SRAM_BYTES = 520000  # ~520 KB SRAM

    def __init__(
        self,
        device_id: str = "esp32_gateway_01",
        ip_address: str = "192.168.4.1",
        firmware_version: str = "v2.1.0-sce",
        seed: int = 42,
    ):
        self.device_id = device_id
        self.ip_address = ip_address
        self.firmware_version = firmware_version
        self.rng = random.Random(seed)

        # Hardware specs
        self.free_sram: int = 280000
        self.core0_load_pct: float = 8.0   # Protocol/WiFi stack
        self.core1_load_pct: float = 12.0  # Application tasks
        self.firmware_integrity: float = 1.0
        self.is_online: bool = True
        self.is_isolated: bool = False
        self.compromised: bool = False
        self.crypto_engine_active: bool = True
        self.connected_endpoints: List[str] = []

    def step(self, routed_traffic_kbs: float = 0.0) -> Dict[str, Any]:
        """Update gateway state with packet routing load."""
        if not self.is_online:
            return self.status()

        # Core 0 handles network throughput
        self.core0_load_pct = min(100.0, 5.0 + routed_traffic_kbs * 2.5)
        # Core 1 handles crypto and edge analysis
        crypto_overhead = 15.0 if self.crypto_engine_active else 0.0
        self.core1_load_pct = min(100.0, 8.0 + crypto_overhead + self.rng.uniform(0, 4))

        return self.status()

    def verify_and_apply_ota(self, image_bytes: bytes, signature: str) -> bool:
        """Simulate secure signed Over-The-Air (OTA) firmware upgrade verification."""
        digest = hashlib.sha256(image_bytes).hexdigest()
        # Simulated verification: valid signature prefix matching digest prefix
        is_valid = signature.startswith(digest[:8])
        if is_valid:
            self.firmware_integrity = 1.0
            self.compromised = False
            self.firmware_version = "v2.2.0-secure"
            return True
        return False

    def isolate(self):
        self.is_isolated = True

    def reconnect(self):
        self.is_isolated = False

    def status(self) -> Dict[str, Any]:
        return {
            "device_id": self.device_id,
            "ip": self.ip_address,
            "online": self.is_online,
            "isolated": self.is_isolated,
            "compromised": self.compromised,
            "firmware_integrity": self.firmware_integrity,
            "free_sram_bytes": self.free_sram,
            "core0_load": self.core0_load_pct,
            "core1_load": self.core1_load_pct,
            "connected_endpoints": len(self.connected_endpoints),
        }
