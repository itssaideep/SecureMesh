# securemesh_sce/environment/iot/esp8266_sim.py
"""ESP8266 Constrained IoT Endpoint Simulator.

Simulates an ultra-constrained micro-controller (80MHz, ~35KB free heap)
transmitting sensor telemetry via MQTT/HTTP. Highly susceptible to resource
exhaustion, connection drops under load, and firmware compromise.
"""

from __future__ import annotations

import time
import random
from typing import Dict, Any, Optional


class ESP8266Simulator:
    """Simulated ESP8266 IoT sensor node."""

    TOTAL_HEAP_BYTES = 36000  # ~36 KB usable heap

    def __init__(
        self,
        device_id: str = "esp8266_01",
        ip_address: str = "192.168.4.15",
        firmware_version: str = "v1.0.4-sce",
        seed: int = 42,
    ):
        self.device_id = device_id
        self.ip_address = ip_address
        self.firmware_version = firmware_version
        self.rng = random.Random(seed)

        # Hardware state
        self.free_heap: int = self.TOTAL_HEAP_BYTES
        self.cpu_load_pct: float = 12.0
        self.firmware_integrity: float = 1.0  # 1.0 = clean, 0.0 = compromised
        self.is_online: bool = True
        self.is_isolated: bool = False
        self.compromised: bool = False
        self.uptime_seconds: int = 0
        self.packet_drop_rate: float = 0.0

        # Sensor readings
        self.temperature: float = 22.5
        self.humidity: float = 45.0

    def step(self, incoming_requests: int = 0) -> Dict[str, Any]:
        """Advance time by one step. Simulates resource pressure under request load."""
        if not self.is_online:
            return self.status()

        self.uptime_seconds += 1

        # Natural sensor fluctuation
        self.temperature += self.rng.uniform(-0.2, 0.2)
        self.humidity = max(10.0, min(95.0, self.humidity + self.rng.uniform(-0.5, 0.5)))

        # Resource degradation under attack traffic / request volume
        heap_consumed = incoming_requests * 1200
        self.free_heap = max(0, self.TOTAL_HEAP_BYTES - 4000 - heap_consumed)
        self.cpu_load_pct = min(100.0, 10.0 + incoming_requests * 15.0)

        # Packet drop if heap critical
        if self.free_heap < 4000:
            self.packet_drop_rate = min(1.0, (4000 - self.free_heap) / 4000.0)
            if self.free_heap <= 500 and self.rng.random() < 0.4:
                # OOM watchdog reboot
                self._reboot()
        else:
            self.packet_drop_rate = 0.0

        return self.status()

    def inject_compromise(self, malicious_payload_size: int = 1024):
        """Simulate malicious exploit / shellcode injection."""
        self.compromised = True
        self.firmware_integrity = max(0.0, self.firmware_integrity - 0.4)
        self.free_heap = max(0, self.free_heap - malicious_payload_size)

    def isolate(self):
        """Defender action: disconnect from mesh/WiFi network."""
        self.is_isolated = True

    def reconnect(self):
        """Defender action: restore network connectivity."""
        self.is_isolated = False

    def patch_firmware(self):
        """Defender action: reflash clean firmware."""
        self.compromised = False
        self.firmware_integrity = 1.0
        self._reboot()

    def _reboot(self):
        self.free_heap = self.TOTAL_HEAP_BYTES - 4000
        self.cpu_load_pct = 15.0
        self.uptime_seconds = 0
        self.packet_drop_rate = 0.0

    def get_telemetry_payload(self) -> Dict[str, Any]:
        """Generate MQTT telemetry packet matching physical ESP8266 schema."""
        return {
            "device_id": self.device_id,
            "ip": self.ip_address,
            "temp": round(self.temperature, 2),
            "humidity": round(self.humidity, 2),
            "heap": self.free_heap,
            "uptime": self.uptime_seconds,
            "compromised": self.compromised,
        }

    def status(self) -> Dict[str, Any]:
        return {
            "device_id": self.device_id,
            "ip": self.ip_address,
            "online": self.is_online,
            "isolated": self.is_isolated,
            "compromised": self.compromised,
            "firmware_integrity": self.firmware_integrity,
            "free_heap_bytes": self.free_heap,
            "cpu_load_pct": self.cpu_load_pct,
            "packet_drop_rate": self.packet_drop_rate,
        }
