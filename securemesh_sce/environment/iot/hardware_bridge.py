# securemesh_sce/environment/iot/hardware_bridge.py
"""Physical Hardware Bridge for ESP8266 / ESP32 devices.

Enables SecureMesh-SCE experiments to query physical IoT hardware directly
over HTTP/Serial, replacing or augmenting software simulations.
"""

from __future__ import annotations

import time
import requests
from typing import Dict, Any, Optional


class PhysicalESP8266Bridge:
    """Bridge for querying and interacting with a physical ESP8266 honeypot device."""

    def __init__(self, device_ip: str, port: int = 80, timeout: float = 2.0):
        self.device_ip = device_ip
        self.port = port
        self.timeout = timeout
        self.base_url = f"http://{device_ip}:{port}"

    def is_reachable(self) -> bool:
        """Check if the physical device is online on the local network."""
        try:
            r = requests.get(f"{self.base_url}/status", timeout=self.timeout)
            return r.status_code == 200
        except Exception:
            return False

    def get_hardware_telemetry(self) -> Optional[Dict[str, Any]]:
        """Fetch live free heap, RSSI, and uptime directly from physical hardware."""
        try:
            r = requests.get(f"{self.base_url}/status", timeout=self.timeout)
            if r.status_code == 200:
                data = r.json()
                data["reachable"] = True
                data["timestamp"] = time.time()
                return data
        except Exception:
            pass
        return {"reachable": False, "device": "ESP8266", "error": "Unreachable"}

    def test_honeypot_login(self, username: str = "admin", password: str = "123456") -> Dict[str, Any]:
        """Simulate an attack probe against the physical device."""
        try:
            r = requests.post(
                f"{self.base_url}/login",
                data={"username": username, "password": password},
                timeout=self.timeout,
            )
            return {"status_code": r.status_code, "response": r.text}
        except Exception as e:
            return {"error": str(e)}
