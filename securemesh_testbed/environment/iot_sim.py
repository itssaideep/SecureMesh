# securemesh_testbed/environment/iot_sim.py
"""ESP32 and ESP8266 IoT endpoint simulators.

Extends the base IoTDevice from network_sim with richer behaviour:
  - Fake login portal (credential capture, mimicking the real ESP8266 sketch)
  - MQTT topic subscription / publish (exploitable topic injection)
  - CoAP resource endpoints
  - OTA firmware update endpoint (simulated vulnerability)
  - Periodic sensor telemetry generation
"""

from __future__ import annotations
import random
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .network_sim import IoTDevice


@dataclass
class SensorReading:
    """A single telemetry reading from a simulated IoT sensor."""
    temperature: float = 0.0
    humidity: float = 0.0
    timestamp: int = 0


class ESP8266Sim(IoTDevice):
    """Simulates the existing ESP8266 honeypot behaviour in software.

    The real hardware hosts a fake 'Smart Camera Administration' login
    portal and logs credential-guessing attempts.  This class reproduces
    that logic so experiments can run without physical hardware.
    """

    def __init__(self, device_id: str = "esp8266-sim",
                 firmware_hash: str = "abc123",
                 vulnerable: bool = True):
        super().__init__(device_id=device_id,
                         firmware_hash=firmware_hash,
                         vulnerable=vulnerable)
        self.credential_log: List[Dict[str, str]] = []
        self.login_portal_active: bool = True

    # --- Fake login portal ---------------------------------------------------
    def portal_login_attempt(self, username: str, password: str,
                             source_ip: str) -> bool:
        """Record a credential-guessing attempt and return whether it
        'succeeds' (always False — it is a honeypot)."""
        self.credential_log.append({
            "username": username,
            "password": password,
            "source_ip": source_ip,
        })
        return False  # honeypot never grants real access

    # --- Sensor telemetry -----------------------------------------------------
    def generate_telemetry(self, step: int) -> SensorReading:
        return SensorReading(
            temperature=20.0 + random.gauss(0, 2),
            humidity=50.0 + random.gauss(0, 5),
            timestamp=step,
        )

    def reset(self):
        """Reset device state for a new episode."""
        self.compromised = False
        self.isolated = False
        self.credential_log.clear()


class ESP32Sim(IoTDevice):
    """Simulates an ESP32 honeypot with MQTT, CoAP, and OTA surfaces.

    Attack surfaces
    ---------------
    1. MQTT topic injection — attacker publishes to privileged topics.
    2. CoAP resource manipulation — attacker reads/writes sensor config.
    3. OTA firmware update — attacker pushes malicious firmware image.
    """

    def __init__(self, device_id: str = "esp32-sim",
                 firmware_hash: str = "def456",
                 vulnerable: bool = True):
        super().__init__(device_id=device_id,
                         firmware_hash=firmware_hash,
                         vulnerable=vulnerable)
        # MQTT state
        self.mqtt_subscriptions: List[str] = ["sensors/temperature",
                                               "sensors/humidity"]
        self.mqtt_published: List[Dict] = []
        self.mqtt_injected: bool = False
        # CoAP state
        self.coap_resources: Dict[str, str] = {
            "/sensor/config": '{"interval": 60}',
            "/sensor/data": '{"temp": 22.0}',
        }
        self.coap_tampered: bool = False
        # OTA state
        self.ota_endpoint_open: bool = vulnerable
        self.firmware_tampered: bool = False

    # --- MQTT ---------------------------------------------------------------
    def mqtt_publish(self, topic: str, payload: str, source: str) -> bool:
        """Publish to a topic.  Returns True if the topic was privileged."""
        self.mqtt_published.append({
            "topic": topic, "payload": payload, "source": source,
        })
        if topic.startswith("admin/") or topic.startswith("cmd/"):
            self.mqtt_injected = True
            return True
        return False

    # --- CoAP ---------------------------------------------------------------
    def coap_get(self, path: str) -> Optional[str]:
        return self.coap_resources.get(path)

    def coap_put(self, path: str, payload: str, source: str) -> bool:
        """Write a CoAP resource.  Returns True if a sensitive resource
        was overwritten."""
        if path in self.coap_resources:
            self.coap_resources[path] = payload
            if path == "/sensor/config":
                self.coap_tampered = True
                return True
        return False

    # --- OTA ----------------------------------------------------------------
    def ota_update(self, firmware_blob: bytes, source: str) -> bool:
        """Attempt an OTA firmware push.  Succeeds if the endpoint is
        open and the device is vulnerable."""
        if not self.ota_endpoint_open:
            return False
        if not self.vulnerable:
            return False
        new_hash = hashlib.sha256(firmware_blob).hexdigest()[:12]
        self.firmware_hash = new_hash
        self.firmware_tampered = True
        self.compromised = True
        return True

    # --- Sensor telemetry ---------------------------------------------------
    def generate_telemetry(self, step: int) -> SensorReading:
        return SensorReading(
            temperature=21.0 + random.gauss(0, 1.5),
            humidity=55.0 + random.gauss(0, 4),
            timestamp=step,
        )

    def reset(self):
        self.compromised = False
        self.isolated = False
        self.mqtt_published.clear()
        self.mqtt_injected = False
        self.coap_resources = {
            "/sensor/config": '{"interval": 60}',
            "/sensor/data": '{"temp": 22.0}',
        }
        self.coap_tampered = False
        self.firmware_tampered = False
