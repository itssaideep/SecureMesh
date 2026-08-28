# securemesh_testbed/environment/network_sim.py
"""Network simulation core.

Provides classes to model hosts, services and IoT devices and to manage
state transitions during the game.
"""

from typing import List, Dict
import random

class Service:
    def __init__(self, name: str, port: int, vulnerability: float):
        self.name = name
        self.port = port
        self.vulnerability = vulnerability  # probability that an exploit succeeds
        self.compromised = False
        self.active_sessions: List[str] = []  # list of attacker IDs currently connected

    def attempt_exploit(self, attacker_id: str) -> bool:
        """Return True if exploit succeeds, marking service compromised."""
        if self.compromised:
            return True
        if random.random() < self.vulnerability:
            self.compromised = True
            self.active_sessions.append(attacker_id)
            return True
        return False

    def reset(self):
        self.compromised = False
        self.active_sessions.clear()

class Host:
    def __init__(self, hostname: str, services: List[Service]):
        self.hostname = hostname
        self.services = {svc.name: svc for svc in services}
        self.isolated = False

    def get_service(self, name: str) -> Service:
        return self.services[name]

    def isolate(self):
        self.isolated = True

    def restore(self):
        self.isolated = False
        for svc in self.services.values():
            svc.reset()

class IoTDevice:
    def __init__(self, device_id: str, firmware_hash: str, vulnerable: bool = True):
        self.device_id = device_id
        self.firmware_hash = firmware_hash
        self.vulnerable = vulnerable
        self.compromised = False
        self.isolated = False

    def attempt_compromise(self, attacker_id: str) -> bool:
        if self.isolated:
            return False
        if not self.vulnerable:
            return False
        # Simplified: 30% chance to compromise per attempt
        if random.random() < 0.3:
            self.compromised = True
            return True
        return False

    def isolate(self):
        self.isolated = True

    def restore(self):
        self.isolated = False
        self.compromised = False

class NetworkSimulator:
    """Container for hosts and IoT devices.

    Provides methods used by the Markov game to query and mutate the environment.
    """

    def __init__(self, hosts: List[Host], iot_devices: List[IoTDevice]):
        self.hosts: Dict[str, Host] = {h.hostname: h for h in hosts}
        self.iot_devices: Dict[str, IoTDevice] = {d.device_id: d for d in iot_devices}

    def get_host(self, name: str) -> Host:
        return self.hosts[name]

    def get_iot(self, device_id: str) -> IoTDevice:
        return self.iot_devices[device_id]

    def reset(self):
        for h in self.hosts.values():
            h.restore()
        for d in self.iot_devices.values():
            d.restore()

    def snapshot(self) -> dict:
        """Return a lightweight representation of the full network state."""
        return {
            "hosts": {
                h.hostname: {
                    "isolated": h.isolated,
                    "services": {n: {"compromised": s.compromised, "active_sessions": s.active_sessions}
                                 for n, s in h.services.items()},
                }
                for h in self.hosts.values()
            },
            "iot_devices": {
                d.device_id: {
                    "isolated": d.isolated,
                    "compromised": d.compromised,
                }
                for d in self.iot_devices.values()
            },
        }
