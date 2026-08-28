# securemesh_testbed/environment/ids_engine.py
"""Simple Zero‑Day IDS engine.

The engine maintains a list of blocked IPs, a rate‑limit counter and a
sensitivity threshold (0..1). Detection methods return `True` when the
simulated IDS would raise an alert.
"""

import random

class IDSEngine:
    def __init__(self, signature_mode: bool = True):
        self.signature_mode = signature_mode
        self.blocked_ips = set()
        self.rate_limited_ips = set()
        self.sensitivity = 0.5  # base detection probability for anomaly
        self.scan_detection_prob = 0.2 if signature_mode else 0.1
        self.auth_detection_base = 0.3
        self.exploit_detection_base = 0.4
        self.malware_detection_base = 0.5
        self.persistence_detection_base = 0.45

    def reset(self):
        self.blocked_ips.clear()
        self.rate_limited_ips.clear()
        self.sensitivity = 0.5

    # ----- Utility helpers -----
    def _rand_detect(self, base: float) -> bool:
        """Return True with probability `base * sensitivity`.
        """
        return random.random() < (base * self.sensitivity)

    # ----- Public API used by the environment -----
    def block_ip(self, ip: str):
        self.blocked_ips.add(ip)

    def rate_limit_ip(self, ip: str):
        self.rate_limited_ips.add(ip)

    def is_ip_blocked(self, ip: str) -> bool:
        return ip in self.blocked_ips

    def increase_sensitivity(self):
        self.sensitivity = min(1.0, self.sensitivity + 0.1)

    def adjust_threshold(self, delta: float):
        self.sensitivity = max(0.0, min(1.0, self.sensitivity + delta))

    # ----- Detection checks -----
    def check_scan(self) -> bool:
        # Scans are relatively noisy; use a lower base probability
        return self._rand_detect(self.scan_detection_prob)

    def check_auth(self, failed_attempts: int) -> bool:
        # More failures raise detection probability
        prob = self.auth_detection_base + 0.05 * min(failed_attempts, 5)
        return self._rand_detect(prob)

    def check_exploit(self, succeeded: bool) -> bool:
        if not succeeded:
            return False
        return self._rand_detect(self.exploit_detection_base)

    def check_malware(self, succeeded: bool) -> bool:
        if not succeeded:
            return False
        return self._rand_detect(self.malware_detection_base)

    def check_persistence(self, succeeded: bool) -> bool:
        if not succeeded:
            return False
        return self._rand_detect(self.persistence_detection_base)
