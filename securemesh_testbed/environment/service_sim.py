# securemesh_testbed/environment/service_sim.py
"""Service simulation utilities.

Each service type can define custom behaviour for authentication attempts,
command execution, and file upload. The base Service class from network_sim
covers generic exploit logic; specialized subclasses add richer logging.
"""

import random

from .network_sim import Service

class SSHService(Service):
    """Simulated SSH service with password authentication.

    `vulnerability` represents probability that a guessed password succeeds.
    """

    def __init__(self, port: int = 22, vulnerability: float = 0.5):
        super().__init__(name="ssh", port=port, vulnerability=vulnerability)
        self.failed_attempts: dict[str, int] = {}

    def auth_attempt(self, username: str, password: str, attacker_id: str) -> bool:
        """Return True if authentication succeeds.

        The method records failed attempts per attacker to allow the defender to
        observe brute‑force patterns.
        """
        # Simplified: success probability = vulnerability * (1 / (1 + attempts))
        attempts = self.failed_attempts.get(attacker_id, 0)
        success_chance = self.vulnerability / (1 + attempts)
        if random.random() < success_chance:
            self.compromised = True
            self.active_sessions.append(attacker_id)
            return True
        else:
            self.failed_attempts[attacker_id] = attempts + 1
            return False

class HTTPService(Service):
    """Simulated HTTP service exposing a fake web‑app login page.

    It records credential submissions and can be exploited via file upload.
    """

    def __init__(self, port: int = 80, vulnerability: float = 0.4):
        super().__init__(name="http", port=port, vulnerability=vulnerability)
        self.logged_credentials: list[dict] = []

    def submit_login(self, username: str, password: str, attacker_id: str):
        self.logged_credentials.append({"attacker": attacker_id, "user": username, "pass": password})
        # No immediate compromise; attacker may later upload malware.

    def upload_file(self, filename: str, content: bytes, attacker_id: str) -> bool:
        """Return True if the uploaded file leads to compromise.
        The chance is proportional to service vulnerability.
        """
        if random.random() < self.vulnerability:
            self.compromised = True
            self.active_sessions.append(attacker_id)
            return True
        return False

# Additional services (FTP, MQTT, CoAP) can be added similarly.
