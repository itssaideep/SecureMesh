# securemesh_sce/environment/services/ssh.py
"""SSH Service Simulator (Cowrie-compatible semantics).

Simulates an SSH service/honeypot, tracking connection attempts, brute-force
credentials, interactive shell commands, and telemetry events.
"""

from __future__ import annotations

import time
import uuid
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field


@dataclass
class SSHSession:
    session_id: str
    src_ip: str
    src_port: int
    start_time: float
    authenticated: bool = False
    username: Optional[str] = None
    commands: List[str] = field(default_factory=list)
    active: bool = True


class SSHServiceSimulator:
    """Simulated SSH service with honeypot logging capabilities."""

    DEFAULT_CREDS = {
        ("root", "root"),
        ("root", "123456"),
        ("admin", "admin"),
        ("pi", "raspberry"),
        ("user", "user"),
        ("guest", "guest"),
    }

    def __init__(
        self,
        port: int = 22,
        banner: str = "SSH-2.0-OpenSSH_8.2p1 Ubuntu-4ubuntu0.5",
        is_honeypot: bool = False,
    ):
        self.port = port
        self.banner = banner
        self.is_honeypot = is_honeypot
        self.sessions: Dict[str, SSHSession] = {}
        self.failed_attempts: List[Dict[str, Any]] = []
        self.event_log: List[Dict[str, Any]] = []
        self.is_running: bool = True

    def connect(self, src_ip: str, src_port: int = 44122) -> str:
        """Create new incoming SSH connection session."""
        session_id = str(uuid.uuid4())[:8]
        sess = SSHSession(
            session_id=session_id,
            src_ip=src_ip,
            src_port=src_port,
            start_time=time.time(),
        )
        self.sessions[session_id] = sess
        self._log_event("cowrie.session.connect", session_id, {"src_ip": src_ip, "port": self.port})
        return session_id

    def authenticate(self, session_id: str, username: str, password: str) -> bool:
        """Attempt authentication within a session."""
        if session_id not in self.sessions or not self.is_running:
            return False

        sess = self.sessions[session_id]
        if not sess.active:
            return False

        success = (username, password) in self.DEFAULT_CREDS
        if success:
            sess.authenticated = True
            sess.username = username
            self._log_event("cowrie.login.success", session_id, {"username": username, "password": password})
        else:
            self.failed_attempts.append({
                "session_id": session_id,
                "src_ip": sess.src_ip,
                "username": username,
                "password": password,
                "timestamp": time.time(),
            })
            self._log_event("cowrie.login.failed", session_id, {"username": username, "password": password})

        return success

    def execute_command(self, session_id: str, command: str) -> Tuple[bool, str]:
        """Simulate execution of a shell command."""
        if session_id not in self.sessions:
            return False, "Session not found"

        sess = self.sessions[session_id]
        if not sess.authenticated or not sess.active:
            return False, "Not authenticated"

        sess.commands.append(command)
        self._log_event("cowrie.command.input", session_id, {"input": command})

        # Emulated responses for typical attacker reconnaissance
        cmd = command.strip().lower()
        if cmd.startswith("uname"):
            output = "Linux securemesh-gateway 5.4.0-104-generic #118-Ubuntu SMP x86_64 GNU/Linux"
        elif cmd.startswith("cat /etc/issue"):
            output = "Ubuntu 20.04.4 LTS \\n \\l"
        elif cmd.startswith("id") or cmd.startswith("whoami"):
            output = f"uid=0({sess.username}) gid=0({sess.username}) groups=0({sess.username})"
        elif cmd.startswith("ps"):
            output = "PID TTY          TIME CMD\n    1 ?        00:00:02 systemd\n  521 ?        00:00:00 sshd"
        elif "wget" in cmd or "curl" in cmd:
            output = "Connecting to 192.168.1.55:8000... connected. HTTP request sent, awaiting response... 200 OK"
        else:
            output = f"{cmd}: command executed"

        return True, output

    def close_session(self, session_id: str):
        if session_id in self.sessions:
            self.sessions[session_id].active = False
            self._log_event("cowrie.session.closed", session_id, {})

    def _log_event(self, event_type: str, session_id: str, data: Dict[str, Any]):
        entry = {
            "eventid": event_type,
            "session": session_id,
            "timestamp": time.time(),
            "data": data,
        }
        self.event_log.append(entry)

    @property
    def active_session_count(self) -> int:
        return sum(1 for s in self.sessions.values() if s.active)

    def reset(self):
        self.sessions.clear()
        self.failed_attempts.clear()
        self.event_log.clear()
        self.is_running = True
