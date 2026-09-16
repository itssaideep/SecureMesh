# securemesh_sce/environment/services/http.py
"""HTTP Web Service & IoT API Simulator.

Simulates an embedded HTTP web server with endpoints, authentication,
vulnerabilities (command injection, path traversal, brute force),
and structured request logging.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field


@dataclass
class HTTPRequest:
    method: str
    path: str
    headers: Dict[str, str] = field(default_factory=dict)
    body: str = ""
    src_ip: str = "127.0.0.1"
    timestamp: float = field(default_factory=time.time)


@dataclass
class HTTPResponse:
    status_code: int
    headers: Dict[str, str] = field(default_factory=dict)
    body: str = ""


class HTTPServiceSimulator:
    """Embedded HTTP service simulator with vulnerability hooks."""

    def __init__(
        self,
        port: int = 80,
        server_header: str = "nginx/1.18.0 (Ubuntu)",
        vulnerability_score: float = 0.4,
    ):
        self.port = port
        self.server_header = server_header
        self.vulnerability_score = vulnerability_score
        self.is_running: bool = True
        self.request_log: List[Dict[str, Any]] = []
        self.compromised: bool = False
        self.active_webshells: List[str] = []

    def handle_request(self, req: HTTPRequest) -> HTTPResponse:
        """Process incoming HTTP request and simulate behavior."""
        if not self.is_running:
            return HTTPResponse(status_code=503, body="Service Unavailable")

        path = req.path.split("?")[0]
        query = req.path.split("?")[1] if "?" in req.path else ""

        # Default headers
        headers = {
            "Server": self.server_header,
            "Content-Type": "text/html; charset=UTF-8",
        }

        # Check command injection in CGI endpoint
        if "/cgi-bin/" in path:
            if ";" in query or "|" in query or "`" in query or "$(" in query or "cmd=" in query:
                if self.vulnerability_score > 0.2:
                    self.compromised = True
                    self.active_webshells.append(req.src_ip)
                    self._record(req, 200, "EXPLOIT_CMD_INJECTION")
                    return HTTPResponse(200, headers, "uid=33(www-data) gid=33(www-data)")

        # Path traversal
        if ".." in path or "etc/passwd" in query:
            if self.vulnerability_score > 0.3:
                self._record(req, 200, "PATH_TRAVERSAL")
                return HTTPResponse(200, headers, "root:x:0:0:root:/root:/bin/bash\nwww-data:x:33:33:www-data:/var/www:/usr/sbin/nologin")

        # API Auth
        if path == "/api/login":
            if req.method == "POST":
                if "admin" in req.body and "password123" in req.body:
                    self._record(req, 200, "AUTH_SUCCESS")
                    return HTTPResponse(200, {"Content-Type": "application/json"}, '{"token": "jwt-token-9982"}')
                else:
                    self._record(req, 401, "AUTH_FAIL")
                    return HTTPResponse(401, {"Content-Type": "application/json"}, '{"error": "Unauthorized"}')

        # Admin panel
        if path.startswith("/admin"):
            self._record(req, 403, "ADMIN_FORBIDDEN")
            return HTTPResponse(403, headers, "<h1>403 Forbidden</h1>")

        # Root endpoint
        if path == "/":
            self._record(req, 200, "OK")
            return HTTPResponse(200, headers, "<html><body><h1>SecureMesh IoT Gateway</h1></body></html>")

        self._record(req, 404, "NOT_FOUND")
        return HTTPResponse(404, headers, "<h1>404 Not Found</h1>")

    def _record(self, req: HTTPRequest, status: int, category: str):
        self.request_log.append({
            "src_ip": req.src_ip,
            "method": req.method,
            "path": req.path,
            "status": status,
            "category": category,
            "timestamp": time.time(),
        })

    def reset(self):
        self.request_log.clear()
        self.compromised = False
        self.active_webshells.clear()
        self.is_running = True
