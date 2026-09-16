# securemesh_sce/environment/network/topology.py
"""Network topology, subnets, and host models.

Provides structural representation of the IoT testbed network,
supporting isolation, firewall rules, service vulnerability mapping,
and generation of network state snapshots matching game requirements.
"""

from __future__ import annotations

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any
import numpy as np


class SubnetType(Enum):
    DMZ = "dmz"
    IOT_LAN = "iot_lan"
    CORP_NET = "corp_net"
    HONEYNET = "honeynet"


class HostType(Enum):
    GATEWAY = "gateway"
    SERVER = "server"
    WORKSTATION = "workstation"
    HONEYPOT = "honeypot"
    IOT_ENDPOINT = "iot_endpoint"


@dataclass
class ServiceInstance:
    name: str
    port: int
    vulnerability_score: float = 0.3
    compromised: bool = False
    active_sessions: int = 0
    is_honeypot: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Host:
    hostname: str
    ip_address: str
    host_type: HostType
    subnet: SubnetType
    services: Dict[str, ServiceInstance] = field(default_factory=dict)
    isolated: bool = False
    compromised: bool = False
    firewall_blocklist: Set[str] = field(default_factory=set)

    def add_service(
        self,
        name: str,
        port: int,
        vulnerability: float = 0.3,
        is_honeypot: bool = False,
    ) -> ServiceInstance:
        svc = ServiceInstance(
            name=name,
            port=port,
            vulnerability_score=vulnerability,
            is_honeypot=is_honeypot,
        )
        self.services[name] = svc
        return svc

    def isolate(self):
        self.isolated = True

    def reconnect(self):
        self.isolated = False


class NetworkTopology:
    """Network topology model containing subnets, routing, hosts, and IoT nodes."""

    def __init__(self, name: str = "SecureMesh-Lab"):
        self.name = name
        self.hosts: Dict[str, Host] = {}
        self.iot_devices: Dict[str, Dict[str, Any]] = {}
        self.firewall_rules: List[Dict[str, Any]] = []

    def add_host(
        self,
        hostname: str,
        ip_address: str,
        host_type: HostType = HostType.SERVER,
        subnet: SubnetType = SubnetType.DMZ,
    ) -> Host:
        host = Host(
            hostname=hostname,
            ip_address=ip_address,
            host_type=host_type,
            subnet=subnet,
        )
        self.hosts[hostname] = host
        return host

    def add_iot_device(
        self,
        device_id: str,
        device_type: str = "esp8266",
        firmware_integrity: float = 1.0,
        isolated: bool = False,
    ):
        self.iot_devices[device_id] = {
            "device_type": device_type,
            "isolated": isolated,
            "compromised": False,
            "firmware_integrity": firmware_integrity,
            "ip_address": f"192.168.4.{len(self.iot_devices) + 10}",
        }

    def isolate_host(self, hostname: str):
        if hostname in self.hosts:
            self.hosts[hostname].isolate()

    def isolate_iot(self, device_id: str):
        if device_id in self.iot_devices:
            self.iot_devices[device_id]["isolated"] = True

    def block_ip(self, ip_address: str):
        self.firewall_rules.append({"action": "DROP", "src_ip": ip_address})
        for host in self.hosts.values():
            host.firewall_blocklist.add(ip_address)

    def snapshot(self) -> Dict[str, Any]:
        """Generate snapshot compatible with flatten_network_state()."""
        host_dict = {}
        for hname, h in self.hosts.items():
            svcs = {}
            for sname, s in h.services.items():
                svcs[sname] = {
                    "compromised": s.compromised,
                    "active_sessions": s.active_sessions,
                    "port": s.port,
                    "vulnerability": s.vulnerability_score,
                }
            host_dict[hname] = {
                "isolated": h.isolated,
                "services": svcs,
            }

        iot_dict = {}
        for did, d in self.iot_devices.items():
            iot_dict[did] = {
                "isolated": d.get("isolated", False),
                "compromised": d.get("compromised", False),
                "firmware_integrity": d.get("firmware_integrity", 1.0),
            }

        return {"hosts": host_dict, "iot_devices": iot_dict}

    @classmethod
    def create_default(
        cls,
        n_hosts: int = 3,
        services_per_host: int = 2,
        n_iot_devices: int = 2,
        base_vulnerability: float = 0.3,
        rng: Optional[np.random.RandomState] = None,
    ) -> NetworkTopology:
        """Create a canonical testbed network matching SCEScenario specifications."""
        r = rng or np.random.RandomState(42)
        topo = cls()

        # Gateway / DMZ
        for i in range(n_hosts):
            hname = f"host_{i}"
            ip = f"10.0.1.{10 + i}"
            htype = HostType.GATEWAY if i == 0 else (HostType.HONEYPOT if i == 1 else HostType.SERVER)
            subnet = SubnetType.DMZ if i < 2 else SubnetType.CORP_NET
            host = topo.add_host(hname, ip, host_type=htype, subnet=subnet)

            for j in range(services_per_host):
                sname = f"svc_{i}_{j}"
                port = 22 if j == 0 else (80 if j == 1 else 1883)
                vuln = float(np.clip(base_vulnerability + r.normal(0, 0.05), 0.05, 0.95))
                is_hp = (htype == HostType.HONEYPOT)
                host.add_service(sname, port, vulnerability=vuln, is_honeypot=is_hp)

        for k in range(n_iot_devices):
            did = f"iot_{k}"
            dtype = "esp8266" if k % 2 == 0 else "esp32"
            topo.add_iot_device(did, device_type=dtype, firmware_integrity=1.0)

        return topo
