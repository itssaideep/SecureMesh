# securemesh_sce/environment/services/mqtt.py
"""MQTT Broker Simulator (Mosquitto-compatible semantics).

Simulates an MQTT broker handling IoT message publish/subscribe,
topic subscriptions, client authentication, and traffic telemetry.
"""

from __future__ import annotations

import time
from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass, field


@dataclass
class MQTTMessage:
    topic: str
    payload: str
    qos: int = 0
    retain: bool = False
    sender_client_id: str = "unknown"
    timestamp: float = field(default_factory=time.time)


class MQTTBrokerSimulator:
    """Simulated MQTT broker for IoT telemetry and command traffic."""

    def __init__(self, port: int = 1883, allow_anonymous: bool = True):
        self.port = port
        self.allow_anonymous = allow_anonymous
        self.subscriptions: Dict[str, Set[str]] = {}  # topic -> set of client_ids
        self.retained_messages: Dict[str, MQTTMessage] = {}
        self.clients: Dict[str, Dict[str, Any]] = {}
        self.message_history: List[MQTTMessage] = []
        self.is_running: bool = True

    def connect_client(
        self, client_id: str, username: Optional[str] = None, password: Optional[str] = None
    ) -> bool:
        if not self.is_running:
            return False

        if not self.allow_anonymous and not username:
            return False

        self.clients[client_id] = {
            "connected_at": time.time(),
            "username": username,
            "subscriptions": set(),
        }
        return True

    def subscribe(self, client_id: str, topic: str) -> bool:
        if client_id not in self.clients or not self.is_running:
            return False

        if topic not in self.subscriptions:
            self.subscriptions[topic] = set()

        self.subscriptions[topic].add(client_id)
        self.clients[client_id]["subscriptions"].add(topic)
        return True

    def publish(
        self,
        client_id: str,
        topic: str,
        payload: str,
        qos: int = 0,
        retain: bool = False,
    ) -> int:
        """Publish a message to a topic. Returns number of receiving subscribers."""
        if client_id not in self.clients or not self.is_running:
            return 0

        msg = MQTTMessage(
            topic=topic,
            payload=payload,
            qos=qos,
            retain=retain,
            sender_client_id=client_id,
        )
        self.message_history.append(msg)

        if retain:
            self.retained_messages[topic] = msg

        # Count matched subscribers (simple exact + wildcard matching)
        subscribers = 0
        for sub_topic, clients in self.subscriptions.items():
            if self._topic_matches(sub_topic, topic):
                subscribers += len(clients)

        return subscribers

    @staticmethod
    def _topic_matches(sub_topic: str, pub_topic: str) -> bool:
        if sub_topic == pub_topic or sub_topic == "#":
            return True
        if sub_topic.endswith("/#"):
            prefix = sub_topic[:-2]
            return pub_topic.startswith(prefix)
        return False

    def disconnect_client(self, client_id: str):
        if client_id in self.clients:
            for t in self.clients[client_id]["subscriptions"]:
                if t in self.subscriptions:
                    self.subscriptions[t].discard(client_id)
            del self.clients[client_id]

    def reset(self):
        self.subscriptions.clear()
        self.retained_messages.clear()
        self.clients.clear()
        self.message_history.clear()
        self.is_running = True
