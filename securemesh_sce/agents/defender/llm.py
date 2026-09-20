# securemesh_sce/agents/defender/llm.py
"""Level 4: LLM-Assisted Autonomous Defender Agent.

Uses Large Language Models (Google Gemini 3.5 Flash) to analyze network telemetry,
detect cyber threats, and select optimal mitigation actions balancing security
efficacy against service availability.
"""

from __future__ import annotations

import logging
import numpy as np
from typing import Dict, Any, Optional, List

from ..common.llm_client import LLMClient
from ..attacker.scripted import BaseAgent
from ...game.actions import DefenderAction, N_DEFENDER_ACTIONS

logger = logging.getLogger("securemesh_sce.defender.llm")

# Action enum to 0-indexed integer mapping
DEFENDER_ACTION_MAP = {
    "MONITOR": 0,
    "INCREASE_MONITORING": 1,
    "ADJUST_IDS_THRESHOLD": 2,
    "RATE_LIMIT": 3,
    "TERMINATE_SESSION": 4,
    "RESET_CREDENTIALS": 5,
    "HONEYPOT_DEPLOY": 6,
    "BLOCK_IP": 7,
    "ISOLATE_SERVICE": 8,
    "ISOLATE_IOT": 9,
    "PATCH_SERVICE": 10,
    "RESTORE_SERVICE": 11,
    "NETWORK_SHUTDOWN": 12,
    "NOOP": 13,
}

SYSTEM_PROMPT = """You are an autonomous defensive blue-team cyber security agent operating in an IoT/Mesh honeypot testbed.
Your mission is to protect the network from active attacks while MAINTAINING MAXIMUM SERVICE AVAILABILITY.

CRITICAL OPERATIONAL CONSTRAINT:
Do NOT trigger destructive actions (ISOLATE_SERVICE, ISOLATE_IOT, NETWORK_SHUTDOWN) on mere recon or routine probes.
Premature shutdowns heavily penalize availability.
Use MONITOR, INCREASE_MONITORING, RATE_LIMIT, and BLOCK_IP for early and medium threats.
Only isolate services or devices when active compromise or malware drops are detected.
If services are degraded or down, consider RESTORE_SERVICE.

Available defender actions:
0: MONITOR (Passive telemetry collection, 0 cost)
1: INCREASE_MONITORING (Enhance logging & IDS sampling rate)
2: ADJUST_IDS_THRESHOLD (Lower anomaly detection threshold)
3: RATE_LIMIT (Throttle incoming connection rates)
4: TERMINATE_SESSION (Drop active suspicious connections)
5: RESET_CREDENTIALS (Force credential rotation)
6: HONEYPOT_DEPLOY (Redirect suspicious traffic to decoy)
7: BLOCK_IP (Add attacker IP to firewall blocklist)
8: ISOLATE_SERVICE (Take compromised server service offline)
9: ISOLATE_IOT (Disconnect compromised IoT device from mesh)
10: PATCH_SERVICE (Apply security patch; temporary downtime)
11: RESTORE_SERVICE (Bring isolated or offline service back online)
12: NETWORK_SHUTDOWN (Emergency complete network isolation - high penalty)
13: NOOP (Idle / maintain steady state)

Analyze the system state and output your reasoning and chosen action.
Respond ONLY with a valid JSON object in this exact schema:
{
  "reasoning": "<short explanation of defensive tactical strategy>",
  "action_name": "<ACTION_NAME>",
  "action_id": <0-13>
}
"""


class LLMDefender(BaseAgent):
    """LLM-driven cyber defense agent."""

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        fallback_strategy: str = "static",
        seed: int = 42,
    ):
        self.client = LLMClient(provider=provider, model=model, api_key=api_key)
        self.fallback_strategy = fallback_strategy
        self.rng = np.random.RandomState(seed)
        self.step_count = 0
        self.history: List[int] = []
        self.trace_logs: List[Dict[str, Any]] = []

    def reset(self):
        self.step_count = 0
        self.history.clear()
        self.trace_logs.clear()

    def _state_to_text(self, observation: np.ndarray) -> str:
        """Translate numeric observation into SOC-level telemetry context."""
        obs = np.asarray(observation, dtype=np.float32).ravel()
        obs_mean = float(obs.mean()) if len(obs) > 0 else 0.0
        obs_max = float(obs.max()) if len(obs) > 0 else 0.0

        recent_actions = [
            list(DEFENDER_ACTION_MAP.keys())[a] for a in self.history[-5:]
        ] if self.history else ["None (System Normal)"]

        return (
            f"Step: {self.step_count}\n"
            f"Recent defender actions: {', '.join(recent_actions)}\n"
            f"Telemetry anomaly signal (mean={obs_mean:.3f}, max={obs_max:.3f})\n"
            f"Network status: {'Normal / Low Activity' if obs_max < 0.3 else 'Elevated Threat / Suspicious Activity detected'}"
        )

    def select_action(self, observation: np.ndarray) -> int:
        self.step_count += 1
        prompt = self._state_to_text(observation)

        data, raw = self.client.generate_json(
            prompt=prompt,
            system_instruction=SYSTEM_PROMPT,
            temperature=0.2,
        )

        chosen_action: Optional[int] = None
        reasoning: str = ""

        if data and isinstance(data, dict):
            act_id = data.get("action_id")
            act_name = data.get("action_name", "")
            reasoning = data.get("reasoning", "")

            if isinstance(act_id, int) and 0 <= act_id < N_DEFENDER_ACTIONS:
                chosen_action = act_id
            elif act_name in DEFENDER_ACTION_MAP:
                chosen_action = DEFENDER_ACTION_MAP[act_name]

        if chosen_action is None:
            chosen_action = self._fallback_action(observation)
            reasoning = f"Fallback ({self.fallback_strategy}) due to LLM error: {raw[:80] if raw else 'None'}"
            logger.debug(reasoning)

        self.history.append(chosen_action)
        self.trace_logs.append({
            "step": self.step_count,
            "action": chosen_action,
            "action_name": list(DEFENDER_ACTION_MAP.keys())[chosen_action],
            "reasoning": reasoning,
            "llm_used": bool(data is not None),
        })

        return chosen_action

    def _fallback_action(self, observation: np.ndarray) -> int:
        """Heuristic defensive fallback: Monitor early, block if high signal, preserve availability."""
        obs = np.asarray(observation, dtype=np.float32).ravel()
        obs_max = float(obs.max()) if len(obs) > 0 else 0.0
        if obs_max > 0.6:
            return DEFENDER_ACTION_MAP["BLOCK_IP"]
        elif obs_max > 0.3:
            return DEFENDER_ACTION_MAP["RATE_LIMIT"]
        elif self.step_count % 5 == 0:
            return DEFENDER_ACTION_MAP["INCREASE_MONITORING"]
        else:
            return DEFENDER_ACTION_MAP["MONITOR"]
