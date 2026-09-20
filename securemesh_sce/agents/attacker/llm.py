# securemesh_sce/agents/attacker/llm.py
"""Level 4: LLM-Assisted Offensive Planning Agent.

Formulates cyber kill-chain actions dynamically using Large Language Models
(Google Gemini / Local Ollama), with semantic state interpretation and robust
heuristic fallback.
"""

from __future__ import annotations

import logging
import numpy as np
from typing import Dict, Any, Optional, List

from ..common.llm_client import LLMClient
from .scripted import BaseAgent
from ...game.actions import AttackerAction, N_ATTACKER_ACTIONS

logger = logging.getLogger("securemesh_sce.attacker.llm")

# Action enum to 0-indexed integer mapping
ACTION_MAP = {
    "RECON_SCAN": 0,
    "RECON_FINGERPRINT": 1,
    "AUTH_BRUTEFORCE": 2,
    "AUTH_CREDENTIAL_STUFF": 3,
    "EXPLOIT_SERVICE": 4,
    "EXPLOIT_IOT": 5,
    "MALWARE_DROP": 6,
    "PERSIST_BACKDOOR": 7,
    "PERSIST_C2": 8,
    "EVADE_OBFUSCATE": 9,
    "EVADE_SLOWDOWN": 10,
    "LATERAL_MOVE": 11,
    "NOOP": 12,
}

SYSTEM_PROMPT = """You are an autonomous offensive red-team cyber security agent operating in an IoT/Mesh honeypot testbed.
Your objective is to execute an effective cyber kill-chain while adapting to defender mitigations.

Available actions:
0: RECON_SCAN (Fast port/service scan)
1: RECON_FINGERPRINT (Slow, stealthy OS/service fingerprint)
2: AUTH_BRUTEFORCE (High-volume credential brute-force)
3: AUTH_CREDENTIAL_STUFF (Targeted credential stuffing)
4: EXPLOIT_SERVICE (Exploit vulnerable server daemon)
5: EXPLOIT_IOT (Exploit embedded IoT firmware vulnerability)
6: MALWARE_DROP (Upload payload/dropper to compromised target)
7: PERSIST_BACKDOOR (Establish persistent reverse shell/backdoor)
8: PERSIST_C2 (Configure Command & Control beacon)
9: EVADE_OBFUSCATE (Obfuscate traffic/signatures to evade IDS)
10: EVADE_SLOWDOWN (Temporarily reduce attack rate to reduce anomaly score)
11: LATERAL_MOVE (Pivot to adjacent network host)
12: NOOP (Idle/observe)

Analyze the current state and provide your strategic reasoning and selected action.
Respond ONLY with a valid JSON object in this exact schema:
{
  "reasoning": "<short explanation of tactical strategy>",
  "action_name": "<ACTION_NAME>",
  "action_id": <0-12>
}
"""


class LLMAttacker(BaseAgent):
    """LLM-driven cyber kill-chain attacker."""

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        fallback_strategy: str = "scripted",
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
        """Translate numeric observation into high-level cybersecurity context."""
        obs = np.asarray(observation, dtype=np.float32).ravel()
        # Basic state feature interpretation
        obs_mean = float(obs.mean()) if len(obs) > 0 else 0.0
        obs_max = float(obs.max()) if len(obs) > 0 else 0.0
        recent_actions = [
            list(ACTION_MAP.keys())[a] for a in self.history[-5:]
        ] if self.history else ["None (Start of attack)"]

        return (
            f"Step: {self.step_count}\n"
            f"Recent actions taken: {', '.join(recent_actions)}\n"
            f"Observed network activity signal (mean={obs_mean:.3f}, max={obs_max:.3f})\n"
            f"Current phase: {'Reconnaissance / Initial Access' if self.step_count < 15 else 'Exploitation / Persistence / Evasion'}"
        )

    def select_action(self, observation: np.ndarray) -> int:
        self.step_count += 1
        prompt = self._state_to_text(observation)

        data, raw = self.client.generate_json(
            prompt=prompt,
            system_instruction=SYSTEM_PROMPT,
            temperature=0.3,
        )

        chosen_action: Optional[int] = None
        reasoning: str = ""

        if data and isinstance(data, dict):
            # Try action_id first
            act_id = data.get("action_id")
            act_name = data.get("action_name", "")
            reasoning = data.get("reasoning", "")

            if isinstance(act_id, int) and 0 <= act_id < N_ATTACKER_ACTIONS:
                chosen_action = act_id
            elif act_name in ACTION_MAP:
                chosen_action = ACTION_MAP[act_name]

        if chosen_action is None:
            # Fallback heuristic if LLM fails or is unauthenticated
            chosen_action = self._fallback_action()
            reasoning = f"Fallback ({self.fallback_strategy}) due to LLM error or unauthenticated key: {raw[:80] if raw else 'None'}"
            logger.debug(reasoning)

        self.history.append(chosen_action)
        self.trace_logs.append({
            "step": self.step_count,
            "action": chosen_action,
            "action_name": list(ACTION_MAP.keys())[chosen_action],
            "reasoning": reasoning,
            "llm_used": bool(data is not None),
        })

        return chosen_action

    def _fallback_action(self) -> int:
        """Deterministic heuristic kill-chain fallback."""
        s = self.step_count % 15
        if s in (1, 2):
            return ACTION_MAP["RECON_SCAN"]
        elif s == 3:
            return ACTION_MAP["RECON_FINGERPRINT"]
        elif s in (4, 5, 6):
            return ACTION_MAP["AUTH_BRUTEFORCE"]
        elif s in (7, 8):
            return ACTION_MAP["EXPLOIT_SERVICE"]
        elif s == 9:
            return ACTION_MAP["EXPLOIT_IOT"]
        elif s == 10:
            return ACTION_MAP["MALWARE_DROP"]
        elif s == 11:
            return ACTION_MAP["PERSIST_BACKDOOR"]
        elif s == 12:
            return ACTION_MAP["EVADE_OBFUSCATE"]
        elif s == 13:
            return ACTION_MAP["EVADE_SLOWDOWN"]
        else:
            return ACTION_MAP["NOOP"]
