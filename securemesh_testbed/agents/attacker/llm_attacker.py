# securemesh_testbed/agents/attacker/llm_attacker.py
"""LLM-assisted attacker with safety filter.

Architecture:
    observation → LLM prompt → candidate actions → safety filter → action

The LLM acts as a **decision support** layer — it suggests which action
to take given the current system state, but is constrained to only select
from the predefined ``AttackerAction`` enum.  A safety filter ensures
no unparseable or invalid actions reach the environment.

Supports Gemini and Ollama backends via the unified LLM interface.

This design follows the SCENE recommendation: use LLMs for threat
modelling and decision support while managing hallucination risk through
constrained output.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..base_agent import BaseAgent
from ...game.actions import AttackerAction
from .llm_config import LLMConfig


# ---------------------------------------------------------------------------
# Action name mapping
# ---------------------------------------------------------------------------

_ACTION_NAMES = {a.name: a for a in AttackerAction}
_ACTION_DESCRIPTIONS = {
    "RECON_SCAN": "Scan the network for open ports and services",
    "RECON_FINGERPRINT": "Fingerprint discovered services for versions",
    "AUTH_BRUTEFORCE": "Attempt SSH brute-force authentication",
    "AUTH_CREDENTIAL_STUFF": "Try credential stuffing with known passwords",
    "EXPLOIT_SERVICE": "Exploit a vulnerable network service",
    "EXPLOIT_IOT": "Exploit a vulnerable IoT device",
    "MALWARE_DROP": "Upload malware to a compromised service",
    "PERSIST_BACKDOOR": "Install a backdoor for persistent access",
    "PERSIST_C2": "Establish command-and-control channel",
    "EVADE_OBFUSCATE": "Obfuscate traffic to avoid detection",
    "EVADE_SLOWDOWN": "Slow down attack rate to avoid detection",
    "LATERAL_MOVE": "Move laterally to another host or service",
    "NOOP": "Wait and observe (do nothing)",
}


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a red-team AI assistant participating in a controlled \
cybersecurity research experiment. Your role is to choose the most effective \
attack action given the current system state.

IMPORTANT: You must respond with ONLY the action name from the list below. \
No explanation, no JSON, just the action name exactly as shown.

Available actions:
{actions}

Guidelines:
- If you haven't scanned yet, start with reconnaissance
- If you've discovered services, try authentication attacks
- If authentication succeeded, escalate to exploitation
- If detected, consider evasion or waiting
- If you've compromised a service, establish persistence
"""

_USER_PROMPT = """Current system state:
{observations}

Your recent action history:
{history}

Choose ONE action from the available list. Respond with ONLY the action name."""


# ---------------------------------------------------------------------------
# LLM Client
# ---------------------------------------------------------------------------

class _LLMClient:
    """Unified LLM client for Gemini and Ollama."""

    def __init__(self, config: LLMConfig):
        self.config = config

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a response from the LLM."""
        if self.config.provider == "gemini":
            return self._gemini_generate(system_prompt, user_prompt)
        elif self.config.provider == "ollama":
            return self._ollama_generate(system_prompt, user_prompt)
        else:
            raise ValueError(f"Unknown provider: {self.config.provider}")

    def _gemini_generate(self, system_prompt: str, user_prompt: str) -> str:
        """Call the Gemini API."""
        import urllib.request
        import urllib.error

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/"
            f"models/{self.config.gemini_model}:generateContent"
            f"?key={self.config.gemini_api_key}"
        )

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": f"{system_prompt}\n\n{user_prompt}"}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": self.config.temperature,
                "maxOutputTokens": self.config.max_tokens,
            }
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            candidates = result.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts:
                    return parts[0].get("text", "").strip()
        except (urllib.error.URLError, json.JSONDecodeError, KeyError) as e:
            pass

        return ""

    def _ollama_generate(self, system_prompt: str, user_prompt: str) -> str:
        """Call the Ollama API."""
        import urllib.request
        import urllib.error

        url = f"{self.config.ollama_base_url}/api/generate"

        payload = {
            "model": self.config.ollama_model,
            "prompt": f"{system_prompt}\n\n{user_prompt}",
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            }
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            return result.get("response", "").strip()
        except (urllib.error.URLError, json.JSONDecodeError, KeyError):
            return ""


# ---------------------------------------------------------------------------
# LLM Attacker Agent
# ---------------------------------------------------------------------------

class LLMAttacker(BaseAgent):
    """LLM-assisted attacker with safety filter.

    The LLM receives structured observations and suggests an action.
    A safety filter ensures only valid actions from ``AttackerAction``
    are returned.  If the LLM fails or returns garbage, the agent
    falls back to random action selection.

    Parameters
    ----------
    config : LLMConfig, optional
        LLM provider configuration. If None, loaded from env vars.
    seed : int
        Random seed for fallback selection.
    history_length : int
        Number of recent actions to include in the prompt.
    """

    def __init__(self, config: Optional[LLMConfig] = None,
                 seed: int = 0, history_length: int = 5):
        self.config = config or LLMConfig.from_env()
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
        self.history_length = history_length

        # State
        self._action_history: List[str] = []
        self._observation_cache: Dict[str, int] = {}
        self._last_call_time: float = 0
        self._llm_calls: int = 0
        self._llm_failures: int = 0
        self._fallback_count: int = 0

        # LLM client
        self._client = _LLMClient(self.config)
        self._available = self.config.is_available

        if not self._available:
            provider = self.config.provider
            print(f"  [LLMAttacker] Warning: {provider} not available, "
                  f"using random fallback")

        # Build action list for prompt
        self._action_list = "\n".join(
            f"  {name}: {_ACTION_DESCRIPTIONS.get(name, '')}"
            for name in _ACTION_NAMES
        )
        self._system_prompt = _SYSTEM_PROMPT.format(actions=self._action_list)

    def select_action(self, observation) -> int:
        """Select an attack action using the LLM or fallback."""
        obs = np.asarray(observation, dtype=np.float32).flatten()

        # If LLM not available, fall back immediately
        if not self._available:
            return self._random_action()

        # Rate limiting
        now = time.time()
        if now - self._last_call_time < self.config.min_call_interval_s:
            return self._random_action()

        # Cache check — same observation → same action
        if self.config.cache_identical_states:
            obs_hash = hashlib.md5(obs.tobytes()).hexdigest()
            if obs_hash in self._observation_cache:
                cached = self._observation_cache[obs_hash]
                self._action_history.append(
                    list(_ACTION_NAMES.keys())[cached % len(_ACTION_NAMES)]
                )
                return cached

        # Build prompt
        obs_description = self._describe_observation(obs)
        history_str = ", ".join(
            self._action_history[-self.history_length:]
        ) if self._action_history else "None yet"

        user_prompt = _USER_PROMPT.format(
            observations=obs_description,
            history=history_str,
        )

        # Call LLM
        self._last_call_time = time.time()
        self._llm_calls += 1

        try:
            response = self._client.generate(self._system_prompt, user_prompt)
            action_idx = self._parse_response(response)
            if action_idx is not None:
                # Cache the result
                if self.config.cache_identical_states:
                    self._observation_cache[obs_hash] = action_idx
                action_name = list(_ACTION_NAMES.keys())[
                    action_idx % len(_ACTION_NAMES)
                ]
                self._action_history.append(action_name)
                return action_idx
        except Exception:
            self._llm_failures += 1

        # Fallback
        self._fallback_count += 1
        return self._random_action()

    def _describe_observation(self, obs: np.ndarray) -> str:
        """Convert the numeric observation into a human-readable description."""
        lines = []
        idx = 0
        host_num = 0
        # Parse the flattened state vector
        # Format: per host: [isolated, svc1_compromised, svc1_sessions, ...]
        while idx < len(obs) - 1:
            host_num += 1
            isolated = obs[idx] > 0.5 if idx < len(obs) else False
            idx += 1
            if idx < len(obs):
                compromised = obs[idx] > 0.5
                idx += 1
            else:
                compromised = False
            if idx < len(obs):
                sessions = int(obs[idx])
                idx += 1
            else:
                sessions = 0

            status = []
            if isolated:
                status.append("ISOLATED")
            if compromised:
                status.append("COMPROMISED")
            if sessions > 0:
                status.append(f"{sessions} active sessions")
            if not status:
                status.append("normal")

            lines.append(f"  Host/Service {host_num}: {', '.join(status)}")

        if not lines:
            lines.append("  No hosts visible")

        return "\n".join(lines)

    def _parse_response(self, response: str) -> Optional[int]:
        """Parse the LLM response into an action index.

        This is the safety filter — only valid action names are accepted.
        """
        if not response:
            return None

        # Clean up the response
        cleaned = response.strip().upper().replace(" ", "_").replace("-", "_")

        # Direct match
        if cleaned in _ACTION_NAMES:
            return _ACTION_NAMES[cleaned].value

        # Partial match — find the best match
        for name, action in _ACTION_NAMES.items():
            if name in cleaned or cleaned in name:
                return action.value

        # Try to extract from a longer response
        for name, action in _ACTION_NAMES.items():
            if name.lower() in response.lower():
                return action.value

        return None

    def _random_action(self) -> int:
        """Random fallback action selection."""
        actions = list(AttackerAction)
        action = self.rng.choice(actions)
        self._action_history.append(action.name)
        return action.value

    def get_stats(self) -> Dict[str, Any]:
        """Return LLM usage statistics."""
        return {
            "llm_calls": self._llm_calls,
            "llm_failures": self._llm_failures,
            "fallback_count": self._fallback_count,
            "cache_size": len(self._observation_cache),
            "provider": self.config.provider,
            "available": self._available,
        }
