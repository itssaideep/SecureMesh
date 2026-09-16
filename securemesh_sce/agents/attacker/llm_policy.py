# securemesh_sce/agents/attacker/llm_policy.py
"""Level 4: LLM-assisted attacker.

Uses a large language model (Gemini or Ollama) to select attacker actions
based on textual state descriptions. Provides the most flexible and
creative attack strategies for red-team testing.
"""

from __future__ import annotations

import os
import json
import time
import numpy as np
from typing import Optional, Dict, Any

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from .scripted import BaseAgent
from ...game.actions import AttackerAction, N_ATTACKER_ACTIONS


# Valid action names for LLM output parsing
ACTION_NAMES = [a.name for a in AttackerAction]


class LLMAttacker(BaseAgent):
    """Level 4: LLM-assisted attacker.

    Queries an LLM with a structured prompt describing the current state,
    and parses the response into an attacker action.
    """

    def __init__(
        self,
        provider: str = "gemini",
        model: str = "gemini-2.0-flash",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        fallback_seed: int = 0,
        seed: Optional[int] = None,
        temperature: float = 0.7,
        max_retries: int = 3,
    ):
        fallback_seed = seed if seed is not None else fallback_seed
        """
        Parameters
        ----------
        provider
            "gemini" or "ollama".
        model
            Model identifier (e.g., "gemini-2.0-flash", "llama3.1:8b").
        api_key
            API key (for Gemini). If None, reads from GEMINI_API_KEY env var.
        base_url
            Base URL for Ollama (default: http://localhost:11434).
        fallback_seed
            Random seed for fallback action when LLM fails.
        temperature
            Sampling temperature.
        max_retries
            Number of retries on LLM failure.
        """
        self.provider = provider.lower()
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.rng = np.random.RandomState(fallback_seed)

        if self.provider == "gemini":
            self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        elif self.provider == "ollama":
            self.base_url = base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        else:
            raise ValueError(f"Unknown provider: {provider}")

        self._history: list = []
        self._step = 0

    def _build_prompt(self, observation: np.ndarray) -> str:
        """Build a structured prompt for the LLM."""
        obs_summary = {
            "obs_vector_length": len(observation),
            "obs_mean": float(observation.mean()),
            "obs_max": float(observation.max()),
            "step": self._step,
        }

        prompt = f"""You are a red-team security tester in an IoT Security Chaos Engineering experiment.

Current system state (step {self._step}):
{json.dumps(obs_summary, indent=2)}

Your goal is to discover vulnerabilities and test the resilience of the IoT network.
You must choose ONE action from this list:
{json.dumps(ACTION_NAMES, indent=2)}

Recent actions: {json.dumps(self._history[-5:])}

Respond with ONLY the action name (e.g., "RECON_SCAN"). No explanation needed."""

        return prompt

    def _query_llm(self, prompt: str) -> Optional[str]:
        """Query the LLM and return the raw response."""
        try:
            if self.provider == "gemini":
                return self._query_gemini(prompt)
            elif self.provider == "ollama":
                return self._query_ollama(prompt)
        except Exception:
            return None

    def _query_gemini(self, prompt: str) -> Optional[str]:
        """Query Gemini API."""
        import urllib.request
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={self.api_key}"
        )
        data = json.dumps({
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": self.temperature, "maxOutputTokens": 50},
        }).encode()
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read())
        return result["candidates"][0]["content"]["parts"][0]["text"].strip()

    def _query_ollama(self, prompt: str) -> Optional[str]:
        """Query Ollama API."""
        import urllib.request
        url = f"{self.base_url}/api/generate"
        data = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.temperature, "num_predict": 50},
        }).encode()
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())
        return result.get("response", "").strip()

    def _parse_action(self, response: str) -> Optional[int]:
        """Parse LLM response into an action index."""
        if not response:
            return None
        response_upper = response.upper().strip().strip('"').strip("'")
        for i, name in enumerate(ACTION_NAMES):
            if name in response_upper:
                return i
        return None

    def select_action(self, observation: np.ndarray) -> int:
        """Query LLM for action; fall back to random on failure."""
        prompt = self._build_prompt(observation)

        for attempt in range(self.max_retries):
            response = self._query_llm(prompt)
            action_idx = self._parse_action(response) if response else None
            if action_idx is not None:
                self._history.append(ACTION_NAMES[action_idx])
                self._step += 1
                return action_idx
            time.sleep(0.5 * (attempt + 1))

        # Fallback: random action
        action_idx = int(self.rng.randint(0, N_ATTACKER_ACTIONS))
        self._history.append(f"FALLBACK_{ACTION_NAMES[action_idx]}")
        self._step += 1
        return action_idx

    def reset(self):
        self._history.clear()
        self._step = 0
