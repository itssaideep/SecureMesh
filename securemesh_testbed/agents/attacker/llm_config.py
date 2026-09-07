# securemesh_testbed/agents/attacker/llm_config.py
"""LLM provider configuration for the AI-assisted attacker.

Supports two backends:
  * **Gemini** — Google's API (requires GEMINI_API_KEY env var)
  * **Ollama** — Local model server (requires Ollama running on localhost)

The configuration is loaded from environment variables with sensible defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class LLMProvider(str, Enum):
    GEMINI = "gemini"
    OLLAMA = "ollama"


@dataclass
class LLMConfig:
    """Configuration for the LLM attacker backend."""

    provider: str = "gemini"
    # Gemini settings
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"
    # Ollama settings
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    # Common settings
    temperature: float = 0.7
    max_tokens: int = 256
    # Rate limiting
    min_call_interval_s: float = 0.5  # min seconds between LLM calls
    cache_identical_states: bool = True
    # Fallback
    fallback_to_random: bool = True

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Load config from environment variables."""
        return cls(
            provider=os.environ.get("LLM_PROVIDER", "gemini"),
            gemini_api_key=os.environ.get("GEMINI_API_KEY", ""),
            gemini_model=os.environ.get("GEMINI_MODEL", "gemini-2.0-flash"),
            ollama_base_url=os.environ.get("OLLAMA_BASE_URL",
                                            "http://localhost:11434"),
            ollama_model=os.environ.get("OLLAMA_MODEL", "llama3.2"),
            temperature=float(os.environ.get("LLM_TEMPERATURE", "0.7")),
            max_tokens=int(os.environ.get("LLM_MAX_TOKENS", "256")),
        )

    @property
    def is_available(self) -> bool:
        """Check if the configured provider is available."""
        if self.provider == "gemini":
            return bool(self.gemini_api_key)
        elif self.provider == "ollama":
            try:
                import urllib.request
                req = urllib.request.Request(
                    f"{self.ollama_base_url}/api/tags",
                    method="GET",
                )
                with urllib.request.urlopen(req, timeout=2) as resp:
                    return resp.status == 200
            except Exception:
                return False
        return False
