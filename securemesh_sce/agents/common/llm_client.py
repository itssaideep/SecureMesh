# securemesh_sce/agents/common/llm_client.py
"""LLM client abstraction for SecureMesh-SCE.

Supports:
1. Google Gemini via REST API (e.g., gemini-2.0-flash, gemini-1.5-flash).
2. Local Ollama via REST API (e.g., llama3.1:8b).
3. Graceful fallback on network errors, rate limits, or invalid credentials.
"""

from __future__ import annotations

import json
import logging
import os
import requests
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger("securemesh_sce.llm_client")


class LLMClient:
    """Unified client for invoking LLMs (Gemini / Ollama) with structured output."""

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        self.provider = (provider or os.getenv("LLM_PROVIDER", "gemini")).lower()
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.timeout = timeout if timeout is not None else float(os.getenv("LLM_TIMEOUT", "15.0"))
        self.ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")

        logger.info(
            f"Initialized LLMClient (provider={self.provider}, model={self.model}, "
            f"has_key={bool(self.api_key)})"
        )

    def generate_json(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.2,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Generate response constrained to JSON format.

        Returns (parsed_json_dict, raw_text). If failed, returns (None, error_or_raw).
        """
        if self.provider == "gemini":
            return self._call_gemini(prompt, system_instruction, temperature)
        elif self.provider in ("ollama", "llama"):
            return self._call_ollama(prompt, system_instruction, temperature)
        else:
            logger.warning(f"Unsupported LLM provider: {self.provider}")
            return None, f"Unsupported provider: {self.provider}"

    def _call_gemini(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.2,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Call Google Gemini REST API."""
        if not self.api_key:
            return None, "Missing GEMINI_API_KEY"

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent"
        )
        headers = {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json",
        }

        contents = []
        if system_instruction:
            contents.append({
                "role": "user",
                "parts": [{"text": f"System Instructions:\n{system_instruction}\n\nTask:\n{prompt}"}]
            })
        else:
            contents.append({
                "role": "user",
                "parts": [{"text": prompt}]
            })

        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "responseMimeType": "application/json",
            }
        }

        models_to_try = [self.model]
        for fallback in ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-flash-latest"]:
            if fallback not in models_to_try:
                models_to_try.append(fallback)

        last_error = ""
        for m in models_to_try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent"
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
                if resp.status_code == 200:
                    data = resp.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        raw_text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                        try:
                            parsed = json.loads(raw_text)
                            return parsed, raw_text
                        except json.JSONDecodeError:
                            cleaned = raw_text.strip().removeprefix("```json").removesuffix("```").strip()
                            return json.loads(cleaned), raw_text
                
                last_error = f"Model {m} returned HTTP {resp.status_code}: {resp.text[:150]}"
                if resp.status_code not in (429, 503):
                    # Hard error (e.g. 400 bad request) — don't keep trying
                    break

            except Exception as e:
                last_error = f"Model {m} error: {e}"

        logger.warning(f"All Gemini models failed. Last error: {last_error}")
        return None, last_error

    def _call_ollama(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.2,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Call local Ollama REST API."""
        url = f"{self.ollama_url}/api/generate"
        full_prompt = f"{system_instruction}\n\n{prompt}" if system_instruction else prompt

        payload = {
            "model": self.model if "llama" in self.model else "llama3.1:8b",
            "prompt": full_prompt,
            "format": "json",
            "stream": False,
            "options": {"temperature": temperature},
        }

        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
            if resp.status_code != 200:
                return None, f"Ollama HTTP {resp.status_code}: {resp.text[:200]}"

            raw_text = resp.json().get("response", "")
            return json.loads(raw_text), raw_text
        except Exception as e:
            logger.warning(f"Ollama call failed: {e}")
            return None, str(e)
