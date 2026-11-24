"""Minimal OpenAI-compatible chat client (works with OpenAI, Groq, Hugging Face router, OpenRouter, ...)."""
from __future__ import annotations

import time
from dataclasses import dataclass

import requests

from .config import get_secret


@dataclass
class LLMResult:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0


class LLMClient:
    def __init__(self, api_key: str | None, base_url: str | None, model: str | None, timeout: int = 60):
        self.api_key, self.model, self.timeout = api_key, model, timeout
        self.base_url = (base_url or "").rstrip("/")

    @classmethod
    def from_secrets(cls) -> "LLMClient":
        return cls(get_secret("LLM_API_KEY"), get_secret("LLM_BASE_URL"), get_secret("LLM_MODEL"))

    @property
    def available(self) -> bool:
        return bool(self.api_key and self.base_url and self.model)

    def chat(self, messages: list[dict], temperature: float = 0.0, max_tokens: int = 700) -> LLMResult:
        if not self.available:
            raise RuntimeError("LLM is not configured (set LLM_API_KEY, LLM_BASE_URL, LLM_MODEL).")
        payload = {"model": self.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        start, last_error = time.perf_counter(), None
        for attempt in range(3):
            try:
                r = requests.post(f"{self.base_url}/chat/completions", json=payload, headers=headers, timeout=self.timeout)
                if r.status_code in (429, 500, 502, 503, 504):
                    last_error = f"HTTP {r.status_code}"
                    time.sleep(2 ** attempt)
                    continue
                r.raise_for_status()
                data = r.json()
                usage = data.get("usage") or {}
                return LLMResult(
                    text=(data["choices"][0]["message"]["content"] or "").strip(),
                    prompt_tokens=int(usage.get("prompt_tokens", 0)),
                    completion_tokens=int(usage.get("completion_tokens", 0)),
                    latency_ms=(time.perf_counter() - start) * 1000,
                )
            except requests.RequestException as exc:
                # never include headers/body in the message: they may contain the API key or document text
                last_error = type(exc).__name__
                time.sleep(2 ** attempt)
        raise RuntimeError(f"LLM request failed ({last_error}).")
