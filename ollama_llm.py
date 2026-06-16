"""
llm/ollama_llm.py — Ollama LLM provider (local inference).

Responsibility
--------------
• Implement BaseLLM using the Ollama HTTP API.
• Handle connection errors, timeouts, and non-200 responses.
• Log every call for observability.

Interactions
------------
← Constructed by the LLM factory using Settings.
← Called by RAGPipeline.generate(prompt).
→ Makes HTTP POST to the Ollama /api/generate endpoint.
"""

from __future__ import annotations

import json

import httpx

from app.config import get_settings
from app.llm.base_llm import BaseLLM
from app.utils import get_logger, LLMError, LLMTimeoutError

logger = get_logger(__name__)


class OllamaLLM(BaseLLM):
    """
    Calls a locally-running Ollama server.

    Default base URL  : http://localhost:11434
    Default model     : llama3
    Both are overridable via Settings / .env.
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        timeout: int | None = None,
    ) -> None:
        settings = get_settings()
        self._model = model or settings.ollama_model
        self._base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self._timeout = timeout or settings.ollama_timeout

    # ------------------------------------------------------------------
    # BaseLLM interface
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        return self._model

    def generate(self, prompt: str) -> str:
        url = f"{self._base_url}/api/generate"
        payload = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
        }

        logger.debug("Calling Ollama model=%r url=%s", self._model, url)

        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.post(url, json=payload)
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(
                f"Ollama timed out after {self._timeout}s "
                f"(model={self._model!r})"
            ) from exc
        except httpx.RequestError as exc:
            raise LLMError(
                f"Cannot reach Ollama at {self._base_url}: {exc}"
            ) from exc

        if response.status_code != 200:
            raise LLMError(
                f"Ollama returned HTTP {response.status_code}: {response.text[:200]}"
            )

        try:
            data = response.json()
        except json.JSONDecodeError as exc:
            raise LLMError(f"Ollama returned non-JSON body: {exc}") from exc

        text: str = data.get("response", "").strip()
        if not text:
            raise LLMError("Ollama returned an empty response body.")

        logger.debug("Ollama response length: %d chars", len(text))
        return text
