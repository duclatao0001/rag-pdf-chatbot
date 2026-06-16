"""
llm/openai_llm.py — OpenAI LLM provider (cloud inference).

Responsibility
--------------
• Implement BaseLLM using the OpenAI Chat Completions API.
• Read credentials and model settings from Settings.
• Translate OpenAI-specific errors into domain LLMError / LLMTimeoutError.

Interactions
------------
← Constructed by the LLM factory using Settings.
← Called by RAGPipeline.generate(prompt).
→ Makes API calls to api.openai.com.
"""

from __future__ import annotations

from app.config import get_settings
from app.llm.base_llm import BaseLLM
from app.utils import get_logger, LLMError, LLMTimeoutError

logger = get_logger(__name__)


class OpenAILLM(BaseLLM):
    """
    Calls the OpenAI Chat Completions API (gpt-4o-mini by default).

    Requires OPENAI_API_KEY in environment / .env file.
    """

    def __init__(
        self,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: int | None = None,
    ) -> None:
        settings = get_settings()
        self._model = model or settings.openai_model
        self._temperature = temperature if temperature is not None else settings.openai_temperature
        self._max_tokens = max_tokens or settings.openai_max_tokens
        self._timeout = timeout or settings.openai_timeout
        self._api_key = settings.openai_api_key

        if not self._api_key:
            raise LLMError(
                "OPENAI_API_KEY is not set. "
                "Add it to your .env file or environment."
            )

    # ------------------------------------------------------------------
    # BaseLLM interface
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        return self._model

    def generate(self, prompt: str) -> str:
        try:
            import openai  # type: ignore[import]
        except ImportError as exc:
            raise LLMError(
                "openai package is not installed. Run: pip install openai"
            ) from exc

        client = openai.OpenAI(api_key=self._api_key, timeout=self._timeout)

        logger.debug("Calling OpenAI model=%r", self._model)

        try:
            completion = client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
        except openai.APITimeoutError as exc:
            raise LLMTimeoutError(
                f"OpenAI timed out after {self._timeout}s (model={self._model!r})"
            ) from exc
        except openai.AuthenticationError as exc:
            raise LLMError(f"OpenAI authentication failed: {exc}") from exc
        except openai.RateLimitError as exc:
            raise LLMError(f"OpenAI rate limit exceeded: {exc}") from exc
        except openai.OpenAIError as exc:
            raise LLMError(f"OpenAI API error: {exc}") from exc

        text = completion.choices[0].message.content or ""
        text = text.strip()

        if not text:
            raise LLMError("OpenAI returned an empty completion.")

        logger.debug(
            "OpenAI response: model=%r, tokens=%d, length=%d chars",
            self._model,
            completion.usage.total_tokens if completion.usage else -1,
            len(text),
        )
        return text
