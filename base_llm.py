"""
llm/base_llm.py — Abstract base class for all LLM providers.

Responsibility
--------------
• Define the minimal contract every LLM implementation must satisfy.
• Keep the RAG pipeline fully decoupled from any concrete LLM.

Interactions
------------
→ Implemented by OllamaLLM and OpenAILLM.
← Used by RAGPipeline via dependency injection.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseLLM(ABC):
    """
    Interface contract for LLM providers.

    Any class that satisfies this interface can be dropped into the
    RAGPipeline without changing any pipeline code.
    """

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Send *prompt* to the model and return the generated text.

        Parameters
        ----------
        prompt : str – fully assembled prompt (system + context + question)

        Returns
        -------
        str – model output, stripped of leading/trailing whitespace

        Raises
        ------
        LLMError        – on any provider-level error
        LLMTimeoutError – when the provider does not respond in time
        """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable identifier of the underlying model."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name!r})"
