"""
rag/rag_pipeline.py — End-to-end Retrieve-then-Generate pipeline.

Responsibility
--------------
• Orchestrate retrieval → prompt construction → LLM call.
• Know NOTHING about which LLM or which vector store is used.
• Be fully testable by injecting mocks for BaseLLM and Retriever.

Interactions
------------
← Retriever provides ranked context chunks.
← BaseLLM (Ollama or OpenAI) generates the final answer.
→ Returns a RAGResponse with answer + source metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.llm.base_llm import BaseLLM
from app.retrieval.retriever import Retriever, RetrievedChunk
from app.utils import get_logger, RAGPipelineError

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Response model
# ---------------------------------------------------------------------------

@dataclass
class SourceChunk:
    """Minimal representation of a retrieved source chunk."""
    text: str
    score: float
    metadata: dict = field(default_factory=dict)


@dataclass
class RAGResponse:
    """What the pipeline returns to the API layer."""
    question: str
    answer: str
    sources: list[SourceChunk]
    model: str


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_PROMPT_TEMPLATE = """\
You are a helpful assistant. Answer the user's question using ONLY the \
context provided below. If the context does not contain enough information \
to answer, say so clearly — do not make up facts.

=== CONTEXT ===
{context}
=== END OF CONTEXT ===

Question: {question}

Answer:"""


class RAGPipeline:
    """
    The core RAG orchestrator.

    Deliberately depends only on the BaseLLM interface and the Retriever.
    Swapping Ollama for OpenAI (or FAISS for another vector store) requires
    zero changes here.

    Parameters
    ----------
    retriever : Retriever – encapsulates vector search + embedding
    llm       : BaseLLM  – any conforming LLM implementation
    """

    def __init__(self, retriever: Retriever, llm: BaseLLM) -> None:
        self._retriever = retriever
        self._llm = llm
        logger.info("RAGPipeline initialised with LLM=%r", self._llm)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(self, question: str) -> RAGResponse:
        """
        Run the full RAG cycle for *question*.

        Steps
        -----
        1. Retrieve top-k context chunks via Retriever.
        2. Build a structured prompt embedding those chunks.
        3. Call the LLM to generate an answer.
        4. Return a RAGResponse with answer + provenance.

        Raises
        ------
        RAGPipelineError – wraps any downstream error with context
        """
        if not question.strip():
            raise RAGPipelineError("Question must not be empty.")

        logger.info("RAG query: %r", question[:120])

        # 1. Retrieve
        try:
            chunks: list[RetrievedChunk] = self._retriever.retrieve(question)
        except Exception as exc:  # noqa: BLE001
            raise RAGPipelineError(f"Retrieval step failed: {exc}") from exc

        if not chunks:
            logger.warning("No relevant chunks found — answering with empty context.")

        # 2. Build prompt
        prompt = self._build_prompt(question, chunks)
        logger.debug("Prompt length: %d chars", len(prompt))

        # 3. Generate
        try:
            answer = self._llm.generate(prompt)
        except Exception as exc:  # noqa: BLE001
            raise RAGPipelineError(f"LLM generation step failed: {exc}") from exc

        # 4. Assemble response
        sources = [
            SourceChunk(
                text=chunk.text[:300],   # truncate for API response size
                score=round(chunk.score, 4),
                metadata=chunk.metadata,
            )
            for chunk in chunks
        ]

        logger.info("RAG answer generated (%d chars).", len(answer))
        return RAGResponse(
            question=question,
            answer=answer,
            sources=sources,
            model=self._llm.model_name,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
        """Assemble the context block and inject it into the template."""
        if not chunks:
            context = "(No relevant context found in the knowledge base.)"
        else:
            parts: list[str] = []
            for i, chunk in enumerate(chunks, start=1):
                source = chunk.metadata.get("filename", "unknown")
                page = chunk.metadata.get("page", "?")
                parts.append(
                    f"[{i}] Source: {source}, page {page}\n{chunk.text}"
                )
            context = "\n\n".join(parts)

        return _PROMPT_TEMPLATE.format(context=context, question=question)
