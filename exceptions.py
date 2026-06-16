"""
utils/exceptions.py — Domain-specific exceptions for the RAG system.

Catching these at API boundaries allows clean HTTP error mapping without
leaking internal implementation details.
"""

from __future__ import annotations


class RAGBaseError(Exception):
    """Root exception for all RAG system errors."""


# ---------------------------------------------------------------------------
# Ingestion layer
# ---------------------------------------------------------------------------

class DocumentLoadError(RAGBaseError):
    """Raised when a source document cannot be loaded or parsed."""


class ChunkingError(RAGBaseError):
    """Raised when document chunking fails."""


# ---------------------------------------------------------------------------
# Embedding layer
# ---------------------------------------------------------------------------

class EmbeddingError(RAGBaseError):
    """Raised when the embedding service returns an error."""


# ---------------------------------------------------------------------------
# Vector store layer
# ---------------------------------------------------------------------------

class VectorStoreError(RAGBaseError):
    """Raised when a FAISS (or other vector-store) operation fails."""


class VectorStoreNotInitializedError(VectorStoreError):
    """Raised when the vector store is used before being loaded/built."""


# ---------------------------------------------------------------------------
# LLM layer
# ---------------------------------------------------------------------------

class LLMError(RAGBaseError):
    """Raised when an LLM call fails."""


class LLMTimeoutError(LLMError):
    """Raised when an LLM call exceeds its timeout."""


# ---------------------------------------------------------------------------
# Retrieval / RAG layer
# ---------------------------------------------------------------------------

class RetrievalError(RAGBaseError):
    """Raised when retrieval returns no results or encounters an error."""


class RAGPipelineError(RAGBaseError):
    """Raised when the end-to-end RAG pipeline fails."""
