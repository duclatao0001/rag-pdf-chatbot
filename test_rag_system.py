"""
tests/test_rag_system.py — Unit and integration tests for the RAG system.

Tests are isolated: no actual LLM or PDF file is required.
Mocks are injected via dependency injection to keep tests fast and stable.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.ingestion.pdf_loader import Document
from app.ingestion.text_cleaner import TextCleaner
from app.ingestion.chunker import RecursiveTextChunker
from app.llm.base_llm import BaseLLM
from app.rag.rag_pipeline import RAGPipeline
from app.retrieval.retriever import Retriever, RetrievedChunk
from app.utils.exceptions import (
    ChunkingError,
    RAGPipelineError,
    RetrievalError,
    VectorStoreNotInitializedError,
)
from app.vectorstore.faiss_store import FaissVectorStore


# ---------------------------------------------------------------------------
# Fixtures / stubs
# ---------------------------------------------------------------------------

def _make_doc(text: str, page: int = 1) -> Document:
    return Document(page_content=text, metadata={"source": "test.pdf", "page": page})


class _StubLLM(BaseLLM):
    """LLM stub that echoes the prompt length as answer."""

    @property
    def model_name(self) -> str:
        return "stub-model"

    def generate(self, prompt: str) -> str:
        return f"Answer based on {len(prompt)} chars of context."


class _StubRetriever:
    """Retriever stub returning pre-canned chunks."""

    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self._chunks = chunks

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        return self._chunks


# ---------------------------------------------------------------------------
# TextCleaner tests
# ---------------------------------------------------------------------------

class TestTextCleaner:
    def test_removes_excessive_whitespace(self):
        doc = _make_doc("Hello   World\n\n\n\nBye")
        cleaned = TextCleaner().clean([doc])
        assert len(cleaned) == 1
        assert "   " not in cleaned[0].page_content
        assert "\n\n\n" not in cleaned[0].page_content

    def test_replaces_ligatures(self):
        doc = _make_doc("ef\ufb01cient")  # 'efi' ligature
        cleaned = TextCleaner().clean([doc])
        assert "fi" in cleaned[0].page_content

    def test_skips_blank_documents(self):
        doc = _make_doc("   \n  \t  ")
        result = TextCleaner().clean([doc])
        assert result == []

    def test_preserves_metadata(self):
        doc = _make_doc("some text", page=7)
        cleaned = TextCleaner().clean([doc])
        assert cleaned[0].metadata["page"] == 7


# ---------------------------------------------------------------------------
# Chunker tests
# ---------------------------------------------------------------------------

class TestChunker:
    def test_splits_long_text(self):
        text = "A" * 1200
        doc = _make_doc(text)
        chunks = RecursiveTextChunker(chunk_size=512, chunk_overlap=64).split([doc])
        assert len(chunks) > 1
        for c in chunks:
            assert len(c.page_content) <= 512

    def test_single_chunk_for_short_text(self):
        text = "Short text."
        doc = _make_doc(text)
        chunks = RecursiveTextChunker(chunk_size=512, chunk_overlap=64).split([doc])
        assert len(chunks) == 1

    def test_chunk_metadata_preserved(self):
        doc = _make_doc("B" * 600, page=3)
        chunks = RecursiveTextChunker(chunk_size=512, chunk_overlap=64).split([doc])
        assert all(c.metadata["page"] == 3 for c in chunks)
        assert all("chunk_index" in c.metadata for c in chunks)

    def test_invalid_overlap_raises(self):
        with pytest.raises(ChunkingError):
            RecursiveTextChunker(chunk_size=100, chunk_overlap=200)


# ---------------------------------------------------------------------------
# FaissVectorStore tests
# ---------------------------------------------------------------------------

class TestFaissVectorStore:
    DIM = 8

    def _store(self) -> FaissVectorStore:
        return FaissVectorStore(dimension=self.DIM)

    def _random_vecs(self, n: int) -> np.ndarray:
        rng = np.random.default_rng(42)
        vecs = rng.standard_normal((n, self.DIM)).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / norms

    def test_add_and_search(self):
        store = self._store()
        docs = [_make_doc(f"doc {i}") for i in range(5)]
        vecs = self._random_vecs(5)
        store.add_documents(docs, vecs)

        results = store.search(vecs[0], top_k=3)
        assert len(results) == 3
        # Top result should be the doc itself (cosine 1.0 with itself)
        assert results[0][0].page_content == "doc 0"
        assert results[0][1] == pytest.approx(1.0, abs=1e-5)

    def test_empty_store_raises(self):
        store = self._store()
        with pytest.raises(VectorStoreNotInitializedError):
            store.search(self._random_vecs(1)[0])

    def test_save_and_load(self, tmp_path):
        store = self._store()
        docs = [_make_doc("hello world")]
        vecs = self._random_vecs(1)
        store.add_documents(docs, vecs)
        store.save(tmp_path)

        store2 = FaissVectorStore(dimension=self.DIM)
        store2.load(tmp_path)
        assert store2.total_documents == 1
        results = store2.search(vecs[0], top_k=1)
        assert results[0][0].page_content == "hello world"

    def test_mismatch_raises(self):
        store = self._store()
        docs = [_make_doc("x"), _make_doc("y")]
        vecs = self._random_vecs(1)   # only 1 vector for 2 docs
        with pytest.raises(Exception):
            store.add_documents(docs, vecs)


# ---------------------------------------------------------------------------
# RAGPipeline tests
# ---------------------------------------------------------------------------

class TestRAGPipeline:
    def _pipeline(self, chunks: list[RetrievedChunk] | None = None) -> RAGPipeline:
        chunks = chunks or [
            RetrievedChunk(
                document=_make_doc("Paris is the capital of France."),
                score=0.95,
            )
        ]
        return RAGPipeline(
            retriever=_StubRetriever(chunks),  # type: ignore[arg-type]
            llm=_StubLLM(),
        )

    def test_returns_answer(self):
        pipeline = self._pipeline()
        response = pipeline.query("What is the capital of France?")
        assert "Answer" in response.answer
        assert response.question == "What is the capital of France?"

    def test_sources_populated(self):
        pipeline = self._pipeline()
        response = pipeline.query("capital?")
        assert len(response.sources) == 1
        assert response.sources[0].score == pytest.approx(0.95, abs=1e-3)

    def test_model_name_reported(self):
        pipeline = self._pipeline()
        response = pipeline.query("anything?")
        assert response.model == "stub-model"

    def test_empty_question_raises(self):
        pipeline = self._pipeline()
        with pytest.raises(RAGPipelineError):
            pipeline.query("   ")

    def test_empty_retrieval_still_answers(self):
        pipeline = self._pipeline(chunks=[])
        response = pipeline.query("something?")
        assert isinstance(response.answer, str)

    def test_retriever_error_wrapped(self):
        class _FailRetriever:
            def retrieve(self, query: str):
                raise RetrievalError("index corrupted")

        pipeline = RAGPipeline(retriever=_FailRetriever(), llm=_StubLLM())  # type: ignore[arg-type]
        with pytest.raises(RAGPipelineError, match="Retrieval step failed"):
            pipeline.query("test?")

    def test_llm_error_wrapped(self):
        from app.utils.exceptions import LLMError

        class _FailLLM(BaseLLM):
            @property
            def model_name(self) -> str:
                return "fail"
            def generate(self, prompt: str) -> str:
                raise LLMError("model offline")

        pipeline = RAGPipeline(
            retriever=_StubRetriever([RetrievedChunk(_make_doc("x"), 0.9)]),  # type: ignore[arg-type]
            llm=_FailLLM(),
        )
        with pytest.raises(RAGPipelineError, match="LLM generation step failed"):
            pipeline.query("test?")
