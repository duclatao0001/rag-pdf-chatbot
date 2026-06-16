"""
retrieval/retriever.py — Retrieves the most relevant chunks for a query.

Responsibility
--------------
• Embed the user query using EmbeddingService.
• Delegate vector search to FaissVectorStore.
• Return the top-k (Document, score) pairs to the RAG pipeline.
• Apply optional minimum-score filtering.

Interactions
------------
← Receives query string from RAGPipeline.
← Uses EmbeddingService to embed the query.
← Uses FaissVectorStore.search() to rank chunks.
→ Returns list[tuple[Document, float]] to RAGPipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.config import get_settings
from app.embeddings.embedding_service import EmbeddingService
from app.ingestion.pdf_loader import Document
from app.utils import get_logger, RetrievalError
from app.vectorstore.faiss_store import FaissVectorStore

logger = get_logger(__name__)


@dataclass
class RetrievedChunk:
    """A document chunk together with its similarity score."""

    document: Document
    score: float

    @property
    def text(self) -> str:
        return self.document.page_content

    @property
    def metadata(self) -> dict:
        return self.document.metadata


class Retriever:
    """
    Combines embedding and vector search into a single retrieve() call.

    Parameters
    ----------
    vector_store      : FaissVectorStore – pre-loaded/built vector store
    embedding_service : EmbeddingService – to embed the query
    top_k             : int              – number of results (default from Settings)
    min_score         : float            – minimum cosine score (0–1, 0 = no filter)
    """

    def __init__(
        self,
        vector_store: FaissVectorStore,
        embedding_service: EmbeddingService,
        top_k: int | None = None,
        min_score: float = 0.0,
    ) -> None:
        settings = get_settings()
        self._store = vector_store
        self._embedder = embedding_service
        self._top_k = top_k or settings.retrieval_top_k
        self._min_score = min_score

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        """
        Embed *query* and return the top-k most similar chunks.

        Parameters
        ----------
        query : str – natural-language question

        Returns
        -------
        list of RetrievedChunk ordered by descending similarity

        Raises
        ------
        RetrievalError – when the store is empty or no results pass the filter
        """
        if not query.strip():
            raise RetrievalError("Retriever received an empty query.")

        if self._store.is_empty:
            raise RetrievalError(
                "Vector store is empty. "
                "Ingest documents before querying."
            )

        logger.debug("Retrieving top-%d chunks for query: %r", self._top_k, query[:80])

        # 1. Embed query
        query_vector = self._embedder.embed_query(query)

        # 2. Vector search
        raw_results = self._store.search(query_vector, top_k=self._top_k)

        # 3. Optional score filtering
        results = [
            RetrievedChunk(document=doc, score=score)
            for doc, score in raw_results
            if score >= self._min_score
        ]

        if not results:
            logger.warning(
                "No chunks passed the min_score filter (%.2f) for query: %r",
                self._min_score,
                query[:80],
            )

        logger.info(
            "Retrieved %d chunk(s) (min_score=%.2f).",
            len(results),
            self._min_score,
        )
        return results
