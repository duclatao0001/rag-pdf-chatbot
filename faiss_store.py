"""
vectorstore/faiss_store.py — FAISS-backed vector store.

Responsibility
--------------
• All FAISS-specific code lives here and ONLY here.
• Accepts numpy arrays from EmbeddingService.
• Stores documents (text + metadata) alongside their vectors.
• Persists index and metadata to disk; loads them back.
• Returns (Document, score) tuples on search.

Interactions
------------
← EmbeddingService provides float32 vectors.
← Retriever calls search() with a query vector.
← Ingestion pipeline calls add_documents() / save().
→ Returns ranked Document objects to Retriever.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Sequence

import faiss  # type: ignore[import]
import numpy as np
from numpy.typing import NDArray

from app.ingestion.pdf_loader import Document
from app.utils import get_logger, VectorStoreError, VectorStoreNotInitializedError

logger = get_logger(__name__)

FloatArray = NDArray[np.float32]

# Filenames written alongside the FAISS index
_INDEX_FILENAME = "index.faiss"
_DOCS_FILENAME = "documents.pkl"
_META_FILENAME = "metadata.json"


class FaissVectorStore:
    """
    FAISS inner-product index wrapped in a clean domain interface.

    Using inner product (IP) with L2-normalised vectors is equivalent to
    cosine similarity and gives slightly better recall than raw L2.

    Parameters
    ----------
    dimension : int – vector dimension (must match embedding model)
    """

    def __init__(self, dimension: int) -> None:
        self._dimension = dimension
        self._index: faiss.Index | None = None
        self._documents: list[Document] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_documents(
        self,
        documents: list[Document],
        embeddings: FloatArray,
    ) -> None:
        """
        Add documents and their pre-computed embeddings to the index.

        Parameters
        ----------
        documents  : list of Document objects (text + metadata)
        embeddings : (N, D) float32 array — one row per document
        """
        if len(documents) != embeddings.shape[0]:
            raise VectorStoreError(
                f"Mismatch: {len(documents)} documents vs "
                f"{embeddings.shape[0]} embeddings."
            )

        if self._index is None:
            logger.info("Initialising FAISS index (dim=%d).", self._dimension)
            self._index = faiss.IndexFlatIP(self._dimension)

        vectors = self._normalise(embeddings)
        self._index.add(vectors)  # type: ignore[union-attr]
        self._documents.extend(documents)

        logger.info(
            "Added %d documents. Total in store: %d.",
            len(documents),
            self._index.ntotal,
        )

    def search(
        self,
        query_vector: FloatArray,
        top_k: int = 5,
    ) -> list[tuple[Document, float]]:
        """
        Return the top-k most similar documents and their cosine scores.

        Parameters
        ----------
        query_vector : (D,) float32 array
        top_k        : number of results to return

        Returns
        -------
        list of (Document, score) sorted by descending similarity
        """
        self._require_initialised()

        query = self._normalise(query_vector.reshape(1, -1))
        k = min(top_k, self._index.ntotal)  # type: ignore[union-attr]

        scores, indices = self._index.search(query, k)  # type: ignore[union-attr]
        results: list[tuple[Document, float]] = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:  # FAISS sentinel for "not enough results"
                continue
            results.append((self._documents[idx], float(score)))

        logger.debug("Search returned %d result(s) (top_k=%d).", len(results), top_k)
        return results

    def save(self, path: str | Path) -> None:
        """Persist the FAISS index and document store to *path* directory."""
        self._require_initialised()
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, str(save_dir / _INDEX_FILENAME))

        with open(save_dir / _DOCS_FILENAME, "wb") as fh:
            pickle.dump(self._documents, fh)

        with open(save_dir / _META_FILENAME, "w", encoding="utf-8") as fh:
            json.dump({"dimension": self._dimension, "total": self._index.ntotal}, fh)  # type: ignore[union-attr]

        logger.info("Saved FAISS index (%d vectors) to %s.", self._index.ntotal, save_dir)  # type: ignore[union-attr]

    def load(self, path: str | Path) -> None:
        """Load a previously saved index from *path* directory."""
        load_dir = Path(path)
        index_file = load_dir / _INDEX_FILENAME
        docs_file = load_dir / _DOCS_FILENAME

        if not index_file.exists() or not docs_file.exists():
            raise VectorStoreError(
                f"Index files not found in {load_dir}. "
                "Run the ingestion pipeline first."
            )

        self._index = faiss.read_index(str(index_file))

        with open(docs_file, "rb") as fh:
            self._documents = pickle.load(fh)  # noqa: S301

        logger.info(
            "Loaded FAISS index (%d vectors) from %s.",
            self._index.ntotal,
            load_dir,
        )

    @property
    def is_empty(self) -> bool:
        return self._index is None or self._index.ntotal == 0

    @property
    def total_documents(self) -> int:
        return 0 if self._index is None else self._index.ntotal

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _require_initialised(self) -> None:
        if self._index is None or not self._documents:
            raise VectorStoreNotInitializedError(
                "Vector store is empty. Call add_documents() or load() first."
            )

    @staticmethod
    def _normalise(vectors: FloatArray) -> FloatArray:
        """L2-normalise rows so inner product == cosine similarity."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)  # avoid division by zero
        return (vectors / norms).astype(np.float32)
