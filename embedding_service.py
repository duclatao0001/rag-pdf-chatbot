"""
embeddings/embedding_service.py — Converts text into dense vectors.

Responsibility
--------------
• Provide a single interface for embedding documents and queries.
• Support sentence-transformers (local, zero-cost) and OpenAI embeddings.
• Batch documents to avoid OOM errors.
• Cache the model handle to amortise loading cost.

Interactions
------------
← Receives text (chunks or query strings) from FaissVectorStore / Retriever.
→ Returns numpy float32 arrays consumed by FaissVectorStore.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from app.config import get_settings, EmbeddingProvider
from app.utils import get_logger, EmbeddingError

logger = get_logger(__name__)

# Type alias
FloatArray = NDArray[np.float32]


class EmbeddingService:
    """
    Unified embedding interface.

    Supports two backends (controlled by Settings.embedding_provider):
      • sentence_transformers — runs locally, no API key required
      • openai              — calls the OpenAI Embeddings API

    The backend is lazily initialised on first use.
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._model: object | None = None  # lazy init

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_documents(self, texts: list[str], batch_size: int = 64) -> FloatArray:
        """
        Embed a list of document chunks.

        Parameters
        ----------
        texts      : list of strings (chunk texts)
        batch_size : number of texts to embed per API / model call

        Returns
        -------
        (N, D) float32 numpy array where N = len(texts), D = embedding dim
        """
        if not texts:
            raise EmbeddingError("embed_documents received an empty text list.")

        logger.info("Embedding %d document chunk(s).", len(texts))
        return self._embed_batched(texts, batch_size)

    def embed_query(self, query: str) -> FloatArray:
        """
        Embed a single query string.

        Returns
        -------
        (D,) float32 numpy array
        """
        if not query.strip():
            raise EmbeddingError("embed_query received an empty query.")

        logger.debug("Embedding query: %r", query[:80])
        result = self._embed_batched([query], batch_size=1)
        return result[0]

    # ------------------------------------------------------------------
    # Backend dispatch
    # ------------------------------------------------------------------

    def _embed_batched(self, texts: list[str], batch_size: int) -> FloatArray:
        """Embed texts in batches; dispatches to the configured provider."""
        provider = self._settings.embedding_provider

        if provider == EmbeddingProvider.SENTENCE_TRANSFORMERS:
            return self._embed_st(texts, batch_size)
        if provider == EmbeddingProvider.OPENAI:
            return self._embed_openai(texts, batch_size)

        raise EmbeddingError(f"Unknown embedding provider: {provider!r}")

    # ------------------------------------------------------------------
    # Sentence-Transformers backend
    # ------------------------------------------------------------------

    def _get_st_model(self) -> object:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore[import]
            except ImportError as exc:
                raise EmbeddingError(
                    "sentence-transformers is not installed. "
                    "Run: pip install sentence-transformers"
                ) from exc

            model_name = self._settings.embedding_model
            logger.info("Loading SentenceTransformer model: %s", model_name)
            self._model = SentenceTransformer(model_name)
        return self._model

    def _embed_st(self, texts: list[str], batch_size: int) -> FloatArray:
        model = self._get_st_model()
        try:
            embeddings = model.encode(  # type: ignore[union-attr]
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        except Exception as exc:  # noqa: BLE001
            raise EmbeddingError(f"SentenceTransformer encoding failed: {exc}") from exc

        return embeddings.astype(np.float32)

    # ------------------------------------------------------------------
    # OpenAI backend
    # ------------------------------------------------------------------

    def _embed_openai(self, texts: list[str], batch_size: int) -> FloatArray:
        try:
            import openai  # type: ignore[import]
        except ImportError as exc:
            raise EmbeddingError(
                "openai package is not installed. Run: pip install openai"
            ) from exc

        api_key = self._settings.openai_api_key
        if not api_key:
            raise EmbeddingError(
                "OPENAI_API_KEY is not set. Add it to your .env file."
            )

        client = openai.OpenAI(api_key=api_key)
        all_embeddings: list[FloatArray] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            try:
                response = client.embeddings.create(
                    model=self._settings.embedding_model,
                    input=batch,
                )
            except openai.OpenAIError as exc:
                raise EmbeddingError(f"OpenAI embeddings API error: {exc}") from exc

            batch_vecs = np.array(
                [item.embedding for item in response.data], dtype=np.float32
            )
            all_embeddings.append(batch_vecs)
            logger.debug("Embedded batch %d/%d", i + len(batch), len(texts))

        return np.vstack(all_embeddings)
