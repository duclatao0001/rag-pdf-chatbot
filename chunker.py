"""
ingestion/chunker.py — Splits cleaned Documents into fixed-size chunks.

Responsibility
--------------
• Receive clean Documents from TextCleaner.
• Split each document's text using a sliding-window strategy
  (chunk_size / chunk_overlap from Settings).
• Propagate and enrich metadata (chunk index, char offsets).
• Return a flat list of chunk Documents ready for embedding.

Interactions
------------
← Receives cleaned Documents from TextCleaner.
→ Passes chunk Documents to EmbeddingService.
"""

from __future__ import annotations

from app.config import get_settings
from app.ingestion.pdf_loader import Document
from app.utils import get_logger, ChunkingError

logger = get_logger(__name__)


class RecursiveTextChunker:
    """
    Splits documents by character count with overlap.

    The algorithm walks the text with a sliding window:
      • step = chunk_size - chunk_overlap
      • each window becomes one chunk

    Parameters
    ----------
    chunk_size    : int – target maximum characters per chunk
    chunk_overlap : int – number of characters shared between adjacent chunks
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        settings = get_settings()
        self._chunk_size = chunk_size or settings.chunk_size
        self._chunk_overlap = chunk_overlap or settings.chunk_overlap

        if self._chunk_overlap >= self._chunk_size:
            raise ChunkingError(
                f"chunk_overlap ({self._chunk_overlap}) must be less than "
                f"chunk_size ({self._chunk_size})."
            )

    def split(self, documents: list[Document]) -> list[Document]:
        """Return a flat list of chunk Documents."""
        chunks: list[Document] = []
        for doc in documents:
            try:
                doc_chunks = self._split_document(doc)
                chunks.extend(doc_chunks)
            except Exception as exc:  # noqa: BLE001
                raise ChunkingError(
                    f"Chunking failed for document {doc.metadata}: {exc}"
                ) from exc

        logger.info(
            "Split %d document(s) into %d chunk(s) "
            "(size=%d, overlap=%d).",
            len(documents),
            len(chunks),
            self._chunk_size,
            self._chunk_overlap,
        )
        return chunks

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _split_document(self, doc: Document) -> list[Document]:
        text = doc.page_content
        step = self._chunk_size - self._chunk_overlap
        chunks: list[Document] = []
        start = 0
        chunk_index = 0

        while start < len(text):
            end = min(start + self._chunk_size, len(text))
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(
                    Document(
                        page_content=chunk_text,
                        metadata={
                            **doc.metadata,
                            "chunk_index": chunk_index,
                            "char_start": start,
                            "char_end": end,
                        },
                    )
                )
                chunk_index += 1

            if end == len(text):
                break

            start += step

        return chunks
