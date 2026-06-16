"""
ingestion/text_cleaner.py — Normalises raw extracted text.

Responsibility
--------------
• Receive raw Document objects from PDFLoader.
• Remove noise: headers/footers artifacts, excessive whitespace, ligatures,
  non-printable characters, etc.
• Return cleaned Document objects — content is altered, metadata is preserved.

Interactions
------------
← Receives raw Documents from PDFLoader.
→ Passes cleaned Documents to Chunker.
"""

from __future__ import annotations

import re
import unicodedata

from app.ingestion.pdf_loader import Document
from app.utils import get_logger, ChunkingError

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Ligature / Unicode normalisation map
# ---------------------------------------------------------------------------

_LIGATURE_MAP: dict[str, str] = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\u2019": "'",
    "\u2018": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2013": "-",
    "\u2014": "--",
    "\u2026": "...",
}

_LIGATURE_TABLE = str.maketrans(_LIGATURE_MAP)


class TextCleaner:
    """
    Stateless text normaliser.

    All cleaning steps are composable: add / remove individual steps without
    changing the public interface.
    """

    def clean(self, documents: list[Document]) -> list[Document]:
        """Return a new list of Documents with cleaned page_content."""
        cleaned: list[Document] = []
        for doc in documents:
            try:
                text = self._clean_text(doc.page_content)
                if not text:
                    logger.debug("Document from %s page %s is empty after cleaning — skipping.",
                                 doc.metadata.get("source"), doc.metadata.get("page"))
                    continue
                cleaned.append(Document(page_content=text, metadata=doc.metadata.copy()))
            except Exception as exc:  # noqa: BLE001
                raise ChunkingError(f"Text cleaning failed for {doc.metadata}: {exc}") from exc

        logger.info("Cleaned %d/%d documents.", len(cleaned), len(documents))
        return cleaned

    # ------------------------------------------------------------------
    # Private pipeline
    # ------------------------------------------------------------------

    def _clean_text(self, text: str) -> str:
        text = self._replace_ligatures(text)
        text = self._normalise_unicode(text)
        text = self._remove_non_printable(text)
        text = self._fix_hyphenation(text)
        text = self._collapse_whitespace(text)
        return text.strip()

    @staticmethod
    def _replace_ligatures(text: str) -> str:
        return text.translate(_LIGATURE_TABLE)

    @staticmethod
    def _normalise_unicode(text: str) -> str:
        return unicodedata.normalize("NFKC", text)

    @staticmethod
    def _remove_non_printable(text: str) -> str:
        # Keep newlines and tabs; strip other control chars
        return re.sub(r"[^\x09\x0A\x0D\x20-\x7E\x80-\xFF]", " ", text)

    @staticmethod
    def _fix_hyphenation(text: str) -> str:
        """Rejoin words broken across lines with a soft hyphen."""
        return re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    @staticmethod
    def _collapse_whitespace(text: str) -> str:
        # Collapse runs of spaces/tabs to a single space
        text = re.sub(r"[ \t]+", " ", text)
        # Collapse 3+ consecutive newlines to 2
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text
