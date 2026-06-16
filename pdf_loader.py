"""
ingestion/pdf_loader.py — Loads raw text from PDF files.

Responsibility
--------------
• Accept a file path (or list of paths / a directory).
• Extract raw text page-by-page using PyMuPDF (fitz).
• Return a list of Document objects containing text + metadata.
• No cleaning or chunking — that is delegated downstream.

Interactions
------------
→ Called by the ingestion pipeline / API route to start the pipeline.
→ Passes raw Document objects to TextCleaner.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import fitz  # PyMuPDF

from app.utils import get_logger, DocumentLoadError

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Domain model
# ---------------------------------------------------------------------------

@dataclass
class Document:
    """Represents a single page (or chunk) of source text."""

    page_content: str
    metadata: dict = field(default_factory=dict)

    @property
    def doc_id(self) -> str:
        """Stable SHA-256 id derived from content + source metadata."""
        key = f"{self.metadata.get('source', '')}:{self.metadata.get('page', 0)}:{self.page_content}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

class PDFLoader:
    """
    Loads one or more PDF files and returns a flat list of Document objects,
    one per page.

    Parameters
    ----------
    paths : str | Path | Sequence[str | Path]
        A single PDF path, a list of paths, or a directory (all .pdf files
        inside will be loaded).
    """

    def __init__(self, paths: str | Path | Sequence[str | Path]) -> None:
        self._paths: list[Path] = self._resolve_paths(paths)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> list[Document]:
        """Return a flat list of Documents across all resolved PDF files."""
        documents: list[Document] = []
        for pdf_path in self._paths:
            documents.extend(self._load_single(pdf_path))
        logger.info("Loaded %d pages from %d PDF file(s).", len(documents), len(self._paths))
        return documents

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_paths(paths: str | Path | Sequence[str | Path]) -> list[Path]:
        if isinstance(paths, (str, Path)):
            p = Path(paths)
            if p.is_dir():
                return sorted(p.glob("**/*.pdf"))
            return [p]
        return [Path(p) for p in paths]

    def _load_single(self, path: Path) -> list[Document]:
        if not path.exists():
            raise DocumentLoadError(f"File not found: {path}")
        if path.suffix.lower() != ".pdf":
            raise DocumentLoadError(f"Not a PDF file: {path}")

        logger.debug("Loading PDF: %s", path)
        documents: list[Document] = []
        try:
            with fitz.open(str(path)) as pdf:
                for page_num, page in enumerate(pdf, start=1):
                    text = page.get_text("text")  # type: ignore[arg-type]
                    if not text.strip():
                        logger.debug("Skipping blank page %d in %s", page_num, path.name)
                        continue
                    documents.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": str(path),
                                "filename": path.name,
                                "page": page_num,
                                "total_pages": len(pdf),
                            },
                        )
                    )
        except fitz.FileDataError as exc:
            raise DocumentLoadError(f"Failed to parse {path}: {exc}") from exc

        logger.debug("Extracted %d page(s) from %s", len(documents), path.name)
        return documents
