"""
api/routes.py — FastAPI router exposing the RAG system over HTTP.

Responsibility
--------------
• Define REST endpoints: /ingest, /query, /health.
• Translate HTTP requests into domain calls (ingestion pipeline, RAGPipeline).
• Map domain exceptions to appropriate HTTP status codes.
• Validate input with Pydantic request/response models.

Interactions
------------
← Receives HTTP requests from clients.
→ Calls PDFLoader / TextCleaner / Chunker / EmbeddingService / FaissVectorStore
  for ingestion.
→ Calls RAGPipeline.query() for question answering.
"""

from __future__ import annotations

import time
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File, status
from pydantic import BaseModel, Field

from app.config import get_settings
from app.embeddings.embedding_service import EmbeddingService
from app.ingestion.chunker import RecursiveTextChunker
from app.ingestion.pdf_loader import PDFLoader
from app.ingestion.text_cleaner import TextCleaner
from app.llm import get_llm
from app.rag.rag_pipeline import RAGPipeline, RAGResponse
from app.retrieval.retriever import Retriever
from app.utils import (
    get_logger,
    DocumentLoadError,
    ChunkingError,
    EmbeddingError,
    VectorStoreError,
    VectorStoreNotInitializedError,
    RetrievalError,
    RAGPipelineError,
)
from app.vectorstore.faiss_store import FaissVectorStore

logger = get_logger(__name__)
router = APIRouter()

# ---------------------------------------------------------------------------
# Shared singletons (created once at module import)
# ---------------------------------------------------------------------------

_settings = get_settings()
_embedding_service = EmbeddingService()
_vector_store = FaissVectorStore(dimension=_settings.embedding_dimension)
_retriever = Retriever(
    vector_store=_vector_store,
    embedding_service=_embedding_service,
    top_k=_settings.retrieval_top_k,
)
_rag_pipeline = RAGPipeline(retriever=_retriever, llm=get_llm())

# Attempt to load an existing index on startup
try:
    _vector_store.load(_settings.vector_store_path)
    logger.info("Vector store loaded from disk at startup.")
except VectorStoreError:
    logger.info("No existing vector store found — starting fresh.")


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)


class SourceChunkOut(BaseModel):
    text: str
    score: float
    metadata: dict


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceChunkOut]
    model: str
    latency_ms: float


class IngestResponse(BaseModel):
    filename: str
    chunks_ingested: int
    message: str


class HealthResponse(BaseModel):
    status: str
    vector_store_total: int
    llm_provider: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    """Liveness + readiness check."""
    return HealthResponse(
        status="ok",
        vector_store_total=_vector_store.total_documents,
        llm_provider=_settings.llm_provider.value,
    )


@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["ingestion"],
)
async def ingest_pdf(file: UploadFile = File(...)) -> IngestResponse:
    """
    Upload a PDF file and ingest it into the vector store.

    The file is saved temporarily, processed through the ingestion pipeline
    (load → clean → chunk → embed → store), then removed.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are accepted.",
        )

    # Save upload to data dir
    _settings.data_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = _settings.data_dir / file.filename

    try:
        content = await file.read()
        tmp_path.write_bytes(content)
        logger.info("Received upload: %s (%d bytes)", file.filename, len(content))

        # Ingestion pipeline
        documents = PDFLoader(tmp_path).load()
        cleaned = TextCleaner().clean(documents)
        chunks = RecursiveTextChunker().split(cleaned)

        if not chunks:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="No text could be extracted from the PDF.",
            )

        texts = [c.page_content for c in chunks]
        embeddings = _embedding_service.embed_documents(texts)
        _vector_store.add_documents(chunks, embeddings)
        _vector_store.save(_settings.vector_store_path)

        return IngestResponse(
            filename=file.filename,
            chunks_ingested=len(chunks),
            message=f"Successfully ingested {len(chunks)} chunks.",
        )

    except (DocumentLoadError, ChunkingError, EmbeddingError, VectorStoreError) as exc:
        logger.error("Ingestion error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


@router.post("/query", response_model=QueryResponse, tags=["rag"])
def query(request: QueryRequest) -> QueryResponse:
    """
    Answer a natural-language question using the ingested knowledge base.
    """
    start = time.perf_counter()

    # Honour per-request top_k override
    _retriever._top_k = request.top_k  # noqa: SLF001 (intentional override)

    try:
        result: RAGResponse = _rag_pipeline.query(request.question)
    except VectorStoreNotInitializedError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Knowledge base is empty. Please ingest documents first.",
        ) from exc
    except (RetrievalError, RAGPipelineError) as exc:
        logger.error("Pipeline error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    latency = round((time.perf_counter() - start) * 1000, 1)

    return QueryResponse(
        question=result.question,
        answer=result.answer,
        sources=[
            SourceChunkOut(text=s.text, score=s.score, metadata=s.metadata)
            for s in result.sources
        ],
        model=result.model,
        latency_ms=latency,
    )
