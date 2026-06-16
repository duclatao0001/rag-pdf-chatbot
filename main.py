"""
main.py — Application entrypoint.

Responsibility
--------------
• Create the FastAPI application.
• Register routers and global exception handlers.
• Start Uvicorn when executed directly.

Usage
-----
  # Development
  python main.py

  # Production (via gunicorn with uvicorn workers)
  gunicorn main:app -k uvicorn.workers.UvicornWorker -w 4 --bind 0.0.0.0:8000
"""

from __future__ import annotations

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.config import get_settings
from app.utils import get_logger, RAGBaseError

logger = get_logger(__name__)
settings = get_settings()

# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title=settings.app_name,
    description="Production-ready Retrieval-Augmented Generation system.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---------------------------------------------------------------------------
# CORS (adjust origins for your deployment)
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global exception handler — catches any un-caught RAGBaseError
# ---------------------------------------------------------------------------

@app.exception_handler(RAGBaseError)
async def rag_exception_handler(request: Request, exc: RAGBaseError) -> JSONResponse:
    logger.error("Unhandled RAG error on %s %s: %s", request.method, request.url, exc)
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "type": type(exc).__name__},
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

app.include_router(router, prefix="/api/v1")

# ---------------------------------------------------------------------------
# Dev server
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        log_level=settings.log_level.lower(),
    )
