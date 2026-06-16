# Production RAG System

A clean-architecture, production-ready Retrieval-Augmented Generation (RAG) system in Python. Every responsibility is isolated, every dependency is injected, and every LLM provider is swappable without touching pipeline code.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          FastAPI  (api/routes.py)               │
│              POST /ingest          POST /query                   │
└───────────────┬─────────────────────────┬───────────────────────┘
                │                         │
   ┌────────────▼───────────┐    ┌────────▼────────────┐
   │   Ingestion Pipeline   │    │    RAG Pipeline      │
   │  PDFLoader             │    │  RAGPipeline         │
   │  TextCleaner           │    │  (rag_pipeline.py)   │
   │  RecursiveTextChunker  │    └────────┬────────────┘
   └────────────┬───────────┘             │
                │                ┌────────▼────────────┐
   ┌────────────▼───────────┐    │      Retriever       │
   │   EmbeddingService     │◄───┤  (retriever.py)      │
   │  (embed_documents)     │    └────────┬────────────┘
   └────────────┬───────────┘             │
                │                         │
   ┌────────────▼─────────────────────────▼────────────┐
   │              FaissVectorStore                      │
   │   add_documents  search  save  load                │
   └────────────────────────────────────────────────────┘
                                          │
                               ┌──────────▼──────────┐
                               │       BaseLLM        │
                               │  ┌──────────────┐   │
                               │  │  OllamaLLM   │   │
                               │  │  OpenAILLM   │   │
                               │  └──────────────┘   │
                               └─────────────────────┘
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env — set LLM_PROVIDER, API keys, etc.
```

### 3. Start the API server

```bash
python main.py
# or for production:
gunicorn main:app -k uvicorn.workers.UvicornWorker -w 4 --bind 0.0.0.0:8000
```

### 4. Ingest a PDF

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -F "file=@/path/to/your/document.pdf"
```

### 5. Query

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic of the document?", "top_k": 5}'
```

---

## Switching LLM Providers

Edit `.env` — **no code changes required**:

```bash
# Use local Ollama (default)
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3

# Switch to OpenAI
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
```

---

## Project Structure

```
project_root/
│
├── app/
│   ├── config/
│   │   └── settings.py          # All config via env vars / .env
│   │
│   ├── ingestion/
│   │   ├── pdf_loader.py        # PDF → raw Document objects
│   │   ├── text_cleaner.py      # Normalise extracted text
│   │   └── chunker.py           # Sliding-window chunking
│   │
│   ├── embeddings/
│   │   └── embedding_service.py # sentence-transformers or OpenAI
│   │
│   ├── vectorstore/
│   │   └── faiss_store.py       # ALL FAISS code lives here only
│   │
│   ├── retrieval/
│   │   └── retriever.py         # embed query → vector search → top-k
│   │
│   ├── llm/
│   │   ├── base_llm.py          # Abstract interface
│   │   ├── ollama_llm.py        # Ollama implementation
│   │   └── openai_llm.py        # OpenAI implementation
│   │
│   ├── rag/
│   │   └── rag_pipeline.py      # Orchestrate retrieve → prompt → generate
│   │
│   ├── api/
│   │   └── routes.py            # FastAPI endpoints
│   │
│   └── utils/
│       ├── logger.py            # Centralised logging
│       └── exceptions.py        # Domain exceptions
│
├── data/                        # Uploaded PDFs + FAISS index
├── tests/
│   └── test_rag_system.py       # Unit tests (no LLM/PDF required)
├── main.py                      # FastAPI app + Uvicorn entrypoint
├── requirements.txt
├── .env.example
└── README.md
```

---

## Running Tests

```bash
pytest tests/ -v
```

All tests are fully isolated — no external services, no PDF files, no API keys required.

---

## Design Principles

| Principle | How it's applied |
|---|---|
| **Single Responsibility** | Each class has one job; no mixing of concerns |
| **Open/Closed** | Add a new LLM by implementing `BaseLLM`; nothing else changes |
| **Dependency Inversion** | `RAGPipeline` depends on `BaseLLM`, never on `OllamaLLM` directly |
| **Config-Driven** | All tunables live in `settings.py`; zero hard-coding |
| **Strong Typing** | Pydantic models + type annotations throughout |
| **Error Handling** | Domain exceptions map cleanly to HTTP status codes |
| **Observability** | Structured logging at every layer via `get_logger(__name__)` |

---

## Adding a New LLM Provider

1. Create `app/llm/my_llm.py` implementing `BaseLLM`.
2. Add the new provider to `LLMProvider` enum in `settings.py`.
3. Add a branch in `app/llm/__init__.py::get_llm()`.
4. Set `LLM_PROVIDER=my_provider` in `.env`.

No other file needs to change.

---

## Adding a New Vector Store

1. Create `app/vectorstore/my_store.py` implementing the same four methods: `add_documents`, `search`, `save`, `load`.
2. Wire it into `api/routes.py` alongside `FaissVectorStore`.

The `Retriever` and `RAGPipeline` require zero changes.
