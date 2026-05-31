# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multi-modal RAG system for enterprise document intelligence. Ingests PDFs (reports, scans, financial charts, design drafts), extracts text + visual content via MinerU, builds cross-modal vector databases in Milvus, and serves mixed text-image QA via LLM.

Full architecture: see [DESIGN.md](./DESIGN.md).

## Tech Stack

| Component | Choice |
|-----------|--------|
| Language | Python 3.11+ |
| PDF Parser | MinerU (magic-pdf) — cloud API at mineru.net |
| Text Embedding | BGE large zh v1.5 (1024-d, Moark API) |
| Image Embedding | Jina CLIP v2 (1024-d, Moark API) |
| Reranker | BCE reranker base v1 (Moark sentence-similarity API) |
| Small Model (routing/rewrite) | Qwen3-0.6B (Moark API, disable thinking) |
| Vector DB | Milvus |
| Message Queue | Kafka (confluent-kafka) |
| State DB | MySQL (SQLAlchemy) — parse_tasks, knowledge_bases, agent_sessions, agent_messages |
| Session DB | MySQL (same DB, separate tables) |
| Agent SDK | openai-agents |
| LLM (QA) | MiniMax-M2.7 (`https://api.minimaxi.com/v1`) |
| API | FastAPI + uvicorn |
| Deployment | Docker Compose |

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest

# Run single test file
pytest tests/test_parser/test_mineru.py

# Run with coverage
pytest --cov=src/multidal --cov-report=term-missing

# Start services (all Docker services)
docker-compose up -d

# Start API server
uvicorn src.multidal.api.app:app --reload --port 8000

# Start worker (Kafka consumer)
python -m src.multidal.queue.consumer

# CLI: ingest a PDF
python scripts/ingest.py --file /path/to/doc.pdf --kb kb_finance

# CLI: query
python scripts/query.py --question "Q1营收增长了多少？" --kb kb_finance
```

## Package Layout

```
src/multidal/
├── config/          # Pydantic Settings (configs/settings.yaml)
├── schema/          # Pydantic data models — cross-stage data contracts & JSON validation
├── pipeline/        # Stage ABC + Orchestrator
├── queue/          # Kafka producer / consumer
├── parser/          # MinerU cloud API wrapper + chart detector
├── embedder/       # Text (BGE) + Image (Jina CLIP) embedders + VLCaptioner + ModelRegistry
├── store/          # MilvusStore + MultiPathRetriever + Reranker
├── db/             # MySQL models (parse_tasks, knowledge_bases) + repository
├── agents/         # BaseAgent → QueryAgent / IngestAgent + router_agent / rewriter_agent / sessions / tools
├── kb/             # KBManager + IntentRouter (alias) + QueryRewriter (alias)
├── llm/            # Qwen3-0.6B small model wrapper via Moark API
├── api/            # FastAPI app + routes (ingest, status, query, kb, doc, sessions, health)
└── utils/          # Image preprocessing, logging
```

## Data Flow

```
Upload:  POST /api/ingest → save file → MySQL task pending → Kafka parse.request → Consumer
         → MinerUParser → TextEmbedder → ImageEmbedder(?) → MilvusStore
         → update MySQL parse_tasks (pending→processing→completed)

Query:   POST /api/query/stream → IntentRouter (Qwen3-0.6B) → QueryRewriter (Qwen3-0.6B)
         → dual-path recall (BGE text + CLIP image) → merge dedup → BCE-reranker
         → context build → QueryAgent (MiniMax-M2.7) → SSE stream
```

## Key Design Rules

- Each pipeline stage implements `Stage` ABC with `process()` and `validate()`
- VectorStore is behind an abstract interface (Milvus primary; Chroma/FAISS for dev)
- Small model agents (KB routing, query rewriting, session naming) use openai-agents SDK with `enable_thinking=False`
- `configs/settings.yaml` holds all config — environment variables override it
- Images passed to LLM as MinerU-generated text captions (no vision LLM required)
- Parse failures auto-retry via Kafka re-publish with exponential backoff (30s → 60s → 90s)
- Each KB = isolated Milvus Collection pair (`{kb_id}_text` + `{kb_id}_image`)
- Type hints on all public functions; pytest

## Important Notes

- **Only A+B dual-path recall exists** — no BM25, no "description search"
- **Config is single file**: `configs/settings.yaml` (not default.yaml + models.yaml)
- **State DB is MySQL**: all tables (parse_tasks, knowledge_bases, agent_sessions, agent_messages) in one MySQL instance
- **ImageEmbedder is optional**: inserted only when `validate()` returns true (CUDA + Jina CLIP available)
- **API base for most models**: Moark API (`https://api.moark.com/v1`) — text embedding, image embedding, reranker, small model all use Moark
- **LLM API is separate**: MiniMax API (`https://api.minimaxi.com/v1`) for the main QA model
- **Agent tools are real**: `search_knowledge_base` and `get_doc_info` in `agents/tools.py` are production implementations using MilvusStore and MySQL repository