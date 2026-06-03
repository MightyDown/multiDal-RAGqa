# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multi-modal RAG system for enterprise document intelligence. Ingests PDFs (reports, scans, financial charts, design drafts), extracts text + visual content via MinerU, builds cross-modal vector databases in Milvus, and serves mixed text-image QA via LLM.

Full architecture: see [DESIGN.md](./DESIGN.md). Evaluation plan: [EVAL.md](./EVAL.md). Interview narrative: [INTERVIEW.md](./INTERVIEW.md).

## Tech Stack

| Component | Choice |
|-----------|--------|
| Language | Python 3.11+ |
| PDF Parser | MinerU (magic-pdf, `vlm`) — cloud API at mineru.net |
| Text Embedding | BGE large zh v1.5 (1024-d, Moark API) |
| Image Embedding | Jina CLIP v2 (1024-d, Moark API) — 2 paths per image (pixel + description) |
| Image Caption Enhancer (optional) | Qwen2.5-VL-7B-Instruct (Moark API) |
| Reranker | BCE reranker base v1 (Moark sentence-similarity API) |
| Small Model (routing/rewrite/naming) | Qwen3-0.6B (Moark API, `enable_thinking=False`) |
| Vector DB | Milvus (IVF_FLAT, IP metric) |
| Message Queue | Kafka (confluent-kafka) |
| State DB | MySQL (SQLAlchemy) — `parse_tasks` / `knowledge_bases` / `agent_sessions` / `agent_messages` |
| Agent SDK | openai-agents (`Agent` + `Runner` + `function_tool`) |
| LLM (QA) | MiniMax-M2.7 (`https://api.minimaxi.com/v1`) |
| API | FastAPI + uvicorn |
| Frontend | Vue 3 + Vite (SPA, built to `frontend/dist`) |
| Deployment | Docker Compose (8 services) |

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

# Integration test (TestClient, in-process)
python main_demo.py

# Integration test against running server
python main_demo.py --live --host http://localhost:8000

# Start services (all Docker services)
docker-compose up -d

# Start API server
uvicorn src.multidal.api.app:app --reload --port 8000
# or
python main.py

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
├── config/          # Pydantic Settings (configs/settings.yaml, single file)
├── schema/          # Pydantic data models — cross-stage data contracts & JSON validation
├── pipeline/        # Stage ABC + Orchestrator (driver)
├── queue/          # Kafka producer / consumer (manual commit, exponential-backoff retry)
├── parser/          # MinerU cloud API wrapper + chart_detector
├── embedder/       # Text (BGE) + Image (Jina CLIP, dual-path) + VLCaptioner (optional)
├── store/          # MilvusStore + MultiPathRetriever + Reranker
├── db/             # MySQL models (parse_tasks, knowledge_bases, agent_sessions, agent_messages) + repository
├── agents/         # BaseAgent → QueryAgent (main LLM) / KBRouterAgent / QueryRewriterAgent (small LLM) + MySQLSession + function_tool
├── kb/             # KBManager + IntentRouter + QueryRewriter (thin aliases over agents/)
├── llm/            # Qwen3-0.6B small model wrapper
├── api/            # FastAPI app + routes (ingest, status, query, kb, doc, sessions, health)
└── utils/          # Image preprocessing, logging
```

## Data Flow

```
Upload:  POST /api/ingest → save file → MySQL task pending → Kafka parse.request
         → Consumer → MinerUParser → TextEmbedder → ImageEmbedder(±VLCaptioner)
         → MilvusStore → update MySQL parse_tasks (pending→processing→completed)

Query:   POST /api/query/stream
         → KBRouterAgent (Qwen3-0.6B, optional search/get_doc_info tools)
         → QueryRewriterAgent (Qwen3-0.6B, 2-3 sub-queries)
         → dual-path recall (BGE text + CLIP image) → merge dedup
         → BCE Reranker (text rerank score, image keeps CLIP score, unified sort)
         → context build → QueryAgent (MiniMax-M2.7) → SSE stream
         → on first message completion: Qwen3-0.6B auto-names the session
```

## Key Design Rules

- Each pipeline stage implements `Stage` ABC with `process()` and `validate()`; `Orchestrator` just chains them
- `VectorStore` is behind an abstract interface (Milvus is the only implementation now; Chroma/FAISS removed as dead code)
- Small model agents (KB routing, query rewriting, session naming) use openai-agents SDK with `enable_thinking=False`
- `configs/settings.yaml` holds all config — environment variables override it
- Images passed to LLM as **text captions** (MinerU default or Qwen2.5-VL enhanced) — no vision LLM required
- Parse failures auto-retry via Kafka re-publish with exponential backoff (30s → 60s → 90s, max 3 retries)
- Each KB = isolated Milvus Collection pair (`{kb_id}_text` + `{kb_id}_image`); physical isolation
- Two vector spaces (BGE + CLIP) → **cannot be ranked together by recall score**; BCE reranker is mandatory, with images retaining their CLIP score (text gets re-scored)
- `MySQLSession` mirrors `SQLiteSession` from openai-agents SDK; business code is storage-agnostic
- Type hints on all public functions; pytest
- Logs: `logs/worker.log` (consumer) + `logs/api.log` (FastAPI) + stdout, configured in `consumer.py:167-174` and `api/app.py:40-48`

## Important Notes

- **Only A+B dual-path recall exists** — no BM25, no "description search" path
- **Config is single file**: `configs/settings.yaml` (no default.yaml + models.yaml split)
- **State DB is MySQL** (migrated from SQLite in 2026-05): all 4 tables in one MySQL instance
- **ImageEmbedder is conditional**: only runs if `validate()` returns true (API reachable + key valid)
- **VLCaptioner is optional**: falls back to MinerU caption if API unavailable
- **API base for most models**: Moark API (`https://api.moark.com/v1`) — text embedding, image embedding, reranker, small model, VL caption
- **LLM API is separate**: MiniMax API (`https://api.minimaxi.com/v1`) for the main QA model
- **Agent tools are real**: `search_knowledge_base` and `get_doc_info` in `agents/tools.py` are production implementations using MilvusStore and MySQL repository
- **Routes** mounted at `/api/*` (ingest, status, query, query/stream, kb, doc, sessions, health) and `/raw/*` (StaticFiles for parsed images)
