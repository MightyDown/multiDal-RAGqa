from __future__ import annotations

import json
import logging

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from src.multidal.agents.query_agent import QueryAgent
from src.multidal.agents.sessions import (
    generate_session_name,
    get_session,
    list_sessions,
    set_session_name,
    set_session_sources,
    get_session_sources,
)
from src.multidal.api.schemas import QueryRequest, QueryResponse
from src.multidal.config import settings
from src.multidal.embedder.text_embedder import TextEmbedder
from src.multidal.embedder.image_embedder import ImageEmbedder
from src.multidal.kb.manager import KBManager
from src.multidal.kb.rewriter import QueryRewriter
from src.multidal.kb.router import IntentRouter
from src.multidal.store.milvus_store import MilvusStore
from src.multidal.store.reranker import Reranker
from src.multidal.store.retriever import MultiPathRetriever

logger = logging.getLogger(__name__)

router = APIRouter()

# 全局复用，启动时自动连接 Milvus
_store = MilvusStore()
_retriever = MultiPathRetriever(_store)
_reranker = Reranker()
_text_embedder = TextEmbedder()
_image_embedder = ImageEmbedder()


@router.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest) -> QueryResponse:
    _, context, _sources_meta = await _retrieve(req)

    if req.session_id:
        from src.multidal.agents.sessions import _session_next_idx
        idx = _session_next_idx.get(req.session_id, 0)
        set_session_sources(req.session_id, _sources_meta, idx)
        _session_next_idx[req.session_id] = idx + 2  # user + assistant

    agent = QueryAgent()
    session = get_session(req.session_id) if req.session_id else None
    answer = await agent.run(req.question, context, session=session)

    return QueryResponse(answer=answer, sources=_sources_meta)


async def _retrieve(req: QueryRequest) -> tuple[list, str, list[dict]]:
    """返回 (candidates, context, sources)。"""
    # 未选择 KB 且禁用检索时，直接跳过 RAG
    if not req.retrieval:
        return [], "", []

    kb_mgr = KBManager()
    router_ = IntentRouter(kb_mgr)
    # 优先用用户选择的 KB，否则意图识别路由
    kb_ids = req.kb_ids if req.kb_ids else await router_.route(req.question)

    queries = [req.question]
    if req.rewrite_query:
        rewriter = QueryRewriter()
        queries = await rewriter.rewrite(req.question)

    candidates: list = []
    seen = set()
    for q in queries[:3]:
        try:
            text_vec = _text_embedder.embed_query(q)     # bge → text collection
            image_vec = _image_embedder.embed_query(q)   # CLIP → image collection
            batch = _retriever.recall(text_vec, image_vec, kb_ids, top_k=settings.top_k_recall)
            for c in batch:
                key = f"{c.kb_id}:{c.chunk_id}"
                if key not in seen:
                    seen.add(key)
                    candidates.append(c)
        except Exception:
            logger.exception("Recall failed for query: %s", q)

    if not candidates:
        return [], "（知识库中未检索到相关内容）", []

    ranked = _reranker.rerank(req.question, candidates, top_k=settings.top_k_final)

    lines = []
    for r in ranked:
        modality_tag = "[图片]" if r.modality == "image" else "[文本]"
        lines.append(
            f"{modality_tag} | {r.kb_id} | p{r.page} | score={r.score:.3f}\n{r.content}"
        )
    context = "\n\n---\n\n".join(lines)

    sources = [
        {
            "chunk_id": r.chunk_id,
            "kb_id": r.kb_id,
            "doc_id": r.doc_id,
            "page": r.page,
            "score": r.score,
            "modality": r.modality,
            "image_path": r.image_path,
            "content": r.content[:500],
        }
        for r in ranked
    ]
    return candidates, context, sources


@router.post("/query/stream")
async def query_stream(req: QueryRequest):
    """流式问答：先发 sources，再流式输出 LLM 回答。"""
    _, context, sources = await _retrieve(req)

    # 判断是否需要自动命名（首个问题）
    _need_name = bool(req.session_id)

    async def event_stream():
        yield f"data: {json.dumps({'type': 'sources', 'sources': sources}, ensure_ascii=False)}\n\n"

        agent = QueryAgent()
        session = get_session(req.session_id) if req.session_id else None
        try:
            async for delta in agent.run_streamed(req.question, context, session=session):
                yield f"data: {json.dumps({'type': 'delta', 'content': delta}, ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"

        if req.session_id:
            set_session_sources(req.session_id, sources)

        yield "data: {\"type\":\"done\"}\n\n"

        # 自动命名：首个问题完成后用 LLM 生成会话名
        if req.session_id and _need_name:
            try:
                sessions = list_sessions()
                current = next((s for s in sessions if s["session_id"] == req.session_id), None)
                if current and not (current.get("session_name") or "").strip():
                    name = await generate_session_name(req.question)
                    if name:
                        set_session_name(req.session_id, name)
            except Exception:
                pass

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
