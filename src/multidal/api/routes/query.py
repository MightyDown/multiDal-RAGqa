from __future__ import annotations

import json
import logging

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from src.multidal.agents.query_agent import QueryAgent, VLMAgent, _has_images
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
    candidates, ranked, context, _sources_meta = await _retrieve(req)

    if req.session_id:
        from src.multidal.agents.sessions import _session_next_idx
        idx = _session_next_idx.get(req.session_id, 0)
        set_session_sources(req.session_id, _sources_meta, idx)
        _session_next_idx[req.session_id] = idx + 2  # user + assistant

    session = get_session(req.session_id) if req.session_id else None

    # 实验用 text_only 模式：强制走纯文本召回 + 纯文本 LLM 路径
    if req.text_only:
        logger.info("text_only mode: filtering image sources, forcing text LLM path")
        ranked = [r for r in ranked if r.modality != "image"]
        if req.session_id:
            set_session_sources(req.session_id, [s for s in _sources_meta if s.get("modality") != "image"])

    # 判断是否需要走 VLM 多模态路径
    if _has_images(ranked):
        logger.info("Using VLM path (images detected in %d ranked candidates)", len(ranked))
        vlm = VLMAgent()
        try:
            answer = vlm.generate(req.question, ranked)
        except Exception:
            # VLM 模型在某些平台可能已下线（如 GLM-4.6V-Flash 在 Moark 已停用），
            # 失败时降级为文本 LLM：去掉 image 模态的候选，仅用文本上下文生成答案。
            logger.exception("VLM path failed, falling back to text LLM (dropping image sources)")
            text_ranked = [r for r in ranked if r.modality != "image"]
            text_context_lines = []
            for r in text_ranked:
                tag = "[文本]"
                text_context_lines.append(f"{tag} | {r.kb_id} | p{r.page} | score={r.score:.3f}\n{r.content}")
            text_context = "\n\n---\n\n".join(text_context_lines)
            agent = QueryAgent()
            answer = await agent.run(req.question, text_context or "(no text context)", session=session)
    else:
        agent = QueryAgent()
        answer = await agent.run(req.question, context, session=session)

    return QueryResponse(answer=answer, sources=_sources_meta)


async def _retrieve(req: QueryRequest) -> tuple[list, list, str, list[dict]]:
    """返回 (candidates, ranked, context, sources)。"""
    # 未选择 KB 且禁用检索时，直接跳过 RAG
    if not req.retrieval:
        return [], [], "", []

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
            # text_only 模式：跳过 image collection 检索，整路 CLIP search 都不发起
            image_vec = _image_embedder.embed_query(q) if not req.text_only else None
            if req.text_only:
                # 直接调底层 store 只搜 text collection,模拟"无图"路径
                batch: list = []
                for kb in kb_ids:
                    try:
                        batch.extend(_retriever._store.search(f"{kb}_text", text_vec, top_k=settings.top_k_recall))
                    except Exception:
                        logger.warning("Text-only search failed for %s_text", kb)
                # 按 kb:chunk 去重 + 排序
                dedup: dict = {}
                for r in batch:
                    k = f"{r.kb_id}:{r.chunk_id}"
                    if k not in dedup or r.score > dedup[k].score:
                        dedup[k] = r
                batch = sorted(dedup.values(), key=lambda x: x.score, reverse=True)
            else:
                batch = _retriever.recall(text_vec, image_vec, kb_ids, top_k=settings.top_k_recall)
            for c in batch:
                key = f"{c.kb_id}:{c.chunk_id}"
                if key not in seen:
                    seen.add(key)
                    candidates.append(c)
        except Exception:
            logger.exception("Recall failed for query: %s", q)

    if not candidates:
        return [], [], "（知识库中未检索到相关内容）", []

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
    return candidates, ranked, context, sources


@router.post("/query/stream")
async def query_stream(req: QueryRequest):
    """流式问答：先发 sources，再流式输出 LLM 回答。

    当检索结果包含图片时，自动切换为 VLM (GLM-4.6V-Flash) 多模态路径。
    """
    candidates, ranked, context, sources = await _retrieve(req)

    _need_name = bool(req.session_id)
    _use_vlm = _has_images(ranked)

    logger.info(
        "query_stream: ranked=%d, use_vlm=%s, modalities=%s",
        len(ranked), _use_vlm,
        [r.modality for r in ranked],
    )

    if _use_vlm:
        logger.info("Using VLM stream path (images detected in %d ranked candidates)", len(ranked))

    async def event_stream():
        yield f"data: {json.dumps({'type': 'sources', 'sources': sources}, ensure_ascii=False)}\n\n"

        fullAnswer = ""

        if _use_vlm:
            # VLM 流式路径
            vlm = VLMAgent()
            try:
                for delta in vlm.generate_stream(req.question, ranked):
                    fullAnswer += delta
                    yield f"data: {json.dumps({'type': 'delta', 'content': delta}, ensure_ascii=False)}\n\n"
            except Exception as e:
                logger.exception("VLM stream failed")
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"
        else:
            # 纯文本 LLM 路径（原逻辑）
            agent = QueryAgent()
            session = get_session(req.session_id) if req.session_id else None
            try:
                async for delta in agent.run_streamed(req.question, context, session=session):
                    fullAnswer += delta
                    yield f"data: {json.dumps({'type': 'delta', 'content': delta}, ensure_ascii=False)}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"

        if req.session_id:
            set_session_sources(req.session_id, sources)

        yield "data: {\"type\":\"done\"}\n\n"

        if req.session_id and _need_name and fullAnswer:
            try:
                name = await generate_session_name(req.question, fullAnswer)
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
