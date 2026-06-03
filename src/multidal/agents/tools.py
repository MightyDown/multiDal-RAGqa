"""智能体可调用的 function_tool 工具集。

本模块实现两个生产级 function_tool:
    - ``search_knowledge_base``: 跨 KB 文本检索(返回最相关 chunk 摘要)。
    - ``get_doc_info``: 按 doc_id 查询文档元信息(文件名、页数、状态等)。

两者均使用 ``@function_tool`` 装饰器(openai-agents SDK 风格),可挂载到
任意 Agent(如 KBRouterAgent)上,模型会自动决定何时调用。
"""

from __future__ import annotations

import logging

from agents import function_tool
from src.multidal.config import settings
from src.multidal.db.repository import get_doc
from src.multidal.embedder.text_embedder import TextEmbedder
from src.multidal.store.milvus_store import MilvusStore

logger = logging.getLogger(__name__)

# 模块级单例:Embedder 与 Store 内部都做了惰性连接,这里直接复用避免重复构造。
_text_embedder = TextEmbedder()
_milvus_store = MilvusStore()


@function_tool
def search_knowledge_base(query: str, kb_ids: list[str], top_k: int = 5) -> dict:
    """跨模态检索知识库,返回最相关的文本 chunks。

    Args:
        query: 搜索查询词(中文问题或关键词)。
        kb_ids: 目标知识库 ID 列表,例如 ``["kb_finance", "kb_tech"]``。
        top_k: 每个 KB 返回的结果数,默认 5。

    Returns:
        dict: 形如 ``{"query": ..., "kb_ids": [...], "results": [...], "error"?: str}``。
              ``results`` 中每条含 ``kb_id`` / ``chunk_id`` / ``doc_id`` / ``page`` /
              ``score`` / ``content``(前 300 字)。
    """
    try:
        # 1) 把 query 嵌入为 BGE 向量
        query_vec = _text_embedder.embed_query(query)
    except Exception as e:
        logger.warning("Failed to embed query: %s", e)
        return {"query": query, "kb_ids": kb_ids, "results": [], "error": str(e)}

    all_results = []
    # 用 set 去重,key 为 kb_id:chunk_id(理论上 KB 间不会重复,KB 内有概率重复)
    seen = set()

    for kb_id in kb_ids:
        try:
            coll_name = f"{kb_id}_text"
            hits = _milvus_store.search(coll_name, query_vec, top_k=top_k)
            for hit in hits:
                key = f"{hit.kb_id}:{hit.chunk_id}"
                if key not in seen:
                    seen.add(key)
                    all_results.append({
                        "kb_id": hit.kb_id,
                        "chunk_id": hit.chunk_id,
                        "doc_id": hit.doc_id,
                        "page": hit.page,
                        "score": round(hit.score, 4),
                        # 截断到 300 字,避免 prompt 爆炸
                        "content": hit.content[:300],
                    })
        except Exception as e:
            # 单 KB 失败不中断其他 KB
            logger.warning("Search failed for kb=%s: %s", kb_id, e)
            continue

    # 全局按 score 降序,数量限制为 top_k × KB 数(避免极端情况)
    all_results.sort(key=lambda x: x["score"], reverse=True)
    return {"query": query, "kb_ids": kb_ids, "results": all_results[:top_k * len(kb_ids)]}


@function_tool
def get_doc_info(doc_id: str) -> dict:
    """获取指定文档的元信息(文件名、页数、入库时间等)。

    Args:
        doc_id: 文档 ID(对应 MySQL parse_tasks.task_id)。

    Returns:
        dict: 形如 ``{"doc_id": ..., "found": bool, "filename"?: str, "file_path"?: str,
              "page_count"?: int, "kb_id"?: str, "status"?: str, "created_at"?: str,
              "error"?: str}``。未找到时 ``found=False``。
    """
    try:
        doc = get_doc(doc_id)
        if doc is None:
            return {"doc_id": doc_id, "found": False, "error": "文档不存在"}
        return {
            "doc_id": doc_id,
            "found": True,
            "filename": doc.filename,
            "file_path": doc.file_path,
            "page_count": doc.page_count,
            "kb_id": doc.kb_id,
            "status": doc.status,
            "created_at": str(doc.created_at),
        }
    except Exception as e:
        logger.warning("get_doc_info failed for doc_id=%s: %s", doc_id, e)
        return {"doc_id": doc_id, "found": False, "error": str(e)}
