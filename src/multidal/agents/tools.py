from __future__ import annotations

from agents import function_tool


@function_tool
def search_knowledge_base(query: str, kb_ids: list[str], top_k: int = 5) -> dict:
    """跨模态检索知识库。
    Args:
        query: 搜索查询词
        kb_ids: 目标知识库 ID 列表
        top_k: 每个 KB 返回的结果数
    """
    return {"query": query, "kb_ids": kb_ids, "results": []}


@function_tool
def get_doc_info(doc_id: str) -> dict:
    """获取指定文档的元信息（文件名、页数、入库时间等）。"""
    return {"doc_id": doc_id, "filename": "", "page_count": 0}
