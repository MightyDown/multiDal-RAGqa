from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, HTTPException

from src.multidal.db.repository import delete_task, get_task, list_tasks
from src.multidal.store.milvus_store import MilvusStore

router = APIRouter()


@router.get("/kb/{kb_id}/docs")
async def list_kb_docs(kb_id: str):
    """列出某知识库下的所有文档（仅已完成的）。"""
    tasks = list_tasks(kb_id=kb_id)
    return {
        "docs": [
            {
                "task_id": t.task_id,
                "filename": t.filename,
                "status": t.status,
                "page_count": t.page_count,
                "created_at": t.created_at.isoformat() if t.created_at else "",
            }
            for t in tasks
        ],
        "total": len(tasks),
    }


@router.get("/docs/{task_id}")
async def get_doc_content(task_id: str):
    """获取文档完整内容（markdown）+ chunk 列表。"""
    task = get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    # 从 Milvus 查 chunks（用于检索，但页面展示用 full_text）
    store = MilvusStore()
    try:
        rows = store.query_by_doc(task.kb_id, task_id)
    except Exception as e:
        rows = []

    return {
        "task_id": task_id,
        "filename": task.filename,
        "kb_id": task.kb_id,
        "page_count": task.page_count,
        "full_text": task.full_text or "",
        "chunks": [
            {
                "chunk_id": r.get("chunk_id", ""),
                "content": r.get("content", ""),
                "modality": r.get("modality", "text"),
                "page": r.get("page", 1),
            }
            for r in rows
        ],
        "chunk_count": len(rows),
    }


@router.delete("/docs/{task_id}")
async def delete_doc(task_id: str):
    """删除文档：SQLite 记录 + Milvus 向量 + 本地文件。"""
    task = get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    # 1. 从 Milvus 删除向量
    try:
        store = MilvusStore()
        store.delete_by_doc(task.kb_id, task_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Milvus delete failed: {e}")

    # 2. 删除本地 PDF 文件
    file_path = task.file_path
    if file_path and os.path.exists(file_path):
        os.remove(file_path)

    # 3. 删除 SQLite 记录
    delete_task(task_id)

    return {"task_id": task_id, "deleted": True}
