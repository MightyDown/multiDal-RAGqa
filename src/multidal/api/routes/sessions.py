from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException

from src.multidal.agents.sessions import delete_session, generate_session_name, get_session, get_session_sources, list_sessions, set_session_name

router = APIRouter()


@router.get("/sessions")
async def get_sessions():
    """列出所有历史会话。"""
    sessions = list_sessions()
    return {"sessions": sessions}


@router.delete("/sessions/{session_id}")
async def remove_session(session_id: str):
    """删除指定会话。"""
    await delete_session(session_id)
    return {"ok": True}


@router.get("/sessions/{session_id}/messages")
async def get_session_messages(session_id: str):
    """获取会话中的所有消息（调试用）。"""
    session = get_session(session_id)
    items = await session.get_items()
    sources_map = get_session_sources(session_id)
    # Attach sources to each assistant message by DB row id
    for item in items:
        if item.get("role") == "assistant":
            item["sources"] = sources_map.get(item.get("_db_id"), [])
    return {"messages": items}


@router.patch("/sessions/{session_id}/messages/last")
async def patch_last_message(session_id: str, body: dict):
    """修改会话中最后一条消息的 content。"""
    session = get_session(session_id)
    items = await session.get_items()
    if not items:
        raise HTTPException(status_code=404, detail="No messages in session")

    last = items[-1]
    new_content = body.get("content")
    if new_content is None:
        raise HTTPException(status_code=400, detail="content field is required")

    # 移除最后一条，重新插入（SQLiteSession 不支持原地更新）
    await session.pop_item()
    items[-1]["content"] = new_content
    await session.add_items([last])
    return {"ok": True}


@router.patch("/sessions/{session_id}")
async def rename_session(session_id: str, body: dict):
    """自动命名会话：根据首个问答对生成会话主题。"""
    session = get_session(session_id)
    items = await session.get_items()
    if not items:
        raise HTTPException(status_code=404, detail="No messages in session")

    user_msg = next((item for item in items if item.get("role") == "user"), None)
    ai_msg = next((item for item in items if item.get("role") == "assistant"), None)

    if not user_msg:
        raise HTTPException(status_code=400, detail="No user message found")

    user_content = user_msg.get("content", "")
    # strip RAG prompt suffix if present
    user_content = user_content.split("\n请基于以上文档内容回答")[0].strip()

    ai_content = ""
    if ai_msg:
        raw = ai_msg.get("content", "")
        try:
            arr = json.loads(raw) if isinstance(raw, str) else raw
            if isinstance(arr, list) and arr and isinstance(arr[0], dict) and arr[0].get("text"):
                ai_content = "".join(item.get("text", "") for item in arr)
            else:
                ai_content = raw[:200]
        except Exception:
            ai_content = raw[:200]

    try:
        name = await generate_session_name(user_content, ai_content)
        set_session_name(session_id, name)
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to generate session name")

    return {"ok": True, "session_name": name}
