from __future__ import annotations

from fastapi import APIRouter, HTTPException

from src.multidal.agents.sessions import delete_session, get_session, get_session_sources, list_sessions, set_session_sources

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
    # Attach sources to each assistant message by index
    for i, item in enumerate(items):
        if item.get("role") == "assistant":
            item["sources"] = sources_map.get(i, [])
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
