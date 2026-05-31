from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path

from agents import SQLiteSession

from src.multidal.config import settings

logger = logging.getLogger(__name__)

DEFAULT_SESSIONS_TABLE = "agent_sessions"
DEFAULT_MESSAGES_TABLE = "agent_messages"

_db_path: Path | None = None

# session_id -> {msg_index: sources}
_session_sources: dict[str, dict[int, list]] = {}
# session_id -> next expected msg index for that session
_session_next_idx: dict[str, int] = {}


def _get_db_path() -> Path:
    global _db_path
    if _db_path is None:
        dir_ = settings.project_root / "data"
        dir_.mkdir(parents=True, exist_ok=True)
        _db_path = dir_ / "sessions.db"
    return _db_path


def _ensure_schema() -> None:
    """创建 sessions 表 + agent_messages 表，并确保 session_name 列存在。"""
    db = str(_get_db_path())
    conn = sqlite3.connect(db)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {DEFAULT_SESSIONS_TABLE} (
            session_id TEXT PRIMARY KEY,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {DEFAULT_MESSAGES_TABLE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            message_data TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES {DEFAULT_SESSIONS_TABLE} (session_id)
                ON DELETE CASCADE
        )
    """)
    conn.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_{DEFAULT_MESSAGES_TABLE}_session_id
        ON {DEFAULT_MESSAGES_TABLE} (session_id, id)
    """)
    try:
        conn.execute("ALTER TABLE agent_sessions ADD COLUMN session_name TEXT DEFAULT ''")
    except sqlite3.OperationalError:
        pass  # 列已存在
    try:
        conn.execute("ALTER TABLE agent_messages ADD COLUMN sources TEXT DEFAULT '[]'")
    except sqlite3.OperationalError:
        pass  # 列已存在
    conn.commit()
    conn.close()


def get_session(session_id: str) -> SQLiteSession:
    """获取或创建持久化的 SQLiteSession。"""
    _ensure_schema()
    return SQLiteSession(session_id, db_path=str(_get_db_path()))


def set_session_name(session_id: str, name: str) -> None:
    """为会话设置一个人类可读的名称。"""
    _ensure_schema()
    db = str(_get_db_path())
    conn = sqlite3.connect(db)
    # 行可能还不存在（尚未写入消息），用 INSERT OR REPLACE 兜底
    conn.execute("""
        INSERT INTO agent_sessions (session_id, session_name, created_at, updated_at)
        VALUES (?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        ON CONFLICT(session_id) DO UPDATE SET session_name = excluded.session_name, updated_at = CURRENT_TIMESTAMP
    """, (session_id, name))
    conn.commit()
    conn.close()


def _update_last_assistant_sources(session_id: str, sources: list) -> None:
    """将最新一条 assistant 消息的 sources 写入 DB。"""
    db = str(_get_db_path())
    conn = sqlite3.connect(db)
    try:
        conn.execute("""
            UPDATE agent_messages
            SET sources = ?
            WHERE session_id = ? AND id = (
                SELECT id FROM agent_messages
                WHERE session_id = ? AND json_extract(message_data, '$.role') = 'assistant'
                ORDER BY id DESC LIMIT 1
            )
        """, (json.dumps(sources), session_id, session_id))
        conn.commit()
    finally:
        conn.close()


def set_session_sources(session_id: str, sources: list, msg_idx: int | None = None) -> None:
    if session_id not in _session_sources:
        _session_sources[session_id] = {}
    if msg_idx is not None:
        _session_sources[session_id][msg_idx] = sources
    else:
        _session_sources[session_id][-1] = sources
    _update_last_assistant_sources(session_id, sources)


def get_session_sources(session_id: str) -> dict[int, list]:
    """返回 {msg_index: sources}，按消息顺序索引。"""
    result = {}
    db = str(_get_db_path())
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT id, sources FROM agent_messages
        WHERE session_id = ? AND sources IS NOT NULL AND sources != '[]' AND sources != ''
        ORDER BY id ASC
    """, (session_id,)).fetchall()
    conn.close()
    # 按行号作为 index（与 messages 列表顺序一致）
    # 注意：query 过滤了 non-empty sources，所以 row['id'] 就是 assistant 消息的真实序号
    for row in rows:
        try:
            result[row["id"] - 1] = json.loads(row["sources"])
        except Exception:
            pass
    return result


def list_sessions() -> list[dict]:
    """列出所有会话（含名称、时间、消息数）。"""
    _ensure_schema()
    db = str(_get_db_path())
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT
            s.session_id,
            s.session_name,
            s.created_at,
            s.updated_at,
            (SELECT COUNT(*) FROM agent_messages m WHERE m.session_id = s.session_id) AS msg_count
        FROM agent_sessions s
        ORDER BY s.updated_at DESC
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


async def delete_session(session_id: str) -> None:
    """清空会话消息并从列表中移除。"""
    s = get_session(session_id)
    try:
        await s.clear_session()
    except Exception:
        pass
    db = str(_get_db_path())
    conn = sqlite3.connect(db)
    conn.execute("DELETE FROM agent_sessions WHERE session_id = ?", (session_id,))
    conn.commit()
    conn.close()


async def generate_session_name(question: str) -> str:
    """用 LLM 根据第一条问题生成简短会话名（3-10 字）。"""
    from src.multidal.agents.base import _get_chat_model
    from agents import Agent, Runner

    try:
        agent = Agent(
            name="Session Namer",
            model=_get_chat_model(),
            instructions="你是一个会话命名助手。根据用户的第一条问题，生成一个简短的会话名称（3-10个中文字符）。只返回名称本身，不要添加任何解释或标点。",
        )
        result = await Runner.run(agent, f"用户问题: {question}\n\n请给这个对话起一个简短的名字：")
        name = (result.final_output or "").strip()
        # 清理掉可能的引号和多余空白
        name = name.replace('"', '').replace('"', '').replace('"', '').replace('《', '').replace('》', '')
        return name[:20]  # 最长 20 字符
    except Exception:
        logger.warning("Failed to generate session name", exc_info=True)
        # 降级：截取问题前 15 字
        return question.strip()[:15] + ("..." if len(question) > 15 else "")
