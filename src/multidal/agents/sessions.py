from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

from sqlalchemy import Column, DateTime, Integer, String, Text, ForeignKey
from sqlalchemy.dialects.mysql import LONGTEXT
from sqlalchemy import text

from src.multidal.config import settings
from src.multidal.db.models import Base, SessionLocal

logger = logging.getLogger(__name__)

DEFAULT_SESSIONS_TABLE = "agent_sessions"
DEFAULT_MESSAGES_TABLE = "agent_messages"

# session_id -> {msg_index: sources}
_session_sources: dict[str, dict[int, list]] = {}
# session_id -> next expected msg index for that session
_session_next_idx: dict[str, int] = {}


class SessionModel(Base):
    __tablename__ = DEFAULT_SESSIONS_TABLE

    session_id = Column(String(64), primary_key=True)
    session_name = Column(String(256), default="")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class MessageModel(Base):
    __tablename__ = DEFAULT_MESSAGES_TABLE

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), ForeignKey("agent_sessions.session_id", ondelete="CASCADE"), nullable=False)
    message_data = Column(LONGTEXT, nullable=False)
    sources = Column(Text, default="[]")
    created_at = Column(DateTime, default=datetime.utcnow)


def _ensure_schema() -> None:
    from src.multidal.db.models import _engine
    Base.metadata.create_all(_engine)


class MySQLSession:
    """MySQL-backed session for openai-agents SDK.

    Implements the same interface as the SDK's SQLiteSession so that
    QueryAgent can use it transparently for get_items / add_items.
    """

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.db_path = str(settings.project_root / "data" / "sessions.db")
        _ensure_schema()

    async def get_items(self, limit: int | None = None) -> list[dict]:
        with SessionLocal() as db:
            q = db.query(MessageModel).filter(MessageModel.session_id == self.session_id).order_by(MessageModel.id.asc())
            if limit is not None:
                q = db.query(MessageModel).filter(MessageModel.session_id == self.session_id).order_by(MessageModel.id.desc()).limit(limit)
                rows = list(reversed(q.all()))
            else:
                rows = q.all()
            items = []
            for row in rows:
                try:
                    item = json.loads(row.message_data)
                    item["_db_id"] = row.id
                    items.append(item)
                except Exception:
                    pass
            return items

    async def add_items(self, items: list[dict]) -> None:
        if not items:
            return
        with SessionLocal() as db:
            # Ensure session row exists (upsert)
            sess = db.get(SessionModel, self.session_id)
            if not sess:
                sess = SessionModel(session_id=self.session_id)
                db.add(sess)
                db.commit()
                db.flush()
            for item in items:
                msg = MessageModel(session_id=self.session_id, message_data=json.dumps(item))
                db.add(msg)
            db.commit()

    async def pop_item(self) -> dict | None:
        with SessionLocal() as db:
            row = db.query(MessageModel).filter(MessageModel.session_id == self.session_id).order_by(MessageModel.id.desc()).first()
            if not row:
                return None
            try:
                item = json.loads(row.message_data)
            except Exception:
                item = None
            db.delete(row)
            db.commit()
            return item

    async def clear_session(self) -> None:
        with SessionLocal() as db:
            db.query(MessageModel).filter(MessageModel.session_id == self.session_id).delete()
            db.query(SessionModel).filter(SessionModel.session_id == self.session_id).delete()
            db.commit()

    def close(self) -> None:
        pass


def get_session(session_id: str) -> MySQLSession:
    return MySQLSession(session_id)


def set_session_name(session_id: str, name: str) -> None:
    _ensure_schema()
    with SessionLocal() as db:
        sess = db.get(SessionModel, session_id)
        if sess:
            sess.session_name = name
            sess.updated_at = datetime.utcnow()
        else:
            sess = SessionModel(session_id=session_id, session_name=name)
            db.add(sess)
        db.commit()


def _update_last_assistant_sources(session_id: str, sources: list) -> None:
    with SessionLocal() as db:
        row = db.query(MessageModel).filter(
            MessageModel.session_id == session_id,
            text(f"JSON_EXTRACT(message_data, '$.role') = 'assistant'")
        ).order_by(MessageModel.id.desc()).first()
        if row:
            row.sources = json.dumps(sources)
            db.commit()


def set_session_sources(session_id: str, sources: list, msg_idx: int | None = None) -> None:
    if session_id not in _session_sources:
        _session_sources[session_id] = {}
    if msg_idx is not None:
        _session_sources[session_id][msg_idx] = sources
    else:
        _session_sources[session_id][-1] = sources
    _update_last_assistant_sources(session_id, sources)


def get_session_sources(session_id: str) -> dict[int, list]:
    """Returns {db_row_id: sources} mapping for messages with non-empty sources."""
    result = {}
    with SessionLocal() as db:
        rows = db.execute(
            text(f"""SELECT id, sources FROM {DEFAULT_MESSAGES_TABLE}
                WHERE session_id = :sid AND sources IS NOT NULL AND sources != '[]' AND sources != ''
                ORDER BY id ASC"""),
            {"sid": session_id}
        ).fetchall()
    for row in rows:
        try:
            result[row[0]] = json.loads(row[1])
        except Exception:
            pass
    return result


def list_sessions() -> list[dict]:
    _ensure_schema()
    with SessionLocal() as db:
        rows = db.execute(
            text(f"""SELECT s.session_id, s.session_name, s.created_at, s.updated_at,
                       (SELECT COUNT(*) FROM {DEFAULT_MESSAGES_TABLE} m WHERE m.session_id = s.session_id) AS msg_count
                FROM {DEFAULT_SESSIONS_TABLE} s
                ORDER BY s.updated_at DESC""")
        ).fetchall()
    return [dict(r._mapping) for r in rows]


async def delete_session(session_id: str) -> None:
    s = get_session(session_id)
    try:
        await s.clear_session()
    except Exception:
        pass
    with SessionLocal() as db:
        sess = db.get(SessionModel, session_id)
        if sess:
            db.delete(sess)
            db.commit()


async def generate_session_name(question: str, answer: str = "") -> str:
    from src.multidal.agents.base import _get_small_agent
    from agents import Runner

    content = f"用户问题: {question}"
    if answer:
        content += f"\nAI回答: {answer[:200]}"
    content += "\n\n请给这个对话起一个简短的名字（3-10个中文字符），只返回名称："

    try:
        agent = _get_small_agent(
            name="Session Namer",
            instructions="你是一个会话命名助手。根据对话内容，生成一个简短的会话名称（3-10个中文字符）。只返回名称本身，不要添加任何解释或标点。",
        )
        result = await Runner.run(agent, content)
        name = (result.final_output or "").strip()
        # 移除思考标签
        name = name.replace('<think>', '').replace('', '')
        name = name.replace('"', '').replace('"', '').replace('"', '').replace('《', '').replace('》', '')
        return name[:20].strip()
    except Exception:
        logger.warning("Failed to generate session name", exc_info=True)
        return question.strip()[:15] + ("..." if len(question) > 15 else "")