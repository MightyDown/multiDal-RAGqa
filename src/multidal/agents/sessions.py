from __future__ import annotations

import json
import logging
from datetime import datetime

from sqlalchemy import Column, DateTime, Integer, String, Text, ForeignKey
from sqlalchemy.orm import relationship, sessionmaker

from agents import SQLiteSession

from src.multidal.config import settings
from src.multidal.db.models import Base

logger = logging.getLogger(__name__)

DEFAULT_SESSIONS_TABLE = "agent_sessions"
DEFAULT_MESSAGES_TABLE = "agent_messages"

_db_path: str | None = None

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

    messages = relationship("MessageModel", back_populates="session", cascade="all, delete-orphan")


class MessageModel(Base):
    __tablename__ = DEFAULT_MESSAGES_TABLE

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), ForeignKey("agent_sessions.session_id", ondelete="CASCADE"), nullable=False)
    message_data = Column(Text, nullable=False)
    sources = Column(Text, default="[]")
    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("SessionModel", back_populates="messages")


def _get_session_local():
    from src.multidal.db.models import SessionLocal
    return SessionLocal()


def _ensure_schema() -> None:
    from src.multidal.db.models import _engine
    Base.metadata.create_all(_engine)


def get_session(session_id: str) -> SQLiteSession:
    _ensure_schema()
    return SQLiteSession(session_id, db_path=str(settings.project_root / "data" / "sessions.db"))


def set_session_name(session_id: str, name: str) -> None:
    _ensure_schema()
    Session = _get_session_local()
    with Session() as session:
        sess = session.get(SessionModel, session_id)
        if sess:
            sess.session_name = name
            sess.updated_at = datetime.utcnow()
        else:
            sess = SessionModel(session_id=session_id, session_name=name)
            session.add(sess)
        session.commit()


def _update_last_assistant_sources(session_id: str, sources: list) -> None:
    Session = _get_session_local()
    with Session() as session:
        result = session.execute(
            f"""SELECT id FROM {DEFAULT_MESSAGES_TABLE}
                WHERE session_id = :sid AND JSON_EXTRACT(message_data, '$.role') = 'assistant'
                ORDER BY id DESC LIMIT 1""",
            {"sid": session_id}
        )
        row = result.fetchone()
        if row:
            msg = session.get(MessageModel, row[0])
            if msg:
                msg.sources = json.dumps(sources)
                session.commit()


def set_session_sources(session_id: str, sources: list, msg_idx: int | None = None) -> None:
    if session_id not in _session_sources:
        _session_sources[session_id] = {}
    if msg_idx is not None:
        _session_sources[session_id][msg_idx] = sources
    else:
        _session_sources[session_id][-1] = sources
    _update_last_assistant_sources(session_id, sources)


def get_session_sources(session_id: str) -> dict[int, list]:
    result = {}
    Session = _get_session_local()
    with Session() as session:
        rows = session.execute(
            f"""SELECT id, sources FROM {DEFAULT_MESSAGES_TABLE}
                WHERE session_id = :sid AND sources IS NOT NULL AND sources != '[]' AND sources != ''
                ORDER BY id ASC""",
            {"sid": session_id}
        ).fetchall()
    for row in rows:
        try:
            result[row[0] - 1] = json.loads(row[1])
        except Exception:
            pass
    return result


def list_sessions() -> list[dict]:
    _ensure_schema()
    Session = _get_session_local()
    with Session() as session:
        rows = session.execute(
            f"""SELECT s.session_id, s.session_name, s.created_at, s.updated_at,
                       (SELECT COUNT(*) FROM {DEFAULT_MESSAGES_TABLE} m WHERE m.session_id = s.session_id) AS msg_count
                FROM {DEFAULT_SESSIONS_TABLE} s
                ORDER BY s.updated_at DESC"""
        ).fetchall()
    return [dict(r._mapping) for r in rows]


async def delete_session(session_id: str) -> None:
    s = get_session(session_id)
    try:
        await s.clear_session()
    except Exception:
        pass
    Session = _get_session_local()
    with Session() as session:
        sess = session.get(SessionModel, session_id)
        if sess:
            session.delete(sess)
            session.commit()


async def generate_session_name(question: str) -> str:
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
        name = name.replace('"', '').replace('"', '').replace('"', '').replace('《', '').replace('》', '')
        return name[:20]
    except Exception:
        logger.warning("Failed to generate session name", exc_info=True)
        return question.strip()[:15] + ("..." if len(question) > 15 else "")