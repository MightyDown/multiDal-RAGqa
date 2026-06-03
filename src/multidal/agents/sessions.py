"""基于 MySQL 的智能体会话存储。

本模块提供与 openai-agents SDK ``SQLiteSession`` 接口对齐的 ``MySQLSession``,
使得上层 (QueryAgent 等) 可在不改业务代码的情况下切换到 MySQL 持久化。

表结构:
    - ``agent_sessions``: 会话元信息(session_id 主键 + 名称 + 时间)。
    - ``agent_messages``: 单条消息(LONGTEXT 存 JSON,Text 存 sources)。

辅助能力:
    - 会话命名(``generate_session_name``):用小模型给会话起 3-10 字中文名。
    - 引用源记录(``set_session_sources`` / ``get_session_sources``):记录每条
      assistant 消息对应的检索来源,供前端展示"参考了哪些文档"。
"""

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

# 默认表名(集中管理,便于统一修改)。
DEFAULT_SESSIONS_TABLE = "agent_sessions"
DEFAULT_MESSAGES_TABLE = "agent_messages"

# 进程内缓存:session_id -> {msg_index -> sources}
# 设计与 MySQL 表并存:用于高频写入(避免每条消息都 UPDATE),定期落库。
_session_sources: dict[str, dict[int, list]] = {}
# session_id -> 下一个预期的消息索引(用于按写入顺序记录 sources)。
_session_next_idx: dict[str, int] = {}


class SessionModel(Base):
    """``agent_sessions`` 表的 ORM 映射。

    Attributes:
        session_id: 会话主键(通常由 API 层生成 UUID)。
        session_name: 人类可读的会话名(由 ``generate_session_name`` 生成)。
        created_at: 创建时间(UTC)。
        updated_at: 最近一次更新时间(UTC),``onupdate`` 自动维护。
    """

    __tablename__ = DEFAULT_SESSIONS_TABLE

    session_id = Column(String(64), primary_key=True)
    session_name = Column(String(256), default="")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class MessageModel(Base):
    """``agent_messages`` 表的 ORM 映射。

    Attributes:
        id: 自增主键(DB 内部 ID,前端通过此 ID 关联 sources)。
        session_id: 外键,关联 ``SessionModel.session_id``,删除时级联清理。
        message_data: 整条消息的 JSON 字符串(role / content / tool_calls 等)。
        sources: 该消息的引用源列表(JSON 字符串,默认空数组)。
        created_at: 创建时间(UTC)。
    """

    __tablename__ = DEFAULT_MESSAGES_TABLE

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), ForeignKey("agent_sessions.session_id", ondelete="CASCADE"), nullable=False)
    message_data = Column(LONGTEXT, nullable=False)
    sources = Column(Text, default="[]")
    created_at = Column(DateTime, default=datetime.utcnow)


def _ensure_schema() -> None:
    """惰性建表:首次访问会话相关接口时确保表存在(``Base.metadata.create_all``)。"""
    from src.multidal.db.models import _engine
    Base.metadata.create_all(_engine)


class MySQLSession:
    """MySQL 持久化的 openai-agents 会话实现,接口与 ``SQLiteSession`` 对齐。

    Attributes:
        session_id: 会话 ID。
        db_path: 保留字段(为兼容 SDK 内部调用),实际未使用 SQLite。
    """

    def __init__(self, session_id: str) -> None:
        """构造一个会话句柄(不立即建表,首次读写时再 ``_ensure_schema``)。

        Args:
            session_id: 会话 ID(主键)。
        """
        self.session_id = session_id
        # 保留字段,实际写入走 MySQL
        self.db_path = str(settings.project_root / "data" / "sessions.db")
        _ensure_schema()

    async def get_items(self, limit: int | None = None) -> list[dict]:
        """读取会话中的消息列表(SDK 调用以构造 LLM context)。

        Args:
            limit: 仅返回最近 N 条(按 id 倒序取再回正);None 时返回全部。

        Returns:
            list[dict]: 消息字典列表,每条额外带 ``_db_id`` 字段(DB 主键,
                        便于后续关联 sources)。
        """
        with SessionLocal() as db:
            q = db.query(MessageModel).filter(MessageModel.session_id == self.session_id).order_by(MessageModel.id.asc())
            if limit is not None:
                # 取最近 N 条再反序,保持时间正序
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
                    # 跳过坏数据,避免单条损坏阻塞整次读取
                    pass
            return items

    async def add_items(self, items: list[dict]) -> None:
        """批量写入消息(SDK 在 LLM 返回后调用)。

        流程:
            1. 确保 ``SessionModel`` 行存在(upsert);
            2. 逐条插入 ``MessageModel``;
            3. 一次性 commit 提升性能。

        Args:
            items: 消息字典列表(SDK 标准格式)。
        """
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
        """弹出最近一条消息并删除(SDK 的"撤销"功能使用)。

        Returns:
            dict | None: 被弹出的消息;无消息时返回 None。
        """
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
        """清空整个会话的消息与会话行(谨慎使用)。"""
        with SessionLocal() as db:
            db.query(MessageModel).filter(MessageModel.session_id == self.session_id).delete()
            db.query(SessionModel).filter(MessageModel.session_id == self.session_id).delete()
            db.commit()

    def close(self) -> None:
        """占位方法(对齐 SDK 接口,无实际资源需要释放)。"""
        pass


def get_session(session_id: str) -> MySQLSession:
    """工厂函数:按 ID 获取 ``MySQLSession`` 实例。

    Args:
        session_id: 会话 ID。

    Returns:
        MySQLSession: 同一 ID 多次调用会得到多个实例(本身无内部状态,无副作用)。
    """
    return MySQLSession(session_id)


def set_session_name(session_id: str, name: str) -> None:
    """为会话设置/更新名称(upsert)。

    Args:
        session_id: 会话 ID。
        name: 人类可读的会话名(中文 3-10 字)。
    """
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
    """把 sources 写到"最近一条 assistant 消息"的 ``sources`` 字段(私有助手)。

    通过 ``JSON_EXTRACT`` 定位 role=='assistant' 的最新消息。

    Args:
        session_id: 会话 ID。
        sources: 来源列表(检索 chunk 摘要等)。
    """
    with SessionLocal() as db:
        row = db.query(MessageModel).filter(
            MessageModel.session_id == session_id,
            text(f"JSON_EXTRACT(message_data, '$.role') = 'assistant'")
        ).order_by(MessageModel.id.desc()).first()
        if row:
            row.sources = json.dumps(sources)
            db.commit()


def set_session_sources(session_id: str, sources: list, msg_idx: int | None = None) -> None:
    """记录某条消息的 sources(同时写入进程缓存与 MySQL)。

    Args:
        session_id: 会话 ID。
        sources: 来源列表。
        msg_idx: 消息索引(为 None 时默认写到最近一条 assistant 消息)。
    """
    if session_id not in _session_sources:
        _session_sources[session_id] = {}
    if msg_idx is not None:
        _session_sources[session_id][msg_idx] = sources
    else:
        _session_sources[session_id][-1] = sources
    _update_last_assistant_sources(session_id, sources)


def get_session_sources(session_id: str) -> dict[int, list]:
    """从 MySQL 读取某会话下"含 sources"的消息映射。

    Returns:
        dict[int, list]: ``{db_row_id: sources_list}``,只包含 sources 非空的行。
    """
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
    """列出全部会话,按 ``updated_at`` 倒序,并附带消息数。

    Returns:
        list[dict]: 每条含 ``session_id`` / ``session_name`` / ``created_at`` /
                    ``updated_at`` / ``msg_count``。
    """
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
    """删除整个会话(消息 + 会话行)。

    Args:
        session_id: 要删除的会话 ID。
    """
    s = get_session(session_id)
    try:
        await s.clear_session()
    except Exception:
        # clear_session 失败时仍尝试删会话行(可能为脏数据)
        pass
    with SessionLocal() as db:
        sess = db.get(SessionModel, session_id)
        if sess:
            db.delete(sess)
            db.commit()


async def generate_session_name(question: str, answer: str = "") -> str:
    """用小模型给对话起一个 3-10 字的中文会话名(降级用问题前 15 字)。

    Args:
        question: 用户首条问题。
        answer: AI 回答(可选,提供可让命名更贴切)。

    Returns:
        str: 清洗后的会话名(去除 think 标签、引号、书名号等)。
    """
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
        # 移除思考标签(部分小模型即便关闭 thinking 仍可能输出残留)
        name = name.replace('<think>', '').replace('', '')
        name = name.replace('"', '').replace('"', '').replace('"', '').replace('《', '').replace('》', '')
        return name[:20].strip()
    except Exception:
        logger.warning("Failed to generate session name", exc_info=True)
        # 降级:用问题前 15 字作名(超过则加省略号)
        return question.strip()[:15] + ("..." if len(question) > 15 else "")
