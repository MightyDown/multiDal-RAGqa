from datetime import datetime

from sqlalchemy import Column, DateTime, Integer, String, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.dialects.mysql import LONGTEXT

from src.multidal.config import settings

Base = declarative_base()


class ParseTaskModel(Base):
    __tablename__ = "parse_tasks"

    task_id = Column(String(64), primary_key=True)
    filename = Column(String(256), nullable=False)
    file_path = Column(String(512), default="")
    file_size = Column(Integer, default=0)
    page_count = Column(Integer, default=0)
    kb_id = Column(String(64), default="")

    status = Column(String(32), default="pending")
    stage = Column(String(64), nullable=True)
    error_message = Column(Text, default="")
    full_text = Column(LONGTEXT, default="")
    retry_count = Column(Integer, default=0)
    max_retries = Column(Integer, default=3)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class KnowledgeBaseModel(Base):
    __tablename__ = "knowledge_bases"

    kb_id = Column(String(64), primary_key=True)
    name = Column(String(256), nullable=False)
    description = Column(Text, default="")
    text_collection = Column(String(128), nullable=False)
    image_collection = Column(String(128), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


_engine = create_engine(
    f"mysql+pymysql://{settings.mysql_user}:{settings.mysql_password}@{settings.mysql_host}:{settings.mysql_port}/{settings.mysql_database}?charset=utf8mb4",
    connect_args={"connect_timeout": 10},
    echo=False,
    pool_pre_ping=True,
)
SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False)


def init_db() -> None:
    # Import session models so they're registered on Base before create_all()
    from src.multidal.agents.sessions import SessionModel, MessageModel  # noqa: F401
    Base.metadata.create_all(_engine)
