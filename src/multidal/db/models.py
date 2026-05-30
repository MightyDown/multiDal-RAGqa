from datetime import datetime

from sqlalchemy import Column, DateTime, Integer, String, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from src.multidal.config import settings

Base = declarative_base()


class ParseTaskModel(Base):
    __tablename__ = "parse_tasks"

    task_id = Column(String, primary_key=True)
    filename = Column(String, nullable=False)
    file_path = Column(String, default="")
    file_size = Column(Integer, default=0)
    page_count = Column(Integer, default=0)
    kb_id = Column(String, default="")

    status = Column(String, default="pending")
    stage = Column(String, nullable=True)
    error_message = Column(Text, default="")
    full_text = Column(Text, default="")
    retry_count = Column(Integer, default=0)
    max_retries = Column(Integer, default=3)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class KnowledgeBaseModel(Base):
    __tablename__ = "knowledge_bases"

    kb_id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(Text, default="")
    text_collection = Column(String, nullable=False)
    image_collection = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


_engine = create_engine(
    f"sqlite:///{settings.project_root / settings.db_path}",
    connect_args={"check_same_thread": False},
    echo=False,
)
SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False)


def init_db() -> None:
    Base.metadata.create_all(_engine)
    # 兼容已有数据库：加 full_text 列
    try:
        from sqlalchemy import text
        with _engine.connect() as conn:
            conn.execute(text("ALTER TABLE parse_tasks ADD COLUMN full_text TEXT DEFAULT ''"))
            conn.commit()
    except Exception:
        pass
