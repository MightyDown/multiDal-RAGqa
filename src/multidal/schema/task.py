from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    EXHAUSTED = "exhausted"


class TaskStage(str, Enum):
    PARSER = "parser"
    EMBEDDER = "embedder"
    STORE = "store"


class ParseTask(BaseModel):
    """解析任务状态，对应 SQLite parse_tasks 表。"""

    task_id: str = Field(...)
    filename: str = Field(..., min_length=1)
    file_path: str = Field("")
    file_size: int = Field(0, ge=0)
    page_count: int = Field(0, ge=0)
    kb_id: str = Field("")

    status: TaskStatus = Field(TaskStatus.PENDING)
    stage: TaskStage | None = Field(None)
    error_message: str = Field("")
    retry_count: int = Field(0, ge=0)
    max_retries: int = Field(3, ge=1)

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
