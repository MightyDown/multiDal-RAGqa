from datetime import datetime

from pydantic import BaseModel, Field


class ParseRequest(BaseModel):
    """Kafka parse.request 消息体。"""

    task_id: str = Field(...)
    file_path: str = Field(...)
    filename: str = Field(...)
    kb_id: str = Field(...)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ParseResponse(BaseModel):
    """Consumer 处理完成后的确认消息。"""

    task_id: str = Field(...)
    status: str = Field(...)
    message: str = Field("")
    completed_at: datetime = Field(default_factory=datetime.utcnow)
