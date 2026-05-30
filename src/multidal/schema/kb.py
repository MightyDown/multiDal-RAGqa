from datetime import datetime

from pydantic import BaseModel, Field


class KBCreateRequest(BaseModel):
    """创建知识库请求。"""

    name: str = Field(..., min_length=1, max_length=128)
    description: str = Field("", max_length=512)


class KBResponse(BaseModel):
    """知识库信息响应。"""

    kb_id: str = Field(...)
    name: str = Field(...)
    description: str = Field("")
    doc_count: int = Field(0, ge=0)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class KBListResponse(BaseModel):
    """知识库列表响应。"""

    kbs: list[KBResponse] = Field(default_factory=list)
    total: int = Field(0, ge=0)
