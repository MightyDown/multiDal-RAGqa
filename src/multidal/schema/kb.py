"""KB(知识库)相关的请求/响应数据模型。

本模块定义了知识库管理 API 所使用的所有 Pydantic 模型,
涵盖创建请求、详情响应、列表响应三类典型场景。

约定:
    - kb_id 由后端生成(通常为 uuid4 短串),不通过请求传入,避免 ID 冲突。
    - doc_count 由 Repository 实时统计,响应中携带便于前端展示。
    - 列表响应使用 total + kbs 模式,前端可基于 total 实现分页。
"""

from datetime import datetime

from pydantic import BaseModel, Field


class KBCreateRequest(BaseModel):
    """创建知识库的请求体。

    Attributes:
        name: 知识库名称,长度 1-128 字符,前端应做重复名校验但不强制。
        description: 可选描述,长度上限 512 字符,默认空串。
    """

    name: str = Field(..., min_length=1, max_length=128)
    description: str = Field("", max_length=512)


class KBResponse(BaseModel):
    """知识库详情响应。

    Attributes:
        kb_id: 后端生成的知识库唯一 ID(同时是 Milvus Collection 名前缀)。
        name: 知识库名称。
        description: 知识库描述。
        doc_count: 当前 KB 内已成功入库的文档数量(>=0)。
        created_at: 创建时间(UTC),由后端填充。
    """

    kb_id: str = Field(...)
    name: str = Field(...)
    description: str = Field("")
    doc_count: int = Field(0, ge=0)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class KBListResponse(BaseModel):
    """知识库列表响应,采用 total + items 模式。

    Attributes:
        kbs: 当前页的知识库列表,默认空列表。
        total: 满足条件的知识库总数(用于前端分页)。
    """

    kbs: list[KBResponse] = Field(default_factory=list)
    total: int = Field(0, ge=0)
