from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    """单路召回请求。"""

    query: str = Field(..., min_length=1)
    kb_ids: list[str] = Field(..., min_length=1)
    modality: str = Field("hybrid", pattern="^(text|image|hybrid|keyword|image_desc)$")
    top_k: int = Field(10, ge=1, le=50)


class SearchResult(BaseModel):
    """单条检索结果。"""

    chunk_id: str = Field(...)
    content: str = Field(...)
    modality: str = Field(..., pattern="^(text|image)$")
    kb_id: str = Field(...)
    doc_id: str = Field(...)
    page: int = Field(..., ge=1)
    score: float = Field(..., ge=0.0, le=1.0)
    metadata: dict = Field(default_factory=dict)


class RecallResult(BaseModel):
    """多路召回合并后的候选结果。"""

    chunk_id: str = Field(...)
    content: str = Field(...)
    modality: str = Field(...)
    source: str = Field(..., description="召回路径: dense_text / dense_image / bm25 / image_desc")
    score: float = Field(..., ge=0.0, le=1.0)
    kb_id: str = Field(...)
    doc_id: str = Field(...)
    page: int = Field(..., ge=1)
    image_path: str | None = Field(default=None)


class RerankResult(BaseModel):
    """Rerank 后的精排结果。"""

    chunk_id: str = Field(...)
    content: str = Field(...)
    modality: str = Field(...)
    score: float = Field(..., ge=0.0, le=1.0)
    rank: int = Field(..., ge=1)
    kb_id: str = Field(...)
    doc_id: str = Field(...)
    page: int = Field(..., ge=1)
