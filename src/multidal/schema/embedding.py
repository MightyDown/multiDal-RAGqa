from typing import Any

from pydantic import BaseModel, Field


class Embedding(BaseModel):
    """单个向量。"""

    model_name: str = Field(..., description="模型名称，如 Qwen/Qwen-Embedding-0.6B")
    dim: int = Field(..., gt=0, description="向量维度")
    vector: list[float] = Field(..., description="向量值")


class EmbeddedChunk(BaseModel):
    """向量化后的数据块，进入 Store 阶段。"""

    chunk_id: str = Field(...)
    content: str = Field(..., description="文本内容或图片描述")
    embedding: Embedding
    modality: str = Field(..., pattern="^(text|image)$")
    kb_id: str = Field(..., description="所属知识库 ID")
    doc_id: str = Field(...)
    page: int = Field(..., ge=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
    image_path: str | None = Field(default=None, description="图片原始路径，仅 image modality 有值")
