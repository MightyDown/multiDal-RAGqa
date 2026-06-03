"""Embedder 阶段输出与中间数据结构。

本模块定义向量化后的数据单元,直接对应 Milvus 的一条记录。
通过将 ``Embedding`` 单独建模,可在多模型共存时记录所用模型及维度,
便于后续按模型过滤或回溯。
"""

from typing import Any

from pydantic import BaseModel, Field


class Embedding(BaseModel):
    """单条向量表示。

    Attributes:
        model_name: 产生该向量的模型名称(如 ``BAAI/bge-large-zh-v1.5``)，
                     用于在多模型场景下做来源追溯。
        dim: 向量维度,必须为正整数(常见:1024 / 768 / 1536)。
        vector: 实际的浮点向量值列表,长度应等于 ``dim``。
    """

    model_name: str = Field(..., description="模型名称，如 Qwen/Qwen-Embedding-0.6B")
    dim: int = Field(..., gt=0, description="向量维度")
    vector: list[float] = Field(..., description="向量值")


class EmbeddedChunk(BaseModel):
    """已向量化的数据块,作为 Store 阶段写入 Milvus 的最小单元。

    Attributes:
        chunk_id: 唯一标识(对应 Milvus 主键),由 Parser 阶段透传。
        content: 文本内容,文本块即为原文,图片块为 caption/描述文本。
        embedding: 包含向量本身、维度与模型名。
        modality: 模态类型,取 ``text`` 或 ``image``,通过正则严格约束。
        kb_id: 所属知识库 ID(决定写入哪个 Milvus Collection)。
        doc_id: 来源文档 ID,用于按文档聚合与权限控制。
        page: 原始页码(从 1 开始),用于前端引用与高亮定位。
        metadata: 附属元数据(如 chunk_type、bbox、image_path、置信度等)。
        image_path: 图片本地路径,仅 ``image`` 模态时有值,用于前端回显与 LLM 引用。
    """

    chunk_id: str = Field(...)
    content: str = Field(..., description="文本内容或图片描述")
    embedding: Embedding
    modality: str = Field(..., pattern="^(text|image)$")
    kb_id: str = Field(..., description="所属知识库 ID")
    doc_id: str = Field(...)
    page: int = Field(..., ge=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
    image_path: str | None = Field(default=None, description="图片原始路径，仅 image modality 有值")
