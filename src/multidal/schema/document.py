from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ImageRegion(BaseModel):
    """图片区域，从 MinerU 解析结果中提取。"""

    image_id: str = Field(..., description="图片唯一标识")
    page: int = Field(..., ge=1)
    bbox: tuple[float, float, float, float] | None = Field(None, description="包围框 (x1, y1, x2, y2)")
    caption: str = Field("", description="MinerU 生成的图片描述")
    label: str = Field("", description="图片类型标签 (chart/photo/diagram)")
    width: int = Field(0, ge=0)
    height: int = Field(0, ge=0)
    image_path: str = Field("", description="图片文件在磁盘上的路径")


class TextChunk(BaseModel):
    """文本块，从 MinerU 解析结果中提取。"""

    chunk_id: str = Field(..., description="文本块唯一标识")
    content: str = Field(..., min_length=1)
    page: int = Field(..., ge=1)
    bbox: tuple[float, float, float, float] | None = Field(None)
    chunk_type: str = Field("paragraph", description="paragraph / heading / caption / footnote")


class TableChunk(BaseModel):
    """表格块，HTML 结构。"""

    table_id: str = Field(..., description="表格唯一标识")
    html: str = Field(..., min_length=1, description="表格 HTML 结构")
    page: int = Field(..., ge=1)
    bbox: tuple[float, float, float, float] | None = Field(None)
    caption: str = Field("")


class ParsedDocument(BaseModel):
    """MinerU 解析完成的文档，进入 Embedder 阶段。"""

    doc_id: str = Field(..., description="文档唯一标识")
    filename: str = Field(..., min_length=1)
    file_path: str = Field("")
    page_count: int = Field(..., ge=1)
    text_chunks: list[TextChunk] = Field(default_factory=list)
    images: list[ImageRegion] = Field(default_factory=list)
    tables: list[TableChunk] = Field(default_factory=list)
    full_text: str = Field("", description="MinerU 输出的完整 markdown")
    parsed_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)
