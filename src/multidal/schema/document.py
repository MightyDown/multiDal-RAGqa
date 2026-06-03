"""MinerU 解析产物的数据模型。

本模块定义了 PDF 解析阶段的输出结构,所有模型由 Parser 阶段产出,
供下游 Embedder 阶段消费。

设计要点:
    - ``ParsedDocument`` 是顶层容器,聚合了文本块、图片区域、表格块与完整 markdown。
    - 每种 chunk/region 都带 ``page`` 与可选 ``bbox``,用于回溯定位。
    - ``chunk_id`` / ``image_id`` / ``table_id`` 全局唯一,在 Embedder 阶段会作为
      Milvus 主键的组成部分。
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ImageRegion(BaseModel):
    """PDF 中提取的一张图片区域,作为 Image Embedder 的输入。

    Attributes:
        image_id: 图片唯一标识(跨文档全局),作为向量化主键前缀。
        page: 图片所在 PDF 页码(从 1 开始)。
        bbox: 包围框 ``(x1, y1, x2, y2)``(PDF 坐标,可能为 None)。
        caption: MinerU 自带的图片描述(中文/英文混合),优先用于语义召回。
        label: 图片类型标签,常见取值 ``chart`` / ``photo`` / ``diagram``;
               来自 MinerU 内部分类,可为空字符串。
        width: 图片宽度(像素),用于前端展示。
        height: 图片高度(像素),用于前端展示。
        image_path: 图片文件在磁盘上的路径(MinerU 抽图产物),
                    为空表示未落盘(纯描述型图片)。
    """

    image_id: str = Field(..., description="图片唯一标识")
    page: int = Field(..., ge=1)
    bbox: tuple[float, float, float, float] | None = Field(None, description="包围框 (x1, y1, x2, y2)")
    caption: str = Field("", description="MinerU 生成的图片描述")
    label: str = Field("", description="图片类型标签 (chart/photo/diagram)")
    width: int = Field(0, ge=0)
    height: int = Field(0, ge=0)
    image_path: str = Field("", description="图片文件在磁盘上的路径")


class TextChunk(BaseModel):
    """PDF 中切分出的一段文本块,作为 Text Embedder 的输入。

    Attributes:
        chunk_id: 文本块唯一标识(跨文档全局)。
        content: 文本内容(MinerU 抽取后清洗过的纯文本,可能含中文)。
        page: 所在页码(从 1 开始)。
        bbox: 包围框 ``(x1, y1, x2, y2)``,可能为 None(横跨多列时)。
        chunk_type: 块类型,常见取值 ``paragraph`` / ``heading`` / ``caption`` /
                    ``footnote``;用于 Embedder 选择不同的处理策略。
    """

    chunk_id: str = Field(..., description="文本块唯一标识")
    content: str = Field(..., min_length=1)
    page: int = Field(..., ge=1)
    bbox: tuple[float, float, float, float] | None = Field(None)
    chunk_type: str = Field("paragraph", description="paragraph / heading / caption / footnote")


class TableChunk(BaseModel):
    """PDF 中识别出的表格块,以 HTML 字符串表示结构。

    Attributes:
        table_id: 表格唯一标识(跨文档全局)。
        html: 表格 HTML 结构(``<table>...``),可直接渲染或解析为 DataFrame。
        page: 所在页码(从 1 开始)。
        bbox: 包围框 ``(x1, y1, x2, y2)``,可能为 None。
        caption: 表格标题/说明文字,默认空串。
    """

    table_id: str = Field(..., description="表格唯一标识")
    html: str = Field(..., min_length=1, description="表格 HTML 结构")
    page: int = Field(..., ge=1)
    bbox: tuple[float, float, float, float] | None = Field(None)
    caption: str = Field("")


class ParsedDocument(BaseModel):
    """MinerU 解析完成的整篇文档,作为 Parser 阶段的标准输出。

    Attributes:
        doc_id: 文档唯一 ID(对应 MySQL parse_tasks.task_id)。
        filename: 原始文件名(用户上传时的文件名)。
        file_path: PDF 落盘路径(可用于回查/重处理)。
        page_count: 文档总页数(>=1)。
        text_chunks: 切分出的文本块列表,按页内顺序排列。
        images: 抽出的图片区域列表。
        tables: 识别出的表格块列表。
        full_text: MinerU 输出的完整 markdown 文本,作为兜底(全量上下文)。
        parsed_at: 解析完成时间(UTC)。
        metadata: 附加元数据(如 MinerU 版本、模型 ID、解析耗时等)。
    """

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
