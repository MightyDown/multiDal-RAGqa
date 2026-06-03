"""Parser 阶段的 dataclass 形态中间模型(早期实现)。

本模块在 ``schema.document`` 引入 Pydantic 模型之前是解析产物的唯一载体。
当前生产路径已统一使用 ``schema.document`` 中的 Pydantic 模型(支持 JSON 序列化、
字段校验、IDE 提示),本模块的 dataclass 形态仅作历史参考。

保留原因:
    - 早期单测可能仍直接 import 此处类型;
    - 作为无依赖的 dataclass 形态,可被需要"轻量类型"的工具脚本引用。
"""

from dataclasses import dataclass, field


@dataclass
class TextChunk:
    """文本块(dataclass 形态)。

    Attributes:
        content: 文本内容。
        page: 所在页码(从 1 开始)。
        bbox: 包围框 ``(x1, y1, x2, y2)``,可为 None。
    """

    content: str
    page: int
    bbox: tuple[float, float, float, float] | None = None


@dataclass
class ImageRegion:
    """图片区域(dataclass 形态,bytes 存图)。

    Attributes:
        data: 图片的原始字节流(MinerU 抽取后的二进制)。
        page: 所在页码(从 1 开始)。
        bbox: 包围框 ``(x1, y1, x2, y2)``,可为 None。
        caption: 图片描述(可能为空)。
        label: 图片类型标签,如 ``chart`` / ``photo``。
    """

    data: bytes
    page: int
    bbox: tuple[float, float, float, float] | None = None
    caption: str = ""
    label: str = ""


@dataclass
class TableChunk:
    """表格块(dataclass 形态)。

    Attributes:
        html: 表格 HTML 结构字符串。
        page: 所在页码(从 1 开始)。
        bbox: 包围框 ``(x1, y1, x2, y2)``,可为 None。
    """

    html: str
    page: int
    bbox: tuple[float, float, float, float] | None = None


@dataclass
class ParsedDocument:
    """解析产物的根容器(dataclass 形态)。

    Attributes:
        filename: 原始文件名。
        text_chunks: 文本块列表。
        images: 图片区域列表。
        tables: 表格块列表。
        page_count: 文档总页数,默认 0(待回填)。
    """

    filename: str
    text_chunks: list[TextChunk] = field(default_factory=list)
    images: list[ImageRegion] = field(default_factory=list)
    tables: list[TableChunk] = field(default_factory=list)
    page_count: int = 0
