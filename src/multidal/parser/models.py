from dataclasses import dataclass, field


@dataclass
class TextChunk:
    content: str
    page: int
    bbox: tuple[float, float, float, float] | None = None


@dataclass
class ImageRegion:
    data: bytes
    page: int
    bbox: tuple[float, float, float, float] | None = None
    caption: str = ""
    label: str = ""


@dataclass
class TableChunk:
    html: str
    page: int
    bbox: tuple[float, float, float, float] | None = None


@dataclass
class ParsedDocument:
    filename: str
    text_chunks: list[TextChunk] = field(default_factory=list)
    images: list[ImageRegion] = field(default_factory=list)
    tables: list[TableChunk] = field(default_factory=list)
    page_count: int = 0
