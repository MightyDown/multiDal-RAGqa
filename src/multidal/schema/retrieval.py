"""检索链路的数据模型。

本模块定义从召回(Recall)到重排(Rerank)全链路的请求/结果模型,
覆盖 Store 单路查询、多路合并、重排精排三个环节。

类型层级:
    SearchRequest  ->  Store 内部查询参数
    SearchResult   ->  单条召回命中(来自单一路径)
    RecallResult   ->  多路径合并后的候选(去重前/去重后均可)
    RerankResult   ->  Reranker 精排后的最终排序结果
"""

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    """单路召回请求(由 MultiPathRetriever 拆解后下发到各路径)。

    Attributes:
        query: 查询文本(用户问题或改写后的问题),不能为空。
        kb_ids: 目标知识库列表(至少 1 个),决定检索范围。
        modality: 检索模式,可选 ``text`` / ``image`` / ``hybrid`` / ``keyword`` /
                  ``image_desc``,通过正则严格约束(本版本主要使用 ``hybrid``)。
        top_k: 单路返回的 Top-K,范围 1-50,默认 10。
    """

    query: str = Field(..., min_length=1)
    kb_ids: list[str] = Field(..., min_length=1)
    modality: str = Field("hybrid", pattern="^(text|image|hybrid|keyword|image_desc)$")
    top_k: int = Field(10, ge=1, le=50)


class SearchResult(BaseModel):
    """单条召回命中(从单个 Milvus Collection 或单条路径得到)。

    Attributes:
        chunk_id: 命中的 chunk 唯一标识。
        content: 命中的文本内容(图片时为 caption)。
        modality: 命中条目的模态(``text`` 或 ``image``)。
        kb_id: 所属知识库 ID。
        doc_id: 来源文档 ID。
        page: 所在页码(从 1 开始)。
        score: 召回相似度,范围 0-1(由 Store 内部归一化)。
        metadata: 附加元数据(透传自 EmbeddedChunk.metadata)。
    """

    chunk_id: str = Field(...)
    content: str = Field(...)
    modality: str = Field(..., pattern="^(text|image)$")
    kb_id: str = Field(...)
    doc_id: str = Field(...)
    page: int = Field(..., ge=1)
    score: float = Field(..., ge=0.0, le=1.0)
    metadata: dict = Field(default_factory=dict)


class RecallResult(BaseModel):
    """多路召回合并后的候选结果(用于送入 Reranker)。

    Attributes:
        chunk_id: 命中 chunk 唯一标识(可能与其他路径重复,后续去重)。
        content: 文本或图片描述。
        modality: 命中条目模态。
        source: 召回路径,常见取值 ``dense_text`` / ``dense_image`` /
                ``bm25`` / ``image_desc``,便于分析各路径贡献。
        score: 该路径下的相似度分值(0-1)。
        kb_id: 所属知识库 ID。
        doc_id: 来源文档 ID。
        page: 所在页码(从 1 开始)。
        image_path: 图片本地路径(仅 image 模态)。
    """

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
    """Reranker 精排后的最终排序结果(送入 LLM 构造上下文)。

    Attributes:
        chunk_id: 命中 chunk 唯一标识。
        content: 文本或图片描述。
        modality: 命中条目模态。
        score: Reranker 给出的精排分(0-1),越大越相关。
        rank: 在最终结果中的排名(从 1 开始递增)。
        kb_id: 所属知识库 ID。
        doc_id: 来源文档 ID。
        page: 所在页码(从 1 开始)。
        image_path: 图片本地路径(仅 image 模态)。
    """

    chunk_id: str = Field(...)
    content: str = Field(...)
    modality: str = Field(...)
    score: float = Field(..., ge=0.0, le=1.0)
    rank: int = Field(..., ge=1)
    kb_id: str = Field(...)
    doc_id: str = Field(...)
    page: int = Field(..., ge=1)
    image_path: str | None = Field(default=None)
