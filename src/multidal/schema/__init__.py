"""Schema 包 - 跨阶段数据契约与 JSON 校验模型。

本包使用 Pydantic BaseModel 定义整个 RAG 流水线中流转的所有数据结构,
覆盖从任务创建、文档解析、向量化、召回重排到 KB 管理的完整生命周期。

设计原则:
    - 严格的类型约束:通过 Field 约束长度、范围、正则等,确保上下游接口一致。
    - 阶段解耦:Parser、Embedder、Store、Agent 各阶段只依赖本包的数据模型,
      不直接引用对方模块的实现细节。
    - 不可变契约:任何字段调整都需同步更新对应的生产/消费端与持久化层。

子模块分工:
    - task:    解析任务的状态机(pending/processing/completed/failed/exhausted)。
    - queue:   Kafka 消息体(parse.request / parse.response)。
    - kb:      知识库元数据(创建请求、详情响应、列表响应)。
    - document: MinerU 解析产物(ImageRegion / TextChunk / TableChunk / ParsedDocument)。
    - embedding: 向量化产物(Embedding 向量本身、EmbeddedChunk 索引文档)。
    - retrieval: 检索链路模型(SearchRequest、SearchResult、RecallResult、RerankResult)。
"""

from src.multidal.schema.document import ImageRegion, ParsedDocument, TableChunk, TextChunk
from src.multidal.schema.embedding import EmbeddedChunk, Embedding
from src.multidal.schema.kb import KBCreateRequest, KBListResponse, KBResponse
from src.multidal.schema.queue import ParseRequest, ParseResponse
from src.multidal.schema.retrieval import RecallResult, RerankResult, SearchRequest, SearchResult
from src.multidal.schema.task import ParseTask, TaskStage, TaskStatus
