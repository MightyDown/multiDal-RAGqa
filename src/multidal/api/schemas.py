from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    task_id: str
    status: str = "pending"


class TaskStatusResponse(BaseModel):
    task_id: str
    filename: str
    status: str
    stage: str | None = None
    error_message: str = ""
    retry_count: int = 0
    max_retries: int = 3


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    kb_ids: list[str] = Field(default_factory=list)
    retrieval: bool = Field(True, description="是否执行检索；False 则跳过 RAG 直接 LLM 回答")
    rewrite_query: bool = True
    text_only: bool = Field(
        False,
        description="实验用：True 时只走文本召回（忽略 image collection）且强制 text LLM 路径，关闭 VLM 多模态分支。",
    )
    session_id: str = Field("", description="会话 ID，用于多轮对话记忆。留空则每次独立问答。")


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict] = Field(default_factory=list)


class KBCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=128)
    description: str = Field("", max_length=512)


class KBResponse(BaseModel):
    kb_id: str
    name: str
    description: str = ""
    doc_count: int = 0
