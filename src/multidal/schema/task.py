"""解析任务的持久化模型与状态机。

本模块定义 MySQL ``parse_tasks`` 表对应的 Pydantic 模型以及任务状态/阶段枚举。
任务在整个解析流水线中按以下状态推进:

    PENDING -> PROCESSING -> COMPLETED
                       \\-> FAILED -> (重试) -> PROCESSING ...
                                          \\-> (重试耗尽) -> EXHAUSTED

阶段(stage)用于定位失败发生在哪一段(Parser / Embedder / Store),
便于运维快速排查。
"""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    """解析任务的状态枚举。

    取值:
        PENDING: 已写入数据库,等待 Kafka Consumer 拉取。
        PROCESSING: 已被 Consumer 取出,流水线进行中。
        COMPLETED: 全流程完成(Embedder + Store 全部成功)。
        FAILED: 本次处理失败,等待指数退避后重试。
        EXHAUSTED: 达到 ``max_retries`` 上限,不再重试,需要人工介入。
    """

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    EXHAUSTED = "exhausted"


class TaskStage(str, Enum):
    """任务当前所处的流水线阶段(用于失败定位)。

    取值:
        PARSER:  MinerU PDF 解析阶段。
        EMBEDDER: Text/Image 向量化阶段。
        STORE:   Milvus 写入阶段。
    """

    PARSER = "parser"
    EMBEDDER = "embedder"
    STORE = "store"


class ParseTask(BaseModel):
    """解析任务的运行时模型,对应 MySQL ``parse_tasks`` 表的一行。

    Attributes:
        task_id: 任务唯一 ID(主键),由 API 层生成后贯穿全流程。
        filename: 用户上传时的原始文件名,用于展示与日志。
        file_path: PDF 落盘后的绝对路径,Parser 阶段读取此文件。
        file_size: 文件大小(字节),>=0,便于监控与配额。
        page_count: 文档总页数,默认 0(解析完成后回填)。
        kb_id: 目标知识库 ID,默认空串(待绑定)。

        status: 任务当前状态(状态机见模块文档)。
        stage: 任务当前所处的流水线阶段(失败时定位失败点)。
        error_message: 失败时的错误描述,成功时为空串。
        retry_count: 已重试次数,>=0。
        max_retries: 最大重试次数,>=1,默认 3。

        created_at: 任务创建时间(UTC)。
        updated_at: 最近一次状态更新时间(UTC),每次写库需刷新。
    """

    task_id: str = Field(...)
    filename: str = Field(..., min_length=1)
    file_path: str = Field("")
    file_size: int = Field(0, ge=0)
    page_count: int = Field(0, ge=0)
    kb_id: str = Field("")

    status: TaskStatus = Field(TaskStatus.PENDING)
    stage: TaskStage | None = Field(None)
    error_message: str = Field("")
    retry_count: int = Field(0, ge=0)
    max_retries: int = Field(3, ge=1)

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
