"""Kafka 消息体的数据契约。

本模块定义了异步流水线中两类 Kafka 消息的 Pydantic 模型:
    - ParseRequest:  生产者(API 层) -> Consumer(Parser worker)的解析请求。
    - ParseResponse: Consumer -> 持久化层的处理结果回执(目前用于日志/监控)。

消息流转:
    POST /api/ingest -> 落盘 + 写 MySQL pending -> Producer(发送 ParseRequest)
    -> Consumer(MinerU 解析 + Embedder + Store) -> 更新 MySQL 状态
"""

from datetime import datetime

from pydantic import BaseModel, Field


class ParseRequest(BaseModel):
    """Kafka 主题 ``parse.request`` 的消息体。

    Attributes:
        task_id: 解析任务唯一 ID(对应 MySQL parse_tasks.task_id),
                 消费者用此 ID 定位并更新任务状态。
        file_path: 已落盘的 PDF 绝对路径,Consumer 读取后送入 MinerU。
        filename: 原始文件名,用于在结果/日志中显示。
        kb_id: 目标知识库 ID,Consumer 完成 Embedder 后会写入
               ``{kb_id}_text`` 与 ``{kb_id}_image`` 两个 Milvus Collection。
        timestamp: 消息生成时间(UTC),便于排障与延迟分析。
    """

    task_id: str = Field(...)
    file_path: str = Field(...)
    filename: str = Field(...)
    kb_id: str = Field(...)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ParseResponse(BaseModel):
    """Consumer 处理完成后的确认消息(主题 ``parse.response``)。

    Attributes:
        task_id: 对应的解析任务 ID,用于关联回原始 ParseRequest。
        status: 终态字符串,通常取 ``"completed"`` 或 ``"failed"``;
                与 ``TaskStatus`` 枚举保持语义一致(本字段不强制枚举以兼容扩展)。
        message: 可选的附加信息(失败原因、成功耗时等),默认空串。
        completed_at: 完成时间(UTC),由 Consumer 在写消息前填充。
    """

    task_id: str = Field(...)
    status: str = Field(...)
    message: str = Field("")
    completed_at: datetime = Field(default_factory=datetime.utcnow)
