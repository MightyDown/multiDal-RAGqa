"""流水线的基础抽象。

本模块定义 RAG 流水线的两个核心抽象:
    - ``PipelineContext``: 在各阶段间传递的上下文对象(类似 Blackboard 模式)。
    - ``Stage``: 流水线阶段的抽象基类,所有具体阶段(Parser / Embedder / Store)
      都应继承并实现 ``validate`` 与 ``process``。

设计动机:
    流水线以"可插拔阶段 + 统一上下文"为核心。每个阶段只关心自己产出的字段,
    不直接依赖其他阶段的实现。这种松耦合让单元测试可以单独 mock 任一阶段,
    也方便按需增删阶段(例如纯文本场景可跳过 Image Embedder)。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PipelineContext:
    """在流水线各阶段间流转的共享上下文(黑板模式)。

    Attributes:
        task_id: 解析任务唯一 ID,贯穿全流程,用于日志关联与状态回写。
        kb_id: 目标知识库 ID,Parser 阶段无需,Embedder/Store 阶段需要。
        file_path: 原始 PDF 的落盘路径,Parser 阶段读取。
        filename: 原始文件名,用于日志与展示。

        parsed: Parser 阶段的产出(``ParsedDocument``),由 Parser 阶段填入。
        embedded: Embedder 阶段的产出(``list[EmbeddedChunk]``),
                  由 Embedder 阶段填入,供 Store 阶段消费。

        meta: 各阶段自由扩展的元数据字典(如阶段耗时、模型版本、token 数等),
              避免主类字段爆炸。
    """

    task_id: str = ""
    kb_id: str = ""
    file_path: str = ""
    filename: str = ""

    # 各阶段产出(阶段完成后填入)
    parsed: Any = None       # 类型: ParsedDocument,延迟导入避免循环依赖
    embedded: Any = None     # 类型: list[EmbeddedChunk]

    # 元数据自由扩展
    meta: dict[str, Any] = field(default_factory=dict)


class Stage(ABC):
    """流水线阶段抽象基类(模板方法模式)。

    所有具体阶段(Parser / Embedder / Store / ...)必须继承本类并实现两个抽象方法。
    基类不规定状态机,具体阶段自行决定如何读写 ``PipelineContext``。

    Attributes:
        name: 阶段显示名,用于日志与错误信息(如 ``"parser"`` / ``"embedder"``)。
    """

    name: str = ""

    @abstractmethod
    def validate(self) -> bool:
        """阶段启动前的自检钩子。

        典型检查项:
            - 所需模型/服务是否已成功加载(Embedder 需要 Model 已就绪)。
            - 外部依赖是否可达(Milvus / Kafka / MinerU API)。
            - 必要配置是否齐全(API Key、Collection 名称等)。

        Returns:
            bool: True 表示阶段可以安全执行;False 表示阶段不具备执行条件,
                  Orchestrator 会抛 ``RuntimeError`` 终止流水线。
        """
        ...

    @abstractmethod
    def process(self, ctx: PipelineContext) -> PipelineContext:
        """执行本阶段的核心逻辑。

        阶段应:
            1. 从 ``ctx`` 读取所需输入(如 ``ctx.file_path``、``ctx.parsed``)。
            2. 执行实际工作(MinerU 解析 / Embedding / Milvus 写入等)。
            3. 将结果回写到 ``ctx`` 的对应字段(如 ``ctx.parsed``、``ctx.embedded``)。
            4. 返回更新后的 ``ctx``(通常是同一个对象,但允许替换)。

        Args:
            ctx: 流水线当前上下文。

        Returns:
            PipelineContext: 更新后的上下文(传给下一阶段)。
        """
        ...
