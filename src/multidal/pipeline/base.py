from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PipelineContext:
    """在流水线各阶段间流转的上下文。"""

    task_id: str = ""
    kb_id: str = ""
    file_path: str = ""
    filename: str = ""

    # 各阶段产出（阶段完成后填入）
    parsed: Any = None       # ParsedDocument
    embedded: Any = None     # list[EmbeddedChunk]

    # 元数据自由扩展
    meta: dict[str, Any] = field(default_factory=dict)


class Stage(ABC):
    """流水线阶段抽象基类。"""

    name: str = ""

    @abstractmethod
    def validate(self) -> bool:
        """检查该阶段是否就绪（模型已加载 / 服务已连接等）。"""
        ...

    @abstractmethod
    def process(self, ctx: PipelineContext) -> PipelineContext:
        """执行本阶段逻辑，返回更新后的上下文。"""
        ...
