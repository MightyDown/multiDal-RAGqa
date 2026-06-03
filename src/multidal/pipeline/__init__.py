"""Pipeline 包 - 流水线阶段抽象与编排。

对外暴露:
    - ``PipelineContext``: 阶段间共享上下文(数据载体)。
    - ``Stage``: 阶段抽象基类(模板方法)。
    - ``Orchestrator``: 顺序执行阶段的驱动器。
"""

from src.multidal.pipeline.base import PipelineContext, Stage
from src.multidal.pipeline.orchestrator import Orchestrator
