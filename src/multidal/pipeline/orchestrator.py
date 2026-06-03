"""流水线编排器。

本模块提供 ``Orchestrator`` 类,负责按顺序执行 ``Stage`` 列表中的所有阶段,
并处理阶段间日志、校验与异常传播。它是流水线的"驱动器",
不持有任何业务逻辑。
"""

from __future__ import annotations

import logging

from src.multidal.pipeline.base import PipelineContext, Stage

logger = logging.getLogger(__name__)


class Orchestrator:
    """顺序执行流水线阶段,负责传递 ``PipelineContext`` 与日志记录。

    使用方式::

        orchestrator = Orchestrator(stages=[ParserStage(), EmbedderStage(), StoreStage()])
        final_ctx = orchestrator.run(ctx)

    Attributes:
        stages: 待执行的阶段列表,按列表顺序依次运行。
    """

    def __init__(self, stages: list[Stage]):
        """初始化编排器。

        Args:
            stages: 流水线阶段列表,顺序敏感(前一阶段的输出是后一阶段的输入)。
        """
        self.stages = stages

    def run(self, ctx: PipelineContext) -> PipelineContext:
        """按顺序执行所有阶段。

        执行流程:
            1. 遍历 ``self.stages``;
            2. 记录 ``start`` 日志(含 task_id 与阶段名);
            3. 调用 ``stage.validate()`` 进行自检,失败则抛出 ``RuntimeError``;
            4. 调用 ``stage.process(ctx)`` 执行阶段逻辑,得到更新后的 ctx;
            5. 记录 ``done`` 日志。

        注意:
            - 任何阶段抛出的异常都会中断流水线(异常向上传播)。
            - 阶段可读可写 ``ctx``,但不应持有引用跨阶段共享可变对象。

        Args:
            ctx: 流水线初始上下文(``task_id``、``file_path`` 等通常已填充)。

        Returns:
            PipelineContext: 全部阶段执行完毕后的最终上下文。

        Raises:
            RuntimeError: 任一阶段 ``validate()`` 返回 ``False`` 时抛出,
                          错误信息包含 ``task_id`` 与失败的阶段名,便于定位。
        """
        for stage in self.stages:
            # 阶段开始日志:便于排障时还原流水线时序
            logger.info(
                "[%s] stage=%s start", ctx.task_id, stage.name
            )
            if not stage.validate():
                # 阶段不具备执行条件(模型未加载、外部服务不可达等),
                # 直接失败让上层(Kafka Consumer)进入重试逻辑
                raise RuntimeError(
                    f"[{ctx.task_id}] stage={stage.name} validate failed"
                )
            ctx = stage.process(ctx)
            # 阶段完成日志:与 start 配对,便于计算阶段耗时(由日志采集层完成)
            logger.info(
                "[%s] stage=%s done", ctx.task_id, stage.name
            )
        return ctx
