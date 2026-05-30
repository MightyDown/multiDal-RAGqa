from __future__ import annotations

import logging

from src.multidal.pipeline.base import PipelineContext, Stage

logger = logging.getLogger(__name__)


class Orchestrator:
    """顺序执行流水线阶段，传递 PipelineContext。"""

    def __init__(self, stages: list[Stage]):
        self.stages = stages

    def run(self, ctx: PipelineContext) -> PipelineContext:
        for stage in self.stages:
            logger.info(
                "[%s] stage=%s start", ctx.task_id, stage.name
            )
            if not stage.validate():
                raise RuntimeError(
                    f"[{ctx.task_id}] stage={stage.name} validate failed"
                )
            ctx = stage.process(ctx)
            logger.info(
                "[%s] stage=%s done", ctx.task_id, stage.name
            )
        return ctx
