import pytest

from src.multidal.pipeline.base import PipelineContext, Stage
from src.multidal.pipeline.orchestrator import Orchestrator


class _FakeStage(Stage):
    def __init__(self, name: str, valid: bool = True, fail_on_process: bool = False):
        self.name = name
        self._valid = valid
        self._fail = fail_on_process
        self.processed = False

    def validate(self) -> bool:
        return self._valid

    def process(self, ctx: PipelineContext) -> PipelineContext:
        if self._fail:
            raise ValueError(f"{self.name} failed")
        self.processed = True
        return ctx


class TestOrchestrator:
    def test_empty_stages(self):
        orch = Orchestrator([])
        ctx = PipelineContext(task_id="t1")
        result = orch.run(ctx)
        assert result is ctx

    def test_single_stage(self):
        s = _FakeStage("parser")
        orch = Orchestrator([s])
        orch.run(PipelineContext(task_id="t1"))
        assert s.processed

    def test_multiple_stages(self):
        s1 = _FakeStage("parser")
        s2 = _FakeStage("embedder")
        orch = Orchestrator([s1, s2])
        orch.run(PipelineContext(task_id="t1"))
        assert s1.processed and s2.processed

    def test_validate_fails_raises(self):
        orch = Orchestrator([_FakeStage("bad", valid=False)])
        with pytest.raises(RuntimeError, match="validate failed"):
            orch.run(PipelineContext(task_id="t1"))

    def test_process_failure_propagates(self):
        orch = Orchestrator([_FakeStage("crash", fail_on_process=True)])
        with pytest.raises(ValueError, match="crash failed"):
            orch.run(PipelineContext(task_id="t1"))

    def test_context_flows_between_stages(self):
        class AccumStage(Stage):
            name = "accum"
            def validate(self) -> bool: return True
            def process(self, ctx: PipelineContext) -> PipelineContext:
                ctx.meta.setdefault("visited", []).append(self.name)
                return ctx

        orch = Orchestrator([AccumStage(), AccumStage()])
        result = orch.run(PipelineContext())
        assert result.meta["visited"] == ["accum", "accum"]
