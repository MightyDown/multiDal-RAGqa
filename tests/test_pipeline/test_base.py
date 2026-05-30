import pytest

from src.multidal.pipeline.base import PipelineContext, Stage


class TestPipelineContext:
    def test_defaults(self):
        ctx = PipelineContext()
        assert ctx.task_id == ""
        assert ctx.kb_id == ""
        assert ctx.file_path == ""
        assert ctx.filename == ""
        assert ctx.parsed is None
        assert ctx.embedded is None
        assert ctx.meta == {}

    def test_full_init(self):
        ctx = PipelineContext(
            task_id="t1", kb_id="kb1", file_path="/tmp/a.pdf", filename="a.pdf",
            meta={"pages": 10},
        )
        assert ctx.task_id == "t1"
        assert ctx.kb_id == "kb1"
        assert ctx.meta == {"pages": 10}

    def test_meta_mutable(self):
        ctx = PipelineContext()
        ctx.meta["key"] = "val"
        assert ctx.meta["key"] == "val"

    def test_stages_produce_output(self):
        ctx = PipelineContext(task_id="t1")
        ctx.parsed = {"text_chunks": []}
        ctx.embedded = [{"chunk_id": "c1"}]
        assert ctx.parsed is not None
        assert ctx.embedded is not None


class TestStage:
    def test_abstract(self):
        with pytest.raises(TypeError):
            Stage()

    def test_concrete_subclass(self):
        class MyStage(Stage):
            name = "my"
            def validate(self) -> bool:
                return True
            def process(self, ctx: PipelineContext) -> PipelineContext:
                ctx.meta["done"] = True
                return ctx

        stage = MyStage()
        assert stage.name == "my"
        assert stage.validate() is True
        ctx = PipelineContext()
        result = stage.process(ctx)
        assert result.meta["done"] is True

    def test_name_default(self):
        class NoNameStage(Stage):
            def validate(self) -> bool: return True
            def process(self, ctx): return ctx

        assert NoNameStage.name == ""
