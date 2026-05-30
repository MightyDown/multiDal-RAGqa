import pytest
from pydantic import ValidationError

from src.multidal.schema.task import ParseTask, TaskStage, TaskStatus


class TestTaskStatus:
    def test_values(self):
        assert TaskStatus.PENDING == "pending"
        assert TaskStatus.PROCESSING == "processing"
        assert TaskStatus.COMPLETED == "completed"
        assert TaskStatus.FAILED == "failed"
        assert TaskStatus.EXHAUSTED == "exhausted"


class TestTaskStage:
    def test_values(self):
        assert TaskStage.PARSER == "parser"
        assert TaskStage.EMBEDDER == "embedder"
        assert TaskStage.STORE == "store"


class TestParseTask:
    def test_defaults(self):
        task = ParseTask(task_id="t1", filename="f.pdf")
        assert task.status == TaskStatus.PENDING
        assert task.stage is None
        assert task.error_message == ""
        assert task.retry_count == 0
        assert task.max_retries == 3
        assert task.kb_id == ""

    def test_filename_min_length(self):
        with pytest.raises(ValidationError):
            ParseTask(task_id="t1", filename="")

    def test_retry_count_ge_0(self):
        with pytest.raises(ValidationError):
            ParseTask(task_id="t1", filename="f.pdf", retry_count=-1)

    def test_max_retries_ge_1(self):
        with pytest.raises(ValidationError):
            ParseTask(task_id="t1", filename="f.pdf", max_retries=0)
