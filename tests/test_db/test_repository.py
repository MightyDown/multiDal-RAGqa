import tempfile
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.multidal.db.models import Base, KnowledgeBaseModel, ParseTaskModel
from src.multidal.db.repository import (
    count_docs_in_kb,
    create_kb,
    create_task,
    delete_kb,
    get_kb,
    get_task,
    list_kbs,
    list_tasks,
    update_task,
)
from src.multidal.schema.task import TaskStatus


@pytest.fixture
def db():
    """创建临时 SQLite 数据库。"""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, expire_on_commit=False)

    import src.multidal.db.repository as repo
    _orig = repo.SessionLocal
    repo.SessionLocal = Session
    yield Session
    repo.SessionLocal = _orig
    Base.metadata.drop_all(engine)


class TestParseTask:
    def test_create_task(self, db):
        task = create_task("a.pdf", "/tmp/a.pdf", "kb1", file_size=1024)
        assert task.task_id
        assert task.filename == "a.pdf"
        assert task.status == TaskStatus.PENDING.value
        assert task.retry_count == 0

    def test_get_task(self, db):
        task = create_task("b.pdf", "/tmp/b.pdf", "kb1")
        fetched = get_task(task.task_id)
        assert fetched is not None
        assert fetched.filename == "b.pdf"

    def test_get_task_not_found(self, db):
        assert get_task("nonexistent") is None

    def test_update_task_status(self, db):
        task = create_task("c.pdf", "/tmp/c.pdf", "kb1")
        updated = update_task(task.task_id, status=TaskStatus.PROCESSING.value, stage="parser")
        assert updated is not None
        assert updated.status == TaskStatus.PROCESSING.value
        assert updated.stage == "parser"

    def test_update_task_error(self, db):
        task = create_task("d.pdf", "/tmp/d.pdf", "kb1")
        updated = update_task(
            task.task_id,
            status=TaskStatus.FAILED.value,
            error_message="OOM",
            retry_count=1,
        )
        assert updated is not None
        assert updated.error_message == "OOM"
        assert updated.retry_count == 1

    def test_update_task_not_found(self, db):
        assert update_task("nope", status="completed") is None

    def test_list_tasks(self, db):
        create_task("a.pdf", "/tmp/a.pdf", "kb1")
        create_task("b.pdf", "/tmp/b.pdf", "kb2")
        create_task("c.pdf", "/tmp/c.pdf", "kb1")
        all_tasks = list_tasks()
        assert len(all_tasks) == 3
        kb1_tasks = list_tasks(kb_id="kb1")
        assert len(kb1_tasks) == 2

    def test_list_tasks_by_status(self, db):
        t = create_task("a.pdf", "/tmp/a.pdf", "kb1")
        update_task(t.task_id, status=TaskStatus.COMPLETED.value)
        assert len(list_tasks(status=TaskStatus.COMPLETED.value)) == 1
        assert len(list_tasks(status=TaskStatus.PENDING.value)) == 0


class TestKnowledgeBase:
    def test_create_and_get_kb(self, db):
        kb = create_kb("财务报告", "公司年报")
        assert kb.kb_id.startswith("kb_")
        assert kb.name == "财务报告"
        assert kb.text_collection == f"{kb.kb_id}_text"
        assert kb.image_collection == f"{kb.kb_id}_image"

        fetched = get_kb(kb.kb_id)
        assert fetched is not None
        assert fetched.name == "财务报告"

    def test_get_kb_not_found(self, db):
        assert get_kb("nonexistent") is None

    def test_list_kbs(self, db):
        create_kb("kb1")
        create_kb("kb2")
        kbs = list_kbs()
        assert len(kbs) == 2

    def test_delete_kb(self, db):
        kb = create_kb("to_delete")
        assert delete_kb(kb.kb_id) is True
        assert get_kb(kb.kb_id) is None

    def test_delete_kb_not_found(self, db):
        assert delete_kb("nope") is False

    def test_count_docs_in_kb(self, db):
        kb = create_kb("test_kb")
        assert count_docs_in_kb(kb.kb_id) == 0
        t = create_task("a.pdf", "/tmp/a.pdf", kb.kb_id)
        update_task(t.task_id, status=TaskStatus.COMPLETED.value)
        assert count_docs_in_kb(kb.kb_id) == 1
