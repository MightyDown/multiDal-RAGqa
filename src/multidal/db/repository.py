from __future__ import annotations

import uuid
from datetime import datetime
from typing import Generator

from sqlalchemy.orm import Session

from src.multidal.db.models import KnowledgeBaseModel, ParseTaskModel, SessionLocal
from src.multidal.schema.task import TaskStatus


def get_session() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ── ParseTask ──────────────────────────────────────────────

def create_task(
    filename: str,
    file_path: str,
    kb_id: str,
    file_size: int = 0,
    task_id: str = "",
    session: Session | None = None,
) -> ParseTaskModel:
    task = ParseTaskModel(
        task_id=task_id or uuid.uuid4().hex[:12],
        filename=filename,
        file_path=file_path,
        file_size=file_size,
        kb_id=kb_id,
        status=TaskStatus.PENDING.value,
    )
    if session:
        session.add(task)
        session.commit()
        session.refresh(task)
    else:
        with SessionLocal() as s:
            s.add(task)
            s.commit()
    return task


def get_task(task_id: str) -> ParseTaskModel | None:
    with SessionLocal() as s:
        return s.query(ParseTaskModel).filter(ParseTaskModel.task_id == task_id).first()


def delete_task(task_id: str) -> ParseTaskModel | None:
    with SessionLocal() as s:
        task = s.query(ParseTaskModel).filter(ParseTaskModel.task_id == task_id).first()
        if not task:
            return None
        s.delete(task)
        s.commit()
        return task


def update_task(
    task_id: str,
    status: str | None = None,
    stage: str | None = None,
    error_message: str | None = None,
    full_text: str | None = None,
    retry_count: int | None = None,
    page_count: int | None = None,
) -> ParseTaskModel | None:
    with SessionLocal() as s:
        task = s.query(ParseTaskModel).filter(ParseTaskModel.task_id == task_id).first()
        if not task:
            return None
        if status is not None:
            task.status = status
        if stage is not None:
            task.stage = stage
        if error_message is not None:
            task.error_message = error_message
        if full_text is not None:
            task.full_text = full_text
        if retry_count is not None:
            task.retry_count = retry_count
        if page_count is not None:
            task.page_count = page_count
        task.updated_at = datetime.utcnow()
        s.commit()
        s.refresh(task)
        return task


def list_tasks(kb_id: str | None = None, status: str | None = None) -> list[ParseTaskModel]:
    with SessionLocal() as s:
        q = s.query(ParseTaskModel)
        if kb_id:
            q = q.filter(ParseTaskModel.kb_id == kb_id)
        if status:
            q = q.filter(ParseTaskModel.status == status)
        return q.order_by(ParseTaskModel.created_at.desc()).all()


# ── KnowledgeBase ──────────────────────────────────────────

def create_kb(name: str, description: str = "") -> KnowledgeBaseModel:
    kb_id = "kb_" + uuid.uuid4().hex[:8]
    kb = KnowledgeBaseModel(
        kb_id=kb_id,
        name=name,
        description=description,
        text_collection=f"{kb_id}_text",
        image_collection=f"{kb_id}_image",
    )
    with SessionLocal() as s:
        s.add(kb)
        s.commit()
        s.refresh(kb)
    return kb


def get_kb(kb_id: str) -> KnowledgeBaseModel | None:
    with SessionLocal() as s:
        return s.query(KnowledgeBaseModel).filter(KnowledgeBaseModel.kb_id == kb_id).first()


def list_kbs() -> list[KnowledgeBaseModel]:
    with SessionLocal() as s:
        return s.query(KnowledgeBaseModel).order_by(KnowledgeBaseModel.created_at.desc()).all()


def delete_kb(kb_id: str) -> bool:
    with SessionLocal() as s:
        kb = s.query(KnowledgeBaseModel).filter(KnowledgeBaseModel.kb_id == kb_id).first()
        if not kb:
            return False
        s.delete(kb)
        s.commit()
        return True


def count_docs_in_kb(kb_id: str) -> int:
    with SessionLocal() as s:
        return s.query(ParseTaskModel).filter(
            ParseTaskModel.kb_id == kb_id,
            ParseTaskModel.status == TaskStatus.COMPLETED.value,
        ).count()
