from __future__ import annotations

import os
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, UploadFile

from src.multidal.api.schemas import IngestResponse
from src.multidal.config import settings
from src.multidal.db.repository import create_task
from src.multidal.queue.producer import KafkaProducer

router = APIRouter()

UPLOAD_DIR = settings.project_root / "docs"


@router.post("/ingest", response_model=IngestResponse)
async def ingest(
    file: UploadFile = File(...),
    kb_id: str = Form("default"),
) -> IngestResponse:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    task_id = uuid.uuid4().hex[:12]
    ext = Path(file.filename or "doc.pdf").suffix
    saved_path = UPLOAD_DIR / f"{task_id}{ext}"

    content = await file.read()
    saved_path.write_bytes(content)

    task = create_task(
        filename=file.filename or "unknown",
        file_path=str(saved_path),
        kb_id=kb_id,
        file_size=len(content),
        task_id=task_id,
    )

    producer = KafkaProducer()
    producer.send_parse_request(
        task_id=task.task_id,
        file_path=str(saved_path),
        filename=task.filename,
        kb_id=kb_id,
    )

    return IngestResponse(task_id=task.task_id, status="pending")
