from __future__ import annotations

from fastapi import APIRouter, HTTPException

from src.multidal.api.schemas import TaskStatusResponse
from src.multidal.db.repository import get_task

router = APIRouter()


@router.get("/ingest/{task_id}", response_model=TaskStatusResponse)
async def task_status(task_id: str) -> TaskStatusResponse:
    task = get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return TaskStatusResponse(
        task_id=task.task_id,
        filename=task.filename,
        status=task.status,
        stage=task.stage,
        error_message=task.error_message,
        retry_count=task.retry_count,
        max_retries=task.max_retries,
    )
