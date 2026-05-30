from __future__ import annotations

from fastapi import APIRouter, HTTPException

from src.multidal.api.schemas import KBCreateRequest, KBResponse
from src.multidal.kb.manager import KBManager

router = APIRouter()


@router.post("/kb/create", response_model=KBResponse)
async def create_kb(req: KBCreateRequest) -> KBResponse:
    mgr = KBManager()
    kb = mgr.create(req.name, req.description)
    return KBResponse(kb_id=kb.kb_id, name=kb.name, description=kb.description)


@router.get("/kb/list")
async def list_kbs():
    mgr = KBManager()
    return mgr.list_all()


@router.delete("/kb/{kb_id}")
async def delete_kb(kb_id: str) -> dict:
    mgr = KBManager()
    ok = mgr.delete(kb_id)
    if not ok:
        raise HTTPException(status_code=404, detail="KB not found")
    return {"status": "deleted", "kb_id": kb_id}
