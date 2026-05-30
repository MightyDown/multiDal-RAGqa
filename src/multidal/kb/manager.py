from __future__ import annotations

import logging

from src.multidal.db.repository import count_docs_in_kb, create_kb, delete_kb, list_kbs
from src.multidal.schema.kb import KBListResponse, KBResponse

logger = logging.getLogger(__name__)


class KBManager:
    def create(self, name: str, description: str = "") -> KBResponse:
        kb = create_kb(name, description)
        return KBResponse(kb_id=kb.kb_id, name=kb.name, description=kb.description)

    def list_all(self) -> KBListResponse:
        kbs = []
        for kb in list_kbs():
            kbs.append(
                KBResponse(
                    kb_id=kb.kb_id,
                    name=kb.name,
                    description=kb.description,
                    doc_count=count_docs_in_kb(kb.kb_id),
                )
            )
        return KBListResponse(kbs=kbs, total=len(kbs))

    def list_all_ids(self) -> list[str]:
        return [kb.kb_id for kb in list_kbs()]

    def delete(self, kb_id: str) -> bool:
        return delete_kb(kb_id)
