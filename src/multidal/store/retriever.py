from __future__ import annotations

import logging

from src.multidal.schema.retrieval import RecallResult
from src.multidal.store.milvus_store import MilvusStore

logger = logging.getLogger(__name__)


class MultiPathRetriever:
    """双语义空间召回：
    - text_vec（bge 空间）→ 搜 {kb}_text  → 文本 chunks
    - image_vec（CLIP 空间）→ 搜 {kb}_image → 图片 chunks
    """

    def __init__(self, store: MilvusStore):
        self._store = store

    def recall(
        self,
        text_vec: list[float],
        image_vec: list[float],
        kb_ids: list[str],
        top_k: int = 10,
    ) -> list[RecallResult]:
        results: list[RecallResult] = []

        for kb_id in kb_ids:
            # 文本路径：bge 向量 → 搜 text collection
            try:
                results.extend(
                    self._store.search(f"{kb_id}_text", text_vec, top_k=top_k)
                )
            except Exception:
                logger.warning("Text search failed for %s_text", kb_id)

            # 图片路径：CLIP 向量 → 搜 image collection
            try:
                results.extend(
                    self._store.search(f"{kb_id}_image", image_vec, top_k=top_k)
                )
            except Exception:
                logger.warning("Image search failed for %s_image", kb_id)

        # 按 kb_id + chunk_id 去重，保留最高分
        seen: dict[str, RecallResult] = {}
        for r in results:
            key = f"{r.kb_id}:{r.chunk_id}"
            if key not in seen or r.score > seen[key].score:
                seen[key] = r

        deduped = sorted(seen.values(), key=lambda x: x.score, reverse=True)
        for r in deduped:
            r.source = "multi_path"

        return deduped
