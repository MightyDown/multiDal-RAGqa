"""双语义空间召回器。

本模块实现 ``MultiPathRetriever``,对多个 KB 同时做"文本路径 + 图片路径"
两路召回,并对结果去重排序。

召回流程:
    对每个 kb_id:
        1. 用 BGE 文本向量检索 ``{kb}_text`` Collection;
        2. 用 CLIP 图片向量检索 ``{kb}_image`` Collection;
        3. 单条路径失败不中断整体,只记录 warning。
    最后:按 ``kb_id:chunk_id`` 去重,保留最高分,按 score 降序输出。
"""

from __future__ import annotations

import logging

from src.multidal.schema.retrieval import RecallResult
from src.multidal.store.milvus_store import MilvusStore

logger = logging.getLogger(__name__)


class MultiPathRetriever:
    """双语义空间召回器(BGE 文本 + CLIP 图片)。

    Attributes:
        _store: 底层向量存储(当前固定为 ``MilvusStore``,理论上可换成 ChromaStore)。
    """

    def __init__(self, store: MilvusStore):
        """注入底层存储。

        Args:
            store: ``MilvusStore`` 实例(或任何实现了 ``search`` 方法的存储)。
        """
        self._store = store

    def recall(
        self,
        text_vec: list[float],
        image_vec: list[float],
        kb_ids: list[str],
        top_k: int = 10,
    ) -> list[RecallResult]:
        """跨多 KB 的双路召回,合并去重后返回。

        Args:
            text_vec: BGE 文本向量(用于搜 ``_text`` Collection)。
            image_vec: CLIP 图片向量(用于搜 ``_image`` Collection)。
            kb_ids: 目标 KB 列表(逐个做双路)。
            top_k: 单 KB 单路径的返回上限,最终结果可能多于该值(多 KB 累加)。

        Returns:
            list[RecallResult]: 去重 + 按相似度降序的候选集。
        """
        results: list[RecallResult] = []

        for kb_id in kb_ids:
            # 文本路径:bge 向量 → 搜 text collection
            try:
                results.extend(
                    self._store.search(f"{kb_id}_text", text_vec, top_k=top_k)
                )
            except Exception:
                # 单 KB 失败不阻塞其他 KB,只记 warning
                logger.warning("Text search failed for %s_text", kb_id)

            # 图片路径:CLIP 向量 → 搜 image collection
            try:
                results.extend(
                    self._store.search(f"{kb_id}_image", image_vec, top_k=top_k)
                )
            except Exception:
                logger.warning("Image search failed for %s_image", kb_id)

        # 按 kb_id + chunk_id 去重,保留最高分
        # 用 dict 而非 set:因为要按 score 比较(同 chunk 可能来自多条路径,分数不同)
        seen: dict[str, RecallResult] = {}
        for r in results:
            key = f"{r.kb_id}:{r.chunk_id}"
            if key not in seen or r.score > seen[key].score:
                seen[key] = r

        # 全局按相似度降序
        deduped = sorted(seen.values(), key=lambda x: x.score, reverse=True)
        # 标记来源便于上层(例如日志/分析)识别
        for r in deduped:
            r.source = "multi_path"

        return deduped
