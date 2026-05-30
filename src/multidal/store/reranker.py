from __future__ import annotations

import logging

import requests

from src.multidal.config import settings
from src.multidal.schema.retrieval import RecallResult, RerankResult

logger = logging.getLogger(__name__)


class Reranker:
    """使用云端 rerank API（Moark sentence-similarity）对候选集精排。"""

    def __init__(self) -> None:
        self._api_base = settings.reranker_api_base
        self._model = settings.reranker_model
        self._api_key = settings.reranker_api_key

    def validate(self) -> bool:
        """检查 rerank API 是否可达。"""
        try:
            headers = {"Content-Type": "application/json"}
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"
            r = requests.post(
                f"{self._api_base}/sentence-similarity",
                json={
                    "model": self._model,
                    "inputs": {"source_sentence": "ping", "sentences": ["pong"]},
                    "normalize": True,
                },
                headers=headers,
                timeout=10,
            )
            return r.status_code == 200
        except Exception:
            return False

    def rerank(
        self, query: str, candidates: list[RecallResult], top_k: int = 5
    ) -> list[RerankResult]:
        if not candidates:
            return []

        text_cands = [c for c in candidates if c.modality == "text"]
        image_cands = [c for c in candidates if c.modality == "image"]

        scores_map: dict[str, float] = {}
        if text_cands:
            try:
                raw_scores = self._score(query, text_cands)
                for c, s in zip(text_cands, raw_scores):
                    scores_map[c.chunk_id] = float(s)
            except Exception:
                logger.exception("Reranker API failed for text candidates")
                for c in text_cands:
                    scores_map[c.chunk_id] = c.score

        for c in image_cands:
            scores_map[c.chunk_id] = c.score

        # 全部候选一起排序，图片按 CLIP 分数与文本竞争
        all_candidates = text_cands + image_cands
        all_candidates.sort(key=lambda x: scores_map.get(x.chunk_id, x.score), reverse=True)
        all_candidates = all_candidates[:top_k]

        return [
            RerankResult(
                chunk_id=c.chunk_id,
                content=c.content,
                modality=c.modality,
                score=scores_map.get(c.chunk_id, c.score),
                rank=i + 1,
                kb_id=c.kb_id,
                doc_id=c.doc_id,
                page=c.page,
                image_path=c.image_path,
            )
            for i, c in enumerate(all_candidates)
        ]

    def _score(self, query: str, candidates: list[RecallResult]) -> list[float]:
        """调用 Moark sentence-similarity API。"""
        docs = [c.content for c in candidates]
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        r = requests.post(
            f"{self._api_base}/sentence-similarity",
            json={
                "model": self._model,
                "inputs": {
                    "source_sentence": query,
                    "sentences": docs,
                },
                "normalize": True,
            },
            headers=headers,
            timeout=60,
        )
        r.raise_for_status()
        data = r.json()

        if isinstance(data, list):
            if all(isinstance(x, (int, float)) for x in data):
                return data
            if data and isinstance(data[0], dict):
                return [it.get("score", it.get("relevance_score", 0.0)) for it in data]
        if isinstance(data, dict):
            if "scores" in data:
                return data["scores"]
            if "data" in data:
                items = data["data"]
                if isinstance(items, list):
                    if items and isinstance(items[0], dict):
                        return [it.get("score", it.get("relevance_score", 0.0)) for it in items]
                    return items
            if "results" in data:
                items = sorted(data["results"], key=lambda x: x.get("index", 0))
                return [it.get("relevance_score", it.get("score", 0.0)) for it in items]

        logger.warning("Reranker: unknown response format: %s", type(data))
        return [0.0] * len(candidates)