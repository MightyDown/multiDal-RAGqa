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

        try:
            scores = self._score(query, candidates)
        except Exception:
            logger.exception("Reranker API failed, falling back to recall scores")
            scores = [c.score for c in candidates]

        ranked = []
        for i, (c, s) in enumerate(zip(candidates, scores)):
            ranked.append(
                RerankResult(
                    chunk_id=c.chunk_id,
                    content=c.content,
                    modality=c.modality,
                    score=float(s),
                    rank=i + 1,
                    kb_id=c.kb_id,
                    doc_id=c.doc_id,
                    page=c.page,
                )
            )
        ranked.sort(key=lambda x: x.score, reverse=True)
        ranked = ranked[:top_k]
        for i, r in enumerate(ranked):
            r.rank = i + 1
        return ranked

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

        # 兼容多种返回格式
        # Moark API 直接返回 float 数组
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
