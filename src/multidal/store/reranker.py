"""云端 Rerank 精排器。

本模块实现 ``Reranker``,基于 Moark sentence-similarity API 对候选集做精排。
文本候选与图片候选分开处理:文本走 rerank 精排,图片沿用 CLIP 相似度,
最后统一排序,允许图片与文本在同一榜单中竞争。

设计要点:
    - 文本/图片分开 rerank:Rerank API 只能处理文本,图片直接用原 CLIP 分;
    - 降级策略:Rerank API 失败时,文本回退到召回分(不阻断整体);
    - 统一截断:全部候选一起按精排分排序,取 top_k。
"""

from __future__ import annotations

import json
import logging

import requests

from src.multidal.config import settings
from src.multidal.schema.retrieval import RecallResult, RerankResult

logger = logging.getLogger(__name__)


class Reranker:
    """使用云端 rerank API(Moark sentence-similarity)对候选集精排。

    Attributes:
        _api_base: rerank API 根地址。
        _model: rerank 模型名称(默认 BCE reranker base v1)。
        _api_key: Bearer Token(部分本地部署可能无 key,允许为空)。
    """

    def __init__(self) -> None:
        """从全局配置读取 rerank API 凭据。"""
        self._api_base = settings.reranker_api_base
        self._model = settings.reranker_model
        self._api_key = settings.reranker_api_key

    def validate(self) -> bool:
        """通过最小请求("ping" vs "pong")探测 API 可用性。

        Returns:
            bool: HTTP 200 时为 True;网络异常或非 200 一律 False。
        """
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
        """对候选集精排,返回 top_k。

        Args:
            query: 原始查询文本。
            candidates: 多路召回的候选集(文本 + 图片混合)。
            top_k: 最终保留的结果数(默认 5)。

        Returns:
            list[RerankResult]: 按精排分降序,带 rank 编号的结果列表。
        """
        if not candidates:
            return []

        # 文本/图片分桶:Rerank 模型只能处理文本,图片直接保留原 CLIP 分
        text_cands = [c for c in candidates if c.modality == "text"]
        image_cands = [c for c in candidates if c.modality == "image"]

        scores_map: dict[str, float] = {}
        if text_cands:
            try:
                raw_scores = self._score(query, text_cands)
                for c, s in zip(text_cands, raw_scores):
                    scores_map[c.chunk_id] = float(s)
            except Exception:
                # Rerank 失败时降级:文本回退到召回分,保证整体流程不中断
                logger.exception("Reranker API failed for text candidates")
                for c in text_cands:
                    scores_map[c.chunk_id] = c.score

        # 图片候选无精排,沿用召回分
        for c in image_cands:
            scores_map[c.chunk_id] = c.score

        # 全部候选一起排序,图片按 CLIP 分数与文本竞争
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
        """调用 Moark sentence-similarity API,兼容多种返回结构。

        兼容的响应格式:
            - 顶层 list[float]:直接返回;
            - 顶层 list[dict]:取 ``score`` 或 ``relevance_score`` 字段;
            - 顶层 dict 含 ``scores``:取 ``scores`` 字段;
            - 顶层 dict 含 ``data``:取 ``data`` 内的 score 字段;
            - 顶层 dict 含 ``results``:按 ``index`` 排序后取 score 字段。

        Args:
            query: 查询文本。
            candidates: 文本候选(图片候选不调用本方法)。

        Returns:
            list[float]: 与 candidates 一一对应的分值列表。

        Raises:
            requests.HTTPError: API 返回非 2xx 时(由调用方捕获并降级)。
        """
        docs = [c.content for c in candidates]
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        # Moark sentence-similarity API 限制单次最多 25 条；
        # 多 query 重写（最多 3 个子问题）会把候选堆到 ~30，超过限制就 400。
        # 分批调用，最后把每批的分数按原 candidates 顺序拼接。
        BATCH_SIZE = 25
        all_scores: list[float] = []
        for i in range(0, len(docs), BATCH_SIZE):
            batch = docs[i : i + BATCH_SIZE]
            r = requests.post(
                f"{self._api_base}/sentence-similarity",
                json={
                    "model": self._model,
                    "inputs": {
                        "source_sentence": query,
                        "sentences": batch,
                    },
                    "normalize": True,
                },
                headers=headers,
                timeout=60,
            )
            if not r.ok:
                logger.error(
                    "Reranker batch %d/%d failed: query=%s | response: %s",
                    i // BATCH_SIZE + 1,
                    (len(docs) + BATCH_SIZE - 1) // BATCH_SIZE,
                    query[:120],
                    r.text[:500],
                )
                raise requests.HTTPError(
                    f"reranker batch failed status={r.status_code}"
                )
            batch_scores = self._parse_scores(r.json())
            # 防御：分批返回的长度必须等于本批 candidates 数
            if len(batch_scores) != len(batch):
                logger.warning(
                    "Reranker batch returned %d scores for %d docs, padding with 0",
                    len(batch_scores), len(batch),
                )
                batch_scores = (batch_scores + [0.0] * len(batch))[: len(batch)]
            all_scores.extend(batch_scores)
        return all_scores

    @staticmethod
    def _parse_scores(data) -> list[float]:
        """兼容多种 sentence-similarity 返回结构。"""
        if isinstance(data, list):
            if all(isinstance(x, (int, float)) for x in data):
                return list(data)
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
                # results 可能乱序,按 index 还原
                items = sorted(data["results"], key=lambda x: x.get("index", 0))
                return [it.get("relevance_score", it.get("score", 0.0)) for it in items]

        # 未知格式:返回全 0，由上层做防御性 padding
        logger.warning("Reranker: unknown response format: %s", type(data))
        return []
