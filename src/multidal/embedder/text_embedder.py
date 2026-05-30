from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from src.multidal.config import settings
from src.multidal.pipeline.base import PipelineContext, Stage
from src.multidal.schema.embedding import EmbeddedChunk, Embedding

logger = logging.getLogger(__name__)

BATCH_SIZE = 20


class TextEmbedder(Stage):
    """通过 OpenAI-compatible API 做文本向量化（云端批量并发）。"""

    name = "embedder_text"

    def __init__(self) -> None:
        self._api_base = settings.text_embedding_api_base
        self._model = settings.text_embedding_model
        self._dim = settings.text_embedding_dim
        self._key = settings.text_embedding_api_key

    def validate(self) -> bool:
        try:
            r = requests.post(
                f"{self._api_base}/embeddings",
                json={"model": self._model, "input": "ping"},
                headers={"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"},
                timeout=10,
            )
            return r.status_code == 200
        except Exception:
            logger.warning("Text embedding API not reachable at %s", self._api_base)
            return False

    def process(self, ctx: PipelineContext) -> PipelineContext:
        if ctx.parsed is None:
            raise ValueError("PipelineContext.parsed is None, run Parser first")

        texts = [tc.content for tc in ctx.parsed.text_chunks]
        vecs = self._embed_batch(texts)

        chunks = []
        for tc, vec in zip(ctx.parsed.text_chunks, vecs):
            chunks.append(
                EmbeddedChunk(
                    chunk_id=tc.chunk_id,
                    content=tc.content,
                    embedding=vec,
                    modality="text",
                    kb_id=ctx.kb_id,
                    doc_id=ctx.parsed.doc_id,
                    page=tc.page,
                )
            )
        ctx.embedded = (ctx.embedded or []) + chunks
        logger.info("Text embedder: %d chunks", len(chunks))
        return ctx

    def embed_query(self, text: str) -> list[float]:
        """公开方法：嵌入查询文本，返回纯向量列表供检索使用。"""
        r = requests.post(
            f"{self._api_base}/embeddings",
            json={"model": self._model, "input": text},
            headers={"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()["data"][0]["embedding"]

    def _embed_batch(self, texts: list[str]) -> list[Embedding]:
        batches = [texts[i : i + BATCH_SIZE] for i in range(0, len(texts), BATCH_SIZE)]
        logger.info("Embedding %d chunks in %d batches (batch_size=%d)", len(texts), len(batches), BATCH_SIZE)

        results: dict[int, list[Embedding]] = {}

        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {
                pool.submit(_do_batch, self._api_base, self._model, self._key, i, b): i
                for i, b in enumerate(batches)
            }
            for f in as_completed(futures):
                idx, embeds = f.result()
                results[idx] = embeds

        all_vecs: list[Embedding] = []
        for i in range(len(batches)):
            all_vecs.extend(results[i])
        return all_vecs


def _do_batch(
    api_base: str, model: str, api_key: str, idx: int, batch: list[str]
) -> tuple[int, list[Embedding]]:
    r = requests.post(
        f"{api_base}/embeddings",
        json={"model": model, "input": batch},
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        timeout=120,
    )
    if not r.ok:
        logger.error("Batch %d API error %d: %s", idx, r.status_code, r.text[:500])
    r.raise_for_status()
    data = r.json()
    return idx, [Embedding(model_name=model, dim=len(item["embedding"]), vector=item["embedding"]) for item in data["data"]]
