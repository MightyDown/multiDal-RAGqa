"""文本向量化器。

本模块实现 ``TextEmbedder``(继承自 ``Stage``),通过 OpenAI-compatible
``/embeddings`` 接口调用云端文本嵌入模型(默认 BGE Large zh v1.5)。

并发策略:
    - 将输入文本按 ``BATCH_SIZE`` 切片;
    - 使用 ``ThreadPoolExecutor`` 并发提交多个 batch 请求(max_workers=3);
    - 通过 ``as_completed`` 收集结果,按原 batch 索引还原顺序。
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from src.multidal.config import settings
from src.multidal.pipeline.base import PipelineContext, Stage
from src.multidal.schema.embedding import EmbeddedChunk, Embedding

logger = logging.getLogger(__name__)

# 单次 API 请求最多提交的文本数;超过会被截断或失败,需根据模型 QPS 调整。
BATCH_SIZE = 20


class TextEmbedder(Stage):
    """通过 OpenAI-compatible API 做文本向量化(云端批量并发)。

    Attributes:
        name: 阶段名,固定为 ``"embedder_text"``。
        _api_base: 嵌入 API 根地址(从配置读取)。
        _model: 模型名称(从配置读取,默认 BGE 大模型)。
        _dim: 预期向量维度(从配置读取,仅用于日志/校验)。
        _key: API Key(Bearer Token)。
    """

    name = "embedder_text"

    def __init__(self) -> None:
        """从全局配置读取 API 凭据与模型信息。"""
        self._api_base = settings.text_embedding_api_base
        self._model = settings.text_embedding_model
        self._dim = settings.text_embedding_dim
        self._key = settings.text_embedding_api_key

    def validate(self) -> bool:
        """探测 API 连通性:发送一个 ``"ping"`` 文本,期望 200。

        Returns:
            bool: API 可达且返回 200 时为 True。
        """
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
        """对 ``ctx.parsed.text_chunks`` 全部文本块做向量化。

        步骤:
            1. 校验 ``ctx.parsed`` 已就绪;
            2. 提取所有文本块内容;
            3. 调用 ``_embed_batch`` 批量嵌入(并发);
            4. 还原顺序,与 ``TextChunk`` 一一对应,构造 ``EmbeddedChunk``;
            5. 追加到 ``ctx.embedded``(保留 ImageEmbedder 之前的产出)。

        Args:
            ctx: 流水线上下文,需含 ``parsed`` 与 ``kb_id``。

        Returns:
            PipelineContext: 已填充 ``embedded`` 的上下文。

        Raises:
            ValueError: ``ctx.parsed`` 为 None 时。
        """
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
        # 累加而非覆盖:ImageEmbedder 可能先于本阶段执行(取决于 Orchestrator 顺序)
        ctx.embedded = (ctx.embedded or []) + chunks
        logger.info("Text embedder: %d chunks", len(chunks))
        return ctx

    def embed_query(self, text: str) -> list[float]:
        """公开方法:将查询文本编码为纯向量,供检索阶段调用。

        Args:
            text: 查询文本(用户问题或改写后的问题)。

        Returns:
            list[float]: 向量列表,长度等于模型维度。

        Raises:
            requests.HTTPError: API 返回非 2xx 时。
        """
        # Moark API 在容器内偶发 30s+ 才建立连接，使用更长 timeout。
        # 增加 retry 减少偶发 timeout 引起的整链路失败。
        import time as _t
        last_err = None
        for attempt in range(3):
            try:
                r = requests.post(
                    f"{self._api_base}/embeddings",
                    json={"model": self._model, "input": text},
                    headers={"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"},
                    timeout=(10, 60),  # connect 10s, read 60s
                )
                r.raise_for_status()
                return r.json()["data"][0]["embedding"]
            except Exception as e:
                last_err = e
                logger.warning("embed_query attempt %d failed: %s", attempt + 1, str(e)[:120])
                _t.sleep(1 + attempt * 2)
        raise last_err

    def _embed_batch(self, texts: list[str]) -> list[Embedding]:
        """将文本列表分批并发嵌入,按原顺序返回结果。

        流程:
            1. 切片为多个 batch;
            2. 提交到线程池(3 并发);
            3. 用 ``as_completed`` 收集,按 batch 索引放入 ``results``;
            4. 顺序遍历 batch 索引,拼接结果,确保与输入顺序一致。

        Args:
            texts: 文本列表。

        Returns:
            list[Embedding]: 与 ``texts`` 一一对应的 ``Embedding`` 列表。
        """
        batches = [texts[i : i + BATCH_SIZE] for i in range(0, len(texts), BATCH_SIZE)]
        logger.info("Embedding %d chunks in %d batches (batch_size=%d)", len(texts), len(batches), BATCH_SIZE)

        # 用 dict 而非 list 收集:多线程完成顺序不可预测,需按 idx 还原
        results: dict[int, list[Embedding]] = {}

        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {
                pool.submit(_do_batch, self._api_base, self._model, self._key, i, b): i
                for i, b in enumerate(batches)
            }
            for f in as_completed(futures):
                idx, embeds = f.result()
                results[idx] = embeds

        # 按原 batch 顺序拼接,保证最终顺序与 texts 一致
        all_vecs: list[Embedding] = []
        for i in range(len(batches)):
            all_vecs.extend(results[i])
        return all_vecs


def _do_batch(
    api_base: str, model: str, api_key: str, idx: int, batch: list[str]
) -> tuple[int, list[Embedding]]:
    """线程池工作函数:执行单个 batch 的嵌入请求。

    Args:
        api_base: API 根地址。
        model: 模型名称。
        api_key: Bearer Token。
        idx: batch 索引(用于回调时定位)。
        batch: 当前 batch 的文本列表。

    Returns:
        tuple[int, list[Embedding]]: ``(batch_idx, embedding 列表)``。
    """
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
