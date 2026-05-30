from __future__ import annotations

import logging
import time

import requests
from fastapi import APIRouter
from pymilvus import connections as milvus_connections, utility as milvus_utility

from src.multidal.config import settings
from src.multidal.embedder.text_embedder import TextEmbedder
from src.multidal.store.reranker import Reranker

logger = logging.getLogger(__name__)

router = APIRouter()

_embedder = TextEmbedder()
_reranker = Reranker()


def _check_milvus() -> dict:
    try:
        # 用别名避免重复连接
        alias = f"health_{int(time.time())}"
        milvus_connections.connect(
            alias=alias,
            host=settings.milvus_host,
            port=settings.milvus_port,
            timeout=5,
        )
        ok = milvus_utility.get_server_version(using=alias) is not None
        milvus_connections.disconnect(alias)
        return {"ok": ok, "detail": settings.milvus_host}
    except Exception as e:
        return {"ok": False, "detail": str(e)[:200]}


def _check_http(name: str, url: str, timeout: float = 5) -> dict:
    try:
        r = requests.get(url, timeout=timeout)
        return {"ok": r.status_code < 500, "detail": f"{r.status_code}"}
    except Exception as e:
        return {"ok": False, "detail": str(e)[:200]}


def _check_kafka() -> dict:
    try:
        from confluent_kafka import Consumer
        c = Consumer({
            "bootstrap.servers": settings.kafka_bootstrap_servers,
            "group.id": f"health_{int(time.time())}",
            "enable.auto.commit": False,
            "session.timeout.ms": 6000,
        })
        # 获取集群元数据即表示连接成功
        meta = c.list_topics(timeout=5)
        c.close()
        return {"ok": True, "detail": f"{len(meta.topics)} topics"}
    except Exception as e:
        return {"ok": False, "detail": str(e)[:200]}


@router.get("/health")
async def health():
    """检查所有下游服务的连通性。"""
    results = {}

    # Milvus
    results["milvus"] = _check_milvus()

    # Kafka
    results["kafka"] = _check_kafka()

    # MinerU
    results["mineru"] = _check_http(
        "mineru",
        f"{settings.mineru_api_base}/api/v4/extract/task",
        timeout=8,
    )

    # Text Embedding
    results["embedding"] = {"ok": _embedder.validate(), "detail": settings.text_embedding_api_base}

    # Reranker
    results["reranker"] = {"ok": _reranker.validate(), "detail": settings.reranker_api_base}

    # LLM
    results["llm"] = _check_http(
        "llm",
        settings.llm_base_url.rstrip("/"),
        timeout=5,
    )

    all_ok = all(v["ok"] for v in results.values())
    return {"status": "healthy" if all_ok else "degraded", "components": results}
