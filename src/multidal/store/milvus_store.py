from __future__ import annotations

import logging

from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility

from src.multidal.config import settings
from src.multidal.pipeline.base import PipelineContext, Stage
from src.multidal.schema.embedding import EmbeddedChunk
from src.multidal.schema.retrieval import RecallResult
from src.multidal.store.base import VectorStore

logger = logging.getLogger(__name__)

_PRIMARY = "chunk_id"
_VECTOR = "vector"


def _schema(dim: int) -> CollectionSchema:
    return CollectionSchema(
        fields=[
            FieldSchema(name=_PRIMARY, dtype=DataType.VARCHAR, is_primary=True, max_length=64),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=8192),
            FieldSchema(name="modality", dtype=DataType.VARCHAR, max_length=16),
            FieldSchema(name="kb_id", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="page", dtype=DataType.INT64),
            FieldSchema(name=_VECTOR, dtype=DataType.FLOAT_VECTOR, dim=dim),
        ]
    )


class MilvusStore(Stage, VectorStore):
    name = "store"

    def __init__(self) -> None:
        self._host = settings.milvus_host
        self._port = str(settings.milvus_port)
        self._connected = False
        self._dim_text = settings.text_embedding_dim
        self._dim_image = settings.image_embedding_dim

    def validate(self) -> bool:
        try:
            self._connect()
            return True
        except Exception:
            logger.warning("Milvus not reachable at %s:%s", self._host, self._port)
            return False

    def process(self, ctx: PipelineContext) -> PipelineContext:
        self._connect()
        if ctx.embedded is None:
            raise ValueError("PipelineContext.embedded is None")
        self.insert(ctx.embedded, ctx.kb_id)
        logger.info("Stored %d chunks into kb=%s", len(ctx.embedded), ctx.kb_id)
        return ctx

    def insert(self, chunks: list[EmbeddedChunk], kb_id: str) -> list[str]:
        text_coll = self._ensure_collection(f"{kb_id}_text", self._dim_text)
        image_coll = self._ensure_collection(f"{kb_id}_image", self._dim_image)

        text_rows, image_rows, ids = [], [], []
        for c in chunks:
            row = {
                _PRIMARY: c.chunk_id,
                "content": c.content[:8192],
                "modality": c.modality,
                "kb_id": c.kb_id,
                "doc_id": c.doc_id,
                "page": c.page,
                _VECTOR: c.embedding.vector,
            }
            if c.modality == "text":
                text_rows.append(row)
            else:
                image_rows.append(row)
            ids.append(c.chunk_id)

        if text_rows:
            text_coll.insert(text_rows)
            text_coll.flush()
        if image_rows:
            image_coll.insert(image_rows)
            image_coll.flush()
        return ids

    def search(
        self, collection_name: str, query_vector: list[float], top_k: int = 10
    ) -> list[RecallResult]:
        self._connect()
        coll = Collection(collection_name)
        coll.load()
        results = coll.search(
            data=[query_vector],
            anns_field=_VECTOR,
            param={"metric_type": "IP", "params": {"nprobe": 10}},
            limit=top_k,
            output_fields=["content", "modality", "kb_id", "doc_id", "page"],
        )
        out = []
        for hit in results[0]:
            out.append(
                RecallResult(
                    chunk_id=hit.id,
                    content=hit.entity.get("content", ""),
                    modality=hit.entity.get("modality", "text"),
                    source="milvus",
                    score=float(hit.distance),
                    kb_id=hit.entity.get("kb_id", ""),
                    doc_id=hit.entity.get("doc_id", ""),
                    page=hit.entity.get("page", 1),
                )
            )
        return out

    def query_by_doc(self, kb_id: str, doc_id: str, limit: int = 500) -> list[dict]:
        """按 doc_id 查询某个 KB 下所有 chunk（文本 + 图片）。"""
        self._connect()
        rows = []
        for suffix in ("_text", "_image"):
            name = f"{kb_id}{suffix}"
            if not utility.has_collection(name):
                continue
            coll = Collection(name)
            coll.load()
            try:
                results = coll.query(
                    expr=f'doc_id == "{doc_id}"',
                    output_fields=["chunk_id", "content", "modality", "kb_id", "doc_id", "page"],
                    limit=limit,
                )
                rows.extend(results)
            except Exception:
                logger.warning("query_by_doc failed for %s", name)
        rows.sort(key=lambda r: r.get("page", 1))
        return rows

    def delete_by_doc(self, kb_id: str, doc_id: str) -> int:
        """删除某个文档的所有向量。"""
        self._connect()
        deleted = 0
        for suffix in ("_text", "_image"):
            name = f"{kb_id}{suffix}"
            if not utility.has_collection(name):
                continue
            coll = Collection(name)
            coll.load()
            try:
                results = coll.query(
                    expr=f'doc_id == "{doc_id}"',
                    output_fields=["chunk_id"],
                    limit=10000,
                )
                ids = [r["chunk_id"] for r in results]
                if ids:
                    coll.delete(f'chunk_id in {ids}')
                    coll.flush()
                    deleted += len(ids)
            except Exception:
                logger.warning("delete_by_doc failed for %s", name)
        return deleted

    def delete_collection(self, collection_name: str) -> None:
        self._connect()
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)

    def _connect(self) -> None:
        if self._connected:
            return
        connections.connect(host=self._host, port=self._port)
        self._connected = True

    def _ensure_collection(self, name: str, dim: int) -> Collection:
        if utility.has_collection(name):
            return Collection(name)
        coll = Collection(name, schema=_schema(dim))
        coll.create_index(_VECTOR, {
            "metric_type": "IP", "index_type": "IVF_FLAT", "params": {"nlist": 128},
        })
        return coll
