"""ChromaDB vector store — lightweight alternative to Milvus for dev.

Uses ChromaDB's persistent client with local storage. No Docker needed.
Memory ~150MB vs Milvus ~2-4GB.
"""

from __future__ import annotations

import logging
import uuid

import chromadb
from chromadb.config import Settings as ChromaSettings

from src.multidal.config import settings
from src.multidal.pipeline.base import PipelineContext, Stage
from src.multidal.schema.embedding import EmbeddedChunk
from src.multidal.schema.retrieval import RecallResult
from src.multidal.store.base import VectorStore

logger = logging.getLogger(__name__)


class ChromaStore(Stage, VectorStore):
    name = "store"

    def __init__(self, persist_dir: str = "data/chroma") -> None:
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

    def validate(self) -> bool:
        try:
            _ = self._client.heartbeat()
            return True
        except Exception:
            return False

    def process(self, ctx: PipelineContext) -> PipelineContext:
        if ctx.embedded is None:
            raise ValueError("PipelineContext.embedded is None")
        self.insert(ctx.embedded, ctx.kb_id)
        logger.info("ChromaDB stored %d chunks into kb=%s", len(ctx.embedded), ctx.kb_id)
        return ctx

    def insert(self, chunks: list[EmbeddedChunk], kb_id: str) -> list[str]:
        text_coll = self._ensure_collection(f"{kb_id}_text")
        image_coll = self._ensure_collection(f"{kb_id}_image")

        ids = []
        text_embeds, text_metas, text_docs, text_ids = [], [], [], []
        image_embeds, image_metas, image_docs, image_ids = [], [], [], []

        for c in chunks:
            cid = c.chunk_id or uuid.uuid4().hex[:8]
            meta = {
                "modality": c.modality,
                "kb_id": c.kb_id,
                "doc_id": c.doc_id,
                "page": c.page,
            }
            if c.modality == "text":
                text_embeds.append(c.embedding.vector)
                text_metas.append(meta)
                text_docs.append(c.content)
                text_ids.append(cid)
            else:
                image_embeds.append(c.embedding.vector)
                image_metas.append(meta)
                image_docs.append(c.content)
                image_ids.append(cid)
            ids.append(cid)

        if text_ids:
            text_coll.add(ids=text_ids, embeddings=text_embeds, metadatas=text_metas, documents=text_docs)
        if image_ids:
            image_coll.add(ids=image_ids, embeddings=image_embeds, metadatas=image_metas, documents=image_docs)

        return ids

    def search(
        self, collection_name: str, query_vector: list[float], top_k: int = 10
    ) -> list[RecallResult]:
        coll = self._client.get_collection(collection_name)
        results = coll.query(query_embeddings=[query_vector], n_results=top_k)
        out = []
        ids_list = results.get("ids", [[]])[0]
        metas_list = results.get("metadatas", [[]])[0] or []
        docs_list = results.get("documents", [[]])[0] or []
        distances_list = results.get("distances", [[]])[0] or []
        for i, cid in enumerate(ids_list):
            meta = metas_list[i] if i < len(metas_list) else {}
            doc = docs_list[i] if i < len(docs_list) else ""
            dist = distances_list[i] if i < len(distances_list) else 0.0
            out.append(RecallResult(
                chunk_id=cid,
                content=doc,
                modality=meta.get("modality", "text"),
                source="chromadb",
                score=float(1.0 / (1.0 + dist)),  # convert L2 distance to similarity
                kb_id=meta.get("kb_id", ""),
                doc_id=meta.get("doc_id", ""),
                page=meta.get("page", 1),
            ))
        return out

    def delete_collection(self, collection_name: str) -> None:
        try:
            self._client.delete_collection(collection_name)
        except Exception:
            pass

    def _ensure_collection(self, name: str) -> chromadb.Collection:
        try:
            return self._client.get_collection(name)
        except Exception:
            return self._client.create_collection(name, metadata={"hnsw:space": "cosine"})
