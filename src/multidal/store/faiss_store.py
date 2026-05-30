"""FAISS-backed vector store — local replacement for Milvus.

Uses FAISS (already installed as pymilvus dependency) with on-disk persistence.
Swap back to MilvusStore by changing one import when Docker is available.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from pathlib import Path

import faiss
import numpy as np

from src.multidal.config import settings
from src.multidal.pipeline.base import PipelineContext, Stage
from src.multidal.schema.embedding import EmbeddedChunk
from src.multidal.schema.retrieval import RecallResult
from src.multidal.store.base import VectorStore

logger = logging.getLogger(__name__)


class FAISSStore(Stage, VectorStore):
    """Local vector store using FAISS. Persists to disk at data/faiss/."""

    name = "store"

    def __init__(self, persist_dir: str = "data/faiss") -> None:
        self._dir = Path(persist_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._indices: dict[str, faiss.IndexFlatIP] = {}
        self._metadata: dict[str, list[dict]] = {}
        self._load_existing()

    def validate(self) -> bool:
        return True

    def _load_existing(self) -> None:
        for idx_file in self._dir.glob("*.index"):
            coll_name = idx_file.stem
            index = faiss.read_index(str(idx_file))
            self._indices[coll_name] = faiss.IndexFlatIP(index.d)
            meta_file = self._dir / f"{coll_name}.json"
            if meta_file.exists():
                with open(meta_file, encoding="utf-8") as f:
                    self._metadata[coll_name] = json.load(f)
            # Migrate stored vectors
            if index.ntotal > 0:
                vectors = np.zeros((index.ntotal, index.d), dtype=np.float32)
                for i in range(index.ntotal):
                    vectors[i] = index.reconstruct(i)
                self._indices[coll_name].add(vectors)
            logger.info("Loaded collection %s: %d vectors", coll_name, index.ntotal)

    def _save(self, coll_name: str) -> None:
        if coll_name in self._indices:
            faiss.write_index(self._indices[coll_name], str(self._dir / f"{coll_name}.index"))
        if coll_name in self._metadata:
            with open(self._dir / f"{coll_name}.json", "w", encoding="utf-8") as f:
                json.dump(self._metadata[coll_name], f, ensure_ascii=False)

    def process(self, ctx: PipelineContext) -> PipelineContext:
        if ctx.embedded is None:
            raise ValueError("PipelineContext.embedded is None")
        self.insert(ctx.embedded, ctx.kb_id)
        logger.info("FAISS stored %d chunks into kb=%s", len(ctx.embedded), ctx.kb_id)
        return ctx

    def insert(self, chunks: list[EmbeddedChunk], kb_id: str) -> list[str]:
        text_coll = f"{kb_id}_text"
        image_coll = f"{kb_id}_image"

        text_vecs, text_metas = [], []
        image_vecs, image_metas = [], []
        ids = []

        for c in chunks:
            cid = c.chunk_id or uuid.uuid4().hex[:8]
            meta = {"chunk_id": cid, "content": c.content, "modality": c.modality,
                    "kb_id": c.kb_id, "doc_id": c.doc_id, "page": c.page}
            vec = np.array(c.embedding.vector, dtype=np.float32)
            if c.modality == "text":
                text_vecs.append(vec)
                text_metas.append(meta)
            else:
                image_vecs.append(vec)
                image_metas.append(meta)
            ids.append(cid)

        for coll, vecs, metas in [(text_coll, text_vecs, text_metas),
                                    (image_coll, image_vecs, image_metas)]:
            if not vecs:
                continue
            arr = np.array(vecs, dtype=np.float32)
            if coll not in self._indices:
                self._indices[coll] = faiss.IndexFlatIP(arr.shape[1])
            self._indices[coll].add(arr)
            self._metadata.setdefault(coll, []).extend(metas)
            self._save(coll)

        return ids

    def search(
        self, collection_name: str, query_vector: list[float], top_k: int = 10
    ) -> list[RecallResult]:
        if collection_name not in self._indices:
            return []
        index = self._indices[collection_name]
        all_metas = self._metadata.get(collection_name, [])
        if index.ntotal == 0:
            return []

        vec = np.array([query_vector], dtype=np.float32)
        top_k = min(top_k, index.ntotal)
        distances, indices = index.search(vec, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(all_metas):
                continue
            meta = all_metas[idx]
            results.append(RecallResult(
                chunk_id=meta.get("chunk_id", ""),
                content=meta.get("content", ""),
                modality=meta.get("modality", "text"),
                source="faiss",
                score=float(dist),
                kb_id=meta.get("kb_id", ""),
                doc_id=meta.get("doc_id", ""),
                page=meta.get("page", 1),
            ))
        return results

    def delete_collection(self, collection_name: str) -> None:
        self._indices.pop(collection_name, None)
        self._metadata.pop(collection_name, None)
        for f in self._dir.glob(f"{collection_name}.*"):
            f.unlink()
