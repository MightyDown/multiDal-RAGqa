from __future__ import annotations

import base64
import io
import logging
import uuid
from pathlib import Path

import requests
from PIL import Image

from src.multidal.config import settings
from src.multidal.pipeline.base import PipelineContext, Stage
from src.multidal.schema.embedding import EmbeddedChunk, Embedding

logger = logging.getLogger(__name__)

MAX_IMAGE_SIZE = 168
JPEG_QUALITY = 30


class ImageEmbedder(Stage):
    """CLIP 双路向量化：图片像素走 image encoder，描述文字走 text encoder，
    两者都在 CLIP 共享空间中，存入 {kb}_image collection。

    可选：配合 VLCaptioner 为图片生成 VL 描述，显著提升图片检索质量。"""

    name = "embedder_image"

    def __init__(self, vl_captioner=None) -> None:
        self._api_base = settings.image_embedding_api_base
        self._model = settings.image_embedding_model
        self._dim = settings.image_embedding_dim
        self._key = settings.image_embedding_api_key
        self._vl_captioner = vl_captioner  # optional VLCaptioner for better captions

    # ── Stage 接口 ──────────────────────────────────────────

    def validate(self) -> bool:
        return self._ping()

    def process(self, ctx: PipelineContext) -> PipelineContext:
        if ctx.parsed is None:
            raise ValueError("PipelineContext.parsed is None, run Parser first")

        if not ctx.parsed.images:
            return ctx

        chunks: list[EmbeddedChunk] = []
        for img in ctx.parsed.images:
            caption = self._get_caption(img)

            # 路径1: 图片像素 → CLIP image encoder
            image_uri = self._load_image_uri(img.image_path)
            if image_uri:
                img_vec = self._encode(image_uri)
                if img_vec is not None:
                    chunks.append(self._make_chunk(img, caption, img_vec, ctx))

            # 路径2: 图片描述文字 → CLIP text encoder（同一 CLIP 空间）
            desc_vec = self._encode(caption)
            if desc_vec is not None:
                chunks.append(self._make_chunk(img, caption, desc_vec, ctx, suffix="_desc"))

        ctx.embedded = (ctx.embedded or []) + chunks
        logger.info("Image embedder: %d vectors (%d images × 2 paths)", len(chunks), len(ctx.parsed.images))
        return ctx

    def _get_caption(self, img) -> str:
        """生成图片 caption：优先用 VL 模型，其次用 MinerU 原 caption。"""
        if self._vl_captioner:
            vl_cap = self._vl_captioner.caption_image(img.image_path)
            if vl_cap:
                return vl_cap
        return img.caption or f"image page {img.page}"

    # ── 检索时调用 ──────────────────────────────────────────

    def embed_query(self, text: str) -> list[float]:
        """CLIP text encoder：将查询文本映射到 CLIP 空间，用于搜图片库。"""
        vec = self._encode(text)
        if vec is None:
            raise RuntimeError("CLIP query embedding failed")
        return vec

    # ── 内部方法 ────────────────────────────────────────────

    def _encode(self, input_data: str) -> list[float] | None:
        """统一编码入口：data URI → image encoder，纯文本 → text encoder。"""
        try:
            r = requests.post(
                f"{self._api_base}/embeddings",
                json={"model": self._model, "input": input_data, "encoding_format": "float"},
                headers={"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"},
                timeout=60,
            )
            r.raise_for_status()
            return r.json()["data"][0]["embedding"]
        except requests.HTTPError:
            logger.warning("CLIP API HTTP %d: %s", r.status_code, r.text[:200])
            return None
        except Exception:
            logger.warning("CLIP API error", exc_info=True)
            return None

    def _ping(self) -> bool:
        try:
            r = requests.post(
                f"{self._api_base}/embeddings",
                json={"model": self._model, "input": "ping"},
                headers={"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"},
                timeout=10,
            )
            return r.status_code == 200
        except Exception:
            logger.warning("Image embedding API not reachable at %s", self._api_base)
            return False

    @staticmethod
    def _make_chunk(img, content: str, vec: list[float], ctx: PipelineContext, suffix: str = "") -> EmbeddedChunk:
        return EmbeddedChunk(
            chunk_id=img.image_id + suffix,
            content=content,
            embedding=Embedding(model_name=settings.image_embedding_model, dim=len(vec), vector=vec),
            modality="image",
            kb_id=ctx.kb_id,
            doc_id=ctx.parsed.doc_id,
            page=img.page,
            image_path=img.image_path,
        )

    @staticmethod
    def _load_image_uri(image_path: str) -> str | None:
        if not image_path:
            return None
        p = Path(image_path)
        if not p.exists():
            return None
        try:
            img = Image.open(p).convert("RGB")
            w, h = img.size
            if max(w, h) > MAX_IMAGE_SIZE:
                ratio = MAX_IMAGE_SIZE / max(w, h)
                img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=JPEG_QUALITY)
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            return f"data:image/jpeg;base64,{b64}"
        except Exception:
            logger.warning("Failed to read image %s", image_path)
            return None
