"""图片向量化器(CLIP 双路)。

本模块实现 ``ImageEmbedder``(继承自 ``Stage``),对每张图片同时产出两条向量:
    1. 图片像素向量 -> CLIP image encoder;
    2. 图片描述向量 -> CLIP text encoder。

两条向量在 CLIP 共享空间中,可与 ``embed_query`` 编码的查询文本直接计算余弦相似度。
这种"双路"策略兼顾了"以图搜图"和"以文搜图"两种检索意图。

可选增强:
    接入 ``VLCaptioner``(Qwen2.5-VL)为图片生成更准确的中文描述,
    显著提升"以文搜图"路径的召回质量。
"""

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

# 图片最大边长:CLIP 模型对 224x224 输入最友好;这里取 168 作为保守值,
# 兼顾长图/宽图识别质量与 base64 体积。
MAX_IMAGE_SIZE = 168
# JPEG 压缩质量:30 已足够保留视觉特征,可显著降低 base64 体积(节省 API 流量)。
JPEG_QUALITY = 30


class ImageEmbedder(Stage):
    """CLIP 双路向量化器。

    对每个图片生成两条向量:
        - 路径1(像素):``{kb}_image`` collection 的一条记录;
        - 路径2(描述):同上 collection 的另一条记录(``chunk_id`` 加 ``_desc`` 后缀)。

    Attributes:
        name: 阶段名,固定为 ``"embedder_image"``。
        _api_base: 嵌入 API 根地址。
        _model: CLIP 模型名称。
        _dim: 预期向量维度(从配置读取)。
        _key: API Key。
        _vl_captioner: 可选的 VL caption 增强器,为 None 时使用 MinerU 自带 caption。
    """

    name = "embedder_image"

    def __init__(self, vl_captioner=None) -> None:
        """初始化图片嵌入器。

        Args:
            vl_captioner: 可选的 ``VLCaptioner`` 实例,用于生成更准确的图片描述。
        """
        self._api_base = settings.image_embedding_api_base
        self._model = settings.image_embedding_model
        self._dim = settings.image_embedding_dim
        self._key = settings.image_embedding_api_key
        self._vl_captioner = vl_captioner  # optional VLCaptioner for better captions

    # ── Stage 接口 ──────────────────────────────────────────

    def validate(self) -> bool:
        """通过 ``_ping`` 探测 API 可达性。

        Returns:
            bool: API 可达且鉴权成功时为 True。
        """
        return self._ping()

    def process(self, ctx: PipelineContext) -> PipelineContext:
        """对 ``ctx.parsed.images`` 中每张图片生成双路向量并写入 ``ctx.embedded``。

        步骤:
            1. 校验 ``ctx.parsed`` 已就绪;
            2. 无图片时直接短路返回(不报错);
            3. 遍历每张图片,先取 caption(优先 VL,其次 MinerU),再分别走像素/描述双路;
            4. 每路成功则追加一条 ``EmbeddedChunk``;
            5. 累加到 ``ctx.embedded``(保留 TextEmbedder 的产出)。

        Args:
            ctx: 流水线上下文,需含 ``parsed`` 与 ``kb_id``。

        Returns:
            PipelineContext: 已填充 ``embedded`` 的上下文。

        Raises:
            ValueError: ``ctx.parsed`` 为 None 时。
        """
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

            # 路径2: 图片描述文字 → CLIP text encoder(同一 CLIP 空间)
            desc_vec = self._encode(caption)
            if desc_vec is not None:
                chunks.append(self._make_chunk(img, caption, desc_vec, ctx, suffix="_desc"))

        ctx.embedded = (ctx.embedded or []) + chunks
        logger.info("Image embedder: %d vectors (%d images × 2 paths)", len(chunks), len(ctx.parsed.images))
        return ctx

    def _get_caption(self, img) -> str:
        """为图片取 caption,优先级:VL 模型 > MinerU caption > 兜底文本。

        Args:
            img: ``ImageRegion`` 对象(来自 Parser)。

        Returns:
            str: 最终使用的 caption(非空)。
        """
        if self._vl_captioner:
            vl_cap = self._vl_captioner.caption_image(img.image_path)
            if vl_cap:
                return vl_cap
        # 兜底:MinerU 抽出的 caption(可能为空)或简单的"page N"描述
        return img.caption or f"image page {img.page}"

    # ── 检索时调用 ──────────────────────────────────────────

    def embed_query(self, text: str) -> list[float]:
        """公开方法:将查询文本编码到 CLIP 空间,供"以文搜图"使用。

        Args:
            text: 查询文本。

        Returns:
            list[float]: CLIP 向量。

        Raises:
            RuntimeError: 编码失败(API 错误)时。
        """
        vec = self._encode(text)
        if vec is None:
            raise RuntimeError("CLIP query embedding failed")
        return vec

    # ── 内部方法 ────────────────────────────────────────────

    def _encode(self, input_data: str) -> list[float] | None:
        """统一编码入口。

        通过文件内容自动判别:data URI(``data:image/...``)走 image encoder,
        否则走 text encoder。

        Args:
            input_data: data URI 字符串(图片)或纯文本(描述)。

        Returns:
            list[float] | None: 编码成功时返回向量;失败(网络/限流)时返回 None,
                                调用方决定是否忽略该路径(不抛异常以保证"双路降级")。
        """
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
        """轻量探活:发送一个 ``"ping"`` 文本,期望 200。

        Returns:
            bool: API 可达且鉴权成功时为 True。
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
            logger.warning("Image embedding API not reachable at %s", self._api_base)
            return False

    @staticmethod
    def _make_chunk(img, content: str, vec: list[float], ctx: PipelineContext, suffix: str = "") -> EmbeddedChunk:
        """根据图片与向量构造 ``EmbeddedChunk``。

        Args:
            img: ``ImageRegion`` 源对象。
            content: caption 文本。
            vec: 向量列表。
            ctx: 流水线上下文(取 kb_id / doc_id)。
            suffix: ``chunk_id`` 后缀,描述路径固定为 ``"_desc"``,像素路径为空。

        Returns:
            EmbeddedChunk: 待写入 Store 的记录。
        """
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
        """读取本地图片,缩放并压缩后转为 base64 data URI。

        Args:
            image_path: 图片本地路径。

        Returns:
            str | None: data URI 字符串(``data:image/jpeg;base64,...``);
                        路径为空/文件不存在/读取失败时返回 None。
        """
        if not image_path:
            return None
        p = Path(image_path)
        if not p.exists():
            return None
        try:
            img = Image.open(p).convert("RGB")
            w, h = img.size
            if max(w, h) > MAX_IMAGE_SIZE:
                # 等比缩放到最长边不超过 MAX_IMAGE_SIZE,保留纵横比
                ratio = MAX_IMAGE_SIZE / max(w, h)
                img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=JPEG_QUALITY)
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            return f"data:image/jpeg;base64,{b64}"
        except Exception:
            logger.warning("Failed to read image %s", image_path)
            return None
