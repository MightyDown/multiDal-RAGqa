"""VL(视觉语言)Caption 生成器。

本模块实现 ``VLCaptioner``,使用 Qwen2.5-VL-7B-Instruct 为图片生成中文描述。
生成的 caption 替代 MinerU 自带的英文 caption,作为 CLIP text encoder 的输入,
可显著提升中文场景下"以文搜图"的召回质量。

使用方式(典型):
    captioner = VLCaptioner()
    if captioner.validate():
        embedder = ImageEmbedder(vl_captioner=captioner)
"""

from __future__ import annotations

import base64
import io
import logging
from pathlib import Path

import requests
from PIL import Image

from src.multidal.config import settings

logger = logging.getLogger(__name__)

# VL 模型的最大边长(像素):超过此值会等比缩放,减少 base64 体积与推理耗时。
MAX_IMAGE_SIZE = 1024


class VLCaptioner:
    """使用 Qwen2.5-VL 为图片生成中文描述,caption 用于 CLIP text encoder 检索。

    Attributes:
        _api_base: VL API 根地址(OpenAI-compatible /chat/completions)。
        _model: 模型名称(从配置读取)。
        _key: API Key。
    """

    def __init__(self) -> None:
        """从全局配置读取 API 凭据与模型信息。"""
        self._api_base = settings.vl_caption_api_base
        self._model = settings.vl_caption_model
        self._key = settings.vl_caption_api_key

    def validate(self) -> bool:
        """通过发送一个最小图片请求探测 API 可用性。

        策略:发送一张 10x10 占位图 + "hello" 文本,期望 status < 500。
        不要求 200,因为部分 API 在 prompt 极简时可能返回 4xx 但服务本身可用。

        Returns:
            bool: 服务端无 5xx 时为 True。
        """
        try:
            r = requests.post(
                f"{self._api_base}/chat/completions",
                json={
                    "model": self._model,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": "https://placehold.co/10x10.png"}},
                                {"type": "text", "text": "hello"},
                            ],
                        }
                    ],
                    "max_tokens": 10,
                },
                headers={"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"},
                timeout=15,
            )
            return r.status_code < 500
        except Exception:
            logger.warning("VL caption API not reachable at %s", self._api_base)
            return False

    def caption_image(self, image_path: str) -> str | None:
        """为本地图片生成中文 caption。

        Args:
            image_path: 图片绝对路径。

        Returns:
            str | None: 生成的 caption 文本(已 strip);失败时返回 None。
        """
        try:
            b64 = self._encode_image(image_path)
            if not b64:
                return None

            data_url = f"data:image/jpeg;base64,{b64}"
            r = requests.post(
                f"{self._api_base}/chat/completions",
                json={
                    "model": self._model,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": data_url}},
                                {
                                    "type": "text",
                                    "text": "请简洁描述这张图片的内容，不要超过50个字。用中文回答。",
                                },
                            ],
                        }
                    ],
                    "max_tokens": 100,
                },
                headers={"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"},
                timeout=30,
            )
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
            return content.strip()
        except Exception:
            # 任何异常(超时/限流/JSON 异常)只记录 warning,不抛给上层
            # ImageEmbedder 会回退到 MinerU caption,不影响整体流程
            logger.warning("VL caption failed for %s", image_path)
            return None

    def _encode_image(self, image_path: str) -> str | None:
        """读取本地图片,等比缩放并压缩为 base64 JPEG 字符串。

        Args:
            image_path: 图片路径。

        Returns:
            str | None: base64 字符串;路径不存在或读取失败时返回 None。
        """
        try:
            p = Path(image_path)
            if not p.exists():
                return None
            img = Image.open(p).convert("RGB")
            w, h = img.size
            if max(w, h) > MAX_IMAGE_SIZE:
                # 等比缩放,避免大图超出 VL 模型输入限制
                ratio = MAX_IMAGE_SIZE / max(w, h)
                img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            return base64.b64encode(buf.getvalue()).decode("ascii")
        except Exception:
            logger.warning("Failed to encode image %s", image_path)
            return None
