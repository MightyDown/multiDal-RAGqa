"""VL Caption 生成器：使用 Qwen2.5-VL-7B-Instruct 为图片生成中文描述。"""

from __future__ import annotations

import base64
import io
import logging
from pathlib import Path

import requests
from PIL import Image

from src.multidal.config import settings

logger = logging.getLogger(__name__)

MAX_IMAGE_SIZE = 1024  # VL 模型最大边长 1024px


class VLCaptioner:
    """使用 Qwen2.5-VL 为图片生成描述，caption 用于 CLIP text encoder 检索。"""

    def __init__(self) -> None:
        self._api_base = settings.vl_caption_api_base
        self._model = settings.vl_caption_model
        self._key = settings.vl_caption_api_key

    def validate(self) -> bool:
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
        """生成图片描述，返回中文 caption。失败返回 None。"""
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
            logger.warning("VL caption failed for %s", image_path)
            return None

    def _encode_image(self, image_path: str) -> str | None:
        """把图片转成 base64 JPEG，用于 API 传输。"""
        try:
            p = Path(image_path)
            if not p.exists():
                return None
            img = Image.open(p).convert("RGB")
            w, h = img.size
            if max(w, h) > MAX_IMAGE_SIZE:
                ratio = MAX_IMAGE_SIZE / max(w, h)
                img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            return base64.b64encode(buf.getvalue()).decode("ascii")
        except Exception:
            logger.warning("Failed to encode image %s", image_path)
            return None