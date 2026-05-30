from __future__ import annotations

import logging

from PIL import Image

from src.multidal.schema.document import ImageRegion

logger = logging.getLogger(__name__)

CHART_SIGNALS = ["chart", "graph", "plot", "diagram", "figure", "tableau"]


def classify_image(image_bytes: bytes, caption: str = "") -> str:
    """根据 caption 文本粗略分类图片类型。"""
    lower = caption.lower()
    for signal in CHART_SIGNALS:
        if signal in lower:
            return "chart"
    return "photo"


def enrich_image_region(img: ImageRegion, raw_bytes: bytes | None = None) -> ImageRegion:
    """补充图片的 label 和尺寸信息。"""
    if raw_bytes:
        try:
            with Image.open(__import__("io").BytesIO(raw_bytes)) as pil_img:
                img.width = pil_img.width
                img.height = pil_img.height
        except Exception:
            logger.warning("Failed to read image size for %s", img.image_id)
    img.label = classify_image(b"", img.caption)
    return img
