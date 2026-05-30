from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path

from PIL import Image


def load_image(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def resize_image(img: Image.Image, max_size: int = 1024) -> Image.Image:
    w, h = img.size
    if max(w, h) <= max_size:
        return img
    ratio = max_size / max(w, h)
    return img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)


def encode_base64(img: Image.Image, fmt: str = "PNG") -> str:
    buf = BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def image_to_data_uri(img: Image.Image, fmt: str = "PNG") -> str:
    b64 = encode_base64(img, fmt)
    return f"data:image/{fmt.lower()};base64,{b64}"
