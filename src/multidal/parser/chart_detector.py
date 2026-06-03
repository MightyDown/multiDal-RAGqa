"""图片类型粗分类与元数据补全工具。

本模块提供两个纯函数:
    - ``classify_image``: 基于 caption 文本判断图片是否为"图表类"。
    - ``enrich_image_region``: 读取图片字节流以补全 ``ImageRegion`` 的宽高与 label。

实现策略:
    - 不调用任何机器学习模型,仅依赖关键词匹配,目的是在 Embedder 之前
      给图片一个粗略标签,便于路由与统计。
    - 关键词集合刻意保持精简(``chart`` / ``graph`` / ``plot`` / ``diagram`` /
      ``figure`` / ``tableau``),避免误判(例如 "image of a chart of accounts")。
"""

from __future__ import annotations

import logging

from PIL import Image

from src.multidal.schema.document import ImageRegion

logger = logging.getLogger(__name__)

# 触发 "chart" 分类的关键词子串(大小写不敏感)。
# 这些词通常出现在图表类图片的 caption 中(如 "revenue chart" / "bar graph")。
CHART_SIGNALS = ["chart", "graph", "plot", "diagram", "figure", "tableau"]


def classify_image(image_bytes: bytes, caption: str = "") -> str:
    """根据 caption 文本粗略分类图片类型。

    分类规则:
        - 若 caption 包含任意 ``CHART_SIGNALS`` 关键词(大小写不敏感),返回 ``"chart"``;
        - 否则一律返回 ``"photo"``。

    注意:此函数不读取 ``image_bytes``,参数保留仅为接口对称(便于后续扩展为视觉特征分类)。

    Args:
        image_bytes: 图片字节流(当前未使用)。
        caption: 图片描述文本,可为空。

    Returns:
        str: ``"chart"`` 或 ``"photo"``。
    """
    lower = caption.lower()
    for signal in CHART_SIGNALS:
        if signal in lower:
            return "chart"
    return "photo"


def enrich_image_region(img: ImageRegion, raw_bytes: bytes | None = None) -> ImageRegion:
    """为 ImageRegion 补全 label 与尺寸信息(原地修改并返回)。

    操作:
        1. 若提供 ``raw_bytes``,用 PIL 读取真实宽高并写入 ``img.width``/``img.height``。
        2. 调用 ``classify_image`` 根据 ``img.caption`` 推断 ``img.label``。

    Args:
        img: 待补全的 ``ImageRegion`` 对象(会被原地修改)。
        raw_bytes: 图片二进制内容,为 None 时跳过尺寸补全。

    Returns:
        ImageRegion: 同一个 ``img`` 对象(便于链式调用)。
    """
    if raw_bytes:
        try:
            # 延迟 import io,避免在模块加载时引入额外符号
            with Image.open(__import__("io").BytesIO(raw_bytes)) as pil_img:
                img.width = pil_img.width
                img.height = pil_img.height
        except Exception:
            # 图片字节不合法(裁切坏/格式异常)时只记录 warning,
            # 不阻断流水线(尺寸留 0 即可,前端会兜底显示)
            logger.warning("Failed to read image size for %s", img.image_id)
    # label 直接覆盖:MinerU 自带的 type 不一定准确(经常是 "image"),
    # 用 caption 关键词二次校正
    img.label = classify_image(b"", img.caption)
    return img
