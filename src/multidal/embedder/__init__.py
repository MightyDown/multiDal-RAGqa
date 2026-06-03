"""Embedder 包 - 文本与图片向量化。

对外暴露:
    - ``TextEmbedder``: 继承 ``Stage`` 的文本嵌入器(BGE 等)。
    - ``ImageEmbedder``: 继承 ``Stage`` 的图片双路嵌入器(CLIP 像素 + CLIP 描述)。

辅助模块(本包内,不导出):
    - ``vl_captioner``: 可选的视觉语言描述生成器,作为 ImageEmbedder 的"增强 caption"来源。
"""

from src.multidal.embedder.image_embedder import ImageEmbedder
from src.multidal.embedder.text_embedder import TextEmbedder
