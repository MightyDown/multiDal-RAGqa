"""Parser 包 - PDF 解析与图片分类。

对外暴露:
    - ``MinerUParser``: 继承自 ``Stage`` 的 MinerU 云 API 解析阶段。
    - ``classify_image``: 基于 caption 关键词的图片粗分类器。
    - ``enrich_image_region``: 为 ImageRegion 补全尺寸与 label 的工具函数。

辅助:
    - ``models``: 早期 dataclass 形态的中间模型(与 ``schema.document`` 重叠,
      当前生产代码未使用,保留作为参考实现)。
"""

from src.multidal.parser.chart_detector import classify_image, enrich_image_region
from src.multidal.parser.mineru_parser import MinerUParser
