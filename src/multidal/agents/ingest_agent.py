from __future__ import annotations

import logging

from agents import Runner

from src.multidal.agents.base import BaseAgent

logger = logging.getLogger(__name__)

INSTRUCTIONS = """你是文档解析质量审核助手。收到 MinerU 解析结果后：
1. 检查文本输出是否可读、完整
2. 检查提取的图片/表格是否合理
3. 输出质量合格 → 回复 APPROVED
4. 输出质量差 → 回复 REVIEW_NEEDED: <具体原因>
"""


class IngestAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="MultiDal Ingest Reviewer", instructions=INSTRUCTIONS)

    async def run(self, parse_summary: str) -> str:
        result = await Runner.run(self.agent, parse_summary)
        return result.final_output
