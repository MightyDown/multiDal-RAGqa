from __future__ import annotations

import logging

from agents import Agent, Runner

from src.multidal.agents.base import _get_chat_model

logger = logging.getLogger(__name__)


class QueryRewriter:
    """将用户问题扩展为 3-5 个多角度搜索词。"""

    def __init__(self) -> None:
        self._agent = Agent(
            name="QueryRewriter",
            model=_get_chat_model(),
            instructions=(
                "你是查询改写助手。将用户问题改写为 3-5 个不同角度的搜索关键词，"
                "覆盖核心概念、同义表达、相关指标、时间维度。"
                "每行一个关键词，不要编号，不要额外解释。"
            ),
        )

    async def rewrite(self, question: str) -> list[str]:
        result = await Runner.run(self._agent, question)
        queries = [q.strip() for q in result.final_output.split("\n") if q.strip()]
        # 保留原问题在第一位
        if question not in queries:
            queries.insert(0, question)
        return queries
