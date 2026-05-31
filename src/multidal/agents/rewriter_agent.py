from __future__ import annotations

import logging

from agents import Agent, Runner

from src.multidal.agents.base import _get_small_agent

logger = logging.getLogger(__name__)

REWRITE_PROMPT = """你是查询改写助手。将用户问题改写为 2-3 个子问题，覆盖核心查询角度。
每个子问题占一行，不要编号，不要额外解释。"""


class QueryRewriterAgent:
    """将用户问题改写为 2-3 个子问题（使用小模型，禁用思考）。"""

    def __init__(self) -> None:
        self._agent = _get_small_agent(name="QueryRewriter", instructions=REWRITE_PROMPT)

    async def rewrite(self, question: str) -> list[str]:
        try:
            result = await Runner.run(self._agent, question)
            queries = [q.strip() for q in (result.final_output or "").split("\n") if q.strip()]
            if question not in queries:
                queries.insert(0, question)
            return queries[:3]
        except Exception:
            logger.warning("QueryRewriterAgent failed, returning original")
            return [question]


# Alias for backward compatibility
QueryRewriter = QueryRewriterAgent