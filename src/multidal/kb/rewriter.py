from __future__ import annotations

import logging

from src.multidal.agents.rewriter_agent import QueryRewriterAgent

logger = logging.getLogger(__name__)


class QueryRewriter:
    """将用户问题改写为 2-3 个子问题（代理封装）。"""

    def __init__(self) -> None:
        self._agent = QueryRewriterAgent()

    async def rewrite(self, question: str) -> list[str]:
        return await self._agent.rewrite(question)