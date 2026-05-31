from __future__ import annotations

import json
import logging

from src.multidal.agents.router_agent import KBRouterAgent, _format_kb_list, _parse_kb_ids
from src.multidal.kb.manager import KBManager

logger = logging.getLogger(__name__)


class IntentRouter:
    """根据用户问题自动路由到目标 KB（代理封装）。"""

    def __init__(self, kb_manager: KBManager) -> None:
        self._agent = KBRouterAgent(kb_manager)

    async def route(
        self, question: str, kb_ids: list[str] | None = None, auto_route: bool = True
    ) -> list[str]:
        return await self._agent.route(question, kb_ids, auto_route)