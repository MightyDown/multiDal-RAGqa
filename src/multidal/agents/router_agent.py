from __future__ import annotations

import json
import logging

from agents import Agent, Runner

from src.multidal.agents.base import _get_small_agent
from src.multidal.agents.tools import get_doc_info, search_knowledge_base
from src.multidal.kb.manager import KBManager

logger = logging.getLogger(__name__)

KB_LIST_PROMPT = """你是知识库路由助手。根据用户问题，判断需要查询哪些知识库。
严格只返回 JSON 数组格式: ["kb_id1", "kb_id2"]。不要有任何其他文字。"""


class KBRouterAgent:
    """根据用户问题自动路由到目标 KB（使用小模型，禁用思考）。"""

    def __init__(self, kb_manager: KBManager) -> None:
        self._kb_manager = kb_manager
        self._agent = _get_small_agent(
            name="KBRouter",
            instructions=KB_LIST_PROMPT,
            tools=[search_knowledge_base, get_doc_info],
        )

    async def route(
        self, question: str, kb_ids: list[str] | None = None, auto_route: bool = True
    ) -> list[str]:
        if kb_ids:
            return kb_ids
        if not auto_route:
            return self._kb_manager.list_all_ids()

        kbs = self._kb_manager.list_all()
        if kbs.total == 0:
            return []

        kb_list = _format_kb_list(kbs)
        prompt = f"""可用知识库:\n{kb_list}\n\n用户问题: {question}\n\n只返回 JSON 数组。"""
        try:
            result = await Runner.run(self._agent, prompt)
            return _parse_kb_ids(result.final_output or "")
        except Exception:
            logger.warning("KBRouterAgent failed, using all KBs")
            return self._kb_manager.list_all_ids()


def _format_kb_list(kbs) -> str:
    lines = []
    for kb in kbs.kbs:
        lines.append(f"- {kb.kb_id}: {kb.name} ({kb.description})")
    return "\n".join(lines)


def _parse_kb_ids(text: str) -> list[str]:
    text = text.strip()
    if "[" in text:
        start = text.index("[")
        end = text.rindex("]") + 1
        text = text[start:end]
    try:
        return json.loads(text)
    except Exception:
        return []


# Alias for backward compatibility
IntentRouter = KBRouterAgent