from __future__ import annotations

import json
import logging

from agents import Agent, Runner

from src.multidal.agents.base import _get_chat_model
from src.multidal.kb.manager import KBManager

logger = logging.getLogger(__name__)


class IntentRouter:
    """根据用户问题自动路由到目标 KB。"""

    def __init__(self, kb_manager: KBManager) -> None:
        self._kb_manager = kb_manager
        self._agent = Agent(
            name="IntentRouter",
            model=_get_chat_model(),
            instructions=(
                "你是知识库路由助手。根据用户问题，判断需要查询哪些知识库。"
                "严格只返回 JSON 数组格式: [\"kb_id1\", \"kb_id2\"]"
            ),
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

        prompt = f"""可用知识库:
{_format_kb_list(kbs)}

用户问题: {question}

只返回 JSON 数组。"""
        result = await Runner.run(self._agent, prompt)
        try:
            return _parse_kb_ids(result.final_output)
        except Exception:
            logger.warning("IntentRouter parse failed, using all KBs")
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
    return json.loads(text)
