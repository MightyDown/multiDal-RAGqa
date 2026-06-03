"""KB 路由智能体(KBRouter Agent)。

本模块实现 ``KBRouterAgent``,根据用户问题自动选择要检索的 KB 列表。
底层使用小模型(``_get_small_agent``,关闭 thinking),通过 function_tool 让
模型能主动调用 ``search_knowledge_base`` / ``get_doc_info`` 做更精细的判断。

典型工作流:
    1. 用户问题 + 候选 KB 列表 -> 拼成 prompt 交给 Agent;
    2. Agent 输出严格 JSON 数组(``["kb_id1", "kb_id2"]``);
    3. ``_parse_kb_ids`` 解析 JSON,容错处理多余文字。

兼容别名:
    - ``IntentRouter = KBRouterAgent``(旧名,保留以兼容调用方)。
"""

from __future__ import annotations

import json
import logging

from agents import Agent, Runner

from src.multidal.agents.base import _get_small_agent
from src.multidal.agents.tools import get_doc_info, search_knowledge_base
from src.multidal.kb.manager import KBManager

logger = logging.getLogger(__name__)

# KB 路由的提示词:严格要求只返回 JSON 数组,无其他文字。
# 配合"enable_thinking=False",小模型能稳定输出可解析的 JSON。
KB_LIST_PROMPT = """你是知识库路由助手。根据用户问题，判断需要查询哪些知识库。
严格只返回 JSON 数组格式: ["kb_id1", "kb_id2"]。不要有任何其他文字。"""


class KBRouterAgent:
    """根据用户问题自动路由到目标 KB(使用小模型,禁用思考)。

    Attributes:
        _kb_manager: KB 管理器,提供 KB 列表与全量 ID 查询。
        _agent: 配置好的小模型 Agent(含 search/get_doc_info 工具)。
    """

    def __init__(self, kb_manager: KBManager) -> None:
        """构造 KB 路由智能体。

        Args:
            kb_manager: ``KBManager`` 实例,用于枚举与查询 KB。
        """
        self._kb_manager = kb_manager
        self._agent = _get_small_agent(
            name="KBRouter",
            instructions=KB_LIST_PROMPT,
            tools=[search_knowledge_base, get_doc_info],
        )

    async def route(
        self, question: str, kb_ids: list[str] | None = None, auto_route: bool = True
    ) -> list[str]:
        """对单个问题做 KB 路由。

        决策流程:
            1. 若显式传 ``kb_ids``,直接返回(优先权最高,跳过路由);
            2. 若 ``auto_route=False``,返回所有 KB ID(不调模型);
            3. 若 KB 总数为 0,返回空列表;
            4. 否则用小模型判断,失败时降级到"全量 KB"。

        Args:
            question: 用户问题。
            kb_ids: 显式指定的目标 KB 列表(为 None 时走自动路由)。
            auto_route: 是否启用模型自动路由(False 时回退到全量 KB)。

        Returns:
            list[str]: 选中的 KB ID 列表。
        """
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
            # 路由失败时降级为"全量 KB",确保不会因为路由错误让用户得不到答案
            logger.warning("KBRouterAgent failed, using all KBs")
            return self._kb_manager.list_all_ids()


def _format_kb_list(kbs) -> str:
    """把 KB 列表格式化为"行式"文本,作为 prompt 的一部分。

    Args:
        kbs: ``KBListResponse`` 类型的对象(含 ``kbs`` 列表)。

    Returns:
        str: 多行字符串,每行形如 ``"- kb_id: 名称 (描述)"``。
    """
    lines = []
    for kb in kbs.kbs:
        lines.append(f"- {kb.kb_id}: {kb.name} ({kb.description})")
    return "\n".join(lines)


def _parse_kb_ids(text: str) -> list[str]:
    """从模型输出中解析 KB ID 列表(容错处理多余文字)。

    处理流程:
        1. 截取首个 ``[`` 到末尾 ``]`` 之间的子串(剥离前后说明文字);
        2. 尝试 ``json.loads``;
        3. 失败返回空列表(由上层决定降级策略)。

    Args:
        text: 模型原始输出。

    Returns:
        list[str]: 解析出的 KB ID 列表;解析失败时为空列表。
    """
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
