"""Agents 包 - LLM 驱动的智能体(查询路由、问题改写、问答)。

对外暴露:
    - ``BaseAgent``: 封装 openai-agents SDK Agent 的基类(主 LLM,默认 MiniMax-M2.7)。
    - ``QueryAgent``: 文档问答智能体,支持普通与流式两种模式。
    - ``KBRouterAgent``(``IntentRouter`` 别名): 根据问题自动路由到目标 KB。
    - ``QueryRewriterAgent``(``QueryRewriter`` 别名): 将问题改写为 2-3 个子查询。
    - ``search_knowledge_base`` / ``get_doc_info``: openai-agents function_tool 工具。
    - ``MySQLSession`` / ``get_session`` / 等: 兼容 openai-agents SDK 的 MySQL 会话实现。

辅助模块(本包内):
    - ``sessions``: 会话持久化(基于 MySQL 的 agent_sessions + agent_messages 表)。
"""

from src.multidal.agents.base import BaseAgent
from src.multidal.agents.query_agent import QueryAgent
from src.multidal.agents.tools import get_doc_info, search_knowledge_base
