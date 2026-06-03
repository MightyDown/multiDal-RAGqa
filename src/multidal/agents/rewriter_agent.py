"""查询改写智能体(QueryRewriter Agent)。

本模块实现 ``QueryRewriterAgent``,将用户的单条问题改写为 2-3 个子查询,
提升召回的"角度覆盖度"。底层使用小模型(关闭 thinking)以保证输出稳定。

使用示例::

    rewriter = QueryRewriterAgent()
    queries = await rewriter.rewrite("Q1 营收增长了多少？")
    # -> ["Q1 营收增长了多少？", "营业收入同比增速", "利润总额变化"]

兼容别名:
    - ``QueryRewriter = QueryRewriterAgent``(旧名)。
"""

from __future__ import annotations

import logging

from agents import Agent, Runner

from src.multidal.agents.base import _get_small_agent

logger = logging.getLogger(__name__)

# 改写提示词:要求按行输出多个子问题,不要编号,不要解释。
# 配合"enable_thinking=False",输出格式稳定。
REWRITE_PROMPT = """你是查询改写助手。将用户问题改写为 2-3 个子问题，覆盖核心查询角度。
每个子问题占一行，不要编号，不要额外解释。"""


class QueryRewriterAgent:
    """将用户问题改写为 2-3 个子问题(使用小模型,禁用思考)。

    Attributes:
        _agent: 小模型 Agent 实例。
    """

    def __init__(self) -> None:
        """构造改写智能体(无工具,纯生成)。"""
        self._agent = _get_small_agent(name="QueryRewriter", instructions=REWRITE_PROMPT)

    async def rewrite(self, question: str) -> list[str]:
        """改写单条问题。

        处理流程:
            1. 调用 Agent,按换行拆分输出为多个子查询;
            2. 若原始问题不在结果中,前置插入(保证至少包含原问题);
            3. 截断到 3 条以内,避免召回量爆炸。

        Args:
            question: 原始用户问题。

        Returns:
            list[str]: 改写后的子查询列表(<=3 条,包含原问题)。
        """
        try:
            result = await Runner.run(self._agent, question)
            queries = [q.strip() for q in (result.final_output or "").split("\n") if q.strip()]
            if question not in queries:
                # 原问题一定保留,作为"零改写"基线
                queries.insert(0, question)
            return queries[:3]
        except Exception:
            # 改写失败时只返回原问题,降级不影响主流程
            logger.warning("QueryRewriterAgent failed, returning original")
            return [question]


# Alias for backward compatibility
QueryRewriter = QueryRewriterAgent
