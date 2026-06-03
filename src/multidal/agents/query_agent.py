"""文档问答智能体(Query Agent)。

本模块实现 ``QueryAgent``,是用户与系统对话的主入口,负责:
    1. 接收用户问题与检索上下文(由 API 层预先做 KB 路由 + 召回 + Rerank 拼好);
    2. 调用主 LLM(MiniMax-M2.7)生成答案;
    3. 支持普通与流式(SSE)两种返回模式。

注意:本智能体不直接做检索,检索由 API 层通过 ``search_knowledge_base``
工具或外部 MultiPathRetriever 完成后,再以"context"字符串形式传入。
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator

from agents import Runner, SQLiteSession

from src.multidal.agents.base import BaseAgent

logger = logging.getLogger(__name__)

# QueryAgent 的系统提示词(中文),定义 LLM 行为准则:
#   - 闲聊不检索;需要时按上下文回答;引用文档与页码;诚实告知空结果;
#   - 生成 Mermaid 图时严格单行样式(streaming 兼容性约束)。
INSTRUCTIONS = """你是企业文档问答助手，可以检索已入库的 PDF 文档来回答问题。
1. 如果问题是闲聊或不需要文档知识，直接回答
2. 需要检索时，根据上下文中的检索结果回答
3. 回答时引用具体的文档名和页码
4. 如果知识库中没有相关信息，诚实告知用户
5. 生成 Mermaid 流程图时：
   - 必须用 ```mermaid 代码块包裹，不能明文输出
   - 使用单行内联样式，不要换行写 style/classDef/click 指令
   - 推荐写法示例：
     ```mermaid
     flowchart LR
         A[Input] --> B[Process] --> C[Output]
         style A fill:#e6f7ff,stroke:#1890ff
         style B fill:#fff0f6,stroke:#eb2f96
         style C fill:#d6e4ff,stroke:#1890ff
     ```
   - 禁止多行 classDef，所有节点样式应紧凑写在一行
   - 这样可以保证 streaming 时不会因换行导致渲染失败
"""


class QueryAgent(BaseAgent):
    """文档问答智能体,基于主 LLM 生成最终答案。"""

    def __init__(self) -> None:
        """构造一个名为 ``"MultiDal QA"`` 的查询智能体。"""
        super().__init__(name="MultiDal QA", instructions=INSTRUCTIONS)

    async def run(self, question: str, context: str, session: SQLiteSession | None = None) -> str:
        """非流式问答:返回完整答案字符串。

        Args:
            question: 用户问题。
            context: 由检索链路(MultiPathRetriever + Reranker)拼好的文档片段文本。
            session: 可选的多轮会话(``SQLiteSession`` / ``MySQLSession``),为 None 时无记忆。

        Returns:
            str: LLM 生成的最终答案。
        """
        prompt = _build_prompt(question, context)
        result = await Runner.run(self.agent, prompt, session=session)
        return result.final_output

    async def run_streamed(
        self, question: str, context: str, session: SQLiteSession | None = None
    ) -> AsyncGenerator[str, None]:
        """流式问答:每次 yield 一段文本增量(用于 SSE 前端展示)。

        通过 ``Runner.run_streamed`` 拿到流式事件,仅消费 ``raw_response_event`` 类型
        并尝试取 ``delta`` 字段;其他类型事件(``run_started`` / ``tool_called`` 等)忽略。

        Args:
            question: 用户问题。
            context: 检索上下文。
            session: 可选的会话对象。

        Yields:
            str: LLM 输出的文本增量。
        """
        prompt = _build_prompt(question, context)
        result = Runner.run_streamed(self.agent, prompt, session=session)
        async for event in result.stream_events():
            if event.type == "raw_response_event":
                try:
                    # 部分 SDK 版本 delta 不存在,需 getattr 兜底
                    chunk = getattr(event.data, "delta", "")
                    if chunk:
                        yield chunk
                except Exception:
                    # 单个事件解析失败不阻断流,继续消费后续事件
                    pass


def _build_prompt(question: str, context: str) -> str:
    """构造发给 LLM 的完整 prompt:检索上下文 + 用户问题 + 指令。

    Args:
        question: 用户问题。
        context: 检索到的相关文档片段(已按 Rerank 分排序)。

    Returns:
        str: 拼接好的 prompt。
    """
    return f"""检索到的相关文档内容:

{context}

---
用户问题: {question}

请基于以上文档内容回答问题。"""
