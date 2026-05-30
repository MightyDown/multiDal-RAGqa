from __future__ import annotations

import logging
from collections.abc import AsyncGenerator

from agents import Runner, SQLiteSession

from src.multidal.agents.base import BaseAgent

logger = logging.getLogger(__name__)

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
    def __init__(self) -> None:
        super().__init__(name="MultiDal QA", instructions=INSTRUCTIONS)

    async def run(self, question: str, context: str, session: SQLiteSession | None = None) -> str:
        prompt = _build_prompt(question, context)
        result = await Runner.run(self.agent, prompt, session=session)
        return result.final_output

    async def run_streamed(
        self, question: str, context: str, session: SQLiteSession | None = None
    ) -> AsyncGenerator[str, None]:
        """流式生成，每次 yield 一段文本增量。"""
        prompt = _build_prompt(question, context)
        result = Runner.run_streamed(self.agent, prompt, session=session)
        async for event in result.stream_events():
            if event.type == "raw_response_event":
                try:
                    chunk = getattr(event.data, "delta", "")
                    if chunk:
                        yield chunk
                except Exception:
                    pass


def _build_prompt(question: str, context: str) -> str:
    return f"""检索到的相关文档内容:

{context}

---
用户问题: {question}

请基于以上文档内容回答问题。"""
