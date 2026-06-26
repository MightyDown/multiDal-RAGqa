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
from src.multidal.config import settings

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
                    # 部分模型(开启 thinking)的实际回答会落在 reasoning_content 字段
                    chunk = (
                        getattr(event.data, "delta", "")
                        or getattr(event.data, "reasoning_content", "")
                    )
                    if chunk:
                        yield chunk
                except Exception:
                    # 单个事件解析失败不阻断流,继续消费后续事件
                    pass


class VLMAgent:
    """多模态问答智能体 — 直接调用 Moark VLM API，支持图片+文本混合输入。

    与 QueryAgent（纯文本 LLM）的区别：
    - 输入不是 context 字符串，而是 OpenAI Vision 格式的 content 数组
    - 图片以 base64 data URL 形式直接传给模型，不走文字描述
    - 用于检索结果包含图片候选时，实现真正的"图文并茂"问答
    """

    def __init__(self) -> None:
        self._api_base = settings.vlm_api_base
        self._api_key = settings.vlm_api_key
        self._model = settings.vlm_model
        self._max_tokens = settings.vlm_max_tokens
        self._temperature = settings.vlm_temperature

    @staticmethod
    def _image_to_base64(image_path: str) -> str | None:
        """将本地图片路径转为 base64 data URL。

        image_path 可能是两种格式之一：
          - 绝对磁盘路径：``/app/docs/{task_id}/images/filename.jpg``
          - URL 路径：``/raw/{task_id}/images/filename.jpg``
        实际文件位置：``{project_root}/docs/{task_id}/images/filename.jpg``
        """
        import base64
        from pathlib import Path as _Path

        try:
            p = _Path(image_path)
            # 已存在 → 直接使用
            if p.is_absolute() and p.exists():
                disk_path = p
            elif image_path.startswith("/raw/"):
                rel = image_path.replace("/raw/", "", 1)
                disk_path = settings.project_root / "docs" / rel
            else:
                # 兜底：去掉前导斜杠拼到 docs 下
                rel = image_path.lstrip("/")
                disk_path = settings.project_root / "docs" / rel

            if not disk_path.exists():
                logger.warning("VLM image not found: %s", disk_path)
                return None

            with open(disk_path, "rb") as f:
                img_data = base64.b64encode(f.read()).decode("utf-8")

            # 根据扩展名推断 MIME 类型
            ext = disk_path.suffix.lower()
            mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                        ".png": "image/png", ".gif": "image/gif",
                        ".webp": "image/webp", ".bmp": "image/bmp"}
            mime = mime_map.get(ext, "image/jpeg")
            return f"data:{mime};base64,{img_data}"
        except Exception:
            logger.exception("VLM: failed to encode image %s", image_path)
            return None

    @staticmethod
    def build_messages(
        question: str,
        ranked: list,
    ) -> list[dict]:
        """用精排结果构建 OpenAI Vision 格式的 messages。

        文本段 → type: text
        图片段 → type: image_url (base64 data URL)
        """
        content: list[dict] = []

        for r in ranked:
            if r.modality == "image" and r.image_path:
                b64_url = VLMAgent._image_to_base64(r.image_path)
                if b64_url:
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": b64_url, "detail": "low"},
                    })
                    # 图片后面跟一段简短的文字说明来源
                    content.append({
                        "type": "text",
                        "text": f"[参考图片 | {r.kb_id} | 第{r.page}页]",
                    })
                else:
                    # 图片读不到，降级为文字描述
                    content.append({
                        "type": "text",
                        "text": f"[图片描述 | {r.kb_id} | p{r.page}]\n{r.content}",
                    })
            else:
                content.append({
                    "type": "text",
                    "text": f"[文档 | {r.kb_id} | p{r.page}]\n{r.content}",
                })

        # 最终 prompt
        content.append({
            "type": "text",
            "text": (
                "以上是检索到的相关文档内容（包含文本和图片）。\n"
                "请基于以上内容回答问题。引用时注明来源的页码。\n\n"
                f"用户问题: {question}"
            ),
        })

        return [{"role": "user", "content": content}]

    def generate(self, question: str, ranked: list) -> str:
        """非流式 VLM 调用,返回完整回答。

        同时兼容 ``content`` 与 ``reasoning_content`` 字段(thinking 模式下
        最终回答会落在 ``content``,但部分早期版本会落在 ``reasoning_content``)。
        """
        import requests as _req

        messages = self.build_messages(question, ranked)

        resp = _req.post(
            f"{self._api_base}/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            json={
                "model": self._model,
                "messages": messages,
                "max_tokens": self._max_tokens,
                "temperature": self._temperature,
                "stream": False,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=120,
        )
        if not resp.ok:
            logger.error(
                "VLM %s returned %d: %s",
                self._model, resp.status_code, resp.text[:800],
            )
        resp.raise_for_status()
        data = resp.json()
        msg = data["choices"][0]["message"]
        return msg.get("content") or msg.get("reasoning_content", "")

    def generate_stream(self, question: str, ranked: list):
        """流式 VLM 调用，yield 文本增量。

        GLM-4.6V-Flash 等推理类 VLM 默认开启 thinking 模式,会把推理过程输出到
        ``reasoning_content`` 字段,而把最终回答输出到 ``content`` 字段。
        为兼容两种模式(无论是否开启 thinking),本方法同时读取这两个字段。
        """
        import requests as _req

        messages = self.build_messages(question, ranked)

        resp = _req.post(
            f"{self._api_base}/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            json={
                "model": self._model,
                "messages": messages,
                "max_tokens": self._max_tokens,
                "temperature": self._temperature,
                "stream": True,
                # 尝试关闭 thinking(Qwen3/GLM 系列支持),让回答直接落到 content
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=120,
            stream=True,
        )
        resp.raise_for_status()

        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            # iter_lines() 默认返回 bytes,统一解码为 str
            line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
            if line.startswith("data: "):
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                try:
                    import json as _json
                    chunk = _json.loads(data_str)
                    delta = chunk["choices"][0].get("delta", {})
                    # 同时读 content 与 reasoning_content,兼容 thinking 模式
                    content = delta.get("content", "") or delta.get("reasoning_content", "")
                    if content:
                        yield content
                except Exception:
                    continue


def _has_images(ranked: list) -> bool:
    """判断精排结果中是否包含图片模态的候选。"""
    return any(r.modality == "image" and r.image_path for r in ranked)


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
