"""智能体基础设施 - LLM 客户端与 Agent 基类。

本模块集中管理与 openai-agents SDK 相关的全局状态:
    - ``_get_chat_model``: 主对话模型(用于 QueryAgent / IngestAgent)。
    - ``_get_small_chat_model``: 小模型(用于 KB 路由、问题改写、会话命名,关闭 thinking)。
    - ``_get_small_agent``: 小模型 Agent 工厂。
    - ``BaseAgent``: 主 LLM Agent 的基类封装。
"""

from __future__ import annotations

import logging
import os

from agents import Agent, ModelSettings, set_default_openai_client
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from agents.tracing import set_tracing_disabled
from openai import AsyncOpenAI

from src.multidal.config import settings

logger = logging.getLogger(__name__)

# openai-agents SDK 要求设置此环境变量才能正常 import。
os.environ.setdefault("OPENAI_API_KEY", settings.llm_api_key)

# 关闭 Tracing：openai-agents 默认会把 span 异步上报到 platform.openai.com，
# 容器内不可达，会触发 "Tracing: request failed: timed out" 并导致 Runner.run 抛 400。
# 项目使用国内 MiniMax + Moark，无需 OpenAI 官方 trace。
set_tracing_disabled(True)

# 主对话 LLM 客户端与模型(全局单例)。
_openai_client: AsyncOpenAI | None = None
_chat_model: OpenAIChatCompletionsModel | None = None

# 小模型客户端与模型(全局单例,用于 KB 路由 / 改写 / 命名)。
_small_client: AsyncOpenAI | None = None
_small_model: OpenAIChatCompletionsModel | None = None


def _get_chat_model() -> OpenAIChatCompletionsModel:
    """惰性初始化主对话模型并注册为 SDK 默认客户端。

    流程:
        1. 首次调用时创建 ``AsyncOpenAI`` 客户端(指向 ``settings.llm_base_url``);
        2. 调用 ``set_default_openai_client`` 让 SDK 后续创建 Agent 时复用;
        3. 构造 ``OpenAIChatCompletionsModel`` 实例并缓存。

    Returns:
        OpenAIChatCompletionsModel: 主 LLM 模型包装对象。
    """
    global _openai_client, _chat_model
    if _chat_model is None:
        _openai_client = AsyncOpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
        )
        set_default_openai_client(_openai_client)
        _chat_model = OpenAIChatCompletionsModel(
            model=settings.llm_model,
            openai_client=_openai_client,
        )
        logger.info("LLM configured: base_url=%s model=%s", settings.llm_base_url, settings.llm_model)
    return _chat_model


def _get_small_chat_model() -> OpenAIChatCompletionsModel:
    """惰性初始化小模型(默认 Qwen3-0.6B)客户端与模型对象。

    小模型与主 LLM 使用不同 base_url(``text_embedding_api_base``),因为
    Moark 平台将嵌入 API 与生成 API 共用同一网关。

    Returns:
        OpenAIChatCompletionsModel: 小模型包装对象。
    """
    global _small_client, _small_model
    if _small_model is None:
        _small_client = AsyncOpenAI(
            api_key=settings.text_embedding_api_key,
            base_url=settings.text_embedding_api_base,
        )
        set_default_openai_client(_small_client)
        _small_model = OpenAIChatCompletionsModel(
            model=settings.small_llm_model,
            openai_client=_small_client,
        )
        logger.info("Small LLM configured: base_url=%s model=%s", settings.text_embedding_api_base, settings.small_llm_model)
    return _small_model


def _get_small_agent(name: str, instructions: str, tools: list | None = None) -> Agent:
    """小模型 Agent 工厂函数。

    Args:
        name: Agent 名称(显示用)。
        instructions: 系统提示词。
        tools: 可挂载的 function_tool 列表(KBRouter 用到 search/get_doc_info)。

    Returns:
        Agent: 配置好的小模型 Agent 实例。
    """
    model = _get_small_chat_model()
    return Agent(
        name=name,
        model=model,
        # 关闭思考:小模型只做"分类/改写/命名"等轻量任务,无需 CoT
        model_settings=ModelSettings(extra_body={"enable_thinking": False}),
        instructions=instructions,
        tools=tools or [],
    )


class BaseAgent:
    """主 LLM Agent 基类,封装 openai-agents SDK Agent 的构建与基本访问。

    使用方式::

        class MyAgent(BaseAgent):
            def __init__(self):
                super().__init__(name="...", instructions="...")

            async def run(self, ...):
                result = await Runner.run(self.agent, ...)

    Attributes:
        name: Agent 显示名。
        _agent: 内部 openai-agents Agent 实例。
    """

    def __init__(self, name: str, instructions: str) -> None:
        """构造 Agent 实例(模型自动通过 ``_get_chat_model`` 惰性初始化)。

        Args:
            name: Agent 名称。
            instructions: 系统提示词。
        """
        self.name = name
        self._agent = Agent(
            name=name,
            model=_get_chat_model(),
            instructions=instructions,
            # 关闭主 LLM 的思考链模式,确保回答直接落到 content 字段,
            # 避免 reasoning_content 与 content 二选一导致前端收不到流式 delta。
            # 与小模型 _get_small_agent 保持一致。
            model_settings=ModelSettings(extra_body={"enable_thinking": False}),
        )

    @property
    def agent(self) -> Agent:
        """暴露内部 Agent 给子类与外部调用方(openai-agents Runner 需要)。"""
        return self._agent
