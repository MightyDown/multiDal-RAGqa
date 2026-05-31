from __future__ import annotations

import logging
import os

from agents import Agent, ModelSettings, set_default_openai_client
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from openai import AsyncOpenAI

from src.multidal.config import settings

logger = logging.getLogger(__name__)

# openai-agents SDK 要求设置此环境变量
os.environ.setdefault("OPENAI_API_KEY", settings.llm_api_key)

# 配置 openai-agents SDK 使用自定义 LLM 端点
_openai_client: AsyncOpenAI | None = None
_chat_model: OpenAIChatCompletionsModel | None = None

_small_client: AsyncOpenAI | None = None
_small_model: OpenAIChatCompletionsModel | None = None


def _get_chat_model() -> OpenAIChatCompletionsModel:
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
    """小模型 Agent，自动禁用思考过程。"""
    model = _get_small_chat_model()
    return Agent(
        name=name,
        model=model,
        model_settings=ModelSettings(extra_body={"enable_thinking": False}),
        instructions=instructions,
        tools=tools or [],
    )


class BaseAgent:
    """参考 keywords.py 模式：封装 openai-agents SDK 的 Agent 构建与运行。"""

    def __init__(self, name: str, instructions: str) -> None:
        self.name = name
        self._agent = Agent(
            name=name,
            model=_get_chat_model(),
            instructions=instructions,
        )

    @property
    def agent(self) -> Agent:
        return self._agent
