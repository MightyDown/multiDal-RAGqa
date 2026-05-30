from __future__ import annotations

import logging
import os

from agents import Agent, set_default_openai_client
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from openai import AsyncOpenAI

from src.multidal.config import settings

logger = logging.getLogger(__name__)

# openai-agents SDK 要求设置此环境变量
os.environ.setdefault("OPENAI_API_KEY", settings.llm_api_key)

# 配置 openai-agents SDK 使用自定义 LLM 端点
_openai_client: AsyncOpenAI | None = None
_chat_model: OpenAIChatCompletionsModel | None = None


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
