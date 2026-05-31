from __future__ import annotations

import logging
import time

import requests

from src.multidal.config import settings

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 30
DEFAULT_MAX_TOKENS = 512


def chat(prompt: str, system: str = "", max_tokens: int = DEFAULT_MAX_TOKENS, **kwargs) -> str:
    """Call Qwen3-0.6B via Moark API, return full content as string."""
    url = f"{settings.text_embedding_api_base}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {settings.text_embedding_api_key}",
    }

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": settings.small_llm_model,
        "messages": messages,
        "stream": True,
        "max_tokens": max_tokens,
        "temperature": kwargs.get("temperature", 0.7),
        "top_p": kwargs.get("top_p", 0.9),
        "frequency_penalty": kwargs.get("frequency_penalty", 0),
        "presence_penalty": kwargs.get("presence_penalty", 0),
        "stop": kwargs.get("stop"),
    }

    for attempt in range(2):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=DEFAULT_TIMEOUT, stream=True)
            resp.raise_for_status()
            content = _read_stream(resp)
            return content
        except Exception as e:
            logger.warning("small_model.chat attempt %d failed: %s", attempt + 1, e)
            if attempt == 0:
                time.sleep(1)
                continue
            raise

    return ""


def _read_stream(resp: requests.Response) -> str:
    import json

    full = []
    for line in resp.iter_lines():
        if not line or line.strip() == "data: [DONE]":
            continue
        if line.startswith("data: "):
            try:
                data = json.loads(line[6:])
            except Exception:
                continue
            if delta := data.get("choices", [{}])[0].get("delta", {}).get("content"):
                full.append(delta)
    return "".join(full)