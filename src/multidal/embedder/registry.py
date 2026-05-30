from __future__ import annotations

import logging
import threading

import torch.nn as nn

logger = logging.getLogger(__name__)


class ModelRegistry:
    """线程安全的模型单例注册中心。"""

    _instance: ModelRegistry | None = None
    _lock = threading.Lock()

    def __new__(cls) -> ModelRegistry:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._models = {}
        return cls._instance

    def get(self, name: str) -> nn.Module | None:
        return self._models.get(name)

    def put(self, name: str, model: nn.Module) -> None:
        self._models[name] = model
        logger.info("Model registered: %s", name)

    def remove(self, name: str) -> None:
        self._models.pop(name, None)
        logger.info("Model unloaded: %s", name)

    @property
    def loaded(self) -> list[str]:
        return list(self._models.keys())
