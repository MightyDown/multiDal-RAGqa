from __future__ import annotations

from abc import ABC, abstractmethod

from src.multidal.schema.embedding import EmbeddedChunk
from src.multidal.schema.retrieval import RecallResult


class VectorStore(ABC):
    @abstractmethod
    def insert(self, collection: str, chunks: list[EmbeddedChunk]) -> list[str]: ...

    @abstractmethod
    def search(
        self, collection: str, query_vector: list[float], top_k: int = 10
    ) -> list[RecallResult]: ...

    @abstractmethod
    def delete_collection(self, collection: str) -> None: ...
