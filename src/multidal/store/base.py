"""向量存储抽象基类。

本模块定义 ``VectorStore`` ABC,作为 Milvus / Chroma / FAISS 等多种后端的
统一接口。理论上所有具体后端都应实现 ``insert`` / ``search`` / ``delete_collection``
三方法。

注意:当前 ``MilvusStore`` / ``ChromaStore`` 的 ``insert`` 签名是
``(chunks, kb_id)``(根据 kb_id 自动构造 ``_text`` / ``_image`` 双 Collection),
而本 ABC 定义的是 ``(collection, chunks)``,存在语义差异。
具体后端根据自身 API 灵活适配,Orchestrator 侧只关心 ``search`` 行为一致。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.multidal.schema.embedding import EmbeddedChunk
from src.multidal.schema.retrieval import RecallResult


class VectorStore(ABC):
    """向量存储后端的统一抽象接口。"""

    @abstractmethod
    def insert(self, collection: str, chunks: list[EmbeddedChunk]) -> list[str]:
        """批量插入向量到指定 Collection。

        Args:
            collection: Collection 名称(具体后端对命名规则可能有约束)。
            chunks: 已向量化的数据块列表。

        Returns:
            list[str]: 实际写入的主键 ID 列表(可能与 ``chunks`` 顺序一致)。
        """
        ...

    @abstractmethod
    def search(
        self, collection: str, query_vector: list[float], top_k: int = 10
    ) -> list[RecallResult]:
        """向量相似度检索。

        Args:
            collection: 目标 Collection 名称。
            query_vector: 查询向量(维度必须与 Collection 一致)。
            top_k: 返回的最大结果数。

        Returns:
            list[RecallResult]: 按相似度降序的命中结果。
        """
        ...

    @abstractmethod
    def delete_collection(self, collection: str) -> None:
        """删除整个 Collection(谨慎使用,不可恢复)。

        Args:
            collection: 要删除的 Collection 名称。
        """
        ...
