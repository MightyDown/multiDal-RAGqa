"""Milvus 向量存储。

本模块实现 ``MilvusStore``(继承自 ``Stage`` 与 ``VectorStore``),封装 pymilvus 的
Collection 管理、IVF_FLAT 索引创建、向量检索等核心操作。

Collection 设计:
    - 每个 KB 对应两个 Collection:``{kb_id}_text``(BGE 空间) + ``{kb_id}_image``(CLIP 空间)。
    - 维度按 Collection 分别设置(从 settings 读取)。
    - 检索 metric:内积(IP)——BGE/CLIP 向量归一化后,IP 等价于 cosine。
    - 索引类型:IVF_FLAT,nlist=128(中小规模 KB 适用,百万级以下无需调大)。
"""

from __future__ import annotations

import logging

from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility

from src.multidal.config import settings
from src.multidal.pipeline.base import PipelineContext, Stage
from src.multidal.schema.embedding import EmbeddedChunk
from src.multidal.schema.retrieval import RecallResult
from src.multidal.store.base import VectorStore

logger = logging.getLogger(__name__)

# Milvus Collection 字段名常量(集中管理,避免散落字符串字面量)。
_PRIMARY = "chunk_id"
_VECTOR = "vector"


def _schema(dim: int) -> CollectionSchema:
    """构造标准 Milvus Collection Schema。

    字段:
        - chunk_id: 主键(VARCHAR,长度 64)。
        - content: 文本/描述内容(VARCHAR,长度 8192,超出将被截断)。
        - modality: 模态(``text`` / ``image``)。
        - kb_id / doc_id / page: 来源元数据,用于过滤与回溯。
        - image_path: 仅图片模态有值,前端回显。
        - vector: FLOAT_VECTOR,维度由参数决定。

    Args:
        dim: 向量维度(应与 ``text_embedding_dim`` 或 ``image_embedding_dim`` 一致)。

    Returns:
        CollectionSchema: 可用于 ``Collection(name, schema=...)``。
    """
    return CollectionSchema(
        fields=[
            FieldSchema(name=_PRIMARY, dtype=DataType.VARCHAR, is_primary=True, max_length=64),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=8192),
            FieldSchema(name="modality", dtype=DataType.VARCHAR, max_length=16),
            FieldSchema(name="kb_id", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="page", dtype=DataType.INT64),
            FieldSchema(name="image_path", dtype=DataType.VARCHAR, max_length=512, is_nullable=True),
            FieldSchema(name=_VECTOR, dtype=DataType.FLOAT_VECTOR, dim=dim),
        ]
    )


class MilvusStore(Stage, VectorStore):
    """Milvus 向量存储实现,也是 ``Stage`` 子类(供 Orchestrator 调用)。"""

    name = "store"

    def __init__(self) -> None:
        """从全局配置读取 Milvus 连接信息与各模态的向量维度。"""
        self._host = settings.milvus_host
        self._port = str(settings.milvus_port)
        self._connected = False
        self._dim_text = settings.text_embedding_dim
        self._dim_image = settings.image_embedding_dim

    def validate(self) -> bool:
        """通过尝试建立连接验证 Milvus 可用性。

        Returns:
            bool: 连接成功为 True,任何异常都降级为 False。
        """
        try:
            self._connect()
            return True
        except Exception:
            logger.warning("Milvus not reachable at %s:%s", self._host, self._port)
            return False

    def process(self, ctx: PipelineContext) -> PipelineContext:
        """Stage 入口:将 ``ctx.embedded`` 全部 chunk 写入对应 KB 的双 Collection。

        Args:
            ctx: 流水线上下文,需含 ``embedded`` 与 ``kb_id``。

        Returns:
            PipelineContext: 透传上下文(不修改任何字段)。

        Raises:
            ValueError: ``ctx.embedded`` 为 None 时。
        """
        self._connect()
        if ctx.embedded is None:
            raise ValueError("PipelineContext.embedded is None")
        self.insert(ctx.embedded, ctx.kb_id)
        logger.info("Stored %d chunks into kb=%s", len(ctx.embedded), ctx.kb_id)
        return ctx

    def insert(self, chunks: list[EmbeddedChunk], kb_id: str) -> list[str]:
        """批量插入 chunks,按模态拆分到 ``{kb}_text`` 与 ``{kb}_image`` 双 Collection。

        Args:
            chunks: 待写入的已向量化数据块。
            kb_id: 目标知识库 ID(同时是 Collection 名前缀)。

        Returns:
            list[str]: 写入的主键 ID 列表(与 chunks 一一对应)。
        """
        text_coll = self._ensure_collection(f"{kb_id}_text", self._dim_text)
        image_coll = self._ensure_collection(f"{kb_id}_image", self._dim_image)

        # 按模态分桶:避免对同一 Collection 混合不同维度的向量
        text_rows, image_rows, ids = [], [], []
        for c in chunks:
            row = {
                _PRIMARY: c.chunk_id,
                "content": c.content[:8192],  # 截断到 schema 限制内
                "modality": c.modality,
                "kb_id": c.kb_id,
                "doc_id": c.doc_id,
                "page": c.page,
                "image_path": c.image_path or "",
                _VECTOR: c.embedding.vector,
            }
            if c.modality == "text":
                text_rows.append(row)
            else:
                image_rows.append(row)
            ids.append(c.chunk_id)

        # 显式 flush:确保数据立即可查(Milvus 默认有 buffer,可能延迟可见)
        if text_rows:
            text_coll.insert(text_rows)
            text_coll.flush()
        if image_rows:
            image_coll.insert(image_rows)
            image_coll.flush()
        return ids

    def search(
        self, collection_name: str, query_vector: list[float], top_k: int = 10
    ) -> list[RecallResult]:
        """在指定 Collection 上做向量近邻检索。

        检索参数:
            - metric_type=IP(内积):等价于 cosine(向量已归一化)。
            - nprobe=10:IVF 索引的探测桶数,精度/速度折中。

        Args:
            collection_name: Collection 名称(完整,含 ``_text`` / ``_image`` 后缀)。
            query_vector: 查询向量。
            top_k: 返回的最大结果数。

        Returns:
            list[RecallResult]: 按相似度降序的命中结果。
        """
        self._connect()
        coll = Collection(collection_name)
        # load 将 Collection 从磁盘加载到内存(query 必需)
        coll.load()
        results = coll.search(
            data=[query_vector],
            anns_field=_VECTOR,
            param={"metric_type": "IP", "params": {"nprobe": 10}},
            limit=top_k,
            output_fields=["content", "modality", "kb_id", "doc_id", "page", "image_path"],
        )
        out = []
        for hit in results[0]:
            img_path = hit.entity.get("image_path", "")
            out.append(
                RecallResult(
                    chunk_id=hit.id,
                    content=hit.entity.get("content", ""),
                    modality=hit.entity.get("modality", "text"),
                    source="milvus",
                    score=float(hit.distance),
                    kb_id=hit.entity.get("kb_id", ""),
                    doc_id=hit.entity.get("doc_id", ""),
                    page=hit.entity.get("page", 1),
                    image_path=img_path if img_path else None,
                )
            )
        return out

    def query_by_doc(self, kb_id: str, doc_id: str, limit: int = 500) -> list[dict]:
        """按 doc_id 拉取某个 KB 下某文档的全部 chunk(文本 + 图片)。

        用于"按文档列出所有引用"、"删除前的预览"等场景。

        Args:
            kb_id: 知识库 ID。
            doc_id: 文档 ID。
            limit: 单 Collection 的最大返回行数。

        Returns:
            list[dict]: 原始 Milvus 行字典,按 page 升序。
        """
        self._connect()
        rows = []
        for suffix in ("_text", "_image"):
            name = f"{kb_id}{suffix}"
            if not utility.has_collection(name):
                continue
            coll = Collection(name)
            coll.load()
            try:
                results = coll.query(
                    expr=f'doc_id == "{doc_id}"',
                    output_fields=["chunk_id", "content", "modality", "kb_id", "doc_id", "page"],
                    limit=limit,
                )
                rows.extend(results)
            except Exception:
                logger.warning("query_by_doc failed for %s", name)
        rows.sort(key=lambda r: r.get("page", 1))
        return rows

    def delete_by_doc(self, kb_id: str, doc_id: str) -> int:
        """删除某个 KB 下某文档的所有 chunk(文本 + 图片)。

        Args:
            kb_id: 知识库 ID。
            doc_id: 文档 ID。

        Returns:
            int: 实际删除的向量条数。
        """
        self._connect()
        deleted = 0
        for suffix in ("_text", "_image"):
            name = f"{kb_id}{suffix}"
            if not utility.has_collection(name):
                continue
            coll = Collection(name)
            coll.load()
            try:
                # 先 query 拿到所有 chunk_id,再 delete(避免 delete 表达式过长)
                results = coll.query(
                    expr=f'doc_id == "{doc_id}"',
                    output_fields=["chunk_id"],
                    limit=10000,
                )
                ids = [r["chunk_id"] for r in results]
                if ids:
                    coll.delete(f'chunk_id in {ids}')
                    coll.flush()
                    deleted += len(ids)
            except Exception:
                logger.warning("delete_by_doc failed for %s", name)
        return deleted

    def delete_collection(self, collection_name: str) -> None:
        """删除整个 Collection(慎用)。

        Args:
            collection_name: Collection 名称。
        """
        self._connect()
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)

    def _connect(self) -> None:
        """惰性连接:仅在首次访问时建立 pymilvus 连接,后续操作复用。

        避免在 ``__init__`` 时连接,降低 import 失败的风险。
        """
        if self._connected:
            return
        connections.connect(host=self._host, port=self._port)
        self._connected = True

    def _ensure_collection(self, name: str, dim: int) -> Collection:
        """按需创建 Collection 与 IVF_FLAT 索引。

        Args:
            name: Collection 名称。
            dim: 向量维度(必须与该 Collection 内已有数据一致)。

        Returns:
            Collection: 已就绪的 Collection 对象。
        """
        if utility.has_collection(name):
            return Collection(name)
        coll = Collection(name, schema=_schema(dim))
        # IVF_FLAT:精确检索,无压缩,适合中小规模(召回质量优先)
        coll.create_index(_VECTOR, {
            "metric_type": "IP", "index_type": "IVF_FLAT", "params": {"nlist": 128},
        })
        return coll
