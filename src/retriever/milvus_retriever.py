"""Milvus 向量召回器：将文档写入 Milvus collection，通过向量相似度搜索召回。

与 VectorRetriever（numpy 内存实现）的核心差异：
  - 索引持久化在 Milvus 服务端，进程重启不丢失
  - 支持增量 upsert / 按 doc_id 删除，不必每次全量重建
  - 适合大规模向量（>10万），单机内存不再是瓶颈

依赖：pymilvus>=2.4，延迟导入（首次实例化时才 import）。
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np

from src.common.exceptions import EmbeddingError, RetrievalError
from src.document_parser.base import Document
from src.retriever.base import BaseRetriever, RetrieveResult
from src.retriever.registry import retriever_registry

EmbedFunc = Callable[[List[str]], List[List[float]]]


def _fake_embed(texts: List[str], dim: int = 8) -> List[List[float]]:
    """测试用假嵌入：随机生成单位向量，保证可复现性较差，仅用于集成测试。"""
    rng = np.random.default_rng(hash(texts[0]) % (2**32) if texts else 42)
    raw = rng.normal(size=(len(texts), dim)).astype(np.float32)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (raw / norms).tolist()


@retriever_registry.register("milvus")
class MilvusRetriever(BaseRetriever):
    """基于 Milvus 向量数据库的召回器。

    典型用法::

        retriever = MilvusRetriever(
            embed_func=my_embed_func,
            dim=1024,
            host="localhost",
            port=19530,
            collection="rag_chunks",
        )
        retriever.build_index(documents)       # 写入 Milvus
        results = retriever.retrieve("你好")    # 从 Milvus 搜索
    """

    def __init__(
        self,
        embed_func: Optional[EmbedFunc] = None,
        dim: int = 1024,
        host: str = "localhost",
        port: int = 19530,
        user: str = "",
        password: str = "",
        database: str = "default",
        collection: str = "rag_chunks",
        metric_type: str = "COSINE",
        index_type: str = "IVF_FLAT",
        nlist: int = 1024,
        nprobe: int = 16,
        top_k: int = 8,
        **_kwargs: Any,
    ) -> None:
        """
        Args:
            embed_func: 句嵌入函数，签名 ``Callable[[List[str]], List[List[float]]]``。
                        为 None 时使用内置的随机假嵌入（仅用于单元测试）。
            dim:        向量维度，必须和 embed_func 输出一致。
            host:       Milvus 服务地址。
            port:       Milvus 服务端口。
            user:       用户名（Milvus 2.4+ 鉴权，单机版可留空）。
            password:   密码。
            database:   Milvus database 名，默认 "default"。
            collection: collection 名，不存在则自动创建。
            metric_type: 相似度度量：COSINE / IP / L2。
            index_type: 向量索引类型：IVF_FLAT / HNSW / DISKANN 等。
            nlist:      IVF_FLAT 聚类数，数据量小时调小如 64。
            nprobe:     搜索时探测的聚类数，越大越准但越慢。
            top_k:      默认返回的 top_k。
        """
        # --- 延迟导入 pymilvus，没装也不影响其他模块 ---
        try:
            from pymilvus import (  # noqa: WPS433
                Collection,
                CollectionSchema,
                DataType,
                FieldSchema,
                connections,
                utility,
            )
        except ImportError as e:
            raise ImportError(
                "pymilvus 未安装，请先执行: pip install pymilvus>=2.4"
            ) from e

        self._Collection = Collection
        self._CollectionSchema = CollectionSchema
        self._DataType = DataType
        self._FieldSchema = FieldSchema
        self._connections = connections
        self._utility = utility

        self.embed_func: EmbedFunc = embed_func or (lambda texts: _fake_embed(texts, dim))
        self.dim = int(dim)
        self.host = host
        self.port = int(port)
        self.user = user
        self.password = password
        self.database = database
        self.collection_name = collection
        self.metric_type = metric_type
        self.index_type = index_type
        self.nlist = int(nlist)
        self.nprobe = int(nprobe)
        self.top_k = int(top_k)

        self._collection: Optional[Collection] = None
        self._alias: str = f"rag_{database}_{collection}"

    # ================================================================ 连接 & Collection 管理
    def _connect(self) -> None:
        """建立到 Milvus 的连接（如果已存在则复用别名）。"""
        if self._connections.has_connection(self._alias):
            return
        self._connections.connect(
            alias=self._alias,
            host=self.host,
            port=str(self.port),
            user=self.user or None,
            password=self.password or None,
            db_name=self.database,
        )

    def _ensure_collection(self) -> None:
        """确保 collection 存在，不存在则创建并建索引。"""
        self._connect()
        if self._utility.has_collection(self.collection_name, using=self._alias):
            self._collection = self._Collection(self.collection_name, using=self._alias)
            return

        fields = [
            self._FieldSchema(
                name="doc_id",
                dtype=self._DataType.VARCHAR,
                is_primary=True,
                max_length=512,
            ),
            self._FieldSchema(
                name="content",
                dtype=self._DataType.VARCHAR,
                max_length=65535,
            ),
            self._FieldSchema(
                name="metadata",
                dtype=self._DataType.JSON,
            ),
            self._FieldSchema(
                name="vector",
                dtype=self._DataType.FLOAT_VECTOR,
                dim=self.dim,
            ),
        ]
        schema = self._CollectionSchema(fields, description="RAG chunks")
        self._collection = self._Collection(
            self.collection_name, schema, using=self._alias
        )

        # 建向量索引
        index_params = self._collection.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type=self.index_type,
            metric_type=self.metric_type,
            params={"nlist": self.nlist},
        )
        self._collection.create_index(index_params)

    # ================================================================ 数据写入
    def build_index(
        self,
        documents: List[Document],
        drop_existing: bool = False,
        batch_size: int = 256,
    ) -> None:
        """把文档写入 Milvus collection 并加载到内存。

        Args:
            documents:     待写入的文档列表
            drop_existing: True 时先删后建（全量重建）；False 时 upsert
            batch_size:    每批写入的文档数，Milvus 单次 insert < 256MB 更安全
        """
        if not documents:
            # 空文档不报错，保证幂等
            return

        if drop_existing:
            self._connect()
            if self._utility.has_collection(self.collection_name, using=self._alias):
                self._Collection(self.collection_name, using=self._alias).drop()
            self._collection = None

        if self._collection is None:
            self._ensure_collection()

        assert self._collection is not None  # mypy 友好

        # 分批嵌入 + 写入
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            embeddings = self.embed_func([d.content for d in batch])

            if len(embeddings) != len(batch):
                raise EmbeddingError(
                    f"embed_func 返回数量 ({len(embeddings)}) 与文档数 ({len(batch)}) 不一致"
                )

            if embeddings and len(embeddings[0]) != self.dim:
                raise EmbeddingError(
                    f"embed_func 输出维度 {len(embeddings[0])} 与配置 dim={self.dim} 不匹配"
                )

            data = [
                [d.doc_id for d in batch],
                [d.content for d in batch],
                [d.metadata for d in batch],
                embeddings,
            ]
            self._collection.insert(data)

        self._collection.flush()
        self._collection.load()

    def upsert(self, documents: List[Document]) -> None:
        """增量 upsert（doc_id 已存在则更新，不存在则新增）。"""
        self.build_index(documents, drop_existing=False)

    def delete_by_ids(self, doc_ids: List[str]) -> None:
        """按 doc_id 删除指定文档。"""
        if self._collection is None:
            raise RetrievalError("Milvus collection 未初始化")
        if not doc_ids:
            return
        ids_str = ", ".join(f'"{did}"' for did in doc_ids)
        self._collection.delete(expr=f'doc_id in [{ids_str}]')
        self._collection.flush()

    # ================================================================ 召回
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[RetrieveResult]:
        if self._collection is None:
            raise RetrievalError("Milvus collection 未初始化，请先调用 build_index()")

        k = top_k or self.top_k
        q_emb = self.embed_func([query])
        q_vec = np.array(q_emb, dtype=np.float32).flatten()

        if q_vec.shape[0] != self.dim:
            raise EmbeddingError(
                f"query 向量维度 {q_vec.shape[0]} 与 dim={self.dim} 不匹配"
            )

        search_params = {
            "metric_type": self.metric_type,
            "params": {"nprobe": self.nprobe},
        }

        results = self._collection.search(
            data=[q_vec.tolist()],
            anns_field="vector",
            param=search_params,
            limit=k,
            output_fields=["content", "metadata"],
        )

        # Milvus search 返回的 hits 已按度量排序（COSINE/IP 降序、L2 升序），
        # 过滤规则需按度量区分：
        #   - COSINE / IP：相似度，<=0 视为不相关，跳过
        #   - L2：距离，>=0 都有效（0 = 完全相同向量），不按正负过滤
        is_l2 = self.metric_type.upper() == "L2"

        output: List[RetrieveResult] = []
        for hits in results:
            for hit in hits:
                score = float(hit.score)
                if not is_l2 and score <= 0:
                    continue
                meta: Dict[str, Any] = hit.entity.get("metadata", {}) or {}
                meta["retriever"] = "milvus"
                output.append(RetrieveResult(
                    doc_id=hit.id,
                    content=hit.entity.get("content", ""),
                    score=score,
                    metadata=meta,
                ))
        return output