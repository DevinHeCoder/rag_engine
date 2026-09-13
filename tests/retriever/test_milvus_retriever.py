"""MilvusRetriever 单元测试 —— 全量 mock pymilvus，无需真实 Milvus 服务。"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from src.document_parser.base import Document
from src.retriever.base import BaseRetriever, RetrieveResult
from src.retriever.milvus_retriever import MilvusRetriever, _fake_embed
from src.retriever.registry import retriever_registry
from src.common.exceptions import EmbeddingError, RetrievalError


# ====================================================================== Milvus Mock 实现
def _build_mock_pymilvus():
    """构造一套自包含的 mock pymilvus 模块结构，注入 sys.modules。"""
    from unittest.mock import MagicMock

    class MockCollection:
        _instances: dict = {}

        def __init__(self, name, schema=None, using="default"):
            self.name = name
            self.using = using
            self.schema = schema
            self._loaded = False
            self._index_params = None
            self._data = []
            MockCollection._instances[(using, name)] = self

        def prepare_index_params(self):
            return MagicMock()

        def create_index(self, index_params):
            self._index_params = index_params

        def insert(self, data):
            self._data.append(data)

        def flush(self):
            pass

        def load(self):
            self._loaded = True

        def is_loaded(self):
            return self._loaded

        def search(self, **kwargs):
            # 默认返回 1 条 COSINE 风格的正分数结果
            return [[
                MagicMock(id="doc_1", score=0.92, entity={
                    "content": "你好世界", "metadata": {"page": 1},
                }),
            ]]

        def delete(self, expr=""):
            pass

        def drop(self):
            MockCollection._instances.pop((self.using, self.name), None)

    class MockFieldSchema:
        def __init__(self, **kw):
            for k, v in kw.items():
                setattr(self, k, v)

    class MockCollectionSchema:
        def __init__(self, fields, description=""):
            self.fields = fields
            self.description = description

    MockDataType = MagicMock()
    MockDataType.VARCHAR = 0
    MockDataType.JSON = 1
    MockDataType.FLOAT_VECTOR = 2

    mock_connections = MagicMock()
    mock_connections.has_connection = lambda alias: False

    class MockUtility:
        @staticmethod
        def has_collection(name, using="default"):
            return (using, name) in MockCollection._instances

        @staticmethod
        def drop_collection(name, using="default"):
            MockCollection._instances.pop((using, name), None)

    return {
        "Collection": MockCollection,
        "CollectionSchema": MockCollectionSchema,
        "DataType": MockDataType,
        "FieldSchema": MockFieldSchema,
        "connections": mock_connections,
        "utility": MockUtility(),
    }


@pytest.fixture()
def mock_pymilvus(monkeypatch):
    """把整套 mock pymilvus 注入 sys.modules，并清理实例缓存。"""
    from unittest.mock import MagicMock

    mm = _build_mock_pymilvus()

    fake_pymilvus = MagicMock()
    fake_pymilvus.Collection = mm["Collection"]
    fake_pymilvus.CollectionSchema = mm["CollectionSchema"]
    fake_pymilvus.DataType = mm["DataType"]
    fake_pymilvus.FieldSchema = mm["FieldSchema"]
    fake_pymilvus.connections = mm["connections"]
    fake_pymilvus.utility = mm["utility"]

    sys.modules["pymilvus"] = fake_pymilvus
    sys.modules["pymilvus.connections"] = fake_pymilvus
    sys.modules["pymilvus.collection"] = fake_pymilvus
    sys.modules["pymilvus.schema"] = fake_pymilvus
    sys.modules["pymilvus.utility"] = fake_pymilvus

    mm["Collection"]._instances.clear()
    yield mm

    for k in list(sys.modules):
        if "pymilvus" in k:
            del sys.modules[k]


def _make_docs_and_embed(n: int = 5, dim: int = 8):
    docs = [
        Document(doc_id=f"doc_{i}", content=f"这是第 {i} 篇测试文档的内容", metadata={"source": f"test_{i}.txt"})
        for i in range(n)
    ]

    def embed(texts):
        return _fake_embed(texts, dim=dim)

    return docs, embed, dim


# ====================================================================== Fixtures
@pytest.fixture()
def docs_and_embed():
    return _make_docs_and_embed(n=5, dim=8)


# ====================================================================== 测试
class TestMilvusRegistration:
    def test_registered(self):
        assert retriever_registry.has("milvus")

    def test_is_base_subclass(self):
        assert issubclass(MilvusRetriever, BaseRetriever)

    def test_registry_create(self, mock_pymilvus):
        r = retriever_registry.create(
            "milvus",
            embed_func=lambda t: _fake_embed(t, 8),
            dim=8,
            collection="reg_test",
        )
        assert isinstance(r, MilvusRetriever)
        assert r.collection_name == "reg_test"
        assert r.dim == 8


class TestBuildIndex:
    def test_empty_noop(self, mock_pymilvus):
        r = MilvusRetriever(embed_func=lambda t: _fake_embed(t, 8), dim=8, collection="empty_test")
        r.build_index([])

    def test_creates_collection(self, mock_pymilvus, docs_and_embed):
        docs, embed_func, dim = docs_and_embed
        r = MilvusRetriever(embed_func=embed_func, dim=dim, collection="first_create")
        r.build_index(docs)

        mm = mock_pymilvus
        mm["connections"].connect.assert_called_once()
        assert (r._alias, "first_create") in mm["Collection"]._instances
        assert r._collection is not None
        assert r._collection._loaded is True

    def test_insert_batches(self, mock_pymilvus, docs_and_embed):
        docs, embed_func, dim = docs_and_embed
        r = MilvusRetriever(embed_func=embed_func, dim=dim, collection="batch_test")
        r.build_index(docs, batch_size=2)

        col = r._collection
        assert len(col._data) == 3  # 5 docs / 2 = 3 batches
        all_vecs = [v for batch in col._data for v in batch[3]]
        assert len(all_vecs) == 5
        assert len(all_vecs[0]) == dim

    def test_drop_existing_rebuilds(self, mock_pymilvus, docs_and_embed):
        docs, embed_func, dim = docs_and_embed
        r = MilvusRetriever(embed_func=embed_func, dim=dim, collection="drop_test")
        r.build_index(docs)
        r.build_index(docs, drop_existing=True)

        mm = mock_pymilvus
        assert (r._alias, "drop_test") in mm["Collection"]._instances
        # 旧 collection 已 drop，新 collection 的 data 又是新一轮 insert
        assert len(r._collection._data) > 0

    def test_embed_count_mismatch_raises(self, mock_pymilvus):
        def bad_embed(texts):
            return [[0.1, 0.2]]

        docs = [
            Document(doc_id="a", content="aaa", metadata={}),
            Document(doc_id="b", content="bbb", metadata={}),
        ]
        r = MilvusRetriever(embed_func=bad_embed, dim=2, collection="count_mismatch")
        with pytest.raises(EmbeddingError, match="返回数量"):
            r.build_index(docs)

    def test_dim_mismatch_raises(self, mock_pymilvus):
        def embed_dim3(texts):
            return _fake_embed(texts, dim=3)

        docs = [Document(doc_id="x", content="xx", metadata={})]
        r = MilvusRetriever(embed_func=embed_dim3, dim=8, collection="dim_mismatch")
        with pytest.raises(EmbeddingError, match="维度"):
            r.build_index(docs)


class TestRetrieve:
    def test_returns_retrieve_results(self, mock_pymilvus, docs_and_embed):
        docs, embed_func, dim = docs_and_embed
        r = MilvusRetriever(embed_func=embed_func, dim=dim, collection="ret_test")
        r.build_index(docs)

        results = r.retrieve("查询一下")
        assert isinstance(results, list)
        assert len(results) == 1
        item = results[0]
        assert isinstance(item, RetrieveResult)
        assert item.doc_id == "doc_1"
        assert item.content == "你好世界"
        assert item.score == 0.92
        assert item.metadata.get("retriever") == "milvus"

    def test_custom_top_k(self, mock_pymilvus, docs_and_embed):
        docs, embed_func, dim = docs_and_embed
        r = MilvusRetriever(embed_func=embed_func, dim=dim, collection="topk_test")
        r.build_index(docs)

        mock_pymilvus["Collection"]._instances[(r._alias, "topk_test")].search = MagicMock(
            return_value=[[
                MagicMock(id="a", score=0.9, entity={"content": "A", "metadata": {}}),
                MagicMock(id="b", score=0.8, entity={"content": "B", "metadata": {}}),
                MagicMock(id="c", score=0.7, entity={"content": "C", "metadata": {}}),
            ]]
        )
        results = r.retrieve("查询", top_k=3)
        assert len(results) == 3

    def test_skip_zero_score(self, mock_pymilvus, docs_and_embed):
        docs, embed_func, dim = docs_and_embed
        r = MilvusRetriever(embed_func=embed_func, dim=dim, collection="zero_test")
        r.build_index(docs)

        mock_pymilvus["Collection"]._instances[(r._alias, "zero_test")].search = MagicMock(
            return_value=[[
                MagicMock(id="a", score=0.0, entity={}),
                MagicMock(id="b", score=0.8, entity={"content": "B", "metadata": {}}),
            ]]
        )
        results = r.retrieve("q")
        assert len(results) == 1
        assert results[0].doc_id == "b"

    def test_retrieve_before_build_raises(self, mock_pymilvus):
        r = MilvusRetriever(embed_func=lambda t: _fake_embed(t, 8), dim=8, collection="no_build")
        with pytest.raises(RetrievalError, match="未初始化"):
            r.retrieve("q")


class TestHelpers:
    def test_upsert_delegates(self, mock_pymilvus, docs_and_embed):
        docs, embed_func, dim = docs_and_embed
        r = MilvusRetriever(embed_func=embed_func, dim=dim, collection="upsert_test")
        with patch.object(r, "build_index") as m:
            r.upsert(docs)
            m.assert_called_once_with(docs, drop_existing=False)

    def test_delete_by_ids(self, mock_pymilvus, docs_and_embed):
        docs, embed_func, dim = docs_and_embed
        r = MilvusRetriever(embed_func=embed_func, dim=dim, collection="del_test")
        r.build_index(docs)

        col = mock_pymilvus["Collection"]._instances[(r._alias, "del_test")]
        col.delete = MagicMock()
        r.delete_by_ids(["doc_1", "doc_2"])

        col.delete.assert_called_once()
        call_args = col.delete.call_args
        expr = call_args.kwargs.get("expr") if call_args.kwargs else call_args.args[0]
        assert "doc_1" in expr and "doc_2" in expr

    def test_delete_empty_noop(self, mock_pymilvus, docs_and_embed):
        docs, embed_func, dim = docs_and_embed
        r = MilvusRetriever(embed_func=embed_func, dim=dim, collection="del_empty")
        r.build_index(docs)
        r.delete_by_ids([])


class TestDefaultFakeEmbed:
    def test_no_embed_func_uses_fake(self, mock_pymilvus):
        r = MilvusRetriever(dim=16, collection="fake_embed_test")
        docs = [Document(doc_id=f"d{i}", content=f"text {i}", metadata={}) for i in range(3)]
        r.build_index(docs)
        results = r.retrieve("q")
        assert isinstance(results, list)


class TestInterfaceConsistency:
    def test_all_retrievers_share_same_base(self):
        from src.retriever.vector_retriever import VectorRetriever
        from src.retriever.bm25_retriever import BM25Retriever

        assert issubclass(MilvusRetriever, BaseRetriever)
        assert issubclass(VectorRetriever, BaseRetriever)
        assert issubclass(BM25Retriever, BaseRetriever)

    def test_hybrid_can_take_milvus(self, mock_pymilvus):
        """HybridRetriever 接收 BaseRetriever，MilvusRetriever 作为 vector 分支直接可用。"""
        from src.retriever.bm25_retriever import BM25Retriever
        from src.retriever.hybrid_retriever import HybridRetriever

        vector_docs, embed_func, dim = _make_docs_and_embed(n=5, dim=8)
        r_milvus = MilvusRetriever(embed_func=embed_func, dim=dim, collection="hybrid_test")
        r_milvus.build_index(vector_docs)

        bm25_docs, _, _ = _make_docs_and_embed(n=3, dim=8)
        r_bm25 = BM25Retriever(documents=bm25_docs)

        hybrid = HybridRetriever(vector_retriever=r_milvus, bm25_retriever=r_bm25)
        results = hybrid.retrieve("测试一下", top_k=5)
        assert isinstance(results, list)


class TestFactory:
    """build_retriever("milvus") 工厂接线：配置透传 + 自动建索引 + 缺参校验。"""

    def test_factory_creates_and_builds_index(self, mock_pymilvus, docs_and_embed):
        from src.retriever import build_retriever

        docs, embed_func, dim = docs_and_embed
        cfg = {
            "host": "10.0.0.8",
            "port": 19531,
            "collection": "factory_test",
            "dim": dim,
            "nprobe": 8,
        }
        r = build_retriever(
            "milvus", docs, embed_func=embed_func, top_k=5, milvus_cfg=cfg
        )
        assert isinstance(r, MilvusRetriever)
        # 配置已透传
        assert r.host == "10.0.0.8"
        assert r.port == 19531
        assert r.collection_name == "factory_test"
        assert r.dim == dim
        assert r.nprobe == 8
        # documents 已自动写入 Milvus
        col = mock_pymilvus["Collection"]._instances[(r._alias, "factory_test")]
        assert col is not None
        assert col._loaded is True

    def test_factory_requires_embed_func(self, mock_pymilvus, docs_and_embed):
        from src.common.exceptions import RetrievalError
        from src.retriever import build_retriever

        docs, _, dim = docs_and_embed
        with pytest.raises(RetrievalError, match="embed_func"):
            build_retriever("milvus", docs, embed_func=None, milvus_cfg={"dim": dim})

    def test_factory_empty_docs_no_index(self, mock_pymilvus):
        from src.retriever import build_retriever

        r = build_retriever(
            "milvus",
            None,
            embed_func=lambda t: _fake_embed(t, 8),
            milvus_cfg={"dim": 8, "collection": "factory_empty"},
        )
        assert r._collection is None  # 未建索引，retrieve 会抛 RetrievalError



class TestL2Metric:
    """L2 距离度量下，距离 0（完全相同向量）也应被保留。"""

    def test_l2_keeps_zero_distance(self, mock_pymilvus, docs_and_embed):
        docs, embed_func, dim = docs_and_embed
        r = MilvusRetriever(
            embed_func=embed_func, dim=dim, collection="l2_test", metric_type="L2"
        )
        r.build_index(docs)

        mock_pymilvus["Collection"]._instances[(r._alias, "l2_test")].search = MagicMock(
            return_value=[[
                MagicMock(id="a", score=0.0, entity={"content": "A", "metadata": {}}),
                MagicMock(id="b", score=0.35, entity={"content": "B", "metadata": {}}),
            ]]
        )
        results = r.retrieve("q")
        assert len(results) == 2  # L2 距离 0.0 也有效（完全相同向量），不应被过滤
        assert {x.doc_id for x in results} == {"a", "b"}

    def test_cosine_still_filters_non_positive(self, mock_pymilvus, docs_and_embed):
        docs, embed_func, dim = docs_and_embed
        r = MilvusRetriever(
            embed_func=embed_func, dim=dim, collection="cosine_test", metric_type="COSINE"
        )
        r.build_index(docs)

        mock_pymilvus["Collection"]._instances[(r._alias, "cosine_test")].search = MagicMock(
            return_value=[[
                MagicMock(id="a", score=-0.2, entity={"content": "A", "metadata": {}}),
                MagicMock(id="b", score=0.8, entity={"content": "B", "metadata": {}}),
            ]]
        )
        results = r.retrieve("q")
        assert len(results) == 1
        assert results[0].doc_id == "b"
