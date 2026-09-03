"""P3 召回层单测：BM25 / Vector / Hybrid(RRF) 三种召回器 + 注册表 + 工厂函数。"""
from typing import List

import pytest

from src.common.exceptions import EmbeddingError, RetrievalError
from src.document_parser.base import Document
from src.retriever import build_retriever, retriever_registry
from src.retriever.bm25_retriever import BM25Retriever, default_tokenizer
from src.retriever.hybrid_retriever import HybridRetriever
from src.retriever.vector_retriever import VectorRetriever


# ---------------- 测试语料与假嵌入 ----------------
@pytest.fixture
def sample_docs() -> List[Document]:
    return [
        Document(doc_id="d1", content="苹果是一种红色的水果，富含维生素", metadata={"source": "fruit"}),
        Document(doc_id="d2", content="香蕉是一种黄色的热带水果", metadata={"source": "fruit"}),
        Document(doc_id="d3", content="汽车是一种陆地交通工具，使用燃油或电力", metadata={"source": "vehicle"}),
        Document(doc_id="d4", content="飞机是一种空中交通工具，速度很快", metadata={"source": "vehicle"}),
    ]


@pytest.fixture
def fake_embed():
    """基于关键词词表的假嵌入：包含某词则对应维度为 1。"""
    vocab = ["苹果", "香蕉", "汽车", "飞机", "水果", "交通工具"]

    def _embed(texts: List[str]) -> List[List[float]]:
        return [[1.0 if w in t else 0.0 for w in vocab] for t in texts]

    return _embed


# ---------------- 注册表测试 ----------------
class TestRegistry:
    def test_all_retrievers_registered(self):
        assert retriever_registry.has("bm25")
        assert retriever_registry.has("vector")
        assert retriever_registry.has("hybrid")
        assert set(retriever_registry.names) == {"bm25", "vector", "hybrid"}


# ---------------- BM25 召回器 ----------------
class TestBM25Retriever:
    def test_build_and_retrieve(self, sample_docs):
        retriever = BM25Retriever(sample_docs, top_k=4)
        results = retriever.retrieve("苹果水果", top_k=4)
        assert len(results) >= 1
        # d1（苹果）应排在最前
        assert results[0].doc_id == "d1"
        assert results[0].score > 0
        assert results[0].metadata["retriever"] == "bm25"

    def test_top_k_limit(self, sample_docs):
        retriever = BM25Retriever(sample_docs, top_k=2)
        results = retriever.retrieve("水果", top_k=2)
        assert len(results) <= 2

    def test_default_top_k(self, sample_docs):
        retriever = BM25Retriever(sample_docs, top_k=2)
        results = retriever.retrieve("水果")  # 不传 top_k 用实例默认
        assert len(results) <= 2

    def test_no_index_raises(self):
        retriever = BM25Retriever()
        with pytest.raises(RetrievalError):
            retriever.retrieve("查询")

    def test_empty_corpus(self):
        retriever = BM25Retriever([])
        with pytest.raises(RetrievalError):
            retriever.retrieve("查询")

    def test_default_tokenizer_mixed(self):
        tokens = default_tokenizer("Hello世界 123")
        assert "Hello" in tokens
        assert "世" in tokens
        assert "界" in tokens
        assert "123" in tokens

    def test_metadata_preserved(self, sample_docs):
        retriever = BM25Retriever(sample_docs, top_k=4)
        results = retriever.retrieve("汽车", top_k=1)
        assert results[0].metadata["source"] == "vehicle"

    def test_small_corpus_negative_idf_not_filtered(self):
        """回归测试：小语料下匹配文档可能得负分（常见词 IDF 为负），不应被过滤。"""
        docs = [
            Document(doc_id="d1", content="苹果是红色水果，富含维生素", metadata={}),
            Document(doc_id="d2", content="香蕉是黄色水果，富含钾", metadata={}),
        ]
        retriever = BM25Retriever(docs, top_k=2)
        results = retriever.retrieve("水果维生素", top_k=2)
        # 两个文档都含"水果"，应都被返回（即使分数为负）
        assert len(results) == 2
        assert {r.doc_id for r in results} == {"d1", "d2"}


# ---------------- Vector 召回器 ----------------
class TestVectorRetriever:
    def test_build_and_retrieve(self, sample_docs, fake_embed):
        retriever = VectorRetriever(sample_docs, embed_func=fake_embed, top_k=4)
        results = retriever.retrieve("苹果水果", top_k=4)
        assert len(results) >= 1
        # d1 包含"苹果"和"水果"，与查询最相似
        assert results[0].doc_id == "d1"
        assert 0 < results[0].score <= 1.0  # 余弦相似度范围
        assert results[0].metadata["retriever"] == "vector"

    def test_vehicle_query_ranks_vehicle(self, sample_docs, fake_embed):
        retriever = VectorRetriever(sample_docs, embed_func=fake_embed, top_k=4)
        results = retriever.retrieve("汽车交通工具", top_k=4)
        assert results[0].doc_id in ("d3", "d4")

    def test_no_embed_func_raises(self, sample_docs):
        with pytest.raises(EmbeddingError):
            VectorRetriever(sample_docs, embed_func=None)

    def test_no_index_raises(self, fake_embed):
        retriever = VectorRetriever(embed_func=fake_embed)
        with pytest.raises(RetrievalError):
            retriever.retrieve("查询")

    def test_top_k_limit(self, sample_docs, fake_embed):
        retriever = VectorRetriever(sample_docs, embed_func=fake_embed, top_k=2)
        results = retriever.retrieve("水果", top_k=2)
        assert len(results) <= 2

    def test_scores_descending(self, sample_docs, fake_embed):
        retriever = VectorRetriever(sample_docs, embed_func=fake_embed, top_k=4)
        results = retriever.retrieve("水果", top_k=4)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)


# ---------------- Hybrid 召回器（RRF 融合） ----------------
class TestHybridRetriever:
    def test_rrf_fusion(self, sample_docs, fake_embed):
        vec = VectorRetriever(sample_docs, embed_func=fake_embed, top_k=8)
        bm = BM25Retriever(sample_docs, top_k=8)
        hybrid = HybridRetriever(vector_retriever=vec, bm25_retriever=bm, rrf_k=60, top_k=4)
        results = hybrid.retrieve("苹果水果", top_k=4)
        assert len(results) >= 1
        assert results[0].doc_id == "d1"
        # RRF 融合分数应为正
        assert all(r.score > 0 for r in results)

    def test_no_duplicate_doc_ids(self, sample_docs, fake_embed):
        vec = VectorRetriever(sample_docs, embed_func=fake_embed, top_k=8)
        bm = BM25Retriever(sample_docs, top_k=8)
        hybrid = HybridRetriever(vector_retriever=vec, bm25_retriever=bm, top_k=4)
        results = hybrid.retrieve("水果", top_k=4)
        doc_ids = [r.doc_id for r in results]
        assert len(doc_ids) == len(set(doc_ids))  # 无重复

    def test_top_k_limit(self, sample_docs, fake_embed):
        vec = VectorRetriever(sample_docs, embed_func=fake_embed, top_k=8)
        bm = BM25Retriever(sample_docs, top_k=8)
        hybrid = HybridRetriever(vector_retriever=vec, bm25_retriever=bm, top_k=2)
        results = hybrid.retrieve("水果", top_k=2)
        assert len(results) <= 2

    def test_rrf_score_independent_of_raw_scale(self, sample_docs, fake_embed):
        """RRF 基于排名而非原始分数，两路分数尺度不同不影响融合。"""
        vec = VectorRetriever(sample_docs, embed_func=fake_embed, top_k=8)
        bm = BM25Retriever(sample_docs, top_k=8)
        hybrid = HybridRetriever(vector_retriever=vec, bm25_retriever=bm, rrf_k=60, top_k=4)
        results = hybrid.retrieve("交通工具", top_k=4)
        # d3/d4（交通工具）应排在前面
        top_ids = [r.doc_id for r in results[:2]]
        assert set(top_ids) == {"d3", "d4"}


# ---------------- 工厂函数 build_retriever ----------------
class TestBuildRetriever:
    def test_build_bm25(self, sample_docs):
        retriever = build_retriever("bm25", sample_docs, top_k=4)
        assert isinstance(retriever, BM25Retriever)
        results = retriever.retrieve("苹果", top_k=1)
        assert results[0].doc_id == "d1"

    def test_build_vector(self, sample_docs, fake_embed):
        retriever = build_retriever("vector", sample_docs, embed_func=fake_embed, top_k=4)
        assert isinstance(retriever, VectorRetriever)
        results = retriever.retrieve("苹果", top_k=1)
        assert results[0].doc_id == "d1"

    def test_build_hybrid(self, sample_docs, fake_embed):
        retriever = build_retriever("hybrid", sample_docs, embed_func=fake_embed, top_k=4)
        assert isinstance(retriever, HybridRetriever)
        results = retriever.retrieve("苹果水果", top_k=2)
        assert len(results) >= 1
        assert results[0].doc_id == "d1"

    def test_build_hybrid_without_embed_raises(self, sample_docs):
        with pytest.raises(EmbeddingError):
            build_retriever("hybrid", sample_docs, embed_func=None)

    def test_unknown_name_raises(self, sample_docs):
        with pytest.raises(Exception):
            build_retriever("nonexistent", sample_docs)
