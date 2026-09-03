"""P4 重排器单测：LLM 重排器用 mock LLM 验证打分、排序、top_k、元数据。"""
from typing import List

import pytest

from src.common.exceptions import RerankError
from src.llm_client.base import BaseLLMClient
from src.reranker import build_reranker, reranker_registry
from src.reranker.llm_reranker import LLMReranker
from src.retriever.base import RetrieveResult


class MockLLMClient(BaseLLMClient):
    """测试用 mock LLM：按预设分数列表依次返回。"""

    def __init__(self, responses: List[str] = None, fail: bool = False):
        self.responses = responses or []
        self.fail = fail
        self.call_count = 0

    def chat(self, messages: List[dict], **kwargs) -> str:
        self.call_count += 1
        if self.fail:
            raise Exception("LLM 调用失败")
        if self.call_count <= len(self.responses):
            return self.responses[self.call_count - 1]
        return "0.5"

    def embed(self, text: str, **kwargs) -> List[float]:
        return [0.0]


def _candidate(doc_id: str, content: str, score: float = 0.5) -> RetrieveResult:
    return RetrieveResult(doc_id=doc_id, content=content, score=score, metadata={"source": "test"})


class TestRegistry:
    def test_llm_reranker_registered(self):
        assert reranker_registry.has("llm")


class TestBuildReranker:
    def test_default_creates_llm_reranker(self):
        llm = MockLLMClient()
        reranker = build_reranker("llm", llm_client=llm, top_k=3)
        assert isinstance(reranker, LLMReranker)
        assert reranker.top_k == 3


class TestLLMReranker:
    def test_rerank_reorders_by_score(self):
        llm = MockLLMClient(responses=["0.9", "0.3", "0.6"])
        reranker = LLMReranker(llm_client=llm, top_k=3)
        candidates = [
            _candidate("d1", "相关文档一"),
            _candidate("d2", "不相关文档"),
            _candidate("d3", "中等相关文档"),
        ]
        results = reranker.rerank("查询", candidates)
        assert [r.doc_id for r in results] == ["d1", "d3", "d2"]
        assert results[0].score == 0.9
        assert results[1].score == 0.6
        assert results[2].score == 0.3

    def test_top_k_limit(self):
        llm = MockLLMClient(responses=["0.9", "0.8", "0.7", "0.6"])
        reranker = LLMReranker(llm_client=llm, top_k=2)
        candidates = [_candidate(f"d{i}", f"文档{i}") for i in range(4)]
        results = reranker.rerank("查询", candidates, top_k=2)
        assert len(results) == 2

    def test_empty_candidates_returns_empty(self):
        llm = MockLLMClient()
        reranker = LLMReranker(llm_client=llm)
        assert reranker.rerank("查询", []) == []

    def test_metadata_preserved_with_rerank_score(self):
        llm = MockLLMClient(responses=["0.75"])
        reranker = LLMReranker(llm_client=llm, top_k=1)
        candidates = [_candidate("d1", "文档", score=0.4)]
        results = reranker.rerank("查询", candidates)
        assert results[0].metadata["source"] == "test"
        assert results[0].metadata["rerank_score"] == 0.75
        assert results[0].metadata["reranker"] == "llm"
        assert results[0].score == 0.75  # score 被重排分数覆盖

    def test_score_parsing_various_formats(self):
        # LLM 可能返回带文字的回复，应能提取数字
        llm = MockLLMClient(responses=[
            "相关度：0.85",
            "0.9 分",
            "大概 0.7 左右",
            "完全不相关",  # 无数字 -> 0
        ])
        reranker = LLMReranker(llm_client=llm, top_k=4)
        candidates = [_candidate(f"d{i}", f"文档{i}") for i in range(4)]
        results = reranker.rerank("查询", candidates)
        scores = [r.score for r in results]
        assert 0.85 in scores
        assert 0.9 in scores
        assert 0.7 in scores
        assert 0.0 in scores

    def test_score_clamped_to_0_1(self):
        llm = MockLLMClient(responses=["1.5", "-0.3"])
        reranker = LLMReranker(llm_client=llm, top_k=2)
        candidates = [_candidate("d1", "文档1"), _candidate("d2", "文档2")]
        results = reranker.rerank("查询", candidates)
        assert results[0].score == 1.0  # 1.5 被钳制到 1.0
        assert results[1].score == 0.0  # -0.3 被钳制到 0.0

    def test_llm_failure_raises_rerank_error(self):
        llm = MockLLMClient(fail=True)
        reranker = LLMReranker(llm_client=llm)
        candidates = [_candidate("d1", "文档")]
        with pytest.raises(RerankError, match="LLM 调用失败"):
            reranker.rerank("查询", candidates)

    def test_content_truncated(self):
        llm = MockLLMClient(responses=["0.5"])
        reranker = LLMReranker(llm_client=llm, top_k=1, max_content_chars=10)
        long_content = "这是一个非常长的文档内容，应该被截断" * 10
        candidates = [_candidate("d1", long_content)]
        results = reranker.rerank("查询", candidates)
        assert len(results) == 1
        # 验证 LLM 收到的 prompt 中内容被截断
        assert llm.call_count == 1
