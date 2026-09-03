"""P6 评估层单测：指标纯函数 + 数据集 + Evaluator 集成 + 忠实度。"""
import json
from typing import List

import pytest

from src.common.exceptions import EvaluationError
from src.evaluation import (
    Evaluator,
    evaluate_faithfulness,
    hit_rate,
    load_dataset,
    mrr,
    precision_at_k,
    recall_at_k,
    root_key,
    save_dataset,
)
from src.evaluation.dataset import EvalSample
from src.llm_client.base import BaseLLMClient
from src.retriever.base import RetrieveResult


# ---------------- 指标纯函数 ----------------

class TestRootKey:
    def test_plain(self):
        assert root_key("apple") == "apple"

    def test_chunk_suffix(self):
        assert root_key("apple::chunk-000") == "apple"

    def test_parser_section_suffix(self):
        assert root_key("kb::000::chunk-000") == "kb"

    def test_page_suffix(self):
        assert root_key("report::page-1::chunk-000") == "report"


class TestRecallAtK:
    def test_basic(self):
        # 相关 {a, b}，检索 [a, c, b]，k=3 -> 命中 a,b = 2/2
        assert recall_at_k(["a", "c", "b"], ["a", "b"], 3) == 1.0

    def test_partial(self):
        assert recall_at_k(["a", "c"], ["a", "b"], 2) == 0.5

    def test_k_limits(self):
        assert recall_at_k(["a", "b"], ["a", "b"], 1) == 0.5

    def test_chunk_ids_normalized(self):
        # 检索结果带 chunk 后缀，也能匹配
        assert recall_at_k(["doc-x::chunk-000", "doc-y::chunk-000"], ["doc-x"], 2) == 1.0

    def test_empty_relevant(self):
        assert recall_at_k(["a"], [], 1) == 0.0


class TestPrecisionAtK:
    def test_basic(self):
        assert precision_at_k(["a", "b", "c"], ["a"], 3) == pytest.approx(1 / 3)

    def test_zero_k(self):
        assert precision_at_k(["a"], ["a"], 0) == 0.0


class TestMRR:
    def test_first_position(self):
        assert mrr(["a", "b"], ["a"]) == 1.0

    def test_second_position(self):
        assert mrr(["x", "a"], ["a"]) == 0.5

    def test_not_found(self):
        assert mrr(["x", "y"], ["a"]) == 0.0

    def test_chunk_normalized(self):
        assert mrr(["x::chunk-000", "a::chunk-000"], ["a"]) == 0.5


class TestHitRate:
    def test_hit(self):
        assert hit_rate(["a", "x"], ["a"], 2) == 1.0

    def test_miss(self):
        assert hit_rate(["x", "y"], ["a"], 2) == 0.0


# ---------------- 数据集 ----------------

class TestDataset:
    def test_load_valid(self, tmp_path):
        p = tmp_path / "eval.json"
        p.write_text(json.dumps([
            {"question": "q1", "relevant": ["a", "b"]},
            {"question": "q2", "relevant": ["c"]},
        ]), encoding="utf-8")
        samples = load_dataset(str(p))
        assert len(samples) == 2
        assert samples[0].question == "q1"
        assert samples[0].relevant == ["a", "b"]

    def test_load_missing_file(self):
        with pytest.raises(EvaluationError, match="不存在"):
            load_dataset("C:/no/such/file.json")

    def test_load_invalid_json(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("not json", encoding="utf-8")
        with pytest.raises(EvaluationError, match="JSON 解析失败"):
            load_dataset(str(p))

    def test_load_empty_list(self, tmp_path):
        p = tmp_path / "empty.json"
        p.write_text("[]", encoding="utf-8")
        with pytest.raises(EvaluationError, match="至少"):
            load_dataset(str(p))

    def test_load_missing_question(self, tmp_path):
        p = tmp_path / "bad2.json"
        p.write_text(json.dumps([{"relevant": ["a"]}]), encoding="utf-8")
        with pytest.raises(EvaluationError, match="question"):
            load_dataset(str(p))

    def test_save_and_reload(self, tmp_path):
        p = tmp_path / "out.json"
        save_dataset([EvalSample("q", ["a"])], str(p))
        assert load_dataset(str(p))[0].question == "q"


# ---------------- 检索评估集成 ----------------

class _StubRetriever:
    """固定返回结果的假召回器。"""

    def __init__(self, results_by_query: dict):
        self.results_by_query = results_by_query

    def retrieve(self, query: str, top_k: int) -> List[RetrieveResult]:
        ids = self.results_by_query.get(query, [])
        return [RetrieveResult(doc_id=i, content=f"content of {i}", score=1.0, metadata={}) for i in ids[:top_k]]


class TestEvaluatorRetrieval:
    def test_metrics_computed(self):
        samples = [
            EvalSample("q1", ["a", "b"]),
            EvalSample("q2", ["x"]),
        ]
        retriever = _StubRetriever({
            "q1": ["a::chunk-000", "c::chunk-000", "b::chunk-000"],  # recall 1.0, mrr 1.0
            "q2": ["y::chunk-000", "x::chunk-000"],                   # recall 1.0, mrr 0.5
        })
        report = Evaluator(metrics=["recall@k", "mrr"], default_k=3).evaluate_retrieval(samples, retriever, top_k=3)

        assert report["num_queries"] == 2
        assert report["top_k"] == 3
        assert report["metrics"]["recall@k"] == pytest.approx(1.0)
        assert report["metrics"]["mrr"] == pytest.approx(0.75)  # (1.0 + 0.5) / 2

    def test_partial_recall(self):
        samples = [EvalSample("q", ["a", "b"])]
        retriever = _StubRetriever({"q": ["a::chunk-000"]})  # 只召回 1/2
        report = Evaluator(metrics=["recall@k"], default_k=2).evaluate_retrieval(samples, retriever, top_k=2)
        assert report["metrics"]["recall@k"] == pytest.approx(0.5)

    def test_progress_callback(self):
        samples = [EvalSample("q1", ["a"]), EvalSample("q2", ["b"])]
        retriever = _StubRetriever({"q1": ["a"], "q2": ["b"]})
        calls = []
        Evaluator(metrics=["recall@k"]).evaluate_retrieval(
            samples, retriever, top_k=1, progress=lambda done, total: calls.append((done, total))
        )
        assert calls == [(1, 2), (2, 2)]

    def test_unknown_metric_raises(self):
        with pytest.raises(EvaluationError, match="不支持"):
            Evaluator(metrics=["nope"])


# ---------------- 忠实度 ----------------

class MockLLM(BaseLLMClient):
    def __init__(self, response="0.9"):
        self.response = response

    def chat(self, messages: List[dict], **kwargs) -> str:
        return self.response

    def embed(self, text: str, **kwargs) -> List[float]:
        return [0.0]


class TestFaithfulness:
    def _ctx(self, doc_id="d1", content="苹果富含维生素C"):
        return [RetrieveResult(doc_id=doc_id, content=content, score=1.0, metadata={})]

    def test_with_llm(self):
        score = evaluate_faithfulness(
            "问题", "苹果富含维生素C", self._ctx(), llm_client=MockLLM("0.85")
        )
        assert score == pytest.approx(0.85)

    def test_with_llm_invalid_response(self):
        score = evaluate_faithfulness(
            "问题", "答案", self._ctx(), llm_client=MockLLM("完全符合要求")
        )
        assert score == 0.0

    def test_heuristic_supported(self):
        score = evaluate_faithfulness(
            "问题", "苹果是水果，富含维生素C", self._ctx("d1", "苹果是水果，富含维生素C和钾")
        )
        assert score > 0.0

    def test_heuristic_unsupported(self):
        score = evaluate_faithfulness(
            "问题", "火星是红色的行星", self._ctx("d1", "苹果是水果")
        )
        assert score < 0.5

    def test_evaluator_faithfulness_requires_llm(self):
        with pytest.raises(EvaluationError, match="llm_client"):
            Evaluator(llm_client=None, metrics=["faithfulness"]).evaluate_faithfulness(
                [EvalSample("q", ["a"])], lambda q: ("answer", self._ctx())
            )

    def test_evaluator_faithfulness_with_llm(self):
        samples = [EvalSample("q", ["a"])]
        evaluator = Evaluator(llm_client=MockLLM("1.0"), metrics=["faithfulness"])
        report = evaluator.evaluate_faithfulness(samples, lambda q: ("答案忠于上下文", self._ctx()))
        assert report["faithfulness"] == pytest.approx(1.0)
        assert report["num_queries"] == 1
