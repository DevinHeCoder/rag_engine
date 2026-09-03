"""P6 评估层：核心评估器。

支持两类评估：
    1. 检索质量（无需 LLM）：recall@k / precision@k / MRR / hit@k
    2. 生成忠实度（可选，需 LLM）：faithfulness
"""
import statistics
from typing import Callable, Dict, List, Optional

from src.common.exceptions import EvaluationError
from src.evaluation import faithfulness as _faithfulness
from src.evaluation.dataset import EvalSample
from src.evaluation.metrics import (
    hit_rate,
    mrr,
    precision_at_k,
    recall_at_k,
    root_key,
)
from src.llm_client.base import BaseLLMClient
from src.retriever.base import BaseRetriever, RetrieveResult

# 支持的指标名（与 config/settings.yaml 中 evaluation.metrics 对齐）
SUPPORTED_METRICS = ["recall@k", "precision@k", "mrr", "hit@k"]


class Evaluator:
    """RAG 评估器。"""

    def __init__(
        self,
        llm_client: Optional[BaseLLMClient] = None,
        metrics: Optional[List[str]] = None,
        default_k: int = 8,
    ) -> None:
        self.llm_client = llm_client
        self.default_k = default_k
        self.metrics = metrics or SUPPORTED_METRICS
        unknown = set(self.metrics) - set(SUPPORTED_METRICS) - {"faithfulness"}
        if unknown:
            raise EvaluationError(f"不支持的指标: {sorted(unknown)}")

    # ---------------- 检索质量评估 ----------------

    def evaluate_retrieval(
        self,
        samples: List[EvalSample],
        retriever: BaseRetriever,
        top_k: Optional[int] = None,
        progress: Optional[Callable[[int, int], None]] = None,
    ) -> Dict:
        """评估检索质量，返回各指标均值 + 逐条明细。

        只计算检索类指标（recall@k/precision@k/mrr/hit@k），faithfulness 走 evaluate_faithfulness。
        """
        k = top_k or self.default_k
        metrics = [m for m in self.metrics if m in SUPPORTED_METRICS]

        per_query: List[Dict] = []
        agg: Dict[str, float] = {}

        for i, sample in enumerate(samples):
            results = retriever.retrieve(sample.question, top_k=k)
            retrieved_ids = [r.doc_id for r in results]

            row: Dict = {"question": sample.question}
            for metric in metrics:
                row[metric] = _metric_value(metric, retrieved_ids, sample.relevant, k)
            per_query.append(row)

            if progress is not None:
                progress(i + 1, len(samples))

        for metric in metrics:
            agg[metric] = statistics.mean(r[metric] for r in per_query)

        return {
            "top_k": k,
            "num_queries": len(samples),
            "metrics": agg,
            "per_query": per_query,
        }

    # ---------------- 忠实度评估 ----------------

    def evaluate_faithfulness(
        self,
        samples: List[EvalSample],
        generate_fn: Callable[[str], tuple[str, List[RetrieveResult]]],
    ) -> Dict:
        """评估生成答案忠实度。

        generate_fn(question) 应返回 (answer, contexts) —— 与 RAGService.query 的返回结构一致。
        """
        if self.llm_client is None:
            raise EvaluationError("faithfulness 指标需要提供 llm_client")

        per_query: List[Dict] = []
        scores: List[float] = []

        for sample in samples:
            answer, contexts = generate_fn(sample.question)
            score = _faithfulness.evaluate_faithfulness(
                sample.question, answer, contexts, llm_client=self.llm_client
            )
            per_query.append({"question": sample.question, "faithfulness": score})
            scores.append(score)

        return {
            "num_queries": len(samples),
            "faithfulness": statistics.mean(scores) if scores else 0.0,
            "per_query": per_query,
        }

    # ---------------- 一站式评估 ----------------

    def evaluate(
        self,
        samples: List[EvalSample],
        retriever: BaseRetriever,
        top_k: Optional[int] = None,
        generate_fn: Optional[Callable[[str], tuple[str, List[RetrieveResult]]]] = None,
    ) -> Dict:
        """一站式评估：检索指标 +（可选）忠实度。"""
        report = {
            "retrieval": self.evaluate_retrieval(samples, retriever, top_k=top_k),
        }
        if "faithfulness" in self.metrics:
            if generate_fn is None:
                raise EvaluationError("faithfulness 指标需要 generate_fn")
            report["faithfulness"] = self.evaluate_faithfulness(samples, generate_fn)
        return report


def _metric_value(metric: str, retrieved_ids: List[str], relevant: List[str], k: int) -> float:
    if metric == "recall@k":
        return recall_at_k(retrieved_ids, relevant, k)
    if metric == "precision@k":
        return precision_at_k(retrieved_ids, relevant, k)
    if metric == "mrr":
        return mrr(retrieved_ids, relevant)
    if metric == "hit@k":
        return hit_rate(retrieved_ids, relevant, k)
    raise EvaluationError(f"不支持的指标: {metric}")


__all__ = ["Evaluator", "SUPPORTED_METRICS", "root_key"]
