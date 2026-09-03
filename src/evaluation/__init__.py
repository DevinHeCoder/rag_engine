"""P6 评估层：统一入口。

用法::

    from src.evaluation import Evaluator, load_dataset
    samples = load_dataset("data/eval_set.json")
    report = Evaluator(metrics=["recall@k", "mrr"]).evaluate_retrieval(samples, retriever, top_k=8)
"""
from src.evaluation.dataset import EvalSample, load_dataset, save_dataset
from src.evaluation.evaluator import Evaluator, SUPPORTED_METRICS
from src.evaluation.faithfulness import evaluate_faithfulness
from src.evaluation.metrics import (
    hit_rate,
    mrr,
    precision_at_k,
    recall_at_k,
    root_key,
)

__all__ = [
    "Evaluator",
    "EvalSample",
    "SUPPORTED_METRICS",
    "load_dataset",
    "save_dataset",
    "evaluate_faithfulness",
    "recall_at_k",
    "precision_at_k",
    "mrr",
    "hit_rate",
    "root_key",
]
