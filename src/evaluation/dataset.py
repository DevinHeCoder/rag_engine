"""P6 评估层：评估数据集加载与校验。

数据集格式（JSON）::

    [
        {
            "question": "什么水果富含维生素？",
            "relevant": ["apple-doc", "banana-doc"]
        },
        ...
    ]

    - question: 查询文本（必填）
    - relevant: 相关文档的根 doc_id 列表（必填，至少 1 个）
"""
import json
from pathlib import Path
from typing import List, Optional

from src.common.exceptions import EvaluationError


class EvalSample:
    """单条评估样本。"""

    __slots__ = ("question", "relevant")

    def __init__(self, question: str, relevant: List[str]) -> None:
        self.question = question
        self.relevant = list(relevant)


def load_dataset(path: str) -> List[EvalSample]:
    """从 JSON 文件加载评估数据集并校验。"""
    p = Path(path)
    if not p.exists():
        raise EvaluationError(f"评估数据集不存在: {path}")

    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise EvaluationError(f"评估数据集 JSON 解析失败: {e}") from e

    if not isinstance(raw, list) or len(raw) == 0:
        raise EvaluationError("评估数据集必须是包含至少 1 条样本的数组")

    samples: List[EvalSample] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise EvaluationError(f"第 {i} 条样本必须是对象")
        question = item.get("question")
        relevant = item.get("relevant")
        if not question or not isinstance(question, str):
            raise EvaluationError(f"第 {i} 条样本缺少 question（非空字符串）")
        if not relevant or not isinstance(relevant, list):
            raise EvaluationError(f"第 {i} 条样本缺少 relevant（非空列表）")
        samples.append(EvalSample(question=question, relevant=[str(x) for x in relevant]))

    return samples


def save_dataset(samples: List[EvalSample], path: str) -> None:
    """将样本列表保存为 JSON 文件。"""
    data = [{"question": s.question, "relevant": s.relevant} for s in samples]
    Path(path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
