"""P6 评估层：检索质量指标纯函数。

约定：
    - 检索结果的 doc_id 可能带层级后缀（如 ``demo-1::chunk-000`` 或 ``kb::000::chunk-000``），
      评估按"根 doc_id"匹配 —— 即第一个 ``::`` 之前的部分（``demo-1`` / ``kb``）。
    - 相关文档集合用根 doc_id 表示。
"""
from typing import List, Set, Union


def root_key(doc_id: str) -> str:
    """提取 doc_id 的根标识（第一个 :: 之前的部分）。"""
    return doc_id.split("::", 1)[0]


def normalize_ids(doc_ids: Union[List[str], Set[str]]) -> Set[str]:
    """将一批 doc_id 规约到根标识集合。"""
    return {root_key(d) for d in doc_ids}


def recall_at_k(
    retrieved_ids: List[str],
    relevant_ids: Union[List[str], Set[str]],
    k: int,
) -> float:
    """Recall@k：top-k 中命中的相关文档数 / 相关文档总数。"""
    relevant = normalize_ids(relevant_ids)
    if not relevant:
        return 0.0
    retrieved = normalize_ids(retrieved_ids[:k])
    hit = len(retrieved & relevant)
    return hit / len(relevant)


def precision_at_k(
    retrieved_ids: List[str],
    relevant_ids: Union[List[str], Set[str]],
    k: int,
) -> float:
    """Precision@k：top-k 中命中的相关文档数 / k。"""
    if k <= 0:
        return 0.0
    relevant = normalize_ids(relevant_ids)
    retrieved = normalize_ids(retrieved_ids[:k])
    hit = len(retrieved & relevant)
    return hit / k


def mrr(
    retrieved_ids: List[str],
    relevant_ids: Union[List[str], Set[str]],
) -> float:
    """MRR：第一个相关文档的倒数排名的期望（单条）。"""
    relevant = normalize_ids(relevant_ids)
    if not relevant:
        return 0.0
    for i, rid in enumerate(retrieved_ids, start=1):
        if root_key(rid) in relevant:
            return 1.0 / i
    return 0.0


def hit_rate(
    retrieved_ids: List[str],
    relevant_ids: Union[List[str], Set[str]],
    k: int,
) -> float:
    """HitRate@k：top-k 是否至少命中一个相关文档（0/1，常用于衡量"有没有找到"）。"""
    relevant = normalize_ids(relevant_ids)
    if not relevant:
        return 0.0
    retrieved = normalize_ids(retrieved_ids[:k])
    return 1.0 if retrieved & relevant else 0.0
