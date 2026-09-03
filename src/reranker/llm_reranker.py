"""LLM 重排器：用 LLM 对召回候选逐一打分（0-1 相关度），按分数重排取 top_k。

cross-encoder 版本依赖 sentence-transformers，可后置实现；当前 LLM 重排器
不依赖额外模型，接入任意 OpenAI 兼容 LLM 即可工作。
"""
import re
from typing import List, Optional

from src.common.exceptions import RerankError
from src.llm_client.base import BaseLLMClient
from src.reranker.base import BaseReranker
from src.reranker.registry import reranker_registry
from src.retriever.base import RetrieveResult

# 从 LLM 回复中提取第一个浮点数
_NUMBER_PATTERN = re.compile(r"[-+]?\d*\.?\d+")


@reranker_registry.register("llm")
class LLMReranker(BaseReranker):
    """基于 LLM 打分的重排器。"""

    def __init__(
        self,
        llm_client: BaseLLMClient,
        top_k: int = 5,
        max_content_chars: int = 800,
    ) -> None:
        self.llm = llm_client
        self.top_k = top_k
        self.max_content_chars = max_content_chars

    def rerank(
        self,
        query: str,
        candidates: List[RetrieveResult],
        top_k: Optional[int] = None,
    ) -> List[RetrieveResult]:
        if not candidates:
            return []
        k = top_k or self.top_k

        scored: List[tuple[float, RetrieveResult]] = []
        for cand in candidates:
            score = self._score_one(query, cand)
            scored.append((score, cand))

        # 按分数降序
        scored.sort(key=lambda x: x[0], reverse=True)

        results: List[RetrieveResult] = []
        for score, cand in scored[:k]:
            meta = dict(cand.metadata)
            meta["rerank_score"] = score
            meta["reranker"] = "llm"
            results.append(RetrieveResult(
                doc_id=cand.doc_id,
                content=cand.content,
                score=score,
                metadata=meta,
            ))
        return results

    def _score_one(self, query: str, candidate: RetrieveResult) -> float:
        """让 LLM 判断 query 与 candidate 的相关度，返回 0-1 浮点数。"""
        content = candidate.content[:self.max_content_chars]
        prompt = (
            "你是一个相关性评估专家。请判断以下查询与文档片段的相关程度，"
            "只输出一个 0 到 1 之间的小数，越相关越接近 1，不要输出其他文字。\n\n"
            f"查询：{query}\n\n"
            f"文档片段：{content}\n\n"
            "相关度："
        )
        try:
            resp = self.llm.chat([{"role": "user", "content": prompt}])
        except Exception as e:
            raise RerankError(f"LLM 重排打分失败: {e}") from e

        match = _NUMBER_PATTERN.search(resp or "")
        if not match:
            return 0.0
        try:
            score = float(match.group())
        except ValueError:
            return 0.0
        # 钳制到 [0, 1]
        return max(0.0, min(1.0, score))
