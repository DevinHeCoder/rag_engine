"""
reranker 包：统一重排入口。

用法::

    from src.reranker import build_reranker
    reranker = build_reranker("llm", llm_client=my_llm, top_k=5)
    results = reranker.rerank(query, candidates)
"""
from typing import Optional

from src.llm_client.base import BaseLLMClient
from src.reranker.base import BaseReranker
from src.reranker.registry import reranker_registry

# 导入实现以触发 @register 装饰器注册
from src.reranker import llm_reranker  # noqa: E402,F401

__all__ = [
    "BaseReranker",
    "reranker_registry",
    "build_reranker",
]


def build_reranker(
    name: str = "llm",
    *,
    llm_client: Optional[BaseLLMClient] = None,
    top_k: int = 5,
    **kwargs,
) -> BaseReranker:
    """工厂函数：创建重排器。

    参数:
        name: 注册表中的实现名（默认 llm）
        llm_client: LLM 客户端实例（llm 重排器必需）
        top_k: 返回 top_k 个结果
        **kwargs: 透传给构造函数
    """
    params = dict(kwargs)
    if llm_client is not None:
        params["llm_client"] = llm_client
    params["top_k"] = top_k
    return reranker_registry.create(name, **params)
