"""
retriever 包：统一召回入口。

用法::

    from src.retriever import build_retriever
    retriever = build_retriever("hybrid", docs, embed_func=my_embed)
    results = retriever.retrieve("查询词", top_k=5)
"""
from typing import Callable, List, Optional

from src.document_parser.base import Document
from src.retriever.base import BaseRetriever, RetrieveResult
from src.retriever.registry import retriever_registry

# 导入实现以触发 @register 装饰器注册（必须在 registry 定义之后）
from src.retriever import bm25_retriever  # noqa: E402,F401
from src.retriever import vector_retriever  # noqa: E402,F401
from src.retriever import hybrid_retriever  # noqa: E402,F401

__all__ = [
    "BaseRetriever",
    "RetrieveResult",
    "retriever_registry",
    "build_retriever",
]


def build_retriever(
    name: str,
    documents: Optional[List[Document]] = None,
    *,
    embed_func: Optional[Callable[[List[str]], List[List[float]]]] = None,
    top_k: int = 8,
    rrf_k: int = 60,
) -> BaseRetriever:
    """工厂函数：创建召回器并构建索引。

    - ``bm25``：关键词召回，无需 embed_func
    - ``vector``：向量召回，必须提供 embed_func
    - ``hybrid``：RRF 融合 vector + bm25，必须提供 embed_func
    """
    if name == "hybrid":
        vec = retriever_registry.create(
            "vector", documents=documents, embed_func=embed_func, top_k=top_k
        )
        bm = retriever_registry.create(
            "bm25", documents=documents, top_k=top_k
        )
        return retriever_registry.create(
            "hybrid",
            vector_retriever=vec,
            bm25_retriever=bm,
            rrf_k=rrf_k,
            top_k=top_k,
        )
    if name == "vector":
        return retriever_registry.create(
            "vector", documents=documents, embed_func=embed_func, top_k=top_k
        )
    if name == "bm25":
        return retriever_registry.create("bm25", documents=documents, top_k=top_k)
    # 其他已注册的实现，透传通用参数
    return retriever_registry.create(name, documents=documents, top_k=top_k)
