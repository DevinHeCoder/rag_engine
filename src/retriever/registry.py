"""retriever 层的可插拔注册表实例。

注意：不设默认实现——hybrid 需要传入两个子召回器，vector 需要 embed_func，
调用方应通过 build_retriever() 工厂函数显式创建并构建索引。
"""
from src.common.registry import Registry
from src.retriever.base import BaseRetriever

retriever_registry = Registry(
    name="retriever",
    base_type=BaseRetriever,
)
