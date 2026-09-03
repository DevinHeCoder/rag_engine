"""reranker 层的可插拔注册表实例。"""
from src.common.registry import Registry
from src.reranker.base import BaseReranker

reranker_registry = Registry(
    name="reranker",
    base_type=BaseReranker,
)
