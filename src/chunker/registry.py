"""chunker 层的可插拔注册表实例。"""
from src.chunker.base import BaseChunker
from src.common.registry import Registry

chunker_registry = Registry(
    name="chunker",
    base_type=BaseChunker,
    default="fixed",
)
