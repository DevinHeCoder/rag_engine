"""chunker 包：统一分块入口。

用法::

    from src.chunker import split_documents, chunker_registry
    chunks = split_documents(docs)                      # 默认 fixed
    chunks = split_documents(docs, "hierarchy")         # 指定层级分块
    chunks = split_documents(docs, "fixed", chunk_size=256, chunk_overlap=32)
"""
from typing import List, Optional

from src.chunker.base import BaseChunker
from src.chunker.registry import chunker_registry

# 导入实现以触发 @register 装饰器注册（必须在 registry 定义之后）
from src.chunker import fixed_chunker  # noqa: E402,F401
from src.chunker import hierarchy_chunker  # noqa: E402,F401
from src.chunker import semantic_chunker  # noqa: E402,F401
from src.document_parser.base import Document

__all__ = [
    "BaseChunker",
    "chunker_registry",
    "split_documents",
]


def split_documents(
    docs: List[Document],
    chunker_name: Optional[str] = None,
    **kwargs,
) -> List[Document]:
    """按名字创建分块器并分块。chunker_name 为 None 时用默认 fixed；kwargs 透传给构造函数。"""
    chunker = chunker_registry.create(chunker_name, **kwargs)
    return chunker.split(docs)
