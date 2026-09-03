"""document_parser 包：统一文档解析入口。

用法::

    from src.document_parser import parse_file, document_parser_registry
    docs = parse_file("path/to/file.pdf")
"""
from pathlib import Path
from typing import Dict, List

from src.common.exceptions import DocumentParsingError
from src.document_parser.base import BaseDocumentParser, Document
from src.document_parser.registry import document_parser_registry

# 导入实现以触发 @register 装饰器注册（必须在 registry 定义之后）
from src.document_parser import md_parser  # noqa: E402,F401
from src.document_parser import pdf_parser  # noqa: E402,F401
from src.document_parser import word_parser  # noqa: E402,F401

# 扩展名 -> 注册表实现名
_EXTENSION_MAP: Dict[str, str] = {
    ".md": "markdown",
    ".markdown": "markdown",
    ".pdf": "pdf",
    ".docx": "word",
}

__all__ = [
    "Document",
    "BaseDocumentParser",
    "document_parser_registry",
    "parse_file",
    "get_parser_name",
]


def get_parser_name(extension: str) -> str:
    """按文件扩展名返回注册表中的解析器实现名。"""
    ext = extension.lower()
    if ext not in _EXTENSION_MAP:
        raise DocumentParsingError(
            f"不支持的文件扩展名: '{extension}'；支持: {sorted(_EXTENSION_MAP)}"
        )
    return _EXTENSION_MAP[ext]


def parse_file(file_path: str) -> List[Document]:
    """按文件扩展名自动选择解析器，解析为统一 Document 列表。"""
    p = Path(file_path)
    if not p.is_file():
        raise DocumentParsingError(f"文件不存在: {p}")
    parser_name = get_parser_name(p.suffix)
    parser = document_parser_registry.create(parser_name)
    return parser.parse(str(p))
