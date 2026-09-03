"""
Markdown 解析器：按一级标题（# ）分块，输出统一 Document 列表。

设计说明：parser 层只做"文本提取 + 逻辑分块"，不做渲染；
更细粒度的切分交给后续 chunker 层。当前用正则原生解析，不依赖 markdown 库。
"""
import re
from pathlib import Path
from typing import List, Tuple

from src.common.exceptions import DocumentParsingError
from src.document_parser.base import BaseDocumentParser, Document
from src.document_parser.registry import document_parser_registry

# 匹配一级标题行：行首 # + 空格 + 标题文本
_H1_PATTERN = re.compile(r"(?m)^#\s+(.+?)\s*$")


@document_parser_registry.register("markdown")
class MarkdownParser(BaseDocumentParser):
    """按一级标题分块的 Markdown 解析器。"""

    def parse(self, file_path: str) -> List[Document]:
        p = Path(file_path)
        if not p.is_file():
            raise DocumentParsingError(f"Markdown 文件不存在: {p}")
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError as e:
            raise DocumentParsingError(f"Markdown 文件编码不是 UTF-8: {p}") from e

        blocks = self._split_by_h1(text)
        docs: List[Document] = []
        for heading, content in blocks:
            content = content.strip()
            if not content:
                continue
            docs.append(Document(
                doc_id=f"{p.stem}::{len(docs):03d}",
                content=content,
                metadata={
                    "source": str(p),
                    "file_name": p.name,
                    "file_type": "md",
                    "section": heading,
                    "index": len(docs),
                },
            ))
        return docs

    @staticmethod
    def _split_by_h1(text: str) -> List[Tuple[str, str]]:
        """按一级标题切分，返回 [(heading_or_None, content), ...]。"""
        matches = list(_H1_PATTERN.finditer(text))
        if not matches:
            return [(None, text)]  # type: ignore[list-item]

        blocks: List[Tuple[str, str]] = []
        # 第一个标题之前的前言
        if matches[0].start() > 0:
            preamble = text[:matches[0].start()].strip()
            if preamble:
                blocks.append((None, preamble))  # type: ignore[list-item]

        for i, m in enumerate(matches):
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            blocks.append((m.group(1).strip(), text[start:end]))
        return blocks
