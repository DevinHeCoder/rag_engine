"""PDF 解析器：按页分块，每页一个 Document。"""
from pathlib import Path
from typing import List

from pypdf import PdfReader

from src.common.exceptions import DocumentParsingError
from src.document_parser.base import BaseDocumentParser, Document
from src.document_parser.registry import document_parser_registry


@document_parser_registry.register("pdf")
class PdfParser(BaseDocumentParser):
    """按页提取文本的 PDF 解析器；空页自动跳过，单页提取失败不中断整个文件。"""

    def parse(self, file_path: str) -> List[Document]:
        p = Path(file_path)
        if not p.is_file():
            raise DocumentParsingError(f"PDF 文件不存在: {p}")
        try:
            reader = PdfReader(str(p))
        except Exception as e:
            raise DocumentParsingError(f"读取 PDF 失败: {p}，原因: {e}") from e

        docs: List[Document] = []
        for page_num, page in enumerate(reader.pages, start=1):
            try:
                text = (page.extract_text() or "").strip()
            except Exception:
                # 单页提取失败不中断整个文件，记为空页跳过
                text = ""
            if not text:
                continue
            docs.append(Document(
                doc_id=f"{p.stem}::page-{page_num:03d}",
                content=text,
                metadata={
                    "source": str(p),
                    "file_name": p.name,
                    "file_type": "pdf",
                    "page": page_num,
                    "index": len(docs),
                },
            ))
        return docs
