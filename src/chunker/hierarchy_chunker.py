"""层级分块器：按 Markdown 标题层级（# / ## / ###）递归切分，超过 max_chunk_size 的节继续下沉到下一级标题，最终用定长分块兜底。"""
import re
from typing import List, Tuple

from src.chunker.base import BaseChunker
from src.chunker.fixed_chunker import FixedChunker
from src.chunker.registry import chunker_registry
from src.document_parser.base import Document


@chunker_registry.register("hierarchy")
class HierarchyChunker(BaseChunker):
    """按标题层级递归分块的分块器。"""

    # 标题层级从高到低（level 数字越大标题越细）
    DEFAULT_HEADING_PATTERNS: List[Tuple[int, re.Pattern]] = [
        (1, re.compile(r"(?m)^#\s+.+$")),
        (2, re.compile(r"(?m)^##\s+.+$")),
        (3, re.compile(r"(?m)^###\s+.+$")),
    ]

    def __init__(
        self,
        max_chunk_size: int = 512,
        chunk_overlap: int = 64,
        heading_patterns: List[Tuple[int, re.Pattern]] = None,
    ) -> None:
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
        self.heading_patterns = (
            heading_patterns if heading_patterns is not None
            else list(self.DEFAULT_HEADING_PATTERNS)
        )
        # 最终兜底：定长分块（overlap 自动钳制，避免超过 chunk_size）
        self._fallback = FixedChunker(
            chunk_size=max_chunk_size,
            chunk_overlap=min(chunk_overlap, max_chunk_size // 2),
        )

    def split(self, docs: List[Document]) -> List[Document]:
        result: List[Document] = []
        for doc in docs:
            chunks = self._split_one(doc.content, level_idx=0)
            for i, chunk in enumerate(chunks):
                meta = dict(doc.metadata)
                meta.update({
                    "chunk_index": i,
                    "chunk_total": len(chunks),
                    "parent_doc_id": doc.doc_id,
                    "chunker": "hierarchy",
                })
                result.append(Document(
                    doc_id=f"{doc.doc_id}::chunk-{i:03d}",
                    content=chunk,
                    metadata=meta,
                ))
        return result

    # ---------------- 内部 ----------------
    def _split_one(self, text: str, level_idx: int) -> List[str]:
        """按当前层级标题递归切分；无标题或已到最底层时用定长兜底。"""
        text = text or ""
        if not text.strip():
            return []
        if len(text) <= self.max_chunk_size:
            return [text]
        if level_idx >= len(self.heading_patterns):
            return self._fallback._split_one(text)

        _, pattern = self.heading_patterns[level_idx]
        matches = list(pattern.finditer(text))
        if not matches:
            # 当前层级没有标题，下沉到下一层级
            return self._split_one(text, level_idx + 1)

        chunks: List[str] = []
        # 第一个标题之前的前言
        if matches[0].start() > 0:
            preamble = text[:matches[0].start()].strip()
            if preamble:
                chunks.extend(self._split_one(preamble, level_idx + 1))

        for i, m in enumerate(matches):
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            section = text[start:end]
            if len(section) <= self.max_chunk_size:
                chunks.append(section)
            else:
                # 本节超长，下沉到下一层级继续切
                chunks.extend(self._split_one(section, level_idx + 1))
        return chunks
