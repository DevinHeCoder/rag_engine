"""定长分块器：按 chunk_size 滑动窗口切分，优先在自然边界（段落/句子）处断开，相邻块重叠 chunk_overlap 字符。"""
from pathlib import Path
from typing import List

from src.chunker.base import BaseChunker
from src.chunker.registry import chunker_registry
from src.common.exceptions import ChunkingError
from src.document_parser.base import Document


@chunker_registry.register("fixed")
class FixedChunker(BaseChunker):
    """定长 + 重叠分块器，切点优先落在分隔符上。"""

    # 从强到弱的分隔符优先级（用于寻找自然切点）
    DEFAULT_SEPARATORS = ["\n\n", "\n", "。", "！", "？", ".", "!", "?", "；", ";"]

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        separators: List[str] = None,
    ) -> None:
        if chunk_size <= 0:
            raise ChunkingError(f"chunk_size 必须为正整数，当前: {chunk_size}")
        if chunk_overlap >= chunk_size:
            raise ChunkingError(
                f"chunk_overlap ({chunk_overlap}) 必须小于 chunk_size ({chunk_size})"
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators if separators is not None else list(self.DEFAULT_SEPARATORS)

    def split(self, docs: List[Document]) -> List[Document]:
        result: List[Document] = []
        for doc in docs:
            chunks = self._split_one(doc.content)
            for i, chunk in enumerate(chunks):
                meta = dict(doc.metadata)
                meta.update({
                    "chunk_index": i,
                    "chunk_total": len(chunks),
                    "parent_doc_id": doc.doc_id,
                    "chunker": "fixed",
                })
                result.append(Document(
                    doc_id=f"{doc.doc_id}::chunk-{i:03d}",
                    content=chunk,
                    metadata=meta,
                ))
        return result

    # ---------------- 内部 ----------------
    def _split_one(self, text: str) -> List[str]:
        text = text or ""
        if not text.strip():
            return []
        if len(text) <= self.chunk_size:
            return [text]

        chunks: List[str] = []
        start = 0
        n = len(text)
        while start < n:
            end = min(start + self.chunk_size, n)
            if end < n:
                end = self._find_cut(text, start, end)
            chunks.append(text[start:end])
            if end >= n:
                break  # 已到文本末尾，结束（避免 overlap 回退产生碎片块）
            # 下一块起点回退 overlap，但保证至少前进 1 字符，避免死循环
            start = max(end - self.chunk_overlap, start + 1)
        return chunks

    def _find_cut(self, text: str, start: int, end: int) -> int:
        """在窗口 [start, end) 内找最靠后的分隔符作为切点；找不到则硬切 at end。"""
        window = text[start:end]
        best = -1
        for sep in self.separators:
            idx = window.rfind(sep)
            if idx != -1:
                pos = start + idx + len(sep)
                # 切点不能太靠近起点（否则会产生过小块），至少大于 overlap
                if pos > start + self.chunk_overlap:
                    best = max(best, pos)
        return best if best != -1 else end
