"""
语义分块器：按句子嵌入的相邻相似度距离突变点切分。

依赖外部传入的 embed_func（句子嵌入函数），不绑定具体 embedding 实现；
P3 向量召回层就绪后可直接传入 embedding 客户端。
"""
import re
from typing import Callable, List, Optional

from src.chunker.base import BaseChunker
from src.chunker.fixed_chunker import FixedChunker
from src.chunker.registry import chunker_registry
from src.common.exceptions import ChunkingError
from src.document_parser.base import Document

# 句子分隔：句末标点（保留在前一句末尾）或换行
_SENTENCE_SPLIT = re.compile(r"(?<=[。！？.!?])\s*|\n+")


@chunker_registry.register("semantic")
class SemanticChunker(BaseChunker):
    """基于句子嵌入相似度的语义分块器。"""

    def __init__(
        self,
        embed_func: Optional[Callable[[List[str]], List[List[float]]]] = None,
        chunk_size: int = 512,
        breakpoint_percentile: float = 90.0,
    ) -> None:
        self.embed_func = embed_func
        self.chunk_size = chunk_size
        self.breakpoint_percentile = breakpoint_percentile
        self._fallback = FixedChunker(
            chunk_size=chunk_size,
            chunk_overlap=min(64, chunk_size // 4),
        )

    def split(self, docs: List[Document]) -> List[Document]:
        if self.embed_func is None:
            raise ChunkingError(
                "SemanticChunker 需要 embed_func 参数（句子嵌入函数），"
                "请在 embedding 客户端就绪后传入，例如 chunker_registry.create('semantic', embed_func=my_embed)"
            )
        result: List[Document] = []
        for doc in docs:
            chunks = self._split_one(doc.content)
            for i, chunk in enumerate(chunks):
                meta = dict(doc.metadata)
                meta.update({
                    "chunk_index": i,
                    "chunk_total": len(chunks),
                    "parent_doc_id": doc.doc_id,
                    "chunker": "semantic",
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

        sentences = self._split_sentences(text)
        if len(sentences) <= 1:
            return [text]

        embeddings = self.embed_func(sentences)
        if len(embeddings) != len(sentences):
            raise ChunkingError(
                f"embed_func 返回数量 ({len(embeddings)}) 与句子数 ({len(sentences)}) 不一致"
            )

        distances = self._cosine_distances(embeddings)
        threshold = self._percentile(distances, self.breakpoint_percentile) if distances else 0.0

        chunks: List[str] = []
        current = [sentences[0]]
        for i, dist in enumerate(distances):
            if dist > threshold:
                chunks.append("".join(current))
                current = [sentences[i + 1]]
            else:
                current.append(sentences[i + 1])
        if current:
            chunks.append("".join(current))

        # 超长语义块再用定长兜底切分
        final: List[str] = []
        for chunk in chunks:
            if len(chunk) > self.chunk_size:
                final.extend(self._fallback._split_one(chunk))
            else:
                final.append(chunk)
        return final

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        parts = _SENTENCE_SPLIT.split(text)
        return [p.strip() for p in parts if p.strip()]

    @staticmethod
    def _cosine_distances(embeddings: List[List[float]]) -> List[float]:
        """计算相邻句子嵌入的余弦距离（1 - 余弦相似度）。"""
        distances: List[float] = []
        for i in range(len(embeddings) - 1):
            a, b = embeddings[i], embeddings[i + 1]
            dot = sum(x * y for x, y in zip(a, b))
            na = sum(x * x for x in a) ** 0.5
            nb = sum(x * x for x in b) ** 0.5
            sim = dot / (na * nb) if na > 0 and nb > 0 else 0.0
            distances.append(1.0 - sim)
        return distances

    @staticmethod
    def _percentile(data: List[float], p: float) -> float:
        """线性插值百分位数（纯 Python，避免 numpy 依赖）。"""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        k = (len(sorted_data) - 1) * p / 100.0
        f = int(k)
        c = min(f + 1, len(sorted_data) - 1)
        return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])
