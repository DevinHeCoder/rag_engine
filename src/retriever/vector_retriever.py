"""向量召回器：基于 embedding 余弦相似度，本地 numpy 实现。

embed_func 由外部注入（P3 阶段可用假嵌入验证，P4 后接入真实 embedding 客户端），
不绑定具体模型或服务。
"""
from typing import Callable, List, Optional

import numpy as np

from src.common.exceptions import EmbeddingError, RetrievalError
from src.document_parser.base import Document
from src.retriever.base import BaseRetriever, RetrieveResult
from src.retriever.registry import retriever_registry

EmbedFunc = Callable[[List[str]], List[List[float]]]


@retriever_registry.register("vector")
class VectorRetriever(BaseRetriever):
    """基于向量余弦相似度的召回器。"""

    def __init__(
        self,
        documents: Optional[List[Document]] = None,
        embed_func: Optional[EmbedFunc] = None,
        top_k: int = 8,
    ) -> None:
        self.embed_func = embed_func
        self.top_k = top_k
        self.docs: List[Document] = []
        self._vectors: Optional[np.ndarray] = None
        if documents:
            self.build_index(documents)

    def build_index(self, documents: List[Document]) -> None:
        """对所有文档内容做 embedding 并构建归一化向量索引。"""
        if self.embed_func is None:
            raise EmbeddingError("VectorRetriever 需要 embed_func 参数（句子嵌入函数）")
        if not documents:
            self.docs = []
            self._vectors = None
            return
        self.docs = list(documents)
        embeddings = self.embed_func([d.content for d in self.docs])
        if len(embeddings) != len(self.docs):
            raise EmbeddingError(
                f"embed_func 返回数量 ({len(embeddings)}) 与文档数 ({len(self.docs)}) 不一致"
            )
        vectors = np.array(embeddings, dtype=np.float32)
        # L2 归一化，后续点积即余弦相似度
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        self._vectors = vectors / norms

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[RetrieveResult]:
        if self._vectors is None:
            raise RetrievalError("向量索引未构建，请先调用 build_index()")
        k = top_k or self.top_k

        q_emb = self.embed_func([query])
        q_vec = np.array(q_emb, dtype=np.float32).flatten()
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm

        scores = self._vectors @ q_vec  # 归一化后点积 = 余弦相似度

        # 取 top_k 索引（降序）
        if len(scores) <= k:
            top_indices = np.argsort(scores)[::-1]
        else:
            top_indices = np.argpartition(scores, -k)[-k:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        results: List[RetrieveResult] = []
        for idx in top_indices:
            score = float(scores[idx])
            if score <= 0:
                continue
            d = self.docs[int(idx)]
            meta = dict(d.metadata)
            meta["retriever"] = "vector"
            results.append(RetrieveResult(
                doc_id=d.doc_id,
                content=d.content,
                score=score,
                metadata=meta,
            ))
        return results
