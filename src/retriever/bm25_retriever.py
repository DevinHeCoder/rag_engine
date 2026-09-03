"""BM25 关键词召回器：基于 rank-bm25，支持中英文混合分词。"""
import re
from typing import Callable, List, Optional

from rank_bm25 import BM25Okapi

from src.common.exceptions import RetrievalError
from src.document_parser.base import Document
from src.retriever.base import BaseRetriever, RetrieveResult
from src.retriever.registry import retriever_registry

# 默认分词：英文/数字按词，中文按字
_TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]")


def default_tokenizer(text: str) -> List[str]:
    """中英文混合分词：英文数字按词，中文按单字。"""
    tokens = _TOKEN_PATTERN.findall(text)
    return tokens if tokens else list(text)


@retriever_registry.register("bm25")
class BM25Retriever(BaseRetriever):
    """BM25 关键词召回器。"""

    def __init__(
        self,
        documents: Optional[List[Document]] = None,
        top_k: int = 8,
        tokenizer: Optional[Callable[[str], List[str]]] = None,
    ) -> None:
        self.top_k = top_k
        self.tokenizer = tokenizer or default_tokenizer
        self.docs: List[Document] = []
        self._bm25: Optional[BM25Okapi] = None
        if documents:
            self.build_index(documents)

    def build_index(self, documents: List[Document]) -> None:
        """从文档列表构建 BM25 索引。"""
        if not documents:
            self.docs = []
            self._bm25 = None
            return
        self.docs = list(documents)
        corpus = [self.tokenizer(d.content) for d in self.docs]
        self._bm25 = BM25Okapi(corpus)

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[RetrieveResult]:
        if self._bm25 is None:
            raise RetrievalError("BM25 索引未构建，请先调用 build_index()")
        k = top_k or self.top_k
        tokens = self.tokenizer(query)
        scores = self._bm25.get_scores(tokens)

        # 按分数降序取 top_k
        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]
        results: List[RetrieveResult] = []
        for idx, score in indexed:
            # 精确 0 分 = 无任何匹配词；负分是小语料下常见词 IDF 为负的已知现象，仍应保留
            if float(score) == 0.0:
                continue
            d = self.docs[idx]
            meta = dict(d.metadata)
            meta["retriever"] = "bm25"
            results.append(RetrieveResult(
                doc_id=d.doc_id,
                content=d.content,
                score=float(score),
                metadata=meta,
            ))
        return results
