from typing import List
import numpy as np
from src.retriever.base import BaseRetriever, RetrieveResult
from src.retriever.registry import retriever_registry


@retriever_registry.register("hybrid")
class HybridRetriever(BaseRetriever):
    def __init__(
        self,
        vector_retriever: BaseRetriever,
        bm25_retriever: BaseRetriever,
        rrf_k: int = 60,       # RRF超参，常用60
        top_k: int = 8
    ):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.rrf_k = rrf_k
        self.top_k = top_k

    def retrieve(self, query: str, top_k: int = None) -> List[RetrieveResult]:
        if top_k is None:
            top_k = self.top_k

        # 1.两路独立召回
        vec_results = self.vector_retriever.retrieve(query, top_k=top_k*2)
        bm25_results = self.bm25_retriever.retrieve(query, top_k=top_k*2)

        # 2.RRF打分融合
        rrf_score = {}

        def add_rrf(items: List[RetrieveResult]):
            for rank, item in enumerate(items):
                doc_id = item.doc_id
                # RRF公式：score += 1 / (rank + k)
                s = 1.0 / (rank + 1 + self.rrf_k)
                if doc_id in rrf_score:
                    rrf_score[doc_id] += s
                else:
                    rrf_score[doc_id] = s

        add_rrf(vec_results)
        add_rrf(bm25_results)

        # 3.把原始对象拿出来，按RRF分数降序
        all_docs = {item.doc_id: item for item in vec_results + bm25_results}
        sorted_doc_ids = sorted(rrf_score.keys(), key=lambda x: rrf_score[x], reverse=True)

        output = []
        for did in sorted_doc_ids[:top_k]:
            origin_item = all_docs[did]
            output.append(RetrieveResult(
                doc_id=origin_item.doc_id,
                content=origin_item.content,
                score=rrf_score[did],   # 使用RRF融合后的分数
                metadata=origin_item.metadata
            ))
        return output
