"""
召回器抽象基类，不同召回器实现子类

关键点：HybridRetriever继承BaseRetriever，内部调用向量 retriever + bm25 retriever，做 RRF 权重融合
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

@dataclass
class RetrieveResult:
    doc_id: str
    content: str
    score: float
    metadata: dict

class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int) -> List[RetrieveResult]:
        """输入query，返回召回结果列表"""
        pass