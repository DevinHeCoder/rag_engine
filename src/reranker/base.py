from abc import ABC, abstractmethod
from typing import List
from src.retriever.base import RetrieveResult

class BaseReranker(ABC):
    @abstractmethod
    def rerank(self, query: str, candidates: List[RetrieveResult], top_k:int) -> List[RetrieveResult]:
        pass