from abc import ABC, abstractmethod
from typing import List
from src.document_parser.base import Document

class BaseChunker(ABC):
    @abstractmethod
    def split(self, docs: List[Document]) -> List[Document]:
        """对文档做分块，返回分块后的Document列表"""
        pass