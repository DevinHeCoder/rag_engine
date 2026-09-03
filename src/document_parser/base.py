from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

@dataclass
class Document:
    doc_id: str
    content: str
    metadata: dict  # 来源、页码、文件名等

class BaseDocumentParser(ABC):
    """文档解析抽象基类，不同格式实现子类"""
    @abstractmethod
    def parse(self, file_path: str) -> List[Document]:
        """输入文件路径，输出统一Document列表"""
        pass