"""LLM 客户端抽象基类：不同供应商（OpenAI/DeepSeek/通义/豆包等）实现同一接口。"""
from abc import ABC, abstractmethod
from typing import List


class BaseLLMClient(ABC):
    """LLM 客户端抽象基类。"""

    @abstractmethod
    def chat(self, messages: List[dict], **kwargs) -> str:
        """发送聊天消息，返回文本回复。

        参数:
            messages: OpenAI 格式的消息列表，如 [{"role": "user", "content": "..."}]
            **kwargs: 透传给具体实现的参数（model/temperature/max_tokens 等）
        """
        pass

    @abstractmethod
    def embed(self, text: str, **kwargs) -> List[float]:
        """获取文本嵌入向量。

        参数:
            text: 待嵌入的文本
            **kwargs: 透传参数（如 embedding_model）
        """
        pass
