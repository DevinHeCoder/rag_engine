"""OpenAI 兼容客户端：支持所有 OpenAI 兼容 API（DeepSeek、通义、豆包、本地 vLLM 等）。

通过 base_url 切换供应商，api_key 从配置或环境变量注入。
"""
from typing import Generator, List, Optional

from openai import OpenAI

from src.common.exceptions import LLMError
from src.llm_client.base import BaseLLMClient
from src.llm_client.registry import llm_client_registry


@llm_client_registry.register("openai_compatible")
class OpenAICompatibleClient(BaseLLMClient):
    """OpenAI 兼容 API 客户端。"""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.2,
        max_tokens: int = 1024,
        embedding_model: Optional[str] = None,
        timeout: float = 60.0,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.embedding_model = embedding_model
        try:
            self.client = OpenAI(
                base_url=base_url,
                api_key=api_key or "sk-placeholder",
                timeout=timeout,
            )
        except Exception as e:
            raise LLMError(f"初始化 OpenAI 客户端失败: {e}") from e

    def chat(self, messages: List[dict], **kwargs) -> str:
        try:
            resp = self.client.chat.completions.create(
                model=kwargs.get("model", self.model),
                messages=messages,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
            )
            content = resp.choices[0].message.content
            return content if content else ""
        except Exception as e:
            raise LLMError(f"LLM 聊天调用失败: {e}") from e

    def chat_stream(self, messages: List[dict], **kwargs) -> Generator[str, None, None]:
        try:
            stream = self.client.chat.completions.create(
                model=kwargs.get("model", self.model),
                messages=messages,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                stream=True,
            )
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except Exception as e:
            raise LLMError(f"LLM 流式调用失败: {e}") from e

    def embed(self, text: str, **kwargs) -> List[float]:
        model = kwargs.get("model", self.embedding_model)
        if not model:
            raise LLMError("embed 需要指定 embedding_model（构造参数或 kwargs['model']）")
        try:
            resp = self.client.embeddings.create(model=model, input=text)
            return list(resp.data[0].embedding)
        except Exception as e:
            raise LLMError(f"LLM 嵌入调用失败: {e}") from e
