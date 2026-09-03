"""
llm_client 包：统一 LLM 客户端入口。

用法::

    from src.llm_client import build_llm_client
    llm = build_llm_client("openai_compatible", base_url="https://api.deepseek.com", api_key="sk-xxx", model="deepseek-chat")
    answer = llm.chat([{"role": "user", "content": "你好"}])
"""
from typing import Optional

from src.llm_client.base import BaseLLMClient
from src.llm_client.registry import llm_client_registry

# 导入实现以触发 @register 装饰器注册
from src.llm_client import openai_client  # noqa: E402,F401

__all__ = [
    "BaseLLMClient",
    "llm_client_registry",
    "build_llm_client",
]


def build_llm_client(
    name: str = "openai_compatible",
    *,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs,
) -> BaseLLMClient:
    """工厂函数：创建 LLM 客户端。

    参数:
        name: 注册表中的实现名（默认 openai_compatible）
        base_url: API 地址（切换供应商）
        api_key: API 密钥
        model: 聊天模型名
        **kwargs: 透传给构造函数（temperature/max_tokens/embedding_model/timeout 等）
    """
    params = dict(kwargs)
    if base_url is not None:
        params["base_url"] = base_url
    if api_key is not None:
        params["api_key"] = api_key
    if model is not None:
        params["model"] = model
    return llm_client_registry.create(name, **params)
