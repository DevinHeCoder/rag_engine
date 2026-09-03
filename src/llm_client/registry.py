"""llm_client 层的可插拔注册表实例。"""
from src.common.registry import Registry
from src.llm_client.base import BaseLLMClient

llm_client_registry = Registry(
    name="llm_client",
    base_type=BaseLLMClient,
)
