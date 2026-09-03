"""P4 LLM 客户端单测：注册表 + 工厂 + OpenAI 兼容客户端（mock 避免真实 API 调用）。"""
from unittest.mock import MagicMock, patch

import pytest

from src.common.exceptions import LLMError
from src.llm_client import build_llm_client, llm_client_registry
from src.llm_client.openai_client import OpenAICompatibleClient


class TestRegistry:
    def test_openai_compatible_registered(self):
        assert llm_client_registry.has("openai_compatible")


class TestBuildLLMClient:
    def test_default_creates_openai_compatible(self):
        client = build_llm_client(api_key="sk-test", model="test-model")
        assert isinstance(client, OpenAICompatibleClient)
        assert client.model == "test-model"

    def test_custom_base_url(self):
        client = build_llm_client(
            base_url="https://api.deepseek.com",
            api_key="sk-deepseek",
            model="deepseek-chat",
        )
        assert isinstance(client, OpenAICompatibleClient)
        assert client.model == "deepseek-chat"


class TestOpenAICompatibleClient:
    def test_init_with_params(self):
        client = OpenAICompatibleClient(
            base_url="https://api.example.com",
            api_key="sk-123",
            model="gpt-test",
            temperature=0.5,
            max_tokens=512,
        )
        assert client.model == "gpt-test"
        assert client.temperature == 0.5
        assert client.max_tokens == 512

    def test_chat_with_mock(self):
        client = OpenAICompatibleClient(api_key="sk-test", model="test-model")
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = "你好，我是测试回复"
        client.client.chat.completions.create = MagicMock(return_value=mock_resp)

        result = client.chat([{"role": "user", "content": "你好"}])
        assert result == "你好，我是测试回复"
        client.client.chat.completions.create.assert_called_once()
        call_kwargs = client.client.chat.completions.create.call_args
        assert call_kwargs.kwargs["model"] == "test-model"
        assert call_kwargs.kwargs["messages"] == [{"role": "user", "content": "你好"}]

    def test_chat_override_model(self):
        client = OpenAICompatibleClient(api_key="sk-test", model="default-model")
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = "ok"
        client.client.chat.completions.create = MagicMock(return_value=mock_resp)

        client.chat([{"role": "user", "content": "hi"}], model="override-model")
        assert client.client.chat.completions.create.call_args.kwargs["model"] == "override-model"

    def test_chat_empty_content_returns_empty(self):
        client = OpenAICompatibleClient(api_key="sk-test", model="test-model")
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = None
        client.client.chat.completions.create = MagicMock(return_value=mock_resp)
        assert client.chat([{"role": "user", "content": "hi"}]) == ""

    def test_chat_failure_raises_llm_error(self):
        client = OpenAICompatibleClient(api_key="sk-test", model="test-model")
        client.client.chat.completions.create = MagicMock(side_effect=Exception("API 超时"))
        with pytest.raises(LLMError, match="API 超时"):
            client.chat([{"role": "user", "content": "hi"}])

    def test_embed_without_model_raises(self):
        client = OpenAICompatibleClient(api_key="sk-test", model="test-model")
        with pytest.raises(LLMError, match="embedding_model"):
            client.embed("测试文本")

    def test_embed_with_mock(self):
        client = OpenAICompatibleClient(
            api_key="sk-test", model="test-model", embedding_model="embed-v1"
        )
        mock_resp = MagicMock()
        mock_resp.data[0].embedding = [0.1, 0.2, 0.3]
        client.client.embeddings.create = MagicMock(return_value=mock_resp)

        result = client.embed("测试")
        assert result == [0.1, 0.2, 0.3]
        client.client.embeddings.create.assert_called_once_with(
            model="embed-v1", input="测试"
        )
