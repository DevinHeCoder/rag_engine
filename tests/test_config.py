"""P0 配置模块单测：YAML 加载、${ENV} 展开、深合并、环境变量路径覆盖。"""
from src.common.exceptions import ConfigError
from src.utils.config_loader import (
    deep_merge,
    expand_env,
    load_config,
    load_yaml,
)


class TestLoadYaml:
    def test_normal(self, tmp_path):
        p = tmp_path / "a.yaml"
        p.write_text("a: 1\nb:\n  c: hello\n", encoding="utf-8")
        assert load_yaml(p) == {"a": 1, "b": {"c": "hello"}}

    def test_empty_returns_empty_dict(self, tmp_path):
        p = tmp_path / "empty.yaml"
        p.write_text("", encoding="utf-8")
        assert load_yaml(p) == {}

    def test_missing_raises(self):
        import pytest
        with pytest.raises(ConfigError):
            load_yaml("/no/such/file.yaml")

    def test_bad_root_type(self, tmp_path):
        import pytest
        p = tmp_path / "list.yaml"
        p.write_text("- a\n- b\n", encoding="utf-8")
        with pytest.raises(ConfigError):
            load_yaml(p)

    def test_bad_syntax(self, tmp_path):
        import pytest
        p = tmp_path / "bad.yaml"
        p.write_text("a: [1, 2\n", encoding="utf-8")
        with pytest.raises(ConfigError):
            load_yaml(p)


class TestExpandEnv:
    def test_replace(self, monkeypatch):
        monkeypatch.setenv("RAG_TEST_XYZ", "abc")
        assert expand_env("${RAG_TEST_XYZ}") == "abc"
        assert expand_env({"k": ["${RAG_TEST_XYZ}", 1]}) == {"k": ["abc", 1]}

    def test_default_used_when_unset(self, monkeypatch):
        monkeypatch.delenv("RAG_TEST_UNSET_VAR", raising=False)
        assert expand_env("${RAG_TEST_UNSET_VAR:fallback}") == "fallback"

    def test_missing_without_default_raises(self, monkeypatch):
        monkeypatch.delenv("RAG_TEST_UNSET_VAR", raising=False)
        import pytest
        with pytest.raises(ConfigError):
            expand_env("${RAG_TEST_UNSET_VAR}")


class TestDeepMerge:
    def test_nested_merge_and_type_override(self):
        base = {"a": 1, "b": {"x": 1, "y": 2}, "c": [1]}
        override = {"b": {"y": 3}, "c": [2], "d": 4}
        assert deep_merge(base, override) == {
            "a": 1, "b": {"x": 1, "y": 3}, "c": [2], "d": 4,
        }
        assert base["b"]["y"] == 2  # 不修改入参


class TestLoadConfig:
    def test_default_config(self, monkeypatch):
        # 屏蔽外部 RAG_ 环境变量干扰，加载项目默认配置
        for key in list(__import__("os").environ):
            if key.startswith("RAG_"):
                monkeypatch.delenv(key, raising=False)
        cfg = load_config()
        assert cfg["app"]["name"] == "rag_engine"
        assert cfg["retrieval"]["retriever"] == "hybrid"
        assert cfg["ingestion"]["chunker"] == "fixed"
        # LLM 密钥未设置时默认展开为空串（不会报错）
        assert cfg["llm"]["api_key"] == ""

    def test_env_path_override(self, monkeypatch):
        monkeypatch.setenv("RAG_RETRIEVAL__TOP_K", "16")
        monkeypatch.setenv("RAG_RERANK__ENABLED", "false")
        cfg = load_config()
        assert cfg["retrieval"]["top_k"] == 16
        assert cfg["rerank"]["enabled"] is False
