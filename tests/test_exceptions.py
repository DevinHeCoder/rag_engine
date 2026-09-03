"""P0 异常层级单测：统一继承 RAGEngineError，可携带 cause。"""
from src.common.exceptions import (
    ChunkingError,
    ConfigError,
    DocumentParsingError,
    EmbeddingError,
    EvaluationError,
    LLMError,
    RAGEngineError,
    RegistryError,
    RerankError,
    RetrievalError,
)


def test_all_exceptions_subclass_base():
    for exc in (
        ConfigError,
        RegistryError,
        DocumentParsingError,
        ChunkingError,
        RetrievalError,
        EmbeddingError,
        RerankError,
        LLMError,
        EvaluationError,
    ):
        assert issubclass(exc, RAGEngineError)


def test_message_and_cause():
    cause = ValueError("inner")
    e = ConfigError("outer failed", cause=cause)
    assert e.message == "outer failed"
    assert e.cause is cause
    assert str(e) == "outer failed"
