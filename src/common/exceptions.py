"""
自定义异常层级：所有模块异常统一继承 RAGEngineError，便于上层统一捕获与回滚。
"""
from typing import Optional


class RAGEngineError(Exception):
    """引擎所有异常的基类。"""

    def __init__(self, message: str = "", *, cause: Optional[BaseException] = None) -> None:
        super().__init__(message)
        self.message = message
        self.cause = cause


# --- 配置与基础设施 ---
class ConfigError(RAGEngineError):
    """配置加载 / 解析失败（文件缺失、YAML 语法错误、环境变量未设置等）。"""


class RegistryError(RAGEngineError):
    """可插拔注册表操作失败（实现未注册 / 重复注册 / 类型不符 / 参数不匹配）。"""


# --- 数据侧（P1 / P2）---
class DocumentParsingError(RAGEngineError):
    """文档解析失败（文件损坏、格式不支持、读取异常）。"""


class ChunkingError(RAGEngineError):
    """文档分块失败。"""


# --- 检索侧（P3 / P4）---
class RetrievalError(RAGEngineError):
    """召回失败。"""


class EmbeddingError(RAGEngineError):
    """向量化 / embedding 失败。"""


class RerankError(RAGEngineError):
    """重排失败。"""


class LLMError(RAGEngineError):
    """LLM 调用失败（网络、限流、鉴权、解析等）。"""


# --- 质量（P6）---
class EvaluationError(RAGEngineError):
    """评估失败。"""
