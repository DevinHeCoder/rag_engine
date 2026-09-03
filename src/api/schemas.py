"""P5 API 层：Pydantic 请求/响应模型。"""
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------- 请求模型 ----------------

class IngestTextRequest(BaseModel):
    """以纯文本方式摄入文档。"""
    content: str = Field(..., min_length=1, description="文档正文内容")
    doc_id: Optional[str] = Field(None, description="文档标识，默认取文件名/时间戳")
    chunker: Optional[str] = Field(None, description="分块策略，覆盖全局配置")


class QueryRequest(BaseModel):
    """查询请求。"""
    question: str = Field(..., min_length=1, description="用户问题")
    top_k: Optional[int] = Field(None, ge=1, le=50, description="召回候选数，覆盖全局配置")
    rerank: Optional[bool] = Field(None, description="是否启用重排，默认取全局配置")


# ---------------- 响应模型 ----------------

class ChunkItem(BaseModel):
    doc_id: str
    content: str
    metadata: Dict[str, object] = Field(default_factory=dict)


class IngestResponse(BaseModel):
    success: bool = True
    doc_id: str
    num_chunks: int
    chunks: List[ChunkItem] = Field(default_factory=list)


class QueryResultItem(BaseModel):
    doc_id: str
    content: str
    score: float
    metadata: Dict[str, object] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    answer: str
    num_results: int
    results: List[QueryResultItem] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    app: str
    version: str
    docs_indexed: int
    llm_configured: bool
    retriever: Optional[str] = None
    reranker: Optional[str] = None
