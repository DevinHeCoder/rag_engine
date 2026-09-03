"""P5 API 层：系统路由（健康检查 / 信息）。"""
from fastapi import APIRouter, Depends

from src.api.dependencies import get_service
from src.api.schemas import HealthResponse
from src.api.service import RAGService

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
async def health(service: RAGService = Depends(get_service)):
    """健康检查。"""
    return HealthResponse(
        status="ok",
        app=service.config.get("app", {}).get("name", "rag_engine"),
        version=service.config.get("app", {}).get("version", "0.0.0"),
        docs_indexed=service.num_chunks,
        llm_configured=service.llm_configured,
        retriever=service.config.get("retrieval", {}).get("retriever"),
        reranker=service.config.get("rerank", {}).get("provider"),
    )


@router.get("/")
async def root(service: RAGService = Depends(get_service)):
    """服务信息。"""
    return {
        "service": service.config.get("app", {}).get("name", "rag_engine"),
        "version": service.config.get("app", {}).get("version", "0.0.0"),
        "docs_indexed": service.num_chunks,
        "endpoints": ["/health", "/ingest/text", "/ingest/file", "/query"],
    }
