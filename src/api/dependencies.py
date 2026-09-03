"""P5 API 层：依赖注入。从 app.state 取全局 RAGService 单例。"""
from fastapi import Request

from src.api.service import RAGService


def get_service(request: Request) -> RAGService:
    """FastAPI 依赖：获取全局 RAG 服务实例。"""
    service: RAGService = request.app.state.service
    return service
