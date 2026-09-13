"""P5 API 层：query 路由。"""
from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool

from src.api.dependencies import get_service
from src.api.schemas import QueryRequest, QueryResponse, QueryResultItem
from src.api.service import RAGService
from src.common.exceptions import RAGEngineError
from src.utils.logger import get_logger

router = APIRouter(prefix="/query", tags=["query"])

logger = get_logger("rag_engine.api.query")


@router.post("", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    service: RAGService = Depends(get_service),
):
    """端到端查询：召回 → 重排 → 生成。"""
    try:
        answer, results = await run_in_threadpool(
            service.query,
            request.question,
            request.top_k,
            request.rerank,
        )
    except RAGEngineError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # 500 只返回通用消息，内部异常记录到服务端日志（避免泄漏内部细节）
        logger.exception("查询未预期异常: question=%r", request.question)
        raise HTTPException(status_code=500, detail="查询失败，请稍后重试")

    return QueryResponse(
        answer=answer,
        num_results=len(results),
        results=[
            QueryResultItem(
                doc_id=r.doc_id,
                content=r.content,
                score=r.score,
                metadata=r.metadata,
            )
            for r in results
        ],
    )
