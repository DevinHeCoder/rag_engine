"""P5 API 层：query 路由。"""
import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse

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


@router.post("/stream")
async def query_stream(
    request: QueryRequest,
    service: RAGService = Depends(get_service),
):
    """流式查询：先返回候选来源（JSON），再逐段推送回答文本（SSE）。"""

    async def event_generator():
        try:
            candidates, token_gen = await run_in_threadpool(
                service.query_stream,
                request.question,
                request.top_k,
                request.rerank,
            )
            sources = [
                {"doc_id": c.doc_id, "content": c.content, "score": c.score}
                for c in candidates
            ]
            yield f"data: {json.dumps({'type': 'sources', 'data': sources}, ensure_ascii=False)}\n\n"
            for token in token_gen:
                yield f"data: {json.dumps({'type': 'token', 'data': token}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        except RAGEngineError as e:
            yield f"data: {json.dumps({'type': 'error', 'data': str(e)}, ensure_ascii=False)}\n\n"
        except Exception as e:
            logger.exception("流式查询异常")
            yield f"data: {json.dumps({'type': 'error', 'data': '查询失败'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
