"""P5 API 层：ingest 路由。"""
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from src.api.dependencies import get_service
from src.api.schemas import ChunkItem, IngestResponse, IngestTextRequest
from src.api.service import RAGService
from src.common.exceptions import RAGEngineError

router = APIRouter(prefix="/ingest", tags=["ingest"])


@router.post("/text", response_model=IngestResponse)
async def ingest_text(
    request: IngestTextRequest,
    service: RAGService = Depends(get_service),
):
    """摄入纯文本内容。"""
    try:
        chunks = await run_in_threadpool(
            service.ingest_text,
            request.content,
            request.doc_id,
            request.chunker,
        )
    except RAGEngineError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # 兜底：分块器等未预期异常
        raise HTTPException(status_code=500, detail=f"摄入失败: {e}")

    resolved_doc_id = request.doc_id
    if not resolved_doc_id:
        resolved_doc_id = chunks[0].doc_id.split("::", 1)[0]

    return IngestResponse(
        doc_id=resolved_doc_id,
        num_chunks=len(chunks),
        chunks=[ChunkItem(doc_id=c.doc_id, content=c.content, metadata=c.metadata) for c in chunks],
    )


@router.post("/file", response_model=IngestResponse)
async def ingest_file(
    file: UploadFile = File(...),
    chunker: str = Form(None),
    service: RAGService = Depends(get_service),
):
    """上传文件（.md/.pdf/.docx）摄入。"""
    # 校验扩展名
    suffix = "." + (file.filename or "").rsplit(".", 1)[-1].lower() if "." in (file.filename or "") else ""
    if suffix not in (".md", ".markdown", ".pdf", ".docx"):
        raise HTTPException(status_code=400, detail=f"不支持的文件类型: {suffix or '未知'}")

    from pathlib import Path

    # 以原始文件名 stem 作为 doc_id（而非临时文件随机名）
    resolved_doc_id = Path(file.filename).stem

    # 保存到临时文件后解析
    content_bytes = await file.read()
    tmp_path = _write_temp(file.filename or "upload", content_bytes)
    try:
        chunks = await run_in_threadpool(service.ingest_file, tmp_path, resolved_doc_id, chunker)
    except RAGEngineError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"摄入失败: {e}")
    finally:
        _remove_temp(tmp_path)

    return IngestResponse(
        doc_id=resolved_doc_id,
        num_chunks=len(chunks),
        chunks=[ChunkItem(doc_id=c.doc_id, content=c.content, metadata=c.metadata) for c in chunks],
    )


def _write_temp(filename: str, content: bytes) -> str:
    import tempfile
    from pathlib import Path

    suffix = Path(filename).suffix or ".md"
    fd, path = tempfile.mkstemp(suffix=suffix, prefix="rag_ingest_")
    with open(fd, "wb") as f:
        f.write(content)
    return path


def _remove_temp(path: str) -> None:
    import os

    try:
        os.remove(path)
    except OSError:
        pass
