"""P5 API 层：FastAPI 应用入口。

启动:
    python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000

或（项目根）:
    uvicorn src.api.main:app --reload
"""
from typing import Optional

from fastapi import FastAPI

from src.api.routes import api_router
from src.api.service import RAGService
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger("rag_engine.api.main")


def create_app(config: Optional[dict] = None, service: Optional[RAGService] = None) -> FastAPI:
    """创建 FastAPI 应用。可注入 config / service 便于测试。"""
    cfg = config or load_config()
    app_cfg = cfg.get("app", {})

    app = FastAPI(
        title=app_cfg.get("name", "RAG Engine"),
        version=app_cfg.get("version", "0.1.0"),
        description="企业可插拔式 RAG 检索引擎 API",
    )

    # 全局单例服务
    app.state.service = service or RAGService(cfg)

    # 路由
    app.include_router(api_router)
    return app


# 默认应用实例（uvicorn 入口）
app = create_app()
