# ==================== RAG Engine 部署镜像 ====================
# 多阶段构建：依赖层（venv） + 运行层
# 构建:  docker build -t rag-engine .
# 运行:  docker run -p 8000:8000 -e RAG_LLM_API_KEY=sk-xxx rag-engine

# --- 阶段 1：构建依赖到虚拟环境 ---
FROM python:3.13-slim AS builder
WORKDIR /build
COPY requirements.txt .
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# --- 阶段 2：运行层（仅复制 venv + 应用代码，镜像更小） ---
FROM python:3.13-slim AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    RAG_CONFIG_PATH=config/settings.yaml

WORKDIR /app

# 从构建阶段复制已安装的依赖
COPY --from=builder /opt/venv /opt/venv

# 复制应用代码与配置
COPY src/ ./src/
COPY config/ ./config/
COPY requirements.txt .

# 安全实践：非 root 运行
RUN useradd --create-home --uid 1000 raguser \
    && mkdir -p /app/logs \
    && chown -R raguser:raguser /app
USER raguser

EXPOSE 8000

# 健康检查（Python 标准库，无需 curl）
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1

CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
