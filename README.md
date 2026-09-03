# 企业可插拔式 RAG 检索引擎

从零实现的**可插拔式企业 RAG 检索引擎**（AI 应用工程师方向作品集项目）。每层抽象 + 注册表 + 配置选型，新增实现只需写类加装饰器，上层零改动。

```
文档 → 解析(P1) → 分块(P2) → 召回(P3) → 重排(P4) → 生成(P4) → API(P5) → 评估(P6) → 部署(P7)
```

## 特性

- **可插拔分层架构**：文档解析 / 分块 / 召回 / 重排 / LLM 客户端 五层，均基于「抽象基类 + 注册表 + YAML 选型」
- **三种召回策略**：BM25 关键词召回、向量语义召回、Hybrid（RRF 融合）
- **三种分块策略**：Fixed（滑动窗口自然边界）、Hierarchy（标题层级）、Semantic（嵌入突变）
- **三种文档解析**：Markdown / PDF / Word
- **LLM 重排 + 生成**：OpenAI 兼容接口，一行切换 DeepSeek / 通义 / 豆包 / 本地 vLLM
- **FastAPI 服务**：`/ingest`（文本/文件上传）、`/query`（端到端）、`/health`
- **离线评估**：recall@k / precision@k / MRR / hit@k + 可选 LLM 忠实度
- **156 个单元测试**，全链路闭环验证

## 架构

```
┌─────────────────────────────────────────────────────────┐
│  API 层 (P5)   FastAPI / RAGService（依赖注入）           │
│  /ingest /query /health                                  │
├─────────────────────────────────────────────────────────┤
│  评估层 (P6)   Evaluator（recall@k / MRR / faithfulness）│
├──────────────┬──────────────┬──────────────┬───────────┤
│  解析 (P1)   │  分块 (P2)   │  召回 (P3)   │ 重排 (P4) │
│  md/pdf/word │ fixed/hier/  │ bm25/vector/ │  llm /    │
│              │ semantic     │ hybrid(RRF)  │ cross-enc │
├──────────────┴──────────────┴──────────────┴───────────┤
│  LLM 客户端 (P4)  OpenAI 兼容（DeepSeek/通义/豆包/vLLM）  │
├─────────────────────────────────────────────────────────┤
│  公共层 (P0)  Registry / Exceptions / Config / Logger    │
└─────────────────────────────────────────────────────────┘
```

**可插拔机制**：每层一个 `Registry[T]`（`src/common/registry.py`），实现类加 `@registry.register("name")` 装饰器即自动注册；`config/settings.yaml` 的 `name` 字段决定选型，支持 `RAG_*` 环境变量深路径覆盖。

## 目录结构

```
rag_engine/
├── src/
│   ├── common/          # P0: Registry / Exceptions / Constants
│   ├── utils/           # P0: config_loader（YAML+环境变量）/ logger
│   ├── document_parser/ # P1: md / pdf / word 解析器
│   ├── chunker/         # P2: fixed / hierarchy / semantic 分块器
│   ├── retriever/       # P3: bm25 / vector / hybrid 召回器
│   ├── llm_client/      # P4: OpenAI 兼容客户端
│   ├── reranker/        # P4: LLM 重排器
│   ├── evaluation/      # P6: 指标 / 数据集 / 忠实度 / 评估器
│   └── api/             # P5: FastAPI 应用 / 路由 / 服务层
├── config/              # settings.yaml / logging.yaml
├── examples/            # query_rag.py（端到端）/ evaluate.py（评估 CLI）
├── tests/               # 156 个单元测试
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

## 快速开始

### 本地运行

```bash
# 1. 安装依赖（Python 3.13）
pip install -r requirements.txt

# 2. 配置 LLM（可选，不配也能启动，但 /query 需要）
# Windows PowerShell:
$env:RAG_LLM_BASE_URL="https://api.deepseek.com"
$env:RAG_LLM_API_KEY="sk-xxx"
$env:RAG_LLM_MODEL="deepseek-chat"

# 3. 启动 API 服务
python -m uvicorn src.api.main:app --reload
# 打开 http://127.0.0.1:8000/docs 查看 Swagger 文档
```

### Docker 部署

```bash
# 方式一：docker compose（推荐）
cp .env.example .env   # 填入 LLM 密钥
docker compose up -d --build

# 方式二：直接构建
docker build -t rag-engine .
docker run -p 8000:8000 -e RAG_LLM_API_KEY=sk-xxx rag-engine
```

## API 文档

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 健康检查 + 索引状态 + 选型信息 |
| POST | `/ingest/text` | 摄入纯文本，返回分块结果 |
| POST | `/ingest/file` | 上传 `.md` / `.pdf` / `.docx` 摄入 |
| POST | `/query` | 端到端查询：召回 → 重排 → 生成 |

```bash
# 摄入文本
curl -X POST http://localhost:8000/ingest/text \
  -H "Content-Type: application/json" \
  -d '{"content":"苹果是红色水果，富含维生素C","doc_id":"kb-1"}'

# 端到端查询
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"什么水果富含维生素？"}'
```

## 可插拔扩展指南

新增一个召回算法（以 `vector` 为参考）：

```python
from src.common.registry import registry_aware  # 或直接使用层内注册表
from src.retriever.base import BaseRetriever
from src.retriever.registry import retriever_registry

@retriever_registry.register("my_retriever")
class MyRetriever(BaseRetriever):
    def __init__(self, documents, **kwargs):
        ...
    def retrieve(self, query, top_k):
        ...
```

然后在 `config/settings.yaml` 把 `retrieval.retriever` 改为 `my_retriever` 即可，上层（服务层/API/评估）零改动。

各层注册表：

| 层 | 注册表 | 已注册实现 |
|---|---|---|
| 解析 | `document_parser_registry` | md / pdf / word |
| 分块 | `chunker_registry` | fixed / hierarchy / semantic |
| 召回 | `retriever_registry` | bm25 / vector / hybrid |
| 重排 | `reranker_registry` | llm |
| LLM | `llm_client_registry` | openai_compatible |

## 离线评估

```bash
# 仅检索指标（无需 LLM）
python examples/evaluate.py <文档或目录> <评估数据集.json>

# 含生成忠实度（需配置 LLM 环境变量）
python examples/evaluate.py <文档或目录> <评估数据集.json> --faithful
```

评估数据集格式（JSON）：

```json
[
  {"question": "什么水果富含维生素？", "relevant": ["apple"]},
  {"question": "汽车用什么驱动？", "relevant": ["car"]}
]
```

## 端到端示例

```bash
python examples/query_rag.py path/to/document.md "你的问题"
```

## 配置

- `config/settings.yaml`：全局配置（各层可插拔选型 + LLM 密钥占位）
- 环境变量覆盖：`RAG_LLM__MODEL=xxx`（`RAG_` 前缀 + 双下划线分层）
- 密钥注入：`${RAG_LLM_API_KEY:}`（未设置且无默认值时启动报错，避免静默用错配置）

## 测试

```bash
python -m pytest tests -q   # 156 passed
```

## 技术栈

Python 3.13 · FastAPI · Pydantic · numpy · rank-bm25 · pypdf · python-docx · openai · PyYAML
