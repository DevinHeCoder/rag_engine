"""RAG 端到端查询示例：文档解析 → 分块 → 召回 → 重排 → LLM 生成答案。

使用前请配置环境变量：
    RAG_LLM_BASE_URL  - API 地址（如 https://api.deepseek.com）
    RAG_LLM_API_KEY   - API 密钥
    RAG_LLM_MODEL     - 模型名（如 deepseek-chat）

运行：
    python examples/query_rag.py path/to/document.md "你的问题"
"""
import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chunker import split_documents
from src.document_parser import parse_file
from src.llm_client import build_llm_client
from src.reranker import build_reranker
from src.retriever import build_retriever
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger("rag_engine.examples.query_rag")


def build_rag_pipeline(config: dict, docs: list):
    """构建 RAG 流水线：召回器 + 重排器 + LLM 客户端。"""
    llm_cfg = config.get("llm", {})
    llm = build_llm_client(
        llm_cfg.get("provider", "openai_compatible"),
        base_url=llm_cfg.get("base_url") or None,
        api_key=llm_cfg.get("api_key") or None,
        model=llm_cfg.get("model") or None,
        embedding_model=llm_cfg.get("embedding_model") or None,
        temperature=llm_cfg.get("temperature", 0.2),
        max_tokens=llm_cfg.get("max_tokens", 1024),
    )

    # 用 LLM 的 embed 方法作为 vector_retriever 的嵌入函数
    def embed_func(texts):
        return [llm.embed(t) for t in texts]

    retrieval_cfg = config.get("retrieval", {})
    retriever = build_retriever(
        retrieval_cfg.get("retriever", "hybrid"),
        docs,
        embed_func=embed_func,
        top_k=retrieval_cfg.get("top_k", 8),
        rrf_k=retrieval_cfg.get("rrf_k", 60),
    )

    rerank_cfg = config.get("rerank", {})
    reranker = None
    if rerank_cfg.get("enabled", True):
        reranker = build_reranker(
            rerank_cfg.get("provider", "llm"),
            llm_client=llm,
            top_k=rerank_cfg.get("top_n", 5),
        )

    return retriever, reranker, llm


def generate_answer(llm, query: str, contexts: list) -> str:
    """用检索到的上下文生成答案。"""
    context_text = "\n\n---\n\n".join(
        f"[{i+1}] {c.content}" for i, c in enumerate(contexts)
    )
    messages = [
        {"role": "system", "content": "你是一个知识助手。请根据以下参考文档回答用户问题，如果文档中没有相关信息，请如实说明。"},
        {"role": "user", "content": f"参考文档：\n{context_text}\n\n问题：{query}\n\n回答："},
    ]
    return llm.chat(messages)


def main():
    if len(sys.argv) < 3:
        print("用法: python examples/query_rag.py <文档路径> <问题>")
        sys.exit(1)

    doc_path = sys.argv[1]
    query = sys.argv[2]

    config = load_config()

    # 1. 解析文档
    logger.info("解析文档: %s", doc_path)
    raw_docs = parse_file(doc_path)

    # 2. 分块
    ingestion_cfg = config.get("ingestion", {})
    chunks = split_documents(
        raw_docs,
        ingestion_cfg.get("chunker", "fixed"),
        chunk_size=ingestion_cfg.get("chunk_size", 512),
        chunk_overlap=ingestion_cfg.get("chunk_overlap", 64),
    )
    logger.info("分块完成: %d 个块", len(chunks))

    # 3. 构建流水线
    retriever, reranker, llm = build_rag_pipeline(config, chunks)

    # 4. 召回
    retrieval_cfg = config.get("retrieval", {})
    candidates = retriever.retrieve(query, top_k=retrieval_cfg.get("top_k", 8))
    logger.info("召回 %d 个候选", len(candidates))

    # 5. 重排
    if reranker and candidates:
        candidates = reranker.rerank(query, candidates)
        logger.info("重排后保留 %d 个", len(candidates))

    # 6. 生成答案
    answer = generate_answer(llm, query, candidates[:3])

    print("\n" + "=" * 60)
    print(f"问题: {query}")
    print("=" * 60)
    print(f"\n答案:\n{answer}")
    print("\n" + "-" * 60)
    print("参考来源:")
    for i, c in enumerate(candidates[:3]):
        print(f"  [{i+1}] {c.doc_id} (score={c.score:.4f})")


if __name__ == "__main__":
    main()