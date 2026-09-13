"""RAG 评估 CLI：对召回/生成质量做离线评估。

用法::

    # 仅检索指标（无需 LLM）
    python examples/evaluate.py <文档或目录> <评估数据集.json>

    # 检索 + 忠实度（需配置 LLM 环境变量）
    python examples/evaluate.py <文档或目录> <评估数据集.json> --faithful

评估数据集格式见 src/evaluation/dataset.py（[{"question","relevant"}]）。
"""
import argparse
import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chunker import split_documents
from src.document_parser import parse_file
from src.evaluation import Evaluator, SUPPORTED_METRICS, load_dataset
from src.llm_client import build_llm_client
from src.retriever import build_retriever
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger("rag_engine.examples.evaluate")


def _load_documents(path: str, config: dict) -> list:
    """加载单个文件或目录下所有支持的文档。"""
    p = Path(path)
    files = [p] if p.is_file() else sorted(
        f for f in p.rglob("*") if f.suffix.lower() in (".md", ".markdown", ".pdf", ".docx")
    )
    if not files:
        raise SystemExit(f"未找到支持的文档: {path}")

    docs = []
    for f in files:
        logger.info("解析: %s", f)
        docs.extend(parse_file(str(f)))

    ingestion_cfg = config.get("ingestion", {})
    return split_documents(
        docs,
        ingestion_cfg.get("chunker", "fixed"),
        chunk_size=ingestion_cfg.get("chunk_size", 512),
        chunk_overlap=ingestion_cfg.get("chunk_overlap", 64),
    )


def main():
    parser = argparse.ArgumentParser(description="RAG 检索/生成质量评估")
    parser.add_argument("docs", help="文档文件或目录")
    parser.add_argument("dataset", help="评估数据集 JSON")
    parser.add_argument("--top-k", type=int, default=None, help="评估 top_k（默认取配置）")
    parser.add_argument("--faithful", action="store_true", help="额外评估生成忠实度（需 LLM）")
    args = parser.parse_args()

    config = load_config()
    retrieval_cfg = config.get("retrieval", {})
    top_k = args.top_k or retrieval_cfg.get("top_k", 8)

    # 1. 加载文档并分块
    chunks = _load_documents(args.docs, config)
    logger.info("共 %d 个块", len(chunks))

    # 2. 构建召回器
    embed_func = None
    llm = None
    if retrieval_cfg.get("retriever", "hybrid") in ("hybrid", "vector", "milvus"):
        llm_cfg = config.get("llm", {})
        llm = build_llm_client(
            llm_cfg.get("provider", "openai_compatible"),
            base_url=llm_cfg.get("base_url") or None,
            api_key=llm_cfg.get("api_key") or None,
            model=llm_cfg.get("model") or None,
            embedding_model=llm_cfg.get("embedding_model") or None,
        )
        embed_func = lambda texts: [llm.embed(t) for t in texts]

    retriever = build_retriever(
        retrieval_cfg.get("retriever", "hybrid"),
        chunks,
        embed_func=embed_func,
        top_k=top_k,
        rrf_k=retrieval_cfg.get("rrf_k", 60),
        milvus_cfg=retrieval_cfg.get("milvus", {}),
    )

    # 3. 加载评估集
    samples = load_dataset(args.dataset)

    # 4. 评估：默认只算检索指标；--faithful 时追加忠实度
    metrics = [m for m in config.get("evaluation", {}).get("metrics", ["recall@k", "mrr"]) if m in SUPPORTED_METRICS]
    if args.faithful and "faithfulness" not in metrics:
        metrics.append("faithfulness")

    evaluator = Evaluator(llm_client=llm if args.faithful else None, metrics=metrics, default_k=top_k)

    generate_fn = None
    if args.faithful:
        def generate_fn(q):
            cands = retriever.retrieve(q, top_k=top_k)
            return ("(由 CLI 生成的答案，请传入真实生成器)", cands)

    report = evaluator.evaluate(samples, retriever, top_k=top_k, generate_fn=generate_fn)

    # 5. 打印报告
    print("\n" + "=" * 60)
    print("评估报告")
    print("=" * 60)
    ret = report["retrieval"]
    print(f"查询数: {ret['num_queries']}  top_k: {ret['top_k']}")
    print("-" * 60)
    for metric, value in ret["metrics"].items():
        print(f"  {metric:12s}: {value:.4f}")
    if "faithfulness" in report:
        print("-" * 60)
        print(f"  faithfulness : {report['faithfulness']['faithfulness']:.4f}")
    print("=" * 60)

    # 逐条明细
    print("\n逐条明细:")
    for row in ret["per_query"]:
        vals = "  ".join(f"{m}={row[m]:.3f}" for m in ret["metrics"])
        print(f"  {row['question'][:30]:32s} {vals}")


if __name__ == "__main__":
    main()
