"""批量摄入文档示例：解析 → 分块 → 打印结果统计。

用法::

    python examples/ingest_docs.py path/to/document.md          # 单个文件
    python examples/ingest_docs.py path/to/docs_dir             # 目录（递归）
"""
import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chunker import split_documents
from src.document_parser import parse_file
from src.utils.config_loader import load_config

SUPPORTED = (".md", ".markdown", ".pdf", ".docx")


def main():
    if len(sys.argv) < 2:
        print("用法: python examples/ingest_docs.py <文档或目录> [chunker]")
        sys.exit(1)

    path = Path(sys.argv[1])
    chunker = sys.argv[2] if len(sys.argv) > 2 else None

    files = [path] if path.is_file() else sorted(
        f for f in path.rglob("*") if f.suffix.lower() in SUPPORTED
    )
    if not files:
        print(f"未找到支持的文档（{SUPPORTED}）: {path}")
        sys.exit(1)

    config = load_config()
    ingestion_cfg = config.get("ingestion", {})
    chunker = chunker or ingestion_cfg.get("chunker", "fixed")

    total_docs = 0
    total_chunks = 0
    for f in files:
        docs = parse_file(str(f))
        chunks = split_documents(
            docs,
            chunker,
            chunk_size=ingestion_cfg.get("chunk_size", 512),
            chunk_overlap=ingestion_cfg.get("chunk_overlap", 64),
        )
        print(f"{f.name:24s} -> {len(docs)} 段 -> {len(chunks)} 块")
        total_docs += len(docs)
        total_chunks += len(chunks)

    print("-" * 50)
    print(f"共 {len(files)} 个文件, {total_docs} 段, {total_chunks} 块 (chunker={chunker})")


if __name__ == "__main__":
    main()
