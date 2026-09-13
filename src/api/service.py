"""P5 API 层：RAG 服务层（不依赖 HTTP 的纯业务逻辑，便于单元测试）。

职责：
    - 管理内存文档索引（ingest 后切块并保存）
    - 懒加载 LLM 客户端 / 召回器 / 重排器（按配置可插拔选型）
    - 提供 ingest / query 两个核心操作
"""
import threading
import time
from typing import Callable, Dict, List, Optional

from src.chunker import split_documents
from src.common.exceptions import RAGEngineError
from src.document_parser import parse_file
from src.document_parser.base import Document
from src.llm_client import build_llm_client
from src.llm_client.base import BaseLLMClient
from src.reranker import build_reranker
from src.retriever import build_retriever
from src.retriever.base import RetrieveResult
from src.utils.logger import get_logger

logger = get_logger("rag_engine.api.service")

EmbedFunc = Callable[[List[str]], List[List[float]]]


class RAGService:
    """RAG 服务：内存版文档索引 + 端到端查询。

    支持注入 llm_client / embed_func / retriever / reranker 以便测试与替换；
    未注入时按 config 懒加载真实组件。
    """

    def __init__(
        self,
        config: dict,
        llm_client: Optional[BaseLLMClient] = None,
        embed_func: Optional[EmbedFunc] = None,
        retriever=None,
        reranker=None,
    ) -> None:
        self.config = config
        self._llm = llm_client
        self._embed_func = embed_func
        self._retriever = retriever
        self._reranker = reranker

        self._chunks: List[Document] = []
        self._lock = threading.Lock()

    # ---------------- 只读信息 ----------------

    @property
    def num_chunks(self) -> int:
        return len(self._chunks)

    @property
    def llm_configured(self) -> bool:
        return self._llm is not None or bool(
            self.config.get("llm", {}).get("api_key")
        )

    # ---------------- 组件懒加载 ----------------

    def _get_llm(self) -> BaseLLMClient:
        if self._llm is None:
            llm_cfg = self.config.get("llm", {})
            self._llm = build_llm_client(
                llm_cfg.get("provider", "openai_compatible"),
                base_url=llm_cfg.get("base_url") or None,
                api_key=llm_cfg.get("api_key") or None,
                model=llm_cfg.get("model") or None,
                embedding_model=llm_cfg.get("embedding_model") or None,
                temperature=llm_cfg.get("temperature", 0.2),
                max_tokens=llm_cfg.get("max_tokens", 1024),
            )
            logger.info("懒加载 LLM 客户端: %s", llm_cfg.get("provider", "openai_compatible"))
        return self._llm

    def _get_embed_func(self) -> EmbedFunc:
        if self._embed_func is None:
            llm = self._get_llm()
            self._embed_func = lambda texts: [llm.embed(t) for t in texts]
        return self._embed_func

    def _get_retriever(self):
        if self._retriever is None:
            retrieval_cfg = self.config.get("retrieval", {})
            self._retriever = build_retriever(
                retrieval_cfg.get("retriever", "hybrid"),
                self._chunks,
                embed_func=self._get_embed_func(),
                top_k=retrieval_cfg.get("top_k", 8),
                rrf_k=retrieval_cfg.get("rrf_k", 60),
                milvus_cfg=retrieval_cfg.get("milvus", {}),
            )
            logger.info("懒加载召回器: %s", retrieval_cfg.get("retriever", "hybrid"))
        return self._retriever

    def _get_reranker(self):
        if self._reranker is None:
            rerank_cfg = self.config.get("rerank", {})
            if rerank_cfg.get("enabled", True):
                self._reranker = build_reranker(
                    rerank_cfg.get("provider", "llm"),
                    llm_client=self._get_llm(),
                    top_k=rerank_cfg.get("top_n", 5),
                )
                logger.info("懒加载重排器: %s", rerank_cfg.get("provider", "llm"))
        return self._reranker

    # ---------------- 摄入 ----------------

    def ingest_text(
        self,
        content: str,
        doc_id: Optional[str] = None,
        chunker: Optional[str] = None,
    ) -> List[Document]:
        """按文本摄入：切块并加入内存索引。"""
        if not content or not content.strip():
            raise RAGEngineError("content 不能为空")

        doc_id = doc_id or f"doc-{int(time.time() * 1000)}"
        docs = [Document(doc_id=doc_id, content=content, metadata={"source": "text"})]

        ingestion_cfg = self.config.get("ingestion", {})
        chunks = split_documents(
            docs,
            chunker or ingestion_cfg.get("chunker", "fixed"),
            chunk_size=ingestion_cfg.get("chunk_size", 512),
            chunk_overlap=ingestion_cfg.get("chunk_overlap", 64),
        )
        if not chunks:
            raise RAGEngineError(f"文档 {doc_id} 分块后为空")

        with self._lock:
            self._chunks.extend(chunks)
            self._retriever = None  # 索引变化，召回器需重建

        logger.info("摄入 %s: %d 块，当前共 %d 块", doc_id, len(chunks), self.num_chunks)
        return chunks

    def ingest_file(
        self,
        file_path: str,
        doc_id: Optional[str] = None,
        chunker: Optional[str] = None,
    ) -> List[Document]:
        """按文件摄入：解析 → 切块 → 加入内存索引。

        doc_id 用于覆盖解析器默认的 doc_id 前缀（通常传原始文件名 stem）。
        """
        docs = parse_file(file_path)
        if doc_id:
            for d in docs:
                seq = d.doc_id.rsplit("::", 1)[-1]
                d.doc_id = f"{doc_id}::{seq}"

        ingestion_cfg = self.config.get("ingestion", {})
        chunks = split_documents(
            docs,
            chunker or ingestion_cfg.get("chunker", "fixed"),
            chunk_size=ingestion_cfg.get("chunk_size", 512),
            chunk_overlap=ingestion_cfg.get("chunk_overlap", 64),
        )
        if not chunks:
            raise RAGEngineError(f"文件 {file_path} 分块后为空")

        with self._lock:
            self._chunks.extend(chunks)
            self._retriever = None

        logger.info("摄入文件 %s: %d 块，当前共 %d 块", file_path, len(chunks), self.num_chunks)
        return chunks

    # ---------------- 查询 ----------------

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        rerank: Optional[bool] = None,
    ) -> tuple[str, List[RetrieveResult]]:
        """端到端查询：召回 → 重排 → 生成，返回 (答案, 结果列表)。"""
        if not self._chunks:
            raise RAGEngineError("索引为空，请先摄入文档")

        retrieval_cfg = self.config.get("retrieval", {})
        k = top_k or retrieval_cfg.get("top_k", 8)

        # 1. 召回
        retriever = self._get_retriever()
        candidates = retriever.retrieve(question, top_k=k)
        logger.info("召回 %d 个候选", len(candidates))

        # 2. 重排
        rerank_cfg = self.config.get("rerank", {})
        use_rerank = rerank if rerank is not None else rerank_cfg.get("enabled", True)
        if use_rerank and candidates:
            reranker = self._get_reranker()
            if reranker is not None:
                candidates = reranker.rerank(question, candidates)
                logger.info("重排后保留 %d 个", len(candidates))

        # 3. 生成
        answer = self._generate(question, candidates)
        return answer, candidates

    def _generate(self, question: str, candidates: List[RetrieveResult]) -> str:
        llm = self._get_llm()
        context_text = "\n\n---\n\n".join(
            f"[{i+1}] {c.content}" for i, c in enumerate(candidates[:3])
        )
        messages = [
            {"role": "system", "content": (
                "你是一个知识助手。请根据以下参考文档回答用户问题，"
                "如果文档中没有相关信息，请如实说明，不要编造。"
            )},
            {"role": "user", "content": f"参考文档：\n{context_text}\n\n问题：{question}\n\n回答："},
        ]
        return llm.chat(messages)