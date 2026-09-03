"""P5 API 层单测：TestClient + mock LLM 验证 /health、/ingest/text、/ingest/file、/query。"""
import io
from typing import List

import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.api.service import RAGService
from src.llm_client.base import BaseLLMClient


class MockLLM(BaseLLMClient):
    """测试用 mock LLM：embed 按词表匹配，chat 返回固定答案。"""

    VOCAB = ["苹果", "香蕉", "汽车", "飞机", "水果", "交通工具", "维生素"]

    def __init__(self, fail: bool = False):
        self.fail = fail

    def chat(self, messages: List[dict], **kwargs) -> str:
        if self.fail:
            raise Exception("LLM 调用失败")
        return "苹果是富含维生素C的红色水果。"

    def embed(self, text: str, **kwargs) -> List[float]:
        if self.fail:
            raise Exception("LLM 调用失败")
        return [1.0 if w in text else 0.0 for w in self.VOCAB]


@pytest.fixture
def config():
    """测试配置：bm25 召回 + 关闭重排（避免依赖重排打分），LLM 无 key。"""
    return {
        "app": {"name": "rag_engine", "version": "0.1.0"},
        "ingestion": {"chunker": "fixed", "chunk_size": 100, "chunk_overlap": 10},
        "retrieval": {"retriever": "bm25", "top_k": 5, "rrf_k": 60},
        "rerank": {"enabled": False, "top_n": 3, "provider": "llm"},
        "llm": {"provider": "openai_compatible", "api_key": "", "model": "test", "temperature": 0.2, "max_tokens": 128},
        "server": {"host": "0.0.0.0", "port": 8000},
    }


@pytest.fixture
def client(config):
    """带 mock LLM 的测试客户端。"""
    llm = MockLLM()
    service = RAGService(
        config,
        llm_client=llm,
        embed_func=lambda texts: [llm.embed(t) for t in texts],
    )
    app = create_app(config=config, service=service)
    return TestClient(app)


class TestHealth:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["docs_indexed"] == 0
        assert data["retriever"] == "bm25"

    def test_root(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "endpoints" in resp.json()


class TestIngestText:
    def test_ingest_text(self, client):
        resp = client.post(
            "/ingest/text",
            json={"content": "苹果是红色水果，富含维生素C。香蕉是黄色热带水果。", "doc_id": "test-1"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["doc_id"] == "test-1"
        assert data["num_chunks"] >= 1
        assert len(data["chunks"]) == data["num_chunks"]

    def test_ingest_updates_health(self, client):
        client.post("/ingest/text", json={"content": "测试内容"})
        resp = client.get("/health")
        assert resp.json()["docs_indexed"] >= 1

    def test_ingest_empty_content_400(self, client):
        resp = client.post("/ingest/text", json={"content": ""})
        assert resp.status_code == 422  # pydantic 校验

    def test_ingest_whitespace_content_400(self, client):
        resp = client.post("/ingest/text", json={"content": "   "})
        assert resp.status_code == 400  # 业务层拒绝


class TestQuery:
    def test_query_end_to_end(self, client):
        # 先摄入
        client.post(
            "/ingest/text",
            json={"content": "苹果是红色水果，富含维生素C。香蕉是黄色热带水果，富含钾。", "doc_id": "d1"},
        )
        resp = client.post("/query", json={"question": "什么水果富含维生素？"})
        assert resp.status_code == 200
        data = resp.json()
        assert "苹果" in data["answer"]
        assert data["num_results"] >= 1
        assert data["results"][0]["doc_id"].startswith("d1")

    def test_query_empty_index_400(self, client):
        resp = client.post("/query", json={"question": "你好"})
        assert resp.status_code == 400
        assert "索引为空" in resp.json()["detail"]

    def test_query_missing_question_422(self, client):
        resp = client.post("/query", json={})
        assert resp.status_code == 422

    def test_query_top_k_validation(self, client):
        client.post("/ingest/text", json={"content": "测试文档内容"})
        resp = client.post("/query", json={"question": "测试", "top_k": 999})
        assert resp.status_code == 422  # ge=1, le=50


class TestIngestFile:
    def test_ingest_markdown_file(self, client):
        md_content = "# 标题\n\n苹果是红色水果，富含维生素C。\n\n## 子节\n\n香蕉是黄色热带水果。\n"
        resp = client.post(
            "/ingest/file",
            files={"file": ("kb.md", io.BytesIO(md_content.encode("utf-8")), "text/markdown")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["num_chunks"] >= 1
        assert data["doc_id"].startswith("kb")

    def test_ingest_unsupported_type_400(self, client):
        resp = client.post(
            "/ingest/file",
            files={"file": ("evil.exe", io.BytesIO(b"MZ"), "application/octet-stream")},
        )
        assert resp.status_code == 400
        assert "不支持" in resp.json()["detail"]

    def test_ingest_pdf_file(self, client, tmp_path):
        # 用测试辅助函数生成最小 PDF（辅助函数用 latin-1 编码，仅支持 ASCII 文本）
        from tests.test_parser import _make_pdf

        pdf_path = tmp_path / "doc.pdf"
        _make_pdf(pdf_path, "Apple is a red fruit rich in vitamin C.")
        pdf_bytes = pdf_path.read_bytes()
        resp = client.post(
            "/ingest/file",
            files={"file": ("doc.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
        )
        assert resp.status_code == 200
        assert resp.json()["num_chunks"] >= 1


class TestServiceErrors:
    def test_llm_failure_raises_500(self, config):
        llm = MockLLM(fail=True)
        service = RAGService(config, llm_client=llm)
        service.ingest_text("苹果是红色水果。")
        app = create_app(config=config, service=service)
        tc = TestClient(app)
        resp = tc.post("/query", json={"question": "水果"})
        assert resp.status_code == 500
        assert "查询失败" in resp.json()["detail"]
