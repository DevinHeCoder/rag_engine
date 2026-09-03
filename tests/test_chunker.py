"""P2 分块层单测：fixed / hierarchy / semantic 三种分块器 + 注册表 + dispatcher。"""
from typing import List

import pytest

from src.chunker import chunker_registry, split_documents
from src.chunker.fixed_chunker import FixedChunker
from src.chunker.hierarchy_chunker import HierarchyChunker
from src.chunker.semantic_chunker import SemanticChunker
from src.common.exceptions import ChunkingError
from src.document_parser.base import Document


def _doc(content: str, doc_id: str = "doc", **meta) -> Document:
    return Document(doc_id=doc_id, content=content, metadata=dict(meta))


# ---------------- 注册表测试 ----------------
class TestRegistry:
    def test_all_chunkers_registered(self):
        assert chunker_registry.has("fixed")
        assert chunker_registry.has("hierarchy")
        assert chunker_registry.has("semantic")
        assert set(chunker_registry.names) == {"fixed", "hierarchy", "semantic"}

    def test_default_is_fixed(self):
        assert isinstance(chunker_registry.create(), FixedChunker)


# ---------------- FixedChunker ----------------
class TestFixedChunker:
    def test_short_text_single_chunk(self):
        docs = [_doc("短文本，不需要切分。")]
        result = FixedChunker(chunk_size=512).split(docs)
        assert len(result) == 1
        assert result[0].content == "短文本，不需要切分。"

    def test_long_text_splits(self):
        text = "句子一。" * 100  # ~400 chars
        docs = [_doc(text)]
        result = FixedChunker(chunk_size=100, chunk_overlap=10).split(docs)
        assert len(result) > 1
        # 所有块拼起来应覆盖原文（允许重叠）
        assert all(len(c.content) <= 100 + 10 for c in result)

    def test_overlap_between_chunks(self):
        text = "a" * 200
        docs = [_doc(text)]
        result = FixedChunker(chunk_size=80, chunk_overlap=20).split(docs)
        assert len(result) >= 2
        # 相邻块有重叠：前一块末尾 == 后一块开头
        assert result[0].content[-20:] == result[1].content[:20]

    def test_cut_at_separator(self):
        # 构造在 chunk_size 附近有句号的文本
        text = "第一句。" * 10 + "第二句。" * 10
        docs = [_doc(text)]
        result = FixedChunker(chunk_size=30, chunk_overlap=5).split(docs)
        # 切点应落在句号后（块以句号结尾或接近句号）
        assert any(c.content.rstrip().endswith("。") for c in result[:-1])

    def test_metadata_inherited_and_augmented(self):
        docs = [_doc("测试内容。", doc_id="src1", source="file.md", file_type="md")]
        result = FixedChunker(chunk_size=512).split(docs)
        assert len(result) == 1
        m = result[0].metadata
        assert m["source"] == "file.md"
        assert m["file_type"] == "md"
        assert m["chunk_index"] == 0
        assert m["chunk_total"] == 1
        assert m["parent_doc_id"] == "src1"
        assert m["chunker"] == "fixed"
        assert result[0].doc_id == "src1::chunk-000"

    def test_empty_document_returns_empty(self):
        docs = [_doc("")]
        assert FixedChunker().split(docs) == []

    def test_invalid_params_raise(self):
        with pytest.raises(ChunkingError):
            FixedChunker(chunk_size=0)
        with pytest.raises(ChunkingError):
            FixedChunker(chunk_size=100, chunk_overlap=100)

    def test_no_infinite_loop(self):
        # 极端：无任何分隔符的长文本
        text = "a" * 1000
        docs = [_doc(text)]
        result = FixedChunker(chunk_size=100, chunk_overlap=10).split(docs)
        assert len(result) > 1
        # 总覆盖（允许重叠，不要求精确等于）
        assert sum(len(c.content) for c in result) >= len(text)

    def test_no_trailing_shards(self):
        # 末尾不应产生越来越小的碎片块（回归测试）
        text = "句子一。" * 30  # ~120 chars
        docs = [_doc(text)]
        result = FixedChunker(chunk_size=50, chunk_overlap=10).split(docs)
        # 最后一块不应小于 overlap 的合理比例（至少 5 字符）
        assert len(result[-1].content) >= 5
        # 所有块都应非空
        assert all(len(c.content) > 0 for c in result)


# ---------------- HierarchyChunker ----------------
class TestHierarchyChunker:
    def test_split_by_h2(self):
        text = "# 大标题\n\n前言。\n\n## 第一节\n\n内容一。\n\n## 第二节\n\n内容二。"
        docs = [_doc(text)]
        # max_chunk_size 设小，强制按标题切分
        result = HierarchyChunker(max_chunk_size=20).split(docs)
        # 前言 + 第一节 + 第二节
        assert len(result) == 3
        assert "前言" in result[0].content
        assert "第一节" in result[1].content
        assert "第二节" in result[2].content

    def test_large_section_recurses_to_h3(self):
        # 第一节超长（含 ### 子标题），应递归到 ### 切分
        text = (
            "# 大标题\n\n"
            "## 第一节\n\n"
            "### 子节A\n\n" + "内容A。" * 50 + "\n\n"
            "### 子节B\n\n" + "内容B。" * 50 + "\n\n"
            "## 第二节\n\n短内容。"
        )
        docs = [_doc(text)]
        result = HierarchyChunker(max_chunk_size=100).split(docs)
        # 第一节被拆成子节A、子节B，第二节独立
        assert any("子节A" in c.content for c in result)
        assert any("子节B" in c.content for c in result)
        assert any("第二节" in c.content for c in result)

    def test_no_headings_fallback_fixed(self):
        text = "这是一段没有任何标题的长文本。" * 50
        docs = [_doc(text)]
        result = HierarchyChunker(max_chunk_size=50, chunk_overlap=5).split(docs)
        assert len(result) > 1
        assert all(c.metadata["chunker"] == "hierarchy" for c in result)

    def test_short_text_single_chunk(self):
        text = "# 标题\n\n短内容。"
        docs = [_doc(text)]
        result = HierarchyChunker(max_chunk_size=512).split(docs)
        assert len(result) == 1

    def test_metadata(self):
        docs = [_doc("# H1\n\n内容。", doc_id="hdoc", source="a.md")]
        result = HierarchyChunker().split(docs)
        assert result[0].metadata["parent_doc_id"] == "hdoc"
        assert result[0].metadata["chunker"] == "hierarchy"
        assert result[0].doc_id == "hdoc::chunk-000"


# ---------------- SemanticChunker ----------------
class TestSemanticChunker:
    def test_no_embed_func_raises(self):
        with pytest.raises(ChunkingError):
            SemanticChunker().split([_doc("测试。")])

    def test_semantic_split_at_boundary(self):
        # 假嵌入：含"猫"的句子向量偏 [1,0]，含"狗"的偏 [0,1]，其余 [0.5,0.5]
        def fake_embed(sentences: List[str]) -> List[List[float]]:
            vecs = []
            for s in sentences:
                if "猫" in s:
                    vecs.append([1.0, 0.0])
                elif "狗" in s:
                    vecs.append([0.0, 1.0])
                else:
                    vecs.append([0.5, 0.5])
            return vecs

        text = "我喜欢养猫。猫很可爱。猫会抓老鼠。我也喜欢养狗。狗很忠诚。狗会看家。"
        docs = [_doc(text)]
        result = SemanticChunker(
            embed_func=fake_embed,
            chunk_size=512,
            breakpoint_percentile=50,
        ).split(docs)
        # 应在"猫"话题和"狗"话题之间断开，至少 2 块
        assert len(result) >= 2
        cat_chunk = next(c for c in result if "猫" in c.content)
        dog_chunk = next(c for c in result if "狗" in c.content)
        assert cat_chunk is not None
        assert dog_chunk is not None

    def test_single_sentence_single_chunk(self):
        def fake_embed(sentences):
            return [[0.1, 0.2] for _ in sentences]
        docs = [_doc("只有一句话。")]
        result = SemanticChunker(embed_func=fake_embed).split(docs)
        assert len(result) == 1

    def test_empty_document(self):
        def fake_embed(sentences):
            return []
        assert SemanticChunker(embed_func=fake_embed).split([_doc("")]) == []

    def test_embed_count_mismatch_raises(self):
        def bad_embed(sentences):
            return [[0.1, 0.2]]  # 只返回 1 个，但句子有多个
        docs = [_doc("第一句。第二句。第三句。")]
        with pytest.raises(ChunkingError):
            SemanticChunker(embed_func=bad_embed).split(docs)

    def test_metadata(self):
        def fake_embed(sentences):
            return [[0.1, 0.2] for _ in sentences]
        docs = [_doc("测试语义分块。元数据应正确。", doc_id="sdoc")]
        result = SemanticChunker(embed_func=fake_embed).split(docs)
        assert result[0].metadata["chunker"] == "semantic"
        assert result[0].metadata["parent_doc_id"] == "sdoc"


# ---------------- dispatcher: split_documents ----------------
class TestSplitDocuments:
    def test_default_fixed(self):
        docs = [_doc("短文本。")]
        result = split_documents(docs)
        assert len(result) == 1
        assert result[0].metadata["chunker"] == "fixed"

    def test_specified_hierarchy(self):
        docs = [_doc("# H1\n\n内容。")]
        result = split_documents(docs, "hierarchy")
        assert result[0].metadata["chunker"] == "hierarchy"

    def test_kwargs_passed_to_constructor(self):
        text = "a" * 500
        docs = [_doc(text)]
        result = split_documents(docs, "fixed", chunk_size=100, chunk_overlap=10)
        assert len(result) > 1
        assert all(len(c.content) <= 110 for c in result)

    def test_unknown_name_raises(self):
        with pytest.raises(Exception):
            split_documents([_doc("x")], "nonexistent")
