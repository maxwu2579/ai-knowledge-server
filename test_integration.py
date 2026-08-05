"""
正式上传链路（方案C切块 → 入库 → 检索）的真实集成测试。

与 test_api.py 的区别：不 mock 切块、不 mock 向量库——
用真实的 load_document_paragraphs + 真实的 all-MiniLM-L6-v2 + 临时 ChromaDB，
验证"新上传的 PDF/TXT 走方案C、上传后能 /search 检索、source/page 正确、
重复上传行为正确"。

设计约束：
- 只写临时 ChromaDB（tempfile 自动清理），绝不碰正式 chroma_data / chroma_data_v2；
- embedding 模型 module-scope 只加载一次；
- 检索逻辑与 store.search 一致（cosine、阈值 0.85）。
"""

import shutil
import tempfile
from pathlib import Path

import chromadb
import pytest
from chromadb.utils import embedding_functions

from chunker import load_document_paragraphs

THRESHOLD = 0.85  # 与 store.search 一致（不修改正式配置）


@pytest.fixture(scope="module")
def real_chroma():
    """真实 embedding + 独立临时库（module 级只加载一次模型）。"""
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    db_dir = tempfile.mkdtemp(prefix="test_integration_")
    col = chromadb.PersistentClient(path=db_dir).get_or_create_collection(
        name="documents",
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )
    yield col
    shutil.rmtree(db_dir, ignore_errors=True)


def add_chunks(col, chunks) -> None:
    """与 store.add_chunks 同构的写入（临时库）。"""
    col.upsert(
        ids=[c.chunk_id(i) for i, c in enumerate(chunks)],
        documents=[c.text for c in chunks],
        metadatas=[{"source": c.source, "page": c.page} for c in chunks],
    )


def search(col, question: str, top_k: int = 5) -> list[dict]:
    """与 store.search 同构的检索（临时库，阈值 0.85）。"""
    if col.count() == 0:
        return []
    res = col.query(query_texts=[question], n_results=min(top_k, col.count()))
    hits = []
    for text, meta, dist in zip(
        res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        if dist <= THRESHOLD:
            hits.append(
                {"text": text, "source": meta["source"], "page": meta["page"], "distance": dist}
            )
    return hits


TXT_CONTENT = (
    "First paragraph about project alpha.\n\n"
    "Second paragraph about project beta.\n\n"
    "Third paragraph about project gamma."
)


@pytest.fixture
def txt_file(tmp_path) -> Path:
    p = tmp_path / "upload_note.txt"
    p.write_text(TXT_CONTENT, encoding="utf-8")
    return p


class TestPdfUploadUsesParagraphChunking:
    def test_pdf_chunks_carry_source_and_page(self):
        pdf = Path(__file__).parent / "docs" / "university letter concerning d internship.pdf"
        chunks = load_document_paragraphs(pdf)
        assert chunks
        for c in chunks:
            assert c.source == pdf.name
            assert c.page == 1
            assert c.text.strip()

    def test_pdf_chunk_count_matches_candidate_library(self):
        """候选库构建时的块数（6）就是正式上传的预期块数。"""
        pdf = Path(__file__).parent / "docs" / "university letter concerning d internship.pdf"
        assert len(load_document_paragraphs(pdf)) == 6

    def test_pdf_uploaded_chunks_retrievable_by_search(self, real_chroma):
        """真实 PDF 切块入库后，能像 /search 一样检索到内容。"""
        pdf = Path(__file__).parent / "docs" / "university letter concerning d internship.pdf"
        chunks = load_document_paragraphs(pdf)
        add_chunks(real_chroma, chunks)

        hits = search(real_chroma, "16 WEEKS OF COMPULSORY INTERNSHIP")
        assert hits
        assert any("16 WEEKS" in h["text"] for h in hits)
        assert hits[0]["source"] == pdf.name
        assert hits[0]["page"] == 1


class TestTxtUploadUsesParagraphChunking:
    def test_txt_paragraph_window_overlap(self, txt_file):
        """方案C特征：3 段 → 2 块，相邻块共享段落内容（旧策略只会切出 1 块）。"""
        chunks = load_document_paragraphs(txt_file)
        assert len(chunks) == 2
        texts = [c.text for c in chunks]
        assert texts[0] == (
            "First paragraph about project alpha. Second paragraph about project beta."
        )
        assert texts[1] == (
            "Second paragraph about project beta. Third paragraph about project gamma."
        )

    def test_txt_chunks_carry_source_and_page(self, txt_file):
        chunks = load_document_paragraphs(txt_file)
        for c in chunks:
            assert c.source == txt_file.name
            assert c.page == 1

    def test_txt_uploaded_chunks_retrievable_by_search(self, real_chroma, txt_file):
        """真实 TXT 切块入库后能检索（/search 语义）。"""
        chunks = load_document_paragraphs(txt_file)
        add_chunks(real_chroma, chunks)

        hits = search(real_chroma, "project beta")
        assert hits
        assert any(h["source"] == txt_file.name for h in hits)
        assert any("project beta" in h["text"] for h in hits)


class TestReuploadBehavior:
    def test_same_document_upload_twice_does_not_duplicate(self, real_chroma, txt_file):
        """同名文档重复上传（delete + add）：块数不叠加。"""
        chunks = load_document_paragraphs(txt_file)
        add_chunks(real_chroma, chunks)
        first_count = real_chroma.count()

        # 模拟 api 上传流程：先删旧数据再入库
        real_chroma.delete(where={"source": txt_file.name})
        add_chunks(real_chroma, chunks)

        assert real_chroma.count() == first_count
        hits = search(real_chroma, "project beta")
        assert hits  # 重新入库后仍能检索

    def test_reupload_leaves_no_other_source_touched(self, real_chroma, txt_file):
        """重复上传只影响同名文档，其他 source 的 chunk 不受影响。"""
        before = real_chroma.count()
        txt_chunks = load_document_paragraphs(txt_file)

        real_chroma.delete(where={"source": txt_file.name})
        after_delete = real_chroma.count()
        add_chunks(real_chroma, txt_chunks)

        # 删除只减少 TXT 的块；重加后库总量恢复原状
        assert after_delete == before - len(txt_chunks)
        assert real_chroma.count() == before
        # 其他 source（此前测试写入的 PDF）内容仍可检索
        pdf_hits = search(real_chroma, "16 WEEKS")
        assert pdf_hits
