"""
chunker.py 方案 C 正式切块函数（段落 + 相邻段重叠）的边界测试。

运行：pytest test_chunker.py -v

覆盖：空文本 / 单段 / 多段 / 超长段落 / PDF 换页 / 不同 source /
      source·page 元数据 / 不产生重复或空白 chunk。
纯文本构造，不加载 embedding 模型、不调用 DeepSeek。
"""

import pytest

from chunker import (
    chunk_paragraphs,
    chunk_paragraphs_overlap,
    load_document_paragraphs,
    split_paragraphs,
)


# ---------------------------------------------------------------------------
# split_paragraphs：空文本与段落切分
# ---------------------------------------------------------------------------


class TestSplitParagraphs:
    def test_empty_text_returns_no_paragraphs(self):
        assert split_paragraphs("") == []
        assert split_paragraphs("   \n\n  \n ") == []
        assert split_paragraphs("\n\n\n") == []

    def test_single_paragraph(self):
        assert split_paragraphs("One paragraph only.") == ["One paragraph only."]

    def test_multiple_paragraphs(self):
        paras = split_paragraphs("AAA.\n\nBBB.\n\n\nCCC.")
        assert paras == ["AAA.", "BBB.", "CCC."]

    def test_inner_whitespace_collapsed(self):
        assert split_paragraphs("AAA   bb \n cc.\n\nDDD")[0] == "AAA bb cc."

    def test_no_blank_line_means_single_paragraph(self):
        # 段内换行（单个 \n）不切分——pypdf 提取文本的典型形态
        assert split_paragraphs("Line one\nLine two\nLine three") == [
            "Line one Line two Line three"
        ]


# ---------------------------------------------------------------------------
# chunk_paragraphs_overlap（方案 C）：空 / 单段 / 多段
# ---------------------------------------------------------------------------


class TestOverlapChunking:
    def test_empty_pages_returns_no_chunks(self):
        assert chunk_paragraphs_overlap([], source="x.pdf") == []
        assert chunk_paragraphs_overlap([(1, ""), (2, "   ")], source="x.pdf") == []

    def test_single_paragraph_single_chunk(self):
        chunks = chunk_paragraphs_overlap([(1, "Only one paragraph.")], source="x.pdf")
        assert len(chunks) == 1
        assert chunks[0].text == "Only one paragraph."

    def test_multiple_paragraphs_share_overlap(self):
        """滑动窗口：块0=段0+段1、块1=段1+段2——相邻块共享下一段内容。"""
        chunks = chunk_paragraphs_overlap(
            [(1, "PARA ONE.\n\nPARA TWO.\n\nPARA THREE.")], source="x.pdf"
        )
        assert len(chunks) == 2
        assert chunks[0].text == "PARA ONE. PARA TWO."
        assert chunks[1].text == "PARA TWO. PARA THREE."

    def test_paragraph_order_preserved_within_chunks(self):
        """窗口合并不会打乱段落顺序。"""
        chunks = chunk_paragraphs_overlap(
            [(1, "A.\n\nB.\n\nC.\n\nD.")], source="x.pdf"
        )
        texts = [c.text for c in chunks]
        assert texts == ["A. B.", "B. C.", "C. D."]


# ---------------------------------------------------------------------------
# 超长段落：安全拆分、不产生空 chunk、内容不丢失
# ---------------------------------------------------------------------------


class TestLongParagraph:
    LONG = "First sentence about work. Second sentence about hours. " * 20

    def test_overlap_split_long_paragraph_internally(self):
        chunks = chunk_paragraphs_overlap([(1, self.LONG)], size=200, source="x.pdf")
        assert len(chunks) >= 2
        flat = " ".join(self.LONG.split())
        assert all(c.text in flat for c in chunks)      # 每块都来自原段落
        assert all(c.text.strip() for c in chunks)      # 没有空白块

    def test_plain_chunking_splits_long_paragraph(self):
        chunks = chunk_paragraphs([(1, self.LONG)], size=200, source="x.pdf")
        assert len(chunks) >= 2
        assert all(c.text.strip() for c in chunks)

    def test_merge_over_limit_falls_back_to_single(self):
        """合并超 max_join 退回单段，且最后一段不丢失。"""
        para1 = "X" * 400
        para2 = "Y" * 200
        chunks = chunk_paragraphs_overlap(
            [(1, f"{para1}\n\n{para2}")], size=500, max_join=500, source="x.pdf"
        )
        assert [c.text for c in chunks] == [para1, para2]

    def test_merged_chunk_never_exceeds_max_join(self):
        para1 = "X" * 400
        para2 = "Y" * 200
        joined = f"{para1} {para2}"  # 601 字符
        chunks = chunk_paragraphs_overlap(
            [(1, f"{para1}\n\n{para2}")], size=500, max_join=601, source="x.pdf"
        )
        assert [c.text for c in chunks] == [joined]
        assert all(len(c.text) <= 601 for c in chunks)


# ---------------------------------------------------------------------------
# PDF 换页：不跨页合并，page 元数据准确
# ---------------------------------------------------------------------------


class TestPageBoundary:
    def test_no_merge_across_pages(self):
        pages = [(1, "P1 ONE.\n\nP1 TWO."), (2, "P2 ONE.")]
        chunks = chunk_paragraphs_overlap(pages, source="x.pdf")
        p1_texts = [c.text for c in chunks if c.page == 1]
        p2_texts = [c.text for c in chunks if c.page == 2]
        assert all("P2 ONE" not in t for t in p1_texts)   # 第 1 页末尾不吞第 2 页
        assert all("P1" not in t for t in p2_texts)

    def test_page_numbers_attached_correctly(self):
        pages = [(1, "AAA."), (2, "BBB."), (3, "CCC.")]
        chunks = chunk_paragraphs_overlap(pages, source="x.pdf")
        assert [c.page for c in chunks] == [1, 2, 3]

    def test_multi_page_overlap_within_each_page(self):
        pages = [(1, "A1.\n\nA2."), (2, "B1.\n\nB2.")]
        chunks = chunk_paragraphs_overlap(pages, source="x.pdf")
        assert [c.text for c in chunks] == ["A1. A2.", "B1. B2."]


# ---------------------------------------------------------------------------
# 不同 source：互不污染，source 元数据正确
# ---------------------------------------------------------------------------


class TestSourceMetadata:
    def test_different_sources_keep_own_names(self):
        c1 = chunk_paragraphs_overlap([(1, "AAA.")], source="a.pdf")
        c2 = chunk_paragraphs_overlap([(1, "BBB.")], source="b.pdf")
        assert all(c.source == "a.pdf" for c in c1)
        assert all(c.source == "b.pdf" for c in c2)

    def test_every_chunk_has_source_and_page(self):
        chunks = chunk_paragraphs_overlap(
            [(1, "A.\n\nB."), (2, "C.")], source="handbook.pdf"
        )
        for c in chunks:
            assert c.source == "handbook.pdf"
            assert isinstance(c.page, int) and c.page >= 1
            assert c.text.strip()

    def test_chunk_ids_unique_across_sources(self):
        """不同 source 用同一 id 前缀方案互不冲突。"""
        from chunker import Chunk

        all_chunks = [
            *chunk_paragraphs_overlap([(1, "A.")], source="a.pdf"),
            *chunk_paragraphs_overlap([(1, "B.")], source="b.pdf"),
        ]
        ids = [c.chunk_id(i) for i, c in enumerate(all_chunks)]
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# 不产生重复或空白 chunk
# ---------------------------------------------------------------------------


class TestNoDuplicatesNoBlanks:
    def test_no_blank_chunks_from_mixed_whitespace_pages(self):
        pages = [(1, "  \n\nAAA.\n\n \n\nBBB.\n\n  ")]
        chunks = chunk_paragraphs_overlap(pages, source="x.pdf")
        assert chunks and all(c.text.strip() for c in chunks)

    def test_consecutive_duplicate_paragraphs_merged(self):
        """三连相同段（"X X X" 模式）不会产出重复 chunk。"""
        pages = [(1, "X\n\nX\n\nX")]
        chunks = chunk_paragraphs_overlap(pages, source="x.pdf")
        texts = [c.text for c in chunks]
        assert len(texts) == len(set(texts))   # 无重复 chunk
        assert "X X X" in texts[0]

    def test_two_duplicate_paragraphs_no_duplicate_chunks(self):
        pages = [(1, "X\n\nX")]
        chunks = chunk_paragraphs_overlap(pages, source="x.pdf")
        texts = [c.text for c in chunks]
        assert len(texts) == len(set(texts))

    def test_no_chunk_equals_another_in_normal_document(self):
        """正常多段文档（无重复段）的所有 chunk 互不相同。"""
        pages = [(1, "A.\n\nB.\n\nC.\n\nD.\n\nE.")]
        chunks = chunk_paragraphs_overlap(pages, source="x.pdf")
        texts = [c.text for c in chunks]
        assert len(texts) == len(set(texts))


# ---------------------------------------------------------------------------
# load_document_paragraphs（正式入口）：真实 PDF 走通 + 与旧入口共存
# ---------------------------------------------------------------------------


class TestLoadDocumentParagraphs:
    def test_real_pdf_keeps_source_and_page(self):
        from pathlib import Path

        path = Path(__file__).parent / "docs" / "university letter concerning d internship.pdf"
        chunks = load_document_paragraphs(path)
        assert chunks
        for c in chunks:
            assert c.source == path.name
            assert c.page == 1
            assert c.text.strip()

    def test_old_and_new_entries_coexist(self):
        """旧入口 load_document 与方案 C 入口都可用（兼容要求）。"""
        from pathlib import Path

        from chunker import load_document

        path = Path(__file__).parent / "docs" / "university letter concerning d internship.pdf"
        old = load_document(path)
        new = load_document_paragraphs(path)
        assert old and new
        assert all(c.source == path.name for c in old)
        assert all(c.source == path.name for c in new)

    def test_new_chunking_never_duplicates_within_document(self):
        from pathlib import Path

        path = Path(__file__).parent / "docs" / "university letter concerning d internship.pdf"
        chunks = load_document_paragraphs(path)
        texts = [c.text for c in chunks]
        assert len(texts) == len(set(texts))
