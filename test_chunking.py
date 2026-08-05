"""
切块策略实验（eval_chunking.py）与失败类型补充问题（EXTRA_QUESTIONS）的测试。

运行：pytest test_chunking.py -v

设计约束（与 test_eval.py 一致）：
- 不加载任何 embedding 模型（load_corpus 只读打开 chroma_data，不需要 embedding 函数）；
- 不调用 DeepSeek；
- EXTRA_QUESTIONS 的标准答案 fragment 必须在真实语料中可唯一定位。
"""

import re

import pytest

from eval_chunking import (
    chunk_pages,
    chunk_pages_a,
    chunk_pages_b,
    chunk_pages_c,
    split_paragraphs,
)
from eval_questions import EXTRA_QUESTIONS, QUESTIONS
from eval_retrieval import load_corpus


# ---------------------------------------------------------------------------
# 三种切块策略的行为（纯函数，不加载模型）
# ---------------------------------------------------------------------------


SAMPLE_PAGES = [
    (
        1,
        "OWN PLACEMENT INFORMATION\n\nDate : 08/02/2026\n\n"
        "Student Name : WU ZHONGHENG\n\nThis letter is to certify that the student "
        "is a full-time student.\n\nA letter of confirmation would be required.",
    ),
    (2, "Page two paragraph only."),
]


class TestSplitParagraphs:
    def test_splits_on_blank_lines(self):
        paras = split_paragraphs("AAA\n\nBBB\n\n\nCCC")
        assert paras == ["AAA", "BBB", "CCC"]

    def test_collapses_inner_whitespace(self):
        paras = split_paragraphs("AAA  bb \n cc\n\nDDD")
        assert paras[0] == "AAA bb cc"

    def test_drops_empty_paragraphs(self):
        paras = split_paragraphs("AAA\n \n \n\nBBB")
        assert paras == ["AAA", "BBB"]


class TestChunkPagesA:
    def test_matches_current_chunker_behavior(self):
        """方案A 必须与线上 chunker.load_document 逐条一致（实验语料与线上同源）。"""
        from pathlib import Path

        from chunker import load_document, read_pdf

        path = Path(__file__).parent / "docs" / "university letter concerning d internship.pdf"
        pages = [(no, txt) for no, txt in read_pdf(path)]
        mine = chunk_pages_a(pages, source=path.name)
        theirs = load_document(path)
        assert [c.text for c in mine] == [c.text for c in theirs]
        assert [(c.source, c.page) for c in mine] == [(c.source, c.page) for c in theirs]

    def test_keeps_source_and_page(self):
        chunks = chunk_pages_a(SAMPLE_PAGES, source="x.pdf")
        assert all(c.source == "x.pdf" for c in chunks)
        assert {c.page for c in chunks} == {1, 2}


class TestChunkPagesB:
    def test_one_chunk_per_paragraph(self):
        chunks = chunk_pages_b(SAMPLE_PAGES, source="x.pdf")
        # 第 1 页 5 段 + 第 2 页 1 段（第一页最后两段都短，不触发超长切分）
        assert len(chunks) == 6
        assert all(c.source == "x.pdf" for c in chunks)
        assert chunks[0].page == 1 and chunks[-1].page == 2

    def test_long_paragraph_is_split_internally(self):
        """超长段落内部按断点切成多块，块内容仍来自原段落。"""
        long_para = "First sentence about work. Second sentence about hours. " * 20
        chunks = chunk_pages_b([(1, long_para)], size=200, source="x.pdf")
        assert len(chunks) >= 2
        flat = " ".join(long_para.split())
        for c in chunks:
            assert c.text in flat  # 每块都是原段落的一部分

    def test_no_overlap_between_sibling_paragraphs(self):
        """相邻段落各成一块，互不包含。"""
        chunks = chunk_pages_b(SAMPLE_PAGES, source="x.pdf")
        for i in range(len(chunks) - 1):
            assert chunks[i + 1].text not in chunks[i].text
            assert chunks[i].text not in chunks[i + 1].text


class TestChunkPagesC:
    def test_adjacent_paragraphs_share_overlap(self):
        """相邻块共享下一段内容（滑动窗口，步长 1 段）。"""
        pages = [(1, "PARA ONE.\n\nPARA TWO.\n\nPARA THREE.")]
        chunks = chunk_pages_c(pages, source="x.pdf")
        assert len(chunks) == 2  # PARA1+PARA2、PARA2+PARA3
        assert "PARA ONE" in chunks[0].text and "PARA TWO" in chunks[0].text
        assert "PARA TWO" in chunks[1].text and "PARA THREE" in chunks[1].text
        assert all(c.source == "x.pdf" for c in chunks)
        assert all(c.page == 1 for c in chunks)

    def test_single_paragraph_stays_one_chunk(self):
        chunks = chunk_pages_c([(1, "ONLY ONE PARAGRAPH.")], source="x.pdf")
        assert len(chunks) == 1
        assert chunks[0].text == "ONLY ONE PARAGRAPH."

    def test_merge_over_limit_falls_back_to_single(self):
        """合并后超 max_join 时退回单段，且最后一段不丢失。"""
        para1 = "X" * 400  # 一段不超过 size，不触发内部切分
        para2 = "Y" * 200
        pages = [(1, f"{para1}\n\n{para2}")]
        chunks = chunk_pages_c(pages, size=500, max_join=500, source="x.pdf")
        # para1+para2 合并后 601 > 500 → 各自单块（最后一段由回退分支补上）
        assert [c.text for c in chunks] == [para1, para2]

    def test_does_not_merge_across_pages(self):
        """相邻段落只同页合并，不跨页，保持 page 元数据准确。"""
        pages = [(1, "P1 ONE.\n\nP1 TWO."), (2, "P2 ONE.")]
        chunks = chunk_pages_c(pages, source="x.pdf")
        p1_texts = [c.text for c in chunks if c.page == 1]
        p2_texts = [c.text for c in chunks if c.page == 2]
        assert all("P2 ONE" not in t for t in p1_texts)
        assert all("P1" not in t for t in p2_texts)


class TestChunkPagesDispatch:
    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError):
            chunk_pages([(1, "x")], "z", source="x.pdf")

    def test_all_strategies_keep_source_and_page(self):
        for strategy in ("a", "b", "c"):
            chunks = chunk_pages(SAMPLE_PAGES, strategy, source="x.pdf")
            assert chunks, strategy
            assert all(c.source == "x.pdf" for c in chunks)
            assert {c.page for c in chunks} == {1, 2}

    def test_chunk_ids_unique(self):
        for strategy in ("a", "b", "c"):
            chunks = chunk_pages(SAMPLE_PAGES, strategy, source="x.pdf")
            ids = [c.chunk_id(i) for i, c in enumerate(chunks)]
            assert len(ids) == len(set(ids)), strategy


# ---------------------------------------------------------------------------
# EXTRA_QUESTIONS：结构与基线一致 + 接地 + 唯一性（不加载模型，只读语料）
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def corpus():
    return load_corpus()


class TestExtraQuestions:

    def test_has_10_questions_5_5_0(self):
        assert len(EXTRA_QUESTIONS) == 10
        counts = {}
        for q in EXTRA_QUESTIONS:
            counts[q["lang"]] = counts.get(q["lang"], 0) + 1
        assert counts == {"en": 5, "zh": 5}

    def test_each_question_has_required_fields(self):
        for q in EXTRA_QUESTIONS:
            assert q["question"].strip(), q
            assert q["expected_source"].strip(), q
            assert q["expected_fragment"].strip(), q
            assert q["lang"] in ("en", "zh"), q
            if q["lang"] == "zh":
                assert q.get("rewritten_en", "").strip(), q

    def test_questions_distinct_from_baseline(self):
        baseline = {q["question"] for q in QUESTIONS}
        extra = [q["question"] for q in EXTRA_QUESTIONS]
        assert len(extra) == len(set(extra))
        assert not (baseline & set(extra))

    def test_expected_source_exists_in_corpus(self, corpus):
        sources = {c["source"] for c in corpus}
        for q in EXTRA_QUESTIONS:
            assert q["expected_source"] in sources, q

    def test_fragment_grounded_in_expected_source(self, corpus):
        """每个 fragment 至少出现在预期来源文件的一块 chunk 里。"""
        by_source = {}
        for c in corpus:
            by_source.setdefault(c["source"], []).append(c["text"])
        for q in EXTRA_QUESTIONS:
            texts = by_source[q["expected_source"]]
            assert any(q["expected_fragment"] in t for t in texts), q

    def test_fragment_unique_corpus_wide(self, corpus):
        """每个 fragment 在整个语料中只出现一次——答案段落无歧义。"""
        for q in EXTRA_QUESTIONS:
            n = sum(1 for c in corpus if q["expected_fragment"] in c["text"])
            assert n == 1, (q["expected_fragment"], n)

    def test_rewrite_is_pure_english(self):
        """中文问题的英文改写必须是纯英文，不残留中文。"""
        for q in EXTRA_QUESTIONS:
            if q["lang"] == "zh":
                assert not re.search(r"[一-鿿]", q["rewritten_en"]), q

    def test_rewrite_does_not_leak_fragment(self):
        """改写不能复制标准答案片段。"""
        for q in EXTRA_QUESTIONS:
            if q["lang"] == "zh":
                assert q["expected_fragment"] not in q["rewritten_en"], q


# ---------------------------------------------------------------------------
# 报告格式化（纯函数，轻量验证）
# ---------------------------------------------------------------------------


def _fake_result(strategy: str) -> dict:
    return {
        "strategy": strategy,
        "scheme": f"方案{strategy}",
        "chunks": 12,
        "model_init_s": 2.0,
        "index_s": 1.0,
        "first_load_s": 3.5,
        "avg_query_s": 0.01,
        "top1": {"n": 50, "hits": 40, "rate": 0.8},
        "top3": {"n": 50, "hits": 45, "rate": 0.9},
        "by_lang": {
            lang: {"n": n, "top1": 3, "top3": 4, "avg_top1_dist": 0.6}
            for lang, n in (("en", 20), ("zh", 20), ("mixed", 10))
        },
        "avg_top1_distance": 0.55,
        "failures_top1": [
            {
                "lang": "zh",
                "question": "实习生的直属上司是谁？",
                "rank": 2,
                "returned_sources": ["a.pdf"],
                "top1_preview": "some text",
            }
        ],
        "failures_top3": [],
        "per_q": [],
    }


class TestChunkingReport:
    def test_format_table_covers_all_required_metrics(self):
        from eval_chunking import format_table

        table = format_table([_fake_result("a"), _fake_result("b"), _fake_result("c")])
        for keyword in (
            "chunk 数量",
            "Top-1 命中率",
            "Top-3 命中率",
            "英文",
            "中文",
            "混合",
            "distance",
            "建库时间",
            "平均查询时间",
        ):
            assert keyword in table

    def test_format_failures_lists_missed_questions(self):
        from eval_chunking import format_failures

        out = format_failures(_fake_result("a"))
        assert "==" in out
        assert "实习生的直属上司是谁？" in out
