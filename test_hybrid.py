"""
混合检索实验（eval_hybrid.py）的纯逻辑测试：分词、BM25 评分、
排名融合、空结果处理。

运行：pytest test_hybrid.py -v

不加载 embedding 模型、不调用 DeepSeek、不触碰正式数据库。
"""

import pytest

from eval_hybrid import (
    BM25Index,
    minmax_normalize,
    rrf_merge,
    tokenize,
    weighted_merge,
)

DOCS = [
    {"id": "d0", "text": "The intern reports to Director Mr Khor Kai Dat.",
     "source": "a.pdf", "page": 1},
    {"id": "d1", "text": "You are accepted as AI Programmer to develop in house AI Server.",
     "source": "a.pdf", "page": 1},
    {"id": "d2", "text": "Your allowance is RM1,000.00 per month, paid not later than the 7th day.",
     "source": "b.pdf", "page": 2},
    {"id": "d3", "text": "Confidential information must stay strictly confidential.",
     "source": "b.pdf", "page": 2},
]


# ---------------------------------------------------------------------------
# 分词
# ---------------------------------------------------------------------------


class TestTokenize:
    def test_lowercase_and_alnum_only(self):
        assert tokenize("AI Programmer Khor Kai Dat") == [
            "ai", "programmer", "khor", "kai", "dat",
        ]

    def test_numbers_and_money(self):
        assert tokenize("RM1,000.00 per month") == ["rm1", "000", "00", "per", "month"]

    def test_punctuation_removed(self):
        assert tokenize("What's the working hours? 9am-6pm!") == [
            "what", "s", "the", "working", "hours", "9am", "6pm",
        ]

    def test_empty_text(self):
        assert tokenize("") == []
        assert tokenize("   \t\n  ") == []

    def test_non_ascii_ignored(self):
        """中文 token 不进入英文检索词集合（查询均为英文改写）。"""
        assert tokenize("实习期是多久？") == []


# ---------------------------------------------------------------------------
# BM25 评分
# ---------------------------------------------------------------------------


class TestBM25:
    def test_docs_keep_source_and_page(self):
        idx = BM25Index(DOCS)
        hits = idx.results("intern reports to director")
        assert hits
        for h in hits:
            assert h["source"] in {"a.pdf", "b.pdf"}
            assert h["page"] in {1, 2}
        assert hits[0]["source"] == "a.pdf" and hits[0]["page"] == 1

    def test_matching_doc_outscores_unrelated(self):
        idx = BM25Index(DOCS)
        hits = idx.results("Khor Kai Dat")
        assert hits and hits[0]["id"] == "d0"
        hits = idx.results("AI Programmer AI Server")
        assert hits and hits[0]["id"] == "d1"
        hits = idx.results("allowance paid month")
        assert hits and hits[0]["id"] == "d2"

    def test_rare_term_has_higher_idf(self):
        idx = BM25Index(DOCS)
        assert idx.idf("khor") > idx.idf("the")  # 罕见词权重大
        assert idx.idf("dat") > 0

    def test_more_matches_score_higher(self):
        idx = BM25Index(DOCS)
        s_single = idx.score("intern", 0)
        s_double = idx.score("intern reports", 0)
        assert s_double > s_single  # 命中的词越多分越高

    def test_unmatched_query_scores_zero(self):
        idx = BM25Index(DOCS)
        assert idx.score("zzzqqq", 0) == 0.0

    def test_search_sorted_desc(self):
        idx = BM25Index(DOCS)
        hits = idx.search("the intern")
        scores = [s for _, s in hits]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# 空结果
# ---------------------------------------------------------------------------


class TestEmptyResults:
    def test_bm25_empty_index(self):
        idx = BM25Index([])
        assert idx.search("anything") == []
        assert idx.results("anything") == []
        assert idx.avgdl == 0.0

    def test_bm25_no_term_overlap(self):
        idx = BM25Index(DOCS)
        assert idx.search("completely unrelated zz") == []

    def test_bm25_empty_query(self):
        idx = BM25Index(DOCS)
        assert idx.search("") == []
        assert idx.search("。！？") == []  # 无英文 token

    def test_rrf_with_empty_rankings(self):
        assert rrf_merge([]) == []
        assert rrf_merge([[], []]) == []

    def test_weighted_merge_with_empty_lists(self):
        assert weighted_merge([], []) == []

    def test_rrf_ignores_empty_route(self):
        """一路为空时，融合结果等于另一路排名。"""
        assert rrf_merge([[], [2, 0, 1]]) == [2, 0, 1]
        assert rrf_merge([[1, 0], []]) == [1, 0]


# ---------------------------------------------------------------------------
# 排名融合
# ---------------------------------------------------------------------------


class TestRRF:
    def test_interleaves_two_rankings(self):
        merged = rrf_merge([[0, 1, 2], [2, 1, 0]])
        # 两路都排第 1 的 doc 得最高分，但排序应包含全部 doc
        assert set(merged) == {0, 1, 2}
        assert merged[0] in (0, 2)

    def test_agreement_pushes_to_top(self):
        # 两路都认为 d0 第一 → d0 融合后第一
        merged = rrf_merge([[0, 1, 2, 3], [0, 2, 1, 3]])
        assert merged[0] == 0

    def test_top_rank_but_solo_route_loses_to_agreement(self):
        # d1 在一路第 1、另一路第 2；d0 两路第 2/第 3 —— RRF 惩罚低名次
        merged = rrf_merge([[1, 0, 2, 3], [0, 1, 2, 3]])
        assert merged[0] == 1

    def test_duplicate_rankings_no_error(self):
        assert set(rrf_merge([[0, 1], [0, 1]])) == {0, 1}

    def test_custom_k(self):
        merged = rrf_merge([[0, 1, 2], [2, 1, 0]], k=1)
        assert set(merged) == {0, 1, 2}


class TestWeightedMerge:
    def test_minmax_normalize(self):
        assert minmax_normalize([0.0, 5.0, 10.0]) == [0.0, 0.5, 1.0]
        assert minmax_normalize([3.0, 3.0, 3.0]) == [0.0, 0.0, 0.0]
        assert minmax_normalize([]) == []

    def test_weighted_merge_prefers_good_in_both(self):
        # d0 向量距离最小且 BM25 分最高 → 融合第一
        merged = weighted_merge(
            [(0, 0.3), (1, 0.5), (2, 0.7)],   # 向量距离（越小越好）
            [(2, 9.0), (0, 8.0), (1, 1.0)],   # BM25 分数（越大越好）
        )
        assert merged[0] == 0

    def test_weighted_merge_returns_all_ids(self):
        merged = weighted_merge([(0, 0.3)], [(1, 2.0)])
        assert set(merged) == {0, 1}


# ---------------------------------------------------------------------------
# 失败类型映射完整性
# ---------------------------------------------------------------------------


class TestFailureTypeMapping:
    def test_every_failure_question_exists_in_eval_set(self):
        from eval_questions import EXTRA_QUESTIONS, QUESTIONS

        all_q = {q["question"] for q in QUESTIONS + EXTRA_QUESTIONS}
        from eval_hybrid import FAILURE_TYPES

        mapped = [q for qs in FAILURE_TYPES.values() for q in qs]
        assert len(mapped) == len(set(mapped))  # 无重复映射
        for q in mapped:
            assert q in all_q, f"映射了不存在的题目：{q}"

    def test_failure_types_cover_all_planned_categories(self):
        from eval_hybrid import FAILURE_TYPES

        assert set(FAILURE_TYPES) == {
            "汇报对象", "岗位/job title", "负责开发什么", "回传文件", "津贴/福利/专名",
        }
        total = sum(len(qs) for qs in FAILURE_TYPES.values())
        assert total >= 15  # 覆盖所有点名的失败类型题目
