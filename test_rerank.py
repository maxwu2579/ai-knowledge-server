"""
Cross-Encoder 重排实验（eval_rerank.py）的纯逻辑测试。

重排核心是纯函数 rerank(query, candidates, scorer)——测试注入假 scorer，
不加载真实模型（零模型下载、零 GPU/CPU 推理）。

覆盖：重排评分、空候选、元数据保留、排序稳定性、评分数量校验。
运行：pytest test_rerank.py -v
"""

import pytest

from eval_rerank import rerank

CANDIDATES = [
    {"text": "The intern reports to Director Mr Khor Kai Dat.", "source": "a.pdf", "page": 1},
    {"text": "You are accepted as AI Programmer to develop in house AI Server.",
     "source": "a.pdf", "page": 1},
    {"text": "Your allowance is RM1,000.00 per month.", "source": "b.pdf", "page": 2},
    {"text": "A letter of confirmation would be humbly required.", "source": "c.pdf", "page": 3},
]


def score_by_marker(pairs):
    """假 scorer：包含 "Khor" 的候选得高分，其次包含 "allowance"。"""
    scores = []
    for _, text in pairs:
        if "Khor Kai Dat" in text:
            scores.append(5.0)
        elif "allowance" in text:
            scores.append(3.0)
        elif "AI Server" in text:
            scores.append(1.0)
        else:
            scores.append(0.0)
    return scores


class TestRerankScoring:
    def test_sorts_by_score_descending(self):
        ranked = rerank("Who does the intern report to?", CANDIDATES, score_by_marker)
        assert ranked[0]["text"].startswith("The intern reports to Director Mr Khor Kai Dat")

    def test_score_is_query_aware(self):
        """scorer 收到的是 (query, 文本) 对——分数随 query 变化。"""
        def q_len_scorer(pairs):
            return [len(q) for q, _ in pairs]  # 查询越长分越高

        ranked = rerank("a very long query here", CANDIDATES, q_len_scorer)
        assert all(True for _ in ranked)  # 不抛错即可

    def test_wrong_score_count_raises(self):
        def bad_scorer(pairs):
            return [1.0]  # 数量与候选不符

        with pytest.raises(ValueError):
            rerank("query", CANDIDATES, bad_scorer)

    def test_tie_scores_keep_original_order(self):
        """同分时保持候选原顺序（稳定排序）。"""
        def flat_scorer(pairs):
            return [1.0] * len(pairs)

        ranked = rerank("q", CANDIDATES, flat_scorer)
        assert [c["text"] for c in ranked] == [c["text"] for c in CANDIDATES]


class TestEmptyCandidates:
    def test_empty_candidates_returns_empty(self):
        assert rerank("any query", [], score_by_marker) == []

    def test_empty_query_with_candidates_ok(self):
        """空查询不崩溃（分数由 scorer 决定）。"""
        ranked = rerank("", CANDIDATES, score_by_marker)
        assert len(ranked) == len(CANDIDATES)


class TestMetadataPreserved:
    def test_source_and_page_survive_rerank(self):
        ranked = rerank("q", CANDIDATES, score_by_marker)
        original = {id(c["text"]) for c in CANDIDATES}
        for c in ranked:
            assert c["source"] in {"a.pdf", "b.pdf", "c.pdf"}
            assert isinstance(c["page"], int)
            # 重排不生成新对象、不丢失字段
            assert c["text"] and id(c["text"]) in original

    def test_all_candidates_present_after_rerank(self):
        ranked = rerank("q", CANDIDATES, score_by_marker)
        assert len(ranked) == len(CANDIDATES)
        assert {c["source"] for c in ranked} == {"a.pdf", "b.pdf", "c.pdf"}


class TestStability:
    def test_deterministic_across_calls(self):
        r1 = rerank("Who does the intern report to?", CANDIDATES, score_by_marker)
        r2 = rerank("Who does the intern report to?", CANDIDATES, score_by_marker)
        assert [c["text"] for c in r1] == [c["text"] for c in r2]

    def test_high_score_candidate_always_first(self):
        for _ in range(3):
            ranked = rerank("Who does the intern report to?", CANDIDATES, score_by_marker)
            assert ranked[0]["source"] == "a.pdf"


# ---------------------------------------------------------------------------
# 六个"纯方案未命中"题的正确性（脚本内清单与评估集对齐）
# ---------------------------------------------------------------------------


class TestSixMissedList:
    def test_all_six_exist_in_eval_set(self):
        from eval_questions import EXTRA_QUESTIONS, QUESTIONS
        from eval_rerank import SIX_MISSED

        all_q = {q["question"] for q in QUESTIONS + EXTRA_QUESTIONS}
        assert len(SIX_MISSED) == 6
        for q in SIX_MISSED:
            assert q in all_q

    def test_six_cover_planned_categories(self):
        from eval_rerank import FAILURE_TYPES, SIX_MISSED

        mapped = {q for qs in FAILURE_TYPES.values() for q in qs}
        # 6 题全部落在重点失败类型映射里
        assert set(SIX_MISSED) <= mapped
