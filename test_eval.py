"""
评估脚本的测试：测试集结构、标准答案接地、命中判定与报告逻辑。

运行：pytest test_eval.py -v

设计约束：
- 不加载任何 embedding 模型（load_corpus 只读打开 chroma_data，不需要 embedding 函数）；
- 不调用 DeepSeek；
- 测试集的"标准答案"必须在真实语料中可唯一定位，否则评估结果不可信。
"""

import pathlib

import pytest

from eval_questions import LANG_COUNTS, QUESTIONS
from eval_retrieval import format_failures, format_table, is_hit, load_corpus


# ---------------------------------------------------------------------------
# 测试集结构
# ---------------------------------------------------------------------------


class TestQuestionSet:
    def test_has_40_questions_with_15_15_10_split(self):
        assert len(QUESTIONS) == 40
        counts = {}
        for q in QUESTIONS:
            counts[q["lang"]] = counts.get(q["lang"], 0) + 1
        assert counts == LANG_COUNTS
        assert counts == {"en": 15, "zh": 15, "mixed": 10}

    def test_each_question_has_required_fields(self):
        for q in QUESTIONS:
            assert q["question"].strip(), q
            assert q["expected_source"].strip(), q
            assert q["expected_fragment"].strip(), q
            assert q["lang"] in ("en", "zh", "mixed"), q

    def test_questions_are_distinct(self):
        questions = [q["question"] for q in QUESTIONS]
        assert len(questions) == len(set(questions))


# ---------------------------------------------------------------------------
# 英文改写（方案 C）：为中文/混合问题提供的独立检索改写
# ---------------------------------------------------------------------------


class TestRewrites:
    def test_all_zh_mixed_have_rewrite(self):
        for q in QUESTIONS:
            if q["lang"] in ("zh", "mixed"):
                assert q.get("rewritten_en", "").strip(), q

    def test_en_questions_have_no_rewrite(self):
        """英文问题直接检索，不需要改写。"""
        for q in QUESTIONS:
            if q["lang"] == "en":
                assert "rewritten_en" not in q, q

    def test_rewrite_is_pure_english(self):
        """改写必须是英文，不能残留中文。"""
        import re

        for q in QUESTIONS:
            if q["lang"] in ("zh", "mixed"):
                assert not re.search(r"[一-鿿]", q["rewritten_en"]), q

    def test_rewrite_does_not_leak_standard_answer(self):
        """改写不能复制 expected_fragment，也不能包含标准答案的原文。"""
        for q in QUESTIONS:
            if q["lang"] not in ("zh", "mixed"):
                continue
            rw = q["rewritten_en"]
            assert rw != q["expected_fragment"], q
            assert q["expected_fragment"] not in rw, q
            # 改写应与原问题不同（确实是重写，而不是原样照抄）
            assert rw != q["question"], q

    def test_rewrite_uses_plain_english_words(self):
        """改写里不应直接出现 fragment 的核心词（防止通过查询泄露答案）。"""
        # 抽样抽查：几个含答案关键词的改写，必须用同义词避开答案原文
        cases = {
            "实习期多长？": "16 WEEKS",
            "实习津贴是多少？": "RM1,000.00",
            "实习生向谁汇报工作？": "Khor Kai Dat",
            "实习生的 job title 是什么？": "AI Programmer",
            "公司全称是什么？": "AURAPLEX",
        }
        for question, answer_word in cases.items():
            q = next(x for x in QUESTIONS if x["question"] == question)
            assert answer_word not in q["rewritten_en"], q

    def test_retrieval_modules_do_not_import_deepseek_client(self):
        """
        检索评估脚本（eval_retrieval.py / eval_questions.py）不得依赖 DeepSeek，
        保证检索评估本身零 API 费用。eval_rewrite.py 是唯一允许调用 DeepSeek 的模块。
        """
        base = pathlib.Path(__file__).parent
        for name in ("eval_retrieval.py", "eval_questions.py"):
            src = (base / name).read_text(encoding="utf-8")
            assert "import ask" not in src
            assert "from ask" not in src
            assert "httpx" not in src


# ---------------------------------------------------------------------------
# 标准答案接地：fragment 必须能在真实语料中唯一定位到答案段落
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def corpus():
    return load_corpus()


class TestGrounding:

    def test_corpus_not_empty(self, corpus):
        assert len(corpus) >= 1
        assert {c["source"] for c in corpus}

    def test_expected_source_exists_in_corpus(self, corpus):
        sources = {c["source"] for c in corpus}
        for q in QUESTIONS:
            assert q["expected_source"] in sources, q

    def test_fragment_grounded_in_expected_source(self, corpus):
        """每个 fragment 至少出现在预期来源文件的一块 chunk 里。"""
        by_source = {}
        for c in corpus:
            by_source.setdefault(c["source"], []).append(c["text"])
        for q in QUESTIONS:
            texts = by_source[q["expected_source"]]
            assert any(q["expected_fragment"] in t for t in texts), q

    def test_fragment_unique_corpus_wide(self, corpus):
        """每个 fragment 在整个语料中只出现在一块里——答案段落无歧义。"""
        for q in QUESTIONS:
            n = sum(1 for c in corpus if q["expected_fragment"] in c["text"])
            assert n == 1, (q["expected_fragment"], n)


# ---------------------------------------------------------------------------
# 方案 D：DeepSeek 自动改写的静态检查（不调用 API）
# ---------------------------------------------------------------------------


class TestAutoRewrite:
    def test_rewrite_prompt_forbids_answering(self):
        """提示词必须禁止回答问题、禁止补充事实、只输出翻译。"""
        from eval_rewrite import SYSTEM_PROMPT

        p = SYSTEM_PROMPT.lower()
        assert "translate" in p
        assert "do not answer" in p
        assert "do not add" in p
        assert "output only" in p

    def test_validate_accepts_english(self):
        from eval_rewrite import validate_rewrite

        q = next(x for x in QUESTIONS if x["lang"] == "zh")
        assert (
            validate_rewrite(q["question"], "How long is the internship?")
            == "How long is the internship?"
        )

    def test_validate_rejects_cjk(self):
        from eval_rewrite import validate_rewrite

        q = next(x for x in QUESTIONS if x["lang"] == "zh")
        with pytest.raises(ValueError):
            validate_rewrite(q["question"], "实习期多长？")

    def test_validate_rejects_empty(self):
        from eval_rewrite import validate_rewrite

        q = next(x for x in QUESTIONS if x["lang"] == "zh")
        with pytest.raises(ValueError):
            validate_rewrite(q["question"], '""')

    def test_validate_rejects_fragment_leak(self):
        """改写里出现标准答案片段（如金额）必须被拒绝。"""
        from eval_rewrite import validate_rewrite

        q = next(x for x in QUESTIONS if x["question"] == "实习津贴是多少？")
        with pytest.raises(ValueError):
            validate_rewrite(q["question"], "The allowance is RM1,000.00 per month")

    def test_validate_strips_quotes(self):
        from eval_rewrite import validate_rewrite

        q = next(x for x in QUESTIONS if x["lang"] == "zh")
        assert validate_rewrite(q["question"], '"When does the internship start?"') == (
            "When does the internship start?"
        )

    def test_cache_roundtrip(self, tmp_path):
        from eval_rewrite import load_cache, save_cache

        p = tmp_path / "cache.json"
        save_cache({"a": "b"}, path=p)
        assert load_cache(path=p) == {"a": "b"}

    def test_cached_auto_rewrites_comply(self):
        """若已生成改写缓存，所有自动改写必须通过校验（纯英文、无答案泄漏）。"""
        from eval_rewrite import load_cache, validate_rewrite

        cache = load_cache()
        if not cache:
            pytest.skip("尚无改写缓存（自动改写未生成），跳过")
        for question, rw in cache.items():
            validate_rewrite(question, rw)


# ---------------------------------------------------------------------------
# 查询选择与人工/自动对比逻辑（纯函数）
# ---------------------------------------------------------------------------


class TestQueryPicking:
    def test_pick_query_english_never_rewritten(self):
        from eval_retrieval import REWRITE_AUTO, REWRITE_MANUAL, REWRITE_ORIGINAL, pick_query

        en_q = next(q for q in QUESTIONS if q["lang"] == "en")
        for mode in (REWRITE_ORIGINAL, REWRITE_MANUAL, REWRITE_AUTO):
            assert pick_query(en_q, mode, {}) == en_q["question"]

    def test_pick_query_manual_uses_rewritten_en(self):
        from eval_retrieval import REWRITE_MANUAL, pick_query

        q = next(x for x in QUESTIONS if x["lang"] == "zh")
        assert pick_query(q, REWRITE_MANUAL, {}) == q["rewritten_en"]

    def test_pick_query_auto_uses_cache_and_falls_back(self):
        from eval_retrieval import REWRITE_AUTO, pick_query

        q = next(x for x in QUESTIONS if x["lang"] == "zh")
        assert pick_query(q, REWRITE_AUTO, {q["question"]: "CUSTOM"}) == "CUSTOM"
        assert pick_query(q, REWRITE_AUTO, {}) == q["question"]


class TestManualVsAutoCompare:
    def test_rank_delta_logic(self):
        from eval_retrieval import _rank_delta

        assert _rank_delta(0, 0) == 0
        assert _rank_delta(0, -1) == -1   # 人工命中、自动未命中 → 变差
        assert _rank_delta(-1, 0) == 1    # 人工未命中、自动命中 → 变好
        assert _rank_delta(2, 1) == 1     # 自动更靠前 → 变好
        assert _rank_delta(1, 2) == -1
        assert _rank_delta(-1, -1) == 0

    def test_compare_manual_vs_auto(self):
        from eval_retrieval import compare_manual_vs_auto

        def perq(question, rank):
            return {"question": question, "lang": "zh", "rank": rank, "query_used": "q"}

        manual = {"per_q": [perq("a", 0), perq("b", -1), perq("c", 2), perq("d", 1)]}
        auto = {"per_q": [perq("a", -1), perq("b", 0), perq("c", 1), perq("d", 1)]}
        worse, better, same = compare_manual_vs_auto(manual, auto)
        assert [w["question"] for w in worse] == ["a"]
        assert [b["question"] for b in better] == ["b", "c"]
        assert same == 1

    def test_schemes_are_the_three_required(self):
        from eval_retrieval import SCHEMES

        assert [s[0] for s in SCHEMES] == [
            "方案A 原始+L6-v2",
            "方案C 人工改写+L6-v2",
            "方案D DeepSeek改写+L6-v2",
        ]


# ---------------------------------------------------------------------------
# 命中判定与报告逻辑（纯函数，不需要模型）
# ---------------------------------------------------------------------------


class TestHitLogic:
    def test_is_hit_returns_rank(self):
        texts = ["aaa", "bbb expected ccc", "ddd"]
        assert is_hit(texts, "expected") == 1

    def test_is_hit_top1(self):
        texts = ["the 16 WEEKS answer", "other"]
        assert is_hit(texts, "16 WEEKS") == 0

    def test_is_hit_miss_returns_minus_1(self):
        assert is_hit(["aaa", "bbb"], "zzz") == -1

    def test_is_hit_empty_results(self):
        assert is_hit([], "zzz") == -1

    def test_fragment_with_whitespace_insensitive_not_required(self):
        """判定要求与原文逐字符一致（这是有意的严格口径）。"""
        assert is_hit(["Monday – Friday 9am"], "Monday – Friday 9am") == 0
        assert is_hit(["Monday - Friday 9am"], "Monday – Friday 9am") == -1


def _fake_result(model: str) -> dict:
    return {
        "model": model,
        "use_rewrite": False,
        "scheme": f"方案X {model}",
        "corpus_chunks": 10,
        "first_load_s": 1.0,
        "model_init_s": 0.5,
        "index_s": 0.5,
        "avg_query_s": 0.01,
        "rss_before_mb": 1.0,
        "rss_after_index_mb": 2.0,
        "rss_delta_mb": 1.0,
        "top1": {"n": 15, "hits": 10, "rate": 0.6667},
        "top3": {"n": 15, "hits": 12, "rate": 0.8},
        "by_lang": {
            lang: {"n": 5, "top1": 3, "top3": 4, "avg_top1_dist": 0.6}
            for lang in ("en", "zh", "mixed")
        },
        "avg_top1_distance": 0.6,
        "failures_top1": [
            {
                "lang": "zh",
                "question": "q?",
                "rank": 2,
                "returned_sources": ["a.pdf"],
                "top1_preview": "some text",
            }
        ],
        "failures_top3": [],
    }


class TestReportFormatting:
    def test_format_table_covers_all_required_metrics(self):
        table = format_table([_fake_result("m1"), _fake_result("m2")])
        for keyword in (
            "Top-1 命中率",
            "Top-3 命中率",
            "英文",
            "中文",
            "混合",
            "distance",
            "首次加载时间",
            "平均查询时间",
            "内存占用",
        ):
            assert keyword in table

    def test_format_failures_lists_missed_questions(self):
        out = format_failures(_fake_result("m1"))
        assert "==" in out
        assert "q?" in out

    def test_format_failures_all_hits(self):
        r = _fake_result("m1")
        r["failures_top1"] = []
        out = format_failures(r)
        assert "Top-1 全部命中" in out
