"""
Cross-Encoder 重排离线实验（基线 vs 向量召回+重排），零 DeepSeek 费用。

语料：chroma_data_v2 的 15 个 chunk（只读打开，不修改正式库）。
查询：50 题（40 基线 + 10 失败补充），中文/混合用人工英文改写
      （eval_questions.rewritten_en），英文用原问题。

方案：
    A 纯向量检索（基线，与 /search 同构，全量排名）
    B 向量召回 Top-5 → cross-encoder/ms-marco-MiniLM-L-6-v2 重排
    C 向量召回 Top-10 → 同一模型重排

模型只用于推理，不训练、不微调；模型缓存位于用户目录
（~/.cache/huggingface），不在项目内、不进入 Git。

指标：整体 Top-1 / Top-3、按语言、重点失败类型、回归案例（B/C 相对 A 变差）、
      模型加载时间、每题额外延迟（重排打分）、内存占用（Windows Working Set）。

重点检查：三个纯检索方案 Top-3 都未命中的 6 题——若正确 chunk 不在
向量 Top-5 / Top-10 召回内，重排无法挽救（明确输出覆盖情况）。

设计约束：不改 store.py / api.py / ask.py / .env / chroma_data_v2 /
          embedding / 0.85 阈值；不调用 DeepSeek。

用法：
    py eval_rerank.py
"""

import sys
import time
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

from eval_questions import EXTRA_QUESTIONS, QUESTIONS
from eval_retrieval import process_working_set_mb

DB_V2 = Path(__file__).parent / "chroma_data_v2"
COLLECTION_NAME = "documents"
EMBED_MODEL = "all-MiniLM-L6-v2"
CE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

LANG_LABEL = {"en": "英文", "zh": "中文", "mixed": "混合"}

# 重点失败类型 → 题目映射（与 eval_hybrid 一致）
FAILURE_TYPES = {
    "汇报对象": [
        "Who does the intern report to?",
        "实习生的直属上司是谁？",
        "实习生向谁汇报工作？",
    ],
    "岗位/job title": [
        "What position was the intern hired for?",
        "实习生担任什么职位？",
        "实习生的 job title 是什么？",
    ],
    "负责开发什么": [
        "What is the intern responsible for developing?",
        "实习生负责开发什么系统？",
        "实习生要 develop 什么？",
    ],
    "回传文件": [
        "What letter does the university need from the company?",
        "What does the university ask the company to send back?",
        "公司需要回传什么文件？",
        "实习结束后实习生要归还什么？",
    ],
    "津贴/福利/专名": [
        "What allowance does the company provide?",
        "公司每个月发多少津贴？",
        "monthly allowance",
        "实习津贴是多少？",
        "实习津贴是 RM 多少？",
        "Which company is offering the internship?",
        "公司全称是什么？",
        "公司叫什么名字？",
        "student name",
        "实习生叫什么名字？",
        "WU ZHONGHENG 的 Student ID 是多少？",
        "Which university does the student attend?",
        "学生来自哪所大学？",
    ],
}

# 三个纯检索方案 Top-3 都未命中的 6 题（切块与混合实验反复确认）
SIX_MISSED = [
    "Who does the intern report to?",
    "实习生的直属上司是谁？",
    "What position was the intern hired for?",
    "实习生担任什么职位？",
    "实习生的 job title 是什么？",
    "实习生要 develop 什么？",
]


# ---------------------------------------------------------------------------
# 重排（纯函数，scorer 可注入，供测试复用）
# ---------------------------------------------------------------------------


def rerank(query: str, candidates: list[dict], scorer) -> list[dict]:
    """用 scorer 对候选打分并按分数降序重排，保留 source / page 元数据。

    candidates: [{"text": str, "source": str, "page": int}, ...]
    scorer:     callable(list[tuple[str, str]]) -> list[float]，返回 (query, 文本) 对的相关度分数
    """
    if not candidates:
        return []
    pairs = [(query, c["text"]) for c in candidates]
    scores = scorer(pairs)
    if len(scores) != len(candidates):
        raise ValueError(f"scorer 返回 {len(scores)} 个分数，候选有 {len(candidates)} 个")
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [c for c, _ in ranked]


# ---------------------------------------------------------------------------
# 评估
# ---------------------------------------------------------------------------


def pick_query(q: dict) -> str:
    return q.get("rewritten_en") or q["question"]


def bench_all(threshold: float | None = None) -> dict:
    # ---- 语料与向量库（只读） ---------------------------------------------
    col = chromadb.PersistentClient(path=str(DB_V2)).get_or_create_collection(
        COLLECTION_NAME
    )
    got = col.get(include=["documents", "metadatas"])
    corpus = [
        {"id": cid, "text": doc, "source": meta["source"], "page": meta["page"]}
        for cid, doc, meta in zip(got["ids"], got["documents"], got["metadatas"])
    ]
    idx_of = {c["id"]: i for i, c in enumerate(corpus)}
    # 正确答案所在的 chunk 索引
    frag_idx = {
        q["question"]: next(i for i, c in enumerate(corpus)
                            if q["expected_fragment"] in c["text"])
        for q in QUESTIONS + EXTRA_QUESTIONS
    }

    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )
    vec_col = chromadb.PersistentClient(path=str(DB_V2)).get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )
    vec_col.query(query_texts=[pick_query(QUESTIONS[0])], n_results=3)  # 预热

    # ---- 加载 Cross-Encoder（只推理），计时 + 内存 ---------------------------
    from sentence_transformers import CrossEncoder

    t_ce0 = time.perf_counter()
    rss_before = process_working_set_mb()
    ce = CrossEncoder(CE_MODEL)
    t_ce_load = time.perf_counter() - t_ce0
    rss_after = process_working_set_mb()

    def ce_scorer(pairs):
        return list(ce.predict(pairs, show_progress_bar=False))

    questions = QUESTIONS + EXTRA_QUESTIONS
    t_query = 0.0
    t_rerank = 0.0
    per_q = []
    for q in questions:
        query_text = pick_query(q)

        t0 = time.perf_counter()
        # 正式口径：向量召回 Top-10（与 store.search 一致）；离线口径：全量 15
        n_results = 10 if threshold is not None else len(corpus)
        res = vec_col.query(query_texts=[query_text], n_results=n_results)
        t_query += time.perf_counter() - t0
        vector_ids = [idx_of[i] for i in res["ids"][0]]
        dists = res["distances"][0]

        cands = [
            {"text": corpus[i]["text"], "source": corpus[i]["source"],
             "page": corpus[i]["page"]}
            for i in vector_ids
        ]
        if threshold is not None:
            # 正式语义：距离 > 阈值视为无关，不进入重排（与 store.search 一致）
            cands = [c for c, d in zip(cands, dists) if d <= threshold]

        # 方案 B：向量 Top-5 → CE 重排
        t0 = time.perf_counter()
        top5 = rerank(query_text, cands[:5], ce_scorer)
        t_rerank += time.perf_counter() - t0
        # 方案 C：向量 Top-10 → CE 重排
        t0 = time.perf_counter()
        top10 = rerank(query_text, cands[:10], ce_scorer)
        t_rerank += time.perf_counter() - t0

        def rank_of(cands_ranked: list[dict], fragment: str) -> int:
            for r, c in enumerate(cands_ranked):
                if fragment in c["text"]:
                    return r
            return -1

        correct_idx = frag_idx[q["question"]]
        vector_rank = vector_ids.index(correct_idx) if correct_idx in vector_ids else -1
        per_q.append({
            "question": q["question"],
            "lang": q["lang"],
            "rank_a": rank_of(cands, q["expected_fragment"]),
            "rank_b": rank_of(top5, q["expected_fragment"]),
            "rank_c": rank_of(top10, q["expected_fragment"]),
            "vector_rank": vector_rank,  # -1 = 不在召回内
            "in_top5": 0 <= vector_rank < 5,
            "in_top10": 0 <= vector_rank < 10,
            "cands_after_threshold": len(cands),
            "correct_in_cands": (
                correct_idx in vector_ids[:len(cands)] if threshold is not None else True
            ),
        })
    n = len(per_q)

    def summarize(rank_key: str) -> dict:
        top1 = sum(1 for r in per_q if r[rank_key] == 0)
        top3 = sum(1 for r in per_q if 0 <= r[rank_key] < 3)
        by_lang, by_type = {}, {}
        for lang in ("en", "zh", "mixed"):
            sub = [r for r in per_q if r["lang"] == lang]
            by_lang[lang] = {
                "n": len(sub),
                "top1": sum(1 for r in sub if r[rank_key] == 0),
                "top3": sum(1 for r in sub if 0 <= r[rank_key] < 3),
            }
        for ftype, fqs in FAILURE_TYPES.items():
            sub = [r for r in per_q if r["question"] in fqs]
            by_type[ftype] = {
                "n": len(sub),
                "top1": sum(1 for r in sub if r[rank_key] == 0),
                "top3": sum(1 for r in sub if 0 <= r[rank_key] < 3),
            }
        return {
            "n": n, "top1": top1, "top3": top3,
            "rate1": round(top1 / n, 4), "rate3": round(top3 / n, 4),
            "by_lang": by_lang, "by_type": by_type,
        }

    six = [
        {**r, "correct_rank": r["vector_rank"]}
        for r in per_q if r["question"] in SIX_MISSED
    ]

    return {
        "chunks": len(corpus),
        "threshold": threshold,
        "ce_model": CE_MODEL,
        "ce_load_s": round(t_ce_load, 2),
        "rss_before_mb": round(rss_before, 1),
        "rss_after_mb": round(rss_after, 1),
        "rss_delta_mb": round(rss_after - rss_before, 1),
        "avg_query_s": round(t_query / n, 4),
        "avg_rerank_s": round(t_rerank / (2 * n), 4),  # 每题两次重排（B 和 C）
        "threshold_dropped": sum(1 for r in per_q if r["cands_after_threshold"] == 0),
        "threshold_correct_dropped": sum(
            1 for r in per_q
            if threshold is not None and not r["correct_in_cands"]
        ),
        "summary_a": summarize("rank_a"),
        "summary_b": summarize("rank_b"),
        "summary_c": summarize("rank_c"),
        "per_q": per_q,
        "six_missed": six,
    }


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------


def format_table(r: dict) -> str:
    col_w = 22
    labels = ("A 纯向量", "B Top5+CE", "C Top10+CE")
    headers = "".join(label.ljust(col_w) for label in labels)
    s = [r["summary_a"], r["summary_b"], r["summary_c"]]
    lines = [
        f"语料：{r['chunks']} 块（chroma_data_v2）；题数：{r['summary_a']['n']}；"
        f"重排模型：{r['ce_model']}",
        "",
        f"{'指标':<26}{headers}",
        "-" * (26 + col_w * 3),
    ]

    def row(metric: str, values: list[str]) -> None:
        lines.append(metric.ljust(26) + "".join(v.ljust(col_w) for v in values))

    row("Top-1 命中率", [f"{x['top1']}/{x['n']} ({x['rate1']:.0%})" for x in s])
    row("Top-3 命中率", [f"{x['top3']}/{x['n']} ({x['rate3']:.0%})" for x in s])
    for lang in ("en", "zh", "mixed"):
        row(f"  {LANG_LABEL[lang]} Top-1 / Top-3",
            [f"{x['by_lang'][lang]['top1']}/{x['by_lang'][lang]['n']} / "
             f"{x['by_lang'][lang]['top3']}/{x['by_lang'][lang]['n']}" for x in s])
    for ftype in FAILURE_TYPES:
        row(f"  {ftype} Top-1 / Top-3",
            [f"{x['by_type'][ftype]['top1']}/{x['by_type'][ftype]['n']} / "
             f"{x['by_type'][ftype]['top3']}/{x['by_type'][ftype]['n']}" for x in s])
    row("模型加载（秒）", [str(r["ce_load_s"])] * 3)
    row("内存增量（MB）", [str(r["rss_delta_mb"])] * 3)
    row("向量查询（秒/题）", [str(r["avg_query_s"])] * 3)
    row("重排额外延迟（秒/题）", ["-", str(r["avg_rerank_s"]), str(r["avg_rerank_s"])])
    return "\n".join(lines)


def format_six_missed(r: dict) -> str:
    lines = ["== 三纯方案 Top-3 都未命中的 6 题：正确 chunk 是否进入向量召回池 =="]
    for item in r["six_missed"]:
        verdict = (
            f"向量排名 {item['correct_rank']}（Top-5 {'✓' if item['in_top5'] else '✗'} / "
            f"Top-10 {'✓' if item['in_top10'] else '✗'}）"
        )
        note = "" if item["in_top5"] else (
            "" if item["in_top10"] else " → 重排无法挽救（不在 Top-10 召回内）"
        )
        lines.append(f"  [{LANG_LABEL[item['lang']]}] {item['question']}：{verdict}{note}")
    return "\n".join(lines)


def format_regressions(r: dict) -> str:
    lines = ["== 回归案例（B / C 相对 A 的 Top-1 命中变差） =="]
    any_reg = False
    for key, label in (("rank_b", "B"), ("rank_c", "C")):
        for p in r["per_q"]:
            if p["rank_a"] == 0 and p[key] != 0:
                any_reg = True
                lines.append(f"  [{label}] {p['question']}：Top-1 命中 → rank={p[key]}")
    if not any_reg:
        lines.append("  无（重排未让任何 A Top-1 命中的题变差）✓")
    return "\n".join(lines)


def run(threshold: float | None = None) -> None:
    r = bench_all(threshold=threshold)
    print()
    print(format_table(r))
    if r["threshold"] is not None:
        print()
        print(f"== 阈值模式（正式口径，distance <= {r['threshold']} 才进入重排）==")
        print(f"  全候选被阈值剔除（返回空）的题数：{r['threshold_dropped']}")
        print(f"  正确 chunk 被阈值剔除（重排无救）的题数：{r['threshold_correct_dropped']}")
    print()
    print(format_six_missed(r))
    print()
    print(format_regressions(r))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=None,
                        help="候选过滤阈值（正式口径 0.85；缺省=离线口径不过滤）")
    args = parser.parse_args()
    run(threshold=args.threshold)

