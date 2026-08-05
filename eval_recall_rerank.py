"""
候选补召回 + Cross-Encoder 重排离线实验，零 DeepSeek 费用。

目标：解决正确 chunk 排在向量 Top-10 之外的问题（如"实习生的直属上司是谁？"
向量排名 11），同时不破坏当前正式方案（Top-1 82% / Top-3 92%）。

语料：chroma_data_v2 的 15 个 chunk（只读）。查询：50 题，中文/混合用
人工英文改写（rewritten_en），英文用原问题。

方案：
    A 当前正式：向量 Top-10 → 0.85 过滤 → CE 重排
    B 扩大召回：向量 Top-15 → 0.85 过滤 → CE 重排
    C 混合补召回：向量 Top-10 ∪ BM25 Top-5（按 chunk 去重）
      → BM25 独有候选补算真实向量距离（1 - cosine(query, chunk)，绝不伪造）
      → 0.85 可靠性过滤（与 A/B 同一规则，不静默绕过）
      → CE 统一重排

可靠性规则说明（方案 C）：
- 来自向量 Top-10 的候选保留原始 distance；
- BM25 独有候选没有向量 distance——用 embedding 模型对 (查询, chunk) 真实编码
  计算 cosine 距离，得到的是真实距离而非伪造值；
- 所有候选统一执行 distance <= 0.85 才进入重排；
- 若 BM25 独有候选全部超过 0.85（常见情形：它不在向量 Top-10 正是因为
  距离较大），则它们被剔除，方案 C 与方案 A 等价——如实报告，不强行放行。

指标：整体 Top-1 / Top-3、分语言、重点失败类型、回归案例（B/C 相对 A 变差）、
      候选数量、每题延迟（向量 / BM25 / 补算 / 重排）、内存占用。

重点验证：
    Who does the intern report to?（向量排名 9，Top-10 内但 CE 未救回）
    实习生的直属上司是谁？（向量排名 11，Top-10 外）

用法：
    py eval_recall_rerank.py
"""

import time
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

from eval_hybrid import BM25Index
from eval_questions import EXTRA_QUESTIONS, QUESTIONS
from eval_rerank import FAILURE_TYPES, rerank
from eval_retrieval import process_working_set_mb

DB_V2 = Path(__file__).parent / "chroma_data_v2"
COLLECTION_NAME = "documents"
EMBED_MODEL = "all-MiniLM-L6-v2"
CE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
THRESHOLD = 0.85  # 与正式 store.search 一致（不修改正式配置）

LANG_LABEL = {"en": "英文", "zh": "中文", "mixed": "混合"}

# 单独验证的两题
FOCUS_QUESTIONS = [
    "Who does the intern report to?",
    "实习生的直属上司是谁？",
]


def pick_query(q: dict) -> str:
    return q.get("rewritten_en") or q["question"]


def cosine_distance(a: list[float], b: list[float]) -> float:
    """1 - cosine 相似度（与 ChromaDB 的 cosine 距离口径一致）。"""
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return 1.0 - dot / (na * nb)


def build_union(
    vector_pairs: list[tuple[int, float]],
    bm25_ids: list[int],
    corpus: list[dict],
    query_emb: list[float],
    emb: list[list[float]],
    threshold: float,
) -> tuple[list[dict], list[dict]]:
    """构建混合补召回候选（方案 C 核心，纯函数供测试）。

    vector_pairs: 向量 Top-10 的 (corpus 索引, 原始 distance)。
    bm25_ids:     BM25 Top-5 的 corpus 索引。
    返回 (过滤后的候选列表, BM25 独有候选列表)。

    可靠性规则：
    - 向量侧候选保留 ChromaDB 返回的原始 distance；
    - BM25 独有候选没有向量 distance——用 (查询, chunk) 真实编码计算
      cosine 距离（不伪造）；
    - 并集统一执行 distance <= threshold 过滤（与方案 A/B 同一规则）。
    """
    union = {i: {**corpus[i], "distance": d} for i, d in vector_pairs}
    extra = []
    for i in bm25_ids:
        if i not in union:
            d = cosine_distance(query_emb, emb[i])
            extra.append({**corpus[i], "distance": d})
    filtered = [c for c in list(union.values()) + extra if c["distance"] <= threshold]
    return filtered, extra


def bench_all() -> dict:
    # ---- 语料与向量库（只读） ---------------------------------------------
    col = chromadb.PersistentClient(path=str(DB_V2)).get_or_create_collection(
        COLLECTION_NAME
    )
    got = col.get(include=["documents", "metadatas", "embeddings"])
    corpus = [
        {"id": cid, "text": doc, "source": meta["source"], "page": meta["page"]}
        for cid, doc, meta in zip(got["ids"], got["documents"], got["metadatas"])
    ]
    emb = got["embeddings"]  # 与 corpus 顺序对应的全库向量
    idx_of = {c["id"]: i for i, c in enumerate(corpus)}
    bm25 = BM25Index(corpus)

    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )
    vec_col = chromadb.PersistentClient(path=str(DB_V2)).get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )
    vec_col.query(query_texts=[pick_query(QUESTIONS[0])], n_results=3)  # 预热

    from sentence_transformers import CrossEncoder

    t_ce0 = time.perf_counter()
    rss_before = process_working_set_mb()
    ce = CrossEncoder(CE_MODEL)
    t_ce_load = time.perf_counter() - t_ce0
    rss_after = process_working_set_mb()

    def ce_scorer(pairs):
        return list(ce.predict(pairs, show_progress_bar=False))

    questions = QUESTIONS + EXTRA_QUESTIONS
    n = len(questions)
    t_query = t_bm25 = t_dist = t_ce = 0.0
    per_q = []

    for q in questions:
        query_text = pick_query(q)

        # ---- A：向量 Top-10 → 过滤 → 重排 -----------------------------------
        t0 = time.perf_counter()
        res = vec_col.query(query_texts=[query_text], n_results=10)
        t_query += time.perf_counter() - t0
        v_ids = [idx_of[i] for i in res["ids"][0]]
        v_dists = res["distances"][0]
        a_cands = [
            {**corpus[i], "distance": d}
            for i, d in zip(v_ids, v_dists) if d <= THRESHOLD
        ]

        t0 = time.perf_counter()
        a_ranked = rerank(query_text, a_cands, ce_scorer)
        t_ce += time.perf_counter() - t0

        # ---- B：向量 Top-15（全量）→ 过滤 → 重排 -----------------------------
        t0 = time.perf_counter()
        res15 = vec_col.query(query_texts=[query_text], n_results=15)
        t_query += time.perf_counter() - t0
        v15_ids = [idx_of[i] for i in res15["ids"][0]]
        v15_dists = res15["distances"][0]
        b_cands = [
            {**corpus[i], "distance": d}
            for i, d in zip(v15_ids, v15_dists) if d <= THRESHOLD
        ]

        t0 = time.perf_counter()
        b_ranked = rerank(query_text, b_cands, ce_scorer)
        t_ce += time.perf_counter() - t0

        # ---- C：向量 Top-10 ∪ BM25 Top-5 → 补算距离 → 过滤 → 重排 -------------
        t0 = time.perf_counter()
        bm25_hits = bm25.search(query_text)[:5]
        t_bm25 += time.perf_counter() - t0
        bm25_ids = [i for i, _ in bm25_hits]

        # 并集：向量侧保留原始 distance；BM25 独有候选补算真实向量距离
        t0 = time.perf_counter()
        query_emb = embed_fn([query_text])[0]
        c_cands, extra = build_union(
            list(zip(v_ids, v_dists)), bm25_ids, corpus, query_emb, emb, THRESHOLD
        )
        t_dist += time.perf_counter() - t0
        t0 = time.perf_counter()
        c_ranked = rerank(query_text, c_cands, ce_scorer)
        t_ce += time.perf_counter() - t0

        def rank_of(ranked_cands, fragment):
            for r, c in enumerate(ranked_cands):
                if fragment in c["text"]:
                    return r
            return -1

        per_q.append({
            "question": q["question"],
            "lang": q["lang"],
            "rank_a": rank_of(a_ranked, q["expected_fragment"]),
            "rank_b": rank_of(b_ranked, q["expected_fragment"]),
            "rank_c": rank_of(c_ranked, q["expected_fragment"]),
            "cands_a": len(a_cands),
            "cands_b": len(b_cands),
            "cands_c": len(c_cands),
            "bm25_extra": len(extra),
            "bm25_extra_survived": sum(1 for c in extra if c["distance"] <= THRESHOLD),
        })

    def summarize(rank_key):
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

    return {
        "chunks": len(corpus),
        "ce_load_s": round(t_ce_load, 2),
        "rss_delta_mb": round(rss_after - rss_before, 1),
        "avg_query_s": round(t_query / n, 4),
        "avg_bm25_s": round(t_bm25 / n, 4),
        "avg_extra_dist_s": round(t_dist / n, 4),
        "avg_rerank_s": round(t_ce / (3 * n), 4),  # 每题三次重排（A/B/C）
        "avg_cands": {
            "a": round(sum(r["cands_a"] for r in per_q) / n, 1),
            "b": round(sum(r["cands_b"] for r in per_q) / n, 1),
            "c": round(sum(r["cands_c"] for r in per_q) / n, 1),
        },
        "bm25_extra_total": sum(r["bm25_extra"] for r in per_q),
        "bm25_extra_survived_total": sum(r["bm25_extra_survived"] for r in per_q),
        "summary_a": summarize("rank_a"),
        "summary_b": summarize("rank_b"),
        "summary_c": summarize("rank_c"),
        "per_q": per_q,
        "focus": [
            {**r} for r in per_q if r["question"] in FOCUS_QUESTIONS
        ],
    }


def format_table(r: dict) -> str:
    col_w = 22
    labels = ("A Top10+CE", "B Top15+CE", "C 混合补召回+CE")
    headers = "".join(label.ljust(col_w) for label in labels)
    s = [r["summary_a"], r["summary_b"], r["summary_c"]]
    lines = [
        f"语料：{r['chunks']} 块；题数：{r['summary_a']['n']}；"
        f"阈值 0.85 在重排前统一执行；BM25 独有候选补算真实向量距离",
        "",
        f"{'指标':<26}{headers}",
        "-" * (26 + col_w * 3),
    ]

    def row(metric, values):
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
    row("平均候选数（过滤后）", [str(r["avg_cands"][k]) for k in ("a", "b", "c")])
    row("BM25 补入候选（总/过阈值）",
        [str(r["bm25_extra_total"]), "-", f"{r['bm25_extra_total']}/{r['bm25_extra_survived_total']}"])
    row("CE 加载（秒）", [str(r["ce_load_s"])] * 3)
    row("内存增量（MB）", [str(r["rss_delta_mb"])] * 3)
    row("向量查询（秒/题）", [str(r["avg_query_s"])] * 3)
    row("BM25/补算/重排（秒/题）",
        [str(r["avg_bm25_s"]), str(r["avg_extra_dist_s"]), str(round(r["avg_bm25_s"] + r["avg_extra_dist_s"] + r["avg_rerank_s"], 4))])
    return "\n".join(lines)


def format_focus(r: dict) -> str:
    lines = ["== 重点验证：两题在各方案中的排名 =="]
    for item in r["focus"]:
        lines.append(f"  [{LANG_LABEL[item['lang']]}] {item['question']}")
        lines.append(f"      A rank={item['rank_a']}  B rank={item['rank_b']}  C rank={item['rank_c']}"
                     f"  （候选数 A/B/C = {item['cands_a']}/{item['cands_b']}/{item['cands_c']}）")
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
        lines.append("  无 ✓")
    return "\n".join(lines)


def run() -> None:
    r = bench_all()
    print()
    print(format_table(r))
    print()
    print(format_focus(r))
    print()
    print(format_regressions(r))


if __name__ == "__main__":
    run()
