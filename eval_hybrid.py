"""
混合检索离线对比实验（纯向量 vs 纯 BM25 vs RRF 融合），零 DeepSeek 费用。

语料：chroma_data_v2 的 15 个 chunk（与正式库完全一致，只读打开，不修改）。
查询：中文/混合问题沿用 eval_questions.py 的人工英文改写（rewritten_en），
      英文问题用原问题——全部以英文检索，与线上 /query 的改写行为一致。

方案：
    A 纯向量检索（collection.query，与 /search 同构，cosine 距离升序排名）
    B 纯 BM25 关键词检索（chunk 文本分词建索引，保留 source / page）
    C 向量 + BM25 融合（RRF：score = Σ 1/(k + rank_i)，k=60）

指标：整体 Top-1 / Top-3、按语言分组、重点失败类型分组
      （汇报对象 / 岗位 / 开发内容 / 回传文件 / 津贴福利专名）、
      平均查询时间、失败案例与回归案例（方案C 相对方案A 变差的题）。

设计约束：
- 不改 store.py / api.py / ask.py / .env / 正式数据库 / embedding / 0.85 阈值；
- 评估集 50 题（40 基线 + 10 失败类型补充），标准答案原文判定，不迎合结果；
- 不调用 DeepSeek。

用法：
    py eval_hybrid.py            # 单进程跑三个方案，输出对比表 + 失败/回归案例
"""

import math
import re
import time
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

from eval_questions import EXTRA_QUESTIONS, QUESTIONS

DB_V2 = Path(__file__).parent / "chroma_data_v2"
COLLECTION_NAME = "documents"
EMBED_MODEL = "all-MiniLM-L6-v2"

LANG_LABEL = {"en": "英文", "zh": "中文", "mixed": "混合"}

# BM25 参数（标准默认值）
K1 = 1.5
B = 0.75
RRF_K = 60

# 重点失败类型 → 题目映射（按问题原文，与 50 题评估集对应）
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


# ---------------------------------------------------------------------------
# 分词与 BM25（纯函数，供测试复用）
# ---------------------------------------------------------------------------

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """英文检索词分词：小写、只保留字母数字 token。"""
    return TOKEN_RE.findall(text.lower())


class BM25Index:
    """从 chunk 文本构建的 BM25 索引，保留 source / page 元数据。"""

    def __init__(self, docs: list[dict], k1: float = K1, b: float = B):
        """
        docs: [{"id": str, "text": str, "source": str, "page": int}, ...]
        """
        self.k1 = k1
        self.b = b
        self.docs = docs
        self.n = len(docs)
        self.doc_tokens: list[list[str]] = [tokenize(d["text"]) for d in docs]
        self.doc_len = [len(t) for t in self.doc_tokens]
        self.avgdl = (sum(self.doc_len) / self.n) if self.n else 0.0
        # 词频表与文档频率
        self.tf: list[dict[str, int]] = []
        self.df: dict[str, int] = {}
        for tokens in self.doc_tokens:
            counts: dict[str, int] = {}
            for t in tokens:
                counts[t] = counts.get(t, 0) + 1
            self.tf.append(counts)
            for t in set(tokens):
                self.df[t] = self.df.get(t, 0) + 1

    def idf(self, term: str) -> float:
        """IDF：ln(1 + (N - df + 0.5) / (df + 0.5))，罕见词权重大。"""
        df = self.df.get(term, 0)
        if df == 0:
            return 0.0
        return math.log(1.0 + (self.n - df + 0.5) / (df + 0.5))

    def score(self, query: str, doc_idx: int) -> float:
        """BM25 评分：只统计查询词与文档的共现词。"""
        total = 0.0
        for term in set(tokenize(query)):
            tf = self.tf[doc_idx].get(term, 0)
            if tf == 0:
                continue
            denom = tf + self.k1 * (
                1.0 - self.b + self.b * self.doc_len[doc_idx] / self.avgdl
            )
            total += self.idf(term) * tf * (self.k1 + 1.0) / denom
        return total

    def search(self, query: str) -> list[tuple[int, float]]:
        """返回 [(doc_idx, score), ...]，按分数降序；无任何词命中时为空。"""
        query_terms = set(tokenize(query))
        if not query_terms:
            return []
        scored = [
            (i, self.score(query, i)) for i in range(self.n) if self.score(query, i) > 0
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def results(self, query: str, top_k: int | None = None) -> list[dict]:
        """BM25 检索结果，保留 source / page。"""
        hits = self.search(query)
        if top_k is not None:
            hits = hits[:top_k]
        return [
            {
                "id": self.docs[i]["id"],
                "text": self.docs[i]["text"],
                "source": self.docs[i]["source"],
                "page": self.docs[i]["page"],
                "bm25_score": round(score, 4),
            }
            for i, score in hits
        ]


# ---------------------------------------------------------------------------
# 排名融合（RRF 与 min-max 归一化加权）
# ---------------------------------------------------------------------------


def rrf_merge(rankings: list[list[int]], k: float = RRF_K) -> list[int]:
    """Reciprocal Rank Fusion：多路排名融合。

    rankings 是若干 doc_id 排名列表（排前 = 相关），每路对排名 r 贡献 1/(k+r)，
    按总分会聚排序返回全部 doc_id。空列表（某路无结果）安全忽略。
    """
    scores: dict[int, float] = {}
    for rank_list in rankings:
        for r, doc_id in enumerate(rank_list):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + r + 1)
    return [doc_id for doc_id, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


def minmax_normalize(values: list[float]) -> list[float]:
    """min-max 归一化到 [0, 1]；全同值时返回全 0。"""
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi == lo:
        return [0.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def weighted_merge(vector_dists: list[tuple[int, float]],
                   bm25_scores: list[tuple[int, float]],
                   w_vector: float = 0.5,
                   w_bm25: float = 0.5) -> list[int]:
    """归一化加权融合：向量距离归一化（越小越好 → 转 1-x）与 BM25 分数
    归一化后加权求和。返回按总分降序的 doc_id 列表。空路自动忽略。"""
    v = dict(vector_dists)
    b = dict(bm25_scores)
    all_ids = sorted(set(v) | set(b))
    if not all_ids:
        return []

    v_norm = minmax_normalize([v.get(i, float("inf")) for i in all_ids])
    b_norm = minmax_normalize([b.get(i, 0.0) for i in all_ids])
    total = {
        i: w_vector * (1.0 - vn) + w_bm25 * bn
        for i, vn, bn in zip(all_ids, v_norm, b_norm)
    }
    return [i for i, _ in sorted(total.items(), key=lambda x: x[1], reverse=True)]


# ---------------------------------------------------------------------------
# 语料：只读打开 chroma_data_v2
# ---------------------------------------------------------------------------


def load_v2_corpus() -> list[dict]:
    """只读导出 chroma_data_v2 的全部 chunk（text + source + page）。"""
    col = chromadb.PersistentClient(path=str(DB_V2)).get_or_create_collection(
        COLLECTION_NAME
    )
    got = col.get(include=["documents", "metadatas"])
    return [
        {
            "id": cid,
            "text": doc,
            "source": meta["source"],
            "page": meta["page"],
        }
        for cid, doc, meta in zip(got["ids"], got["documents"], got["metadatas"])
    ]


# ---------------------------------------------------------------------------
# 三方案评估（单进程：一次向量查询即可同时得到三种排名）
# ---------------------------------------------------------------------------


def pick_query(q: dict) -> str:
    return q.get("rewritten_en") or q["question"]


def bench_all() -> dict:
    corpus = load_v2_corpus()
    idx_of = {c["id"]: i for i, c in enumerate(corpus)}  # doc id → corpus 索引
    bm25 = BM25Index(corpus)

    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )
    col = chromadb.PersistentClient(path=str(DB_V2)).get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )
    # 预热（含首次 embedding）
    col.query(query_texts=[pick_query(QUESTIONS[0])], n_results=3)

    questions = QUESTIONS + EXTRA_QUESTIONS
    t_query = 0.0
    t_bm25 = 0.0
    t_hybrid = 0.0

    per_q = []
    for q in questions:
        query_text = pick_query(q)

        # ---- 方案 A：纯向量 -------------------------------------------------
        t0 = time.perf_counter()
        res = col.query(query_texts=[query_text], n_results=len(corpus))
        t_query += time.perf_counter() - t0
        # query 返回顺序即距离升序 → 映射回 corpus 索引得到向量排名
        vector_ids = [idx_of[i] for i in res["ids"][0]]

        # ---- 方案 B：纯 BM25 -------------------------------------------------
        t0 = time.perf_counter()
        bm25_hits = bm25.search(query_text)
        t_bm25 += time.perf_counter() - t0
        bm25_ids = [i for i, _ in bm25_hits]

        # ---- 方案 C：RRF 融合 -------------------------------------------------
        t0 = time.perf_counter()
        hybrid_ids = rrf_merge([vector_ids, bm25_ids])
        # 变体：归一化加权融合（向量距离对 / BM25 分数对）
        dist_pairs = list(zip(vector_ids, res["distances"][0]))
        score_pairs = [(i, s) for i, s in bm25_hits]
        w50_ids = weighted_merge(dist_pairs, score_pairs, 0.5, 0.5)
        w70_ids = weighted_merge(dist_pairs, score_pairs, 0.3, 0.7)
        t_hybrid += time.perf_counter() - t0

        def rank_of(ids: list[int], fragment: str) -> int:
            for r, i in enumerate(ids):
                if fragment in corpus[i]["text"]:
                    return r
            return -1

        per_q.append({
            "question": q["question"],
            "lang": q["lang"],
            "query_used": query_text,
            "expected_source": q["expected_source"],
            "expected_fragment": q["expected_fragment"],
            "rank_a": rank_of(vector_ids, q["expected_fragment"]),
            "rank_b": rank_of(bm25_ids, q["expected_fragment"]),
            "rank_c": rank_of(hybrid_ids, q["expected_fragment"]),
            "rank_w50": rank_of(w50_ids, q["expected_fragment"]),
            "rank_w70": rank_of(w70_ids, q["expected_fragment"]),
            "top1_source_a": corpus[vector_ids[0]]["source"],
            "top1_source_c": corpus[hybrid_ids[0]]["source"] if hybrid_ids else None,
            "bm25_hit_count": len(bm25_hits),
            "bm25_top1_preview": (
                corpus[bm25_hits[0][0]]["text"][:60] if bm25_hits else "(空)"
            ),
        })
    n = len(per_q)

    def summarize(rank_key: str) -> dict:
        top1 = sum(1 for r in per_q if r[rank_key] == 0)
        top3 = sum(1 for r in per_q if 0 <= r[rank_key] < 3)
        by_lang = {}
        for lang in ("en", "zh", "mixed"):
            sub = [r for r in per_q if r["lang"] == lang]
            by_lang[lang] = {
                "n": len(sub),
                "top1": sum(1 for r in sub if r[rank_key] == 0),
                "top3": sum(1 for r in sub if r[rank_key] >= 0),
            }
        by_type = {}
        for ftype, fqs in FAILURE_TYPES.items():
            sub = [r for r in per_q if r["question"] in fqs]
            by_type[ftype] = {
                "n": len(sub),
                "top1": sum(1 for r in sub if r[rank_key] == 0),
                "top3": sum(1 for r in sub if r[rank_key] >= 0),
            }
        return {
            "n": n,
            "top1": top1,
            "top3": top3,
            "rate1": round(top1 / n, 4),
            "rate3": round(top3 / n, 4),
            "by_lang": by_lang,
            "by_type": by_type,
        }

    return {
        "chunks": len(corpus),
        "avg_query_s": round(t_query / n, 4),
        "avg_bm25_s": round(t_bm25 / n, 4),
        "avg_hybrid_s": round(t_hybrid / n, 4),
        "summary_a": summarize("rank_a"),
        "summary_b": summarize("rank_b"),
        "summary_c": summarize("rank_c"),
        "summary_w50": summarize("rank_w50"),
        "summary_w70": summarize("rank_w70"),
        "per_q": per_q,
        "failures": {
            "a": [r for r in per_q if r["rank_a"] != 0],
            "b": [r for r in per_q if r["rank_b"] != 0],
            "c": [r for r in per_q if r["rank_c"] != 0],
        },
        "miss_top3": {
            "a": [r for r in per_q if not (0 <= r["rank_a"] < 3)],
            "b": [r for r in per_q if not (0 <= r["rank_b"] < 3)],
            "c": [r for r in per_q if not (0 <= r["rank_c"] < 3)],
        },
    }


# ---------------------------------------------------------------------------
# 对比表（单结果：A/B/C 三列）
# ---------------------------------------------------------------------------


def format_table(result: dict) -> str:
    col_w = 22
    labels = ("A 纯向量", "B 纯BM25", "C-RRF", "C-w50(1:1)", "C-w70(3:7)")
    headers = "".join(label.ljust(col_w) for label in labels)
    lines = [
        f"语料：{result['chunks']} 块（chroma_data_v2，方案C切块）；"
        f"题数：{result['summary_c']['n']}（40 基线 + 10 失败补充）",
        "",
        f"{'指标':<24}{headers}",
        "-" * (24 + col_w * 5),
    ]

    def row(metric: str, values: list[str]) -> None:
        lines.append(metric.ljust(24) + "".join(v.ljust(col_w) for v in values))

    s = [result["summary_a"], result["summary_b"], result["summary_c"],
         result["summary_w50"], result["summary_w70"]]
    row("Top-1 命中率", [f"{x['top1']}/{x['n']} ({x['rate1']:.0%})" for x in s])
    row("Top-3 命中率", [f"{x['top3']}/{x['n']} ({x['rate3']:.0%})" for x in s])
    for lang in ("en", "zh", "mixed"):
        row(
            f"  {LANG_LABEL[lang]} Top-1 / Top-3",
            [f"{x['by_lang'][lang]['top1']}/{x['by_lang'][lang]['n']} / "
             f"{x['by_lang'][lang]['top3']}/{x['by_lang'][lang]['n']}" for x in s],
        )
    for ftype in FAILURE_TYPES:
        row(
            f"  {ftype} Top-1 / Top-3",
            [f"{x['by_type'][ftype]['top1']}/{x['by_type'][ftype]['n']} / "
             f"{x['by_type'][ftype]['top3']}/{x['by_type'][ftype]['n']}" for x in s],
        )
    row("平均向量查询（秒/题）", [str(result["avg_query_s"])] * 5)
    row("平均融合耗时（秒/题）", [str(result["avg_bm25_s"])] * 2 +
                                 [str(round(result["avg_bm25_s"] + result["avg_hybrid_s"], 4))] * 3)
    return "\n".join(lines)


def format_failures(result: dict) -> str:
    lines = ["== Top-3 完全未命中 =="]
    for scheme_key, label in (("a", "A 纯向量"), ("b", "B 纯BM25"), ("c", "C 融合")):
        missed = result["miss_top3"][scheme_key]
        lines.append(f"[{label}] {len(missed)} 题：")
        for r in missed:
            lines.append(f"    [{LANG_LABEL[r['lang']]}] {r['question']}")
    return "\n".join(lines)


def format_regressions(result: dict) -> str:
    """方案 C 相对方案 A 的回归案例（A 命中而 C 未命中 / 排名变差）。"""
    lines = ["== 回归案例（C 相对 A 变差） =="]
    regressions = []
    for r in result["per_q"]:
        a_hit = r["rank_a"] >= 0
        c_hit = r["rank_c"] >= 0
        if a_hit and not c_hit:
            regressions.append((r, "Top-3 命中 → Top-3 未命中"))
        elif r["rank_a"] == 0 and r["rank_c"] != 0:
            regressions.append((r, f"Top-1 命中 → rank={r['rank_c']}"))
    if not regressions:
        lines.append("  无（C 未让任何 A 命中的题变差）✓")
    for r, note in regressions:
        lines.append(f"  [{LANG_LABEL[r['lang']]}] {r['question']}：{note}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


def run() -> None:
    result = bench_all()
    print()
    print(format_table(result))
    print()
    print(format_failures(result))
    print()
    print(format_regressions(result))


if __name__ == "__main__":
    run()
