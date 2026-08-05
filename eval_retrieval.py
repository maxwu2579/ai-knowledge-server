"""
三种检索方案对比评估（40 题）。

用法：
    py eval_retrieval.py            # 跑三个方案，输出对比表 + 失败案例
    py eval_retrieval.py --model X --mode {original,manual,auto}  # 子进程模式

三种方案（均用当前正式模型 all-MiniLM-L6-v2）：
    方案A  原始问题          + L6-v2（当前正式方案）
    方案C  人工英文改写问题  + L6-v2（中文/混合问题用 eval_questions.py 里
           人工提供的 rewritten_en；英文问题不改写）
    方案D  DeepSeek 自动改写 + L6-v2（改写来自 eval_rewrite_cache.json，
           由 py eval_rewrite.py 生成；英文问题不改写）

设计约束：
- 不改 store.py / api.py / .env / chroma_data（只读导出语料）、不改相关性阈值；
- /search 保持零 DeepSeek 费用：本脚本检索过程不调用 DeepSeek，
  自动改写结果来自 eval_rewrite.py 生成的缓存文件；
- 每个方案使用独立的临时 ChromaDB 目录（tempfile.mkdtemp），结束后自动清理。

评估语料：从现有 chroma_data 只读导出的全部 chunk（与线上索引内容一致）。
命中判定：标准答案 expected_fragment 出现在返回段落文本中。

指标：Top-1 / Top-3 命中率（按语言分组）、Top-1 distance、首次加载时间、
      平均查询时间、内存占用（进程 Working Set 增量，ctypes 读 Windows API）。
"""

import argparse
import ctypes
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

from eval_questions import QUESTIONS

DB_DIR = Path(__file__).parent / "chroma_data"
COLLECTION_NAME = "documents"
THRESHOLD = 0.85  # 与 store.search 保持一致，仅用于参考统计，不参与命中判定

MODEL_CURRENT = "all-MiniLM-L6-v2"                       # 当前正式模型

REWRITE_ORIGINAL = "original"   # 用原问题
REWRITE_MANUAL = "manual"       # 用 eval_questions.py 的人工英文改写
REWRITE_AUTO = "auto"           # 用 eval_rewrite_cache.json 的 DeepSeek 自动改写

# (方案名, 模型, 改写模式)
SCHEMES = [
    ("方案A 原始+L6-v2", MODEL_CURRENT, REWRITE_ORIGINAL),
    ("方案C 人工改写+L6-v2", MODEL_CURRENT, REWRITE_MANUAL),
    ("方案D DeepSeek改写+L6-v2", MODEL_CURRENT, REWRITE_AUTO),
]

LANG_LABEL = {"en": "英文", "zh": "中文", "mixed": "混合"}

AUTO_CACHE_FILE = Path(__file__).parent / "eval_rewrite_cache.json"


def load_auto_rewrites() -> dict:
    """读取 DeepSeek 自动改写缓存 {问题: 英文改写}。"""
    if AUTO_CACHE_FILE.exists():
        return json.loads(AUTO_CACHE_FILE.read_text(encoding="utf-8"))
    return {}


def pick_query(q: dict, rewrite_mode: str, auto_cache: dict) -> str:
    """按改写模式选择实际检索用的查询。英文问题任何模式下都用原问题。"""
    if rewrite_mode == REWRITE_MANUAL:
        return q.get("rewritten_en") or q["question"]
    if rewrite_mode == REWRITE_AUTO:
        return auto_cache.get(q["question"]) or q["question"]
    return q["question"]


# ---------------------------------------------------------------------------
# 内存占用（Windows）
# ---------------------------------------------------------------------------


def process_working_set_mb() -> float:
    """当前进程的 Working Set（RSS），MB。读 Windows psapi，无第三方依赖。"""
    try:
        import ctypes as ct
        from ctypes import wintypes

        class PROCESS_MEMORY_COUNTERS(ct.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ct.c_size_t),
                ("WorkingSetSize", ct.c_size_t),
                ("QuotaPeakPagedPoolUsage", ct.c_size_t),
                ("QuotaPagedPoolUsage", ct.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ct.c_size_t),
                ("QuotaNonPagedPoolUsage", ct.c_size_t),
                ("PagefileUsage", ct.c_size_t),
                ("PeakPagefileUsage", ct.c_size_t),
            ]

        # 注意：必须显式声明 restype/argtypes。
        # 默认 restype 是 32 位 c_int，会把 64 位 HANDLE 截断成无效句柄。
        kernel32 = ct.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE

        psapi = ct.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ct.POINTER(PROCESS_MEMORY_COUNTERS),
            wintypes.DWORD,
        ]

        pmc = PROCESS_MEMORY_COUNTERS()
        pmc.cb = ct.sizeof(pmc)
        ok = psapi.GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(),
            ct.byref(pmc),
            pmc.cb,
        )
        if not ok:
            return 0.0
        return pmc.WorkingSetSize / 1024 / 1024
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# 语料：从现有 chroma_data 只读导出（不需要 embedding 函数，不会加载模型）
# ---------------------------------------------------------------------------


def load_corpus() -> list[dict]:
    client = chromadb.PersistentClient(path=str(DB_DIR))
    col = client.get_or_create_collection(COLLECTION_NAME)
    got = col.get(include=["documents", "metadatas"])
    return [
        {"id": cid, "text": doc, "source": meta["source"], "page": meta["page"]}
        for cid, doc, meta in zip(got["ids"], got["documents"], got["metadatas"])
    ]


# ---------------------------------------------------------------------------
# 命中判定（纯逻辑，供测试复用）
# ---------------------------------------------------------------------------


def is_hit(returned_texts: list[str], fragment: str) -> int:
    """标准答案是否在返回结果中。命中返回 rank（0 起），未命中返回 -1。"""
    for i, text in enumerate(returned_texts):
        if fragment in text:
            return i
    return -1


# ---------------------------------------------------------------------------
# 单模型评估（在独立子进程中运行，保证内存数据互不污染）
# ---------------------------------------------------------------------------


def bench_model(model_name: str, rewrite_mode: str = REWRITE_ORIGINAL) -> dict:
    corpus = load_corpus()
    db_dir = tempfile.mkdtemp(prefix="eval_chroma_")
    auto_cache = load_auto_rewrites() if rewrite_mode == REWRITE_AUTO else {}

    def pick_query_for(q: dict) -> str:
        return pick_query(q, rewrite_mode, auto_cache)

    try:
        rss_before = process_working_set_mb()

        t0 = time.perf_counter()
        embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_name
        )
        t_model = time.perf_counter()

        col = chromadb.PersistentClient(path=db_dir).get_or_create_collection(
            name="eval",
            embedding_function=embed_fn,
            metadata={"hnsw:space": "cosine"},
        )
        col.add(
            ids=[c["id"] for c in corpus],
            documents=[c["text"] for c in corpus],
            metadatas=[{"source": c["source"], "page": c["page"]} for c in corpus],
        )
        t_index = time.perf_counter()

        # 第一次查询：完成模型预热（含首次 embedding）
        col.query(query_texts=[pick_query_for(QUESTIONS[0])], n_results=3)
        t_first = time.perf_counter()
        rss_after = process_working_set_mb()

        # 逐题查询
        per_q = []
        for q in QUESTIONS:
            query_text = pick_query_for(q)
            tq = time.perf_counter()
            res = col.query(query_texts=[query_text], n_results=3)
            elapsed = time.perf_counter() - tq

            texts = res["documents"][0]
            metas = res["metadatas"][0]
            dists = res["distances"][0]
            rank = is_hit(texts, q["expected_fragment"])
            per_q.append({
                "question": q["question"],
                "query_used": query_text,
                "lang": q["lang"],
                "expected_source": q["expected_source"],
                "rank": rank,
                "top1_dist": dists[0],
                "top1_preview": texts[0][:80].replace("\n", " "),
                "returned_sources": [m["source"] for m in metas],
                "passed_threshold": sum(1 for d in dists if d <= THRESHOLD),
            })
        t_done = time.perf_counter()

        # ---- 汇总指标 --------------------------------------------------------
        n = len(per_q)
        top1_hits = sum(1 for r in per_q if r["rank"] == 0)
        top3_hits = sum(1 for r in per_q if r["rank"] >= 0)
        by_lang = {}
        for lang in ("en", "zh", "mixed"):
            sub = [r for r in per_q if r["lang"] == lang]
            by_lang[lang] = {
                "n": len(sub),
                "top1": sum(1 for r in sub if r["rank"] == 0),
                "top3": sum(1 for r in sub if r["rank"] >= 0),
                "avg_top1_dist": round(
                    sum(r["top1_dist"] for r in sub) / len(sub), 3
                )
                if sub
                else None,
            }

        return {
            "model": model_name,
            "rewrite_mode": rewrite_mode,
            "scheme": scheme_label(model_name, rewrite_mode),
            "corpus_chunks": len(corpus),
            "first_load_s": round(t_first - t0, 2),   # 模型加载 + 建库 + 首次查询
            "model_init_s": round(t_model - t0, 2),
            "index_s": round(t_index - t_model, 2),
            "avg_query_s": round((t_done - t_first) / n, 4),
            "rss_before_mb": round(rss_before, 1),
            "rss_after_index_mb": round(rss_after, 1),
            "rss_delta_mb": round(rss_after - rss_before, 1),
            "top1": {"n": n, "hits": top1_hits, "rate": round(top1_hits / n, 4)},
            "top3": {"n": n, "hits": top3_hits, "rate": round(top3_hits / n, 4)},
            "by_lang": by_lang,
            "avg_top1_distance": round(
                sum(r["top1_dist"] for r in per_q) / n, 3
            ),
            "failures_top1": [r for r in per_q if r["rank"] != 0],
            "failures_top3": [r for r in per_q if r["rank"] == -1],
            "per_q": per_q,
            "auto_missing": [
                q["question"]
                for q in QUESTIONS
                if q["lang"] in ("zh", "mixed")
                and q["question"] not in auto_cache
            ]
            if rewrite_mode == REWRITE_AUTO
            else [],
        }
    finally:
        shutil.rmtree(db_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# 对比表
# ---------------------------------------------------------------------------


def scheme_label(model_name: str, rewrite_mode: str) -> str:
    for name, model, mode in SCHEMES:
        if model == model_name and mode == rewrite_mode:
            return name
    return model_name


def format_table(results: list[dict]) -> str:
    col_w = 30
    headers = "".join(r["scheme"].ljust(col_w) for r in results)

    lines = [
        f"语料：{results[0]['corpus_chunks']} 块（从现有 chroma_data 只读导出，与线上索引一致）",
        "",
        f"{'指标':<26}{headers}",
        "-" * (26 + col_w * len(results)),
    ]

    def row(metric: str, values: list[str]) -> None:
        lines.append(metric.ljust(26) + "".join(v.ljust(col_w) for v in values))

    row(
        "Top-1 命中率",
        [f"{r['top1']['hits']}/{r['top1']['n']} ({r['top1']['rate']:.0%})" for r in results],
    )
    row(
        "Top-3 命中率",
        [f"{r['top3']['hits']}/{r['top3']['n']} ({r['top3']['rate']:.0%})" for r in results],
    )
    for lang in ("en", "zh", "mixed"):
        row(
            f"  {LANG_LABEL[lang]} Top-1 / Top-3",
            [f"{r['by_lang'][lang]['top1']}/{r['by_lang'][lang]['n']} / "
             f"{r['by_lang'][lang]['top3']}/{r['by_lang'][lang]['n']}" for r in results],
        )
        row(
            f"  {LANG_LABEL[lang]} 平均 Top-1 distance",
            [str(r["by_lang"][lang]["avg_top1_dist"]) for r in results],
        )
    row("Top-1 平均 distance（越小越相关）", [str(r["avg_top1_distance"]) for r in results])
    row("首次加载时间（含建库+首查，秒）", [str(r["first_load_s"]) for r in results])
    row("  其中模型初始化（秒）", [str(r["model_init_s"]) for r in results])
    row("  其中语料建库（秒）", [str(r["index_s"]) for r in results])
    row("平均查询时间（秒/题）", [str(r["avg_query_s"]) for r in results])
    row("内存占用（RSS 增量，MB）", [str(r["rss_delta_mb"]) for r in results])
    return "\n".join(lines)


def format_failures(result: dict) -> str:
    lines = [f"== {result['scheme']} =="]

    if not result["failures_top1"]:
        lines.append("  Top-1 全部命中 ✓")
    for r in result["failures_top1"]:
        rank = r["rank"] if r["rank"] >= 0 else "Top-3 未命中"
        src = r["returned_sources"][0] if r["returned_sources"] else "?"
        preview = r.get("top1_preview", "")
        lines.append(f"  [{LANG_LABEL[r['lang']]}] {r['question']}")
        lines.append(
            f"      → 标准答案 rank={rank}；Top-1 返回 [{src}] 「{preview}…」"
        )

    if result["failures_top3"]:
        lines.append("  其中 Top-3 完全未命中的问题：")
        for r in result["failures_top3"]:
            lines.append(f"    [{LANG_LABEL[r['lang']]}] {r['question']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


def run_single(model_name: str, rewrite_mode: str) -> None:
    """子进程模式：输出一行 JSON，供主流程解析。"""
    result = bench_model(model_name, rewrite_mode=rewrite_mode)
    print(json.dumps(result, ensure_ascii=False))


def _rank_delta(manual_rank: int, auto_rank: int) -> int:
    """auto 相对 manual 的好坏：1=更好，-1=更差，0=相同。rank=-1 表示未命中。"""
    if manual_rank == auto_rank:
        return 0
    if manual_rank == -1:
        return 1
    if auto_rank == -1:
        return -1
    return 1 if auto_rank < manual_rank else -1


def compare_manual_vs_auto(r_manual: dict, r_auto: dict) -> tuple[list, list, int]:
    """对中文/混合问题逐题比较人工改写与自动改写的命中 rank。"""
    manual_by_q = {r["question"]: r for r in r_manual["per_q"]}
    worse, better, same = [], [], 0
    for r in r_auto["per_q"]:
        if r["lang"] == "en":
            continue
        m = manual_by_q[r["question"]]
        delta = _rank_delta(m["rank"], r["rank"])
        if delta == 0:
            same += 1
        elif delta < 0:
            worse.append({
                "question": r["question"],
                "lang": r["lang"],
                "manual_rank": m["rank"],
                "auto_rank": r["rank"],
                "auto_query": r["query_used"],
            })
        else:
            better.append({
                "question": r["question"],
                "lang": r["lang"],
                "manual_rank": m["rank"],
                "auto_rank": r["rank"],
                "auto_query": r["query_used"],
            })
    return worse, better, same


def run_all() -> None:
    results = []
    for scheme, model, rewrite_mode in SCHEMES:
        print(f"评估 {scheme} ...", file=sys.stderr)
        proc = subprocess.run(
            [sys.executable, str(__file__), "--model", model, "--mode", rewrite_mode],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=3600,
        )
        if proc.returncode != 0:
            print(proc.stderr, file=sys.stderr)
            sys.exit(f"{scheme} 评估失败")
        results.append(json.loads(proc.stdout.strip().splitlines()[-1]))

    print()
    print(format_table(results))
    print()
    print("== 失败案例 ==")
    for result in results:
        print(format_failures(result))
        print()

    r_manual = next(r for r in results if r["rewrite_mode"] == REWRITE_MANUAL)
    r_auto = next(r for r in results if r["rewrite_mode"] == REWRITE_AUTO)

    # 逐条输出自动改写，供检查泄漏与语义变化
    print("== 自动改写（方案D）逐条输出（中文/混合，实际用于检索的查询）==")
    for r in r_auto["per_q"]:
        if r["lang"] == "en":
            continue
        print(f"  [{LANG_LABEL[r['lang']]}] {r['question']} -> \"{r['query_used']}\"")
    if r_auto["auto_missing"]:
        print("  警告：以下问题没有缓存改写，回退用了原问题：")
        for q in r_auto["auto_missing"]:
            print(f"    {q}")

    worse, better, same = compare_manual_vs_auto(r_manual, r_auto)
    print()
    print("== 方案D vs 方案C（人工改写）逐题对比（中文/混合）==")
    print(f"  变好 {len(better)} 题、变差 {len(worse)} 题、相同 {same} 题")
    if worse:
        print("  自动改写比人工改写变差的案例：")
        for w in worse:
            rank_c = w["manual_rank"] if w["manual_rank"] >= 0 else "Top-3 未命中"
            rank_d = w["auto_rank"] if w["auto_rank"] >= 0 else "Top-3 未命中"
            print(
                f"    [{LANG_LABEL[w['lang']]}] {w['question']}: "
                f"人工 rank={rank_c} -> 自动 rank={rank_d} "
                f"（自动查询：\"{w['auto_query']}\"）"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="只评估指定模型（子进程模式）")
    parser.add_argument(
        "--mode",
        choices=[REWRITE_ORIGINAL, REWRITE_MANUAL, REWRITE_AUTO],
        default=REWRITE_ORIGINAL,
        help="查询改写模式：original=原问题 / manual=人工改写 / auto=DeepSeek 自动改写",
    )
    args = parser.parse_args()

    if args.model:
        run_single(args.model, rewrite_mode=args.mode)
    else:
        run_all()
