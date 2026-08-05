"""
切块策略离线对比评估（只评估检索，零 DeepSeek 费用）。

对比三种切块策略（同一个 all-MiniLM-L6-v2 模型，同一份原始文档）：

    方案A  当前正式策略：整页压平 + 500 字符窗口 + 自然断点 + 50 字符重叠
          （等价于 chunker.load_document 的默认行为）
    方案B  按页面与自然段落切块：每页按空行分成自然段落，一段一块；
          超长段落内部按句号再切（段落内不重叠）
    方案C  段落切块 + 相邻内容重叠：在 B 的基础上按滑动窗口合并相邻两段，
          每块含下一段，相邻块共享段落内容

三种策略的正式实现已收编进 chunker.py（方案C = load_document_paragraphs），
本脚本只做薄包装、临时库构建与指标汇总。另有 --db 模式可直接评估
已建好的候选库（如 chroma_data_v2），用于复现离线实验结论。

设计约束：
- 不改 store.py / api.py / .env / chroma_data（只读导出语料）与相关性阈值；
- 每个方案使用独立的临时 ChromaDB 目录（tempfile.mkdtemp），结束后自动清理；
- 不调用 DeepSeek，不产生 API 费用；
- 每种方案保留 source / page 元数据，与线上格式一致。

评估语料：原始文档 = docs/ 下的 PDF + 桌面上的 Letter of Appointment PDF
+ 从正式 chroma_data 导出的 cloud_doc.txt（一句话，无段落结构，三方案都是 1 块）。
脚本开头会校验方案A切出的块与正式库 chunk 逐条一致，确保实验语料与线上同源。

查询方式：中文/混合问题用 eval_questions.py 的人工英文改写（rewritten_en），
英文问题用原问题——全部问题以英文检索，排除查询语言干扰，聚焦切块策略差异。

指标：Top-1 / Top-3 命中率（总体 + 按语言）、平均 Top-1 distance、
      chunk 数量、建库时间、平均查询时间、失败案例。

用法：
    py eval_chunking.py                    # 跑三个方案，输出对比表 + 失败案例
    py eval_chunking.py --strategy a       # 子进程模式：只评估单个方案
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

from chunker import (
    Chunk,
    chunk_paragraphs,
    chunk_paragraphs_overlap,
    read_pdf,
    split_paragraphs,
    split_text,
)
from eval_questions import EXTRA_QUESTIONS, QUESTIONS
from eval_retrieval import DB_DIR, load_corpus

EMBED_MODEL = "all-MiniLM-L6-v2"          # 与线上完全一致，不换模型
THRESHOLD = 0.85                          # 与 store.search 一致，仅参考统计
COLLECTION_NAME = "documents"

# 原始文档输入（只读）。cloud_doc.txt 已不在磁盘，运行时从正式库导出。
RAW_DOCS = [
    Path(__file__).parent / "docs" / "university letter concerning d internship.pdf",
    Path(r"C:\Users\25795\Desktop\Letter of Appointment Internship - WU ZHONGHENG.pdf"),
]
CLOUD_DOC_SOURCE = "cloud_doc.txt"

STRATEGY_A = "a"   # 当前正式策略
STRATEGY_B = "b"   # 按页面与自然段落切块
STRATEGY_C = "c"   # 段落切块 + 相邻内容重叠（chunker 正式推荐）

SCHEMES = [
    ("方案A 当前策略(500字符+断点+50重叠)", STRATEGY_A),
    ("方案B 页面自然段落", STRATEGY_B),
    ("方案C 段落+相邻段重叠", STRATEGY_C),
]

LANG_LABEL = {"en": "英文", "zh": "中文", "mixed": "混合"}


# ---------------------------------------------------------------------------
# 三种切块策略（薄包装：正式实现已收编进 chunker.py，实验脚本只做调度）
# ---------------------------------------------------------------------------


def chunk_pages_a(pages: list[tuple[int, str]], size: int = 500, overlap: int = 50, source: str = "") -> list[Chunk]:
    """方案A：整页压平 + 字符窗口 + 自然断点 + 重叠（等价 chunker.load_document）。"""
    chunks: list[Chunk] = []
    for page_no, page_text in pages:
        for piece in split_text(page_text, size, overlap):
            chunks.append(Chunk(text=piece, source=source, page=page_no))
    return chunks


def chunk_pages_b(pages: list[tuple[int, str]], size: int = 500, source: str = "") -> list[Chunk]:
    """方案B：按页面与自然段落切块（chunker.chunk_paragraphs）。"""
    return chunk_paragraphs(pages, size=size, source=source)


def chunk_pages_c(pages: list[tuple[int, str]], size: int = 500, source: str = "", max_join: int = 800) -> list[Chunk]:
    """方案C：段落切块 + 相邻内容重叠（chunker.chunk_paragraphs_overlap，正式推荐）。"""
    return chunk_paragraphs_overlap(pages, size=size, source=source, max_join=max_join)


def chunk_pages(pages: list[tuple[int, str]], strategy: str, source: str = "") -> list[Chunk]:
    """统一入口：按策略切一页列表，返回带 source / page 的 chunk。"""
    if strategy == STRATEGY_A:
        return chunk_pages_a(pages, source=source)
    if strategy == STRATEGY_B:
        return chunk_pages_b(pages, source=source)
    if strategy == STRATEGY_C:
        return chunk_pages_c(pages, source=source)
    raise ValueError(f"未知切块策略：{strategy}")


# ---------------------------------------------------------------------------
# 原始文档：只读 PDF + 从正式库导出 cloud_doc.txt
# ---------------------------------------------------------------------------


def load_raw_docs() -> list[tuple[str, list[tuple[int, str]]]]:
    """返回 [(文件名, [(页码, 文本), ...]), ...]，与正式库三份源文件一致。"""
    docs: list[tuple[str, list[tuple[int, str]]]] = []
    for path in RAW_DOCS:
        pages = [(no, txt) for no, txt in read_pdf(path)]
        docs.append((path.name, pages))

    # cloud_doc.txt 已不在磁盘：从正式库只读导出原文，内容与线上逐字符一致
    cloud = next(c for c in load_corpus() if c["source"] == CLOUD_DOC_SOURCE)
    docs.append((CLOUD_DOC_SOURCE, [(1, cloud["text"])]))
    return docs


def verify_matches_production(docs: list[tuple[str, list[tuple[int, str]]]]) -> None:
    """校验方案A切出的块与正式库逐条一致（实验语料与线上同源才可比）。"""
    prod = {}
    for c in load_corpus():
        prod.setdefault(c["source"], []).append(c["text"])

    ok = True
    for source, pages in docs:
        mine = [c.text for c in chunk_pages_a(pages, source=source)]
        theirs = prod.get(source, [])
        if mine != theirs:
            ok = False
            print(
                f"警告：{source} 方案A切块与正式库不一致（实验 {len(mine)} 块 vs "
                f"正式 {len(theirs)} 块），实验语料可能与线上有出入",
                file=sys.stderr,
            )
    if ok:
        print("校验：方案A切块与正式 chroma_data 逐条一致 ✓", file=sys.stderr)


# ---------------------------------------------------------------------------
# 单方案评估（子进程模式运行，保证各方案内存互不污染）
# ---------------------------------------------------------------------------


def bench_strategy(strategy: str) -> dict:
    docs = load_raw_docs()
    verify_matches_production(docs)

    db_dir = tempfile.mkdtemp(prefix="eval_chunking_")
    try:
        t0 = time.perf_counter()
        embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBED_MODEL
        )
        t_model = time.perf_counter()

        col = chromadb.PersistentClient(path=db_dir).get_or_create_collection(
            name="eval",
            embedding_function=embed_fn,
            metadata={"hnsw:space": "cosine"},
        )

        all_chunks: list[Chunk] = []
        for source, pages in docs:
            all_chunks.extend(chunk_pages(pages, strategy, source=source))
        n_chunks = len(all_chunks)

        col.add(
            ids=[c.chunk_id(i) for i, c in enumerate(all_chunks)],
            documents=[c.text for c in all_chunks],
            metadatas=[{"source": c.source, "page": c.page} for c in all_chunks],
        )
        t_index = time.perf_counter()

        questions = QUESTIONS + EXTRA_QUESTIONS

        def pick_query(q: dict) -> str:
            return q.get("rewritten_en") or q["question"]

        # 首次查询：预热（含首次 embedding）
        col.query(query_texts=[pick_query(questions[0])], n_results=3)
        t_first = time.perf_counter()

        per_q = []
        for q in questions:
            query_text = pick_query(q)
            tq = time.perf_counter()
            res = col.query(query_texts=[query_text], n_results=3)
            elapsed = time.perf_counter() - tq

            texts = res["documents"][0]
            metas = res["metadatas"][0]
            dists = res["distances"][0]
            rank = next((i for i, t in enumerate(texts) if q["expected_fragment"] in t), -1)
            per_q.append({
                "question": q["question"],
                "lang": q["lang"],
                "query_used": query_text,
                "expected_source": q["expected_source"],
                "expected_fragment": q["expected_fragment"],
                "rank": rank,
                "top1_dist": dists[0],
                "top1_preview": texts[0][:80].replace("\n", " "),
                "returned_sources": [m["source"] for m in metas],
                "passed_threshold": sum(1 for d in dists if d <= THRESHOLD),
            })
        t_done = time.perf_counter()

        n = len(per_q)
        by_lang = {}
        for lang in ("en", "zh", "mixed"):
            sub = [r for r in per_q if r["lang"] == lang]
            by_lang[lang] = {
                "n": len(sub),
                "top1": sum(1 for r in sub if r["rank"] == 0),
                "top3": sum(1 for r in sub if r["rank"] >= 0),
                "avg_top1_dist": round(sum(r["top1_dist"] for r in sub) / len(sub), 3)
                if sub else None,
            }

        return {
            "strategy": strategy,
            "scheme": scheme_label(strategy),
            "chunks": n_chunks,
            "model_init_s": round(t_model - t0, 2),
            "index_s": round(t_index - t_model, 2),
            "first_load_s": round(t_first - t0, 2),
            "avg_query_s": round((t_done - t_first) / n, 4),
            "top1": {"n": n, "hits": sum(1 for r in per_q if r["rank"] == 0),
                     "rate": round(sum(1 for r in per_q if r["rank"] == 0) / n, 4)},
            "top3": {"n": n, "hits": sum(1 for r in per_q if r["rank"] >= 0),
                     "rate": round(sum(1 for r in per_q if r["rank"] >= 0) / n, 4)},
            "by_lang": by_lang,
            "avg_top1_distance": round(sum(r["top1_dist"] for r in per_q) / n, 3),
            "failures_top1": [r for r in per_q if r["rank"] != 0],
            "failures_top3": [r for r in per_q if r["rank"] == -1],
            "per_q": per_q,
        }
    finally:
        shutil.rmtree(db_dir, ignore_errors=True)


def scheme_label(strategy: str) -> str:
    for name, s in SCHEMES:
        if s == strategy:
            return name
    return strategy


# ---------------------------------------------------------------------------
# 评估已建好的候选库（--db 模式：不重建、不切块，直接逐题查询）
# ---------------------------------------------------------------------------


def bench_existing_db(db_path: str) -> dict:
    """对已建好的 ChromaDB 目录（如 chroma_data_v2）运行 50 题评估。

    指标结构与 bench_strategy 一致，chunks = 库内实际 chunk 数，
    index_s 填 0.0（没有建库阶段）。查询方式与切块实验相同：
    中文/混合用人工英文改写，英文用原问题。
    """
    t0 = time.perf_counter()
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )
    t_model = time.perf_counter()

    col = chromadb.PersistentClient(path=db_path).get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )
    n_chunks = col.count()
    questions = QUESTIONS + EXTRA_QUESTIONS

    def pick_query(q: dict) -> str:
        return q.get("rewritten_en") or q["question"]

    # 首次查询：预热（含首次 embedding）
    col.query(query_texts=[pick_query(questions[0])], n_results=3)
    t_first = time.perf_counter()

    per_q = []
    for q in questions:
        query_text = pick_query(q)
        tq = time.perf_counter()
        res = col.query(query_texts=[query_text], n_results=3)
        elapsed = time.perf_counter() - tq

        texts = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]
        rank = next((i for i, t in enumerate(texts) if q["expected_fragment"] in t), -1)
        per_q.append({
            "question": q["question"],
            "lang": q["lang"],
            "query_used": query_text,
            "expected_source": q["expected_source"],
            "expected_fragment": q["expected_fragment"],
            "rank": rank,
            "top1_dist": dists[0],
            "top1_preview": texts[0][:80].replace("\n", " "),
            "returned_sources": [m["source"] for m in metas],
            "passed_threshold": sum(1 for d in dists if d <= THRESHOLD),
        })
    t_done = time.perf_counter()

    n = len(per_q)
    by_lang = {}
    for lang in ("en", "zh", "mixed"):
        sub = [r for r in per_q if r["lang"] == lang]
        by_lang[lang] = {
            "n": len(sub),
            "top1": sum(1 for r in sub if r["rank"] == 0),
            "top3": sum(1 for r in sub if r["rank"] >= 0),
            "avg_top1_dist": round(sum(r["top1_dist"] for r in sub) / len(sub), 3)
            if sub else None,
        }

    return {
        "strategy": "db",
        "scheme": f"候选库 {Path(db_path).name}",
        "chunks": n_chunks,
        "model_init_s": round(t_model - t0, 2),
        "index_s": 0.0,  # 已有库没有建库阶段
        "first_load_s": round(t_first - t0, 2),
        "avg_query_s": round((t_done - t_first) / n, 4),
        "top1": {"n": n, "hits": sum(1 for r in per_q if r["rank"] == 0),
                 "rate": round(sum(1 for r in per_q if r["rank"] == 0) / n, 4)},
        "top3": {"n": n, "hits": sum(1 for r in per_q if r["rank"] >= 0),
                 "rate": round(sum(1 for r in per_q if r["rank"] >= 0) / n, 4)},
        "by_lang": by_lang,
        "avg_top1_distance": round(sum(r["top1_dist"] for r in per_q) / n, 3),
        "failures_top1": [r for r in per_q if r["rank"] != 0],
        "failures_top3": [r for r in per_q if r["rank"] == -1],
        "per_q": per_q,
    }


# ---------------------------------------------------------------------------
# 对比表
# ---------------------------------------------------------------------------


def format_table(results: list[dict]) -> str:
    col_w = 30
    headers = "".join(r["scheme"].ljust(col_w) for r in results)

    lines = [
        f"评估题数：{results[0]['top1']['n']}（40 题 + 失败类型补充 "
        f"{results[0]['top1']['n'] - 40} 题）；查询：中文/混合用人工英文改写，英文用原问题",
        "",
        f"{'指标':<26}{headers}",
        "-" * (26 + col_w * len(results)),
    ]

    def row(metric: str, values: list[str]) -> None:
        lines.append(metric.ljust(26) + "".join(v.ljust(col_w) for v in values))

    row("chunk 数量", [str(r["chunks"]) for r in results])
    row("Top-1 命中率", [f"{r['top1']['hits']}/{r['top1']['n']} ({r['top1']['rate']:.0%})" for r in results])
    row("Top-3 命中率", [f"{r['top3']['hits']}/{r['top3']['n']} ({r['top3']['rate']:.0%})" for r in results])
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
    row("Top-1 平均 distance", [str(r["avg_top1_distance"]) for r in results])
    row("模型初始化（秒）", [str(r["model_init_s"]) for r in results])
    row("建库时间（秒）", [str(r["index_s"]) for r in results])
    row("首次加载（含首查，秒）", [str(r["first_load_s"]) for r in results])
    row("平均查询时间（秒/题）", [str(r["avg_query_s"]) for r in results])
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
        lines.append(f"      → 标准答案 rank={rank}；Top-1 返回 [{src}] 「{preview}…」")
    if result["failures_top3"]:
        lines.append("  其中 Top-3 完全未命中的问题：")
        for r in result["failures_top3"]:
            lines.append(f"    [{LANG_LABEL[r['lang']]}] {r['question']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


def run_single(strategy: str) -> None:
    """子进程模式：输出一行 JSON，供主流程解析。"""
    result = bench_strategy(strategy)
    print(json.dumps(result, ensure_ascii=False))


def run_all() -> None:
    results = []
    for scheme, strategy in SCHEMES:
        print(f"评估 {scheme} ...", file=sys.stderr)
        proc = subprocess.run(
            [sys.executable, str(__file__), "--strategy", strategy],
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=[STRATEGY_A, STRATEGY_B, STRATEGY_C],
                        help="只评估指定策略（子进程模式）")
    parser.add_argument("--db", help="评估已建好的候选库目录（如 chroma_data_v2），不重建")
    args = parser.parse_args()

    if args.db:
        result = bench_existing_db(args.db)
        print(json.dumps(result, ensure_ascii=False))
        print(file=sys.stderr)
        print(format_table([result]), file=sys.stderr)
        print(file=sys.stderr)
        print(format_failures(result), file=sys.stderr)
    elif args.strategy:
        run_single(args.strategy)
    else:
        run_all()
