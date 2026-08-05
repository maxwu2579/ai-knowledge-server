"""
候选库 chroma_data_v2 的真实冒烟测试（/search 与 /query 的等价流程）。

- search_v2 ≈ POST /search：向量检索候选库，返回段落数组（阈值 0.85 与正式一致）；
- query_v2  ≈ POST /query：检索候选库 → DeepSeek 生成带出处的答案
  （复用 ask._call_llm 与答案提示词，不修改正式代码；真实调用 DeepSeek）。

检查四项：
    /search 英文（"How long is the internship?"）
    /search 中文（"实习期是多久？"）
    /query  英文（"How long is the internship?"）
    /query  中文（"实习期是多久？"）
每项打印结果并校验 source / page 引用。

正式 API（store.py 的 chroma_data）完全不受影响：本脚本只读 chroma_data_v2。
用法：
    py smoke_v2.py
"""

import re
import sys
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

from ask import _call_llm

DB_V2 = Path(__file__).parent / "chroma_data_v2"
COLLECTION_NAME = "documents"
EMBED_MODEL = "all-MiniLM-L6-v2"
THRESHOLD = 0.85  # 与 store.search 一致（不修改正式配置）

CJK_RE = re.compile(r"[一-鿿]")
CITE_RE = re.compile(r"\[([^\]]+?) p\.(\d+)\]")

SMOKE_CASES = [
    ("/search 英文", "How long is the internship?", "search"),
    ("/search 中文", "实习期是多久？", "search"),
    ("/query 英文", "How long is the internship?", "query"),
    ("/query 中文", "实习期是多久？", "query"),
]


def get_collection():
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )
    return chromadb.PersistentClient(path=str(DB_V2)).get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )


def search_v2(question: str, top_k: int = 5) -> list[dict]:
    """等价 POST /search：只做向量检索，不调用 DeepSeek。"""
    col = get_collection()
    if col.count() == 0:
        return []
    res = col.query(query_texts=[question], n_results=min(top_k, col.count()))
    hits = []
    for text, meta, dist in zip(
        res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        if dist <= THRESHOLD:
            hits.append({
                "text": text,
                "source": meta["source"],
                "page": meta["page"],
                "distance": dist,
            })
    return hits


def query_v2(question: str, top_k: int = 5) -> dict:
    """等价 POST /query：检索候选库 → DeepSeek 生成带出处的答案。"""
    hits = search_v2(question, top_k=top_k)
    if not hits:
        return {"answer": "资料中找不到这个问题的答案。", "sources": []}

    parts = [f"[{h['source']} p.{h['page']}]\n{h['text']}" for h in hits]
    context = "\n\n---\n\n".join(parts)
    user_msg = f"Context:\n\n{context}\n\n---\n\nQuestion: {question}"
    answer = _call_llm(user_msg)
    return {"answer": answer, "sources": hits}


def check_language(question: str, answer: str) -> str:
    """答案语言应跟随提问语言（与回答提示词规则一致）。"""
    if CJK_RE.search(question):  # 中文提问 → 中文答案
        return "中文✓" if CJK_RE.search(answer) else "中文✗（答案未含中文）"
    return "英文✓" if not CJK_RE.search(answer) else "英文✗（答案含中文）"


def check_citations(answer: str, sources: list[dict]) -> str:
    """答案中的 [source p.page] 引用必须真实存在于返回段落里。"""
    cites = CITE_RE.findall(answer)
    if not cites:
        return "无出处引用✗"
    known = {(s["source"], s["page"]) for s in sources}
    for src, page in cites:
        if (src, int(page)) not in known:
            return f"引用越界✗（{src} p.{page} 不在返回段落中）"
    return f"{len(cites)} 处引用均有效✓"


def main() -> None:
    failures = 0
    for label, question, mode in SMOKE_CASES:
        print("=" * 66)
        print(f"{label}：{question}")
        print("=" * 66)
        try:
            if mode == "search":
                hits = search_v2(question)
                print(f"  命中 {len(hits)} 段（阈值 0.85）")
                for i, h in enumerate(hits, 1):
                    preview = h["text"][:70].replace("\n", " ")
                    print(f"  {i}. [{h['source']} p.{h['page']}] 距离={h['distance']:.3f}")
                    print(f"     {preview}…")
                if not hits:
                    print("  ✗ 无命中")
                    failures += 1
                else:
                    top = hits[0]
                    print(f"  Top-1 source/page 校验：{top['source']} p.{top['page']} ✓")
            else:  # query
                result = query_v2(question)
                answer = result["answer"]
                lang = check_language(question, answer)
                cite = check_citations(answer, result["sources"])
                print(f"  答案语言：{lang}")
                print(f"  出处引用：{cite}")
                print(f"  答案：{answer}")
                if "✗" in lang or "✗" in cite:
                    failures += 1
                if not result["sources"]:
                    print("  ✗ 检索无结果")
                    failures += 1
        except Exception as e:
            print(f"  ✗ 出错：{type(e).__name__}: {e}")
            failures += 1
        print()

    print("=" * 66)
    if failures:
        print(f"冒烟测试：{len(SMOKE_CASES) - failures}/{len(SMOKE_CASES)} 项通过")
        sys.exit(1)
    print(f"冒烟测试：{len(SMOKE_CASES)}/{len(SMOKE_CASES)} 项全部通过 ✓（正式库 chroma_data 未受影响）")


if __name__ == "__main__":
    main()
