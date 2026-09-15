"""
向量库这一层。负责把 chunk 存进去、根据问题找出最相关的几段。

这里用 PersistentClient 而不是内存模式，
数据会写到磁盘上的 data/chroma_data_v2/ 文件夹，程序重启后还在。
（第一周验收要求"ChromaDB 重启后数据仍存在"，靠的就是这个。）

2026-08：正式数据库已切换为方案 C 切块构建的 data/chroma_data_v2/。
旧库 data/chroma_data/（旧切块）保留未动，作为回滚库：
    store.py 的 DB_DIR 改回 data/chroma_data 并重启服务即可回滚。

embedding 模型用本地的 all-MiniLM-L6-v2：
- CPU 就能跑，不用显卡
- 模型只有 80MB 左右
- 免费，不需要 API key
- 第二周要做 ONNX 的时候，正好可以把这个模型导出成 ONNX 来练手

注意 DeepSeek 没有 embeddings 接口，所以向量化这步只能本地做，
DeepSeek 只负责最后生成答案那一步。
"""

import time
from functools import lru_cache
from pathlib import Path

import chromadb
from chromadb.errors import NotFoundError
from chromadb.utils import embedding_functions
from langfuse import get_client, observe

from chunker import Chunk
from reranker import rerank

DB_DIR = Path(__file__).parent / "data" / "chroma_data_v2"  # 正式库（方案C切块）
COLLECTION_NAME = "documents"
EMBED_MODEL = "all-MiniLM-L6-v2"
RERANK_RECALL = 10  # 去重后的候选上限（Top-10），再交给 Cross-Encoder 重排
RAW_CANDIDATE_LIMIT = max(RERANK_RECALL * 2, RERANK_RECALL + 10)


@lru_cache(maxsize=1)
def get_embedding_function():
    """延迟创建并复用正式检索使用的 embedding function。"""
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )


def get_collection():
    """拿到（或创建）存文档的 collection。"""
    client = chromadb.PersistentClient(path=str(DB_DIR))
    embed_fn = get_embedding_function()

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        # cosine 距离更适合文本相似度，chroma 默认是 l2
        metadata={"hnsw:space": "cosine"},
    )


def warm_up_vector_components() -> dict:
    """只读预热 embedding、首次 encode 与现有 Chroma/HNSW 索引。"""
    embedding_started = time.perf_counter()
    embed_fn = get_embedding_function()
    warmup_embedding = embed_fn(["startup warmup query"])[0]
    embedding_ms = (time.perf_counter() - embedding_started) * 1000

    hnsw_started = time.perf_counter()
    client = chromadb.PersistentClient(path=str(DB_DIR))
    try:
        collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=embed_fn,
        )
    except NotFoundError:
        return {
            "embedding_ms": embedding_ms,
            "hnsw_ms": (time.perf_counter() - hnsw_started) * 1000,
            "hnsw_status": "collection-missing",
        }

    if collection.count() == 0:
        return {
            "embedding_ms": embedding_ms,
            "hnsw_ms": (time.perf_counter() - hnsw_started) * 1000,
            "hnsw_status": "empty",
        }

    collection.query(
        query_embeddings=[warmup_embedding],
        n_results=1,
    )
    return {
        "embedding_ms": embedding_ms,
        "hnsw_ms": (time.perf_counter() - hnsw_started) * 1000,
        "hnsw_status": "warmed",
    }


def get_metadata_collection():
    """打开现有 collection，仅用于 count/get 等不需要 embedding 的操作。"""
    client = chromadb.PersistentClient(path=str(DB_DIR))
    return client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=None,
    )


def add_chunks(chunks: list[Chunk]) -> int:
    """把 chunk 写进向量库。同一份文件重复导入会覆盖，不会产生重复数据。"""
    if not chunks:
        return 0

    collection = get_collection()

    collection.upsert(
        ids=[c.chunk_id(i) for i, c in enumerate(chunks)],
        documents=[c.text for c in chunks],
        metadatas=[{"source": c.source, "page": c.page} for c in chunks],
    )
    return len(chunks)


@observe(name="vector-search")
def _vector_search(question: str, threshold: float) -> list[dict]:
    collection = get_collection()

    if collection.count() == 0:
        get_client().update_current_span(
            metadata={
                "retrieved_candidate_count": 0,
                "threshold": threshold,
                "recall_limit": RERANK_RECALL,
            }
        )
        return []

    result = collection.query(
        query_texts=[question],
        n_results=min(RAW_CANDIDATE_LIMIT, collection.count()),
    )

    hits = []
    seen: set[tuple[str, int, str]] = set()
    for text, meta, dist in zip(
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0],
    ):
        if dist <= threshold:
            identity = (meta["source"], meta["page"], text)
            if identity in seen:
                continue
            seen.add(identity)
            hits.append({
                "text": text,
                "source": meta["source"],
                "page": meta["page"],
                "distance": dist,
            })
            if len(hits) == RERANK_RECALL:
                break

    get_client().update_current_span(
        metadata={
            "retrieved_candidate_count": len(hits),
            "threshold": threshold,
            "recall_limit": RERANK_RECALL,
            "raw_candidate_limit": RAW_CANDIDATE_LIMIT,
        }
    )

    return hits


def search(question: str, top_k: int = 5, threshold: float = 0.85) -> list[dict]:
    """
    根据问题找出最相关的 top_k 段（向量召回 + Cross-Encoder 重排）。

    流程：向量召回最多 20 条原始候选 → 按 threshold 过滤（distance <= threshold
    才视为可靠）→ 按 source/page/text 稳定去重并保留前 10 条 →
    Cross-Encoder 重排 → 返回重排后的前 top_k。
    返回 [{text, source, page, distance}, ...]；distance 仍是原始向量距离
    （不因重排改变），排序为重排后的相关度顺序。

    重排模型加载/推理失败时自动回退纯向量排序（不抛异常）；
    没有任何候选通过阈值时返回 []（“无可靠结果”语义与旧版一致）。

    threshold: cosine 距离阈值，只返回距离 <= threshold 的结果。
               0.85 对于 all-MiniLM-L6-v2 是一个合理的默认值。
    """
    hits = _vector_search(question, threshold)

    if not hits:
        return []

    # Cross-Encoder 重排（失败自动回退原向量顺序）；top_k 截断
    return rerank(question, hits, top_k=top_k)


def delete_source(source: str) -> None:
    """删掉某个文件的全部 chunk。第二周验收要用到。"""
    get_collection().delete(where={"source": source})


def stats() -> dict:
    """看看库里现在有多少 chunk、来自哪些文件。"""
    try:
        collection = get_metadata_collection()
    except NotFoundError:
        return {"chunks": 0, "sources": []}

    count = collection.count()
    if count == 0:
        return {"chunks": 0, "sources": []}

    all_meta = collection.get(include=["metadatas"])["metadatas"]
    sources = sorted({m["source"] for m in all_meta})
    return {"chunks": count, "sources": sources}
