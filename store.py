"""
向量库这一层。负责把 chunk 存进去、根据问题找出最相关的几段。

这里用 PersistentClient 而不是内存模式，
数据会写到磁盘上的 chroma_data_v2/ 文件夹，程序重启后还在。
（第一周验收要求"ChromaDB 重启后数据仍存在"，靠的就是这个。）

2026-08：正式数据库已切换为方案 C 切块构建的 chroma_data_v2/。
旧库 chroma_data/（旧切块）保留未动，作为回滚库：
    store.py 的 DB_DIR 改回 chroma_data 并重启服务即可回滚。

embedding 模型用本地的 all-MiniLM-L6-v2：
- CPU 就能跑，不用显卡
- 模型只有 80MB 左右
- 免费，不需要 API key
- 第二周要做 ONNX 的时候，正好可以把这个模型导出成 ONNX 来练手

注意 DeepSeek 没有 embeddings 接口，所以向量化这步只能本地做，
DeepSeek 只负责最后生成答案那一步。
"""

from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

from chunker import Chunk
from reranker import rerank

DB_DIR = Path(__file__).parent / "chroma_data_v2"  # 正式库（方案C切块）
COLLECTION_NAME = "documents"
EMBED_MODEL = "all-MiniLM-L6-v2"
RERANK_RECALL = 10  # 向量召回上限（Top-10），再交给 Cross-Encoder 重排


def get_collection():
    """拿到（或创建）存文档的 collection。"""
    client = chromadb.PersistentClient(path=str(DB_DIR))

    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        # cosine 距离更适合文本相似度，chroma 默认是 l2
        metadata={"hnsw:space": "cosine"},
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


def search(question: str, top_k: int = 5, threshold: float = 0.85) -> list[dict]:
    """
    根据问题找出最相关的 top_k 段（向量召回 + Cross-Encoder 重排）。

    流程：向量召回最多 Top-10 候选 → 按 threshold 过滤（distance <= threshold
    才视为可靠）→ Cross-Encoder 重排 → 返回重排后的前 top_k。
    返回 [{text, source, page, distance}, ...]；distance 仍是原始向量距离
    （不因重排改变），排序为重排后的相关度顺序。

    重排模型加载/推理失败时自动回退纯向量排序（不抛异常）；
    没有任何候选通过阈值时返回 []（“无可靠结果”语义与旧版一致）。

    threshold: cosine 距离阈值，只返回距离 <= threshold 的结果。
               0.85 对于 all-MiniLM-L6-v2 是一个合理的默认值。
    """
    collection = get_collection()

    if collection.count() == 0:
        return []

    result = collection.query(
        query_texts=[question],
        n_results=min(RERANK_RECALL, collection.count()),
    )

    hits = []
    for text, meta, dist in zip(
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0],
    ):
        if dist <= threshold:
            hits.append({
                "text": text,
                "source": meta["source"],
                "page": meta["page"],
                "distance": dist,
            })

    if not hits:
        return []

    # Cross-Encoder 重排（失败自动回退原向量顺序）；top_k 截断
    return rerank(question, hits, top_k=top_k)


def delete_source(source: str) -> None:
    """删掉某个文件的全部 chunk。第二周验收要用到。"""
    get_collection().delete(where={"source": source})


def stats() -> dict:
    """看看库里现在有多少 chunk、来自哪些文件。"""
    collection = get_collection()
    count = collection.count()
    if count == 0:
        return {"chunks": 0, "sources": []}

    all_meta = collection.get(include=["metadatas"])["metadatas"]
    sources = sorted({m["source"] for m in all_meta})
    return {"chunks": count, "sources": sources}
