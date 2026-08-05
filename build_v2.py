"""
用方案 C（段落 + 相邻段重叠）构建候选知识库 chroma_data_v2。

候选库与正式库完全同构（collection 名 "documents"、同一 all-MiniLM-L6-v2、
cosine 空间、metadatas 含 source/page），正式 API 将来只需把 DB_DIR 指向
chroma_data_v2 即可切换，不需要改代码。

本脚本不修改、不覆盖、不删除正式 chroma_data：
- 原始文档只读（docs/ 下的 PDF + 桌面上的 Letter of Appointment PDF +
  cloud_doc.txt 从正式库只读导出，逐字符一致）；
- 输出只写入 chroma_data_v2/ 新目录。

用法：
    py build_v2.py
"""

import sys
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

from chunker import load_document_paragraphs
from eval_retrieval import load_corpus

DB_V2 = Path(__file__).parent / "chroma_data_v2"
COLLECTION_NAME = "documents"
EMBED_MODEL = "all-MiniLM-L6-v2"

# 原始文档输入（与 eval_chunking.py 一致）。cloud_doc.txt 已不在磁盘，
# 从正式 chroma_data 导出原文后重建（内容与线上逐字符一致）。
RAW_DOCS = [
    Path(__file__).parent / "docs" / "university letter concerning d internship.pdf",
    Path(r"C:\Users\25795\Desktop\Letter of Appointment Internship - WU ZHONGHENG.pdf"),
]
CLOUD_DOC_SOURCE = "cloud_doc.txt"


def main() -> None:
    # ---- 1. 收集原始文档 ------------------------------------------------
    paths: list[Path] = []
    for p in RAW_DOCS:
        if not p.exists():
            print(f"找不到原始文档：{p}", file=sys.stderr)
            sys.exit(1)
        paths.append(p)

    cloud = next((c for c in load_corpus() if c["source"] == CLOUD_DOC_SOURCE), None)
    if cloud is None:
        print(f"正式库中找不到 {CLOUD_DOC_SOURCE}，无法重建", file=sys.stderr)
        sys.exit(1)
    cloud_path = Path(cloud["source"])
    cloud_path.write_text(cloud["text"], encoding="utf-8")  # 临时重建，用完删除
    try:
        paths.append(cloud_path)

        # ---- 2. 方案 C 切块 -----------------------------------------------
        all_chunks = []
        for p in paths:
            chunks = load_document_paragraphs(p)
            print(f"  {p.name}：{len(chunks)} 块（方案C：段落+相邻段重叠）")
            all_chunks.extend(chunks)

        if DB_V2.exists():
            print(f"警告：{DB_V2} 已存在，将重新写入（正式 chroma_data 不受影响）")

        # ---- 3. 写入候选库（与正式 store.py 同构） --------------------------
        embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBED_MODEL
        )
        col = chromadb.PersistentClient(path=str(DB_V2)).get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=embed_fn,
            metadata={"hnsw:space": "cosine"},
        )
        col.upsert(
            ids=[c.chunk_id(i) for i, c in enumerate(all_chunks)],
            documents=[c.text for c in all_chunks],
            metadatas=[{"source": c.source, "page": c.page} for c in all_chunks],
        )

        print(f"\n完成：{DB_V2.name}/ 共 {len(all_chunks)} 块（正式库 chroma_data 未动）")
    finally:
        cloud_path.unlink(missing_ok=True)  # 清理临时重建的 cloud_doc.txt


if __name__ == "__main__":
    main()
