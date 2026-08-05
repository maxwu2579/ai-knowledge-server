"""
把文档导入知识库。

用法：
    py ingest.py docs/handbook.pdf
    py ingest.py docs/            # 导入整个文件夹
    py ingest.py --stats          # 看看库里现在有什么
"""

import sys
from pathlib import Path

from chunker import load_document_paragraphs
from store import add_chunks, stats

SUPPORTED = {".pdf", ".txt", ".md"}


def ingest_path(target: Path) -> None:
    if target.is_dir():
        files = [f for f in sorted(target.iterdir()) if f.suffix.lower() in SUPPORTED]
        if not files:
            print(f"{target} 里没有找到 pdf / txt / md 文件")
            return
    else:
        files = [target]

    total = 0
    for f in files:
        try:
            chunks = load_document_paragraphs(f)  # 方案C：段落 + 相邻段重叠
        except Exception as e:
            print(f"  跳过 {f.name}：{e}")
            continue

        if not chunks:
            print(f"  跳过 {f.name}：没提取到文字（可能是扫描版 PDF）")
            continue

        n = add_chunks(chunks)
        pages = len({c.page for c in chunks})
        print(f"  {f.name}：{pages} 页 -> {n} 块")
        total += n

    print(f"\n完成，共导入 {total} 块")


def main() -> None:
    args = sys.argv[1:]

    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        return

    if args[0] == "--stats":
        s = stats()
        print(f"库里有 {s['chunks']} 块，来自 {len(s['sources'])} 份文件：")
        for src in s["sources"]:
            print(f"  - {src}")
        return

    target = Path(args[0])
    if not target.exists():
        print(f"找不到：{target}")
        sys.exit(1)

    print(f"正在导入 {target} ...")
    ingest_path(target)


if __name__ == "__main__":
    main()
