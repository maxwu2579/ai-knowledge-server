"""
把文档读成一段一段的 chunk，每个 chunk 都要记住它来自哪个文件、第几页。

这一步是整个 RAG 里最容易返工的地方：
如果切块的时候没有把页码带上，后面想让答案标注来源就没辙了，
只能整个重来。所以从第一天就把 source 和 page 存好。

切块策略（2026-08 切块实验后定稿）：
- 旧策略（保留）：load_document / split_text —— 整页压平 + 字符窗口 + 自然断点。
  仅供兼容旧流程与对比实验使用，新导入请用方案 C。
- 方案 C（正式推荐）：load_document_paragraphs —— 按页面与自然段落切块，
  相邻段滑动窗口合并（步长 1 段，每块含下一段），相邻块共享段落内容；
  相邻段不跨 PDF 页面合并，连续相同段落会预合并去重，不产生重复/空白 chunk。
  经 50 题离线评估：Top-1 62% / Top-3 80%，优于旧策略（56% / 68%）。
"""

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Chunk:
    text: str      # 这一块的内容
    source: str    # 来自哪个文件，比如 "handbook.pdf"
    page: int      # 来自第几页（txt/md 文件统一记为 1）

    def chunk_id(self, index: int) -> str:
        """给 ChromaDB 用的唯一 id。同一份文件重新导入时会覆盖而不是重复插入。"""
        return f"{self.source}::p{self.page}::{index}"


def read_pdf(path: Path) -> list[tuple[int, str]]:
    """返回 [(页码, 该页文本), ...]，页码从 1 开始。"""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((i, text))
    return pages


def read_text_file(path: Path) -> list[tuple[int, str]]:
    """txt / md 没有页的概念，整份当作第 1 页。"""
    return [(1, path.read_text(encoding="utf-8", errors="ignore"))]


def split_text(text: str, size: int = 500, overlap: int = 50) -> list[str]:
    """
    把一段长文本切成小块。

    size 是每块大约多少个字符，overlap 是相邻两块重叠多少。
    留重叠是为了避免一句话正好被切断，导致两边都读不懂。

    切的时候优先在句号、换行这些自然断点上切，
    而不是硬按字数砍，这样每块读起来是完整的。
    """
    text = " ".join(text.split())  # 把多余的空白和换行压成单个空格
    if not text:
        return []

    breakpoints = "。！？.!?\n"
    chunks = []
    start = 0

    while start < len(text):
        end = start + size

        if end >= len(text):
            chunks.append(text[start:].strip())
            break

        # 在 end 附近往回找一个自然断点，最多回退 size 的一半
        cut = end
        for i in range(end, max(start + size // 2, start), -1):
            if text[i - 1] in breakpoints:
                cut = i
                break

        chunk = text[start:cut].strip()
        if chunk:
            chunks.append(chunk)

        # 下一块从 cut 往回退 overlap 个字符开始，制造重叠
        start = max(cut - overlap, start + 1)

    return chunks


def load_document(path: Path, size: int = 500, overlap: int = 50) -> list[Chunk]:
    """读一份文档，返回带来源信息的 chunk 列表（旧策略：字符窗口切块）。"""
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        pages = read_pdf(path)
    elif suffix in (".txt", ".md"):
        pages = read_text_file(path)
    else:
        raise ValueError(f"暂不支持的文件类型：{suffix}（目前支持 .pdf .txt .md）")

    chunks = []
    for page_no, page_text in pages:
        for piece in split_text(page_text, size, overlap):
            chunks.append(Chunk(text=piece, source=path.name, page=page_no))

    return chunks


# ---------------------------------------------------------------------------
# 方案 C：按页面与自然段落切块 + 相邻段重叠（正式推荐，2026-08）
# ---------------------------------------------------------------------------

# pypdf 提取的文本里段落间是连续空行（可能含空格），段内换行只有一个 \n。
# 用「换行 + 仅空白行 + 换行」切分自然段落。
PARA_BREAK_RE = re.compile(r"\n[ \t]*\n")


def split_paragraphs(page_text: str) -> list[str]:
    """把一页文本按空行切成自然段落。

    段落内空白压成单空格；strip 后为空的段落（纯空行）丢弃。
    """
    paras = [re.sub(r"\s+", " ", p).strip() for p in PARA_BREAK_RE.split(page_text)]
    return [p for p in paras if p]


def _merge_consecutive_duplicates(units: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """把连续内容完全相同的段落预合并成一段，避免滑动窗口产生重复 chunk。"""
    out: list[tuple[int, str]] = []
    for page_no, text in units:
        if out and text == out[-1][1]:
            out[-1] = (out[-1][0], f"{out[-1][1]} {text}")
        else:
            out.append((page_no, text))
    return out


def chunk_paragraphs(
    pages: list[tuple[int, str]], size: int = 500, source: str = ""
) -> list[Chunk]:
    """方案 B（参考用）：按页面与自然段落切块。

    一段一块；超长段落内部按自然断点再切（段落内不重叠）。保留 source / page。
    """
    chunks: list[Chunk] = []
    for page_no, page_text in pages:
        for para in split_paragraphs(page_text):
            pieces = split_text(para, size, overlap=0) if len(para) > size else [para]
            for piece in pieces:
                chunks.append(Chunk(text=piece, source=source, page=page_no))
    return chunks


def chunk_paragraphs_overlap(
    pages: list[tuple[int, str]],
    size: int = 500,
    source: str = "",
    max_join: int = 800,
) -> list[Chunk]:
    """方案 C（正式推荐）：段落切块 + 相邻内容重叠。

    把每页的自然段落序列化后按滑动窗口合并：每块 = 当前段 + 下一段
    （步长 1 段 → 相邻块共享下一段内容）。约束：
    - 只合并同页相邻段落，绝不跨 PDF 页面（page 元数据始终准确）；
    - 合并后超过 max_join 字符时退回单段，最后一段由回退分支补上，不丢失；
    - 超长段落内部按自然断点先切好（复用 split_text，不产生空块）；
    - 连续相同段落预合并，滑动窗口不会产出重复 chunk。
    每个 chunk 保留 source / page。
    """
    units: list[tuple[int, str]] = []
    for page_no, page_text in pages:
        for para in split_paragraphs(page_text):
            pieces = split_text(para, size, overlap=0) if len(para) > size else [para]
            for piece in pieces:
                units.append((page_no, piece))
    units = _merge_consecutive_duplicates(units)

    # 按页分组，每页内部跑滑动窗口——页边界处不合并、不产生多余单段块
    page_paras: dict[int, list[str]] = {}
    for page_no, piece in units:
        page_paras.setdefault(page_no, []).append(piece)

    chunks: list[Chunk] = []
    for page_no in page_paras:  # 保持 units 的页顺序（dict 按插入序）
        paras = page_paras[page_no]
        m = len(paras)
        if m == 0:
            continue
        if m == 1:
            chunks.append(Chunk(text=paras[0], source=source, page=page_no))
            continue
        for i in range(m - 1):
            joined = f"{paras[i]} {paras[i + 1]}"
            if len(joined) <= max_join:
                chunks.append(Chunk(text=joined, source=source, page=page_no))
                continue
            chunks.append(Chunk(text=paras[i], source=source, page=page_no))
            if i == m - 2:  # 最后一段被窗口吞并失败时补上，避免丢内容
                chunks.append(Chunk(text=paras[i + 1], source=source, page=page_no))
    return chunks


def load_document_paragraphs(
    path: Path, size: int = 500, max_join: int = 800
) -> list[Chunk]:
    """方案 C 正式入口：读一份文档，按「段落 + 相邻段重叠」切块。

    与 load_document 同签名风格（size 为超长段内部切分上限），
    每个 chunk 保留 source / page。新导入请用本函数。
    """
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        pages = read_pdf(path)
    elif suffix in (".txt", ".md"):
        pages = read_text_file(path)
    else:
        raise ValueError(f"暂不支持的文件类型：{suffix}（目前支持 .pdf .txt .md）")

    return chunk_paragraphs_overlap(pages, size=size, source=path.name, max_join=max_join)
