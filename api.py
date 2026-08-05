"""
FastAPI 接口：健康检查、文档上传、问答。

启动：
    uvicorn api:app --reload --host 0.0.0.0 --port 8000
"""

import logging
import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from ask import (
    LLMAuthError,
    LLMConnectionError,
    LLMRateLimitError,
    LLMServerError,
    ask,
)
from chunker import load_document_paragraphs
from store import add_chunks, delete_source, search as vector_search, stats

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
logger = logging.getLogger("ai_server")
if not logger.handlers:
    # 独立配置结构化请求日志（不干扰 uvicorn 自带日志与测试的 caplog）
    logger.setLevel(logging.INFO)
    _console = logging.StreamHandler()
    _console.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    logger.addHandler(_console)

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}
# 上传大小上限：默认 10 MB，可用环境变量 MAX_UPLOAD_BYTES 覆盖（单位：字节）
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))

# ---------------------------------------------------------------------------
# FastAPI 应用
# ---------------------------------------------------------------------------
app = FastAPI(
    title="AI Knowledge Server",
    description="上传文档建立知识库，提问时检索相关内容并生成带出处的答案。",
    version="0.1.0",
)


# ---------------------------------------------------------------------------
# 安全文件名
# ---------------------------------------------------------------------------
def secure_filename(filename: str) -> str:
    """
    从用户提供的文件名中提取安全的文件名部分。

    防御路径穿越：阻止 .. / \\ 和绝对路径。先检查原始输入再取最后一段。
    """
    raw = filename.strip()
    if not raw:
        raise HTTPException(status_code=400, detail="文件名不能为空。")

    # 先检查原始文件名中的路径穿越模式
    if ".." in raw or "/" in raw or "\\" in raw:
        raise HTTPException(
            status_code=400,
            detail="文件名包含非法字符（.. 或路径分隔符）。",
        )

    # 取最后一个路径组件（防御先归一化再检查的绕过）
    name = Path(filename).name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="文件名不能为空。")

    return name


# ---------------------------------------------------------------------------
# Pydantic 模型
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        description="用户问题",
        json_schema_extra={"example": "实习期是多久？"},
    )

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("问题不能为空或全为空白字符。")
        return stripped


class SearchRequest(BaseModel):
    query: str = Field(
        ...,
        description="搜索问题",
        json_schema_extra={"example": "实习期是多久？"},
    )
    top_k: int = Field(
        5,
        ge=1,
        le=20,
        description="返回结果数量，范围 1-20",
    )

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("query 不能为空或全为空白字符。")
        return stripped


class SourceInfo(BaseModel):
    source: str = Field(..., description="文件名")
    page: int = Field(..., description="页码")
    text: str = Field(..., description="检索到的段落全文")
    distance: float = Field(..., description="向量距离（cosine），越小越相关")


class QueryResponse(BaseModel):
    answer: str = Field(..., description="LLM 生成的答案，每句带 [文件名 p.页码] 出处")
    sources: list[SourceInfo] = Field(default_factory=list, description="检索到的来源段落")


class UploadResponse(BaseModel):
    filename: str = Field(..., description="上传的文件名")
    chunks: int = Field(..., description="切出的块数")
    message: str = Field(..., description="导入结果说明")


class HealthResponse(BaseModel):
    status: str
    chunks: int
    sources: list[str]


class ErrorDetail(BaseModel):
    detail: str


# ---------------------------------------------------------------------------
# 结构化请求日志 + request_id
# ---------------------------------------------------------------------------


@app.middleware("http")
async def request_logging(request: Request, call_next):
    """为每个请求生成 request_id（响应头 X-Request-ID 返回），
    记录方法/路径/状态码/耗时。不记录请求体（不含文档内容与敏感信息）。"""
    request_id = uuid.uuid4().hex[:12]
    request.state.request_id = request_id
    start = time.perf_counter()

    response = await call_next(request)

    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "request_id=%s method=%s path=%s status=%d duration_ms=%.1f",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    response.headers["X-Request-ID"] = request_id
    return response


# ---------------------------------------------------------------------------
# 异常处理
# ---------------------------------------------------------------------------


@app.exception_handler(LLMAuthError)
async def llm_auth_handler(request: Request, exc: LLMAuthError):
    return JSONResponse(status_code=502, content={"detail": str(exc)})


@app.exception_handler(LLMRateLimitError)
async def llm_ratelimit_handler(request: Request, exc: LLMRateLimitError):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(LLMServerError)
async def llm_server_handler(request: Request, exc: LLMServerError):
    return JSONResponse(status_code=502, content={"detail": str(exc)})


@app.exception_handler(LLMConnectionError)
async def llm_connection_handler(request: Request, exc: LLMConnectionError):
    return JSONResponse(status_code=502, content={"detail": str(exc)})


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    # 完整堆栈只进服务端日志（含 request_id 可关联）；响应不暴露堆栈与本地路径
    logger.exception(
        "未处理异常 request_id=%s method=%s path=%s",
        getattr(request.state, "request_id", "-"),
        request.method,
        request.url.path,
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "服务器内部错误"},
    )


# ---------------------------------------------------------------------------
# 接口
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
def health():
    """检查服务状态，返回向量库中存储的块数和文件列表。"""
    s = stats()
    return HealthResponse(
        status="ok",
        chunks=s["chunks"],
        sources=s["sources"],
    )


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """
    提交一个问题，返回带出处的答案和检索到的来源段落。

    如果没有检索到可靠段落（低于相关性阈值），不会调用 LLM，
    直接返回 "资料中找不到这个问题的答案。"。

    错误码：
    - 422: 问题为空或全为空白字符
    - 502: LLM 认证失败 / 服务异常 / 连接错误
    - 503: LLM 频率限制
    """
    result = ask(req.question)

    sources = [
        SourceInfo(
            source=s["source"],
            page=s["page"],
            text=s["text"],
            distance=s["distance"],
        )
        for s in result["sources"]
    ]

    return QueryResponse(answer=result["answer"], sources=sources)


@app.post("/search", response_model=list[SourceInfo])
def search(req: SearchRequest):
    """
    只做向量检索，不调用 DeepSeek（不产生任何 API 费用）。

    返回按 distance 升序排列的段落数组（越小越相关）；
    低于相关性阈值的结果被过滤，没有命中时返回空数组。

    错误码：
    - 422: query 为空或全为空白字符 / top_k 不在 1-20 范围
    """
    hits = vector_search(req.query, top_k=req.top_k)
    return [
        SourceInfo(
            source=h["source"],
            page=h["page"],
            text=h["text"],
            distance=h["distance"],
        )
        for h in hits
    ]


@app.post("/documents/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    上传 PDF / TXT / MD 文件，切块后存入向量库。

    支持格式：.pdf  .txt  .md
    大小限制：32 MB（边读边检查，超限立即返回 413 不完整写入）

    同名文件再上传会先删除旧 chunks 再入库，保证不残留。

    错误码：
    - 400: 不支持的类型 / 路径穿越 / 扫描版 PDF / 空文件
    - 413: 文件超过 32 MB
    """
    # --- 安全文件名 -----------------------------------------------------------
    raw_filename = file.filename or "unknown"
    filename = secure_filename(raw_filename)
    suffix = Path(filename).suffix.lower()

    # --- 校验扩展名 -----------------------------------------------------------
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"不支持的文件类型「{suffix}」，目前支持：{' '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )

    # --- 流式读取并写入临时文件，边读边检查大小 --------------------------------
    tmp_dir = tempfile.mkdtemp()
    tmp_path = Path(tmp_dir) / filename

    try:
        total_read = 0
        with open(tmp_path, "wb") as f:
            while True:
                chunk = await file.read(8192)  # 8 KB per chunk
                if not chunk:
                    break
                total_read += len(chunk)
                if total_read > MAX_UPLOAD_BYTES:
                    # 立即停止，不写后面的内容
                    raise HTTPException(
                        status_code=413,
                        detail=f"文件过大，最大允许 {MAX_UPLOAD_BYTES // 1024 // 1024} MB。",
                    )
                f.write(chunk)

        if total_read == 0:
            raise HTTPException(status_code=400, detail="上传的文件为空。")

    except HTTPException:
        # 清理并继续抛出
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail="文件读取失败，请重试。")

    try:
        # --- 切块（方案C：自然段落 + 相邻段重叠）-------------------------------
        chunks = load_document_paragraphs(tmp_path)
        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="没从文件里提取到文字（可能是扫描版 PDF 或空文件）。",
            )

        # --- 入库（先删旧数据，防止残留）--------------------------------------
        delete_source(filename)
        n = add_chunks(chunks)
    finally:
        # 清理临时目录
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return UploadResponse(
        filename=filename,
        chunks=n,
        message=f"已导入 {n} 块",
    )
