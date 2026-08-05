# Enterprise AI Knowledge Server

上传文档建立知识库，提问时检索相关内容并生成带出处的答案。

当前版本：命令行问答、FastAPI 接口、自动化测试、中文查询自动改写都已就绪；
Docker Compose 尚未添加。

## 跑起来

```bash
py -m venv .venv
.venv\Scripts\activate
py -m pip install -r requirements.txt
```

装 sentence-transformers 会顺带拉 PyTorch，包比较大，第一次要等几分钟。

在项目根目录创建 `.env`，把 DeepSeek 的 key 填进去：

```env
DEEPSEEK_API_KEY=your_api_key_here
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-v4-pro
```

`.env` 已经加入 `.gitignore`，不要把真实 API key 提交到 Git 仓库。

## 用法

把要问的文档放进 `docs/`，然后导入：

```bash
py ingest.py docs/
```

第一次运行会自动下载 embedding 模型（约 80MB），之后就有缓存了。

看看库里有什么：

```bash
py ingest.py --stats
```

提问：

```bash
py ask.py "实习期是多久？"
```

会打印答案，答案里每句话后面带 `[文件名 p.页码]` 的出处，
下面再列出实际检索到的段落和距离，方便你判断检索准不准。

## 各文件在做什么

| 文件 | 职责 |
|---|---|
| `chunker.py` | 读 PDF/TXT/MD，切成块，每块记住来自哪个文件第几页；含两套切块（旧字符窗口 + 方案C段落重叠） |
| `store.py` | 向量库这一层：存进去、查出来、按文件删除 |
| `ingest.py` | 导入文档的命令行入口（方案C切块） |
| `ask.py` | 检索 + 中文问题自动英文改写 + 调 DeepSeek 生成答案 |
| `api.py` | FastAPI 接口：`/health`、`/query`、`/search`、`/documents/upload` |
| `test_api.py` | 接口自动化测试，`pytest test_api.py -v` 运行 |
| `test_chunker.py` | 方案C切块边界测试（空文本/单段/多段/超长段/换页/source/page/去重） |
| `eval_questions.py` | 评估测试集：40 题（15 英文 / 15 中文 / 10 中英混合）+ 标准答案 + 人工英文改写；另含失败类型补充题 `EXTRA_QUESTIONS`（10 题） |
| `eval_retrieval.py` | 评估脚本：临时向量库对比各方案命中率/耗时/内存，输出失败案例 |
| `eval_rewrite.py` | 离线生成 DeepSeek 自动改写并缓存（仅供评估，不进入正式服务） |
| `eval_chunking.py` | 切块策略对比实验（方案A/B/C，50 题）；`--db` 模式直接评估已建候选库 |
| `build_v2.py` | 用方案C重建 `chroma_data_v2/`（旧库 `chroma_data/` 保留为回滚库） |
| `smoke_v2.py` | 候选库真实冒烟：`/search`、`/query` 等价流程（英/中各一题） |

## 两个设计上的取舍

**为什么 embedding 用本地模型而不是 API**

DeepSeek 没有 embeddings 接口，所以向量化只能本地做。
用 all-MiniLM-L6-v2：CPU 能跑、免费、80MB。
第二周做 ONNX 时正好可以把它导出成 ONNX 格式来练手，前后能接上。

**为什么切块时就要存页码**

如果切块阶段没把 source 和 page 带上，后面想让答案标注出处就没辙了，
只能整个重来。所以 `Chunk` 从一开始就带这两个字段。

## 系统流程

```
中文/中英混合问题：
  用户问题（中文）
    → [DeepSeek] 自动改写成英文检索查询（只翻译，不回答问题、不补充信息）
    → 向量检索（all-MiniLM-L6-v2，本地，零 API 费用）
    → [DeepSeek] 用原始问题生成答案（语言跟随提问语言，带出处）

英文问题：
  用户问题（英文）
    → 向量检索（all-MiniLM-L6-v2，本地，零 API 费用）
    → [DeepSeek] 生成答案（带出处）

POST /search（独立检索接口）：
  任意语言查询 → 向量检索 → 直接返回段落数组（不调用 DeepSeek，零费用）
```

## API 接口（uvicorn api:app）

### 启动

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

启动后打开 http://localhost:8000/docs 可以看 Swagger 文档和在线调试。

### GET /health

检查服务状态：

```bash
curl -s http://localhost:8000/health | python -m json.tool
```

```json
{
  "status": "ok",
  "chunks": 4,
  "sources": ["university letter concerning d internship.pdf"]
}
```

### POST /query

提交问题，返回带出处的答案和检索段落：

```bash
curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "实习期是多久？"}' | python -m json.tool
```

```json
{
  "answer": "实习期为16周…[university letter concerning d internship.pdf p.1]",
  "sources": [
    {
      "source": "university letter concerning d internship.pdf",
      "page": 1,
      "text": "…",
      "distance": 0.751
    }
  ]
}
```

**错误情况（均返回 200，通过 answer 字段区分）：**

- 知识库为空时 answer = `"知识库是空的，请先用 POST /documents/upload 上传文档。"`
- 没有段落通过相关性阈值时 answer = `"资料中找不到这个问题的答案。"`

**DeepSeek API 调用次数（费用提示）：**

- **中文/中英混合问题：通常 2 次调用**——第 1 次把问题改写成英文检索查询（只翻译，不回答问题、不补充信息），第 2 次生成答案
- **英文问题：通常 1 次调用**（只生成答案，不改写）
- 改写失败、超时或返回空时，自动回退用原问题检索，`/query` 不会报错，此时中文问题只产生 1 次调用
- 最终答案始终使用提问语言：中文提问返回中文答案（改写只用于检索，不改变回答语言）

**其他错误码：**

| 状态码 | 场景 |
|--------|------|
| 422 | 问题为空或全为空白字符 |
| 502 | DeepSeek API Key 无效 / 服务异常 / 连接错误 |
| 503 | DeepSeek 频率限制 |

### POST /search

只做向量检索，**不调用 DeepSeek，不产生任何 API 费用**。
即使 `query` 是中文也不做改写、不调用任何 DeepSeek 函数，
适合前端先拉取相关段落做预览或二次过滤：

```bash
curl -s -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "实习期是多久？", "top_k": 5}' | python -m json.tool
```

```json
[
  {
    "source": "university letter concerning d internship.pdf",
    "page": 1,
    "text": "…",
    "distance": 0.751
  }
]
```

- `query`：必填，去除空白后为空返回 422
- `top_k`：可选，缺省 5，范围 1–20，超出返回 422
- 结果按 `distance` 升序（越小越相关），低于相关性阈值（0.85）的段落会被过滤
- 没有任何命中时返回空数组 `[]`

### POST /documents/upload

上传并索引文档：

```bash
curl -s -X POST http://localhost:8000/documents/upload \
  -F "file=@docs/你的文件.pdf" | python -m json.tool
```

```json
{
  "filename": "你的文件.pdf",
  "chunks": 12,
  "message": "已导入 12 块"
}
```

**错误码：**

| 状态码 | 场景 |
|--------|------|
| 400 | 不支持的文件类型 / 扫描版 PDF 提取不到文字 / 空文件 |
| 413 | 文件超过 32 MB |
| 422 | 缺少 file 参数 |
| 500 | 服务器内部错误 |

## 测试

```bash
pytest test_api.py test_eval.py test_chunking.py test_chunker.py
```

最终结果：**130 passed**（44 个接口/改写测试 + 35 个评估脚本测试 +
25 个切块实验测试 + 26 个方案C边界测试）。

真实冒烟测试结果（真实 DeepSeek + 真实向量库）：

| 场景 | 结果 | DeepSeek 调用次数 |
|---|---|---|
| 英文 `POST /query`（"How long is the internship?"） | 200，英文答案带出处：*The internship is 16 weeks long, from 14 September 2026 to 8 January 2027* [university letter concerning d internship.pdf p.1] | 1 次（只回答） |
| 中文 `POST /query`（"实习期是多久？"） | 200，中文答案带出处：*实习期为16周，从2026年9月14日开始至2027年1月8日结束* [university letter concerning d internship.pdf p.1] | 2 次（改写 + 回答） |
| 中文 `POST /search` | 200，毫秒级返回段落 | 0 次 |

回答语言严格跟随提问语言：英文提问必返回英文，中文提问必返回中文
（回答阶段的提示词按中文/英文/其他语言分别写明强制规则，改写只用于检索，
不改变回答语言）。

## 切块策略：方案 C（段落 + 相邻段重叠）已正式启用

2026-08 切块实验（50 题：40 基线 + 10 失败类型补充）对比了三种策略：

| 方案 | Top-1 | Top-3 | chunk 数 |
|---|---|---|---|
| A 旧策略（500字符窗口+断点+50重叠） | 56% | 68% | 10 |
| B 页面自然段落 | 58% | 76% | 17 |
| **C 段落 + 相邻段重叠（已启用）** | **62%** | **80%** | **15** |

**2026-08 正式启用**：
- 上传流程（`/documents/upload` 与 `ingest.py`）统一走方案 C
  （`chunker.load_document_paragraphs`：自然段落切块 + 相邻段重叠，
  不跨 PDF 页面、保留 source/page）；
- 正式数据库切换为 `store.py` 的 `DB_DIR = chroma_data_v2/`（15 块：6+8+1）；
- 对 v2 库重跑 50 题评估与离线实验逐项一致（62% / 80%）；
  真实冒烟（`/health`、上传临时文档、`/search`、`/query` 英中）全部通过，
  新上传文档确认写入 v2、旧库未受影响。

**回滚方式**：把 `store.py` 的 `DB_DIR` 改回 `chroma_data`（旧库从未被
覆盖/改名/删除，仍为 10 块旧切块数据），重启 uvicorn 即可回滚。
注意：回滚后新上传仍按方案 C 切块（切块入口已换），如需完整回滚到旧切块，
还需把 `api.py` / `ingest.py` 的 `load_document_paragraphs` 换回 `load_document`。

## 进度

- [x] 包成 FastAPI：`GET /health`、`POST /query`、`POST /search`、`POST /documents/upload`
- [x] 加自动化测试（130 个测试全部通过，见「测试」）
- [x] 自动中文查询改写实验（40 题评估集：15 英文 / 15 中文 / 10 中英混合；
      对比原始查询、多语言模型、人工改写、DeepSeek 自动改写）
- [x] 自动中文查询改写接入 `/query`（中文问题先英文改写再检索，
      答案仍用原语言；改写失败自动回退原问题）
- [x] 切块策略实验并选定方案 C（段落+相邻段重叠）：正式函数入 `chunker.py`，
      候选库 `chroma_data_v2/` 已建并通过 50 题复现与真实冒烟
- [x] 正式启用方案 C 与 `chroma_data_v2/`：上传流程（API + ingest）统一走
      `load_document_paragraphs`，`store.py` 的 `DB_DIR` 已切换；
      旧库 `chroma_data/` 保留为回滚库（见「切块策略」回滚方式）
- [ ] Docker Compose 起 API + ChromaDB

## 第二周补充计划：多语言 Embedding 与 ONNX

- [x] 准备一组中文、英文和中英混合的测试问题，并标记正确来源（40 题，见 `eval_questions.py`）
- [x] 对比当前 `all-MiniLM-L6-v2` 与多语言模型（优先尝试 `paraphrase-multilingual-MiniLM-L12-v2`）
- [x] 记录各模型的检索命中率、响应时间和内存占用（`eval_retrieval.py`，含人工/自动改写的方案对比）
- [x] 选择模型方案：保留 `all-MiniLM-L6-v2`（多语言模型英文回归明显且无统计优势），
      改用「中文查询自动改写」提升中文检索（见「系统流程」）
- [ ] 导出 `all-MiniLM-L6-v2` 为 ONNX 并在 ONNX Runtime 上运行，比较转换前后的速度和命中率
