# AI Knowledge Server

## What it does

AI Knowledge Server is a small FastAPI RAG service with a Next.js MAX chat UI.
It indexes PDF/TXT/Markdown documents, retrieves relevant passages, reranks them
locally, and asks an LLM to produce a grounded answer with source citations.

Chinese questions receive Chinese answers, English questions receive English
answers, and mixed questions containing Chinese receive Chinese answers. Chinese
and mixed questions may be rewritten to English for retrieval only.

## Architecture

```text
Documents
→ paragraph-aware chunking
→ all-MiniLM-L6-v2 embedding
→ ChromaDB
→ vector retrieval and threshold filtering
→ Cross-Encoder rerank
→ DeepSeek-compatible LLM
→ cited answer and source chunks
```

Langfuse observes the request pipeline; it is not part of retrieval or ranking.

## Project structure

| Path | Purpose |
|---|---|
| `api.py` | FastAPI app, schemas, upload handling, errors, startup warm-up |
| `ask.py` | Query rewrite, retrieval orchestration, grounded answer generation |
| `store.py` | ChromaDB, embedding, vector search, stats, vector warm-up |
| `reranker.py` | Cross-Encoder singleton, reranking, vector-only fallback |
| `chunker.py` | PDF/TXT/MD parsing and paragraph-aware chunks |
| `ingest.py` | Command-line document ingestion and knowledge-base stats |
| `frontend/` | Next.js + TypeScript App Router frontend |
| `evaluation/scripts/` | Offline evaluation runners and experiments |
| `evaluation/data/` | Evaluation question sets and rewrite cache |
| `evaluation/reports/` | Generated evaluation results and reports |
| `tests/` | Python test suite |
| `data/` | Local ChromaDB stores (ignored by Git) |
| `docs/API.md` | Developer-facing HTTP API documentation |
| `evaluation/scripts/eval_acceptance.py` | End-to-end `/query` acceptance runner |
| `evaluation/data/eval_acceptance_questions.example.json` | Fictional example evaluation questions |
| `sample_docs/` | Fictional document for trying the public repository |

This public repository contains no real appointment letters, private evaluation
questions/results, Chroma databases, or model weights. Local experiments are not
part of the published snapshot.

## Setup

Python 3.12 is used by the Docker image; use a current supported Python locally.

```powershell
py -m venv .venv
& .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` in the project root and set your own keys. Do not
commit `.env` or real keys.

```env
DEEPSEEK_API_KEY=your_key_here
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-v4-pro

LANGFUSE_PUBLIC_KEY=your_public_key
LANGFUSE_SECRET_KEY=your_secret_key
LANGFUSE_BASE_URL=https://cloud.langfuse.com
LANGFUSE_TRACING_ENVIRONMENT=development

MAX_UPLOAD_BYTES=10485760
```

`LANGFUSE_TRACING_ENABLED=false` disables trace export when needed. `HF_HOME` may
be set to choose the local Hugging Face model cache directory.

## Add documents

Upload through `POST /documents/upload`, or place your own files in `docs/` and use:

```powershell
python ingest.py docs\
python ingest.py --stats
```

For a first run with fictional data, use `python ingest.py sample_docs\`.

Supported upload formats are `.pdf`, `.txt`, and `.md`. Uploading writes to the
knowledge base; `/health`, `/query`, `/search`, and the acceptance runner are read-only.

## Start backend

```powershell
python -m uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

Open `http://127.0.0.1:8000/docs` for Swagger UI.

Startup synchronously warms the local embedding model, first embedding encode,
existing Chroma/HNSW query when available, and Cross-Encoder inference. The service
does not become ready until warm-up completes. A cold start can take tens of seconds,
depending on model cache, disk, CPU, and network availability.

Warm-up is read-only, never calls the external LLM, and does not create normal
Langfuse RAG traces. If essential vector warm-up fails, startup fails; reranker
failure retains the existing vector-only fallback behavior.

## Start frontend

```powershell
cd frontend
npm install
Copy-Item .env.local.example .env.local
npm run dev
```

If PowerShell blocks `npm.ps1`, use:

```powershell
npm.cmd run dev
```

Set the server-only FastAPI URL in `frontend/.env.local`:

```env
FASTAPI_URL=http://localhost:8000
```

Open `http://localhost:3000`. The browser posts questions to the Next.js Route
Handler at `/api/chat`; that server-side handler validates the question and forwards
it to the existing FastAPI `POST /query` endpoint. `FASTAPI_URL` is not included in
the browser bundle.

For LAN testing, start Next.js on all interfaces:

```powershell
npm.cmd run dev -- --host 0.0.0.0
```

Another device on the same LAN can then open:

```text
http://<HOST_LAN_IP>:3000
```

LAN access is not public internet exposure; firewall and network rules still apply.

## API documentation

See [docs/API.md](docs/API.md) for every current endpoint, real request/response
schema, examples, and error behavior.

Current public endpoints:

- `GET /health`
- `POST /query`
- `POST /search`
- `POST /documents/upload`

There is currently no public document-delete endpoint.

## Langfuse

Normal `/query` traces retain this structure:

```text
rag-query
├── knowledge-base-stats
├── query-rewrite        # Chinese/mixed questions only
├── retrieval
│   ├── vector-search
│   └── rerank
└── answer-generation
```

Useful scalar metadata includes configured model, response language, rewrite
occurrence, candidate/result counts, outcome, and reranker fallback state. No API
keys, Authorization headers, or `.env` contents are added to metadata. Existing
observations may contain user questions, retrieved chunks, and final prompts, so use
appropriate Langfuse access controls and retention for private company documents.

Startup warm-up uses standard Python logging only and creates no user RAG trace.

## Acceptance evaluation

Validate the fictional example question file without calling the API or LLM:

```powershell
python -m evaluation.scripts.eval_acceptance --questions evaluation/data/eval_acceptance_questions.example.json --validate-only
```

After importing `sample_docs/` and starting the backend, optionally run the example
evaluation. This makes real `/query` calls to the configured LLM and may incur cost:

```powershell
python -m evaluation.scripts.eval_acceptance --questions evaluation/data/eval_acceptance_questions.example.json --base-url http://127.0.0.1:8000
```

To use your own private question set, pass its path with `--questions`; do not commit
private questions or generated results. The runner writes:

- `evaluation/reports/eval_acceptance_results.json` — machine-readable per-question results
- `evaluation/reports/eval_acceptance_report.md` — failures first, then totals and latency summary

The runner checks answer keywords, expected source filenames, refusal behavior,
average latency, P50, and P95. It never changes threshold, Top-K, prompt, or RAG data.

## Tests

Run the published, self-contained tests:

```powershell
python -m pytest -q tests/test_api.py tests/test_reranker.py tests/test_reliability.py tests/test_docker_config.py tests/test_eval_acceptance.py
```

If Windows denies access to pytest's shared temporary directory, use a dedicated
directory inside the project:

```powershell
python -m pytest -q --basetemp=.pytest-tmp tests/test_api.py tests/test_reranker.py tests/test_reliability.py tests/test_docker_config.py tests/test_eval_acceptance.py
```

Frontend production build:

```powershell
cd frontend
npm run build
```

## Docker

The repository includes `Dockerfile`, `docker-compose.yml`, and static Docker
configuration tests. Compose binds the API to `127.0.0.1:8000`, loads `.env` at
runtime, persists Chroma/model cache in named volumes, and includes a healthcheck.

```powershell
docker compose up -d --build
docker compose logs -f
docker compose down
```

Do not use `docker compose down -v` unless intentionally deleting the Docker-managed
Chroma and model-cache volumes. Docker runtime behavior has not been verified on the
current machine; the checked-in configuration is covered by static tests.

## Production notes

This repository is a local demo system. The Next.js Route Handler expects the
FastAPI service configured by `FASTAPI_URL`; deployment and a managed gateway are
not included. A future deployment may add authentication, TLS, an internal LLM,
and managed infrastructure. Do not expose local development ports to the public
internet.
