# AI Knowledge Server API

Base URL for local development: `http://127.0.0.1:8000`

Interactive OpenAPI documentation:

- Swagger UI: `GET /docs`
- ReDoc: `GET /redoc`
- OpenAPI schema: `GET /openapi.json`

Every handled HTTP response includes an `X-Request-ID` header for correlation with
server logs. The service currently has no public document-delete endpoint.

## Source object

`POST /query` and `POST /search` use the same source schema:

```json
{
  "source": "sample_handbook.md",
  "page": 1,
  "text": "The retrieved passage text...",
  "distance": 0.4123
}
```

| Field | Type | Meaning |
|---|---|---|
| `source` | string | Original uploaded filename |
| `page` | integer | Source page number |
| `text` | string | Full retrieved chunk text |
| `distance` | number | Original cosine vector distance; smaller means more similar |

The final source array is ordered by Cross-Encoder reranking. `distance` remains
the original vector distance, so it is not guaranteed to be ascending after reranking.

## `GET /health`

Returns service and knowledge-base status. The endpoint is available only after
the synchronous local model warm-up has completed and FastAPI startup has yielded.

### Request

No parameters or request body.

```bash
curl http://127.0.0.1:8000/health
```

### Successful response — `200`

```json
{
  "status": "ok",
  "chunks": 3,
  "sources": [
    "sample_handbook.md"
  ]
}
```

### Important errors

- `500`: unexpected Chroma/database failure; response body is
  `{"detail":"服务器内部错误"}` and the server log contains the request ID.

### Notes

This is a health/status endpoint, not a public readiness state machine. During
startup warm-up the application does not accept requests yet.

## `POST /query`

Runs the complete grounded RAG answer flow:

```text
original question
→ optional English query rewrite for Chinese/mixed input
→ vector retrieval
→ Cross-Encoder rerank
→ Top-K context
→ answer generation
→ cited answer and source chunks
```

Retrieval language and answer language are separate:

- Chinese question → Chinese answer
- English question → English answer
- Mixed question containing Chinese → Chinese answer
- Answer language is always determined from the **original question**
- A Chinese/mixed question may be rewritten to English only for retrieval

### Request body

```json
{
  "question": "实习期是多久？"
}
```

| Field | Type | Required | Validation |
|---|---|---|---|
| `question` | string | yes | Must not be empty or whitespace-only |

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"实习期是多久？"}'
```

### Successful response — `200`

```json
{
  "answer": "实习期为16周。[sample_handbook.md p.1]",
  "sources": [
    {
      "source": "sample_handbook.md",
      "page": 1,
      "text": "16 WEEKS OF COMPULSORY INTERNSHIP...",
      "distance": 0.4123
    }
  ]
}
```

The following are also successful `200` responses because they are valid RAG outcomes:

```json
{
  "answer": "I couldn't find the answer to this question in the available documents.",
  "sources": []
}
```

```json
{
  "answer": "资料中找不到这个问题的答案。",
  "sources": []
}
```

An empty knowledge base similarly returns a language-aware explanation and an
empty `sources` array.

### Important errors

| Status | Meaning |
|---|---|
| `422` | Missing, empty, whitespace-only, or invalid request body |
| `502` | LLM authentication, connection, response, or server failure |
| `503` | LLM rate limit |
| `500` | Unexpected internal error |

Error body:

```json
{"detail": "Error description"}
```

### Notes

- English questions normally use one LLM call for answer generation.
- Chinese/mixed questions normally use one rewrite call and one answer call.
- Rewrite failure falls back to the original query and does not itself fail `/query`.
- Answers are instructed to use only retrieved context and preserve citations.

## `POST /search`

Runs local vector retrieval, threshold filtering, and Cross-Encoder reranking.
It does not call the external LLM and does not perform query rewriting.

### Request body

```json
{
  "query": "working hours",
  "top_k": 5
}
```

| Field | Type | Required | Validation |
|---|---|---|---|
| `query` | string | yes | Must not be empty or whitespace-only |
| `top_k` | integer | no | Default `5`; allowed range `1`–`20` |

```bash
curl -X POST http://127.0.0.1:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query":"working hours","top_k":5}'
```

### Successful response — `200`

```json
[
  {
    "source": "sample_handbook.md",
    "page": 1,
    "text": "Monday – Friday 9am to 6pm...",
    "distance": 0.3912
  }
]
```

No reliable result returns `[]` with status `200`.

### Important errors

- `422`: invalid JSON, blank `query`, non-integer `top_k`, or `top_k` outside `1`–`20`.
- `500`: unexpected embedding, Chroma, or internal failure.

### Notes

The response order reflects reranking, while `distance` remains the vector distance.

## `POST /documents/upload`

Uploads one PDF, TXT, or Markdown document, extracts text, chunks it, and indexes
the chunks in the existing Chroma collection.

### Request

Content type: `multipart/form-data`

| Field | Type | Required | Notes |
|---|---|---|---|
| `file` | file | yes | Extensions: `.pdf`, `.txt`, `.md` |

```bash
curl -X POST http://127.0.0.1:8000/documents/upload \
  -F "file=@docs/example.pdf"
```

### Successful response — `200`

```json
{
  "filename": "example.pdf",
  "chunks": 12,
  "message": "已导入 12 块"
}
```

### Important errors

| Status | Meaning |
|---|---|
| `400` | Empty file, unsafe filename, unreadable file, or no extractable text |
| `413` | File exceeds `MAX_UPLOAD_BYTES` (default 10 MiB) |
| `415` | File extension is not `.pdf`, `.txt`, or `.md` |
| `422` | Multipart `file` field is missing or malformed |
| `500` | Unexpected internal indexing failure |

### Notes

- Upload is the only public mutation endpoint.
- Re-uploading the same filename replaces that source's existing chunks.
- Temporary upload files are removed after processing.
- The API does not currently expose document deletion.
- Uploading modifies Chroma data; `/health`, `/query`, and `/search` are read-only.
