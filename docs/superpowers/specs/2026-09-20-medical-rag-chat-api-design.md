# Medical RAG — Chat API and UI Design

Scope: connect the mockup UI (`UI_template/medical_rag_ui.html`) to the backend. Add hybrid retrieval with optional reranking, prompt building and generation, an HTTP API, and a UI that renders the answer and its sources dynamically. Single-turn only.

## Decisions

- **Reranking is optional.** It runs only when `RERANK_URL` is set. The reranker lives on Kaggle behind an ephemeral tunnel, so the app must work without it. If it is unset or a call fails, results keep the Qdrant fusion order and `score` is `null`.
- **One process serves API and UI.** FastAPI serves `index.html` at `/` and the API under `/api`. Same origin, so no CORS.
- **Single-turn chat.** Each question is independent. Multi-turn is future work (see below).

## Architecture

```
backend/medical_rag/
  retrieval.py     # new: Hit, RerankClient, Retriever
  generation.py    # new: build_messages, answer
  api.py           # new: FastAPI app
  static/index.html  # moved from UI_template/medical_rag_ui.html
  encoders.py      # EmbedClient.embed gains a `task` argument
```

### retrieval.py

- `Hit` is a dataclass: `id`, `text`, `type`, `article_title`, `article_url`, `section_path`, `score: float | None`.
- `Retriever(client, embed, rerank=None).search(question, k=5) -> list[Hit]`. `embed` is `(texts, task="document") -> vectors` (production: `EmbedClient.embed`); `rerank` is `(query, documents) -> list[float]` (production: `RerankClient.rerank`), or `None`. Callables, like `ingest`'s `embedder`, keep the tests free of HTTP:
  1. `segment(question)`, then `embed([segmented], task="query")`.
  2. Sparse vector from `Bm25Encoder().encode_query(segmented)`. `encode_query` needs no `fit`.
  3. `client.query_points` with two `prefetch` entries (dense and sparse, limit 20 each) and `FusionQuery(Fusion.RRF)`. If the sparse query has no indices, only the dense prefetch is sent.
  4. Without a reranker: return the first `k` points with `score=None`. The RRF score is a rank-fusion value, not a relevance a user can read, so it is not shown.
  5. With a reranker: fuse to 20 candidates, call `/rerank` with the original question and each hit's `text`, sort by score, keep `k`. `score` is the raw reranker logit (unbounded, may be negative), so the UI shows it with 2 decimals and no percent. On `RerankError` log a warning and fall back to step 4.
- `RerankClient(base_url, timeout)` posts `{"query", "documents"}` to `/rerank` and returns `list[float]`. It validates the length and raises `RerankError` on any HTTP, network or shape problem. No retry loop: a failure falls back straight away.

### encoders.py

`EmbedClient.embed(segmented_texts, task="document")`. The `task` field is added to the request body only when it is `"query"`. Ingestion and the existing tests send the same body as before. The local `embed_server.py` reads `task`; the Kaggle server ignores it.

### generation.py

- `build_messages(question, hits) -> list[dict]`. The system prompt is in Vietnamese: answer only from the numbered context, cite sources as `[n]`, say so when the context is not enough, no diagnosis. The user message lists each hit as `[n] <article_title> — <section_path>` followed by its `text`, then the question.
- `answer(retriever, question) -> dict` returns `{"answer": str, "sources": [...]}`. It calls `retriever.search`, `llm.chat(build_messages(...))`, and shapes each hit into `{n, title, url, section_path, type, text, score}` with `n` starting at 1.
- With no hits, it skips the LLM and returns a fixed "no information found" answer with `sources=[]`.

### api.py

- Lifespan: read env, open Qdrant at `QDRANT_PATH` (default `qdrant_data`), fail with a clear message if the collection does not exist. Build `EmbedClient(EMBED_URL)` and, if `RERANK_URL` is set, `RerankClient`. `EMBED_URL` is required.
- `POST /api/chat`, body `{"question": str}` (1 to 1000 characters after strip), returns the `answer` result above.
- `GET /api/health` returns `{"status": "ok", "points": <count>}`.
- `GET /` returns `static/index.html`.
- Errors: `EmbedError` and LLM failures (`RuntimeError`, `openai.OpenAIError`) become HTTP 502 with a short `detail`. The `question` length rule gives 422 through pydantic.
- Handlers are plain `def`, so the blocking clients run in FastAPI's threadpool.
- Run from `backend/`: `uvicorn medical_rag.api:app --env-file ../.env`.

Environment: `EMBED_URL` (required), `OPENROUTER_API_KEY`, `LLM_MODEL`, `QDRANT_PATH`, `RERANK_URL` (optional).

## Frontend

The file moves to `backend/medical_rag/static/index.html`. The hard-coded conversation and source cards are removed.

- `sendMessage`: the user bubble is built with `textContent` (the mockup interpolated raw input into `innerHTML`, an XSS hole). Show a pending assistant bubble, disable the send button, `fetch('/api/chat')`.
- Answer rendering: escape HTML, keep line breaks (`white-space: pre-wrap`), turn `[n]` into a citation chip only when `1 <= n <= sources.length`. Errors render as an error bubble and re-enable the button.
- Sources panel: title as a link to `url` (new tab), a `type` badge, `section_path`, and the first 300 characters of `text`. The "Reranker score" row shows only when `score != null`. The mockup's PDF and page fields are dropped because the data is web articles.
- Status pill: on load call `/api/health`; show the chunk count, or "Offline" on failure.
- Sidebar stays a mockup. "New Chat" clears the messages and sources.

## Testing

- `test_retrieval.py`: in-memory Qdrant built with `ingest` and the fake embedder (same pattern as `test_validate_ingest.py`). Cases: hybrid returns relevant points, rerank reorders, rerank failure falls back to fusion order, empty sparse query still works.
- `test_encoders.py`: `task="query"` reaches the request body, the default does not add the field.
- `test_generation.py`: the prompt contains every numbered source and the question; `answer` shapes `sources`; the empty-hits path does not call the LLM. The LLM is faked.
- `test_api.py`: FastAPI `TestClient` with fake retriever and fake `chat`. Cases: 200 shape, 422 on empty question, 502 on embed failure, `/api/health`.
- The frontend is checked by running the app against the real collection and asking a question. No JS tests.

## Dependencies

Add `fastapi` and `uvicorn` to `dependencies`, `httpx` to the `dev` extra (needed by `TestClient`).

## Out of scope / future work

- **Multi-turn chat.** The client sends the history with each request; the server rewrites a follow-up into a standalone question with one LLM call before retrieval; the sidebar persists conversations.
- Streaming answers (SSE).
- Top-k tuning and a full hybrid + rerank evaluation.
- Qdrant local mode holds a file lock, so ingestion and the API cannot run at the same time. A Qdrant server removes this.
