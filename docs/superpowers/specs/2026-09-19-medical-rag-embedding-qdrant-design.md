# Medical RAG — Embedding and Qdrant Ingestion Design

Date: 2026-09-19
Status: approved in brainstorming (sections 1-4); written spec awaiting user review.
Scope: turn chunk dicts from `chunk_article` into points in a local Qdrant collection with dense and sparse (BM25) vectors. Retrieval, reranking and generation are separate sub-projects.

## Context and decisions

The chunking sub-project (`docs/superpowers/specs/2026-09-17-medical-rag-chunking-design.md`) produces chunk dicts with keys `id`, `type`, `article_slug`, `article_title`, `article_url`, `section_path`, `text`, `token_count`, `split_part`. This spec covers the next stage: embedding and storage.

Decisions made with the user:

| Decision | Choice | Why |
|---|---|---|
| Vector DB host | Qdrant **local on-disk mode** (`QdrantClient(path="qdrant_data")`) | No Docker or server on the dev machine; about 9k chunks is small; the client API is identical to server mode, so moving to a server later means changing one constructor. |
| Embedding compute | Call the existing **Kaggle `/embed` endpoint** through a Cloudflare tunnel | Local torch is broken and the GPU has 4 GB; the notebook `serve_model/serve_qwen3_kaggle.ipynb` already serves the model. |
| Sparse/hybrid design | **Approach A**: Qdrant-native hybrid, one shared pyvi segmentation pass feeding both vectors | One source of truth, server-side fusion later, no persisted vocabulary. Rejected: fastembed `Qdrant/bm25` (English-oriented defaults, black box) and in-memory `rank-bm25` (index outside Qdrant, hand-written fusion). |
| Repo layout | Monorepo with `backend/` and `frontend/` | The user will build a backend API and a frontend later. |

## Serving and network note (Cloudflare)

Embedding, reranker and LLM are **not** run locally. They run on Kaggle (T4 x2) from `serve_model/serve_qwen3_kaggle.ipynb` and are reached from the local machine through **Cloudflare quick tunnels**:

- Embed and rerank: FastAPI on port 8001, `POST /embed {"texts": [...]}` returns `{"embeddings": [[...768 floats...]]}`; `POST /rerank {"query", "documents"}` returns `{"scores": [...]}`.
- LLM: OpenAI-compatible vLLM on port 8000.
- Tunnel URLs (`*.trycloudflare.com`) change every Kaggle session and are unauthenticated. They are read from configuration, never hardcoded and never committed.

Consequences for this design:

- `EMBED_URL` (and later `LLM_URL`) come from environment variables, overridable with `--embed-url`.
- `EmbedClient` treats tunnel errors (`502`, `504`, `524`, `530`, `1033`) and connection errors or timeouts as transient: up to 3 retries with increasing backoff. `4xx` responses fail immediately.
- Cloudflare cuts requests after about 100 s (`524`). Each request has its own timeout below that, the default batch is 32 chunks, and on `524` the client halves the batch and retries.
- A startup health check calls `/embed` with one short text and verifies a 768-dimension vector. On failure it prints: "tunnel not responding, check the Kaggle notebook and update EMBED_URL".
- The `/rerank` endpoint and `LLM_URL` reuse the same configuration mechanism in later sub-projects.

## Repository layout

```
Medical_RAG/
  backend/
    pyproject.toml            # dependencies and pytest config; install with `pip install -e backend`
    medical_rag/
      ingestion/
        chunking.py           # moved from repo root (already merged)
        ingest.py             # new: CLI orchestration
      encoders.py             # new: segment, EmbedClient, Bm25Encoder (shared with retrieval later)
      store.py                # new: Qdrant collection and article replacement (shared with retrieval later)
    tests/                    # test_chunking.py moves here, plus new tests
    scripts/
      validate_chunking.py    # moved from repo root
      validate_ingest.py      # new: acceptance check
  frontend/                   # name reserved only; nothing is created until frontend work starts
  docs/                       # stays at the repo root
  data/ test_case/ testset/ vietmed_crawled/ qdrant_data/   # datasets and index, gitignored, stay at the root
```

- `ingestion/` holds code used only for loading data. `encoders.py` and `store.py` sit outside it because retrieval imports them. `retrieval/`, `generation/` and `api/` are added next to `ingestion/` when their specs land, not before.
- All commands run from the repository root so the relative paths `data/` and `test_case/` keep working.
- The first plan task moves the three existing files with `git mv`, changes imports (`from chunking import ...` becomes `from medical_rag.ingestion.chunking import ...`), and re-runs the 5 existing tests and `validate_chunking` to prove nothing broke. Old specs and plans under `docs/` are historical records and are not rewritten for the new paths.
- `qdrant_data/` is added to `.gitignore`. `serve_model/` is the user's untracked notebook and is left where it is.

## Components and interfaces

### `encoders.py`

- `segment(text: str) -> str` — `pyvi.ViTokenizer.tokenize`. The single word-segmentation function for both dense and sparse encoding, for documents and for queries. The model card of `dangvantuan/vietnamese-embedding` requires pyvi segmentation before encoding; the Kaggle server does not segment, so the client does.
- `embed_input(chunk: dict) -> str` — the part of a chunk sent to the encoders. Default: `chunk["text"]` unchanged (the chunking spec's decision). It is a separate function because the `Nguồn: https://...` line in the injected header may be noise for the embedding; changing that later is a one-line edit.
- `EmbedClient(base_url: str, batch_size: int = 32, timeout: float = 60.0, retries: int = 3)`
  - `embed(segmented_texts: list[str]) -> list[list[float]]` — `POST {base_url}/embed`, batches, retries as described above, raises on a wrong vector dimension.
  - `health_check() -> None` — raises a clear error if the tunnel is down or the dimension is not 768.
- `Bm25Encoder(k1: float = 1.2, b: float = 0.75)`
  - `fit(segmented_docs: list[str]) -> None` — computes `avgdl`; raises on an empty corpus.
  - `encode_doc(segmented: str) -> tuple[list[int], list[float]]` — `(indices, values)`.
  - `encode_query(segmented: str) -> tuple[list[int], list[float]]`.

### `store.py`

- `COLLECTION = "medical_rag"`, `DENSE_DIM = 768`.
- `open_client(path: str | None) -> QdrantClient` — `None` means `:memory:` (used by tests).
- `ensure_collection(client, recreate: bool = False) -> None` — creates the collection if missing; if it exists with a different dense size it raises and does not recreate; `recreate=True` drops and rebuilds.
- `replace_article(client, article_type: str, article_slug: str, points: list[PointStruct]) -> None` — deletes existing points matching `type` and `article_slug`, then upserts `points`.

### `ingestion/ingest.py`

- `ingest(data_dir: str, client, embedder: Callable[[list[str]], list[list[float]]], token_counter: Callable[[str], int], batch_size: int = 32) -> IngestReport`, where `IngestReport` holds `articles_ok: int`, `articles_failed: list[str]`, `points_upserted: int`. The embedder is injected so tests can pass a deterministic fake; the CLI passes `EmbedClient.embed`.
- `main()` CLI: `--data-dir` (default `data`), `--qdrant-path` (default `qdrant_data`), `--embed-url` (default `$EMBED_URL`), `--recreate`. Exit codes: `0` success; `1` at least one article failed; `2` configuration or health-check failure.

## Ingest flow

1. **Startup**: read configuration, run `health_check()`, open the client, `ensure_collection`.
2. **Pass 1 (CPU, local)**: `iter_articles` then `chunk_article` with the real tokenizer (transformers tokenizers only, no torch), then `segment(embed_input(chunk))` for every chunk, then `Bm25Encoder.fit` over all segmented documents. Everything stays in RAM (about 9.3k chunks; the full 1932-article crawl is still small).
3. **Pass 2 (per article)**: embed the article's segmented chunks in batches, build sparse vectors, build points, then `replace_article`. Embedding finishes **before** any deletion, so a failed embedding leaves the article's old points untouched. A failed article is recorded and processing continues.
4. **Finish**: print totals (articles ok/failed, points upserted) and compare the collection's point count with the expected chunk count. Re-running is idempotent (same ids, points overwritten).

Chunk ids are positional (`uuid5(ns, "type:slug:idx")`), so if an article's sections change, ids shift. This is why re-ingestion always replaces by `type` plus `article_slug` instead of relying on id overwrite alone.

Not built (upgrade path): skipping already-embedded articles on re-run. If Kaggle drops mid-run the whole run is repeated, which is acceptable at this scale.

## Qdrant collection `medical_rag`

- Point id: `chunk["id"]`.
- Vector `dense`: size 768, cosine (Qdrant normalizes; the client does not).
- Vector `sparse`: `SparseVectorParams(modifier=Modifier.IDF)`.
- Payload: all chunk fields (`type`, `article_slug`, `article_title`, `article_url`, `section_path`, `text`, `token_count`, `split_part`). Keyword payload indexes on `type` and `article_slug`.
- A point whose chunk has no tokens omits the `sparse` vector and keeps `dense`.

## Sparse BM25 encoder

- **Tokenization** of the segmented string: Unicode NFC normalization (composed and decomposed Vietnamese diacritics otherwise differ), lowercase, tokens by regex `\w+`. `_` is in `\w`, so a segmented compound such as `bệnh_nhân` stays one term. Digits are kept (dosages matter). No stopword removal: the IDF modifier already lowers common terms.
- **Index**: `zlib.crc32(token.encode("utf-8")) & 0xFFFFFFFF` (u32); no vocabulary is stored. With about 30k terms the expected number of colliding pairs is below one; a collision only merges two terms into one dimension. Repeated indices within one document have their term frequencies summed.
- **Document weight** (BM25 TF part): `w = tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))` with `k1 = 1.2`, `b = 0.75`; `dl` is the chunk's token count; `avgdl` comes from `fit`.
- **IDF** is applied by Qdrant at query time: `ln(1 + (N - n + 0.5) / (n + 0.5))`, the Lucene BM25 variant.
- **Query weight**: 1.0 per distinct query term; no `avgdl` needed, so the query encoder has no dependency on ingest state.
- **Limitation**: `avgdl` is correct only for the corpus of the ingest run that produced it. Ingesting a small subset later uses a slightly different `avgdl`; accepted.

## Testing and validation

### 1. pytest, offline (no Kaggle)

No mocks of Qdrant or HTTP.

- `test_encoders.py`: `segment` with real pyvi (compounds joined by `_`, deterministic); BM25 weights checked against hand-computed values, longer document gets a smaller weight for equal `tf`, hash stable, NFC-equivalent inputs equal, empty input; `EmbedClient` against a **real HTTP server on localhost** in a thread simulating success, one `524` then success (batch halves), repeated `530` (raises after 3 tries), `400` (no retry), wrong dimension (raises).
- `test_store.py`: Qdrant `:memory:` — collection creation, dense-size mismatch raises, `replace_article` replaces only that article's points, re-running does not duplicate. A rare term outranks a common term in a sparse query (proves `Modifier.IDF` works in local mode), and a hybrid `query_points` with two prefetches and RRF fusion runs.
- `test_ingest.py`: whole flow on a temporary data directory, a deterministic 768-dimension fake embedder and `:memory:` Qdrant — point counts, payload fields, one article's embedding failing leaves the others ingested and its old points intact, exit code non-zero.
- The 5 existing chunking tests and `validate_chunking` stay green after the move.

### 2. `validate_ingest.py` (acceptance; needs live Kaggle and real data)

Run after a real ingest.

- Invariants: point count equals the number of chunks `chunk_article` produces over `data/` (recomputed independently); dense dimension 768; no duplicate ids; every point has `dense` and has `sparse` unless its chunk is empty; per-`type` counts match; **the token count of every segmented `embed_input` is at most 512** (the chunk cap of 400 was measured on unsegmented text, and segmentation can change token counts, so this is checked directly with the local tokenizer and the distribution is reported).
- Smoke retrieval: a seeded sample of 30 questions per type from `test_case/*.csv`; embed each question (segment then `/embed`), then dense top-5 and sparse top-5 separately; report hit@5 at article level (`article_slug` matches) and at chunk level (normalized `context` is contained in one of the 5 chunks). This is a smoke test only; the full hybrid plus rerank evaluation belongs to the retrieval sub-project.
- **No pre-committed thresholds.** The chunking sub-project showed that a baseline estimated with a different method than the shipped system is meaningless. Floors are set only after the first real run, with a 3-point margin like `validate_chunking`.

### 3. Human-provided preconditions

Start the Kaggle notebook, copy the Cloudflare tunnel URL, `export EMBED_URL=...`, run `pip install -e backend`. The plan's acceptance task states this explicitly because an agent cannot do it.

### Done criteria

All tests green; real ingest of `data/` finishes with 0 failed articles; `validate_ingest` invariants pass; measured hit@5 numbers are recorded in this spec.

## Risks and limitations

- **Local-mode IDF support is unverified.** A grep shows local mode handles `Modifier`, `prefetch` and `SparseVector`, but no behaviour was run. The first plan task is a short spike test. Fallback if IDF is not applied: compute document frequency in `Bm25Encoder.fit` and bake IDF into the document weights on the client.
- **pyvi is an approximation** of the segmenter the model was trained with; embedding quality may differ slightly from the published numbers. It is the tool named in the model card and installs on Python 3.13 (a `cp313` Windows wheel exists for `python-crfsuite`).
- **Header noise**: `Nguồn: https://youmed.vn/...` is embedded and tokenized into BM25 terms in every chunk. The URL words have very low IDF, and slug words may even help, but it is unmeasured. `embed_input` is the seam to change it.
- **Local mode holds a file lock**: only one process can open `qdrant_data/` at a time, so ingestion and a running retrieval service cannot overlap. Moving to a server later removes this.
- **Tunnels are flaky and public**: quick tunnels have no SLA and no authentication; never commit URLs.
- **Positional chunk ids**: handled by replace-by-article on every ingest (see the ingest flow).

## Out of scope

- Search, hybrid fusion orchestration, reranking (`/rerank`) and top-k tuning: retrieval sub-project.
- LLM prompting and generation (`LLM_URL`): generation sub-project.
- Backend API and frontend: later sub-projects; only the folder layout is prepared here.
- Resume or skip of already-embedded articles.
- Parent-child (small-to-big) chunking, still deferred from the chunking spec.
