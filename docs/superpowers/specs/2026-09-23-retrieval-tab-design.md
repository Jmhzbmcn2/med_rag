# Retrieval Tab Design

## Goal

Add a retrieval-only tab where a user can submit a medical query, choose dense, sparse, or hybrid retrieval, and inspect every returned chunk's raw retrieval score separately from its reranker score. This flow does not call the LLM or persist query history.

## Scope

- Add `Chat` and `Retrieval` navigation to the existing header.
- Serve the retrieval UI at `/retrieval`; keep the current chat at `/`.
- Add a retrieval-only API endpoint and reuse the existing `Retriever`.
- Support `dense`, `sparse`, and `hybrid` modes.
- Return 10 results by default and accept `k` from 1 through 20.
- Show chunk rank, article title and URL, section path, article type, text, raw retrieval score, and reranker score when available.

The feature does not add filters, pagination, query history, side-by-side mode comparison, configurable fusion, or new dependencies.

## Backend Design

### Retrieval modes

Extend `Retriever.search` to accept `mode`, defaulting to `"hybrid"` so the chat path keeps its current behavior.

- `dense`: segment and embed the question, then query Qdrant's named `dense` vector.
- `sparse`: segment and BM25-encode the question, then query Qdrant's named `sparse` vector. Do not call the embedding service. If tokenization produces no sparse terms, return no hits.
- `hybrid`: build the dense and, when available, sparse prefetches and fuse them with Qdrant RRF. A question with no sparse terms falls back to dense retrieval.

Request `CANDIDATES` points when a reranker is configured and otherwise request only `k`. The existing reranker reorders candidates when successful. If it fails, retain retrieval order and leave reranker scores absent.

### Scores

Add `retrieval_score` to `Hit` and populate it from each Qdrant scored point. Keep the existing `score` field as the reranker score so current chat responses and the `SourcesPanel` contract remain compatible.

Raw scores have mode-specific meanings: cosine similarity for dense, Qdrant sparse/BM25 score for sparse, and RRF fusion score for hybrid. They must not be normalized or presented as directly comparable across modes.

### API contract

Add `POST /api/retrieve`.

Request:

```json
{
  "question": "Triệu chứng viêm phổi là gì?",
  "mode": "hybrid",
  "k": 10
}
```

Validation:

- `question`: trimmed, 1 through 1000 characters.
- `mode`: one of `dense`, `sparse`, or `hybrid`.
- `k`: integer from 1 through 20, default 10.

Response:

```json
{
  "mode": "hybrid",
  "hits": [
    {
      "id": "chunk-id",
      "text": "Chunk text",
      "type": "disease",
      "article_title": "Viêm phổi",
      "article_url": "https://youmed.vn/tin-tuc/viem-phoi/",
      "section_path": "Triệu chứng",
      "retrieval_score": 0.032786,
      "rerank_score": 0.9123
    }
  ]
}
```

Map embedding failures to `502`, matching the chat endpoint. Invalid request fields use FastAPI's standard `422` response. Reranker failures are already handled inside `Retriever` and do not fail the endpoint.

## Frontend Design

Use Next.js file routing rather than adding client-side tab state:

- `/` renders the existing chat.
- `/retrieval` renders a new client component.
- The shared header renders links for `Chat` and `Retrieval` and marks the current route active.

The retrieval page contains:

- A single question input with the existing 1000-character limit.
- A three-option native control for dense, sparse, and hybrid mode; hybrid is the default.
- A submit button with disabled/loading state.
- A readable error message for network, timeout, validation, or backend failures.
- A result count followed by ranked chunk cards.

Each card shows article metadata, the full chunk text, the raw retrieval score to six decimal places, and the reranker score when present. The score label names the selected retrieval mode and notes that scores are not comparable across modes. The frontend sends `k: 10`; no result-count selector is added.

Add a typed `retrieve` function beside the existing `chat` and `health` clients. It uses the same timeout and backend error-detail handling as `chat`.

## Data Flow

1. The user enters a question, selects a mode, and submits the retrieval form.
2. The frontend posts the validated values to `/api/retrieve` through the existing Next.js API rewrite.
3. FastAPI validates the request and calls `Retriever.search(question, k, mode)`.
4. The retriever performs only the encodings required by the selected mode, queries Qdrant, and records raw point scores.
5. When configured, the reranker scores and reorders the candidate set without replacing raw retrieval scores.
6. The endpoint serializes both scores and chunk metadata; the frontend renders the ranked cards.

## Error and Empty States

- Before the first request, explain that results will appear after a query.
- A successful query with no matches displays an explicit empty-results message.
- Prevent duplicate submission while a request is pending.
- Preserve the current form values after an error so the user can retry.
- A missing reranker is normal: omit the reranker score instead of showing an error.

## Verification

Backend tests cover:

- Dense mode uses only dense query inputs and returns Qdrant raw scores.
- Sparse mode avoids the embedding service and returns no hits for a termless query.
- Hybrid mode preserves the current RRF behavior and dense fallback.
- Successful reranking preserves `retrieval_score`, sets the reranker score, and reorders hits.
- Reranker failure preserves retrieval ordering and leaves reranker scores absent.
- The retrieval endpoint validates question, mode, and `k`, serializes both scores, and maps embedding failures to `502`.
- The existing chat API remains hybrid by default and keeps its current source score contract.

Frontend tests cover the typed retrieval request, response parsing, backend detail errors, generic errors, and timeout handling. `npm run build` verifies the new route and components compile. Completion gates are the full backend `pytest` suite, frontend Vitest suite, and frontend production build.

