# Retrieval Tab Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a retrieval-only tab that runs dense, sparse, or hybrid search and shows each chunk's raw retrieval score separately from its reranker score.

**Architecture:** Extend the existing `Retriever` with a mode parameter and retain Qdrant point scores, then expose that behavior through a dedicated FastAPI endpoint. Add a Next.js `/retrieval` route that reuses the shared header and API proxy; the existing chat remains on `/` and keeps hybrid retrieval as its default.

**Tech Stack:** Python 3.11, FastAPI, Pydantic, Qdrant client, pytest, Next.js 15, React 19, TypeScript, Vitest, plain CSS.

**Spec:** `docs/superpowers/specs/2026-09-23-retrieval-tab-design.md`

## Global Constraints

- Do not call the LLM or persist query history from the retrieval tab.
- Support exactly `dense`, `sparse`, and `hybrid`; default existing callers and the new UI to `hybrid`.
- Keep Qdrant retrieval score and reranker score as separate fields; do not normalize or compare scores across modes.
- Accept `k` from 1 through 20, default 10 at the API, and keep the UI fixed at 10.
- Keep `/api/chat` response compatibility: its existing `score` remains the reranker score.
- Add no dependency, filter, pagination, configurable fusion, history, or side-by-side mode comparison.
- Prefix repository shell commands with `rtk` as required by `RTK.md`.

## Review Focus

- A sparse query containing no indexable terms returns no hits and never calls the embedding service; Task 1 pins this behavior.
- A hybrid query containing no sparse terms still performs dense retrieval; Task 1 pins this fallback.
- A reranker outage preserves raw scores and retrieval order while leaving reranker scores absent; Task 1 pins this fallback.
- Invalid modes and non-strict or out-of-range `k` values return `422`; Task 2 pins the request boundary.
- A frontend timeout remains distinguishable from backend detail and generic HTTP errors; Task 3 pins all three paths.

---

### Task 1: Mode-aware retrieval and raw scores

**Files:**
- Modify: `backend/medical_rag/retrieval.py:1-100`
- Modify: `backend/tests/test_retrieval.py:47-98`
- Modify: `backend/tests/test_api.py:9-16`
- Modify: `backend/tests/test_generation.py:1-15`

**Interfaces:**
- Produces: `SearchMode = Literal["dense", "sparse", "hybrid"]`.
- Produces: `Hit.retrieval_score: float` for the Qdrant score and preserves `Hit.score: float | None` for reranking.
- Produces: `Retriever.search(question: str, k: int = 5, mode: SearchMode = "hybrid") -> list[Hit]`.
- Preserves: callers that omit `mode` use the current hybrid pipeline.

- [ ] **Step 1: Add failing mode and score tests**

In `backend/tests/test_retrieval.py`, extend the hybrid assertions and add the focused mode tests:

```python
def test_hybrid_search_merges_dense_and_sparse_hits(store):
    hits = Retriever(store, fake_embed).search(QUESTION)
    ids = [hit.id for hit in hits]
    assert set(ids[:2]) == {"1", "3"}
    assert ids[2] == "2"
    assert all(isinstance(hit.retrieval_score, float) for hit in hits)
    assert all(hit.score is None for hit in hits)


def test_dense_search_returns_raw_scores(store):
    hits = Retriever(store, fake_embed).search(QUESTION, mode="dense")
    assert hits[0].id == "1"
    assert hits[0].retrieval_score == pytest.approx(1.0)


def test_sparse_search_does_not_embed(store):
    def fail_embed(*args, **kwargs):
        raise AssertionError("sparse mode must not call the embedding service")

    hits = Retriever(store, fail_embed).search(QUESTION, mode="sparse")
    assert hits[0].id == "3"
    assert hits[0].retrieval_score > 0


def test_sparse_question_without_terms_returns_no_hits(store):
    def fail_embed(*args, **kwargs):
        raise AssertionError("sparse mode must not call the embedding service")

    assert Retriever(store, fail_embed).search("???", mode="sparse") == []


def test_hybrid_question_without_terms_falls_back_to_dense(store):
    hits = Retriever(store, fake_embed).search("???", mode="hybrid")
    assert hits[0].id == "1"
    assert hits[0].retrieval_score == pytest.approx(1.0)
```

Update the rerank tests so they prove the two scores remain independent:

```python
def test_rerank_reorders_and_sets_scores(store):
    seen = {}

    def rerank(query, documents):
        seen["query"], seen["documents"] = query, documents
        return [9.0 if "tiểu đường" in doc else -1.5 for doc in documents]

    hits = Retriever(store, fake_embed, rerank).search(QUESTION)
    assert hits[0].id == "2"
    assert hits[0].score == 9.0
    assert all(isinstance(hit.retrieval_score, float) for hit in hits)
    assert seen["query"] == QUESTION
    assert sorted(seen["documents"]) == sorted(DOCS)


def test_rerank_failure_keeps_retrieval_order_and_scores(store):
    def broken(query, documents):
        raise RerankError("tunnel down")

    plain = Retriever(store, fake_embed).search(QUESTION)
    fallback = Retriever(store, fake_embed, broken).search(QUESTION)
    assert [h.id for h in fallback] == [h.id for h in plain]
    assert [h.retrieval_score for h in fallback] == [h.retrieval_score for h in plain]
    assert all(hit.score is None for hit in fallback)
```

Add `retrieval_score=0.25` to the manually constructed `Hit` values in `backend/tests/test_api.py` and `backend/tests/test_generation.py` so the dataclass contract is explicit.

- [ ] **Step 2: Run the focused tests and confirm failure**

Run:

```powershell
rtk python -m pytest backend/tests/test_retrieval.py backend/tests/test_generation.py backend/tests/test_api.py -q
```

Expected: failures because `Hit` has no `retrieval_score` and `Retriever.search` has no `mode` parameter.

- [ ] **Step 3: Implement the minimum mode branches**

Update the typing import and add the public mode type in `backend/medical_rag/retrieval.py`:

```python
from typing import Callable, Literal

SearchMode = Literal["dense", "sparse", "hybrid"]
```

Make raw score required on `Hit` and populate it from Qdrant:

```python
@dataclass
class Hit:
    id: str
    text: str
    type: str
    article_title: str
    article_url: str
    section_path: str
    retrieval_score: float
    score: float | None = None


def _hit(point) -> Hit:
    p = point.payload
    return Hit(
        id=str(point.id),
        text=p["text"],
        type=p["type"],
        article_title=p["article_title"],
        article_url=p["article_url"],
        section_path=p["section_path"],
        retrieval_score=float(point.score),
    )
```

Replace `Retriever.search` with a single branch per supported Qdrant query shape:

```python
def search(self, question: str, k: int = 5, mode: SearchMode = "hybrid") -> list[Hit]:
    segmented = segment(question)
    limit = CANDIDATES if self.rerank else k

    if mode in ("dense", "hybrid"):
        dense = self.embed([segmented], task="query")[0]

    if mode == "dense":
        points = self.client.query_points(
            COLLECTION, query=dense, using="dense", limit=limit, with_payload=True
        ).points
    else:
        indices, values = self._bm25.encode_query(segmented)
        if mode == "sparse":
            if not indices:
                return []
            points = self.client.query_points(
                COLLECTION,
                query=models.SparseVector(indices=indices, values=values),
                using="sparse",
                limit=limit,
                with_payload=True,
            ).points
        elif not indices:
            points = self.client.query_points(
                COLLECTION, query=dense, using="dense", limit=limit, with_payload=True
            ).points
        else:
            points = self.client.query_points(
                COLLECTION,
                prefetch=[
                    models.Prefetch(query=dense, using="dense", limit=CANDIDATES),
                    models.Prefetch(
                        query=models.SparseVector(indices=indices, values=values),
                        using="sparse",
                        limit=CANDIDATES,
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=limit,
                with_payload=True,
            ).points

    hits = [_hit(point) for point in points]
    if self.rerank and hits:
        try:
            scores = self.rerank(question, [hit.text for hit in hits])
        except RerankError as error:
            log.warning("rerank failed, keeping retrieval order: %s", error)
            return hits[:k]
        for hit, score in zip(hits, scores):
            hit.score = score
        hits.sort(key=lambda hit: hit.score, reverse=True)
    return hits[:k]
```

The `else` branch is safe because `SearchMode` is validated at the API boundary; do not add strategy classes or a mode registry.

- [ ] **Step 4: Run retrieval and regression tests**

Run:

```powershell
rtk python -m pytest backend/tests/test_retrieval.py backend/tests/test_generation.py backend/tests/test_api.py -q
```

Expected: all selected tests pass, including the existing chat-generation tests.

- [ ] **Step 5: Commit the retrieval engine change**

```powershell
rtk git add backend/medical_rag/retrieval.py backend/tests/test_retrieval.py backend/tests/test_api.py backend/tests/test_generation.py
rtk git commit -m "feat: support selectable retrieval modes and raw scores"
```

---

### Task 2: Retrieval API contract

**Files:**
- Modify: `backend/medical_rag/api.py:4-58`
- Modify: `backend/tests/test_api.py:19-68`
- Modify: `README.md:54-58`

**Interfaces:**
- Consumes: `SearchMode`, `Hit.retrieval_score`, and `Retriever.search(question, k, mode)` from Task 1.
- Produces: `POST /api/retrieve` with `{question, mode, k}` input.
- Produces: `{mode, hits}` where each hit has chunk metadata, `retrieval_score`, and `rerank_score`.

- [ ] **Step 1: Make the fake retriever record the new call shape**

Update `FakeRetriever` in `backend/tests/test_api.py`:

```python
class FakeRetriever:
    def __init__(self, hits=None, error=None):
        self.hits = [HIT] if hits is None else hits
        self.error = error
        self.calls = []

    def search(self, question, k=5, mode="hybrid"):
        self.calls.append((question, k, mode))
        if self.error:
            raise self.error
        return self.hits
```

This keeps `generation.answer()` compatible because it still omits `mode` and receives the hybrid default.

- [ ] **Step 2: Add failing endpoint tests**

Add `from dataclasses import replace` to `backend/tests/test_api.py`, then add:

```python
def test_retrieve_returns_scores_and_forwards_options(client):
    retriever = FakeRetriever(hits=[replace(HIT, score=0.75)])
    api.app.state.retriever = retriever

    resp = client.post(
        "/api/retrieve",
        json={"question": "  đau đầu?  ", "mode": "dense", "k": 7},
    )

    assert resp.status_code == 200
    assert resp.json() == {
        "mode": "dense",
        "hits": [
            {
                "id": "1",
                "text": "Nội dung.",
                "type": "disease",
                "article_title": "Bài 1",
                "article_url": "https://example.test/1",
                "section_path": "Mục",
                "retrieval_score": 0.25,
                "rerank_score": 0.75,
            }
        ],
    }
    assert retriever.calls == [("đau đầu?", 7, "dense")]


@pytest.mark.parametrize(
    "payload",
    [
        {"question": "", "mode": "hybrid", "k": 10},
        {"question": "q", "mode": "other", "k": 10},
        {"question": "q", "mode": "dense", "k": 0},
        {"question": "q", "mode": "dense", "k": 21},
        {"question": "q", "mode": "dense", "k": True},
        {"question": "q", "mode": "dense", "k": "10"},
    ],
)
def test_retrieve_rejects_invalid_requests(client, payload):
    assert client.post("/api/retrieve", json=payload).status_code == 422


def test_retrieve_defaults_to_hybrid_and_ten_hits(client):
    retriever = FakeRetriever()
    api.app.state.retriever = retriever
    assert client.post("/api/retrieve", json={"question": "q"}).status_code == 200
    assert retriever.calls == [("q", 10, "hybrid")]


def test_retrieve_maps_embed_failure_to_502(client):
    api.app.state.retriever = FakeRetriever(error=EmbedError("down"))
    resp = client.post("/api/retrieve", json={"question": "q"})
    assert resp.status_code == 502
    assert "down" in resp.json()["detail"]
```

- [ ] **Step 3: Run the API tests and confirm failure**

```powershell
rtk python -m pytest backend/tests/test_api.py -q
```

Expected: `/api/retrieve` returns `404` because the endpoint is not implemented.

- [ ] **Step 4: Implement strict request validation and serialization**

Update imports in `backend/medical_rag/api.py`:

```python
from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints

from medical_rag.retrieval import RerankClient, Retriever, SearchMode
```

Add the request model beside `ChatRequest`:

```python
class RetrievalRequest(ChatRequest):
    mode: SearchMode = "hybrid"
    k: Annotated[int, Field(strict=True, ge=1, le=20)] = 10
```

Add the endpoint before `/api/health`:

```python
@app.post("/api/retrieve")
def retrieve(body: RetrievalRequest, request: Request):
    try:
        hits = request.app.state.retriever.search(body.question, k=body.k, mode=body.mode)
    except EmbedError as error:
        raise HTTPException(502, f"embedding service error: {error}") from error
    return {
        "mode": body.mode,
        "hits": [
            {
                "id": hit.id,
                "text": hit.text,
                "type": hit.type,
                "article_title": hit.article_title,
                "article_url": hit.article_url,
                "section_path": hit.section_path,
                "retrieval_score": hit.retrieval_score,
                "rerank_score": hit.score,
            }
            for hit in hits
        ],
    }
```

Do not route through `generation.answer`; that would call the LLM and change the response shape.

- [ ] **Step 5: Document and verify the endpoint**

Add this row to the endpoint table in `README.md`:

```markdown
| `POST /api/retrieve` | `{"question": "...", "mode": "dense|sparse|hybrid", "k": 10}` returns ranked chunks with separate `retrieval_score` and `rerank_score` |
```

Run:

```powershell
rtk python -m pytest backend/tests/test_api.py backend/tests/test_retrieval.py -q
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit the API contract**

```powershell
rtk git add backend/medical_rag/api.py backend/tests/test_api.py README.md
rtk git commit -m "feat: expose retrieval-only API"
```

---

### Task 3: Typed frontend retrieval client

**Files:**
- Modify: `frontend/lib/api.ts:1-44`
- Modify: `frontend/lib/api.test.ts:1-64`

**Interfaces:**
- Consumes: `POST /api/retrieve` from Task 2.
- Produces: `RetrievalMode`, `RetrievalHit`, `RetrievalResponse`, and `retrieve(question, mode)` for the UI.
- Preserves: existing `chat()` and `health()` behavior.

- [ ] **Step 1: Add failing client tests**

Update the import in `frontend/lib/api.test.ts`:

```typescript
import { chat, health, retrieve } from "./api";
```

Add:

```typescript
describe("retrieve", () => {
  it("posts the selected mode and fixed result count", async () => {
    const payload = {
      mode: "sparse",
      hits: [
        {
          id: "1",
          text: "Nội dung",
          type: "disease",
          article_title: "Bài 1",
          article_url: "https://example.test/1",
          section_path: "Mục",
          retrieval_score: 1.25,
          rerank_score: null,
        },
      ],
    };
    const fetchMock = vi.fn().mockResolvedValue(json(payload));
    vi.stubGlobal("fetch", fetchMock);

    expect(await retrieve("đau đầu?", "sparse")).toEqual(payload);
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("/api/retrieve");
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body)).toEqual({ question: "đau đầu?", mode: "sparse", k: 10 });
  });

  it("uses a backend detail error", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(json({ detail: "embedding service error: down" }, 502)));
    await expect(retrieve("q", "dense")).rejects.toThrow("embedding service error: down");
  });

  it("uses a generic error for a non-string detail", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(json({ detail: [] }, 422)));
    await expect(retrieve("q", "hybrid")).rejects.toThrow("Request failed (422)");
  });

  it("maps a timeout to a readable message", async () => {
    const timeout = Object.assign(new Error("timed out"), { name: "TimeoutError" });
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(timeout));
    await expect(retrieve("q", "hybrid")).rejects.toThrow("Request timed out");
  });
});
```

- [ ] **Step 2: Run the frontend test and confirm failure**

```powershell
rtk npm test -- lib/api.test.ts
```

Run from `frontend/`. Expected: failure because `retrieve` is not exported.

- [ ] **Step 3: Add types and the minimal fetch client**

Add to `frontend/lib/api.ts`:

```typescript
export type RetrievalMode = "dense" | "sparse" | "hybrid";

export type RetrievalHit = {
  id: string;
  text: string;
  type: string;
  article_title: string;
  article_url: string;
  section_path: string;
  retrieval_score: number;
  rerank_score: number | null;
};

export type RetrievalResponse = { mode: RetrievalMode; hits: RetrievalHit[] };
```

Rename `CHAT_TIMEOUT_MS` to `REQUEST_TIMEOUT_MS`, keep its value at `60_000`, update `chat`, and add:

```typescript
export async function retrieve(question: string, mode: RetrievalMode): Promise<RetrievalResponse> {
  let res: Response;
  try {
    res = await fetch("/api/retrieve", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, mode, k: 10 }),
      signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
    });
  } catch (err) {
    if (err instanceof Error && err.name === "TimeoutError") throw new Error("Request timed out");
    throw err;
  }
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(typeof data.detail === "string" ? data.detail : `Request failed (${res.status})`);
  }
  return data as RetrievalResponse;
}
```

Do not introduce a generic fetch wrapper in this feature; two small request functions are cheaper than an unrelated API refactor.

- [ ] **Step 4: Run the frontend API tests**

```powershell
rtk npm test -- lib/api.test.ts
```

Run from `frontend/`. Expected: all API tests pass.

- [ ] **Step 5: Commit the frontend client**

```powershell
rtk git add frontend/lib/api.ts frontend/lib/api.test.ts
rtk git commit -m "feat: add typed retrieval API client"
```

---

### Task 4: Retrieval route, navigation, and result cards

**Files:**
- Create: `frontend/app/retrieval/page.tsx`
- Create: `frontend/components/RetrievalApp.tsx`
- Modify: `frontend/components/Header.tsx:1-28`
- Modify: `frontend/app/globals.css:22-183`

**Interfaces:**
- Consumes: `retrieve(question, mode)`, `RetrievalMode`, and `RetrievalResponse` from Task 3.
- Consumes: the existing `Header`, `health`, and `isHttpUrl` utilities.
- Produces: `/retrieval` and shared `Chat | Retrieval` navigation.

- [ ] **Step 1: Add the route first as a compile-time check**

Create `frontend/app/retrieval/page.tsx`:

```tsx
import { RetrievalApp } from "../../components/RetrievalApp";

export default function RetrievalPage() {
  return <RetrievalApp />;
}
```

- [ ] **Step 2: Run the production build and confirm failure**

```powershell
rtk npm run build
```

Run from `frontend/`. Expected: module resolution fails because `components/RetrievalApp.tsx` does not exist yet.

- [ ] **Step 3: Implement the retrieval client component**

Create `frontend/components/RetrievalApp.tsx`:

```tsx
"use client";

import { type FormEvent, useState } from "react";
import { isHttpUrl } from "../lib/sources";
import { retrieve } from "../lib/api";
import type { RetrievalMode, RetrievalResponse } from "../lib/api";
import { Header } from "./Header";

export function RetrievalApp() {
  const [question, setQuestion] = useState("");
  const [mode, setMode] = useState<RetrievalMode>("hybrid");
  const [result, setResult] = useState<RetrievalResponse | null>(null);
  const [error, setError] = useState("");
  const [pending, setPending] = useState(false);

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    const value = question.trim();
    if (!value || pending) return;
    setPending(true);
    setError("");
    setResult(null);
    try {
      setResult(await retrieve(value, mode));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Request failed");
    } finally {
      setPending(false);
    }
  };

  return (
    <div className="app">
      <Header />
      <main className="retrieval-main">
        <section className="retrieval-shell">
          <div className="retrieval-heading">
            <h1>Retrieval Inspector</h1>
            <p>Inspect ranked chunks without generating an answer.</p>
          </div>

          <form className="retrieval-form" onSubmit={submit}>
            <div className="retrieval-field">
              <label htmlFor="retrieval-question">Question</label>
              <input
                id="retrieval-question"
                value={question}
                maxLength={1000}
                required
                placeholder="Enter a medical query..."
                onChange={(event) => setQuestion(event.target.value)}
              />
            </div>
            <div className="retrieval-field">
              <label htmlFor="retrieval-mode">Mode</label>
              <select
                id="retrieval-mode"
                value={mode}
                onChange={(event) => setMode(event.target.value as RetrievalMode)}
              >
                <option value="hybrid">Hybrid</option>
                <option value="dense">Dense</option>
                <option value="sparse">Sparse</option>
              </select>
            </div>
            <button type="submit" disabled={pending || !question.trim()}>
              {pending ? "Searching…" : "Search"}
            </button>
          </form>

          <p className="score-note">
            Retrieval scores use the selected mode's scale and are not comparable across modes.
          </p>
          {error && <div className="retrieval-error" role="alert">{error}</div>}
          {!result && !error && !pending && (
            <div className="retrieval-empty">Results appear here after you run a query.</div>
          )}
          {result && (
            <section className="retrieval-results" aria-live="polite">
              <h2>{result.hits.length} chunks</h2>
              {result.hits.length === 0 && <div className="retrieval-empty">No matching chunks found.</div>}
              {result.hits.map((hit, index) => (
                <article className="retrieval-card" key={hit.id}>
                  <div className="retrieval-card-head">
                    <span className="retrieval-rank">#{index + 1}</span>
                    <div>
                      <h3>
                        {isHttpUrl(hit.article_url) ? (
                          <a href={hit.article_url} target="_blank" rel="noopener noreferrer">
                            {hit.article_title}
                          </a>
                        ) : hit.article_title}
                      </h3>
                      <div className="meta">{hit.section_path} · {hit.type}</div>
                    </div>
                  </div>
                  <div className="retrieval-scores">
                    <span>{result.mode} score <strong>{hit.retrieval_score.toFixed(6)}</strong></span>
                    {hit.rerank_score !== null && (
                      <span>reranker score <strong>{hit.rerank_score.toFixed(4)}</strong></span>
                    )}
                  </div>
                  <div className="retrieval-text">{hit.text}</div>
                </article>
              ))}
            </section>
          )}
        </section>
      </main>
    </div>
  );
}
```

The `pending` guard and disabled button prevent duplicate submissions. The question and selected mode are never cleared, including on failure.

- [ ] **Step 4: Add route-aware header navigation**

Update `frontend/components/Header.tsx` imports:

```tsx
import Link from "next/link";
import { usePathname } from "next/navigation";
```

Inside `Header`, add `const pathname = usePathname();`, then render the brand and navigation together before the existing status pill:

```tsx
<div className="header-left">
  <div className="brand">
    <div className="brand-badge">✚</div>
    <span>MediRAG</span>
  </div>
  <nav className="app-nav" aria-label="Primary">
    <Link className={pathname === "/" ? "active" : ""} href="/">Chat</Link>
    <Link className={pathname === "/retrieval" ? "active" : ""} href="/retrieval">Retrieval</Link>
  </nav>
</div>
```

Keep the current health request and status pill unchanged.

- [ ] **Step 5: Add only the CSS required by the new route**

Append these rules before the existing media queries in `frontend/app/globals.css`:

```css
.header-left,.app-nav{display:flex;align-items:center}
.header-left{gap:28px}
.app-nav{gap:4px}
.app-nav a{
  color:var(--muted);text-decoration:none;font-size:14px;font-weight:700;
  padding:8px 12px;border-radius:9px;
}
.app-nav a:hover{background:#f3f6fa;color:var(--text)}
.app-nav a.active{background:var(--primary-soft);color:#0f5db6}
.retrieval-main{min-height:0;overflow:auto;padding:28px;background:var(--bg)}
.retrieval-shell{max-width:960px;margin:0 auto}
.retrieval-heading h1{margin:0 0 5px;font-size:24px}
.retrieval-heading p,.score-note{color:var(--muted);font-size:13px}
.retrieval-form{
  display:grid;grid-template-columns:1fr 150px auto;gap:8px 12px;
  align-items:end;margin:22px 0 10px;padding:18px;background:var(--panel);
  border:1px solid var(--border);border-radius:14px;box-shadow:var(--shadow);
}
.retrieval-field{display:grid;gap:6px}
.retrieval-form label{font-size:12px;font-weight:700}
.retrieval-form input,.retrieval-form select{
  min-height:42px;border:1px solid #d9e2ec;border-radius:10px;
  padding:0 12px;background:white;font:inherit;
}
.retrieval-form button{
  min-height:42px;border:0;border-radius:10px;padding:0 18px;
  background:var(--primary);color:white;font-weight:800;cursor:pointer;
}
.retrieval-form button:disabled{opacity:.5;cursor:default}
.score-note{margin:0 0 18px}
.retrieval-error,.retrieval-empty{padding:16px;border-radius:12px;background:white;border:1px solid var(--border)}
.retrieval-error{color:#991b1b;background:#fef2f2;border-color:#fecaca}
.retrieval-results h2{font-size:16px;margin:0 0 12px}
.retrieval-card{
  margin-bottom:14px;padding:18px;background:white;border:1px solid var(--border);
  border-radius:14px;box-shadow:var(--shadow);
}
.retrieval-card-head{display:flex;gap:12px;align-items:flex-start}
.retrieval-rank{font-weight:800;color:var(--primary)}
.retrieval-card h3{font-size:15px;margin:0 0 4px}
.retrieval-card h3 a{color:inherit;text-decoration:none}
.retrieval-card h3 a:hover{text-decoration:underline}
.retrieval-scores{display:flex;gap:18px;margin:14px 0 10px;font-size:12px;color:var(--muted)}
.retrieval-scores strong{margin-left:4px;color:#0f5db6}
.retrieval-text{white-space:pre-wrap;overflow-wrap:anywhere;font-size:14px;line-height:1.55}
```

Extend the existing `max-width:700px` media query:

```css
  header{padding:0 12px}
  .header-left{gap:10px}
  .brand span,.status,.status.offline{display:none}
  .retrieval-main{padding:16px}
  .retrieval-form{grid-template-columns:1fr}
  .retrieval-scores{flex-direction:column;gap:4px}
```

- [ ] **Step 6: Run frontend verification**

From `frontend/`, run:

```powershell
rtk npm test
rtk npm run build
```

Expected: all Vitest tests pass and Next.js builds both `/` and `/retrieval` successfully.

- [ ] **Step 7: Perform a browser smoke check**

With the existing backend, embedding server, Qdrant data, and frontend running:

1. Open `/`; verify chat still loads, sends a question, and shows its existing source score behavior.
2. Open `/retrieval`; submit the same question in hybrid, dense, and sparse modes.
3. Verify each response shows 10 or fewer ranked chunks, full chunk text, and a six-decimal raw score.
4. If `RERANK_URL` is configured, verify reranker scores appear separately and ranking follows them; otherwise verify that row is absent.
5. Submit punctuation-only text in sparse mode and verify the empty-results message; repeat in hybrid mode and verify dense results.
6. Stop the embedding service, submit dense mode, verify the readable error, and verify the form remains populated and usable.

- [ ] **Step 8: Run the full completion gates and commit**

From the repository root:

```powershell
rtk python -m pytest backend/tests -q
rtk npm --prefix frontend test
rtk npm --prefix frontend run build
```

Expected: all backend tests and frontend tests pass; the production build succeeds.

Then commit:

```powershell
rtk git add frontend/app/retrieval/page.tsx frontend/components/RetrievalApp.tsx frontend/components/Header.tsx frontend/app/globals.css
rtk git commit -m "feat: add retrieval inspection tab"
```

---

## Plan Self-Review

- **Spec coverage:** Tasks 1–4 cover every backend mode, independent score, API field, route, UI state, error path, regression gate, and stated non-goal in the approved design.
- **Placeholder scan:** Every test and implementation step includes concrete code or an exact verification command; no deferred implementation markers remain.
- **Type consistency:** `SearchMode` maps to `RetrievalMode`; backend `retrieval_score` and `score` serialize as frontend `retrieval_score` and `rerank_score`; the UI consumes the exact Task 3 types.
- **Review focus:** Each of the five high-risk cases is exercised in Task 1, Task 2, or Task 3 before the full build and browser smoke check.
