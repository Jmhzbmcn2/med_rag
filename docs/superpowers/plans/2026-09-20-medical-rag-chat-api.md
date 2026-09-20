# Medical RAG Chat API and UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Connect the mockup UI to the backend: hybrid retrieval with optional reranking, LLM answer with numbered sources, a FastAPI service, and a UI that renders the answer and sources dynamically.

**Architecture:** `Retriever` runs dense + BM25 prefetch fused with RRF in Qdrant, then optionally reranks through a Kaggle `/rerank` endpoint. `generation.answer` builds a numbered-context prompt and calls the existing `llm.chat`. `api.py` exposes `POST /api/chat`, `GET /api/health` and serves the UI at `/`. The UI is a single static HTML file that calls the API with `fetch`.

**Tech Stack:** Python 3.11, FastAPI, uvicorn, qdrant-client (local mode), requests, openai SDK (OpenRouter), pytest, httpx (TestClient), plain HTML/JS.

**Spec:** `docs/superpowers/specs/2026-09-20-medical-rag-chat-api-design.md`

## Global Constraints

- Work from `backend/`. Run tests with `python -m pytest -q` (baseline: 57 passed). `filterwarnings = error::UserWarning` is on.
- Env vars: `EMBED_URL` (required), `OPENROUTER_API_KEY`, `LLM_MODEL`, `QDRANT_PATH` (default `qdrant_data`), `RERANK_URL` (optional). Never hardcode URLs or keys.
- Reranking is optional: unset `RERANK_URL` or a failing `/rerank` call must fall back to the RRF order with `score=None`.
- Retrieval defaults: `k=5`, 20 candidates per prefetch (`CANDIDATES = 20`).
- `question`: 1 to 1000 characters after strip. Errors from the embed or LLM services map to HTTP 502.
- The frontend must never assign user or model text through `innerHTML`.
- Single-turn only. No history, no streaming.
- Commit messages end with `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>`. Stage files by name, never `git add -A` (the tree holds unrelated untracked work such as `.claude/`, `serve_model/`).
- Style: match the surrounding code (short functions, few comments, callables instead of classes for injected behavior).

## File Structure

```
backend/medical_rag/encoders.py       # modify: EmbedClient.embed/_post gain `task`
backend/medical_rag/retrieval.py      # create: Hit, RerankError, RerankClient, Retriever
backend/medical_rag/generation.py     # create: build_messages, answer
backend/medical_rag/api.py            # create: FastAPI app
backend/medical_rag/static/index.html # moved from UI_template/medical_rag_ui.html, then rewritten
backend/pyproject.toml                # modify: deps + package-data
backend/tests/test_encoders.py        # modify: task test
backend/tests/test_retrieval.py       # create
backend/tests/test_generation.py      # create
backend/tests/test_api.py             # create
```

---

### Task 1: `task` argument on `EmbedClient.embed` and dependencies

**Files:**
- Modify: `backend/medical_rag/encoders.py` (`embed`, `_post`)
- Modify: `backend/pyproject.toml`
- Test: `backend/tests/test_encoders.py`

**Interfaces:**
- Produces: `EmbedClient.embed(segmented_texts: list[str], task: str = "document") -> list[list[float]]`. The request body gains `"task": "query"` only when `task == "query"`.

- [ ] **Step 1: Write the failing test**

Append to `backend/tests/test_encoders.py`:

```python
def test_embed_sends_task_only_for_queries():
    client = EmbedClient("http://embed.test")
    bodies = []

    class Response:
        status_code = 200

        def json(self):
            return ok_body(bodies[-1]["texts"])

    def fake_post(url, json, timeout):
        bodies.append(json)
        return Response()

    client._session.post = fake_post
    client.embed(["a"])
    client.embed(["b"], task="query")
    assert bodies == [{"texts": ["a"]}, {"texts": ["b"], "task": "query"}]
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_encoders.py::test_embed_sends_task_only_for_queries -v`
Expected: FAIL with `TypeError: embed() got an unexpected keyword argument 'task'`.

- [ ] **Step 3: Implement**

In `backend/medical_rag/encoders.py` change three lines:

```python
    def embed(self, segmented_texts: list[str], task: str = "document") -> list[list[float]]:
```
```python
                vectors.extend(self._post(batch, task))
```
```python
    def _post(self, batch: list[str], task: str = "document") -> list[list[float]]:
        body = {"texts": batch}
        if task == "query":
            body["task"] = task  # only embed_server.py reads it; the Kaggle server ignores it
        last_error = None
```
and in `_post` replace `json={"texts": batch}` with `json=body`:

```python
                resp = self._session.post(self.url, json=body, timeout=self.timeout)
```

- [ ] **Step 4: Add dependencies to `backend/pyproject.toml`**

Add `"fastapi",` and `"uvicorn",` to `dependencies`, `"httpx"` to `dev`, and package data:

```toml
dependencies = [
    "langchain-text-splitters",
    "transformers",
    "pandas",
    "qdrant-client>=1.17,<2",
    "pyvi==0.1.1",
    "requests",
    "openai",
    "fastapi",
    "uvicorn",
]

[project.optional-dependencies]
dev = ["pytest", "httpx"]

[tool.setuptools.package-data]
medical_rag = ["static/*.html"]
```

- [ ] **Step 5: Run the whole suite**

Run: `python -m pytest -q`
Expected: 58 passed.

- [ ] **Step 6: Commit**

```bash
git add backend/medical_rag/encoders.py backend/pyproject.toml backend/tests/test_encoders.py
git commit -m "feat: EmbedClient sends task=query for search questions

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

(This also commits two pre-existing uncommitted edits in those files: the `health_check` message and the `openai` dependency. That is expected.)

---

### Task 2: Retrieval (`Hit`, `RerankClient`, `Retriever`)

**Files:**
- Create: `backend/medical_rag/retrieval.py`
- Test: `backend/tests/test_retrieval.py`

**Interfaces:**
- Consumes: `segment`, `Bm25Encoder.encode_query` from `medical_rag.encoders`; `COLLECTION` from `medical_rag.store`.
- Produces:
  - `Hit(id: str, text: str, type: str, article_title: str, article_url: str, section_path: str, score: float | None = None)` dataclass.
  - `RerankError(Exception)`.
  - `RerankClient(base_url: str, timeout: float = 30.0).rerank(query: str, documents: list[str]) -> list[float]`.
  - `Retriever(client, embed, rerank=None).search(question: str, k: int = 5) -> list[Hit]` where `embed(texts, task="document")` and `rerank(query, documents)` are callables.

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/test_retrieval.py`:

```python
import pytest
import requests

from medical_rag.encoders import DENSE_DIM, Bm25Encoder, segment
from medical_rag.retrieval import RerankClient, RerankError, Retriever
from medical_rag.store import COLLECTION, build_point, ensure_collection, open_client

DOCS = [
    "Thuốc hạ sốt paracetamol dùng khi sốt cao.",
    "Bệnh tiểu đường cần kiểm soát đường huyết.",
    "Viêm phổi gây ho và khó thở.",
]
QUESTION = "Viêm phổi gây ho và khó thở"  # matches DOCS[2] by BM25 terms


def one_hot(i):
    return [1.0 if j == i else 0.0 for j in range(DENSE_DIM)]


def fake_embed(texts, task="document"):
    return [one_hot(0) for _ in texts]  # the dense side always prefers DOCS[0]


@pytest.fixture
def store():
    client = open_client(None)
    ensure_collection(client)
    segmented = [segment(text) for text in DOCS]
    encoder = Bm25Encoder()
    encoder.fit(segmented)
    points = []
    for i, (text, seg) in enumerate(zip(DOCS, segmented)):
        chunk = {
            "id": i + 1,
            "type": "disease",
            "article_slug": f"a{i}",
            "article_title": f"Bài {i}",
            "article_url": f"https://example.test/{i}",
            "section_path": "Mục",
            "text": text,
        }
        points.append(build_point(chunk, one_hot(i), encoder.encode_doc(seg)))
    client.upsert(COLLECTION, points=points)
    return client


def test_hybrid_search_merges_dense_and_sparse_hits(store):
    hits = Retriever(store, fake_embed).search(QUESTION)
    ids = [hit.id for hit in hits]
    assert set(ids[:2]) == {"1", "3"}  # dense favourite + BM25 favourite beat the unrelated doc
    assert ids[2] == "2"
    assert all(hit.score is None for hit in hits)
    assert hits[0].article_title in {"Bài 0", "Bài 2"}
    assert hits[0].article_url.startswith("https://example.test/")


def test_search_respects_k(store):
    assert len(Retriever(store, fake_embed).search(QUESTION, k=1)) == 1


def test_question_without_terms_falls_back_to_dense_only(store):
    hits = Retriever(store, fake_embed).search("???")
    assert hits[0].id == "1"


def test_search_embeds_the_segmented_question_as_a_query(store):
    calls = []

    def embed(texts, task="document"):
        calls.append((texts, task))
        return fake_embed(texts)

    Retriever(store, embed).search(QUESTION)
    assert calls == [([segment(QUESTION)], "query")]


def test_rerank_reorders_and_sets_scores(store):
    seen = {}

    def rerank(query, documents):
        seen["query"], seen["documents"] = query, documents
        return [9.0 if "tiểu đường" in doc else -1.5 for doc in documents]

    hits = Retriever(store, fake_embed, rerank).search(QUESTION)
    assert hits[0].id == "2"
    assert hits[0].score == 9.0
    assert seen["query"] == QUESTION
    assert sorted(seen["documents"]) == sorted(DOCS)


def test_rerank_failure_keeps_fusion_order(store):
    def broken(query, documents):
        raise RerankError("tunnel down")

    plain = Retriever(store, fake_embed).search(QUESTION)
    fallback = Retriever(store, fake_embed, broken).search(QUESTION)
    assert [h.id for h in fallback] == [h.id for h in plain]
    assert all(hit.score is None for hit in fallback)


class _Response:
    def __init__(self, body, status=200):
        self.body, self.status = body, status

    def raise_for_status(self):
        if self.status >= 400:
            raise requests.HTTPError(f"HTTP {self.status}")

    def json(self):
        return self.body


def test_rerank_client_returns_scores(monkeypatch):
    sent = {}

    def post(url, json, timeout):
        sent.update(url=url, json=json)
        return _Response({"scores": [0.5, -2.0]})

    monkeypatch.setattr(requests, "post", post)
    assert RerankClient("http://rerank.test/").rerank("q", ["a", "b"]) == [0.5, -2.0]
    assert sent == {"url": "http://rerank.test/rerank", "json": {"query": "q", "documents": ["a", "b"]}}


@pytest.mark.parametrize(
    "response",
    [_Response({"scores": [1.0]}), _Response({"nope": 1}), _Response({"scores": ["x", "y"]}), _Response({}, status=530)],
)
def test_rerank_client_rejects_bad_responses(monkeypatch, response):
    monkeypatch.setattr(requests, "post", lambda url, json, timeout: response)
    with pytest.raises(RerankError):
        RerankClient("http://rerank.test").rerank("q", ["a", "b"])


def test_rerank_client_wraps_network_errors(monkeypatch):
    def post(url, json, timeout):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(requests, "post", post)
    with pytest.raises(RerankError):
        RerankClient("http://rerank.test").rerank("q", ["a"])
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_retrieval.py -v`
Expected: collection error `ModuleNotFoundError: No module named 'medical_rag.retrieval'`.

- [ ] **Step 3: Implement `backend/medical_rag/retrieval.py`**

```python
import logging
from dataclasses import dataclass
from typing import Callable

import requests
from qdrant_client import models

from medical_rag.encoders import Bm25Encoder, segment
from medical_rag.store import COLLECTION

CANDIDATES = 20
log = logging.getLogger(__name__)


class RerankError(Exception):
    pass


class RerankClient:
    def __init__(self, base_url: str, timeout: float = 30.0):
        self.url = base_url.rstrip("/") + "/rerank"
        self.timeout = timeout

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        try:
            resp = requests.post(self.url, json={"query": query, "documents": documents}, timeout=self.timeout)
            resp.raise_for_status()
            scores = resp.json()["scores"]
        except (requests.RequestException, ValueError, KeyError, TypeError) as e:
            raise RerankError(str(e)) from e
        if (
            not isinstance(scores, list)
            or len(scores) != len(documents)
            or any(isinstance(s, bool) or not isinstance(s, (int, float)) for s in scores)
        ):
            raise RerankError(f"expected {len(documents)} numeric scores from /rerank")
        return scores


@dataclass
class Hit:
    id: str
    text: str
    type: str
    article_title: str
    article_url: str
    section_path: str
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
    )


class Retriever:
    def __init__(
        self,
        client,
        embed: Callable[..., list[list[float]]],
        rerank: Callable[[str, list[str]], list[float]] | None = None,
    ):
        self.client = client
        self.embed = embed
        self.rerank = rerank
        self._bm25 = Bm25Encoder()  # queries need no fit()

    def search(self, question: str, k: int = 5) -> list[Hit]:
        segmented = segment(question)
        dense = self.embed([segmented], task="query")[0]
        indices, values = self._bm25.encode_query(segmented)
        prefetch = [models.Prefetch(query=dense, using="dense", limit=CANDIDATES)]
        if indices:
            sparse = models.SparseVector(indices=indices, values=values)
            prefetch.append(models.Prefetch(query=sparse, using="sparse", limit=CANDIDATES))
        points = self.client.query_points(
            COLLECTION,
            prefetch=prefetch,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=CANDIDATES if self.rerank else k,
            with_payload=True,
        ).points
        hits = [_hit(point) for point in points]
        if self.rerank and hits:
            try:
                scores = self.rerank(question, [hit.text for hit in hits])
            except RerankError as error:
                log.warning("rerank failed, keeping fusion order: %s", error)
                return hits[:k]
            for hit, score in zip(hits, scores):
                hit.score = score
            hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:k]
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_retrieval.py -v`
Expected: all pass. If `test_hybrid_search_merges_dense_and_sparse_hits` fails on the ordering, print `[(h.id, h.text) for h in hits]` first: the cause is the pyvi segmentation of `QUESTION` differing from the doc's segmentation, so adjust `QUESTION` to a phrase that segments identically in both, not the assertion.

- [ ] **Step 5: Full suite and commit**

Run: `python -m pytest -q` (expect all pass)

```bash
git add backend/medical_rag/retrieval.py backend/tests/test_retrieval.py
git commit -m "feat: hybrid retrieval with optional rerank

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 3: Generation (`build_messages`, `answer`)

**Files:**
- Create: `backend/medical_rag/generation.py`
- Test: `backend/tests/test_generation.py`

**Interfaces:**
- Consumes: `Hit`, `Retriever.search(question, k)` from Task 2; `medical_rag.llm.chat(messages) -> str`.
- Produces:
  - `NO_RESULT: str`.
  - `build_messages(question: str, hits: list[Hit]) -> list[dict]`.
  - `answer(retriever, question: str, k: int = 5) -> dict` returning `{"answer": str, "sources": [{"n", "title", "url", "section_path", "type", "text", "score"}]}`.
  - The module imports `chat` by name (`from medical_rag.llm import chat`), so tests patch `medical_rag.generation.chat`.

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/test_generation.py`:

```python
from medical_rag import generation
from medical_rag.generation import NO_RESULT, answer, build_messages
from medical_rag.retrieval import Hit


def _hit(i, score=None):
    return Hit(
        id=str(i),
        text=f"[Chủ đề: T{i}]\n\nNội dung {i}.",
        type="disease",
        article_title=f"Bài {i}",
        article_url=f"https://example.test/{i}",
        section_path=f"Mục {i}",
        score=score,
    )


class FakeRetriever:
    def __init__(self, hits):
        self.hits = hits
        self.calls = []

    def search(self, question, k=5):
        self.calls.append((question, k))
        return self.hits


def test_build_messages_numbers_every_source_and_ends_with_the_question():
    messages = build_messages("đau đầu là gì?", [_hit(1), _hit(2)])
    assert [m["role"] for m in messages] == ["system", "user"]
    user = messages[1]["content"]
    assert "[1] Bài 1 — Mục 1\n[Chủ đề: T1]" in user
    assert "[2] Bài 2 — Mục 2" in user
    assert "Nội dung 2." in user
    assert user.endswith("Câu hỏi: đau đầu là gì?")


def test_answer_returns_text_and_numbered_sources(monkeypatch):
    seen = {}

    def fake_chat(messages, **kwargs):
        seen["messages"] = messages
        return "Trả lời [1]"

    monkeypatch.setattr(generation, "chat", fake_chat)
    retriever = FakeRetriever([_hit(1, score=0.5), _hit(2)])
    result = answer(retriever, "câu hỏi", k=3)

    assert retriever.calls == [("câu hỏi", 3)]
    assert seen["messages"] == build_messages("câu hỏi", retriever.hits)
    assert result["answer"] == "Trả lời [1]"
    assert result["sources"][0] == {
        "n": 1,
        "title": "Bài 1",
        "url": "https://example.test/1",
        "section_path": "Mục 1",
        "type": "disease",
        "text": "[Chủ đề: T1]\n\nNội dung 1.",
        "score": 0.5,
    }
    assert [s["n"] for s in result["sources"]] == [1, 2]
    assert result["sources"][1]["score"] is None


def test_answer_without_hits_skips_the_llm(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("the LLM must not be called without sources")

    monkeypatch.setattr(generation, "chat", boom)
    assert answer(FakeRetriever([]), "câu hỏi") == {"answer": NO_RESULT, "sources": []}
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_generation.py -v`
Expected: collection error `ModuleNotFoundError: No module named 'medical_rag.generation'`.

- [ ] **Step 3: Implement `backend/medical_rag/generation.py`**

```python
from medical_rag.llm import chat
from medical_rag.retrieval import Hit

SYSTEM_PROMPT = (
    "Bạn là trợ lý y khoa nói tiếng Việt. Chỉ trả lời dựa trên các đoạn tài liệu được đánh số trong tin nhắn của người dùng. "
    "Sau mỗi ý lấy từ tài liệu, ghi số nguồn dạng [n]. "
    "Nếu tài liệu không đủ thông tin, hãy nói rõ là chưa có đủ thông tin, không tự suy đoán. "
    "Không chẩn đoán bệnh cho người dùng; khuyên đi khám bác sĩ khi triệu chứng nặng hoặc kéo dài."
)
NO_RESULT = "Không tìm thấy thông tin liên quan trong cơ sở dữ liệu."


def build_messages(question: str, hits: list[Hit]) -> list[dict]:
    context = "\n\n".join(
        f"[{n}] {hit.article_title} — {hit.section_path}\n{hit.text}" for n, hit in enumerate(hits, 1)
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Tài liệu:\n{context}\n\nCâu hỏi: {question}"},
    ]


def answer(retriever, question: str, k: int = 5) -> dict:
    hits = retriever.search(question, k=k)
    if not hits:
        return {"answer": NO_RESULT, "sources": []}
    sources = [
        {
            "n": n,
            "title": hit.article_title,
            "url": hit.article_url,
            "section_path": hit.section_path,
            "type": hit.type,
            "text": hit.text,
            "score": hit.score,
        }
        for n, hit in enumerate(hits, 1)
    ]
    return {"answer": chat(build_messages(question, hits)), "sources": sources}
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_generation.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/medical_rag/generation.py backend/tests/test_generation.py
git commit -m "feat: build RAG prompt and answer with numbered sources

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 4: FastAPI app and static file move

**Files:**
- Create: `backend/medical_rag/api.py`
- Move: `UI_template/medical_rag_ui.html` → `backend/medical_rag/static/index.html` (content unchanged in this task)
- Test: `backend/tests/test_api.py`

**Interfaces:**
- Consumes: `answer` (Task 3), `Retriever`, `RerankClient` (Task 2), `EmbedClient`, `EmbedError`, `open_client`, `COLLECTION`.
- Produces: `api.app` (FastAPI). `app.state.retriever` (has `.search(question, k)`) and `app.state.client` (Qdrant client) are set by the lifespan. Routes: `POST /api/chat` → `{"answer", "sources"}`; `GET /api/health` → `{"status": "ok", "points": int}`; `GET /` → the HTML file.

- [ ] **Step 1: Move the HTML**

```bash
mkdir -p medical_rag/static
mv ../UI_template/medical_rag_ui.html medical_rag/static/index.html
rmdir ../UI_template 2>/dev/null || true
```

(`rmdir` only succeeds when the folder is empty; leaving it is fine.)

- [ ] **Step 2: Write the failing tests**

Create `backend/tests/test_api.py`:

```python
import pytest
from fastapi.testclient import TestClient

from medical_rag import api, generation
from medical_rag.encoders import EmbedError
from medical_rag.retrieval import Hit
from medical_rag.store import ensure_collection, open_client

HIT = Hit(
    id="1",
    text="Nội dung.",
    type="disease",
    article_title="Bài 1",
    article_url="https://example.test/1",
    section_path="Mục",
)


class FakeRetriever:
    def __init__(self, hits=None, error=None):
        self.hits = [HIT] if hits is None else hits
        self.error = error

    def search(self, question, k=5):
        if self.error:
            raise self.error
        return self.hits


@pytest.fixture
def client(monkeypatch):
    store = open_client(None)
    ensure_collection(store)
    api.app.state.client = store
    api.app.state.retriever = FakeRetriever()
    monkeypatch.setattr(generation, "chat", lambda messages, **kw: "Trả lời [1]")
    return TestClient(api.app)  # not used as a context manager, so the lifespan is skipped


def test_chat_returns_answer_and_sources(client):
    resp = client.post("/api/chat", json={"question": "  đau đầu?  "})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "Trả lời [1]"
    assert body["sources"][0]["title"] == "Bài 1"
    assert body["sources"][0]["score"] is None


@pytest.mark.parametrize("question", ["", "   ", "x" * 1001])
def test_chat_rejects_bad_questions(client, question):
    assert client.post("/api/chat", json={"question": question}).status_code == 422


def test_chat_maps_embed_failure_to_502(client):
    api.app.state.retriever = FakeRetriever(error=EmbedError("down"))
    resp = client.post("/api/chat", json={"question": "q"})
    assert resp.status_code == 502
    assert "down" in resp.json()["detail"]


def test_chat_maps_llm_failure_to_502(client, monkeypatch):
    def boom(messages, **kw):
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    monkeypatch.setattr(generation, "chat", boom)
    resp = client.post("/api/chat", json={"question": "q"})
    assert resp.status_code == 502
    assert "OPENROUTER_API_KEY" in resp.json()["detail"]


def test_health_reports_point_count(client):
    assert client.get("/api/health").json() == {"status": "ok", "points": 0}


def test_index_page_is_served(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "MediRAG" in resp.text


def test_startup_requires_embed_url(monkeypatch):
    monkeypatch.delenv("EMBED_URL", raising=False)
    with pytest.raises(RuntimeError, match="EMBED_URL"):
        with TestClient(api.app):
            pass


def test_startup_requires_an_ingested_collection(monkeypatch, tmp_path):
    monkeypatch.setenv("EMBED_URL", "http://embed.test")
    monkeypatch.setenv("QDRANT_PATH", str(tmp_path / "empty"))
    with pytest.raises(RuntimeError, match="collection"):
        with TestClient(api.app):
            pass
```

- [ ] **Step 3: Run to verify failure**

Run: `python -m pytest tests/test_api.py -v`
Expected: collection error `ImportError: cannot import name 'api' from 'medical_rag'`.

- [ ] **Step 4: Implement `backend/medical_rag/api.py`**

```python
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import openai
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, StringConstraints

from medical_rag.encoders import EmbedClient, EmbedError
from medical_rag.generation import answer
from medical_rag.retrieval import RerankClient, Retriever
from medical_rag.store import COLLECTION, open_client

INDEX = Path(__file__).parent / "static" / "index.html"


@asynccontextmanager
async def lifespan(app: FastAPI):
    embed_url = os.environ.get("EMBED_URL")
    if not embed_url:
        raise RuntimeError("EMBED_URL is not set, e.g. http://localhost:8001 (serve_model/embed_server.py)")
    client = open_client(os.environ.get("QDRANT_PATH", "qdrant_data"))
    if not client.collection_exists(COLLECTION):
        client.close()
        raise RuntimeError(f"collection {COLLECTION!r} not found; run `python -m medical_rag.ingestion.ingest` first")
    rerank_url = os.environ.get("RERANK_URL")
    rerank = RerankClient(rerank_url).rerank if rerank_url else None
    app.state.client = client
    app.state.retriever = Retriever(client, EmbedClient(embed_url).embed, rerank)
    yield
    client.close()


app = FastAPI(lifespan=lifespan)


class ChatRequest(BaseModel):
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1000)]


@app.post("/api/chat")
def chat(body: ChatRequest, request: Request):
    try:
        return answer(request.app.state.retriever, body.question)
    except EmbedError as error:
        raise HTTPException(502, f"embedding service error: {error}") from error
    except (RuntimeError, openai.OpenAIError) as error:
        raise HTTPException(502, f"LLM error: {error}") from error


@app.get("/api/health")
def health(request: Request):
    return {"status": "ok", "points": request.app.state.client.count(COLLECTION, exact=True).count}


@app.get("/")
def index():
    return FileResponse(INDEX)
```

- [ ] **Step 5: Run to verify pass**

Run: `python -m pytest tests/test_api.py -v`
Expected: all pass (10 tests including the parametrized ones).

- [ ] **Step 6: Full suite and commit**

Run: `python -m pytest -q` (expect all pass)

```bash
git add backend/medical_rag/api.py backend/medical_rag/static/index.html backend/tests/test_api.py
git commit -m "feat: FastAPI chat endpoint serving the UI

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 5: Dynamic frontend

**Files:**
- Modify: `backend/medical_rag/static/index.html`

**Interfaces:**
- Consumes: `POST /api/chat` returns `{answer: string, sources: [{n, title, url, section_path, type, text, score}]}` (`score` is a number or `null`); errors return `{detail: string}`; `GET /api/health` returns `{status, points}`.

There is no JS test framework. Verification is the manual run in Step 5.

- [ ] **Step 1: Add CSS**

Insert before `@media (max-width:980px){` in the `<style>` block:

```css
    .bubble{white-space:pre-wrap;overflow-wrap:anywhere}
    .bubble.pending{color:var(--muted)}
    .bubble.error{background:#fef2f2;border-color:#fecaca;color:#991b1b}
    .send:disabled{opacity:.5;cursor:default}
    .badge{
      display:inline-block;margin-top:4px;padding:1px 8px;border-radius:999px;
      background:#f1f5f9;color:#475569;font-size:11px;font-weight:700;
    }
    .source-card h3 a{color:inherit;text-decoration:none}
    .source-card h3 a:hover{text-decoration:underline}
    .status.offline{color:#991b1b;background:#fef2f2;border-color:#fecaca}
    .status.offline .dot{background:#dc2626}
    .empty{font-size:13px;color:var(--muted)}
```

- [ ] **Step 2: Replace the status pill**

Replace `<div class="status"><span class="dot"></span>Knowledge Base Ready</div>` with:

```html
      <div class="status" id="status"><span class="dot"></span><span id="status-text">Connecting…</span></div>
```

- [ ] **Step 3: Replace the static conversation, sources and send button**

- Give the New Chat button an id: `<button class="new-chat" id="new-chat">＋ New Chat</button>`.
- Replace the whole `<div class="messages" id="messages"> … </div>` (both hard-coded rows) with `<div class="messages" id="messages"></div>`.
- Replace the send button with `<button class="send" id="send">➤</button>` (no inline `onclick`).
- Replace the whole `<section class="sources"> … </section>` (both hard-coded cards) with:

```html
      <section class="sources">
        <h2>Sources</h2>
        <div id="sources"></div>
      </section>
```

- [ ] **Step 4: Replace the whole `<script>` block**

```html
  <script>
    const $ = id => document.getElementById(id);
    const messages = $('messages'), sourcesEl = $('sources'), input = $('question'), sendBtn = $('send');
    const EMPTY_SOURCES = 'Sources appear here after you ask a question.';

    function el(tag, className, text){
      const node = document.createElement(tag);
      if(className) node.className = className;
      if(text !== undefined) node.textContent = text;
      return node;
    }

    function addRow(role, avatar){
      const row = el('div', 'row ' + role);
      const bubble = el('div', 'bubble');
      const face = el('div', 'avatar', avatar);
      if(role === 'user') row.append(bubble, face); else row.append(face, bubble);
      messages.appendChild(row);
      messages.scrollTop = messages.scrollHeight;
      return bubble;
    }

    // Text nodes and citation chips only, never innerHTML, so model output cannot inject markup.
    function renderAnswer(bubble, text, count){
      bubble.textContent = '';
      text.split(/(\[\d+\])/).forEach(part => {
        const m = part.match(/^\[(\d+)\]$/);
        if(m && +m[1] >= 1 && +m[1] <= count) bubble.appendChild(el('span', 'citation', m[1]));
        else if(part) bubble.appendChild(document.createTextNode(part));
      });
    }

    function renderSources(sources){
      sourcesEl.textContent = '';
      if(!sources.length){
        sourcesEl.appendChild(el('div', 'empty', EMPTY_SOURCES));
        return;
      }
      sources.forEach(s => {
        const card = el('div', 'source-card');
        const top = el('div', 'source-top');
        const info = el('div');
        const title = el('h3');
        const label = `[${s.n}] ${s.title}`;
        if(/^https?:\/\//.test(s.url)){
          const link = el('a', '', label);
          link.href = s.url;
          link.target = '_blank';
          link.rel = 'noopener';
          title.appendChild(link);
        } else {
          title.textContent = label;
        }
        info.append(title, el('div', 'meta', s.section_path), el('span', 'badge', s.type));
        top.append(el('div', 'doc-icon', '📄'), info);
        card.appendChild(top);
        if(s.score !== null){
          const score = el('div', 'score');
          score.append(el('span', '', 'Reranker score'), el('strong', '', s.score.toFixed(2)));
          card.appendChild(score);
        }
        const body = s.text.replace(/^\[[^\]]*\]\s*/, '');  // drop the "[Chủ đề: … | Nguồn: …]" header
        card.appendChild(el('div', 'passage', body.length > 300 ? body.slice(0, 300) + '…' : body));
        sourcesEl.appendChild(card);
      });
    }

    async function sendMessage(){
      const question = input.value.trim();
      if(!question || sendBtn.disabled) return;
      addRow('user', '👤').textContent = question;
      input.value = '';
      sendBtn.disabled = true;
      const bubble = addRow('assistant', '🩺');
      bubble.classList.add('pending');
      bubble.textContent = 'Thinking…';
      try{
        const res = await fetch('/api/chat', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({question}),
        });
        const data = await res.json().catch(() => ({}));
        if(!res.ok) throw new Error(typeof data.detail === 'string' ? data.detail : `Request failed (${res.status})`);
        bubble.classList.remove('pending');
        renderAnswer(bubble, data.answer, data.sources.length);
        renderSources(data.sources);
      }catch(err){
        bubble.classList.remove('pending');
        bubble.classList.add('error');
        bubble.textContent = err.message;
      }finally{
        sendBtn.disabled = false;
        messages.scrollTop = messages.scrollHeight;
        input.focus();
      }
    }

    sendBtn.addEventListener('click', sendMessage);
    input.addEventListener('keydown', e => {
      if(e.key === 'Enter' && !e.isComposing) sendMessage();  // isComposing: Vietnamese IMEs confirm text with Enter
    });
    $('new-chat').addEventListener('click', () => {
      messages.textContent = '';
      renderSources([]);
    });

    renderSources([]);
    fetch('/api/health')
      .then(r => r.ok ? r.json() : Promise.reject())
      .then(h => { $('status-text').textContent = `Knowledge base ready · ${h.points} chunks`; })
      .catch(() => {
        $('status').classList.add('offline');
        $('status-text').textContent = 'Offline';
      });
  </script>
```

- [ ] **Step 5: Manual end-to-end check**

1. Start the embed server (own venv): from `serve_model/`, `.venv\Scripts\activate`, then `uvicorn embed_server:app --port 8001 --env-file ../.env`.
2. Check the collection matches the current embed model. From `backend/`:
   `python -c "from medical_rag.store import *; c=open_client('qdrant_data'); print(c.count(COLLECTION, exact=True).count)"`
   If it errors or prints 0, ingest first: `python -m medical_rag.ingestion.ingest --embed-url http://localhost:8001 --recreate` (needs the embed server; takes a while). The memory note says the collection must be rebuilt after the switch to `embeddinggemma-300m`.
3. Start the API from `backend/` (the ingest process must have exited: local Qdrant holds a file lock):
   `set EMBED_URL=http://localhost:8001` (or use the `.env` via `--env-file`), then
   `uvicorn medical_rag.api:app --env-file ../.env --port 8000`
4. Open `http://localhost:8000/`. Verify:
   - Status pill shows "Knowledge base ready · N chunks".
   - Ask "Triệu chứng của bệnh tiểu đường là gì?": a "Thinking…" bubble, then an answer with `[n]` chips and matching source cards (article link opens in a new tab, no "Reranker score" row while `RERANK_URL` is unset).
   - Type `<img src=x onerror=alert(1)>` as the question: it appears as text, no alert.
   - Stop the embed server and ask again: a red error bubble, the button re-enables.
   - New Chat clears messages and sources.
5. Optional, with a running Kaggle reranker: set `RERANK_URL` and restart; source cards show "Reranker score".

- [ ] **Step 6: Commit**

```bash
git add backend/medical_rag/static/index.html
git commit -m "feat: render chat answers and sources dynamically in the UI

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Self-Review

- **Spec coverage:** optional rerank + fallback (Task 2), `task="query"` (Task 1), hybrid RRF + dense-only fallback (Task 2), prompt/`answer`/empty-hits (Task 3), API routes, 422/502, lifespan failures, `/` (Task 4), UI rendering, XSS fix, citation chips, score only when non-null, status pill, New Chat (Task 5), deps and package-data (Task 1), future work is in the spec only. The spec's "UI shows the type badge, section_path, 300-char snippet" is covered in `renderSources`.
- **Placeholders:** none. The one conditional note in Task 2 Step 4 gives a concrete diagnosis path.
- **Type consistency:** `Hit` fields, `Retriever(client, embed, rerank)`, `answer(retriever, question, k)`, source dict keys (`n,title,url,section_path,type,text,score`) match between Tasks 2–5 and the JS reads the same keys.
