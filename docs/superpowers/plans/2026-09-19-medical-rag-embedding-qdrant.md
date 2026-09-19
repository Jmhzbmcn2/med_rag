# Medical RAG Embedding and Qdrant Ingestion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn chunk dicts from `chunk_article` into points in a local on-disk Qdrant collection, each with a dense vector (from the Kaggle `/embed` endpoint) and a sparse BM25 vector, and prove it with an acceptance script.

**Architecture:** `backend/medical_rag/` package. `encoders.py` (pyvi segmentation, BM25 sparse encoder, resilient HTTP embed client) and `store.py` (Qdrant collection, per-article replace) are shared with the future retrieval stage; `ingestion/ingest.py` orchestrates: chunk everything, fit BM25, then per article embed first and only then delete-and-upsert. The existing chunking code moves into the package first.

**Tech Stack:** Python 3.13, `qdrant-client` (local on-disk mode), `pyvi`, `requests`, `transformers` (tokenizer only), `langchain-text-splitters`, `pandas`, `pytest`.

**Spec:** `docs/superpowers/specs/2026-09-19-medical-rag-embedding-qdrant-design.md`

## Global Constraints

- Work on git branch `feat/embedding-qdrant` (already created; do not switch branches). Do not touch the untracked `.claude/` and `serve_model/` directories.
- Run every command from the **repository root** (`C:\Users\VUDUYLINH\PycharmProjects\Medical_RAG`) so the relative paths `data/`, `test_case/`, `qdrant_data/` resolve. Test command form: `python -m pytest backend/tests/<file> -v`.
- Shell is Git Bash on Windows. For scripts that print Vietnamese text, prefix `PYTHONIOENCODING=utf-8`.
- Dense vector size `768`, cosine. Collection name `medical_rag`. Named vectors `dense` and `sparse`; sparse uses `Modifier.IDF`.
- BM25 defaults `k1 = 1.2`, `b = 0.75`. Term index = `zlib.crc32(term.encode("utf-8")) & 0xFFFFFFFF`. Query weight `1.0` per distinct term.
- `EmbedClient` defaults: `batch_size=32`, `timeout=60.0`, `retries=3`, `backoff=1.0`. Transient HTTP statuses `{502, 503, 504, 530}` plus `524` (which halves the batch); connection errors and timeouts are transient. Any other non-200 status fails immediately.
- All file reads/writes use `encoding="utf-8"` explicitly.
- Tests never mock Qdrant or HTTP: use Qdrant `:memory:` (`open_client(None)`) and a real `ThreadingHTTPServer` on `127.0.0.1`. The embedding endpoint itself is replaced only by an injected fake embedder callable in ingest/validate tests.
- Tunnel URLs (`*.trycloudflare.com`) come from `EMBED_URL` / `--embed-url` only. Never hardcode or commit one.
- Test output must be pristine. The only tolerated warning is the pre-existing `PytestDeprecationWarning` about `asyncio_default_fixture_loop_scope`; a Qdrant `UserWarning` about payload indexes must not appear.
- Do not pre-commit hit-rate thresholds in `validate_ingest.py`; they are set in Task 8 from a real run.

## Execution Model (for the controller)

Use superpowers:subagent-driven-development. One fresh implementer per task, strictly **sequential** (every task edits files the next one imports; never dispatch two implementers at once). After each task: task review (spec + quality), fix loop if needed. After Task 7: final whole-branch review on the most capable model. Task 8 needs a live Kaggle tunnel and a human; the controller runs it with the user, it is not dispatched.

| Task | Deliverable | Depends on | Implementer model | Why this tier |
|---|---|---|---|---|
| 1 | `backend/` layout, `pyproject.toml`, imports fixed, old tests green | none | sonnet | path/import fixes and two long verification runs need judgment |
| 2 | `segment`, `embed_input`, `DENSE_DIM` | 1 | sonnet | complete code given |
| 3 | `Bm25Encoder` | 2 | sonnet | complete code given |
| 4 | `EmbedClient` + `EmbedError` | 3 | sonnet | threaded test server, retry logic, likely debugging |
| 5 | `store.py` | 4 | sonnet | complete code given |
| 6 | `ingest.py` (`ingest`, CLI `main`) | 5 | sonnet | orchestration and CLI edge cases |
| 7 | `validate_ingest.py` + tests | 6 | sonnet | script plus in-memory Qdrant tests |
| 8 | live ingest, measured hit@5, floors | 7 | controller + user | needs live Kaggle |

User ruling (2026-09-19): every subagent runs on `sonnet`, never `haiku`. Task reviewers and scoped re-reviews `sonnet`; final whole-branch reviewer `opus`.

---

### Task 1: Restructure into `backend/` and add `pyproject.toml`

**Files:**
- Move: `chunking.py` → `backend/medical_rag/ingestion/chunking.py`
- Move: `test_chunking.py` → `backend/tests/test_chunking.py`
- Move: `validate_chunking.py` → `backend/scripts/validate_chunking.py`
- Create: `backend/pyproject.toml`, `backend/medical_rag/__init__.py` (empty), `backend/medical_rag/ingestion/__init__.py` (empty)
- Modify: `.gitignore`, the import lines of the two moved Python files

**Interfaces:**
- Produces: importable package `medical_rag` with `medical_rag.ingestion.chunking` exposing the unchanged functions `parse_article`, `split_sections`, `inject_context_header`, `guard_rail_split`, `chunk_article`, `iter_articles`; pytest config so `python -m pytest backend/tests` works from the repo root with `medical_rag` and the `scripts/` modules importable.

This is a refactor, not TDD: prove nothing broke by running the old tests and the old acceptance script before and after.

- [ ] **Step 1: Confirm the baseline is green before moving anything**

Run: `git branch --show-current && git status --short && python -m pytest test_chunking.py -q`
Expected: branch `feat/embedding-qdrant`; status shows only `?? .claude/` and `?? serve_model/`; `5 passed`.

- [ ] **Step 2: Create directories and move the three files with git**

```bash
mkdir -p backend/medical_rag/ingestion backend/tests backend/scripts
git mv chunking.py backend/medical_rag/ingestion/chunking.py
git mv test_chunking.py backend/tests/test_chunking.py
git mv validate_chunking.py backend/scripts/validate_chunking.py
```

- [ ] **Step 3: Create the package files**

`backend/medical_rag/__init__.py` and `backend/medical_rag/ingestion/__init__.py`: empty files.

`backend/pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "medical-rag"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "langchain-text-splitters",
    "transformers",
    "pandas",
    "qdrant-client>=1.10",
    "pyvi",
    "requests",
]

[project.optional-dependencies]
dev = ["pytest"]

[tool.setuptools.packages.find]
include = ["medical_rag*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = [".", "scripts"]
```

- [ ] **Step 4: Fix the two import lines**

In `backend/tests/test_chunking.py`, line 1 currently starts `from chunking import ...`. Change `from chunking import` to `from medical_rag.ingestion.chunking import`, keeping the imported names exactly as they are.

In `backend/scripts/validate_chunking.py`, change `from chunking import chunk_article, iter_articles` to `from medical_rag.ingestion.chunking import chunk_article, iter_articles`.

- [ ] **Step 5: Ignore the local index and build artifacts**

Append two lines to `.gitignore`:

```
qdrant_data/
*.egg-info/
```

- [ ] **Step 6: Install the package in editable mode**

Run: `python -m pip install -e backend`
Expected: ends with `Successfully installed ... medical-rag-0.1.0 ... pyvi-...` (dependencies already present are reported as satisfied). If a wheel build fails for `python-crfsuite`, stop and report BLOCKED with the pip output.

- [ ] **Step 7: Verify the moved tests pass**

Run: `python -m pytest backend/tests -q`
Expected: `5 passed`, no warning other than the tolerated `PytestDeprecationWarning`.

- [ ] **Step 8: Verify the chunking acceptance script still passes**

Run: `PYTHONIOENCODING=utf-8 python backend/scripts/validate_chunking.py`
Expected (takes a few minutes): four lines starting `OK:` for `disease`, `medicine`, `drug`, `body-part`, and exit code `0` (`echo $?` prints `0`). No `FAIL:` line.

- [ ] **Step 9: Commit**

```bash
git add .gitignore backend
git status --short
git commit -m "refactor: move chunking into backend/medical_rag package"
```

Expected `git status --short` before the commit: renames (`R`) for the three moved files, added `backend/pyproject.toml` and the two `__init__.py`, modified `.gitignore`; no `egg-info` and no `__pycache__` entries.

---

### Task 2: `segment`, `embed_input`, `DENSE_DIM`

**Files:**
- Create: `backend/medical_rag/encoders.py`
- Create: `backend/tests/test_encoders.py`

**Interfaces:**
- Produces:
  - `DENSE_DIM: int = 768`
  - `segment(text: str) -> str` — `pyvi.ViTokenizer.tokenize`; the one word-segmentation function for documents and queries.
  - `embed_input(chunk: dict) -> str` — returns `chunk["text"]` unchanged.

- [ ] **Step 1: Write the failing test**

`backend/tests/test_encoders.py`:

```python
from medical_rag.encoders import DENSE_DIM, embed_input, segment


def test_dense_dim_matches_the_embedding_model():
    assert DENSE_DIM == 768


def test_segment_joins_compounds_and_is_deterministic():
    assert segment("Bệnh nhân bị đau đầu.") == "Bệnh_nhân bị đau_đầu ."
    assert segment("Bệnh nhân bị đau đầu.") == segment("Bệnh nhân bị đau đầu.")
    assert segment("") == ""


def test_embed_input_is_the_chunk_text_unchanged():
    chunk = {"text": "[Chủ đề: T | Nguồn: U]\n\nThân bài."}
    assert embed_input(chunk) == "[Chủ đề: T | Nguồn: U]\n\nThân bài."
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest backend/tests/test_encoders.py -v`
Expected: collection ERROR `ModuleNotFoundError: No module named 'medical_rag.encoders'`.

- [ ] **Step 3: Write minimal implementation**

`backend/medical_rag/encoders.py`:

```python
from pyvi import ViTokenizer

DENSE_DIM = 768


def segment(text: str) -> str:
    return ViTokenizer.tokenize(text)


def embed_input(chunk: dict) -> str:
    return chunk["text"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest backend/tests/test_encoders.py -v`
Expected: `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add backend/medical_rag/encoders.py backend/tests/test_encoders.py
git commit -m "feat: add pyvi segmentation and embed_input"
```

---

### Task 3: `Bm25Encoder`

**Files:**
- Modify: `backend/medical_rag/encoders.py`
- Modify: `backend/tests/test_encoders.py`

**Interfaces:**
- Consumes: `segment` conventions from Task 2 (input to the encoder is an already-segmented string).
- Produces:
  - `terms(segmented: str) -> list[str]` — NFC-normalize, lowercase, regex `\w+`.
  - `term_index(term: str) -> int` — `zlib.crc32(term.encode("utf-8")) & 0xFFFFFFFF`.
  - `Bm25Encoder(k1: float = 1.2, b: float = 0.75)` with `.avgdl` (None until fitted), `.fit(segmented_docs: list[str]) -> None`, `.encode_doc(segmented: str) -> tuple[list[int], list[float]]`, `.encode_query(segmented: str) -> tuple[list[int], list[float]]`. Both encode methods return indices in ascending order, unique. `fit` raises `ValueError` on an empty corpus or a corpus with zero terms; `encode_doc` before `fit` raises `RuntimeError`; an empty doc returns `([], [])`.

- [ ] **Step 1: Write the failing tests**

In `backend/tests/test_encoders.py`, replace the first line (the import) with:

```python
import pytest

from medical_rag.encoders import (
    DENSE_DIM,
    Bm25Encoder,
    embed_input,
    segment,
    term_index,
    terms,
)
```

and append:

```python
def test_terms_normalizes_nfc_lowercases_and_keeps_compounds():
    assert terms("Bệnh_Nhân đau 325 mg") == ["bệnh_nhân", "đau", "325", "mg"]
    assert terms("e\u0301") == terms("\u00e9") == ["\u00e9"]
    assert terms("") == []


def test_term_index_is_stable_u32():
    assert term_index("đau") == term_index("đau")
    assert 0 <= term_index("đau") <= 0xFFFFFFFF
    assert term_index("đau") != term_index("sốt")


def test_bm25_doc_weights_match_hand_computed_values():
    enc = Bm25Encoder(k1=1.2, b=0.75)
    enc.fit(["a b", "a c c"])  # avgdl = (2 + 3) / 2 = 2.5

    indices, values = enc.encode_doc("a b")  # dl=2, norm=1.2*(0.25+0.75*2/2.5)=1.02
    weights = dict(zip(indices, values))
    assert weights[term_index("a")] == pytest.approx(2.2 / 2.02, abs=1e-6)
    assert weights[term_index("b")] == pytest.approx(2.2 / 2.02, abs=1e-6)

    indices, values = enc.encode_doc("a c c")  # dl=3, norm=1.2*(0.25+0.75*3/2.5)=1.38
    weights = dict(zip(indices, values))
    assert weights[term_index("a")] == pytest.approx(2.2 / 2.38, abs=1e-6)
    assert weights[term_index("c")] == pytest.approx(4.4 / 3.38, abs=1e-6)


def test_bm25_longer_doc_gets_smaller_weight_for_equal_tf():
    enc = Bm25Encoder()
    enc.fit(["a b", "a b c d e f g h"])
    _, short = enc.encode_doc("a b")
    long_idx, long_vals = enc.encode_doc("a b c d e f g h")
    long_weights = dict(zip(long_idx, long_vals))
    assert long_weights[term_index("a")] < short[0]


def test_bm25_indices_are_sorted_unique_and_empty_doc_is_empty():
    enc = Bm25Encoder()
    enc.fit(["x y z"])
    indices, values = enc.encode_doc("z y x x")
    assert indices == sorted(set(indices))
    assert len(indices) == len(values) == 3
    assert enc.encode_doc("") == ([], [])


def test_bm25_query_uses_weight_one_per_distinct_term_without_fit():
    enc = Bm25Encoder()
    indices, values = enc.encode_query("đau đau sốt")
    assert indices == sorted({term_index("đau"), term_index("sốt")})
    assert values == [1.0, 1.0]


def test_bm25_errors_on_empty_corpus_and_encode_before_fit():
    with pytest.raises(ValueError):
        Bm25Encoder().fit([])
    with pytest.raises(ValueError):
        Bm25Encoder().fit(["", "  "])
    with pytest.raises(RuntimeError):
        Bm25Encoder().encode_doc("a")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest backend/tests/test_encoders.py -v`
Expected: collection ERROR `ImportError: cannot import name 'Bm25Encoder'`.

- [ ] **Step 3: Write the implementation**

In `backend/medical_rag/encoders.py`, replace the first line (`from pyvi import ViTokenizer`) with:

```python
import re
import unicodedata
import zlib
from collections import Counter

from pyvi import ViTokenizer
```

and append at the end of the file:

```python


_TERM_RE = re.compile(r"\w+")


def terms(segmented: str) -> list[str]:
    return _TERM_RE.findall(unicodedata.normalize("NFC", segmented).lower())


def term_index(term: str) -> int:
    return zlib.crc32(term.encode("utf-8")) & 0xFFFFFFFF


class Bm25Encoder:
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.avgdl = None

    def fit(self, segmented_docs: list[str]) -> None:
        lengths = [len(terms(doc)) for doc in segmented_docs]
        if not lengths or sum(lengths) == 0:
            raise ValueError("cannot fit BM25 on an empty corpus")
        self.avgdl = sum(lengths) / len(lengths)

    def encode_doc(self, segmented: str) -> tuple[list[int], list[float]]:
        if self.avgdl is None:
            raise RuntimeError("call fit() before encode_doc()")
        tokens = terms(segmented)
        if not tokens:
            return [], []
        dl = len(tokens)
        tf = Counter(term_index(t) for t in tokens)
        norm = self.k1 * (1 - self.b + self.b * dl / self.avgdl)
        indices = sorted(tf)
        return indices, [tf[i] * (self.k1 + 1) / (tf[i] + norm) for i in indices]

    def encode_query(self, segmented: str) -> tuple[list[int], list[float]]:
        indices = sorted({term_index(t) for t in terms(segmented)})
        return indices, [1.0] * len(indices)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest backend/tests/test_encoders.py -v`
Expected: `10 passed`.

- [ ] **Step 5: Commit**

```bash
git add backend/medical_rag/encoders.py backend/tests/test_encoders.py
git commit -m "feat: add BM25 sparse encoder"
```

---

### Task 4: `EmbedClient` and `EmbedError`

**Files:**
- Modify: `backend/medical_rag/encoders.py`
- Modify: `backend/tests/test_encoders.py`

**Interfaces:**
- Consumes: `DENSE_DIM` (Task 2).
- Produces:
  - `EmbedError(Exception)`
  - `EmbedClient(base_url: str, batch_size: int = 32, timeout: float = 60.0, retries: int = 3, backoff: float = 1.0)`
    - `.embed(segmented_texts: list[str]) -> list[list[float]]` — `POST {base_url}/embed` with JSON `{"texts": [...]}`, expects `{"embeddings": [[768 floats], ...]}`; batches by `batch_size`; empty input makes no request. On HTTP `524` with more than one text in the batch, halves the batch (`max(1, len(batch)//2)`) and re-sends without using a retry; on `524` with one text, or on `502/503/504/530`, or on connection error/timeout, retries up to `retries` attempts total with sleeps `backoff * 2**(attempt-1)` before attempts 2, 3, ...; raises `EmbedError("gave up after N attempts: ...")`. Any other non-200 status raises `EmbedError("HTTP <code>: ...")` immediately. A malformed body, a wrong vector count, or a vector length other than `DENSE_DIM` raises `EmbedError` (message contains `dimension` for the size case).
    - `.health_check() -> None` — embeds one short text; on `EmbedError` re-raises `EmbedError("tunnel not responding, check the Kaggle notebook and update EMBED_URL: <detail>")`.

- [ ] **Step 1: Write the failing tests**

In `backend/tests/test_encoders.py`, replace the import block at the top (from the first line through the closing `)` of the `from medical_rag.encoders import (...)`) with:

```python
import contextlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from medical_rag.encoders import (
    DENSE_DIM,
    Bm25Encoder,
    EmbedClient,
    EmbedError,
    embed_input,
    segment,
    term_index,
    terms,
)
```

and append:

```python
def _ok_body(texts):
    return {"embeddings": [[0.5] * DENSE_DIM for _ in texts]}


@contextlib.contextmanager
def fake_embed_server(script):
    """script(texts, request_number) -> (status_code, json_body)"""
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            texts = body["texts"]
            status, payload = script(texts, len(calls))
            calls.append(len(texts))
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", calls
    finally:
        server.shutdown()
        server.server_close()


def test_embed_client_batches_requests():
    with fake_embed_server(lambda texts, n: (200, _ok_body(texts))) as (url, calls):
        vectors = EmbedClient(url, batch_size=2).embed(["a", "b", "c", "d", "e"])
    assert len(vectors) == 5
    assert calls == [2, 2, 1]


def test_embed_client_empty_input_makes_no_request():
    with fake_embed_server(lambda texts, n: (200, _ok_body(texts))) as (url, calls):
        assert EmbedClient(url).embed([]) == []
    assert calls == []


def test_embed_client_halves_batch_after_gateway_timeout():
    def script(texts, n):
        return (524, {}) if n == 0 else (200, _ok_body(texts))

    with fake_embed_server(script) as (url, calls):
        vectors = EmbedClient(url, batch_size=4).embed(["a", "b", "c", "d"])
    assert len(vectors) == 4
    assert calls == [4, 2, 2]


def test_embed_client_keeps_halving_down_to_single_texts():
    def script(texts, n):
        return (524, {}) if len(texts) > 1 else (200, _ok_body(texts))

    with fake_embed_server(script) as (url, calls):
        vectors = EmbedClient(url, batch_size=4).embed(["a", "b", "c"])
    assert len(vectors) == 3
    assert calls == [3, 1, 1, 1]


def test_embed_client_retries_transient_errors_then_succeeds():
    def script(texts, n):
        return (530, {}) if n < 2 else (200, _ok_body(texts))

    with fake_embed_server(script) as (url, calls):
        vectors = EmbedClient(url, retries=3, backoff=0).embed(["a"])
    assert len(vectors) == 1
    assert calls == [1, 1, 1]


def test_embed_client_gives_up_after_retries():
    with fake_embed_server(lambda texts, n: (530, {})) as (url, calls):
        with pytest.raises(EmbedError, match="gave up after 3 attempts"):
            EmbedClient(url, retries=3, backoff=0).embed(["a"])
    assert calls == [1, 1, 1]


def test_embed_client_does_not_retry_client_errors():
    with fake_embed_server(lambda texts, n: (400, {"detail": "bad"})) as (url, calls):
        with pytest.raises(EmbedError, match="HTTP 400"):
            EmbedClient(url, retries=3, backoff=0).embed(["a"])
    assert calls == [1]


def test_embed_client_rejects_wrong_dimension():
    with fake_embed_server(lambda texts, n: (200, {"embeddings": [[0.1, 0.2, 0.3]]})) as (url, _):
        with pytest.raises(EmbedError, match="dimension"):
            EmbedClient(url).embed(["a"])


def test_embed_client_retries_connection_errors_then_raises():
    with pytest.raises(EmbedError, match="gave up after 2 attempts"):
        EmbedClient("http://127.0.0.1:1", retries=2, backoff=0, timeout=2).embed(["a"])


def test_health_check_success_and_failure_message():
    with fake_embed_server(lambda texts, n: (200, _ok_body(texts))) as (url, _):
        EmbedClient(url).health_check()
    with pytest.raises(EmbedError, match="update EMBED_URL"):
        EmbedClient("http://127.0.0.1:1", retries=1, backoff=0, timeout=2).health_check()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest backend/tests/test_encoders.py -v`
Expected: collection ERROR `ImportError: cannot import name 'EmbedClient'`.

- [ ] **Step 3: Write the implementation**

In `backend/medical_rag/encoders.py`, replace everything from the first line through the line `DENSE_DIM = 768` with:

```python
import re
import time
import unicodedata
import zlib
from collections import Counter

import requests
from pyvi import ViTokenizer

DENSE_DIM = 768
_TRANSIENT_STATUS = {502, 503, 504, 530}  # Cloudflare error 1033 arrives as HTTP 530
_GATEWAY_TIMEOUT = 524
```

and append at the end of the file:

```python


class EmbedError(Exception):
    pass


class _GatewayTimeout(Exception):
    pass


class EmbedClient:
    def __init__(
        self,
        base_url: str,
        batch_size: int = 32,
        timeout: float = 60.0,
        retries: int = 3,
        backoff: float = 1.0,
    ):
        self.url = base_url.rstrip("/") + "/embed"
        self.batch_size = batch_size
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff

    def embed(self, segmented_texts: list[str]) -> list[list[float]]:
        vectors = []
        i, size = 0, self.batch_size
        while i < len(segmented_texts):
            batch = segmented_texts[i:i + size]
            try:
                vectors.extend(self._post(batch))
            except _GatewayTimeout:
                size = max(1, len(batch) // 2)
                continue
            i += len(batch)
        return vectors

    def health_check(self) -> None:
        try:
            self.embed(["xin chào"])
        except EmbedError as e:
            raise EmbedError(
                f"tunnel not responding, check the Kaggle notebook and update EMBED_URL: {e}"
            ) from e

    def _post(self, batch: list[str]) -> list[list[float]]:
        last_error = None
        for attempt in range(self.retries):
            if attempt:
                time.sleep(self.backoff * 2 ** (attempt - 1))
            try:
                resp = requests.post(self.url, json={"texts": batch}, timeout=self.timeout)
            except (requests.ConnectionError, requests.Timeout) as e:
                last_error = e
                continue
            if resp.status_code == _GATEWAY_TIMEOUT and len(batch) > 1:
                raise _GatewayTimeout()
            if resp.status_code == _GATEWAY_TIMEOUT or resp.status_code in _TRANSIENT_STATUS:
                last_error = EmbedError(f"HTTP {resp.status_code}")
                continue
            if resp.status_code != 200:
                raise EmbedError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            try:
                vectors = resp.json()["embeddings"]
            except (ValueError, KeyError) as e:
                raise EmbedError(f"malformed /embed response: {e}") from e
            if len(vectors) != len(batch) or any(len(v) != DENSE_DIM for v in vectors):
                raise EmbedError(
                    f"expected {len(batch)} vectors of dimension {DENSE_DIM} from /embed"
                )
            return vectors
        raise EmbedError(f"gave up after {self.retries} attempts: {last_error}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest backend/tests/test_encoders.py -v`
Expected: `20 passed` (10 from Tasks 2-3 plus 10 new), well under a minute.

- [ ] **Step 5: Commit**

```bash
git add backend/medical_rag/encoders.py backend/tests/test_encoders.py
git commit -m "feat: add resilient Kaggle embed client"
```

---

### Task 5: `store.py`

**Files:**
- Create: `backend/medical_rag/store.py`
- Create: `backend/tests/test_store.py`

**Interfaces:**
- Consumes: `DENSE_DIM`, `Bm25Encoder`, `segment` from `medical_rag.encoders`.
- Produces:
  - `COLLECTION: str = "medical_rag"`
  - `open_client(path: str | None) -> QdrantClient` — `None` gives `:memory:`.
  - `ensure_collection(client, recreate: bool = False) -> None` — creates the collection (dense 768 cosine + sparse IDF + keyword payload indexes on `type`, `article_slug`, suppressing Qdrant's local-mode "Payload indexes have no effect" `UserWarning`); if it exists: with `recreate=True` drops and rebuilds, otherwise returns silently when the `dense` size is 768 and raises `ValueError` (message contains `dense size <n>`) when it differs or there is no named `dense` vector.
  - `build_point(chunk: dict, dense: list[float], sparse: tuple[list[int], list[float]]) -> models.PointStruct` — id `chunk["id"]`; payload = every chunk key except `id`; vector dict has `dense`, plus `sparse` only when the indices list is non-empty.
  - `replace_article(client, article_type: str, article_slug: str, points: list[models.PointStruct]) -> None` — deletes points whose payload `type` and `article_slug` both match, then upserts `points` if there are any.

- [ ] **Step 1: Write the failing tests**

`backend/tests/test_store.py`:

```python
import uuid

import pytest
from qdrant_client import models

from medical_rag.encoders import DENSE_DIM, Bm25Encoder, segment
from medical_rag.store import (
    COLLECTION,
    build_point,
    ensure_collection,
    open_client,
    replace_article,
)


def _dense(axis: int) -> list[float]:
    vector = [0.0] * DENSE_DIM
    vector[axis] = 1.0
    return vector


def _chunk(name: str, article_type: str = "drug", slug: str = "a", text: str = "nội dung") -> dict:
    return {
        "id": str(uuid.uuid5(uuid.NAMESPACE_URL, name)),
        "type": article_type,
        "article_slug": slug,
        "article_title": "T",
        "article_url": "U",
        "section_path": "S",
        "text": text,
        "token_count": 3,
        "split_part": None,
    }


def _count(client) -> int:
    return client.count(COLLECTION, exact=True).count


def test_ensure_collection_creates_and_is_idempotent():
    client = open_client(None)
    ensure_collection(client)
    ensure_collection(client)
    params = client.get_collection(COLLECTION).config.params
    assert params.vectors["dense"].size == DENSE_DIM
    assert params.vectors["dense"].distance == models.Distance.COSINE
    assert params.sparse_vectors["sparse"].modifier == models.Modifier.IDF


def test_ensure_collection_raises_on_dense_size_mismatch():
    client = open_client(None)
    client.create_collection(
        COLLECTION,
        vectors_config={"dense": models.VectorParams(size=4, distance=models.Distance.COSINE)},
    )
    with pytest.raises(ValueError, match="dense size 4"):
        ensure_collection(client)


def test_ensure_collection_recreate_drops_existing_points():
    client = open_client(None)
    ensure_collection(client)
    client.upsert(COLLECTION, points=[build_point(_chunk("x"), _dense(0), ([], []))])
    assert _count(client) == 1
    ensure_collection(client, recreate=True)
    assert _count(client) == 0


def test_build_point_payload_excludes_id_and_sparse_is_omitted_when_empty():
    chunk = _chunk("x")
    with_sparse = build_point(chunk, _dense(0), ([5, 9], [0.5, 1.5]))
    assert with_sparse.id == chunk["id"]
    assert "id" not in with_sparse.payload
    assert with_sparse.payload["article_slug"] == "a"
    assert set(with_sparse.vector) == {"dense", "sparse"}
    assert with_sparse.vector["sparse"].indices == [5, 9]

    without_sparse = build_point(chunk, _dense(0), ([], []))
    assert set(without_sparse.vector) == {"dense"}


def test_replace_article_replaces_only_that_article_and_is_idempotent():
    client = open_client(None)
    ensure_collection(client)
    a = [_chunk("a1", slug="a"), _chunk("a2", slug="a"), _chunk("a3", slug="a")]
    b = [_chunk("b1", slug="b")]
    same_slug_other_type = [_chunk("c1", article_type="disease", slug="a")]
    for article_type, slug, chunks in (
        ("drug", "a", a),
        ("drug", "b", b),
        ("disease", "a", same_slug_other_type),
    ):
        replace_article(client, article_type, slug, [build_point(c, _dense(0), ([], [])) for c in chunks])
    assert _count(client) == 5

    replace_article(client, "drug", "a", [build_point(a[0], _dense(0), ([], []))])
    assert _count(client) == 3  # a shrank from 3 to 1; b and disease/a untouched

    replace_article(client, "drug", "a", [build_point(a[0], _dense(0), ([], []))])
    assert _count(client) == 3  # re-running does not duplicate

    replace_article(client, "drug", "a", [])
    assert _count(client) == 2  # an article that became empty is removed


def test_sparse_query_ranks_rare_term_above_common_term():
    docs = ["bệnh_nhân đau", "bệnh_nhân sốt", "bệnh_nhân ho"]
    encoder = Bm25Encoder()
    encoder.fit(docs)
    client = open_client(None)
    ensure_collection(client)
    chunks = [_chunk(f"d{i}", slug=f"s{i}") for i in range(3)]
    client.upsert(
        COLLECTION,
        points=[build_point(c, _dense(i), encoder.encode_doc(d)) for i, (c, d) in enumerate(zip(chunks, docs))],
    )

    indices, values = encoder.encode_query("bệnh_nhân sốt")
    result = client.query_points(
        COLLECTION, query=models.SparseVector(indices=indices, values=values), using="sparse", limit=3
    )
    assert result.points[0].payload["article_slug"] == "s1"  # the only doc with the rare term "sốt"


def test_hybrid_query_with_rrf_fusion_returns_the_doc_both_signals_agree_on():
    docs = ["bệnh_nhân đau", "bệnh_nhân sốt", "bệnh_nhân ho"]
    encoder = Bm25Encoder()
    encoder.fit(docs)
    client = open_client(None)
    ensure_collection(client)
    chunks = [_chunk(f"h{i}", slug=f"s{i}") for i in range(3)]
    client.upsert(
        COLLECTION,
        points=[build_point(c, _dense(i), encoder.encode_doc(d)) for i, (c, d) in enumerate(zip(chunks, docs))],
    )

    indices, values = encoder.encode_query(segment("sốt"))
    result = client.query_points(
        COLLECTION,
        prefetch=[
            models.Prefetch(query=_dense(1), using="dense", limit=3),
            models.Prefetch(query=models.SparseVector(indices=indices, values=values), using="sparse", limit=3),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=3,
    )
    assert result.points[0].payload["article_slug"] == "s1"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest backend/tests/test_store.py -v`
Expected: collection ERROR `ModuleNotFoundError: No module named 'medical_rag.store'`.

- [ ] **Step 3: Write the implementation**

`backend/medical_rag/store.py`:

```python
import warnings

from qdrant_client import QdrantClient, models

from medical_rag.encoders import DENSE_DIM

COLLECTION = "medical_rag"
_INDEXED_FIELDS = ("type", "article_slug")


def open_client(path: str | None) -> QdrantClient:
    return QdrantClient(":memory:") if path is None else QdrantClient(path=path)


def ensure_collection(client: QdrantClient, recreate: bool = False) -> None:
    if client.collection_exists(COLLECTION):
        if not recreate:
            vectors = client.get_collection(COLLECTION).config.params.vectors
            if not isinstance(vectors, dict) or "dense" not in vectors:
                raise ValueError(f"collection {COLLECTION!r} has no named 'dense' vector")
            if vectors["dense"].size != DENSE_DIM:
                raise ValueError(
                    f"collection {COLLECTION!r} has dense size {vectors['dense'].size}, "
                    f"expected {DENSE_DIM}; pass recreate=True to rebuild it"
                )
            return
        client.delete_collection(COLLECTION)

    client.create_collection(
        COLLECTION,
        vectors_config={"dense": models.VectorParams(size=DENSE_DIM, distance=models.Distance.COSINE)},
        sparse_vectors_config={"sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )
    with warnings.catch_warnings():
        # local mode ignores payload indexes; they matter once this moves to a Qdrant server
        warnings.filterwarnings("ignore", message="Payload indexes have no effect")
        for field in _INDEXED_FIELDS:
            client.create_payload_index(COLLECTION, field, models.PayloadSchemaType.KEYWORD)


def build_point(
    chunk: dict, dense: list[float], sparse: tuple[list[int], list[float]]
) -> models.PointStruct:
    vector = {"dense": dense}
    indices, values = sparse
    if indices:
        vector["sparse"] = models.SparseVector(indices=indices, values=values)
    payload = {key: value for key, value in chunk.items() if key != "id"}
    return models.PointStruct(id=chunk["id"], vector=vector, payload=payload)


def replace_article(
    client: QdrantClient,
    article_type: str,
    article_slug: str,
    points: list[models.PointStruct],
) -> None:
    article_filter = models.Filter(
        must=[
            models.FieldCondition(key="type", match=models.MatchValue(value=article_type)),
            models.FieldCondition(key="article_slug", match=models.MatchValue(value=article_slug)),
        ]
    )
    client.delete(COLLECTION, points_selector=models.FilterSelector(filter=article_filter))
    if points:
        client.upsert(COLLECTION, points=points)
```

- [ ] **Step 4: Run tests to verify they pass, with warnings as errors**

Run: `python -m pytest backend/tests/test_store.py -v -W error::UserWarning`
Expected: `7 passed`. (`-W error::UserWarning` proves the payload-index warning is suppressed.)

- [ ] **Step 5: Commit**

```bash
git add backend/medical_rag/store.py backend/tests/test_store.py
git commit -m "feat: add Qdrant store with per-article replacement"
```

---

### Task 6: `ingest.py` (orchestration and CLI)

**Files:**
- Create: `backend/medical_rag/ingestion/ingest.py`
- Create: `backend/tests/test_ingest.py`

**Interfaces:**
- Consumes: `chunk_article`, `iter_articles` (chunking); `Bm25Encoder`, `EmbedClient`, `EmbedError`, `embed_input`, `segment` (encoders); `COLLECTION`, `build_point`, `ensure_collection`, `open_client`, `replace_article` (store).
- Produces:
  - `TOKENIZER_NAME = "dangvantuan/vietnamese-embedding"`
  - `IngestReport` dataclass: `articles_ok: int = 0`, `articles_failed: list[str]`, `points_upserted: int = 0`, `expected_chunks: int = 0`.
  - `ingest(data_dir: str, client, embedder: Callable[[list[str]], list[list[float]]], token_counter: Callable[[str], int]) -> IngestReport` — pass 1 chunks + segments every article and fits BM25 over all segmented docs; pass 2 per article: `embedder(segmented)` first (an `EmbedError`, or a wrong vector count, records `"<type>/<slug>: <error>"` in `articles_failed` and moves on without touching Qdrant), then `replace_article`. Prints one progress line per article.
  - `main(argv: list[str] | None = None) -> int` — CLI flags `--data-dir` (default `data`), `--qdrant-path` (default `qdrant_data`), `--embed-url` (default `$EMBED_URL`), `--recreate`. Exit codes: `2` when the URL is missing or the health check / collection check fails (message printed), `1` when any article failed, `0` otherwise. Run as `python -m medical_rag.ingestion.ingest`.

- [ ] **Step 1: Write the failing tests**

`backend/tests/test_ingest.py`:

```python
from medical_rag.encoders import DENSE_DIM, EmbedError
from medical_rag.ingestion import ingest as ingest_module
from medical_rag.ingestion.ingest import ingest
from medical_rag.store import COLLECTION, ensure_collection, open_client


def _word_count(text: str) -> int:
    return len(text.split())


def fake_embedder(texts):
    return [[1.0 + len(text) % 5] + [0.0] * (DENSE_DIM - 1) for text in texts]


def _article(slug: str, intro: str, sections: list[tuple[str, str]]) -> str:
    body = "".join(f"\n## {heading}\n\n{text}\n" for heading, text in sections)
    return (
        f"# SOURCE_URL: https://youmed.vn/tin-tuc/{slug}/\n\n# Tiêu đề {slug}\n\n"
        f"Nội dung bài viết\n\n**{intro}**\n{body}"
    )


def _write(root, article_type: str, slug: str, text: str) -> None:
    folder = root / article_type
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{slug}.txt").write_text(text, encoding="utf-8")


def _setup(tmp_path):
    _write(
        tmp_path, "drug", "thuoc-a",
        _article("thuoc-a", "Giới thiệu thuốc A.", [("Công dụng", "Giảm đau."), ("Cách dùng", "Uống 1 viên.")]),
    )
    _write(
        tmp_path, "disease", "benh-b",
        _article("benh-b", "Giới thiệu bệnh B FAILME.", [("Triệu chứng", "Sốt và ho.")]),
    )
    client = open_client(None)
    ensure_collection(client)
    return client


def _count(client) -> int:
    return client.count(COLLECTION, exact=True).count


def test_ingest_loads_every_chunk_with_payload_and_both_vectors(tmp_path):
    client = _setup(tmp_path)
    report = ingest(str(tmp_path), client, fake_embedder, _word_count)

    assert report.articles_ok == 2
    assert report.articles_failed == []
    assert report.points_upserted == 5  # thuoc-a: intro + 2 sections, benh-b: intro + 1 section
    assert report.expected_chunks == 5
    assert _count(client) == 5

    points, _ = client.scroll(COLLECTION, limit=10, with_payload=True, with_vectors=True)
    assert {p.payload["type"] for p in points} == {"drug", "disease"}
    for point in points:
        assert set(point.vector) == {"dense", "sparse"}
        assert len(point.vector["dense"]) == DENSE_DIM
        assert "id" not in point.payload
        assert {"article_slug", "article_title", "article_url", "section_path", "text"} <= set(point.payload)


def test_ingest_is_idempotent_and_removes_stale_chunks(tmp_path):
    client = _setup(tmp_path)
    ingest(str(tmp_path), client, fake_embedder, _word_count)
    ingest(str(tmp_path), client, fake_embedder, _word_count)
    assert _count(client) == 5

    _write(
        tmp_path, "drug", "thuoc-a",
        _article("thuoc-a", "Giới thiệu thuốc A.", [("Công dụng", "Giảm đau.")]),
    )
    ingest(str(tmp_path), client, fake_embedder, _word_count)
    assert _count(client) == 4  # thuoc-a shrank from 3 chunks to 2; positional ids shifted


def test_failed_article_keeps_old_points_and_others_still_load(tmp_path):
    client = _setup(tmp_path)
    ingest(str(tmp_path), client, fake_embedder, _word_count)
    assert _count(client) == 5

    _write(
        tmp_path, "drug", "thuoc-a",
        _article("thuoc-a", "Giới thiệu thuốc A.", [("Công dụng", "Giảm đau.")]),
    )

    def flaky_embedder(texts):
        if any("FAILME" in text for text in texts):
            raise EmbedError("tunnel down")
        return fake_embedder(texts)

    report = ingest(str(tmp_path), client, flaky_embedder, _word_count)

    assert report.articles_ok == 1
    assert len(report.articles_failed) == 1
    assert report.articles_failed[0].startswith("disease/benh-b")
    assert _count(client) == 4  # thuoc-a replaced (3 -> 2), benh-b's 2 old points untouched


def test_main_exits_2_when_embed_url_is_missing(monkeypatch, capsys):
    monkeypatch.delenv("EMBED_URL", raising=False)
    assert ingest_module.main(["--data-dir", "nowhere"]) == 2
    assert "EMBED_URL" in capsys.readouterr().out


def test_main_exits_2_when_tunnel_is_unreachable(monkeypatch, capsys):
    assert ingest_module.main(["--embed-url", "http://127.0.0.1:1"]) == 2
    assert "update EMBED_URL" in capsys.readouterr().out
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest backend/tests/test_ingest.py -v`
Expected: collection ERROR `ModuleNotFoundError: No module named 'medical_rag.ingestion.ingest'`.

- [ ] **Step 3: Write the implementation**

`backend/medical_rag/ingestion/ingest.py`:

```python
import argparse
import os
import sys
from dataclasses import dataclass, field
from typing import Callable

from transformers import AutoTokenizer

from medical_rag.encoders import (
    Bm25Encoder,
    EmbedClient,
    EmbedError,
    embed_input,
    segment,
)
from medical_rag.ingestion.chunking import chunk_article, iter_articles
from medical_rag.store import COLLECTION, build_point, ensure_collection, open_client, replace_article

TOKENIZER_NAME = "dangvantuan/vietnamese-embedding"


@dataclass
class IngestReport:
    articles_ok: int = 0
    articles_failed: list[str] = field(default_factory=list)
    points_upserted: int = 0
    expected_chunks: int = 0


def ingest(
    data_dir: str,
    client,
    embedder: Callable[[list[str]], list[list[float]]],
    token_counter: Callable[[str], int],
) -> IngestReport:
    articles = []
    for article_type, slug, text in iter_articles(data_dir):
        chunks = chunk_article(text, article_type, slug, token_counter)
        segmented = [segment(embed_input(chunk)) for chunk in chunks]
        articles.append((article_type, slug, chunks, segmented))

    encoder = Bm25Encoder()
    encoder.fit([doc for _, _, _, segmented in articles for doc in segmented])

    report = IngestReport(expected_chunks=sum(len(chunks) for _, _, chunks, _ in articles))
    for number, (article_type, slug, chunks, segmented) in enumerate(articles, 1):
        try:
            dense = embedder(segmented)
            if len(dense) != len(chunks):
                raise EmbedError(f"embedder returned {len(dense)} vectors for {len(chunks)} chunks")
        except EmbedError as error:
            report.articles_failed.append(f"{article_type}/{slug}: {error}")
            print(f"[{number}/{len(articles)}] FAILED {article_type}/{slug}: {error}", flush=True)
            continue
        points = [
            build_point(chunk, vector, encoder.encode_doc(doc))
            for chunk, vector, doc in zip(chunks, dense, segmented)
        ]
        replace_article(client, article_type, slug, points)
        report.articles_ok += 1
        report.points_upserted += len(points)
        print(f"[{number}/{len(articles)}] {article_type}/{slug} ({len(points)} chunks)", flush=True)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Embed chunks and load them into Qdrant.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--qdrant-path", default="qdrant_data")
    parser.add_argument("--embed-url", default=os.environ.get("EMBED_URL"))
    parser.add_argument("--recreate", action="store_true")
    args = parser.parse_args(argv)

    if not args.embed_url:
        print("EMBED_URL is not set (or pass --embed-url): copy the Cloudflare tunnel URL from the Kaggle notebook.")
        return 2
    embed_client = EmbedClient(args.embed_url)
    try:
        embed_client.health_check()
        qdrant = open_client(args.qdrant_path)
        ensure_collection(qdrant, recreate=args.recreate)
    except (EmbedError, ValueError) as error:
        print(f"startup failed: {error}")
        return 2

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    token_counter = lambda text: len(tokenizer.encode(text, add_special_tokens=False))  # noqa: E731
    report = ingest(args.data_dir, qdrant, embed_client.embed, token_counter)

    total = qdrant.count(COLLECTION, exact=True).count
    print(
        f"articles ok={report.articles_ok} failed={len(report.articles_failed)} "
        f"points upserted={report.points_upserted} collection points={total} "
        f"expected chunks={report.expected_chunks}"
    )
    for failure in report.articles_failed:
        print(f"  failed: {failure}")
    return 1 if report.articles_failed else 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests to verify they pass, with warnings as errors**

Run: `python -m pytest backend/tests/test_ingest.py -v -W error::UserWarning`
Expected: `5 passed` (about 30 s; the unreachable-tunnel test waits through 3 attempts with backoff).

- [ ] **Step 5: Run the whole suite**

Run: `python -m pytest backend/tests -q -W error::UserWarning`
Expected: `37 passed` (5 chunking + 20 encoders + 7 store + 5 ingest).

- [ ] **Step 6: Commit**

```bash
git add backend/medical_rag/ingestion/ingest.py backend/tests/test_ingest.py
git commit -m "feat: add Qdrant ingest pipeline and CLI"
```

---

### Task 7: `validate_ingest.py` acceptance script

**Files:**
- Create: `backend/scripts/validate_ingest.py`
- Create: `backend/tests/test_validate_ingest.py`

**Interfaces:**
- Consumes: everything above, plus `normalize` and `slug_from_url` from `backend/scripts/validate_chunking.py` (importable because pytest `pythonpath` includes `scripts`, and because Python puts the script's own directory on `sys.path` when run as `python backend/scripts/validate_ingest.py`).
- Produces (module `validate_ingest`):
  - `MODEL_MAX_TOKENS = 512`
  - `load_expected(data_dir: str, token_counter) -> list[dict]` — every chunk from `chunk_article` plus a `"segmented"` key holding `segment(embed_input(chunk))`.
  - `check_invariants(client, expected: list[dict]) -> list[str]` — list of problem strings, empty when fine. Checks: point count equals `len(expected)`; duplicate ids; missing ids; unexpected ids; each point has a `dense` vector of size 768; a `sparse` vector is present exactly when `terms(chunk["segmented"])` is non-empty; payload `type` and `article_slug` equal the chunk's.
  - `max_segmented_tokens(expected: list[dict], token_counter) -> int`
  - `smoke_retrieval(client, embed, test_case_dir: str, per_type: int = 30, seed: int = 42, k: int = 5) -> dict` — for each `<type>.csv` in the directory (columns `question`, `context`, `article_url`): a `random_state=seed` sample of `per_type` rows; dense top-k and sparse top-k separately; returns `{type: {"dense": {"article": rate, "chunk": rate}, "sparse": {...}}}` where `article` means a hit whose payload `type` equals the CSV name and `article_slug` equals the slug of `article_url`, and `chunk` means `normalize(context)` is contained in `normalize(payload["text"])` of a hit.
  - `main(argv=None) -> int` — flags `--data-dir`, `--test-case-dir` (default `test_case`), `--qdrant-path`, `--embed-url` (default `$EMBED_URL`), `--per-type` (30), `--seed` (42). Prints `chunks expected=... longest segmented chunk=... tokens`, up to 20 `FAIL: ...` lines, then one `hit@5` line per type and mode. Exit `1` if any invariant problem or the longest segmented chunk exceeds 512 tokens, `2` if the URL is missing, else `0`. It contains **no** hit-rate thresholds (Task 8 adds them).

- [ ] **Step 1: Write the failing tests**

`backend/tests/test_validate_ingest.py`:

```python
import pandas as pd
from qdrant_client import models

from medical_rag.encoders import DENSE_DIM
from medical_rag.ingestion.ingest import ingest
from medical_rag.store import COLLECTION, ensure_collection, open_client
from validate_ingest import check_invariants, load_expected, max_segmented_tokens, smoke_retrieval


def _word_count(text: str) -> int:
    return len(text.split())


def fake_embedder(texts):
    return [[1.0 + len(text) % 5] + [0.0] * (DENSE_DIM - 1) for text in texts]


def _write_articles(root) -> None:
    for article_type, slug, intro, sections in (
        ("drug", "thuoc-a", "Giới thiệu thuốc A.", [("Công dụng", "Giảm đau."), ("Cách dùng", "Uống 1 viên.")]),
        ("disease", "benh-b", "Giới thiệu bệnh B.", [("Triệu chứng", "Sốt và ho.")]),
    ):
        folder = root / article_type
        folder.mkdir(parents=True)
        body = "".join(f"\n## {heading}\n\n{text}\n" for heading, text in sections)
        (folder / f"{slug}.txt").write_text(
            f"# SOURCE_URL: https://youmed.vn/tin-tuc/{slug}/\n\n# Tiêu đề {slug}\n\n"
            f"Nội dung bài viết\n\n**{intro}**\n{body}",
            encoding="utf-8",
        )


def _ingested(tmp_path):
    _write_articles(tmp_path)
    client = open_client(None)
    ensure_collection(client)
    ingest(str(tmp_path), client, fake_embedder, _word_count)
    return client, load_expected(str(tmp_path), _word_count)


def test_check_invariants_passes_on_a_clean_ingest(tmp_path):
    client, expected = _ingested(tmp_path)
    assert len(expected) == 5
    assert check_invariants(client, expected) == []


def test_check_invariants_reports_a_missing_point_and_a_missing_sparse_vector(tmp_path):
    client, expected = _ingested(tmp_path)
    victim = expected[0]["id"]
    client.delete(COLLECTION, points_selector=models.PointIdsList(points=[victim]))
    problems = check_invariants(client, expected)
    assert any("point count 4 != expected chunk count 5" in p for p in problems)
    assert any(f"missing point for chunk {victim}" in p for p in problems)

    client.upsert(
        COLLECTION,
        points=[
            models.PointStruct(
                id=victim,
                vector={"dense": [1.0] * DENSE_DIM},
                payload={"type": expected[0]["type"], "article_slug": expected[0]["article_slug"]},
            )
        ],
    )
    problems = check_invariants(client, expected)
    assert any("sparse vector presence" in p for p in problems)


def test_max_segmented_tokens_uses_the_segmented_text(tmp_path):
    _, expected = _ingested(tmp_path)
    assert max_segmented_tokens(expected, _word_count) == max(_word_count(c["segmented"]) for c in expected)


def test_smoke_retrieval_finds_the_right_article_with_sparse_search(tmp_path):
    client, _ = _ingested(tmp_path)
    case_dir = tmp_path / "cases"
    case_dir.mkdir()
    pd.DataFrame(
        [{
            "question": "Thuốc nào giảm đau?",
            "context": "Giảm đau.",
            "article_url": "https://youmed.vn/tin-tuc/thuoc-a/",
        }]
    ).to_csv(case_dir / "drug.csv", index=False, encoding="utf-8")

    results = smoke_retrieval(client, fake_embedder, str(case_dir), per_type=30, seed=42, k=1)

    assert set(results) == {"drug"}
    assert results["drug"]["sparse"] == {"article": 1.0, "chunk": 1.0}
    for levels in results["drug"].values():
        assert all(0.0 <= value <= 1.0 for value in levels.values())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest backend/tests/test_validate_ingest.py -v`
Expected: collection ERROR `ModuleNotFoundError: No module named 'validate_ingest'`.

- [ ] **Step 3: Write the implementation**

`backend/scripts/validate_ingest.py`:

```python
import argparse
import os
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from qdrant_client import models
from transformers import AutoTokenizer

from medical_rag.encoders import (
    DENSE_DIM,
    Bm25Encoder,
    EmbedClient,
    embed_input,
    segment,
    terms,
)
from medical_rag.ingestion.chunking import chunk_article, iter_articles
from medical_rag.store import COLLECTION, open_client
from validate_chunking import normalize, slug_from_url

TOKENIZER_NAME = "dangvantuan/vietnamese-embedding"
MODEL_MAX_TOKENS = 512


def load_expected(data_dir: str, token_counter) -> list[dict]:
    expected = []
    for article_type, slug, text in iter_articles(data_dir):
        for chunk in chunk_article(text, article_type, slug, token_counter):
            expected.append({**chunk, "segmented": segment(embed_input(chunk))})
    return expected


def _scroll_all(client) -> list:
    points, offset = [], None
    while True:
        batch, offset = client.scroll(
            COLLECTION, limit=256, offset=offset, with_payload=True, with_vectors=True
        )
        points.extend(batch)
        if offset is None:
            return points


def check_invariants(client, expected: list[dict]) -> list[str]:
    problems = []
    points = _scroll_all(client)
    if len(points) != len(expected):
        problems.append(f"point count {len(points)} != expected chunk count {len(expected)}")

    by_id = {chunk["id"]: chunk for chunk in expected}
    seen = Counter(str(point.id) for point in points)
    for point_id, times in seen.items():
        if times > 1:
            problems.append(f"duplicate id {point_id} ({times}x)")
    for missing in sorted(set(by_id) - set(seen)):
        problems.append(f"missing point for chunk {missing}")

    for point in points:
        chunk = by_id.get(str(point.id))
        if chunk is None:
            problems.append(f"unexpected point {point.id}")
            continue
        dense = point.vector.get("dense") if isinstance(point.vector, dict) else None
        if dense is None or len(dense) != DENSE_DIM:
            problems.append(f"point {point.id}: dense vector missing or not {DENSE_DIM}-dimensional")
        should_have_sparse = bool(terms(chunk["segmented"]))
        if should_have_sparse != ("sparse" in point.vector):
            problems.append(f"point {point.id}: sparse vector presence does not match its text")
        for key in ("type", "article_slug"):
            if point.payload.get(key) != chunk[key]:
                problems.append(f"point {point.id}: payload {key} differs from the chunk")
    return problems


def max_segmented_tokens(expected: list[dict], token_counter) -> int:
    return max(token_counter(chunk["segmented"]) for chunk in expected)


def smoke_retrieval(client, embed, test_case_dir: str, per_type: int = 30, seed: int = 42, k: int = 5) -> dict:
    encoder = Bm25Encoder()
    results = {}
    for path in sorted(Path(test_case_dir).glob("*.csv")):
        article_type = path.stem
        frame = pd.read_csv(path, encoding="utf-8")
        sample = frame.sample(n=min(per_type, len(frame)), random_state=seed)
        questions = [str(question) for question in sample["question"]]
        vectors = embed([segment(question) for question in questions])

        hits = {mode: {"article": 0, "chunk": 0} for mode in ("dense", "sparse")}
        for (_, row), vector in zip(sample.iterrows(), vectors):
            slug = slug_from_url(row["article_url"])
            context = normalize(str(row["context"]))
            indices, values = encoder.encode_query(segment(str(row["question"])))
            found = {
                "dense": client.query_points(COLLECTION, query=vector, using="dense", limit=k).points,
                "sparse": (
                    client.query_points(
                        COLLECTION,
                        query=models.SparseVector(indices=indices, values=values),
                        using="sparse",
                        limit=k,
                    ).points
                    if indices
                    else []
                ),
            }
            for mode, points in found.items():
                if any(p.payload["type"] == article_type and p.payload["article_slug"] == slug for p in points):
                    hits[mode]["article"] += 1
                if any(context in normalize(p.payload["text"]) for p in points):
                    hits[mode]["chunk"] += 1
        results[article_type] = {
            mode: {level: count / len(sample) for level, count in levels.items()}
            for mode, levels in hits.items()
        }
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Acceptance check for a real Qdrant ingest.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--test-case-dir", default="test_case")
    parser.add_argument("--qdrant-path", default="qdrant_data")
    parser.add_argument("--embed-url", default=os.environ.get("EMBED_URL"))
    parser.add_argument("--per-type", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if not args.embed_url:
        print("EMBED_URL is not set (or pass --embed-url).")
        return 2

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    token_counter = lambda text: len(tokenizer.encode(text, add_special_tokens=False))  # noqa: E731
    expected = load_expected(args.data_dir, token_counter)
    client = open_client(args.qdrant_path)

    problems = check_invariants(client, expected)
    longest = max_segmented_tokens(expected, token_counter)
    if longest > MODEL_MAX_TOKENS:
        problems.append(f"longest segmented chunk has {longest} tokens > model max {MODEL_MAX_TOKENS}")
    print(f"chunks expected={len(expected)} longest segmented chunk={longest} tokens")
    for problem in problems[:20]:
        print(f"FAIL: {problem}")

    embed_client = EmbedClient(args.embed_url)
    results = smoke_retrieval(client, embed_client.embed, args.test_case_dir, args.per_type, args.seed)
    for article_type, modes in results.items():
        for mode, levels in modes.items():
            print(
                f"{article_type:10s} {mode:6s} hit@5 article={levels['article']:.1%} chunk={levels['chunk']:.1%}"
            )
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests to verify they pass, with warnings as errors**

Run: `python -m pytest backend/tests/test_validate_ingest.py -v -W error::UserWarning`
Expected: `4 passed`.

- [ ] **Step 5: Run the whole suite and check the script imports**

Run: `python -m pytest backend/tests -q -W error::UserWarning && PYTHONIOENCODING=utf-8 python backend/scripts/validate_ingest.py --help`
Expected: `41 passed` (37 + 4) and the argparse usage text listing `--data-dir`, `--test-case-dir`, `--qdrant-path`, `--embed-url`, `--per-type`, `--seed`.

- [ ] **Step 6: Commit**

```bash
git add backend/scripts/validate_ingest.py backend/tests/test_validate_ingest.py
git commit -m "test: add validate_ingest acceptance script"
```

---

### Task 8: Live acceptance run (controller with the user; not dispatched)

**Files:**
- Modify: `backend/scripts/validate_ingest.py` (hit-rate floors, after measuring)
- Modify: `docs/superpowers/specs/2026-09-19-medical-rag-embedding-qdrant-design.md` (record measured numbers)

**Interfaces:**
- Consumes: everything above plus a live Kaggle notebook (`serve_model/serve_qwen3_kaggle.ipynb`, embed server on port 8001 behind a Cloudflare tunnel).

This task needs a person: an agent cannot start the notebook or obtain the tunnel URL.

- [ ] **Step 1: Human precondition**

The user starts the Kaggle notebook, waits for the embed server, copies the embed tunnel URL, and in the Git Bash session that will run the commands: `export EMBED_URL="https://<random>.trycloudflare.com"`.

- [ ] **Step 2: Health check**

Run: `python -c "import os; from medical_rag.encoders import EmbedClient; EmbedClient(os.environ['EMBED_URL']).health_check(); print('embed ok')"`
Expected: `embed ok`. On failure the message says to check the notebook and update `EMBED_URL`.

- [ ] **Step 3: Real ingest**

Run: `PYTHONIOENCODING=utf-8 python -m medical_rag.ingestion.ingest --recreate`
Expected: 584 progress lines, then `articles ok=584 failed=0 points upserted=9294 collection points=9294 expected chunks=9294`, exit code 0 (roughly 10-20 minutes: about 90 s of local chunking and segmentation plus about 290 batched embedding requests). If some articles fail, re-run without `--recreate`; a re-run is idempotent.

- [ ] **Step 4: Acceptance script**

Run: `PYTHONIOENCODING=utf-8 python backend/scripts/validate_ingest.py`
Expected: `chunks expected=9294 longest segmented chunk=363 tokens`, no `FAIL:` line, exit code 0, and eight `hit@5` lines (4 types x dense/sparse). Close any other process holding `qdrant_data/` first (local mode allows one opener).

- [ ] **Step 5: Record the measured numbers**

Add a "Measured results (date)" subsection at the end of the spec's Validation section listing the eight `hit@5` lines exactly as printed, plus the ingest duration and the counts from Step 3.

- [ ] **Step 6: Set floors from the measurement**

In `backend/scripts/validate_ingest.py`, add above `def main`:

```python
FLOORS = {}  # {(type, mode): minimum chunk-level hit@5}, measured on <date>, margin 0.03
```

and fill `FLOORS` with one entry per printed line: the printed chunk-level rate minus `0.03`, rounded down to 3 decimals (for example a printed `chunk=63.3%` gives `("drug", "dense"): 0.603`). In `main`, after the `hit@5` print loop, add:

```python
    for (article_type, mode), floor in FLOORS.items():
        if results[article_type][mode]["chunk"] < floor:
            print(f"FAIL: {article_type} {mode} chunk hit@5 below floor {floor:.3f}")
            problems.append("hit-rate floor")
```

Re-run Step 4: it must still exit `0`. Then run `python -m pytest backend/tests -q -W error::UserWarning` (expect `41 passed`).

- [ ] **Step 7: Commit**

```bash
git add backend/scripts/validate_ingest.py docs/superpowers/specs/2026-09-19-medical-rag-embedding-qdrant-design.md
git commit -m "test: record measured hit@5 and set validate_ingest floors"
```

- [ ] **Step 8: Final review and finish**

Run the final whole-branch review (most capable model) over `git merge-base master HEAD..HEAD`, fix findings in one wave, then superpowers:finishing-a-development-branch.

---

## Self-Review Notes

- **Spec coverage:** repo layout and move (Task 1); `segment`/`embed_input`/`DENSE_DIM` (2); BM25 sparse encoder with hashing, weights, query weights, edge cases (3); `EmbedClient` retry, halving, health check, dimension check (4); collection schema, IDF, payload indexes with the local-mode warning handled, `build_point`, replace-by-article (5); ingest flow with embed-before-delete, failure isolation, CLI exit codes, env var URL (6); acceptance invariants including the 512-token segmented check, smoke retrieval, no pre-committed thresholds (7); live run, recorded numbers, floors set after measurement (8). Out-of-scope items (retrieval, rerank, generation, resume) have no task.
- **Verified before writing:** all code and tests in Tasks 2-7 were run in a scratch copy: 41 tests pass with `-W error::UserWarning` (5 moved chunking + 20 + 7 + 5 + 4), and the full pipeline ran over the 584 real files with a fake embedder (9294 points, 0 invariant problems).
- **Placeholder scan:** none; Task 8's floor values are derived from a live measurement by an explicit formula, by design.
- **Type consistency:** `EmbedClient.embed` and the ingest `embedder` callable share `list[str] -> list[list[float]]`; `build_point(chunk, dense, sparse)` takes the `(indices, values)` tuple that `Bm25Encoder.encode_doc` returns; `check_invariants` relies on the `"segmented"` key that `load_expected` adds; `IngestReport` fields match their use in `main`.
