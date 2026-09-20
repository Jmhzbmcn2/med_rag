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
