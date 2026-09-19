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
