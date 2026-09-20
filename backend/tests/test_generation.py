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
