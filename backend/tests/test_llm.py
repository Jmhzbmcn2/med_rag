import pytest

from medical_rag.llm import chat


def test_chat_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        chat([{"role": "user", "content": "hi"}])
