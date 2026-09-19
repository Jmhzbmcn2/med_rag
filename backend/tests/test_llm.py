from types import SimpleNamespace

import pytest

from medical_rag import llm
from medical_rag.llm import chat


def test_chat_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        chat([{"role": "user", "content": "hi"}])


def fake_openai(monkeypatch, choices):
    made = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            made.update(kwargs)
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: SimpleNamespace(choices=choices)))

    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.setattr(llm, "OpenAI", FakeOpenAI)
    return made


def message(content):
    return [SimpleNamespace(message=SimpleNamespace(content=content))]


def test_chat_returns_content_and_sets_timeout(monkeypatch):
    made = fake_openai(monkeypatch, message("xin chào"))
    assert chat([{"role": "user", "content": "hi"}]) == "xin chào"
    assert made["timeout"] == 60


def test_chat_rejects_empty_choices(monkeypatch):
    fake_openai(monkeypatch, [])
    with pytest.raises(RuntimeError, match="empty completion"):
        chat([{"role": "user", "content": "hi"}])


def test_chat_rejects_null_content(monkeypatch):
    fake_openai(monkeypatch, message(None))
    with pytest.raises(RuntimeError, match="empty completion"):
        chat([{"role": "user", "content": "hi"}])
