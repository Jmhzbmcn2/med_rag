import os

from openai import OpenAI

OPENROUTER_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "google/gemini-2.5-flash"


def chat(messages: list[dict], model: str | None = None, **kwargs) -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    client = OpenAI(base_url=OPENROUTER_URL, api_key=key, timeout=60)
    model_name = model or os.environ.get("LLM_MODEL", DEFAULT_MODEL)
    resp = client.chat.completions.create(model=model_name, messages=messages, **kwargs)
    choice = resp.choices[0] if resp.choices else None
    if choice is None or choice.message.content is None:
        raise RuntimeError(f"empty completion from {model_name}")
    return choice.message.content
