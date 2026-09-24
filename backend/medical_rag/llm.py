import logging
import os

from openai import NotFoundError, OpenAI

log = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1"
GROQ_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "google/gemini-2.5-flash"


def chat(messages: list[dict], model: str | None = None, **kwargs) -> str:
    model_name = model or os.environ.get("LLM_MODEL", DEFAULT_MODEL)
    groq_key = os.environ.get("GROQ_API_KEY")
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    provider = os.environ.get("LLM_PROVIDER", "").lower()

    # Determine whether to target Groq:
    is_groq_target = provider == "groq" or (
        bool(groq_key)
        and (
            model_name.startswith("llama-3.1-8b-instant")
            or model_name.startswith("openai/gpt-oss")
            or model_name.startswith("qwen/")
        )
    )

    if is_groq_target:
        if not groq_key:
            raise RuntimeError("GROQ_API_KEY is not set")
        client = OpenAI(base_url=GROQ_URL, api_key=groq_key, timeout=60)
        try:
            resp = client.chat.completions.create(model=model_name, messages=messages, **kwargs)
        except NotFoundError as err:
            # If the model is not found/accessible on Groq, fallback to OpenRouter if available
            if openrouter_key:
                fallback = "meta-llama/llama-3.1-8b-instruct" if "llama" in model_name.lower() else DEFAULT_MODEL
                log.warning(
                    "Model %s not accessible on Groq (%s), falling back to %s on OpenRouter",
                    model_name,
                    err,
                    fallback,
                )
                fallback_client = OpenAI(base_url=OPENROUTER_URL, api_key=openrouter_key, timeout=60)
                resp = fallback_client.chat.completions.create(model=fallback, messages=messages, **kwargs)
                model_name = fallback
            else:
                raise
    else:
        if not openrouter_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        client = OpenAI(base_url=OPENROUTER_URL, api_key=openrouter_key, timeout=60)
        resp = client.chat.completions.create(model=model_name, messages=messages, **kwargs)

    choice = resp.choices[0] if resp.choices else None
    if choice is None or choice.message.content is None:
        raise RuntimeError(f"empty completion from {model_name}")
    return choice.message.content
