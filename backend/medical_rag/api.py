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
    qdrant_path = os.environ.get("QDRANT_PATH", "qdrant_data")
    if not Path(qdrant_path).exists():  # open_client would silently create an empty store
        raise RuntimeError(
            f"QDRANT_PATH {qdrant_path!r} does not exist (relative to {Path.cwd()}); "
            "run from the repo root or set QDRANT_PATH"
        )
    client = open_client(qdrant_path)
    if not client.collection_exists(COLLECTION):
        client.close()
        raise RuntimeError(f"collection {COLLECTION!r} not found; run `python -m medical_rag.ingestion.ingest` first")
    rerank_url = os.environ.get("RERANK_URL")
    rerank = RerankClient(rerank_url).rerank if rerank_url else None
    app.state.client = client
    app.state.retriever = Retriever(client, EmbedClient(embed_url, timeout=15, retries=2).embed, rerank)
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
