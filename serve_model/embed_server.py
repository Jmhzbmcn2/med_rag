"""Local embedding server, same /embed contract as the Kaggle one (EmbedClient works unchanged).

Run: uvicorn embed_server:app --port 8001
"""
from typing import Literal

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

MODEL = "google/embeddinggemma-300m"  # gated: accept the license on HF and run `hf auth login`
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI()
model = SentenceTransformer(MODEL, device=DEVICE)  # fp32: gemma3 embeddings break in fp16


class EmbedReq(BaseModel):
    texts: list[str]
    # embeddinggemma is trained with task prefixes; index chunks as "document", search with "query"
    task: Literal["document", "query"] = "document"


@app.post("/embed")
def embed(req: EmbedReq):
    # pyvi segmentation joins syllables with "_"; gemma's tokenizer wants plain text
    texts = [t.replace("_", " ") for t in req.texts]
    encode = model.encode_query if req.task == "query" else model.encode_document
    return {"embeddings": encode(texts, batch_size=16).tolist()}
