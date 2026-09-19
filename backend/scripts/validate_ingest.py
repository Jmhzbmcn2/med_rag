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
