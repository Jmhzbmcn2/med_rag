"""Compare dense, sparse and hybrid (RRF) retrieval on test_case/*.csv.

Needs the embed server, and exclusive access to the Qdrant store (stop the API first).
"""
import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from qdrant_client import models

from medical_rag.encoders import Bm25Encoder, EmbedClient, EmbedError, segment
from medical_rag.retrieval import Retriever
from medical_rag.store import COLLECTION, open_client
from validate_chunking import normalize, slug_from_url

MODES = ("dense", "sparse", "hybrid")


def _first_hits(rows, article_type, slug, context):
    """1-based rank of the first row from the right article, and of the first row holding the context."""
    article = chunk = None
    for rank, (row_type, row_slug, text) in enumerate(rows, 1):
        if article is None and row_type == article_type and row_slug == slug:
            article = rank
        if chunk is None and context in normalize(text):
            chunk = rank
    return article, chunk


def _payload_rows(points):
    return [(p.payload["type"], p.payload["article_slug"], p.payload["text"]) for p in points]


def _summarize(records: list[tuple], k: int) -> dict:
    n = len(records)
    articles, chunks = [r[0] for r in records], [r[1] for r in records]
    return {
        "n": n,
        f"article@{k}": sum(a is not None for a in articles) / n,
        "chunk@1": sum(c == 1 for c in chunks) / n,
        f"chunk@{k}": sum(c is not None for c in chunks) / n,
        "mrr": sum(1 / c for c in chunks if c) / n,
    }


def evaluate(client, embed, test_case_dir: str, per_type: int = 50, seed: int = 42, k: int = 5) -> dict:
    encoder = Bm25Encoder()
    records = {}  # (article_type, mode) -> [(article_rank, chunk_rank)]
    for path in sorted(Path(test_case_dir).glob("*.csv")):
        frame = pd.read_csv(path, encoding="utf-8")
        if frame.empty:
            continue
        sample = frame.sample(n=min(per_type, len(frame)), random_state=seed)
        questions = [str(q) for q in sample["question"]]
        vectors = embed([segment(q) for q in questions], task="query")
        for (_, row), question, vector in zip(sample.iterrows(), questions, vectors):
            slug, context = slug_from_url(row["article_url"]), normalize(str(row["context"]))
            indices, values = encoder.encode_query(segment(question))
            found = {
                "dense": _payload_rows(client.query_points(COLLECTION, query=vector, using="dense", limit=k).points),
                "sparse": _payload_rows(
                    client.query_points(
                        COLLECTION,
                        query=models.SparseVector(indices=indices, values=values),
                        using="sparse",
                        limit=k,
                    ).points
                    if indices
                    else []
                ),
                # same query vector as "dense", so the modes differ only in the search itself
                "hybrid": [
                    (h.type, slug_from_url(h.article_url), h.text)
                    for h in Retriever(client, lambda texts, task="document", v=vector: [v]).search(question, k=k)
                ],
            }
            for mode, rows in found.items():
                records.setdefault((path.stem, mode), []).append(_first_hits(rows, path.stem, slug, context))

    results = {}
    for (article_type, mode), recs in records.items():
        results.setdefault(article_type, {})[mode] = _summarize(recs, k)
    if results:
        results["all"] = {
            mode: _summarize([r for (_, m), recs in records.items() if m == mode for r in recs], k) for mode in MODES
        }
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare dense, sparse and hybrid retrieval.")
    parser.add_argument("--test-case-dir", default="test_case")
    parser.add_argument("--qdrant-path", default="qdrant_data")
    parser.add_argument("--embed-url", default=os.environ.get("EMBED_URL"))
    parser.add_argument("--per-type", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args(argv)
    if not args.embed_url:
        print("EMBED_URL is not set (or pass --embed-url).")
        return 2
    if not Path(args.qdrant_path).exists():
        print(f"qdrant path {args.qdrant_path!r} does not exist (relative to {Path.cwd()}).")
        return 2

    embed_client = EmbedClient(args.embed_url)
    try:
        embed_client.health_check()
        client = open_client(args.qdrant_path)
        results = evaluate(client, embed_client.embed, args.test_case_dir, args.per_type, args.seed, args.k)
    except (EmbedError, OSError, RuntimeError, ValueError) as error:
        print(f"eval aborted: {error}")
        return 2

    k = args.k
    print(f"{'type':10s} {'mode':7s} {'n':>4s} {'article@' + str(k):>10s} {'chunk@1':>8s} {'chunk@' + str(k):>8s} {'mrr':>6s}")
    for article_type, modes in results.items():
        for mode, row in modes.items():
            print(
                f"{article_type:10s} {mode:7s} {row['n']:4d} {row[f'article@{k}']:10.1%} "
                f"{row['chunk@1']:8.1%} {row[f'chunk@{k}']:8.1%} {row['mrr']:6.3f}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
