"""Compare dense, sparse and hybrid (RRF) retrieval on test_case/*.csv across IR metrics.

Metrics evaluated:
- Recall@K: có lấy đủ thông tin không? (Fraction of ground truth relevant chunks retrieved in top K)
- Precision@K: lấy về có nhiều rác không? (Fraction of retrieved chunks in top K that are relevant)
- HitRate@K: có lấy được ít nhất một tài liệu đúng không? (Whether at least 1 relevant chunk is in top K)
- MRR@K / MRR: tài liệu đúng đầu tiên nằm cao không? (Reciprocal rank of first relevant chunk)
- nDCG@K: chất lượng ranking tổng thể tốt không? (Normalized Discounted Cumulative Gain)

Needs the embed server, and exclusive access to the Qdrant store (stop the API first).
"""
import argparse
import math
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
DEFAULT_K_VALUES = (1, 3, 5, 10)


def _eval_query_hits(rows, article_type: str, slug: str, context: str, k_values: tuple[int, ...] | list[int]) -> dict:
    """Evaluate relevance of retrieved rows against ground truth article and context for multiple k values."""
    rel_chunks = [1 if context in normalize(text) else 0 for (_, _, text) in rows]
    rel_articles = [1 if (row_type == article_type and row_slug == slug) else 0 for (row_type, row_slug, _) in rows]

    first_chunk_rank = rel_chunks.index(1) + 1 if 1 in rel_chunks else None
    reciprocal_rank = (1.0 / first_chunk_rank) if first_chunk_rank is not None else 0.0
    total_chunk_matches = max(1, sum(rel_chunks))

    first_article_rank = rel_articles.index(1) + 1 if 1 in rel_articles else None

    metrics = {
        "first_chunk_rank": first_chunk_rank,
        "first_article_rank": first_article_rank,
        "reciprocal_rank": reciprocal_rank,
        "total_chunk_matches": total_chunk_matches,
    }

    for k in k_values:
        sub_chunks = rel_chunks[:k]
        sub_articles = rel_articles[:k]
        chunk_matches_k = sum(sub_chunks)
        article_matches_k = sum(sub_articles)

        # HitRate@K: 1 if at least one matching chunk is in top k, else 0
        hit_k = 1 if chunk_matches_k > 0 else 0
        metrics[f"hit@{k}"] = hit_k
        # Precision@K: matching chunks in top k divided by k
        metrics[f"prec@{k}"] = chunk_matches_k / k
        # Recall@K: matching chunks in top k divided by total matching chunks
        metrics[f"recall@{k}"] = chunk_matches_k / total_chunk_matches

        # MRR@K: 1/rank if first hit is within top k, else 0
        first_k = sub_chunks.index(1) + 1 if 1 in sub_chunks else None
        metrics[f"mrr@{k}"] = (1.0 / first_k) if first_k is not None else 0.0

        # nDCG@K
        dcg_k = sum(rel / math.log2(rank + 1) for rank, rel in enumerate(sub_chunks, 1))
        idcg_k = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(total_chunk_matches, k) + 1))
        metrics[f"ndcg@{k}"] = (dcg_k / idcg_k) if idcg_k > 0 else 0.0

        # Article@K: 1 if target article is in top k, else 0
        metrics[f"article@{k}"] = 1 if article_matches_k > 0 else 0

    return metrics


def _payload_rows(points):
    return [(p.payload["type"], p.payload["article_slug"], p.payload["text"]) for p in points]


def _summarize(query_metrics: list[dict], k_values: tuple[int, ...] | list[int]) -> dict:
    n = len(query_metrics)
    if n == 0:
        return {"n": 0}

    row = {
        "n": n,
        "mrr": sum(m["reciprocal_rank"] for m in query_metrics) / n,
    }

    for k in k_values:
        row[f"article@{k}"] = sum(m[f"article@{k}"] for m in query_metrics) / n
        row[f"chunk@{k}"] = sum(m[f"hit@{k}"] for m in query_metrics) / n
        row[f"hit_rate@{k}"] = sum(m[f"hit@{k}"] for m in query_metrics) / n
        row[f"recall@{k}"] = sum(m[f"recall@{k}"] for m in query_metrics) / n
        row[f"precision@{k}"] = sum(m[f"prec@{k}"] for m in query_metrics) / n
        row[f"mrr@{k}"] = sum(m[f"mrr@{k}"] for m in query_metrics) / n
        row[f"ndcg@{k}"] = sum(m[f"ndcg@{k}"] for m in query_metrics) / n

    if 1 in k_values:
        row["chunk@1"] = row["chunk@1"]

    return row


def evaluate(
    client,
    embed,
    test_case_dir: str,
    per_type: int = 50,
    seed: int = 42,
    k: int = 5,
    k_values: tuple[int, ...] | list[int] = DEFAULT_K_VALUES,
) -> dict:
    all_k = tuple(sorted(set(k_values).union({k, 1})))
    max_k = max(all_k)

    encoder = Bm25Encoder()
    records = {}  # (article_type, mode) -> [query_metric_dict]

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
                "dense": _payload_rows(
                    client.query_points(COLLECTION, query=vector, using="dense", limit=max_k).points
                ),
                "sparse": _payload_rows(
                    client.query_points(
                        COLLECTION,
                        query=models.SparseVector(indices=indices, values=values),
                        using="sparse",
                        limit=max_k,
                    ).points
                    if indices
                    else []
                ),
                # same query vector as "dense", so the modes differ only in the search itself
                "hybrid": [
                    (h.type, slug_from_url(h.article_url), h.text)
                    for h in Retriever(client, lambda texts, task="document", v=vector: [v]).search(question, k=max_k)
                ],
            }
            for mode, rows in found.items():
                metrics = _eval_query_hits(rows, path.stem, slug, context, all_k)
                records.setdefault((path.stem, mode), []).append(metrics)

    results = {}
    for (article_type, mode), recs in records.items():
        results.setdefault(article_type, {})[mode] = _summarize(recs, all_k)
    if results:
        results["all"] = {
            mode: _summarize([r for (_, m), recs in records.items() if m == mode for r in recs], all_k)
            for mode in MODES
        }
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare dense, sparse and hybrid retrieval across IR metrics.")
    parser.add_argument("--test-case-dir", default="test_case")
    parser.add_argument("--qdrant-path", default="qdrant_data")
    parser.add_argument("--embed-url", default=os.environ.get("EMBED_URL"))
    parser.add_argument("--per-type", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--k-values", default="1,3,5,10", help="Comma-separated k values, e.g. 1,3,5,10")
    args = parser.parse_args(argv)

    if not args.embed_url:
        print("EMBED_URL is not set (or pass --embed-url).")
        return 2
    if not Path(args.qdrant_path).exists():
        print(f"qdrant path {args.qdrant_path!r} does not exist (relative to {Path.cwd()}).")
        return 2

    k_values = tuple(sorted(set(int(x.strip()) for x in args.k_values.split(",") if x.strip())))
    all_k = tuple(sorted(set(k_values).union({args.k, 1})))

    embed_client = EmbedClient(args.embed_url)
    try:
        embed_client.health_check()
        client = open_client(args.qdrant_path)
        results = evaluate(
            client,
            embed_client.embed,
            args.test_case_dir,
            per_type=args.per_type,
            seed=args.seed,
            k=args.k,
            k_values=all_k,
        )
    except (EmbedError, OSError, RuntimeError, ValueError) as error:
        print(f"eval aborted: {error}")
        return 2

    print("=" * 95)
    print(f"RETRIEVAL EVALUATION REPORT (per_type={args.per_type}, seed={args.seed})")
    print("=" * 95)

    for k in k_values:
        print(f"\n[ Cutoff K = {k} ]")
        print(
            f"{'category':12s} {'mode':8s} {'n':>4s} {'Recall@' + str(k):>10s} {'Prec@' + str(k):>9s} "
            f"{'HitRate@' + str(k):>11s} {'MRR@' + str(k):>8s} {'nDCG@' + str(k):>8s} {'Article@' + str(k):>11s}"
        )
        print("-" * 88)
        for article_type, modes in results.items():
            for mode, row in modes.items():
                print(
                    f"{article_type:12s} {mode:8s} {row['n']:4d} "
                    f"{row[f'recall@{k}']:10.1%} "
                    f"{row[f'precision@{k}']:9.1%} "
                    f"{row[f'hit_rate@{k}']:11.1%} "
                    f"{row[f'mrr@{k}']:8.3f} "
                    f"{row[f'ndcg@{k}']:8.3f} "
                    f"{row[f'article@{k}']:11.1%}"
                )

    print("\n" + "=" * 95)
    print("OVERALL SUMMARY (category = 'all'):")
    print("=" * 95)
    header = f"{'Metric':15s}" + "".join(f"{'K=' + str(k):>12s}" for k in k_values) + f"{'MRR (overall)':>15s}"
    for mode in MODES:
        row = results["all"][mode]
        print(f"\n--- Mode: {mode.upper()} (n = {row['n']}) ---")
        print(header)
        print("-" * len(header))
        print(f"{'Recall@K':15s}" + "".join(f"{row[f'recall@{k}']:12.1%}" for k in k_values) + f"{'-':>15s}")
        print(f"{'Precision@K':15s}" + "".join(f"{row[f'precision@{k}']:12.1%}" for k in k_values) + f"{'-':>15s}")
        print(f"{'HitRate@K':15s}" + "".join(f"{row[f'hit_rate@{k}']:12.1%}" for k in k_values) + f"{'-':>15s}")
        print(f"{'MRR@K':15s}" + "".join(f"{row[f'mrr@{k}']:12.3f}" for k in k_values) + f"{row['mrr']:15.3f}")
        print(f"{'nDCG@K':15s}" + "".join(f"{row[f'ndcg@{k}']:12.3f}" for k in k_values) + f"{'-':>15s}")
        print(f"{'Article@K':15s}" + "".join(f"{row[f'article@{k}']:12.1%}" for k in k_values) + f"{'-':>15s}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
