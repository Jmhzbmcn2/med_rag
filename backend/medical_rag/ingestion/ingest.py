import argparse
import os
import sys
from dataclasses import dataclass, field
from typing import Callable

from transformers import AutoTokenizer

from medical_rag.encoders import (
    Bm25Encoder,
    EmbedClient,
    EmbedError,
    embed_input,
    segment,
)
from medical_rag.ingestion.chunking import chunk_article, iter_articles
from medical_rag.store import COLLECTION, build_point, ensure_collection, open_client, replace_article

TOKENIZER_NAME = "dangvantuan/vietnamese-embedding"


@dataclass
class IngestReport:
    articles_ok: int = 0
    articles_failed: list[str] = field(default_factory=list)
    points_upserted: int = 0
    expected_chunks: int = 0


def ingest(
    data_dir: str,
    client,
    embedder: Callable[[list[str]], list[list[float]]],
    token_counter: Callable[[str], int],
) -> IngestReport:
    articles = []
    for article_type, slug, text in iter_articles(data_dir):
        chunks = chunk_article(text, article_type, slug, token_counter)
        segmented = [segment(embed_input(chunk)) for chunk in chunks]
        articles.append((article_type, slug, chunks, segmented))

    encoder = Bm25Encoder()
    encoder.fit([doc for _, _, _, segmented in articles for doc in segmented])

    report = IngestReport(expected_chunks=sum(len(chunks) for _, _, chunks, _ in articles))
    for number, (article_type, slug, chunks, segmented) in enumerate(articles, 1):
        try:
            dense = embedder(segmented)
            if len(dense) != len(chunks):
                raise EmbedError(f"embedder returned {len(dense)} vectors for {len(chunks)} chunks")
        except EmbedError as error:
            report.articles_failed.append(f"{article_type}/{slug}: {error}")
            print(f"[{number}/{len(articles)}] FAILED {article_type}/{slug}: {error}", flush=True)
            continue
        points = [
            build_point(chunk, vector, encoder.encode_doc(doc))
            for chunk, vector, doc in zip(chunks, dense, segmented)
        ]
        try:
            replace_article(client, article_type, slug, points)
        except Exception as error:  # per-article isolation boundary
            report.articles_failed.append(f"{article_type}/{slug}: store error: {error!r}")
            print(
                f"[{number}/{len(articles)}] STORE ERROR {article_type}/{slug}: {error!r}; "
                "stopping, the local index may be inconsistent, re-run with --recreate",
                flush=True,
            )
            break
        report.articles_ok += 1
        report.points_upserted += len(points)
        print(f"[{number}/{len(articles)}] {article_type}/{slug} ({len(points)} chunks)", flush=True)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Embed chunks and load them into Qdrant.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--qdrant-path", default="qdrant_data")
    parser.add_argument("--embed-url", default=os.environ.get("EMBED_URL"))
    parser.add_argument("--recreate", action="store_true")
    args = parser.parse_args(argv)

    if not args.embed_url:
        print("EMBED_URL is not set (or pass --embed-url): copy the Cloudflare tunnel URL from the Kaggle notebook.")
        return 2
    embed_client = EmbedClient(args.embed_url)
    try:
        embed_client.health_check()
        qdrant = open_client(args.qdrant_path)
        ensure_collection(qdrant, recreate=args.recreate)
    except (EmbedError, ValueError) as error:
        print(f"startup failed: {error}")
        return 2

    try:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
        token_counter = lambda text: len(tokenizer.encode(text, add_special_tokens=False))  # noqa: E731
        report = ingest(args.data_dir, qdrant, embed_client.embed, token_counter)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"ingest aborted: {error}")
        return 2

    total = qdrant.count(COLLECTION, exact=True).count
    print(
        f"articles ok={report.articles_ok} failed={len(report.articles_failed)} "
        f"points upserted={report.points_upserted} collection points={total} "
        f"expected chunks={report.expected_chunks}"
    )
    for failure in report.articles_failed:
        print(f"  failed: {failure}")
    return 1 if report.articles_failed else 0


if __name__ == "__main__":
    sys.exit(main())
