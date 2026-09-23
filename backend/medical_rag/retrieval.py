import logging
from dataclasses import dataclass
from typing import Callable, Literal

import requests
from qdrant_client import models

from medical_rag.encoders import Bm25Encoder, segment
from medical_rag.store import COLLECTION

CANDIDATES = 20
log = logging.getLogger(__name__)

SearchMode = Literal["dense", "sparse", "hybrid"]


class RerankError(Exception):
    pass


class RerankClient:
    def __init__(self, base_url: str, timeout: float = 30.0):
        self.url = base_url.rstrip("/") + "/rerank"
        self.timeout = timeout

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        try:
            resp = requests.post(self.url, json={"query": query, "documents": documents}, timeout=self.timeout)
            resp.raise_for_status()
            scores = resp.json()["scores"]
        except (requests.RequestException, ValueError, KeyError, TypeError) as e:
            raise RerankError(str(e)) from e
        if (
            not isinstance(scores, list)
            or len(scores) != len(documents)
            or any(isinstance(s, bool) or not isinstance(s, (int, float)) for s in scores)
        ):
            raise RerankError(f"expected {len(documents)} numeric scores from /rerank")
        return scores


@dataclass
class Hit:
    id: str
    text: str
    type: str
    article_title: str
    article_url: str
    section_path: str
    retrieval_score: float
    score: float | None = None


def _hit(point) -> Hit:
    p = point.payload
    return Hit(
        id=str(point.id),
        text=p["text"],
        type=p["type"],
        article_title=p["article_title"],
        article_url=p["article_url"],
        section_path=p["section_path"],
        retrieval_score=float(point.score),
    )


class Retriever:
    def __init__(
        self,
        client,
        embed: Callable[..., list[list[float]]],
        rerank: Callable[[str, list[str]], list[float]] | None = None,
    ):
        self.client = client
        self.embed = embed
        self.rerank = rerank
        self._bm25 = Bm25Encoder()  # queries need no fit()

    def search(self, question: str, k: int = 5, mode: SearchMode = "hybrid") -> list[Hit]:
        segmented = segment(question)
        limit = CANDIDATES if self.rerank else k

        if mode in ("dense", "hybrid"):
            dense = self.embed([segmented], task="query")[0]

        if mode == "dense":
            points = self.client.query_points(
                COLLECTION, query=dense, using="dense", limit=limit, with_payload=True
            ).points
        else:
            indices, values = self._bm25.encode_query(segmented)
            if mode == "sparse":
                if not indices:
                    return []
                points = self.client.query_points(
                    COLLECTION,
                    query=models.SparseVector(indices=indices, values=values),
                    using="sparse",
                    limit=limit,
                    with_payload=True,
                ).points
            elif not indices:
                points = self.client.query_points(
                    COLLECTION, query=dense, using="dense", limit=limit, with_payload=True
                ).points
            else:
                points = self.client.query_points(
                    COLLECTION,
                    prefetch=[
                        models.Prefetch(query=dense, using="dense", limit=CANDIDATES),
                        models.Prefetch(
                            query=models.SparseVector(indices=indices, values=values),
                            using="sparse",
                            limit=CANDIDATES,
                        ),
                    ],
                    query=models.FusionQuery(fusion=models.Fusion.RRF),
                    limit=limit,
                    with_payload=True,
                ).points

        hits = [_hit(point) for point in points]
        if self.rerank and hits:
            try:
                scores = self.rerank(question, [hit.text for hit in hits])
            except RerankError as error:
                log.warning("rerank failed, keeping retrieval order: %s", error)
                return hits[:k]
            for hit, score in zip(hits, scores):
                hit.score = score
            hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:k]
