import logging
from dataclasses import dataclass
from typing import Callable

import requests
from qdrant_client import models

from medical_rag.encoders import Bm25Encoder, segment
from medical_rag.store import COLLECTION

CANDIDATES = 20
log = logging.getLogger(__name__)


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

    def search(self, question: str, k: int = 5) -> list[Hit]:
        segmented = segment(question)
        dense = self.embed([segmented], task="query")[0]
        indices, values = self._bm25.encode_query(segmented)
        prefetch = [models.Prefetch(query=dense, using="dense", limit=CANDIDATES)]
        if indices:
            sparse = models.SparseVector(indices=indices, values=values)
            prefetch.append(models.Prefetch(query=sparse, using="sparse", limit=CANDIDATES))
        points = self.client.query_points(
            COLLECTION,
            prefetch=prefetch,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=CANDIDATES if self.rerank else k,
            with_payload=True,
        ).points
        hits = [_hit(point) for point in points]
        if self.rerank and hits:
            try:
                scores = self.rerank(question, [hit.text for hit in hits])
            except RerankError as error:
                log.warning("rerank failed, keeping fusion order: %s", error)
                return hits[:k]
            for hit, score in zip(hits, scores):
                hit.score = score
            hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:k]
