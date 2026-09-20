import re
import time
import unicodedata
import zlib
from collections import Counter

import requests
from pyvi import ViTokenizer

DENSE_DIM = 768
_TRANSIENT_STATUS = {502, 503, 504, 530}  # Cloudflare error 1033 arrives as HTTP 530
_GATEWAY_TIMEOUT = 524


def segment(text: str) -> str:
    return ViTokenizer.tokenize(text)


def embed_input(chunk: dict) -> str:
    return chunk["text"]


_TERM_RE = re.compile(r"\w+")


def terms(segmented: str) -> list[str]:
    return _TERM_RE.findall(unicodedata.normalize("NFC", segmented).lower())


def term_index(term: str) -> int:
    return zlib.crc32(term.encode("utf-8")) & 0xFFFFFFFF


class Bm25Encoder:
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.avgdl = None

    def fit(self, segmented_docs: list[str]) -> None:
        lengths = [len(terms(doc)) for doc in segmented_docs]
        if not lengths or sum(lengths) == 0:
            raise ValueError("cannot fit BM25 on an empty corpus")
        self.avgdl = sum(lengths) / len(lengths)

    def encode_doc(self, segmented: str) -> tuple[list[int], list[float]]:
        if self.avgdl is None:
            raise RuntimeError("call fit() before encode_doc()")
        tokens = terms(segmented)
        if not tokens:
            return [], []
        dl = len(tokens)
        tf = Counter(term_index(t) for t in tokens)
        norm = self.k1 * (1 - self.b + self.b * dl / self.avgdl)
        indices = sorted(tf)
        return indices, [tf[i] * (self.k1 + 1) / (tf[i] + norm) for i in indices]

    def encode_query(self, segmented: str) -> tuple[list[int], list[float]]:
        indices = sorted({term_index(t) for t in terms(segmented)})
        return indices, [1.0] * len(indices)


class EmbedError(Exception):
    pass


class _GatewayTimeout(Exception):
    pass


class EmbedClient:
    def __init__(
        self,
        base_url: str,
        batch_size: int = 32,
        timeout: float = 60.0,
        retries: int = 3,
        backoff: float = 1.0,
    ):
        self.url = base_url.rstrip("/") + "/embed"
        self.batch_size = batch_size
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff
        self._session = requests.Session()

    def embed(self, segmented_texts: list[str], task: str = "document") -> list[list[float]]:
        vectors = []
        i, size = 0, self.batch_size
        while i < len(segmented_texts):
            batch = segmented_texts[i:i + size]
            try:
                vectors.extend(self._post(batch, task))
            except _GatewayTimeout:
                # size stays halved for the rest of this call; embed() is called once per article, so the effect is bounded
                size = max(1, len(batch) // 2)
                continue
            i += len(batch)
        return vectors

    def health_check(self) -> None:
        try:
            self.embed(["xin chào"])
        except EmbedError as e:
            raise EmbedError(
                f"embed server not responding, start serve_model/embed_server.py and update EMBED_URL: {e}"
            ) from e

    def _post(self, batch: list[str], task: str = "document") -> list[list[float]]:
        body = {"texts": batch}
        if task == "query":
            body["task"] = task  # only embed_server.py reads it; the Kaggle server ignores it
        last_error = None
        for attempt in range(self.retries):
            if attempt:
                time.sleep(self.backoff * 2 ** (attempt - 1))
            try:
                resp = self._session.post(self.url, json=body, timeout=self.timeout)
            except (requests.ConnectionError, requests.Timeout) as e:
                last_error = e
                continue
            except requests.RequestException as e:
                raise EmbedError(f"invalid request to {self.url}: {e}") from e
            if resp.status_code == _GATEWAY_TIMEOUT and len(batch) > 1:
                raise _GatewayTimeout()
            if resp.status_code == _GATEWAY_TIMEOUT or resp.status_code in _TRANSIENT_STATUS:
                last_error = EmbedError(f"HTTP {resp.status_code}")
                continue
            if resp.status_code != 200:
                raise EmbedError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            try:
                vectors = resp.json()["embeddings"]
            except (ValueError, KeyError, TypeError) as e:
                raise EmbedError(f"malformed /embed response: {e}") from e
            if (
                not isinstance(vectors, list)
                or len(vectors) != len(batch)
                or any(not isinstance(v, list) or len(v) != DENSE_DIM for v in vectors)
            ):
                raise EmbedError(
                    f"expected {len(batch)} vectors of dimension {DENSE_DIM} from /embed"
                )
            return vectors
        raise EmbedError(f"gave up after {self.retries} attempts: {last_error}")
