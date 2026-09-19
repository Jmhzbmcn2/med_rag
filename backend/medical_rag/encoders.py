import re
import unicodedata
import zlib
from collections import Counter

from pyvi import ViTokenizer

DENSE_DIM = 768


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
