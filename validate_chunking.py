import re
import sys

import pandas as pd
from transformers import AutoTokenizer

from chunking import chunk_article, iter_articles

MAX_TOKENS = 400
BASELINE_HIT_RATE = {
    "disease": 0.819,
    "medicine": 0.873,
    "drug": 0.899,
    "body-part": 0.818,
}
REGRESSION_MARGIN = 0.03


def normalize(s: str) -> str:
    s = re.sub(r"\*+", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def slug_from_url(url: str) -> str:
    return str(url).rstrip("/").rsplit("/", 1)[-1]


def main() -> int:
    tokenizer = AutoTokenizer.from_pretrained("dangvantuan/vietnamese-embedding")
    token_counter = lambda s: len(tokenizer.encode(s, add_special_tokens=False))

    chunks_by_type_slug: dict[tuple[str, str], list[dict]] = {}
    all_chunks: list[dict] = []
    for article_type, slug, text in iter_articles("data"):
        chunks = chunk_article(text, article_type, slug, token_counter, MAX_TOKENS)
        chunks_by_type_slug[(article_type, slug)] = chunks
        all_chunks.extend(chunks)

    ok = True

    # invariant: token cap respected
    over_cap = [c for c in all_chunks if c["token_count"] > MAX_TOKENS]
    if over_cap:
        print(f"FAIL: {len(over_cap)} chunks exceed {MAX_TOKENS} tokens")
        ok = False

    # invariant: no empty body (text always starts with a non-empty
    # "[Chủ đề: ...]" header, so checking the whole text is a tautology —
    # check the body after the header/body-separating blank line instead)
    empty = [c for c in all_chunks if not c["text"].split("\n\n", 1)[-1].strip()]
    if empty:
        print(f"FAIL: {len(empty)} chunks have empty body")
        ok = False

    # invariant: unique ids
    ids = [c["id"] for c in all_chunks]
    if len(set(ids)) != len(ids):
        print(f"FAIL: duplicate chunk ids ({len(ids) - len(set(ids))} dupes)")
        ok = False

    # regression check: context containment hit-rate per type
    for article_type, baseline in BASELINE_HIT_RATE.items():
        df = pd.read_csv(f"test_case/{article_type}.csv", encoding="utf-8")
        hit, total = 0, 0
        for _, row in df.iterrows():
            slug = slug_from_url(row["article_url"])
            chunks = chunks_by_type_slug.get((article_type, slug))
            if chunks is None:
                continue
            total += 1
            ctx_norm = normalize(str(row["context"]))
            if any(ctx_norm in normalize(c["text"]) for c in chunks):
                hit += 1
        rate = hit / total if total else 0.0
        status = "OK" if rate >= baseline - REGRESSION_MARGIN else "FAIL"
        print(f"{status}: {article_type} hit-rate {hit}/{total} = {rate:.1%} (baseline {baseline:.1%})")
        if status == "FAIL":
            ok = False

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
