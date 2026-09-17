# Medical RAG — Chunking Strategy Design

Date: 2026-09-17
Status: approved (chunking sub-project only; retrieval/generation are separate future sub-projects)

## Context

`data/<type>/*.txt` (type = disease, medicine, drug, body-part) holds crawled
YouMed articles, one file per article. `test_case/<type>.csv` holds QA rows
(`question`, `answer`, `context`, `article_url`, ...) exported from the
ViMedAQA dataset (`testset/`), scoped to the same articles as `data/`. This
spec defines how `data/*.txt` gets split into chunks for ingestion into
Qdrant, using `dangvantuan/vietnamese-embedding` (PhoBERT-based, 768-dim,
512-token max sequence length) as the embedding model. The pipeline is
standalone (no LightRAG dependency) — a plain function producing chunk
dicts that get embedded and upserted into Qdrant directly.

A sibling project (`KLTN/LightRAG/lightrag/chunking_semantic.py`) already
solves the same shape of problem (same `# SOURCE_URL:` / H1 / `##` file
format) for a different embedding model (2048-token cap). This design
reuses its pipeline shape and libraries, adapted for our much smaller
512-token cap.

## Source data shape

Every `.txt` file:
```
# SOURCE_URL: <url>

# <Article title>

Nội dung bài viết

**<bold intro paragraph>**

## <H2 section>
...
### <H3 subsection>
...
#### <H4 subsection>
...
```

Measured across all 584 sampled files (`data/`):

| type | files | sections/doc (all header levels) | median section | p90 |
|---|---|---|---|---|
| disease | 127 | 13.7 | 449 chars | 1103 |
| medicine | 130 | 13.6 | 266 chars | 795 |
| drug | 168 | 17.3 | 271 chars | 668 |
| body-part | 159 | 17.1 | 373 chars | 969 |

`test_case/<type>.csv`'s `context` field is a whitespace/markdown-normalized
copy of one section's body (paragraph + its list), not a full article and
not an arbitrary span. Validated by containment check (see Validation).

## Decision: split at every header level (H2–H6), not H2-only

Two options were measured directly against this data:

1. **H2-only split** (what the LightRAG sibling project does): median
   chunk 700–940 chars, but with our 400-token cap, 4–33% of chunks need
   further guard-rail splitting (character-based, no markdown awareness).
2. **All-level split** (H2 through H6, each header a candidate chunk
   boundary): median chunk 270–450 chars, only 0.8–3.4% need guard-rail
   splitting.

Both gave the *same* context-containment rate (see Validation) — the ~10%
containment misses are contexts spanning two sections, not information
lost by splitting deeper. Since going deeper costs nothing in containment
and avoids most guard-rail intervention, **all-level splitting wins** given
our small token cap. (H2-only would be the right call for a model with a
much larger context window, like the sibling project's 2048-token model.)

## Algorithm

1. **Strip `# SOURCE_URL:`** line → `article_url`.
2. **Extract H1** → `article_title` (first `# ` line that isn't
   `# SOURCE_URL:`).
3. **Split** the remaining markdown with
   `langchain_text_splitters.MarkdownHeaderTextSplitter`, configured for
   `["##", "###", "####", "#####"]`, `strip_headers=False`. Each resulting
   doc's `metadata` gives the breadcrumb (H2 > H3 > H4 > H5 present at that
   point); `page_content` is the leaf section body. Drop docs with empty
   `page_content` (a header that only leads into child headers, no text of
   its own — no salvageable content).
4. **Inject context header**: prepend
   `[Chủ đề: <title> | Mục: <breadcrumb> | Nguồn: <url>]` (breadcrumb
   omitted if the chunk is the pre-first-header intro) to each chunk body.
   This is the text actually embedded and shown to the LLM — one field,
   no separate "clean" vs "annotated" copies.
5. **Guard-rail**: if a chunk (context header + body) exceeds 400 tokens
   (measured with the real `dangvantuan/vietnamese-embedding` tokenizer,
   not a char estimate — leaves ~110 tokens of headroom under the model's
   512 cap for the context header and any tokenizer overhead), re-split the
   *body* with `RecursiveCharacterTextSplitter(chunk_size=400,
   chunk_overlap=0, length_function=<tokenizer-based>, separators=["\n\n",
   "\n", ". ", " ", ""])`, then re-prefix every resulting sub-chunk with the
   same context header (breadcrumb gets a `(phần i/n)` suffix). No overlap
   between normal section chunks — sections are already independent
   semantic units; overlap only applies inside a guard-rail split, and even
   there we default to 0 (no evidence yet that splitting a single
   over-long section needs it — revisit if the guard-rail path is measured
   to hurt recall).

## Metadata / Qdrant payload

One Qdrant point per chunk:

- `id`: UUID5 derived from `f"{type}:{article_slug}:{chunk_index}"`
  (deterministic → re-ingestion overwrites instead of duplicating).
- payload:
  - `type` — disease / medicine / drug / body-part
  - `article_slug`, `article_title`, `article_url`
  - `section_path` — breadcrumb string, e.g. `"Ai không nên sử dụng X > Đối tượng thận trọng"`
  - `text` — context-header + body (embedded as-is, returned to LLM as-is)
  - `token_count`
  - `split_part` — `"i/n"` if guard-rail split this section, else `null`

Retrieval is decided (see Out of scope) as hybrid search: dense cosine
(`dangvantuan/vietnamese-embedding`) + sparse BM25, both computed from the
same `text` field — Qdrant stores dense and sparse vectors on the same
point, so no extra payload field or second copy of the text is needed for
this.

## Re-ranking

Hybrid search candidates get reranked with a cross-encoder,
`AITeamVN/Vietnamese_Reranker` (fine-tuned from `bge-reranker-v2-m3`,
XLM-RoBERTa-based, sequence-classification / cross-encoder — scores a
query-passage pair directly rather than comparing separate embeddings).
Max sequence length 2304 tokens (256 query + 2048 passage), well above our
400-token chunk cap — no truncation risk from chunk size.

Flow: hybrid search returns top-N candidates (N to be tuned, e.g. 20–50) →
reranker scores each `(question, chunk.text)` pair → keep top-k (e.g. 5) by
reranker score for the LLM context. Reranking parameters (N, k, whether to
batch) and eval belong to the retrieval sub-project's own spec — noted here
only because the model choice is now decided and it doesn't affect the
chunking design (chunk `text` field is already what gets passed as the
passage).

## Known limitations (accepted, not engineered around)

- ~10% of eval `context` spans cross a section boundary (e.g. answer built
  from the tail of one section + head of the next). Per-section chunking
  cannot retrieve these in one chunk. Accepted: fixing this would need
  cross-section overlap or answer-aware chunking, disproportionate to the
  benefit for a first version. Revisit only if retrieval eval on the real
  question set shows this class of miss matters in practice.
- Guard-rail split (0.8–3.4% of chunks) uses generic paragraph/sentence
  boundaries, not markdown-aware — acceptable because it's rare and still
  respects paragraph boundaries first.

## Validation

`validate_chunking.py` (repo root): runs the chunker over every file in
`data/`, then for each row in `test_case/<type>.csv`, normalizes `context`
(strip `**`/`*`, collapse whitespace) and checks it's a substring of at
least one chunk's body from the same article. Prints hit-rate per type and
asserts:
- hit-rate per type stays at or above the measured baseline minus a small
  margin (disease 88.8%, medicine 91.9%, drug 90.9%, body-part 86.4%; using
  −3pp as the regression margin) — catches future edits to the chunker
  that quietly break section boundaries.
- every emitted chunk's `token_count <= 400`.
- no chunk has empty `text`.
- no duplicate `id`.

This script is both the acceptance test for this design and the ongoing
regression check for the chunker implementation.

## Out of scope (future sub-projects)

- Embedding + Qdrant upsert pipeline (this spec only produces chunk dicts).
- Retrieval strategy: **decided as hybrid search (dense cosine + sparse
  BM25, fused e.g. RRF) reranked with `AITeamVN/Vietnamese_Reranker`**
  (see Re-ranking section); top-N/top-k tuning and filtering by `type` are
  still open and belong to the retrieval sub-project's own spec.
- Answer generation / LLM prompting.
- Parent-child (small-to-big) chunking — flagged as a later upgrade path if
  section-level retrieval precision proves insufficient on real queries.
