import uuid

import pytest
from qdrant_client import models

from medical_rag.encoders import DENSE_DIM, Bm25Encoder, segment
from medical_rag.store import (
    COLLECTION,
    build_point,
    ensure_collection,
    open_client,
    replace_article,
)


def _dense(axis: int) -> list[float]:
    vector = [0.0] * DENSE_DIM
    vector[axis] = 1.0
    return vector


def _chunk(name: str, article_type: str = "drug", slug: str = "a", text: str = "nội dung") -> dict:
    return {
        "id": str(uuid.uuid5(uuid.NAMESPACE_URL, name)),
        "type": article_type,
        "article_slug": slug,
        "article_title": "T",
        "article_url": "U",
        "section_path": "S",
        "text": text,
        "token_count": 3,
        "split_part": None,
    }


def _count(client) -> int:
    return client.count(COLLECTION, exact=True).count


def test_ensure_collection_creates_and_is_idempotent():
    client = open_client(None)
    ensure_collection(client)
    ensure_collection(client)
    params = client.get_collection(COLLECTION).config.params
    assert params.vectors["dense"].size == DENSE_DIM
    assert params.vectors["dense"].distance == models.Distance.COSINE
    assert params.sparse_vectors["sparse"].modifier == models.Modifier.IDF


def test_ensure_collection_raises_on_dense_size_mismatch():
    client = open_client(None)
    client.create_collection(
        COLLECTION,
        vectors_config={"dense": models.VectorParams(size=4, distance=models.Distance.COSINE)},
    )
    with pytest.raises(ValueError, match="dense size 4"):
        ensure_collection(client)


def test_ensure_collection_recreate_drops_existing_points():
    client = open_client(None)
    ensure_collection(client)
    client.upsert(COLLECTION, points=[build_point(_chunk("x"), _dense(0), ([], []))])
    assert _count(client) == 1
    ensure_collection(client, recreate=True)
    assert _count(client) == 0


def test_build_point_payload_excludes_id_and_sparse_is_omitted_when_empty():
    chunk = _chunk("x")
    with_sparse = build_point(chunk, _dense(0), ([5, 9], [0.5, 1.5]))
    assert with_sparse.id == chunk["id"]
    assert "id" not in with_sparse.payload
    assert with_sparse.payload["article_slug"] == "a"
    assert set(with_sparse.vector) == {"dense", "sparse"}
    assert with_sparse.vector["sparse"].indices == [5, 9]

    without_sparse = build_point(chunk, _dense(0), ([], []))
    assert set(without_sparse.vector) == {"dense"}


def test_replace_article_replaces_only_that_article_and_is_idempotent():
    client = open_client(None)
    ensure_collection(client)
    a = [_chunk("a1", slug="a"), _chunk("a2", slug="a"), _chunk("a3", slug="a")]
    b = [_chunk("b1", slug="b")]
    same_slug_other_type = [_chunk("c1", article_type="disease", slug="a")]
    for article_type, slug, chunks in (
        ("drug", "a", a),
        ("drug", "b", b),
        ("disease", "a", same_slug_other_type),
    ):
        replace_article(client, article_type, slug, [build_point(c, _dense(0), ([], [])) for c in chunks])
    assert _count(client) == 5

    replace_article(client, "drug", "a", [build_point(a[0], _dense(0), ([], []))])
    assert _count(client) == 3  # a shrank from 3 to 1; b and disease/a untouched

    replace_article(client, "drug", "a", [build_point(a[0], _dense(0), ([], []))])
    assert _count(client) == 3  # re-running does not duplicate

    replace_article(client, "drug", "a", [])
    assert _count(client) == 2  # an article that became empty is removed


def test_sparse_query_ranks_rare_term_above_common_term():
    docs = ["bệnh_nhân đau", "bệnh_nhân sốt", "bệnh_nhân ho"]
    encoder = Bm25Encoder()
    encoder.fit(docs)
    client = open_client(None)
    ensure_collection(client)
    chunks = [_chunk(f"d{i}", slug=f"s{i}") for i in range(3)]
    client.upsert(
        COLLECTION,
        points=[build_point(c, _dense(i), encoder.encode_doc(d)) for i, (c, d) in enumerate(zip(chunks, docs))],
    )

    indices, values = encoder.encode_query("bệnh_nhân sốt")
    result = client.query_points(
        COLLECTION, query=models.SparseVector(indices=indices, values=values), using="sparse", limit=3
    )
    assert result.points[0].payload["article_slug"] == "s1"  # the only doc with the rare term "sốt"


def test_hybrid_query_with_rrf_fusion_returns_the_doc_both_signals_agree_on():
    docs = ["bệnh_nhân đau", "bệnh_nhân sốt", "bệnh_nhân ho"]
    encoder = Bm25Encoder()
    encoder.fit(docs)
    client = open_client(None)
    ensure_collection(client)
    chunks = [_chunk(f"h{i}", slug=f"s{i}") for i in range(3)]
    client.upsert(
        COLLECTION,
        points=[build_point(c, _dense(i), encoder.encode_doc(d)) for i, (c, d) in enumerate(zip(chunks, docs))],
    )

    indices, values = encoder.encode_query(segment("sốt"))
    result = client.query_points(
        COLLECTION,
        prefetch=[
            models.Prefetch(query=_dense(1), using="dense", limit=3),
            models.Prefetch(query=models.SparseVector(indices=indices, values=values), using="sparse", limit=3),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=3,
    )
    assert result.points[0].payload["article_slug"] == "s1"
