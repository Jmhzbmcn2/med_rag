import warnings

from qdrant_client import QdrantClient, models

from medical_rag.encoders import DENSE_DIM

COLLECTION = "medical_rag"
_INDEXED_FIELDS = ("type", "article_slug")


def open_client(path: str | None) -> QdrantClient:
    return QdrantClient(":memory:") if path is None else QdrantClient(path=path)


def ensure_collection(client: QdrantClient, recreate: bool = False) -> None:
    if client.collection_exists(COLLECTION):
        if not recreate:
            vectors = client.get_collection(COLLECTION).config.params.vectors
            if not isinstance(vectors, dict) or "dense" not in vectors:
                raise ValueError(f"collection {COLLECTION!r} has no named 'dense' vector")
            if vectors["dense"].size != DENSE_DIM:
                raise ValueError(
                    f"collection {COLLECTION!r} has dense size {vectors['dense'].size}, "
                    f"expected {DENSE_DIM}; pass recreate=True to rebuild it"
                )
            return
        client.delete_collection(COLLECTION)

    client.create_collection(
        COLLECTION,
        vectors_config={"dense": models.VectorParams(size=DENSE_DIM, distance=models.Distance.COSINE)},
        sparse_vectors_config={"sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )
    with warnings.catch_warnings():
        # local mode ignores payload indexes; they matter once this moves to a Qdrant server
        warnings.filterwarnings("ignore", message="Payload indexes have no effect")
        for field in _INDEXED_FIELDS:
            client.create_payload_index(COLLECTION, field, models.PayloadSchemaType.KEYWORD)


def build_point(
    chunk: dict, dense: list[float], sparse: tuple[list[int], list[float]]
) -> models.PointStruct:
    vector = {"dense": dense}
    indices, values = sparse
    if indices:
        vector["sparse"] = models.SparseVector(indices=indices, values=values)
    payload = {key: value for key, value in chunk.items() if key != "id"}
    return models.PointStruct(id=chunk["id"], vector=vector, payload=payload)


def replace_article(
    client: QdrantClient,
    article_type: str,
    article_slug: str,
    points: list[models.PointStruct],
) -> None:
    article_filter = models.Filter(
        must=[
            models.FieldCondition(key="type", match=models.MatchValue(value=article_type)),
            models.FieldCondition(key="article_slug", match=models.MatchValue(value=article_slug)),
        ]
    )
    client.delete(COLLECTION, points_selector=models.FilterSelector(filter=article_filter))
    if points:
        client.upsert(COLLECTION, points=points)
