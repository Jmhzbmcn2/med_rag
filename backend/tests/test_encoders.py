import pytest
from fake_server import fake_embed_server, ok_body

from medical_rag.encoders import (
    DENSE_DIM,
    Bm25Encoder,
    EmbedClient,
    EmbedError,
    embed_input,
    segment,
    term_index,
    terms,
)


def test_dense_dim_matches_the_embedding_model():
    assert DENSE_DIM == 768


def test_segment_joins_compounds_and_is_deterministic():
    assert segment("Bệnh nhân bị đau đầu.") == "Bệnh_nhân bị đau_đầu ."
    assert segment("Bệnh nhân bị đau đầu.") == segment("Bệnh nhân bị đau đầu.")
    assert segment("") == ""


def test_embed_input_is_the_chunk_text_unchanged():
    chunk = {"text": "[Chủ đề: T | Nguồn: U]\n\nThân bài."}
    assert embed_input(chunk) == "[Chủ đề: T | Nguồn: U]\n\nThân bài."


def test_terms_normalizes_nfc_lowercases_and_keeps_compounds():
    assert terms("Bệnh_Nhân đau 325 mg") == ["bệnh_nhân", "đau", "325", "mg"]
    assert terms("é") == terms("é") == ["é"]
    assert terms("") == []


def test_term_index_is_stable_u32():
    assert term_index("đau") == term_index("đau")
    assert 0 <= term_index("đau") <= 0xFFFFFFFF
    assert term_index("đau") != term_index("sốt")


def test_bm25_doc_weights_match_hand_computed_values():
    enc = Bm25Encoder(k1=1.2, b=0.75)
    enc.fit(["a b", "a c c"])  # avgdl = (2 + 3) / 2 = 2.5

    indices, values = enc.encode_doc("a b")  # dl=2, norm=1.2*(0.25+0.75*2/2.5)=1.02
    weights = dict(zip(indices, values))
    assert weights[term_index("a")] == pytest.approx(2.2 / 2.02, abs=1e-6)
    assert weights[term_index("b")] == pytest.approx(2.2 / 2.02, abs=1e-6)

    indices, values = enc.encode_doc("a c c")  # dl=3, norm=1.2*(0.25+0.75*3/2.5)=1.38
    weights = dict(zip(indices, values))
    assert weights[term_index("a")] == pytest.approx(2.2 / 2.38, abs=1e-6)
    assert weights[term_index("c")] == pytest.approx(4.4 / 3.38, abs=1e-6)


def test_bm25_longer_doc_gets_smaller_weight_for_equal_tf():
    enc = Bm25Encoder()
    enc.fit(["a b", "a b c d e f g h"])
    _, short = enc.encode_doc("a b")
    long_idx, long_vals = enc.encode_doc("a b c d e f g h")
    long_weights = dict(zip(long_idx, long_vals))
    assert long_weights[term_index("a")] < short[0]


def test_bm25_indices_are_sorted_unique_and_empty_doc_is_empty():
    enc = Bm25Encoder()
    enc.fit(["x y z"])
    indices, values = enc.encode_doc("z y x x")
    assert indices == sorted(set(indices))
    assert len(indices) == len(values) == 3
    assert enc.encode_doc("") == ([], [])


def test_bm25_query_uses_weight_one_per_distinct_term_without_fit():
    enc = Bm25Encoder()
    indices, values = enc.encode_query("đau đau sốt")
    assert indices == sorted({term_index("đau"), term_index("sốt")})
    assert values == [1.0, 1.0]


def test_bm25_errors_on_empty_corpus_and_encode_before_fit():
    with pytest.raises(ValueError):
        Bm25Encoder().fit([])
    with pytest.raises(ValueError):
        Bm25Encoder().fit(["", "  "])
    with pytest.raises(RuntimeError):
        Bm25Encoder().encode_doc("a")


def test_embed_client_batches_requests():
    with fake_embed_server(lambda texts, n: (200, ok_body(texts))) as (url, calls):
        vectors = EmbedClient(url, batch_size=2).embed(["a", "b", "c", "d", "e"])
    assert len(vectors) == 5
    assert calls == [2, 2, 1]


def test_embed_client_empty_input_makes_no_request():
    with fake_embed_server(lambda texts, n: (200, ok_body(texts))) as (url, calls):
        assert EmbedClient(url).embed([]) == []
    assert calls == []


def test_embed_client_halves_batch_after_gateway_timeout():
    def script(texts, n):
        return (524, {}) if n == 0 else (200, ok_body(texts))

    with fake_embed_server(script) as (url, calls):
        vectors = EmbedClient(url, batch_size=4).embed(["a", "b", "c", "d"])
    assert len(vectors) == 4
    assert calls == [4, 2, 2]


def test_embed_client_keeps_halving_down_to_single_texts():
    def script(texts, n):
        return (524, {}) if len(texts) > 1 else (200, ok_body(texts))

    with fake_embed_server(script) as (url, calls):
        vectors = EmbedClient(url, batch_size=4).embed(["a", "b", "c"])
    assert len(vectors) == 3
    assert calls == [3, 1, 1, 1]


def test_embed_client_retries_transient_errors_then_succeeds():
    def script(texts, n):
        return (530, {}) if n < 2 else (200, ok_body(texts))

    with fake_embed_server(script) as (url, calls):
        vectors = EmbedClient(url, retries=3, backoff=0).embed(["a"])
    assert len(vectors) == 1
    assert calls == [1, 1, 1]


def test_embed_client_gives_up_after_retries():
    with fake_embed_server(lambda texts, n: (530, {})) as (url, calls):
        with pytest.raises(EmbedError, match="gave up after 3 attempts"):
            EmbedClient(url, retries=3, backoff=0).embed(["a"])
    assert calls == [1, 1, 1]


def test_embed_client_does_not_retry_client_errors():
    with fake_embed_server(lambda texts, n: (400, {"detail": "bad"})) as (url, calls):
        with pytest.raises(EmbedError, match="HTTP 400"):
            EmbedClient(url, retries=3, backoff=0).embed(["a"])
    assert calls == [1]


def test_embed_client_rejects_wrong_dimension():
    with fake_embed_server(lambda texts, n: (200, {"embeddings": [[0.1, 0.2, 0.3]]})) as (url, _):
        with pytest.raises(EmbedError, match="dimension"):
            EmbedClient(url).embed(["a"])


def test_embed_client_retries_connection_errors_then_raises():
    with pytest.raises(EmbedError, match="gave up after 2 attempts"):
        EmbedClient("http://127.0.0.1:1", retries=2, backoff=0, timeout=2).embed(["a"])


def test_health_check_success_and_failure_message():
    with fake_embed_server(lambda texts, n: (200, ok_body(texts))) as (url, _):
        EmbedClient(url).health_check()
    with pytest.raises(EmbedError, match="update EMBED_URL"):
        EmbedClient("http://127.0.0.1:1", retries=1, backoff=0, timeout=2).health_check()


def test_embed_client_rejects_invalid_url_without_retrying():
    with pytest.raises(EmbedError, match="invalid request"):
        EmbedClient("not-a-url", retries=3, backoff=0).embed(["a"])


def test_embed_client_rejects_a_non_object_json_body():
    with fake_embed_server(lambda texts, n: (200, ["not", "an", "object"])) as (url, _):
        with pytest.raises(EmbedError, match="malformed"):
            EmbedClient(url).embed(["a"])
