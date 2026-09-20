from fake_server import fake_embed_server, ok_body
from medical_rag.encoders import DENSE_DIM, EmbedError
from medical_rag.ingestion import ingest as ingest_module
from medical_rag.ingestion.ingest import ingest
from medical_rag.store import COLLECTION, ensure_collection, open_client


def _word_count(text: str) -> int:
    return len(text.split())


def fake_embedder(texts):
    return [[1.0 + len(text) % 5] + [0.0] * (DENSE_DIM - 1) for text in texts]


def _article(slug: str, intro: str, sections: list[tuple[str, str]]) -> str:
    body = "".join(f"\n## {heading}\n\n{text}\n" for heading, text in sections)
    return (
        f"# SOURCE_URL: https://youmed.vn/tin-tuc/{slug}/\n\n# Tiêu đề {slug}\n\n"
        f"Nội dung bài viết\n\n**{intro}**\n{body}"
    )


def _write(root, article_type: str, slug: str, text: str) -> None:
    folder = root / article_type
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{slug}.txt").write_text(text, encoding="utf-8")


def _setup(tmp_path):
    _write(
        tmp_path, "drug", "thuoc-a",
        _article("thuoc-a", "Giới thiệu thuốc A.", [("Công dụng", "Giảm đau."), ("Cách dùng", "Uống 1 viên.")]),
    )
    _write(
        tmp_path, "disease", "benh-b",
        _article("benh-b", "Giới thiệu bệnh B FAILME.", [("Triệu chứng", "Sốt và ho.")]),
    )
    client = open_client(None)
    ensure_collection(client)
    return client


def _count(client) -> int:
    return client.count(COLLECTION, exact=True).count


def test_ingest_loads_every_chunk_with_payload_and_both_vectors(tmp_path):
    client = _setup(tmp_path)
    report = ingest(str(tmp_path), client, fake_embedder, _word_count)

    assert report.articles_ok == 2
    assert report.articles_failed == []
    assert report.points_upserted == 5  # thuoc-a: intro + 2 sections, benh-b: intro + 1 section
    assert report.expected_chunks == 5
    assert _count(client) == 5

    points, _ = client.scroll(COLLECTION, limit=10, with_payload=True, with_vectors=True)
    assert {p.payload["type"] for p in points} == {"drug", "disease"}
    for point in points:
        assert set(point.vector) == {"dense", "sparse"}
        assert len(point.vector["dense"]) == DENSE_DIM
        assert "id" not in point.payload
        assert {"article_slug", "article_title", "article_url", "section_path", "text"} <= set(point.payload)


def test_ingest_is_idempotent_and_removes_stale_chunks(tmp_path):
    client = _setup(tmp_path)
    ingest(str(tmp_path), client, fake_embedder, _word_count)
    ingest(str(tmp_path), client, fake_embedder, _word_count)
    assert _count(client) == 5

    _write(
        tmp_path, "drug", "thuoc-a",
        _article("thuoc-a", "Giới thiệu thuốc A.", [("Công dụng", "Giảm đau.")]),
    )
    ingest(str(tmp_path), client, fake_embedder, _word_count)
    assert _count(client) == 4  # thuoc-a shrank from 3 chunks to 2; positional ids shifted


def test_failed_article_keeps_old_points_and_others_still_load(tmp_path):
    client = _setup(tmp_path)
    ingest(str(tmp_path), client, fake_embedder, _word_count)
    assert _count(client) == 5

    _write(
        tmp_path, "drug", "thuoc-a",
        _article("thuoc-a", "Giới thiệu thuốc A.", [("Công dụng", "Giảm đau.")]),
    )

    def flaky_embedder(texts):
        if any("FAILME" in text for text in texts):
            raise EmbedError("tunnel down")
        return fake_embedder(texts)

    report = ingest(str(tmp_path), client, flaky_embedder, _word_count)

    assert report.articles_ok == 1
    assert len(report.articles_failed) == 1
    assert report.articles_failed[0].startswith("disease/benh-b")
    assert _count(client) == 4  # thuoc-a replaced (3 -> 2), benh-b's 2 old points untouched


def test_main_exits_2_when_embed_url_is_missing(monkeypatch, capsys):
    monkeypatch.delenv("EMBED_URL", raising=False)
    assert ingest_module.main(["--data-dir", "nowhere"]) == 2
    assert "EMBED_URL" in capsys.readouterr().out


def test_main_exits_2_when_tunnel_is_unreachable(monkeypatch, capsys):
    assert ingest_module.main(["--embed-url", "http://127.0.0.1:1"]) == 2
    assert "update EMBED_URL" in capsys.readouterr().out


def _main_args(url, data_dir, tmp_path):
    return ["--embed-url", url, "--data-dir", str(data_dir), "--qdrant-path", str(tmp_path / "q")]


def test_main_exits_2_when_data_dir_is_missing(tmp_path, capsys):
    with fake_embed_server(lambda texts, n: (200, ok_body(texts))) as (url, _):
        code = ingest_module.main(_main_args(url, tmp_path / "nowhere", tmp_path))
    assert code == 2
    out = capsys.readouterr().out
    assert "ingest aborted" in out
    assert "nowhere" in out  # names the missing directory, so a tokenizer failure cannot satisfy this


def test_main_exits_2_when_data_dir_has_no_articles(tmp_path, capsys):
    (tmp_path / "empty").mkdir()
    with fake_embed_server(lambda texts, n: (200, ok_body(texts))) as (url, _):
        code = ingest_module.main(_main_args(url, tmp_path / "empty", tmp_path))
    assert code == 2
    assert "empty corpus" in capsys.readouterr().out


def test_main_exits_2_on_a_url_without_scheme(tmp_path, capsys):
    assert ingest_module.main(_main_args("not-a-url", tmp_path, tmp_path)) == 2
    assert "update EMBED_URL" in capsys.readouterr().out


def test_store_failure_is_recorded_and_stops_the_run(tmp_path):
    client = _setup(tmp_path)

    def picky_embedder(texts):
        if any("FAILME" in text for text in texts):
            return [[1.0, 0.0, 0.0, 0.0] for _ in texts]  # wrong size: Qdrant rejects the upsert
        return fake_embedder(texts)

    report = ingest(str(tmp_path), client, picky_embedder, _word_count)

    assert report.articles_ok == 0
    assert len(report.articles_failed) == 1
    assert report.articles_failed[0].startswith("disease/benh-b")
    assert "store error" in report.articles_failed[0]


def test_main_exits_2_when_the_qdrant_path_is_locked(tmp_path, capsys):
    holder = open_client(str(tmp_path / "q"))  # keeps the local storage lock
    try:
        with fake_embed_server(lambda texts, n: (200, ok_body(texts))) as (url, _):
            code = ingest_module.main(_main_args(url, tmp_path, tmp_path))
    finally:
        holder.close()
    assert code == 2
    assert "startup failed" in capsys.readouterr().out


def _write_many(tmp_path, count):
    for n in range(count):
        _write(
            tmp_path, "drug", f"thuoc-{n}",
            _article(f"thuoc-{n}", "Giới thiệu thuốc.", [("Công dụng", "Giảm đau.")]),
        )


def test_five_consecutive_embed_failures_stop_the_run(tmp_path):
    _write_many(tmp_path, 7)
    client = open_client(None)
    ensure_collection(client)

    def dead_embedder(texts):
        raise EmbedError("tunnel down")

    report = ingest(str(tmp_path), client, dead_embedder, _word_count)

    assert report.articles_ok == 0
    assert len(report.articles_failed) == 5  # stopped after 5, the last 2 articles were never tried


def test_embed_failures_that_are_not_consecutive_do_not_stop_the_run(tmp_path):
    _write_many(tmp_path, 7)
    client = open_client(None)
    ensure_collection(client)
    calls = []

    def alternating_embedder(texts):
        calls.append(1)
        if len(calls) % 2 == 1:
            raise EmbedError("blip")
        return fake_embedder(texts)

    report = ingest(str(tmp_path), client, alternating_embedder, _word_count)

    assert report.articles_ok == 3
    assert len(report.articles_failed) == 4
