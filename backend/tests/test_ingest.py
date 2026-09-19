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
