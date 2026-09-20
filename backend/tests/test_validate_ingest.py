import pandas as pd
from qdrant_client import models

from medical_rag.encoders import DENSE_DIM
from medical_rag.ingestion.ingest import ingest
from medical_rag.store import COLLECTION, ensure_collection, open_client
from fake_server import fake_embed_server, ok_body
from validate_ingest import check_invariants, load_expected, max_segmented_tokens, smoke_retrieval
from validate_ingest import main as validate_main


def _word_count(text: str) -> int:
    return len(text.split())


def fake_embedder(texts):
    return [[1.0 + len(text) % 5] + [0.0] * (DENSE_DIM - 1) for text in texts]


def _write_articles(root) -> None:
    for article_type, slug, intro, sections in (
        ("drug", "thuoc-a", "Giới thiệu thuốc A.", [("Công dụng", "Giảm đau."), ("Cách dùng", "Uống 1 viên.")]),
        ("disease", "benh-b", "Giới thiệu bệnh B.", [("Triệu chứng", "Sốt và ho.")]),
    ):
        folder = root / article_type
        folder.mkdir(parents=True)
        body = "".join(f"\n## {heading}\n\n{text}\n" for heading, text in sections)
        (folder / f"{slug}.txt").write_text(
            f"# SOURCE_URL: https://youmed.vn/tin-tuc/{slug}/\n\n# Tiêu đề {slug}\n\n"
            f"Nội dung bài viết\n\n**{intro}**\n{body}",
            encoding="utf-8",
        )


def _ingested(tmp_path):
    _write_articles(tmp_path)
    client = open_client(None)
    ensure_collection(client)
    ingest(str(tmp_path), client, fake_embedder, _word_count)
    return client, load_expected(str(tmp_path), _word_count)


def test_check_invariants_passes_on_a_clean_ingest(tmp_path):
    client, expected = _ingested(tmp_path)
    assert len(expected) == 5
    assert check_invariants(client, expected) == []


def test_check_invariants_reports_a_missing_point_and_a_missing_sparse_vector(tmp_path):
    client, expected = _ingested(tmp_path)
    victim = expected[0]["id"]
    client.delete(COLLECTION, points_selector=models.PointIdsList(points=[victim]))
    problems = check_invariants(client, expected)
    assert any("point count 4 != expected chunk count 5" in p for p in problems)
    assert any(f"missing point for chunk {victim}" in p for p in problems)

    client.upsert(
        COLLECTION,
        points=[
            models.PointStruct(
                id=victim,
                vector={"dense": [1.0] * DENSE_DIM},
                payload={"type": expected[0]["type"], "article_slug": expected[0]["article_slug"]},
            )
        ],
    )
    problems = check_invariants(client, expected)
    assert any("sparse vector presence" in p for p in problems)


def test_max_segmented_tokens_uses_the_segmented_text(tmp_path):
    _, expected = _ingested(tmp_path)
    assert max_segmented_tokens(expected, _word_count) == max(_word_count(c["segmented"]) for c in expected)


def test_smoke_retrieval_finds_the_right_article_with_sparse_search(tmp_path):
    client, _ = _ingested(tmp_path)
    case_dir = tmp_path / "cases"
    case_dir.mkdir()
    pd.DataFrame(
        [{
            "question": "Thuốc nào giảm đau?",
            "context": "Giảm đau.",
            "article_url": "https://youmed.vn/tin-tuc/thuoc-a/",
        }]
    ).to_csv(case_dir / "drug.csv", index=False, encoding="utf-8")

    results = smoke_retrieval(client, fake_embedder, str(case_dir), per_type=30, seed=42, k=1)

    assert set(results) == {"drug"}
    assert results["drug"]["sparse"] == {"article": 1.0, "chunk": 1.0}
    for levels in results["drug"].values():
        assert all(0.0 <= value <= 1.0 for value in levels.values())


def test_smoke_retrieval_skips_an_empty_csv(tmp_path):
    client, _ = _ingested(tmp_path)
    case_dir = tmp_path / "cases"
    case_dir.mkdir()
    pd.DataFrame(columns=["question", "context", "article_url"]).to_csv(
        case_dir / "drug.csv", index=False, encoding="utf-8"
    )

    assert smoke_retrieval(client, fake_embedder, str(case_dir)) == {}


def test_main_exits_2_when_embed_url_is_missing(monkeypatch, capsys):
    monkeypatch.delenv("EMBED_URL", raising=False)
    assert validate_main(["--data-dir", "nowhere"]) == 2
    assert "EMBED_URL" in capsys.readouterr().out


def test_main_exits_2_when_the_tunnel_is_dead(capsys):
    assert validate_main(["--embed-url", "http://127.0.0.1:1"]) == 2
    assert "update EMBED_URL" in capsys.readouterr().out


def test_main_exits_2_when_the_collection_does_not_exist(tmp_path, capsys):
    (tmp_path / "empty").mkdir()
    with fake_embed_server(lambda texts, n: (200, ok_body(texts))) as (url, _):
        code = validate_main([
            "--embed-url", url,
            "--data-dir", str(tmp_path / "empty"),
            "--qdrant-path", str(tmp_path / "q"),
        ])
    assert code == 2
    out = capsys.readouterr().out
    assert "validate aborted" in out
    assert "medical_rag not found" in out  # names the missing collection, so a tokenizer failure cannot satisfy this
