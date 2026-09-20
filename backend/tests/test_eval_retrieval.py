import pandas as pd
from eval_retrieval import MODES, evaluate
from eval_retrieval import main as eval_main
from test_validate_ingest import _ingested, fake_embedder


def embed(texts, task="document"):
    return fake_embedder(texts)


def _cases(tmp_path):
    case_dir = tmp_path / "cases"
    case_dir.mkdir()
    pd.DataFrame(
        [{"question": "Thuốc nào giảm đau?", "context": "Giảm đau.", "article_url": "https://youmed.vn/tin-tuc/thuoc-a/"}]
    ).to_csv(case_dir / "drug.csv", index=False, encoding="utf-8")
    pd.DataFrame(columns=["question", "context", "article_url"]).to_csv(
        case_dir / "disease.csv", index=False, encoding="utf-8"
    )
    return str(case_dir)


def test_evaluate_reports_every_mode_per_type_and_overall(tmp_path):
    client, _ = _ingested(tmp_path)
    results = evaluate(client, embed, _cases(tmp_path), per_type=30, seed=42, k=5)

    assert set(results) == {"drug", "all"}  # the empty disease.csv is skipped
    for scores in results.values():
        assert set(scores) == set(MODES)
        for row in scores.values():
            assert row["n"] == 1
            assert {"article@5", "chunk@1", "chunk@5", "recall@5", "precision@5", "mrr"} <= set(row)
            assert all(0.0 <= row[key] <= 1.0 for key in row if key != "n")
    assert results["drug"]["sparse"]["chunk@1"] == 1.0  # BM25 alone finds "Giảm đau."
    assert results["drug"]["hybrid"]["chunk@5"] == 1.0  # 5 chunks in the store, k=5
    assert results["all"] == results["drug"]


def test_main_exits_2_when_embed_url_is_missing(monkeypatch):
    monkeypatch.delenv("EMBED_URL", raising=False)
    assert eval_main(["--qdrant-path", "nowhere"]) == 2


def test_main_exits_2_and_creates_nothing_for_a_missing_store(tmp_path, capsys):
    missing = tmp_path / "missing"
    assert eval_main(["--embed-url", "http://embed.test", "--qdrant-path", str(missing)]) == 2
    assert "does not exist" in capsys.readouterr().out
    assert not missing.exists()
