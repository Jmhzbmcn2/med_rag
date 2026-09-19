from medical_rag.encoders import DENSE_DIM, embed_input, segment


def test_dense_dim_matches_the_embedding_model():
    assert DENSE_DIM == 768


def test_segment_joins_compounds_and_is_deterministic():
    assert segment("Bệnh nhân bị đau đầu.") == "Bệnh_nhân bị đau_đầu ."
    assert segment("Bệnh nhân bị đau đầu.") == segment("Bệnh nhân bị đau đầu.")
    assert segment("") == ""


def test_embed_input_is_the_chunk_text_unchanged():
    chunk = {"text": "[Chủ đề: T | Nguồn: U]\n\nThân bài."}
    assert embed_input(chunk) == "[Chủ đề: T | Nguồn: U]\n\nThân bài."
