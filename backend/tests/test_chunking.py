from medical_rag.ingestion.chunking import parse_article, split_sections, inject_context_header, guard_rail_split, chunk_article


def test_parse_article_strips_metadata_lines():
    text = (
        "# SOURCE_URL: https://youmed.vn/tin-tuc/thuoc-x/\n"
        "\n"
        "# Thuốc X là gì?\n"
        "\n"
        "Nội dung bài viết\n"
        "\n"
        "**Đoạn giới thiệu về thuốc X.**\n"
        "\n"
        "## Công dụng\n"
        "\n"
        "Thuốc X dùng để giảm đau.\n"
    )
    url, title, clean_body = parse_article(text)
    assert url == "https://youmed.vn/tin-tuc/thuoc-x/"
    assert title == "Thuốc X là gì?"
    assert clean_body == (
        "**Đoạn giới thiệu về thuốc X.**\n"
        "\n"
        "## Công dụng\n"
        "\n"
        "Thuốc X dùng để giảm đau."
    )


def test_split_sections_breadcrumb_and_empty_drop():
    body = (
        "**Đoạn giới thiệu ngắn về thuốc X.**\n"
        "\n"
        "## Công dụng\n"
        "\n"
        "Thuốc X dùng để giảm đau.\n"
        "\n"
        "## Cách dùng\n"
        "\n"
        "### Liều dùng\n"
        "\n"
        "Uống 1 viên mỗi ngày.\n"
        "\n"
        "### Lưu ý rỗng\n"
        "\n"
        "## Tác dụng phụ\n"
        "\n"
        "Có thể gây buồn ngủ.\n"
    )
    sections = split_sections(body)

    assert [s["breadcrumb"] for s in sections] == [
        "",
        "Công dụng",
        "Cách dùng > Liều dùng",
        "Tác dụng phụ",
    ]
    assert sections[0]["content"] == "**Đoạn giới thiệu ngắn về thuốc X.**"
    assert "Thuốc X dùng để giảm đau." in sections[1]["content"]
    assert sections[1]["content"].startswith("## Công dụng")
    assert "Uống 1 viên mỗi ngày." in sections[2]["content"]
    assert "Có thể gây buồn ngủ." in sections[3]["content"]


def test_inject_context_header_with_and_without_breadcrumb():
    sections = [
        {"breadcrumb": "", "content": "Đoạn intro."},
        {"breadcrumb": "Công dụng", "content": "## Công dụng\nThuốc X dùng để giảm đau."},
    ]
    result = inject_context_header(
        sections, title="Thuốc X là gì?", url="https://youmed.vn/tin-tuc/thuoc-x/"
    )

    assert result[0]["text"] == (
        "[Chủ đề: Thuốc X là gì? | Nguồn: https://youmed.vn/tin-tuc/thuoc-x/]\n\n"
        "Đoạn intro."
    )
    assert result[1]["text"] == (
        "[Chủ đề: Thuốc X là gì? | Mục: Công dụng | Nguồn: https://youmed.vn/tin-tuc/thuoc-x/]\n\n"
        "## Công dụng\nThuốc X dùng để giảm đau."
    )
    assert result[1]["breadcrumb"] == "Công dụng"
    assert result[1]["content"] == "## Công dụng\nThuốc X dùng để giảm đau."


def _word_count(s: str) -> int:
    return len(s.split())


def test_guard_rail_split_passes_short_and_splits_long():
    # max_tokens=70 with the header (~9 words) and safety margin (30 words)
    # reserved leaves a body budget of ~31 words — big enough that each of
    # the two paragraphs below fits in one piece, so the split below is
    # driven by the _SEPARATORS paragraph boundary ("\n\n"), not by a
    # degenerate near-1-word clamp.
    short_text = "[Chủ đề: T | Mục: A | Nguồn: U]\n\n## A\nMột hai ba."
    paragraph_1 = (
        "Đoạn một có nhiều từ để vượt qua ngưỡng token được đặt ra cho bài "
        "kiểm tra này nhằm đảm bảo việc phân chia đoạn văn xảy ra đúng"
    )
    paragraph_2 = (
        "Đoạn hai cũng dài tương tự để đảm bảo việc cắt chia hoạt động đúng "
        "theo ranh giới đoạn văn và không bị cắt giữa câu khi kiểm thử guard rail"
    )
    long_content = f"## B\n\n{paragraph_1}\n\n{paragraph_2}"
    long_text = f"[Chủ đề: T | Mục: B | Nguồn: U]\n\n{long_content}"

    sections = [
        {"breadcrumb": "A", "content": "## A\nMột hai ba.", "text": short_text},
        {"breadcrumb": "B", "content": long_content, "text": long_text},
    ]

    result = guard_rail_split(
        sections, title="T", url="U", token_counter=_word_count, max_tokens=70
    )

    # section A fits under the cap -> passes through untouched
    a_results = [r for r in result if r["breadcrumb"] == "A"]
    assert len(a_results) == 1
    assert a_results[0]["text"] == short_text
    assert a_results[0]["split_part"] is None

    # section B is over the cap -> splits into exactly 2 parts, right at the
    # "\n\n" paragraph boundary between paragraph_1 and paragraph_2 (not a
    # further sentence/word split within either paragraph)
    b_results = [r for r in result if r["breadcrumb"] == "B"]
    assert len(b_results) == 2
    assert b_results[0]["split_part"] == "1/2"
    assert b_results[1]["split_part"] == "2/2"
    assert b_results[0]["text"] == (
        "[Chủ đề: T | Mục: B (phần 1/2) | Nguồn: U]\n\n"
        f"## B\n\n{paragraph_1}"
    )
    assert b_results[1]["text"] == (
        "[Chủ đề: T | Mục: B (phần 2/2) | Nguồn: U]\n\n"
        f"{paragraph_2}"
    )
    for r in b_results:
        assert r["breadcrumb"] == "B"
        assert _word_count(r["text"]) <= 70
        assert r["token_count"] == _word_count(r["text"])


def test_chunk_article_end_to_end_and_deterministic_ids():
    text = (
        "# SOURCE_URL: https://youmed.vn/tin-tuc/thuoc-x/\n"
        "\n"
        "# Thuốc X là gì?\n"
        "\n"
        "Nội dung bài viết\n"
        "\n"
        "**Đoạn giới thiệu về thuốc X.**\n"
        "\n"
        "## Công dụng\n"
        "\n"
        "Thuốc X dùng để giảm đau.\n"
        "\n"
        "## Cách dùng\n"
        "\n"
        "Uống 1 viên mỗi ngày.\n"
    )

    chunks = chunk_article(
        text, article_type="drug", article_slug="thuoc-x",
        token_counter=_word_count, max_tokens=1000,
    )

    assert len(chunks) == 3  # intro, Cong dung, Cach dung
    for chunk in chunks:
        assert chunk["type"] == "drug"
        assert chunk["article_slug"] == "thuoc-x"
        assert chunk["article_title"] == "Thuốc X là gì?"
        assert chunk["article_url"] == "https://youmed.vn/tin-tuc/thuoc-x/"
        assert chunk["split_part"] is None

    assert chunks[1]["section_path"] == "Công dụng"
    assert chunks[2]["section_path"] == "Cách dùng"

    # ids are unique and deterministic across repeated runs
    ids = [c["id"] for c in chunks]
    assert len(set(ids)) == len(ids)
    chunks_again = chunk_article(
        text, article_type="drug", article_slug="thuoc-x",
        token_counter=_word_count, max_tokens=1000,
    )
    assert [c["id"] for c in chunks_again] == ids
