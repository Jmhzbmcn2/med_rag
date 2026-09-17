from chunking import parse_article, split_sections, inject_context_header


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
