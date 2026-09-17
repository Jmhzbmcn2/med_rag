from chunking import parse_article


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
