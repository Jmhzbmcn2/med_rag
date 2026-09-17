import re


def parse_article(text: str) -> tuple[str, str, str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    url_match = re.search(r"^# SOURCE_URL:\s*(.*)$", text, re.M)
    url = url_match.group(1).strip() if url_match else ""
    after_url = text[url_match.end():] if url_match else text

    title_match = re.search(r"^#\s+(.*)$", after_url, re.M)
    title = title_match.group(1).strip() if title_match else ""
    after_title = after_url[title_match.end():] if title_match else after_url

    after_title = re.sub(
        r"^\s*Nội dung bài viết\s*\n",
        "",
        after_title.lstrip("\n"),
        count=1,
    )
    return url, title, after_title.strip()
