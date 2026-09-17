import re

from langchain_text_splitters import MarkdownHeaderTextSplitter

_HEADER_LEVELS = [("##", "H2"), ("###", "H3"), ("####", "H4"), ("#####", "H5")]
_HEADER_LINE = re.compile(r"^#{1,6}\s")


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


def _is_empty_section(content: str) -> bool:
    body = "\n".join(
        line for line in content.split("\n")
        if not _HEADER_LINE.match(line.strip())
    ).strip()
    return not body


def split_sections(clean_body: str) -> list[dict]:
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=_HEADER_LEVELS,
        strip_headers=False,
    )
    docs = splitter.split_text(clean_body)

    sections = []
    for doc in docs:
        if _is_empty_section(doc.page_content):
            continue
        breadcrumb = " > ".join(
            doc.metadata[key] for _, key in _HEADER_LEVELS if key in doc.metadata
        )
        sections.append({"breadcrumb": breadcrumb, "content": doc.page_content.strip()})
    return sections


def _context_header(title: str, breadcrumb: str, url: str) -> str:
    parts = [f"Chủ đề: {title}"]
    if breadcrumb:
        parts.append(f"Mục: {breadcrumb}")
    parts.append(f"Nguồn: {url}")
    return "[" + " | ".join(parts) + "]"


def inject_context_header(sections: list[dict], title: str, url: str) -> list[dict]:
    result = []
    for section in sections:
        header = _context_header(title, section["breadcrumb"], url)
        result.append({**section, "text": f"{header}\n\n{section['content']}"})
    return result
