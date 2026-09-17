import re
import uuid

from pathlib import Path
from typing import Callable, Iterator

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

_HEADER_LEVELS = [("##", "H2"), ("###", "H3"), ("####", "H4"), ("#####", "H5")]
_HEADER_LINE = re.compile(r"^#{1,6}\s")
_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]
_HEADER_BUDGET_SAFETY_MARGIN = 30

CHUNK_NAMESPACE = uuid.UUID("f47b6a3e-3f0e-4b8a-9c2e-2f8e6a1d7c50")


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


def guard_rail_split(
    sections: list[dict],
    title: str,
    url: str,
    token_counter: Callable[[str], int],
    max_tokens: int = 400,
) -> list[dict]:
    result = []
    for section in sections:
        n = token_counter(section["text"])
        if n <= max_tokens:
            result.append({
                "breadcrumb": section["breadcrumb"],
                "text": section["text"],
                "token_count": n,
                "split_part": None,
            })
            continue

        sample_header = _context_header(title, f"{section['breadcrumb']} (phần 1/1)".strip(), url)
        header_budget = token_counter(sample_header)
        body_budget = max(max_tokens - header_budget - _HEADER_BUDGET_SAFETY_MARGIN, 1)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=body_budget,
            chunk_overlap=0,
            length_function=token_counter,
            separators=_SEPARATORS,
        )
        parts = splitter.split_text(section["content"])
        total = len(parts)
        for i, part in enumerate(parts, 1):
            suffix = f"(phần {i}/{total})"
            breadcrumb_display = f"{section['breadcrumb']} {suffix}".strip()
            header = _context_header(title, breadcrumb_display, url)
            text = f"{header}\n\n{part.strip()}"
            result.append({
                "breadcrumb": section["breadcrumb"],
                "text": text,
                "token_count": token_counter(text),
                "split_part": f"{i}/{total}",
            })
    return result


def chunk_article(
    text: str,
    article_type: str,
    article_slug: str,
    token_counter: Callable[[str], int],
    max_tokens: int = 400,
) -> list[dict]:
    url, title, clean_body = parse_article(text)
    sections = split_sections(clean_body)
    sections = inject_context_header(sections, title, url)
    sections = guard_rail_split(sections, title, url, token_counter, max_tokens)

    chunks = []
    for idx, section in enumerate(sections):
        chunk_id = str(uuid.uuid5(CHUNK_NAMESPACE, f"{article_type}:{article_slug}:{idx}"))
        chunks.append({
            "id": chunk_id,
            "type": article_type,
            "article_slug": article_slug,
            "article_title": title,
            "article_url": url,
            "section_path": section["breadcrumb"],
            "text": section["text"],
            "token_count": section["token_count"],
            "split_part": section["split_part"],
        })
    return chunks


def iter_articles(data_dir: str = "data") -> Iterator[tuple[str, str, str]]:
    for type_dir in sorted(p for p in Path(data_dir).iterdir() if p.is_dir()):
        for file_path in sorted(type_dir.glob("*.txt")):
            yield type_dir.name, file_path.stem, file_path.read_text(encoding="utf-8")
