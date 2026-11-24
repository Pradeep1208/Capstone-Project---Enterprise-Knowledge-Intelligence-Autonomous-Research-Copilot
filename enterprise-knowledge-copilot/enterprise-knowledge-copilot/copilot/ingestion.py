"""Document loaders: PDF, DOCX, TXT, Markdown, HTML, CSV, JSON -> common Document structure."""
from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".markdown", ".html", ".htm", ".csv", ".json"}

# "Item 1A. Risk Factors" style headings used by SEC 10-K filings (also useful for many reports)
ITEM_RE = re.compile(r"^\s*(item\s+\d+[a-c]?\.?\s+[A-Za-z][^\n]{2,90})$", re.IGNORECASE | re.MULTILINE)
MD_HEADING_RE = re.compile(r"^(#{1,3})\s+(.+?)\s*$", re.MULTILINE)


@dataclass
class Page:
    text: str
    page: int | None = None        # 1-based page number when the format has pages
    section: str | None = None


@dataclass
class Document:
    doc_id: str
    title: str
    source: str
    pages: list[Page]
    metadata: dict = field(default_factory=dict)

    @property
    def n_chars(self) -> int:
        return sum(len(p.text) for p in self.pages)


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t\r\f\v\u00a0]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _detect_section(text: str, current: str | None) -> str | None:
    matches = ITEM_RE.findall(text)
    return re.sub(r"\s+", " ", matches[-1]).strip()[:100] if matches else current


# ----------------------------------------------------------------------------- loaders
def _load_pdf(data: bytes) -> list[Page]:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    if reader.is_encrypted:
        if not reader.decrypt(""):
            raise ValueError("Encrypted PDF cannot be read.")
    pages, section = [], None
    for number, page in enumerate(reader.pages, start=1):
        try:
            raw = page.extract_text() or ""
        except Exception:  # malformed page: skip it rather than failing the whole file
            raw = ""
        text = clean_text(raw)
        if not text:
            continue
        section = _detect_section(text, section)
        pages.append(Page(text=text, page=number, section=section))
    return pages


def _load_docx(data: bytes) -> list[Page]:
    from docx import Document as DocxDocument

    doc = DocxDocument(io.BytesIO(data))
    pages, buf, section = [], [], None

    def flush():
        text = clean_text("\n".join(buf))
        if text:
            pages.append(Page(text=text, section=section))
        buf.clear()

    for para in doc.paragraphs:
        if para.style is not None and para.style.name.lower().startswith("heading") and para.text.strip():
            flush()
            section = para.text.strip()[:100]
        elif para.text.strip():
            buf.append(para.text)
    for table in doc.tables:
        for row in table.rows:
            buf.append(" | ".join(c.text.strip() for c in row.cells))
    flush()
    return pages


def _split_by_regex_headings(text: str, regex: re.Pattern, group: int = 1) -> list[Page]:
    marks = [(m.start(), m.group(group).strip()) for m in regex.finditer(text)]
    if not marks:
        return [Page(text=clean_text(text))] if text.strip() else []
    pages = []
    head = clean_text(text[: marks[0][0]])
    if head:
        pages.append(Page(text=head))
    for i, (start, title) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        body = clean_text(text[start:end])
        if body:
            pages.append(Page(text=body, section=re.sub(r"\s+", " ", title).strip()[:100]))
    return pages


def _load_markdown(data: bytes) -> list[Page]:
    return _split_by_regex_headings(data.decode("utf-8", errors="replace"), MD_HEADING_RE, group=2)


def _load_text(data: bytes) -> list[Page]:
    text = clean_text(data.decode("utf-8", errors="replace"))
    return [Page(text=text)] if text else []


def _load_html(data: bytes) -> list[Page]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(data, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    for tag in soup.find_all(lambda t: t.name and t.name.lower() == "ix:header"):  # hidden XBRL block
        tag.decompose()
    text = soup.get_text("\n")
    return _split_by_regex_headings(clean_text(text), ITEM_RE, group=1)


def _rows_to_pages(rows: list[str], per_page: int = 20) -> list[Page]:
    pages = []
    for start in range(0, len(rows), per_page):
        block = rows[start : start + per_page]
        pages.append(Page(text="\n".join(block), section=f"rows {start + 1}-{start + len(block)}"))
    return pages


def _load_csv(data: bytes) -> list[Page]:
    import pandas as pd

    df = pd.read_csv(io.BytesIO(data), nrows=5000, dtype=str).fillna("")
    rows = ["; ".join(f"{c}: {v}" for c, v in row.items() if v) for row in df.to_dict("records")]
    return _rows_to_pages(rows)


def _load_json(data: bytes) -> list[Page]:
    obj = json.loads(data.decode("utf-8", errors="replace"))
    if isinstance(obj, list) and obj and all(isinstance(x, dict) for x in obj):
        rows = ["; ".join(f"{k}: {v}" for k, v in x.items()) for x in obj[:5000]]
        return _rows_to_pages(rows)
    return _load_text(json.dumps(obj, indent=2, ensure_ascii=False)[:500_000].encode("utf-8"))


LOADERS = {
    ".pdf": _load_pdf, ".docx": _load_docx, ".txt": _load_text, ".md": _load_markdown,
    ".markdown": _load_markdown, ".html": _load_html, ".htm": _load_html,
    ".csv": _load_csv, ".json": _load_json,
}


# ----------------------------------------------------------------------------- public API
def load_document(
    data: bytes,
    filename: str,
    doc_id: str | None = None,
    source: str | None = None,
    max_mb: float = 50,
    **metadata,
) -> Document:
    """Parse raw file bytes into a Document. Raises ValueError for unsupported/oversized/empty files."""
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type '{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
    if len(data) > max_mb * 1024 * 1024:
        raise ValueError(f"{filename} is larger than {max_mb} MB.")
    try:
        pages = LOADERS[ext](data)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Could not parse {filename}: {type(exc).__name__}") from exc
    if not pages:
        raise ValueError(f"No extractable text found in {filename} (scanned PDFs need OCR).")
    stem = Path(filename).stem
    return Document(
        doc_id=doc_id or stem,
        title=metadata.pop("title", None) or stem.replace("_", " "),
        source=source or filename,
        pages=pages,
        metadata=metadata,
    )


def load_path(path: str | Path, **kwargs) -> Document:
    path = Path(path)
    return load_document(path.read_bytes(), path.name, **kwargs)
