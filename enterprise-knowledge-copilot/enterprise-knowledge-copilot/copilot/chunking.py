"""Chunking strategies. Chunks never cross page/section boundaries so citations stay accurate."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from .ingestion import Document

MIN_CHUNK_CHARS = 30


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    source: str
    page: int | None
    section: str | None
    text: str
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def citation_label(self) -> str:
        parts = [self.title]
        if self.page is not None:
            parts.append(f"p. {self.page}")
        if self.section:
            parts.append(self.section)
        return " — ".join(parts)


# ------------------------------------------------------------------ fixed-size
def fixed_split(text: str, size: int, overlap: int) -> list[str]:
    chunks, i = [], 0
    while i < len(text):
        end = min(i + size, len(text))
        if end < len(text):
            space = text.rfind(" ", i, end)
            if space > i + size // 2:
                end = space
        piece = text[i:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        i = max(end - overlap, i + 1)
    return chunks


# ------------------------------------------------------------------ recursive
_LEVELS = [
    (re.compile(r"\n\s*\n"), "\n\n"),
    (re.compile(r"\n"), "\n"),
    (re.compile(r"(?<=[.!?])\s+"), " "),
    (re.compile(r" "), " "),
]


def _pieces(text: str, size: int, level: int = 0) -> list[tuple[str, str]]:
    """Break text into (piece, joiner) tuples where every piece is <= size characters."""
    if len(text) <= size:
        return [(text, "")]
    if level >= len(_LEVELS):
        return [(text[i : i + size], "") for i in range(0, len(text), size)]
    regex, joiner = _LEVELS[level]
    out: list[tuple[str, str]] = []
    for part in regex.split(text):
        if not part.strip():
            continue
        for piece, _ in _pieces(part, size, level + 1) if len(part) > size else [(part, "")]:
            out.append((piece, joiner))
    return out


def _tail(text: str, overlap: int) -> str:
    if overlap <= 0:
        return ""
    tail = text[-overlap:]
    return tail[tail.find(" ") + 1 :] if " " in tail else tail


def recursive_split(text: str, size: int, overlap: int) -> list[str]:
    chunks, current = [], ""
    for piece, joiner in _pieces(text, size):
        if current and len(current) + len(joiner) + len(piece) > size:
            chunks.append(current)
            tail = _tail(current, overlap)
            current = f"{tail}{joiner}{piece}" if tail else piece
        else:
            current = f"{current}{joiner}{piece}" if current else piece
    if current:
        chunks.append(current)
    return [c.strip() for c in chunks if c.strip()]


STRATEGIES = {"fixed": fixed_split, "recursive": recursive_split}


def chunk_document(doc: Document, strategy: str = "recursive", size: int = 1000, overlap: int = 150) -> list[Chunk]:
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown chunking strategy '{strategy}'. Choose from {list(STRATEGIES)}")
    if overlap >= size:
        raise ValueError("chunk overlap must be smaller than chunk size")
    split = STRATEGIES[strategy]
    chunks: list[Chunk] = []
    for page in doc.pages:
        for n, text in enumerate(split(page.text, size, overlap)):
            if len(text) < MIN_CHUNK_CHARS:
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}::p{page.page or 0}::{len(chunks)}",
                    doc_id=doc.doc_id,
                    title=doc.title,
                    source=doc.source,
                    page=page.page,
                    section=page.section,
                    text=text,
                    metadata={**doc.metadata, "chunker": strategy, "chunk_size": size, "overlap": overlap},
                )
            )
    return chunks
