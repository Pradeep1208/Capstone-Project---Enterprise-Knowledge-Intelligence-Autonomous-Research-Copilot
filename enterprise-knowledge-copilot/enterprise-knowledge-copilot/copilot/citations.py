"""Citation parsing and a heuristic verification layer.

Verification checks (1) every [n] refers to a retrieved passage and (2) each cited sentence shares
enough content words with the passages it cites. It is a lexical heuristic: it flags weak support,
it does not prove an answer is correct.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .index import tokenize
from .retrieval import Hit

CITE_RE = re.compile(r"\[(\d+)\]")


@dataclass
class VerificationReport:
    cleaned_text: str
    valid_ids: list[int] = field(default_factory=list)
    invalid_ids: list[int] = field(default_factory=list)
    sentences: list[dict] = field(default_factory=list)   # {"text", "cited", "supported"}

    @property
    def unsupported_rate(self) -> float:
        judged = [s for s in self.sentences if s["supported"] is not None]
        return sum(1 for s in judged if not s["supported"]) / len(judged) if judged else 0.0


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


def verify_answer(text: str, hits: list[Hit], min_overlap: float = 0.5) -> VerificationReport:
    n = len(hits)
    cited = [int(x) for x in CITE_RE.findall(text)]
    invalid = sorted({c for c in cited if c < 1 or c > n})
    valid = sorted({c for c in cited if 1 <= c <= n})
    cleaned = CITE_RE.sub(lambda m: m.group(0) if 1 <= int(m.group(1)) <= n else "", text)

    sentences = []
    for sent in split_sentences(cleaned):
        ids = [int(x) for x in CITE_RE.findall(sent) if 1 <= int(x) <= n]
        words = set(tokenize(CITE_RE.sub("", sent)))
        if not ids or not words:
            sentences.append({"text": sent, "cited": ids, "supported": None})
            continue
        evidence = set()
        for i in ids:
            evidence |= set(tokenize(hits[i - 1].chunk.text))
        supported = len(words & evidence) / len(words) >= min_overlap
        sentences.append({"text": sent, "cited": ids, "supported": supported})
    return VerificationReport(cleaned, valid, invalid, sentences)
