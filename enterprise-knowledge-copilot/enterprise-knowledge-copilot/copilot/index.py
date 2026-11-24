"""Hybrid index = dense vectors (exact cosine search, NumPy) + sparse BM25.

Exact NumPy search is deliberately used instead of a vector database: it has no native
dependencies, is deterministic for evaluation, and is fast for up to ~100k chunks.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from .chunking import Chunk

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")
_STOP = frozenset(
    "a an the of in on at to for and or is are was were be been by with as that this it its from what which who how "
    "when where why do does did has have had not no".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP]


class HybridIndex:
    def __init__(self, chunks: list[Chunk], embeddings: np.ndarray, embedding_model: str):
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        self.chunks = chunks
        self.embeddings = embeddings.astype("float32")
        self.embedding_model = embedding_model
        self._tokens = [tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(self._tokens) if chunks else None

    def __len__(self) -> int:
        return len(self.chunks)

    # ------------------------------------------------------------------ search
    def vector_search(self, query_vec: np.ndarray, k: int) -> list[tuple[int, float]]:
        if not len(self):
            return []
        sims = self.embeddings @ query_vec
        return self._top(sims, k)

    def bm25_search(self, query: str, k: int) -> list[tuple[int, float]]:
        tokens = tokenize(query)
        if not tokens or self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokens)
        return [(i, s) for i, s in self._top(scores, k) if s > 0]

    @staticmethod
    def _top(scores: np.ndarray, k: int) -> list[tuple[int, float]]:
        k = min(k, len(scores))
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [(int(i), float(scores[i])) for i in idx]

    # ------------------------------------------------------------------ mutation
    def with_added(self, chunks: list[Chunk], embeddings: np.ndarray) -> "HybridIndex":
        """Return a NEW index containing existing + new chunks (the original is untouched)."""
        if not chunks:
            return self
        merged = np.vstack([self.embeddings, embeddings]) if len(self) else embeddings
        return HybridIndex(self.chunks + chunks, merged, self.embedding_model)

    # ------------------------------------------------------------------ persistence
    def save(self, directory: str | Path, extra_meta: dict | None = None) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "chunks.jsonl", "w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
        np.save(d / "embeddings.npy", self.embeddings)
        meta = {"embedding_model": self.embedding_model, "n_chunks": len(self), **(extra_meta or {})}
        (d / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path) -> "HybridIndex":
        d = Path(directory)
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        with open(d / "chunks.jsonl", encoding="utf-8") as f:
            chunks = [Chunk(**json.loads(line)) for line in f if line.strip()]
        embeddings = np.load(d / "embeddings.npy")
        return cls(chunks, embeddings, meta["embedding_model"])

    @classmethod
    def exists(cls, directory: str | Path) -> bool:
        d = Path(directory)
        return all((d / n).exists() for n in ("chunks.jsonl", "embeddings.npy", "meta.json"))

    @classmethod
    def build(cls, chunks: list[Chunk], embedder, batch_size: int = 32, show_progress: bool = False) -> "HybridIndex":
        vectors = embedder.encode([c.text for c in chunks], batch_size=batch_size, show_progress=show_progress)
        return cls(chunks, vectors, embedder.model_name)

    def doc_summary(self) -> list[dict]:
        out: dict[str, dict] = {}
        for c in self.chunks:
            row = out.setdefault(c.doc_id, {"doc_id": c.doc_id, "title": c.title, "source": c.source, "chunks": 0})
            row["chunks"] += 1
        return list(out.values())
