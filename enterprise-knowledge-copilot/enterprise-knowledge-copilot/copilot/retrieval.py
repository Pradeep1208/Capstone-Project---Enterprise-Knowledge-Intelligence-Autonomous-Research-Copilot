"""Vector, BM25, hybrid (Reciprocal Rank Fusion) and hybrid + cross-encoder reranking."""
from __future__ import annotations

from dataclasses import dataclass

from .chunking import Chunk
from .index import HybridIndex

MODES = ("vector", "bm25", "hybrid", "hybrid_rerank")
MODE_LABELS = {
    "vector": "Vector only (baseline)",
    "bm25": "BM25 only",
    "hybrid": "Hybrid (BM25 + vector, RRF)",
    "hybrid_rerank": "Hybrid + cross-encoder reranker",
}


@dataclass
class Hit:
    chunk: Chunk
    score: float
    vector_rank: int | None = None
    bm25_rank: int | None = None
    rerank_score: float | None = None   # raw cross-encoder logit


def reciprocal_rank_fusion(rank_lists: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """RRF: score(d) = sum over lists of 1 / (k + rank(d)), ranks starting at 1."""
    scores: dict[int, float] = {}
    for ranking in rank_lists:
        for rank, idx in enumerate(ranking, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


class Retriever:
    def __init__(self, index: HybridIndex, embedder, reranker=None, rrf_k: int = 60):
        self.index, self.embedder, self.reranker, self.rrf_k = index, embedder, reranker, rrf_k

    def search(self, query: str, mode: str = "hybrid_rerank", top_k: int = 5, pool: int = 50) -> list[Hit]:
        if mode not in MODES:
            raise ValueError(f"Unknown retrieval mode '{mode}'")
        if not len(self.index) or not query.strip():
            return []
        pool = max(pool, top_k)

        vec, bm = [], []
        if mode in ("vector", "hybrid", "hybrid_rerank"):
            qv = self.embedder.encode([query])[0]
            vec = self.index.vector_search(qv, pool)
        if mode in ("bm25", "hybrid", "hybrid_rerank"):
            bm = self.index.bm25_search(query, pool)
        vrank = {i: r for r, (i, _) in enumerate(vec, 1)}
        brank = {i: r for r, (i, _) in enumerate(bm, 1)}

        if mode == "vector":
            ranked = vec
        elif mode == "bm25":
            ranked = bm
        else:
            ranked = reciprocal_rank_fusion([[i for i, _ in vec], [i for i, _ in bm]], self.rrf_k)

        hits = [Hit(self.index.chunks[i], s, vrank.get(i), brank.get(i)) for i, s in ranked[:pool]]

        if mode == "hybrid_rerank":
            if self.reranker is None:
                raise RuntimeError("hybrid_rerank requires a reranker")
            scores = self.reranker.score(query, [h.chunk.text for h in hits])
            for h, s in zip(hits, scores):
                h.rerank_score = float(s)
                h.score = float(s)
            hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]
