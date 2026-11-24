#!/usr/bin/env python
"""Compare vector-only / BM25 / hybrid / hybrid+rerank on a BEIR dataset (human relevance judgements).

  python scripts/download_data.py beir --name scifact
  python scripts/eval_beir.py --name scifact
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from copilot.chunking import Chunk
from copilot.config import SETTINGS
from copilot.embeddings import Embedder
from copilot.evaluation import evaluate_retrieval
from copilot.index import HybridIndex
from copilot.reranker import CrossEncoderReranker
from copilot.retrieval import MODES, Retriever


def load_beir(name: str, split: str = "test"):
    base = ROOT / "data/beir" / name
    corpus = [json.loads(l) for l in open(base / "corpus.jsonl", encoding="utf-8")]
    queries = {q["_id"]: q["text"] for q in map(json.loads, open(base / "queries.jsonl", encoding="utf-8"))}
    qrels: dict[str, dict[str, int]] = {}
    with open(base / "qrels" / f"{split}.tsv", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            qrels.setdefault(row["query-id"], {})[row["corpus-id"]] = int(row["score"])
    return corpus, {q: queries[q] for q in qrels if q in queries}, qrels


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="scifact")
    p.add_argument("--limit", type=int, default=0, help="evaluate only the first N queries (quick test)")
    p.add_argument("--modes", default=",".join(MODES))
    a = p.parse_args()

    corpus, queries, qrels = load_beir(a.name)
    if a.limit:
        queries = dict(list(queries.items())[: a.limit])
    print(f"BEIR/{a.name}: {len(corpus)} documents, {len(queries)} test queries")

    chunks = [Chunk(d["_id"], d["_id"], d.get("title", ""), f"BEIR/{a.name}", None, None,
                    (d.get("title", "") + ". " + d["text"]).strip(".  "), {}) for d in corpus]
    embedder = Embedder(SETTINGS.embedding_model)
    cache = ROOT / "data/beir" / a.name / f"embeddings_{SETTINGS.embedding_model.split('/')[-1]}.npy"
    if cache.exists():
        vectors = np.load(cache)
    else:
        print("Embedding corpus (one-off, cached) ...")
        vectors = embedder.encode([c.text for c in chunks], show_progress=True)
        np.save(cache, vectors)
    retriever = Retriever(HybridIndex(chunks, vectors, SETTINGS.embedding_model), embedder,
                          CrossEncoderReranker(SETTINGS.reranker_model), SETTINGS.rrf_k)

    views = {"doc": (lambda c: c.doc_id, qrels)}
    SETTINGS.results_dir.mkdir(exist_ok=True)
    summaries = []
    for mode in a.modes.split(","):
        print(f"Evaluating mode: {mode}")
        per_query, summary = evaluate_retrieval(retriever, queries, views, mode, k=10, pool=SETTINGS.candidate_pool)
        per_query.to_csv(SETTINGS.results_dir / f"beir_{a.name}_{mode}_per_query.csv", index=False)
        summaries.append({"suite": f"BEIR/{a.name}", "config": mode, **summary})
        print({k: v for k, v in summary.items() if "ndcg" in k or "recall@10" in k or "p50" in k})
    out = SETTINGS.results_dir / f"beir_{a.name}_summary.csv"
    pd.DataFrame(summaries).to_csv(out, index=False)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
