"""Evaluation helpers shared by the BEIR and FinanceBench scripts."""
from __future__ import annotations

import time
from typing import Callable

import pandas as pd

from .metrics import mrr_at_k, ndcg_at_k, percentile, precision_at_k, recall_at_k
from .retrieval import Retriever

View = tuple[Callable, dict]   # (chunk -> relevance key, {query_id: {key: grade}})


def _dedupe(keys: list[str], k: int) -> list[str]:
    seen, out = set(), []
    for key in keys:
        if key not in seen:
            seen.add(key)
            out.append(key)
        if len(out) == k:
            break
    return out


def evaluate_retrieval(
    retriever: Retriever,
    queries: dict[str, str],
    views: dict[str, View],
    mode: str,
    k: int = 10,
    pool: int = 50,
) -> tuple[pd.DataFrame, dict]:
    """Run every query once, score each 'view' (e.g. page-level and document-level relevance)."""
    rows, latencies = [], []
    for qid, text in queries.items():
        t0 = time.perf_counter()
        hits = retriever.search(text, mode, top_k=k * 4, pool=max(pool, k * 4))
        latencies.append((time.perf_counter() - t0) * 1000)
        row = {"query_id": qid, "latency_ms": latencies[-1]}
        for name, (key_fn, qrels) in views.items():
            ranked = _dedupe([key_fn(h.chunk) for h in hits], k)
            rel = qrels.get(qid, {})
            row[f"{name}_recall@5"] = recall_at_k(ranked, rel, 5)
            row[f"{name}_recall@{k}"] = recall_at_k(ranked, rel, k)
            row[f"{name}_precision@5"] = precision_at_k(ranked, rel, 5)
            row[f"{name}_mrr@{k}"] = mrr_at_k(ranked, rel, k)
            row[f"{name}_ndcg@{k}"] = ndcg_at_k(ranked, rel, k)
        rows.append(row)
    df = pd.DataFrame(rows)
    summary = {c: round(float(df[c].mean()), 4) for c in df.columns if c not in ("query_id", "latency_ms")}
    summary.update(n_queries=len(df), p50_ms=round(percentile(latencies, 50), 1), p95_ms=round(percentile(latencies, 95), 1))
    return df, summary


JUDGE_PROMPT = """You are a strict grader. Reply with JSON only: {"correct": true|false}.
Mark correct=true only if the CANDIDATE answer states the same facts/numbers as the GOLD answer (wording may differ,
rounding within 1% is fine). QUESTION: %s
GOLD: %s
CANDIDATE: %s"""

FAITH_PROMPT = """You are a strict grader. Reply with JSON only: {"faithful": true|false}.
faithful=true only if every factual claim in ANSWER is supported by CONTEXT.
CONTEXT:
%s

ANSWER: %s"""


def judge_json(llm, prompt: str, key: str) -> bool | None:
    import json
    import re

    try:
        raw = llm.chat([{"role": "user", "content": prompt}], max_tokens=30).text
        return bool(json.loads(re.search(r"\{.*\}", raw, re.S).group(0))[key])
    except Exception:
        return None
