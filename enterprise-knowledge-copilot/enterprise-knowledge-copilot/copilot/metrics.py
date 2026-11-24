"""Information-retrieval metrics. `ranked` is a list of unique relevance keys; `qrels` maps key -> grade."""
from __future__ import annotations

import math
import re

import numpy as np


def recall_at_k(ranked: list[str], qrels: dict[str, int], k: int) -> float:
    relevant = {key for key, g in qrels.items() if g > 0}
    return len(relevant & set(ranked[:k])) / len(relevant) if relevant else 0.0


def precision_at_k(ranked: list[str], qrels: dict[str, int], k: int) -> float:
    relevant = {key for key, g in qrels.items() if g > 0}
    return len([r for r in ranked[:k] if r in relevant]) / k if k else 0.0


def mrr_at_k(ranked: list[str], qrels: dict[str, int], k: int) -> float:
    for rank, key in enumerate(ranked[:k], start=1):
        if qrels.get(key, 0) > 0:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked: list[str], qrels: dict[str, int], k: int) -> float:
    dcg = sum((2 ** qrels.get(key, 0) - 1) / math.log2(i + 2) for i, key in enumerate(ranked[:k]))
    ideal = sorted(qrels.values(), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else 0.0


# ----------------------------------------------------------------------------- answer-level helpers
_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def extract_numbers(text: str) -> list[float]:
    out = []
    for m in _NUM_RE.findall(text.replace("$", " ")):
        try:
            out.append(float(m.replace(",", "")))
        except ValueError:
            pass
    return out


def numbers_match(gold: str, pred: str, rel_tol: float = 0.01) -> bool | None:
    """True/False if the gold answer has numbers; None if it has none (use another metric)."""
    g = extract_numbers(gold)
    if not g:
        return None
    p = extract_numbers(pred)
    return all(any(abs(x - y) <= rel_tol * max(abs(x), 1e-9) or x == y for y in p) for x in g)


def token_f1(gold: str, pred: str) -> float:
    gt, pt = re.findall(r"\w+", gold.lower()), re.findall(r"\w+", pred.lower())
    if not gt or not pt:
        return 0.0
    common = {}
    for t in set(gt) & set(pt):
        common[t] = min(gt.count(t), pt.count(t))
    overlap = sum(common.values())
    if not overlap:
        return 0.0
    p, r = overlap / len(pt), overlap / len(gt)
    return 2 * p * r / (p + r)
