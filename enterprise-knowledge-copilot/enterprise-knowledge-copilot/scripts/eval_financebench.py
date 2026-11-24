#!/usr/bin/env python
"""Evaluate retrieval (and optionally generation) on FinanceBench: 150 expert-written questions over real SEC filings.

  python scripts/eval_financebench.py --index-dir index/demo
  python scripts/eval_financebench.py --index-dir index/demo --generate --limit 40 --judge   # needs LLM_* env vars

Page numbers: FinanceBench stores 0-based page indices; this repo stores 1-based pages, so expected_page = evidence_page_num + 1.
Questions whose filing is NOT in your index are used as a proxy for 'unanswerable' (the corpus lacks that filing); this is
a proxy because a different filing of the same company may occasionally contain the figure.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from copilot.config import SETTINGS
from copilot.embeddings import Embedder
from copilot.evaluation import FAITH_PROMPT, JUDGE_PROMPT, evaluate_retrieval, judge_json
from copilot.index import HybridIndex
from copilot.llm import LLMClient
from copilot.metrics import numbers_match, percentile, token_f1
from copilot.rag import RAGPipeline
from copilot.reranker import CrossEncoderReranker
from copilot.retrieval import MODES, Retriever


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--index-dir", default=str(SETTINGS.index_dir))
    p.add_argument("--tag", default="demo")
    p.add_argument("--modes", default=",".join(MODES))
    p.add_argument("--generate", action="store_true")
    p.add_argument("--gen-mode", default="hybrid_rerank", choices=MODES)
    p.add_argument("--limit", type=int, default=40, help="max answerable questions for generation eval")
    p.add_argument("--ooc", type=int, default=15, help="out-of-corpus questions for abstention check")
    p.add_argument("--judge", action="store_true", help="LLM-as-judge for correctness and faithfulness")
    a = p.parse_args()

    questions = [json.loads(l) for l in open(ROOT / "data/eval/financebench_questions.jsonl", encoding="utf-8")]
    index = HybridIndex.load(a.index_dir)
    indexed_docs = {c.doc_id for c in index.chunks}
    answerable = [q for q in questions if q["doc_name"] in indexed_docs]
    ooc = [q for q in questions if q["doc_name"] not in indexed_docs]
    print(f"{len(index)} chunks | {len(indexed_docs)} filings | {len(answerable)} answerable, {len(ooc)} out-of-corpus questions")

    embedder = Embedder(index.embedding_model)
    retriever = Retriever(index, embedder, CrossEncoderReranker(SETTINGS.reranker_model), SETTINGS.rrf_k)

    # ---------------------------------------------------------------- retrieval evaluation
    qtext = {q["financebench_id"]: q["question"] for q in answerable}
    doc_q = {q["financebench_id"]: {q["doc_name"]: 1} for q in answerable}
    page_q = {q["financebench_id"]: {f'{e["doc_name"]}::{int(e["evidence_page_num"]) + 1}': 1 for e in q["evidence"]}
              for q in answerable}
    views = {"page": (lambda c: f"{c.doc_id}::{c.page}", page_q), "doc": (lambda c: c.doc_id, doc_q)}
    SETTINGS.results_dir.mkdir(exist_ok=True)
    summaries = []
    for mode in a.modes.split(","):
        print(f"Retrieval eval: {mode}")
        per_query, summary = evaluate_retrieval(retriever, qtext, views, mode, k=10, pool=SETTINGS.candidate_pool)
        per_query.to_csv(SETTINGS.results_dir / f"financebench_{a.tag}_{mode}_per_query.csv", index=False)
        summaries.append({"suite": "FinanceBench", "config": mode, **summary})
        print({k: v for k, v in summary.items() if k.startswith("page_") and ("recall@10" in k or "ndcg" in k)})
    pd.DataFrame(summaries).to_csv(SETTINGS.results_dir / f"financebench_{a.tag}_retrieval_summary.csv", index=False)

    # ---------------------------------------------------------------- generation evaluation
    if not a.generate:
        return
    llm = LLMClient.from_secrets()
    if not llm.available:
        sys.exit("Set LLM_API_KEY, LLM_BASE_URL and LLM_MODEL to run generation evaluation.")
    pipe = RAGPipeline(retriever, llm, SETTINGS)
    rows = []
    for kind, qs in (("answerable", answerable[: a.limit]), ("out_of_corpus", ooc[: a.ooc])):
        for i, q in enumerate(qs, 1):
            ans = pipe.answer(q["question"], mode=a.gen_mode, top_k=SETTINGS.top_k, corrective=True)
            row = {"id": q["financebench_id"], "kind": kind, "abstained": ans.abstained, "latency_ms": ans.latency_ms,
                   "tokens": ans.prompt_tokens + ans.completion_tokens, "relevance": ans.relevance}
            if kind == "answerable" and not ans.abstained:
                nm = numbers_match(q["answer"], ans.text)
                row["correct_lexical"] = nm if nm is not None else token_f1(q["answer"], ans.text) >= 0.5
                gold_pages = {(e["doc_name"], int(e["evidence_page_num"]) + 1) for e in q["evidence"]}
                cited = [ans.hits[i - 1].chunk for i in (ans.verification.valid_ids if ans.verification else [])]
                row["citation_correct"] = any((c.doc_id, c.page) in gold_pages for c in cited)
                row["unsupported_rate"] = ans.verification.unsupported_rate if ans.verification else None
                if a.judge:
                    row["correct_judge"] = judge_json(llm, JUDGE_PROMPT % (q["question"], q["answer"], ans.text), "correct")
                    ctx = "\n\n".join(h.chunk.text for h in ans.hits)
                    row["faithful_judge"] = judge_json(llm, FAITH_PROMPT % (ctx, ans.text), "faithful")
            rows.append(row)
            print(f"[{kind} {i}/{len(qs)}] abstained={ans.abstained} {ans.latency_ms:.0f} ms")
    df = pd.DataFrame(rows)
    df.to_csv(SETTINGS.results_dir / f"financebench_{a.tag}_generation_per_question.csv", index=False)
    A, O = df[df.kind == "answerable"], df[df.kind == "out_of_corpus"]
    answered = A[~A.abstained.astype(bool)]
    summary = {
        "config": a.gen_mode + "+corrective", "n_answerable": len(A), "n_out_of_corpus": len(O),
        "answer_rate": round(1 - A.abstained.mean(), 3) if len(A) else None,
        "correct_lexical": round(answered["correct_lexical"].astype(float).mean(), 3) if len(answered) else None,
        "citation_correct": round(answered["citation_correct"].astype(float).mean(), 3) if len(answered) else None,
        "mean_unsupported_rate": round(answered["unsupported_rate"].astype(float).mean(), 3) if len(answered) else None,
        "ooc_abstention_rate": round(O.abstained.astype(bool).mean(), 3) if len(O) else None,
        "p50_latency_ms": round(percentile(df.latency_ms.tolist(), 50)), "p95_latency_ms": round(percentile(df.latency_ms.tolist(), 95)),
        "mean_tokens_per_query": round(df.tokens.mean()),
    }
    if a.judge and len(answered):
        summary["correct_judge"] = round(answered["correct_judge"].dropna().astype(float).mean(), 3)
        summary["faithful_judge"] = round(answered["faithful_judge"].dropna().astype(float).mean(), 3)
    pd.DataFrame([summary]).to_csv(SETTINGS.results_dir / f"financebench_{a.tag}_generation_summary.csv", index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
