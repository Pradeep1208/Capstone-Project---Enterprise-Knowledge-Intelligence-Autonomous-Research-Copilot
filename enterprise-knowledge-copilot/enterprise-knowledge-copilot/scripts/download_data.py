#!/usr/bin/env python
"""Download REAL public datasets (nothing here is synthetic or AI-generated).

  python scripts/download_data.py financebench --docs 12     # 150 expert Q&A + N SEC filing PDFs (recommended)
  python scripts/download_data.py beir --name scifact        # BEIR retrieval benchmark (human relevance labels)
  python scripts/download_data.py nist                       # NIST AI / cybersecurity PDFs (US Government)
  python scripts/download_data.py sec --tickers AAPL,MSFT    # latest 10-K filings straight from SEC EDGAR
"""
from __future__ import annotations

import argparse
import collections
import io
import json
import os
import sys
import time
import zipfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
FB_RAW = "https://raw.githubusercontent.com/patronus-ai/financebench/main"
BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"

NIST_DOCS = [  # (file stem, URL) -- US Government publications, free to download
    ("NIST_AI_100-1", "https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf"),
    ("NIST_AI_600-1", "https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf"),
    ("NIST_CSWP_29", "https://nvlpubs.nist.gov/nistpubs/CSWP/NIST.CSWP.29.pdf"),
    ("NIST_SP_800-218", "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-218.pdf"),
    ("NIST_SP_800-61r3", "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-61r3.pdf"),
]
CIKS = {"AAPL": 320193, "MSFT": 789019, "GOOGL": 1652044, "AMZN": 1018724, "NVDA": 1045810}


def get(url: str, headers: dict | None = None, stream: bool = False, retries: int = 3) -> requests.Response:
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=headers, timeout=120, stream=stream)
            r.raise_for_status()
            return r
        except requests.RequestException as exc:
            if attempt == retries - 1:
                raise
            print(f"  retry {attempt + 1} after error: {type(exc).__name__}")
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("unreachable")


def save(url: str, dest: Path, headers: dict | None = None) -> bool:
    if dest.exists() and dest.stat().st_size > 1000:
        print(f"  have {dest.name}")
        return True
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(get(url, headers).content)
        print(f"  saved {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
        return True
    except requests.RequestException as exc:
        print(f"  FAILED {dest.name}: {type(exc).__name__} - check the URL on the publisher's site")
        return False


# ----------------------------------------------------------------------------- FinanceBench
def financebench(n_docs: int) -> None:
    out_dir, eval_dir = ROOT / "data/corpus/financebench", ROOT / "data/eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    questions = [json.loads(l) for l in get(f"{FB_RAW}/data/financebench_open_source.jsonl").text.splitlines() if l.strip()]
    counts = collections.Counter(q["doc_name"] for q in questions)
    chosen = [d for d, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:n_docs]]
    print(f"{len(questions)} questions over {len(counts)} filings; downloading the {len(chosen)} filings with most questions")
    for doc in chosen:
        save(f"{FB_RAW}/pdfs/{doc}.pdf", out_dir / f"{doc}.pdf")
    with open(eval_dir / "financebench_questions.jsonl", "w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    (eval_dir / "financebench_demo_docs.json").write_text(json.dumps(chosen, indent=2))
    n_in = sum(counts[d] for d in chosen)
    print(f"{n_in} of {len(questions)} questions are answerable from the downloaded subset.")


# ----------------------------------------------------------------------------- BEIR
def beir(name: str) -> None:
    target = ROOT / "data/beir"
    if (target / name / "corpus.jsonl").exists():
        print(f"BEIR/{name} already downloaded")
        return
    print(f"Downloading BEIR/{name} ...")
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(get(BEIR_URL.format(name=name)).content)) as z:
        z.extractall(target)
    print(f"Extracted to {target / name}")


# ----------------------------------------------------------------------------- NIST
def nist() -> None:
    for stem, url in NIST_DOCS:
        save(url, ROOT / "data/corpus/nist" / f"{stem}.pdf")


# ----------------------------------------------------------------------------- SEC EDGAR
def sec(tickers: list[str], per_company: int) -> None:
    ua = os.environ.get("SEC_USER_AGENT")
    if not ua:
        sys.exit('SEC requires a declared User-Agent. Example:  export SEC_USER_AGENT="Your Name your.email@example.com"')
    headers = {"User-Agent": ua, "Accept-Encoding": "gzip, deflate"}
    for ticker in tickers:
        cik = CIKS.get(ticker.upper())
        if not cik:
            print(f"Unknown ticker {ticker}; known: {', '.join(CIKS)}")
            continue
        recent = get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json", headers).json()["filings"]["recent"]
        time.sleep(0.2)   # SEC fair access: max 10 requests/second
        found = 0
        for form, acc, doc, period in zip(recent["form"], recent["accessionNumber"], recent["primaryDocument"], recent["reportDate"]):
            if form != "10-K":
                continue
            url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc.replace('-', '')}/{doc}"
            save(url, ROOT / "data/corpus/sec" / f"{ticker.upper()}_{period}_10K.htm", headers)
            time.sleep(0.2)
            found += 1
            if found >= per_company:
                break


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("financebench"); s.add_argument("--docs", type=int, default=12)
    s = sub.add_parser("beir"); s.add_argument("--name", default="scifact", help="scifact | fiqa | nfcorpus | ...")
    sub.add_parser("nist")
    s = sub.add_parser("sec"); s.add_argument("--tickers", default="AAPL,MSFT"); s.add_argument("--per-company", type=int, default=2)
    a = p.parse_args()
    {"financebench": lambda: financebench(a.docs), "beir": lambda: beir(a.name), "nist": nist,
     "sec": lambda: sec(a.tickers.split(","), a.per_company)}[a.cmd]()


if __name__ == "__main__":
    main()
