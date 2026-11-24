# Enterprise Knowledge Intelligence & Autonomous Research Copilot

MSc Data Science capstone — hybrid-retrieval RAG with reranking, corrective retrieval, citation verification,
systematic evaluation and a Streamlit front end. All datasets are **real, human-produced public data** (no AI-generated data).

```
Documents ─► ingestion.py ─► chunking.py ─► embeddings.py ─┬► index.py (vectors + BM25) ─► retrieval.py (RRF) ─► reranker.py
 (PDF/DOCX/HTML/MD/CSV/JSON/TXT)                           │                                                      │
                                                           ▼                                                      ▼
                          app.py (Streamlit) ◄── citations.py ◄── llm.py ◄── rag.py (relevance check → rewrite/retry → abstain)
                                                                              evaluation.py + metrics.py  (offline benchmarks)
```

## 1. Datasets (all real, all free)

| Role | Dataset | Link | Used by |
|---|---|---|---|
| **Knowledge corpus + QA benchmark** (recommended core) | **FinanceBench** — 150 expert-written questions, gold answers and evidence pages over real SEC 10-K/10-Q filings | https://github.com/patronus-ai/financebench · https://huggingface.co/datasets/PatronusAI/financebench | `download_data.py financebench` |
| Retrieval benchmark | **BEIR** (SciFact: 5k docs / 300 queries; FiQA: finance QA) with human relevance judgements | https://github.com/beir-cellar/beir · zips: https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/ | `download_data.py beir` |
| Extra corpus: US filings | **SEC EDGAR** 10-K filings (public record; declare a User-Agent, max 10 requests/s) | https://www.sec.gov/os/accessing-edgar-data | `download_data.py sec` |
| Extra corpus: standards | **NIST** AI RMF, GenAI profile, CSF 2.0, SSDF, incident-response guide | https://csrc.nist.gov/publications · https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf · https://nvlpubs.nist.gov/nistpubs/CSWP/NIST.CSWP.29.pdf | `download_data.py nist` |
| Optional fine-tuning data | **HotpotQA** (human questions + supporting facts) | https://hotpotqa.github.io/ | `finetune/prepare_hotpot.py` |

Licences: check each source before redistributing. FinanceBench's open sample is reported as CC-BY-NC-4.0 (non-commercial; confirm on its repo/HF page and cite
Islam et al., 2023, arXiv:2311.11944). NIST publications are US Government works; SEC filings are public records. BEIR distributes data as-is and does not vouch for licences.

## 2. Setup (Python 3.12 recommended)

```bash
python3.12 -m venv .venv
source .venv/bin/activate            # Windows PowerShell:  .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pytest -q                  # 18 fast tests, no model downloads
```

## 3. Build the demo corpus and index

```bash
python scripts/download_data.py financebench --docs 12      # 150 questions + the 12 filings with most questions
python scripts/download_data.py nist                        # optional extra documents
python scripts/build_index.py --corpus data/corpus/financebench --corpus data/corpus/nist --out index/demo
```
First run downloads two small Hugging Face models (embedding ≈ 90 MB, reranker ≈ 90 MB). Indexing ~10k chunks on CPU takes roughly 5–15 minutes.
Optional SEC download: `export SEC_USER_AGENT="Your Name you@example.com"` then `python scripts/download_data.py sec --tickers AAPL,MSFT`.

## 4. Run the app locally

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # then edit: LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
streamlit run app.py
```
Without an LLM key the app still works in **extractive mode** (shows the best passages). Any OpenAI-compatible endpoint works
(Groq, Hugging Face router, OpenAI, OpenRouter…); take the current model name from your provider's model list.

## 5. Evaluate (this produces your report's results)

```bash
python scripts/download_data.py beir --name scifact
python scripts/eval_beir.py --name scifact                              # vector vs BM25 vs hybrid vs hybrid+rerank
python scripts/eval_financebench.py --index-dir index/demo              # page-level + document-level retrieval metrics
# generation + abstention (needs LLM_* variables in your shell):
export LLM_API_KEY=... LLM_BASE_URL=... LLM_MODEL=...
python scripts/eval_financebench.py --index-dir index/demo --generate --limit 40 --judge
```
Results land in `results/*.csv`; the app's **Evaluation** page displays every `*_summary.csv`. **Chunking comparison:** build a second index with
`--chunker fixed --out index/fixed`, run `eval_financebench.py --index-dir index/fixed --tag fixed`, and compare with the recursive run.

Metrics: Recall@5/10, Precision@5, MRR@10, NDCG@10, P50/P95 retrieval latency; generation: lexical/judge correctness, citation correctness,
unsupported-statement rate, faithfulness (judge), abstention on out-of-corpus questions, tokens per query.
Notes: FinanceBench page numbers are 0-based (this repo converts to 1-based); out-of-corpus questions are a *proxy* for "unanswerable".
Calibrate `relevance_threshold` in `copilot/config.py` on your own benchmark (compare answer rate vs. abstention).

## 6. Deploy on Streamlit Community Cloud

1. Create a GitHub repo and push this folder (`git init && git add . && git commit -m "init" && git branch -M main && git remote add origin <url> && git push -u origin main`).
   `data/corpus/` is git-ignored; **do commit `index/demo/` and `results/`** (keep files < 100 MB each).
2. Go to https://share.streamlit.io → **New app** → pick the repo, branch `main`, main file `app.py`.
3. Click **Advanced settings** → choose **Python 3.12** → paste your secrets (`LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`) → Deploy.
4. Python version cannot be changed after deployment — delete and redeploy the app to change it. Community Cloud has been reported to default to very new Python releases
   that some ML packages do not support yet, which is why 3.12 is chosen explicitly.
5. Never commit `.streamlit/secrets.toml`. The app limits uploads (10 MB, allowed types only) and questions per session (30).

If the build runs out of memory, use a smaller index (fewer filings) or switch `requirements.txt` to a lighter embedding stack.

## 7. Optional: fine-tuning (LoRA/QLoRA)
Run on a GPU (e.g. Colab T4), never on Streamlit Cloud: `pip install -r finetune/requirements-finetune.txt`,
`python finetune/prepare_hotpot.py`, `python finetune/lora_train.py`. Fine-tuning teaches *format and behaviour* (cited JSON answers); facts stay in RAG.
Evaluate base vs adapted model on identical prompts before claiming an improvement.

## 8. Design notes and limitations
* Vector search is exact NumPy cosine search (no vector DB): zero native dependencies, deterministic experiments, ample for ≤100k chunks. Swap in Chroma/pgvector behind `HybridIndex` if needed.
* Citation verification is a lexical heuristic; it flags weak support but cannot prove correctness.
* PDF tables are extracted as plain text; scanned PDFs need OCR (not included).
* Retrieved text is treated as untrusted (delimited sources, instruction-like text neutralised) — this reduces, not eliminates, prompt-injection risk.
