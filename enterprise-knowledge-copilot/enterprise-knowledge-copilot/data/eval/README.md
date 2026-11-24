# Evaluation data

* `financebench_questions.jsonl` (created by `scripts/download_data.py financebench`) — 150 expert-written questions with
  gold answers and evidence page numbers, over real SEC filings. Source: https://github.com/patronus-ai/financebench
* `my_questions_template.csv` — header for questions YOU write by hand against your own NIST/SEC documents.
  Do not generate questions or answers with an AI model: read the document, write the question, copy the evidence
  passage and page number, and have a second person spot-check a sample. Aim for all five types in your proposal:
  direct factual, multi-document synthesis, keyword-sensitive, ambiguous, unanswerable (`answerable=False`).
