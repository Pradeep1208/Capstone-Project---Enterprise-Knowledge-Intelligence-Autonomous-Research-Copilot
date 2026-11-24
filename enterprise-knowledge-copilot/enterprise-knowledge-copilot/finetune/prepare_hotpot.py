#!/usr/bin/env python
"""Build an instruction-tuning set from HotpotQA (human-written questions, answers and supporting-fact labels).

The model learns the BEHAVIOUR we want (use only the numbered sources, answer concisely, return JSON with citations) --
not facts. Nothing here is AI-generated. Official download page: https://hotpotqa.github.io/
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import requests

URL = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"   # check hotpotqa.github.io if it moves
PROMPT = (
    "Answer the question using ONLY the numbered sources. Reply with JSON: "
    '{"answer": "<short answer>", "citations": [<source numbers>]}.\n\n%s\n\nQuestion: %s'
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=3000)
    p.add_argument("--out", default="finetune/outputs/data")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    raw = out / "hotpot_dev_distractor_v1.json"
    if not raw.exists():
        raw.write_bytes(requests.get(URL, timeout=300).content)
    items = json.loads(raw.read_text(encoding="utf-8"))
    rng = random.Random(42)
    rng.shuffle(items)
    rows = []
    for it in items:
        gold = {t for t, _ in it["supporting_facts"]}
        paras = it["context"]                       # [[title, [sentences]], ...]
        keep = [x for x in paras if x[0] in gold] + [x for x in paras if x[0] not in gold][:2]   # gold + 2 distractors
        rng.shuffle(keep)
        ctx = "\n".join(f"[{i}] {t}: {' '.join(s)}" for i, (t, s) in enumerate(keep, 1))
        cites = [i for i, (t, _) in enumerate(keep, 1) if t in gold]
        rows.append({"prompt": PROMPT % (ctx, it["question"]),
                     "response": json.dumps({"answer": it["answer"], "citations": cites})})
        if len(rows) >= a.n:
            break
    cut = int(len(rows) * 0.9)
    for name, part in (("train.jsonl", rows[:cut]), ("val.jsonl", rows[cut:])):
        with open(out / name, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in part)
    print(f"Wrote {cut} train / {len(rows) - cut} validation examples to {out}")


if __name__ == "__main__":
    main()
