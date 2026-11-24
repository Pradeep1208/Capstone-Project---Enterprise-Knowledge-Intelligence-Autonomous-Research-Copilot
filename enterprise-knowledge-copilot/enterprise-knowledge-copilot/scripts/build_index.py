#!/usr/bin/env python
"""Parse -> chunk -> embed -> save a hybrid index.

  python scripts/build_index.py --corpus data/corpus/financebench --out index/demo
  python scripts/build_index.py --corpus data/corpus/financebench --corpus data/corpus/nist --chunker fixed --out index/fixed
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from copilot.chunking import chunk_document
from copilot.config import SETTINGS
from copilot.embeddings import Embedder
from copilot.index import HybridIndex
from copilot.ingestion import SUPPORTED_EXTENSIONS, load_path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", action="append", required=True, help="folder of documents (repeatable)")
    p.add_argument("--out", default=str(SETTINGS.index_dir))
    p.add_argument("--chunker", choices=["recursive", "fixed"], default=SETTINGS.chunker)
    p.add_argument("--size", type=int, default=SETTINGS.chunk_size)
    p.add_argument("--overlap", type=int, default=SETTINGS.chunk_overlap)
    p.add_argument("--model", default=SETTINGS.embedding_model)
    a = p.parse_args()

    files = sorted(f for folder in a.corpus for f in Path(folder).rglob("*") if f.suffix.lower() in SUPPORTED_EXTENSIONS)
    if not files:
        sys.exit("No supported documents found. Run scripts/download_data.py first.")
    chunks = []
    for f in files:
        try:
            doc = load_path(f, doc_id=f.stem)
        except ValueError as exc:
            print(f"SKIP {f.name}: {exc}")
            continue
        new = chunk_document(doc, a.chunker, a.size, a.overlap)
        chunks.extend(new)
        print(f"{f.name}: {len(doc.pages)} pages -> {len(new)} chunks")
    print(f"\nEmbedding {len(chunks)} chunks with {a.model} (CPU: this can take several minutes) ...")
    t0 = time.time()
    index = HybridIndex.build(chunks, Embedder(a.model), show_progress=True)
    index.save(a.out, {"chunker": a.chunker, "chunk_size": a.size, "overlap": a.overlap, "n_documents": len(files)})
    print(f"Saved index to {a.out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
