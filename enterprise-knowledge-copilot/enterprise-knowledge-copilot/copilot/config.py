"""Central configuration. Secrets come from environment variables or Streamlit secrets."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def get_secret(name: str, default: str | None = None) -> str | None:
    """Read a secret from the environment first, then from st.secrets (if available)."""
    value = os.environ.get(name)
    if value:
        return value
    try:
        import streamlit as st

        return st.secrets.get(name, default)
    except Exception:  # no secrets file, or Streamlit not installed/running
        return default


@dataclass(frozen=True)
class Settings:
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    chunker: str = "recursive"          # "recursive" or "fixed"
    chunk_size: int = 1000              # characters
    chunk_overlap: int = 150            # characters
    candidate_pool: int = 50            # candidates passed to the reranker
    top_k: int = 5                      # passages sent to the LLM
    rrf_k: int = 60                     # Reciprocal Rank Fusion constant
    relevance_threshold: float = 0.20   # sigmoid(cross-encoder) below this = weak evidence
    max_upload_mb: int = 10
    max_context_chars: int = 9000
    max_queries_per_session: int = 30   # protects your API budget on a public demo
    index_dir: Path = ROOT / "index" / "demo"
    results_dir: Path = ROOT / "results"


SETTINGS = Settings()
