import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from copilot.chunking import Chunk
from copilot.index import HybridIndex, tokenize


class FakeEmbedder:
    """Deterministic hashing bag-of-words embedder so tests never download a model."""
    model_name = "fake-hash-embedder"

    def encode(self, texts, batch_size=32, show_progress=False):
        out = np.zeros((len(texts), 256), dtype="float32")
        for i, t in enumerate(texts):
            for tok in tokenize(t):
                out[i, int(hashlib.md5(tok.encode()).hexdigest(), 16) % 256] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1, norms)


class FakeReranker:
    """Scores by word overlap, returned as logits."""
    def score(self, query, texts, batch_size=16):
        q = set(tokenize(query))
        return np.array([8.0 * len(q & set(tokenize(t))) / max(len(q), 1) - 4.0 for t in texts])


class FakeLLM:
    available = True
    model = "fake"

    def __init__(self, reply="Revenue was 100 million [1]."):
        self.reply, self.calls = reply, []

    def chat(self, messages, temperature=0.0, max_tokens=700):
        from copilot.llm import LLMResult
        self.calls.append(messages)
        return LLMResult(self.reply, 10, 5, 1.0)


DOCS = [
    ("annual", 3, "Item 7. Management Discussion", "Total revenue for fiscal 2023 was 100 million dollars, up from 90 million in fiscal 2022."),
    ("annual", 4, "Item 1A. Risk Factors", "Cybersecurity incidents could disrupt operations and damage our reputation with customers."),
    ("manual", 1, None, "To reset the router hold the recessed button for ten seconds until the light blinks."),
    ("manual", 2, None, "The warranty covers manufacturing defects for twenty four months from purchase."),
]


@pytest.fixture
def chunks():
    return [Chunk(f"{d}::p{p}::0", d, d.title(), f"{d}.pdf", p, s, t, {}) for d, p, s, t in DOCS]


@pytest.fixture
def index(chunks):
    return HybridIndex.build(chunks, FakeEmbedder())
