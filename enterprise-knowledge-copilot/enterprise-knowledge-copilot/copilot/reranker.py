"""Cross-encoder reranker (query + passage -> relevance score)."""
from __future__ import annotations

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype="float64")))


class CrossEncoderReranker:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, max_length=512, device="cpu")
        return self._model

    def score(self, query: str, texts: list[str], batch_size: int = 16) -> np.ndarray:
        """Raw logits; apply sigmoid() for a 0-1 relevance probability."""
        if not texts:
            return np.zeros(0)
        pairs = [(query, t) for t in texts]
        return np.asarray(self.model.predict(pairs, batch_size=batch_size, show_progress_bar=False))
