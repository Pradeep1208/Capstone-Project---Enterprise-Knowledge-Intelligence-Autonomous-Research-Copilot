"""Sentence-embedding wrapper (lazy-loaded so the app starts fast)."""
from __future__ import annotations

import numpy as np


class Embedder:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device="cpu")
        return self._model

    def encode(self, texts: list[str], batch_size: int = 32, show_progress: bool = False) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype="float32")
        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,   # unit vectors: dot product == cosine similarity
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        )
        return vectors.astype("float32")
