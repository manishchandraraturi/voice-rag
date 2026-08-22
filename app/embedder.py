"""Bridge between eval loop and our core ONNX embedder."""
from __future__ import annotations

import numpy as np
from core.embedder import Embedder

_embedder: Embedder | None = None


def get_model() -> Embedder:
    """Initialize and return the global embedder model."""
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder


def embed_one(text: str) -> np.ndarray:
    """Embed a single query/text into a 1D vector (dim,)."""
    return get_model().encode_query(text)


def embed(texts: list[str]) -> np.ndarray:
    """Embed a list of passage texts into a 2D array (len(texts), dim)."""
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    return get_model().encode_passages(texts)
