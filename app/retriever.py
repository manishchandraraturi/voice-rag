"""Retrieval wrapper — bridges the user's benchmark script to core/.

The project uses hnswlib (not FAISS) for dense search and bm25s for sparse.
This module wraps the core embedder into a self-contained retriever that
builds an in-memory hnswlib index from a small synthetic corpus, so the
benchmark can run without the full 241K-chunk index on disk.

Returns a SearchResult with per-stage timing (embed_ms, search_ms, total_ms)
matching the contract expected by app/benchmark.py.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import numpy as np

os.environ.setdefault("E5_VARIANT", "int8_x86")

from core.embedder import Embedder, EmbedderConfig

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Hit:
    text: str
    score: float


@dataclass(slots=True)
class SearchResult:
    query: str = ""
    hits: list[Hit] = field(default_factory=list)
    embed_ms: float = 0.0
    search_ms: float = 0.0
    total_ms: float = 0.0


# ---------------------------------------------------------------------------
# Synthetic corpus — small enough to live in memory, large enough to be real
# ---------------------------------------------------------------------------

_CORPUS = [
    "FAISS is a library for efficient similarity search and clustering of dense vectors.",
    "HNSW (Hierarchical Navigable Small World) indexing creates a multi-layered graph for approximate nearest neighbour search.",
    "Retrieval augmented generation (RAG) grounds LLM outputs in external knowledge by retrieving relevant documents before generation.",
    "The all-MiniLM-L6-v2 model produces 384-dimensional embeddings and is fast enough for CPU-only serving.",
    "RAG latency can be reduced by quantising embeddings, using approximate search indexes, and caching frequent queries.",
    "efSearch controls the number of neighbours inspected during HNSW search — higher values increase recall but also latency.",
    "Normalising embeddings before indexing lets cosine similarity be computed as a dot product, which is cheaper.",
    "A RAG pipeline typically has four stages: query encoding, retrieval, reranking, and generation.",
    "Vector quantisation reduces memory by compressing float32 embeddings to int8 or binary codes.",
    "Hybrid retrieval combines dense (semantic) and sparse (keyword) search for better recall across query types.",
    "BM25 is a term-frequency scoring function used in sparse retrieval. It is fast but misses synonyms.",
    "Reciprocal Rank Fusion (RRF) merges ranked lists from multiple retrievers by summing inverse ranks.",
    "Matryoshka embeddings allow truncating the vector to fewer dimensions while retaining most of the quality.",
    "Speculative prefetching predicts the next query from conversation context and pre-computes retrieval results.",
    "FlashRank is a lightweight cross-encoder reranker that runs on CPU in under 10ms per query.",
    "LLMLingua-2 compresses retrieved context before sending it to the LLM, reducing generation tokens and latency.",
    "Model Racing sends the same prompt to multiple LLMs in parallel and accepts the fastest valid response.",
    "RadixAttention (from SGLang) caches KV states for shared prompt prefixes across requests.",
    "Speculative RAG uses a small draft model to generate candidate answers, then verifies with a larger model.",
    "Semantic caching stores query-answer pairs keyed by embedding similarity, avoiding redundant retrieval and generation.",
    "The multilingual-e5-small model handles Devanagari script natively, unlike English-only encoders.",
    "Dynamic int8 quantisation on ONNX Runtime achieves near-fp32 accuracy with 2-3x throughput gain on AVX-512.",
    "Groq LPU hardware accelerates autoregressive decoding by eliminating the memory-bandwidth bottleneck of GPUs.",
    "Token-level chunking with overlap preserves context at chunk boundaries, reducing information loss.",
    "Sentence-boundary chunking respects linguistic units, producing more coherent passages for retrieval.",
    "Metadata-enriched chunks prepend query-type hints (e.g., [numeric], [location]) to improve retrieval precision.",
    "Document-level chunking reconstructs full documents from passage shards for broader context windows.",
    "The virama and matra problem in Hindi BM25 tokenisation was solved by splitting on separators instead of \\w+.",
    "Guardrails gate both the input (intent) and output (grounding) of a RAG pipeline to prevent hallucination.",
    "Abstaining on low-confidence queries is better than answering incorrectly — silence beats a wrong answer.",
]

# ---------------------------------------------------------------------------
# Module-level singletons (initialised by warmup)
# ---------------------------------------------------------------------------

_embedder: Embedder | None = None
_corpus_vecs: np.ndarray | None = None
_corpus_texts: list[str] = []

try:
    import hnswlib as _hnsw

    _HAS_HNSW = True
except ImportError:
    _HAS_HNSW = False

_hnsw_index = None


def warmup() -> None:
    """Load the embedding model and build the in-memory index."""
    global _embedder, _corpus_vecs, _corpus_texts, _hnsw_index

    if _embedder is not None:
        return

    cfg = EmbedderConfig(threads=int(os.getenv("ORT_THREADS", "0")))
    _embedder = Embedder(cfg)
    _corpus_texts = list(_CORPUS)

    # Encode corpus
    _corpus_vecs = _embedder.encode_passages(_corpus_texts)

    # Build hnswlib index
    if _HAS_HNSW:
        dim = _corpus_vecs.shape[1]
        idx = _hnsw.Index(space="ip", dim=dim)
        idx.init_index(max_elements=len(_corpus_vecs), ef_construction=200, M=16)
        idx.add_items(_corpus_vecs, list(range(len(_corpus_vecs))))
        idx.set_ef(64)
        _hnsw_index = idx


def search(query: str, top_k: int = 5) -> SearchResult:
    """Embed a query and search the in-memory index."""
    if _embedder is None:
        warmup()

    t0 = time.perf_counter()

    # Stage 1: embed
    qv = _embedder.encode_query(query)
    t_embed = time.perf_counter()

    # Stage 2: search
    if _hnsw_index is not None:
        labels, distances = _hnsw_index.knn_query(qv, k=min(top_k, len(_corpus_texts)))
        hits = [
            Hit(text=_corpus_texts[int(i)], score=round(float(d), 4))
            for i, d in zip(labels[0], distances[0])
        ]
    else:
        # Fallback: brute-force dot product
        scores = (_corpus_vecs @ qv.T).flatten()
        top_idx = np.argsort(scores)[::-1][:top_k]
        hits = [Hit(text=_corpus_texts[int(i)], score=round(float(scores[i]), 4)) for i in top_idx]

    t_search = time.perf_counter()

    return SearchResult(
        query=query,
        hits=hits,
        embed_ms=round((t_embed - t0) * 1000, 3),
        search_ms=round((t_search - t_embed) * 1000, 3),
        total_ms=round((t_search - t0) * 1000, 3),
    )
