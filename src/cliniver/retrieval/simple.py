"""Fixed reference retrievers used to *compare chunkings* in NB02 (not the final retrieval stack; NB05).

* ``BM25Index``  — bm25s (Lucene-style BM25, k1=1.5, b=0.75), English stopwords, lowercase.
* ``DenseIndex`` — a sentence-transformers bi-encoder with exact (brute-force) cosine search.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np


@dataclass
class SearchResult:
    ids: np.ndarray  # (n_queries, k) indices into the indexed texts
    scores: np.ndarray
    index_s: float
    query_s: float


class BM25Index:
    def __init__(self, texts: list[str]) -> None:
        import bm25s
        self._bm25s = bm25s
        t0 = time.perf_counter()
        self.model = bm25s.BM25(k1=1.5, b=0.75)
        self.model.index(bm25s.tokenize(texts, stopwords="en", show_progress=False), show_progress=False)
        self.n = len(texts)
        self.index_s = time.perf_counter() - t0

    def search(self, queries: list[str], k: int) -> SearchResult:
        t0 = time.perf_counter()
        k = min(k, self.n)
        ids, scores = self.model.retrieve(self._bm25s.tokenize(queries, stopwords="en", show_progress=False), k=k,
                                          show_progress=False, n_threads=0)
        return SearchResult(np.asarray(ids), np.asarray(scores), self.index_s, time.perf_counter() - t0)


class DenseIndex:
    def __init__(self, texts: list[str], model, *, batch_size: int = 64, query_prefix: str = "") -> None:
        import torch
        self.model, self.query_prefix, self.torch = model, query_prefix, torch
        t0 = time.perf_counter()
        emb = model.encode(texts, batch_size=batch_size, normalize_embeddings=True, convert_to_tensor=True,
                           show_progress_bar=False)
        self.emb = emb.half() if emb.is_cuda else emb
        self.index_s = time.perf_counter() - t0
        self.n = len(texts)

    def search(self, queries: list[str], k: int) -> SearchResult:
        t0 = time.perf_counter()
        q = self.model.encode([self.query_prefix + x for x in queries], batch_size=128, normalize_embeddings=True,
                              convert_to_tensor=True, show_progress_bar=False).to(self.emb.dtype)
        k = min(k, self.n)
        ids, scores = [], []
        for i in range(0, len(q), 256):
            s = q[i:i + 256] @ self.emb.T
            top = self.torch.topk(s.float(), k, dim=1)
            ids.append(top.indices.cpu().numpy()); scores.append(top.values.cpu().numpy())
        return SearchResult(np.vstack(ids), np.vstack(scores), self.index_s, time.perf_counter() - t0)
