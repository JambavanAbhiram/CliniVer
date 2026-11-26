"""Rank / score fusion for combining separate retrievals (NB04 strategies D and E; reused in NB05)."""

from __future__ import annotations

import numpy as np


def rrf(rankings: list[np.ndarray], k: int = 60, top: int = 100) -> np.ndarray:
    """Reciprocal-rank fusion of several ranked id arrays (Cormack et al., 2009)."""
    score: dict[int, float] = {}
    for r in rankings:
        for rank, i in enumerate(r):
            score[int(i)] = score.get(int(i), 0.0) + 1.0 / (k + rank + 1)
    return np.array(sorted(score, key=lambda i: (-score[i], i))[:top])


def weighted(ids_a: np.ndarray, sc_a: np.ndarray, ids_b: np.ndarray, sc_b: np.ndarray, w: float, top: int = 100) -> np.ndarray:
    """w · minmax(a) + (1 − w) · minmax(b) over the union of both candidate lists (missing score = 0)."""
    def mm(s):
        s = np.asarray(s, dtype=float)
        rng = s.max() - s.min()
        return (s - s.min()) / rng if rng > 0 else np.ones_like(s)
    score: dict[int, float] = {}
    for ids, sc, wt in ((ids_a, mm(sc_a), w), (ids_b, mm(sc_b), 1 - w)):
        for i, s in zip(ids, sc):
            score[int(i)] = score.get(int(i), 0.0) + wt * float(s)
    return np.array(sorted(score, key=lambda i: (-score[i], i))[:top])
