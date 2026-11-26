"""Document-level retrieval metrics (binary relevance): Recall@k, P@k, MRR, nDCG@k."""

from __future__ import annotations

import numpy as np


def doc_metrics(ranked: list[str], relevant: set[str], ks=(5, 10, 20)) -> dict[str, float]:
    hits = [d in relevant for d in ranked]
    out: dict[str, float] = {}
    for k in ks:
        out[f"recall@{k}"] = sum(hits[:k]) / max(1, len(relevant))
        out[f"p@{k}"] = sum(hits[:k]) / k
    first = next((i for i, h in enumerate(hits) if h), None)
    out["mrr"] = 1.0 / (first + 1) if first is not None else 0.0
    dcg = sum(1 / np.log2(i + 2) for i, h in enumerate(hits[:10]) if h)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(10, len(relevant))))
    out["ndcg@10"] = dcg / idcg if idcg else 0.0
    return out
