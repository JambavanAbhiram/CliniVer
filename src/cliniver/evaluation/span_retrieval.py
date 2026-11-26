"""Span-level retrieval metrics for comparing chunkings (NB02).

A question has one gold answer span (doc_id, start, end) in the document text. A retrieved chunk is
*relevant* if it contains the whole span (strict); its *coverage* is the fraction of the span's
characters it contains. Comparing chunk sizes at a fixed k favours large chunks trivially, so every
metric is also reported at a fixed **token budget**: retrieved chunks are taken in rank order while
their cumulative token count stays ≤ the budget (what a verifier with a fixed context could read).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Gold:
    qid: str
    doc_id: str
    start: int
    end: int


def _cover(c_doc, c_start, c_end, g: Gold) -> float:
    if c_doc != g.doc_id:
        return 0.0
    inter = min(c_end, g.end) - max(c_start, g.start)
    return max(0, inter) / max(1, g.end - g.start)


def evaluate(golds: list[Gold], ranked: np.ndarray, chunk_doc: list[str], chunk_start: np.ndarray,
             chunk_end: np.ndarray, chunk_tokens: np.ndarray, ks=(1, 5, 10), budgets=(512, 1024, 2048)) -> pd.DataFrame:
    """One row per question. ``ranked[i]`` = chunk indices for question i, best first."""
    by_doc: dict[str, list[int]] = {}
    for j, d in enumerate(chunk_doc):
        by_doc.setdefault(d, []).append(j)
    rows = []
    for g, r in zip(golds, ranked):
        cov = np.array([_cover(chunk_doc[j], chunk_start[j], chunk_end[j], g) for j in r])
        full = cov >= 1.0 - 1e-9
        row = {"qid": g.qid, "doc_id": g.doc_id,
               # is the gold span inside ANY chunk of its document? (upper bound for strict hits)
               "gold_in_one_chunk": any(chunk_start[j] <= g.start and chunk_end[j] >= g.end for j in by_doc.get(g.doc_id, []))}
        first = int(np.argmax(full)) + 1 if full.any() else 0
        row["rr"] = 1.0 / first if first else 0.0
        for k in ks:
            row[f"hit@{k}"] = bool(full[:k].any())
            row[f"cover@{k}"] = _union_cover(g, [(chunk_doc[j], chunk_start[j], chunk_end[j]) for j in r[:k]])
        gains = cov[:10]
        dcg = float(np.sum(gains / np.log2(np.arange(2, len(gains) + 2))))
        row["ndcg@10"] = dcg  # ideal DCG = 1 (one chunk containing the whole span at rank 1)
        cum = np.cumsum(chunk_tokens[r])
        for B in budgets:
            n = int(np.searchsorted(cum, B, side="right"))
            row[f"hit@{B}t"] = bool(full[:n].any())
            row[f"cover@{B}t"] = _union_cover(g, [(chunk_doc[j], chunk_start[j], chunk_end[j]) for j in r[:n]])
        rows.append(row)
    return pd.DataFrame(rows)


def _union_cover(g: Gold, spans) -> float:
    iv = sorted((max(a, g.start), min(b, g.end)) for d, a, b in spans if d == g.doc_id and min(b, g.end) > max(a, g.start))
    tot, cur_a, cur_b = 0, None, None
    for a, b in iv:
        if cur_b is None or a > cur_b:
            if cur_b is not None:
                tot += cur_b - cur_a
            cur_a, cur_b = a, b
        else:
            cur_b = max(cur_b, b)
    if cur_b is not None:
        tot += cur_b - cur_a
    return tot / max(1, g.end - g.start)


def cluster_bootstrap_diff(a: pd.Series, b: pd.Series, clusters: pd.Series, n_boot: int = 2000, seed: int = 0) -> dict:
    """Paired difference mean(a) - mean(b) with a 95% CI from resampling *clusters* (articles), because
    questions about the same article are not independent."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"d": a.astype(float).values - b.astype(float).values, "c": clusters.values})
    g = df.groupby("c")["d"].agg(["sum", "count"])
    s, n = g["sum"].to_numpy(), g["count"].to_numpy()
    idx = rng.integers(0, len(g), size=(n_boot, len(g)))
    boots = s[idx].sum(1) / n[idx].sum(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"diff": float(df.d.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "p_two_sided": float(min(1.0, 2 * min((boots <= 0).mean(), (boots >= 0).mean()))), "n_clusters": int(len(g))}
