"""BM25 with tokenizer variants (NB05 Part C).

* ``default``  — bm25s built-in: lowercase, English stopwords, tokens of ≥ 2 word characters (drops "5", splits
                 "SGLT-2" into "sglt", and loses "<", "%").
* ``clinical`` — keeps numbers, units and hyphenated/slashed names as single tokens ("sglt-2", "5", "mg/dl",
                 "1.73", "≥30"), lowercase, the same English stopword list, no stemming.
"""

from __future__ import annotations

import re
import time

import numpy as np

_CLIN = re.compile(r"[≥≤<>]?\d+(?:[.,]\d+)?%?|[a-z0-9]+(?:[-/][a-z0-9]+)*", re.I)


def clinical_tokenize(text: str, stop: set[str]) -> list[str]:
    return [t for t in (m.group(0).lower() for m in _CLIN.finditer(text)) if t not in stop]


class BM25:
    def __init__(self, texts: list[str], variant: str = "clinical", k1: float = 1.5, b: float = 0.75) -> None:
        import bm25s
        from bm25s.stopwords import STOPWORDS_EN
        self.bm25s, self.variant = bm25s, variant
        self.stop = set(STOPWORDS_EN)
        t0 = time.perf_counter()
        self.model = bm25s.BM25(k1=k1, b=b)
        self.model.index(self._tok(texts), show_progress=False)
        self.index_s = time.perf_counter() - t0
        self.n = len(texts)

    def _tok(self, texts):
        if self.variant == "default":
            return self.bm25s.tokenize(texts, stopwords="en", show_progress=False)
        return [clinical_tokenize(t, self.stop) for t in texts]

    def search(self, queries: list[str], k: int) -> tuple[np.ndarray, np.ndarray]:
        q = self._tok(queries)
        if self.variant != "default":
            q = [x if x else ["∅"] for x in q]  # bm25s needs at least one token per query
        ids, scores = self.model.retrieve(q, k=min(k, self.n), show_progress=False, n_threads=0)
        return np.asarray(ids), np.asarray(scores)
