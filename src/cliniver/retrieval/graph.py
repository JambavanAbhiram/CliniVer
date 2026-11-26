"""Concept–document graph and Personalized PageRank (NB05 Part D; ADR-0003; HippoRAG-style).

Nodes: documents (or chunks) and concepts (normalized entity identifiers, e.g. MeSH / NCBI Gene IDs from
PubTator). Edges: document —mentions— concept (weight = log(1 + mention count)), and concept —relation—
concept from PubTator relations (weight = log(1 + number of documents asserting it)). A query is linked to
concepts; PPR with restart on those seed concepts spreads mass over the graph; documents are ranked by their
PPR mass. Edges are retrieval signals only, never evidence (similarity ≠ verification).
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict

import numpy as np
import scipy.sparse as sp


class ConceptGraph:
    def __init__(self, doc_concepts: dict[str, Counter], relations: Counter | None = None) -> None:
        self.docs = sorted(doc_concepts)
        concepts = sorted({c for cs in doc_concepts.values() for c in cs} | {c for pair in (relations or {}) for c in pair})
        self.concepts = concepts
        self.c_index = {c: i for i, c in enumerate(concepts)}
        nd, nc = len(self.docs), len(concepts)
        rows, cols, vals = [], [], []
        for di, d in enumerate(self.docs):
            for c, n in doc_concepts[d].items():
                ci = nd + self.c_index[c]
                w = float(np.log1p(n))
                rows += [di, ci]; cols += [ci, di]; vals += [w, w]
        for (a, b), n in (relations or Counter()).items():
            if a in self.c_index and b in self.c_index and a != b:
                ia, ib = nd + self.c_index[a], nd + self.c_index[b]
                w = float(np.log1p(n))
                rows += [ia, ib]; cols += [ib, ia]; vals += [w, w]
        A = sp.csr_matrix((vals, (rows, cols)), shape=(nd + nc, nd + nc))
        deg = np.asarray(A.sum(axis=1)).ravel()
        deg[deg == 0] = 1.0
        self.P = (sp.diags(1.0 / deg) @ A).T.tocsr()  # column-stochastic transition matrix
        self.nd, self.nc = nd, nc
        self.n_edges = A.nnz // 2

    def ppr(self, seeds: list[str], damping: float = 0.5, iters: int = 30) -> np.ndarray:
        """Document scores (length n_docs) from PPR restarted at ``seeds``; zeros when no seed is in the graph."""
        idx = [self.nd + self.c_index[s] for s in seeds if s in self.c_index]
        if not idx:
            return np.zeros(self.nd)
        r = np.zeros(self.nd + self.nc)
        r[idx] = 1.0 / len(idx)
        x = r.copy()
        for _ in range(iters):
            x = (1 - damping) * r + damping * (self.P @ x)
        return x[:self.nd]


class EntityLinker:
    """Dictionary linker built from corpus mentions: surface form → most frequent concept id.

    Matching is by **word n-gram hash lookup** (longest match first, up to ``max_words`` words, case-insensitive,
    words = alphanumeric runs incl. inner hyphens), so a dictionary of 10^5 surface forms links 10^5 documents in
    minutes. Surface forms shorter than ``min_len`` characters or seen fewer than ``min_count`` times are ignored
    (ambiguity control)."""

    _W = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")

    def __init__(self, mentions: list[tuple[str, str]], min_len: int = 3, min_count: int = 2, max_words: int = 6) -> None:
        by_surface: dict[str, Counter] = defaultdict(Counter)
        for surface, cid in mentions:
            key = " ".join(w.lower() for w in self._W.findall(surface))
            if key:
                by_surface[key][cid] += 1
        self.surface = {k: cs.most_common(1)[0][0] for k, cs in by_surface.items()
                        if len(k) >= min_len and sum(cs.values()) >= min_count}
        self.max_words = min(max_words, max((k.count(" ") + 1 for k in self.surface), default=1))
        totals = {k: sum(cs.values()) for k, cs in by_surface.items()}
        self.surfaces_of: dict[str, list[str]] = defaultdict(list)
        for k, c in self.surface.items():
            self.surfaces_of[c].append(k)
        for c in self.surfaces_of:  # most frequent surface form first
            self.surfaces_of[c].sort(key=lambda k: -totals[k])

    def link(self, text: str) -> list[tuple[str, str]]:
        """[(surface as written, concept id)] in order of appearance, longest match first, non-overlapping."""
        words = [(m.group(0), m.start(), m.end()) for m in self._W.finditer(text)]
        low = [w.lower() for w, _, _ in words]
        out, i = [], 0
        while i < len(words):
            for n in range(min(self.max_words, len(words) - i), 0, -1):
                key = " ".join(low[i:i + n])
                if key in self.surface:
                    out.append((text[words[i][1]:words[i + n - 1][2]], self.surface[key])); i += n
                    break
            else:
                i += 1
        return out
