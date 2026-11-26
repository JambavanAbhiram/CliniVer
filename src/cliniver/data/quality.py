"""Corpus-quality measurements used in Notebook 01 (and re-used for the Stage B corpus)."""

from __future__ import annotations

import re
import zlib
from collections import defaultdict
from itertools import combinations
from typing import Iterable

import numpy as np
import pandas as pd

from cliniver.schemas.document import Document

_TOKEN = re.compile(r"\w+", re.UNICODE)


def docs_to_frame(docs: Iterable[Document]) -> pd.DataFrame:
    rows = []
    for d in docs:
        rows.append({
            "doc_id": d.doc_id, "source": d.source, "role": d.role, "title": d.title,
            "n_chars": len(d.text), "n_words": len(_TOKEN.findall(d.text)), "n_sections": len(d.sections),
            "pub_date": pd.Timestamp(d.pub_date) if d.pub_date else pd.NaT, "date_precision": d.date_precision,
            "year": d.pub_date.year if d.pub_date else None, "publication_types": d.publication_types,
            "n_mesh": len(d.mesh_terms), "journal": d.journal, "language": d.language, "license": d.license,
            "url": d.url, "conditions": tuple(sorted(d.conditions)), "doi": d.identifiers.get("doi"),
            "pmid": d.identifiers.get("pmid"), **{f"x_{k}": v for k, v in d.extra.items()
                                                    if isinstance(v, (bool, int, float, str)) or v is None},
        })
    return pd.DataFrame(rows)


FIELDS = {
    "title": lambda d: bool(d.title.strip()),
    "text": lambda d: bool(d.text.strip()),
    "pub_date": lambda d: d.pub_date is not None,
    "date_day_precision": lambda d: d.date_precision == "day",
    "publication_types": lambda d: bool(d.publication_types),
    "mesh_terms": lambda d: bool(d.mesh_terms),
    "journal": lambda d: bool(d.journal),
    "language": lambda d: bool(d.language),
    "license": lambda d: bool(d.license),
    "url": lambda d: bool(d.url),
    "doi": lambda d: bool(d.identifiers.get("doi")),
}


def completeness(docs: Iterable[Document]) -> pd.DataFrame:
    """Fraction of documents with each field populated, per source (rows) × field (columns)."""
    acc: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for d in docs:
        for f, test in FIELDS.items():
            acc[d.source][f].append(test(d))
    return pd.DataFrame({src: {f: sum(v) / len(v) for f, v in fs.items()} for src, fs in acc.items()}).T


def shingles(text: str, k: int = 5) -> set[tuple[str, ...]]:
    """Word k-shingles. Texts shorter than k give one shingle (the whole text); empty text gives none
    (otherwise two empty texts would share the empty shingle and score Jaccard 1.0)."""
    toks = [t.lower() for t in _TOKEN.findall(text)]
    if not toks:
        return set()
    return {tuple(toks[i:i + k]) for i in range(max(len(toks) - k + 1, 1))}


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b) if len(a) <= len(b) else len(b & a)
    return inter / (len(a) + len(b) - inter)  # |A∪B| without materialising the union


def near_duplicate_pairs(docs: list[Document], group_key, threshold: float = 0.8, k: int = 5) -> pd.DataFrame:
    """Pairwise word-k-shingle Jaccard *within* groups (e.g. labels of the same drug).

    Quadratic per group, which is fine for Stage A samples. Stage B will need MinHash/LSH.
    """
    groups: dict[str, list[Document]] = defaultdict(list)
    for d in docs:
        groups[group_key(d)].append(d)
    rows = []
    for g, members in groups.items():
        sh = {d.doc_id: shingles(d.text, k) for d in members}
        for a, b in combinations(members, 2):
            j = jaccard(sh[a.doc_id], sh[b.doc_id])
            rows.append({"group": g, "doc_a": a.doc_id, "doc_b": b.doc_id, "jaccard": j, "near_dup": j >= threshold})
    return pd.DataFrame(rows, columns=["group", "doc_a", "doc_b", "jaccard", "near_dup"])


# --------------------------------------------------------------------------- MinHash + LSH
_P32 = np.uint64(4294967291)  # largest prime < 2^32; with 32-bit hashes and a,b < 2^32, a*h+b < 2^64 (no overflow)


def _shingle_hashes(text: str, k: int) -> np.ndarray:
    toks = [t.lower() for t in _TOKEN.findall(text)]
    grams = {" ".join(toks[i:i + k]) for i in range(max(len(toks) - k + 1, 1))} if toks else set()
    # crc32: deterministic across processes (Python's hash() is salted per process)
    return np.fromiter((zlib.crc32(g.encode()) for g in grams), dtype=np.uint64, count=len(grams))


def minhash_signatures(texts: list[str], num_perm: int = 128, k: int = 5, seed: int = 0) -> np.ndarray:
    """(n_docs, num_perm) MinHash signatures over word k-shingles (deterministic for a given seed)."""
    rng = np.random.default_rng(seed)
    a = rng.integers(1, int(_P32), size=num_perm, dtype=np.uint64)
    b = rng.integers(0, int(_P32), size=num_perm, dtype=np.uint64)
    sigs = np.full((len(texts), num_perm), np.iinfo(np.uint64).max, dtype=np.uint64)
    for i, text in enumerate(texts):
        h = _shingle_hashes(text, k)
        if h.size:
            sigs[i] = ((h[:, None] * a[None, :] + b[None, :]) % _P32).min(axis=0)
    return sigs


def lsh_candidate_pairs(sigs: np.ndarray, bands: int = 32) -> set[tuple[int, int]]:
    """Index pairs sharing at least one identical band (r = num_perm / bands rows per band).

    With 128 permutations and 32 bands (r = 4), pairs at Jaccard 0.8 become candidates with
    probability 1 - (1 - 0.8^4)^32 ≈ 1.0; candidates are then verified exactly.
    """
    n, m = sigs.shape
    if m % bands:
        raise ValueError("num_perm must be divisible by bands")
    r = m // bands
    pairs: set[tuple[int, int]] = set()
    for bnd in range(bands):
        buckets: dict[bytes, list[int]] = defaultdict(list)
        block = sigs[:, bnd * r:(bnd + 1) * r]
        for i in range(n):
            buckets[block[i].tobytes()].append(i)
        for members in buckets.values():
            if 1 < len(members):
                pairs.update(combinations(members, 2))
    return pairs


def near_duplicate_clusters(ids: list[str], texts: list[str], threshold: float = 0.8, k: int = 5,
                            num_perm: int = 128, bands: int = 32, seed: int = 0
                            ) -> tuple[pd.DataFrame, list[list[str]]]:
    """Scalable near-duplicate detection: MinHash/LSH candidates → exact Jaccard check → clusters.

    Clusters are connected components of the graph "Jaccard ≥ threshold". A candidate pair whose two
    documents are ALREADY in the same component is skipped: verifying it cannot change any component.
    Without this, a group where nearly everything is a duplicate (e.g. 555 generic labels of one drug)
    costs O(n²) exact comparisons (~530k in Stage B, > 1 h); with it, about O(n).

    Returns (verified edges with Jaccard ≥ threshold, i.e. the edges that formed the clusters — a
    spanning subset, not every duplicate pair; clusters of ≥ 2 ids). ``pairs.attrs`` also records the
    number of LSH candidates and of exact checks performed.
    """
    if len(ids) != len(texts):
        raise ValueError("ids and texts must align")
    sigs = minhash_signatures(texts, num_perm=num_perm, k=k, seed=seed)
    cand = lsh_candidate_pairs(sigs, bands=bands)
    cache: dict[int, set] = {}

    def sh(i: int) -> set:
        if i not in cache:
            cache[i] = shingles(texts[i], k)
        return cache[i]

    rows = []
    parent = list(range(len(ids)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    n_checked = 0
    for i, j in sorted(cand):  # sorted: deterministic edge order
        ri, rj = find(i), find(j)
        if ri == rj:
            continue  # already connected: this edge cannot change the components
        n_checked += 1
        jac = jaccard(sh(i), sh(j))
        if jac >= threshold:
            rows.append({"doc_a": ids[i], "doc_b": ids[j], "jaccard": jac})
            parent[ri] = rj
    groups: dict[int, list[str]] = defaultdict(list)
    for i in range(len(ids)):
        groups[find(i)].append(ids[i])
    clusters = [g for g in groups.values() if len(g) > 1]
    pairs = pd.DataFrame(rows, columns=["doc_a", "doc_b", "jaccard"])
    pairs.attrs["n_candidates"] = len(cand)
    pairs.attrs["n_exact_checks"] = n_checked
    return pairs, clusters


def merge_duplicate_ids(docs: Iterable[Document]) -> tuple[list[Document], int]:
    """Collapse identical doc_ids retrieved under several conditions (union of ``conditions``)."""
    merged: dict[str, Document] = {}
    dups = 0
    for d in docs:
        if d.doc_id in merged:
            dups += 1
            prev = merged[d.doc_id]
            merged[d.doc_id] = prev.model_copy(update={"conditions": sorted(set(prev.conditions) | set(d.conditions))})
        else:
            merged[d.doc_id] = d
    return list(merged.values()), dups
