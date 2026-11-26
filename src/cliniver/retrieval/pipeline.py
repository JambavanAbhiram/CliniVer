"""The selected evidence retriever (ADR-0012): BM25 + MedCPT dense (RRF) → MedCPT cross-encoder re-ranking.

Retrieval units (ADR-0009 + amendment 2026-11-21): whole PubMed abstracts; ``paragraph-128`` chunks for openFDA
labels, Europe PMC full texts and ClinicalTrials.gov records.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from cliniver.retrieval.encoders import Encoder
from cliniver.retrieval.fusion import rrf
from cliniver.retrieval.sparse import BM25


@dataclass
class Unit:
    unit_id: str
    doc_id: str
    source: str
    title: str
    text: str          # what the verifier reads
    index_text: str    # what BM25 indexes
    pub_date: str | None = None


def build_units(documents_jsonl: Path, chunks_jsonl: Path) -> tuple[list[Unit], list[int]]:
    """Whole PubMed abstracts + non-PubMed chunks. Returns (units, row index of each non-PubMed unit in the chunk
    file, or -1 for abstracts) so cached chunk embeddings can be reused."""
    from cliniver.processing.long_docs import iter_documents
    units, rows = [], []
    meta = {}
    for d in iter_documents(documents_jsonl):
        meta[d["doc_id"]] = (d.get("title", ""), d.get("pub_date"))
        if d["source"] == "pubmed" and d["text"].strip():
            units.append(Unit(f"{d['doc_id']}#whole", d["doc_id"], "pubmed", d["title"], d["text"], f"{d['title']}. {d['text']}", d.get("pub_date")))
            rows.append(-1)
    with chunks_jsonl.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            c = json.loads(line)
            if c["source"] == "pubmed":
                continue
            t, pdate = meta.get(c["doc_id"], ("", None))
            units.append(Unit(c["chunk_id"], c["doc_id"], c["source"], t, c["text"], c["index_text"], pdate))
            rows.append(i)
    return units, rows


class EvidenceRetriever:
    def __init__(self, units: list[Unit], dense: Encoder, unit_emb: torch.Tensor, cross_encoder, *, depth: int = 100,
                 pool: int = 50, rrf_k: int = 60, bm25_variant: str = "default") -> None:
        self.units, self.dense, self.emb, self.ce = units, dense, unit_emb, cross_encoder
        self.depth, self.pool, self.rrf_k = depth, pool, rrf_k
        t0 = time.perf_counter()
        self.bm25 = BM25([u.index_text for u in units], variant=bm25_variant)
        self.bm25_index_s = time.perf_counter() - t0

    def search(self, query: str, top: int = 50) -> tuple[list[int], list[float]]:
        bi, _ = self.bm25.search([query], self.depth)
        di, _ = Encoder.search(self.dense.encode_queries([query]), self.emb, self.depth)
        cand = list(rrf([bi[0], di[0]], k=self.rrf_k, top=self.depth)[:self.pool])
        sc = np.asarray(self.ce.predict([(query, self.units[j].text) for j in cand], batch_size=64, show_progress_bar=False),
                        dtype=float).reshape(len(cand), -1)[:, -1]
        order = np.argsort(-sc)[:top]
        return [cand[k] for k in order], [float(sc[k]) for k in order]

    def search_many(self, queries: list[str], top: int = 50) -> list[tuple[list[int], list[float]]]:
        """``search`` for several queries with one BM25 call, one query-encoder batch and one cross-encoder call
        (latency lever L3, EXP-08-003). Same candidates and scoring as ``search``."""
        if not queries:
            return []
        bi, _ = self.bm25.search(queries, self.depth)
        di, _ = Encoder.search(self.dense.encode_queries(queries), self.emb, self.depth)
        cands = [list(rrf([bi[i], di[i]], k=self.rrf_k, top=self.depth)[:self.pool]) for i in range(len(queries))]
        pairs = [(q, self.units[j].text) for q, c in zip(queries, cands) for j in c]
        sc = np.asarray(self.ce.predict(pairs, batch_size=64, show_progress_bar=False), dtype=float).reshape(len(pairs), -1)[:, -1]
        out, s = [], 0
        for c in cands:
            part = sc[s:s + len(c)]; s += len(c)
            order = np.argsort(-part)[:top]
            out.append(([c[k] for k in order], [float(part[k]) for k in order]))
        return out


def unit_embeddings(units: list[Unit], rows: list[int], dense: Encoder, chunk_emb_file: Path, cache_file: Path) -> torch.Tensor:
    """Embeddings for the mixed unit index: reuse cached chunk embeddings for non-PubMed units; encode whole
    abstracts (title, abstract) with the article encoder (batch ≤ 16 to stay inside the GPU budget, ADR-0012).
    The abstract part is cached in ``cache_file``."""
    if cache_file.exists():
        abs_emb = torch.from_numpy(np.load(cache_file))
    else:
        idx = [i for i, r in enumerate(rows) if r < 0]
        old_bs, dense.batch_size = dense.batch_size, min(dense.batch_size, 16)
        abs_emb = dense.encode_docs([units[i].text for i in idx], [units[i].title for i in idx]).cpu()
        dense.batch_size = old_bs
        np.save(cache_file, abs_emb.numpy())
    chunk_emb = np.load(chunk_emb_file, mmap_mode="r")
    keep = np.array([r for r in rows if r >= 0])
    other = torch.from_numpy(np.ascontiguousarray(chunk_emb[keep]))
    out = torch.empty((len(units), abs_emb.shape[1]), dtype=torch.float16)
    a = torch.tensor([i for i, r in enumerate(rows) if r < 0]); o = torch.tensor([i for i, r in enumerate(rows) if r >= 0])
    out[a] = abs_emb.to(torch.float16); out[o] = other.to(torch.float16)
    return out


def scifact_primary_corpus(sf_raw: bytes, documents_jsonl: Path) -> tuple[list[str], list[str], list[str], int]:
    """The NB05 primary benchmark corpus in its exact order (so cached NB05 embeddings line up): SciFact abstracts,
    then CliniVer PubMed abstracts whose normalized title does not match a SciFact title.
    Returns (doc_ids, titles, abstracts, n_scifact)."""
    import re as _re
    from cliniver.ingestion.scifact import corpus_titles, parse_release
    from cliniver.processing.long_docs import iter_documents
    sf_corpus, _ = parse_release(sf_raw); sf_titles = corpus_titles(sf_raw)
    norm = lambda t: _re.sub(r"\W+", " ", t.lower()).strip()
    keys = {norm(t) for t in sf_titles.values()}
    ids = list(sf_corpus); titles = [sf_titles[d] for d in ids]
    abstracts = [sf_corpus[d][len(sf_titles[d]) + 2:] if sf_corpus[d].startswith(sf_titles[d]) else sf_corpus[d] for d in ids]
    n_sf = len(ids)
    for d in iter_documents(documents_jsonl, {"pubmed"}):
        if norm(d["title"]) in keys:
            continue
        ids.append(f"pm:{d['source_id']}"); titles.append(d["title"]); abstracts.append(d["text"])
    return ids, titles, abstracts, n_sf
