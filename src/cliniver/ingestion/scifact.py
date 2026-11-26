"""SciFact (Wadden et al., 2020): expert-written scientific claims with abstract-level evidence labels.

Official release (allenai/scifact, ``data.tar.gz``): claims and evidence annotations CC BY 4.0; corpus
abstracts ODC-By 1.0 (from S2ORC). Test-split labels were never released, so only train/dev claims with
evidence are usable. A document is *relevant* to a claim if it carries evidence for it (SUPPORT or CONTRADICT:
both are what a verifier must find).
"""

from __future__ import annotations

import io
import re
import json
import tarfile
from dataclasses import dataclass


@dataclass(frozen=True)
class SciFactClaim:
    claim_id: str
    split: str
    text: str
    relevant: frozenset[str]
    labels: tuple[str, ...]


def parse_release(raw: bytes) -> tuple[dict[str, str], list[SciFactClaim]]:
    """Returns ({doc_id: "title. abstract"}, claims with ≥ 1 evidence document)."""
    tf = tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz")

    def read(suffix: str) -> list[dict]:
        name = next(n for n in tf.getnames() if n.endswith(suffix) and "/._" not in n and "cross_validation" not in n)
        return [json.loads(l) for l in tf.extractfile(name).read().decode("utf-8").splitlines() if l.strip()]

    corpus = {str(d["doc_id"]): f"{d['title'].strip()}. {' '.join(s.strip() for s in d['abstract'])}" for d in read("corpus.jsonl")}
    claims = []
    for split in ("train", "dev"):
        for c in read(f"claims_{split}.jsonl"):
            ev = c.get("evidence") or {}
            if not ev:
                continue
            labels = tuple(sorted({s["label"] for d in ev.values() for s in d}))
            claims.append(SciFactClaim(f"scifact:{c['id']}", split, c["claim"].strip(), frozenset(str(d) for d in ev), labels))
    return corpus, claims


_TITLE_JUNK = re.compile(r'["\[\]\(\)\{\}:;]|\b(?:AND|OR|NOT)\b')


def title_query(title: str) -> str:
    """PubMed exact-title query; characters that break the esearch syntax are replaced by spaces."""
    t = " ".join(_TITLE_JUNK.sub(" ", title.rstrip(".")).split())
    return f'"{t}"[ti]'


def map_pmids(pubmed_client, titles: dict[str, str], workers: int = 2) -> dict[str, str]:
    """doc_id → PMID by exact-title search (archived per document). Only unambiguous single hits are kept;
    callers should additionally verify the mapping (e.g. against PubTator's title)."""
    from cliniver.ingestion.http import parallel_map
    ids = list(titles)
    res = parallel_map(lambda d: pubmed_client.search_ids(title_query(titles[d]), 3, f"scifact_title_{d}"), ids, workers)
    return {d: r[0] for d, r in zip(ids, res) if len(r) == 1}


def corpus_titles(raw: bytes) -> dict[str, str]:
    tf = tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz")
    name = next(n for n in tf.getnames() if n.endswith("corpus.jsonl") and "/._" not in n)
    return {str(d["doc_id"]): d["title"].strip() for d in
            (json.loads(l) for l in tf.extractfile(name).read().decode("utf-8").splitlines() if l.strip())}
