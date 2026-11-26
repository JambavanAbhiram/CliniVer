"""Loading the long-document subset of the frozen corpus for NB02 (Europe PMC OA full text, openFDA labels,
and ClinicalTrials.gov records above a token threshold), normalized, with section spans."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from cliniver.processing.normalize import normalize
from cliniver.processing.segment import SectionSpan, locate_sections


@dataclass
class LongDoc:
    doc_id: str
    source: str
    title: str
    text: str  # normalized
    sections: list[SectionSpan]
    n_sections_declared: int


def iter_documents(path: Path, sources: set[str] | None = None):
    """Stream raw JSON dicts from documents.jsonl (fast pre-filter on the source field)."""
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if sources is not None and not any(f'"source":"{s}"' in line[:400] for s in sources):
                continue
            yield json.loads(line)


def to_long_doc(d: dict) -> LongDoc:
    n = normalize(d["text"])
    secs = [(s["name"], normalize(s["text"]).text) for s in d.get("sections", [])]
    if d["source"] == "openfda_label":
        # label text is "name: section text" per line; locate the body, then widen to include the "name: " prefix
        located = locate_sections(n.text, secs)
        located = [SectionSpan(s.name, max(0, s.start - len(s.name) - 2), s.end)
                   if n.text[max(0, s.start - len(s.name) - 2):s.start] == f"{s.name}: " else s for s in located]
    else:
        located = locate_sections(n.text, secs)
    return LongDoc(d["doc_id"], d["source"], d.get("title", ""), n.text, located, len(secs))
