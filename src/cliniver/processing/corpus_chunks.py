"""Apply the selected chunking to the whole frozen corpus and attach entity mentions to chunks (NB02 §9).

Entity sources
* **PubTator 3** (PubMed documents): offsets refer to PubTator's own title + abstract text, which can differ
  from ours (e.g. structured-abstract labels such as "AIM:" are in PubTator's text but not in ours). Mentions
  are therefore *aligned*: each mention's surface text is searched near its expected position, tracking the
  running offset difference. Unaligned mentions are counted, never guessed.
* **RxNorm dictionary** (all documents): case-insensitive whole-word matches of in-scope drug names and
  their RxNorm synonyms (brand names, incl. combination brands). Dictionary matching has no context, so
  e.g. a brand name used as an English word would be a false positive; the dictionary is small and
  domain-specific, which keeps this rare (not measured).

Title mentions: our chunk text excludes the title. Title mentions are attached to every chunk of the
document with ``in_title = True`` (the title describes the whole document).
"""

from __future__ import annotations

import re
from dataclasses import asdict
from typing import Iterable

from cliniver.processing.chunking import ChunkConfig, TokenIndex, make_chunks
from cliniver.processing.long_docs import to_long_doc


class DrugMatcher:
    def __init__(self, concepts: Iterable[dict]) -> None:
        names: dict[str, str] = {}
        for c in concepts:
            if c.get("source") != "rxnorm" or c.get("kind") != "drug":
                continue
            for n in [c["name"], *c.get("synonyms", [])]:
                n = n.strip()
                if len(n) >= 4:  # very short synonyms are ambiguous
                    names.setdefault(n.lower(), c["concept_id"])
        self.names = names
        alt = "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True))
        self.rx = re.compile(rf"(?<![\w-])(?:{alt})(?![\w-])", re.I)

    def find(self, text: str) -> list[tuple[str, int, int, str]]:
        return [(self.names[m.group(0).lower()], m.start(), m.end(), m.group(0)) for m in self.rx.finditer(text)]


def align_mentions(text: str, mentions: list[tuple], window: int = 400) -> tuple[list[tuple], int]:
    """``mentions``: (local_offset, length, surface, payload) sorted by offset, offsets in the source's text.
    Returns ([(start, end, payload)], n_unaligned) with spans in ``text``."""
    out, delta, unaligned = [], 0, 0
    for off, length, surface, payload in sorted(mentions, key=lambda m: m[0]):
        exp = off + delta
        if 0 <= exp and text[exp:exp + length] == surface:
            out.append((exp, exp + length, payload)); continue
        lo = max(0, exp - window)
        best, pos = None, text.find(surface, lo, exp + window + length)
        while pos >= 0:
            if best is None or abs(pos - exp) < abs(best - exp):
                best = pos
            pos = text.find(surface, pos + 1, exp + window + length)
        if best is None:
            unaligned += 1
            continue
        out.append((best, best + length, payload)); delta = best - off
    return out, unaligned


def chunk_document(d: dict, cfg: ChunkConfig, tokenizer, matcher: DrugMatcher, pubtator: list[tuple] | None = None,
                   embed=None):
    """Returns (chunk records, entity records, stats) for one corpus document (raw JSON dict)."""
    ld = to_long_doc(d)  # normalized text + located sections
    tok = TokenIndex(ld.text, tokenizer)
    chunks = make_chunks(d["doc_id"], ld.text, cfg, tok, title=ld.title, sections=ld.sections, embed=embed)
    recs = []
    for c in chunks:
        r = asdict(c); r.pop("meta")
        r["chunk_id"] = f"{c.doc_id}#{c.idx}"; r["source"] = d["source"]
        recs.append(r)
    ents = []
    stats = {"pubtator_mentions": 0, "pubtator_unaligned": 0, "rxnorm_mentions": 0, "mentions_cut_by_chunking": 0}

    def owner(a: int, b: int) -> list[int]:
        return [i for i, c in enumerate(chunks) if c.start <= a and b <= c.end] if chunks else []

    for cid, a, b, surf in matcher.find(ld.text):
        own = owner(a, b)
        stats["mentions_cut_by_chunking"] += not own
        for i in own:
            ents.append({"chunk_id": recs[i]["chunk_id"], "source": "rxnorm_dict", "entity_type": "Chemical",
                         "identifier": cid, "text": surf, "start": a, "end": b, "in_title": False})
            stats["rxnorm_mentions"] += 1
    if pubtator:
        title = d.get("title", "")
        abstract = [(o - len(title) - 1, ln, s, (t, i)) for (p, t, i, s, o, ln) in pubtator if p != "title"]
        aligned, unal = align_mentions(ld.text, abstract)
        stats["pubtator_mentions"] += len(abstract); stats["pubtator_unaligned"] += unal
        for a, b, (etype, ident) in aligned:
            own = owner(a, b)
            stats["mentions_cut_by_chunking"] += not own
            for i in own:
                ents.append({"chunk_id": recs[i]["chunk_id"], "source": "pubtator", "entity_type": etype,
                             "identifier": ident, "text": ld.text[a:b], "start": a, "end": b, "in_title": False})
        for (p, t, i, s, o, ln) in pubtator:
            if p == "title":
                for r in recs:
                    ents.append({"chunk_id": r["chunk_id"], "source": "pubtator", "entity_type": t, "identifier": i,
                                 "text": s, "start": None, "end": None, "in_title": True})
    return recs, ents, stats
