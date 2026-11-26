"""Chunking strategies compared in Notebook 02 (RQ1 / H1).

Every strategy returns ``Chunk`` spans into the (normalized) document text, so evaluation can check
whether a gold span or a qualifier span is cut by a chunk boundary, and chunks can inherit entity
annotations by offset. Token counts use the tokenizer of the fixed NB02 retriever.

Strategies
* ``fixed``      — windows of ``size`` tokens with fractional ``overlap``; ignores all structure.
* ``sentence``   — consecutive sentences packed up to ``size`` tokens.
* ``paragraph``  — one chunk per paragraph (line); paragraphs above ``size`` are split recursively.
* ``recursive``  — LangChain-style: blank-line blocks → paragraphs → sentences → tokens, then greedy
                   packing of neighbours up to ``size`` (may cross section boundaries).
* ``section``    — like ``recursive`` but never crosses a section boundary (structure-aware).
* ``semantic``   — sentence embeddings; a boundary where the cosine distance between neighbouring
                   sentences exceeds the doc's ``pct`` percentile; segments above ``size`` split recursively.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from cliniver.processing.segment import SectionSpan, Span, blocks, paragraphs, sentences


@dataclass(frozen=True)
class ChunkConfig:
    strategy: str  # fixed | sentence | paragraph | recursive | section | semantic
    size: int = 256  # max tokens per chunk (target window for "fixed")
    overlap: float = 0.0  # fixed only
    pct: float = 90.0  # semantic only
    header: bool = False  # prefix "title / section" to the text that is indexed (offsets unaffected)

    @property
    def name(self) -> str:
        s = f"{self.strategy}-{self.size}"
        if self.strategy == "fixed":
            s += f"-o{int(round(self.overlap * 100))}"
        if self.strategy == "semantic":
            s += f"-p{int(self.pct)}"
        return s + ("+hdr" if self.header else "")


@dataclass
class Chunk:
    doc_id: str
    idx: int
    start: int
    end: int
    section: str
    n_tokens: int
    text: str = ""  # document text of the span
    index_text: str = ""  # what is indexed (text, optionally with a header)
    meta: dict = field(default_factory=dict)


class TokenIndex:
    """Token character offsets of one document, for O(log n) token counts of any span."""

    def __init__(self, text: str, tokenizer) -> None:
        enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
        offs = [(a, b) for a, b in enc["offset_mapping"] if b > a]
        self.starts = [a for a, _ in offs]
        self.ends = [b for _, b in offs]

    def count(self, a: int, b: int) -> int:
        return max(0, bisect.bisect_right(self.ends, b) - bisect.bisect_left(self.starts, a))

    def windows(self, a: int, b: int, size: int, stride: int) -> list[Span]:
        i0, i1 = bisect.bisect_left(self.starts, a), bisect.bisect_right(self.ends, b)
        if i1 <= i0:
            return [(a, b)] if b > a else []
        out = []
        for i in range(i0, i1, stride):
            j = min(i + size, i1)
            out.append((self.starts[i], self.ends[j - 1]))
            if j == i1:
                break
        return out


def _merge(pieces: list[Span], tok: TokenIndex, size: int) -> list[Span]:
    out: list[Span] = []
    for a, b in pieces:
        if out and tok.count(out[-1][0], b) <= size:
            out[-1] = (out[-1][0], b)
        else:
            out.append((a, b))
    return out


_LEVELS: list[Callable] = [blocks, paragraphs, sentences]


def split_to_fit(text: str, a: int, b: int, tok: TokenIndex, size: int, level: int = 0, merge: bool = True) -> list[Span]:
    """Recursive split of [a, b) into spans of ≤ size tokens, preferring the coarsest boundary."""
    if tok.count(a, b) <= size:
        return [(a, b)]
    if level >= len(_LEVELS):
        return tok.windows(a, b, size, size)
    units = _LEVELS[level](text, a, b)
    if len(units) <= 1:
        ua, ub = units[0] if units else (a, b)
        return split_to_fit(text, ua, ub, tok, size, level + 1, merge)
    pieces: list[Span] = []
    for ua, ub in units:
        pieces += split_to_fit(text, ua, ub, tok, size, level + 1, merge)
    return _merge(pieces, tok, size) if merge else pieces


def _section_of(sections: Sequence[SectionSpan], pos: int) -> str:
    starts = [s.start for s in sections]
    i = bisect.bisect_right(starts, pos) - 1
    return sections[i].name if i >= 0 and pos < sections[i].end else ""


def _gaps_as_sections(text: str, sections: Sequence[SectionSpan]) -> list[SectionSpan]:
    """Located sections plus unnamed sections for any uncovered text, in order."""
    out, pos = [], 0
    for s in sorted(sections, key=lambda s: s.start):
        if s.start > pos and text[pos:s.start].strip():
            out.append(SectionSpan("", pos, s.start))
        out.append(s); pos = max(pos, s.end)
    if pos < len(text) and text[pos:].strip():
        out.append(SectionSpan("", pos, len(text)))
    return out


def chunk_spans(text: str, cfg: ChunkConfig, tok: TokenIndex, sections: Sequence[SectionSpan] = (),
                embed: Callable[[list[str]], np.ndarray] | None = None) -> list[tuple[int, int, str]]:
    n = len(text)
    secs = _gaps_as_sections(text, sections) if sections else [SectionSpan("", 0, n)]
    if cfg.strategy == "fixed":
        stride = max(1, int(round(cfg.size * (1 - cfg.overlap))))
        spans = tok.windows(0, n, cfg.size, stride)
    elif cfg.strategy == "sentence":
        pieces: list[Span] = []
        for a, b in sentences(text):
            pieces += split_to_fit(text, a, b, tok, cfg.size, level=len(_LEVELS))
        spans = _merge(pieces, tok, cfg.size)
    elif cfg.strategy == "paragraph":
        spans = []
        for a, b in paragraphs(text):
            spans += split_to_fit(text, a, b, tok, cfg.size, level=2)
    elif cfg.strategy == "recursive":
        spans = split_to_fit(text, 0, n, tok, cfg.size)
    elif cfg.strategy == "section":
        out = []
        for s in secs:
            out += [(a, b, s.name) for a, b in split_to_fit(text, s.start, s.end, tok, cfg.size)]
        return out
    elif cfg.strategy == "semantic":
        if embed is None:
            raise ValueError("semantic chunking needs an embed function")
        sents = sentences(text)
        if len(sents) <= 1:
            spans = split_to_fit(text, 0, n, tok, cfg.size)
        else:
            e = embed([text[a:b] for a, b in sents])
            dist = 1 - np.sum(e[:-1] * e[1:], axis=1)  # embeddings are L2-normalized
            cut = np.percentile(dist, cfg.pct)
            spans, seg_start = [], sents[0][0]
            for k in range(1, len(sents) + 1):
                if k == len(sents) or dist[k - 1] > cut:
                    seg_end = sents[k - 1][1]
                    spans += split_to_fit(text, seg_start, seg_end, tok, cfg.size, level=2)
                    if k < len(sents):
                        seg_start = sents[k][0]
    else:
        raise ValueError(f"unknown strategy {cfg.strategy!r}")
    return [(a, b, _section_of(secs, a)) for a, b in spans]


def make_chunks(doc_id: str, text: str, cfg: ChunkConfig, tok: TokenIndex, *, title: str = "",
                sections: Sequence[SectionSpan] = (), embed=None) -> list[Chunk]:
    out = []
    for i, (a, b, sec) in enumerate(chunk_spans(text, cfg, tok, sections, embed)):
        body = text[a:b]
        hdr = " / ".join(x for x in (title.strip(), sec.strip()) if x)
        out.append(Chunk(doc_id, i, a, b, sec, tok.count(a, b), body, f"{hdr}\n{body}" if cfg.header and hdr else body))
    return out
