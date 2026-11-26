"""Span-based segmentation: every unit is a (start, end) character span into the document text, so
chunks keep exact offsets (needed for gold-span evaluation and for inheriting entity annotations)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

Span = tuple[int, int]

_BLOCK = re.compile(r"\n\s*\n")  # blank line
_LINE = re.compile(r"\n")


def _split(text: str, sep: re.Pattern, start: int = 0, end: int | None = None) -> list[Span]:
    end = len(text) if end is None else end
    spans, pos = [], start
    for m in sep.finditer(text, start, end):
        spans.append((pos, m.start())); pos = m.end()
    spans.append((pos, end))
    return [s for s in (_strip(text, a, b) for a, b in spans) if s[1] > s[0]]


def _strip(text: str, a: int, b: int) -> Span:
    while a < b and text[a].isspace():
        a += 1
    while b > a and text[b - 1].isspace():
        b -= 1
    return a, b


def blocks(text: str, start: int = 0, end: int | None = None) -> list[Span]:
    """Blank-line separated blocks."""
    return _split(text, _BLOCK, start, end)


def paragraphs(text: str, start: int = 0, end: int | None = None) -> list[Span]:
    """Line-separated paragraphs (Europe PMC joins paragraphs with "\\n", sections with "\\n\\n")."""
    return _split(text, _LINE, start, end)


@lru_cache(maxsize=1)
def _segmenter():
    import pysbd
    return pysbd.Segmenter(language="en", clean=False, char_span=True)


def sentences(text: str, start: int = 0, end: int | None = None) -> list[Span]:
    """Sentence spans (pysbd, run per paragraph for speed), with one repair: a 'sentence' that starts
    with a lowercase letter is merged into the previous one. pysbd splits after unknown abbreviations
    (e.g. "pts. with eGFR < 30"), which would otherwise cut a population qualifier in two."""
    out: list[Span] = []
    for pa, pb in paragraphs(text, start, end):
        out.extend(_para_sentences(text, pa, pb))
    return out


@lru_cache(maxsize=500_000)
def _para_sentences(text: str, pa: int, pb: int) -> tuple[Span, ...]:
    """Cached per paragraph: every chunking strategy re-segments the same paragraphs."""
    seg = text[pa:pb]
    spans: list[Span] = []
    for s in _segmenter().segment(seg):
        a, b = _strip(text, pa + s.start, pa + s.end)
        if b <= a:
            continue
        if spans and text[a].islower():
            spans[-1] = (spans[-1][0], b)
        else:
            spans.append((a, b))
    return tuple(spans)


@dataclass(frozen=True)
class SectionSpan:
    name: str
    start: int
    end: int


def locate_sections(text: str, sections: list[tuple[str, str]]) -> list[SectionSpan]:
    """Find each (name, section_text) in ``text`` in order. Sections that cannot be located are skipped
    (the caller reports how many). Text not covered by any located section is not lost: chunkers treat
    the gaps as unnamed sections."""
    out, pos = [], 0
    for name, body in sections:
        body = body.strip()
        if not body:
            continue
        i = text.find(body, pos)
        if i < 0:
            continue
        out.append(SectionSpan(name, i, i + len(body))); pos = i + len(body)
    return out


_HEADING_END = re.compile(r"[.:;?!,]$")


def detect_headings(text: str, max_words: int = 12) -> list[SectionSpan]:
    """Heuristic sections for plain text without markup (used for COVID-QA contexts): a blank-line block
    of ≤ ``max_words`` words that does not end in punctuation is a heading; the section runs to the next
    heading. Text before the first heading is an unnamed section."""
    bl = blocks(text)
    heads = [(a, b) for a, b in bl if len(text[a:b].split()) <= max_words and not _HEADING_END.search(text[a:b])
             and "\n" not in text[a:b]]
    out, prev_start, prev_name = [], 0, ""
    for a, b in heads:
        if a > prev_start:
            out.append(SectionSpan(prev_name, prev_start, a))
        prev_start, prev_name = a, text[a:b]
    out.append(SectionSpan(prev_name, prev_start, len(text)))
    return [s for s in out if s.end > s.start]
