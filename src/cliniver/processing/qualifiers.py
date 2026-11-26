"""Rule-based detection of *qualifier spans*: the parts of a clinical statement whose loss changes its
meaning (research_foundation §9.2). A chunking strategy that cuts such a span in two can make a
retrieved chunk appear to support a claim that the full sentence does not support.

Types
* ``negation``   — negation cue + its scope up to the end of the clause (max 12 words)
* ``quantity``   — number (or range) + unit, with an optional frequency ("500 mg twice daily", "18 years",
                   "12.5%"). The "95%" of a "95% CI" label is excluded
* ``threshold``  — comparator + number (+ unit): "≥ 65 years", "eGFR < 30 mL/min/1.73 m2", "less than 5%";
                   and p-values ("p = 0.04", "P<0.001"). A bare "=" is not a comparator ("n = 12" is a count)
* ``effect``     — an effect estimate with its confidence interval: "HR 0.86 (95% CI 0.74–0.99)". Abbreviations
                   are case-sensitive (the conjunction "or" is not an odds ratio) and the CI must contain a number
* ``population`` — "in/among/for patients with …" up to the end of the clause (max 12 words)

The rules are deliberately transparent and conservative. Their precision is measured on a manually
inspected sample in NB02 (not a clinician annotation; see the notebook's limitations). A first inspection
round led to the revisions noted above; precision is reported on a second, fresh sample.
"""

from __future__ import annotations

import bisect
import re
from dataclasses import dataclass

_NUM = r"\d+(?:[.,]\d+)?(?:\s?(?:-|–|—|to)\s?\d+(?:[.,]\d+)?)?"
_UNIT = (r"(?:mg/kg(?:/day)?|mg/dL|mg/dl|mmol/L|mmol/mol|µmol/L|μmol/L|mL/min(?:/1\.73\s?m(?:2|²))?|mEq/L|mmHg|mm\s?Hg"
         r"|mcg|µg|μg|ng/mL|mg|g|kg|mL|ml|L|IU|units?|U|%|kg/m(?:2|²)|years?|yrs?|months?|weeks?|days?|hours?|h)")
_FREQ = (r"(?:\s?(?:once|twice|three times|four times)\s(?:a\s)?(?:daily|day|weekly|week)|\s?(?:daily|weekly|b\.?i\.?d\.?"
         r"|t\.?i\.?d\.?|q\.?d\.?|per day|/day|/d|a day))?")
_CMP = r"(?:≥|≤|>=|<=|<|>|less than|greater than|more than|fewer than|at least|at most|above|below|under|over|up to)"
# "." and "," end a clause only when not inside a number ("1.73 m2", "1,000 patients")
_CLAUSE_END = re.compile(r"[;:()\[\]]|[.,](?!\d)|\s(?:but|whereas|while|although|however)\s")

_RULES: dict[str, re.Pattern] = {
    "quantity": re.compile(rf"(?i:(?<![\w.]){_NUM}\s?{_UNIT}(?![\w/]){_FREQ})(?!\s?(?:CI\b|[Cc]onfidence))"),
    "threshold": re.compile(rf"(?<!\w){_CMP}\s?{_NUM}(?:\s?{_UNIT}(?![\w/]))?|\b[pP]\s?(?:<|>|=|≤|≥)\s?0?\.\d+", re.I),
    "effect": re.compile(r"(?:\b(?:HR|RR|OR|aHR|aOR|IRR|MD)\b|(?i:\b(?:hazard ratio|odds ratio|relative risk|risk ratio|"
                         r"rate ratio|mean difference)\b))[^.;()]{0,30}?\d+(?:\.\d+)?[^.;()]{0,10}?[(\[]\s?95\s?%\s?"
                         r"(?:CI|[Cc]onfidence [Ii]nterval)[^)\]]{0,60}?\d[^)\]]{0,40}[)\]]"),
}
_NEG_CUE = re.compile(r"\b(?:no|not|without|neither|nor|never|absence of|lack of|lacked|failed to|did not|does not|"
                      r"do not|was not|were not|is not|are not|cannot|unable to|non-?significant(?:ly)?|"
                      r"contraindicated)\b", re.I)
_POP_CUE = re.compile(r"\b(?:in|among|for)\s(?:the\s)?(?:patients|adults|subjects|participants|individuals|women|men|"
                      r"children|people|persons|those|elderly|older adults|infants|neonates|pregnant women)\b", re.I)


@dataclass(frozen=True)
class Qualifier:
    kind: str
    start: int
    end: int


def _scoped(text: str, cue: re.Match, max_words: int = 12) -> tuple[int, int]:
    a, pos, words = cue.start(), cue.end(), 0
    m = _CLAUSE_END.search(text, pos)
    limit = m.start() if m else len(text)
    nl = text.find("\n", pos)
    if 0 <= nl < limit:
        limit = nl
    for w in re.finditer(r"\S+", text[pos:limit]):
        words += 1
        if words > max_words:
            return a, pos + w.start() - 1
    return a, limit


def detect(text: str) -> list[Qualifier]:
    out: list[Qualifier] = []
    for kind, rx in _RULES.items():
        out += [Qualifier(kind, m.start(), m.end()) for m in rx.finditer(text)]
    for kind, rx in (("negation", _NEG_CUE), ("population", _POP_CUE)):
        for m in rx.finditer(text):
            if kind == "negation" and text[max(0, m.start() - 8):m.start()].lower().endswith("with or "):
                continue  # "with or without X" is a disjunction, not a negation
            a, b = _scoped(text, m)
            while b > a and text[b - 1].isspace():
                b -= 1
            if b > m.end():  # a cue with an empty scope ("no." at a clause end) is not a qualifier
                out.append(Qualifier(kind, a, b))
    return sorted(set(out), key=lambda q: (q.start, q.end, q.kind))


def split_mask(spans: list[tuple[int, int]], chunks: list[tuple[int, int]]) -> list[bool]:
    """True for each span that no single chunk contains completely (i.e. the chunking cut it)."""
    ch = sorted(chunks)
    starts = [a for a, _ in ch]
    # running max of chunk ends over chunks sorted by start: does any chunk starting ≤ s reach e?
    best_end, run = [], -1
    for _, b in ch:
        run = max(run, b); best_end.append(run)
    out = []
    for s, e in spans:
        i = bisect.bisect_right(starts, s) - 1
        out.append(not (i >= 0 and best_end[i] >= e))
    return out


def truncated_mask(spans: list[tuple[int, int]], chunks: list[tuple[int, int]]) -> list[bool]:
    """True for each span that appears CUT in at least one chunk (a chunk overlaps the span without
    containing it). Without overlapping chunks this equals ``split_mask``. With overlapping windows a span
    can be contained in one window and cut in its neighbour; a retriever may return the neighbour, so the
    reader sees a truncated qualifier. This is the retrieval-unit view of qualifier damage."""
    ch = sorted(chunks)
    starts = [a for a, _ in ch]
    out = []
    for s, e in spans:
        hi = bisect.bisect_left(starts, e)  # chunks starting before the span ends
        cut = False
        for a, b in ch[max(0, hi - 64):hi]:  # chunks are short; 64 predecessors covers any overlap pattern used
            if b > s and (a > s or b < e):
                cut = True
                break
        out.append(cut)
    return out
