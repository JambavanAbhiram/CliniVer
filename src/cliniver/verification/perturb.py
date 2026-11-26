"""Labels-by-construction perturbations of SUPPORTED claim–evidence pairs (research_foundation §10.2). SYNTHETIC.

Rule-based only (no LLM), so every label follows from the edit:
* ``negation``   — insert/remove a negation in the main verb phrase              → CON
* ``number``     — multiply the first number by 3 (or add 10 to percentages)   → CON
* ``hedge``      — remove a hedge ("may", "might", "can", "could", "possibly") → PART (claim more certain than evidence)
* ``population`` — delete a population qualifier ("in patients with …")         → PART (claim broader than evidence)
* ``entity``     — replace the claim's subject with an unrelated drug           → INS
* ``paraphrase`` — light meaning-preserving edit (contractions, synonyms)        → SUP (control)

A perturbation is produced only when its rule applies; otherwise the pair is skipped for that type (counts reported).
"""

from __future__ import annotations

import re

_NEG_IN = [(re.compile(r"\b(is|are|was|were)\b(?! not)", re.I), r"\1 not"),
           (re.compile(r"\b(increases|reduces|decreases|improves|causes|prevents|lowers|raises|inhibits|induces|affects|"
                       r"promotes|predicts|protects|enhances|impairs|correlates|associates)\b", re.I), None)]
_NEG_OUT = [(re.compile(r"\b(is|are|was|were|does|do|did|can|could|has|have) not\b", re.I), r"\1"),
            (re.compile(r"\bno\s+(?=\w)", re.I), "")]
_VERB_BASE = {"increases": "increase", "reduces": "reduce", "decreases": "decrease", "improves": "improve", "causes": "cause",
              "prevents": "prevent", "lowers": "lower", "raises": "raise", "inhibits": "inhibit", "induces": "induce",
              "affects": "affect", "promotes": "promote", "predicts": "predict", "protects": "protect", "enhances": "enhance",
              "impairs": "impair", "correlates": "correlate", "associates": "associate"}
_HEDGE = re.compile(r"\b(may|might|can|could|possibly|potentially|likely)\s+", re.I)
_POP = re.compile(r"\s*\b(?:in|among|for)\s+(?:the\s+)?(?:patients|adults|subjects|participants|individuals|women|men|children|"
                  r"people|mice|rats|elderly|older adults|infants|smokers|non-smokers)\b[^,.;]*", re.I)
_NUM = re.compile(r"(?<![A-Za-z\d.])(\d+(?:\.\d+)?)(\s?%)?(?![A-Za-z\d])")  # a standalone number, not part of "HbA1c" or "T2D"
_UNRELATED = ["warfarin", "levothyroxine", "omeprazole", "salbutamol", "sertraline", "amoxicillin"]


def negate(c: str) -> str | None:
    for rx, repl in _NEG_OUT:
        if rx.search(c):
            return rx.sub(repl, c, count=1)
    m = _NEG_IN[0][0].search(c)
    if m:
        return c[:m.end()] + " not" + c[m.end():]
    m = _NEG_IN[1][0].search(c)
    if m:
        v = m.group(0)
        return c[:m.start()] + "does not " + _VERB_BASE[v.lower()] + c[m.end():]
    return None


def change_number(c: str) -> str | None:
    m = _NUM.search(c)
    if not m:
        return None
    x = float(m.group(1)); pct = m.group(2)
    y = x + 10 if pct and x <= 90 else x * 3
    s = f"{y:g}"
    return c[:m.start()] + s + (pct or "") + c[m.end():]


def remove_hedge(c: str) -> str | None:
    m = _HEDGE.search(c)
    if not m:
        return None
    rest = c[m.end():]
    return c[:m.start()] + rest


def widen_population(c: str) -> str | None:
    m = _POP.search(c)
    if not m:
        return None
    out = (c[:m.start()] + c[m.end():]).strip()
    return out if len(out.split()) >= 4 else None


def swap_entity(c: str, entity: str | None, k: int = 0) -> str | None:
    if not entity or entity.lower() not in c.lower():
        return None
    repl = next(u for u in _UNRELATED[k % len(_UNRELATED):] + _UNRELATED if u not in c.lower())
    return re.sub(re.escape(entity), repl, c, count=1, flags=re.I)


_PARA = [(r"\bdo not\b", "don't"), (r"\bdoes not\b", "doesn't"), (r"\bis not\b", "isn't"), (r"\bincreases\b", "raises"),
         (r"\breduces\b", "lowers"), (r"\bpatients\b", "people"), (r"\bassociated with\b", "linked to")]


def paraphrase(c: str) -> str | None:
    for a, b in _PARA:
        if re.search(a, c, re.I):
            return re.sub(a, b, c, count=1, flags=re.I)
    return None


PERTURBATIONS = {"negation": (negate, "CON"), "number": (change_number, "CON"), "hedge": (remove_hedge, "PART"),
                 "population": (widen_population, "PART"), "paraphrase": (paraphrase, "SUP")}
