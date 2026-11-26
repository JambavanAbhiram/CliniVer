"""Claim-extraction metrics against physician-written reference statements (NB03, research_foundation §9.3).

* ``coverage``      — share of reference statements entailed by at least one extracted claim (NLI, claim ⇒ statement);
                      also a *set* variant where the premise is all claims joined (statements can be compound).
* ``faithfulness``  — share of extracted claims entailed by the source answer (answer ⇒ claim);
                      1 − faithfulness is the hallucinated-claim rate of the extractor.
* ``qualifier_preservation`` — share of rule-detected qualifiers in the answer that survive in the claims
                      (heuristic, per type; see ``qualifier_kept``).
* atomicity proxies — claims per answer sentence, words per claim.

The NLI judge is itself a model; its agreement with manual reading is spot-checked in the notebook.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

_NUM = re.compile(r"\d+(?:[.,]\d+)?")
_WORD = re.compile(r"[A-Za-z][A-Za-z\-]+")
_NEG = re.compile(r"\b(?:no|not|without|never|cannot|can't|don't|doesn't|isn't|aren't|wasn't|weren't|neither|nor|"
                  r"absence|lack|avoid|contraindicated|non)\b", re.I)
_STOP = set("the a an of in on for to and or with by is are was were be been as at from that this these those it its "
            "patients patient people who which may can".split())


@dataclass
class NLIJudge:
    """Cross-encoder NLI (premise, hypothesis) → P(entailment)."""
    model_name: str
    revision: str
    device: str = "cuda"
    batch_size: int = 32
    max_length: int = 512

    def __post_init__(self) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(self.model_name, revision=self.revision)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, revision=self.revision).to(self.device).eval()
        labels = {v.lower(): int(k) for k, v in self.model.config.id2label.items()}
        self.ent = next(i for lab, i in labels.items() if lab.startswith("entail"))

    def entail_prob(self, premises: list[str], hypotheses: list[str]) -> np.ndarray:
        out = []
        with self.torch.inference_mode():
            for i in range(0, len(premises), self.batch_size):
                enc = self.tok(premises[i:i + self.batch_size], hypotheses[i:i + self.batch_size], truncation="only_first",
                               max_length=self.max_length, padding=True, return_tensors="pt").to(self.device)
                p = self.torch.softmax(self.model(**enc).logits.float(), dim=-1)[:, self.ent]
                out.append(p.cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)


def content_words(t: str) -> set[str]:
    return {w.lower() for w in _WORD.findall(t) if w.lower() not in _STOP and len(w) > 2}


def qualifier_kept(kind: str, span: str, claims: list[str]) -> bool:
    """Heuristic survival test of one source qualifier in the extracted claims.
    quantity/threshold/effect: all numbers of the span appear in one claim;
    negation: one claim has a negation cue and shares ≥ 1 content word of the scope (after the cue);
    population: one claim contains ≥ 50% of the span's content words."""
    if kind in ("quantity", "threshold", "effect"):
        nums = _NUM.findall(span)
        return any(all(n in c for n in nums) for c in claims) if nums else True
    if kind == "negation":
        scope = content_words(_NEG.sub(" ", span))
        return any(_NEG.search(c) and (not scope or scope & content_words(c)) for c in claims)
    words = content_words(span)
    return any(len(words & content_words(c)) >= 0.5 * len(words) for c in claims) if words else True


def qualifier_kept_strict(kind: str, span: str, claims: list[str]) -> bool:
    """Stricter, POST-HOC survival test (added after EXP-03-001 showed a ceiling effect of ``qualifier_kept``;
    reported as exploratory, never as the pre-registered H2 metric).
    quantity/threshold/effect: numbers AND the span's other words (units, comparators) in the SAME claim;
    negation: one claim has a negation cue AND ≥ 60% of the scope's content words;
    population: one claim contains ≥ 80% of the span's content words."""
    low = [c.lower() for c in claims]
    if kind in ("quantity", "threshold", "effect"):
        nums = _NUM.findall(span)
        words = [w.lower() for w in _WORD.findall(span)]
        return any(all(n in c for n in nums) and all(w in c for w in words) for c in low)
    if kind == "negation":
        scope = content_words(_NEG.sub(" ", span))
        return any(_NEG.search(c) and (not scope or len(scope & content_words(c)) >= 0.6 * len(scope)) for c in claims)
    words = content_words(span)
    return any(len(words & content_words(c)) >= 0.8 * len(words) for c in claims) if words else True
