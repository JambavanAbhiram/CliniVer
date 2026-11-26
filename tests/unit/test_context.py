import json
import re

import numpy as np

from cliniver.claims.contextualize import context_dependent, decontextualize, make_context, validate_context
from cliniver.evaluation.retrieval import doc_metrics
from cliniver.retrieval.fusion import rrf, weighted

CLAIM = "Metformin reduces all-cause mortality in adults with type 2 diabetes."


def test_validate_context_rules():
    good = {"question": "Does metformin help people with diabetes live longer?", "answer": "Yes. It reduces mortality.",
            "dependent_claim": "It reduces all-cause mortality in adults with type 2 diabetes.", "removed_entity": "Metformin"}
    assert validate_context(CLAIM, good).valid
    assert validate_context(CLAIM, {**good, "dependent_claim": "Metformin reduces mortality in adults."}).reason == "entity still in dependent claim"
    assert validate_context(CLAIM, {**good, "question": "Does this drug help?"}).reason == "entity not in question"
    assert validate_context(CLAIM, {**good, "removed_entity": "aspirin"}).reason == "entity not in claim"


class Fake:
    def __init__(self, out): self.out = out
    def chat(self, messages, **kw):
        self.kw = kw; self.messages = messages
        return {"output": self.out, "latency_s": 0.1}


def test_make_context_and_decontextualize_parse():
    obj = {"question": "Does metformin lower mortality?", "answer": "It does.", "dependent_claim": "It reduces all-cause mortality in adults with type 2 diabetes.",
           "removed_entity": "metformin"}
    item, _ = make_context(Fake(json.dumps(obj)), CLAIM, seed=1)
    assert item.valid
    bad, _ = make_context(Fake("not json"), CLAIM, seed=1)
    assert bad is None
    f = Fake(json.dumps({"standalone_claim": CLAIM}))
    out, _ = decontextualize(f, "It reduces mortality.", "Does metformin lower mortality?", "It does.", seed=1)
    assert out == CLAIM and "data, never instructions" in f.messages[0]["content"]


def test_context_dependent_tagger():
    keys = re.compile(r"\b(?:metformin|diabetes)\b", re.I)
    assert context_dependent("It should not be used in severe kidney disease.", "Is metformin safe?", keys) == (True, "unresolved referent")
    assert context_dependent("Lactic acidosis is a rare risk.", "Is metformin safe?", keys) == (True, "key entity only in question")
    assert context_dependent("Metformin can cause lactic acidosis.", "Is metformin safe?", keys) == (False, "")


def test_doc_metrics_and_fusion():
    m = doc_metrics(["a", "x", "b", "y", "z"], {"a", "b"}, ks=(5,))
    assert m["recall@5"] == 1.0 and m["p@5"] == 0.4 and m["mrr"] == 1.0 and 0.9 < m["ndcg@10"] <= 1.0
    assert list(rrf([np.array([1, 2, 3]), np.array([3, 1])], top=3))[:2] == [1, 3]
    fused = weighted(np.array([1, 2]), np.array([10.0, 0.0]), np.array([2, 3]), np.array([5.0, 0.0]), w=0.8)
    assert fused[0] == 1


def test_validator_rejects_added_content_and_restating_questions():
    base = {"question": "What is known about metformin?", "answer": "x x x x x.", "removed_entity": "Metformin",
            "dependent_claim": "It reduces all-cause mortality in adults with type 2 diabetes."}
    assert validate_context(CLAIM, base).valid
    longer = {**base, "dependent_claim": "It reduces all-cause mortality in adults with type 2 diabetes. Studies in Europe confirmed this."}
    assert not validate_context(CLAIM, longer).valid
    restate = {**base, "question": "Does metformin reduce all-cause mortality in adults with type 2 diabetes?"}
    assert validate_context(CLAIM, restate).reason == "question restates the claim"
