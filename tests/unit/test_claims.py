import json

import pytest

from cliniver.claims.extract import (ATOMIC_SCHEMA, QUALIFIED_SCHEMA, extract_atomic, extract_qualified, extract_sentence,
                                     numbered_sentences)
from cliniver.evaluation.extraction import content_words, qualifier_kept
from cliniver.ingestion.qa_datasets import domain_regex, parse_jsonl


class FakeClient:
    def __init__(self, output): self.output, self.calls = output, []
    def chat(self, messages, **kw):
        self.calls.append((messages, kw))
        return {"output": self.output, "latency_s": 0.1, "completion_tokens": 10}


ANSWER = "Metformin is first-line therapy. It should not be used if eGFR is below 30 mL/min."


def test_sentence_baseline_and_numbering():
    r = extract_sentence(ANSWER)
    assert [c.text for c in r.claims] == ["Metformin is first-line therapy.", "It should not be used if eGFR is below 30 mL/min."]
    numbered, _ = numbered_sentences(ANSWER)
    assert numbered.startswith("[1] Metformin") and "\n[2] It should not" in numbered


def test_atomic_and_qualified_parse_and_schema_passed():
    c = FakeClient(json.dumps({"claims": ["Metformin is first-line therapy."]}))
    r = extract_atomic(c, ANSWER, seed=1)
    assert r.valid_json and r.claims[0].text == "Metformin is first-line therapy."
    assert c.calls[0][1]["json_schema"] is ATOMIC_SCHEMA and "<answer>" in c.calls[0][0][1]["content"]
    q = {"claims": [{"text": "Metformin should not be used if eGFR is below 30 mL/min.", "source_sentence": 2,
                     "claim_type": "contraindication", "drug": "metformin", "condition": None, "population": "eGFR < 30",
                     "negated": True, "dose": None, "frequency": None, "temporal": None, "conditional": "eGFR below 30",
                     "certainty": "definite", "is_verifiable": True}]}
    c2 = FakeClient(json.dumps(q))
    r2 = extract_qualified(c2, ANSWER, seed=1)
    assert r2.valid_json and r2.claims[0].source_sentence == 2 and r2.claims[0].fields["negated"] is True
    assert c2.calls[0][1]["json_schema"] is QUALIFIED_SCHEMA


def test_invalid_json_is_recorded_not_raised():
    r = extract_atomic(FakeClient('{"claims": ["unterminated'), ANSWER, seed=1)
    assert not r.valid_json and r.claims == [] and "JSONDecodeError" in r.error


@pytest.mark.parametrize("kind,span,claims,kept", [
    ("quantity", "500 mg", ["Take 500 mg twice daily."], True),
    ("quantity", "500 mg", ["Take metformin twice daily."], False),
    ("negation", "should not be used if eGFR is below 30", ["Metformin should not be used when eGFR is below 30."], True),
    ("negation", "should not be used if eGFR is below 30", ["Metformin should be used when eGFR is below 30."], False),
    ("population", "in patients with severe kidney disease", ["Avoid in severe kidney disease."], True),
])
def test_qualifier_kept(kind, span, claims, kept):
    assert qualifier_kept(kind, span, claims) is kept


def test_qa_parsing_and_domain_filter():
    raw = (json.dumps({"Question": "Can I take  Eliquis with aspirin?", "Free_form_answer": "Usually not.",
                       "Must_have": "['Apixaban plus aspirin raises bleeding risk']", "Nice_to_have": "[]"}) + "\n").encode()
    items = parse_jsonl(raw, "kqa_golden", human=True)
    assert items[0].question == "Can I take Eliquis with aspirin?" and items[0].must_have == ("Apixaban plus aspirin raises bleeding risk",)
    rx = domain_regex({"eliquis", "apixaban"})
    assert rx.search(items[0].question) and rx.search("my blood sugar is high") and not rx.search("Is Lexapro safe?")
    assert content_words("The patients with kidney disease") == {"kidney", "disease"}


def test_chat_client_replays_from_log(tmp_path, monkeypatch):
    from cliniver.llm import runtime as rt

    class Srv:
        url = "http://x"
    posted = []

    class Resp:
        status_code = 200
        def raise_for_status(self): pass
        def json(self): return {"choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
                                "usage": {"prompt_tokens": 3, "completion_tokens": 1}, "timings": {"predicted_per_second": 50}}
    monkeypatch.setattr(rt.requests, "post", lambda *a, **k: posted.append(k) or Resp())
    log = tmp_path / "calls.jsonl"
    c = rt.ChatClient(Srv(), log, model_key="m", model_file="m.gguf", release="b1")
    msgs = [{"role": "user", "content": "q"}]
    a = c.chat(msgs, seed=1)
    b = rt.ChatClient(None, log, model_key="m", model_file="m.gguf", release="b1").chat(msgs, seed=1)  # replay, no server
    assert a["output"] == b["output"] == "hi" and b["cached"] and len(posted) == 1
    with pytest.raises(LookupError):
        rt.ChatClient(None, log, model_key="m", model_file="m.gguf", release="b1").chat(msgs, seed=2)  # different seed


def test_wordless_segments_are_not_claims():
    r = extract_sentence("Take it with food. . - Call your doctor.")
    assert all(any(ch.isalpha() for ch in c.text) for c in r.claims) and len(r.claims) == 2


def test_strict_qualifier_check_catches_partial_population_and_detached_negation():
    from cliniver.evaluation.extraction import qualifier_kept_strict
    pop = "for children who weigh less than 15 kg and for pregnant or nursing individuals"
    partial = ["Permethrin cream is the preferred treatment for children who weigh less than 15 kg."]
    assert qualifier_kept(pop.split()[0] and "population", pop, partial) and not qualifier_kept_strict("population", pop, partial)
    neg = "do not require treatment with antibiotics"
    detached = ["Acute sinus infections are usually self-limited.", "Treatment with antibiotics is not covered."]
    assert not qualifier_kept_strict("negation", neg, ["Sinus infections require treatment with antibiotics."])
    assert qualifier_kept_strict("negation", neg, ["Acute sinus infections do not require treatment with antibiotics."])
    assert qualifier_kept_strict("quantity", "500 mg", ["Take 500 mg daily."]) and not qualifier_kept_strict("quantity", "500 mg", ["Take 500 daily."])
