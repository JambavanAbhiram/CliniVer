"""Service layer: configuration parsing, provenance links, and the expert-review store (no models loaded)."""

import json
from types import SimpleNamespace

import pytest

from cliniver.config.service_config import load_service_config
from cliniver.service.engine import parse_config_name, source_link
from cliniver.service.review import ReviewStore


def test_parse_config_name():
    assert parse_config_name("v2|guard=1|tau=0.2|so=llm_label") == {"verifier": "v2", "guard": True, "tau_q": 0.2, "so": "llm_label"}
    assert parse_config_name("v1r|guard=0|tau=None|so=abstain") == {"verifier": "v1r", "guard": False, "tau_q": None, "so": "abstain"}


def test_source_link():
    assert source_link("pubmed:12345") == "https://pubmed.ncbi.nlm.nih.gov/12345/"
    assert source_link("clinicaltrials:NCT01234567") == "https://clinicaltrials.gov/study/NCT01234567"
    assert source_link("openfda_label:abc") is None
    assert source_link("pubmed:not-a-pmid") is None


def test_service_config_is_localhost_only():
    cfg = load_service_config()
    assert cfg.host in ("127.0.0.1", "localhost")


def _write(p, rows):
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


@pytest.fixture()
def store(tmp_path):
    proc = tmp_path / "proc"; (proc / "responses").mkdir(parents=True)
    v2, v1, meta, resp = [], [], [], []
    for i in range(40):
        cid = f"g:kqa_golden:{i}#c0" if i % 2 else f"g:medlfqa:{i}#c0"
        rid = cid.split("#")[0]
        verdict = ["SUP", "CON", "INS", "ABSTAINED"][(i // 2) % 4]
        v2.append({"claim_id": cid, "claim": f"claim {i}", "verified_text": f"claim {i}", "decisive_unit": "u1", "verifier_label": "SUP",
                   "final_verdict": verdict, "overall_confidence": 0.5, "second_opinion": None, "config": "v2|guard=1|tau=None|so=abstain",
                   "corpus_snapshot": "2026-11-20", "evidence_date": None})
        v1.append({"claim_id": cid, "final_verdict": "SUP"})
        meta.append({"claim_id": cid, "response_id": rid})
        resp.append({"response_id": rid, "question": "q", "answer": "a", "generator": "g", "query_source": rid.split(":")[1]})
    _write(proc / "responses" / "v2.jsonl", v2); _write(proc / "responses" / "v1.jsonl", v1)
    _write(proc / "responses" / "claims_contextualized.jsonl", meta); _write(proc / "responses" / "responses.jsonl", resp)
    cfg = SimpleNamespace(sample_size=16, verdicts_file="v2.jsonl", baseline_verdicts_file="v1.jsonl", annotations_file="ann/review.jsonl")
    return ReviewStore(proc, tmp_path, cfg, seed=42, unit_lookup=lambda uid: None)


def test_review_queue_is_stratified_and_seeded(store):
    strata = {(x["verdict"], x["reference"]) for x in store.queue}
    assert len(store.queue) == 16 and len(strata) == 8          # 4 verdicts × 2 reference sources, 2 each


def test_review_annotation_validation_and_append(store):
    cid = store.queue[0]["claim_id"]
    with pytest.raises(ValueError):
        store.add({"reviewer": "", "claim_id": cid, "assessment": "correct"})
    with pytest.raises(ValueError):
        store.add({"reviewer": "r", "claim_id": cid, "assessment": "incorrect"})          # needs the correct label
    with pytest.raises(ValueError):
        store.add({"reviewer": "r", "claim_id": "nope", "assessment": "correct"})
    store.add({"reviewer": "r", "claim_id": cid, "assessment": "incorrect", "correct_label": "INS"})
    store.add({"reviewer": "r", "claim_id": cid, "assessment": "correct"})                  # revision: appended, latest counts
    assert len(store.annotations()) == 2
    assert store.latest("r")[("r", cid)]["assessment"] == "correct"
    assert store.summary()["reviewed_items"] == 1
