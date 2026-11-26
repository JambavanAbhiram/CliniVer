"""Answer-generation helpers: query expansion, fusion with document diversity, drug-naming check."""

from types import SimpleNamespace

from cliniver.generation.answer import NON_ANSWER, Strategy, Vocabulary, fuse, search_queries

VOC = Vocabulary(condition_terms={"hypertension": ["hypertension", "high blood pressure"]}, condition_names={"hypertension": "high blood pressure"},
                 drugs={"hypertension": ["lisinopril", "amlodipine"]}, classes={"hypertension": ["Angiotensin Converting Enzyme Inhibitor"]},
                 clinical_names={"hypertension": "Hypertension"})


def test_conditions_and_concept_queries():
    q = "What is the medicine for high blood pressure?"
    assert VOC.conditions_in(q) == ["hypertension"]
    s3 = Strategy("G3", "planned_concepts", 8, True, "v2")
    qs = search_queries(s3, q, {"intent": "treatment_choice", "conditions": [], "queries": ["antihypertensive drugs", q]}, VOC)
    assert qs[0] == q and "antihypertensive drugs" in qs and "Angiotensin Converting Enzyme Inhibitor for Hypertension" in qs
    assert len(qs) == len(set(x.lower() for x in qs))                      # no duplicate queries
    assert search_queries(Strategy("G0", "question", 5, False, "v1"), q, None, VOC) == [q]


def test_fuse_keeps_one_passage_per_document():
    units = [SimpleNamespace(doc_id=d) for d in ["a", "a", "b", "c"]]
    res = [([0, 1, 2], [0.9, 0.8, 0.7]), ([1, 3], [0.95, 0.6])]
    out = fuse(res, units, k=3, diverse=True)
    assert [units[j].doc_id for j, _ in out] == ["a", "c", "b"]           # RRF: c (rank 2 in list 2) beats b (rank 3 in list 1)
    assert len(fuse(res, units, k=4, diverse=False)) == 4


def test_drug_naming_and_non_answer():
    assert VOC.names_drug_for("ACE inhibitors such as lisinopril lower blood pressure [1].", "hypertension")
    assert not VOC.names_drug_for("Blood pressure targets are rarely achieved [4].", "hypertension")
    assert NON_ANSWER.search("However, the specific medications are not detailed in the provided passages.")


def test_routed_strategy_picks_by_intent():
    from cliniver.generation.answer import Routed
    variants = {"G0": {"retrieval": "question", "k": 5, "prompt": "v1"}, "G3": {"retrieval": "planned_concepts", "k": 8, "diverse": True, "prompt": "v2"}}
    r = Routed.from_cfg("H1", {"treatment_choice": "G3", "default": "G0"}, variants)
    assert r.pick({"intent": "treatment_choice"}).name == "G3"
    assert r.pick({"intent": "dosing"}).name == "G0" and r.pick(None).name == "G0"
    rec = Routed.from_cfg("H1", {"treatment_choice": {"name": "G3", **variants["G3"]}, "default": {"name": "G0", **variants["G0"]}}, {})
    assert rec.pick({"intent": "treatment_choice"}).k == 8                 # adoption records carry full configs
