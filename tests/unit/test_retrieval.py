from collections import Counter

import numpy as np

from cliniver.retrieval.graph import ConceptGraph, EntityLinker
from cliniver.retrieval.sparse import BM25, clinical_tokenize


def test_clinical_tokenizer_keeps_numbers_units_and_hyphens():
    toks = clinical_tokenize("SGLT-2 inhibitors at 10 mg/dL reduce HbA1c by 0.5% if eGFR ≥30", {"at", "by", "if"})
    assert {"sglt-2", "10", "mg/dl", "0.5%", "≥30", "hba1c", "egfr"} <= set(toks) and "at" not in toks


def test_bm25_variants_retrieve():
    docs = ["metformin lowers HbA1c by 1%", "SGLT-2 inhibitors reduce heart failure admissions", "warfarin and INR"]
    for v in ("default", "clinical"):
        ids, _ = BM25(docs, variant=v).search(["SGLT-2 heart failure"], 2)
        assert ids[0][0] == 1


def test_ppr_ranks_documents_connected_to_seed():
    g = ConceptGraph({"d1": Counter({"C:metformin": 2}), "d2": Counter({"C:insulin": 1}), "d3": Counter({"C:metformin": 1, "C:t2d": 1})},
                     relations=Counter({("C:t2d", "C:insulin"): 3}))
    s = g.ppr(["C:metformin"])
    assert s[g.docs.index("d1")] > s[g.docs.index("d2")] and s[g.docs.index("d3")] > s[g.docs.index("d2")]
    assert np.all(g.ppr(["C:unknown"]) == 0)


def test_entity_linker_longest_match_and_counts():
    m = [("type 2 diabetes", "D1"), ("Type 2 Diabetes", "D1"), ("diabetes", "D2"), ("diabetes", "D2"), ("T2D", "D1"), ("T2D", "D1"), ("xy", "X")]
    lk = EntityLinker(m)
    assert lk.link("Metformin in type 2 diabetes and T2D") == [("type 2 diabetes", "D1"), ("T2D", "D1")]
    assert "xy" not in lk.surface and lk.surfaces_of["D1"][0] in ("type 2 diabetes", "t2d")
    assert lk.link("no-diabetes-here, but diabetes yes") == [("diabetes", "D2")]  # hyphenated word is one token
