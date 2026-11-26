import re

import numpy as np
import pandas as pd
import pytest

from cliniver.evaluation.span_retrieval import Gold, cluster_bootstrap_diff, evaluate
from cliniver.processing.chunking import ChunkConfig, TokenIndex, chunk_spans, make_chunks
from cliniver.processing.normalize import normalize
from cliniver.processing.qualifiers import detect, split_mask
from cliniver.processing.segment import SectionSpan, detect_headings, locate_sections, sentences


class WsTok:
    """Whitespace 'tokenizer' with offsets (keeps tests independent of model downloads)."""
    def __call__(self, text, **_):
        return {"offset_mapping": [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]}


TOK = WsTok()


# ---------------------------------------------------------------- normalization
def test_normalize_keeps_clinical_symbols_and_numbers():
    raw = "HbA1c ≥ 7.0 %, neutrophils 10⁹/L,​ no­effect;  eGFR <30 mL/min/1.73 m²\r\n\r\n\r\nﬁnal"
    n = normalize(raw)
    for keep in ("≥ 7.0 %", "10⁹/L", "<30 mL/min/1.73 m²", "no"):
        assert keep in n.text
    assert " " not in n.text and "​" not in n.text and "  " not in n.text
    assert "\n\n\n" not in n.text and "\r" not in n.text and "final" in n.text
    assert len(n.src) == len(n.text) + 1 and n.src[-1] == len(raw)


def test_normalize_map_span_roundtrip():
    raw = "Metformin   is   contraindicated   in severe renal impairment."
    i = raw.index("contraindicated")
    n = normalize(raw)
    a, b = n.map_span(i, i + len("contraindicated"))
    assert n.text[a:b] == "contraindicated"


# ---------------------------------------------------------------- segmentation
def test_sentences_merge_lowercase_fragment():
    t = "No effect was seen in pts. with eGFR < 30. A second sentence follows."
    spans = sentences(t)
    assert [t[a:b] for a, b in spans] == ["No effect was seen in pts. with eGFR < 30.", "A second sentence follows."]


def test_locate_sections_and_headings():
    t = "Intro text here.\n\nMethods\n\nWe enrolled 100 adults.\n\nResults\n\nHbA1c fell."
    secs = detect_headings(t)
    assert [s.name for s in secs] == ["", "Methods", "Results"]
    loc = locate_sections("A: x\nB: y", [("A", "A: x"), ("B", "B: y"), ("C", "missing")])
    assert [(s.name, s.start, s.end) for s in loc] == [("A", 0, 4), ("B", 5, 9)]


# ---------------------------------------------------------------- chunking
DOC = ("Background\n\n" + " ".join(f"w{i}." for i in range(40)) + "\n\nMethods\n\n"
       + "Para one is here. It has two sentences.\nPara two follows here.\n\nResults\n\n"
       + " ".join(f"r{i}" for i in range(30)) + ".")


@pytest.mark.parametrize("cfg", [ChunkConfig("fixed", 16, 0.25), ChunkConfig("sentence", 16), ChunkConfig("paragraph", 16),
                                 ChunkConfig("recursive", 16), ChunkConfig("section", 16)])
def test_chunks_respect_size_and_cover_text(cfg):
    tok = TokenIndex(DOC, TOK)
    spans = chunk_spans(DOC, cfg, tok, detect_headings(DOC))
    assert all(tok.count(a, b) <= 16 for a, b, _ in spans)
    covered = np.zeros(len(DOC), bool)
    for a, b, _ in spans:
        covered[a:b] = True
    assert all(covered[i] or DOC[i].isspace() for i in range(len(DOC)))  # no non-space text lost


def test_section_chunks_never_cross_sections():
    tok = TokenIndex(DOC, TOK)
    secs = detect_headings(DOC)
    for a, b, name in chunk_spans(DOC, ChunkConfig("section", 64), tok, secs):
        assert any(s.start <= a and b <= s.end for s in secs)
    rec = chunk_spans(DOC, ChunkConfig("recursive", 64), tok, secs)
    assert any(not any(s.start <= a and b <= s.end for s in secs) for a, b, _ in rec)  # recursive may cross


def test_fixed_overlap_and_header():
    tok = TokenIndex(DOC, TOK)
    sp = chunk_spans(DOC, ChunkConfig("fixed", 20, 0.25), tok)
    assert sp[1][0] < sp[0][1]  # consecutive windows overlap
    ch = make_chunks("d", DOC, ChunkConfig("section", 64, header=True), tok, title="T", sections=detect_headings(DOC))
    assert ch[-1].index_text.startswith("T / Results\n") and ch[-1].text == DOC[ch[-1].start:ch[-1].end]


def test_semantic_chunking_uses_breakpoints():
    t = "Cats purr. Cats meow. Dogs bark. Dogs howl."
    def embed(xs):
        v = np.array([[1.0, 0.0] if "Cats" in x else [0.0, 1.0] for x in xs]); return v
    sp = chunk_spans(t, ChunkConfig("semantic", 100, pct=50), TokenIndex(t, TOK), embed=embed)
    assert [t[a:b] for a, b, _ in sp] == ["Cats purr. Cats meow.", "Dogs bark. Dogs howl."]


# ---------------------------------------------------------------- qualifiers
def test_qualifier_detection_examples():
    t = ("Metformin 500 mg twice daily was not associated with lactic acidosis in patients with eGFR ≥ 45 mL/min/1.73 m2, "
         "HR 0.86 (95% CI 0.74–0.99).")
    kinds = {(q.kind, t[q.start:q.end]) for q in detect(t)}
    assert ("quantity", "500 mg twice daily") in kinds and ("quantity", "95%") not in kinds
    assert ("threshold", "≥ 45 mL/min/1.73 m2") in kinds
    # the decimal point in "1.73" must not end the negation scope
    assert ("negation", "was not associated with lactic acidosis in patients with eGFR ≥ 45 mL/min/1.73 m2") in kinds
    assert any(k == "effect" and s.startswith("HR 0.86") and s.endswith(")") for k, s in kinds)
    assert any(k == "population" and s.startswith("in patients with eGFR") for k, s in kinds)


def test_split_mask():
    chunks = [(0, 10), (8, 20), (20, 30)]
    assert split_mask([(2, 9), (9, 19), (15, 25), (21, 29)], chunks) == [False, False, True, False]


# ---------------------------------------------------------------- span retrieval metrics
def test_span_metrics_and_budget():
    golds = [Gold("q1", "d1", 12, 18)]
    doc = ["d1", "d1", "d2"]
    start, end, ntok = np.array([0, 10, 0]), np.array([10, 20, 50]), np.array([300, 300, 600])
    ranked = np.array([[0, 1, 2]])
    r = evaluate(golds, ranked, doc, start, end, ntok, ks=(1, 2), budgets=(512, 1024)).iloc[0]
    assert not r["hit@1"] and r["hit@2"] and r["rr"] == 0.5 and r["gold_in_one_chunk"]
    assert not r["hit@512t"] and r["hit@1024t"]  # second chunk only fits within the larger budget
    g2 = [Gold("q2", "d1", 5, 15)]  # straddles chunks 0 and 1
    r2 = evaluate(g2, ranked, doc, start, end, ntok, ks=(2,), budgets=(1024,)).iloc[0]
    assert not r2["gold_in_one_chunk"] and r2["cover@2"] == 1.0 and not r2["hit@2"]


def test_cluster_bootstrap_detects_difference():
    a = pd.Series([1, 1, 1, 0] * 50); b = pd.Series([0, 0, 1, 0] * 50); c = pd.Series(np.repeat(np.arange(50), 4))
    res = cluster_bootstrap_diff(a, b, c, n_boot=500)
    assert res["diff"] == 0.5 and res["ci_low"] > 0 and res["n_clusters"] == 50


def test_normalize_fast_path_equals_slow_path():
    from cliniver.processing import normalize as nm
    for t in ["clean text.\nNext line.", "a\n b", "a  b", "a \nb", "a\n\n\nb", " lead", "x\u00a0y"]:
        fast = nm.normalize(t)
        nm_re, nm._NEEDS_WORK = nm._NEEDS_WORK, __import__("re").compile(".")  # force the slow path
        try:
            slow = nm.normalize(t)
        finally:
            nm._NEEDS_WORK = nm_re
        assert fast.text == slow.text, t


def test_qualifier_rule_revisions():
    t = ("Stroke or systemic embolism were 0.96 (95% CI 0.71-1.28) and HR of 1.05 (95% CI) (0.81, 1.36) in 70% "
         "of patients (n = 12, p = 0.04) with or without a history of asthma.")
    found = {(q.kind, t[q.start:q.end]) for q in detect(t)}
    assert not any(k == "effect" for k, _ in found)  # "or" is not OR; a CI without numbers is not an effect estimate
    assert ("threshold", "p = 0.04") in found and not any(s.startswith("= 12") for _, s in found)
    assert not any(k == "negation" and s.startswith("without") for k, s in found)
    assert not any(k == "population" for k, _ in found)  # "of patients" is not a population cue


def test_align_mentions_tracks_drift():
    from cliniver.processing.corpus_chunks import align_mentions
    ours = "To assess drugs. Since losartan and MK954 were developed, hypertension care changed."
    # source text had an extra "AIM: " (5 chars) label before the same content
    src = "AIM: " + ours
    ments = [(src.index(w), len(w), w, w) for w in ("losartan", "MK954", "hypertension")]
    out, unal = align_mentions(ours, ments)
    assert unal == 0 and [ours[a:b] for a, b, _ in out] == ["losartan", "MK954", "hypertension"]
    out, unal = align_mentions(ours, [(10, 7, "absent!", None)])
    assert out == [] and unal == 1


def test_drug_matcher_whole_words():
    from cliniver.processing.corpus_chunks import DrugMatcher
    m = DrugMatcher([{"source": "rxnorm", "kind": "drug", "concept_id": "rxnorm:6809", "name": "metformin",
                      "synonyms": ["Glucophage", "Janumet"]}])
    hits = m.find("Metformin (Glucophage) and metformin-like; not metforminX. JANUMET XR.")
    assert [h[3] for h in hits] == ["Metformin", "Glucophage", "JANUMET"]


def test_truncated_mask_counts_cut_copies_under_overlap():
    from cliniver.processing.qualifiers import truncated_mask
    no_overlap = [(0, 10), (10, 20)]
    assert truncated_mask([(2, 8), (8, 12)], no_overlap) == split_mask([(2, 8), (8, 12)], no_overlap) == [False, True]
    overlap = [(0, 10), (7, 17)]  # span (8, 9) is inside window 1 but cut by window 2's start? no: (7,17) contains it
    assert split_mask([(5, 9)], overlap) == [False]       # contained in window 1
    assert truncated_mask([(5, 9)], overlap) == [True]    # but window 2 holds a cut copy
    assert truncated_mask([(8, 9)], overlap) == [False]   # inside both windows: never cut
