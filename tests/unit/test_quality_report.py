from datetime import datetime, timezone

import pytest

from cliniver.data.quality import jaccard, minhash_signatures, near_duplicate_clusters, shingles
from cliniver.data.report import evidence_stratum, quality_flags
from cliniver.ingestion.openfda import is_combination
from cliniver.schemas import Document

BASE = ("metformin is the preferred initial pharmacologic agent for the treatment of type 2 diabetes "
        "and is contraindicated in patients with severe renal impairment because of the risk of lactic "
        "acidosis which is rare but serious and the dose should be reduced when egfr falls below 45")


def test_minhash_deterministic_and_estimates_jaccard():
    other = BASE.replace("rare but serious", "uncommon but important")
    s1 = minhash_signatures([BASE, other], seed=1)
    s2 = minhash_signatures([BASE, other], seed=1)
    assert (s1 == s2).all()  # same seed → identical signatures (reproducible)
    est = float((s1[0] == s1[1]).mean())
    true = jaccard(shingles(BASE), shingles(other))
    assert abs(est - true) < 0.15


def test_near_duplicate_clusters_finds_only_true_duplicates():
    near = BASE + " in adults"
    unrelated = "apixaban reduced the risk of stroke or systemic embolism compared with warfarin in atrial fibrillation " * 2
    pairs, clusters = near_duplicate_clusters(["a", "b", "c"], [BASE, near, unrelated], threshold=0.8)
    assert clusters == [["a", "b"]] or clusters == [["b", "a"]]
    assert set(pairs.columns) == {"doc_a", "doc_b", "jaccard"} and (pairs.jaccard >= 0.8).all()


def test_near_duplicate_handles_empty_text():
    pairs, clusters = near_duplicate_clusters(["a", "b"], ["", ""], threshold=0.8)
    # Empty texts share the "max" signature but exact Jaccard of empty sets is 0 → not duplicates
    assert clusters == [] and pairs.empty


def test_evidence_stratum_priority():
    assert evidence_stratum(["Review", "Randomized Controlled Trial"]) == "rct"  # design beats review tag
    assert evidence_stratum(["Meta-Analysis", "Systematic Review"]) == "meta_analysis"
    assert evidence_stratum(["Journal Article"]) == "other"


@pytest.mark.parametrize("name,expected", [
    ("METFORMIN HYDROCHLORIDE", False), ("SITAGLIPTIN AND METFORMIN HYDROCHLORIDE", True),
    ("sacubitril/valsartan", True), ("INSULIN GLARGINE", False), (None, False)])
def test_is_combination(name, expected):
    assert is_combination(name) is expected


def test_quality_flags_counts_retracted():
    now = datetime.now(timezone.utc)
    mk = lambda i, retracted: Document(doc_id=f"pubmed:{i}", source="pubmed", source_id=str(i), role="evidence",
                                       title="t", text="word " * 60, retrieved_at=now, snapshot="s",
                                       language="eng", extra={"is_retracted": retracted})
    q = quality_flags([mk(1, True), mk(2, False)])
    assert q["pubmed_retracted"] == 1 and q["pubmed_abstract_lt_50_words"] == 0


def test_dense_duplicate_group_is_fast_and_single_cluster():
    """Regression (2026-11-20): ~555 near-identical drug labels made exact verification O(n²) (> 1 h).
    Skipping edges inside an existing component must make a 400-document clique cheap."""
    import time
    base = " ".join(f"w{i}" for i in range(3000))
    texts = [base + f" variant{i}" for i in range(400)]
    t0 = time.perf_counter()
    pairs, clusters = near_duplicate_clusters([str(i) for i in range(400)], texts, threshold=0.8)
    assert len(clusters) == 1 and len(clusters[0]) == 400
    assert pairs.attrs["n_exact_checks"] < 2000  # vs 79,800 pairs without the shortcut
    assert time.perf_counter() - t0 < 60


def test_clusters_equal_bruteforce_components():
    """The shortcut must not change the clusters: compare with brute-force connected components."""
    import random
    from itertools import combinations as comb
    rng = random.Random(3)
    vocab = [f"t{i}" for i in range(60)]
    seeds = [" ".join(rng.choices(vocab, k=80)) for _ in range(6)]
    texts, ids = [], []
    for s_i, s in enumerate(seeds):
        for v in range(5):
            words = s.split()
            for _ in range(rng.randint(0, 6)):
                words[rng.randrange(len(words))] = rng.choice(vocab)
            texts.append(" ".join(words)); ids.append(f"{s_i}-{v}")
    _, clusters = near_duplicate_clusters(ids, texts, threshold=0.6, bands=64)
    parent = list(range(len(ids)))
    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x
    for i, j in comb(range(len(ids)), 2):
        if jaccard(shingles(texts[i]), shingles(texts[j])) >= 0.6:
            parent[find(i)] = find(j)
    groups = {}
    for i in range(len(ids)):
        groups.setdefault(find(i), set()).add(ids[i])
    brute = sorted(sorted(g) for g in groups.values() if len(g) > 1)
    assert sorted(sorted(c) for c in clusters) == brute
