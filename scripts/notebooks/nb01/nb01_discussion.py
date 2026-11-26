DISCUSSION_CELLS = [
r"""
## 9. Discussion
All numbers below are taken from the outputs above, which are written to `artifacts/experiments/EXP-01-001/`.

### 9.1 Population-level findings (count queries; these generalize)
| Condition | PubMed articles (MeSH `/drug therapy`, human, English, with abstract, 1990–) | Completed drug trials (ClinicalTrials.gov) |
|---|---|---|
| Type 2 diabetes | 40,779 | 4,177 |
| Hypertension | 40,306 | 2,971 |
| Heart failure | 18,325 | 1,129 |
| AF / anticoagulation | 15,744 | 537 |
| Dyslipidaemia | 15,107 | 1,360 |

- **Size.** The per-condition sum (130,261) is an *upper bound* on unique articles, because conditions overlap. The full domain literature is small enough to acquire **completely**, so no sampling is needed (projection in §7: ~5.5 min of PubMed requests at the configured rate, ~318 MB of JSONL).
- **Evidence types** (shares are upper bounds because strata overlap): narrative reviews ≤ 25.8%, RCTs ≤ 17.9%, meta-analyses ≤ 4.8%, systematic reviews ≤ 4.0%, other clinical trials ≤ 4.2%, case reports ≤ 4.1%, observational ≤ 3.6%, guidelines ≤ 0.3%. High-tier designs (MA, SR, RCT) make up a substantial minority, which supports a `source_tier` confidence signal (NB07).
- **Time.** Type 2 diabetes literature is still growing (1,520 articles in the 1990s vs. 17,922 in 2020–2026). Hypertension peaked in 2010–2019 (13,480). Evidence recency therefore varies by topic, which supports per-evidence dating (ADR-0005).
- **Trial registry.** Only 27 of 97 sampled completed trials (28%) had posted results. Most registry records describe design, not outcomes, so they are kept as `trial_registry`, not as efficacy evidence.
- Europe PMC OA hit counts come from broader free-text queries and are **not comparable** with the MeSH-restricted PubMed counts.

### 9.2 Parser and schema validation (the purpose of the Stage A sample)
- All five document sources and both normalization sources produced valid records in the unified schema: 494/494 PubMed IDs parsed; 0 openFDA name misses; 0 Europe PMC full-text failures (after retries).
- **Two parser defects were found and fixed** (with regression tests):
  1. 10/494 (2.0%) PubMed records had their English abstract only in `<OtherAbstract Language="eng">`, which left their text empty. These are now recovered (`pubmed_empty_abstract = 0`).
  2. sacubitril, marketed only in combination, received no drug class because of the combination filter. It now falls back to combination classes, flagged with `via_combination`.
- **Truncation check.** Abstracts are not truncated. A known long RCT abstract (EMPA-REG OUTCOME, PMID 26378978) parses to all 297 words in 4 structured sections, matching NCBI's own text format.

### 9.3 The Best Match sample is not representative
Sample vs. population: RCTs 0.6% vs ≤ 17.9%; narrative reviews 48.6% vs ≤ 25.8%. The median sample abstract is 67 words, and 168/494 have fewer than 50 words. Spot checks show news items and short narrative reviews ranked highly. **No statistic from the PubMed sample should be generalized.** Stage B should acquire the full defined population (decision D-NB01-1).

### 9.4 Source-specific issues that need Stage B decisions
- **WHO GHO:** 3,444 statements (85% of Stage A documents) from just 4 indicators × 3 sex categories × years 1980–2022 × global/regional locations. Pooled with the other sources, they would dominate retrieval by count alone.
- **openFDA:** 15 of 30 same-drug label pairs are near-duplicates (word 5-gram Jaccard ≥ 0.8), mostly different manufacturers' labels for the same generic. 6 labels were combination products. Label dates are *revision* dates (all 2026), not dates of the underlying evidence.
- **PubTator 3:**
  - Coverage: 478/494 documents (96.8%) have annotations, and 16 PMIDs are missing (7 of them from 2024, consistent with processing lag). Disease annotations: 90% of documents. Chemical: 57%. Both: 52%.
  - Drug mentions: an exact-name match for an in-scope drug appears in only 101/494 documents. This is a lower bound (salts, classes and abbreviations are missed), but it shows that **normalization quality will limit the graph channel**.
  - Relation labels are coarse statistical types (Negative_Correlation 333 of 567, Positive_Correlation 106, Association 96). They are not "treats/causes" semantics, so **graph edges must never be read as evidence of an effect or its direction** (similarity ≠ verification).
- **Normalization:** RxNorm 49/49, RxClass 49/49 (1 via combination), PubChem 46/49. The three misses (dulaglutide, evolocumab, alirocumab) are biologics, which PubChem's compound database is not designed to hold. RxNorm should therefore be the normalization backbone, with PubChem as a supplementary synonym source for small molecules.
- **Europe PMC OA:** all 10 sampled full texts are from 2026 (ordering bias). Median length 4,299 words. 19 tables are not part of the extracted text, which NB02 must account for.
- **ClinicalTrials.gov:** 47/97 records have no detailed description, only a brief summary.

### 9.5 Acquisition engineering
- **Parallel acquisition is 4.15× faster.** The raw-archive span was 189 s, vs 784 s for the earlier sequential implementation, with the same 408 raw files. The remaining bottleneck is Europe PMC's full-text endpoint (60 s read timeouts, 3 retries); every other job finished in ≤ 44 s.
- **Rate-limit incident.** The first parallel run received **14 HTTP 429 responses from NCBI** at 2.5 req/s with 4 requests in flight. NCBI counts by arrival time, and network jitter bunched arrivals. All 14 succeeded on retry, so no data was lost, but the provider's limit had been exceeded. The fix (2 req/s, ≤ 2 in flight) was validated live in **EXP-01-002**: 70 fresh requests under the same two-job contention, **0 rejections**.
- **Reproducibility.** A replay of the snapshot makes **0 live requests**: 411 archive hits, including 3 archived "not found" answers, which were added after a replay exposed that 404s were not being archived.
""",
r"""
## 10. Limitations
- The Stage A PubMed sample is unrepresentative (by design: it exists to validate parsers). Only the count-query results generalize.
- Population counts depend on MeSH indexing. Very recent articles (indexed automatically and later) may be under-counted. The `/drug therapy` subheading can miss pharmacology papers indexed only under the drug name.
- Publication-type shares are upper bounds because strata overlap.
- Some provider terms (openFDA keyless per-minute limit, ClinicalTrials.gov pacing) were verified from secondary sources because the official pages could not be retrieved programmatically. We use conservative rates.
- Europe PMC timing depends on server latency on the day of acquisition.
- Near-duplicate detection is pairwise (quadratic). That is adequate for 30 labels but not for Stage B (needs MinHash/LSH).
""",
r"""
## 11. Conclusion and decisions

**Conclusion (Stage A).** The multi-source acquisition pipeline is validated:
- every source parses into the unified schema;
- the two parser defects found are fixed and covered by tests;
- acquisition is parallel and within every provider's limits;
- the raw archive reproduces the snapshot exactly.

The domain literature is small enough (≤ 130k PubMed articles) to acquire in full.

**Proposed decisions for Stage B** (to be confirmed by the project owner):

| ID | Decision | Rationale (evidence above) |
|---|---|---|
| D-NB01-1 | PubMed: acquire the **full defined population** (all five condition queries), no sampling | Best Match sample is heavily biased (§9.3). Full acquisition is cheap (§7) |
| D-NB01-2 | PubTator: annotations for all acquired PMIDs; record the missing ones | 96.8% coverage in Stage A |
| D-NB01-3 | openFDA: all 49 drugs, prescription labels; collapse near-duplicate clusters to the most recent label, keeping the others' set-ids as aliases | 15/30 near-duplicate pairs (§9.4) |
| D-NB01-4 | ClinicalTrials.gov: all completed drug trials for the five conditions; keep `has_results`; registry-only records stay `trial_registry` | 28% have posted results |
| D-NB01-5 | Europe PMC OA: a **bounded, year-stratified** long-document subset (~300–500 articles) for the NB02 chunking study, not the full OA set | Long-document study only; 2026-only ordering bias seen |
| D-NB01-6 | WHO GHO: keep all statements, role `statistics`; how (or whether) they enter retrieval is decided in NB05 (e.g., a separate index used only for epidemiological claims) | 85% of Stage A documents from 4 indicators (§9.4) |
| D-NB01-7 | Scale near-duplicate detection with MinHash/LSH | Pairwise approach does not scale |
| D-NB01-8 | RxNorm/RxClass is the normalization backbone; PubChem is supplementary | PubChem 46/49; biologics absent |
""",
r"""
## 12. Next steps
1. Owner review of the Stage A findings and decisions D-NB01-1 … 8.
2. Stage B: scaled acquisition and full-corpus analysis (same sections 6–8, on the full corpus), with MinHash near-duplicate detection.
3. Then NB02 (document processing and chunking) on the long-document subset.
""",
]
