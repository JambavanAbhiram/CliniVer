
# ---- Stage B discussion (written from EXP-01-003 outputs; appended by build_nb01.py after stage_b_section.py)
md(r"""
## B6. Discussion (Stage B)
All figures are copied from the outputs above (`artifacts/experiments/EXP-01-003/results.json`). Where a count refers to a set other than the final corpus, that set is named.

### B6.1 Completeness of acquisition (requested vs received)
| Source | Requested / available | Received | Notes |
|---|---|---|---|
| PubMed | 5 condition populations | **125,022 unique PMIDs, 125,022 parsed (0 missing)** | Per-condition unique counts equal the Stage A population counts **exactly** for all 5 conditions. Largest yearly slice 3,386 < esearch cap 9,999, so no slice was truncated |
| PubTator 3 | 125,022 PMIDs | 124,814 (99.8%) | 208 PMIDs not available from PubTator; recorded |
| openFDA | 49 drugs, Rx labels | 4,514 fetched → **4,114 unique** | All 49 drugs have ≥ 1 Rx label; none hit the 1,000-label cap. 400 fetches were the same combination label returned for a second component drug (merged) |
| ClinicalTrials.gov | completed drug trials, 5 conditions | **9,665 unique** | API totals reconciled per condition (4,177 / 2,971 / 1,360 / 537 / 1,129); 509 trials registered under > 1 condition (merged) |
| Europe PMC OA | 4 year strata × ~100 | **398 full texts** | 99–100 per stratum |
| WHO GHO | Stage A statements | 3,444 | Re-used from the Stage A fetch of the same snapshot day |

**Final corpus (snapshot 2026-11-20): 138,764 documents** (PubMed 124,918; ClinicalTrials.gov 9,665; WHO 3,444; Europe PMC 398; openFDA 339), plus 190 normalization concepts, 3,020,745 PubTator entity mentions and 386,627 PubTator relations.

### B6.2 Exclusions and duplicates
- **Retractions:** 104 retracted PubMed articles excluded (IDs listed in `exclusions_and_aliases.json`). A further **2,330** articles carry a correction or expression-of-concern link; they are kept but flagged, so a verdict citing them can say so.
- **PubMed near-duplicates:** 261 abstracts in 125 clusters (0.2% of the 123,583 abstracts with ≥ 50 words). The inspected top pairs are the same article indexed twice under two PMIDs (e.g. journal-supplement articles re-indexed in a later year). They are **annotated** (`near_dup_group`), not removed (ADR-0008), so NB05 can collapse them at ranking time.
- **openFDA near-duplicates:** the 4,114 unique labels collapse to **339** (200 clusters, largest 207 labels of one drug). About 92% of unique labels are near-identical copies (repackagers and generic manufacturers). The representative is the most recent label; the other set-ids are kept as aliases. Clustering is done **within** one drug (and combination flag) only, so no cluster mixes drugs.
- **Cross-condition overlap** is real and expected: 130,261 condition memberships vs 125,022 unique PMIDs.

### B6.3 What evidence the corpus contains
| Stratum (exclusive, highest tier first) | Corpus docs | Corpus share | Population upper bound (Stage A) |
|---|---|---|---|
| Meta-analysis | 5,418 | 4.3% | 4.8% |
| Systematic review | 1,730 | 1.4% | 4.0% |
| RCT | 22,339 | 17.9% | 17.9% |
| Other clinical trial | 5,306 | 4.2% | 4.2% |
| Observational | 4,526 | 3.6% | 3.6% |
| Guideline | 306 | 0.2% | 0.3% |
| Review | 29,096 | 23.3% | 25.8% |
| Case report | 4,576 | 3.7% | 4.1% |
| Other | 51,621 | 41.3% | — (no population query) |

- The full-population corpus reproduces the population evidence mix (RCT 17.9% vs 17.9%), unlike the Stage A Best Match sample (RCT 0.6%). This confirms D-NB01-1.
- Shares below the upper bound are expected, because the strata are exclusive. For example, an article tagged both systematic review and meta-analysis is counted once, as a meta-analysis, which is why the systematic-review share (1.4%) is well below its upper bound (4.0%).
- **Guidelines are scarce (306).** Claims phrased as recommendations ("first-line", "recommended") will often lack guideline-level support in this corpus. Copyrighted society guidelines are excluded (D2), so INSUFFICIENT EVIDENCE is the correct outcome there, not a failure.

### B6.4 Quality and structure
- **PubMed:** 51% structured abstracts; median 251 words; 1,207 abstracts have < 50 words (1.0%); 34 English abstracts recovered from `OtherAbstract`; 0 empty. DOI present for 93.1%; day-precision dates for 62.7% (the rest are month or year only, so evidence dates must carry their precision).
- **openFDA** (final corpus: median 7,078 words, 13 sections): long, sectioned documents. Section-aware chunking is mandatory in NB02. Among the 4,114 unique labels:
  - 678 are combination products.
  - 48 lack at least one key section.
  - 22 have no named sections and use the `spl_unclassified_section` fallback.
  - 1 label (SOAANZ, torsemide) has **no clinical text in openFDA at all**: only product data and package-panel image captions. This is a gap in the source, not a parser defect, and NB02 drops empty documents explicitly.
- **ClinicalTrials.gov:**
  - 3,300 trials (34.1%) have posted results.
  - 5,036 (52.1%) have no detailed description, so registry text is short (median 97 words) and mostly describes design, not outcomes.
  - 1,135 trials cite a result publication; for **797 (70.2%)** that PMID is in the PubMed corpus. This is a usable trial→publication link.
- **Europe PMC OA:** median 4,169 words and 8 sections, balanced across strata (99–100 each). There are **1,057 tables that are not in the extracted text**: NB02 must measure that loss or handle tables separately. 132 of 398 articles are also in the PubMed corpus (abstract and full text of the same study, useful for the abstract-only limitation analysis).
- **PubTator coverage** (over the 124,918 corpus PubMed documents):
  - any entity 99.3%, Disease 96.6%, Chemical 77.7%, Chemical and Disease 75.6%; 81% of chemical mentions are normalized to an identifier.
  - Exact in-scope drug-name mentions: 38,557 (30.9%), a lower bound.
  - Relations are dominated by coarse statistical types: Negative_Correlation 214,721, Association 84,905, Positive_Correlation 60,324. Drug_Interaction has only 59 relations, so **PubTator cannot serve as an interaction source**, and graph edges remain retrieval signals, never evidence (ADR-0003).

### B6.5 Error analysis: defects found and fixed during Stage B
Each defect was traced to its root cause, fixed in `src/cliniver/`, and (except for item 3, in the runner script) covered by a regression test (50 unit tests pass).
1. **Hung PubMed connections:** a server trickling bytes never trips a per-read timeout. Fix: a total request deadline enforced by a watchdog. The first version deadlocked on Windows (`Response.close` blocks), so the watchdog shuts the socket down instead.
2. **Quadratic near-duplicate verification:** near-identical openFDA labels formed cliques (~530k exact comparisons, > 1 h). Fix: skip candidate pairs already in the same component. This gives identical clusters (tested against brute force), and openFDA dedup now takes about 98 s.
3. **Runner masked a timeout:** the notebook runner reported success after a cell timeout. Fix: any exception or error output sets status 1.
4. **Per-condition counts double-counted** articles that `[dp]` places in two yearly slices. Fix: unique counts per condition, which now equal the population counts exactly.
5. **Combination labels counted under each component drug** (400 duplicates). Fix: merge by label ID before deduplication.
6. **Empty documents:**
   - SPL labels without named sections (now use `spl_unclassified_section`).
   - JATS articles with paragraphs directly under `<body>` (5 previously empty; now 0).

### B6.6 Acquisition cost (live run, 2026-11-20)
- The live run took 2,902 s wall time. The critical path was Europe PMC full text (2,902 s, 117 retries against slow responses). PubMed efetch took 665 s and PubTator 478 s, with **0 retries** for both (the EXP-01-002 pacing held).
- A replay from the raw archive makes **0 network requests**: about 2 min for the archive and about 5.5 min for the whole notebook.
""")

md(r"""
## B7. Limitations (Stage B)
- **Indexing lag:** recent articles (2025–2026) are under-represented, because MeSH indexing lags publication.
- **Abstracts, not full text:** PubMed abstracts can omit qualifiers. Full text covers only the 398-article Europe PMC subset.
- **openFDA:**
  - US labels only. Label dates are revision dates, not evidence dates.
  - Collapsing near-duplicates keeps one representative, so small differences between aliased labels (e.g. strengths, inactive ingredients) are not searchable.
- **Registry records** describe design more than outcomes.
- **WHO statistics** describe populations and cannot verify individual-level claims.
- **One snapshot day:** server-side changes after 2026-11-20 are not reflected. Every verdict must state the snapshot date (ADR-0005).
- **Evidence strata** come from PubMed publication types, which are assigned with a lag and can be incomplete. "Other" (41.3%) is not further classified.
- **Missing annotations:** 208 PubMed documents (0.2%) have no PubTator annotations, so they are invisible to the graph channel but not to BM25 or dense retrieval.
""")

md(r"""
## B8. Conclusion and decision
**Conclusion:**
- The corpus for snapshot **2026-11-20** is complete with respect to its defined populations: every requested PubMed record was received, and the per-condition counts match the population counts exactly. It is deduplicated, with retractions excluded and corrections flagged, and fully reproducible from the raw archive.
- It reproduces the population evidence mix, which the Stage A sample did not.
- Its main structural risks for later notebooks are long sectioned labels, tables missing from full text, scarce guidelines, and coarse PubTator relations.

**Decision (proposed, pending owner approval):** freeze `data/processed/2026-11-20/` as the input to NB02–NB08.

**Next steps (NB02, document processing and chunking):**
1. Compare chunking strategies:
   - by structured-abstract section for PubMed;
   - by SPL section for openFDA;
   - long-document strategies on the Europe PMC subset.
2. Drop empty-text documents explicitly.
3. Measure what is lost with tables.
""")
