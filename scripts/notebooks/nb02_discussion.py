"""Discussion cells for NB02, written AFTER execution from the outputs of EXP-02-001 (results.json and tables)."""

DISCUSSION_CELLS = [
r"""
## 11. Discussion
All figures below are copied from the outputs above (`artifacts/experiments/EXP-02-001/`).

### 11.1 Where chunking matters in our corpus (§3.1)
| Source | Docs | Median tokens | > 128 tokens | > 512 tokens |
|---|---|---|---|---|
| PubMed | 124,918 | 375 | 96.0% | 12.9% |
| ClinicalTrials.gov | 9,665 | 142 | 52.6% | 13.7% |
| WHO GHO | 3,444 | 45 | 0% | 0% |
| Europe PMC OA | 398 | 6,287 | 100% | 100% |
| openFDA labels | 339 | 11,263 | 93.2% | 93.2% |

- The long documents (labels, full text) are where chunking decides what a retrieved unit can say.
- 12.9% of PubMed abstracts exceed the dense model's 512-token input limit, so even "index abstracts whole" is not lossless for a 512-token encoder.

### 11.2 Retrieval (E1, COVID-QA)
- **Smaller is better, for every strategy and both retrievers.** On dev, every 128-token configuration beats its 256- and 512-token versions on `hit@1024t`. With a fixed reading budget, small chunks fill it with more distinct candidate passages.
- **Selected configuration: `paragraph-128`** (dev mean `hit@1024t` 0.679; BM25 0.720, dense 0.637). It is closely followed by `recursive-128` (0.676), `section-128` (0.675), `semantic-128-p90` (0.661) and `sentence-128` (0.661). The best fixed configuration is `fixed-128-o25` (0.650), and the common default `fixed-256-o0` scores 0.548.
- **Test split, selected vs the default baseline `fixed-256-o0`:** BM25 0.643 vs 0.625, dense **0.638 vs 0.493** (descriptive; the pre-registered test is against the best fixed configuration, below).
- **H1 retrieval part, test split, `paragraph-128` vs `fixed-128-o25`:**

| Retriever | Metric | Paragraph-128 | Fixed-128-o25 | Difference [95% CI] |
|---|---|---|---|---|
| BM25 | hit@1024t | 0.643 | 0.655 | −0.012 [−0.053, 0.021] |
| Dense | hit@1024t | 0.638 | 0.609 | +0.029 [−0.028, 0.072] |
| BM25 | hit@5 | 0.556 | 0.610 | **−0.054 [−0.093, −0.022]** |
| Dense | hit@5 | 0.562 | 0.542 | +0.020 [−0.035, 0.067] |

  At the budget level neither difference is significant, so **the retrieval part of H1 is not supported**. The overlapping fixed windows match the structure-aware chunks, but they pay with a **33% larger index** (1.20 M vs 0.90 M indexed tokens). At a fixed top-5, BM25 is significantly *better* with the overlapping windows: their larger chunks contain more of each answer, and k counts chunks, not tokens.
- **Gold answers kept in one chunk:** 89.6% (`paragraph-128`) vs 95.2% (`fixed-128-o25`) vs 92.4% (`fixed-256-o0`). Overlap keeps answers whole by duplicating text. Paragraph chunks cut answers that cross a 128-token packing boundary.
- **Headers hurt.** "Title / section" prefixes lowered `hit@1024t` (e.g. `section-128+hdr` 0.622 vs `section-128` 0.675 on dev). The cost is mostly with the dense retriever (0.550 vs 0.630), and the header also consumes reading budget.
- **Semantic chunking gives no gain** over paragraph boundaries (0.661 vs 0.679) and costs an embedding pass (≈ 30 s for COVID-QA vs 0.05 s).
- **Normalization (E4)** is neutral for retrieval on COVID-QA (largest difference 0.002). Its purpose is safety and exact offsets, not accuracy: 117 of 884 documents changed (net 424 characters removed: invisible characters and whitespace runs), with **0 violations** of the digits/symbols invariant.

### 11.3 Qualifier preservation (E2, our 736 long documents)
- 182,469 qualifiers detected (quantity 106,018; negation 32,835; threshold 22,519; population 20,881; effect estimate 216).
- **Truncation rate** (a qualifier appears cut in at least one chunk):
  - `paragraph-128` **0.43%** vs `fixed-128-o25` **9.54%**: difference −9.1 points, 95% CI [−9.3, −8.9] over 715 documents.
  - The default `fixed-256-o0`: 1.80%.
  - With overlap, the split rate of `fixed-128-o25` is 0.00% while 9.5% of qualifiers sit cut in a neighbouring window. This is why the truncation rate was made the primary measure before the full run (§1.2).
- Structure-aware strategies at 256/512 tokens truncate even less (0.02–0.14%), but retrieve worse. The selected configuration trades 0.4% qualifier truncation for the best retrieval.
- Residual truncations of `paragraph-128` are concentrated in **population** phrases of full-text articles (1.9%), which are long and run up to clause ends.
- **Rule precision** (E3):
  - Round 2 (fresh sample, revised rules): **149/150 = 99.3%**, Wilson 95% CI [96.3%, 99.9%].
  - Round 1, before revision: 86.7%.
  - Inspected by the AI developer, not a human or clinician. Recall not measured.

### 11.4 Error analysis (selected configuration, dev, 736 questions)
| | BM25 | Dense |
|---|---|---|
| Hit | 530 | 469 |
| Wrong document | 94 | 147 |
| Right document, wrong chunk | 77 | 85 |
| Gold split by chunking | 35 | 35 |

- Most misses are **retrieval** errors (the wrong article, or the right article but the wrong passage), not chunking errors.
- Only 35/736 dev answers (4.8%) are cut by the selected chunking.
- The small general-domain dense model misses the article more often than BM25 (147 vs 94). This is a retriever question for NB05, not a chunking one.

### 11.5 Corpus chunking (§10)
- **592,930 chunks** from 138,763 documents (1 empty document skipped), in 683 s:
  - PubMed 485,220 (3.88 per abstract);
  - openFDA 39,963 (118 per label);
  - Europe PMC 33,557;
  - ClinicalTrials.gov 30,746;
  - WHO 3,444.
- **Entities:**
  - PubTator: 2.71 M abstract mentions aligned to our text at **99.4%**.
  - RxNorm dictionary: 326,250 drug mentions.
  - 975 mentions straddle a chunk boundary and are not attached to either chunk.
""",
r"""
## 12. Limitations
- **Domain gap of the benchmark:** COVID-QA is biomedical full text, but not cardiometabolic pharmacotherapy, and its questions are extractive QA, not claims. It measures document-structure effects; the conclusion must be re-checked on claim-shaped retrieval in NB05 (SciFact, BioASQ).
- **Abstracts were not evaluated directly.** `paragraph-128` splits 96% of PubMed abstracts into ~4 pieces. Whether whole-abstract units retrieve better for claim verification is **untested** here. A piece may lose context from elsewhere in the abstract (e.g. a results sentence without the population in the methods), which the qualifier rules (clause-level) do not measure.
- **Fixed reference retrievers.** Conclusions are conditional on BM25 and `bge-small-en-v1.5`. A stronger or biomedical encoder (NB05) could favour different chunk sizes.
- **Qualifier rules:** precision was checked by the AI developer, not by humans; recall is unknown. Clause-level scopes do not capture cross-sentence dependencies.
- **One protocol change:** the qualifier metric was switched from split rate to truncation rate after a code-check run on 40 documents, before the full experiment. Both are reported.
- **Single run:** the chunkers, BM25 and exact dense search are deterministic, so seeds matter only for the dev/test split (one split, seed 42).
""",
r"""
## 13. Conclusion and decision
**Conclusion:**
- Chunk size matters more than strategy: at a fixed reading budget, 128-token units retrieve best across all strategies and both retrievers.
- Among 128-token configurations, structure-aware chunking (`paragraph-128`) **matches** the best fixed-size chunking in retrieval (no significant difference) with a 25% smaller index (0.90 M vs 1.20 M indexed tokens).
- It **truncates 22× fewer qualifiers** (0.43% vs 9.5%), which matters directly for verification: a retrieved chunk with a cut negation, threshold or population can make a claim look supported.

**H1: PARTIALLY SUPPORTED** (pre-registered rule): the qualifier part is supported; the retrieval part is not.

**Decision (proposed, pending owner approval):**
- Adopt **`paragraph-128`** with conservative normalization as the chunking of the corpus snapshot (`data/processed/2026-11-20/chunks/paragraph-128/`), and no section headers in the indexed text.
- Record this in ADR-0009.

**Next steps:**
1. **NB03:** local LLM runtime, response generation, claim extraction.
2. **NB05** must test the open abstract question: whole-abstract vs `paragraph-128` units on claim-shaped benchmarks.
3. **NB05** must also re-check chunk size with the selected (possibly biomedical) encoder.
""",
]
