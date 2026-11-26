"""Discussion cells for NB05, written AFTER execution from the outputs of EXP-05-001."""

DISCUSSION_CELLS = [
r"""
## 13. Discussion
All figures are copied from the outputs above (`artifacts/experiments/EXP-05-001/`).

### 13.1 Benchmark
- 693 SciFact claims (208 tuning / 485 evaluation) over **130,066 abstracts**: 5,183 SciFact + 124,883 CliniVer PubMed distractors (35 removed as title duplicates).
- The distractors restored headroom: the best single encoder falls from Recall@20 0.99 (SciFact only) to 0.94–0.96.
- Absolute scores are not comparable to BEIR (different queries and relevance definition).

### 13.2 Encoders (Part A)
| Encoder | Params | Tuning nDCG@10 | Evaluation nDCG@10 | Evaluation Recall@20 | Docs/s (abstracts) |
|---|---|---|---|---|---|
| MedCPT | 219M | **0.838** | 0.811 | 0.941 | 103 (peak GPU 7.3 GB!) |
| BGE-base | 109M | 0.838 | **0.846** | 0.955 | 194 (1.0 GB) |
| BGE-small | 33M | 0.818 | 0.832 | 0.956 | 648 (0.5 GB) |
| E5-base | 109M | 0.812 | 0.811 | 0.943 | 247 (1.0 GB) |
| MiniLM | 23M | 0.734 | 0.718 | 0.899 | ~2,470 (first attempt; embeddings cached) |

- The pre-registered rule selected **MedCPT** by 0.0005 on tuning, a practical tie with BGE-base, which is better on evaluation and 1.9× faster.
- **Budget breach:** MedCPT's article encoder peaked at **7.3 GB** of GPU memory on full abstracts (batch 64 × 512 tokens), above the 6.0 GB project budget; it ran only because the card has 8 GB. On the short CliniVer chunks it peaked at 2.9 GB. Mitigation: encode long documents with batch ≤ 16.
- The biomedical encoder does not dominate first-stage retrieval. Its value shows in re-ranking (§13.4).
- No train/dev contamination signal: train–dev gaps match BM25's (Part A check).

### 13.3 Sparse, fusion and the graph channel
- **BM25:** default tokenization beat the clinical tokenizer (tuning nDCG@10 0.759 vs 0.733). SciFact claims are rarely number- or dose-driven, so this may differ for dosing claims.
- **H4a, supported:** the hybrid (BM25 + MedCPT, RRF) reaches Recall@20 0.949 vs dense 0.941 (+0.008 [−0.010, 0.028]). The weighted hybrid (dense weight 0.6) is better still (Recall@20 0.956, nDCG@10 0.832) but was not in the pre-registered complexity order.
- **Graph channel:**
  - Dictionary linking covered every document (19–25 concepts each) and 664/693 claims.
  - Graph: 14,565 concepts, 3.27M edges; PPR ≈ 470 ms per claim.
  - Alone, the graph reaches Recall@20 0.40. Fusing it **lowers** quality: tribrid nDCG@10 0.718 vs hybrid 0.816; Recall@20 −0.023 [−0.039, −0.010] overall.
- **H4b, not supported.** Synonym-variant stratum: −0.026 [−0.044, −0.009]. Class-level stratum: no difference, with only 20 evaluation claims.
- **H4c:** the normalization-only control (appending alternative surface forms) **hurts** even more (Recall@20 −0.078 vs hybrid). The "graph structure adds value" criterion (tribrid > hybrid + normalization) is met only because the control is degraded; it is reported but not meaningful.
- **Graph not retained** (ADR-0003 rule: H4b criterion not met).
- The error analysis (§10) shows the expected **hub-concept drift**: claims linked to frequent concepts (e.g. common diseases) pull in topically related abstracts that are not evidence.

### 13.4 Re-ranking
| Re-ranker | Pool | Evaluation nDCG@10 | ms / claim (abstracts) |
|---|---|---|---|
| MedCPT cross-encoder | 50 | **0.871** | 244 |
| MedCPT cross-encoder | 20 | 0.867 | 94 |
| BGE re-ranker base | 20 | 0.836 | 141 |
| MiniLM cross-encoder | 20 | 0.827 | 46 |
| none (hybrid) | — | 0.816 | — |

**H4d, supported:** +0.055 [0.034, 0.076] nDCG@10 for the selected MedCPT-CE@50. Domain-specific re-ranking is the largest single gain in this notebook. Larger pools did not help (pool 100: 0.869).

### 13.5 NB02 open question
On SciFact-only, whole abstracts retrieve slightly better than `paragraph-128` chunks (BM25 nDCG@10 0.827 vs 0.798; MedCPT 0.879 vs 0.874; descriptive, no CI).

### 13.6 Selected stack on the CliniVer corpus
- Indexes over **592,930 chunks:** BM25 in 25 s; MedCPT embeddings in 578 s (869 MB fp16 on the GPU).
- Latency per claim (hybrid + MedCPT-CE@50): **p50 0.086 s, p95 0.110 s**, well inside the 2.0 s budget. Peak GPU memory 2.9 GB.
- 2,777 real claims (NB04 standalone rewrites) → top-50 evidence candidates each (`evidence_candidates.jsonl`).
- Sources of the top-5 candidates: PubMed 56%, openFDA labels 25%, ClinicalTrials.gov 12%, Europe PMC 7%.
- **Retrieval quality on real claims is not measured** (no labels). NB06 verification and NB08 evaluate it indirectly.
""",
r"""
## 14. Limitations
- **One labelled benchmark:** SciFact is general biomedicine with expert-written claims, not answers from patient-facing LLMs. BioASQ (registration required) was **NOT EVALUATED**.
- **Distractors come from our own domain corpus**, so relevant SciFact documents are mostly off-domain for them. This makes the task easier than retrieving among in-domain near-misses.
- **The graph channel was tested only in one form:** a PubTator-derived dictionary linker applied to every document (SciFact has no PMIDs; the exact-title PMID mapping failed, 7/1,052), PubTator relations as concept edges, and single-hop claims. The negative result is specific to this setup.
- **Small strata:** class-level has 20 evaluation claims; synonym-variant is broad (75% of claims), because the dictionary has many surface forms per concept.
- **Near-tie encoder selection:** MedCPT vs BGE-base on tuning, opposite order on evaluation. Re-check in NB08.
- **Protocol notes** (all before any hypothesis result; recorded in the experiment log):
  1. Benchmark changed to add distractors (ceiling).
  2. PMID mapping replaced by uniform dictionary linking.
  3. The class-level regex was broken by YAML escaping and fixed.
  4. The run was restarted with progress logging after an unexplained slowdown.
""",
r"""
## 15. Conclusion and decision
**Conclusion:**
- On this hardware and benchmark, the best evidence retriever is a **BM25 + MedCPT hybrid, re-ranked by the MedCPT cross-encoder** (evaluation nDCG@10 0.871, ~0.1 s per claim on the 593k-chunk corpus).
- Hybrid retrieval matches or beats the single channels (H4a).
- Domain re-ranking gives the largest gain (H4d).
- The concept graph, as built here, adds noise rather than evidence (H4b not supported), so it is **not retained** under the pre-registered rule.

**Decision (proposed, pending owner approval; ADR-0012):** adopt the stack above; evidence candidates for the 2,777 real claims go to NB06.

**Recommendation for the owner (not applied):** index PubMed abstracts whole and keep chunks for long documents (amends ADR-0009 for abstracts).

**Next steps:**
- **NB06 (claim–evidence verification):** NLI and LLM verifiers on SciFact / HealthVer, then on the real claims with these candidates.
- **NB08:** re-check BGE-base vs MedCPT and the weighted hybrid in the full pipeline.
""",
]
