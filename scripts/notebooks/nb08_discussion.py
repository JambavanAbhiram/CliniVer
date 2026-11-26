"""Discussion cells for NB08, written AFTER execution from the outputs of EXP-08-001."""

DISCUSSION_CELLS = [
r"""
## 10. Discussion
All figures are copied from the outputs above (`artifacts/experiments/EXP-08-001/`). Labels are benchmark (SciFact, human), synthetic (NB04) and silver. **Real-claim accuracy is NOT EVALUATED** (D7).

### 10.1 End-to-end results (300 SciFact dev claims, retrieval over 130,066 abstracts)
| Configuration | Coverage | Macro-F1 (answered) | Macro-F1 (abstained = error) | False-support rate (answered) |
|---|---|---|---|---|
| **Pipeline** (ADR-0011 … 0014) | 0.880 | 0.762 | 0.714 | 0.191 |
| NLI only (no calibration / second opinion / abstention) | 1.000 | 0.726 | **0.726** | 0.188 |
| Uncalibrated routing | 0.847 | **0.773** | 0.707 | 0.175 |
| No second opinion (threshold abstention) | 0.810 | 0.753 | 0.673 | 0.171 |
| No abstention (second opinion kept) | 1.000 | 0.715 | 0.715 | 0.199 |
| Weighted hybrid retrieval | 0.877 | 0.760 | 0.711 | 0.185 |
| BGE-base instead of MedCPT | 0.883 | 0.751 | 0.705 | 0.203 |
| Hybrid without re-ranker | 0.873 | 0.711 | 0.664 | 0.211 |
| MedCPT only | 0.873 | 0.727 | 0.679 | 0.184 |
| BM25 only | 0.883 | 0.699 | 0.653 | 0.181 |
| Qwen3-8B only (no NLI) | 1.000 | 0.718 | 0.718 | 0.250 |
| Zero-shot NLI | 1.000 | 0.675 | 0.675 | 0.153 |
| Top-3 naive aggregation | 1.000 | 0.682 | 0.682 | 0.273 |

- **Components that clearly earn their place:**
  - the **MedCPT re-ranker** (+0.05 macro-F1 over the hybrid without it);
  - the **fine-tuned verifier** (+0.05 over zero-shot NLI; +0.01 over Qwen-only, with far lower false support);
  - **top-1 evidence** (+0.04 over top-3 naive).
- **The confidence layer does not pay for itself in distribution.** It raises answered-claim quality (0.726 → 0.762), but abstains on 12% of claims without lowering false support (0.191 vs 0.188). With abstentions counted as errors it is slightly below plain NLI. Its value lies **out of distribution** (NB07: HealthFC 0.28 → 0.54 with the second opinion).
- The top-1 retrieved abstract is a gold evidence abstract for 48.7% of claims. The re-ranker raises this from 44.3%, and BGE-base matches MedCPT (48.3%).

### 10.2 H7: error attribution
- **H7: NOT SUPPORTED.** Of 81 errors without abstention, **47 (58%) are verifier-attributable** and 34 (42%) retrieval-attributable; the CI of the share difference is [−0.38, 0.05].
- Oracle evidence lifts macro-F1 only from 0.726 to **0.786**, so **the verifier is the main bottleneck**.
- **By gold label:**

  | Gold label | Retrieval errors | Verifier errors |
  |---|---|---|
  | SUP | 7 | **21** |
  | CON | 3 | **13** |
  | INS | **24** | 13 |

  Retrieval errors concentrate on claims for which no evidence exists (INS): a topical but non-addressing abstract is retrieved and then judged SUP or CON.

### 10.3 Contextualization end-to-end (476 synthetic context-dependent items; SUP/CON macro-F1)
| Retrieval query | Verifier reads | Macro-F1 |
|---|---|---|
| claim as written | claim as written | 0.715 |
| question + claim | claim as written | 0.794 |
| rewrite F | claim as written (ADR-0011 policy) | 0.820 |
| **rewrite F** | **the rewrite** | **0.896** |
| original claim (oracle) | original claim | 0.897 |

**Verifying the standalone rewrite instead of the claim as written closes almost the whole gap to the oracle** (+0.076). ADR-0011 kept the original text for verification; this result argues for changing that. The original text stays for display.

### 10.4 Silver check of the real claims (Pillar 3)
- Each real claim is compared with its question's reference statements. Silver labels were assigned to 903 of 2,777 claims.

  | Reference statements | Pipeline SUP when silver SUP | Pipeline SUP when silver CON ("silver false support") | Pipeline CON or ABSTAINED when silver CON |
  |---|---|---|---|
  | Physician (K-QA): 58 silver SUP, 31 silver CON | 0.64 | **0.45** | 0.39 |
  | LLM-generated (MedLFQA): 644 silver SUP, 170 silver CON | 0.70 | **0.47** | 0.29 |

- **Interpretation (silver, noisy):** on real answers, the pipeline agrees with supporting reference statements most of the time, but **labels nearly half of the reference-contradicted claims as SUP**. This is consistent with the benchmark findings: false support and numeric blindness are the dominant risks.

### 10.5 Cost, latency and resources
| Stage | Median | p95 |
|---|---|---|
| Answer generation | 3.65 s | 4.84 s |
| Claim extraction (per answer) | 18.8 s | 30.2 s |
| Standalone rewrite (per claim) | 0.88 s | 1.33 s |
| Retrieval + re-ranking (per claim) | — | 0.37 s |
| Second opinion (35% of claims) | 2.33 s each | ≈ 0.82 s expected per claim |

- No LLM call in these stages ended unfinished.
- GPU memory (live): Qwen3-8B server **4.6 GB** above baseline for one short verification call; NB03 measured **5.3 GB** during generation and claim extraction (same server settings; the peak depends on the workload). Both are within the 6 GB budget. Retrieval 2.4 GB.

### 10.6 Final selection (pre-registered rule)
- The rule selects **"uncalibrated"** routing: the highest answered macro-F1 (0.773) among configurations with false support ≤ the pipeline's, within the latency budget.
- This is a weak preference: it is within noise of the pipeline and has lower end-to-end macro-F1 (0.707).
- The rule optimizes answered-claim quality and does not penalize abstention. That is a limitation of the rule, recorded here rather than changed after seeing the results.
""",
r"""
## 11. Limitations
- **No real-claim gold labels** (D7). Silver labels come from NLI against reference statements, which may cover a claim only partially. For most MedLFQA questions the statements are LLM-generated.
- **One end-to-end benchmark** (SciFact dev, 300 claims), so CIs are wide. The synthetic context items favour the rewrite strategy by construction (NB04).
- **Ablations are one-factor-at-a-time**; interactions (e.g. a different verifier with a different retriever) are not explored.
- **The final-selection rule ignores coverage**, as noted in §10.6.
- **Reproducibility incident:** NB03's cache replay had erased its live GPU and latency measurements, because no server started. `ManagedClient.start()` now forces a live measurement; NB03 was re-run to restore them.
""",
r"""
## 12. Conclusion and decision
**Conclusion:**
- End to end, CliniVer reaches macro-F1 ≈ 0.71–0.76 on SciFact claims retrieved from 130k abstracts, at ~0.4 s retrieval plus ~1 s LLM time per claim on a laptop.
- **The verifier, not retrieval, is the bottleneck (H7 not supported).**
- **False support is the dominant failure**, on benchmarks (≈ 0.19) and on real claims (silver ≈ 0.45 of contradicted claims).
- Three design changes are clearly indicated by the evidence:
  1. verify the **standalone rewrite** (+0.076 on context-dependent claims);
  2. a verifier trained to catch **numeric / negation / qualifier changes**;
  3. a confidence layer tuned for **end-to-end** quality and out-of-distribution robustness rather than in-distribution abstention.

**Decision (proposed, pending owner approval; ADR-0015):**
- The approved pipeline stands as the **baseline v1**.
- An **optimization iteration** (owner request 2026-11-21) addresses the three changes on **fresh, untouched evaluation data**, so the gains are not tuned on the NB08 results.
- Real-claim accuracy remains NOT EVALUATED until clinical validation.
""",
]
