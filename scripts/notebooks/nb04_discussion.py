"""Discussion cells for NB04, written AFTER execution from the outputs of EXP-04-001."""

DISCUSSION_CELLS = [
r"""
## 8. Discussion
All figures are copied from the outputs above (`artifacts/experiments/EXP-04-001/`).

### 8.1 Test items (synthetic, by construction)
- 693 SciFact claims → **476 valid items** (68.7%): 155 tuning / 321 evaluation.
- Rejected by the rules: the subject still named in the dependent claim (74), not one sentence of similar length (58), content added (38), entity not in the question (27), entity not in the claim (16), other (4).
- The rules removed exactly the failure modes seen in the probe (invented sentences, questions restating the finding).

### 8.2 Hypotheses (evaluation split, 321 claims)
| | Retriever | Claim only (A) | Question + claim (B) | Difference [95% CI] |
|---|---|---|---|---|
| **H3a** context-dependent, Recall@10 | BM25 | 0.759 | 0.906 | **+0.146 [0.107, 0.185]** |
| | Dense | 0.721 | 0.967 | **+0.246 [0.202, 0.293]** |
| **H3b** self-contained, nDCG@10 | BM25 | 0.835 | 0.820 | −0.015 [−0.033, 0.001] |
| | Dense | 0.901 | 0.890 | **−0.011 [−0.022, −0.000]** |

- **H3a: SUPPORTED.** When a claim lost its subject, adding the question recovers most of the lost recall for both retrievers.
- **H3b: SUPPORTED.** For self-contained claims, adding the question does not help, and with the dense retriever it slightly but significantly **hurts** (P@5 −0.003 [−0.005, −0.001]).
- Separate retrieval + merge (D) is clearly worse on self-contained claims (nDCG@10 0.742 vs 0.901, dense). Rank fusion lets the question's documents displace the claim's.

### 8.3 Strategy selection: F (LLM rewrite), chosen on the tuning split
| Strategy | Tuning nDCG@10 (mean, both strata, both retrievers) |
|---|---|
| F rewrite | **0.868** |
| C question + claim + answer | 0.863 |
| B question + claim | 0.859 |
| E weighted fusion (w = 0.6) | 0.844 |
| D rank fusion | 0.779 |
| A claim only | 0.734 |

- On the evaluation split, F **matches the oracle** on context-dependent claims (dense nDCG@10 0.904 vs oracle 0.901; BM25 0.835 vs 0.835).
- F also does not degrade self-contained claims (0.906 vs 0.901 dense), unlike B.
- Query drift explains why: the F queries stay closest to the original claim (drift 0.004 vs 0.045 for B and 0.092 for C).
- On SciFact, F restored the removed subject in **98.3%** of dependent claims and returned 82.4% of already-standalone claims unchanged (the rest were minor rewordings).

### 8.4 Real claims (NB03 output, no relevance labels)
- **42.7% (1,185 / 2,777) are tagged context-dependent**: 701 because the key entity is only in the question, 484 for an unresolved referent. The rate is the same for both generators (43.5% / 42.0%).
- **Tagger spot check (40 claims, AI-annotated): 80% agreement.**
  - 16/20 "dependent" tags are correct. The false positives are complete statements in which the question happened to name a drug.
  - 4/20 "self-contained" tags are wrong. All four have an unresolved reference *plus* a domain word ("**These factors** increase LDL cholesterol"), which suppresses the referent rule. So the 42.7% is approximate, with errors in both directions.
- **F on real claims is far less complete than on synthetic ones.**
  - It changed 60.6% of the 1,185 dependent claims, and 50.8% are tagged self-contained afterwards (vs 98.3% restoration on SciFact).
  - The unchanged ones are largely generic advice ("Always ensure that you are using a genuine product…"), plus tagger false positives.
  - When F rewrites, it typically restores the drug or condition ("stopping the medication" → "stopping simvastatin"). Occasionally it turns a list fragment into a stated relation ("9. Headache" → "Headache is a sign of low blood pressure"): correct in context, but it introduces a predicate the answer only implied.
  - 83% of the self-contained controls came back unchanged; mean length grows by 21% for dependent claims.
""",
r"""
## 9. Limitations
- **Synthetic, favourable construction:** each dependent claim lacks exactly one entity, and the question names exactly that entity in a formulaic way ("What do we know about X?"). Real dependence is messier (missing populations, conditions, list context), and the real-claim analysis shows F is much less complete there. The H3 effect sizes are therefore an **upper bound** on real-world gains.
- **Benchmark domain:** SciFact is general biomedicine, with claims written from citation contexts, not patient-facing answers.
- **Same model generates and rewrites:** Qwen3-8B wrote the synthetic context and also performs F. It may undo its own phrasing more easily than real phrasing.
- **No retrieval labels for real claims:** retrieval quality of F on real claims is not measured, only how often it rewrites and whether the rewrite still looks context-dependent.
- **Tagger:** precision and recall are only spot-checked (40 items, AI annotation; D5).
- **Fixed small reference retrievers:** NB05 re-checks the strategy with the final retrieval stack (NB08 re-runs winners).
""",
r"""
## 10. Conclusion and decision
**Conclusion:**
- Context matters, but only for claims that need it.
- Adding the question rescues context-dependent claims (+15 to +25 Recall@10 points) and slightly hurts self-contained ones.
- An LLM rewrite into a standalone claim (F) gets the benefit without the cost on the synthetic benchmark, matching the oracle.
- On real answers, 43% of claims need context, and F resolves roughly half of them.

**H3a: SUPPORTED. H3b: SUPPORTED** (pre-registered rules; synthetic test items).

**Decision (proposed, pending owner approval; ADR-0011):**
- Use **strategy F** for every claim: retrieve with the standalone rewrite (`standalone_text` in `data/processed/2026-11-20/responses/claims_contextualized.jsonl`).
- Keep the original claim text for verification and display.
- Do **not** prepend the question by default (it degrades self-contained claims).
- For claims still tagged context-dependent after F, NB05 tests question + claim (B) as a fallback channel.

**Next steps (NB05, retrieval):**
- Sparse, dense, graph and fusion channels plus re-ranking over the CliniVer chunked corpus and SciFact.
- Re-check the chunk-size and whole-abstract questions left open by NB02.
- Test F vs F + B fallback on real claims where labels can be obtained.
""",
]
