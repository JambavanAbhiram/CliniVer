"""Discussion cells for NB06, written AFTER execution from the outputs of EXP-06-001."""

DISCUSSION_CELLS = [
r"""
## 11. Discussion
All figures are copied from the outputs above (`artifacts/experiments/EXP-06-001/`).

### 11.1 Verifier comparison (macro-F1 over SUP / CON / INS; synthetic includes PART)
| Verifier | HealthVer dev | HealthVer test | SciFact dev | HealthFC | Synthetic |
|---|---|---|---|---|---|
| **Fine-tuned NLI** (DeBERTa-v3-base, SciFact + HealthVer train) | **0.834** | **0.777** | **0.786** | 0.282 | 0.154 |
| LLM Qwen3-8B (fixed samples) | 0.687 | 0.571 | 0.762 | **0.695** | **0.285** |
| Zero-shot NLI (DeBERTa-v3-large) | 0.524 | 0.455 | 0.767 | 0.332 | 0.176 |
| Hybrid NLI + qualifier rules | 0.834 | 0.777 | 0.786 | 0.282 | 0.162 |
| MiniCheck-RoBERTa-Large | 0.393 | 0.380 | 0.488 | 0.177 | 0.045 |
| Similarity (MedCPT) | 0.419 | 0.387 | 0.389 | 0.201 | 0.039 |
| LLM Granite 3.3 8B (samples) | 0.439 | 0.383 | — | — | — |
| Majority (INS) | 0.228 | 0.190 | 0.185 | 0.240 | 0.000 |

- **H5a: NOT SUPPORTED** under the pre-registered rule (a gain on both HealthVer test and SciFact dev).
  - Fine-tuning gives a large in-distribution gain on HealthVer (+0.322 [0.294, 0.352]), but none that is significant on SciFact (+0.018 [−0.032, 0.070]).
  - The Qwen LLM verifier beats zero-shot NLI on HealthVer (+0.128 [0.073, 0.180]) but not on SciFact (−0.005 [−0.060, 0.046]).
- **Generalization is the main finding.**
  - The fine-tuned NLI is best wherever the data resemble its training sets, and **collapses on HealthFC** (0.282: expert verdicts on question-form, consumer-style claims).
  - The **Qwen3-8B LLM verifier generalizes far better there (0.695)** and is the only verifier that produces PART.
  - Our real claims come from patient-facing answers, closer to HealthFC than to HealthVer.
- **Granite 3.3 8B** produced no valid JSON in 506 of 600 calls, even with a four-label schema (consistent with NB03); it is not usable in this setup.
- **MiniCheck** is very strict on HealthVer-style indirect evidence: the tuned threshold reached the grid minimum (0.02). MiniCheck cannot express CON, which caps its macro-F1.
- **The hybrid qualifier rule changed no prediction** on the human-labelled sets: its trigger conditions (numbers or populations in the claim missing from the evidence) almost never co-occur with an NLI SUP there.

### 11.2 Error types
- **H5b: NOT SUPPORTED** under the rule (> 50% of the similarity baseline's errors are false support on both sets).
  - On SciFact, false support is **89.8%** of the similarity baseline's errors (false-support rate 0.698): similarity ≠ verification, as expected.
  - On HealthVer it is 40.5%, because many HealthVer CON/INS evidence passages are lexically distant from the claim.
- **False-support rates on HealthVer test:** zero-shot NLI 0.025, fine-tuned NLI 0.125, Qwen 0.309, MiniCheck 0.348, similarity 0.309.
- **False-contradiction rates** stay ≤ 0.082 for every verifier.

### 11.3 Qualifier errors (synthetic, labels by construction)
| Edit | Fine-tuned NLI | Qwen3-8B | Zero-shot NLI |
|---|---|---|---|
| Negation flip → CON | 0.526 | 0.450 | 0.459 |
| **Number change → CON** | **0.053** | **0.061** | **0.118** |
| Hedge removed → PART | 0.000 | 0.562 | 0.000 |
| Population widened → PART (n = 13) | 0.000 | 1.000 | 0.000 |
| Paraphrase → SUP (control) | 0.820 | 0.714 | 0.342 |

- **No verifier reliably detects a changed number** (dose tripled, percentage +10 points): at most 12% are caught. This is the most safety-relevant weakness found so far.
- Only the LLM verifier can express PART, and it catches roughly half of the hedge removals.

### 11.4 Top-k aggregation (H5c, SciFact dev claims, retrieved evidence)
| Evidence | Claim-level macro-F1 |
|---|---|
| top-1 | **0.706** |
| top-3 | 0.667 |
| top-5 | 0.654 |
| oracle (cited abstracts) | 0.786 |

- **H5c: NOT SUPPORTED**, and the effect runs the other way: top-3 vs top-1 is −0.039 [−0.073, −0.005].
- The simple any-SUP / any-CON rule lets lower-ranked, less relevant evidence inject spurious labels.
- The 0.08 gap to oracle evidence is the retrieval share of the error.

### 11.5 Real claims (2,777; no gold labels)
- Verdicts with the selected fine-tuned NLI and top-3 aggregation (as pre-registered): **SUP 81.5%, INS 10.7%, CON 7.8%**, with a conflict flag on 270 (9.7%).
- The weaker generator receives more CON (Qwen3-1.7B 10.1% vs Qwen3-8B 6.1%).
- **These rates are not validated.** Given the verifier's weak transfer to consumer-style claims (HealthFC), its 0.125 false-support rate, its blindness to number changes, and H5c, the SUP share is very likely **inflated**. Each verdict carries the corpus snapshot and the evidence dates (ADR-0005).
- Resources:
  - Retrieval p95 0.37 s per claim (2.4 GB peak).
  - Fine-tuning: 416 s, 7.1 GB peak (above the 6 GB budget; within the 8 GB card).
  - Qwen3-8B verifier: 5.9 GB.
""",
r"""
## 12. Limitations
- **No real-answer gold labels:** verdicts on real claims are unvalidated by design (D5/D7; clinical validation is the final stage).
- **Train/test proximity:** fine-tuning used HealthVer train, so HealthVer test is in-distribution. SciFact dev and HealthFC are the out-of-distribution checks.
- **Samples:** LLM verifiers were evaluated on fixed-seed samples (200–400 pairs per set). The fast verifiers were also reported on the same samples (`verifier_results_on_llm_samples`).
- **Synthetic perturbations are rule-based and clean**, so they measure sensitivity, not the realistic error mix. The population edits are few (13).
- **HealthFC** (CC BY-NC-ND 4.0) is used for evaluation only. Its claims are questions, and its label mapping was checked against the German verdict wording, not re-annotated.
- **GPU budget:** fine-tuning peaked at 7.1 GB and the LLM servers at 6.7–7.1 GB including resident PyTorch memory (6 GB budget, 8 GB card).
- **Protocol notes** (recorded in the experiment log):
  1. The MiniCheck input format was corrected to the reference implementation before the full run.
  2. The PyTorch models are moved to the CPU while the LLM server runs.
  3. A MedCPT encoder memory-retention bug was found and fixed (affects only memory, not values).
""",
r"""
## 13. Conclusion and decision
**Conclusion:**
- With open models, claim verification reaches macro-F1 ≈ 0.78–0.79 on in-distribution scientific and health claims (fine-tuned NLI), and ≈ 0.70 on expert-labelled consumer claims (LLM verifier).
- No single verifier wins everywhere.
- **The dominant risks are false support and blindness to numeric/dose changes, not false contradiction.**
- More evidence (top-3) hurts with naive aggregation.
- **H5a, H5b and H5c are all NOT SUPPORTED** under the pre-registered rules. The detailed results above say why.

**Decision (proposed, pending owner approval; ADR-0013):**
- Keep the pre-registered selection (fine-tuned NLI) as the **fast first-pass** verifier, whose output is in `verdicts.jsonl`.
- Do **not** treat its SUP verdicts as validated.
- NB07 must address:
  - (a) calibration and abstention;
  - (b) top-1 vs aggregated evidence;
  - (c) a **numeric/qualifier check** (the rule-based detectors from NB02/NB03 comparing claim numbers against evidence numbers);
  - (d) an LLM-verifier second opinion, or an ensemble for consumer-style claims.
- Granite is dropped as a verifier.

**Next steps:** NB07 (confidence, calibration, abstention) on these outputs.
""",
]
