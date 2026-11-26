"""Discussion cells for NB07, written AFTER execution from the outputs of EXP-07-001."""

DISCUSSION_CELLS = [
r"""
## 8. Discussion
All figures are copied from the outputs above (`artifacts/experiments/EXP-07-001/`).

### 8.1 Calibration (H6a)
The fine-tuned NLI verifier is **over-confident**; the fitted temperature is T = 2.0.

| ECE (15 equal-mass bins) | HealthVer test | SciFact dev | HealthFC |
|---|---|---|---|
| uncalibrated | 0.156 | 0.141 | 0.565 |
| temperature scaling | 0.062 | **0.060** | 0.423 |
| isotonic (top-label) | **0.045** | 0.067 | **0.399** |
| Platt (top-label) | 0.094 | 0.093 | 0.470 |

- **H6a: SUPPORTED.** Temperature scaling lowers ECE on every held-out set: HealthVer −0.094 [−0.096, −0.082]; SciFact −0.080 [−0.087, −0.034]; HealthFC −0.143 [−0.146, −0.138].
- Isotonic regression is slightly better on two sets. On consumer-style claims (HealthFC) every method stays badly calibrated (ECE ≥ 0.40): calibration fitted on HealthVer does not transfer to a different claim distribution.

### 8.2 Which signals predict correctness (H6b)
- **H6b: NOT SUPPORTED.** The multi-signal confidence gives a tiny AURC gain on HealthVer test (−0.003 [−0.005, −0.001]) and a non-significant loss on SciFact (+0.013 [−0.007, 0.039]).
- The standardized coefficients show the verifier's own **margin and entropy** carry almost all the signal. The numeric-consistency signals have near-zero weight (unit mismatch 0.06; unsupported numbers 0.02), because they are rare or noisy on these datasets.
- **Exploratory (not part of the rule): agreement with the Qwen3-8B verifier is the most informative extra signal.**
  - On HealthFC the NLI verifier is right 61.5% of the time when Qwen agrees and **5.5%** when it disagrees. Adding agreement lowers AURC from 0.618 to 0.461.
  - On SciFact: 90.4% vs 50.0%.

### 8.3 Abstention (H6c)
- **H6c: SUPPORTED on HealthVer test.** At the threshold set on the calibration split for 80% coverage, coverage is 77.1% and the false-support rate falls from 0.125 to **0.091** (−0.034 [−0.047, −0.021]). Risk falls from 0.214 to 0.128.
- On SciFact, coverage is 82.9% and the false-support rate falls from 0.109 to 0.066.
- **On HealthFC abstention backfires.** Coverage is only 58.5%, and the false-support rate among the answered claims *rises* from 0.80 to **0.89**: out of distribution, the verifier is confidently wrong, so a confidence threshold keeps exactly the wrong answers.
- Reaching a false-support rate ≤ 0.05 on the calibration split costs coverage: about 85% is the highest coverage that meets it.

### 8.4 Carry-over decisions
- **Evidence aggregation** (300 SciFact dev claims, retrieved evidence): top-1 0.706 > naive any-rule 0.665 > max-confidence 0.529 → **top-1** is used for real claims (consistent with NB06 H5c).
- **Second opinion** (route the lowest-confidence 25% to Qwen3-8B; fixed samples):
  - HealthFC macro-F1 0.276 → **0.543**, the largest robustness gain in NB06–NB07;
  - SciFact 0.786 → 0.788;
  - HealthVer test 0.756 → 0.732, a small cost where the fine-tuned model is strongest.

### 8.5 Final verdicts for the real claims (2,777; unvalidated)
| Verdict | NLI alone | After calibration, second opinion and abstention |
|---|---|---|
| SUP | 1,919 (69.1%) | 1,807 (65.1%) |
| INS | 664 (23.9%) | 462 (16.6%) |
| CON | 194 (7.0%) | 75 (2.7%) |
| ABSTAINED | — | **433 (15.6%)** |

- 983 claims (35.4%) fell in the low-confidence band, more than the 25% expected from the calibration split, which is a sign of distribution shift. In 433 of them Qwen disagreed and the claim was abstained.
- The decisive evidence is mostly regulatory labels (887), reviews (821) and trial registry records (512). Only 10 decisions rest on a guideline and 15 on a meta-analysis.
- 162 claims have numbers absent from their decisive evidence; this is flagged in the confidence vector.
- The weaker generator has more abstentions (16.8% vs 14.7%) and more CON (3.3% vs 2.3%).
""",
r"""
## 9. Limitations
- **No real-claim gold labels:** the effect of calibration, abstention and the second opinion on real claims is not measured (D5/D7). The benchmark results show that the confidence model does not transfer to consumer-style claims.
- **Fitting on HealthVer dev only:** HealthVer test is in-distribution, SciFact and HealthFC are not. The HealthFC failure is the realistic warning.
- **Exploratory signals:** the LLM-agreement analysis and the combined real-claim policy (second opinion + abstain on disagreement) were not pre-registered as hypotheses. They are design proposals backed by the benchmark samples.
- **Verbalized and sampling-consistency LLM confidence:** NOT EVALUATED (no additional LLM runs); agreement between the two verifiers was used instead.
- **Bug fixed before reporting:** the isotonic and Platt maps were first fitted on temperature-scaled confidences but applied to uncalibrated ones. They were corrected and the notebook replayed (no new LLM calls); H6a–c were unaffected.
""",
r"""
## 10. Conclusion and decision
**Conclusion:**
- The verifier is over-confident, and simple post-hoc calibration fixes this in distribution (H6a supported).
- Abstention reduces false support in distribution (H6c supported).
- Extra hand-crafted signals add little (H6b not supported).
- **Out of distribution (consumer-style claims) confidence fails, and abstention makes false support worse.** There, disagreement between two different verifiers is the most useful warning, and an LLM second opinion roughly doubles macro-F1.

**Decision (proposed, pending owner approval; ADR-0014):**
- Temperature-scaled NLI confidence (isotonic as an alternative).
- **Top-1 evidence.**
- Qwen3-8B second opinion for the low-confidence band, with **ABSTAINED when the two disagree**.
- The full confidence vector is reported with every verdict: `verdicts_calibrated.jsonl`.
- ABSTAINED and INS stay distinct (ADR-0005).

**Next steps (NB08, end-to-end evaluation):**
- Run the whole pipeline with ablations: extraction, contextualization, retrieval (MedCPT vs BGE-base; weighted hybrid), verification (NLI vs routed vs LLM), aggregation, abstention.
- Re-examine the verifier choice in light of the HealthFC results.
- Consider the small human-labelled real-claim set that would turn these verdicts into a measured accuracy.
""",
]
