"""Discussion of NB08 Part B (EXP-08-002), written AFTER execution from its outputs."""

DISCUSSION_CELLS = [
r"""
## 24. Discussion: optimization round 1 (EXP-08-002)
All figures come from the outputs above (`artifacts/experiments/EXP-08-002/`).
- **Fresh tests:**
  - the SciFact-train test split: 241 claims, grouped by abstract, never used for training or selection;
  - PUBHEALTH test.
- **Seen sets:** everything used in NB06–NB08.

### 24.1 Selection and adoption (pre-registered rules)
- The dev split selected **`v2 | no numeric guard | no INS threshold | second opinion → take the LLM label`**.
  - All 8 configurations within 0.01 of the best dev score use v2 + "take the LLM label".
  - The rule then picked the one with the fewest components.
- The adoption rule passed on all three criteria, so this configuration is **baseline v2**.

| Fresh test | Baseline (v1 recipe, v1 policy) | Selected | Δ (95% CI) |
|---|---|---|---|
| SciFact-train test, macro-F1 (abstained = error) | 0.700 | **0.755** | **+0.054 [+0.011, +0.099]** |
| SciFact-train test, false-support rate | 0.231 | **0.124** | **−0.106 [−0.168, −0.052]** |
| PUBHEALTH macro-F1 | 0.436 | **0.472** | **+0.036 [+0.020, +0.054]** |
| PUBHEALTH false-support rate | 0.330 | 0.346 | +0.016 [−0.013, +0.045] (no significant change) |

- On the fresh SciFact split, the optimized pipeline is **better and safer at once**: false support is roughly halved.
- On PUBHEALTH it is better, with no significant change in false support. PUBHEALTH stays hard (0.47): consumer fact-checks with journalist explanations as evidence.

### 24.2 Where the gain comes from (component rows, fresh SciFact test)
| Configuration | Coverage | Macro-F1 | False-support rate |
|---|---|---|---|
| v1r verifier only | 1.00 | 0.689 | 0.226 |
| v2 verifier only | 1.00 | 0.672 | **0.088** |
| v2 + v1 policy (abstain on disagreement) | 0.83 | 0.694 (answered 0.771) | 0.087 |
| **v2 + take the LLM label** | 1.00 | **0.755** | 0.124 |

- **v2 is a conservative verifier.** The augmentation adds rule edits plus 811 hard negatives (topical non-cited abstracts → INS). It cuts false support from 0.23 to 0.09, but loses some SUP/CON recall.
- **The LLM second opinion restores that recall** for the low-confidence band. Taking its label beats abstaining.

### 24.3 Robustness to edits (synthetic; the same rule generators as the v2 training augmentation, so **in-generator**)
| Edit | v1 | v2 |
|---|---|---|
| Changed number caught (→ CON) | 0.053 | **0.935** |
| Negation caught (→ CON) | 0.526 | **0.788** |
| Overstated hedge / population not called SUP | 0.247 | **0.847** |
| Paraphrase kept as SUP | 0.820 | 0.757 |

- The number blindness found in NB06 (≤ 12%) is removed for this kind of edit.
- Because the generators are shared with training, this shows the verifier *can* learn the distinction. It does not show that it generalizes to naturally occurring numeric errors. The fresh-test false-support reduction is the stronger evidence.

### 24.4 Costs and regressions (reported with equal weight)
1. **Context items** (§18b, post-hoc fairness check; macro-F1 on SUP/CON, rewrite verified, verifier only):
   - Unseen by all models: v1 0.806, v1r 0.769, **v2 0.682**. The drop is **real, not a training-exposure artifact.**
   - Mechanism: v2 says INS more often, and this metric scores INS as an error even when the retrieved abstract does not address the claim.
   - In the full pipeline, the second opinion compensates (SciFact dev end to end: 0.714 → 0.739).
2. **Real claims, silver check:**
   - Against LLM-generated reference statements, silver false support improves: 0.465 → 0.424.
   - Against **physician** statements it **worsens: 0.452 → 0.548** (31 silver-CON claims, i.e. about 3 claims).
   - Claims labelled CON or ABSTAINED when silver says CON fall from 0.29–0.39 to 0.15–0.19, because abstention no longer exists.
   - On real claims the LLM overrode the verifier for 519 of 2,777 claims (331 INS → SUP).
   - **Taking the LLM label shifts real-claim verdicts towards SUP.** This is the main open risk of baseline v2.
3. **Latency:** 40.4% of real claims are routed to the second opinion (35.4% in v1), about 1 s more per claim.
4. **Unused levers:**
   - The INS threshold on the cross-encoder score is useless: its scores saturate near 1.0 (dev quantiles 0.10 / 0.99 / 0.998).
   - The numeric guard is redundant once v2 catches number edits.
""",
r"""
## 25. Limitations (Part B)
- **Post-hoc with respect to EXP-08-001:** the levers were chosen after seeing the NB08 errors. The protection is the fresh test data, which played no part in any choice.
- **One fresh benchmark per kind:** 241 SciFact claims (wide CIs) and PUBHEALTH (verifier-level, given evidence, few INS items: 45).
- **The synthetic edit results are in-generator** (§24.3).
- **Real-claim results are silver** and small for physician statements (31 contradicted claims). Real-claim accuracy remains NOT EVALUATED (D7). The expert-review interface (`cliniver.service`) starts collecting human judgements.
- **One seed, one augmentation recipe;** no hyper-parameter search beyond the pre-registered grid.

## 26. Conclusion and decision (Part B)
- **Conclusion:** on fresh data, round 1 raises end-to-end macro-F1 (+0.054) and **halves false support** on scientific claims. It also improves consumer fact-checks, and makes the verifier sensitive to changed numbers, negations and overstatements.
- **The price:**
  - a more conservative verifier (lower SUP/CON recall when used alone);
  - a pipeline that relies on the LLM second opinion;
  - on physician-referenced real claims, that LLM pushes some contradicted claims to SUP.
- **Decision** (pre-registered rule; ADR-0016): the selected configuration is **baseline v2** and powers the demo service.
- **Round 2 targets the open risk:**
  - let v2's INS/CON act as a veto on LLM SUP overrides, instead of always taking the LLM label;
  - restore SUP/CON recall on context items, e.g. with fewer hard negatives;
  - measure real-claim accuracy with the expert review.
""",
]
