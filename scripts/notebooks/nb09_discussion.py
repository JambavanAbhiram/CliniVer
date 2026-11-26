"""Discussion cells for NB09, written AFTER execution from EXP-09-001 … EXP-09-003."""

DISCUSSION_CELLS = [
r"""
## Discussion
All figures come from the outputs above (`artifacts/experiments/EXP-09-001 … 003/`).

### Round 1 (EXP-09-001, pre-registered): G3 fixes the failure, but not safely enough
- **Dev:** G3 (planner + drug-class expansion + direct-answer prompt) was best: reference recall 0.470 vs G0 0.396, contradiction 0.069 vs 0.106, drug naming 100% vs 70%.
- **Test (136 questions):**
  - G3 named a drug for every treatment question (G0: 70%) and cut non-answers from 33% to 10%;
  - **reference contradictions rose 0.048 → 0.092** (+0.044 [+0.010, +0.078]).
- The rule therefore **did not adopt** G3.
- The error analysis traced the real errors to:
  1. facts from outside the passages, with a citation attached;
  2. answering a neighbouring aspect;
  3. off-topic passages from class expansion on **non-treatment** questions.

  About half of the flagged contradictions were judge or reference artefacts.

### Round 2 (EXP-09-002): stricter grounding (G4) costs support
- G4's prompt forbids background knowledge. On the seen test it brought contradictions back to 0.057.
- On the fresh set its verified Supported share fell to 0.775 (G0 0.819), and it said "the passages do not cover…" in 52% of answers. **Not adopted.**
- **On the same fresh set, G3 was best on every measure:** Supported 0.873, Contradicted 0.028, non-answers 7%.

### Round 3 (EXP-09-003): route by intent (H1), adopted
| Criterion | G0 | H1 | Rule |
|---|---|---|---|
| Seen test: reference contradiction | 0.048 | **0.042** (−0.006 [−0.019, +0.004]) | ≤ G0 + 0.02 ✅ |
| Seen test: reference recall | 0.437 | **0.444** (+0.007 [−0.020, +0.036]) | ≥ G0 − 0.01 ✅ |
| Fresh set 2 (58 questions): verified Supported | 0.919 | **0.917** (−0.002 [−0.023, +0.018]) | ≥ G0 − 0.02 ✅ |
| Fresh set 2: verified Contradicted | 0.015 | **0.014** | ≤ G0 + 0.02 ✅ |
| Drug naming, all 40 treatment questions | 72.5% | **100%** | ≥ G0 ✅ |

- H1 uses G3 only when the planner labels a question "treatment_choice" (16% of the seen test, 21% of fresh set 2). Everything else keeps G0's behaviour.
- H1 therefore names medicines whenever it is asked for them, **without** G3's extra contradictions on other questions.
- On real questions that *mention* treatment by keyword, drug naming stays at 0.80 (as G0). Some of them ask about dosing or interactions, and the planner routes those to G0.
""",
r"""
## Limitations
- **Rounds 2 and 3 are post-hoc** with respect to round 1. The protection is the fresh question sets (one per round), built from the corpus vocabulary; they have no human references and are scored by CliniVer's own verifier (baseline v2), whose errors they inherit.
- **Reference statements:** most come from MedLFQA (LLM-generated); only K-QA golden has physician statements. The NLI judge adds noise (about half of the inspected round-1 contradictions were artefacts).
- **Routing depends on the planner's intent label.** A treatment question phrased as dosing or interaction goes to G0.
- **Small question sets** (58–136 per comparison), one seed, one LLM (Qwen3-8B); no human rating of answer quality yet.

## Conclusion and decision
- **Conclusion:** grounding alone does not make an answer *answer the question*. Retrieval has to be steered towards what was asked (planned queries, drug-class vocabulary), and the prompt has to demand a direct answer. Applying that only to treatment questions keeps the safety profile of the plain strategy.
- **Decision (ADR-0018):** the service answers with **H1**: G3 for treatment-choice questions, G0 otherwise. Every generated answer is still verified claim by claim before it is shown.
- **Next:**
  - expert review of generated answers;
  - unit-aware numeric checking (a "g/L vs mg/dL" error passed as Supported in a smoke test);
  - a better intent classifier for routing.
""",
]
