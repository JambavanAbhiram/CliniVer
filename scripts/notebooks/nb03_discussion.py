"""Discussion cells for NB03, written AFTER execution from the outputs of EXP-03-001 (results.json and tables)."""

DISCUSSION_CELLS = [
r"""
## 9. Discussion
All figures are copied from the outputs above (`artifacts/experiments/EXP-03-001/`).

### 9.1 Runtime and models
| Model | Peak GPU memory | Tokens/s | Probe JSON | Role outcome |
|---|---|---|---|---|
| Qwen3-8B (Q4_K_M) | 5.3 GB | 51 | valid | generator; **selected extractor** |
| Granite 3.3 8B (Q4_K_M) | 4.8 GB | 51 | **invalid** | **ineligible** extractor (5% valid JSON in schema mode) |
| Qwen3-1.7B (Q8_0) | 2.4 GB | 140 | valid | weaker generator |

- All three models run fully on the GPU within the 6 GB budget (8-bit KV cache, 8k context).
- The full experiment made **1,392 LLM calls** (830k generated tokens, ≈ 5 GPU-hours). Every call is logged and replayable: re-executing this notebook regenerated nothing (16.7 min, almost all of it NLI scoring).
- **Granite in schema-constrained mode** ran into the 3,072-token limit without emitting JSON in 57/60 dev answers (≈ 64 s each), yet produced valid JSON for the simpler `atomic` schema in 60/60. This is a failure of model + grammar together (likely unbounded whitespace that the grammar permits), not of the runtime. Granite remains a candidate for NB06, where outputs are short labels.

### 9.2 Part A: generated answers
- 186 in-domain questions (MedicationQA 92, HealthSearchQA 60, K-QA 24, LiveQA 10) × 2 generators = **372 answers**, none truncated or empty.
- Qwen3-8B answers are longer (median 145 words vs 84).
- From these answers, the selected extractor produced **2,777 claims** (100% valid JSON): 8.6 per Qwen3-8B answer, 6.4 per Qwen3-1.7B answer. 77.9% are marked verifiable; 413 carry a population and 164 a dose.

### 9.3 Part B: extraction (K-QA test, 141 physician answers, Qwen3-8B)
| Method | Claims/answer | Claims/sentence | Words/claim | Precision | Statement recall | Recall (set) | Pronoun start |
|---|---|---|---|---|---|---|---|
| sentence | 4.6 | 1.00 | 19.5 | 0.986 | 0.741 | 0.958 | 10.9% |
| atomic | 10.9 | 2.51 | 11.1 | 0.990 | 0.608 | 0.934 | 3.3% |
| qualified | 4.9 | 1.11 | 18.3 | 0.987 | 0.739 | 0.948 | 8.1% |

- **`qualified` barely decomposes.** Despite the "one fact per claim" instruction, Qwen3-8B keeps claims at near-sentence granularity (1.1 claims per sentence) and mostly copies sentences with pronouns resolved. **`atomic` really decomposes** (2.5 per sentence, 11 words).
- Per-claim statement recall favours sentence-sized claims (physician statements often combine several facts); set recall shows all methods cover the content (0.93–0.96).
- Precision is ≈ 0.99 for every method, and is a **lower bound** (§9.5).

### 9.4 H2: pre-registered result
| Part | Comparison (test) | Difference [95% CI] | Criterion | Result |
|---|---|---|---|---|
| (a) qualifiers | qualified − atomic | +0.008 [0.000, 0.028] | CI > 0 | **not met** |
| (a) qualifiers | qualified − sentence | 0.000 [0.000, 0.000] | not lower | met |
| (b) precision | qualified − atomic | −0.003 [−0.019, 0.008] | lower bound > −0.03 | **met** |

**H2: PARTIALLY SUPPORTED.** The qualifier part failed against `atomic` because the pre-registered heuristic sits at its ceiling (0.992 vs 1.000).

**Exploratory, post-hoc (not part of the verdict).** With the stricter check (§7.1b), the methods separate:

| Qualifier type | Atomic | Qualified | Sentence |
|---|---|---|---|
| Negation | **89.2%** | 100% | 100% |
| Population | **75.0%** | 100% | 100% |
| All | 91.3% | 100% | 100% |

Qualified − atomic: +8.7 points [3.6, 14.3]. Generic atomic decomposition detaches negations and population limits from the facts they qualify, which is exactly the failure mode the qualifier-aware design targets. This is consistent with H2 but was found after seeing the data, so it is a hypothesis for confirmation, not a confirmed result.

### 9.5 How far to trust the NLI judge (§7.3)
On 30 inspected judgements (all 10 that the judge rated *not entailed*, plus 20 it rated *entailed*):
- **"Entailed" was always right (20/20).**
- **"Not entailed" was wrong in 9/10.** Four of these claims were verbatim answer sentences. Five depended on context ("the medication", "the eye"), which a standalone judgement can fairly reject.
- **The one genuine error** was an extraction error that the judge correctly caught: "Cough and malaise can persist for 2 weeks or longer *in most adults*", where the answer says most adults recover within a week.

Consequences:
- Precision values are lower bounds.
- The `qualified` method produced **1 unfaithful claim in 685** test claims.
- One supported claim dropped a safety qualifier ("only safe short-term, not for chronic use" on the maximum ibuprofen dose). Neither automatic metric catches this kind of omission, because the claim is still literally supported.

### 9.6 Reliability of the schema fields
The extractor fills structured fields, but they are not dependable:
- `negated` stays false for "should not be used".
- **56% of the 2,777 response claims are typed `dosing`.** The model uses it as a default (e.g. "Metformin is commonly used to treat type 2 diabetes" → `dosing`).

Downstream notebooks must read qualifiers from the claim **text**, and treat `claim_type` as unreliable until validated.
""",
r"""
## 10. Limitations
- **Reference statements:** K-QA's physician statements were written to evaluate answers, not as a gold atomic decomposition, so granularity differs. Its questions span general medicine, not only our domain.
- **Few qualifiers in the test set:** 127 rule-detected qualifiers in 141 answers. The pre-registered heuristic was too lenient (ceiling); the stricter check is post-hoc.
- **The NLI judge is a model:** it under-calls entailment of context-dependent claims. The spot check was done by the AI developer, not a human or clinician (D5).
- **One seed, one prompt per method:** prompt wording was not tuned or varied, so the poor decomposition by `qualified` may depend on the prompt.
- **Quantization:** Q4_K_M/Q8_0 models may behave differently from full-precision weights.
- **Generated answers are research material**, not reviewed for clinical correctness. They will be verified, never shown as advice.
""",
r"""
## 11. Conclusion and decision
**Conclusion:**
- A local, zero-cost stack works on this laptop: llama.cpp + 4-bit 8B models within 6 GB, schema-constrained JSON at 100% validity (Qwen), with full call logging and exact replay.
- For extraction, Qwen3-8B with the qualifier-aware schema gives the best statement recall at equal precision, **but behaves like sentence splitting with pronoun resolution plus structured fields**.
- Generic atomic decomposition gives finer claims, but (exploratory) detaches negations and population limits.

**H2: PARTIALLY SUPPORTED** (precision part yes; pre-registered qualifier part no, at ceiling; the stricter post-hoc check favours `qualified`).

**Decision (proposed, pending owner approval; ADR-0010):**
- llama.cpp b12560 runtime.
- Qwen3-8B + Qwen3-1.7B as generators.
- **Qwen3-8B `qualified`** as the claim extractor; the 2,777 response claims are the input to NB04.
- Qualifiers are read from claim text, not from schema fields.
- `atomic` outputs stay cached for an NB06 granularity comparison.

**Next steps (NB04, claim contextualization):** use the question and response context to make claims standalone and retrievable. 8–11% of claims still open with an unresolved pronoun, and the extractor sometimes drops the condition a claim depends on.
""",
]
