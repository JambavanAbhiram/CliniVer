"""Discussion of NB08 Part C (EXP-08-003), written AFTER execution from its outputs."""

DISCUSSION_CELLS = [
r"""
## 27. Discussion: latency (EXP-08-003)
All figures are measured live on this laptop (`artifacts/experiments/EXP-08-003/`).

| Lever | Measurement | Quality / budget check | Decision |
|---|---|---|---|
| L1 Qwen3-0.6B draft (8 / 16 tokens) | extraction 20.5 s → 9.96 / 9.85 s (**2.06× / 2.09×**); draft acceptance 0.76 / 0.61 | claim F1 0.978 / 0.990; **GPU 6,489 MB > 6,144 MB budget** | rejected (budget) |
| L1 n-gram speculation | 20.5 s → **11.35 s (1.81×)**; p95 28.0 → 14.8 s | claim F1 **1.000**; 72.5% byte-identical outputs; GPU unchanged (5,339 MB) | **adopted** |
| L2 re-rank pool 50 → 20 | 236 → **95 ms** per claim | dev macro-F1 Δ +0.0002; top-1 unchanged for 90% of claims | **adopted** |
| L3 batched retrieval | 1.75 → 0.90 s per answer (1.95×) | top-1 identical for **98.35%** (< 99% rule) | rejected |
| L4 warm server, PyTorch resident | removes ~3 s per server start | joint GPU peak **7,641 MB** above idle (> budget) | rejected |
| L5 second opinion 160 tokens | 2.69 → 2.69 s | label agreement 0.853 | rejected |

**End to end** (service engine, 20 fresh test answers, 152–154 claims, every call live):

| | Baseline settings | Adopted levers |
|---|---|---|
| Per answer, median | 41.2 s | **30.4 s (1.36×)** |
| Per answer, p95 | 54.8 s | **38.0 s** |
| Extraction + rewrites (median) | 27.8 s | 18.0 s |
| Retrieval + verification (median) | 2.23 s | 1.01 s |
| Second opinion (median) | 11.1 s | 10.6 s |

- **Agreement:** 85% of answers yield identical claims under both settings; on those, 91% of verdicts are identical.
- **Speculative decoding is not bit-exact on this GPU** (its batched verification changes floating-point order), so verdicts can change. The pre-registered checks bound the quality effect: claim F1 1.000; dev macro-F1 unchanged.
- **The remaining cost** is now the second opinion (~11 s per answer; 40% of claims routed) and the LLM-server starts (one or two per answer, ~3 s each). The GPU budget prevents keeping everything resident.
""",
r"""
## 28. Limitations, conclusion and decision (Part C)
**Limitations:**
- One laptop and one run per variant; dev sets of 40 answers / 162 claims.
- The GPU figures were measured in the notebook process: they include its CUDA context and are an upper bound for the standalone service.
- The rules are strict by design: L3 (98.35% identical top-1) and the draft model (6.49 GB) missed their thresholds narrowly. They are **not** adopted after the fact.

**Conclusion:**
- Two levers, n-gram speculation and the smaller re-rank pool, make the full verification **1.36× faster per answer** (median 41 → 30 s; p95 55 → 38 s), with the quality checks passed.
- Claim extraction alone is 1.8× faster, and re-ranking 2.5× faster.

**Decision (ADR-0016):** the service uses `--spec-type ngram-simple` and re-rank pool 20. Round-2 latency candidates, each needing its own measurement:
1. fp32 re-ranking, to make batched retrieval bit-stable (L3);
2. a smaller draft context, to bring the draft model within budget;
3. a cheaper second-opinion route, e.g. only for claims the verifier would label SUP.
""",
]
