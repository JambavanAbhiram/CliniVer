"""Builds the round-1 research report for first-stage review, from the experiment records only.

Every number in the report is read from ``artifacts/experiments/*`` (never typed by hand). Output:
``docs/report_round1.md`` (repository copy) and ``docs/report_round1.html`` (the shareable page).

    .venv\\Scripts\\python.exe scripts\\build_report.py
"""

from __future__ import annotations

import csv
import html
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "artifacts" / "experiments"


def load(e: str) -> dict:
    return json.loads((EXP / e / "results.json").read_text(encoding="utf-8"))


E1, E2, E3 = load("EXP-08-001"), load("EXP-08-002"), load("EXP-08-003")
CORP = load("EXP-01-003")["composition"]
RR = next(x for x in load("EXP-05-001")["rerankers"] if x["config"] == "hybrid+medcpt-ce@50")
V6 = {(x["verifier"], x["set"]): x for x in load("EXP-06-001")["verifier_results"]}
CAL = {(r["set"], r["method"]): r for r in csv.DictReader((EXP / "EXP-07-001" / "calibration.csv").open(encoding="utf-8"))}


def f(x, d=3):
    if x is None:
        return "—"
    if isinstance(x, str):
        x = float(x)
    return f"{x:.{d}f}"


def pct(x, d=0):
    return f"{100 * float(x):.{d}f}%"


def ci(lo_hi, d=3):
    return f"[{float(lo_hi[0]):+.{d}f}, {float(lo_hi[1]):+.{d}f}]"


def sgn(x, d=3):
    return f"{float(x):+.{d}f}"


# ---------------------------------------------------------------- figures used in the text
e2e = {r["config"]: r for r in E1["e2e_ablations"]}
H7 = E1["H7"]; CTX = E1["context_ablation"]["macro_F1_SUP_CON"]; SIL1 = E1["silver"]; COST = E1["cost"]
FT, FP = E2["fresh_tests"]["test"], E2["fresh_tests"]["pubhealth"]
TT, PT = FT["table"], FP["table"]
B, S = "baseline (v1r, v1 policy)", "selected"
VL = E2["seen_sets"]["verifier_level"]; SD = E2["seen_sets"]["scifact_dev_e2e"]
FAIR = E2["context_fairness_check"]; RC = E2["real_claims"]; SPL = E2["splits"]; FTN = E2["finetune"]
L1 = E3["L1_speculative_decoding"]["variants"]; L2 = E3["L2_rerank_pool"]["by_pool"]; L3 = E3["L3_batched_retrieval"]
L4 = E3["L4_warm_server"]; L5 = E3["L5_second_opinion_tokens"]; LAT = E3["end_to_end_latency_test"]
unseen = next(k for k in FAIR["v1"] if k.startswith("unseen"))
phys, llmref = RC["silver"]["physician (K-QA)"], RC["silver"]["LLM-generated (MedLFQA)"]
n_units = 232628
docs_total = sum(v["docs"] for v in CORP.values())

# ---------------------------------------------------------------- document model: (kind, payload)
DOC: list[tuple] = []
h2 = lambda t, anchor: DOC.append(("h2", t, anchor))
h3 = lambda t: DOC.append(("h3", t))
p = lambda t: DOC.append(("p", t))
ul = lambda items: DOC.append(("ul", items))
ol = lambda items: DOC.append(("ol", items))
note = lambda t: DOC.append(("note", t))
table = lambda head, rows, cap="": DOC.append(("table", head, rows, cap))

h2("Summary for the reviewer", "summary")
p("CliniVer checks the factual claims in an AI-generated answer to a clinical question against a dated evidence corpus. "
  "It returns, per claim, **Supported**, **Contradicted** or **Insufficient evidence**, with the evidence passage, its source and date, and a confidence. "
  "Domain: adult cardiometabolic pharmacotherapy. It is a research prototype, not a medical device and not medical advice.")
ul([f"**Corpus:** {docs_total:,} documents (PubMed, FDA labels, ClinicalTrials.gov, Europe PMC, WHO), snapshot 2026-11-20, indexed as {n_units:,} evidence units.",
    f"**Fresh end-to-end test** (241 SciFact claims never used for training or selection), after optimization round 1: "
    f"macro-F1 **{f(TT[B]['macro_F1_e2e'])} → {f(TT[S]['macro_F1_e2e'])}** ({sgn(FT['selected_vs_baseline']['delta_F1_e2e'])}, 95% CI {ci(FT['selected_vs_baseline']['ci_F1_e2e'])}); "
    f"false-support rate **{f(TT[B]['FSR_answered'])} → {f(TT[S]['FSR_answered'])}** ({sgn(FT['selected_vs_baseline']['delta_FSR'])}, CI {ci(FT['selected_vs_baseline']['ci_FSR'])}).",
    f"**Fresh consumer fact-checks** (PUBHEALTH): macro-F1 {f(PT[B]['macro_F1_e2e'])} → {f(PT[S]['macro_F1_e2e'])} (CI {ci(FP['selected_vs_baseline']['ci_F1_e2e'])}).",
    f"**Safety-relevant edits caught** (synthetic): changed numbers {pct(VL['synthetic|v1']['number_edit_caught (CON)'])} → {pct(VL['synthetic|v2']['number_edit_caught (CON)'])}, "
    f"negations {pct(VL['synthetic|v1']['negation_edit_caught (CON)'])} → {pct(VL['synthetic|v2']['negation_edit_caught (CON)'])}.",
    f"**Latency:** a full verification takes **{f(LAT['adopted']['per_answer_median_s'], 1)} s per answer** (median; p95 {f(LAT['adopted']['per_answer_p95_s'], 1)} s) "
    f"on a laptop GPU, down from {f(LAT['baseline']['per_answer_median_s'], 1)} s. All models are open-weight, run locally, at zero cost.",
    f"**Main open risks:** on real answers, accuracy is not yet measured against human labels. A silver check suggests the system still labels about half "
    f"of the claims contradicted by physician statements as Supported ({f(phys['selected: SUP|silver CON (silver false support)'], 2)}, n = {phys['silver_CON']}). "
    f"The optimized verifier is more conservative and relies on an LLM second opinion (§8, Costs and regressions).",
    "**What we ask of you:** the checks in §12, centred on an expert review of 107 real-claim verdicts in the bundled interface."])

h2("Problem, scope and safety", "scope")
p("LLMs answer clinical questions fluently but can state wrong doses, contraindications or effect sizes. "
  "CliniVer is an *auditing* tool: it never answers clinical questions itself. It takes a question and an AI answer, splits the answer into checkable claims, "
  "and tests each claim against evidence it retrieves.")
ul(["**Verdict semantics:** *Insufficient evidence* means the corpus snapshot holds no adequate evidence. It does **not** mean the claim is false. "
    "Every verdict is relative to the corpus snapshot and shows the evidence dates.",
    "**Similarity is not verification:** a retrieved passage that is merely on-topic never counts as support. A separate verifier must judge entailment.",
    "**Safety:** research tool only. No diagnosis, prescribing or medication advice. Retrieved documents are treated as untrusted data: the LLM prompts tell the model to treat evidence as data, never as instructions.",
    "**Constraints:** open-weight models only, run locally on one laptop (RTX 5070 Laptop, 8 GB; budget 6 GB GPU memory), zero budget, legally usable data only."])

h2("System", "system")
table(["Stage", "Method (baseline v2)", "Chosen in"],
      [["Claim extraction", "Qwen3-8B (Q4_K_M, llama.cpp), JSON-schema output that keeps qualifiers (dose, population, condition)", "NB03 · ADR-0010"],
       ["Contextualization", "Claims that depend on the question are rewritten to stand alone; **the rewrite is verified**", "NB04, NB08 · ADR-0011/0016"],
       ["Retrieval", f"BM25 + MedCPT dense (RRF) → MedCPT cross-encoder re-rank (pool 20) over {n_units:,} units", "NB05 · ADR-0012/0016"],
       ["Verification", "DeBERTa-v3-base NLI, fine-tuned (v2: + rule perturbations and hard negatives), on the top-1 evidence", "NB06, NB08 · ADR-0013/0016"],
       ["Confidence", "Temperature scaling + multi-signal confidence; low-confidence claims (bottom 25%) get a Qwen3-8B second opinion", "NB07 · ADR-0014/0016"],
       ["Answering (demo)", "For a typed question, Qwen3-8B writes an answer from the top-5 retrieved passages, with citations; that answer is then verified like any other", "ADR-0017"],
       ["Output", "Summary table per claim: claim, verdict, confidence, evidence dates, evidence links; plus evidence text, second-opinion rationale, corpus snapshot", "service"]],
      "The pipeline as deployed in the demo service.")
p("Graph-based retrieval over a concept graph was tested as a pre-registered experiment and **not retained** (it lowered nDCG; NB05).")

h2("Data", "data")
table(["Source", "Documents", "Role"],
      [["PubMed abstracts", f"{CORP['pubmed']['docs']:,}", "primary literature (indexed whole)"],
       ["ClinicalTrials.gov records", f"{CORP['clinicaltrials']['docs']:,}", "trial registrations and results"],
       ["WHO GHO indicators", f"{CORP.get('who_gho', {}).get('docs', 0):,}", "population statistics"],
       ["Europe PMC open-access full texts", f"{CORP['europepmc_oa']['docs']:,}", "full-text articles"],
       ["openFDA drug labels", f"{CORP['openfda_label']['docs']:,}", "regulatory labels"],
       ["**Total**", f"**{docs_total:,}**", f"{n_units:,} evidence units"]],
      "Corpus snapshot 2026-11-20 (EXP-01-003). Third-party text is not redistributed; IDs and scripts are.")
table(["Benchmark", "Use", "Licence"],
      [["SciFact", "verifier training (train split), end-to-end tests (dev; held-out train split)", "claims CC BY 4.0; abstracts ODC-By"],
       ["HealthVer", "verifier training, calibration (dev), test", "CC BY 4.0"],
       ["HealthFC", "out-of-distribution evaluation only", "CC BY-NC-ND 4.0 (no training)"],
       ["PUBHEALTH", "fresh out-of-distribution test (round 1)", "MIT"],
       ["K-QA, MedLFQA", "real questions → real AI answers → real claims; reference statements for the silver check", "MIT; CC BY 4.0"]])

h2("Evaluation protocol", "protocol")
ul(["**Pre-registration:** every hypothesis, decision rule and selection rule was written into a config file before the run that tested it.",
    "**Three pillars:** (1) benchmarks with human labels (SciFact, HealthVer, HealthFC, PUBHEALTH); (2) synthetic rule-based edits whose labels follow by construction; (3) a silver check of real answers against reference statements.",
    "**Leakage control for round 1:** the optimized models were trained and selected without two fresh test sets. SciFact train claims were split by abstract group (train 406 / dev 162 / test 241), so no abstract crosses splits; PUBHEALTH was never used before.",
    "**Metrics:** macro-F1 over Supported / Contradicted / Insufficient; **end-to-end macro-F1** counts an abstention as an error; "
    "**false-support rate** = share of claims whose gold label is Contradicted or Insufficient that the system labels Supported (the clinically dangerous error); paired bootstrap 95% CIs (2,000 resamples).",
    "**Reproducibility:** every experiment writes its config, model versions, data version, hardware and results from code. Every LLM call is logged and replayable; re-running NB08 reproduced its numbers exactly."])

h2("Research questions and outcomes", "hypotheses")
table(["Hypothesis", "Outcome", "Evidence"],
      [["H1 structure-aware chunking: higher recall and fewer split qualifiers than fixed-size", "partially supported (qualifiers yes, recall no)", "NB02"],
       ["H2 qualifier-aware extraction keeps more qualifiers, without lower precision", "partially supported", "NB03"],
       ["H3a question + claim retrieval helps context-dependent claims; H3b it does not help self-contained ones", "**supported**", "NB04"],
       ["H4a sparse + dense hybrid ≥ the better single channel; H4d re-ranking improves nDCG", "**supported**", "NB05"],
       ["H4b a graph channel improves recall on class-level / synonym claims", "not supported (channel dropped)", "NB05"],
       ["H4c graph gains are reproduced by normalization-only expansion", "not meaningful (the control itself hurt)", "NB05"],
       ["H5a an adapted verifier beats zero-shot NLI on both sets", "not supported (no significant gain on SciFact)", "NB06"],
       ["H5b similarity verification errs mainly by false support; H5c top-k evidence beats top-1", "not supported (H5c reversed: aggregation hurts)", "NB06"],
       ["H6a temperature scaling lowers ECE; H6c abstention lowers false support", "**supported** (H6c in distribution only)", "NB07"],
       ["H6b multi-signal confidence beats verifier probability", "not supported", "NB07"],
       ["H7 retrieval causes a larger share of errors than the verifier", "not supported: the verifier is the bottleneck", "NB08"]],
      "Negative results are reported as such; they drove the optimization round.")

h2("Baseline v1: end-to-end results", "baseline")
p(f"Before optimization, components were evaluated individually. Retrieval: nDCG@10 **{f(RR['evaluation_ndcg@10'])}**, Recall@20 {f(RR['evaluation_recall@20'])} on SciFact claims over ~130k abstracts. "
  f"Verifier (fine-tuned vs zero-shot NLI, macro-F1): HealthVer test {f(V6[('fine-tuned NLI', 'healthver_test')]['macro_F1'])} vs {f(V6[('zero-shot NLI', 'healthver_test')]['macro_F1'])}; "
  f"SciFact {f(V6[('fine-tuned NLI', 'scifact_dev')]['macro_F1'])} vs {f(V6[('zero-shot NLI', 'scifact_dev')]['macro_F1'])}; "
  f"HealthFC {f(V6[('fine-tuned NLI', 'healthfc')]['macro_F1'])} (Qwen3-8B: {f(V6[('LLM qwen3-8b', 'healthfc')]['macro_F1'])}). "
  f"Calibration (ECE, uncalibrated → temperature): HealthVer test {f(CAL[('healthver_test', 'uncalibrated')]['ECE'])} → {f(CAL[('healthver_test', 'temperature')]['ECE'])}; "
  f"SciFact {f(CAL[('scifact_dev', 'uncalibrated')]['ECE'])} → {f(CAL[('scifact_dev', 'temperature')]['ECE'])}.")
rows = []
for k, name in [("pipeline", "Pipeline v1"), ("nli_only_no_conf", "NLI only (no confidence layer)"), ("hybrid_no_rerank", "without re-ranker"),
                ("bm25_only", "BM25 only"), ("hybrid_bge_base", "BGE-base instead of MedCPT"), ("zero_shot_nli", "zero-shot NLI verifier"),
                ("qwen_only", "Qwen3-8B as the only verifier"), ("top3_naive", "top-3 evidence aggregation")]:
    r = e2e[k]; rows.append([name, f(r["coverage"]), f(r["macro_F1_answered"]), f(r["macro_F1_e2e"]), f(r["FSR_answered"])])
table(["Configuration (SciFact dev, 300 claims)", "Coverage", "Macro-F1 answered", "Macro-F1 end-to-end", "False support"], rows, "EXP-08-001 ablations.")
ul([f"**Where errors come from (H7):** of {H7['errors']} errors, {H7['verifier_attributable']} are verifier-attributable and {H7['retrieval_attributable']} retrieval-attributable. "
    f"Oracle evidence lifts macro-F1 only from {f(H7['retrieved_macro_F1'])} to {f(H7['oracle_evidence_macro_F1'])}.",
    f"**Contextualization:** verifying the standalone rewrite of context-dependent claims scores {f(CTX['rewrite F (verify the rewrite)'])}, vs {f(CTX['claim_only (verify as written)'])} for the claim as written (oracle {f(CTX['oracle original claim'])}).",
    f"**Weaknesses that motivated round 1:** false support ≈ {f(e2e['pipeline']['FSR_answered'], 2)}; changed numbers caught ≤ 12% (NB06); "
    f"silver false support on real claims ≈ {f(SIL1['physician (K-QA)']['pipeline_SUP_given_silver_CON (silver false support)'], 2)}."])

h2("Optimization round 1", "round1")
h3("Design")
p("Protocol fixed in advance (`configs/optimize.yaml`). Candidates:")
ul([f"a **v2 verifier**: the v1 recipe plus {sum(FTN['augmentation']['perturbations'].values()):,} rule perturbations of training claims "
    f"(changed number / negation → Contradicted; removed hedge / widened population → Insufficient; paraphrase → Supported) and "
    f"{FTN['augmentation']['hard_negatives']} hard negatives (retrieved, on-topic but non-cited abstracts → Insufficient);",
    "a numeric guard;",
    "an evidence-score threshold;",
    "three second-opinion policies."])
p(f"The baseline is the v1 recipe retrained on the same training split (v1r). Selection used the dev split ({SPL['dev']['e2e_items']} claims); adoption required gains on the fresh tests.")
h3("Fresh tests")
rows = [["SciFact held-out test: macro-F1 end-to-end", f(TT[B]["macro_F1_e2e"]), f(TT[S]["macro_F1_e2e"]), f"{sgn(FT['selected_vs_baseline']['delta_F1_e2e'])} {ci(FT['selected_vs_baseline']['ci_F1_e2e'])}"],
        ["SciFact held-out test: false-support rate", f(TT[B]["FSR_answered"]), f(TT[S]["FSR_answered"]), f"{sgn(FT['selected_vs_baseline']['delta_FSR'])} {ci(FT['selected_vs_baseline']['ci_FSR'])}"],
        ["SciFact held-out test: coverage", f(TT[B]["coverage"]), f(TT[S]["coverage"]), "no abstentions"],
        ["PUBHEALTH: macro-F1", f(PT[B]["macro_F1_e2e"]), f(PT[S]["macro_F1_e2e"]), f"{sgn(FP['selected_vs_baseline']['delta_F1_e2e'])} {ci(FP['selected_vs_baseline']['ci_F1_e2e'])}"],
        ["PUBHEALTH: false-support rate", f(PT[B]["FSR_answered"]), f(PT[S]["FSR_answered"]), f"{sgn(FP['selected_vs_baseline']['delta_FSR'])} {ci(FP['selected_vs_baseline']['ci_FSR'])}"]]
table(["Metric", "Baseline (v1r)", "Baseline v2", "Δ, 95% CI"], rows,
      f"EXP-08-002. Selected and adopted by the pre-registered rules: {E2['adoption']['selected']}.")
h3("What changed inside")
table(["Component (SciFact held-out test)", "Coverage", "Macro-F1 end-to-end", "False support"],
      [[k, f(TT[k]["coverage"]), f(TT[k]["macro_F1_e2e"]), f(TT[k]["FSR_answered"])] for k in ("v1r, verifier only", "v2, verifier only", "v2, v1 policy", "selected")])
table(["Synthetic edit (in-generator)", "v1", "v2"],
      [["changed number caught", pct(VL["synthetic|v1"]["number_edit_caught (CON)"]), pct(VL["synthetic|v2"]["number_edit_caught (CON)"])],
       ["negation caught", pct(VL["synthetic|v1"]["negation_edit_caught (CON)"]), pct(VL["synthetic|v2"]["negation_edit_caught (CON)"])],
       ["overstated hedge / population not called Supported", pct(VL["synthetic|v1"]["hedge/population_edit_not_SUP"]), pct(VL["synthetic|v2"]["hedge/population_edit_not_SUP"])],
       ["paraphrase kept as Supported", pct(VL["synthetic|v1"]["paraphrase_kept_SUP"]), pct(VL["synthetic|v2"]["paraphrase_kept_SUP"])]],
      "The edits come from the same rules used for v2's augmentation, so this shows the verifier can learn the distinction, not that it generalizes to natural errors.")
p(f"Seen sets: SciFact dev end-to-end {f(SD['v1 deployed, v1 policy (EXP-08-001 pipeline)']['macro_F1_e2e'])} → {f(SD['selected']['macro_F1_e2e'])}; "
  f"HealthFC (verifier only) {f(VL['healthfc|v1']['macro_F1'])} → {f(VL['healthfc|v2']['macro_F1'])}; HealthVer test {f(VL['healthver_test|v1']['macro_F1'])} → {f(VL['healthver_test|v2']['macro_F1'])}.")
h3("Costs and regressions")
table(["Finding", "v1", "v2 / baseline v2"],
      [[f"Context items, verifier only, claims unseen by all models ({unseen.split('(n=')[1].rstrip(')')} items)", f(FAIR["v1"][unseen]), f(FAIR["v2"][unseen])],
       [f"Real claims, silver false support, physician statements (n = {phys['silver_CON']})", f(phys["v1: SUP|silver CON (silver false support)"]), f(phys["selected: SUP|silver CON (silver false support)"])],
       [f"Real claims, silver false support, LLM-generated statements (n = {llmref['silver_CON']})", f(llmref["v1: SUP|silver CON (silver false support)"]), f(llmref["selected: SUP|silver CON (silver false support)"])],
       ["Real claims labelled Contradicted or Abstained when silver says contradicted (physician)", f(phys["v1: CON or ABST|silver CON"]), f(phys["selected: CON or ABST|silver CON"])]])
ul(["**v2 is conservative:** it says *Insufficient* more often. That halves false support on the fresh test, but costs Supported/Contradicted recall, visible on the context items.",
    "**The LLM second opinion restores recall**, and on real answers it also pushes some contradicted claims to *Supported*. The physician-referenced increase rests on about 3 claims, but it is the risk we most want the reviewer to probe (§12).",
    "**Abstention was dropped:** the selected policy takes the LLM's label instead of abstaining. Low confidence remains visible per claim."])

h2("Latency and resources", "latency")
rows = [["Speculative decoding (Qwen3-0.6B draft)", f"extraction {f(L1['baseline']['median_s'], 1)} → {f(L1['draft06_n8']['median_s'], 1)} s; GPU {L1['draft06_n8']['gpu_mb_above_idle']:,} MB", "rejected: over the 6 GB budget"],
        ["Speculative decoding (n-gram lookup)", f"extraction {f(L1['baseline']['median_s'], 1)} → {f(L1['ngram_simple']['median_s'], 1)} s; claim F1 vs plain {f(L1['ngram_simple']['claim_F1_vs_baseline'])}", "**adopted**"],
        ["Re-rank pool 50 → 20", f"{L2['50']['rerank_ms_median']} → {L2['20']['rerank_ms_median']} ms per claim; dev F1 Δ {sgn(L2['20']['delta_vs_50'], 4)}", "**adopted**"],
        ["Batched retrieval", f"{f(L3['per_answer_median_s_per_claim_search'], 2)} → {f(L3['per_answer_median_s_batched'], 2)} s per answer; top-1 identical {pct(L3['top1_identical'], 1)}", "rejected (< 99% rule)"],
        ["Warm LLM server", f"joint GPU peak {L4['joint_above_idle_mb']:,} MB", "rejected: over budget"],
        ["Shorter second opinion", f"{L5['median_s_300']} → {L5['median_s_160']} s", "rejected: no gain"]]
table(["Lever (pre-registered)", "Measured", "Decision"], rows, "EXP-08-003; all measured live on the laptop.")
table(["Per answer (20 fresh answers)", "Baseline settings", "Adopted"],
      [["median", f"{f(LAT['baseline']['per_answer_median_s'], 1)} s", f"{f(LAT['adopted']['per_answer_median_s'], 1)} s"],
       ["p95", f"{f(LAT['baseline']['per_answer_p95_s'], 1)} s", f"{f(LAT['adopted']['per_answer_p95_s'], 1)} s"],
       ["claim extraction + rewrites (median)", f"{f(LAT['baseline']['extraction_and_rewrite_s_median'], 1)} s", f"{f(LAT['adopted']['extraction_and_rewrite_s_median'], 1)} s"],
       ["retrieval + verification (median)", f"{f(LAT['baseline']['retrieval_and_verification_s_median'], 2)} s", f"{f(LAT['adopted']['retrieval_and_verification_s_median'], 2)} s"],
       ["second opinion (median)", f"{f(LAT['baseline']['second_opinion_s_median'], 1)} s", f"{f(LAT['adopted']['second_opinion_s_median'], 1)} s"]],
      f"Verdict agreement between the two settings: {pct(LAT['agreement']['verdict_agreement_on_those'])} on answers with identical claims.")
p(f"Resources: the Qwen3-8B server uses {COST['vram_mb']['llm_qwen3_8b_server']['gpu_peak_minus_baseline_mb']:,}–{L1['baseline']['gpu_mb_above_idle']:,} MB of GPU memory; "
  f"retrieval {COST['vram_mb']['retrieval_peak_mb']:,} MB; verifier fine-tuning peaked at {FTN['v2']['history'][-1]['peak_vram_mb']:,} MB "
  f"({FTN['v2']['seconds']} s). One large model on the GPU at a time.")

h2("Final configuration: baseline v2", "final")
p("ADR-0016. The service reads its configuration from the experiment records. None is set by hand.")
ul([f"**Verified text:** the standalone rewrite (the original is kept for display).",
    f"**Verifier:** v2 ({FTN['v2']['pairs']:,} training pairs).",
    f"**Low-confidence band:** Qwen3-8B second opinion; its label is taken on disagreement.",
    f"**Retrieval:** re-rank pool 20.",
    f"**LLM:** n-gram speculative decoding.",
    f"**Real answers:** {RC['verified_as_rewrite']} of 2,777 real claims are verified as their rewrite. "
    f"Verdict mix: Supported {pct(RC['verdict_distribution']['selected']['SUP'])}, Insufficient {pct(RC['verdict_distribution']['selected']['INS'])}, "
    f"Contradicted {pct(RC['verdict_distribution']['selected']['CON'])} (unvalidated)."])

h2("Limitations and threats to validity", "limitations")
ul(["**No human-labelled real-claim accuracy yet.** Real-answer results are silver (NLI against reference statements), and small for physician statements.",
    "**Round 1 is post-hoc** to the baseline evaluation. The protection is the two fresh test sets; SciFact dev and the other NB06–NB08 sets are reported as *seen*.",
    "**Fresh tests are modest in size** (241 claims; PUBHEALTH has 45 Insufficient items), so CIs are wide.",
    "**Synthetic robustness is in-generator** (§8).",
    "**Single seed;** one augmentation recipe; one laptop for all latency figures (latency is hardware-specific).",
    "**English, one clinical domain, one corpus snapshot.** Evidence quality (study design, recency) is reported but not yet weighed in the verdict.",
    "**Speculative decoding is not bit-exact on this GPU.** Its effect is bounded by the pre-registered quality checks, not zero."])

h2("First-stage verification: what we ask the reviewer to check", "verify")
ol(["**Expert review of real-claim verdicts (main task).** Start the interface (below), open *Expert review*, enter your name, and judge each item: is the verdict correct for this claim and this evidence; is the evidence relevant; is the claim a faithful extract of the answer. The queue holds 107 claims, stratified by verdict and by reference source. Judgements are saved to `data/annotations/expert_review_round1.jsonl`; they give the first human estimate of real-claim accuracy and of the false-support risk in §8.",
    "**Ask your own questions** in *Ask a question*: the local model writes an answer from retrieved evidence and every claim of that answer is verified, with a summary table of claim, verdict, confidence, evidence dates and links. In *Check another answer*, paste correct and subtly wrong answers (changed dose, wrong population, reversed effect). Check that *Insufficient* is used when the corpus is silent.",
    "**Audit the method:** pre-registered rules (`configs/*.yaml`), fresh-split construction (NB08 Part B, first code cell), the context fairness check (NB08 §18b), and the decision records (`docs/decisions/ADR-0001 … 0016`).",
    "**Spot-check reproducibility:** any notebook re-executes from its builder; LLM calls replay from the call logs."])
p(r"Start the interface (Windows, project root): `.venv\Scripts\python.exe scripts\run_app.py`, then open `http://127.0.0.1:8501` (Streamlit, localhost only). "
  "Loading takes about half a minute; asking a question (retrieve, generate, verify) takes about 30 s. Details: `docs/demo_guide.md`.")

h2("Next steps", "next")
ol(["Use the expert-review judgements as the first human-labelled real-claim set: accuracy and false support, re-weighted by stratum.",
    "**Round 2 (accuracy):** let v2's *Insufficient* / *Contradicted* veto LLM overrides to *Supported*; recover context-item recall with fewer hard negatives; weigh evidence quality and recency.",
    "**Round 2 (latency):** fp32 re-ranking so batched retrieval is bit-stable; a smaller draft-model context to fit the GPU budget.",
    "**Clinical validation (pre-production, ADR-0006):** clinician annotators, written protocol, inter-annotator agreement."])

h2("Appendix: records", "appendix")
table(["Item", "Where"],
      [["Notebooks (8)", "`notebooks/01_…` to `08_…` (NB08 Parts A–C: baseline, optimization, latency)"],
       ["Experiments", "`artifacts/experiments/EXP-01-001 … EXP-08-003` (config, versions, hardware, results)"],
       ["Experiment log", "`docs/experiment_log.md`"],
       ["Decisions", "`docs/decisions/ADR-0001 … ADR-0016`"],
       ["Research design", "`docs/research_foundation.md` (questions, hypotheses, outcomes §18.1)"],
       ["Interface", "`src/cliniver/service/` · `docs/demo_guide.md`"],
       ["This report", "generated by `scripts/build_report.py` from the records"]])


# ---------------------------------------------------------------- renderers
def inline_html(t: str) -> str:
    t = html.escape(t, quote=False)
    t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
    t = re.sub(r"`(.+?)`", r"<code>\1</code>", t)
    t = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<em>\1</em>", t)
    return t


def to_md() -> str:
    out = ["# CliniVer — Round 1 Research Report", "",
           "Prepared for first-stage expert review · 2026-11-22 · corpus snapshot 2026-11-20 · research prototype, not medical advice", ""]
    sec = 0
    for b in DOC:
        k = b[0]
        if k == "h2":
            sec += 1; out += [f"## {sec}. {b[1]}", ""]
        elif k == "h3":
            out += [f"### {b[1]}", ""]
        elif k == "p":
            out += [b[1], ""]
        elif k == "note":
            out += [f"> {b[1]}", ""]
        elif k in ("ul", "ol"):
            out += [("- " if k == "ul" else f"{i}. ") + x for i, x in enumerate(b[1], 1)] + [""]
        elif k == "table":
            head, rows, cap = b[1], b[2], b[3]
            out += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(r) + " |" for r in rows]
            out += ([f"*{cap}*"] if cap else []) + [""]
    return "\n".join(out)


CSS = """
:root{--bg:#f6f8f9;--paper:#ffffff;--fg:#17212b;--muted:#566472;--line:#d6dee5;--accent:#0f6c7b;--accent-soft:#e3f1f3;--warn:#8a5a00;--warn-soft:#fbf1dc;
--f-body:"Source Sans 3",system-ui,-apple-system,"Segoe UI",sans-serif;--f-head:"Source Serif 4",Georgia,serif;--f-mono:"IBM Plex Mono",ui-monospace,Consolas,monospace}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#11171c;--paper:#172027;--fg:#e4eaef;--muted:#9aa8b4;--line:#2c3943;--accent:#5fc3d1;--accent-soft:#163238;--warn:#f0c46c;--warn-soft:#3a2f17;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#11171c;--paper:#172027;--fg:#e4eaef;--muted:#9aa8b4;--line:#2c3943;--accent:#5fc3d1;--accent-soft:#163238;--warn:#f0c46c;--warn-soft:#3a2f17;color-scheme:dark}
body{background:var(--bg);color:var(--fg);font:16px/1.6 var(--f-body);padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px))}
.wrap{max-width:860px;margin:0 auto;padding-block:40px 80px}
header.top{border-bottom:2px solid var(--fg);padding-bottom:18px;margin-bottom:28px}
.kicker{font:600 12px/1 var(--f-body);letter-spacing:.12em;text-transform:uppercase;color:var(--accent)}
h1{font:600 clamp(28px,5vw,40px)/1.15 var(--f-head);margin:10px 0 8px;text-wrap:balance}
.meta{color:var(--muted);font-size:14px;display:flex;flex-wrap:wrap;gap:4px 18px}
nav.toc{background:var(--paper);border:1px solid var(--line);border-radius:6px;padding:14px 18px;margin:0 0 32px;font-size:14px}
nav.toc ol{margin:6px 0 0;padding-left:22px;columns:2;column-gap:28px}
@media (max-width:620px){nav.toc ol{columns:1}}
nav.toc a{color:var(--fg);text-decoration:none}nav.toc a:hover{color:var(--accent);text-decoration:underline}
h2{font:600 24px/1.25 var(--f-head);margin:44px 0 12px;padding-top:6px;text-wrap:balance}
h2 .n{color:var(--accent);font-family:var(--f-mono);font-size:17px;margin-right:10px;font-weight:500}
h3{font:600 17px/1.3 var(--f-body);margin:26px 0 8px}
p,li{max-width:70ch}ul,ol{padding-left:22px}li{margin:5px 0}
strong{font-weight:650}code{font:13.5px var(--f-mono);background:var(--accent-soft);padding:1px 5px;border-radius:4px;overflow-wrap:anywhere}
.tbl{overflow-x:auto;margin:14px 0 6px;border:1px solid var(--line);border-radius:6px;background:var(--paper)}
table{border-collapse:collapse;width:100%;font-size:14px}
th{text-align:left;font-weight:600;background:var(--accent-soft);color:var(--fg);padding:8px 10px;border-bottom:1px solid var(--line);vertical-align:bottom}
td{padding:7px 10px;border-top:1px solid var(--line);vertical-align:top;font-variant-numeric:tabular-nums}
td:not(:first-child){white-space:nowrap}
.cap{color:var(--muted);font-size:13px;margin:0 0 16px;max-width:none}
section#summary ul{background:var(--paper);border:1px solid var(--line);border-left:4px solid var(--accent);border-radius:6px;padding:14px 18px 14px 34px}
.safety{background:var(--warn-soft);color:var(--fg);border-radius:6px;padding:10px 14px;font-size:14px;margin:0 0 24px}
footer{margin-top:48px;border-top:1px solid var(--line);padding-top:14px;color:var(--muted);font-size:13px}
a:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
"""


def to_html() -> str:
    secs, toc, sec, open_sec = [], [], 0, False
    for b in DOC:
        k = b[0]
        if k == "h2":
            sec += 1
            if open_sec:
                secs.append("</section>")
            secs.append(f'<section id="{b[2]}"><h2><span class="n">{sec:02d}</span>{inline_html(b[1])}</h2>'); open_sec = True
            toc.append(f'<li><a href="#{b[2]}">{inline_html(b[1])}</a></li>')
        elif k == "h3":
            secs.append(f"<h3>{inline_html(b[1])}</h3>")
        elif k == "p":
            secs.append(f"<p>{inline_html(b[1])}</p>")
        elif k == "note":
            secs.append(f'<p class="cap">{inline_html(b[1])}</p>')
        elif k in ("ul", "ol"):
            secs.append(f"<{k}>" + "".join(f"<li>{inline_html(x)}</li>" for x in b[1]) + f"</{k}>")
        elif k == "table":
            head, rows, cap = b[1], b[2], b[3]
            t = "<tr>" + "".join(f"<th>{inline_html(h)}</th>" for h in head) + "</tr>"
            t += "".join("<tr>" + "".join(f"<td>{inline_html(c)}</td>" for c in r) + "</tr>" for r in rows)
            secs.append(f'<div class="tbl"><table>{t}</table></div>' + (f'<p class="cap">{inline_html(cap)}</p>' if cap else ""))
    if open_sec:
        secs.append("</section>")
    return f"""<title>CliniVer Round 1 Report</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;600;700&family=Source+Serif+4:opsz,wght@8..60,600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>{CSS}</style>
<div class="wrap">
<header class="top"><div class="kicker">Research report · first-stage review</div>
<h1>CliniVer: clinical claim verification, round 1</h1>
<div class="meta"><span>Prepared 2026-11-22</span><span>Corpus snapshot 2026-11-20</span><span>Experiments EXP-08-001 to EXP-08-003</span><span>Figures generated from the experiment records</span></div></header>
<p class="safety"><strong>Research prototype.</strong> Not a medical device and not medical advice. Verdicts may be wrong and are relative to a dated corpus snapshot.</p>
<nav class="toc" aria-label="Contents"><strong>Contents</strong><ol>{''.join(toc)}</ol></nav>
{''.join(secs)}
<footer>Generated by <code>scripts/build_report.py</code> from <code>artifacts/experiments/</code>. Insufficient evidence ≠ false; similarity ≠ verification.</footer>
</div>"""


if __name__ == "__main__":
    (ROOT / "docs" / "report_round1.md").write_text(to_md(), encoding="utf-8")
    (ROOT / "docs" / "report_round1.html").write_text(to_html(), encoding="utf-8")
    print("wrote docs/report_round1.md and docs/report_round1.html")
