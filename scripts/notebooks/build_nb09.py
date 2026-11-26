"""Builds notebooks/09_answer_generation.ipynb (owner-approved 9th notebook, 2026-11-22). Discussion: nb09_discussion.py.

Usage: python scripts/notebooks/build_nb09.py notebooks/09_answer_generation.ipynb
"""
import sys
from pathlib import Path

import nbformat as nbf

out = Path(sys.argv[1])
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# Notebook 09 — Evidence-Grounded Answer Generation
**CliniVer** · Experiment `EXP-09-001` · owner-approved dedicated notebook (2026-11-22)

| | |
|---|---|
| **Objective** | Make the service's own answers *answer the question*, grounded in the corpus, and select the generation strategy by evaluation |
| **Trigger** | "What is the medicine for high blood pressure?" was answered without naming a single drug: the question-only retrieval returned general blood-pressure literature, and the prompt did not demand a direct answer |
| **Research question** | **RQ9:** Do LLM query planning, corpus-vocabulary expansion and a direct-answer prompt make grounded answers more complete (reference statements covered) without more contradictions? |
| **Hypothesis (pre-registered)** | **H9:** the selected strategy covers more reference statements than G0 on held-out questions (paired bootstrap), with contradiction rate ≤ G0 + 0.02, and names a drug for treatment questions at least as often as G0 |
| **Protocol** | `configs/generation.yaml`, written before any variant was built or run |
| **Safety** | The answers are research outputs: no diagnosis, no advice to start, stop or change medication; every answer is verified claim by claim before it is shown |
""")

md(r"""
## 1. Method
- **Questions:**
  - the **186 in-domain real questions** of NB03 (K-QA golden with physician statements; MedLFQA with LLM-generated statements), split dev 60 / test 126 (seeded);
  - **20 treatment-intent questions** built from the corpus vocabulary (5 conditions × 4 phrasings; phrasings 0–1 → dev, 2–3 → test).
- **Variants:**

| | Retrieval | Passages | Prompt |
|---|---|---|---|
| **G0** (current service) | the question | top 5 | v1 |
| **G1** | the question | 8, one per document | v2: direct answer first, names drugs when asked |
| **G2** | question + 3 LLM-planned search queries, RRF fusion | 8, one per document | v2 |
| **G3** | G2 + condition → drug-class queries from the corpus vocabulary | 8, one per document | v2 |

- **Metrics:**
  - **reference recall:** share of reference statements entailed by the answer;
  - **reference contradiction:** share of reference statements contradicted by the answer (hallucination proxy, as in K-QA). The NB03 judge is used, threshold 0.5;
  - **drug naming:** on treatment questions;
  - **non-answers;**
  - **latency;**
  - on a test subset, **CliniVer's own verification** of the answer claims.
- **Reference statements:** a question's *must-have* statements, or its *nice-to-have* statements when it has none.
""")

md("## 2. Setup")
code(r"""
import json, os, random, re, time, collections
from pathlib import Path
from collections import Counter
import numpy as np, pandas as pd, matplotlib.pyplot as plt, torch, yaml
from IPython.display import display

from cliniver.config import load_project_config
from cliniver.config.llm_config import load_llm_config, load_claims_config
from cliniver.utils.seed import set_seed
from cliniver.utils.experiment import Experiment
from cliniver.ingestion.http import RateLimitedClient, RawStore, cached_get
from cliniver.ingestion.qa_datasets import parse_jsonl, domain_regex
from cliniver.evaluation.extraction import NLIJudge
from cliniver.generation.answer import Strategy, Vocabulary, plan, search_queries, fuse, generate, NON_ANSWER
from cliniver.service.engine import Engine

pcfg = load_project_config(); lcfg = load_llm_config(); ccfg = load_claims_config()
ROOT = pcfg.paths.absolute("data_raw").parents[1]
gcfg = yaml.safe_load((ROOT / "configs" / "generation.yaml").read_text(encoding="utf-8"))
dcfg = yaml.safe_load((ROOT / "configs" / "data.yaml").read_text(encoding="utf-8"))
SEED = set_seed(gcfg["seed"]); SNAP = gcfg["snapshot"]
SMOKE = os.environ.get("CLINIVER_SMOKE") == "1"
RAW = pcfg.paths.absolute("data_raw"); PROC = pcfg.paths.absolute("data_processed") / SNAP
LLM_LOG = pcfg.paths.absolute("data_interim") / SNAP / "llm_calls"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
pd.set_option("display.width", 170); pd.set_option("display.max_colwidth", 120)
exp = Experiment(exp_id="EXP-09-000-smoke" if SMOKE else "EXP-09-001", title="Evidence-grounded answer generation", research_question="RQ9 / H9",
                 config={"generation": gcfg}, seed=SEED, packages=["torch", "transformers", "sentence-transformers", "cliniver"])
eng = Engine(log=print, llm_log_dir=LLM_LOG); eng.load()          # the service engine: retriever, verifier, confidence model (baseline v2)
concepts = [json.loads(l) for l in (PROC / "concepts.jsonl").open(encoding="utf-8")]
VOCAB = Vocabulary.build(gcfg, dcfg["drugs"], concepts, dcfg["clinicaltrials"]["conditions"])
STRAT = {k: Strategy.from_cfg(k, v) for k, v in gcfg["variants"].items()}
print("variants:", STRAT); print("drug classes per condition:", {k: len(v) for k, v in VOCAB.classes.items()})
""")

md("## 3. Questions")
code(r"""
http = RateLimitedClient("qa_datasets", min_interval_s=1.0, max_in_flight=1)
kqa = parse_jsonl(cached_get(http, RawStore(RAW, "kqa", SNAP), "questions_w_answers", "jsonl", ccfg.datasets["kqa"]["url"]), "kqa_golden", human=True)
med = []
for f in ccfg.datasets["medlfqa"]["files"]:
    src = f.replace("_test_MedLFQA", "").replace("_wogold", "")
    med += parse_jsonl(cached_get(http, RawStore(RAW, "medlfqa", SNAP), f, "jsonl", f"{ccfg.datasets['medlfqa']['base_url']}/{f}.jsonl"), src, human=False)
drug_names = {n.lower() for c in concepts if c["source"] == "rxnorm" for n in [c["name"], *c["synonyms"]] if len(n) >= 4}
DOM = domain_regex(drug_names)
seen, REAL = set(), []
for it in kqa + med:                                   # the NB03 construction (same order, same filter)
    key = re.sub(r"\W+", " ", it.question.lower()).strip()
    if DOM.search(it.question) and key not in seen:
        seen.add(key); REAL.append(it)
order = list(range(len(REAL))); random.Random(SEED).shuffle(order)
n_dev = gcfg["questions"]["split"]["dev"]
Q = []
for rank, i in enumerate(order):
    it = REAL[i]; refs = list(it.must_have) or list(it.nice_to_have)
    Q.append({"qid": it.qid, "question": it.question, "source": it.source, "refs": refs, "split": "dev" if rank < n_dev else "test", "kind": "real",
              "condition": (VOCAB.conditions_in(it.question) or [None])[0]})
tt = gcfg["questions"]["treatment_templates"]
for cond, name in tt["condition_names"].items():
    for p_i, ph in enumerate(tt["phrasings"]):
        Q.append({"qid": f"tmpl:{cond}:{p_i}", "question": ph.format(condition=name), "source": "template", "refs": [], "kind": "treatment",
                  "split": "dev" if p_i in tt["dev_phrasings"] else "test", "condition": cond})
if SMOKE: Q = [q for sp in ("dev", "test") for kd, n in (("real", 4), ("treatment", 2)) for q in [x for x in Q if x["split"] == sp and x["kind"] == kd][:n]]
qs = pd.DataFrame(Q)
info = {"questions": len(Q), "by_split_kind": qs.groupby(["split", "kind"]).size().to_dict(), "with_refs": int((qs.refs.str.len() > 0).sum()),
        "refs_per_question_median": float(qs[qs.refs.str.len() > 0].refs.str.len().median()), "by_source": qs.source.value_counts().to_dict()}
exp.log_result("questions", {k: (str(v) if k == "by_split_kind" else v) for k, v in info.items()}); print(info)
display(qs[qs.kind == "treatment"][["qid", "question", "split"]].head(6))
""")

md(r"""
## 4. Running the variants (phase by phase, one large model on the GPU at a time)
1. **LLM phase:** query plans for every question.
2. **GPU retrieval:** every distinct search query, cached.
3. **Passages per variant.**
4. **LLM phase:** the answers. Every variant is generated on dev; on test, only G0 and the selected variant (§7).

LLM calls are logged and replayed from `data/interim/<snapshot>/llm_calls/` (a replay reproduces the answers exactly).
""")
code(r"""
needs_plan = any(s.retrieval != "question" for s in STRAT.values())
PLANS, T_PLAN = {}, {}
eng._torch_to("cpu")
with eng._new_llm() as mc:
    if needs_plan:
        for q in Q:
            t0 = time.perf_counter(); PLANS[q["qid"]] = plan(mc, q["question"], seed=SEED, tag=f"gen:plan:{q['qid']}"); T_PLAN[q["qid"]] = time.perf_counter() - t0
eng._torch_to(eng.dev)
SQ = {(v, q["qid"]): search_queries(s, q["question"], PLANS.get(q["qid"]), VOCAB) for v, s in STRAT.items() for q in Q}
RES, T_Q = {}, {}
for qq in sorted({x for v in SQ.values() for x in v}):
    t0 = time.perf_counter(); RES[qq] = eng.retriever.search(qq, top=10); T_Q[qq] = time.perf_counter() - t0
PASS = {}
for (v, qid), qlist in SQ.items():
    s = STRAT[v]
    if s.retrieval == "question" and not s.diverse:
        idx, sc = RES[qlist[0]]; picked = list(zip(idx[:s.k], sc[:s.k]))
    else:
        picked = fuse([RES[x] for x in qlist], eng.units, s.k, s.diverse, rrf_k=60)
    PASS[(v, qid)] = [eng._evidence(eng.units[j], sc) for j, sc in picked]
print(len(RES), "distinct search queries;", "plans:", Counter(p.get("intent") for p in PLANS.values()))
QBY = {q["qid"]: q for q in Q}
ANS = {}
def gen_answers(keys):
    "LLM phase: answers for (variant, qid) keys (the PyTorch models wait on the CPU)."
    eng._torch_to("cpu")
    with eng._new_llm() as mc:
        for v, qid in keys:
            a, info = generate(mc, QBY[qid]["question"], PASS[(v, qid)], prompt=STRAT[v].prompt, seed=SEED, tag=f"gen:answer:{v}:{qid}")
            ANS[(v, qid)] = {"answer": a, **info}
    eng._torch_to(eng.dev)
gen_answers([(v, q["qid"]) for v in STRAT for q in Q if q["split"] == "dev"])      # dev: every variant
print(len(ANS), "dev answers; finish reasons:", Counter(x["finish_reason"] for x in ANS.values()))
""")

md("## 5. Scoring the answers")
code(r"""
JDG = gcfg["judge"]
judge = NLIJudge(JDG["repo"], JDG["revision"], device=DEV)
lab_idx = {v.lower(): int(k) for k, v in judge.model.config.id2label.items()}
CON_I = next(i for l, i in lab_idx.items() if l.startswith("contra"))
def ent_con(premise, hyps):
    if not hyps: return np.zeros(0), np.zeros(0)
    enc = judge.tok([premise] * len(hyps), hyps, truncation="only_first", max_length=512, padding=True, return_tensors="pt").to(DEV)
    with torch.inference_mode():
        pr = torch.softmax(judge.model(**enc).logits.float(), -1).cpu().numpy()
    return pr[:, judge.ent], pr[:, CON_I]
def score(keys):
  rows = []
  for v, qid in keys:
    a = ANS[(v, qid)]; q = QBY[qid]
    e, c = ent_con(a["answer"], q["refs"])
    trt = q["kind"] == "treatment" or (q["condition"] is not None and re.search(r"\b(medicine|medication|drug|treat|pill|therapy)\b", q["question"], re.I))
    lat = T_PLAN.get(qid, 0.0) * (STRAT[v].retrieval != "question") + sum(T_Q[x] for x in SQ[(v, qid)]) + (a["latency_s"] or 0)
    rows.append({"variant": v, "qid": qid, "split": q["split"], "kind": q["kind"], "source": q["source"],
                 "ref_recall": float(np.mean(e >= JDG["threshold"])) if len(e) else np.nan,
                 "ref_contradiction": float(np.mean(c >= JDG["threshold"])) if len(c) else np.nan,
                 "treatment_q": bool(trt), "names_drug": VOCAB.names_drug_for(a["answer"], q["condition"]) if q["condition"] else np.nan,
                 "non_answer": bool(NON_ANSWER.search(a["answer"])), "words": len(a["answer"].split()), "latency_s": lat,
                 "n_passages": len(PASS[(v, qid)]), "n_queries": len(SQ[(v, qid)])})
  return pd.DataFrame(rows)
SC = score(list(ANS))
def summ(df):
    t = df[df.treatment_q & df.names_drug.notna()]
    return pd.Series({"questions": df.qid.nunique(), "ref_recall": df.ref_recall.mean(), "ref_contradiction": df.ref_contradiction.mean(),
                      "drug_naming_templates": df[df.kind == "treatment"].names_drug.astype(float).mean(),
                      "drug_naming_real_treatment_q": t[t.kind == "real"].names_drug.astype(float).mean(),
                      "non_answer": df.non_answer.mean(), "words_median": df.words.median(), "latency_median_s": df.latency_s.median()})
judge.model.to("cpu"); torch.cuda.empty_cache()     # the judge returns for the test answers (§7)
DEVT = SC[SC.split == "dev"].groupby("variant").apply(summ, include_groups=False).round(4)
exp.save_table("dev_variants", DEVT); exp.log_result("dev", DEVT.reset_index().to_dict(orient="records")); display(DEVT)
""")

md("## 6. Selection on dev (pre-registered rule)")
code(r"""
g0 = DEVT.loc["G0"]
ok = DEVT[(DEVT.ref_contradiction <= g0.ref_contradiction + 0.02) & (DEVT.drug_naming_templates.fillna(0) >= (g0.drug_naming_templates if pd.notna(g0.drug_naming_templates) else 0))]
best = ok.ref_recall.max()
near = [v for v in ok.index if ok.loc[v, "ref_recall"] >= best - 0.01]
SELV = min(near, key=lambda v: (STRAT[v].llm_calls, -ok.loc[v, "ref_recall"]))
sel = {"eligible": list(ok.index), "within_0.01_of_best": near, "selected": SELV}
exp.log_result("selection", sel); print(sel)
""")

md("## 7. Test: selected variant vs G0")
code(r"""
test_keys = [(v, q["qid"]) for v in dict.fromkeys(["G0", SELV]) for q in Q if q["split"] == "test"]
gen_answers(test_keys)
judge.model.to(DEV)
SC = pd.concat([SC, score(test_keys)], ignore_index=True)
judge.model.to("cpu"); torch.cuda.empty_cache()
TE = SC[SC.split == "test"]
TT = TE[TE.variant.isin(["G0", SELV])].groupby("variant").apply(summ, include_groups=False).round(4)
display(TT)
def paired(metric, df_a, df_b, n=gcfg["n_boot"]):
    m = df_a[["qid", metric]].merge(df_b[["qid", metric]], on="qid", suffixes=("_a", "_b")).dropna()
    d = (m[f"{metric}_b"].astype(float) - m[f"{metric}_a"].astype(float)).values
    rng = np.random.default_rng(SEED); bs = [rng.choice(d, len(d)).mean() for _ in range(n)]
    return {"delta": round(float(d.mean()), 4), "ci": [round(float(x), 4) for x in np.percentile(bs, [2.5, 97.5])], "n": int(len(d))}
A, Bv = TE[TE.variant == "G0"], TE[TE.variant == SELV]
TEST = {"table": TT.reset_index().to_dict(orient="records"),
        "ref_recall": paired("ref_recall", A[A.kind == "real"], Bv[Bv.kind == "real"]),
        "ref_contradiction": paired("ref_contradiction", A[A.kind == "real"], Bv[Bv.kind == "real"]),
        "drug_naming_templates": {"G0": TT.loc["G0", "drug_naming_templates"], SELV: TT.loc[SELV, "drug_naming_templates"]}}
exp.log_result("test", TEST); print(json.dumps({k: v for k, v in TEST.items() if k != "table"}, indent=1, default=str))
""")

md(r"""
### 7b. CliniVer's own verification of the generated answers (test subset)
Each answer goes through the full baseline-v2 pipeline: claim extraction, standalone rewrite, retrieval, verifier, confidence, second opinion. This is the share of each answer's claims that the corpus supports.
""")
code(r"""
sub_ids = [q["qid"] for q in Q if q["split"] == "test" and q["kind"] == "real"]
sub_ids = sorted(random.Random(SEED).sample(sub_ids, min(gcfg["verification_subset"], len(sub_ids)))) + [q["qid"] for q in Q if q["split"] == "test" and q["kind"] == "treatment"]
if SMOKE: sub_ids = sub_ids[:3]
VER = []
for v in ("G0", SELV) if SELV != "G0" else ("G0",):
    for qid in sub_ids:
        q = next(x for x in Q if x["qid"] == qid); r = eng.verify(q["question"], ANS[(v, qid)]["answer"])
        vc = Counter(c["verdict"] for c in r["claims"]); n = max(1, len(r["claims"]))
        VER.append({"variant": v, "qid": qid, "claims": len(r["claims"]), "supported": vc["SUP"] / n, "contradicted": vc["CON"] / n,
                    "insufficient": vc["INS"] / n, "verify_s": sum(r["timings"].values())})
VR = pd.DataFrame(VER)
VT = VR.groupby("variant")[["claims", "supported", "contradicted", "insufficient", "verify_s"]].mean().round(4)
exp.log_result("verified_support", {"subset": len(sub_ids), "by_variant": VT.reset_index().to_dict(orient="records"),
                                    "supported_delta": paired("supported", VR[VR.variant == "G0"], VR[VR.variant == SELV]) if SELV != "G0" else None})
display(VT)
""")

md("## 8. Adoption decision and examples")
code(r"""
ADOPT = {"selected": SELV,
         "ref_recall_gain": bool(TEST["ref_recall"]["delta"] > 0),
         "contradiction_within_0.02": bool(TEST["ref_contradiction"]["delta"] <= 0.02),
         "drug_naming_not_worse": bool(np.nan_to_num(TEST["drug_naming_templates"][SELV]) >= np.nan_to_num(TEST["drug_naming_templates"]["G0"]))}
ADOPT["adopted"] = bool(SELV != "G0" and all(ADOPT[k] for k in ("ref_recall_gain", "contradiction_within_0.02", "drug_naming_not_worse")))
ADOPT["service_strategy"] = SELV if ADOPT["adopted"] else "G0"
ADOPT["strategy_config"] = gcfg["variants"][ADOPT["service_strategy"]]
exp.log_result("adoption", ADOPT); print(json.dumps(ADOPT, indent=1, default=str))
ex_ids = ["tmpl:hypertension:0", "tmpl:t2d:2", "tmpl:heart_failure:3"] + [q["qid"] for q in Q if q["split"] == "test" and q["kind"] == "real"][:2]
EX = []
for qid in ex_ids:
    if ("G0", qid) not in ANS: continue
    q = next(x for x in Q if x["qid"] == qid)
    for v in dict.fromkeys(["G0", SELV]):
        EX.append({"question": q["question"], "variant": v, "answer": ANS[(v, qid)]["answer"], "queries": SQ[(v, qid)],
                   "passages": [f"{p['source']} {p.get('date')}: {p['title'][:90]}" for p in PASS[(v, qid)]]})
(exp.dir / "examples.json").write_text(json.dumps(EX, indent=1, ensure_ascii=False), encoding="utf-8")
for e in EX[:6]:
    print(f"\n=== {e['variant']} | {e['question']}\n{e['answer']}\nqueries: {e['queries'][:5]}")
fig, ax = plt.subplots(figsize=(7, 3.6))
d = DEVT[["ref_recall", "ref_contradiction", "drug_naming_templates"]]
d.plot.bar(ax=ax); ax.set_title("Dev: answer quality by generation strategy"); ax.set_ylim(0, 1); ax.grid(alpha=.3, axis="y")
exp.save_figure("dev_variants", fig); plt.show()
with (exp.dir / "answers.jsonl").open("w", encoding="utf-8") as fh:
    for (v, qid), a in ANS.items():
        fh.write(json.dumps({"variant": v, "qid": qid, "answer": a["answer"], "queries": SQ[(v, qid)], "passages": [p["unit_id"] for p in PASS[(v, qid)]]}, ensure_ascii=False) + "\n")
exp.finish(); print("results:", exp.dir)
""")

md(r"""
# Part B — Round 2: stricter grounding (EXP-09-002)
- **EXP-09-001:** G3 fixed the observed failure. It named a drug on 100% of the test treatment questions (G0: 70%) and cut non-answers from 33% to 10%. But it raised reference contradictions from 0.048 to 0.092 (CI +0.010 to +0.078), so the pre-registered rule did **not** adopt it.
- **Error analysis** of G3's new contradictions (post-hoc, done on the test answers):
  1. facts from outside the passages, with a citation attached;
  2. answering a neighbouring aspect;
  3. off-topic passages from drug-class expansion on non-treatment questions.

  About half were judge or reference artefacts.
- **G4** = G3 retrieval with drug-class expansion only for "which treatment" questions, 6 passages, and **prompt v3**: answer exactly the aspect asked; every sentence restates its cited passage; no background knowledge.
- **Protocol:** `configs/generation.yaml` → `round2`, written before G4 was built.
- **Because the EXP-09-001 test set has now been read, it is reported as *seen*.** A **fresh** set is built from the corpus vocabulary: side effects and contraindications of 25 sampled drugs, plus new treatment phrasings. It has no human references, so it is scored by CliniVer's own claim verification.
""")
code(r"""
r2 = gcfg["round2"]; fr = r2["evaluation"]["fresh_test"]
exp2 = Experiment(exp_id="EXP-09-000b-smoke" if SMOKE else "EXP-09-002", title="Answer generation, round 2 (stricter grounding)",
                  research_question="RQ9 round 2", config={"round2": r2}, seed=SEED, packages=["torch", "transformers", "cliniver"])
STRAT["G4"] = Strategy.from_cfg("G4", r2["variants"]["G4"])
drug_cond = {d: c for c, ds in dcfg["drugs"].items() for d in ds}
picks = sorted(random.Random(fr["seed"]).sample(sorted(drug_cond), fr["drugs_sampled"]))
FRESH = [{"qid": f"fresh:{d}:{i}", "question": t.format(drug=d), "source": "fresh_template", "refs": [], "kind": "drug", "split": "fresh",
          "condition": drug_cond[d]} for d in picks for i, t in enumerate(fr["drug_questions"])]
FRESH += [{"qid": f"fresh:{c}:t{i}", "question": t.format(condition=n), "source": "fresh_template", "refs": [], "kind": "treatment", "split": "fresh",
           "condition": c} for c, n in tt["condition_names"].items() for i, t in enumerate(fr["treatment_questions"])]
if SMOKE: FRESH = FRESH[:4] + [x for x in FRESH if x["kind"] == "treatment"][:2]
Q += FRESH; QBY.update({q["qid"]: q for q in FRESH})
eng._torch_to("cpu")
with eng._new_llm() as mc:
    for q in Q:
        if q["qid"] not in PLANS:
            t0 = time.perf_counter(); PLANS[q["qid"]] = plan(mc, q["question"], seed=SEED, tag=f"gen:plan:{q['qid']}"); T_PLAN[q["qid"]] = time.perf_counter() - t0
eng._torch_to(eng.dev)
KEYS2 = [("G4", q["qid"]) for q in Q] + [(v, q["qid"]) for v in ("G0", "G3") for q in FRESH]
for v, qid in KEYS2:
    SQ[(v, qid)] = search_queries(STRAT[v], QBY[qid]["question"], PLANS.get(qid), VOCAB)
for qq in sorted({x for k in KEYS2 for x in SQ[k]} - set(RES)):
    t0 = time.perf_counter(); RES[qq] = eng.retriever.search(qq, top=10); T_Q[qq] = time.perf_counter() - t0
for v, qid in KEYS2:
    s_ = STRAT[v]
    if s_.retrieval == "question" and not s_.diverse:
        idx, sc = RES[SQ[(v, qid)][0]]; picked = list(zip(idx[:s_.k], sc[:s_.k]))
    else:
        picked = fuse([RES[x] for x in SQ[(v, qid)]], eng.units, s_.k, s_.diverse, rrf_k=60)
    PASS[(v, qid)] = [eng._evidence(eng.units[j], sc) for j, sc in picked]
gen_answers([k for k in KEYS2 if k not in ANS])
judge.model.to(DEV)
SC2 = score(KEYS2)
judge.model.to("cpu"); torch.cuda.empty_cache()
ALL = pd.concat([SC, SC2], ignore_index=True).drop_duplicates(["variant", "qid"], keep="last")
print(len(FRESH), "fresh questions;", len(KEYS2), "round-2 answers")
""")
code(r"""
def tbl(split, variants):
    return ALL[(ALL.split == split) & ALL.variant.isin(variants)].groupby("variant").apply(summ, include_groups=False).round(4)
DEV2, SEEN2 = tbl("dev", ["G0", "G3", "G4"]), tbl("test", ["G0", "G3", "G4"])
print("dev (sanity check, no selection)"); display(DEV2); print("seen test (EXP-09-001 test questions)"); display(SEEN2)
TS = ALL[ALL.split == "test"]
SEEN_CI = {m: paired(m, TS[(TS.variant == "G0") & (TS.kind == "real")], TS[(TS.variant == "G4") & (TS.kind == "real")]) for m in ("ref_recall", "ref_contradiction")}
TRT = ALL[(ALL.kind == "treatment")]
DRUG_NAMING = TRT.groupby("variant").names_drug.apply(lambda x: float(x.astype(float).mean())).round(4).to_dict()
exp2.log_result("dev", DEV2.reset_index().to_dict(orient="records")); exp2.log_result("seen_test", {"table": SEEN2.reset_index().to_dict(orient="records"), "G4_vs_G0": SEEN_CI})
exp2.log_result("drug_naming_all_treatment_questions", DRUG_NAMING)
print(json.dumps(SEEN_CI, indent=1)); print("drug naming, all treatment questions:", DRUG_NAMING)
""")
md("### Fresh questions: CliniVer verification of every answer")
code(r"""
FV = []
for v in ("G0", "G3", "G4"):
    for q in FRESH:
        r = eng.verify(q["question"], ANS[(v, q["qid"])]["answer"])
        vc = Counter(c["verdict"] for c in r["claims"]); n = max(1, len(r["claims"]))
        FV.append({"variant": v, "qid": q["qid"], "kind": q["kind"], "claims": len(r["claims"]), "supported": vc["SUP"] / n,
                   "contradicted": vc["CON"] / n, "insufficient": vc["INS"] / n})
FVD = pd.DataFrame(FV)
FT = FVD.groupby("variant")[["claims", "supported", "contradicted", "insufficient"]].mean().round(4)
FRESH_CI = {m: paired(m, FVD[FVD.variant == "G0"], FVD[FVD.variant == "G4"]) for m in ("supported", "contradicted")}
NA = ALL[ALL.split == "fresh"].groupby("variant").non_answer.mean().round(4).to_dict()
exp2.log_result("fresh_test", {"questions": len(FRESH), "verified": FT.reset_index().to_dict(orient="records"), "G4_vs_G0": FRESH_CI, "non_answer": NA})
display(FT); print(json.dumps(FRESH_CI, indent=1), "non-answers:", NA)
""")
code(r"""
s_g0, s_g4 = SEEN2.loc["G0"], SEEN2.loc["G4"]; f_g0, f_g4 = FT.loc["G0"], FT.loc["G4"]
ADOPT2 = {"seen_contradiction_ok": bool(s_g4.ref_contradiction <= s_g0.ref_contradiction + 0.02),
          "seen_recall_ok": bool(s_g4.ref_recall >= s_g0.ref_recall - 0.01),
          "fresh_contradicted_ok": bool(f_g4.contradicted <= f_g0.contradicted + 0.02),
          "fresh_supported_ok": bool(f_g4.supported >= f_g0.supported - 0.02),
          "drug_naming_ok": bool(DRUG_NAMING.get("G4", 0) >= DRUG_NAMING.get("G0", 0))}
ADOPT2["adopted"] = all(ADOPT2.values())
ADOPT2["service_strategy"] = "G4" if ADOPT2["adopted"] else "G0"
ADOPT2["strategy_config"] = r2["variants"]["G4"] if ADOPT2["adopted"] else gcfg["variants"]["G0"]
exp2.log_result("adoption", ADOPT2); print(json.dumps(ADOPT2, indent=1, default=str))
EX2 = []
for qid in ["tmpl:hypertension:0", "fresh:hypertension:t0", f"fresh:{picks[0]}:0", f"fresh:{picks[0]}:1"] + [q["qid"] for q in Q if q["split"] == "test" and q["kind"] == "real"][:3]:
    if ("G4", qid) in ANS:
        for v in ("G0", "G4"):
            if (v, qid) in ANS:
                EX2.append({"variant": v, "question": QBY[qid]["question"], "answer": ANS[(v, qid)]["answer"], "queries": SQ.get((v, qid))})
(exp2.dir / "examples.json").write_text(json.dumps(EX2, indent=1, ensure_ascii=False), encoding="utf-8")
for e in EX2[:6]: print(f"\n=== {e['variant']} | {e['question']}\n{e['answer']}")
with (exp2.dir / "answers.jsonl").open("w", encoding="utf-8") as fh:
    for k in KEYS2:
        fh.write(json.dumps({"variant": k[0], "qid": k[1], "answer": ANS[k]["answer"], "queries": SQ[k], "passages": [p["unit_id"] for p in PASS[k]]}, ensure_ascii=False) + "\n")
exp2.finish(); print("results:", exp2.dir)
""")

md(r"""
# Part C — Round 3: route by question intent (EXP-09-003)
- **EXP-09-002:** G4 was not adopted (fresh verified Supported 0.775 vs G0 0.819).
- **On the fresh set, G3 was best on every measure:** Supported 0.873, Contradicted 0.028, non-answers 7%, drug naming 100%.
- **G3's round-1 failure** (contradictions against reference statements) came mostly from **non-treatment** questions, per the error analysis.
- **H1** therefore routes on the planner's intent: **G3 for "treatment_choice" questions, G0 for everything else.**
- **Protocol:** `configs/generation.yaml` → `round3`, written before this part ran.
- **Seen sets** (EXP-09-001 test, EXP-09-002 fresh) reuse the cached G0 / G3 answers.
- **A second fresh set** uses the 24 in-scope drugs not sampled before, with new phrasings, and is scored by CliniVer's own verification.
""")
code(r"""
r3 = gcfg["round3"]; f2 = r3["evaluation"]["fresh_test_2"]
exp3 = Experiment(exp_id="EXP-09-000c-smoke" if SMOKE else "EXP-09-003", title="Answer generation, round 3 (intent routing)",
                  research_question="RQ9 round 3", config={"round3": r3}, seed=SEED, packages=["torch", "transformers", "cliniver"])
ROUTE = r3["variants"]["H1"]["route"]
def h1_of(qid): return ROUTE.get((PLANS.get(qid) or {}).get("intent"), ROUTE["default"])
rest = sorted(set(drug_cond) - set(picks))
FRESH2 = [{"qid": f"fresh2:{d}:{i}", "question": t.format(drug=d), "source": "fresh_template_2", "refs": [], "kind": "drug", "split": "fresh2",
           "condition": drug_cond[d]} for d in rest for i, t in enumerate(f2["drug_questions"])]
FRESH2 += [{"qid": f"fresh2:{c}:t{i}", "question": t.format(condition=n), "source": "fresh_template_2", "refs": [], "kind": "treatment", "split": "fresh2",
            "condition": c} for c, n in tt["condition_names"].items() for i, t in enumerate(f2["treatment_questions"])]
if SMOKE: FRESH2 = FRESH2[:4] + [x for x in FRESH2 if x["kind"] == "treatment"][:2]
Q += FRESH2; QBY.update({q["qid"]: q for q in FRESH2})
eng._torch_to("cpu")
with eng._new_llm() as mc:
    for q in FRESH2:
        if q["qid"] not in PLANS:
            t0 = time.perf_counter(); PLANS[q["qid"]] = plan(mc, q["question"], seed=SEED, tag=f"gen:plan:{q['qid']}"); T_PLAN[q["qid"]] = time.perf_counter() - t0
eng._torch_to(eng.dev)
KEYS3 = [("G0", q["qid"]) for q in FRESH2] + [("G3", q["qid"]) for q in FRESH2 if h1_of(q["qid"]) == "G3"]
for v, qid in KEYS3:
    SQ[(v, qid)] = search_queries(STRAT[v], QBY[qid]["question"], PLANS.get(qid), VOCAB)
for qq in sorted({x for k in KEYS3 for x in SQ[k]} - set(RES)):
    t0 = time.perf_counter(); RES[qq] = eng.retriever.search(qq, top=10); T_Q[qq] = time.perf_counter() - t0
for v, qid in KEYS3:
    s_ = STRAT[v]
    if s_.retrieval == "question" and not s_.diverse:
        idx, sc = RES[SQ[(v, qid)][0]]; picked = list(zip(idx[:s_.k], sc[:s_.k]))
    else:
        picked = fuse([RES[x] for x in SQ[(v, qid)]], eng.units, s_.k, s_.diverse, rrf_k=60)
    PASS[(v, qid)] = [eng._evidence(eng.units[j], sc) for j, sc in picked]
gen_answers([k for k in KEYS3 if k not in ANS])
STRAT["H1"] = STRAT["G0"]          # placeholder for score(): the routed answer is the G0 or G3 answer of that question
for q in Q:
    src = h1_of(q["qid"])
    if (src, q["qid"]) in ANS:
        ANS[("H1", q["qid"])] = ANS[(src, q["qid"])]; PASS[("H1", q["qid"])] = PASS[(src, q["qid"])]; SQ[("H1", q["qid"])] = SQ[(src, q["qid"])]
HKEYS = [k for k in ANS if k[0] == "H1"]
judge.model.to(DEV)
SC3 = score(HKEYS + [k for k in KEYS3 if k[0] == "G0"])
judge.model.to("cpu"); torch.cuda.empty_cache()
ALL3 = pd.concat([ALL, SC3], ignore_index=True).drop_duplicates(["variant", "qid"], keep="last")
routed_share = {sp: round(float(np.mean([h1_of(q["qid"]) == "G3" for q in Q if q["split"] == sp])), 3) for sp in ("dev", "test", "fresh", "fresh2")}
exp3.log_result("routing", {"route": ROUTE, "share_routed_to_G3": routed_share,
                            "intents": {sp: dict(Counter((PLANS.get(q["qid"]) or {}).get("intent") for q in Q if q["split"] == sp)) for sp in ("test", "fresh2")}})
print("share of questions routed to G3:", routed_share)
""")
code(r"""
TS3 = ALL3[ALL3.split == "test"]
SEEN3 = TS3[TS3.variant.isin(["G0", "G3", "H1"])].groupby("variant").apply(summ, include_groups=False).round(4)
SEEN3_CI = {m: paired(m, TS3[(TS3.variant == "G0") & (TS3.kind == "real")], TS3[(TS3.variant == "H1") & (TS3.kind == "real")]) for m in ("ref_recall", "ref_contradiction")}
DRUG3 = ALL3[ALL3.kind == "treatment"].groupby("variant").names_drug.apply(lambda x: float(x.astype(float).mean())).round(4).to_dict()
exp3.log_result("seen_test", {"table": SEEN3.reset_index().to_dict(orient="records"), "H1_vs_G0": SEEN3_CI})
exp3.log_result("drug_naming_all_treatment_questions", DRUG3)
display(SEEN3); print(json.dumps(SEEN3_CI, indent=1)); print("drug naming, all treatment questions:", DRUG3)
FV3 = []
for v in ("G0", "H1"):
    for q in FRESH2:
        r = eng.verify(q["question"], ANS[(v, q["qid"])]["answer"])
        vc = Counter(c["verdict"] for c in r["claims"]); n = max(1, len(r["claims"]))
        FV3.append({"variant": v, "qid": q["qid"], "kind": q["kind"], "claims": len(r["claims"]), "supported": vc["SUP"] / n,
                    "contradicted": vc["CON"] / n, "insufficient": vc["INS"] / n})
FVD3 = pd.DataFrame(FV3)
FT3 = FVD3.groupby("variant")[["claims", "supported", "contradicted", "insufficient"]].mean().round(4)
FRESH3_CI = {m: paired(m, FVD3[FVD3.variant == "G0"], FVD3[FVD3.variant == "H1"]) for m in ("supported", "contradicted")}
NA3 = ALL3[ALL3.split == "fresh2"].groupby("variant").non_answer.mean().round(4).to_dict()
exp3.log_result("fresh_test_2", {"questions": len(FRESH2), "verified": FT3.reset_index().to_dict(orient="records"), "H1_vs_G0": FRESH3_CI, "non_answer": NA3})
display(FT3); print(json.dumps(FRESH3_CI, indent=1), "non-answers:", NA3)
""")
code(r"""
s0, s1 = SEEN3.loc["G0"], SEEN3.loc["H1"]; f0, f1 = FT3.loc["G0"], FT3.loc["H1"]
ADOPT3 = {"seen_contradiction_ok": bool(s1.ref_contradiction <= s0.ref_contradiction + 0.02),
          "seen_recall_ok": bool(s1.ref_recall >= s0.ref_recall - 0.01),
          "fresh2_contradicted_ok": bool(f1.contradicted <= f0.contradicted + 0.02),
          "fresh2_supported_ok": bool(f1.supported >= f0.supported - 0.02),
          "drug_naming_ok": bool(DRUG3.get("H1", 0) >= DRUG3.get("G0", 0))}
ADOPT3["adopted"] = all(ADOPT3.values())
ADOPT3["service_strategy"] = "H1" if ADOPT3["adopted"] else "G0"
ADOPT3["strategy_config"] = ({"route": {k: {"name": v, **gcfg["variants"][v]} for k, v in ROUTE.items()}} if ADOPT3["adopted"] else gcfg["variants"]["G0"])
exp3.log_result("adoption", ADOPT3); print(json.dumps(ADOPT3, indent=1, default=str))
EX3 = []
for qid in ["tmpl:hypertension:0", "fresh2:hypertension:t0", "fresh2:heart_failure:t1", f"fresh2:{rest[0]}:1"]:
    if ("H1", qid) in ANS:
        for v in ("G0", "H1"):
            EX3.append({"variant": v, "routed_to": h1_of(qid) if v == "H1" else None, "question": QBY[qid]["question"], "answer": ANS[(v, qid)]["answer"]})
(exp3.dir / "examples.json").write_text(json.dumps(EX3, indent=1, ensure_ascii=False), encoding="utf-8")
for e in EX3[:4]: print(f"\n=== {e['variant']} ({e['routed_to']}) | {e['question']}\n{e['answer']}")
exp3.finish(); print("results:", exp3.dir)
""")

sys.path.insert(0, str(Path(__file__).parent))
try:
    from nb09_discussion import DISCUSSION_CELLS
    for t in DISCUSSION_CELLS:
        md(t)
except ImportError:
    pass

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"name": "cliniver", "display_name": "CliniVer (Python 3.12)", "language": "python"},
    "language_info": {"name": "python"}})
out.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, out)
print("wrote", out, len(cells), "cells")
