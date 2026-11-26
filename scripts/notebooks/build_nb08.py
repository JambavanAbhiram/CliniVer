"""Builds notebooks/08_end_to_end_evaluation.ipynb. Discussion: nb08_discussion.py (written after execution).

Usage: python scripts/notebooks/build_nb08.py notebooks/08_end_to_end_evaluation.ipynb
"""
import sys
from pathlib import Path

import nbformat as nbf

out = Path(sys.argv[1])
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# Notebook 08 — End-to-End Evaluation and Ablations
**CliniVer** · Experiment `EXP-08-001`

| | |
|---|---|
| **Objective** | Evaluate the complete pipeline (claim → retrieval → verification → calibration → second opinion → abstention), attribute its errors to components, ablate each design decision, and fix the final configuration |
| **Research question** | **RQ7:** How do errors propagate end-to-end? Which component contributes most? |
| **Hypothesis (pre-registered)** | **H7:** Retrieval failure accounts for a larger share of end-to-end errors than verifier error |
| **Decision rules** | `configs/e2e.yaml` (fixed before running): H7 attribution rule; final-selection rule |
| **Labels** | Benchmark (SciFact, human), synthetic (NB04 context items) and **silver** (reference statements). **Real-claim gold labels are not used** (owner decision 2026-11-21, D7): real-claim accuracy is NOT EVALUATED until clinical validation |
""")

md(r"""
## 1. Method
- **End-to-end benchmark:** the 300 SciFact dev claims whose evidence is unanimous (SUP / CON) or absent (INS); claims with mixed evidence are excluded. Evidence is retrieved over the NB05 primary corpus (5,183 SciFact + 124,883 CliniVer abstracts).
- **Pipeline (ADR-0011 … ADR-0014):**
  1. Hybrid BM25 + MedCPT (RRF), MedCPT cross-encoder over the top 50.
  2. Top-1 evidence.
  3. Fine-tuned NLI.
  4. Temperature scaling and the multi-signal confidence (fitted on HealthVer dev).
  5. Qwen3-8B second opinion below the 25th percentile.
  6. ABSTAINED when the two verifiers disagree.
- **Metrics:**
  - macro-F1 (SUP / CON / INS) on the answered claims, coverage, false-support rate (FSR) on the answered claims;
  - **end-to-end macro-F1 with abstentions counted as errors** (a conservative single number);
  - latency per claim.
- **H7 attribution** (pipeline without abstention): an error that disappears when the retrieved top-1 evidence is replaced by **oracle evidence** (a gold evidence abstract; for INS claims, a cited abstract) is *retrieval-attributable*. An error that remains is *verifier-attributable*.
- **Ablations:** retrieval (BM25 only, MedCPT only, hybrid without re-ranker, hybrid with BGE-base, weighted hybrid); evidence (top-3 naive aggregation); verifier (Qwen3-8B only, zero-shot NLI); confidence (uncalibrated, no second opinion, no abstention); contextualization on the NB04 synthetic context-dependent items (claim only, question + claim, rewrite F; oracle = original claim).
- **Silver check of the real claims:** each real claim is compared with its question's reference statements by the NB03 NLI judge. Silver SUP = some statement entails the claim; silver CON = some statement contradicts it.
""")

md("## 2. Setup")
code(r"""
import json, os, random, re, time, collections, io, tarfile, gc
from pathlib import Path
from collections import Counter
import numpy as np, pandas as pd, matplotlib.pyplot as plt, torch, yaml
from IPython.display import display

from cliniver.config import load_project_config
from cliniver.config.llm_config import load_llm_config
from cliniver.utils.seed import set_seed
from cliniver.utils.experiment import Experiment
from cliniver.ingestion.http import RateLimitedClient
from cliniver.ingestion.scifact import parse_release
from cliniver.retrieval.encoders import Encoder
from cliniver.retrieval.sparse import BM25
from cliniver.retrieval.fusion import rrf, weighted
from cliniver.retrieval.pipeline import scifact_primary_corpus
from cliniver.verification.datasets import load_sets
from cliniver.verification.verifiers import NLIVerifier, LLMVerifier
from cliniver.verification.confidence import ConfidenceModel, LAB
from cliniver.llm.runtime import ManagedClient
from sentence_transformers import CrossEncoder
from transformers import AutoModelForSequenceClassification, AutoTokenizer

pcfg = load_project_config(); lcfg = load_llm_config()
ROOT = pcfg.paths.absolute("data_raw").parents[1]
cfg = yaml.safe_load((ROOT / "configs" / "e2e.yaml").read_text(encoding="utf-8"))
vcfg = yaml.safe_load((ROOT / "configs" / "verification.yaml").read_text(encoding="utf-8"))
rcfg = yaml.safe_load((ROOT / "configs" / "retrieval.yaml").read_text(encoding="utf-8"))
ccfg = yaml.safe_load((ROOT / "configs" / "context.yaml").read_text(encoding="utf-8"))
SEED = set_seed(cfg["seed"]); SNAP = cfg["snapshot"]
SMOKE = os.environ.get("CLINIVER_SMOKE") == "1"
RAW = pcfg.paths.absolute("data_raw"); PROC = pcfg.paths.absolute("data_processed") / SNAP
INTERIM = pcfg.paths.absolute("data_interim") / SNAP; EMB = INTERIM / "embeddings"; LLM_LOG = INTERIM / "llm_calls"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
pd.set_option("display.width", 170); pd.set_option("display.max_colwidth", 110)
exp = Experiment(exp_id="EXP-08-000-smoke" if SMOKE else "EXP-08-001", title="End-to-end evaluation and ablations", research_question="RQ7 / H7",
                 config={"e2e": cfg}, seed=SEED, packages=["torch", "transformers", "sentence-transformers", "bm25s", "scikit-learn", "pandas", "cliniver"])
FT = sorted(d for d in (INTERIM / "models").glob("nli_ft_*") if (d / "config.json").exists())[0]
nli = NLIVerifier(str(FT), device=DEV, model=AutoModelForSequenceClassification.from_pretrained(FT), tokenizer=AutoTokenizer.from_pretrained(FT),
                  max_length=vcfg["finetune"]["max_length"])
SETS, sf_raw = load_sets(vcfg, RAW, SNAP, RateLimitedClient("datasets", min_interval_s=1.0, max_in_flight=1), seed=SEED)
CM = ConfidenceModel.fit(nli, SETS["healthver_dev"])
_zc = nli.logits([x["evidence"] for x in SETS["healthver_dev"]], [x["claim"] for x in SETS["healthver_dev"]])
RAW_Q25 = float(np.quantile(np.exp(_zc).max(1) / np.exp(_zc).sum(1), 0.25))   # routing threshold on the UNCALIBRATED scale (ablation)
def managed(key):
    return ManagedClient(lcfg, key, LLM_LOG / f"{key}.jsonl", server_log=pcfg.paths.absolute("logs") / f"llama-server_nb08_{key}.log")
exp.log_result("confidence_model", {"temperature": CM.t, "route_threshold": CM.route_threshold, "abstain_threshold": CM.abstain_threshold})
print("verifier", FT.name, "| T", round(CM.t, 3), "| route <", round(CM.route_threshold, 3), "| smoke", SMOKE)
""")

md("## 3. Benchmark claims and retrieval variants")
code(r"""
DOC_IDS, TITLES, ABSTRACTS, N_SF = scifact_primary_corpus(sf_raw, PROC / "documents.jsonl")
DOC_TEXT = [f"{t}. {a}" for t, a in zip(TITLES, ABSTRACTS)]
sf_corpus, _ = parse_release(sf_raw)
tf = tarfile.open(fileobj=io.BytesIO(sf_raw), mode="r:gz")
dev = [json.loads(l) for l in tf.extractfile(next(x for x in tf.getnames() if x.endswith("claims_dev.jsonl") and "/._" not in x and "cross" not in x)).read().decode().splitlines() if l.strip()]
CLAIMS = []
for c in dev:
    labs = {s["label"] for d in (c.get("evidence") or {}).values() for s in d}
    g = "SUP" if labs == {"SUPPORT"} else "CON" if labs == {"CONTRADICT"} else "INS" if not labs else None
    if g is None: continue
    ev_docs = [d for d in (c.get("evidence") or {}) if d in sf_corpus]
    cited = [str(d) for d in c.get("cited_doc_ids", []) if str(d) in sf_corpus]
    oracle = ev_docs[0] if ev_docs else (cited[0] if cited else None)
    CLAIMS.append({"id": c["id"], "claim": c["claim"], "gold": g, "oracle_text": sf_corpus[oracle] if oracle else ""})
if SMOKE: CLAIMS = CLAIMS[:40]
Q = [c["claim"] for c in CLAIMS]; GOLD = np.array([c["gold"] for c in CLAIMS])
print(len(CLAIMS), "claims", Counter(GOLD))
DEPTH = rcfg["first_stage_depth"]
bm = BM25(DOC_TEXT, variant="default"); BM = bm.search(Q, DEPTH)
dense = {}
for key in ("medcpt", "bge-base"):
    m = rcfg["encoders"][key]
    enc = Encoder(key, m["repo"], m["revision"], m.get("query_prefix", ""), m.get("doc_prefix", ""), m.get("doc_repo"), m.get("doc_revision"), device=DEV)
    D = torch.from_numpy(np.load(EMB / f"{key}_primary.npy")).to(DEV)
    dense[key] = Encoder.search(enc.encode_queries(Q), D, DEPTH)
    enc.unload(); del D; torch.cuda.empty_cache()
ce_ = rcfg["rerankers"]["medcpt-ce"]; CE = CrossEncoder(ce_["repo"], revision=ce_["revision"], device=DEV, max_length=512)
if DEV == "cuda": CE.model.half()
def rerank(cands, pool=50):
    out = []
    for q, c in zip(Q, cands):
        c = list(c[:pool]); s = np.asarray(CE.predict([(q, DOC_TEXT[j]) for j in c], batch_size=64, show_progress_bar=False), dtype=float).reshape(len(c), -1)[:, -1]
        out.append(([c[k] for k in np.argsort(-s)], float(s.max())))
    return out
hyb = [rrf([BM[0][i], dense["medcpt"][0][i]], k=rcfg["fusion"]["rrf_k"], top=DEPTH) for i in range(len(Q))]
t0 = time.perf_counter(); rr = rerank(hyb); rr_ms = (time.perf_counter() - t0) / len(Q) * 1000
RET = {"pipeline": [r[0] for r in rr],
       "bm25_only": [list(BM[0][i]) for i in range(len(Q))], "dense_only": [list(dense["medcpt"][0][i]) for i in range(len(Q))],
       "hybrid_no_rerank": [list(h) for h in hyb],
       "hybrid_bge_base": [r[0] for r in rerank([rrf([BM[0][i], dense["bge-base"][0][i]], k=rcfg["fusion"]["rrf_k"], top=DEPTH) for i in range(len(Q))])],
       "weighted_hybrid": [r[0] for r in rerank([weighted(dense["medcpt"][0][i], dense["medcpt"][1][i], BM[0][i], BM[1][i], 0.6, top=DEPTH) for i in range(len(Q))])]}
REL_HIT = {k: float(np.mean([any(DOC_IDS[j] in {str(d) for d in (dv.get("evidence") or {})} for j in r[:1]) for r, dv in zip(v, [next(x for x in dev if x["id"] == c["id"]) for c in CLAIMS])])) for k, v in RET.items()}
exp.log_result("retrieval", {"rerank_ms_per_claim": round(rr_ms, 1), "top1_is_gold_evidence_doc": REL_HIT}); print("top-1 hits a gold evidence abstract:", REL_HIT)
""")

md("## 4. End-to-end runs")
code(r"""
QWEN_LABELS = {}
def qwen_labels(pairs, tag):
    "Qwen3-8B labels for (claim, evidence) pairs; cached per (claim, evidence)."
    need = [(c, e) for c, e in pairs if (c, e) not in QWEN_LABELS]
    if need:
        for m_ in (nli.model, CE.model): m_.to("cpu")
        torch.cuda.empty_cache()
        with managed("qwen3-8b") as mc:
            v = LLMVerifier(mc, seed=SEED)
            for c, e in need:
                QWEN_LABELS[(c, e)] = v.verify(e, c, tag=f"verify:e2e:{tag}")[0] or "INS"
        for m_ in (nli.model, CE.model): m_.to(DEV)
    return [QWEN_LABELS[p] for p in pairs]

def run(evidence_ranked, *, verifier="nli", top="top1", calibrated=True, second_opinion=True, abstain=True, tag="pipeline"):
    ev1 = [DOC_TEXT[r[0]] if len(r) else "" for r in evidence_ranked]
    if verifier == "qwen":
        lab = np.array([("SUP" if l == "PART" else l) for l in qwen_labels(list(zip(Q, ev1)), tag)])
        return {"pred": lab, "answered": np.ones(len(Q), bool)}
    if verifier == "zero-shot":
        zs = vcfg["models"]["zero_shot_nli"]; v0 = NLIVerifier(zs["repo"], zs["revision"], device=DEV)
        lab, _ = v0.predict(ev1, Q); del v0; torch.cuda.empty_cache()
        return {"pred": np.array(lab), "answered": np.ones(len(Q), bool)}
    if top == "top3_naive":
        labs = []
        for q, r in zip(Q, evidence_ranked):
            ev = [DOC_TEXT[j] for j in r[:3]]; l3, _ = nli.predict(ev, [q] * len(ev))
            labs.append("SUP" if "SUP" in l3 and "CON" not in l3 else "CON" if "CON" in l3 and "SUP" not in l3 else (l3[0] if ("SUP" in l3) else "INS"))
        return {"pred": np.array(labs), "answered": np.ones(len(Q), bool)}
    z = nli.logits(ev1, Q)
    lab, p, overall = CM.score(Q, ev1, z)
    route_thr = CM.route_threshold
    if not calibrated:
        overall = np.exp(z).max(1) / np.exp(z).sum(1); route_thr = RAW_Q25
    pred = np.array(lab); answered = np.ones(len(Q), bool)
    if second_opinion:
        low = overall < route_thr
        idx = np.where(low)[0]
        if len(idx):
            ql = qwen_labels([(Q[i], ev1[i]) for i in idx], tag)
            for i, l in zip(idx, ql):
                l = "SUP" if l == "PART" else l
                if l != pred[i]:
                    if abstain: answered[i] = False
                    else: pred[i] = l
    elif abstain:
        answered = overall >= CM.abstain_threshold
    return {"pred": pred, "answered": answered}

def macro_f1(g, p, labels=("SUP", "CON", "INS")):
    f = []
    for l in labels:
        tp = np.sum((g == l) & (p == l)); fp = np.sum((g != l) & (p == l)); fn = np.sum((g == l) & (p != l))
        f.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return float(np.mean(f))
def summarize(name, r):
    a = r["answered"]; g, p = GOLD, r["pred"]
    e2e = np.where(a, p, "ABSTAINED")
    fs_den = max(1, np.sum(np.isin(g[a], ["CON", "INS"])))
    return {"config": name, "coverage": round(float(a.mean()), 3), "macro_F1_answered": round(macro_f1(g[a], p[a]), 4),
            "macro_F1_e2e": round(macro_f1(g, e2e), 4), "FSR_answered": round(float(np.sum((p[a] == "SUP") & np.isin(g[a], ["CON", "INS"])) / fs_den), 4),
            "accuracy_answered": round(float(np.mean(p[a] == g[a])), 4)}
RUNS = {"pipeline": run(RET["pipeline"])}
for k in cfg["ablations"]["retrieval"]:
    RUNS[k] = run(RET[k], tag=k)
RUNS["top3_naive"] = run(RET["pipeline"], top="top3_naive")
RUNS["qwen_only"] = run(RET["pipeline"], verifier="qwen", tag="qwen_only")
RUNS["zero_shot_nli"] = run(RET["pipeline"], verifier="zero-shot")
RUNS["uncalibrated"] = run(RET["pipeline"], calibrated=False, tag="uncal")
RUNS["no_second_opinion"] = run(RET["pipeline"], second_opinion=False)
RUNS["no_abstention"] = run(RET["pipeline"], abstain=False)
RUNS["nli_only_no_conf"] = run(RET["pipeline"], second_opinion=False, abstain=False)
E2E = pd.DataFrame([summarize(k, v) for k, v in RUNS.items()]).set_index("config")
exp.save_table("e2e_ablations", E2E); exp.log_result("e2e_ablations", E2E.reset_index().to_dict(orient="records")); display(E2E)
""")

md("## 5. H7: where do the errors come from? (oracle-evidence substitution)")
code(r"""
base = RUNS["nli_only_no_conf"]["pred"]                      # the pipeline without abstention / second opinion (pure retrieval + verifier)
oz = nli.logits([c["oracle_text"] for c in CLAIMS], Q); olab = np.array([LAB[k] for k in oz.argmax(1)])
err = base != GOLD
retr = err & (olab == GOLD); verif = err & (olab != GOLD)
rng = np.random.default_rng(SEED); idx = np.arange(len(GOLD)); diffs = []
for _ in range(cfg["n_boot"]):
    s = rng.choice(idx, len(idx)); e = err[s].sum()
    diffs.append((retr[s].sum() - verif[s].sum()) / max(1, e))
lo, hi = np.percentile(diffs, [2.5, 97.5])
H7 = {"errors": int(err.sum()), "retrieval_attributable": int(retr.sum()), "verifier_attributable": int(verif.sum()),
      "retrieval_share": round(float(retr.sum() / max(1, err.sum())), 3), "share_difference_ci": [round(float(lo), 3), round(float(hi), 3)],
      "oracle_evidence_macro_F1": round(macro_f1(GOLD, olab), 4), "retrieved_macro_F1": round(macro_f1(GOLD, base), 4),
      "errors_by_gold": {g: {"retrieval": int((retr & (GOLD == g)).sum()), "verifier": int((verif & (GOLD == g)).sum())} for g in ("SUP", "CON", "INS")}}
H7["supported"] = bool(lo > 0)
exp.log_result("H7", H7); print(json.dumps(H7, indent=1))
""")

md(r"""
## 6. Contextualization ablation (NB04 synthetic context-dependent items)
The NB04 items are rebuilt from the LLM call log (exact replay). Each dependent claim is run through retrieval, top-1 evidence and NLI. The query is the claim as written (A), question + claim (B), or the rewrite (F), with the original claim as the oracle. The verifier reads either the claim as written or the rewrite.
""")
code(r"""
from cliniver.claims.contextualize import make_context, decontextualize
from cliniver.ingestion.scifact import parse_release as _pr
_, sf_claims = _pr(sf_raw)
items = []
with managed(ccfg["llm_model"]) as mc:
    for c in sf_claims:
        it, _ = make_context(mc, c.text, seed=ccfg["seed"], tag=f"ctx:{c.claim_id}")
        if it is None or not it.valid: continue
        fdep, _ = decontextualize(mc, it.dependent_claim, it.question, it.answer, seed=ccfg["seed"], tag=f"dectx:dependent:{c.claim_id}")
        items.append({"claim": c.text, "dep": it.dependent_claim, "q": it.question, "F": fdep or it.dependent_claim, "labels": c.labels})
    ctx_live = mc.stats
items = [x for x in items if x["labels"] in (("SUPPORT",), ("CONTRADICT",))]
if SMOKE: items = items[:40]
G2 = np.array(["SUP" if x["labels"] == ("SUPPORT",) else "CON" for x in items])
def ctx_eval(queries, verify_text):
    bi = bm.search(queries, DEPTH)[0]
    m = rcfg["encoders"]["medcpt"]; enc = Encoder("medcpt", m["repo"], m["revision"], doc_repo=m["doc_repo"], doc_revision=m["doc_revision"], device=DEV)
    D = torch.from_numpy(np.load(EMB / "medcpt_primary.npy")).to(DEV); di = Encoder.search(enc.encode_queries(queries), D, DEPTH)[0]; enc.unload(); del D; torch.cuda.empty_cache()
    top1 = []
    for q, a, b in zip(queries, bi, di):
        c = list(rrf([a, b], k=rcfg["fusion"]["rrf_k"], top=DEPTH)[:50]); s = np.asarray(CE.predict([(q, DOC_TEXT[j]) for j in c], batch_size=64, show_progress_bar=False), dtype=float).reshape(len(c), -1)[:, -1]
        top1.append(DOC_TEXT[c[int(np.argmax(s))]])
    lab, _ = nli.predict(top1, verify_text)
    return round(macro_f1(G2, np.array(lab), ("SUP", "CON")), 4)
ctx = {"claim_only (verify as written)": ctx_eval([x["dep"] for x in items], [x["dep"] for x in items]),
       "question+claim (verify as written)": ctx_eval([f"{x['q']} {x['dep']}" for x in items], [x["dep"] for x in items]),
       "rewrite F (verify as written)": ctx_eval([x["F"] for x in items], [x["dep"] for x in items]),
       "rewrite F (verify the rewrite)": ctx_eval([x["F"] for x in items], [x["F"] for x in items]),
       "oracle original claim": ctx_eval([x["claim"] for x in items], [x["claim"] for x in items])}
exp.log_result("context_ablation", {"items": len(items), "macro_F1_SUP_CON": ctx, "llm_replay_live_server": ctx_live is not None}); print(len(items), "items", json.dumps(ctx, indent=1))
""")

md("## 7. Silver check of the real claims (Pillar 3)")
code(r"""
from cliniver.ingestion.qa_datasets import parse_jsonl
from cliniver.evaluation.extraction import NLIJudge
from cliniver.config.llm_config import load_claims_config
clc = load_claims_config()
kqa = parse_jsonl((RAW / "kqa" / SNAP / "questions_w_answers.jsonl").read_bytes(), "kqa_golden", human=True)
med = []
for f in clc.datasets["medlfqa"]["files"]:
    src = f.replace("_test_MedLFQA", "").replace("_wogold", "")
    med += parse_jsonl((RAW / "medlfqa" / SNAP / f"{f}.jsonl").read_bytes(), src, human=False)
QA = {it.qid: it for it in kqa + med}
resp = {json.loads(l)["response_id"]: json.loads(l) for l in (PROC / "responses" / "responses.jsonl").open(encoding="utf-8")}
vc = [json.loads(l) for l in (PROC / "responses" / "verdicts_calibrated.jsonl").open(encoding="utf-8")]
claims_meta = {json.loads(l)["claim_id"]: json.loads(l) for l in (PROC / "responses" / "claims.jsonl").open(encoding="utf-8")}
for m_ in (nli.model, CE.model): m_.to("cpu")
torch.cuda.empty_cache()
judge = NLIJudge(cfg["silver"]["judge"]["repo"], cfg["silver"]["judge"]["revision"], device=DEV)
from transformers import AutoModelForSequenceClassification as _A
lab_idx = {v.lower(): int(k) for k, v in judge.model.config.id2label.items()}
CON_I = next(i for l, i in lab_idx.items() if l.startswith("contra"))
rows = []
for v in vc:
    rid = claims_meta[v["claim_id"]]["response_id"]; it = QA.get(resp[rid]["qid"])
    if it is None: continue
    st = list(it.must_have) + list(it.nice_to_have)
    if not st: continue
    enc = judge.tok(st, [v["claim"]] * len(st), truncation=True, max_length=512, padding=True, return_tensors="pt").to(DEV)
    with torch.inference_mode():
        pr = torch.softmax(judge.model(**enc).logits.float(), -1).cpu().numpy()
    ent, con = pr[:, judge.ent].max(), pr[:, CON_I].max()
    silver = "SUP" if ent >= cfg["silver"]["entail_threshold"] and ent >= con else "CON" if con >= cfg["silver"]["entail_threshold"] else "UNK"
    rows.append({"claim_id": v["claim_id"], "source": "physician (K-QA)" if it.human_statements else "LLM-generated (MedLFQA)", "silver": silver,
                 "final": v["final_verdict"], "generator": v["generator"]})
SV = pd.DataFrame(rows)
tab = pd.crosstab([SV.source, SV.silver], SV.final)
exp.save_table("silver_crosstab", tab); display(tab)
sil = {}
for src, g in SV.groupby("source"):
    s = g[g.silver != "UNK"]
    sil[src] = {"claims": len(g), "silver_labelled": len(s), "silver_SUP": int((s.silver == "SUP").sum()), "silver_CON": int((s.silver == "CON").sum()),
                "pipeline_SUP_given_silver_SUP": round(float((s[s.silver == "SUP"].final == "SUP").mean()), 3) if (s.silver == "SUP").any() else None,
                "pipeline_SUP_given_silver_CON (silver false support)": round(float((s[s.silver == "CON"].final == "SUP").mean()), 3) if (s.silver == "CON").any() else None,
                "pipeline_CON_or_ABST_given_silver_CON": round(float(s[s.silver == "CON"].final.isin(["CON", "ABSTAINED"]).mean()), 3) if (s.silver == "CON").any() else None}
exp.log_result("silver", sil); print(json.dumps(sil, indent=1))
del judge; torch.cuda.empty_cache()
for m_ in (nli.model, CE.model): m_.to(DEV)
""")

md("## 8. Cost, latency and failure rate of the full pipeline (real claims)")
code(r"""
def log_stats(model, prefix):
    lat, bad = [], 0
    p = LLM_LOG / f"{model}.jsonl"
    for l in p.open(encoding="utf-8"):
        r = json.loads(l)
        if r.get("tag", "").startswith(prefix):
            lat.append(r["latency_s"]); bad += r.get("finish_reason") != "stop"
    return {"calls": len(lat), "median_s": round(float(np.median(lat)), 2) if lat else None, "p95_s": round(float(np.percentile(lat, 95)), 2) if lat else None,
            "not_stopped": bad}
cost = {"answer_generation (NB03)": log_stats("qwen3-8b", "gen:"), "claim_extraction (NB03)": log_stats("qwen3-8b", "claims:"),
        "standalone_rewrite (NB04)": log_stats("qwen3-8b", "dectx:real"), "second_opinion (NB07)": log_stats("qwen3-8b", "verify:real2")}
prev = {k: json.loads((pcfg.paths.absolute("artifacts") / "experiments" / k / "results.json").read_text(encoding="utf-8")) for k in ("EXP-05-001", "EXP-06-001", "EXP-07-001")}
routed_share = prev["EXP-07-001"]["real_claims"]["routed_to_llm"] / prev["EXP-07-001"]["real_claims"]["claims"]
per_claim = {"rewrite_s": cost["standalone_rewrite (NB04)"]["median_s"], "retrieval_p95_s": prev["EXP-06-001"]["real_claims"]["retrieval_latency_p95_s"],
             "second_opinion_expected_s": round(routed_share * (cost["second_opinion (NB07)"]["median_s"] or 0), 2)}
for m_ in (nli.model, CE.model): m_.to("cpu")
torch.cuda.empty_cache()
with managed("qwen3-8b") as mc:   # live measurement of the LLM server (one probe call)
    mc.start(); LLMVerifier(mc, seed=SEED).verify(DOC_TEXT[0], Q[0], tag="verify:e2e:vram_probe"); llm_stats = mc.stats
for m_ in (nli.model, CE.model): m_.to(DEV)
vram = {"llm_qwen3_8b_server": llm_stats, "retrieval_peak_mb": prev["EXP-06-001"]["vram"]["real_claim_retrieval_peak_mb"]}
exp.log_result("cost", {"llm_stage_stats": cost, "per_claim": per_claim, "vram_mb": vram, "routed_share": round(routed_share, 3)})
display(pd.DataFrame(cost).T); print(per_claim, vram)
""")

md("## 9. Final configuration (pre-registered rule)")
code(r"""
budget_ok = {k: True for k in E2E.index}   # all variants share the ADR-0012 retrieval latency (≤ 0.4 s p95) except the LLM-only verifier
budget_ok["qwen_only"] = False              # ~2–4 s per claim on this laptop (LLM on every claim)
fsr_ref = E2E.loc["pipeline", "FSR_answered"]
cand = E2E[(E2E.FSR_answered <= fsr_ref) & E2E.index.map(lambda k: budget_ok[k])]
best = cand.macro_F1_answered.max()
complexity = {"nli_only_no_conf": 0, "no_second_opinion": 1, "no_abstention": 2, "uncalibrated": 2, "pipeline": 3}
near = [k for k in cand.index if cand.loc[k, "macro_F1_answered"] >= best - 0.01]
FINAL = min(near, key=lambda k: complexity.get(k, 3))
final = {"candidates_meeting_constraints": list(cand.index), "within_0.01_of_best": near, "selected": FINAL, "selected_metrics": E2E.loc[FINAL].to_dict()}
exp.log_result("final_selection", final); print(json.dumps(final, indent=1, default=str))
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.scatter(E2E.coverage, E2E.macro_F1_answered, c=["C3" if k == FINAL else "C0" for k in E2E.index])
for k, r in E2E.iterrows(): ax.annotate(k, (r.coverage, r.macro_F1_answered), fontsize=7)
ax.set_xlabel("coverage"); ax.set_ylabel("macro-F1 (answered)"); ax.set_title("End-to-end: quality vs coverage (SciFact dev)"); ax.grid(alpha=.3)
exp.save_figure("e2e_tradeoff", fig); plt.show()
exp.finish(); print("results:", exp.dir)
""")

sys.path.insert(0, str(Path(__file__).parent))
try:
    from nb08_discussion import DISCUSSION_CELLS
    for t in DISCUSSION_CELLS:
        md(t)
except ImportError:
    pass
from nb08_partb import add_cells as add_part_b   # Part B: optimization round 1 (EXP-08-002), owner request 2026-11-21
add_part_b(md, code)
try:
    from nb08b_discussion import DISCUSSION_CELLS as DISCUSSION_B
    for t in DISCUSSION_B:
        md(t)
except ImportError:
    pass
from nb08_partc import add_cells as add_part_c   # Part C: latency (EXP-08-003), owner request 2026-11-22
add_part_c(md, code)
try:
    from nb08c_discussion import DISCUSSION_CELLS as DISCUSSION_C
    for t in DISCUSSION_C:
        md(t)
except ImportError:
    pass

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"name": "cliniver", "display_name": "CliniVer (Python 3.12)", "language": "python"},
    "language_info": {"name": "python"}})
out.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, out)
print("wrote", out, len(cells), "cells")
