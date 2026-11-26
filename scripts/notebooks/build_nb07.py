"""Builds notebooks/07_confidence_calibration_abstention.ipynb. Discussion: nb07_discussion.py (written after execution).

Usage: python scripts/notebooks/build_nb07.py notebooks/07_confidence_calibration_abstention.ipynb
"""
import sys
from pathlib import Path

import nbformat as nbf

out = Path(sys.argv[1])
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# Notebook 07 — Confidence, Calibration and Abstention
**CliniVer** · Experiment `EXP-07-001`

| | |
|---|---|
| **Objective** | Turn verifier outputs into calibrated confidence, decide when to abstain or ask for a second opinion, and produce the final (still unvalidated) verdicts for the real claims |
| **Research question** | **RQ6:** Which confidence signals predict correctness? Does calibrated abstention improve reliability at acceptable coverage? |
| **Hypotheses (pre-registered)** | **H6a** temperature scaling reduces ECE on held-out data · **H6b** multi-signal confidence yields lower AURC than verifier probability alone · **H6c** abstention reduces the false-support rate at ≥ 80% coverage |
| **Decision rules** | `configs/calibration.yaml` (fixed before running). Everything is fitted on HealthVer **dev** only; held-out sets: HealthVer test, SciFact dev, HealthFC |
| **NB06 carry-overs** | numeric/qualifier consistency signal; top-1 vs aggregated evidence; Qwen3-8B second opinion for low-confidence decisions |

> **ABSTAINED ≠ INS (ADR-0005):** abstention means "the system is not confident enough to give a verdict", INS means "the evidence does not settle the claim". Both are reported separately, with the corpus snapshot date.
""")

md("## 1. Setup")
code(r"""
import json, os, random, re, time, collections, io
from pathlib import Path
from collections import Counter
import numpy as np, pandas as pd, matplotlib.pyplot as plt, torch, yaml
from IPython.display import display
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

from cliniver.config import load_project_config
from cliniver.config.llm_config import load_llm_config
from cliniver.utils.seed import set_seed
from cliniver.utils.experiment import Experiment
from cliniver.ingestion.http import RateLimitedClient
from cliniver.ingestion.scifact import corpus_titles
from cliniver.verification.datasets import load_sets, llm_samples
from cliniver.verification.verifiers import NLIVerifier, LLMVerifier
from cliniver.verification.calibration import softmax, fit_temperature, ece, brier, risk_coverage, aurc, selective, threshold_for_coverage
from cliniver.verification.consistency import numeric_consistency, unsupported_numbers
from cliniver.evaluation.extraction import content_words
from cliniver.llm.runtime import ManagedClient

pcfg = load_project_config(); lcfg = load_llm_config()
ROOT = pcfg.paths.absolute("data_raw").parents[1]
cfg = yaml.safe_load((ROOT / "configs" / "calibration.yaml").read_text(encoding="utf-8"))
vcfg = yaml.safe_load((ROOT / "configs" / "verification.yaml").read_text(encoding="utf-8"))
SEED = set_seed(cfg["seed"]); SNAP = cfg["snapshot"]
SMOKE = os.environ.get("CLINIVER_SMOKE") == "1"
RAW = pcfg.paths.absolute("data_raw"); PROC = pcfg.paths.absolute("data_processed") / SNAP
INTERIM = pcfg.paths.absolute("data_interim") / SNAP; LLM_LOG = INTERIM / "llm_calls"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
pd.set_option("display.width", 170); pd.set_option("display.max_colwidth", 110)
exp = Experiment(exp_id="EXP-07-000-smoke" if SMOKE else "EXP-07-001", title="Confidence, calibration, abstention", research_question="RQ6 / H6a-H6c",
                 config={"calibration": cfg, "verification": vcfg}, seed=SEED, packages=["torch", "transformers", "scikit-learn", "scipy", "pandas", "cliniver"])
ft_dirs = sorted((INTERIM / "models").glob("nli_ft_*"))
ft_dirs = [d for d in ft_dirs if (d / "config.json").exists()]
assert len(ft_dirs) >= 1, "the fine-tuned verifier from EXP-06-001 is required"
FT_DIR = ft_dirs[0] if len(ft_dirs) == 1 else max(ft_dirs, key=lambda d: (d / "history.json").stat().st_size)
from transformers import AutoModelForSequenceClassification, AutoTokenizer
nli = NLIVerifier(str(FT_DIR), device=DEV, model=AutoModelForSequenceClassification.from_pretrained(FT_DIR),
                  tokenizer=AutoTokenizer.from_pretrained(FT_DIR), max_length=vcfg["finetune"]["max_length"])
def managed(key):
    return ManagedClient(lcfg, key, LLM_LOG / f"{key}.jsonl", server_log=pcfg.paths.absolute("logs") / f"llama-server_nb07_{key}.log")
print("verifier:", FT_DIR.name, "| device", DEV, "| smoke", SMOKE)
""")

md("## 2. Data and verifier outputs")
code(r"""
SETS, sf_raw = load_sets(vcfg, RAW, SNAP, RateLimitedClient("datasets", min_interval_s=1.0, max_in_flight=1), smoke=SMOKE, seed=SEED)
EVAL = [cfg["calibration_split"]] + cfg["heldout_sets"] + ["synthetic"]
LAB = ["SUP", "CON", "INS"]; L2I = {l: i for i, l in enumerate(LAB)}
Z, Y, GOLD = {}, {}, {}
for s in EVAL:
    items = SETS[s]
    Z[s] = nli.logits([x["evidence"] for x in items], [x["claim"] for x in items])
    GOLD[s] = np.array(["SUP" if x["label"] == "PART" else x["label"] for x in items])   # PART (synthetic) counts as not-SUP-correct below
    Y[s] = np.array([L2I.get(g, 0) for g in GOLD[s]])
PARTGOLD = np.array([x["label"] for x in SETS["synthetic"]])
print({s: len(SETS[s]) for s in EVAL})
""")

md(r"""
## 3. H6a: calibration
Temperature scaling, isotonic regression and Platt scaling (both on the top-label confidence) are fitted on HealthVer dev, then evaluated on the held-out sets. ECE uses 15 equal-mass bins.
""")
code(r"""
CAL = cfg["calibration_split"]
T = fit_temperature(Z[CAL], Y[CAL])
def top(p): return p.max(1), p.argmax(1)
# isotonic / Platt map the UNCALIBRATED top-label confidence to P(correct); fitted on the calibration split
c_cal0, a_cal = top(softmax(Z[CAL])); corr_cal = a_cal == Y[CAL]
iso = IsotonicRegression(out_of_bounds="clip").fit(c_cal0, corr_cal.astype(float))
platt = LogisticRegression().fit(c_cal0.reshape(-1, 1), corr_cal.astype(int))
cal_rows, CONF = [], {}
for s in EVAL:
    if s == "synthetic": continue
    p0 = softmax(Z[s]); p1 = softmax(Z[s], T); c0, a0 = top(p0); c1, _ = top(p1); corr = a0 == Y[s]
    variants = {"uncalibrated": c0, "temperature": c1, "isotonic": iso.predict(c0), "platt": platt.predict_proba(c0.reshape(-1, 1))[:, 1]}
    CONF[s] = variants
    for name, c in variants.items():
        cal_rows.append({"set": s, "method": name, "ECE": round(ece(c, corr, cfg["ece_bins"]), 4),
                         "Brier": round(brier(p1 if name == "temperature" else p0, Y[s]), 4) if name in ("uncalibrated", "temperature") else None,
                         "accuracy": round(float(corr.mean()), 4), "mean_conf": round(float(c.mean()), 4)})
cal_df = pd.DataFrame(cal_rows)
exp.save_table("calibration", cal_df); exp.log_result("temperature", T); display(cal_df.pivot_table(index="method", columns="set", values="ECE"))
print("temperature T =", round(T, 3))
def boot_diff(fn, a, b, corr, n=cfg["n_boot"], seed=SEED):
    rng = np.random.default_rng(seed); idx = np.arange(len(corr))
    d0 = fn(a, corr) - fn(b, corr)
    bs = [fn(a[s], corr[s]) - fn(b[s], corr[s]) for s in (rng.choice(idx, len(idx)) for _ in range(n))]
    lo, hi = np.percentile(bs, [2.5, 97.5]); return {"diff": round(float(d0), 4), "ci_low": round(float(lo), 4), "ci_high": round(float(hi), 4)}
H = {"H6a": {}}
for s in cfg["heldout_sets"]:
    corr = softmax(Z[s]).argmax(1) == Y[s]
    H["H6a"][s] = boot_diff(lambda c, k: ece(c, k, cfg["ece_bins"]), CONF[s]["temperature"], CONF[s]["uncalibrated"], corr)
H["H6a"]["supported"] = all(H["H6a"][s]["ci_high"] < 0 for s in ("healthver_test", "scifact_dev"))
print(json.dumps(H["H6a"], indent=1))
""")
code(r"""
fig, axes = plt.subplots(1, len(cfg["heldout_sets"]), figsize=(15, 4), sharey=True)
for ax, s in zip(axes, cfg["heldout_sets"]):
    corr = (softmax(Z[s]).argmax(1) == Y[s]).astype(float)
    for name in ("uncalibrated", "temperature", "isotonic"):
        c = CONF[s][name]; o = np.argsort(c); parts = np.array_split(o, cfg["ece_bins"])
        ax.plot([c[p].mean() for p in parts], [corr[p].mean() for p in parts], marker="o", label=name)
    ax.plot([0, 1], [0, 1], "k--", lw=.8); ax.set_title(s); ax.set_xlabel("confidence"); ax.grid(alpha=.3)
axes[0].set_ylabel("accuracy"); axes[0].legend(fontsize=8)
plt.tight_layout(); exp.save_figure("reliability", fig); plt.show()
""")

md(r"""
## 4. Confidence signals and H6b (multi-signal confidence)
Signals per claim–evidence pair:
- calibrated verifier confidence; probability margin; entropy;
- the numeric consistency signals (unit mismatch; unsupported bare numbers);
- claim and evidence length; lexical overlap.

A logistic regression predicting "the verifier is correct" is fitted on HealthVer dev. **AURC** (area under the risk–coverage curve; lower is better) is compared with the calibrated verifier confidence alone.
""")
code(r"""
def features(items, z):
    p = softmax(z, T); srt = np.sort(p, 1)
    rows = []
    for x, pi, si in zip(items, p, srt):
        nc = numeric_consistency(x["claim"], x["evidence"]); cw = content_words(x["claim"])
        rows.append({"verifier_conf": si[-1], "margin": si[-1] - si[-2], "entropy": float(-(pi * np.log(pi + 1e-12)).sum()),
                     "unit_mismatch": int(nc["mismatch"] > 0), "unsupported_numbers": len(unsupported_numbers(x["claim"], x["evidence"])),
                     "claim_words": len(x["claim"].split()), "evidence_words": len(x["evidence"].split()),
                     "lexical_overlap": len(cw & content_words(x["evidence"])) / max(1, len(cw))})
    return pd.DataFrame(rows)
FEAT = {s: features(SETS[s], Z[s]) for s in EVAL}
SIG = [s for s in cfg["signals"] if s != "llm_agrees"]
corr_of = lambda s: (softmax(Z[s]).argmax(1) == Y[s])
mu, sd = FEAT[CAL][SIG].mean(), FEAT[CAL][SIG].std().replace(0, 1)
ms = LogisticRegression(max_iter=1000).fit(((FEAT[CAL][SIG] - mu) / sd).values, corr_of(CAL).astype(int))
coef = pd.Series(ms.coef_[0], index=SIG).round(3)
MULTI = {s: ms.predict_proba(((FEAT[s][SIG] - mu) / sd).values)[:, 1] for s in EVAL}
rows = []
for s in EVAL:
    corr = corr_of(s) if s != "synthetic" else (np.array(LAB)[softmax(Z[s]).argmax(1)] == PARTGOLD)
    rows.append({"set": s, "AURC_verifier_conf": round(aurc(FEAT[s]["verifier_conf"].values, corr), 4), "AURC_multi_signal": round(aurc(MULTI[s], corr), 4),
                 "AURC_oracle": round(aurc(corr.astype(float), corr), 4), "error_rate": round(1 - corr.mean(), 4)})
aurc_df = pd.DataFrame(rows).set_index("set")
H["H6b"] = {s: boot_diff(aurc, MULTI[s], FEAT[s]["verifier_conf"].values, corr_of(s)) for s in cfg["heldout_sets"]}
H["H6b"]["supported"] = all(H["H6b"][s]["ci_high"] < 0 for s in ("healthver_test", "scifact_dev"))
exp.save_table("aurc", aurc_df); exp.log_result("multi_signal_coefficients", coef.to_dict())
display(aurc_df); print("coefficients (standardized):", coef.to_dict()); print(json.dumps(H["H6b"], indent=1))
""")
code(r"""
# exploratory: does agreement with the Qwen3-8B verifier (cached NB06 calls; fixed samples) add information?
SAMPLE = llm_samples(SETS, vcfg["llm_samples"], ["healthver_dev", "healthver_test", "scifact_dev", "healthfc", "synthetic"], SEED)
QWEN = {}
nli.model.to("cpu"); torch.cuda.empty_cache()   # if any call is not cached, the LLM server needs the GPU (budget)
with managed("qwen3-8b") as mc:
    v = LLMVerifier(mc, seed=SEED)
    for s in EVAL:
        QWEN[s] = {i: (v.verify(SETS[s][i]["evidence"], SETS[s][i]["claim"], tag=f"verify:{s}:{SETS[s][i]['id']}")[0] or "INS") for i in SAMPLE[s]}
    qwen_live = mc.stats
nli.model.to(DEV)
exp.log_result("qwen_replay", {"live_server_started": qwen_live is not None, "stats": qwen_live})
agree_rows = []
for s in cfg["heldout_sets"]:
    ii = np.array(sorted(QWEN[s])); corr = corr_of(s)[ii]
    nli_lab = np.array(LAB)[softmax(Z[s]).argmax(1)][ii]; q = np.array([QWEN[s][i] for i in ii]); q = np.where(q == "PART", "SUP", q)
    agrees = (q == nli_lab).astype(float)
    agree_rows.append({"set": s, "n": len(ii), "agreement": round(agrees.mean(), 3),
                       "accuracy_when_agree": round(corr[agrees == 1].mean(), 3) if agrees.sum() else None,
                       "accuracy_when_disagree": round(corr[agrees == 0].mean(), 3) if (agrees == 0).sum() else None,
                       "AURC_conf": round(aurc(FEAT[s]["verifier_conf"].values[ii], corr), 4),
                       "AURC_conf_plus_agree": round(aurc(FEAT[s]["verifier_conf"].values[ii] + agrees, corr), 4)})
agree_df = pd.DataFrame(agree_rows); exp.save_table("llm_agreement", agree_df); display(agree_df)
""")

md("## 5. H6c: abstention")
code(r"""
thr80 = threshold_for_coverage(MULTI[CAL], cfg["abstention"]["target_coverage"])
abst_rows = []
for s in cfg["heldout_sets"]:
    pred = np.array(LAB)[softmax(Z[s]).argmax(1)]
    full = selective(np.ones(len(pred)), pred, GOLD[s], 0.0); sel = selective(MULTI[s], pred, GOLD[s], thr80)
    abst_rows.append({"set": s, "coverage": round(sel["coverage"], 3), "FSR_all": round(full["false_support_rate"], 4), "FSR_answered": round(sel["false_support_rate"], 4),
                      "risk_all": round(full["risk"], 4), "risk_answered": round(sel["risk"], 4)})
abst_df = pd.DataFrame(abst_rows).set_index("set")
s = "healthver_test"; pred = np.array(LAB)[softmax(Z[s]).argmax(1)]; g = GOLD[s]
rng = np.random.default_rng(SEED); idx = np.arange(len(g)); diffs = []
for b in range(cfg["n_boot"]):
    k = rng.choice(idx, len(idx))
    diffs.append(selective(MULTI[s][k], pred[k], g[k], thr80)["false_support_rate"] - selective(np.ones(len(k)), pred[k], g[k], 0.0)["false_support_rate"])
lo, hi = np.percentile(diffs, [2.5, 97.5])
H["H6c"] = {"threshold": round(thr80, 4), "healthver_test": {"diff": round(abst_df.loc[s, "FSR_answered"] - abst_df.loc[s, "FSR_all"], 4), "ci_low": round(float(lo), 4),
            "ci_high": round(float(hi), 4), "coverage": abst_df.loc[s, "coverage"]}}
H["H6c"]["supported"] = bool(hi < 0 and abst_df.loc[s, "coverage"] >= 0.75)
# coverage needed on the calibration split to reach FSR ≤ ε
eps = cfg["abstention"]["epsilon_false_support"]; predc = np.array(LAB)[softmax(Z[CAL]).argmax(1)]
cov_eps = max((c for c in np.linspace(0.05, 1.0, 96) if selective(MULTI[CAL], predc, GOLD[CAL], threshold_for_coverage(MULTI[CAL], c))["false_support_rate"] <= eps), default=None)
H["H6c"]["calibration_coverage_at_FSR<=eps"] = cov_eps
exp.save_table("abstention", abst_df); exp.log_result("hypotheses", H); display(abst_df); print(json.dumps(H["H6c"], indent=1))
fig, ax = plt.subplots(figsize=(7, 4))
for s in cfg["heldout_sets"]:
    corr = corr_of(s); cov, risk = risk_coverage(MULTI[s], corr); ax.plot(cov, risk, label=f"{s} multi")
    cov, risk = risk_coverage(FEAT[s]["verifier_conf"].values, corr); ax.plot(cov, risk, "--", label=f"{s} conf")
ax.axvline(0.8, c="grey", lw=.8); ax.set_xlabel("coverage"); ax.set_ylabel("risk (error rate)"); ax.legend(fontsize=7); ax.grid(alpha=.3)
exp.save_figure("risk_coverage", fig); plt.show()
""")

md(r"""
## 6. Carry-over decisions
### 6.1 Second opinion: route low-confidence NLI decisions to Qwen3-8B (fixed samples)
""")
code(r"""
q_thr = float(np.quantile(MULTI[CAL], cfg["second_opinion"]["route_below_quantile"]))
def macro_f1(g, p, labels=("SUP", "CON", "INS")):
    f = []
    for l in labels:
        tp = sum(a == l and b == l for a, b in zip(g, p)); fp = sum(a != l and b == l for a, b in zip(g, p)); fn = sum(a == l and b != l for a, b in zip(g, p))
        f.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return float(np.mean(f))
so_rows = []
for s in cfg["heldout_sets"]:
    ii = np.array(sorted(QWEN[s])); nl = np.array(LAB)[softmax(Z[s]).argmax(1)][ii]; q = np.array([QWEN[s][i] for i in ii]); q = np.where(q == "PART", "SUP", q)
    route = MULTI[s][ii] < q_thr; mixed = np.where(route, q, nl); g = GOLD[s][ii]
    so_rows.append({"set": s, "n": len(ii), "routed_share": round(route.mean(), 3), "F1_nli": round(macro_f1(g, nl), 4), "F1_qwen": round(macro_f1(g, q), 4),
                    "F1_routed": round(macro_f1(g, mixed), 4)})
so_df = pd.DataFrame(so_rows).set_index("set"); exp.save_table("second_opinion", so_df); display(so_df)
""")

md(r"""
### 6.2 Evidence aggregation for claims (SciFact dev claims, retrieved evidence; the NB06 H5c setting)
- `top1`: the label for the top-ranked evidence.
- `naive_any`: the NB06 rule (any SUP / any CON).
- `max_confidence`: the label of the evidence unit with the highest calibrated confidence among the top 3.
""")
code(r"""
from cliniver.retrieval.encoders import Encoder
from cliniver.retrieval.pipeline import build_units, unit_embeddings, EvidenceRetriever, Unit
from sentence_transformers import CrossEncoder
rcfg = yaml.safe_load((ROOT / "configs" / "retrieval.yaml").read_text(encoding="utf-8"))
UNITS, ROWS = build_units(PROC / "documents.jsonl", PROC / "chunks" / "paragraph-128" / "chunks.jsonl")
mc_ = rcfg["encoders"]["medcpt"]
DENSE = Encoder("medcpt", mc_["repo"], mc_["revision"], doc_repo=mc_["doc_repo"], doc_revision=mc_["doc_revision"], device=DEV)
UEMB = unit_embeddings(UNITS, ROWS, DENSE, INTERIM / "embeddings" / "medcpt_cliniver_chunks.npy", INTERIM / "embeddings" / "medcpt_pubmed_abstracts_whole.npy").to(DEV)
ce_ = rcfg["rerankers"]["medcpt-ce"]; CE = CrossEncoder(ce_["repo"], revision=ce_["revision"], device=DEV, max_length=512)
if DEV == "cuda": CE.model.half()
from cliniver.ingestion.scifact import parse_release
sf_corpus, _ = parse_release(sf_raw); sf_titles = corpus_titles(sf_raw)
sf_ids = list(sf_corpus); abs_units = [u for u in UNITS if u.source == "pubmed"]
pool_text = [sf_corpus[d] for d in sf_ids] + [u.text for u in abs_units]
pool_title = [sf_titles[d] for d in sf_ids] + [u.title for u in abs_units]
sf_emb = DENSE.encode_docs([t[len(ti) + 2:] if t.startswith(ti) else t for t, ti in zip(pool_text[:len(sf_ids)], pool_title[:len(sf_ids)])], pool_title[:len(sf_ids)])
pool_emb = torch.cat([sf_emb.to(DEV), UEMB[torch.tensor([u.source == "pubmed" for u in UNITS]).to(UEMB.device)]])
pool_units = [Unit(p, p, "pool", ti, tx, f"{ti}. {tx}") for p, ti, tx in zip([f"sf:{d}" for d in sf_ids] + [u.doc_id for u in abs_units], pool_title, pool_text)]
RETR_SF = EvidenceRetriever(pool_units, DENSE, pool_emb, CE, depth=100, pool=50, rrf_k=rcfg["fusion"]["rrf_k"])
import tarfile
tf = tarfile.open(fileobj=io.BytesIO(sf_raw), mode="r:gz")
dev_claims = [json.loads(l) for l in tf.extractfile(next(x for x in tf.getnames() if x.endswith("claims_dev.jsonl") and "/._" not in x and "cross" not in x)).read().decode().splitlines() if l.strip()]
if SMOKE: dev_claims = dev_claims[:40]
agg_rows, G = {"top1": [], "naive_any": [], "max_confidence": []}, []
for c in dev_claims:
    labs = {s_["label"] for d in (c.get("evidence") or {}).values() for s_ in d}
    g = "SUP" if labs == {"SUPPORT"} else "CON" if labs == {"CONTRADICT"} else "INS" if not labs else None
    if g is None: continue
    ids, _ = RETR_SF.search(c["claim"], top=3); ev = [pool_text[j] for j in ids]
    p = softmax(nli.logits(ev, [c["claim"]] * len(ev)), T); lab = [LAB[k] for k in p.argmax(1)]
    agg_rows["top1"].append(lab[0])
    agg_rows["naive_any"].append("SUP" if "SUP" in lab and "CON" not in lab else "CON" if "CON" in lab and "SUP" not in lab else (lab[int(np.argmax(p.max(1)))] if "SUP" in lab else "INS"))
    agg_rows["max_confidence"].append(lab[int(np.argmax(p.max(1)))])
    G.append(g)
agg = {k: round(macro_f1(G, v), 4) for k, v in agg_rows.items()}
best = max(agg.values()); AGG = "top1" if agg["top1"] >= best - 0.01 else max(agg, key=agg.get)
exp.log_result("aggregation", {"macro_F1": agg, "claims": len(G), "selected": AGG}); print(agg, "→ selected:", AGG)
""")

md(r"""
## 7. Final verdicts for the real claims
For each real claim:
1. The ADR-0012 retriever returns the top-3 evidence units.
2. The fine-tuned NLI labels them; the selected aggregation gives the claim verdict.
3. The multi-signal model gives a calibrated confidence (`overall`).
4. Claims in the low-confidence band (below the calibration-split 25th percentile) get a Qwen3-8B second opinion on the decisive evidence:
   - if it agrees with the NLI label, that label stands;
   - if it disagrees, the claim is **ABSTAINED** (low confidence and disagreeing verifiers).

   This combined rule governs the real-claim output only; H6c is tested on the benchmarks above with the pre-registered threshold.

The confidence vector follows research_foundation §9.7: `verifier_conf`, `retrieval_strength` (re-ranker score), `evidence_agreement`, `source_tier`, `evidence_recency_years` (relative to the snapshot), `claim_complexity`, `overall`. Output: `verdicts_calibrated.jsonl`. Research output only: never advice; every verdict states the snapshot.
""")
code(r"""
real = [json.loads(l) for l in (PROC / "responses" / "verdicts.jsonl").open(encoding="utf-8")]
if SMOKE: real = real[:40]
uid = {u.unit_id: i for i, u in enumerate(UNITS)}
from cliniver.processing.long_docs import iter_documents
from cliniver.data.report import evidence_stratum
ptypes = {d["doc_id"]: d.get("publication_types", []) for d in iter_documents(PROC / "documents.jsonl", {"pubmed", "europepmc_oa"})}
def tier(u):
    if u.source == "openfda_label": return "regulatory_label"
    if u.source == "clinicaltrials": return "trial_registry"
    if u.source == "who_gho": return "population_statistics"
    return evidence_stratum(ptypes.get(u.doc_id, []))
snap_year = int(SNAP[:4])
out_rows, routed = [], []
for r in real:
    units = [UNITS[uid[e["unit_id"]]] for e in r["evidence"] if e["unit_id"] in uid]
    ev = [u.text for u in units]
    z = nli.logits(ev, [r["claim"]] * len(ev)); p = softmax(z, T); lab = [LAB[k] for k in p.argmax(1)]
    k = 0 if AGG == "top1" else int(np.argmax(p.max(1)))
    verdict = lab[k] if AGG != "naive_any" else ("SUP" if "SUP" in lab and "CON" not in lab else "CON" if "CON" in lab and "SUP" not in lab else lab[k])
    f = features([{"claim": r["claim"], "evidence": ev[k]}], z[k:k + 1]).iloc[0]
    overall = float(ms.predict_proba((((f[SIG] - mu) / sd).values).reshape(1, -1))[0, 1])
    yrs = [snap_year - int(u.pub_date[:4]) for u in units if u.pub_date]
    out_rows.append({"claim_id": r["claim_id"], "claim": r["claim"], "nli_verdict": verdict, "decisive_unit": units[k].unit_id,
                     "confidence": {"verifier_conf": round(float(p[k].max()), 4), "retrieval_strength": r["evidence"][k]["retrieval_score"],
                                    "evidence_agreement": round(sum(l == verdict for l in lab) / len(lab), 3), "conflict": ("SUP" in lab and "CON" in lab),
                                    "source_tier": tier(units[k]), "evidence_recency_years": (snap_year - int(units[k].pub_date[:4])) if units[k].pub_date else None,
                                    "claim_complexity": len(r["claim"].split()), "unit_mismatch": int(f["unit_mismatch"]),
                                    "unsupported_numbers": int(f["unsupported_numbers"]), "overall": round(overall, 4)},
                     "generator": r.get("generator"), "is_verifiable": r.get("is_verifiable"), "corpus_snapshot": SNAP, "evidence_dates": [u.pub_date for u in units]})
    if overall < q_thr:
        routed.append(len(out_rows) - 1)
for m_ in (nli.model, DENSE.q_model, DENSE.d_model, CE.model): m_.to("cpu")   # one large model on the GPU at a time
UEMB = UEMB.cpu(); torch.cuda.empty_cache()
with managed(cfg["second_opinion"]["model"]) as mc:
    v = LLMVerifier(mc, seed=SEED)
    for i in routed:
        o = out_rows[i]; u = UNITS[uid[o["decisive_unit"]]]
        lab, info = v.verify(u.text, o["claim"], tag=f"verify:real2:{o['claim_id']}")
        o["second_opinion"] = {"model": cfg["second_opinion"]["model"], "label": lab or "INS", "rationale": info.get("rationale", "")}
    so_stats = mc.stats
for o in out_rows:
    so = o.get("second_opinion")
    if so is None:
        o["final_verdict"] = o["nli_verdict"]
    else:
        q = "SUP" if so["label"] == "PART" else so["label"]
        o["final_verdict"] = o["nli_verdict"] if q == o["nli_verdict"] else "ABSTAINED"
        o["second_opinion"]["agrees"] = q == o["nli_verdict"]
OUT = PROC / "responses" / ("verdicts_calibrated_smoke.jsonl" if SMOKE else "verdicts_calibrated.jsonl")
with OUT.open("w", encoding="utf-8") as fh:
    for o in out_rows: fh.write(json.dumps(o, ensure_ascii=False) + "\n")
fv = pd.DataFrame([{"final": o["final_verdict"], "nli": o["nli_verdict"], "routed": "second_opinion" in o, "tier": o["confidence"]["source_tier"],
                    "gen": o["generator"], "unsupported_numbers": o["confidence"]["unsupported_numbers"] > 0} for o in out_rows])
real_summary = {"claims": len(fv), "final": fv.final.value_counts().to_dict(), "nli_before": fv.nli.value_counts().to_dict(), "routed_to_llm": int(fv.routed.sum()),
                "second_opinion_disagreements_abstained": int(sum(o["final_verdict"] == "ABSTAINED" for o in out_rows)),
                "decisive_source_tier": fv.tier.value_counts().to_dict(), "final_by_generator": fv.groupby("gen").final.value_counts(normalize=True).round(3).unstack(fill_value=0).to_dict(orient="index"),
                "claims_with_unsupported_numbers": int(fv.unsupported_numbers.sum()), "second_opinion_server": so_stats}
exp.log_result("real_claims", real_summary); print(json.dumps(real_summary, indent=1, default=str))
exp.finish(); print("results:", exp.dir)
""")

sys.path.insert(0, str(Path(__file__).parent))
try:
    from nb07_discussion import DISCUSSION_CELLS
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
