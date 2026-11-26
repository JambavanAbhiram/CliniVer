"""Builds notebooks/06_claim_evidence_verification.ipynb. Discussion cells: nb06_discussion.py (written after execution).

Usage: python scripts/notebooks/build_nb06.py notebooks/06_claim_evidence_verification.ipynb
"""
import sys
from pathlib import Path

import nbformat as nbf

out = Path(sys.argv[1])
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# Notebook 06 — Claim–Evidence Verification
**CliniVer** · Experiment `EXP-06-001`

| | |
|---|---|
| **Objective** | Choose the verifier that labels each claim against evidence as SUP / PART / CON / INS, with evidence, and produce verdicts for the real claims |
| **Research question** | **RQ5:** How accurately can claim–evidence relations be classified into SUP / PART / CON / INS with open models? Which errors dominate? |
| **Hypotheses (pre-registered)** | **H5a** an adapted verifier (fine-tuned NLI or a local LLM verifier) beats zero-shot NLI on macro-F1 · **H5b** the dominant error of similarity-heavy verification is false support · **H5c** aggregating top-k evidence beats top-1 |
| **Decision rules** | `configs/verification.yaml` (fixed before running) |

**Label semantics (ADR-0005):**
- **INS** means the evidence does not settle the claim. It is not "false": absence of evidence is INS, never CON.
- **PART** means the core relation is supported but a qualifier is not, or the claim is broader or more certain than the evidence.
- The human-labelled datasets have no PART label, so PART is evaluated on **synthetic** labels-by-construction perturbations.
""")

md(r"""
## 1. Method
**§0 Evidence units:** by owner decision (ADR-0009 amendment), PubMed abstracts are indexed whole. The ADR-0012 retriever (BM25 + MedCPT → MedCPT cross-encoder @50) is rebuilt over 232,628 units: 124,918 abstracts plus 107,710 chunks of labels, full texts and trials.

**Data:**

| Set | Pairs | Use | Licence |
|---|---|---|---|
| SciFact train / dev | cited abstracts: SUPPORT → SUP, CONTRADICT → CON, cited without evidence → INS | train: fine-tuning; dev: evaluation | CC BY 4.0 / ODC-By |
| HealthVer train / dev / test | Supports → SUP, Refutes → CON, Neutral → INS | train: fine-tuning; dev: thresholds and selection; test: evaluation | CC BY 4.0 |
| HealthFC | expert verdicts on question-form claims; evidence = expert-selected sentences | evaluation only | CC BY-NC-ND 4.0 |
| Synthetic | rule perturbations of SUP pairs (negation, number → CON; hedge, population → PART; paraphrase → SUP) | PART / qualifier evaluation | derived |

**Verifiers:**
1. Majority class.
2. Similarity (MedCPT inner product; SUP above a threshold tuned on HealthVer dev, never CON).
3. Zero-shot NLI (DeBERTa-v3-large).
4. Fine-tuned NLI (DeBERTa-v3-base, MNLI/FEVER/ANLI start, 2 epochs on SciFact train + HealthVer train).
5. MiniCheck-RoBERTa-Large: reference input format (`doc</s>claim`, documents chunked); threshold tuned on dev.
6. Hybrid: fine-tuned NLI + a qualifier-mismatch rule → PART.
7. LLM verifiers: Qwen3-8B, and Granite 3.3 8B as a different family; JSON schema over four labels with a quote and a rationale; on fixed-seed samples.

**Metrics:**
- Macro-F1 over SUP / CON / INS (PART counted as SUP where the gold set has no PART), per-class F1, confusion matrices.
- False-support rate (predicted SUP; gold CON or INS) and false-contradiction rate (predicted CON; gold SUP or INS).
- Paired bootstrap CIs.

**H5c:** SciFact dev claims (including NEI claims → INS) retrieved over SciFact + CliniVer abstracts. Claim label from k evidence units by the aggregation rule (`aggregate`): SUP if some evidence is SUP and none CON; CON if the reverse; both → the higher-probability label with `conflict = true`; none → INS. Also: oracle (gold) evidence.
""")

md("## 2. Setup")
code(r"""
import json, os, random, re, time, collections, io, csv
from pathlib import Path
from collections import Counter
import numpy as np, pandas as pd, matplotlib.pyplot as plt, torch, yaml
from IPython.display import display

from cliniver.config import load_project_config
from cliniver.config.llm_config import load_llm_config
from cliniver.utils.seed import set_seed
from cliniver.utils.experiment import Experiment
from cliniver.ingestion.http import RawStore, RateLimitedClient, cached_get
from cliniver.ingestion.scifact import parse_release, corpus_titles
from cliniver.retrieval.encoders import Encoder
from cliniver.retrieval.pipeline import build_units, unit_embeddings, EvidenceRetriever
from cliniver.verification.verifiers import NLIVerifier, finetune_nli, MiniCheckVerifier, LLMVerifier, SimilarityVerifier, qualifier_part
from cliniver.verification.perturb import PERTURBATIONS
from cliniver.llm.runtime import ManagedClient
from sentence_transformers import CrossEncoder

pcfg = load_project_config(); lcfg = load_llm_config()
ROOT = pcfg.paths.absolute("data_raw").parents[1]
cfg = yaml.safe_load((ROOT / "configs" / "verification.yaml").read_text(encoding="utf-8"))
rcfg = yaml.safe_load((ROOT / "configs" / "retrieval.yaml").read_text(encoding="utf-8"))
SEED = set_seed(cfg["seed"]); SNAP = cfg["snapshot"]
SMOKE = os.environ.get("CLINIVER_SMOKE") == "1"   # code check on small subsets; never results
RAW = pcfg.paths.absolute("data_raw"); PROC = pcfg.paths.absolute("data_processed") / SNAP
EMB = pcfg.paths.absolute("data_interim") / SNAP / "embeddings"; LLM_LOG = pcfg.paths.absolute("data_interim") / SNAP / "llm_calls"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
pd.set_option("display.width", 170); pd.set_option("display.max_colwidth", 110)
exp = Experiment(exp_id="EXP-06-000-smoke" if SMOKE else "EXP-06-001", title="Claim-evidence verification", research_question="RQ5 / H5a-H5c",
                 config={"verification": cfg, "retrieval": rcfg}, seed=SEED, packages=["torch", "transformers", "sentence-transformers", "bm25s", "pandas", "cliniver"])
import gc
def offload(*mods):
    "Move PyTorch models/tensors to the CPU while the LLM server needs the GPU (one large model at a time)."
    for m in mods:
        if hasattr(m, "to"): m.to("cpu")
    gc.collect(); torch.cuda.empty_cache()
def onload(*mods):
    for m in mods:
        if hasattr(m, "to"): m.to(DEV)
def peak_reset():
    if DEV == "cuda": torch.cuda.reset_peak_memory_stats()
def peak():
    return round(torch.cuda.max_memory_allocated() / 2**20) if DEV == "cuda" else None
VRAM = {}
def managed(key):
    return ManagedClient(lcfg, key, LLM_LOG / f"{key}.jsonl", server_log=pcfg.paths.absolute("logs") / f"llama-server_nb06_{key}.log")
print("seed", SEED, "| device", DEV, "| smoke", SMOKE)
""")

md("## 3. §0 Evidence units (whole abstracts) and the ADR-0012 retriever")
code(r"""
t0 = time.perf_counter()
UNITS, ROWS = build_units(PROC / "documents.jsonl", PROC / "chunks" / "paragraph-128" / "chunks.jsonl")
mc_ = rcfg["encoders"]["medcpt"]
DENSE = Encoder("medcpt", mc_["repo"], mc_["revision"], doc_repo=mc_["doc_repo"], doc_revision=mc_["doc_revision"], device=DEV,
                progress_log=str(pcfg.paths.absolute("logs") / "encode_progress.log"))
UEMB = unit_embeddings(UNITS, ROWS, DENSE, EMB / "medcpt_cliniver_chunks.npy", EMB / "medcpt_pubmed_abstracts_whole.npy").to(DEV)
ce_ = rcfg["rerankers"]["medcpt-ce"]
CE = CrossEncoder(ce_["repo"], revision=ce_["revision"], device=DEV, max_length=512)
if DEV == "cuda": CE.model.half()
RETR = EvidenceRetriever(UNITS, DENSE, UEMB, CE, depth=rcfg["first_stage_depth"], pool=50, rrf_k=rcfg["fusion"]["rrf_k"])
units_summary = {"units": len(UNITS), "by_source": dict(Counter(u.source for u in UNITS)), "bm25_index_s": round(RETR.bm25_index_s, 1),
                 "dense_index_MB": round(UEMB.numel() * 2 / 2**20), "build_s": round(time.perf_counter() - t0)}
exp.log_result("evidence_units", units_summary); print(json.dumps(units_summary, indent=1))
""")

md("## 4. Data")
code(r"""
http = RateLimitedClient("datasets", min_interval_s=1.0, max_in_flight=1)
sf_raw = cached_get(http, RawStore(RAW, "scifact", SNAP), "data", "tar.gz", cfg["datasets"]["scifact_url"])
sf_corpus, _ = parse_release(sf_raw)
import tarfile
tf = tarfile.open(fileobj=io.BytesIO(sf_raw), mode="r:gz")
def read_claims(split):
    n = next(x for x in tf.getnames() if x.endswith(f"claims_{split}.jsonl") and "/._" not in x and "cross_validation" not in x)
    return [json.loads(l) for l in tf.extractfile(n).read().decode().splitlines() if l.strip()]
def sf_pairs(split):
    out = []
    for c in read_claims(split):
        ev = c.get("evidence") or {}
        for d in c.get("cited_doc_ids", []):
            labs = {s["label"] for s in ev.get(str(d), [])}
            lab = "SUP" if "SUPPORT" in labs else "CON" if "CONTRADICT" in labs else "INS"
            if str(d) in sf_corpus:
                out.append({"id": f"sf{split}:{c['id']}:{d}", "claim": c["claim"], "evidence": sf_corpus[str(d)], "label": lab, "set": f"scifact_{split}"})
    return out
def hv(split):
    raw = cached_get(http, RawStore(RAW, "healthver", SNAP), f"healthver_{split}", "csv",
                     f"https://raw.githubusercontent.com/sarrouti/HealthVer/{cfg['datasets']['healthver']['commit']}/data/healthver_{split}.csv")
    m = {"Supports": "SUP", "Refutes": "CON", "Neutral": "INS"}
    return [{"id": f"hv{split}:{r['id']}", "claim": r["claim"], "evidence": r["evidence"], "label": m[r["label"]], "set": f"healthver_{split}"}
            for r in csv.DictReader(io.StringIO(raw.decode("utf-8")))]
fc_raw = cached_get(http, RawStore(RAW, "healthfc", SNAP), "Datensatz", "csv",
                    f"https://raw.githubusercontent.com/jvladika/HealthFC/{cfg['datasets']['healthfc']['commit']}/Datensatz.csv")
fc_rows = list(csv.DictReader(io.StringIO(fc_raw.decode("utf-8-sig"))))
# verify the HealthFC label mapping against the German verdict wording before using it
fc_check = pd.crosstab(pd.Series([r["label"] for r in fc_rows], name="label"),
                       pd.Series(["yes" if re.search(r"\bja\b|wahrscheinlich$|^möglicherweise$|für die", r["de_verdict"], re.I) and not re.search(r"nein|nicht|gegen", r["de_verdict"], re.I)
                                  else "no" if re.search(r"nein|nicht\b|gegen", r["de_verdict"], re.I) else "unclear/insufficient" for r in fc_rows], name="verdict wording"))
display(fc_check)
FCMAP = cfg["datasets"]["healthfc_label_map"]
healthfc = [{"id": f"fc:{i}", "claim": r["en_claim"], "evidence": r["en_top_sentences"], "label": FCMAP[r["label"]], "set": "healthfc"} for i, r in enumerate(fc_rows)]
SETS = {"scifact_train": sf_pairs("train"), "scifact_dev": sf_pairs("dev"), "healthver_train": hv("train"), "healthver_dev": hv("dev"),
        "healthver_test": hv("test"), "healthfc": healthfc}
if SMOKE:
    SETS = {k: random.Random(SEED).sample(v, min(len(v), 60)) for k, v in SETS.items()}
# synthetic perturbations of SUP pairs (SciFact dev + HealthVer test)
syn = []
for p in [x for x in SETS["scifact_dev"] + SETS["healthver_test"] if x["label"] == "SUP"]:
    for kind, (fn, lab) in PERTURBATIONS.items():
        c2 = fn(p["claim"])
        if c2 and c2 != p["claim"]:
            syn.append({"id": f"syn:{kind}:{p['id']}", "claim": c2, "evidence": p["evidence"], "label": lab, "set": "synthetic", "kind": kind, "orig": p["claim"]})
SETS["synthetic"] = syn
data_summary = {k: {"pairs": len(v), **Counter(x["label"] for x in v)} for k, v in SETS.items()}
data_summary["synthetic_by_kind"] = dict(Counter(x["kind"] for x in syn))
exp.log_result("data", data_summary); display(pd.DataFrame(data_summary).T)
""")

md("## 5. Verifiers on the full sets (fast verifiers)")
code(r"""
EVAL_SETS = ["healthver_dev", "healthver_test", "scifact_dev", "healthfc", "synthetic"]
PRED = {}   # (verifier, set) -> list of labels ; PROB -> arrays
PROB = {}
majority = Counter(x["label"] for x in SETS["healthver_train"] + SETS["scifact_train"]).most_common(1)[0][0]
for s in EVAL_SETS:
    PRED[("majority", s)] = [majority] * len(SETS[s])
# similarity: MedCPT inner product (query encoder for the claim, article encoder for the evidence)
def sims(items):
    q = DENSE.encode_queries([x["claim"] for x in items]); d = DENSE.encode_docs([x["evidence"] for x in items])
    return (q.float() * d.float()).sum(1).cpu().numpy()
SIM = {s: sims(SETS[s]) for s in EVAL_SETS}
def macro_f1(gold, pred, labels=("SUP", "CON", "INS")):
    f = []
    for l in labels:
        tp = sum(g == l and p == l for g, p in zip(gold, pred)); fp = sum(g != l and p == l for g, p in zip(gold, pred)); fn = sum(g == l and p != l for g, p in zip(gold, pred))
        f.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return float(np.mean(f))
gold = {s: [x["label"] for x in SETS[s]] for s in EVAL_SETS}
grid = np.quantile(SIM["healthver_dev"], np.linspace(0.05, 0.95, 37))
T_SIM = max(grid, key=lambda t: macro_f1(gold["healthver_dev"], SimilarityVerifier(t).predict(SIM["healthver_dev"])))
for s in EVAL_SETS:
    PRED[("similarity", s)] = SimilarityVerifier(T_SIM).predict(SIM[s])
# zero-shot NLI
zs = cfg["models"]["zero_shot_nli"]
nli0 = NLIVerifier(zs["repo"], zs["revision"], device=DEV)
for s in EVAL_SETS:
    PRED[("zero-shot NLI", s)], PROB[("zero-shot NLI", s)] = nli0.predict([x["evidence"] for x in SETS[s]], [x["claim"] for x in SETS[s]])
del nli0; torch.cuda.empty_cache()
# MiniCheck (threshold tuned on HealthVer dev)
mcfg = cfg["models"]["minicheck"]
mck = MiniCheckVerifier(mcfg["repo"], mcfg["revision"], device=DEV)
MCP = {s: mck.prob_supported([x["evidence"] for x in SETS[s]], [x["claim"] for x in SETS[s]]) for s in EVAL_SETS}
T_MC = max(np.linspace(0.02, 0.98, 49), key=lambda t: macro_f1(gold["healthver_dev"], ["SUP" if p >= t else "INS" for p in MCP["healthver_dev"]]))
for s in EVAL_SETS:
    PRED[("MiniCheck", s)] = ["SUP" if p >= T_MC else "INS" for p in MCP[s]]
del mck; torch.cuda.empty_cache()
exp.log_result("thresholds", {"similarity": float(T_SIM), "minicheck": float(T_MC), "majority": majority})
""")
code(r"""
# fine-tuned NLI on SciFact train + HealthVer train
ft = cfg["finetune"]; base = cfg["models"]["finetune_base"]
train = [(x["evidence"], x["claim"], x["label"]) for x in SETS["scifact_train"] + SETS["healthver_train"]]
import hashlib
ft_key = hashlib.sha256(json.dumps([base, ft, SEED, len(train), SMOKE], sort_keys=True).encode()).hexdigest()[:12]
FT_DIR = pcfg.paths.absolute("data_interim") / SNAP / "models" / f"nli_ft_{ft_key}"   # cached fine-tuned weights (replays skip training)
t0 = time.perf_counter()
offload(DENSE.q_model, DENSE.d_model, CE.model); UEMB = UEMB.cpu()   # keep the GPU for training only (budget)
if (FT_DIR / "config.json").exists():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    ft_model = AutoModelForSequenceClassification.from_pretrained(FT_DIR).to(DEV).eval(); ft_tok = AutoTokenizer.from_pretrained(FT_DIR)
    ft_hist = json.loads((FT_DIR / "history.json").read_text(encoding="utf-8")); print("loaded cached fine-tuned model", FT_DIR.name)
else:
    ft_model, ft_tok, ft_hist = finetune_nli(base["repo"], base["revision"], train, epochs=ft["epochs"], lr=ft["lr"], batch_size=ft["batch_size"] // 2,
                                             grad_accum=2, max_length=ft["max_length"], seed=SEED, device=DEV)
    FT_DIR.mkdir(parents=True, exist_ok=True); ft_model.save_pretrained(FT_DIR); ft_tok.save_pretrained(FT_DIR)
    (FT_DIR / "history.json").write_text(json.dumps(ft_hist), encoding="utf-8")
onload(DENSE.q_model, DENSE.d_model, CE.model); UEMB = UEMB.to(DEV); RETR.emb = UEMB
ft_s = time.perf_counter() - t0
nli1 = NLIVerifier(base["repo"], base["revision"], device=DEV, model=ft_model, tokenizer=ft_tok, max_length=ft["max_length"])
for s in EVAL_SETS:
    PRED[("fine-tuned NLI", s)], PROB[("fine-tuned NLI", s)] = nli1.predict([x["evidence"] for x in SETS[s]], [x["claim"] for x in SETS[s]])
    PRED[("hybrid NLI+rules", s)] = [qualifier_part(x["claim"], x["evidence"], l) for x, l in zip(SETS[s], PRED[("fine-tuned NLI", s)])]
exp.log_result("finetune", {"pairs": len(train), "seconds": round(ft_s), "history": ft_hist, "micro_batch": ft["batch_size"] // 2, "grad_accum": 2})
print(f"fine-tuning {ft_s:.0f}s", ft_hist)
""")

md("## 6. LLM verifiers (fixed-seed samples)")
code(r"""
S = dict(cfg["llm_samples"])
if SMOKE:
    S = {k: (10 if not isinstance(v, list) else v) for k, v in S.items()}
SAMPLE = {}
for s in EVAL_SETS:
    n = S.get(s, "all")
    idx = list(range(len(SETS[s])))
    SAMPLE[s] = idx if n == "all" or n >= len(idx) else sorted(random.Random(SEED).sample(idx, n))
LLM_INVALID = Counter()
GPU_SIDE = [DENSE.q_model, DENSE.d_model, CE.model, nli1.model]
offload(*GPU_SIDE); UEMB = UEMB.cpu()
for key in cfg["models"]["llm_verifiers"]:
    sets = S["granite_only"] if key.startswith("granite") else EVAL_SETS
    t0 = time.perf_counter()
    with managed(key) as mc:
        v = LLMVerifier(mc, seed=SEED)
        for s in sets:
            preds = {}
            for i in SAMPLE[s]:
                x = SETS[s][i]
                lab, info = v.verify(x["evidence"], x["claim"], tag=f"verify:{s}:{x['id']}")
                if lab is None:
                    LLM_INVALID[(key, s)] += 1; lab = "INS"
                preds[i] = lab
            PRED[(f"LLM {key}", s)] = preds
        stats = mc.stats
    VRAM[f"llm_{key}"] = stats
    print(f"{key}: {time.perf_counter()-t0:.0f}s; server {stats}")
onload(*GPU_SIDE); UEMB = UEMB.to(DEV); RETR.emb = UEMB
exp.log_result("llm_invalid_outputs", {f"{k}|{s}": v for (k, s), v in LLM_INVALID.items()})
""")

md("## 7. Evaluation")
code(r"""
def per_class(g, p):
    out = {}
    for l in ("SUP", "PART", "CON", "INS"):
        tp = sum(a == l and b == l for a, b in zip(g, p)); fp = sum(a != l and b == l for a, b in zip(g, p)); fn = sum(a == l and b != l for a, b in zip(g, p))
        out[f"F1_{l}"] = round(2 * tp / (2 * tp + fp + fn), 3) if (tp + fp + fn) else None
    return out
def collapse(p, s):  # datasets without PART: PART counts as SUP
    return p if s == "synthetic" else ["SUP" if x == "PART" else x for x in p]
rows = []
for (ver, s), p in PRED.items():
    idx = sorted(p) if isinstance(p, dict) else list(range(len(p)))
    pl = collapse([p[i] for i in idx], s); g = [gold[s][i] for i in idx]
    labels = ("SUP", "PART", "CON", "INS") if s == "synthetic" else ("SUP", "CON", "INS")
    fs = sum(b == "SUP" and a in ("CON", "INS") for a, b in zip(g, pl)) / max(1, sum(a in ("CON", "INS") for a in g))
    fc = sum(b == "CON" and a in ("SUP", "INS") for a, b in zip(g, pl)) / max(1, sum(a in ("SUP", "INS") for a in g))
    rows.append({"verifier": ver, "set": s, "n": len(idx), "sample": isinstance(p, dict), "macro_F1": round(macro_f1(g, pl, labels), 4),
                 "accuracy": round(np.mean([a == b for a, b in zip(g, pl)]), 4), "false_support_rate": round(fs, 4), "false_contradiction_rate": round(fc, 4), **per_class(g, pl)})
RES = pd.DataFrame(rows)
exp.save_table("verifier_results", RES); exp.log_result("verifier_results", RES.to_dict(orient="records"))
tab = RES.pivot_table(index="verifier", columns="set", values="macro_F1")
display(tab.round(3))
# all verifiers on the LLM samples too (fair comparison)
srows = []
for (ver, s), p in PRED.items():
    if isinstance(p, dict): continue
    ii = SAMPLE[s]; pl = collapse([p[i] for i in ii], s); g = [gold[s][i] for i in ii]
    srows.append({"verifier": ver, "set": s, "macro_F1_on_LLM_sample": round(macro_f1(g, pl, ("SUP", "PART", "CON", "INS") if s == "synthetic" else ("SUP", "CON", "INS")), 4)})
for (ver, s), p in PRED.items():
    if isinstance(p, dict):
        ii = sorted(p); g = [gold[s][i] for i in ii]
        srows.append({"verifier": ver, "set": s, "macro_F1_on_LLM_sample": round(macro_f1(g, collapse([p[i] for i in ii], s), ("SUP", "PART", "CON", "INS") if s == "synthetic" else ("SUP", "CON", "INS")), 4)})
SRES = pd.DataFrame(srows).pivot_table(index="verifier", columns="set", values="macro_F1_on_LLM_sample")
exp.save_table("verifier_results_on_llm_samples", SRES); display(SRES.round(3))
""")
code(r"""
NB = cfg["decision_rules"]["n_boot"]
def boot_f1(a_pred, b_pred, g, n=NB, seed=SEED):
    rng = np.random.default_rng(seed); idx = np.arange(len(g)); g = np.array(g); a = np.array(a_pred); b = np.array(b_pred)
    d0 = macro_f1(g, a) - macro_f1(g, b)
    bs = [macro_f1(g[s], a[s]) - macro_f1(g[s], b[s]) for s in (rng.choice(idx, len(idx)) for _ in range(n))]
    lo, hi = np.percentile(bs, [2.5, 97.5]); return {"diff": round(d0, 4), "ci_low": round(float(lo), 4), "ci_high": round(float(hi), 4), "n": len(g)}
H = {"H5a": {}}
for s in ("healthver_test", "scifact_dev"):
    g = gold[s]
    H["H5a"][f"fine-tuned_vs_zero-shot|{s}"] = boot_f1(collapse(PRED[("fine-tuned NLI", s)], s), PRED[("zero-shot NLI", s)], g)
    p = PRED[("LLM qwen3-8b", s)]; ii = sorted(p)
    H["H5a"][f"qwen_vs_zero-shot|{s}"] = boot_f1(collapse([p[i] for i in ii], s), [PRED[("zero-shot NLI", s)][i] for i in ii], [g[i] for i in ii])
ok = lambda name: all(H["H5a"][f"{name}|{s}"]["ci_low"] > 0 for s in ("healthver_test", "scifact_dev"))
H["H5a"]["supported"] = ok("fine-tuned_vs_zero-shot") or ok("qwen_vs_zero-shot")
H["H5b"] = {}
for s in ("healthver_test", "scifact_dev"):
    g, p = gold[s], PRED[("similarity", s)]
    errs = [(a, b) for a, b in zip(g, p) if a != b]
    H["H5b"][s] = {"errors": len(errs), "false_support_share": round(sum(b == "SUP" for a, b in errs) / max(1, len(errs)), 4)}
H["H5b"]["supported"] = all(H["H5b"][s]["false_support_share"] > 0.5 for s in ("healthver_test", "scifact_dev"))
exp.log_result("hypotheses_H5a_H5b", H); print(json.dumps(H, indent=1))
""")

md(r"""
### 7.1 Synthetic perturbations: which verifier catches which edit?
Accuracy per perturbation type (synthetic set; PART is only producible by the hybrid and the LLMs).
""")
code(r"""
kinds = [x["kind"] for x in SETS["synthetic"]]
pert = []
for (ver, s), p in PRED.items():
    if s != "synthetic": continue
    ii = sorted(p) if isinstance(p, dict) else range(len(kinds))
    for k in PERTURBATIONS:
        sel = [i for i in ii if kinds[i] == k]
        if sel:
            pert.append({"verifier": ver, "kind": f"{k}→{PERTURBATIONS[k][1]}", "accuracy": round(np.mean([p[i] == gold['synthetic'][i] for i in sel]), 3), "n": len(sel)})
pert = pd.DataFrame(pert).pivot_table(index="verifier", columns="kind", values="accuracy")
exp.save_table("synthetic_by_kind", pert); display(pert)
""")

md("## 8. H5c: top-k aggregation with retrieved evidence (SciFact dev claims)")
code(r"""
def aggregate(labels, probs=None):
    has_s, has_c = "SUP" in labels or "PART" in labels, "CON" in labels
    if has_s and not has_c: return ("PART" if "PART" in labels and "SUP" not in labels else "SUP"), False
    if has_c and not has_s: return "CON", False
    if has_s and has_c:
        if probs is not None:
            best = max(range(len(labels)), key=lambda i: max(probs[i][0], probs[i][1]))
            return ("CON" if probs[best][1] > probs[best][0] else "SUP"), True
        return labels[0], True
    return "INS", False
dev_claims = read_claims("dev")
if SMOKE: dev_claims = dev_claims[:40]
# SciFact + CliniVer abstracts as the evidence pool for this experiment (document-level relevance)
sf_ids = list(sf_corpus); sf_text = [sf_corpus[d] for d in sf_ids]
sf_titles = corpus_titles(sf_raw)
abs_units = [u for u in UNITS if u.source == "pubmed"]
pool_text = sf_text + [u.text for u in abs_units]; pool_title = [sf_titles[d] for d in sf_ids] + [u.title for u in abs_units]
pool_ids = [f"sf:{d}" for d in sf_ids] + [u.doc_id for u in abs_units]
sf_emb = DENSE.encode_docs([t[len(ti) + 2:] if t.startswith(ti) else t for t, ti in zip(sf_text, pool_title[:len(sf_ids)])], pool_title[:len(sf_ids)])
abs_mask = torch.tensor([u.source == "pubmed" for u in UNITS])
pool_emb = torch.cat([sf_emb.to(DEV), UEMB[abs_mask.to(UEMB.device)]])
from cliniver.retrieval.pipeline import Unit
pool_units = [Unit(pid, pid, "pool", ti, tx, f"{ti}. {tx}") for pid, ti, tx in zip(pool_ids, pool_title, pool_text)]
RETR_SF = EvidenceRetriever(pool_units, DENSE, pool_emb, CE, depth=100, pool=50, rrf_k=rcfg["fusion"]["rrf_k"])
claim_gold, ranked = [], []
for c in dev_claims:
    labs = {s["label"] for d in (c.get("evidence") or {}).values() for s in d}
    claim_gold.append("SUP" if labs == {"SUPPORT"} else "CON" if labs == {"CONTRADICT"} else "INS" if not labs else "MIXED")
    ranked.append(RETR_SF.search(c["claim"], top=max(cfg["retrieval_eval"]["ks"]))[0])
keep = [i for i, g in enumerate(claim_gold) if g != "MIXED"]
VER = nli1   # the fine-tuned NLI (fast) is used for the k-sweep; the selected verifier is used for real claims
h5c, agg_pred = {}, {}
for k in cfg["retrieval_eval"]["ks"]:
    preds = []
    for i in keep:
        ev = [pool_text[j] for j in ranked[i][:k]]
        labs, pr = VER.predict(ev, [dev_claims[i]["claim"]] * len(ev))
        preds.append(aggregate(labs, pr)[0])
    agg_pred[k] = ["SUP" if p == "PART" else p for p in preds]
oracle = []
for i in keep:
    c = dev_claims[i]; docs = [str(d) for d in c.get("cited_doc_ids", []) if str(d) in sf_corpus]
    labs, pr = VER.predict([sf_corpus[d] for d in docs], [c["claim"]] * len(docs)) if docs else ([], None)
    oracle.append(aggregate(labs, pr)[0] if docs else "INS")
G = [claim_gold[i] for i in keep]
h5c = {f"k={k}": round(macro_f1(G, agg_pred[k]), 4) for k in agg_pred} | {"oracle_cited_docs": round(macro_f1(G, oracle), 4), "claims": len(keep),
       "gold": dict(Counter(G))}
pk = cfg["retrieval_eval"]["primary_k"]
h5c["top3_vs_top1"] = boot_f1(agg_pred[pk], agg_pred[1], G)
h5c["supported"] = h5c["top3_vs_top1"]["ci_low"] > 0
exp.log_result("H5c", h5c); print(json.dumps(h5c, indent=1))
""")

md("## 9. Selection of the verifier for real claims")
code(r"""
dev = RES[(RES.set == "healthver_dev")].set_index("verifier")["macro_F1"].drop(["majority"], errors="ignore")
speed = {"similarity": 0, "zero-shot NLI": 2, "fine-tuned NLI": 1, "hybrid NLI+rules": 1, "MiniCheck": 2, "LLM qwen3-8b": 5, "LLM granite-3.3-8b": 5}
best = dev.max(); near = [v for v in dev.index if dev[v] >= best - 0.01]
SELECTED = min(near, key=lambda v: speed.get(v, 9))
exp.log_result("selection", {"healthver_dev_macro_F1": dev.round(4).to_dict(), "within_0.01_of_best": near, "selected": SELECTED})
print("selected verifier:", SELECTED, "| dev macro-F1:", dev.round(3).to_dict())
""")

md(r"""
## 10. Verdicts for the real claims (NB03–NB05 output)
Each real claim (its NB04 standalone rewrite is the query) is matched to the top-3 evidence units of the ADR-0012 retriever over the whole-abstract index. The selected verifier labels each pair, and the claim verdict is aggregated with a `conflict` flag. Every verdict carries the **corpus snapshot date** and the **evidence dates** (ADR-0005, D4). Output: `data/processed/<snapshot>/responses/verdicts.jsonl`. These are research outputs, never advice.
""")
code(r"""
real = [json.loads(l) for l in (PROC / "responses" / "claims_contextualized.jsonl").open(encoding="utf-8")]
peak_reset()
if SMOKE: real = real[:30]
K = cfg["real_claims"]["top_k"]
t0 = time.perf_counter(); lat = []
evid = []
for c in real:
    t1 = time.perf_counter(); ids, sc = RETR.search(c["standalone_text"], top=K); lat.append(time.perf_counter() - t1)
    evid.append((ids, sc))
retr_s = time.perf_counter() - t0
VRAM["real_claim_retrieval_peak_mb"] = peak()
verdicts = []
use_llm = SELECTED.startswith("LLM")
if use_llm:
    key = SELECTED.split(" ", 1)[1]
    offload(*GPU_SIDE)
    with managed(key) as mc:
        v = LLMVerifier(mc, seed=SEED)
        labs_all = [[(v.verify(UNITS[j].text, c["text"], tag=f"verify:real:{c['claim_id']}:{UNITS[j].unit_id}")[0] or "INS") for j in ids] for c, (ids, _) in zip(real, evid)]
    probs_all = [None] * len(real)
else:
    verifier = {"fine-tuned NLI": nli1, "hybrid NLI+rules": nli1}.get(SELECTED, nli1)
    labs_all, probs_all = [], []
    for c, (ids, _) in zip(real, evid):
        ev = [UNITS[j].text for j in ids]
        labs, pr = verifier.predict(ev, [c["text"]] * len(ev))
        if SELECTED == "hybrid NLI+rules":
            labs = [qualifier_part(c["text"], e, l) for e, l in zip(ev, labs)]
        labs_all.append(labs); probs_all.append(pr)
for c, (ids, sc), labs, pr in zip(real, evid, labs_all, probs_all):
    lab, conflict = aggregate(labs, pr)
    verdicts.append({"claim_id": c["claim_id"], "claim": c["text"], "query": c["standalone_text"], "verdict": lab, "conflict": conflict,
                     "verifier": SELECTED, "corpus_snapshot": SNAP,
                     "evidence": [{"unit_id": UNITS[j].unit_id, "source": UNITS[j].source, "date": UNITS[j].pub_date, "label": l,
                                   "retrieval_score": round(s, 4)} for j, s, l in zip(ids, sc, labs)],
                     "is_verifiable": c.get("is_verifiable"), "generator": c.get("generator"), "context_dependent": c.get("context_dependent")})
OUTV = PROC / "responses" / ("verdicts_smoke.jsonl" if SMOKE else "verdicts.jsonl")
with OUTV.open("w", encoding="utf-8") as fh:
    for v_ in verdicts: fh.write(json.dumps(v_, ensure_ascii=False) + "\n")
vd = pd.DataFrame(verdicts)
real_summary = {"claims": len(vd), "verdicts": vd.verdict.value_counts().to_dict(), "conflicts": int(vd.conflict.sum()),
                "by_generator": vd.groupby("generator").verdict.value_counts(normalize=True).round(3).unstack(fill_value=0).to_dict(orient="index"),
                "verifiable_only": vd[vd.is_verifiable == True].verdict.value_counts(normalize=True).round(3).to_dict(),
                "retrieval_latency_p95_s": round(float(np.percentile(lat, 95)), 3), "total_s": round(time.perf_counter() - t0)}
exp.log_result("real_claims", real_summary); exp.log_result("vram", VRAM); print(json.dumps(real_summary, indent=1)); print("VRAM:", VRAM)
display(vd[["claim", "verdict", "conflict"]].sample(min(10, len(vd)), random_state=SEED))
exp.finish(); print("results:", exp.dir)
""")

sys.path.insert(0, str(Path(__file__).parent))
try:
    from nb06_discussion import DISCUSSION_CELLS
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
