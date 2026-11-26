"""NB08 Part B: optimization round 1 (EXP-08-002). Cells are added by build_nb08.py after the Part A discussion.

Runs in the same kernel as Part A and reuses its objects (corpus, BM25, cross-encoder, v1 verifier, confidence model,
Qwen label cache, silver labels).
"""


def add_cells(md, code):
    md(r"""
# Part B — Optimization round 1 (EXP-08-002)
**Owner request 2026-11-21:** optimize the results after NB08. **This part is post-hoc with respect to EXP-08-001**: the weaknesses were found on SciFact dev, so SciFact dev and every set used in NB06–NB08 count as *seen*.

The candidates are trained and selected **without** the two fresh test sets:
1. **SciFact-train test split:** 30% of the SciFact *train* claims, grouped by cited abstract so no abstract is shared across splits, and removed from verifier training.
2. **PUBHEALTH test:** public-health fact-checks (MIT licence), never used before.

Protocol and decision rules: `configs/optimize.yaml`, written before running.

| Lever (from EXP-08-001) | Candidate |
|---|---|
| Verifier is the bottleneck; numeric blindness; false support | **v2 verifier:** v1 recipe + rule perturbations of training SUP pairs (negation / number → CON; hedge removal / population widening → INS; paraphrase → SUP) + **hard negatives** (top re-ranked, non-cited abstracts → INS) |
| Retrieval errors fall on INS claims | **INS threshold** on the top-1 cross-encoder score |
| Numbers | **Numeric guard:** a SUP verdict for a claim containing a number absent from the evidence → INS |
| The confidence layer does not help end to end | **Second-opinion policy:** none / abstain on disagreement (v1) / take the LLM label |
| Context (EXP-08-001 §10.3) | **Verify the standalone rewrite** (policy change) |

- **Baseline:** `v1r` = the v1 recipe retrained on the same training split (no augmentation), with the v1 policy. It has not seen the test claims; the deployed v1 has (it was trained on all SciFact train claims).
- **Selection** (dev split): the highest end-to-end macro-F1 (abstained = error) with false-support rate ≤ the baseline's.
""")
    code(r"""
from cliniver.verification.verifiers import finetune_nli
from cliniver.verification.perturb import PERTURBATIONS
from cliniver.verification.consistency import unsupported_numbers
from cliniver.ingestion.http import RawStore, cached_get
import hashlib, itertools
ocfg = yaml.safe_load((ROOT / "configs" / "optimize.yaml").read_text(encoding="utf-8"))
exp2 = Experiment(exp_id="EXP-08-002-smoke" if SMOKE else "EXP-08-002", title="Optimization round 1 (post-hoc, owner request)",
                  research_question="RQ5–RQ7 optimization", config={"optimize": ocfg}, seed=SEED,
                  packages=["torch", "transformers", "sentence-transformers", "bm25s", "scikit-learn", "pandas", "pyarrow", "cliniver"])
_tr = [json.loads(l) for l in tf.extractfile(next(x for x in tf.getnames() if x.endswith("claims_train.jsonl") and "/._" not in x and "cross" not in x)).read().decode().splitlines() if l.strip()]
# group claims that share a cited / evidence abstract (union-find), then split groups: no abstract crosses splits
parent = {}
def find(x):
    while parent.setdefault(x, x) != x: x = parent[x]
    return x
for c in _tr:
    docs = [f"d{d}" for d in c.get("cited_doc_ids", [])] + [f"d{d}" for d in (c.get("evidence") or {})]
    for d in docs: parent[find(d)] = find(f"c{c['id']}")
    find(f"c{c['id']}")
groups = collections.defaultdict(list)
for c in _tr: groups[find(f"c{c['id']}")].append(c["id"])
gkeys = sorted(groups, key=lambda g: min(groups[g])); random.Random(SEED).shuffle(gkeys)
fr = ocfg["splits"]["scifact_train_claims"]; n_tot = len(_tr); SPLIT = {}; acc = 0
for g in gkeys:
    s = "train" if acc < fr["train"] * n_tot else "dev" if acc < (fr["train"] + fr["dev"]) * n_tot else "test"
    for cid in groups[g]: SPLIT[cid] = s
    acc += len(groups[g])
def claim_items(claims):
    out = []
    for c in claims:
        labs = {s["label"] for d in (c.get("evidence") or {}).values() for s in d}
        g = "SUP" if labs == {"SUPPORT"} else "CON" if labs == {"CONTRADICT"} else "INS" if not labs else None
        if g is not None: out.append({"id": c["id"], "claim": c["claim"], "gold": g})
    return out
SPL_ITEMS = {s: claim_items([c for c in _tr if SPLIT[c["id"]] == s]) for s in ("dev", "test")}
if SMOKE: SPL_ITEMS = {k: v[:30] for k, v in SPL_ITEMS.items()}
split_info = {s: {"claims": sum(v == s for v in SPLIT.values()), "e2e_items": len(SPL_ITEMS.get(s, [])),
                  "gold": dict(Counter(x["gold"] for x in SPL_ITEMS.get(s, [])))} for s in ("train", "dev", "test")}
exp2.log_result("splits", split_info); print(json.dumps(split_info))
""")

    md("## 13. Retrieval for the train-split claims (hard negatives) and the dev / test splits")
    code(r"""
def retrieve(queries, pool=50):
    "ADR-0012 retrieval over the NB05 primary corpus: (re-ranked doc indices, their cross-encoder scores) per query."
    bi = bm.search(queries, DEPTH)[0]
    m = rcfg["encoders"]["medcpt"]; enc = Encoder("medcpt", m["repo"], m["revision"], doc_repo=m["doc_repo"], doc_revision=m["doc_revision"], device=DEV)
    D = torch.from_numpy(np.load(EMB / "medcpt_primary.npy")).to(DEV); di = Encoder.search(enc.encode_queries(queries), D, DEPTH)[0]; enc.unload(); del D; torch.cuda.empty_cache()
    out = []
    for q, a, b in zip(queries, bi, di):
        c = list(rrf([a, b], k=rcfg["fusion"]["rrf_k"], top=DEPTH)[:pool])
        s = np.asarray(CE.predict([(q, DOC_TEXT[j]) for j in c], batch_size=64, show_progress_bar=False), dtype=float).reshape(len(c), -1)[:, -1]
        o = np.argsort(-s); out.append(([c[k] for k in o], [float(s[k]) for k in o]))
    return out
tr_claims = [c for c in _tr if SPLIT[c["id"]] == "train"]
if SMOKE: tr_claims = tr_claims[:40]
t0 = time.perf_counter()
R_TR = retrieve([c["claim"] for c in tr_claims])
R_SPL = {s: retrieve([x["claim"] for x in v]) for s, v in SPL_ITEMS.items()}
print(f"retrieval {time.perf_counter() - t0:.0f}s for {len(tr_claims) + sum(len(v) for v in SPL_ITEMS.values())} claims")
""")

    md(r"""
## 14. Training data and the two verifiers
Both verifiers use the same base model and hyper-parameters as ADR-0013. The micro-batch is 4 × accumulation 4 (the same effective batch of 16), so the GPU stays within budget (NB06 peaked at 7.1 GB with 8 × 2).
""")
    code(r"""
tr_ids = {c["id"] for c in tr_claims}
sf_tr = [x for x in SETS["scifact_train"] if int(x["id"].split(":")[1]) in tr_ids]
base_pairs = [(x["evidence"], x["claim"], x["label"]) for x in sf_tr + SETS["healthver_train"]]
rng_aug = random.Random(SEED); aug, kinds = [], Counter()
for x in sf_tr + SETS["healthver_train"]:
    if x["label"] != "SUP" or rng_aug.random() >= ocfg["augmentation"]["perturb_fraction"]: continue
    opts = []
    for k, lab in ocfg["augmentation"]["perturbations"].items():
        c2 = PERTURBATIONS[k][0](x["claim"])
        if c2 and c2 != x["claim"]: opts.append((k, c2, lab))
    if opts:
        k, c2, lab = rng_aug.choice(opts); aug.append((x["evidence"], c2, lab)); kinds[k] += 1
hard = []
for c, (ranked, _) in zip(tr_claims, R_TR):
    gold_docs = {str(d) for d in c.get("cited_doc_ids", [])} | {str(d) for d in (c.get("evidence") or {})}
    neg = [j for j in ranked[:ocfg["augmentation"]["hard_negative_depth"]] if DOC_IDS[j] not in gold_docs][:ocfg["augmentation"]["hard_negatives_per_claim"]]
    hard += [(DOC_TEXT[j], c["claim"], "INS") for j in neg]
TRAIN = {"v1r": base_pairs, "v2": base_pairs + aug + hard}
if SMOKE: TRAIN = {k: random.Random(SEED).sample(v, 300) for k, v in TRAIN.items()}
ft = vcfg["finetune"]; fbase = vcfg["models"]["finetune_base"]
for m_ in (nli.model, CE.model): m_.to("cpu")
torch.cuda.empty_cache()
VER, FT_LOG = {}, {}
for name, data in TRAIN.items():
    key = hashlib.sha256(json.dumps([fbase, ft, SEED, name, len(data), sorted(Counter(l for _, _, l in data).items()), SMOKE], sort_keys=True).encode()).hexdigest()[:12]
    d = INTERIM / "models" / f"nli_opt_{name}_{key}"
    t0 = time.perf_counter()
    if (d / "config.json").exists():
        mdl = AutoModelForSequenceClassification.from_pretrained(d); tok = AutoTokenizer.from_pretrained(d)
        hist = json.loads((d / "history.json").read_text(encoding="utf-8")); print("cached", d.name)
    else:
        mdl, tok, hist = finetune_nli(fbase["repo"], fbase["revision"], data, epochs=1 if SMOKE else ft["epochs"], lr=ft["lr"], batch_size=4, grad_accum=4,
                                      max_length=ft["max_length"], seed=SEED, device=DEV)
        d.mkdir(parents=True, exist_ok=True); mdl.save_pretrained(d); tok.save_pretrained(d); (d / "history.json").write_text(json.dumps(hist), encoding="utf-8")
    mdl.to(DEV).eval()
    VER[name] = NLIVerifier(str(d), device=DEV, model=mdl, tokenizer=tok, max_length=ft["max_length"])
    FT_LOG[name] = {"pairs": len(data), "labels": dict(Counter(l for _, _, l in data)), "seconds": round(time.perf_counter() - t0), "history": hist, "dir": d.name}
for m_ in (nli.model, CE.model): m_.to(DEV)
FT_LOG["augmentation"] = {"perturbations": dict(kinds), "hard_negatives": len(hard)}
exp2.log_result("finetune", FT_LOG); print(json.dumps(FT_LOG, indent=1, default=str))
CMS = {"v1": CM, **{k: ConfidenceModel.fit(v, SETS["healthver_dev"]) for k, v in VER.items()}}   # calibration split = HealthVer dev, as in ADR-0014
VERS = {"v1": nli, **VER}
""")

    md(r"""
## 15. Scoring every set once
Each set is a list of (verified text, evidence) pairs with gold labels, plus the top-1 cross-encoder score when retrieval is involved. For each verifier we keep its label and its multi-signal confidence; the Qwen3-8B second opinion is computed once for every pair that any verifier routes. The LLM labels depend only on the pair and come from the replay cache when available.
""")
    code(r"""
def scored(claims, evidence, gold, top1_score=None, verifiers=("v1", "v1r", "v2")):
    d = {"Q": list(claims), "ev": list(evidence), "gold": np.array(gold), "top1_score": None if top1_score is None else np.array(top1_score), "lab": {}, "overall": {}}
    for k in verifiers:
        z = VERS[k].logits(d["ev"], d["Q"]); lab, _, ov = CMS[k].score(d["Q"], d["ev"], z)
        d["lab"][k] = np.array(lab); d["overall"][k] = ov
    return d
SCO = {}
for s in ("dev", "test"):
    SCO[s] = scored([x["claim"] for x in SPL_ITEMS[s]], [DOC_TEXT[r[0][0]] for r in R_SPL[s]], [x["gold"] for x in SPL_ITEMS[s]], [r[1][0] for r in R_SPL[s]])
SCO["scifact_dev_seen"] = scored(Q, [DOC_TEXT[r[0]] for r in RET["pipeline"]], GOLD, [r[1] for r in rr])
# PUBHEALTH (fresh): claim vs the fact-checkers' explanation
ph_cfg = ocfg["fresh_tests"]["pubhealth"]
ph_raw = cached_get(RateLimitedClient("datasets", min_interval_s=1.0, max_in_flight=1), RawStore(RAW, "pubhealth", SNAP), "test", "parquet",
                    f"https://huggingface.co/datasets/{ph_cfg['repo']}/resolve/{ph_cfg['revision']}/{ph_cfg['file']}")
PH = pd.read_parquet(io.BytesIO(ph_raw)); PH = PH[PH.label.isin(list(ph_cfg["label_map"]))].reset_index(drop=True)
if SMOKE: PH = PH.sample(60, random_state=SEED).reset_index(drop=True)
SCO["pubhealth"] = scored(PH.claim.tolist(), PH[ph_cfg["evidence_field"]].tolist(), PH.label.map(ph_cfg["label_map"]).tolist())
print({k: len(v["Q"]) for k, v in SCO.items()}, "| PUBHEALTH labels", dict(Counter(SCO["pubhealth"]["gold"])))
""")
    code(r"""
# Real claims (no gold): the verified text is the standalone rewrite (policy change); evidence = the ADR-0012 top-1 unit
from cliniver.retrieval.pipeline import build_units
UNITS, _ = build_units(PROC / "documents.jsonl", PROC / "chunks" / "paragraph-128" / "chunks.jsonl")
uid = {u.unit_id: i for i, u in enumerate(UNITS)}
real = [json.loads(l) for l in (PROC / "responses" / "verdicts.jsonl").open(encoding="utf-8")]
ctxd = {json.loads(l)["claim_id"]: json.loads(l) for l in (PROC / "responses" / "claims_contextualized.jsonl").open(encoding="utf-8")}
if SMOKE: real = real[:40]
r_units = [UNITS[uid[r["evidence"][0]["unit_id"]]] for r in real]
r_text = [ctxd[r["claim_id"]]["standalone_text"] or r["claim"] for r in real]
r_ce = [float(np.asarray(CE.predict([(r["query"], u.index_text)], show_progress_bar=False), dtype=float).reshape(-1)[-1]) for r, u in zip(real, r_units)]
SCO["real"] = scored(r_text, [u.text for u in r_units], ["UNK"] * len(real), r_ce, verifiers=("v1r", "v2"))
print(len(real), "real claims;", sum(t != r["claim"] for t, r in zip(r_text, real)), "verified as their standalone rewrite")
""")
    code(r"""
def routed(d, vk):
    return np.where(d["overall"][vk] < CMS[vk].route_threshold)[0]
need = sorted({(d["Q"][i], d["ev"][i]) for d in SCO.values() for vk in d["lab"] for i in routed(d, vk)})
print(len(need), "unique (claim, evidence) pairs routed to the second opinion;", sum(p not in QWEN_LABELS for p in need), "not yet labelled in this kernel")
for v_ in VER.values(): v_.model.to("cpu")
_ = qwen_labels(need, "opt")      # offloads the v1 verifier and the cross-encoder; one server start
for v_ in VER.values(): v_.model.to(DEV)
""")

    md("## 16. Configuration search on the dev split")
    code(r"""
TAU = {q: float(np.quantile(SCO["dev"]["top1_score"], q)) for q in ocfg["search_space"]["ins_threshold_quantile"] if q is not None}
def apply(d, vk, guard, tq, so):
    lab = d["lab"][vk].copy(); ans = np.ones(len(lab), bool)
    if guard:
        for i in np.where(lab == "SUP")[0]:
            if unsupported_numbers(d["Q"][i], d["ev"][i]): lab[i] = "INS"
    if tq is not None and d["top1_score"] is not None:
        lab[d["top1_score"] < TAU[tq]] = "INS"
    if so != "none":
        for i in routed(d, vk):
            l = QWEN_LABELS[(d["Q"][i], d["ev"][i])]; l = "SUP" if l == "PART" else l
            if l != lab[i]:
                if so == "abstain": ans[i] = False
                else: lab[i] = l
    return lab, ans
def metrics(g, p, a):
    e2e = np.where(a, p, "ABSTAINED"); den = max(1, np.sum(np.isin(g[a], ["CON", "INS"])))
    return {"coverage": round(float(a.mean()), 3), "macro_F1_e2e": round(macro_f1(g, e2e), 4), "macro_F1_answered": round(macro_f1(g[a], p[a]), 4),
            "FSR_answered": round(float(np.sum((p[a] == "SUP") & np.isin(g[a], ["CON", "INS"])) / den), 4), "accuracy_e2e": round(float(np.mean(e2e == g)), 4)}
ss = ocfg["search_space"]
GRID = list(itertools.product(ss["verifier"], ss["numeric_guard"], ss["ins_threshold_quantile"], ss["second_opinion"]))
def cname(c): return f"{c[0]}|guard={int(c[1])}|tau={c[2]}|so={c[3]}"
BASE = (ocfg["baseline"]["verifier"], ocfg["baseline"]["numeric_guard"], ocfg["baseline"]["ins_threshold_quantile"], ocfg["baseline"]["second_opinion"])
rows = []
for c in GRID:
    for s in ("dev", "test"):
        p, a = apply(SCO[s], *c); rows.append({"config": cname(c), "split": s, **metrics(SCO[s]["gold"], p, a)})
GR = pd.DataFrame(rows).pivot(index="config", columns="split")
GR.columns = [f"{m}|{s}" for m, s in GR.columns]
fsr_base = GR.loc[cname(BASE), "FSR_answered|dev"]
cand = GR[GR["FSR_answered|dev"] <= fsr_base]
best = cand["macro_F1_e2e|dev"].max()
ncomp = lambda c: int(c[1]) + int(c[2] is not None) + int(c[3] != "none")
near = [c for c in GRID if cname(c) in cand.index and cand.loc[cname(c), "macro_F1_e2e|dev"] >= best - 0.01]
SEL = min(near, key=lambda c: (ncomp(c), -cand.loc[cname(c), "macro_F1_e2e|dev"]))
sel = {"baseline": cname(BASE), "baseline_dev_FSR": fsr_base, "within_0.01_of_best": [cname(c) for c in near], "selected": cname(SEL), "tau_values": TAU}
exp2.save_table("grid", GR); exp2.log_result("selection", sel); print(json.dumps(sel, indent=1))
display(GR.sort_values("macro_F1_e2e|dev", ascending=False).head(15)[[c for c in GR.columns if c.endswith("|dev")]])
""")

    md("## 17. Fresh tests: SciFact-train test split and PUBHEALTH")
    code(r"""
def paired_ci(d, ca, cb, n=ocfg["n_boot"]):
    "95% bootstrap CI of (b − a) for end-to-end macro-F1 and FSR(answered), paired over items."
    g = d["gold"]; pa, aa = apply(d, *ca); pb, ab = apply(d, *cb)
    ea, eb = np.where(aa, pa, "ABSTAINED"), np.where(ab, pb, "ABSTAINED")
    rng = np.random.default_rng(SEED); idx = np.arange(len(g)); df, dfs = [], []
    def fsr(p, a, s):
        m = a[s]; den = max(1, np.sum(np.isin(g[s][m], ["CON", "INS"]))); return np.sum((p[s][m] == "SUP") & np.isin(g[s][m], ["CON", "INS"])) / den
    for _ in range(n):
        s = rng.choice(idx, len(idx)); df.append(macro_f1(g[s], eb[s]) - macro_f1(g[s], ea[s])); dfs.append(fsr(pb, ab, s) - fsr(pa, aa, s))
    return {"delta_F1_e2e": round(macro_f1(g, eb) - macro_f1(g, ea), 4), "ci_F1_e2e": [round(float(x), 4) for x in np.percentile(df, [2.5, 97.5])],
            "delta_FSR": round(float(np.mean(dfs)), 4), "ci_FSR": [round(float(x), 4) for x in np.percentile(dfs, [2.5, 97.5])]}
REPORT = {}
for s in ("test", "pubhealth"):
    d = SCO[s]; tab = {}
    for nm, c in {"baseline (v1r, v1 policy)": BASE, "selected": SEL, "v2, v1 policy": ("v2", False, None, "abstain"),
                  "v2, verifier only": ("v2", False, None, "none"), "v1r, verifier only": ("v1r", False, None, "none")}.items():
        p, a = apply(d, *c); tab[nm] = metrics(d["gold"], p, a)
    if s == "pubhealth":
        for nm, c in {"v1 deployed, v1 policy": ("v1", False, None, "abstain"), "v1 deployed, verifier only": ("v1", False, None, "none")}.items():
            p, a = apply(d, *c); tab[nm] = metrics(d["gold"], p, a)
    REPORT[s] = {"table": tab, "selected_vs_baseline": paired_ci(d, BASE, SEL)}
    print(s); display(pd.DataFrame(tab).T); print(REPORT[s]["selected_vs_baseline"])
exp2.log_result("fresh_tests", REPORT)
""")

    md("## 18. Seen sets (secondary): SciFact dev end to end, verifier-level sets, synthetic edits, context items")
    code(r"""
seen = {}
d = SCO["scifact_dev_seen"]
seen["scifact_dev_e2e"] = {nm: metrics(d["gold"], *apply(d, *c)) for nm, c in
                          {"v1 deployed, v1 policy (EXP-08-001 pipeline)": ("v1", False, None, "abstain"), "baseline (v1r, v1 policy)": BASE, "selected": SEL}.items()}
lvl = {}
for s in ("healthver_test", "healthfc", "synthetic"):
    X = SETS[s] if not SMOKE else SETS[s][:200]
    g = np.array([x["label"] for x in X]); ev = [x["evidence"] for x in X]; cl = [x["claim"] for x in X]
    for k in ("v1", "v1r", "v2"):
        lab, _ = VERS[k].predict(ev, cl); lab = np.array(lab)
        labs = ("SUP", "CON", "INS") if s != "synthetic" else ("SUP", "CON", "INS", "PART")
        lvl[(s, k)] = {"macro_F1": round(macro_f1(g, lab, labs), 4)}
        if s == "synthetic":
            kind = np.array([x["kind"] for x in X])
            for kk in ("number", "negation"):
                lvl[(s, k)][f"{kk}_edit_caught (CON)"] = round(float(np.mean(lab[kind == kk] == "CON")), 3)
            lvl[(s, k)]["hedge/population_edit_not_SUP"] = round(float(np.mean(lab[np.isin(kind, ["hedge", "population"])] != "SUP")), 3)
            lvl[(s, k)]["paraphrase_kept_SUP"] = round(float(np.mean(lab[kind == "paraphrase"] == "SUP")), 3)
LVL = pd.DataFrame(lvl).T; LVL.index.names = ["set", "verifier"]; display(LVL)
seen["verifier_level"] = {f"{s}|{k}": v for (s, k), v in lvl.items()}
_nli_v1 = nli; nli = VERS[SEL[0]]
seen["context_items_verify_rewrite_selected_verifier"] = ctx_eval([x["F"] for x in items], [x["F"] for x in items])
nli = _nli_v1
exp2.log_result("seen_sets", seen); print(json.dumps(seen["scifact_dev_e2e"], indent=1)); print("context items (rewrite verified):", seen["context_items_verify_rewrite_selected_verifier"])
""")

    md(r"""
### 18b. Fairness check of the context-item score (added 2026-11-22, after the first EXP-08-002 run)
- In the first run, the selected verifier scored 0.748 on the NB04 context items (rewrite verified), vs 0.896 for the deployed v1 in EXP-08-001.
- The comparison is confounded:
  - the context items are built from SciFact claims of **both** releases (train and dev);
  - v1 was trained on **all** SciFact train claims, whereas v1r and v2 saw only the train split.
- The items are therefore grouped by training exposure, with the same evidence (the top-1 for the rewrite) for every verifier.
- **Exploratory, post-hoc analysis.**
""")
    code(r"""
tr_all_text = {c["claim"] for c in _tr}
tr_split_text = {c["claim"] for c in _tr if SPLIT[c["id"]] == "train"}
def exposure(t):
    if t not in tr_all_text: return "unseen by all (SciFact dev claims)"
    return "seen by v1, v1r, v2 (train split)" if t in tr_split_text else "seen by v1 only (dev/test splits)"
EXPO = np.array([exposure(x["claim"]) for x in items])
def ctx_top1(queries):
    bi = bm.search(queries, DEPTH)[0]
    m = rcfg["encoders"]["medcpt"]; enc = Encoder("medcpt", m["repo"], m["revision"], doc_repo=m["doc_repo"], doc_revision=m["doc_revision"], device=DEV)
    D = torch.from_numpy(np.load(EMB / "medcpt_primary.npy")).to(DEV); di = Encoder.search(enc.encode_queries(queries), D, DEPTH)[0]; enc.unload(); del D; torch.cuda.empty_cache()
    out = []
    for q, a, b in zip(queries, bi, di):
        c = list(rrf([a, b], k=rcfg["fusion"]["rrf_k"], top=DEPTH)[:50]); s = np.asarray(CE.predict([(q, DOC_TEXT[j]) for j in c], batch_size=64, show_progress_bar=False), dtype=float).reshape(len(c), -1)[:, -1]
        out.append(DOC_TEXT[c[int(np.argmax(s))]])
    return out
F_TOP1 = ctx_top1([x["F"] for x in items]); FQ = [x["F"] for x in items]
fair = {}
for k in ("v1", "v1r", "v2"):
    lab = np.array(VERS[k].predict(F_TOP1, FQ)[0])
    fair[k] = {"all": round(macro_f1(G2, lab, ("SUP", "CON")), 4)}
    for gname in sorted(set(EXPO)):
        msk = EXPO == gname
        fair[k][f"{gname} (n={int(msk.sum())})"] = round(macro_f1(G2[msk], lab[msk], ("SUP", "CON")), 4)
FAIR = pd.DataFrame(fair).T; display(FAIR)
exp2.log_result("context_fairness_check", fair)
""")

    md("## 19. Real claims: re-verification with the selected configuration and the silver check")
    code(r"""
d = SCO["real"]; p, a = apply(d, *SEL); FIN = np.where(a, p, "ABSTAINED")
newv = pd.DataFrame({"claim_id": [r["claim_id"] for r in real], "final_v2": FIN})
SV2 = SV.merge(newv, on="claim_id", how="inner")
sil2 = {}
for src, g in SV2.groupby("source"):
    s = g[g.silver != "UNK"]
    def rate(col, sl, vals): x = s[s.silver == sl]; return round(float(x[col].isin(vals).mean()), 3) if len(x) else None
    sil2[src] = {"silver_SUP": int((s.silver == "SUP").sum()), "silver_CON": int((s.silver == "CON").sum()),
                 "v1: SUP|silver SUP": rate("final", "SUP", ["SUP"]), "selected: SUP|silver SUP": rate("final_v2", "SUP", ["SUP"]),
                 "v1: SUP|silver CON (silver false support)": rate("final", "CON", ["SUP"]), "selected: SUP|silver CON (silver false support)": rate("final_v2", "CON", ["SUP"]),
                 "v1: CON or ABST|silver CON": rate("final", "CON", ["CON", "ABSTAINED"]), "selected: CON or ABST|silver CON": rate("final_v2", "CON", ["CON", "ABSTAINED"])}
dist = {"v1 (EXP-07-001)": SV2.final.value_counts(normalize=True).round(3).to_dict(), "selected": pd.Series(FIN).value_counts(normalize=True).round(3).to_dict()}
exp2.log_result("real_claims", {"silver": sil2, "verdict_distribution": dist, "verified_as_rewrite": int(sum(t != r["claim"] for t, r in zip(r_text, real)))})
display(pd.DataFrame(sil2).T); print(dist)
OUT2 = PROC / "responses" / ("verdicts_v2_smoke.jsonl" if SMOKE else "verdicts_v2.jsonl")
ROUTED_SEL = set(routed(d, SEL[0]).tolist()) if SEL[3] != "none" else set()
with OUT2.open("w", encoding="utf-8") as fh:
    for i, r in enumerate(real):
        q = QWEN_LABELS.get((d["Q"][i], d["ev"][i])) if i in ROUTED_SEL else None
        fh.write(json.dumps({"claim_id": r["claim_id"], "claim": r["claim"], "verified_text": d["Q"][i], "decisive_unit": r_units[i].unit_id,
                             "verifier_label": str(d["lab"][SEL[0]][i]), "final_verdict": str(FIN[i]), "overall_confidence": round(float(d["overall"][SEL[0]][i]), 4),
                             "top1_rerank_score": round(r_ce[i], 3), "second_opinion": q, "config": cname(SEL), "corpus_snapshot": SNAP,
                             "evidence_date": r_units[i].pub_date}, ensure_ascii=False) + "\n")
print("wrote", OUT2.name)
""")

    md("## 20. Adoption decision (pre-registered rule)")
    code(r"""
t, ph = REPORT["test"], REPORT["pubhealth"]
c1 = t["selected_vs_baseline"]["delta_F1_e2e"] > 0
c2 = t["table"]["selected"]["FSR_answered"] <= t["table"]["baseline (v1r, v1 policy)"]["FSR_answered"] + 0.02
c3 = ph["table"]["selected"]["macro_F1_e2e"] >= ph["table"]["baseline (v1r, v1 policy)"]["macro_F1_e2e"] - 0.02
ADOPT = {"selected": cname(SEL), "e2e_F1_gain_on_fresh_scifact": c1, "fsr_within_0.02": c2, "pubhealth_not_worse_than_0.02": c3, "adopted_as_baseline_v2": bool(c1 and c2 and c3)}
exp2.log_result("adoption", ADOPT); print(json.dumps(ADOPT, indent=1))
fig, axs = plt.subplots(1, 2, figsize=(11, 4))
for ax, s in zip(axs, ("test", "pubhealth")):
    tb = pd.DataFrame(REPORT[s]["table"]).T
    ax.barh(tb.index, tb.macro_F1_e2e, color=["C3" if i == "selected" else "C0" for i in tb.index]); ax.set_xlabel("macro-F1 (abstained = error)"); ax.set_title(s); ax.grid(alpha=.3, axis="x")
fig.tight_layout(); exp2.save_figure("fresh_tests", fig); plt.show()
exp2.finish(); print("results:", exp2.dir)
""")
