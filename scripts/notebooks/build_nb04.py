"""Builds notebooks/04_claim_contextualization.ipynb. Discussion cells: nb04_discussion.py (written after execution).

Usage: python scripts/notebooks/build_nb04.py notebooks/04_claim_contextualization.ipynb
"""
import sys
from pathlib import Path

import nbformat as nbf

out = Path(sys.argv[1])
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# Notebook 04 — Claim Contextualization
**CliniVer** · Experiment `EXP-04-001`

| | |
|---|---|
| **Objective** | Decide how a claim is turned into a retrieval query: alone, or together with the user's question and/or the model's answer, or rewritten into a standalone claim |
| **Research question** | **RQ3:** Does including the original query, and in what form, improve evidence retrieval? For which claims? |
| **Hypotheses (pre-registered)** | **H3a:** Query + claim retrieval improves Recall@10 over claim-only for **context-dependent** claims. **H3b:** For **self-contained** claims, adding the query does not improve, and may degrade, precision |
| **Decision rules (fixed before running, `configs/context.yaml`)** | **H3a** supported if Recall@10(B) − Recall@10(A) > 0 (95% CI excludes 0) on context-dependent claims, for **both** retrievers. **H3b** supported if nDCG@10(B) − nDCG@10(A) is **not** significantly positive on self-contained claims, for both retrievers. The pipeline strategy is the one with the highest mean nDCG@10 (both strata, both retrievers) on the **tuning** split, reported on the **evaluation** split |

> **Synthetic data, stated up front.** No public dataset combines questions, context-dependent claims and evidence relevance labels. The context-dependent test items are therefore **constructed** from real SciFact claims with the local LLM and validated by rules. Their relevant documents are the expert evidence labels of the original claims. Results on them are labelled synthetic (research_foundation §10.2).
""")

md(r"""
## 1. Background and method
### 1.1 Strategies
| ID | Query | Notes |
|---|---|---|
| A | claim | baseline |
| B | question + claim | concatenated |
| C | question + claim + answer | "M" in the design is interpreted as the model's **answer** (response) |
| D | question and claim retrieved separately, merged by reciprocal-rank fusion | k = 60 |
| E | weighted score fusion of the two sub-queries | claim weight tuned on the tuning split |
| F | LLM rewrite of the claim into a standalone claim (molecular-facts style), then retrieval with the rewrite | Qwen3-8B, schema-constrained |
| oracle | the original SciFact claim | upper bound for the context-dependent stratum |

### 1.2 Test items
- **Self-contained stratum:** the 693 SciFact train/dev claims that have evidence (expert-written, standalone by design), each paired with a generated question and answer.
- **Context-dependent stratum:** for each claim, Qwen3-8B writes a question that names the claim's main subject without revealing the finding, a 2–3 sentence answer, and a one-sentence version of the claim that refers back to the question ("It reduces…") instead of naming the subject.
- **Validation rules** (`cliniver.claims.contextualize.validate_context`):
  - the removed entity occurs in the claim and in the question, and not in the dependent claim;
  - one sentence, of similar length;
  - ≥ 70% of its content words come from the original claim;
  - the question does not restate the claim.

  Invalid items are excluded and counted.
- **Corpus:** the 5,183 SciFact abstracts (title + abstract, indexed whole). Relevant = documents with SUPPORT or CONTRADICT evidence.

### 1.3 Real claims (NB03 output)
- The 2,777 claims extracted from real generated answers are tagged as context-dependent or not by a rule tagger (unresolved referent, or a key domain entity that appears only in the question), with a manual spot check.
- Strategy F is applied to the context-dependent ones (and to 100 self-contained controls, which should come back unchanged).
- There are no relevance labels for real claims, so this part measures **prevalence** and **rewrite behaviour**, not retrieval quality.

### 1.4 Metrics and statistics
- Recall@10, P@5, MRR, nDCG@10; **query drift** = 1 − cos(query, original claim) with the dense encoder.
- Paired bootstrap over claims (2,000 resamples).
""")

md("## 2. Setup")
code(r"""
import json, os, random, re, time, collections
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt, torch, yaml
from IPython.display import display

from cliniver.config import load_project_config
from cliniver.config.llm_config import load_llm_config
from cliniver.config.processing_config import load_processing_config
from cliniver.utils.seed import set_seed
from cliniver.utils.experiment import Experiment
from cliniver.ingestion.http import RawStore, RateLimitedClient, cached_get
from cliniver.ingestion.scifact import parse_release
from cliniver.ingestion.qa_datasets import domain_regex
from cliniver.llm.runtime import ManagedClient
from cliniver.claims.contextualize import make_context, decontextualize, context_dependent, _REFERENT
from cliniver.retrieval.simple import BM25Index, DenseIndex
from cliniver.retrieval.fusion import rrf, weighted
from cliniver.evaluation.retrieval import doc_metrics
from cliniver.evaluation.span_retrieval import cluster_bootstrap_diff

pcfg = load_project_config(); lcfg = load_llm_config(); prc = load_processing_config()
cfg = yaml.safe_load((pcfg.paths.absolute("data_raw").parents[1] / "configs" / "context.yaml").read_text(encoding="utf-8"))
SEED = set_seed(cfg["seed"]); SNAP = cfg["snapshot"]
SMOKE = os.environ.get("CLINIVER_SMOKE") == "1"   # code check on small subsets; shares the LLM call cache; never results
RAW = pcfg.paths.absolute("data_raw"); PROC = pcfg.paths.absolute("data_processed") / SNAP; ANN = RAW.parent / "annotations"
LLM_LOG = pcfg.paths.absolute("data_interim") / SNAP / "llm_calls"
pd.set_option("display.width", 170); pd.set_option("display.max_colwidth", 140)
exp = Experiment(exp_id="EXP-04-000-smoke" if SMOKE else "EXP-04-001", title="Claim contextualization", research_question="RQ3 / H3a, H3b",
                 config={"context": cfg, "llm_model": lcfg.models[cfg["llm_model"]].model_dump(), "dense": prc.retriever.model_dump()},
                 seed=SEED, packages=["torch", "sentence-transformers", "bm25s", "pandas", "cliniver"])
def managed():
    return ManagedClient(lcfg, cfg["llm_model"], LLM_LOG / f"{cfg['llm_model']}.jsonl", server_log=pcfg.paths.absolute("logs") / "llama-server_nb04.log")
print("seed", SEED, "| LLM", cfg["llm_model"], "| dense", prc.retriever.dense_model)
""")

md("## 3. Data: SciFact claims, corpus, and synthetic context")
code(r"""
raw = cached_get(RateLimitedClient("scifact", min_interval_s=1.0, max_in_flight=1), RawStore(RAW, "scifact", SNAP), "data", "tar.gz", cfg["scifact"]["url"])
corpus, claims = parse_release(raw)
if SMOKE:
    claims = claims[:20]
doc_ids = list(corpus); doc_text = [corpus[d] for d in doc_ids]
ids_sorted = sorted(c.claim_id for c in claims)
rng = random.Random(SEED); rng.shuffle(ids_sorted)
n_tune = round(len(ids_sorted) * cfg["scifact"]["tuning_fraction"])
split_of = {cid: ("tuning" if i < n_tune else "evaluation") for i, cid in enumerate(ids_sorted)}
print(f"corpus {len(corpus)} abstracts | claims with evidence {len(claims)} | relevant docs per claim mean {np.mean([len(c.relevant) for c in claims]):.2f}")
""")
code(r"""
items, invalid = [], collections.Counter()
t0 = time.perf_counter()
with managed() as mc:
    for c in claims:
        it, rec = make_context(mc, c.text, seed=SEED, tag=f"ctx:{c.claim_id}")
        if it is None:
            invalid["invalid JSON"] += 1; continue
        if not it.valid:
            invalid[it.reason] += 1; continue
        items.append({"claim_id": c.claim_id, "split": split_of[c.claim_id], "claim": c.text, "question": it.question, "answer": it.answer,
                      "dependent": it.dependent_claim, "removed_entity": it.removed_entity, "relevant": sorted(c.relevant)})
    gen_stats = mc.stats
print(f"generation {time.perf_counter()-t0:.0f}s; server: {gen_stats}")
synth = {"claims": len(claims), "valid_items": len(items), "invalid": dict(invalid),
         "valid_rate": round(len(items) / len(claims), 3), "by_split": dict(collections.Counter(i["split"] for i in items))}
exp.log_result("synthetic_items", synth); print(json.dumps(synth, indent=1))
display(pd.DataFrame(items)[["claim", "question", "dependent"]].sample(6, random_state=SEED))
""")

md(r"""
### 3.1 Strategy F rewrites (both strata)
The same rewrite prompt is applied to the dependent claim (it should restore the subject) and to the original claim (it should return it essentially unchanged).
""")
code(r"""
with managed() as mc:
    for it in items:
        for key in ("dependent", "claim"):
            out, _ = decontextualize(mc, it[key], it["question"], it["answer"], seed=SEED, tag=f"dectx:{key}:{it['claim_id']}")
            it[f"F_{key}"] = out or it[key]
            it[f"F_{key}_ok"] = out is not None
ent_restored = np.mean([it["removed_entity"].lower() in it["F_dependent"].lower() for it in items])
unchanged = np.mean([re.sub(r"\W+", " ", it["F_claim"].lower()).strip() == re.sub(r"\W+", " ", it["claim"].lower()).strip() for it in items])
fstats = {"valid_json": float(np.mean([it["F_dependent_ok"] and it["F_claim_ok"] for it in items])),
          "dependent_entity_restored": round(float(ent_restored), 3), "self_contained_returned_unchanged": round(float(unchanged), 3)}
exp.log_result("rewrite_F", fstats); print(fstats)
""")

md("## 4. Retrieval experiments (A–F, oracle) × 2 strata × 2 retrievers")
code(r"""
from sentence_transformers import SentenceTransformer
enc = SentenceTransformer(prc.retriever.dense_model, revision=prc.retriever.dense_revision, device="cuda" if torch.cuda.is_available() else "cpu")
bm = BM25Index(doc_text)
de = DenseIndex(doc_text, enc, batch_size=prc.retriever.batch_size, query_prefix=prc.retriever.query_prefix)
DEPTH = cfg["candidate_depth"]
def search(index, queries):
    r = index.search(queries, DEPTH)
    return r.ids, r.scores

STRATA = {"self-contained": "claim", "context-dependent": "dependent"}
runs = {}
for rname, idx in (("bm25", bm), ("dense", de)):
    q_ids, q_sc = search(idx, [it["question"] for it in items])
    for stratum, key in STRATA.items():
        C = [it[key] for it in items]
        c_ids, c_sc = search(idx, C)
        runs[(rname, stratum, "A")] = c_ids
        runs[(rname, stratum, "B")] = search(idx, [f"{it['question']} {c}" for it, c in zip(items, C)])[0]
        runs[(rname, stratum, "C")] = search(idx, [f"{it['question']} {c} {it['answer']}" for it, c in zip(items, C)])[0]
        runs[(rname, stratum, "D")] = [rrf([c_ids[i], q_ids[i]], k=cfg["rrf_k"], top=100) for i in range(len(items))]
        for w in cfg["fusion_weights"]:
            runs[(rname, stratum, f"E{w}")] = [weighted(c_ids[i], c_sc[i], q_ids[i], q_sc[i], w) for i in range(len(items))]
        runs[(rname, stratum, "F")] = search(idx, [it[f"F_{key}"] for it in items])[0]
        if stratum == "context-dependent":
            runs[(rname, stratum, "oracle")] = search(idx, [it["claim"] for it in items])[0]
rows = []
for (rname, stratum, strat), ranked in runs.items():
    for it, r in zip(items, ranked):
        m = doc_metrics([doc_ids[j] for j in r[:100]], set(it["relevant"]))
        rows.append({"retriever": rname, "stratum": stratum, "strategy": strat, "claim_id": it["claim_id"], "split": it["split"], **m})
R = pd.DataFrame(rows)
exp.save_table("per_claim_metrics", R)
# strategy E: choose the claim weight on the tuning split (mean nDCG@10 over strata and retrievers)
e_tune = R[(R.split == "tuning") & R.strategy.str.startswith("E")].groupby("strategy")["ndcg@10"].mean()
E_BEST = e_tune.idxmax()
R = R[~R.strategy.str.startswith("E") | (R.strategy == E_BEST)].replace({"strategy": {E_BEST: "E"}})
exp.log_result("fusion_weight", {"chosen": E_BEST, "tuning_ndcg": e_tune.round(4).to_dict()})
print("strategy E weight:", E_BEST)
""")
code(r"""
METRICS = ["recall@10", "p@5", "mrr", "ndcg@10"]
summary = R.groupby(["split", "stratum", "retriever", "strategy"])[METRICS].mean().round(3)
evals = summary.loc["evaluation"]
exp.save_table("summary", summary)
display(evals.unstack("retriever"))
""")

md("### 4.1 Query drift")
code(r"""
def emb(xs): return enc.encode(xs, batch_size=128, normalize_embeddings=True, show_progress_bar=False)
orig = emb([it["claim"] for it in items])
drift = {}
for stratum, key in STRATA.items():
    C = [it[key] for it in items]
    qs = {"A": C, "B": [f"{it['question']} {c}" for it, c in zip(items, C)], "C": [f"{it['question']} {c} {it['answer']}" for it, c in zip(items, C)],
          "F": [it[f"F_{key}"] for it in items]}
    for s, q in qs.items():
        drift[(stratum, s)] = round(float(np.mean(1 - np.sum(emb(q) * orig, axis=1))), 4)
drift_df = pd.Series(drift).unstack(0)
exp.log_result("query_drift", drift_df.to_dict()); display(drift_df)
""")

md("## 5. Evaluation: H3a, H3b, selection")
code(r"""
def diff(stratum, retriever, a, b, metric, split="evaluation"):
    X = R[(R.split == split) & (R.stratum == stratum) & (R.retriever == retriever)]
    xa = X[X.strategy == a].set_index("claim_id")[metric]; xb = X[X.strategy == b].set_index("claim_id")[metric].loc[xa.index]
    return cluster_bootstrap_diff(xa, xb, pd.Series(xa.index, index=xa.index), n_boot=cfg["decision_rules"]["n_boot"], seed=SEED)
h3 = {"H3a": {}, "H3b": {}}
for r in cfg["retrievers"]:
    h3["H3a"][r] = {s: diff("context-dependent", r, s, "A", "recall@10") for s in ("B", "C", "D", "E", "F", "oracle")}
    h3["H3b"][r] = {m: {s: diff("self-contained", r, s, "A", m) for s in ("B", "C", "D", "E", "F")} for m in ("ndcg@10", "p@5")}
h3a_ok = all(h3["H3a"][r]["B"]["ci_low"] > 0 for r in cfg["retrievers"])
h3b_ok = all(h3["H3b"][r]["ndcg@10"]["B"]["ci_low"] <= 0 for r in cfg["retrievers"])
h3["verdict"] = {"H3a": "SUPPORTED" if h3a_ok else "NOT SUPPORTED", "H3b": "SUPPORTED" if h3b_ok else "NOT SUPPORTED"}
tune = R[R.split == "tuning"].groupby("strategy")["ndcg@10"].mean().drop("oracle", errors="ignore")
h3["selection"] = {"tuning_mean_ndcg@10": tune.round(4).to_dict(), "selected": tune.idxmax()}
exp.log_result("hypotheses", h3)
print(json.dumps(h3["verdict"]), "| selected:", h3["selection"]["selected"])
tab = pd.DataFrame([{"retriever": r, "comparison": f"{s} − A", **{k: round(v, 4) for k, v in h3["H3a"][r][s].items() if k != "n_clusters"}}
                    for r in cfg["retrievers"] for s in h3["H3a"][r]])
print("H3a (context-dependent, recall@10, evaluation split)"); display(tab)
tab_b = pd.DataFrame([{"retriever": r, "metric": m, "comparison": f"{s} − A", **{k: round(v, 4) for k, v in h3["H3b"][r][m][s].items() if k != "n_clusters"}}
                      for r in cfg["retrievers"] for m in ("ndcg@10", "p@5") for s in h3["H3b"][r][m]])
print("H3b (self-contained, evaluation split)"); display(tab_b)
exp.save_table("h3a", tab); exp.save_table("h3b", tab_b)
""")
code(r"""
fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
order = ["A", "B", "C", "D", "E", "F", "oracle"]
for ax, stratum in zip(axes, STRATA):
    d = evals.xs(stratum, level="stratum")["ndcg@10"].unstack("retriever").reindex([o for o in order if (stratum == "context-dependent" or o != "oracle")])
    d.plot.bar(ax=ax, rot=0); ax.set_title(f"{stratum} (evaluation split)"); ax.set_ylabel("nDCG@10"); ax.grid(axis="y", alpha=.3)
plt.tight_layout(); exp.save_figure("strategies", fig); plt.show()
""")

md("## 6. Error analysis")
code(r"""
X = R[(R.split == "evaluation") & (R.stratum == "context-dependent") & (R.retriever == "dense")].pivot(index="claim_id", columns="strategy", values="recall@10")
by_id = {it["claim_id"]: it for it in items}
lost = X[(X["oracle"] > 0) & (X["A"] == 0)].index[:4]
hurt = X[(X["A"] > 0) & (X["B"] < X["A"])].index[:4]
print("Dense, context-dependent: oracle finds evidence, claim-only (A) does not:")
display(pd.DataFrame([{"dependent": by_id[i]["dependent"], "question": by_id[i]["question"], "F": by_id[i]["F_dependent"]} for i in lost]))
print("Adding the question (B) loses evidence that A found:")
display(pd.DataFrame([{"dependent": by_id[i]["dependent"], "question": by_id[i]["question"]} for i in hurt]))
""")

md(r"""
## 7. Real claims (NB03): how many are context-dependent, and what F does to them
Tagger: unresolved referent (pronoun or "the drug/condition/patients…" with no key term), or a key domain term (in-scope drug name or condition word) that appears in the question but not in the claim. **Spot check:** 40 tags inspected by the AI developer (not a human annotation); reported if the annotation file exists, otherwise NOT EVALUATED.
""")
code(r"""
resp = {json.loads(l)["response_id"]: json.loads(l) for l in (PROC / "responses" / "responses.jsonl").open(encoding="utf-8")}
real = [json.loads(l) for l in (PROC / "responses" / "claims.jsonl").open(encoding="utf-8")]
if SMOKE:
    real = real[:200]
concepts = [json.loads(l) for l in (PROC / "concepts.jsonl").open(encoding="utf-8")]
KEYS = domain_regex({n.lower() for c in concepts if c["source"] == "rxnorm" for n in [c["name"], *c["synonyms"]] if len(n) >= 4})
for c in real:
    c["context_dependent"], c["cd_reason"] = context_dependent(c["text"], resp[c["response_id"]]["question"], KEYS)
rc = pd.DataFrame(real)
prev = {"claims": len(rc), "context_dependent": int(rc.context_dependent.sum()), "share": round(float(rc.context_dependent.mean()), 3),
        "by_reason": rc[rc.context_dependent].cd_reason.value_counts().to_dict(),
        "share_by_generator": rc.groupby("generator").context_dependent.mean().round(3).to_dict()}
# tagger check by construction on SciFact (referent rule only; key terms there are not domain terms)
prev["referent_rule_on_synthetic_dependent"] = round(float(np.mean([bool(_REFERENT.search(it["dependent"])) for it in items])), 3)
prev["referent_rule_on_original_scifact"] = round(float(np.mean([bool(_REFERENT.search(it["claim"])) for it in items])), 3)
exp.log_result("real_claims_context_dependence", prev); print(json.dumps(prev, indent=1))
""")
code(r"""
rng = random.Random(SEED)
cd_idx = [i for i, c in enumerate(real) if c["context_dependent"]]; sc_idx = [i for i, c in enumerate(real) if not c["context_dependent"]]
k = cfg["real_claims"]["tagger_spot_check"] // 2
spot = [real[i] | {"question": resp[real[i]["response_id"]]["question"]} for i in sorted(rng.sample(cd_idx, min(k, len(cd_idx))) + rng.sample(sc_idx, min(k, len(sc_idx))))]
spot_file = ANN / ("nb04_tagger_spot_check_smoke.json" if SMOKE else "nb04_tagger_spot_check.json")
if spot_file.exists():
    ann = {a["claim_id"]: a for a in json.loads(spot_file.read_text(encoding="utf-8"))["items"]}
    ok = [ann[s["claim_id"]]["context_dependent"] == s["context_dependent"] for s in spot if s["claim_id"] in ann]
    tp = sum(1 for s in spot if s["claim_id"] in ann and s["context_dependent"] and ann[s["claim_id"]]["context_dependent"])
    spot_res = {"n": len(ok), "agreement": round(float(np.mean(ok)), 3), "precision_of_dependent_tag": f"{tp}/{sum(1 for s in spot if s['context_dependent'])}",
                "missed_dependent_among_self_contained_tags": sum(1 for s in spot if s["claim_id"] in ann and not s["context_dependent"] and ann[s["claim_id"]]["context_dependent"])}
else:
    (ANN / ("nb04_tagger_spot_check_TODO_smoke.json" if SMOKE else "nb04_tagger_spot_check_TODO.json")).write_text(json.dumps([{"claim_id": s["claim_id"], "question": s["question"], "claim": s["text"],
        "tag": s["context_dependent"], "reason": s["cd_reason"]} for s in spot], ensure_ascii=False, indent=1), encoding="utf-8")
    spot_res = "NOT EVALUATED (sample written for annotation)"
exp.log_result("tagger_spot_check", spot_res); print(spot_res)
""")
code(r"""
controls = rng.sample(sc_idx, min(cfg["real_claims"]["self_contained_controls"], len(sc_idx)))
targets = sorted(set(cd_idx) | set(controls))
with managed() as mc:
    for i in targets:
        c = real[i]; rsp = resp[c["response_id"]]
        out, _ = decontextualize(mc, c["text"], rsp["question"], rsp["answer"], seed=SEED, tag=f"dectx:real:{c['claim_id']}")
        c["standalone_text"] = out
norm = lambda t: re.sub(r"\W+", " ", (t or "").lower()).strip()
dep = [real[i] for i in cd_idx]; ctl = [real[i] for i in controls]
real_f = {"rewritten_dependent": len(dep), "dependent_changed": round(float(np.mean([norm(c["standalone_text"]) != norm(c["text"]) for c in dep])), 3),
          "dependent_now_tagged_self_contained": round(float(np.mean([not context_dependent(c["standalone_text"] or c["text"], resp[c["response_id"]]["question"], KEYS)[0] for c in dep])), 3),
          "controls": len(ctl), "controls_unchanged": round(float(np.mean([norm(c["standalone_text"]) == norm(c["text"]) for c in ctl])), 3),
          "length_ratio_dependent": round(float(np.mean([len((c["standalone_text"] or "").split()) / max(1, len(c["text"].split())) for c in dep])), 2)}
exp.log_result("real_claims_rewrite_F", real_f); print(json.dumps(real_f, indent=1))
display(pd.DataFrame([{"question": resp[c["response_id"]]["question"][:90], "claim": c["text"], "F": c["standalone_text"]} for c in dep[:8]]))
with (PROC / "responses" / ("claims_contextualized_smoke.jsonl" if SMOKE else "claims_contextualized.jsonl")).open("w", encoding="utf-8") as fh:
    for c in real:
        fh.write(json.dumps({**c, "standalone_text": c.get("standalone_text") or c["text"],
                             "standalone_source": "F" if c.get("standalone_text") else "original"}, ensure_ascii=False) + "\n")
exp.finish(); print("results:", exp.dir)
""")

sys.path.insert(0, str(Path(__file__).parent))
try:
    from nb04_discussion import DISCUSSION_CELLS
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
