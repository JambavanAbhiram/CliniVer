"""Builds notebooks/05_retrieval.ipynb. Discussion cells: nb05_discussion.py (written after execution).

Usage: python scripts/notebooks/build_nb05.py notebooks/05_retrieval.ipynb
"""
import sys
from pathlib import Path

import nbformat as nbf

out = Path(sys.argv[1])
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# Notebook 05 — Retrieval: sparse, dense, graph, fusion, re-ranking
**CliniVer** · Experiment `EXP-05-001`

| | |
|---|---|
| **Objective** | Choose the evidence retriever (encoder, BM25 variant, fusion, graph channel, re-ranker) with evidence, and retrieve evidence candidates for the real claims |
| **Research question** | **RQ4:** Which retrieval configuration among sparse, dense, graph and their fusions (± re-ranking) gives the best evidence retrieval under 8 GB VRAM? **Does the graph channel add value beyond entity normalization?** |
| **Hypotheses (pre-registered)** | **H4a** hybrid (sparse + dense, RRF) Recall@20 ≥ the better single channel · **H4b** a graph channel (PPR over the concept–document graph) improves Recall@20 over the hybrid on class-level and synonym-variant claims · **H4c** part of any graph gain is reproduced by normalization-only query expansion; graph *structure* adds value only if tribrid > hybrid + normalization · **H4d** cross-encoder re-ranking improves nDCG@10 at a measurable latency cost |
| **Decision rules** | Fixed in `configs/retrieval.yaml` before the main experiment; selections on the **tuning** split, hypotheses on the **evaluation** split. Graph retention rule: ADR-0003 |

> **Benchmark change before the main experiment (recorded):** a Part A check showed SciFact's own 5,183-abstract corpus is near ceiling (Recall@20 0.98–0.99). The primary benchmark is therefore SciFact claims over **SciFact + the 124,918 CliniVer PubMed abstracts as distractors** (de-duplicated by normalized title). SciFact-only is reported as secondary.
""")

md(r"""
## 1. Method
- **Queries:** the 693 SciFact train/dev claims with evidence (expert-written, standalone; NB04 showed rewriting is not needed for standalone claims). Split 30% tuning / 70% evaluation (seed 42, the same procedure as NB04).
- **Relevance:** abstracts with SUPPORT or CONTRADICT evidence. Absolute scores are **not comparable to BEIR**, which uses a different query set and relevance definition.
- **Channels:**
  - *Sparse:* BM25, `default` vs `clinical` tokenization.
  - *Dense:* 5 encoders (MiniLM, E5-base, BGE-small, BGE-base, MedCPT) with exact search.
  - *Graph:* a dictionary of 10^5 surface forms → concept IDs (MeSH / NCBI Gene) is built from the 3M PubTator mentions of the CliniVer PubMed corpus (NB01). **Every** document (SciFact and distractor) and every claim is linked with the same dictionary (word n-gram lookup), so no document set gets a better linker. Concept–concept edges come from PubTator relations. Documents are scored by Personalized PageRank from the claim's concepts.
    - *Why not PubTator for SciFact:* SciFact abstracts have no PMIDs. Exact-title PubMed search found 7 of the first 1,052 titles (PubMed reports long quoted phrases as "not found"), so it was abandoned.
  - *Normalization-only control:* append up to 3 alternative surface forms per linked concept to the query (no graph).
- **Fusion:** RRF (k = 60) and min-max weighted fusion (weight chosen on tuning).
- **Re-ranking:** MiniLM cross-encoder, BGE re-ranker base, MedCPT cross-encoder; pools of 20 / 50 / 100.
- **Strata (pre-registered rules):**
  - *class-level:* a pharmacological class noun;
  - *synonym-variant:* a linked mention that is not the concept's most frequent name;
  - *multi-entity:* ≥ 2 linked concepts;
  - *single-entity:* exactly one, none of the above;
  - *unlinked*.
- **Metrics:** Recall@{5,10,20,50}, P@5, MRR, nDCG@10; latency p50/p95; index size and build time; peak VRAM. Paired bootstrap over claims.
""")

md("## 2. Setup")
code(r"""
import json, os, random, re, time, collections, math
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np, pandas as pd, matplotlib.pyplot as plt, torch, yaml
from IPython.display import display

from cliniver.config import load_project_config
from cliniver.config.data_config import load_data_config
from cliniver.utils.seed import set_seed
from cliniver.utils.experiment import Experiment
from cliniver.ingestion.http import RawStore, RateLimitedClient, cached_get
from cliniver.ingestion.scifact import parse_release, corpus_titles
from cliniver.processing.long_docs import iter_documents
from cliniver.retrieval.encoders import Encoder
from cliniver.retrieval.sparse import BM25
from cliniver.retrieval.graph import ConceptGraph, EntityLinker
from cliniver.retrieval.fusion import rrf, weighted
from cliniver.evaluation.retrieval import doc_metrics
from cliniver.evaluation.span_retrieval import cluster_bootstrap_diff

pcfg = load_project_config(); dcfg = load_data_config()
ROOT = pcfg.paths.absolute("data_raw").parents[1]
cfg = yaml.safe_load((ROOT / "configs" / "retrieval.yaml").read_text(encoding="utf-8"))
SEED = set_seed(cfg["seed"]); SNAP = cfg["snapshot"]
SMOKE = os.environ.get("CLINIVER_SMOKE") == "1"   # code check: subsets, archived title lookups only, 2 encoders, 1 re-ranker; never results
SFX = "_smoke" if SMOKE else ""
if SMOKE:
    cfg["encoders"] = {k: cfg["encoders"][k] for k in ("minilm", "bge-small")}
    cfg["rerankers"] = {"minilm-ce": cfg["rerankers"]["minilm-ce"]}; cfg["rerank"]["pools"] = [20]
RAW = pcfg.paths.absolute("data_raw"); PROC = pcfg.paths.absolute("data_processed") / SNAP
EMB = pcfg.paths.absolute("data_interim") / SNAP / "embeddings"; EMB.mkdir(parents=True, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
pd.set_option("display.width", 170); pd.set_option("display.max_colwidth", 120)
exp = Experiment(exp_id="EXP-05-000-smoke" if SMOKE else "EXP-05-001", title="Retrieval: sparse, dense, graph, fusion, re-ranking", research_question="RQ4 / H4a-H4d",
                 config=cfg, seed=SEED, packages=["torch", "transformers", "sentence-transformers", "bm25s", "scipy", "pandas", "cliniver"])
print("seed", SEED, "| device", DEV)
""")

md("## 3. Data")
code(r"""
raw = cached_get(RateLimitedClient("scifact", min_interval_s=1.0, max_in_flight=1), RawStore(RAW, "scifact", SNAP), "data", "tar.gz", cfg["benchmark"]["scifact_url"])
sf_corpus, claims = parse_release(raw); sf_titles = corpus_titles(raw)
if SMOKE:
    claims = claims[:60]
ids_sorted = sorted(c.claim_id for c in claims); random.Random(SEED).shuffle(ids_sorted)
n_tune = round(len(ids_sorted) * cfg["benchmark"]["tuning_fraction"])
split_of = {cid: ("tuning" if i < n_tune else "evaluation") for i, cid in enumerate(ids_sorted)}
SPLIT = np.array([split_of[c.claim_id] for c in claims])
QUERIES = [c.text for c in claims]; REL = [set(c.relevant) for c in claims]

""")
code(r"""
norm_title = lambda t: re.sub(r"\W+", " ", t.lower()).strip()
sf_title_keys = {norm_title(t) for t in sf_titles.values()}
dist, dropped = {}, Counter()
for d in iter_documents(PROC / "documents.jsonl", {"pubmed"}):
    if norm_title(d["title"]) in sf_title_keys:
        dropped["title_matches_scifact"] += 1; continue
    dist[f"pm:{d['source_id']}"] = (d["title"], d["text"])
    if SMOKE and len(dist) >= 5000:
        break
DOC_IDS = list(sf_corpus) + list(dist)
TITLES = [sf_titles[d] for d in sf_corpus] + [t for t, _ in dist.values()]
ABSTRACTS = [sf_corpus[d][len(sf_titles[d]) + 2:] if sf_corpus[d].startswith(sf_titles[d]) else sf_corpus[d] for d in sf_corpus] + [a for _, a in dist.values()]
N_SF = len(sf_corpus)
data_summary = {"claims": len(claims), "tuning": int((SPLIT == "tuning").sum()), "evaluation": int((SPLIT == "evaluation").sum()),
                "scifact_docs": N_SF, "distractors": len(dist), "distractors_dropped_title_match": dropped.get("title_matches_scifact", 0), "primary_corpus_docs": len(DOC_IDS),
                "relevant_per_claim_mean": round(float(np.mean([len(r) for r in REL])), 2)}
exp.log_result("data", data_summary); print(json.dumps(data_summary, indent=1))
""")

md("### 3.1 Entities for the graph channel")
code(r"""
# One dictionary for everything: surface forms from all CliniVer PubTator mentions (Chemical / Disease / Gene)
mentions, relations = [], Counter()
with (PROC / "pubtator_entities.jsonl").open(encoding="utf-8") as fh:
    for line in fh:
        e = json.loads(line)
        if e["identifier"] and e["entity_type"] in ("Chemical", "Disease", "Gene"):
            mentions.append((e["text"], e["identifier"]))
with (PROC / "pubtator_relations.jsonl").open(encoding="utf-8") as fh:
    for line in fh:
        r = json.loads(line)
        relations[tuple(sorted((r["entity1"], r["entity2"])))] += 1
G = cfg["graph"]
linker = EntityLinker(mentions, min_len=G["linker_min_len"], min_count=G["linker_min_count"])
del mentions
t0 = time.perf_counter()
doc_concepts = {d: Counter(c for _, c in linker.link(f"{t}. {a}")) for d, t, a in zip(DOC_IDS, TITLES, ABSTRACTS)}
link_s = time.perf_counter() - t0
t0 = time.perf_counter()
graph = ConceptGraph({d: doc_concepts.get(d, Counter()) for d in DOC_IDS}, relations)
graph_build_s = time.perf_counter() - t0
gpos = {d: i for i, d in enumerate(graph.docs)}
claim_links = [linker.link(q) for q in QUERIES]
ent_summary = {"dictionary_link_s": round(link_s, 1), "relations": len(relations),
               "scifact_docs_with_concepts": round(sum(1 for d in sf_corpus if doc_concepts.get(d)) / N_SF, 3),
               "distractor_docs_with_concepts": round(sum(1 for d in dist if doc_concepts.get(d)) / max(1, len(dist)), 3),
               "concepts_per_doc_scifact": round(float(np.mean([len(doc_concepts[d]) for d in sf_corpus])), 1),
               "concepts_per_doc_distractor": round(float(np.mean([len(doc_concepts[d]) for d in dist])), 1) if dist else None,
               "concepts": graph.nc, "edges": graph.n_edges, "graph_build_s": round(graph_build_s, 1), "linker_surfaces": len(linker.surface),
               "claims_with_link": sum(1 for l in claim_links if l), "claims_multi_concept": sum(1 for l in claim_links if len({c for _, c in l}) >= 2)}
exp.log_result("entities", ent_summary); print(json.dumps(ent_summary, indent=1))
""")
code(r"""
CLASS_RX = re.compile(cfg["strata"]["class_level_regex"], re.I)
assert CLASS_RX.search("ACE inhibitors reduce mortality") and not CLASS_RX.search("Metformin reduces mortality"), "class-level regex broken"
def strata_of(q, links):
    concepts = {c for _, c in links}
    s = set()
    if CLASS_RX.search(q): s.add("class-level")
    if any(linker.surfaces_of.get(c) and surf.lower() != linker.surfaces_of[c][0] for surf, c in links): s.add("synonym-variant")
    if len(concepts) >= cfg["strata"]["multi_entity_min"]: s.add("multi-entity")
    if len(concepts) == 1 and not s: s.add("single-entity")
    if not concepts: s.add("unlinked")
    return s
STRATA = [strata_of(q, l) for q, l in zip(QUERIES, claim_links)]
strata_counts = Counter(s for ss in STRATA for s in ss)
exp.log_result("strata", dict(strata_counts)); print(strata_counts)
""")

md("## 4. Part A/B: dense encoders (both corpora)")
code(r"""
def metrics_table(ranked_ids, label):
    rows = []
    for i, (r, rel) in enumerate(zip(ranked_ids, REL)):
        rows.append({"config": label, "claim": i, "split": SPLIT[i], **doc_metrics([DOC_IDS[j] for j in r[:100]], rel, ks=(5, 10, 20, 50))})
    return pd.DataFrame(rows)

def emb_docs(enc, key):
    f = EMB / f"{key}_primary{SFX}.npy"
    if f.exists():
        return torch.from_numpy(np.load(f)).to(DEV)
    d = enc.encode_docs(ABSTRACTS, TITLES); np.save(f, d.cpu().numpy())
    exp.log_result(f"encode_stats_{key}", enc.stats)
    return d

DEPTH = cfg["first_stage_depth"]
runs, enc_rows = {}, []
for key, m in cfg["encoders"].items():
    enc = Encoder(key, m["repo"], m["revision"], m.get("query_prefix", ""), m.get("doc_prefix", ""), m.get("doc_repo"), m.get("doc_revision"), device=DEV, progress_log=str(pcfg.paths.absolute("logs") / "encode_progress.log"))
    D = emb_docs(enc, key); Qv = enc.encode_queries(QUERIES)
    t0 = time.perf_counter(); I, S = Encoder.search(Qv, D, DEPTH); q_s = (time.perf_counter() - t0) / len(QUERIES)
    runs[("dense", key)] = (I, S)
    sf_only = [[j for j in r if j < N_SF][:100] for r in Encoder.search(Qv, D[:N_SF], DEPTH)[0]]
    m_p, m_s = metrics_table(I, key), metrics_table(sf_only, key)
    enc_rows.append({"encoder": key, "license": m["license"], "dim": enc.dim, "params_M": round(enc.n_params / 1e6),
                     "index_MB_fp16": round(D.numel() * 2 / 2**20), "query_ms": round(q_s * 1000, 2),
                     "tune_ndcg10_primary": round(m_p[m_p.split == "tuning"]["ndcg@10"].mean(), 4),
                     "tune_recall20_primary": round(m_p[m_p.split == "tuning"]["recall@20"].mean(), 4),
                     "eval_ndcg10_primary": round(m_p[m_p.split == "evaluation"]["ndcg@10"].mean(), 4),
                     "eval_recall20_primary": round(m_p[m_p.split == "evaluation"]["recall@20"].mean(), 4),
                     "eval_ndcg10_scifact_only": round(m_s[m_s.split == "evaluation"]["ndcg@10"].mean(), 4),
                     "eval_recall20_scifact_only": round(m_s[m_s.split == "evaluation"]["recall@20"].mean(), 4)})
    enc.unload(); del enc, D, Qv; torch.cuda.empty_cache()
enc_df = pd.DataFrame(enc_rows).set_index("encoder")
encode_stats = {k: v for k, v in json.loads((exp.dir / "results.json").read_text(encoding="utf-8")).items() if k.startswith("encode_stats_")}
DENSE = enc_df.tune_ndcg10_primary.idxmax()
exp.save_table("encoders", enc_df); exp.log_result("encoders", enc_df.reset_index().to_dict(orient="records")); exp.log_result("dense_selected", DENSE)
display(enc_df); print("selected encoder (tuning nDCG@10, primary corpus):", DENSE)
""")

md("## 5. Part C: BM25 variants")
code(r"""
DOC_TEXT = [f"{t}. {a}" for t, a in zip(TITLES, ABSTRACTS)]
bm_rows = {}
for v in ("default", "clinical"):
    bm = BM25(DOC_TEXT, variant=v)
    t0 = time.perf_counter(); I, S = bm.search(QUERIES, DEPTH); q_s = (time.perf_counter() - t0) / len(QUERIES)
    runs[("bm25", v)] = (I, S)
    mt = metrics_table(I, v)
    bm_rows[v] = {"index_s": round(bm.index_s, 1), "query_ms": round(q_s * 1000, 2),
                  **{f"{s}_{k}": round(mt[mt.split == s][k].mean(), 4) for s in ("tuning", "evaluation") for k in ("ndcg@10", "recall@20")}}
    if v == "clinical":
        BM_CLIN = bm
    else:
        del bm
bm_df = pd.DataFrame(bm_rows).T
SPARSE = bm_df["tuning_ndcg@10"].idxmax()
exp.save_table("bm25_variants", bm_df); exp.log_result("bm25_variants", bm_df.to_dict(orient="index")); exp.log_result("sparse_selected", SPARSE)
display(bm_df); print("selected BM25 variant:", SPARSE)
""")

md("## 6. Part D: graph channel and the normalization-only control")
code(r"""
t0 = time.perf_counter()
graph_runs = []
for links in claim_links:
    s = graph.ppr([c for _, c in links], damping=G["damping"], iters=G["iters"])
    top = np.argsort(-s)[:DEPTH] if s.any() else np.array([], dtype=int)
    graph_runs.append((np.array([DOC_IDS.index(graph.docs[j]) if False else gpos_to_doc[j] for j in top]) if False else top, s[top] if len(top) else np.array([])))
ppr_ms = (time.perf_counter() - t0) / len(QUERIES) * 1000
""")
# NOTE: graph.docs is built in DOC_IDS order? It is sorted. Map explicitly in the next cell.
cells.pop()
code(r"""
doc_index = {d: i for i, d in enumerate(DOC_IDS)}
g2d = np.array([doc_index[d] for d in graph.docs])
t0 = time.perf_counter()
G_I, G_S = [], []
for links in claim_links:
    s = graph.ppr([c for _, c in links], damping=G["damping"], iters=G["iters"])
    if s.any():
        top = np.argsort(-s)[:DEPTH]; G_I.append(g2d[top]); G_S.append(s[top])
    else:
        G_I.append(np.array([], dtype=int)); G_S.append(np.array([]))
ppr_ms = (time.perf_counter() - t0) / len(QUERIES) * 1000
runs[("graph", "ppr")] = (G_I, G_S)
# normalization-only expansion: append alternative surface forms of linked concepts
def expand(q, links):
    extra = []
    for surf, c in links:
        extra += [s for s in linker.surfaces_of.get(c, [])[:G["expansion_surfaces"] + 1] if s != surf.lower()][:G["expansion_surfaces"]]
    return q + (" " + " ".join(dict.fromkeys(extra)) if extra else "")
EXPANDED = [expand(q, l) for q, l in zip(QUERIES, claim_links)]
m = cfg["encoders"][DENSE]
enc = Encoder(DENSE, m["repo"], m["revision"], m.get("query_prefix", ""), m.get("doc_prefix", ""), m.get("doc_repo"), m.get("doc_revision"), device=DEV, progress_log=str(pcfg.paths.absolute("logs") / "encode_progress.log"))
D = emb_docs(enc, DENSE)
runs[("dense+norm", DENSE)] = Encoder.search(enc.encode_queries(EXPANDED), D, DEPTH)
runs[("bm25+norm", SPARSE)] = (BM_CLIN if SPARSE == "clinical" else BM25(DOC_TEXT, variant=SPARSE)).search(EXPANDED, DEPTH)
gm = metrics_table(G_I, "graph")
graph_summary = {"ppr_ms_per_claim": round(ppr_ms, 1), "claims_with_graph_results": sum(1 for x in G_I if len(x)),
                 "graph_eval_recall@20": round(gm[gm.split == "evaluation"]["recall@20"].mean(), 4),
                 "graph_eval_ndcg@10": round(gm[gm.split == "evaluation"]["ndcg@10"].mean(), 4),
                 "queries_expanded": sum(1 for a, b in zip(QUERIES, EXPANDED) if a != b)}
exp.log_result("graph_channel", graph_summary); print(json.dumps(graph_summary, indent=1))
print("expansion example:", next((b for a, b in zip(QUERIES, EXPANDED) if a != b), None))
""")

md("## 7. Part E: fusion")
code(r"""
def fuse_rrf(*rs):
    return [rrf([r[0][i] for r in rs if len(r[0][i])], k=cfg["fusion"]["rrf_k"], top=DEPTH) for i in range(len(QUERIES))]
def fuse_w(a, b, w):
    return [weighted(a[0][i], a[1][i], b[0][i], b[1][i], w, top=DEPTH) if len(b[0][i]) else a[0][i][:DEPTH] for i in range(len(QUERIES))]
SP, DN, GR = runs[("bm25", SPARSE)], runs[("dense", DENSE)], runs[("graph", "ppr")]
SPN, DNN = runs[("bm25+norm", SPARSE)], runs[("dense+norm", DENSE)]
CONFIGS = {"bm25": SP[0], "dense": DN[0], "graph": GR[0],
           "hybrid": fuse_rrf(SP, DN), "hybrid+norm": fuse_rrf(SPN, DNN), "tribrid": fuse_rrf(SP, DN, GR),
           "tribrid+norm": fuse_rrf(SPN, DNN, GR), "bm25+graph": fuse_rrf(SP, GR), "dense+graph": fuse_rrf(DN, GR)}
# weighted hybrid: weight on dense chosen on tuning
wt = {}
for w in cfg["fusion"]["weights_grid"]:
    mt = metrics_table(fuse_w(DN, SP, w), "w")
    wt[w] = mt[mt.split == "tuning"]["ndcg@10"].mean()
W_BEST = max(wt, key=wt.get)
CONFIGS["hybrid-weighted"] = fuse_w(DN, SP, W_BEST)
exp.log_result("hybrid_weight", {"dense_weight": W_BEST, "tuning_ndcg@10": {str(k): round(v, 4) for k, v in wt.items()}})
ALL = pd.concat([metrics_table(r, name) for name, r in CONFIGS.items()], ignore_index=True)
ALL["strata"] = ALL.claim.map(lambda i: STRATA[i])
summ = ALL.groupby(["split", "config"])[["recall@5", "recall@10", "recall@20", "recall@50", "p@5", "mrr", "ndcg@10"]].mean().round(4)
exp.save_table("fusion_summary", summ); display(summ.loc["evaluation"].sort_values("ndcg@10", ascending=False))
""")
code(r"""
def strat_table(metric="recall@20", split="evaluation"):
    rows = []
    for s in ["class-level", "synonym-variant", "multi-entity", "single-entity", "unlinked"]:
        sub = ALL[(ALL.split == split) & ALL.strata.map(lambda x: s in x)]
        rows.append({"stratum": s, "n": sub.claim.nunique(), **sub.groupby("config")[metric].mean().round(4).to_dict()})
    return pd.DataFrame(rows).set_index("stratum")
st = strat_table(); exp.save_table("strata_recall20", st); display(st[["n", "hybrid", "hybrid+norm", "tribrid", "tribrid+norm", "graph"]])
""")

md("## 8. Part F: re-ranking")
code(r"""
from sentence_transformers import CrossEncoder
BASE = "hybrid"  # first stage for re-ranking: the hypothesis-defined hybrid (selection below may differ)
rr_rows, rr_runs = [], {}
for key, m in cfg["rerankers"].items():
    ce = CrossEncoder(m["repo"], revision=m["revision"], device=DEV, max_length=512)
    if DEV == "cuda":
        ce.model.half()
    for pool in cfg["rerank"]["pools"]:
        t0 = time.perf_counter(); out = []
        for i, q in enumerate(QUERIES):
            cand = list(CONFIGS[BASE][i][:pool])
            sc = ce.predict([(q, DOC_TEXT[j]) for j in cand], batch_size=64, show_progress_bar=False)
            sc = np.asarray(sc, dtype=float).reshape(len(cand), -1)[:, -1]
            out.append(np.array(cand)[np.argsort(-sc)])
        ms = (time.perf_counter() - t0) / len(QUERIES) * 1000
        name = f"{BASE}+{key}@{pool}"; rr_runs[name] = out
        mt = metrics_table(out, name)
        rr_rows.append({"config": name, "reranker": key, "pool": pool, "ms_per_claim": round(ms, 1),
                        **{f"{s}_{k}": round(mt[mt.split == s][k].mean(), 4) for s in ("tuning", "evaluation") for k in ("ndcg@10", "recall@20", "p@5")}})
    del ce; torch.cuda.empty_cache()
rr_df = pd.DataFrame(rr_rows).set_index("config")
RR_BEST = rr_df["tuning_ndcg@10"].idxmax()
CONFIGS[RR_BEST] = rr_runs[RR_BEST]
ALL = pd.concat([ALL, metrics_table(rr_runs[RR_BEST], RR_BEST).assign(strata=lambda x: x.claim.map(lambda i: STRATA[i]))], ignore_index=True)
exp.save_table("rerankers", rr_df); exp.log_result("rerankers", rr_df.reset_index().to_dict(orient="records")); exp.log_result("reranker_selected", RR_BEST)
display(rr_df); print("selected re-ranker (tuning nDCG@10):", RR_BEST)
""")

md("## 9. Evaluation: H4a–H4d, graph retention, selection")
code(r"""
NB = cfg["decision_rules"]["n_boot"]
def d(a, b, metric, split="evaluation", stratum=None):
    X = ALL[ALL.split == split]
    if stratum:
        X = X[X.strata.map(lambda s: stratum in s)]
    xa = X[X.config == a].set_index("claim")[metric]; xb = X[X.config == b].set_index("claim")[metric].loc[xa.index]
    r = cluster_bootstrap_diff(xa, xb, pd.Series(xa.index, index=xa.index), n_boot=NB, seed=SEED)
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}
ev = ALL[ALL.split == "evaluation"].groupby("config")["recall@20"].mean()
better_single = "bm25" if ev["bm25"] >= ev["dense"] else "dense"
H = {"H4a": {"vs": better_single, **d("hybrid", better_single, "recall@20")}}
H["H4a"]["supported"] = H["H4a"]["diff"] >= 0 and H["H4a"]["ci_low"] > -0.02
H["H4b"] = {s: d("tribrid", "hybrid", "recall@20", stratum=s) for s in ("class-level", "synonym-variant")}
H["H4b"]["overall"] = d("tribrid", "hybrid", "recall@20")
H["H4b"]["supported"] = any(H["H4b"][s]["ci_low"] > 0 for s in ("class-level", "synonym-variant"))
H["H4c"] = {"norm_vs_hybrid_overall": d("hybrid+norm", "hybrid", "recall@20"),
            "structure_overall": d("tribrid", "hybrid+norm", "recall@20"),
            **{f"structure_{s}": d("tribrid", "hybrid+norm", "recall@20", stratum=s) for s in ("class-level", "synonym-variant", "multi-entity")},
            **{f"norm_{s}": d("hybrid+norm", "hybrid", "recall@20", stratum=s) for s in ("class-level", "synonym-variant", "multi-entity")}}
H["H4c"]["structure_adds_value"] = any(H["H4c"][k]["ci_low"] > 0 for k in H["H4c"] if k.startswith("structure_"))
H["H4d"] = {"reranker": RR_BEST, **d(RR_BEST, "hybrid", "ndcg@10"), "ms_per_claim_scifact_primary": float(rr_df.loc[RR_BEST, "ms_per_claim"])}
H["H4d"]["supported"] = H["H4d"]["ci_low"] > 0
H["graph_retained"] = bool(H["H4b"]["supported"] and H["H4c"]["structure_adds_value"])
exp.log_result("hypotheses", H)
print(json.dumps({k: (v.get("supported") if isinstance(v, dict) and "supported" in v else v) for k, v in H.items() if k != "H4c"}, indent=1),
      "| H4c structure adds value:", H["H4c"]["structure_adds_value"])
""")
code(r"""
order = ["bm25", "dense", "hybrid", "hybrid+norm", "tribrid"] + ([] if H["graph_retained"] else [])
cand = [c for c in order if c in CONFIGS] + [RR_BEST]
tune = ALL[ALL.split == "tuning"].groupby("config")[["ndcg@10", "recall@20"]].mean()
best = tune.loc[cand, "ndcg@10"].idxmax()
ok = []
for c in cand:
    if c == best:
        ok.append(c); continue
    a = d(c, best, "ndcg@10", split="tuning"); b = d(c, best, "recall@20", split="tuning")
    if a["ci_high"] >= 0 and b["ci_high"] >= 0:
        ok.append(c)
if not H["graph_retained"]:
    ok = [c for c in ok if "graph" not in c and "tribrid" not in c]
SELECTED = min(ok, key=lambda c: cand.index(c))
sel = {"best_on_tuning": best, "within_ci_of_best": ok, "selected": SELECTED, "graph_retained": H["graph_retained"]}
exp.log_result("selection", sel); print(json.dumps(sel, indent=1))
""")
code(r"""
fig, axes = plt.subplots(1, 2, figsize=(14, 4.2))
e = ALL[ALL.split == "evaluation"].groupby("config")[["recall@20", "ndcg@10"]].mean().sort_values("ndcg@10")
e.plot.barh(ax=axes[0]); axes[0].set_title("Evaluation split, primary corpus"); axes[0].grid(axis="x", alpha=.3)
axes[1].scatter(rr_df.ms_per_claim, rr_df["evaluation_ndcg@10"], c=["C0" if "minilm" in i else "C1" if "bge" in i else "C2" for i in rr_df.index])
for i, r in rr_df.iterrows():
    axes[1].annotate(i.split("+")[1], (r.ms_per_claim, r["evaluation_ndcg@10"]), fontsize=7)
axes[1].axhline(e.loc["hybrid", "ndcg@10"], ls="--", c="grey"); axes[1].set_xlabel("ms per claim"); axes[1].set_ylabel("nDCG@10"); axes[1].set_title("Re-ranking: quality vs latency")
plt.tight_layout(); exp.save_figure("retrieval_overview", fig); plt.show()
""")

md("## 10. Error analysis: graph drift, synonym misses")
code(r"""
X = ALL[ALL.split == "evaluation"].pivot_table(index="claim", columns="config", values="recall@20")
drift = X[(X["hybrid"] > X["tribrid"])].index[:5]
helped = X[(X["tribrid"] > X["hybrid"])].index[:5]
def hub(i):
    cs = Counter(c for _, c in claim_links[i])
    return [(c, int(np.diff(graph.P[:, graph.nd + graph.c_index[c]].indptr).sum()) if c in graph.c_index else 0) for c in cs]
print("Graph hurt (tribrid < hybrid): claim and its linked concepts (concept, graph degree):")
display(pd.DataFrame([{"claim": QUERIES[i], "links": claim_links[i], "strata": sorted(STRATA[i])} for i in drift]))
print("Graph helped:"); display(pd.DataFrame([{"claim": QUERIES[i], "links": claim_links[i], "strata": sorted(STRATA[i])} for i in helped]))
""")

md(r"""
## 11. NB02 open question: whole abstracts vs `paragraph-128` chunks (SciFact-only corpus)
Each SciFact abstract is chunked with the NB02 configuration; a document's score is its best chunk. Queries and relevance are unchanged.
""")
code(r"""
from cliniver.processing.chunking import ChunkConfig, TokenIndex, make_chunks
from cliniver.processing.normalize import normalize
tok = enc.st.tokenizer if hasattr(enc, "st") else enc.d_tok
ck_text, ck_doc = [], []
for j in range(N_SF):
    t = normalize(DOC_TEXT[j]).text
    for ch in make_chunks(str(j), t, ChunkConfig("paragraph", 128), TokenIndex(t, tok)):
        ck_text.append(ch.text); ck_doc.append(j)
ck_doc = np.array(ck_doc)
def doc_rank(chunk_ids):
    seen, out = set(), []
    for c in chunk_ids:
        dj = ck_doc[c]
        if dj not in seen:
            seen.add(dj); out.append(dj)
    return out
whole = {"bm25": BM25(DOC_TEXT[:N_SF], variant=SPARSE).search(QUERIES, DEPTH)[0], "dense": Encoder.search(enc.encode_queries(QUERIES), D[:N_SF], DEPTH)[0]}
Dc = enc.encode_docs(ck_text)
chunked = {"bm25": [doc_rank(r) for r in BM25(ck_text, variant=SPARSE).search(QUERIES, 400)[0]],
           "dense": [doc_rank(r) for r in Encoder.search(enc.encode_queries(QUERIES), Dc, 400)[0]]}
nb02 = []
for ch in ("bm25", "dense"):
    for unit, rr in (("whole", whole[ch]), ("paragraph-128", chunked[ch])):
        mt = metrics_table(rr, f"{ch}-{unit}")
        nb02.append({"channel": ch, "unit": unit, **mt[mt.split == "evaluation"][["recall@20", "ndcg@10"]].mean().round(4).to_dict()})
nb02 = pd.DataFrame(nb02)
exp.log_result("nb02_units", {"chunks": len(ck_text), "table": nb02.to_dict(orient="records")}); display(nb02)
enc.unload(); del D, Dc; torch.cuda.empty_cache()
""")

md(r"""
## 12. Apply the selected retriever to the CliniVer corpus (real claims → evidence candidates for NB06)
Corpus: the NB02 `paragraph-128` chunks (592,930). Queries: the NB04 standalone rewrites of the 2,777 real claims (ADR-0011). Output: top-50 chunks per claim. Latency is measured per claim on this laptop.
""")
code(r"""
assert "tribrid" not in SELECTED and "graph" not in SELECTED, "a graph-based selection needs the chunk-level graph (not implemented here)"
CK_TEXT, CK_ID = [], []
with (PROC / "chunks" / "paragraph-128" / "chunks.jsonl").open(encoding="utf-8") as fh:
    for line in fh:
        c = json.loads(line); CK_TEXT.append(c["index_text"]); CK_ID.append(c["chunk_id"])
        if SMOKE and len(CK_ID) >= 20000:
            break
real = [json.loads(l) for l in (PROC / "responses" / "claims_contextualized.jsonl").open(encoding="utf-8")]
if SMOKE:
    real = real[:50]
RQ = [c["standalone_text"] for c in real]
t0 = time.perf_counter(); bm_real = BM25(CK_TEXT, variant=SPARSE); bm_index_s = time.perf_counter() - t0
m = cfg["encoders"][DENSE]
enc = Encoder(DENSE, m["repo"], m["revision"], m.get("query_prefix", ""), m.get("doc_prefix", ""), m.get("doc_repo"), m.get("doc_revision"), device=DEV, progress_log=str(pcfg.paths.absolute("logs") / "encode_progress.log"))
f = EMB / f"{DENSE}_cliniver_chunks{SFX}.npy"
if f.exists():
    DCK = torch.from_numpy(np.load(f)).to(DEV); dense_index_s = None
else:
    t0 = time.perf_counter(); DCK = enc.encode_docs(CK_TEXT); dense_index_s = time.perf_counter() - t0; np.save(f, DCK.cpu().numpy())
lat, results = [], []
ce = None
if "+" in SELECTED and "@" in SELECTED:
    rk = SELECTED.split("+")[-1].split("@")[0]; pool = int(SELECTED.split("@")[-1]); mm = cfg["rerankers"][rk]
    ce = CrossEncoder(mm["repo"], revision=mm["revision"], device=DEV, max_length=512)
    if DEV == "cuda": ce.model.half()
for i, q in enumerate(RQ):
    t0 = time.perf_counter()
    bi, bs = bm_real.search([q], DEPTH); di, ds = Encoder.search(enc.encode_queries([q]), DCK, DEPTH)
    fused = rrf([bi[0], di[0]], k=cfg["fusion"]["rrf_k"], top=DEPTH) if "hybrid" in SELECTED else (bi[0] if SELECTED.startswith("bm25") else di[0])
    if ce is not None:
        cand = list(fused[:pool]); sc = np.asarray(ce.predict([(q, CK_TEXT[j]) for j in cand], batch_size=64, show_progress_bar=False), dtype=float).reshape(len(cand), -1)[:, -1]
        fused = np.array(cand)[np.argsort(-sc)]
    lat.append(time.perf_counter() - t0)
    results.append({"claim_id": real[i]["claim_id"], "query": q, "candidates": [CK_ID[j] for j in fused[:50]]})
lat = np.array(lat)
real_summary = {"chunks": len(CK_TEXT), "claims": len(RQ), "bm25_index_s": round(bm_index_s, 1), "dense_index_s": round(dense_index_s, 1) if dense_index_s else "cached",
                "dense_index_MB_fp16": round(DCK.numel() * 2 / 2**20), "latency_p50_s": round(float(np.percentile(lat, 50)), 3),
                "latency_p95_s": round(float(np.percentile(lat, 95)), 3), "meets_budget": bool(np.percentile(lat, 95) <= cfg["latency_budget_s_p95"]),
                "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 2**20) if DEV == "cuda" else None}
with (PROC / "responses" / f"evidence_candidates{SFX}.jsonl").open("w", encoding="utf-8") as fh:
    for r in results:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
exp.log_result("cliniver_retrieval", real_summary); print(json.dumps(real_summary, indent=1))
top1_sources = Counter(cid.split(":")[0] for r in results for cid in r["candidates"][:5])
exp.log_result("cliniver_top5_sources", dict(top1_sources)); print("sources of top-5 candidates:", dict(top1_sources))
exp.finish(); print("results:", exp.dir)
""")

sys.path.insert(0, str(Path(__file__).parent))
try:
    from nb05_discussion import DISCUSSION_CELLS
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
