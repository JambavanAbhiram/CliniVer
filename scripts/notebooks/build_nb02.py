"""Builds notebooks/02_document_processing_and_chunking.ipynb (headless construction; the executed notebook
is the research artifact). Discussion cells are written after execution, from the outputs, in nb02_discussion.py.

Usage: python scripts/notebooks/build_nb02.py notebooks/02_document_processing_and_chunking.ipynb
"""
import sys
from pathlib import Path

import nbformat as nbf

out = Path(sys.argv[1])
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# Notebook 02 — Document Processing and Chunking
**CliniVer** · Experiment `EXP-02-001` · Corpus snapshot `2026-11-20` (frozen, NB01)

| | |
|---|---|
| **Objective** | Choose how documents are normalized and cut into retrieval units ("chunks"), with evidence, and produce the chunked corpus used by NB04–NB08 |
| **Research question** | **RQ1:** How do preprocessing and chunking affect retrieval and qualifier preservation on long documents? |
| **Hypothesis (pre-registered, research_foundation §6)** | **H1:** Structure-aware chunking yields higher Recall@k and fewer split qualifiers than fixed-size chunking on long documents |
| **Decision rule (fixed before running)** | Configurations are selected on a **dev** split; H1 is tested on a held-out **test** split. H1 is *supported* if the best structure-aware configuration beats the best fixed-size configuration on the primary metric for **both** reference retrievers (95% cluster-bootstrap CI of the difference excludes 0) **and** truncates fewer qualifiers on our long documents (CI excludes 0; metric defined in §1.2, item 2). *Partially supported* if only one part holds; *not supported* otherwise |
| **Primary metric** | `hit@1024t`: the gold answer span lies entirely inside one of the top-ranked chunks that fit in a 1,024-token budget |

> **Why a token budget?** At a fixed number of chunks *k*, larger chunks win trivially (they contain more text). Downstream, the verifier (NB06) reads a fixed amount of context, so the fair comparison is: *how often is the evidence inside what the verifier can read?*
""")

md(r"""
## 1. Background and method

### 1.1 Chunking strategies compared
| Strategy | Unit boundaries | Sizes (max tokens) |
|---|---|---|
| `fixed` (baseline family) | none: windows of N tokens, overlap 0 / 15 / 25% | 128, 256, 512 |
| `sentence` | sentences packed up to N | 128, 256, 512 |
| `paragraph` | one paragraph per chunk (oversize paragraphs split by sentences) | 128, 256, 512 |
| `recursive` | blank-line blocks → paragraphs → sentences → tokens, neighbours packed up to N (may cross sections) | 128, 256, 512 |
| `section` | like `recursive` but **never crosses a section boundary** | 128, 256, 512 |
| `section+hdr` | `section`, and the indexed text is prefixed with "title / section" | 128, 256, 512 |
| `semantic` | boundary where neighbouring-sentence embedding distance > 90th percentile of the document | 128, 256, 512 |

"Structure-aware" = `sentence`, `paragraph`, `recursive`, `section`, `section+hdr`, `semantic` (they respect linguistic or document boundaries). Tokens are counted with the tokenizer of the fixed dense retriever.

### 1.2 Evaluation design
1. **Span retrieval on COVID-QA** (Möller et al., 2020; Apache-2.0): 2,019 questions written by biomedical experts over 147 full-text scientific articles, each with an exact gold answer span. It is the closest openly licensed benchmark with **span-level gold on long biomedical full text**. It is *not* our clinical domain (COVID-19 research vs cardiometabolic pharmacotherapy). Here it measures a **document-structure** property (does chunking keep evidence together and retrievable?), not clinical knowledge; the domain gap is a stated limitation.
   - All chunks of all 147 articles form one search pool (realistic distractors).
   - Articles are split **dev 30% / test 70%** (seeded); every question follows its article.
   - Two **fixed reference retrievers**: BM25 (bm25s) and a small dense bi-encoder (`BAAI/bge-small-en-v1.5`, pinned revision). Choosing the final retriever is NB05's job; using two very different retrievers checks that a chunking conclusion is not an artefact of one retriever.
   - Metrics per question: `hit@k` (strict containment in the top-k), `cover@k` (fraction of the gold span covered by the union of the top-k), MRR, nDCG@10 (gain = coverage), and the same at token budgets of 512 / 1,024 / 2,048. Also `gold_in_one_chunk`: whether the chunking keeps the gold span inside *any* single chunk (an upper bound for strict hits).
2. **Qualifier preservation on our own long documents** (398 Europe PMC full texts + 338 openFDA labels with text): rule-detected qualifier spans (negation, quantity, threshold, effect estimate, population; `cliniver.processing.qualifiers`). Two measures:
   - **truncation rate (primary for H1):** share of qualifiers that appear *cut* in at least one chunk (a chunk overlaps the span without containing it). This is what a reader of a retrieved chunk can be misled by.
   - **split rate (secondary):** share of qualifiers that no single chunk contains.

   Without overlapping chunks the two are equal. With overlapping windows a qualifier can be intact in one window and cut in the neighbour, so the split rate drops to ~0 while cut copies remain retrievable.
   > **Protocol change, recorded before the full run:** the split rate was the planned metric. A code-check run on 40 documents showed that 25% overlap makes it trivially 0 (`fixed-128-o25`: 0.0000) although overlapping windows still contain cut copies. The truncation rate was therefore made the primary qualifier metric **before** the full experiment; both are reported.

   Rule precision is estimated on manually inspected samples (§6.3).
3. **Cost**: number of chunks, indexed tokens (overlap inflates the index), chunking / indexing / query time.

Statistics: differences between configurations are paired (same questions) and resampled **by article / document** (cluster bootstrap, 2,000 resamples), because questions on one article and qualifiers in one document are not independent.

### 1.3 Normalization rule
Never strip or rewrite numbers, units, negation cues, comparatives, or symbols (≥, <, %). Only invisible characters, exotic spaces, ligatures and whitespace runs are changed, with an offset map back to the original (`cliniver.processing.normalize`). Unicode NFKC is deliberately **not** used: it rewrites superscripts ("10⁹/L" → "109/L").
""")

md("## 2. Setup")
code(r"""
import json, math, os, random, time, collections, itertools
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from IPython.display import display

from cliniver.config import load_project_config
from cliniver.config.processing_config import load_processing_config
from cliniver.utils.seed import set_seed
from cliniver.utils.hardware import collect_hardware_info
from cliniver.utils.experiment import Experiment
from cliniver.ingestion.http import RawStore
from cliniver.ingestion import covidqa
from cliniver.processing.normalize import normalize
from cliniver.processing.segment import detect_headings, sentences
from cliniver.processing.chunking import ChunkConfig, TokenIndex, make_chunks
from cliniver.processing.qualifiers import detect, split_mask, truncated_mask
from cliniver.processing.long_docs import iter_documents, to_long_doc
from cliniver.retrieval.simple import BM25Index, DenseIndex
from cliniver.evaluation.span_retrieval import Gold, evaluate, cluster_bootstrap_diff

pcfg = load_project_config()
cfg = load_processing_config()
SEED = set_seed(pcfg.reproducibility.seed)
# SMOKE=1: a few configurations on small subsets, written to a separate experiment id (code check only)
SMOKE = os.environ.get("CLINIVER_SMOKE") == "1"
SNAP = cfg.corpus_snapshot
PROC = pcfg.paths.absolute("data_processed") / SNAP
RAW = pcfg.paths.absolute("data_raw")
ANN = pcfg.paths.absolute("data_raw").parent / "annotations"
pd.set_option("display.width", 170); pd.set_option("display.max_columns", 40)

exp = Experiment(exp_id="EXP-02-000-smoke" if SMOKE else "EXP-02-001", title="Document processing and chunking: strategy comparison and corpus chunking",
                 research_question="RQ1 / H1", config=cfg.model_dump(mode="json"), seed=SEED,
                 packages=["torch", "transformers", "sentence-transformers", "tokenizers", "bm25s", "pysbd", "pandas", "cliniver"])
hw = collect_hardware_info().to_dict()
print("seed", SEED, "| snapshot", SNAP, "| GPUs:", [g.get("name") for g in hw.get("gpus", [])], "| torch", torch.__version__)
""")
code(r"""
from sentence_transformers import SentenceTransformer
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
encoder = SentenceTransformer(cfg.retriever.dense_model, revision=cfg.retriever.dense_revision, device=DEVICE)
tokenizer = encoder.tokenizer
assert tokenizer.is_fast, "offset mapping needs a fast tokenizer"
embed = lambda xs: encoder.encode(xs, batch_size=128, normalize_embeddings=True, show_progress_bar=False)
print(cfg.retriever.dense_model, "@", cfg.retriever.dense_revision[:10], "| device", DEVICE, "| max_seq", encoder.max_seq_length)
""")

md(r"""
## 3. Datasets
### 3.1 How long are the corpus documents? (decides where chunking matters)
""")
code(r"""
t0 = time.perf_counter()
lens = collections.defaultdict(list)
batch, srcs = [], []
def flush():
    enc = tokenizer([x for x in batch], add_special_tokens=False, truncation=False)["input_ids"]
    for s, ids in zip(srcs, enc):
        lens[s].append(len(ids))
    batch.clear(); srcs.clear()
for d in (itertools.islice(iter_documents(PROC / "documents.jsonl"), 5000) if SMOKE else iter_documents(PROC / "documents.jsonl")):
    batch.append(d["text"]); srcs.append(d["source"])
    if len(batch) == 2000:
        flush()
flush()
rows = []
for s, v in lens.items():
    v = np.array(v)
    rows.append({"source": s, "docs": len(v), "tokens_median": int(np.median(v)), "tokens_p90": int(np.percentile(v, 90)),
                 "tokens_max": int(v.max()), "share_gt_128": round((v > 128).mean(), 3), "share_gt_256": round((v > 256).mean(), 3),
                 "share_gt_512": round((v > 512).mean(), 3), "total_tokens_M": round(v.sum() / 1e6, 2)})
corpus_len = pd.DataFrame(rows).set_index("source").sort_values("docs", ascending=False)
exp.save_table("corpus_token_lengths", corpus_len); exp.log_result("corpus_token_lengths", corpus_len.to_dict(orient="index"))
display(corpus_len); print(f"tokenized {sum(len(v) for v in lens.values()):,} documents in {time.perf_counter()-t0:.0f}s")
""")

md("### 3.2 Long-document subset of our corpus (qualifier study)")
code(r"""
long_docs = [to_long_doc(d) for d in iter_documents(PROC / "documents.jsonl", {"europepmc_oa", "openfda_label"})]
long_docs = [d for d in long_docs if d.text.strip()]  # 1 openFDA label has no clinical text at source (NB01 §B6.4)
if SMOKE:
    long_docs = long_docs[:20] + long_docs[-20:]
ld_tok = {d.doc_id: TokenIndex(d.text, tokenizer) for d in long_docs}
ld_stats = pd.DataFrame([{"source": d.source, "tokens": len(ld_tok[d.doc_id].starts), "sections_declared": d.n_sections_declared,
                          "sections_located": len(d.sections)} for d in long_docs]).groupby("source").agg(
    docs=("tokens", "size"), tokens_median=("tokens", "median"), sections_declared=("sections_declared", "sum"),
    sections_located=("sections_located", "sum"))
exp.log_result("long_docs", ld_stats.to_dict(orient="index")); display(ld_stats)
""")

md("### 3.3 COVID-QA (span-retrieval benchmark)")
code(r"""
qa_raw = covidqa.fetch(cfg.benchmark.url, RawStore(RAW, "covidqa", SNAP))
qa_docs, qa_qs = covidqa.parse(qa_raw)
qa_norm = {d.doc_id: normalize(d.text) for d in qa_docs}
golds = []
for q in qa_qs:
    n = qa_norm[q.gold.doc_id]; a, b = n.map_span(q.gold.start, q.gold.end)
    golds.append(Gold(q.qid, q.gold.doc_id, a, b))
# The mapped gold span must equal the normalized original span (modulo whitespace; ligatures are expanded)
_raw = {d.doc_id: d.text for d in qa_docs}
_squash = lambda t: "".join(c for c in normalize(t).text if not c.isspace())
assert all(_squash(qa_norm[g.doc_id].text[g.start:g.end]) == _squash(_raw[q.gold.doc_id][q.gold.start:q.gold.end])
           for g, q in zip(golds, qa_qs))
qa_text = {k: v.text for k, v in qa_norm.items()}
qa_tok = {k: TokenIndex(t, tokenizer) for k, t in qa_text.items()}
qa_secs = {k: detect_headings(t) for k, t in qa_text.items()}
qa_title = {k: t.split("\n", 1)[0] for k, t in qa_text.items()}

arts = sorted(qa_text); rng = random.Random(SEED); rng.shuffle(arts)
n_dev = round(len(arts) * cfg.benchmark.dev_fraction)
split_of = {a: ("dev" if i < n_dev else "test") for i, a in enumerate(arts)}
qsplit = np.array([split_of[g.doc_id] for g in golds])
ans_tok = np.array([qa_tok[g.doc_id].count(g.start, g.end) for g in golds])
qa_summary = {"articles": len(qa_docs), "questions": len(golds), "dev_articles": n_dev, "test_articles": len(arts) - n_dev,
              "dev_questions": int((qsplit == "dev").sum()), "test_questions": int((qsplit == "test").sum()),
              "article_tokens_median": int(np.median([len(t.starts) for t in qa_tok.values()])),
              "answer_tokens_median": int(np.median(ans_tok)), "answer_tokens_p90": int(np.percentile(ans_tok, 90)),
              "answer_tokens_max": int(ans_tok.max()), "answers_over_128_tokens": int((ans_tok > 128).sum()),
              "contexts_changed_by_normalization": sum(n.changed for n in qa_norm.values()),
              "detected_sections_median": int(np.median([len(s) for s in qa_secs.values()]))}
exp.log_result("covidqa", qa_summary); print(json.dumps(qa_summary, indent=1))
""")

md(r"""
## 4. Normalization: what changes, and what must not
The invariant is checked on every document of the long subset and on COVID-QA: the multiset of digits and clinically meaningful symbols must be **identical** before and after normalization.
""")
code(r"""
KEEP = set("0123456789%<>≤≥=±×·/.,-–+°µμ²³⁰¹⁴⁵⁶⁷⁸⁹")
def keep_counts(t): return collections.Counter(c for c in t if c in KEEP)
raw_long = {d["doc_id"]: d["text"] for d in iter_documents(PROC / "documents.jsonl", {"europepmc_oa", "openfda_label"})}
viol, changed, removed_chars = 0, 0, 0
for did, t in itertools.chain(raw_long.items(), ((d.doc_id, d.text) for d in qa_docs)):
    n = normalize(t)
    viol += keep_counts(t) != keep_counts(n.text)
    changed += n.changed; removed_chars += len(t) - len(n.text)
norm_check = {"documents_checked": len(raw_long) + len(qa_docs), "documents_changed": changed,
              "chars_removed_total": removed_chars, "invariant_violations": viol}
exp.log_result("normalization_check", norm_check); print(norm_check)
assert viol == 0
ex = "HbA1c ≥ 7.0 %,​ neutrophils 10⁹/L  (no­effect)\r\n\r\n\r\nﬁnal"
print(repr(ex), "→", repr(normalize(ex).text))
""")

md(r"""
## 5. Configurations and baseline
The baseline is `fixed-256-o0` (the common default in RAG tutorials). The full grid is defined in `configs/processing.yaml`.
""")
code(r"""
G = cfg.grid
CONFIGS = [ChunkConfig("fixed", s, o) for s in G.fixed_sizes for o in G.fixed_overlaps]
CONFIGS += [ChunkConfig(st, s) for st in ("sentence", "paragraph", "recursive", "section") for s in G.structured_sizes]
if G.header_variants:
    CONFIGS += [ChunkConfig("section", s, header=True) for s in G.structured_sizes]
CONFIGS += [ChunkConfig("semantic", s, pct=p) for s in G.structured_sizes for p in G.semantic_pcts]
if SMOKE:
    CONFIGS = [c for c in CONFIGS if c.name in ("fixed-256-o0", "fixed-128-o25", "section-256", "section-256+hdr", "semantic-128-p90")]
BASELINE = "fixed-256-o0"
family = lambda name: "fixed" if name.startswith("fixed") else "structure-aware"
print(len(CONFIGS), "configurations:", ", ".join(c.name for c in CONFIGS))
""")

md(r"""
## 6. Experiments
### 6.1 E1: span retrieval on COVID-QA (all configurations × two retrievers)
""")
code(r"""
QUESTIONS = [q.text for q in qa_qs]
TOPK = cfg.benchmark.top_k
_sent_emb_cache = {}
def embed_cached(xs):
    miss = [x for x in xs if x not in _sent_emb_cache]
    if miss:
        for x, e in zip(miss, embed(miss)):
            _sent_emb_cache[x] = e
    return np.stack([_sent_emb_cache[x] for x in xs])

per_q, cost_rows = [], []
for c in CONFIGS:
    t0 = time.perf_counter()
    chunks = [ch for d in arts for ch in make_chunks(d, qa_text[d], c, qa_tok[d], title=qa_title[d], sections=qa_secs[d],
                                                    embed=embed_cached if c.strategy == "semantic" else None)]
    t_chunk = time.perf_counter() - t0
    texts = [ch.index_text for ch in chunks]
    cdoc = [ch.doc_id for ch in chunks]; cst = np.array([ch.start for ch in chunks]); cen = np.array([ch.end for ch in chunks])
    ntok = np.array([ch.n_tokens for ch in chunks])
    if c.header:  # the reader also reads the header: count its tokens against the budget
        hdr = [ch.index_text[:len(ch.index_text) - len(ch.text)] for ch in chunks]
        ntok = ntok + np.array([len(x) for x in tokenizer(hdr, add_special_tokens=False)["input_ids"]])
    for rname, make in (("bm25", lambda: BM25Index(texts)),
                        ("dense", lambda: DenseIndex(texts, encoder, batch_size=cfg.retriever.batch_size, query_prefix=cfg.retriever.query_prefix))):
        idx = make(); res = idx.search(QUESTIONS, TOPK)
        ev = evaluate(golds, res.ids, cdoc, cst, cen, ntok, budgets=tuple(cfg.budgets))
        ev["config"], ev["strategy"], ev["size"], ev["retriever"], ev["split"] = c.name, c.strategy + ("+hdr" if c.header else ""), c.size, rname, qsplit
        per_q.append(ev)
        cost_rows.append({"config": c.name, "retriever": rname, "chunks": len(chunks), "mean_tokens": round(ntok.mean(), 1),
                          "indexed_tokens": int(ntok.sum()), "chunk_s": round(t_chunk, 2), "index_s": round(idx.index_s, 2),
                          "query_s": round(res.query_s, 2)})
        del idx
    torch.cuda.empty_cache() if DEVICE == "cuda" else None
    print(f"{c.name:20s} chunks={len(chunks):5d} {time.perf_counter()-t0:5.1f}s")
E1 = pd.concat(per_q, ignore_index=True)
cost = pd.DataFrame(cost_rows)
exp.save_table("e1_per_question", E1); exp.save_table("e1_cost", cost)
""")
code(r"""
METRICS = ["hit@1", "hit@5", "hit@10", "rr", "ndcg@10", "hit@512t", "hit@1024t", "hit@2048t", "cover@1024t", "gold_in_one_chunk"]
summ = E1.groupby(["split", "retriever", "config"])[METRICS].mean().round(3)
dev = summ.loc["dev"]
dev_primary = dev[cfg.primary_metric].unstack("retriever")
dev_primary["mean"] = dev_primary.mean(axis=1)
dev_primary = dev_primary.join(cost.groupby("config")[["chunks", "mean_tokens", "indexed_tokens"]].first()).sort_values("mean", ascending=False)
exp.save_table("e1_summary", summ); exp.save_table("e1_dev_primary", dev_primary)
display(dev_primary)
""")

md("### 6.2 E2: qualifier preservation on our long documents (all configurations)")
code(r"""
quals = {d.doc_id: detect(d.text) for d in long_docs}
q_rows = []
for c in CONFIGS:
    t0 = time.perf_counter()
    for d in long_docs:
        ch = make_chunks(d.doc_id, d.text, c, ld_tok[d.doc_id], title=d.title, sections=d.sections,
                         embed=embed_cached if c.strategy == "semantic" else None)
        spans_, chs_ = [(q.start, q.end) for q in quals[d.doc_id]], [(x.start, x.end) for x in ch]
        for q, m, t in zip(quals[d.doc_id], split_mask(spans_, chs_), truncated_mask(spans_, chs_)):
            q_rows.append((c.name, d.doc_id, d.source, q.kind, m, t))
    print(f"{c.name:20s} {time.perf_counter()-t0:5.1f}s")
E2 = pd.DataFrame(q_rows, columns=["config", "doc_id", "source", "kind", "split", "truncated"])
q_summary = E2.groupby(["config", "kind"]).truncated.mean().unstack("kind").round(4).add_prefix("trunc_")
q_summary["trunc_all"] = E2.groupby("config").truncated.mean().round(4)
q_summary["split_all"] = E2.groupby("config").split.mean().round(4)
q_summary = q_summary.sort_values("trunc_all")
exp.save_table("e2_qualifier_split", q_summary)
exp.log_result("qualifiers_detected", E2[E2.config == BASELINE].groupby(["source", "kind"]).size().unstack(fill_value=0).to_dict(orient="index"))
display(q_summary)
""")

md(r"""
### 6.3 E3: precision of the qualifier rules (manual inspection)
Two rounds of 150 detections each (30 per type, stratified random sample from the 737 long documents).
- **Round 1** (seed 42, initial rules) revealed systematic errors; the rules were revised (`cliniver.processing.qualifiers` docstring).
- **Round 2** (seed 43, revised rules) is a fresh sample and gives the reported precision.

**Annotator:** the AI developer (Claude), reading each span in context. This is **not** a human or clinician annotation (D5). Recall of the rules is **not measured** (it would require exhaustive annotation).
""")
code(r"""
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return round((c - h) / d, 3), round((c + h) / d, 3)
val = []
for rnd in (1, 2):
    a = json.loads((ANN / f"nb02_qualifier_validation_round{rnd}.json").read_text(encoding="utf-8"))
    df = pd.DataFrame(a["items"]); df["round"] = rnd
    val.append(df)
val = pd.concat(val)
# Round 2 must be reproducible from the CURRENT rules: every sampled span is still a detection
cur = {(d, q.kind, q.start, q.end) for d, qs in quals.items() for q in qs}
r2 = val[val["round"] == 2]
reproduced = sum((r.doc_id, r.kind, r.start, r.end) in cur for r in r2.itertuples())
prec = []
for (rnd, kind), g in val.groupby(["round", "kind"]):
    k, n = int(g.valid.sum()), len(g); lo, hi = wilson(k, n)
    prec.append({"round": rnd, "kind": kind, "valid": k, "n": n, "precision": round(k / n, 3), "ci_low": lo, "ci_high": hi})
for rnd, g in val.groupby("round"):
    k, n = int(g.valid.sum()), len(g); lo, hi = wilson(k, n)
    prec.append({"round": rnd, "kind": "ALL", "valid": k, "n": n, "precision": round(k / n, 3), "ci_low": lo, "ci_high": hi})
prec = pd.DataFrame(prec)
exp.save_table("e3_rule_precision", prec)
exp.log_result("rule_precision", {"round2_reproduced_from_current_rules": f"{reproduced}/{len(r2)}",
                                  "round2_boundary_ok": int(r2.boundary_ok.sum()), "table": prec.to_dict(orient="records")})
print(f"round-2 sample reproduced by current rules: {reproduced}/{len(r2)}")
display(prec)
display(val[~val.valid | (val.get("boundary_ok") == False)][["round", "kind", "span", "note"]])
""")

md(r"""
## 7. Evaluation
### 7.1 Selection on dev (pre-registered rule)
The selected configuration maximizes the primary metric averaged over the two retrievers on **dev**. The best fixed-size and best structure-aware configurations (same rule within each family) are compared on **test** for H1.
""")
code(r"""
SELECTED = dev_primary.index[0]
best_fixed = next(c for c in dev_primary.index if family(c) == "fixed")
best_struct = next(c for c in dev_primary.index if family(c) == "structure-aware")
sel = {"selected": SELECTED, "best_fixed": best_fixed, "best_structure_aware": best_struct, "baseline": BASELINE}
exp.log_result("selection", sel); print(sel)
""")

md("### 7.2 H1 part 1: retrieval (test split)")
code(r"""
test = E1[E1.split == "test"]
h1_rows = []
for r in ("bm25", "dense"):
    a = test[(test.config == best_struct) & (test.retriever == r)].set_index("qid")
    b = test[(test.config == best_fixed) & (test.retriever == r)].set_index("qid").loc[a.index]
    for m in (cfg.primary_metric, "hit@5", "rr", "cover@1024t"):
        res = cluster_bootstrap_diff(a[m], b[m], a["doc_id"], seed=SEED)
        h1_rows.append({"retriever": r, "metric": m, "structure_aware": round(a[m].mean(), 3), "fixed": round(b[m].mean(), 3), **{k: round(v, 4) if isinstance(v, float) else v for k, v in res.items()}})
h1_ret = pd.DataFrame(h1_rows)
exp.save_table("h1_retrieval_test", h1_ret); exp.log_result("h1_retrieval_test", h1_ret.to_dict(orient="records"))
display(h1_ret)
test_summary = summ.loc["test"][METRICS].unstack("retriever")
exp.log_result("test_summary_selected", {c: summ.loc["test"].xs(c, level="config").to_dict(orient="index") for c in {SELECTED, best_fixed, best_struct, BASELINE}})
""")

md("### 7.3 H1 part 2: qualifier splits (our long documents)")
code(r"""
qa_ = E2[E2.config == best_struct].reset_index(drop=True)
qb_ = E2[E2.config == best_fixed].reset_index(drop=True)
assert (qa_[["doc_id", "kind"]].values == qb_[["doc_id", "kind"]].values).all()  # same spans, same order
h1_q = cluster_bootstrap_diff(qa_.truncated, qb_.truncated, qa_.doc_id, seed=SEED)
h1_q = {"metric": "truncation rate", "structure_aware": round(qa_.truncated.mean(), 4), "fixed": round(qb_.truncated.mean(), 4), **h1_q,
        "secondary_split_rate": {"structure_aware": round(qa_.split.mean(), 4), "fixed": round(qb_.split.mean(), 4),
                                 **cluster_bootstrap_diff(qa_.split, qb_.split, qa_.doc_id, seed=SEED)}}
exp.log_result("h1_qualifiers", h1_q); print(json.dumps(h1_q, indent=1))
""")
code(r"""
ret_ok = all((h1_ret[(h1_ret.retriever == r) & (h1_ret.metric == cfg.primary_metric)].ci_low > 0).item() for r in ("bm25", "dense"))
qual_ok = h1_q["ci_high"] < 0
verdict = "SUPPORTED" if ret_ok and qual_ok else ("PARTIALLY SUPPORTED" if ret_ok or qual_ok else "NOT SUPPORTED")
exp.log_result("h1_verdict", {"retrieval_part": ret_ok, "qualifier_part": qual_ok, "verdict": verdict})
print("H1:", verdict, "| retrieval part:", ret_ok, "| qualifier part:", qual_ok)
""")

md("### 7.4 E4: raw vs normalized text (selected configuration, test split)")
code(r"""
c_sel = next(c for c in CONFIGS if c.name == SELECTED)
raw_text = {d.doc_id: d.text for d in qa_docs}
raw_tok = {k: TokenIndex(t, tokenizer) for k, t in raw_text.items()}
raw_golds = [q.gold for q in qa_qs]
e4 = []
for variant, TXT, TOK, GOLD in (("normalized", qa_text, qa_tok, golds), ("raw", raw_text, raw_tok, raw_golds)):
    chunks = [ch for d in arts for ch in make_chunks(d, TXT[d], c_sel, TOK[d], title=TXT[d].split("\n", 1)[0],
                                                    sections=detect_headings(TXT[d]), embed=embed_cached if c_sel.strategy == "semantic" else None)]
    texts = [ch.index_text for ch in chunks]
    args = ([ch.doc_id for ch in chunks], np.array([ch.start for ch in chunks]), np.array([ch.end for ch in chunks]), np.array([ch.n_tokens for ch in chunks]))
    for rname, idx in (("bm25", BM25Index(texts)), ("dense", DenseIndex(texts, encoder, query_prefix=cfg.retriever.query_prefix))):
        ev = evaluate(GOLD, idx.search(QUESTIONS, TOPK).ids, *args, budgets=tuple(cfg.budgets))
        ev = ev[qsplit == "test"]
        e4.append({"variant": variant, "retriever": rname, "chunks": len(chunks), **ev[METRICS].mean().round(3).to_dict()})
e4 = pd.DataFrame(e4).set_index(["variant", "retriever"])
exp.save_table("e4_raw_vs_normalized", e4); exp.log_result("e4_raw_vs_normalized", e4.reset_index().to_dict(orient="records"))
display(e4)
""")

md("## 8. Results and visualizations")
code(r"""
fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
for ax, r in zip(axes[:2], ("bm25", "dense")):
    d = summ.loc["test"].xs(r, level="retriever")[cfg.primary_metric].rename("v").reset_index()
    d["strategy"] = d.config.str.replace(r"-\d+(-o\d+|-p\d+)?", "", regex=True); d["size"] = d.config.str.extract(r"-(\d+)")[0].astype(int)
    for st, g in d.groupby("strategy"):
        g = g.groupby("size").v.max()  # fixed: best overlap per size
        ax.plot(g.index, g.values, marker="o", label=st, lw=2.2 if st != "fixed" else 3, ls="--" if st == "fixed" else "-")
    ax.set_xscale("log", base=2); ax.set_xticks([128, 256, 512]); ax.set_xticklabels([128, 256, 512])
    ax.set_title(f"{r}: {cfg.primary_metric} (test)"); ax.set_xlabel("max chunk tokens"); ax.grid(alpha=.3)
axes[0].legend(fontsize=7)
qs_ = q_summary["trunc_all"].sort_values()
axes[2].barh(qs_.index, qs_.values, color=["#c44e52" if family(i) == "fixed" else "#4c72b0" for i in qs_.index])
axes[2].set_title("qualifier truncation rate (long docs; red = fixed)"); axes[2].tick_params(axis="y", labelsize=6)
plt.tight_layout(); exp.save_figure("results_overview", fig); plt.show()
""")
code(r"""
fig, ax = plt.subplots(figsize=(7, 4))
d = E1[(E1.split == "test")].groupby(["config", "retriever"])[cfg.primary_metric].mean().unstack().join(cost.groupby("config").indexed_tokens.first())
for r, mk in (("bm25", "o"), ("dense", "s")):
    ax.scatter(d.indexed_tokens / 1e3, d[r], marker=mk, label=r, c=["#c44e52" if family(i) == "fixed" else "#4c72b0" for i in d.index])
ax.set_xlabel("indexed tokens (thousands): cost of overlap"); ax.set_ylabel(cfg.primary_metric + " (test)"); ax.legend(); ax.grid(alpha=.3)
ax.set_title("Quality vs index size (red = fixed, blue = structure-aware)")
exp.save_figure("quality_vs_cost", fig); plt.show()
""")

md(r"""
## 9. Error analysis (selected configuration, dev split, BM25 and dense)
Each miss (`hit@1024t` false) is put in one category:
- **split**: no chunk contains the whole gold span (a chunking error);
- **wrong document**: no top-budget chunk is from the gold article (a retrieval error);
- **right document, wrong chunk**: the article was found but not the chunk holding the answer.
""")
code(r"""
ev_dev = E1[(E1.config == SELECTED) & (E1.split == "dev")]
# split "wrong document or chunk" using the ranked lists of the selected configuration
chunks_sel = [ch for d in arts for ch in make_chunks(d, qa_text[d], c_sel, qa_tok[d], title=qa_title[d], sections=qa_secs[d],
                                                    embed=embed_cached if c_sel.strategy == "semantic" else None)]
texts_sel = [ch.index_text for ch in chunks_sel]; doc_sel = np.array([ch.doc_id for ch in chunks_sel]); tok_sel = np.array([ch.n_tokens for ch in chunks_sel])
dev_idx = np.where(qsplit == "dev")[0]
refined = collections.Counter()
examples = []
for rname, idx in (("bm25", BM25Index(texts_sel)), ("dense", DenseIndex(texts_sel, encoder, query_prefix=cfg.retriever.query_prefix))):
    ids = idx.search([QUESTIONS[i] for i in dev_idx], TOPK).ids
    evr = ev_dev[ev_dev.retriever == rname].set_index("qid")
    for i, row in zip(dev_idx, ids):
        g = golds[i]; r = evr.loc[g.qid]
        if r[cfg.primary_metric]:
            refined[(rname, "hit")] += 1; continue
        if not r.gold_in_one_chunk:
            refined[(rname, "split (chunking)")] += 1; continue
        n = int(np.searchsorted(np.cumsum(tok_sel[row]), 1024, side="right"))
        c_ = "right document, wrong chunk" if g.doc_id in set(doc_sel[row[:n]]) else "wrong document"
        refined[(rname, c_)] += 1
        if len(examples) < 6 and c_ == "right document, wrong chunk":
            examples.append({"retriever": rname, "question": QUESTIONS[i], "gold": qa_text[g.doc_id][g.start:g.end][:160]})
err = pd.Series(refined).unstack(0).fillna(0).astype(int)
exp.save_table("error_categories_dev", err); exp.log_result("error_categories_dev", err.to_dict())
display(err); display(pd.DataFrame(examples))
""")
code(r"""
# Which qualifier types does the selected configuration still cut, and where? (long documents)
sel_q = E2[E2.config == SELECTED]
cut_by = sel_q.groupby(["source", "kind"]).truncated.agg(["sum", "count", "mean"]).round(4)
exp.save_table("selected_qualifier_splits", cut_by); display(cut_by)
d0 = next(d for d in long_docs if d.source == "openfda_label")
ch0 = make_chunks(d0.doc_id, d0.text, c_sel, ld_tok[d0.doc_id], title=d0.title, sections=d0.sections, embed=embed_cached if c_sel.strategy == "semantic" else None)
for x in ch0[:3]:
    print(f"--- chunk {x.idx} [{x.section}] {x.n_tokens} tokens\n{x.index_text[:300]}…")
""")

md(r"""
## 10. Apply the selected configuration to the whole corpus
Every document (all sources) is chunked with the selected configuration; documents shorter than the size limit stay whole. Chunks keep exact character offsets into the normalized document text. Entity mentions are attached by offset: PubTator mentions for PubMed (aligned to our text), and RxNorm dictionary matches for all sources (`cliniver.processing.corpus_chunks`).
""")
code(r"""
from cliniver.processing.corpus_chunks import DrugMatcher, chunk_document
from cliniver.data.corpus_io import read_jsonl
concepts = [json.loads(l) for l in (PROC / "concepts.jsonl").open(encoding="utf-8")]
matcher = DrugMatcher(concepts)
t0 = time.perf_counter()
pt = collections.defaultdict(list)
with (PROC / "pubtator_entities.jsonl").open(encoding="utf-8") as fh:
    for line in fh:
        e = json.loads(line)
        pt[e["pmid"]].append((e["passage_type"], e["entity_type"], e["identifier"], e["text"], e["offset"], e["length"]))
print(f"PubTator mentions loaded for {len(pt):,} PMIDs in {time.perf_counter()-t0:.0f}s")
""")
code(r"""
OUTC = PROC / "chunks" / (SELECTED + ("-SMOKE" if SMOKE else ""))
OUTC.mkdir(parents=True, exist_ok=True)
stats, per_src, n_docs = collections.Counter(), collections.defaultdict(list), 0
t0 = time.perf_counter()
with (OUTC / "chunks.jsonl").open("w", encoding="utf-8") as fc, (OUTC / "chunk_entities.jsonl").open("w", encoding="utf-8") as fe:
    for d in (itertools.islice(iter_documents(PROC / "documents.jsonl"), 3000) if SMOKE else iter_documents(PROC / "documents.jsonl")):
        if not d["text"].strip():
            stats["skipped_empty_documents"] += 1; continue
        recs, ents, st = chunk_document(d, c_sel, tokenizer, matcher, pt.get(d["source_id"]) if d["source"] == "pubmed" else None,
                                        embed=embed if c_sel.strategy == "semantic" else None)
        stats.update(st); n_docs += 1
        per_src[d["source"]].append((len(recs), sum(r["n_tokens"] for r in recs)))
        for r in recs:
            fc.write(json.dumps(r, ensure_ascii=False) + "\n")
        for e in ents:
            fe.write(json.dumps(e, ensure_ascii=False) + "\n")
        if n_docs % 20000 == 0:
            print(f"{n_docs:,} docs, {time.perf_counter()-t0:.0f}s")
t_corpus = time.perf_counter() - t0
chunk_stats = pd.DataFrame({s: {"docs": len(v), "chunks": sum(a for a, _ in v), "chunks_per_doc_mean": round(np.mean([a for a, _ in v]), 2),
                                "single_chunk_docs_share": round(np.mean([a == 1 for a, _ in v]), 3), "tokens": sum(b for _, b in v)}
                            for s, v in per_src.items()}).T
manifest = {"config": SELECTED, "snapshot": SNAP, "tokenizer": f"{cfg.retriever.dense_model}@{cfg.retriever.dense_revision}",
            "documents": n_docs, "chunks": int(chunk_stats.chunks.sum()), "seconds": round(t_corpus), **stats,
            "pubtator_alignment_rate": round(1 - stats["pubtator_unaligned"] / max(1, stats["pubtator_mentions"]), 4),
            "files_mb": {p.name: round(p.stat().st_size / 1e6, 1) for p in OUTC.glob("*.jsonl")}}
(OUTC / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
exp.save_table("corpus_chunks", chunk_stats); exp.log_result("corpus_chunking", {**manifest, "by_source": chunk_stats.to_dict(orient="index")})
display(chunk_stats); print(json.dumps(manifest, indent=1))
""")
code(r"""
exp.finish()
print("results:", exp.dir)
""")

sys.path.insert(0, str(Path(__file__).parent))
try:
    from nb02_discussion import DISCUSSION_CELLS  # written after execution, from the outputs
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
