"""Builds notebooks/03_llm_responses_and_claim_extraction.ipynb (headless construction; the executed notebook is
the research artifact). Discussion cells: nb03_discussion.py, written after execution from the outputs.

Usage: python scripts/notebooks/build_nb03.py notebooks/03_llm_responses_and_claim_extraction.ipynb
"""
import sys
from pathlib import Path

import nbformat as nbf

out = Path(sys.argv[1])
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# Notebook 03 — LLM Responses and Claim Extraction
**CliniVer** · Experiment `EXP-03-001` · open-weight models only, run locally (ADR-0004)

| | |
|---|---|
| **Objective** | (A) Produce realistic AI answers to in-domain patient questions with local open-weight LLMs; (B) choose how answers are decomposed into verifiable claims, with evidence |
| **Research question** | **RQ2:** Which open-weight-LLM-based extraction method yields atomic, complete, qualifier-preserving claims? |
| **Hypothesis (pre-registered)** | **H2:** Schema-constrained, qualifier-aware extraction preserves more qualifiers than sentence splitting and generic atomic decomposition, without lower claim precision |
| **Decision rule (fixed before running)** | On the held-out K-QA test answers, with the selected extractor model: **(a)** qualifier preservation of `qualified` − `atomic` > 0 (95% bootstrap CI excludes 0) **and** `qualified` not significantly lower than `sentence` (CI upper bound of the difference ≥ 0); **(b)** claim precision of `qualified` − `atomic`: CI lower bound > −0.03. *Supported* if (a) and (b); *partially* if one; otherwise *not supported* |
| **Model selection** | No model is chosen in advance (research_foundation §9.3). The extractor model is selected on K-QA **dev** answers (highest F1 of claim precision and statement recall, method `qualified`, among models with ≥ 95% valid JSON) and evaluated on **test** |

> **Medical safety.** Generated answers are research material to be *verified*, never advice. They are stored with their model, parameters and seed, and are never shown as recommendations.
""")

md(r"""
## 1. Background and method
### 1.1 Runtime
- **llama.cpp `llama-server`** (release pinned in `configs/llm.yaml`, official Windows CUDA build, SHA-256 recorded), running GGUF 4/8-bit models fully on the GPU.
- **JSON-schema constrained decoding**: the server compiles the schema to a grammar, so syntactically invalid JSON should be impossible. Validity is still measured, since outputs can be truncated by the token limit.
- **One model on the GPU at a time.** Each model's server is started, used and stopped; peak GPU memory is sampled from `nvidia-smi` every 0.5 s.
- Every call is logged with model file, runtime release, parameters, prompt hash, seed, latency and token counts. The log is also a **replay cache**: re-executing this notebook answers identical calls from the log, without a GPU (`cliniver.llm.runtime`).
- Qwen3 "thinking" is disabled (`enable_thinking: false`) for all calls: answers and JSON are produced directly.

### 1.2 Candidate models (all Apache-2.0, weights pinned to a Hugging Face commit)
| Key | Model | Quantization | Role |
|---|---|---|---|
| `qwen3-8b` | Qwen3-8B | Q4_K_M | stronger generator; extractor candidate |
| `qwen3-1.7b` | Qwen3-1.7B | Q8_0 | weaker generator (natural errors, by design); extractor candidate |
| `granite-3.3-8b` | IBM Granite 3.3 8B Instruct | Q4_K_M | second model family; extractor candidate |

### 1.3 Data
- **K-QA** (Manes et al., 2024; MIT): 201 real patient questions with **physician-written answers** and physician-written *Must_have* / *Nice_to_have* statements. These are the **human reference** for extraction. Split 60 dev / 141 test (seeded).
- **MedLFQA** (Jeong et al., 2024; CC-BY-4.0 card): LiveQA, MedicationQA, HealthSearchQA and K-QA silver questions. Used only as **query sources** for Part A (their statements are LLM-generated, not human).
- Part A uses questions about our five conditions or 49 drugs (`cliniver.ingestion.qa_datasets.domain_regex`). Part B uses all K-QA answers: extraction is evaluated on physician text in general medicine. This is a stated domain difference.

### 1.4 Extraction methods (`cliniver.claims.extract`)
1. `sentence`: each answer sentence is a claim (baseline, no model).
2. `atomic`: generic atomic-decomposition prompt, JSON list of strings.
3. `qualified`: qualifier-aware extraction into the CliniVer claim schema (text, source sentence, type, drug, condition, population, negated, dose, frequency, temporal, conditional, certainty, verifiable).

### 1.5 Metrics (`cliniver.evaluation.extraction`)
- **Claim precision (faithfulness):** share of claims entailed by the source answer (NLI: answer ⇒ claim). Each claim is judged **on its own**, so a claim that only makes sense in context ("However, they are most effective…") counts as not entailed. Precision therefore measures "standalone *and* supported", which is what a verifier needs. 1 − precision is the share of claims that are hallucinated or not standalone.
- **Statement recall (coverage):** share of physician statements entailed by at least one claim (claim ⇒ statement); also a *set* variant (all claims joined ⇒ statement), because one physician statement can combine several atomic facts.
- **F1** of precision and recall.
- **Qualifier preservation:** rule-detected qualifiers in the answer (NB02 rules: negation, quantity, threshold, effect, population) that survive in the claims (heuristic per type).
- **Atomicity / standalone proxies:** claims per sentence, words per claim, share of claims opening with an unresolved pronoun ("It", "This", "They", …).
- **Validity and cost:** valid JSON rate, latency, tokens/s, peak GPU memory.
- **NLI judge:** `DeBERTa-v3-large` (MNLI/FEVER/ANLI/LING/WANLI; MIT; pinned revision). It is itself a model, so 40 of its judgements are spot-checked manually (§7.3).
- **Statistics:** paired differences with a bootstrap over answers (2,000 resamples).
""")

md("## 2. Setup")
code(r"""
import json, math, os, random, re, time, collections, itertools, hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from IPython.display import display

from cliniver.config import load_project_config
from cliniver.config.llm_config import load_llm_config, load_claims_config
from cliniver.utils.seed import set_seed
from cliniver.utils.experiment import Experiment
from cliniver.ingestion.http import RawStore, RateLimitedClient, cached_get
from cliniver.ingestion.qa_datasets import parse_jsonl, domain_regex
from cliniver.llm.runtime import ManagedClient, gpu_memory_used_mb
from cliniver.claims.extract import extract_sentence, extract_atomic, extract_qualified, numbered_sentences
from cliniver.evaluation.extraction import NLIJudge, qualifier_kept
from cliniver.processing.qualifiers import detect

pcfg = load_project_config(); lcfg = load_llm_config(); ccfg = load_claims_config()
SEED = set_seed(ccfg.seed)
# SMOKE=1: small model only, tiny subsets, separate experiment id and call log (code check only; never results)
SMOKE = os.environ.get("CLINIVER_SMOKE") == "1"
SNAP = ccfg.snapshot
RAW = pcfg.paths.absolute("data_raw"); PROC = pcfg.paths.absolute("data_processed") / SNAP
ANN = RAW.parent / "annotations"
LLM_LOG = pcfg.paths.absolute("data_interim") / SNAP / ("llm_calls_smoke" if SMOKE else "llm_calls")   # replay cache + call log (one file per model)
LLM_LOG.mkdir(parents=True, exist_ok=True)
pd.set_option("display.width", 170); pd.set_option("display.max_columns", 40); pd.set_option("display.max_colwidth", 120)
exp = Experiment(exp_id="EXP-03-000-smoke" if SMOKE else "EXP-03-001", title="LLM responses and claim extraction", research_question="RQ2 / H2",
                 config={"llm": lcfg.model_dump(mode="json"), "claims": ccfg.model_dump(mode="json")}, seed=SEED,
                 packages=["torch", "transformers", "pandas", "pysbd", "cliniver"])
print("seed", SEED, "| runtime", lcfg.runtime.name, lcfg.runtime.release, "| models:", list(lcfg.models))
MODELS = ["qwen3-1.7b"] if SMOKE else list(lcfg.models)
for k in MODELS:
    p = lcfg.model_path(k); sha = (p.parent / (p.name + ".sha256"))
    print(f"  {k:15s} {p.name:40s} {p.stat().st_size/1e9:5.2f} GB sha256={sha.read_text().strip()[:16] if sha.exists() else 'n/a'}")

def managed(key):
    return ManagedClient(lcfg, key, LLM_LOG / f"{key}.jsonl", server_log=pcfg.paths.absolute("logs") / f"llama-server_{key}.log")
""")

md("## 3. Datasets")
code(r"""
http = RateLimitedClient("qa_datasets", min_interval_s=1.0, max_in_flight=1)
kqa_raw = cached_get(http, RawStore(RAW, "kqa", SNAP), "questions_w_answers", "jsonl", ccfg.datasets["kqa"]["url"])
kqa = parse_jsonl(kqa_raw, "kqa_golden", human=True)
mstore = RawStore(RAW, "medlfqa", SNAP)
med = []
for f in ccfg.datasets["medlfqa"]["files"]:
    src = f.replace("_test_MedLFQA", "").replace("_wogold", "")
    med += parse_jsonl(cached_get(http, mstore, f, "jsonl", f"{ccfg.datasets['medlfqa']['base_url']}/{f}.jsonl"), src, human=False)

concepts = [json.loads(l) for l in (PROC / "concepts.jsonl").open(encoding="utf-8")]
drug_names = {n.lower() for c in concepts if c["source"] == "rxnorm" for n in [c["name"], *c["synonyms"]] if len(n) >= 4}
DOM = domain_regex(drug_names)
seen, queries = set(), []
for it in kqa + med:
    key = re.sub(r"\W+", " ", it.question.lower()).strip()
    if DOM.search(it.question) and key not in seen:
        seen.add(key); queries.append(it)
order = list(range(len(kqa))); random.Random(SEED).shuffle(order)
dev_idx = set(order[:ccfg.extraction.dev_size])
kqa_dev = [kqa[i] for i in sorted(dev_idx)]; kqa_test = [kqa[i] for i in range(len(kqa)) if i not in dev_idx]
if SMOKE:
    kqa_dev, kqa_test, queries = kqa_dev[:4], kqa_test[:4], queries[:4]
ds = {"kqa_answers": len(kqa), "kqa_dev": len(kqa_dev), "kqa_test": len(kqa_test),
      "kqa_statements_must": sum(len(x.must_have) for x in kqa), "kqa_statements_nice": sum(len(x.nice_to_have) for x in kqa),
      "kqa_answer_words_median": int(np.median([len(x.answer.split()) for x in kqa])),
      "medlfqa_questions": dict(collections.Counter(x.source for x in med)),
      "domain_queries": len(queries), "domain_queries_by_source": dict(collections.Counter(x.source for x in queries))}
exp.log_result("datasets", ds); print(json.dumps(ds, indent=1))
display(pd.DataFrame([{"source": q.source, "question": q.question} for q in queries]).sample(min(8, len(queries)), random_state=SEED))
""")

md(r"""
## 4. Runtime check: load time, GPU memory, speed, JSON validity
A fixed prompt per model (one free-text answer + one schema-constrained extraction), before any experiment.
""")
code(r"""
PROBE_Q = "What are the common side effects of metformin, and when should it not be used?"
runtime_rows = []
for key in MODELS:
    with managed(key) as mc:
        mc.start()   # always measure a live server (a replay must not erase hardware measurements)
        r1 = mc.chat([{"role": "system", "content": ccfg.generation.system_prompt}, {"role": "user", "content": PROBE_Q}],
                     temperature=0.0, max_tokens=256, seed=SEED, tag="probe-answer")
        r2 = extract_qualified(mc, r1["output"], seed=SEED, tag="probe-extract")
        runtime_rows.append({"model": key, **(mc.stats or {"load_s": None, "gpu_peak_mb": None}),
                             "tokens_per_s": r1["tokens_per_s"], "answer_tokens": r1["completion_tokens"],
                             "probe_json_valid": r2.valid_json, "probe_claims": len(r2.claims), "replayed": mc.stats is None})
runtime_df = pd.DataFrame(runtime_rows).set_index("model")
exp.save_table("runtime_check", runtime_df); exp.log_result("runtime_check", runtime_df.reset_index().to_dict(orient="records"))
display(runtime_df)
""")

md(r"""
## 5. Part A: answers to in-domain patient questions
Each in-domain question is answered by both generators (`configs/claims.yaml`): temperature 0.7, top-p 0.9, ≤ 512 tokens, seed 42. Answers are stored with full provenance.
""")
code(r"""
G = ccfg.generation
responses = []
for key in (["qwen3-1.7b"] if SMOKE else G.generators):
    t0 = time.perf_counter()
    with managed(key) as mc:
        for q in queries:
            r = mc.chat([{"role": "system", "content": G.system_prompt}, {"role": "user", "content": q.question}],
                        temperature=G.temperature, top_p=G.top_p, max_tokens=G.max_tokens, seed=SEED, tag=f"gen:{q.qid}")
            responses.append({"response_id": f"{key}:{q.qid}", "qid": q.qid, "query_source": q.source, "question": q.question,
                              "generator": key, "model_file": lcfg.models[key].file, "runtime": f"llama.cpp {lcfg.runtime.release}",
                              "params": {"temperature": G.temperature, "top_p": G.top_p, "max_tokens": G.max_tokens, "seed": SEED},
                              "answer": r["output"].strip(), "finish_reason": r["finish_reason"], "latency_s": r["latency_s"],
                              "completion_tokens": r["completion_tokens"], "tokens_per_s": r["tokens_per_s"], "call_key": r["key"]})
        stats = mc.stats
    print(f"{key}: {len(queries)} answers in {time.perf_counter()-t0:.0f}s; server stats: {stats}")
resp_df = pd.DataFrame(responses)
gen_summary = resp_df.groupby("generator").agg(answers=("answer", "size"), words_median=("answer", lambda s: int(s.str.split().str.len().median())),
                                               truncated=("finish_reason", lambda s: int((s == "length").sum())),
                                               empty=("answer", lambda s: int((s.str.len() == 0).sum())),
                                               tokens_per_s=("tokens_per_s", "median"), latency_s_median=("latency_s", "median"))
exp.save_table("generation_summary", gen_summary); exp.log_result("generation_summary", gen_summary.to_dict(orient="index"))
display(gen_summary)
OUTR = PROC / ("responses_smoke" if SMOKE else "responses"); OUTR.mkdir(parents=True, exist_ok=True)
with (OUTR / "responses.jsonl").open("w", encoding="utf-8") as fh:
    for r in responses:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
""")

md(r"""
## 6. Part B: extraction
### 6.1 Candidate models on K-QA dev (selection)
""")
code(r"""
def run_extraction(mc, items, methods, split):
    rows = []
    for it in items:
        for m in methods:
            if m == "sentence":
                res = extract_sentence(it.answer)
            elif m == "atomic":
                res = extract_atomic(mc, it.answer, seed=SEED, tag=f"atomic:{it.qid}")
            else:
                res = extract_qualified(mc, it.answer, seed=SEED, tag=f"qualified:{it.qid}")
            rows.append({"qid": it.qid, "split": split, "method": m, "model": None if m == "sentence" else mc.key, "res": res})
    return rows

ext_rows, cand_stats = [], {}
for key in (["qwen3-1.7b"] if SMOKE else ccfg.extraction.candidates):
    t0 = time.perf_counter()
    with managed(key) as mc:
        ext_rows += run_extraction(mc, kqa_dev, ["atomic", "qualified"], "dev")
        cand_stats[key] = mc.stats
    print(f"{key}: dev extraction {time.perf_counter()-t0:.0f}s")
ext_rows += run_extraction(None, kqa_dev, ["sentence"], "dev")
""")

md(r"""
### 6.2 NLI scoring (the LLM servers are stopped; the NLI judge now uses the GPU)
""")
code(r"""
E = ccfg.extraction
judge = NLIJudge(E.nli_model, E.nli_revision, device="cuda" if torch.cuda.is_available() else "cpu")
items_by_id = {it.qid: it for it in kqa}
_PRON = re.compile(r"^(?:it|this|that|they|these|those|he|she|its|their)\b", re.I)

def score(rows):
    # batch all NLI pairs once
    prem, hyp, idx = [], [], []
    for ri, r in enumerate(rows):
        it, claims = items_by_id[r["qid"]], [c.text for c in r["res"].claims]
        stmts = list(it.must_have) + list(it.nice_to_have)
        for ci, c in enumerate(claims):
            prem.append(it.answer); hyp.append(c); idx.append((ri, "faith", ci))
        for si, s in enumerate(stmts):
            for ci, c in enumerate(claims):
                prem.append(c); hyp.append(s); idx.append((ri, "cov", (si, ci)))
            prem.append(" ".join(claims)); hyp.append(s); idx.append((ri, "covset", si))
    p = judge.entail_prob(prem, hyp) if prem else np.zeros(0)
    acc = collections.defaultdict(lambda: {"faith": {}, "cov": collections.defaultdict(float), "covset": {}})
    for (ri, kind, k), v in zip(idx, p):
        if kind == "cov":
            acc[ri]["cov"][k[0]] = max(acc[ri]["cov"][k[0]], float(v))
        else:
            acc[ri][kind][k] = float(v)
    T = E.entail_threshold
    out = []
    for ri, r in enumerate(rows):
        it, res = items_by_id[r["qid"]], r["res"]
        claims = [c.text for c in res.claims]
        n_st = len(it.must_have) + len(it.nice_to_have)
        quals = detect(it.answer)
        kept = [qualifier_kept(q.kind, it.answer[q.start:q.end], claims) for q in quals] if claims else [False] * len(quals)
        _, sents = numbered_sentences(it.answer)
        a = acc[ri]
        out.append({"qid": r["qid"], "split": r["split"], "method": r["method"], "model": r["model"], "valid_json": res.valid_json,
                    "n_claims": len(claims), "n_statements": n_st, "n_must": len(it.must_have),
                    "precision": np.mean([v >= T for v in a["faith"].values()]) if claims else np.nan,
                    "recall": np.mean([a["cov"].get(si, 0.0) >= T for si in range(n_st)]) if n_st else np.nan,
                    "recall_must": np.mean([a["cov"].get(si, 0.0) >= T for si in range(len(it.must_have))]) if it.must_have else np.nan,
                    "recall_set": np.mean([a["covset"].get(si, 0.0) >= T for si in range(n_st)]) if n_st else np.nan,
                    "n_qualifiers": len(quals), "qual_kept": int(sum(kept)),
                    "claims_per_sentence": len(claims) / max(1, len(sents)),
                    "words_per_claim": np.mean([len(c.split()) for c in claims]) if claims else np.nan,
                    "pronoun_start": np.mean([bool(_PRON.match(c.strip())) for c in claims]) if claims else np.nan,
                    "latency_s": res.latency_s, "completion_tokens": res.completion_tokens})
    return pd.DataFrame(out)

t0 = time.perf_counter()
dev_scores = score(ext_rows)
print(f"NLI scoring {time.perf_counter()-t0:.0f}s")
def summarize(df):
    g = df.groupby(["method", "model"], dropna=False)
    s = g.agg(answers=("qid", "size"), valid_json=("valid_json", "mean"), claims=("n_claims", "mean"),
              precision=("precision", "mean"), recall=("recall", "mean"), recall_must=("recall_must", "mean"),
              recall_set=("recall_set", "mean"), claims_per_sentence=("claims_per_sentence", "mean"),
              words_per_claim=("words_per_claim", "mean"), pronoun_start=("pronoun_start", "mean"), latency_s=("latency_s", "median"))
    q = g.apply(lambda x: x.qual_kept.sum() / max(1, x.n_qualifiers.sum()), include_groups=False).rename("qualifier_preservation")
    s = s.join(q)
    s["f1"] = 2 * s.precision * s.recall / (s.precision + s.recall)
    return s.round(3)
dev_summary = summarize(dev_scores)
exp.save_table("extraction_dev", dev_summary); display(dev_summary)
""")

md("### 6.3 Model selection (pre-registered rule)")
code(r"""
cand = dev_summary.xs("qualified", level="method")
eligible = cand[cand.valid_json >= E.min_json_validity]
SELECTED = eligible[E.selection_metric].idxmax() if len(eligible) else None
sel = {"selected_extractor": SELECTED, "eligible": list(eligible.index), "dev_f1": cand[E.selection_metric].to_dict(),
       "dev_valid_json": cand.valid_json.to_dict(), "server_stats": cand_stats}
exp.log_result("extractor_selection", sel); print(json.dumps(sel, indent=1, default=str))
assert SELECTED is not None, "no candidate reached the JSON-validity threshold"
""")

md("### 6.4 H2 on K-QA test (selected model; three methods)")
code(r"""
test_rows = []
with managed(SELECTED) as mc:
    test_rows += run_extraction(mc, kqa_test, ["atomic", "qualified"], "test")
test_rows += run_extraction(None, kqa_test, ["sentence"], "test")
test_scores = score(test_rows)
test_summary = summarize(test_scores)
exp.save_table("extraction_test", test_summary); exp.log_result("extraction_test", test_summary.reset_index().to_dict(orient="records"))
display(test_summary)
""")
code(r"""
def boot_diff(a_df, b_df, num, den=None, n=int(ccfg.h2["n_boot"]), seed=SEED):
    # paired bootstrap over answers; ratio of sums when den is given, else mean of per-answer values
    a = a_df.set_index("qid"); b = b_df.set_index("qid").loc[a.index]
    rng = np.random.default_rng(seed); ids = np.arange(len(a))
    def stat(x, i):
        return x[num].to_numpy()[i].sum() / max(1, x[den].to_numpy()[i].sum()) if den else np.nanmean(x[num].to_numpy()[i])
    d0 = stat(a, ids) - stat(b, ids)
    bs = [stat(a, s) - stat(b, s) for s in (rng.choice(ids, len(ids)) for _ in range(n))]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"diff": round(float(d0), 4), "ci_low": round(float(lo), 4), "ci_high": round(float(hi), 4)}
T_ = {m: test_scores[test_scores.method == m] for m in ("sentence", "atomic", "qualified")}
h2 = {"qualifier_preservation": {"qualified_vs_atomic": boot_diff(T_["qualified"], T_["atomic"], "qual_kept", "n_qualifiers"),
                                 "qualified_vs_sentence": boot_diff(T_["qualified"], T_["sentence"], "qual_kept", "n_qualifiers")},
      "precision": {"qualified_vs_atomic": boot_diff(T_["qualified"], T_["atomic"], "precision"),
                    "qualified_vs_sentence": boot_diff(T_["qualified"], T_["sentence"], "precision")},
      "recall": {"qualified_vs_atomic": boot_diff(T_["qualified"], T_["atomic"], "recall"),
                 "qualified_vs_sentence": boot_diff(T_["qualified"], T_["sentence"], "recall")}}
part_a = h2["qualifier_preservation"]["qualified_vs_atomic"]["ci_low"] > 0 and h2["qualifier_preservation"]["qualified_vs_sentence"]["ci_high"] >= 0
part_b = h2["precision"]["qualified_vs_atomic"]["ci_low"] > -ccfg.h2["precision_margin"]
verdict = "SUPPORTED" if part_a and part_b else ("PARTIALLY SUPPORTED" if part_a or part_b else "NOT SUPPORTED")
h2["verdict"] = {"qualifier_part": part_a, "precision_part": part_b, "verdict": verdict}
exp.log_result("h2", h2); print(json.dumps(h2, indent=1))
""")

md(r"""
## 7. Results, visualizations and error analysis
### 7.1 Per-qualifier-type preservation (test)
""")
code(r"""
qrows = []
for r in test_rows:
    it = items_by_id[r["qid"]]; claims = [c.text for c in r["res"].claims]
    for q in detect(it.answer):
        qrows.append({"method": r["method"], "kind": q.kind, "kept": qualifier_kept(q.kind, it.answer[q.start:q.end], claims) if claims else False})
qtype = pd.DataFrame(qrows).groupby(["kind", "method"]).kept.agg(["mean", "size"]).unstack("method").round(3)
exp.save_table("qualifier_by_type_test", qtype); display(qtype)
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
ts = test_summary.reset_index()
ts["label"] = ts.method + ts.model.fillna("").map(lambda m: f" ({m})" if m else "")
ts.set_index("label")[["precision", "recall", "recall_set", "qualifier_preservation"]].plot.bar(ax=axes[0], rot=0)
axes[0].set_ylim(0, 1.05); axes[0].set_title("K-QA test: extraction quality"); axes[0].legend(fontsize=7)
dv = dev_summary.reset_index(); dv = dv[dv.method != "sentence"]
for m, g in dv.groupby("method"):
    axes[1].scatter(g.latency_s, g.f1, label=m, s=60)
    for _, r in g.iterrows():
        axes[1].annotate(r.model, (r.latency_s, r.f1), fontsize=7)
axes[1].set_xlabel("median latency per answer (s)"); axes[1].set_ylabel("F1 (dev)"); axes[1].legend(); axes[1].set_title("Dev: quality vs latency")
plt.tight_layout(); exp.save_figure("extraction_overview", fig); plt.show()
""")

md(r"""
### 7.1b Exploratory (post-hoc): a stricter qualifier check
**Not pre-registered.** After the run, inspecting the pre-registered heuristic showed a ceiling effect: a negation counted as kept when *any* claim had *any* negation cue and shared one word with the scope, and a population phrase at 50% word overlap. For example, "for children under 15 kg **and** for pregnant or nursing individuals" passed with the pregnancy part dropped.

The stricter check (`qualifier_kept_strict`) requires:
- a negation in the **same** claim as ≥ 60% of its scope;
- a population phrase ≥ 80% present in one claim;
- numbers together with their units or comparators in one claim.

It is computed on the same cached outputs (no new LLM calls). The H2 verdict above is unchanged by it.
""")
code(r"""
from cliniver.evaluation.extraction import qualifier_kept_strict
srows = []
for r in test_rows:
    it = items_by_id[r["qid"]]; claims = [c.text for c in r["res"].claims]
    for q in detect(it.answer):
        span = it.answer[q.start:q.end]
        srows.append({"qid": r["qid"], "method": r["method"], "kind": q.kind, "kept_strict": qualifier_kept_strict(q.kind, span, claims) if claims else False})
strict = pd.DataFrame(srows)
strict_tab = strict.groupby(["kind", "method"]).kept_strict.mean().unstack("method").round(3)
strict_tab.loc["ALL"] = strict.groupby("method").kept_strict.mean().round(3)
per_ans = strict.groupby(["method", "qid"]).kept_strict.agg(["sum", "size"]).rename(columns={"sum": "qual_kept", "size": "n_qualifiers"}).reset_index()
P = {m: per_ans[per_ans.method == m] for m in ("sentence", "atomic", "qualified")}
strict_h2 = {"qualified_vs_atomic": boot_diff(P["qualified"], P["atomic"], "qual_kept", "n_qualifiers"),
             "qualified_vs_sentence": boot_diff(P["qualified"], P["sentence"], "qual_kept", "n_qualifiers")}
exp.save_table("qualifier_strict_test_exploratory", strict_tab)
exp.log_result("qualifier_strict_exploratory", {"table": strict_tab.to_dict(), "differences": strict_h2})
display(strict_tab); print(json.dumps(strict_h2, indent=1))
""")

md("### 7.2 Error analysis: dropped qualifiers, unfaithful claims, invalid JSON")
code(r"""
def examples(method, n=4):
    out = []
    for r in test_rows:
        if r["method"] != method: continue
        it = items_by_id[r["qid"]]; claims = [c.text for c in r["res"].claims]
        for q in detect(it.answer):
            span = it.answer[q.start:q.end]
            if claims and not qualifier_kept(q.kind, span, claims):
                out.append({"method": method, "kind": q.kind, "qualifier": span[:80], "closest_claims": " | ".join(claims[:3])[:240]})
                break
        if len(out) >= n: break
    return out
display(pd.DataFrame(examples("atomic") + examples("qualified")))
# unfaithful claims (low entailment from the answer) for the qualified method
low = []
for r in test_rows:
    if r["method"] != "qualified" or not r["res"].claims: continue
    it = items_by_id[r["qid"]]
    ps = judge.entail_prob([it.answer] * len(r["res"].claims), [c.text for c in r["res"].claims])
    for c, p in zip(r["res"].claims, ps):
        if p < ccfg.extraction.entail_threshold:
            low.append({"qid": r["qid"], "p_entail": round(float(p), 3), "claim": c.text[:200]})
low_df = pd.DataFrame(low)
exp.log_result("qualified_unfaithful_claims_test", len(low_df)); display(low_df.head(10))
invalid = [(r["qid"], r["model"], r["res"].error, r["res"].raw[-200:]) for r in ext_rows + test_rows if not r["res"].valid_json]
print("invalid JSON outputs:", len(invalid)); display(pd.DataFrame(invalid, columns=["qid", "model", "error", "tail"]).head(8))
""")

md(r"""
### 7.3 NLI judge spot check (manual)
Up to 40 NLI judgements (stratified: up to 20 judged *entailed*, up to 20 *not entailed*; claim-faithfulness pairs, `qualified` method, test) are inspected manually; when fewer than 20 claims were judged not entailed, all of them are included. **Annotator: the AI developer (Claude), not a human or clinician.** The annotation file is written once from this sample; the notebook reports agreement if it exists, otherwise NOT EVALUATED.
""")
code(r"""
rng = random.Random(SEED)
pairs = []
for r in test_rows:
    if r["method"] != "qualified": continue
    it = items_by_id[r["qid"]]
    for c in r["res"].claims:
        pairs.append((r["qid"], it.answer, c.text))
probs = judge.entail_prob([p[1] for p in pairs], [p[2] for p in pairs])
pos = [i for i, p in enumerate(probs) if p >= ccfg.extraction.entail_threshold]; neg = [i for i, p in enumerate(probs) if p < ccfg.extraction.entail_threshold]
k = ccfg.nli_spot_check["sample"] // 2
sample_idx = sorted(rng.sample(pos, min(k, len(pos))) + rng.sample(neg, min(k, len(neg))))
spot = [{"i": i, "qid": pairs[i][0], "claim": pairs[i][2], "nli_entailed": bool(probs[i] >= ccfg.extraction.entail_threshold),
         "p": round(float(probs[i]), 3)} for i in sample_idx]
spot_file = ANN / ("nb03_nli_spot_check_smoke.json" if SMOKE else "nb03_nli_spot_check.json")
if spot_file.exists():
    ann = {(a["qid"], a["claim"]): a for a in json.loads(spot_file.read_text(encoding="utf-8"))["items"]}
    agree = [ann[(s["qid"], s["claim"])]["entailed"] == s["nli_entailed"] for s in spot if (s["qid"], s["claim"]) in ann]
    agree2 = [(ann[(s["qid"], s["claim"])]["entailed"] and ann[(s["qid"], s["claim"])]["standalone"]) == s["nli_entailed"]
              for s in spot if (s["qid"], s["claim"]) in ann]
    spot_res = {"n": len(agree), "agreement": round(float(np.mean(agree)), 3) if agree else None,
                "agreement_vs_entailed_and_standalone": round(float(np.mean(agree2)), 3) if agree2 else None,
                "manual_entailed_among_nli_entailed": f"{sum(ann[(s['qid'], s['claim'])]['entailed'] for s in spot if s['nli_entailed'] and (s['qid'], s['claim']) in ann)}/{sum(1 for s in spot if s['nli_entailed'])}",
                "manual_entailed_among_nli_not": f"{sum(ann[(s['qid'], s['claim'])]['entailed'] for s in spot if not s['nli_entailed'] and (s['qid'], s['claim']) in ann)}/{sum(1 for s in spot if not s['nli_entailed'])}",
                "nli_entailed_but_manual_not": sum(1 for s in spot if (s["qid"], s["claim"]) in ann and s["nli_entailed"] and not ann[(s["qid"], s["claim"])]["entailed"]),
                "nli_not_but_manual_entailed": sum(1 for s in spot if (s["qid"], s["claim"]) in ann and not s["nli_entailed"] and ann[(s["qid"], s["claim"])]["entailed"])}
else:
    (ANN / ("nb03_nli_spot_check_TODO_smoke.json" if SMOKE else "nb03_nli_spot_check_TODO.json")).write_text(json.dumps([{**s, "answer": items_by_id[s["qid"]].answer} for s in spot], ensure_ascii=False, indent=1), encoding="utf-8")
    spot_res = "NOT EVALUATED (sample written for annotation)"
exp.log_result("nli_spot_check", spot_res); print(spot_res)
""")

md(r"""
## 8. Claims from the generated answers (input to NB04–NB08)
The selected extractor runs the `qualified` method on every Part A answer. Claims keep the full schema and provenance (response id, generator, extractor model and prompt hash).
""")
code(r"""
claims_out, ext_valid = [], []
with managed(SELECTED) as mc:
    for r in responses:
        if not r["answer"]:
            continue
        res = extract_qualified(mc, r["answer"], seed=SEED, tag=f"claims:{r['response_id']}")
        ext_valid.append(res.valid_json)
        for j, c in enumerate(res.claims):
            claims_out.append({"claim_id": f"{r['response_id']}#c{j}", "response_id": r["response_id"], "text": c.text,
                               "source_sentence": c.source_sentence, **c.fields, "generator": r["generator"],
                               "extractor": {"method": "qualified", "model": SELECTED, "model_file": lcfg.models[SELECTED].file,
                                             "runtime": f"llama.cpp {lcfg.runtime.release}"}})
with (OUTR / "claims.jsonl").open("w", encoding="utf-8") as fh:
    for c in claims_out:
        fh.write(json.dumps(c, ensure_ascii=False) + "\n")
cdf = pd.DataFrame(claims_out)
claims_summary = {"responses": len(responses), "claims": len(cdf), "valid_json_rate": round(float(np.mean(ext_valid)), 4),
                  "claims_per_response": cdf.groupby("generator").response_id.value_counts().groupby("generator").mean().round(2).to_dict(),
                  "claim_types": cdf.claim_type.value_counts().to_dict(), "verifiable_share": round(float(cdf.is_verifiable.mean()), 3),
                  "negated_share": round(float(cdf.negated.mean()), 3),
                  "with_dose": int(cdf.dose.notna().sum()), "with_population": int(cdf.population.notna().sum()),
                  "certainty": cdf.certainty.value_counts().to_dict()}
exp.log_result("claims_from_responses", claims_summary); print(json.dumps(claims_summary, indent=1))
display(cdf[["response_id", "text", "claim_type", "drug", "population", "negated", "dose", "certainty", "is_verifiable"]].head(12))
""")
code(r"""
calls = pd.concat([pd.read_json(p, lines=True) for p in LLM_LOG.glob("*.jsonl")], ignore_index=True)
call_summary = calls.groupby("model").agg(calls=("key", "size"), completion_tokens=("completion_tokens", "sum"),
                                          gpu_seconds=("latency_s", "sum"), tokens_per_s_median=("tokens_per_s", "median"))
exp.save_table("llm_call_summary", call_summary); exp.log_result("llm_call_summary", call_summary.to_dict(orient="index"))
display(call_summary)
exp.finish(); print("results:", exp.dir)
""")

sys.path.insert(0, str(Path(__file__).parent))
try:
    from nb03_discussion import DISCUSSION_CELLS
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
