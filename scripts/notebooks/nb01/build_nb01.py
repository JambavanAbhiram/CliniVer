"""Builds notebooks/01_data_acquisition_and_analysis.ipynb (Stage A). Discussion cells are added after execution."""
import sys
from pathlib import Path

import nbformat as nbf

out = Path(sys.argv[1])
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# Notebook 01 — Data Acquisition and Corpus Analysis
**CliniVer** · Stage A (controlled sample, parallel acquisition) · Experiment `EXP-01-001`

| | |
|---|---|
| **Objective** | Build and validate a multi-source acquisition pipeline for the cardiometabolic pharmacotherapy evidence corpus (ADR-0001, ADR-0002), and measure the corpus before scaling |
| **Research question** | RQ0 (partial): *What does an openly licensed, multi-source corpus for this domain look like (size, evidence types, dates, metadata quality, normalization coverage)?* Claim-level coverage is measured later, in NB05/NB08 |
| **Hypothesis** | None. This notebook is descriptive. It produces the measurements that later hypotheses depend on |
| **Stage A scope** | A **controlled sample** to validate parsers, schemas and data quality, plus **population counts** (count-only queries) to size Stage B. Scaling happens only after review |

> **Sampling caveat.** The PubMed sample uses NCBI *Best Match* ordering (top 100 per condition). It is a convenience sample for validating parsers, **not** a representative sample. Population-level statements in this notebook use the count-only queries, which cover the full query results.
""")

md(r"""
## 1. Background and methodology

### 1.1 Sources, roles and terms (verified 2026-11-19)
| Source | Role | Access | Terms / limits used |
|---|---|---|---|
| PubMed (E-utilities) | evidence | keyless | ≤ 3 req/s without an API key, counted on arrival. Now 2 req/s with ≤ 2 in flight (the first parallel run used 2.5 req/s and got HTTP 429s; see EXP-01-002); `tool=cliniver`; abstracts may be under publisher copyright, so they are stored locally and **not redistributed** |
| PubTator 3 | graph input (annotations) | keyless | ≤ 3 req/s used |
| Europe PMC OA | evidence (full text) | keyless | Only CC-licensed OA articles; license stored per document |
| openFDA drug labels | regulatory evidence | keyless | CC0 (open.fda.gov/license); ~1,000 req/day keyless |
| ClinicalTrials.gov v2 | trial registry | keyless | No official limit published; paced at ~46 req/min |
| RxNorm / RxClass | normalization (not evidence) | keyless | ≤ 20 req/s per NLM; drug–drug interaction API **discontinued 2024-01-02** |
| PubChem | normalization (not evidence) | keyless | ≤ 5 req/s |
| WHO GHO (OData) | population statistics | keyless | CC BY-NC-SA 3.0 IGO; legacy endpoint live on 2026-11-19 despite announced rework |

**NCBI API key:** not available to the project owner, so PubMed runs at the keyless rate. This affects only speed (≤ 200 records per efetch request), not the data.

### 1.2 Pipeline
`configs/data.yaml` → source clients (`cliniver.ingestion.*`; rate-limited, retrying) → **raw archive** (`data/raw/<source>/<snapshot>/`, unchanged bytes + metadata sidecar with SHA-256) → parsers → unified `Document` / `Concept` / PubTator annotation records (`cliniver.schemas`) → `data/interim/<snapshot>/stageA/*.jsonl` → quality analysis (`cliniver.data.quality`).

Re-running this notebook re-uses the archived raw responses (no new API calls), so results are reproducible for a given snapshot.
""")

md("## 2. Environment and configuration")
code(r"""
import json, time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timezone

import matplotlib.pyplot as plt
import pandas as pd

from cliniver.config import load_project_config
from cliniver.config.data_config import load_data_config
from cliniver.data.corpus_io import write_jsonl
from cliniver.data.quality import completeness, docs_to_frame, merge_duplicate_ids, near_duplicate_clusters, near_duplicate_pairs
from cliniver.data.report import composition, evidence_stratum, pubtator_coverage, quality_flags
from cliniver.ingestion.clinicaltrials import ClinicalTrialsClient, parse_study
from cliniver.ingestion.europepmc import EuropePMCClient, parse_jats
from cliniver.ingestion.http import FetchError, RawStore, parallel_map
from cliniver.ingestion.normalization import PubChemClient, RxNavClient
from cliniver.ingestion.openfda import OpenFDAClient, parse_label
from cliniver.ingestion.pubmed import PubMedClient, parse_pubmed_xml
from cliniver.ingestion.pubtator import PubTatorClient, parse_bioc
from cliniver.ingestion.who_gho import WhoGhoClient, textualize
from cliniver.utils import collect_hardware_info, set_seed, setup_logging
from cliniver.utils.experiment import Experiment

pcfg = load_project_config()
dcfg = load_data_config()
setup_logging(pcfg.logging.level, pcfg.paths.absolute("logs"), run_name="nb01")
SEED = set_seed(pcfg.reproducibility.seed)
SNAP = dcfg.snapshot
RAW = pcfg.paths.absolute("data_raw")
OUT = pcfg.paths.absolute("data_interim") / SNAP / "stageA"
OUT.mkdir(parents=True, exist_ok=True)

exp = Experiment(
    exp_id="EXP-01-001", title="Stage A multi-source acquisition and corpus quality analysis",
    research_question="RQ0 (partial): corpus composition, quality and normalization coverage",
    config=dcfg.model_dump(mode="json"), seed=SEED,
    packages=["torch", "pandas", "requests", "pydantic", "matplotlib", "cliniver"])
store = lambda src: RawStore(RAW, src, SNAP)
pd.set_option("display.width", 160); pd.set_option("display.max_columns", 30)
print("snapshot:", SNAP, "| stage:", dcfg.stage, "| seed:", SEED)
print("hardware:", {k: v for k, v in collect_hardware_info().to_dict().items() if k in ("python_version", "platform", "ram_total_gb")})
""")

md(r"""
## 3. Acquisition (parallel)

**Concurrency design.**
- **Across providers:** each source runs as an independent job, up to `concurrency.sources_in_parallel` at once. Providers have independent rate limits, so they never wait on each other.
- **Within a provider:** up to `workers_per_source` requests can be in flight at once. Request **start times** are still spaced by that provider's `min_interval_s` (a thread-safe limiter shared by every thread using the client), so published limits are never exceeded. Only network latency overlaps.
- **Shared limits:** PubMed population counts and the PubMed sample are separate jobs that share one PubMed client, so their *combined* rate stays at 2.5 req/s.

Dependency: PubTator needs the PubMed sample PMIDs, so the two run in sequence inside one job.

**Population counts** (part of the `pubmed_counts` job) download **no records**. They return NCBI's hit count per query, which gives an unbiased picture of the *full* literature per condition, by publication type and by decade. Stage B is sized from these counts.
""")
code(r"""
W = dcfg.concurrency.workers_per_source
pm = PubMedClient(dcfg.pubmed, store("pubmed"), W)
pt = PubTatorClient(dcfg.pubtator, store("pubtator"), W)
fda = OpenFDAClient(dcfg.openfda, store("openfda"), W)
ct = ClinicalTrialsClient(dcfg.clinicaltrials, store("clinicaltrials"), W)
ep = EuropePMCClient(dcfg.europepmc, store("europepmc"), W)
rx, pc = RxNavClient(dcfg.rxnav, store("rxnav"), W), PubChemClient(dcfg.pubchem, store("pubchem"), W)
who = WhoGhoClient(dcfg.who_gho, store("who_gho"), W)
CONDS = list(dcfg.pubmed.conditions)


def acquire_pubmed_counts():
    qs = []
    for c in CONDS:
        qs.append(("total", c, None, (pm.query(c), f"{c}_total")))
        qs += [("type", c, t, (pm.query(c, f"AND {q}"), f"{c}_type_{t}")) for t, q in dcfg.pubmed.publication_type_strata.items()]
        qs += [("decade", c, f"{a}-{b}", (pm.query(c, f'AND ("{a}/01/01"[dp] : "{b}/12/31"[dp])'), f"{c}_dec_{a}"))
               for a, b in dcfg.pubmed.decades]
    out = {"total": {}, "by_type": defaultdict(dict), "by_decade": defaultdict(dict)}
    for (kind, c, k, _), v in zip(qs, pm.count_many([q[3] for q in qs])):
        if kind == "total": out["total"][c] = v
        else: out["by_type" if kind == "type" else "by_decade"][c][k] = v
    return out


def acquire_pubmed_sample_and_pubtator():
    ids = dict(zip(CONDS, parallel_map(lambda c: pm.search_ids(pm.query(c), dcfg.pubmed.sample_per_condition, f"{c}_sample"), CONDS, W)))
    xml_per_cond = parallel_map(lambda c: pm.fetch_xml(ids[c], f"{c}_sample"), CONDS, W)
    docs = [d for c, xmls in zip(CONDS, xml_per_cond) for x in xmls for d in parse_pubmed_xml(x, snapshot=SNAP, condition=c)]
    docs, dup = merge_duplicate_ids(docs)
    pmids = sorted({d.source_id for d in docs})
    bioc = pt.export(pmids, "stageA")
    ents, rels = [], []
    for b in bioc:
        e, r = parse_bioc(b); ents += e; rels += r
    return {"ids": ids, "docs": docs, "dup": dup, "pmids": pmids, "bioc": bioc, "entities": ents, "relations": rels}


def acquire_openfda():
    pairs = [(c, d) for c, ds in dcfg.drugs.items() for d in ds[: dcfg.openfda.sample_drugs_per_condition]]
    labels = parallel_map(lambda cd: fda.labels_for(cd[1], dcfg.openfda.labels_per_drug), pairs, W)
    docs = [parse_label(l, sections=dcfg.openfda.sections, snapshot=SNAP, query_drug=d, condition=c)
            for (c, d), ls in zip(pairs, labels) for l in ls]
    return {"docs": docs, "missing": [d for (c, d), ls in zip(pairs, labels) if not ls]}


def acquire_clinicaltrials():
    res = parallel_map(lambda c: ct.search(c, dcfg.clinicaltrials.sample_per_condition), CONDS, W)
    docs = [parse_study(s, snapshot=SNAP, condition=c) for c, (studies, _) in zip(CONDS, res) for s in studies]
    docs, dup = merge_duplicate_ids(docs)
    return {"docs": docs, "dup": dup, "totals": {c: t for c, (_, t) in zip(CONDS, res)}}


def acquire_europepmc():
    res = parallel_map(lambda c: ep.search_oa(c, dcfg.europepmc.sample_per_condition), CONDS, W)
    metas = [(c, m) for c, (results, _) in zip(CONDS, res) for m in results]

    def one(cm):
        c, meta = cm
        try:
            return parse_jats(ep.full_text_xml(meta["pmcid"]), meta, snapshot=SNAP, condition=c), None
        except (FetchError, KeyError) as exc:
            return None, (meta.get("pmcid"), str(exc)[:120])

    out = parallel_map(one, metas, W)
    return {"docs": [d for d, _ in out if d], "fail": [e for _, e in out if e], "hits": {c: h for c, (_, h) in zip(CONDS, res)}}


def acquire_normalization():
    def one(cd):
        c, drug = cd
        return c, drug, rx.concepts_for(drug, snapshot=SNAP), pc.concept_for(drug, snapshot=SNAP)
    return parallel_map(one, dcfg.all_drugs(), W)


def acquire_who():
    found = parallel_map(who.find_indicators, dcfg.who_gho.indicator_keywords, W)
    indicators = {i["IndicatorCode"]: i["IndicatorName"] for lst in found for i in lst}
    # Stage A: values only for age-standardized prevalence indicators, to validate the textualizer
    selected = {k: v for k, v in indicators.items() if "age-standardized" in v.lower()}
    rows = parallel_map(who.values, list(selected), W)
    docs = [textualize(r, name, snapshot=SNAP) for (code_, name), rs in zip(selected.items(), rows) for r in rs]
    return {"indicators": indicators, "selected": selected, "docs": docs}


JOBS = {"pubmed_counts": acquire_pubmed_counts, "pubmed_sample+pubtator": acquire_pubmed_sample_and_pubtator,
        "openfda": acquire_openfda, "clinicaltrials": acquire_clinicaltrials, "europepmc_oa": acquire_europepmc,
        "rxnav+pubchem": acquire_normalization, "who_gho": acquire_who}


def _timed(name, fn):
    t0 = time.perf_counter(); res = fn(); return name, res, time.perf_counter() - t0


t_all = time.perf_counter()
R, job_seconds = {}, {}
with ThreadPoolExecutor(max_workers=dcfg.concurrency.sources_in_parallel) as pool:
    for fut in as_completed([pool.submit(_timed, n, f) for n, f in JOBS.items()]):
        name, res, dt = fut.result()  # re-raises any job's exception: failures are never silent
        R[name], job_seconds[name] = res, dt
wall = time.perf_counter() - t_all
print(f"parallel acquisition wall time: {wall:.1f}s | sum of job times: {sum(job_seconds.values()):.1f}s")
""")

md(r"""
### 3.1 Acquisition timing and comparison with the sequential baseline
Both timings are **measured from the raw archive**: the span between the first and last `retrieved_at` timestamps written by `RawStore` (1-second resolution).
- Snapshot `2026-11-19`: the earlier fully sequential implementation.
- The current snapshot: this parallel run.

Both runs fetched fresh from the APIs (no cache hits), so the comparison is like-for-like. Responses per source can differ slightly between days; request counts are shown alongside.
""")
code(r"""
def archive_span(snapshot):
    stamps, n = [], Counter()
    for meta in RAW.glob(f"*/{snapshot}/*.meta.json"):
        m = json.loads(meta.read_text(encoding="utf-8"))
        stamps.append(datetime.fromisoformat(m["retrieved_at"])); n[meta.parent.parent.name] += 1
    return ((max(stamps) - min(stamps)).total_seconds() if stamps else None), sum(n.values())

clients = {"pubmed": pm, "pubtator": pt, "openfda": fda, "clinicaltrials": ct, "europepmc": ep, "rxnav": rx, "pubchem": pc, "who_gho": who}
timing = pd.DataFrame({"seconds": pd.Series(job_seconds).round(1)}).sort_values("seconds", ascending=False)
req = pd.DataFrame({k: {"requests": c.http.n_requests, "retries": c.http.n_retries, "cache_hits": c.http.n_cache_hits,
                        "min_interval_s": c.http.min_interval_s,
                        "floor_s_if_rate_bound": round(c.http.n_requests * c.http.min_interval_s, 1)} for k, c in clients.items()}).T
seq_span, seq_files = archive_span("2026-11-19")
par_span, par_files = archive_span(SNAP)
cmp_ = {"sequential_snapshot": "2026-11-19", "sequential_span_s": seq_span, "sequential_raw_files": seq_files,
        "parallel_snapshot": SNAP, "parallel_span_s": par_span, "parallel_raw_files": par_files,
        "parallel_wall_s_in_notebook": round(wall, 1),
        "speedup_span_ratio": round(seq_span / par_span, 2) if seq_span and par_span else None}
REPLAY = int(req["requests"].sum()) == 0  # every response came from the raw archive
cmp_["note"] = ("REPLAY of the raw archive: job seconds below measure parsing only, not acquisition. Live acquisition timing: "
               "artifacts/experiments/EXP-01-001/live_acquisition_run_2026-11-20_*.") if REPLAY else "live acquisition run"
exp.log_result("replay_timing" if REPLAY else "acquisition_timing", {"jobs_s": job_seconds, "wall_s": wall, "requests": req.to_dict(orient="index")})
exp.log_result("acquisition_span_comparison", cmp_)
exp.save_table("acquisition_timing_jobs", timing); exp.save_table("acquisition_requests", req)
print(json.dumps(cmp_, indent=2))
display(timing); display(req)
""")
code(r"""
# Unpack job results for the analysis sections
pop = R["pubmed_counts"]
pop_total = pop["total"]
pop_type_df = pd.DataFrame(pop["by_type"]).T.loc[CONDS]
pop_type_df.insert(0, "TOTAL", pd.Series(pop_total))
pop_decade_df = pd.DataFrame(pop["by_decade"]).T.loc[CONDS]
exp.save_table("pubmed_population_by_type", pop_type_df); exp.save_table("pubmed_population_by_decade", pop_decade_df)
exp.log_result("pubmed_population_total_per_condition", pop_total)

S = R["pubmed_sample+pubtator"]
pm_ids, pm_docs, pm_dup_ids, all_pmids, bioc, entities, relations = (S[k] for k in ("ids", "docs", "dup", "pmids", "bioc", "entities", "relations"))
annotated = {e.pmid for e in entities}
fda_docs, fda_missing = R["openfda"]["docs"], R["openfda"]["missing"]
ct_docs, ct_dup, ct_totals = R["clinicaltrials"]["docs"], R["clinicaltrials"]["dup"], R["clinicaltrials"]["totals"]
ep_docs, ep_fail, ep_hits = R["europepmc_oa"]["docs"], R["europepmc_oa"]["fail"], R["europepmc_oa"]["hits"]
indicators, selected, who_docs = R["who_gho"]["indicators"], R["who_gho"]["selected"], R["who_gho"]["docs"]
exp.log_result("clinicaltrials_population_completed_drug_trials", ct_totals)
exp.log_result("europepmc_oa_population_hits", ep_hits)
print("Totals are per condition; documents can belong to more than one condition.")
pop_type_df
""")
code(r"""
pop_decade_df
""")

md("## 4. Per-source acquisition results")
md("### 4.1 PubMed sample (top 100 per condition, Best Match) and 4.2 PubTator 3 annotations")
code(r"""
requested = sum(len(v) for v in pm_ids.values())
print(f"requested IDs: {requested} | unique parsed docs: {len(pm_docs)} | same PMID under >1 condition: {pm_dup_ids}")
print(f"PMIDs sent to PubTator: {len(all_pmids)} | returned: {len(bioc)} | with >=1 entity: {len(annotated)}")
print(f"entity mentions: {len(entities)} | relations: {len(relations)}")
""")
md("### 4.3 openFDA drug labels (2 drugs per condition × up to 3 labels)")
code(r"""
print(f"labels: {len(fda_docs)} | drugs with no exact-name label: {fda_missing}")
""")
md("### 4.4 ClinicalTrials.gov (completed, drug-intervention trials; 20 per condition)")
code(r"""
print("population (completed drug trials):", ct_totals)
print(f"sample docs: {len(ct_docs)} (cross-condition duplicates merged: {ct_dup})")
""")
md("### 4.5 Europe PMC open-access full text (license-filtered; 2 per condition)")
code(r"""
print("OA full-text population hits:", ep_hits)
print(f"full texts parsed: {len(ep_docs)} | failures: {ep_fail}")
""")
md("### 4.6 Drug normalization: RxNorm / RxClass and PubChem (all 49 in-scope drugs)")
code(r"""
concepts, norm_rows = [], []
for c, drug, rxc, pcc in R["rxnav+pubchem"]:
    concepts += rxc + ([pcc] if pcc else [])
    drug_c = next((x for x in rxc if x.kind == "drug"), None)
    norm_rows.append({"condition": c, "drug": drug, "rxcui": drug_c.concept_id if drug_c else None,
                      "n_brands": len(drug_c.synonyms) if drug_c else 0,
                      "n_classes": sum(x.kind == "drug_class" for x in rxc),
                      "pubchem_cid": pcc.concept_id if pcc else None,
                      "n_pubchem_synonyms_total": pcc.extra["n_synonyms_total"] if pcc else 0})
norm_df = pd.DataFrame(norm_rows)
# Classes repeat across drugs: keep one record per concept_id with the union of member drugs
uniq = {}
for x in concepts:
    if x.concept_id in uniq and x.kind == "drug_class":
        prev = uniq[x.concept_id]
        members = set(prev.extra.get("members", [prev.extra["member_rxcui"]])) | {x.extra["member_rxcui"]}
        uniq[x.concept_id] = prev.model_copy(update={"extra": {**prev.extra, "members": sorted(members)}})
    else:
        uniq.setdefault(x.concept_id, x)
concepts = list(uniq.values())
exp.save_table("normalization_coverage", norm_df)
print(f"concept records: {len(concepts)}  ({Counter(x.kind for x in concepts)})")
norm_df
""")
md("### 4.7 WHO GHO population statistics (indicator discovery + global/regional values)")
code(r"""
print(f"{len(indicators)} indicators matched keywords | selected (age-standardized): {len(selected)} | statistic documents: {len(who_docs)}")
pd.Series(selected, name="indicator").to_frame()
""")

md("## 5. Persist normalized records")
code(r"""
all_docs = pm_docs + fda_docs + ct_docs + ep_docs + who_docs
counts = {
    "documents": write_jsonl(all_docs, OUT / "documents.jsonl"),
    "concepts": write_jsonl(concepts, OUT / "concepts.jsonl"),
    "pubtator_entities": write_jsonl(entities, OUT / "pubtator_entities.jsonl"),
    "pubtator_relations": write_jsonl(relations, OUT / "pubtator_relations.jsonl"),
}
manifest = {"snapshot": SNAP, "stage": "A", "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "counts": counts, "by_source": dict(Counter(d.source for d in all_docs))}
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
exp.log_result("record_counts", manifest)
manifest
""")

md(r"""
## 6. Evaluation: corpus quality

### 6.1 Composition and text length per source
""")
code(r"""
df = docs_to_frame(all_docs)
comp = composition(df)
exp.save_table("composition", comp); exp.log_result("composition", comp.to_dict(orient="index"))
comp
""")
code(r"""
fig, ax = plt.subplots(figsize=(8, 3.5))
order = comp.index.tolist()
ax.boxplot([df.loc[df.source == s, "n_words"].clip(lower=1) for s in order], tick_labels=order, showfliers=True)
ax.set_yscale("log"); ax.set_ylabel("words per document (log)"); ax.set_title("Document length by source (Stage A sample)")
plt.xticks(rotation=20); exp.save_figure("length_by_source", fig); plt.show()
""")

md("### 6.2 Metadata completeness (fraction of documents with each field)")
code(r"""
compl = completeness(all_docs)
exp.save_table("completeness", compl); exp.log_result("completeness", compl.round(3).to_dict(orient="index"))
fig, ax = plt.subplots(figsize=(9, 3))
im = ax.imshow(compl.values, vmin=0, vmax=1, cmap="viridis", aspect="auto")
ax.set_xticks(range(len(compl.columns)), compl.columns, rotation=40, ha="right"); ax.set_yticks(range(len(compl.index)), compl.index)
for i in range(compl.shape[0]):
    for j in range(compl.shape[1]):
        ax.text(j, i, f"{compl.values[i, j]:.2f}", ha="center", va="center", color="white" if compl.values[i, j] < .6 else "black", fontsize=8)
fig.colorbar(im, ax=ax, fraction=.02); ax.set_title("Field completeness by source")
exp.save_figure("completeness", fig); plt.show()
compl.round(3)
""")

md("### 6.3 Duplicates: exact, cross-source and near-duplicate")
code(r"""
pm_pmids = {d.source_id for d in pm_docs}
cross = [d.doc_id for d in ep_docs if d.identifiers.get("pmid") in pm_pmids]
dois = [d.identifiers.get("doi") for d in pm_docs if d.identifiers.get("doi")]
label_nd = near_duplicate_pairs(fda_docs, group_key=lambda d: d.extra["query_drug"], threshold=0.8)
dup = {"pubmed_same_pmid_multiple_conditions": pm_dup_ids, "clinicaltrials_same_nct_multiple_conditions": ct_dup,
       "pubmed_duplicate_dois": len(dois) - len(set(dois)), "europepmc_oa_also_in_pubmed_sample": len(cross),
       "openfda_label_pairs_same_drug": len(label_nd), "openfda_near_dup_pairs_jaccard>=0.8": int(label_nd.near_dup.sum()) if len(label_nd) else 0}
exp.log_result("duplicates", dup); exp.save_table("openfda_label_similarity", label_nd)
print(json.dumps(dup, indent=2))
label_nd.groupby("group").jaccard.describe()[["count", "mean", "min", "max"]].round(3) if len(label_nd) else "no label pairs"
""")

md("### 6.4 Evidence types: PubMed sample vs. population")
code(r"""
stratum = evidence_stratum  # shared definition (cliniver.data.report)
pm_df = df[df.source == "pubmed"].copy()
pm_df["stratum"] = pm_df.publication_types.map(stratum)
sample_share = pm_df.stratum.value_counts(normalize=True).rename("sample_share")
pop_sum = pop_type_df.drop(columns="TOTAL").sum()
pop_share = (pop_sum / pop_type_df["TOTAL"].sum()).rename("population_share_upper_bound")
evid = pd.concat([sample_share, pop_share], axis=1).fillna(0).round(3)
exp.save_table("evidence_type_sample_vs_population", evid)
print("Population shares: strata may overlap (e.g. an RCT also tagged Review), so the shares are upper bounds and need not sum to 1.")
evid
""")

md("### 6.5 Temporal distribution")
code(r"""
fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
pop_decade_df.T.plot(kind="bar", ax=axes[0]); axes[0].set_title("PubMed population by decade (count queries)"); axes[0].set_ylabel("articles")
for s, g in df[df.year.notna()].groupby("source"):
    axes[1].hist(g.year, bins=range(1990, 2028), alpha=.55, label=s)
axes[1].set_title("Stage A sample: document year by source"); axes[1].legend(fontsize=7)
plt.tight_layout(); exp.save_figure("temporal", fig); plt.show()
yr = df.groupby("source").year.describe()[["count", "min", "50%", "max"]]
exp.save_table("year_by_source", yr); yr
""")

md("### 6.6 Document-quality flags")
code(r"""
q = quality_flags(all_docs)
exp.log_result("quality_flags", q); pd.Series(q, name="value").to_frame()
""")

md(r"""
### 6.7 Coverage

**(a) Drug normalization coverage.** Can every in-scope drug be linked to RxNorm, RxClass and PubChem?
""")
code(r"""
cov = {"drugs": len(norm_df), "rxnorm_found": int(norm_df.rxcui.notna().sum()), "with_rxclass": int((norm_df.n_classes > 0).sum()),
       "pubchem_found": int(norm_df.pubchem_cid.notna().sum()),
       "not_in_pubchem": norm_df.loc[norm_df.pubchem_cid.isna(), "drug"].tolist(),
       "not_in_rxnorm": norm_df.loc[norm_df.rxcui.isna(), "drug"].tolist()}
exp.log_result("normalization_coverage", cov); print(json.dumps(cov, indent=2))
""")
md(r"""
**(b) PubTator entity coverage of the PubMed sample.** This tests whether graph construction (ADR-0003) can rely on precomputed annotations: what share of documents carry at least one Chemical and one Disease annotation, and how many mention an in-scope drug (exact name match against generic and RxNorm brand names; a lower bound)?
""")
code(r"""
drug_names = {d.lower() for _, d in dcfg.all_drugs()} | {s.lower() for x in concepts if x.source == "rxnorm" for s in x.synonyms}
pt_cov = pubtator_coverage(entities, relations, pm_docs, drug_names)
exp.log_result("pubtator_coverage", pt_cov); print(json.dumps(pt_cov, indent=2))
""")

md(r"""
## 7. Stage B sizing (projection from measured counts)
This is a projection, not a measurement. It assumes the measured Stage A request behaviour (≤ 200 records per efetch at 2.5 req/s) and the measured median document size.
""")
code(r"""
union_upper = int(pop_type_df["TOTAL"].sum())  # conditions overlap, so the union is smaller
pm_bytes_per_doc = (OUT / "documents.jsonl").stat().st_size / max(len(all_docs), 1)
proj = {
    "pubmed_population_sum_over_conditions_upper_bound": union_upper,
    "efetch_requests_needed": -(-union_upper // dcfg.pubmed.efetch_batch),
    "est_minutes_pubmed_at_configured_rate": round((union_upper / dcfg.pubmed.efetch_batch + union_upper / 9999) * dcfg.pubmed.min_interval_s / 60, 1),
    "est_minutes_pubtator_at_configured_rate": round(union_upper / dcfg.pubtator.batch * dcfg.pubtator.min_interval_s / 60, 1),
    "pubtator_requests_needed": -(-union_upper // dcfg.pubtator.batch),
    "est_normalized_jsonl_mb": round(union_upper * pm_bytes_per_doc / 1e6, 1),
}
exp.log_result("stageB_projection", proj); print(json.dumps(proj, indent=2))
""")

md("## 8. Error analysis: failures and edge cases encountered")
code(r"""
errors = {"openfda_no_label_for_name": fda_missing, "europepmc_fulltext_failures": ep_fail,
          "pubmed_ids_not_parsed": sorted(set(sum(pm_ids.values(), [])) - {d.source_id for d in pm_docs}),
          "pubtator_pmids_missing": sorted(set(all_pmids) - annotated),
          "http": {name: {"requests": c.http.n_requests, "retries": c.http.n_retries}
                   for name, c in [("pubmed", pm), ("pubtator", pt), ("openfda", fda), ("clinicaltrials", ct),
                                   ("europepmc", ep), ("rxnav", rx), ("pubchem", pc), ("who_gho", who)]}}
exp.log_result("errors", errors); print(json.dumps(errors, indent=2))
""")
code(r"""
# Inspect one record per source so the schema can be checked by eye
for src in ["pubmed", "openfda_label", "clinicaltrials", "europepmc_oa", "who_gho"]:
    d = next(x for x in all_docs if x.source == src)
    print(f"--- {src}: {d.doc_id} | {d.pub_date} ({d.date_precision}) | license: {d.license}")
    print("title:", d.title[:110]); print("sections:", [s.name for s in d.sections][:8]); print("text:", d.text[:240].replace("\n", " "), "\n")
""")
code(r"""
exp.finish()
print("results:", exp.dir)
""")

sys.path.insert(0, str(Path(__file__).parent))
from nb01_discussion import DISCUSSION_CELLS
for t in DISCUSSION_CELLS:
    md(t)

exec(Path(__file__).with_name('stage_b_section.py').read_text(encoding='utf-8'))
exec(Path(__file__).with_name('stage_b_discussion.py').read_text(encoding='utf-8'))

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"name": "cliniver", "display_name": "CliniVer (Python 3.12)", "language": "python"},
    "language_info": {"name": "python"}})
out.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, out)
print("wrote", out, len(cells), "cells")
