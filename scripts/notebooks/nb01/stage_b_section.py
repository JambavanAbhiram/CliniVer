md(r"""
---
# Part B — Stage B: full corpus (Experiment `EXP-01-003`)

**Decisions adopted.** On 2026-11-20 the project owner delegated the Stage B decisions. The proposals from §11 were adopted unchanged, plus one safety rule (recorded in `docs/decisions/ADR-0008-corpus-composition.md`):

| ID | Decision as implemented |
|---|---|
| D-NB01-1 | PubMed: **full** population of the five condition queries, retrieved one publication year at a time (esearch returns ≤ 10k IDs per query; each slice is checked against its reported count) |
| D-NB01-2 | PubTator 3 annotations for every acquired PMID (parsed per batch) |
| D-NB01-3 | openFDA: all 49 drugs, **all** prescription labels (paged; the planned 100-label cap would have truncated 8 drugs, found in the smoke test); near-duplicate clusters collapsed to their most recent label |
| D-NB01-4 | ClinicalTrials.gov: all completed drug-intervention trials (paged; checked against `totalCount`) with trial → publication links |
| D-NB01-5 | Europe PMC OA: year-stratified long-document subset (5 conditions × 4 periods × 20) |
| D-NB01-6 | WHO GHO statements reused from Stage A (same snapshot); retrieval role decided in NB05 |
| D-NB01-7 | MinHash/LSH near-duplicate detection (128 permutations, 32 bands) with an exact Jaccard check |
| D-NB01-8 | RxNorm/RxClass is the normalization backbone; PubChem supplementary (Stage A concept records reused) |
| **Safety** | **Retracted publications are excluded from the evidence corpus** (listed, never used as evidence) |

Stage B raw responses go to `data/raw/<source>/<snapshot>/stageB/`, kept separate from the Stage A archive.
""")
code(r"""
import os
# SMOKE mode (env CLINIVER_NB01_SMOKE=1): a small end-to-end check of this Part B code before the long live
# run. Everything goes to separate *_smoke locations and a separate experiment id, so smoke data can never
# mix with the real Stage B archive (batch cache names would otherwise collide).
SMOKE = os.environ.get("CLINIVER_NB01_SMOKE") == "1"
cfgB = dcfg.stage_b if not SMOKE else dcfg.stage_b.model_copy(update={
    "pubmed_years": (2025, 2026), "openfda_max_labels_per_drug": 5, "clinicaltrials_page_size": 20,
    "europepmc_per_stratum": 1})
TAG = "stageB_smoke" if SMOKE else "stageB"
expB = Experiment(
    exp_id="EXP-01-003" + ("-smoke" if SMOKE else ""), title="Stage B full-corpus acquisition, deduplication and quality analysis",
    research_question="RQ0: composition, quality, duplication and normalization coverage of the full evidence corpus",
    config=dcfg.model_dump(mode="json"), seed=SEED,
    packages=["torch", "numpy", "pandas", "requests", "pydantic", "matplotlib", "cliniver"])
storeB = lambda src: RawStore(RAW, src, f"{SNAP}/{TAG}")
OUTB = pcfg.paths.absolute("data_interim") / SNAP / TAG; OUTB.mkdir(parents=True, exist_ok=True)
PROC = pcfg.paths.absolute("data_processed") / (SNAP + ("_smoke" if SMOKE else "")); PROC.mkdir(parents=True, exist_ok=True)
pmB = PubMedClient(dcfg.pubmed, storeB("pubmed"), W)
ptB = PubTatorClient(dcfg.pubtator, storeB("pubtator"), W)
fdaB = OpenFDAClient(dcfg.openfda, storeB("openfda"), W)
ctB = ClinicalTrialsClient(dcfg.clinicaltrials, storeB("clinicaltrials"), W)
epB = EuropePMCClient(dcfg.europepmc, storeB("europepmc"), W)
YEARS = range(cfgB.pubmed_years[0], cfgB.pubmed_years[1] + 1)
print("MODE:", "SMOKE (not a research result)" if SMOKE else "FULL", "| PubMed years:", YEARS.start, "-", YEARS.stop - 1, "| in-flight caps:",
      {n: c.http.max_in_flight for n, c in [("pubmed", pmB), ("pubtator", ptB), ("openfda", fdaB), ("clinicaltrials", ctB), ("europepmc", epB)]})
""")

md(r"""
## B1. Parallel acquisition
The PMID list is needed by both efetch and PubTator, so those two jobs wait on the `pubmed_ids` job and then run **concurrently** with each other. All other jobs start immediately.
""")
code(r"""
def b_pubmed_ids():
    per_cond = dict(zip(CONDS, parallel_map(lambda c: pmB.all_ids_by_year(c, YEARS, "B"), CONDS, len(CONDS))))
    membership = defaultdict(set)
    for c, by_year in per_cond.items():
        for ids in by_year.values():
            for pid in ids:
                membership[pid].add(c)
    slice_max = max(len(ids) for by_year in per_cond.values() for ids in by_year.values())
    # [dp] matches both electronic and print publication dates, so an article can appear in two yearly
    # slices: count UNIQUE PMIDs per condition (the sum over slices double-counts those articles).
    per_cond_n = {c: len({p for v in by_year.values() for p in v}) for c, by_year in per_cond.items()}
    in_two = {c: sum(len(v) for v in by_year.values()) - per_cond_n[c] for c, by_year in per_cond.items()}
    return {"pmids": sorted(membership, key=int), "membership": membership, "per_condition": per_cond_n,
            "in_more_than_one_year_slice": in_two, "largest_year_slice": slice_max}


def b_pubmed_docs(f_ids):
    ids = f_ids.result()[1]  # _timed future → (name, result, seconds)
    pmids, B = ids["pmids"], dcfg.pubmed.efetch_batch
    batches = [pmids[i:i + B] for i in range(0, len(pmids), B)]
    def one(ix):
        return [d.model_copy(update={"conditions": sorted(ids["membership"][d.source_id])})
                for d in parse_pubmed_xml(pmB.fetch_xml(batches[ix], f"B_full_{ix:05d}")[0], snapshot=SNAP)]
    return [d for part in parallel_map(one, range(len(batches)), pmB.workers) for d in part]


def b_pubtator(f_ids):
    return ptB.export_parsed(f_ids.result()[1]["pmids"], "B_full")


def b_openfda():
    pairs = dcfg.all_drugs()
    res = parallel_map(lambda cd: fdaB.search_all(cd[1], cfgB.openfda_max_labels_per_drug, rx_only=cfgB.openfda_rx_only,
                                                  cache_prefix="labelB", page_size=min(100, cfgB.openfda_max_labels_per_drug)),
                       pairs, W)
    docs = [parse_label(l, sections=dcfg.openfda.sections, snapshot=SNAP, query_drug=d, condition=c)
            for (c, d), (labels, _) in zip(pairs, res) for l in labels]
    totals = {d: t for (_, d), (_, t) in zip(pairs, res)}
    # A combination label (e.g. sitagliptin + metformin) is returned once per component query drug:
    # merge by doc_id, keeping every query drug and condition (56 duplicates were missed before 2026-11-20).
    qd = defaultdict(set)
    for d in docs:
        qd[d.doc_id].add(d.extra["query_drug"])
    n_fetched = len(docs)
    docs, dup = merge_duplicate_ids(docs)
    docs = [d.model_copy(update={"extra": {**d.extra, "query_drug": min(qd[d.doc_id]), "query_drugs": sorted(qd[d.doc_id])}})
            for d in docs]
    return {"docs": docs, "totals": totals, "fetched": n_fetched, "dup": dup}


def b_clinicaltrials():
    search = (lambda c: ctB.search(c, cfgB.clinicaltrials_page_size)) if SMOKE else \
             (lambda c: ctB.search_all(c, cfgB.clinicaltrials_page_size, "studiesB"))
    res = parallel_map(search, CONDS, len(CONDS))
    docs = [parse_study(s, snapshot=SNAP, condition=c) for c, (studies, _) in zip(CONDS, res) for s in studies]
    docs, dup = merge_duplicate_ids(docs)
    return {"docs": docs, "dup": dup, "totals": {c: t for c, (_, t) in zip(CONDS, res)}}


def b_europepmc():
    strata = [(c, tuple(s)) for c in CONDS for s in cfgB.europepmc_year_strata]
    res = parallel_map(lambda cs: epB.search_oa(cs[0], cfgB.europepmc_per_stratum, years=cs[1],
                                                cache_name=f"searchB_{cs[0]}_{cs[1][0]}"), strata, W)
    metas, seen = [], {}
    for (c, s), (results, hits) in zip(strata, res):
        for m in results:
            if m.get("pmcid") and m["pmcid"] not in seen:
                seen[m["pmcid"]] = (c, s); metas.append((c, s, m))
    def one(csm):
        c, s, meta = csm
        try:
            return parse_jats(epB.full_text_xml(meta["pmcid"]), meta, snapshot=SNAP, condition=c), None
        except (FetchError, KeyError, ET.ParseError) as exc:
            return None, (meta.get("pmcid"), type(exc).__name__, str(exc)[:120])
    out = parallel_map(one, metas, W)
    return {"docs": [d for d, _ in out if d], "fail": [e for _, e in out if e],
            "hits": {f"{c}:{s[0]}-{s[1]}": h for (c, s), (_, h) in zip(strata, res)},
            "stratum_of": {m["pmcid"]: f"{s[0]}-{s[1]}" for c, s, m in metas}}


tB = time.perf_counter()
RB, secB = {}, {}
with ThreadPoolExecutor(max_workers=6) as pool:
    f_ids = pool.submit(_timed, "pubmed_ids", b_pubmed_ids)
    futs = [f_ids, pool.submit(_timed, "pubmed_efetch", lambda: b_pubmed_docs(f_ids)),
            pool.submit(_timed, "pubtator", lambda: b_pubtator(f_ids)),
            pool.submit(_timed, "openfda", b_openfda), pool.submit(_timed, "clinicaltrials", b_clinicaltrials),
            pool.submit(_timed, "europepmc_oa", b_europepmc)]
    for fut in as_completed(futs):
        name, res, dt = fut.result()  # any job's exception is re-raised here
        RB[name], secB[name] = res, dt
wallB = time.perf_counter() - tB
clientsB = {"pubmed": pmB, "pubtator": ptB, "openfda": fdaB, "clinicaltrials": ctB, "europepmc": epB}
reqB = pd.DataFrame({k: {"requests": c.http.n_requests, "retries": c.http.n_retries, "cache_hits": c.http.n_cache_hits}
                     for k, c in clientsB.items()}).T
LIVE_B = int(reqB["requests"].sum()) > 0
expB.log_result("acquisition_timing" if LIVE_B else "replay_timing",
                {"jobs_s": secB, "wall_s": wallB, "requests": reqB.to_dict(orient="index")})
print(f"Stage B {'LIVE acquisition' if LIVE_B else 'REPLAY from raw archive'}: wall {wallB:.1f}s")
display(pd.Series(secB, name="seconds").round(1).sort_values(ascending=False).to_frame()); display(reqB)
""")

md("## B2. Reconciliation: requested vs received, and caps")
code(r"""
ids = RB["pubmed_ids"]; pm_full = RB["pubmed_efetch"]; ents_B, rels_B, pt_seen = RB["pubtator"]
fdaR, ctR, epR = RB["openfda"], RB["clinicaltrials"], RB["europepmc_oa"]
pm_parsed = {d.source_id for d in pm_full}
recon = {
    "pubmed_ids_unique": len(ids["pmids"]), "pubmed_ids_sum_over_conditions": sum(ids["per_condition"].values()),
    "pubmed_ids_per_condition": ids["per_condition"], "pubmed_ids_in_two_year_slices": ids["in_more_than_one_year_slice"],
    "pubmed_per_condition_equals_population_count": {c: ids["per_condition"][c] == pop_total[c] for c in CONDS},
    "pubmed_largest_year_slice": ids["largest_year_slice"],
    "pubmed_esearch_cap": PubMedClient.ESEARCH_MAX,
    "pubmed_parsed": len(pm_full), "pubmed_ids_without_record": len(set(ids["pmids"]) - pm_parsed),
    "pubmed_population_counts_stageA": pop_total,
    "pubtator_docs_returned": len(pt_seen), "pubtator_missing": len(set(ids["pmids"]) - pt_seen),
    "openfda_labels_fetched": fdaR["fetched"], "openfda_labels_unique": len(fdaR["docs"]),
    "openfda_labels_under_several_query_drugs": fdaR["dup"], "openfda_drugs_without_rx_label": sorted(d for d, t in fdaR["totals"].items() if t == 0),
    "openfda_drugs_truncated_at_limit": {d: t for d, t in fdaR["totals"].items() if t > cfgB.openfda_max_labels_per_drug},
    "clinicaltrials_totals": ctR["totals"], "clinicaltrials_unique": len(ctR["docs"]), "clinicaltrials_cross_condition_dups": ctR["dup"],
    "europepmc_fulltexts": len(epR["docs"]), "europepmc_failures": epR["fail"],
}
expB.log_result("reconciliation", recon)
print(json.dumps(recon, indent=1, default=str))
""")

md(r"""
## B3. Deduplication and exclusions
- **PubMed:** retracted records are excluded (safety rule). Near-duplicate abstracts (≥ 50 words; Jaccard ≥ 0.8) are *annotated* with a group id but **not removed**: they can be legitimately distinct publications (e.g., duplicate publication, or the same abstract in two journals). Whether to collapse them at retrieval time is an NB05 decision.
- **openFDA:** near-duplicate clusters within each (drug, single-ingredient vs combination) group are collapsed to the most recent label. The other labels' set-ids are kept as aliases (D-NB01-3).
""")
code(r"""
t0 = time.perf_counter()
retracted = [d for d in pm_full if d.extra.get("is_retracted")]
pm_keep = [d for d in pm_full if not (cfgB.exclude_retracted and d.extra.get("is_retracted"))]
long_pm = [d for d in pm_keep if len(d.text.split()) >= cfgB.near_dup_min_words]
pm_pairs, pm_clusters = near_duplicate_clusters([d.doc_id for d in long_pm], [d.text for d in long_pm],
                                                threshold=cfgB.near_dup_threshold, seed=SEED)
group_of = {doc_id: f"pmdup{gi:05d}" for gi, cl in enumerate(pm_clusters) for doc_id in cl}
pm_keep = [d.model_copy(update={"extra": {**d.extra, "near_dup_group": group_of[d.doc_id]}}) if d.doc_id in group_of else d
           for d in pm_keep]
t_pm = time.perf_counter() - t0

t0 = time.perf_counter()
fda_reps, fda_aliases, fda_cluster_sizes = [], {}, []
groups = defaultdict(list)
for d in fdaR["docs"]:
    groups[(d.extra["query_drug"], d.extra["is_combination"])].append(d)
for key, members in groups.items():
    _, clusters = near_duplicate_clusters([d.doc_id for d in members], [d.text for d in members],
                                          threshold=cfgB.near_dup_threshold, seed=SEED)
    in_cluster = set()
    by_id = {d.doc_id: d for d in members}
    for cl in clusters:
        docs_cl = sorted((by_id[i] for i in cl), key=lambda d: (d.pub_date or date.min, d.doc_id), reverse=True)
        rep = docs_cl[0]
        fda_aliases[rep.doc_id] = [x.identifiers["set_id"] for x in docs_cl[1:]]
        fda_reps.append(rep.model_copy(update={"extra": {**rep.extra, "n_near_duplicate_labels": len(docs_cl) - 1}}))
        in_cluster |= set(cl); fda_cluster_sizes.append(len(cl))
    fda_reps += [d for d in members if d.doc_id not in in_cluster]
t_fda = time.perf_counter() - t0

dedup = {
    "pubmed_retracted_excluded": len(retracted), "pubmed_after_exclusion": len(pm_keep),
    "pubmed_near_dup_checked_docs": len(long_pm), "pubmed_lsh_candidate_pairs": pm_pairs.attrs.get("n_candidates"),
    "pubmed_exact_checks": pm_pairs.attrs.get("n_exact_checks"),
    "pubmed_near_dup_edges_verified": len(pm_pairs), "pubmed_near_dup_clusters": len(pm_clusters),
    "pubmed_docs_in_near_dup_clusters": sum(len(c) for c in pm_clusters), "pubmed_near_dup_seconds": round(t_pm, 1),
    "openfda_labels_before": len(fdaR["docs"]), "openfda_labels_after": len(fda_reps),
    "openfda_clusters": len(fda_cluster_sizes), "openfda_largest_cluster": max(fda_cluster_sizes, default=0),
    "openfda_dedup_seconds": round(t_fda, 1),
}
expB.log_result("deduplication", dedup)
expB.save_table("pubmed_near_dup_pairs", pm_pairs)
print(json.dumps(dedup, indent=1))
""")
code(r"""
# Inspect a few PubMed near-duplicate pairs: what kind of duplication is this?
show = pm_pairs.sort_values("jaccard", ascending=False).head(6)
by_id = {d.doc_id: d for d in pm_keep}
for _, r in show.iterrows():
    a, b = by_id[r.doc_a], by_id[r.doc_b]
    print(f"J={r.jaccard:.2f} | {a.doc_id} ({a.pub_date}, {a.journal}) vs {b.doc_id} ({b.pub_date}, {b.journal})")
    print("   A:", a.title[:100]); print("   B:", b.title[:100])
""")

md("## B4. Assemble and persist the corpus")
code(r"""
ct_full, ep_full = ctR["docs"], [d.model_copy(update={"extra": {**d.extra, "year_stratum": epR["stratum_of"][d.source_id]}}) for d in epR["docs"]]
corpus = pm_keep + fda_reps + ct_full + ep_full + who_docs
countsB = {
    "documents_interim_all": write_jsonl(pm_full + fdaR["docs"] + ct_full + ep_full + who_docs, OUTB / "documents_all.jsonl"),
    "documents": write_jsonl(corpus, PROC / "documents.jsonl"),
    "concepts": write_jsonl(concepts, PROC / "concepts.jsonl"),
    "pubtator_entities": write_jsonl(ents_B, PROC / "pubtator_entities.jsonl"),
    "pubtator_relations": write_jsonl(rels_B, PROC / "pubtator_relations.jsonl"),
}
(PROC / "exclusions_and_aliases.json").write_text(json.dumps({
    "retracted_excluded_pmids": sorted(d.source_id for d in retracted),
    "openfda_aliases": fda_aliases, "pubmed_near_dup_clusters": pm_clusters}, indent=1), encoding="utf-8")
sizes = {p.name: round(p.stat().st_size / 1e6, 1) for p in PROC.glob("*.jsonl")}
manifestB = {"snapshot": SNAP, "stage": "B", "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "counts": countsB, "by_source": dict(Counter(d.source for d in corpus)), "file_mb": sizes,
             "licenses": dict(Counter(d.license for d in corpus).most_common(12)),
             "source_snapshot_dates": {"who_gho": SNAP + " (Stage A fetch)", "concepts": SNAP + " (Stage A fetch)",
                                       "others": SNAP + " (Stage B fetch)"}}
(PROC / "manifest.json").write_text(json.dumps(manifestB, indent=1), encoding="utf-8")
expB.log_result("corpus_manifest", manifestB)
print(json.dumps(manifestB, indent=1))
""")

md("## B5. Evaluation: full-corpus quality\n### B5.1 Composition and completeness")
code(r"""
dfB = docs_to_frame(corpus)
compB = composition(dfB); complB = completeness(corpus)
expB.save_table("composition", compB); expB.save_table("completeness", complB)
expB.log_result("composition", compB.to_dict(orient="index")); expB.log_result("completeness", complB.round(3).to_dict(orient="index"))
display(compB); display(complB.round(3))
""")

md("### B5.2 Evidence types in the full PubMed corpus (actual), vs Stage A population upper bounds")
code(r"""
pmB_df = dfB[dfB.source == "pubmed"].copy()
pmB_df["stratum"] = pmB_df.publication_types.map(evidence_stratum)
actual = pmB_df.stratum.value_counts().rename("corpus_docs")
evB = pd.concat([actual, (actual / actual.sum()).round(3).rename("corpus_share"), pop_share.rename("population_upper_bound_share").round(3)], axis=1)
evB["corpus_docs"] = evB["corpus_docs"].fillna(0).astype(int)  # population share stays NaN where no population query exists ("other")
by_cond = (pmB_df.explode("conditions").groupby(["conditions", "stratum"]).size().unstack(fill_value=0))
expB.save_table("evidence_strata", evB); expB.save_table("evidence_strata_by_condition", by_cond)
expB.log_result("evidence_strata", evB.astype(object).where(evB.notna(), None).to_dict(orient="index"))  # NaN → null (valid JSON)
display(evB); display(by_cond)
""")
code(r"""
fig, ax = plt.subplots(figsize=(9, 3.5))
evB[["corpus_share", "population_upper_bound_share"]].sort_values("corpus_share").plot.barh(ax=ax)
ax.set_xlabel("share of PubMed documents"); ax.set_title("Evidence strata: full corpus (exclusive, highest tier first) vs population upper bounds")
expB.save_figure("evidence_strata", fig); plt.show()
""")

md("### B5.3 Temporal distribution and quality flags")
code(r"""
fig, ax = plt.subplots(figsize=(10, 3.5))
for s, g in dfB[dfB.year.notna() & (dfB.source != "who_gho")].groupby("source"):
    ax.hist(g.year, bins=range(1990, 2028), alpha=.5, label=f"{s} (n={len(g)})")
ax.set_yscale("log"); ax.set_ylabel("documents (log)"); ax.legend(fontsize=7); ax.set_title("Full corpus: document year by source")
expB.save_figure("temporal", fig); plt.show()
qB = quality_flags(pm_full + fdaR["docs"] + ct_full + ep_full)  # before exclusions, so retractions are counted
expB.log_result("quality_flags", qB); pd.Series(qB, name="value").to_frame()
""")

md("### B5.4 PubTator coverage, and cross-source links")
code(r"""
drug_names = {d.lower() for _, d in dcfg.all_drugs()} | {s.lower() for x in concepts if x.source == "rxnorm" for s in x.synonyms}
ptB_cov = pubtator_coverage(ents_B, rels_B, pm_keep, drug_names)
pm_ids_set = {d.source_id for d in pm_keep}
links = {
    "trials_with_result_pmids": sum(bool(d.identifiers.get("result_pmids")) for d in ct_full),
    "trials_with_result_pmid_in_corpus": sum(bool(set(d.identifiers.get("result_pmids", [])) & pm_ids_set) for d in ct_full),
    "europepmc_oa_with_pmid_in_corpus": sum(d.identifiers.get("pmid") in pm_ids_set for d in ep_full),
}
expB.log_result("pubtator_coverage", ptB_cov); expB.log_result("cross_source_links", links)
print(json.dumps(ptB_cov, indent=1)); print(json.dumps(links, indent=1))
""")

md("### B5.5 Europe PMC long-document subset (input to NB02)")
code(r"""
epdf = pd.DataFrame([{"pmcid": d.source_id, "stratum": d.extra["year_stratum"], "words": len(d.text.split()),
                      "sections": len(d.sections), "tables": d.extra["n_tables"], "license": d.license} for d in ep_full])
ep_summary = epdf.groupby("stratum").agg(docs=("pmcid", "count"), words_median=("words", "median"), sections_median=("sections", "median"), tables=("tables", "sum"))
expB.save_table("europepmc_subset", ep_summary); expB.log_result("europepmc_subset", ep_summary.to_dict(orient="index"))
display(ep_summary); display(epdf.license.value_counts().to_frame())
""")
code(r"""
expB.finish()
print("results:", expB.dir)
""")
