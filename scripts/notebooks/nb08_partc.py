"""NB08 Part C: optimization round 1, latency (EXP-08-003). Added by build_nb08.py after Part B.

Every figure is measured live on this laptop: the LLM calls use fresh, run-specific call logs, so no cached replay
hides the latency.
"""


def add_cells(md, code):
    md(r"""
# Part C — Optimization round 1: latency (EXP-08-003)
- **Owner request 2026-11-22:** include latency in the optimization.
- **Protocol** (`configs/optimize_latency.yaml`): written before any latency measurement and before the EXP-08-002 results were seen. The n-gram variant of L1 was added before measurement.
- **Rule:** a lever is adopted only if its quality check passes **and** it cuts its stage's median latency by ≥ 10% on the dev data.
- **Baseline stage latencies (EXP-08-001):**
  - extraction 18.8 s per answer;
  - retrieval p95 0.37 s per claim;
  - second opinion 2.33 s per routed claim;
  - demo: + ~3 s per LLM-server start.

| Lever | Stage | Quality check |
|---|---|---|
| L1 speculative decoding (Qwen3-0.6B draft, or n-gram lookup) | claim extraction | outputs identical, else claim F1 ≥ 0.98 vs baseline; GPU ≤ budget |
| L2 smaller re-rank pool (50 → 30 / 20) | re-ranking | dev end-to-end macro-F1 Δ ≥ −0.01 |
| L3 batched retrieval per answer | retrieval | top-1 evidence identical for ≥ 99% of claims |
| L4 warm LLM server, PyTorch resident | demo overhead | joint GPU peak ≤ budget |
| L5 second-opinion max_tokens 300 → 160 | second opinion | label agreement ≥ 0.98 |
""")
    code(r"""
from cliniver.llm.runtime import LlamaServer, ChatClient, gpu_memory_used_mb
from cliniver.claims.extract import extract_qualified
lat_cfg = yaml.safe_load((ROOT / "configs" / "optimize_latency.yaml").read_text(encoding="utf-8"))
exp3 = Experiment(exp_id="EXP-08-003-smoke" if SMOKE else "EXP-08-003", title="Optimization round 1: latency", research_question="latency (owner request 2026-11-22)",
                  config={"optimize_latency": lat_cfg}, seed=SEED, packages=["torch", "transformers", "sentence-transformers", "bm25s", "cliniver"])
RUN_TS = time.strftime("%Y%m%dT%H%M%S")
LAT_LOG = INTERIM / "llm_calls" / "latency" / RUN_TS          # fresh call logs: every LLM call in Part C is live
BUDGET_MB = lcfg.runtime.vram_budget_gb * 1024
med = lambda x: round(float(np.median(x)), 3) if len(x) else None
p95 = lambda x: round(float(np.percentile(x, 95)), 3) if len(x) else None
def qwen_safe(pairs, tag):
    for v_ in VER.values(): v_.model.to("cpu")
    out = qwen_labels(pairs, tag)
    for v_ in VER.values(): v_.model.to(DEV)
    return out
def sync():
    if DEV == "cuda": torch.cuda.synchronize()
print("latency logs:", LAT_LOG)
""")

    md("## 21. L2: re-rank pool size (SciFact-train dev split, selected EXP-08-002 configuration)")
    code(r"""
dq = SCO["dev"]["Q"]
bi = bm.search(dq, DEPTH)[0]
m = rcfg["encoders"]["medcpt"]; enc = Encoder("medcpt", m["repo"], m["revision"], doc_repo=m["doc_repo"], doc_revision=m["doc_revision"], device=DEV)
D = torch.from_numpy(np.load(EMB / "medcpt_primary.npy")).to(DEV); di = Encoder.search(enc.encode_queries(dq), D, DEPTH)[0]; enc.unload(); del D; torch.cuda.empty_cache()
CANDS = [list(rrf([a, b], k=rcfg["fusion"]["rrf_k"], top=DEPTH)) for a, b in zip(bi, di)]
POOL = {}
for pool in lat_cfg["levers"]["L2_rerank_pool"]["values"]:
    lat, top1, s1 = [], [], []
    for q, c in zip(dq, CANDS):
        c = c[:pool]; sync(); t0 = time.perf_counter()
        s = np.asarray(CE.predict([(q, DOC_TEXT[j]) for j in c], batch_size=64, show_progress_bar=False), dtype=float).reshape(len(c), -1)[:, -1]
        sync(); lat.append(time.perf_counter() - t0); k = int(np.argmax(s)); top1.append(c[k]); s1.append(float(s[k]))
    POOL[pool] = {"top1": top1, "score": s1, "lat": lat, "d": scored(dq, [DOC_TEXT[j] for j in top1], SCO["dev"]["gold"], s1, verifiers=(SEL[0],))}
need = sorted({(P["d"]["Q"][i], P["d"]["ev"][i]) for P in POOL.values() for i in routed(P["d"], SEL[0])})
_ = qwen_safe(need, "opt")
L2 = {}
ref = metrics(SCO["dev"]["gold"], *apply(POOL[50]["d"], *SEL))["macro_F1_e2e"]
for pool, P in POOL.items():
    mm = metrics(SCO["dev"]["gold"], *apply(P["d"], *SEL))
    L2[pool] = {"macro_F1_e2e": mm["macro_F1_e2e"], "delta_vs_50": round(mm["macro_F1_e2e"] - ref, 4), "FSR": mm["FSR_answered"],
                "rerank_ms_median": round(1000 * np.median(P["lat"]), 1), "rerank_ms_p95": round(1000 * np.percentile(P["lat"], 95), 1),
                "top1_same_as_pool50": round(float(np.mean(np.array(P["top1"]) == np.array(POOL[50]["top1"]))), 3)}
base_ms = L2[50]["rerank_ms_median"]
ok2 = [p for p in L2 if p != 50 and L2[p]["delta_vs_50"] >= -0.01 and L2[p]["rerank_ms_median"] <= 0.9 * base_ms]
L2_POOL = min(ok2) if ok2 else 50
exp3.log_result("L2_rerank_pool", {"by_pool": L2, "adopted_pool": L2_POOL}); display(pd.DataFrame(L2).T); print("adopted pool:", L2_POOL)
""")

    md(r"""
## 22. L1: speculative decoding for claim extraction (and L5 in the same baseline session)
Dev answers: a seeded sample of NB03 answers, plus one untimed warm-up answer per server. Each variant gets a fresh server and a fresh call log, so every call is generated live. The PyTorch models are on the CPU during this section (one large model on the GPU at a time).
""")
    code(r"""
for m_ in [nli.model, CE.model] + [v.model for v in VER.values()]: m_.to("cpu")
torch.cuda.empty_cache(); IDLE_MB = gpu_memory_used_mb()
resp_all = [json.loads(l) for l in (PROC / "responses" / "responses.jsonl").open(encoding="utf-8")]
n_dev = lat_cfg["dev_answers"]; pick = random.Random(SEED).sample(range(len(resp_all)), n_dev + 21)
WARM = resp_all[pick[0]]; DEV_ANS = [resp_all[i] for i in pick[1:1 + n_dev]]; TEST_ANS = [resp_all[i] for i in pick[1 + n_dev:]]
if SMOKE: DEV_ANS, TEST_ANS = DEV_ANS[:3], TEST_ANS[:2]
rt = lcfg.runtime; MK = "qwen3-8b"; L1c = lat_cfg["levers"]["L1_speculative_decoding"]
draft = lambda n: ["--spec-type", "draft-simple", "--model-draft", str(lcfg.model_path(L1c["draft_model"])), "--spec-draft-n-max", str(n), "-ngld", "999",
                   "--cache-type-k-draft", "q8_0", "--cache-type-v-draft", "q8_0"]
VARIANTS = {"baseline": [], **{f"draft06_n{n}": draft(n) for n in L1c["draft_max"]}, "ngram_simple": ["--spec-type", "ngram-simple"]}
so_pairs = [(SCO["dev"]["Q"][i], SCO["dev"]["ev"][i]) for i in routed(SCO["dev"], SEL[0])]
if SMOKE: so_pairs = so_pairs[:5]
EXTRACT, SRV_STATS, SO_RUN = {}, {}, {}
for name, extra in VARIANTS.items():
    time.sleep(2)   # let the previous server release its port
    srv = LlamaServer(ROOT / rt.dir, lcfg.model_path(MK), MK, rt.release, rt.host, rt.port, rt.ctx_size, rt.gpu_layers,
                      extra_args=list(rt.extra_args) + extra, log_path=pcfg.paths.absolute("logs") / f"llama-server_nb08c_{name}.log")
    try:
        srv.__enter__()
        cl = ChatClient(srv, LAT_LOG / f"{name}.jsonl", model_key=MK, model_file=lcfg.models[MK].file, release=rt.release)
        extract_qualified(cl, WARM["answer"], seed=SEED, tag="lat:warmup")
        rows = []
        for a in DEV_ANS:
            ex = extract_qualified(cl, a["answer"], seed=SEED, tag=f"lat:{name}")
            r_ = list(cl.cache.values())[-1]
            rows.append({"response_id": a["response_id"], "latency_s": ex.latency_s, "tokens": ex.completion_tokens, "valid": ex.valid_json,
                         "raw": ex.raw, "claims": [c.text for c in ex.claims], "draft_n": r_.get("draft_n"), "draft_accepted": r_.get("draft_accepted")})
        EXTRACT[name] = rows
        if name == "baseline":     # L5: second opinion with max_tokens 300 vs 160 on the routed dev pairs, same live server
            for mt in (300, 160):
                v = LLMVerifier(cl, seed=SEED, max_tokens=mt); labs, lats = [], []
                for c, e in so_pairs:
                    l, info = v.verify(e, c, tag=f"lat:so{mt}"); labs.append(("SUP" if l == "PART" else l) or "INS"); lats.append(info["latency_s"])
                SO_RUN[mt] = {"labels": labs, "lat": lats}
        SRV_STATS[name] = srv.stats()
    except Exception as exc:   # a variant this build cannot run is recorded, not hidden
        EXTRACT[name] = {"error": f"{type(exc).__name__}: {exc}"}; print(name, "failed:", exc)
    finally:
        srv.__exit__(None, None, None)
print({k: (len(v) if isinstance(v, list) else v) for k, v in EXTRACT.items()})
""")
    code(r"""
from cliniver.evaluation.extraction import NLIJudge
judge = NLIJudge(cfg["silver"]["judge"]["repo"], cfg["silver"]["judge"]["revision"], device=DEV)
_n = lambda t: re.sub(r"\W+", " ", t.lower()).strip()
def matched(A, B):
    "claims of A with a partner in B: identical text, or bidirectional entailment ≥ 0.5 (NB03 judge)"
    n = 0
    for a in A:
        if any(_n(a) == _n(b) for b in B): n += 1; continue
        if not B: continue
        p1 = judge.entail_prob(list(B), [a] * len(B)); p2 = judge.entail_prob([a] * len(B), list(B))
        n += bool(np.any((p1 >= 0.5) & (p2 >= 0.5)))
    return n
base = EXTRACT["baseline"]; b_med = med([r["latency_s"] for r in base]); b_valid = float(np.mean([r["valid"] for r in base]))
L1 = {"baseline": {"median_s": b_med, "p95_s": p95([r["latency_s"] for r in base]), "gpu_mb_above_idle": SRV_STATS["baseline"]["gpu_peak_minus_baseline_mb"]}}
for name, rows in EXTRACT.items():
    if name == "baseline": continue
    if isinstance(rows, dict): L1[name] = rows; continue
    ident = [r["raw"] == b["raw"] for r, b in zip(rows, base)]
    tp_r = sum(len(b["claims"]) if i else matched(b["claims"], r["claims"]) for r, b, i in zip(rows, base, ident))
    tp_p = sum(len(r["claims"]) if i else matched(r["claims"], b["claims"]) for r, b, i in zip(rows, base, ident))
    R = tp_r / max(1, sum(len(b["claims"]) for b in base)); P = tp_p / max(1, sum(len(r["claims"]) for r in rows))
    dn = [r["draft_n"] for r in rows if r["draft_n"]]; da = [r["draft_accepted"] for r in rows if r["draft_n"]]
    L1[name] = {"median_s": med([r["latency_s"] for r in rows]), "p95_s": p95([r["latency_s"] for r in rows]),
                "speedup_median": round(b_med / max(1e-9, med([r["latency_s"] for r in rows])), 2),
                "identical_outputs": round(float(np.mean(ident)), 3), "claim_F1_vs_baseline": round(2 * P * R / max(1e-9, P + R), 4),
                "valid_json": round(float(np.mean([r["valid"] for r in rows])), 3),
                "draft_acceptance": round(sum(da) / sum(dn), 3) if dn else None,
                "gpu_mb_above_idle": SRV_STATS[name]["gpu_peak_minus_baseline_mb"]}
del judge; torch.cuda.empty_cache()
def l1_ok(k):
    x = L1[k]
    return ("error" not in x and (x["identical_outputs"] == 1.0 or x["claim_F1_vs_baseline"] >= 0.98) and x["valid_json"] >= b_valid
            and (x["gpu_mb_above_idle"] or 0) <= BUDGET_MB and x["median_s"] <= 0.9 * b_med)
ok1 = [k for k in L1 if k != "baseline" and l1_ok(k)]
L1_ADOPT = min(ok1, key=lambda k: L1[k]["median_s"]) if ok1 else None
L1_ARGS = VARIANTS[L1_ADOPT] if L1_ADOPT else []
L5 = {"pairs": len(so_pairs)}
if SO_RUN:
    a300, a160 = np.array(SO_RUN[300]["labels"]), np.array(SO_RUN[160]["labels"])
    L5.update({"agreement": round(float(np.mean(a300 == a160)), 3) if len(a300) else None, "median_s_300": med(SO_RUN[300]["lat"]), "median_s_160": med(SO_RUN[160]["lat"])})
    L5["adopted"] = bool(L5["agreement"] is not None and L5["agreement"] >= 0.98 and L5["median_s_160"] <= 0.9 * L5["median_s_300"])
exp3.log_result("L1_speculative_decoding", {"variants": L1, "adopted": L1_ADOPT, "idle_gpu_mb_incl_notebook_context": IDLE_MB})
exp3.log_result("L5_second_opinion_tokens", L5)
display(pd.DataFrame(L1).T); print("L1 adopted:", L1_ADOPT, "| L5:", L5)
""")

    md(r"""
## 23. L3, L4 and the end-to-end latency of the service engine
- The **service engine** (`cliniver.service.engine`, the code behind the demo) is loaded with the EXP-08-002 configuration.
- **L3:** batched retrieval for the claims of each dev answer vs one search per claim.
- **L4:** joint GPU peak with the LLM server (adopted L1 flags) running while the PyTorch models stay on the GPU.
- **Final:** per-answer latency on 20 **test** answers, first with the baseline settings, then with all adopted levers. Fresh call logs keep every call live.
""")
    code(r"""
from cliniver.service.engine import Engine, BASELINE_LATENCY
del UNITS, uid; gc.collect()
eng = Engine(log=print, latency=dict(BASELINE_LATENCY), llm_log_dir=LAT_LOG / "engine_baseline"); eng.load()
groups = [r["claims"] for r in EXTRACT["baseline"] if r["claims"]]
t_one, t_many, same1, same3 = [], [], [], []
for g in groups:
    sync(); t0 = time.perf_counter(); one = [eng.retriever.search(c, top=3) for c in g]; sync(); t_one.append(time.perf_counter() - t0)
    sync(); t0 = time.perf_counter(); many = eng.retriever.search_many(g, top=3); sync(); t_many.append(time.perf_counter() - t0)
    same1 += [a[0][0] == b[0][0] for a, b in zip(one, many)]; same3 += [a[0] == b[0] for a, b in zip(one, many)]
L3 = {"answers": len(groups), "claims": len(same1), "per_answer_median_s_per_claim_search": med(t_one), "per_answer_median_s_batched": med(t_many),
      "top1_identical": round(float(np.mean(same1)), 4), "top3_identical": round(float(np.mean(same3)), 4)}
L3["adopted"] = bool(L3["top1_identical"] >= 0.99 and L3["per_answer_median_s_batched"] <= 0.9 * L3["per_answer_median_s_per_claim_search"])
exp3.log_result("L3_batched_retrieval", L3); print(L3)
# L4: LLM server (with the adopted L1 flags) running while the PyTorch models stay on the GPU and do retrieval + NLI work
eng.latency["llm_extra_args"] = L1_ARGS
mc = eng._new_llm().start()
flat = [c for g in groups for c in g][:24]
hits = eng.retriever.search_many(flat, top=1)
_ = eng.nli.logits([eng.units[h[0][0]].text for h in hits], flat)
_ = mc.chat([{"role": "user", "content": "Reply with OK."}], max_tokens=8, tag="lat:l4probe")
time.sleep(1.5); st = mc.stats; mc.__exit__(None, None, None)
L4 = {"joint_gpu_peak_mb": st["gpu_peak_mb"], "idle_gpu_mb_incl_notebook_context": IDLE_MB,
      "joint_above_idle_mb": (st["gpu_peak_mb"] - IDLE_MB) if st["gpu_peak_mb"] and IDLE_MB is not None else None, "budget_mb": BUDGET_MB,
      "server_load_s": st["load_s"]}
L4["adopted"] = bool(L4["joint_above_idle_mb"] is not None and L4["joint_above_idle_mb"] <= BUDGET_MB)
exp3.log_result("L4_warm_server", L4); print(L4)
""")
    code(r"""
ADOPTED = {"llm_extra_args": L1_ARGS, "rerank_pool": L2_POOL, "batched_retrieval": L3["adopted"], "warm_llm": L4["adopted"],
           "so_max_tokens": 160 if L5.get("adopted") else 300}
def run_answers(settings, tag):
    eng.close(); eng.latency.update({**BASELINE_LATENCY, **settings}); eng.retriever.pool = int(eng.latency["rerank_pool"])
    eng.llm_log_dir = LAT_LOG / f"engine_{tag}"
    if eng.latency["warm_llm"]:
        eng._torch_to(eng.dev); eng._warm = eng._new_llm().start()
    out = []
    for a in TEST_ANS:
        t0 = time.perf_counter(); r = eng.verify(a["question"], a["answer"]); wall = time.perf_counter() - t0
        out.append({"response_id": a["response_id"], "wall_s": round(wall, 2), "claims": len(r["claims"]), **r["timings"],
                    "verdicts": [c["verdict"] for c in r["claims"]], "claim_texts": [c["claim"] for c in r["claims"]]})
    eng.close()
    return out
FINAL = {"baseline": run_answers(dict(BASELINE_LATENCY), "baseline"), "adopted": run_answers(ADOPTED, "adopted")}
summ = {}
for k, rows in FINAL.items():
    w = [r["wall_s"] for r in rows]; per_claim = [r["wall_s"] / max(1, r["claims"]) for r in rows]
    summ[k] = {"answers": len(rows), "claims": sum(r["claims"] for r in rows), "per_answer_median_s": med(w), "per_answer_p95_s": p95(w),
               "per_claim_median_s": med(per_claim),
               **{f"{s}_median": med([r.get(s, 0) for r in rows]) for s in ("extraction_and_rewrite_s", "retrieval_and_verification_s", "second_opinion_s")}}
same_claims = [a["claim_texts"] == b["claim_texts"] for a, b in zip(FINAL["baseline"], FINAL["adopted"])]
agree = [x == y for a, b, s in zip(FINAL["baseline"], FINAL["adopted"], same_claims) if s for x, y in zip(a["verdicts"], b["verdicts"])]
summ["agreement"] = {"answers_with_identical_claims": round(float(np.mean(same_claims)), 3), "verdict_agreement_on_those": round(float(np.mean(agree)), 3) if agree else None}
summ["speedup_per_answer_median"] = round(summ["baseline"]["per_answer_median_s"] / max(1e-9, summ["adopted"]["per_answer_median_s"]), 2)
exp3.log_result("end_to_end_latency_test", summ)
exp3.log_result("adoption", {"adopted_settings": ADOPTED, "levers": {"L1": L1_ADOPT, "L2_pool": L2_POOL, "L3": L3["adopted"], "L4": L4["adopted"], "L5": L5.get("adopted")}})
display(pd.DataFrame({k: v for k, v in summ.items() if k in ("baseline", "adopted")}).T); print(json.dumps(summ["agreement"]), "speed-up", summ["speedup_per_answer_median"])
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.boxplot([[r["wall_s"] for r in FINAL["baseline"]], [r["wall_s"] for r in FINAL["adopted"]]]); ax.set_xticks([1, 2], ["baseline settings", "adopted levers"])
ax.set_ylabel("seconds per answer (test answers)"); ax.set_title("End-to-end verification latency (service engine)"); ax.grid(alpha=.3)
exp3.save_figure("latency_end_to_end", fig); plt.show()
exp3.finish(); print("results:", exp3.dir)
""")
