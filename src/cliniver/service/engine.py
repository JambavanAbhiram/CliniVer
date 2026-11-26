"""The CliniVer verification engine: the research pipeline as one in-process object (modular monolith, rule 9).

Input: a clinical question and an AI-generated answer. Output: per-claim verdicts (SUP / CON / INS / ABSTAINED),
each with its evidence, provenance (source, title, date, link), confidence and the corpus snapshot.

The configuration is not chosen here. It is read from the optimization experiment's results
(``configs/service.yaml`` → ``configuration_source``):
- if the pre-registered adoption rule adopted the selected configuration, it is used;
- otherwise the baseline (v1 recipe, v1 policy) is used.

GPU discipline (CLAUDE.md, one large model at a time): the PyTorch models are moved to the CPU while the LLM server
runs, and back for retrieval and verification.

Research tool only: it never diagnoses, prescribes or recommends changes to medication. Retrieved documents are
untrusted data (the LLM prompts say so).
"""

from __future__ import annotations

import contextlib
import json
import pickle
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

LABEL_TEXT = {
    "SUP": "Supported by the retrieved evidence",
    "CON": "Contradicted by the retrieved evidence",
    "INS": "Insufficient evidence in the corpus (this does not mean false)",
    "ABSTAINED": "Abstained: low confidence and the two verifiers disagree",
}
BASELINE_LATENCY = {"llm_extra_args": [], "rerank_pool": 50, "batched_retrieval": False, "warm_llm": False, "so_max_tokens": 300}
DISCLAIMER = ("Research prototype. Verdicts are relative to a dated corpus snapshot and may be wrong. "
              "Not medical advice; never use it to diagnose, prescribe or change medication.")


def parse_config_name(name: str) -> dict[str, Any]:
    """'v2|guard=1|tau=0.2|so=llm_label' → {'verifier': 'v2', 'guard': True, 'tau_q': 0.2, 'so': 'llm_label'}."""
    parts = name.split("|")
    kv = dict(p.split("=", 1) for p in parts[1:])
    return {"verifier": parts[0], "guard": kv["guard"] == "1", "tau_q": None if kv["tau"] == "None" else float(kv["tau"]),
            "so": kv["so"]}


def source_link(doc_id: str) -> str | None:
    src, _, ident = doc_id.partition(":")
    if src == "pubmed" and ident.isdigit():
        return f"https://pubmed.ncbi.nlm.nih.gov/{ident}/"
    if src == "clinicaltrials" and ident.upper().startswith("NCT"):
        return f"https://clinicaltrials.gov/study/{ident}"
    if src == "europepmc_oa" and ident.upper().startswith("PMC"):
        return f"https://europepmc.org/article/PMC/{ident}"
    return None


def key_terms_regex(concepts_jsonl: Path) -> re.Pattern:
    """The NB04 key-term pattern (in-scope RxNorm drug names and synonyms, ≥ 4 characters) for context-dependence tagging."""
    from cliniver.ingestion.qa_datasets import domain_regex
    names = set()
    with concepts_jsonl.open(encoding="utf-8") as fh:
        for line in fh:
            c = json.loads(line)
            if c["source"] == "rxnorm":
                names |= {n.lower() for n in [c["name"], *c["synonyms"]] if len(n) >= 4}
    return domain_regex(names)


@dataclass
class ActiveConfig:
    name: str
    verifier: str
    verifier_dir: str
    guard: bool
    tau: float | None
    so: str
    adopted: bool
    source_experiment: str


class Engine:
    def __init__(self, log=print, latency: dict | None = None, llm_log_dir: Path | None = None) -> None:
        from cliniver.config import load_project_config
        from cliniver.config.llm_config import load_llm_config
        from cliniver.config.service_config import load_service_config
        self.log = log
        self.scfg = load_service_config(); self.pcfg = load_project_config(); self.lcfg = load_llm_config()
        root = self.pcfg.paths.absolute("data_raw").parents[1]
        self.root = root
        self.rcfg = yaml.safe_load((root / "configs" / "retrieval.yaml").read_text(encoding="utf-8"))
        self.vcfg = yaml.safe_load((root / "configs" / "verification.yaml").read_text(encoding="utf-8"))
        self.snap = self.scfg.snapshot
        self.proc = self.pcfg.paths.absolute("data_processed") / self.snap
        self.interim = self.pcfg.paths.absolute("data_interim") / self.snap
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.lock = threading.Lock()
        self.ready = False
        self.load_error: str | None = None
        self.active = self._active_config()
        self.latency = {**BASELINE_LATENCY, **(latency if latency is not None else self._adopted_latency())}
        self.llm_log_dir = llm_log_dir or (self.interim / "service" / "llm_calls")
        self._warm = None

    # ------------------------------------------------------------------ configuration
    def _active_config(self) -> ActiveConfig:
        exp = self.scfg.configuration_source
        res = json.loads((self.pcfg.paths.absolute("artifacts") / "experiments" / exp / "results.json").read_text(encoding="utf-8"))
        adopted = bool(res["adoption"]["adopted_as_baseline_v2"])
        name = res["selection"]["selected"] if adopted else res["selection"]["baseline"]
        c = parse_config_name(name)
        tau = None
        if c["tau_q"] is not None:
            tv = res["selection"]["tau_values"]
            tau = float(tv.get(str(c["tau_q"]), tv.get(c["tau_q"])))
        vdir = res["finetune"][c["verifier"]]["dir"]
        return ActiveConfig(name, c["verifier"], vdir, c["guard"], tau, c["so"], adopted, exp)

    def _adopted_latency(self) -> dict:
        """Latency settings adopted by EXP-08-003 (pre-registered rule); the baseline settings if not run yet."""
        p = self.pcfg.paths.absolute("artifacts") / "experiments" / "EXP-08-003" / "results.json"
        if not p.exists():
            return {}
        return json.loads(p.read_text(encoding="utf-8")).get("adoption", {}).get("adopted_settings", {})

    # ------------------------------------------------------------------ loading
    def load(self) -> None:
        try:
            self._load()
            self.ready = True
        except Exception as exc:  # noqa: BLE001 — reported through /api/v1/status, never hidden
            self.load_error = f"{type(exc).__name__}: {exc}"
            raise

    def _load(self) -> None:
        from sentence_transformers import CrossEncoder
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from cliniver.retrieval.encoders import Encoder
        from cliniver.retrieval.pipeline import EvidenceRetriever, build_units, unit_embeddings
        from cliniver.verification.verifiers import NLIVerifier
        t0 = time.perf_counter()
        self.units, rows = build_units(self.proc / "documents.jsonl", self.proc / "chunks" / "paragraph-128" / "chunks.jsonl")
        self.uid = {u.unit_id: i for i, u in enumerate(self.units)}
        self.urls = {}
        with (self.proc / "documents.jsonl").open(encoding="utf-8") as fh:   # canonical source URL per document (PubMed, DailyMed, CT.gov, WHO, Europe PMC)
            for line in fh:
                d = json.loads(line)
                if d.get("url"):
                    self.urls[d["doc_id"]] = d["url"]
        self.log(f"units {len(self.units)} ({time.perf_counter() - t0:.0f}s)")
        m = self.rcfg["encoders"]["medcpt"]; emb = self.interim / "embeddings"
        self.dense = Encoder("medcpt", m["repo"], m["revision"], doc_repo=m["doc_repo"], doc_revision=m["doc_revision"], device=self.dev)
        uemb = unit_embeddings(self.units, rows, self.dense, emb / "medcpt_cliniver_chunks.npy", emb / "medcpt_pubmed_abstracts_whole.npy").to(self.dev)
        self.dense.d_model.to("cpu"); del self.dense.d_model   # documents are pre-encoded; only the query encoder is needed
        self.dense.d_model = None
        ce = self.rcfg["rerankers"]["medcpt-ce"]
        self.ce = CrossEncoder(ce["repo"], revision=ce["revision"], device=self.dev, max_length=512)
        if self.dev == "cuda":
            self.ce.model.half()
        self.retriever = EvidenceRetriever(self.units, self.dense, uemb, self.ce, depth=self.rcfg["first_stage_depth"], pool=int(self.latency["rerank_pool"]),
                                           rrf_k=self.rcfg["fusion"]["rrf_k"])
        self.log(f"retriever ready ({time.perf_counter() - t0:.0f}s)")
        d = self.interim / "models" / self.active.verifier_dir
        self.nli = NLIVerifier(str(d), device=self.dev, model=AutoModelForSequenceClassification.from_pretrained(d),
                               tokenizer=AutoTokenizer.from_pretrained(d), max_length=self.vcfg["finetune"]["max_length"])
        self.cm = self._confidence_model(d.name)
        self.keys = key_terms_regex(self.proc / "concepts.jsonl")
        from cliniver.generation.answer import Vocabulary
        gcfg = yaml.safe_load((self.root / "configs" / "generation.yaml").read_text(encoding="utf-8"))
        dcfg = yaml.safe_load((self.root / "configs" / "data.yaml").read_text(encoding="utf-8"))
        with (self.proc / "concepts.jsonl").open(encoding="utf-8") as fh:
            concepts = [json.loads(l) for l in fh]
        self.vocab = Vocabulary.build(gcfg, dcfg["drugs"], concepts, dcfg["clinicaltrials"]["conditions"])
        self.gen_strategy = self._generation_strategy(gcfg)
        if self.latency["warm_llm"]:
            self._warm = self._new_llm().start()   # kept running; the PyTorch models stay on the GPU (L4: measured within budget)
        self.log(f"engine ready: {self.active.name}; latency settings {self.latency} ({time.perf_counter() - t0:.0f}s)")

    def _generation_strategy(self, gcfg: dict):
        """The answer-generation strategy adopted by EXP-09-001 (pre-registered rule); G0 until that experiment has run."""
        from cliniver.generation.answer import Routed, Strategy
        for exp_id in ("EXP-09-003", "EXP-09-002", "EXP-09-001"):   # the latest round with an adoption record decides
            p = self.pcfg.paths.absolute("artifacts") / "experiments" / exp_id / "results.json"
            if p.exists():
                a = json.loads(p.read_text(encoding="utf-8")).get("adoption", {})
                if a.get("service_strategy"):
                    c = a["strategy_config"]
                    if "route" in c:
                        return Routed.from_cfg(a["service_strategy"], c["route"], {})
                    return Strategy.from_cfg(a["service_strategy"], c)
        return Strategy.from_cfg("G0", gcfg["variants"]["G0"])

    def _confidence_model(self, verifier_dir: str):
        """The ADR-0014 confidence model for the active verifier, fitted on the calibration split (HealthVer dev) and
        cached, so the service starts without refitting."""
        from cliniver.verification.confidence import ConfidenceModel
        cache = self.interim / "service" / f"confidence_{verifier_dir}.pkl"
        if cache.exists():
            return pickle.loads(cache.read_bytes())
        from cliniver.ingestion.http import RateLimitedClient
        from cliniver.verification.datasets import load_sets
        sets, _ = load_sets(self.vcfg, self.pcfg.paths.absolute("data_raw"), self.snap,
                            RateLimitedClient("datasets", min_interval_s=1.0, max_in_flight=1), seed=self.scfg.seed)
        cm = ConfidenceModel.fit(self.nli, sets["healthver_dev"])
        cache.parent.mkdir(parents=True, exist_ok=True); cache.write_bytes(pickle.dumps(cm))
        return cm

    # ------------------------------------------------------------------ GPU discipline
    def _torch_to(self, dev: str) -> None:
        if self.dev != "cuda" or self._warm is not None:
            return
        self.dense.q_model.to(dev); self.dense.device = dev
        self.nli.model.to(dev); self.nli.device = dev
        self.ce.model.to(dev)
        torch.cuda.empty_cache()

    def _new_llm(self):
        from cliniver.llm.runtime import ManagedClient
        key = self.scfg.llm_model
        return ManagedClient(self.lcfg, key, self.llm_log_dir / f"{key}.jsonl",
                             server_log=self.pcfg.paths.absolute("logs") / f"llama-server_service_{key}.log",
                             extra_args_add=list(self.latency["llm_extra_args"]))

    def _llm(self):
        "An LLM client for one phase: the warm server (kept running) or a server started and stopped for this phase."
        return contextlib.nullcontext(self._warm) if self._warm is not None else self._new_llm()

    def close(self) -> None:
        if self._warm is not None:
            self._warm.__exit__(None, None, None); self._warm = None

    # ------------------------------------------------------------------ verification
    def verify(self, question: str, answer: str) -> dict:
        if not self.ready:
            raise RuntimeError(self.load_error or "engine is still loading")
        question, answer = question.strip(), answer.strip()[:self.scfg.max_answer_chars]
        if not answer:
            raise ValueError("an answer to verify is required")
        with self.lock:
            return self._verify(question, answer)

    def ask(self, question: str) -> dict:
        """Answer the question with the local LLM from retrieved evidence (the EXP-09-001 strategy), then verify that
        answer claim by claim against the corpus (the generated answer gets no special trust)."""
        from cliniver.generation.answer import fuse, generate, plan, search_queries
        if not self.ready:
            raise RuntimeError(self.load_error or "engine is still loading")
        question = question.strip()
        if not question:
            raise ValueError("a question is required")
        s = self.gen_strategy
        with self.lock:
            timings, plan_obj = {}, None
            if s.retrieval != "question":                       # LLM phase: search plan (and, for a routed strategy, the intent)
                t0 = time.perf_counter(); self._torch_to("cpu")
                with self._llm() as mc:
                    plan_obj = plan(mc, question, seed=self.scfg.seed, tag="service:plan")
                timings["query_planning_s"] = round(time.perf_counter() - t0, 2)
            routed_from = None
            if hasattr(s, "pick"):
                routed_from, s = s.name, s.pick(plan_obj)
            t0 = time.perf_counter(); self._torch_to(self.dev)
            queries = search_queries(s, question, plan_obj, self.vocab)
            found = [self.retriever.search(x, top=10) for x in queries]
            if s.retrieval == "question" and not s.diverse:
                picked = list(zip(found[0][0][:s.k], found[0][1][:s.k]))
            else:
                picked = fuse(found, self.units, s.k, s.diverse)
            passages = [self._evidence(self.units[j], sc) for j, sc in picked]
            timings["question_retrieval_s"] = round(time.perf_counter() - t0, 2)
            gen: dict = {}

            def produce(mc) -> str:
                ans, info = generate(mc, question, passages, prompt=s.prompt, seed=self.scfg.seed, tag="service:generate")
                gen.update({"answer": ans, **info}); return ans

            res = self._verify(question, None, produce=produce)
        res["generated"] = {"answer": gen.get("answer", ""), "passages": passages, "model": self.scfg.llm_model, "strategy": s.name if routed_from is None else f"{routed_from} → {s.name}",
                            "search_queries": queries, "plan": plan_obj, "latency_s": gen.get("latency_s"), "finish_reason": gen.get("finish_reason")}
        res["timings"] = {**timings, **res["timings"]}
        return res

    def _verify(self, question: str, answer: str | None, produce=None) -> dict:
        from cliniver.claims.contextualize import context_dependent, decontextualize
        from cliniver.claims.extract import extract_qualified, extract_sentence
        from cliniver.verification.consistency import unsupported_numbers
        from cliniver.verification.verifiers import LLMVerifier
        seed, timings, notes = self.scfg.seed, {}, []
        # 1. LLM phase: claim extraction and standalone rewrites
        t0 = time.perf_counter(); self._torch_to("cpu")
        with self._llm() as mc:
            if produce is not None:                    # generation shares the LLM phase (one server start)
                answer = produce(mc)[:self.scfg.max_answer_chars]
                timings["generation_s"] = round(time.perf_counter() - t0, 2); t0 = time.perf_counter()
                if not answer:
                    return {"claims": [], "notes": ["The language model returned no answer."], **self._meta(timings, mc.stats)}
            ex = extract_qualified(mc, answer, seed=seed, tag="service:extract")
            if not ex.valid_json or not ex.claims:
                notes.append("Claim extraction did not return valid output; the answer was split into sentences instead.")
                ex = extract_sentence(answer)
            claims = []
            for c in ex.claims:
                dep, why = context_dependent(c.text, question, self.keys)
                rewrite = None
                if dep:
                    rewrite, _ = decontextualize(mc, c.text, question, answer, seed=seed, tag="service:dectx")
                claims.append({"claim": c.text, "fields": c.fields, "context_dependent": dep, "context_reason": why,
                               "verified_text": (rewrite or c.text).strip()})
            llm_stats = mc.stats
        timings["extraction_and_rewrite_s"] = round(time.perf_counter() - t0, 2)
        if not claims:
            return {"answer": answer, "claims": [], "notes": notes + ["No verifiable claim was found in the answer."], **self._meta(timings, llm_stats)}
        # 2. Retrieval, verification, confidence (PyTorch on the GPU)
        t0 = time.perf_counter(); self._torch_to(self.dev)
        k = self.scfg.evidence_top_k
        if self.latency["batched_retrieval"]:
            found = self.retriever.search_many([c["verified_text"] for c in claims], top=k)
        else:
            found = [self.retriever.search(c["verified_text"], top=k) for c in claims]
        for c, (idx, sc) in zip(claims, found):
            c["evidence"] = [self._evidence(self.units[j], s) for j, s in zip(idx, sc)]
        top = [self.units[self.uid[c["evidence"][0]["unit_id"]]] for c in claims]
        texts = [c["verified_text"] for c in claims]
        z = self.nli.logits([u.text for u in top], texts)
        labels, probs, overall = self.cm.score(texts, [u.text for u in top], z)
        top_ce = [float(np.asarray(self.ce.predict([(t, u.index_text)], show_progress_bar=False), dtype=float).reshape(-1)[-1])
                  for t, u in zip(texts, top)]
        for i, c in enumerate(claims):
            lab, rules = labels[i], []
            if self.active.guard and lab == "SUP":
                miss = unsupported_numbers(texts[i], top[i].text)
                if miss:
                    lab = "INS"; rules.append(f"numeric guard: claim number(s) {', '.join(miss)} not found in the evidence")
            if self.active.tau is not None and top_ce[i] < self.active.tau:
                if lab != "INS":
                    rules.append("weak evidence match: top evidence score below the INS threshold")
                lab = "INS"
            c.update({"verifier_label": labels[i], "label_after_rules": lab, "rules_applied": rules,
                      "probabilities": {n: round(float(p), 4) for n, p in zip(("SUP", "CON", "INS"), probs[i])},
                      "confidence": round(float(overall[i]), 4), "top_evidence_score": round(top_ce[i], 3),
                      "routed_to_second_opinion": bool(self.active.so != "none" and overall[i] < self.cm.route_threshold)})
        timings["retrieval_and_verification_s"] = round(time.perf_counter() - t0, 2)
        # 3. Second opinion for the low-confidence band
        routed = [c for c in claims if c["routed_to_second_opinion"]]
        so_stats = None
        if routed:
            t0 = time.perf_counter(); self._torch_to("cpu")
            with self._llm() as mc:
                v = LLMVerifier(mc, seed=seed, max_tokens=int(self.latency["so_max_tokens"]))
                for c in routed:
                    u = self.units[self.uid[c["evidence"][0]["unit_id"]]]
                    l, info = v.verify(u.text, c["verified_text"], tag="service:verify")
                    l = "SUP" if l == "PART" else (l or "INS")
                    c["second_opinion"] = {"model": self.scfg.llm_model, "label": l, "rationale": info.get("rationale", ""),
                                           "quote": info.get("quote", ""), "agrees": l == c["label_after_rules"]}
                so_stats = mc.stats
            self._torch_to(self.dev)
            timings["second_opinion_s"] = round(time.perf_counter() - t0, 2)
        for c in claims:
            so = c.get("second_opinion")
            if so is None or so["agrees"]:
                final = c["label_after_rules"]
            else:
                final = "ABSTAINED" if self.active.so == "abstain" else so["label"]
            c["verdict"] = final; c["verdict_text"] = LABEL_TEXT[final]
            c["evidence_dates"] = [e["date"] for e in c["evidence"]]
        return {"answer": answer, "claims": claims, "notes": notes, **self._meta(timings, llm_stats or so_stats)}

    def _evidence(self, u, score: float) -> dict:
        return {"unit_id": u.unit_id, "doc_id": u.doc_id, "source": u.source, "title": u.title, "date": u.pub_date,
                "text": u.text, "rerank_score": round(float(score), 3), "link": self.urls.get(u.doc_id) or source_link(u.doc_id)}

    def _meta(self, timings: dict, llm_stats: dict | None) -> dict:
        return {"corpus_snapshot": self.snap, "configuration": self.active.__dict__, "latency_settings": self.latency, "timings": timings,
                "llm_server": llm_stats, "disclaimer": DISCLAIMER}

    def status(self) -> dict:
        return {"ready": self.ready, "error": self.load_error, "corpus_snapshot": self.snap, "configuration": self.active.__dict__,
                "latency_settings": self.latency, "device": self.dev, "units": len(getattr(self, "units", []) or []), "disclaimer": DISCLAIMER}

    def unit(self, unit_id: str):
        i = self.uid.get(unit_id) if self.ready else None
        return None if i is None else self.units[i]
