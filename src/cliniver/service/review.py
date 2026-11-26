"""Expert review round 1: a stratified queue of real-claim verdicts and an append-only annotation log.

The queue is a seeded sample that gives equal weight to each (verdict × reference-source) stratum, so rare verdicts
(CON, ABSTAINED) are reviewed too. Rates estimated from it must therefore be re-weighted by the stratum sizes, which
``summary()`` reports.

Annotations are appended, never overwritten. A reviewer's latest judgement per claim counts. This is a first-stage
expert check, not the clinical validation of ADR-0006 (which needs clinicians, a written protocol and
inter-annotator agreement).
"""

from __future__ import annotations

import json
import random
import threading
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ASSESSMENTS = ("correct", "incorrect", "unsure")
LABELS = ("SUP", "CON", "INS", "PART", "NOT_VERIFIABLE")
EVIDENCE_RELEVANCE = ("relevant", "partly", "irrelevant")


def _jsonl(p: Path) -> list[dict]:
    with p.open(encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


class ReviewStore:
    def __init__(self, proc: Path, root: Path, cfg, seed: int, unit_lookup) -> None:
        self.cfg, self.seed, self.unit_lookup = cfg, seed, unit_lookup
        self.ann_path = root / cfg.annotations_file
        self.lock = threading.Lock()
        v2 = _jsonl(proc / "responses" / cfg.verdicts_file)
        v1 = {r["claim_id"]: r for r in _jsonl(proc / "responses" / cfg.baseline_verdicts_file)}
        meta = {r["claim_id"]: r for r in _jsonl(proc / "responses" / "claims_contextualized.jsonl")}
        resp = {r["response_id"]: r for r in _jsonl(proc / "responses" / "responses.jsonl")}
        items = []
        for r in v2:
            m = meta[r["claim_id"]]; rs = resp[m["response_id"]]
            items.append({"claim_id": r["claim_id"], "question": rs["question"], "answer": rs["answer"], "generator": rs["generator"],
                          "query_source": rs["query_source"],
                          "reference": "physician statements (K-QA)" if rs["query_source"].startswith("kqa") else "LLM-generated statements (MedLFQA)",
                          "claim": r["claim"], "verified_text": r["verified_text"], "verdict": r["final_verdict"],
                          "verifier_label": r["verifier_label"], "confidence": r["overall_confidence"], "second_opinion": r.get("second_opinion"),
                          "decisive_unit": r["decisive_unit"], "evidence_date": r.get("evidence_date"), "config": r["config"],
                          "v1_verdict": v1.get(r["claim_id"], {}).get("final_verdict"), "corpus_snapshot": r["corpus_snapshot"]})
        self.population = Counter((x["verdict"], x["reference"]) for x in items)
        strata = defaultdict(list)
        for x in items:
            strata[(x["verdict"], x["reference"])].append(x)
        rng = random.Random(seed)
        keys = sorted(strata)
        per = max(1, cfg.sample_size // len(keys))
        queue = []
        for k in keys:
            pool = sorted(strata[k], key=lambda x: x["claim_id"])
            queue += rng.sample(pool, min(per, len(pool)))
        rng.shuffle(queue)
        self.queue = queue
        self.by_id = {x["claim_id"]: x for x in queue}

    def item(self, claim_id: str) -> dict | None:
        x = self.by_id.get(claim_id)
        if x is None:
            return None
        u = self.unit_lookup(x["decisive_unit"])
        ev = None if u is None else {"unit_id": u.unit_id, "doc_id": u.doc_id, "source": u.source, "title": u.title, "date": u.pub_date, "text": u.text}
        return {**x, "evidence": ev}

    def annotations(self) -> list[dict]:
        return _jsonl(self.ann_path) if self.ann_path.exists() else []

    def latest(self, reviewer: str | None = None) -> dict[tuple[str, str], dict]:
        out = {}
        for a in self.annotations():
            if reviewer is None or a["reviewer"] == reviewer:
                out[(a["reviewer"], a["claim_id"])] = a
        return out

    def add(self, a: dict) -> dict:
        reviewer = str(a.get("reviewer", "")).strip()[:80]
        if not reviewer:
            raise ValueError("reviewer name is required")
        cid = a.get("claim_id")
        if cid not in self.by_id:
            raise ValueError("unknown claim_id")
        if a.get("assessment") not in ASSESSMENTS:
            raise ValueError(f"assessment must be one of {ASSESSMENTS}")
        corr = a.get("correct_label") or None
        if corr is not None and corr not in LABELS:
            raise ValueError(f"correct_label must be one of {LABELS}")
        if a["assessment"] == "incorrect" and corr is None:
            raise ValueError("an 'incorrect' assessment needs the correct label")
        rel = a.get("evidence_relevance") or None
        if rel is not None and rel not in EVIDENCE_RELEVANCE:
            raise ValueError(f"evidence_relevance must be one of {EVIDENCE_RELEVANCE}")
        ef = a.get("extraction_faithful")
        x = self.by_id[cid]
        rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "reviewer": reviewer, "claim_id": cid,
               "assessment": a["assessment"], "correct_label": corr, "evidence_relevance": rel,
               "extraction_faithful": None if ef is None else bool(ef), "note": str(a.get("note", ""))[:2000],
               "verdict_shown": x["verdict"], "config": x["config"], "corpus_snapshot": x["corpus_snapshot"]}
        with self.lock:
            self.ann_path.parent.mkdir(parents=True, exist_ok=True)
            with self.ann_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    def summary(self) -> dict:
        lat = self.latest()
        by_verdict = defaultdict(Counter)
        for a in lat.values():
            by_verdict[a["verdict_shown"]][a["assessment"]] += 1
        return {"queue": len(self.queue), "reviewed_items": len({cid for _, cid in lat}), "reviewers": sorted({r for r, _ in lat}),
                "assessment_by_verdict": {k: dict(v) for k, v in by_verdict.items()},
                "population_by_stratum": {f"{v} | {r}": n for (v, r), n in sorted(self.population.items())},
                "note": "Stratified sample (equal per verdict × reference stratum): re-weight by population_by_stratum before estimating rates."}
