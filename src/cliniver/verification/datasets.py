"""Verification datasets shared by NB06 and NB07 (same construction and order, so fixed-seed samples and the LLM
call cache line up exactly).

Sets: SciFact train/dev (cited abstracts; SUPPORT → SUP, CONTRADICT → CON, cited without evidence → INS),
HealthVer train/dev/test, HealthFC (evaluation only, CC BY-NC-ND 4.0), and synthetic rule perturbations of SUP pairs.
"""

from __future__ import annotations

import csv
import io
import json
import random
import tarfile

from cliniver.ingestion.http import RawStore, cached_get
from cliniver.ingestion.scifact import parse_release
from cliniver.verification.perturb import PERTURBATIONS


def load_sets(cfg: dict, raw_root, snapshot: str, http, smoke: bool = False, seed: int = 42) -> tuple[dict, bytes]:
    sf_raw = cached_get(http, RawStore(raw_root, "scifact", snapshot), "data", "tar.gz", cfg["datasets"]["scifact_url"])
    sf_corpus, _ = parse_release(sf_raw)
    tf = tarfile.open(fileobj=io.BytesIO(sf_raw), mode="r:gz")

    def read_claims(split):
        n = next(x for x in tf.getnames() if x.endswith(f"claims_{split}.jsonl") and "/._" not in x and "cross_validation" not in x)
        return [json.loads(l) for l in tf.extractfile(n).read().decode().splitlines() if l.strip()]

    def sf_pairs(split):
        out = []
        for c in read_claims(split):
            ev = c.get("evidence") or {}
            for d in c.get("cited_doc_ids", []):
                labs = {s["label"] for s in ev.get(str(d), [])}
                lab = "SUP" if "SUPPORT" in labs else "CON" if "CONTRADICT" in labs else "INS"
                if str(d) in sf_corpus:
                    out.append({"id": f"sf{split}:{c['id']}:{d}", "claim": c["claim"], "evidence": sf_corpus[str(d)], "label": lab, "set": f"scifact_{split}"})
        return out

    def hv(split):
        raw = cached_get(http, RawStore(raw_root, "healthver", snapshot), f"healthver_{split}", "csv",
                         f"https://raw.githubusercontent.com/sarrouti/HealthVer/{cfg['datasets']['healthver']['commit']}/data/healthver_{split}.csv")
        m = {"Supports": "SUP", "Refutes": "CON", "Neutral": "INS"}
        return [{"id": f"hv{split}:{r['id']}", "claim": r["claim"], "evidence": r["evidence"], "label": m[r["label"]], "set": f"healthver_{split}"}
                for r in csv.DictReader(io.StringIO(raw.decode("utf-8")))]

    fc_raw = cached_get(http, RawStore(raw_root, "healthfc", snapshot), "Datensatz", "csv",
                        f"https://raw.githubusercontent.com/jvladika/HealthFC/{cfg['datasets']['healthfc']['commit']}/Datensatz.csv")
    fc_rows = list(csv.DictReader(io.StringIO(fc_raw.decode("utf-8-sig"))))
    fcmap = cfg["datasets"]["healthfc_label_map"]
    healthfc = [{"id": f"fc:{i}", "claim": r["en_claim"], "evidence": r["en_top_sentences"], "label": fcmap[r["label"]], "set": "healthfc"}
                for i, r in enumerate(fc_rows)]
    sets = {"scifact_train": sf_pairs("train"), "scifact_dev": sf_pairs("dev"), "healthver_train": hv("train"), "healthver_dev": hv("dev"),
            "healthver_test": hv("test"), "healthfc": healthfc}
    if smoke:
        sets = {k: random.Random(seed).sample(v, min(len(v), 60)) for k, v in sets.items()}
    syn = []
    for p in [x for x in sets["scifact_dev"] + sets["healthver_test"] if x["label"] == "SUP"]:
        for kind, (fn, lab) in PERTURBATIONS.items():
            c2 = fn(p["claim"])
            if c2 and c2 != p["claim"]:
                syn.append({"id": f"syn:{kind}:{p['id']}", "claim": c2, "evidence": p["evidence"], "label": lab, "set": "synthetic", "kind": kind, "orig": p["claim"]})
    sets["synthetic"] = syn
    return sets, sf_raw


def llm_samples(sets: dict, sizes: dict, eval_sets: list[str], seed: int) -> dict[str, list[int]]:
    """The fixed-seed sample indices used for the LLM verifiers in NB06."""
    out = {}
    for s in eval_sets:
        n = sizes.get(s, "all")
        idx = list(range(len(sets[s])))
        out[s] = idx if n == "all" or n >= len(idx) else sorted(random.Random(seed).sample(idx, n))
    return out
