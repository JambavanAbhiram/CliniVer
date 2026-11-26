"""The per-claim verification summary as a table: claim, verdict, confidence, evidence dates, evidence links.

Pure functions (no Streamlit), so the table the interface shows is unit-tested.
"""

from __future__ import annotations

import csv
import io

SOURCE_LABEL = {"pubmed": "PubMed", "openfda_label": "FDA label", "clinicaltrials": "Trial record", "europepmc_oa": "Europe PMC",
                "who_gho": "WHO data"}
VERDICT_LABEL = {"SUP": "Supported", "CON": "Contradicted", "INS": "Insufficient evidence", "ABSTAINED": "Abstained"}
_MD_SPECIAL = str.maketrans({c: "\\" + c for c in "\\|*_`[]<>#"})


def md_escape(t: str) -> str:
    return " ".join(str(t).split()).translate(_MD_SPECIAL)


def summary_rows(result: dict) -> list[dict]:
    """One row per claim. ``links`` is a list of (label, url | None); the decisive (top-1) evidence comes first."""
    rows = []
    for i, c in enumerate(result.get("claims", []), 1):
        so = c.get("second_opinion")
        decided_by_llm = bool(so) and not so.get("agrees", True) and c["verdict"] == so.get("label")
        docs, seen = [], set()          # several passages of one document (e.g. one FDA label) are listed once
        for e in c.get("evidence", []):
            if e["doc_id"] not in seen:
                seen.add(e["doc_id"]); docs.append(e)
        rows.append({"n": i, "claim": c.get("verified_text") or c["claim"], "verdict": c["verdict"],
                     "verdict_text": VERDICT_LABEL.get(c["verdict"], c["verdict"]) + (" †" if decided_by_llm else ""),
                     "confidence": float(c.get("confidence", 0.0)), "dates": [e["date"] for e in docs if e.get("date")],
                     "links": [(SOURCE_LABEL.get(e["source"], e["source"]), e.get("link")) for e in docs],
                     "titles": [e.get("title") or e["doc_id"] for e in docs]})
    return rows


def summary_markdown(rows: list[dict]) -> str:
    head = "| # | Claim | Verdict | Confidence | Evidence dates | Evidence |\n|---:|---|---|---:|---|---|\n"
    body = []
    for r in rows:
        links = " · ".join(f"[{md_escape(lbl)} {k}]({url})" if url else f"{md_escape(lbl)} {k}" for k, (lbl, url) in enumerate(r["links"], 1))
        body.append(f"| {r['n']} | {md_escape(r['claim'])} | **{md_escape(r['verdict_text'])}** | {r['confidence']:.0%} | "
                    f"{', '.join(r['dates']) or 'undated'} | {links or '—'} |")
    return head + "\n".join(body)


def summary_csv(rows: list[dict]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["#", "claim", "verdict", "confidence", "evidence_dates", "evidence_titles", "evidence_links"])
    for r in rows:
        w.writerow([r["n"], r["claim"], r["verdict_text"], f"{r['confidence']:.4f}", "; ".join(r["dates"]),
                    " | ".join(r["titles"]), " | ".join(u or "" for _, u in r["links"])])
    return buf.getvalue()
