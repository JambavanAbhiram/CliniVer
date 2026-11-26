"""openFDA drug labels (SPL). CC0 public domain (open.fda.gov/license). Regulatory evidence, US-specific."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone

from cliniver.config.data_config import OpenFDACfg
from cliniver.ingestion.http import RateLimitedClient, RawStore, cached_get
from cliniver.schemas.document import SOURCE_ROLES, Document, Section


class OpenFDAClient:
    def __init__(self, cfg: OpenFDACfg, store: RawStore, workers: int = 1) -> None:
        self.cfg = cfg
        self.store = store
        self.workers = workers
        self.http = RateLimitedClient("openfda", cfg.min_interval_s, max_in_flight=workers)

    def labels_for(self, generic_name: str, limit: int) -> list[dict]:
        """Most recent labels whose openFDA generic name matches exactly. Empty list if none (404)."""
        return self.search(generic_name, limit, rx_only=False, cache_prefix="label")[0]

    def search(self, generic_name: str, limit: int, *, rx_only: bool, cache_prefix: str) -> tuple[list[dict], int]:
        """(labels, total matches). ``rx_only`` restricts to HUMAN PRESCRIPTION DRUG labels.

        Returns at most ``limit`` (≤ 1000) most recent labels; ``total`` reveals any truncation.
        """
        query = f'openfda.generic_name:"{generic_name}"'
        if rx_only:
            query += ' AND openfda.product_type:"HUMAN PRESCRIPTION DRUG"'
        params = {"search": query, "limit": limit, "sort": "effective_time:desc"}
        # openFDA answers 404 when nothing matches: an archived, reproducible "no label" result.
        body = cached_get(self.http, self.store, f"{cache_prefix}_{generic_name}", "json", self.cfg.base_url,
                          params, not_found_ok=True)
        if body is None:
            return [], 0
        data = json.loads(body)
        return data.get("results", []), int(data.get("meta", {}).get("results", {}).get("total", 0))

    def search_all(self, generic_name: str, max_labels: int, *, rx_only: bool, cache_prefix: str,
                   page_size: int = 100) -> tuple[list[dict], int]:
        """All matching labels (up to ``max_labels``), paging with ``skip``; returns (labels, total).

        Pages are sequential (each needs the previous total); different drugs run in parallel.
        """
        query = f'openfda.generic_name:"{generic_name}"'
        if rx_only:
            query += ' AND openfda.product_type:"HUMAN PRESCRIPTION DRUG"'
        labels: list[dict] = []
        total, page = None, 0
        while total is None or len(labels) < min(total, max_labels):
            params = {"search": query, "limit": page_size, "skip": page * page_size, "sort": "effective_time:desc"}
            body = cached_get(self.http, self.store, f"{cache_prefix}_{generic_name}_p{page:03d}", "json",
                              self.cfg.base_url, params, not_found_ok=True)
            if body is None:  # 404: no (more) matches
                break
            data = json.loads(body)
            total = int(data.get("meta", {}).get("results", {}).get("total", 0))
            got = data.get("results", [])
            if not got:
                break
            labels += got
            page += 1
        return labels[:max_labels], int(total or 0)


def _first(v) -> str | None:
    return v[0] if isinstance(v, list) and v else (v if isinstance(v, str) else None)


def is_combination(generic_name: str | None) -> bool:
    """openFDA writes multi-ingredient products as e.g. 'SITAGLIPTIN AND METFORMIN HYDROCHLORIDE'."""
    g = f" {(generic_name or '').upper()} "
    return " AND " in g or "/" in g or ", " in g


UNSTRUCTURED_SECTION = "spl_unclassified_section"


def parse_label(label: dict, *, sections: list[str], snapshot: str, query_drug: str, condition: str,
                retrieved_at: datetime | None = None) -> Document:
    of = label.get("openfda", {})
    secs = [Section(name=s, text=" ".join(" ".join(label[s]).split())) for s in sections if label.get(s)]
    # Some SPL labels (often older generics) are not split into named sections: the whole label is in
    # spl_unclassified_section. Without this fallback they became empty documents (30 of 399 in Stage B).
    unstructured = not secs and bool(label.get(UNSTRUCTURED_SECTION))
    if unstructured:
        secs = [Section(name=UNSTRUCTURED_SECTION, text=" ".join(" ".join(label[UNSTRUCTURED_SECTION]).split()))]
    eff = label.get("effective_time")
    pub_date = date(int(eff[:4]), int(eff[4:6]), int(eff[6:8])) if eff and len(eff) >= 8 else None
    brand = _first(of.get("brand_name"))
    generic = _first(of.get("generic_name")) or query_drug
    set_id = label.get("set_id") or label["id"]
    return Document(
        doc_id=f"openfda_label:{label['id']}", source="openfda_label", source_id=label["id"],
        role=SOURCE_ROLES["openfda_label"],
        title=f"{brand or generic} ({generic}) prescribing information",
        text="\n".join(f"{s.name}: {s.text}" for s in secs), sections=secs,
        pub_date=pub_date, date_precision="day" if pub_date else "unknown",
        retrieved_at=retrieved_at or datetime.now(timezone.utc), snapshot=snapshot,
        publication_types=["Drug Label"], license="CC0 1.0 (openFDA)",
        url=f"https://dailymed.nlm.nih.gov/dailymed/lookup.cfm?setid={set_id}",
        identifiers={"set_id": set_id, "spl_id": label["id"], "rxcui": of.get("rxcui", []),
                     "unii": of.get("unii", []), "application_number": of.get("application_number", [])},
        conditions=[condition],
        extra={"query_drug": query_drug, "brand_name": brand, "manufacturer": _first(of.get("manufacturer_name")),
               "product_type": _first(of.get("product_type")), "route": of.get("route", []),
               "n_generic_names": len(of.get("generic_name", [])),
               "generic_name": generic, "is_combination": is_combination(generic),
               "unstructured_label": unstructured},
    )
