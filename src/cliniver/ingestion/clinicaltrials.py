"""ClinicalTrials.gov API v2. Trial-registry records (protocol summaries + whether results are posted)."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone

from cliniver.config.data_config import ClinicalTrialsCfg
from cliniver.ingestion.http import RateLimitedClient, RawStore, cached_get
from cliniver.schemas.document import SOURCE_ROLES, Document, Section


class ClinicalTrialsClient:
    def __init__(self, cfg: ClinicalTrialsCfg, store: RawStore, workers: int = 1) -> None:
        self.cfg = cfg
        self.store = store
        self.workers = workers
        self.http = RateLimitedClient("clinicaltrials", cfg.min_interval_s, max_in_flight=workers)

    def _params(self, condition_key: str, page_size: int) -> dict:
        return {"query.cond": self.cfg.conditions[condition_key],
                # Essie expression on the structured field (query.intr would be free-text search).
                "query.term": f"AREA[InterventionType]{self.cfg.intervention_type}",
                "filter.overallStatus": "COMPLETED", "pageSize": page_size, "countTotal": "true",
                "format": "json"}

    def search(self, condition_key: str, page_size: int) -> tuple[list[dict], int | None]:
        body = json.loads(cached_get(self.http, self.store, f"studies_{condition_key}_p0", "json",
                                     self.cfg.base_url, self._params(condition_key, page_size)))
        return body.get("studies", []), body.get("totalCount")

    def search_all(self, condition_key: str, page_size: int, cache_prefix: str) -> tuple[list[dict], int | None]:
        """Every matching study, following nextPageToken (pages are sequential by construction).

        Raises if the number of studies received differs from the API's totalCount.
        """
        studies: list[dict] = []
        total, token, page = None, None, 0
        while True:
            params = self._params(condition_key, page_size)
            if token:
                params["pageToken"] = token
            body = json.loads(cached_get(self.http, self.store, f"{cache_prefix}_{condition_key}_p{page:03d}",
                                         "json", self.cfg.base_url, params))
            total = body.get("totalCount", total)
            studies += body.get("studies", [])
            token, page = body.get("nextPageToken"), page + 1
            if not token:
                break
        if total is not None and len(studies) != total:
            raise RuntimeError(f"{condition_key}: totalCount {total} but {len(studies)} studies received")
        return studies, total


def _date(s: str | None) -> tuple[date | None, str]:
    if not s:
        return None, "unknown"
    parts = s.split("-")
    if len(parts) == 3:
        return date(int(parts[0]), int(parts[1]), int(parts[2])), "day"
    if len(parts) == 2:
        return date(int(parts[0]), int(parts[1]), 1), "month"
    return date(int(parts[0]), 1, 1), "year"


def parse_study(study: dict, *, snapshot: str, condition: str, retrieved_at: datetime | None = None) -> Document:
    p = study["protocolSection"]
    ident, desc = p.get("identificationModule", {}), p.get("descriptionModule", {})
    status, design = p.get("statusModule", {}), p.get("designModule", {})
    arms = p.get("armsInterventionsModule", {})
    nct = ident["nctId"]
    secs = [Section(name=k, text=" ".join(desc[k].split())) for k in ("briefSummary", "detailedDescription")
            if desc.get(k)]
    # Results date if posted, else completion date: the date the evidence became available.
    pub_date, prec = _date(status.get("resultsFirstPostDateStruct", {}).get("date")
                           or status.get("completionDateStruct", {}).get("date"))
    phases = design.get("phases", [])
    refs = p.get("referencesModule", {}).get("references", [])
    # Trial → publication links (RESULT = publication reporting this trial's results)
    result_pmids = sorted({str(r["pmid"]) for r in refs if r.get("pmid") and r.get("type") == "RESULT"})
    all_pmids = sorted({str(r["pmid"]) for r in refs if r.get("pmid")})
    return Document(
        doc_id=f"clinicaltrials:{nct}", source="clinicaltrials", source_id=nct,
        role=SOURCE_ROLES["clinicaltrials"],
        title=ident.get("officialTitle") or ident.get("briefTitle", nct),
        text="\n".join(s.text for s in secs), sections=secs, pub_date=pub_date, date_precision=prec,
        retrieved_at=retrieved_at or datetime.now(timezone.utc), snapshot=snapshot,
        publication_types=[f"Trial registration ({design.get('studyType', 'UNKNOWN')})"],
        license="ClinicalTrials.gov terms and conditions (US government registry)",
        url=f"https://clinicaltrials.gov/study/{nct}",
        identifiers={"nct_id": nct, "result_pmids": result_pmids, "reference_pmids": all_pmids},
        conditions=[condition],
        extra={
            "overall_status": status.get("overallStatus"), "phases": phases,
            "has_results": bool(study.get("hasResults")),
            "enrollment": design.get("enrollmentInfo", {}).get("count"),
            "conditions": p.get("conditionsModule", {}).get("conditions", []),
            "interventions": [{"type": i.get("type"), "name": i.get("name")} for i in arms.get("interventions", [])],
            "last_update": status.get("lastUpdatePostDateStruct", {}).get("date"),
        },
    )
