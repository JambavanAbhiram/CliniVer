"""WHO Global Health Observatory (OData). Population-level statistics, CC BY-NC-SA 3.0 IGO.

Each (indicator, location, year, sex) value is textualized into a short statement with an
explicit source, so it can be retrieved like any other passage. Role "statistics": these can
check epidemiological claims, but never clinical-effect claims.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone

from cliniver.config.data_config import WhoGhoCfg
from cliniver.ingestion.http import RateLimitedClient, RawStore, cached_get
from cliniver.schemas.document import SOURCE_ROLES, Document, Section


class WhoGhoClient:
    def __init__(self, cfg: WhoGhoCfg, store: RawStore, workers: int = 1) -> None:
        self.cfg = cfg
        self.store = store
        self.workers = workers
        self.http = RateLimitedClient("who_gho", cfg.min_interval_s, max_in_flight=workers)

    def find_indicators(self, keyword: str) -> list[dict]:
        params = {"$filter": f"contains(tolower(IndicatorName),'{keyword.lower()}')"}
        body = cached_get(self.http, self.store, f"indicators_{keyword}", "json", f"{self.cfg.base_url}/Indicator",
                          params)
        return json.loads(body).get("value", [])

    def values(self, code: str) -> list[dict]:
        types = " or ".join(f"SpatialDimType eq '{t}'" for t in self.cfg.spatial_types)
        body = cached_get(self.http, self.store, f"values_{code}", "json", f"{self.cfg.base_url}/{code}",
                          {"$filter": types})
        return json.loads(body).get("value", [])


def textualize(row: dict, indicator_name: str, *, snapshot: str, retrieved_at: datetime | None = None) -> Document:
    year = row.get("TimeDim")
    sex = {"SEX_BTSX": "both sexes", "SEX_MLE": "males", "SEX_FMLE": "females"}.get(row.get("Dim1") or "", row.get("Dim1"))
    ci = (f" (95% uncertainty interval {row['Low']:.1f}–{row['High']:.1f})"
          if isinstance(row.get("Low"), (int, float)) and isinstance(row.get("High"), (int, float)) else "")
    value = row.get("NumericValue")
    stmt = (f"According to the WHO Global Health Observatory, the indicator '{indicator_name}' for "
            f"{row.get('SpatialDim')} in {year}{', ' + sex if sex else ''} was "
            f"{value:.1f}{ci}." if isinstance(value, (int, float)) else
            f"WHO GHO '{indicator_name}' for {row.get('SpatialDim')} in {year}: {row.get('Value')}.")
    rid = str(row.get("Id"))
    return Document(
        doc_id=f"who_gho:{row.get('IndicatorCode')}:{rid}", source="who_gho", source_id=rid,
        role=SOURCE_ROLES["who_gho"], title=f"WHO GHO: {indicator_name}", text=stmt,
        sections=[Section(name="statistic", text=stmt)],
        pub_date=date(int(year), 12, 31) if year else None, date_precision="year" if year else "unknown",
        retrieved_at=retrieved_at or datetime.now(timezone.utc), snapshot=snapshot,
        publication_types=["Population statistic"], license="CC BY-NC-SA 3.0 IGO (WHO)",
        url=f"https://www.who.int/data/gho/data/indicators/indicator-details/GHO/{row.get('IndicatorCode')}",
        identifiers={"indicator_code": str(row.get("IndicatorCode")), "spatial_dim": str(row.get("SpatialDim"))},
        extra={"value": value, "low": row.get("Low"), "high": row.get("High"), "dim1": row.get("Dim1"),
               "spatial_type": row.get("SpatialDimType"), "year": year},
    )
