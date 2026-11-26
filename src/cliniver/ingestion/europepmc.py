"""Europe PMC: open-access full text (JATS XML), license-filtered (ADR-0002).

Used mainly as the long-document corpus for the chunking study (NB02).
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from datetime import date, datetime, timezone

from cliniver.config.data_config import EuropePMCCfg
from cliniver.ingestion.http import RateLimitedClient, RawStore, cached_get
from cliniver.schemas.document import SOURCE_ROLES, Document, Section


class EuropePMCClient:
    def __init__(self, cfg: EuropePMCCfg, store: RawStore, workers: int = 1) -> None:
        self.cfg = cfg
        self.store = store
        self.workers = workers
        self.http = RateLimitedClient("europepmc", cfg.min_interval_s, max_in_flight=workers)

    def search_oa(self, condition_key: str, page_size: int, years: tuple[int, int] | None = None,
                  cache_name: str | None = None) -> tuple[list[dict], int]:
        """License-filtered OA search; ``years`` restricts to PUB_YEAR in [a, b] (for year-stratified subsets)."""
        lic = " OR ".join(f'LICENSE:"{l}"' for l in self.cfg.licenses)
        q = f"({self.cfg.condition_queries[condition_key]}) AND OPEN_ACCESS:Y AND IN_EPMC:Y AND ({lic}) AND SRC:MED"
        if years:
            q += f" AND PUB_YEAR:[{years[0]} TO {years[1]}]"
        body = json.loads(cached_get(self.http, self.store, cache_name or f"search_{condition_key}", "json",
                                     f"{self.cfg.base_url}/search",
                                     {"query": q, "format": "json", "pageSize": page_size, "resultType": "core"}))
        return body["resultList"]["result"], int(body.get("hitCount", 0))

    def full_text_xml(self, pmcid: str) -> bytes:
        return cached_get(self.http, self.store, f"fulltext_{pmcid}", "xml", f"{self.cfg.base_url}/{pmcid}/fullTextXML")


def _text(el: ET.Element | None) -> str:
    return " ".join("".join(el.itertext()).split()) if el is not None else ""


def parse_jats(xml_bytes: bytes, meta: dict, *, snapshot: str, condition: str,
               retrieved_at: datetime | None = None) -> Document:
    """Body sections become Sections (top-level <sec> title → paragraphs). Tables/figures are skipped
    as text but counted, so NB02 can report what is lost."""
    root = ET.fromstring(xml_bytes)
    sections: list[Section] = []
    abstract = _text(root.find(".//abstract"))
    if abstract:
        sections.append(Section(name="ABSTRACT", text=abstract))
    body = root.find(".//body")
    if body is not None:
        # Short articles (letters, editorials) put paragraphs directly under <body> without any <sec>;
        # these were previously dropped, leaving 5 of 398 Stage B full texts empty.
        loose = [_text(p) for p in body.findall("p") if _text(p)]
        if loose:
            sections.append(Section(name="BODY", text="\n".join(loose)))
        for sec in body.findall("sec"):
            title = _text(sec.find("title")) or "SECTION"
            paras = [_text(p) for p in sec.iter("p") if _text(p)]
            if paras:
                sections.append(Section(name=title, text="\n".join(paras)))
    pmcid = meta["pmcid"]
    fp = meta.get("firstPublicationDate")
    pub_date = date.fromisoformat(fp) if fp else None
    return Document(
        doc_id=f"europepmc_oa:{pmcid}", source="europepmc_oa", source_id=pmcid, role=SOURCE_ROLES["europepmc_oa"],
        title=meta.get("title", ""), text="\n\n".join(s.text for s in sections), sections=sections,
        pub_date=pub_date, date_precision="day" if pub_date else "unknown",
        retrieved_at=retrieved_at or datetime.now(timezone.utc), snapshot=snapshot,
        publication_types=(meta.get("pubTypeList") or {}).get("pubType", []),
        journal=(meta.get("journalInfo") or {}).get("journal", {}).get("title"),
        language=meta.get("language"), license=meta.get("license"),
        url=f"https://europepmc.org/article/PMC/{pmcid}",
        identifiers={k: str(meta[k]) for k in ("pmid", "pmcid", "doi") if meta.get(k)}, conditions=[condition],
        extra={"n_tables": len(root.findall(".//table-wrap")), "n_figures": len(root.findall(".//fig")),
               "has_body": body is not None},
    )
