"""PubMed via NCBI E-utilities (esearch / efetch), keyless at <= 3 req/s.

Parsing keeps what verification needs: structured abstract sections (BACKGROUND / METHODS /
RESULTS / CONCLUSIONS), publication types (evidence tier proxy), MeSH, dates with their
precision, and retraction flags.
"""

from __future__ import annotations

import json
import os
import xml.etree.ElementTree as ET
from datetime import date, datetime, timezone

from cliniver.config.data_config import PubMedCfg
from cliniver.ingestion.http import RateLimitedClient, RawStore, cached_get, parallel_map
from cliniver.schemas.document import SOURCE_ROLES, Document, Section

_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}


class PubMedClient:
    def __init__(self, cfg: PubMedCfg, store: RawStore, workers: int = 1) -> None:
        self.cfg = cfg
        self.store = store
        self.workers = cfg.in_flight(workers)  # NCBI: config caps in-flight requests below the default
        self.http = RateLimitedClient("pubmed", cfg.min_interval_s, max_in_flight=self.workers)

    def _params(self, **kw) -> dict:
        p = {"db": "pubmed", "tool": self.cfg.tool, **kw}
        if email := os.environ.get("NCBI_EMAIL", "").strip():
            p["email"] = email  # only if the owner set it; redacted in raw metadata
        if key := os.environ.get("NCBI_API_KEY", "").strip():
            p["api_key"] = key
        return p

    def query(self, condition_key: str, extra: str = "") -> str:
        return f"{self.cfg.conditions[condition_key]} {self.cfg.common_filter} {extra}".strip()

    def count(self, term: str, cache_name: str) -> int:
        body = cached_get(self.http, self.store, f"count_{cache_name}", "json",
                          f"{self.cfg.base_url}/esearch.fcgi",
                          self._params(term=term, retmax=0, retmode="json"))
        return int(json.loads(body)["esearchresult"]["count"])

    def count_many(self, queries: list[tuple[str, str]]) -> list[int]:
        """Concurrent ``count`` for (term, cache_name) pairs; order preserved."""
        return parallel_map(lambda q: self.count(*q), queries, self.workers)

    def search_ids(self, term: str, retmax: int, cache_name: str) -> list[str]:
        body = cached_get(self.http, self.store, f"search_{cache_name}", "json",
                          f"{self.cfg.base_url}/esearch.fcgi",
                          self._params(term=term, retmax=retmax, sort=self.cfg.sample_sort, retmode="json"))
        return json.loads(body)["esearchresult"]["idlist"]

    ESEARCH_MAX = 9999  # PubMed esearch cannot page beyond 10,000 UIDs per query

    def all_ids_by_year(self, condition_key: str, years: range, cache_prefix: str) -> dict[int, list[str]]:
        """Every PMID for a condition, retrieved one publication year at a time.

        PubMed's esearch returns at most ~10k UIDs per query, so the full population is retrieved in
        yearly slices. Each slice's reported count is checked against the IDs actually returned:
        a slice at or above the cap raises instead of silently truncating.
        """
        def one(year: int) -> tuple[int, list[str]]:
            term = self.query(condition_key, f'AND ("{year}"[dp])')
            body = cached_get(self.http, self.store, f"search_{cache_prefix}_{condition_key}_{year}", "json",
                              f"{self.cfg.base_url}/esearch.fcgi",
                              self._params(term=term, retmax=self.ESEARCH_MAX, retmode="json"))
            res = json.loads(body)["esearchresult"]
            ids, count = res["idlist"], int(res["count"])
            if count > self.ESEARCH_MAX:
                raise RuntimeError(f"{condition_key} {year}: {count} hits exceed the esearch cap; split further")
            if len(ids) != count:
                raise RuntimeError(f"{condition_key} {year}: count {count} but {len(ids)} IDs returned")
            return year, ids

        return dict(parallel_map(one, years, self.workers))

    def fetch_xml(self, pmids: list[str], cache_name: str) -> list[bytes]:
        """efetch in batches (fetched concurrently); one raw XML payload per batch, in order."""
        starts = range(0, len(pmids), self.cfg.efetch_batch)

        def one(i: int) -> bytes:
            batch = pmids[i:i + self.cfg.efetch_batch]
            return cached_get(self.http, self.store, f"efetch_{cache_name}_{i // self.cfg.efetch_batch:04d}",
                              "xml", f"{self.cfg.base_url}/efetch.fcgi",
                              self._params(id=",".join(batch), retmode="xml"))

        return parallel_map(one, starts, self.workers)


def _text(el: ET.Element | None) -> str:
    return " ".join("".join(el.itertext()).split()) if el is not None else ""


def _parse_date(article: ET.Element, medline: ET.Element) -> tuple[date | None, str]:
    """Best publication date: electronic ArticleDate (day precision) > journal PubDate."""
    ad = article.find("ArticleDate")
    if ad is not None and ad.findtext("Year"):
        try:
            return date(int(ad.findtext("Year")), int(ad.findtext("Month") or 1), int(ad.findtext("Day") or 1)), "day"
        except ValueError:
            pass
    pd_ = article.find("Journal/JournalIssue/PubDate")
    if pd_ is None:
        return None, "unknown"
    year = pd_.findtext("Year")
    if not year:
        medline_date = pd_.findtext("MedlineDate") or ""  # e.g. "1998 Nov-Dec"
        year = medline_date[:4] if medline_date[:4].isdigit() else None
        if not year:
            return None, "unknown"
        return date(int(year), 1, 1), "year"
    month_raw = (pd_.findtext("Month") or "").strip().lower()
    month = int(month_raw) if month_raw.isdigit() else _MONTHS.get(month_raw[:3])
    day = pd_.findtext("Day")
    if month and day and day.isdigit():
        return date(int(year), month, int(day)), "day"
    if month:
        return date(int(year), month, 1), "month"
    return date(int(year), 1, 1), "year"


def parse_pubmed_xml(xml_bytes: bytes, *, snapshot: str, retrieved_at: datetime | None = None,
                     condition: str | None = None) -> list[Document]:
    root = ET.fromstring(xml_bytes)
    retrieved_at = retrieved_at or datetime.now(timezone.utc)
    docs: list[Document] = []
    for art in root.iter("PubmedArticle"):
        medline = art.find("MedlineCitation")
        article = medline.find("Article")
        pmid = medline.findtext("PMID")

        sections = []
        abstract_source = "Abstract"
        abstract_els = article.findall("Abstract/AbstractText")
        if not abstract_els:
            # Some records carry the English abstract only in <OtherAbstract Language="eng">
            # (found in EXP-01-001: 10/494 sample records were otherwise empty).
            for other in medline.findall("OtherAbstract"):
                if (other.get("Language") or "").lower() == "eng":
                    abstract_els = other.findall("AbstractText")
                    abstract_source = "OtherAbstract"
                    break
        for ab in abstract_els:
            txt = _text(ab)
            if txt:
                label = ab.get("Label") or ab.get("NlmCategory") or "ABSTRACT"
                sections.append(Section(name=label.upper(), text=txt))
        abstract = " ".join(s.text for s in sections)

        pub_types = [_text(p) for p in article.findall("PublicationTypeList/PublicationType")]
        mesh = []
        for mh in medline.findall("MeshHeadingList/MeshHeading"):
            desc = _text(mh.find("DescriptorName"))
            quals = [_text(q) for q in mh.findall("QualifierName")]
            mesh.append(desc + ("/" + "/".join(quals) if quals else ""))
        ids = {}
        for aid in art.findall("PubmedData/ArticleIdList/ArticleId"):
            if aid.get("IdType") in {"doi", "pmc"} and aid.text:
                ids[aid.get("IdType").replace("pmc", "pmcid")] = aid.text.strip()
        ids["pmid"] = pmid

        # Retraction / correction signals: a retracted paper must never support a claim.
        corrections = [c.get("RefType") for c in medline.findall("CommentsCorrectionsList/CommentsCorrections")]
        pub_date, precision = _parse_date(article, medline)

        docs.append(Document(
            doc_id=f"pubmed:{pmid}", source="pubmed", source_id=pmid, role=SOURCE_ROLES["pubmed"],
            title=_text(article.find("ArticleTitle")), text=abstract, sections=sections,
            pub_date=pub_date, date_precision=precision, retrieved_at=retrieved_at, snapshot=snapshot,
            publication_types=pub_types, mesh_terms=mesh,
            keywords=[_text(k) for k in medline.findall("KeywordList/Keyword")],
            journal=_text(article.find("Journal/Title")) or None,
            language=article.findtext("Language"),
            license="NLM PubMed terms; abstract copyright may rest with publisher (not redistributed)",
            url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/", identifiers=ids,
            conditions=[condition] if condition else [],
            extra={
                "is_retracted": "Retracted Publication" in pub_types or "RetractionIn" in corrections,
                "has_correction": any(c in {"ErratumIn", "ExpressionOfConcernIn"} for c in corrections),
                "structured_abstract": any(s.name != "ABSTRACT" for s in sections),
                "abstract_source": abstract_source if sections else None,
            },
        ))
    return docs
