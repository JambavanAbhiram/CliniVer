"""Unified records for the multi-source corpus (ADR-0002).

Two record kinds are deliberately separate:

* ``Document``: text that can serve as *evidence* (or, for WHO, population *statistics*).
* ``Concept``: identity / normalization data (RxNorm, PubChem, RxClass) used for entity
  linking and the concept graph (ADR-0003). It is **never** evidence for a clinical claim.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Source = Literal["pubmed", "europepmc_oa", "openfda_label", "clinicaltrials", "who_gho"]
SourceRole = Literal["evidence", "regulatory_evidence", "trial_registry", "statistics"]

SOURCE_ROLES: dict[str, SourceRole] = {
    "pubmed": "evidence",
    "europepmc_oa": "evidence",
    "openfda_label": "regulatory_evidence",
    "clinicaltrials": "trial_registry",
    "who_gho": "statistics",
}


class Section(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str  # e.g. "BACKGROUND", "contraindications", "brief_summary"
    text: str


class Document(BaseModel):
    """One evidence document. ``text`` is the concatenation of ``sections`` (or the abstract)."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str  # "<source>:<source_id>", globally unique
    source: Source
    source_id: str
    role: SourceRole
    title: str
    text: str
    sections: list[Section] = Field(default_factory=list)
    pub_date: date | None = None  # best available date; precision recorded in date_precision
    date_precision: Literal["day", "month", "year", "unknown"] = "unknown"
    retrieved_at: datetime
    snapshot: str  # acquisition snapshot id (YYYY-MM-DD), shown in reports (ADR-0005)
    publication_types: list[str] = Field(default_factory=list)
    mesh_terms: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    journal: str | None = None
    language: str | None = None
    license: str | None = None  # license / terms identifier as reported by the source
    url: str | None = None
    identifiers: dict[str, str | list[str]] = Field(default_factory=dict)  # pmid, doi, pmcid, nct_id, set_id, rxcui…
    conditions: list[str] = Field(default_factory=list)  # domain condition keys this doc was retrieved for
    extra: dict[str, Any] = Field(default_factory=dict)  # source-specific fields kept for analysis

    @field_validator("doc_id")
    @classmethod
    def _doc_id_format(cls, v: str) -> str:
        if ":" not in v:
            raise ValueError("doc_id must be '<source>:<source_id>'")
        return v


class Concept(BaseModel):
    """Normalization record (drug / class / synonym set). Not evidence."""

    model_config = ConfigDict(extra="forbid")

    concept_id: str  # e.g. "rxnorm:6809", "pubchem:4091", "atc:A10BA02"
    source: Literal["rxnorm", "rxclass", "pubchem"]
    kind: Literal["drug", "drug_class"]
    name: str
    query_name: str | None = None  # the domain drug name that led to this record
    synonyms: list[str] = Field(default_factory=list)
    xrefs: dict[str, list[str]] = Field(default_factory=dict)  # e.g. {"brand_rxcui": [...], "class_ids": [...]}
    retrieved_at: datetime
    snapshot: str
    extra: dict[str, Any] = Field(default_factory=dict)


class EntityAnnotation(BaseModel):
    """A PubTator 3 entity mention inside a PubMed document (graph construction input)."""

    model_config = ConfigDict(extra="forbid")

    pmid: str
    passage_type: str  # "title" | "abstract" | ...
    entity_type: str  # Chemical, Disease, Gene, Species, ...
    identifier: str | None  # e.g. "MESH:D008687"
    text: str
    offset: int
    length: int


class RelationAnnotation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pmid: str
    relation_type: str  # e.g. "Treat", "Negative_Correlation"
    entity1: str
    entity2: str
    score: float | None = None
