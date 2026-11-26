"""Typed model for ``configs/data.yaml`` (Notebook 01)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from cliniver.utils.paths import project_root, resolve


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class _Rated(_Strict):
    base_url: str
    min_interval_s: float = Field(gt=0)
    # Optional per-provider cap on concurrent requests; None = use concurrency.workers_per_source.
    max_in_flight: int | None = Field(default=None, ge=1, le=8)

    def in_flight(self, default: int) -> int:
        return self.max_in_flight or default


class PubMedCfg(_Rated):
    efetch_batch: int = Field(ge=1, le=500)
    tool: str
    common_filter: str
    conditions: dict[str, str]
    publication_type_strata: dict[str, str]
    decades: list[tuple[int, int]]
    sample_per_condition: int = Field(ge=1)
    sample_sort: Literal["relevance", "pub_date", "first_author"]


class OpenFDACfg(_Rated):
    labels_per_drug: int = Field(ge=1, le=100)
    sample_drugs_per_condition: int = Field(ge=1)
    sections: list[str]


class ClinicalTrialsCfg(_Rated):
    sample_per_condition: int = Field(ge=1, le=1000)
    conditions: dict[str, str]
    intervention_type: str


class PubTatorCfg(_Rated):
    batch: int = Field(ge=1, le=100)


class RxNavCfg(_Rated):
    class_sources: list[str]


class PubChemCfg(_Rated):
    max_synonyms: int = Field(ge=1)


class EuropePMCCfg(_Rated):
    licenses: list[str]
    sample_per_condition: int = Field(ge=1)
    condition_queries: dict[str, str]


class WhoGhoCfg(_Rated):
    indicator_keywords: list[str]
    spatial_types: list[str]


class ConcurrencyCfg(_Strict):
    sources_in_parallel: int = Field(ge=1, le=16)
    workers_per_source: int = Field(ge=1, le=8)


class StageBCfg(_Strict):
    pubmed_years: tuple[int, int]
    openfda_rx_only: bool
    openfda_max_labels_per_drug: int = Field(ge=1, le=1000)
    near_dup_threshold: float = Field(gt=0, le=1)
    near_dup_min_words: int = Field(ge=0)
    clinicaltrials_page_size: int = Field(ge=1, le=1000)
    europepmc_year_strata: list[tuple[int, int]]
    europepmc_per_stratum: int = Field(ge=1, le=1000)
    exclude_retracted: bool


class DataConfig(_Strict):
    snapshot: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    stage: Literal["A", "B"]
    stage_b: StageBCfg
    concurrency: ConcurrencyCfg
    pubmed: PubMedCfg
    drugs: dict[str, list[str]]
    openfda: OpenFDACfg
    clinicaltrials: ClinicalTrialsCfg
    pubtator: PubTatorCfg
    rxnav: RxNavCfg
    pubchem: PubChemCfg
    europepmc: EuropePMCCfg
    who_gho: WhoGhoCfg

    @model_validator(mode="after")
    def _same_condition_keys(self) -> "DataConfig":
        keys = set(self.pubmed.conditions)
        for name, other in [("drugs", self.drugs), ("clinicaltrials.conditions", self.clinicaltrials.conditions),
                            ("europepmc.condition_queries", self.europepmc.condition_queries)]:
            if set(other) != keys:
                raise ValueError(f"{name} condition keys {sorted(other)} != pubmed.conditions {sorted(keys)}")
        return self

    def all_drugs(self) -> list[tuple[str, str]]:
        """(condition, drug) pairs; a drug listed under two conditions appears twice."""
        return [(c, d) for c, ds in self.drugs.items() for d in ds]


def load_data_config(path: str | Path | None = None) -> DataConfig:
    p = resolve(path) if path else project_root() / "configs" / "data.yaml"
    with p.open(encoding="utf-8") as fh:
        return DataConfig.model_validate(yaml.safe_load(fh))
