"""Typed model for ``configs/processing.yaml`` (Notebook 02)."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from cliniver.utils.paths import project_root, resolve


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RetrieverCfg(_Strict):
    dense_model: str
    dense_revision: str = Field(min_length=40, max_length=40)
    query_prefix: str
    batch_size: int = Field(ge=1)


class BenchmarkCfg(_Strict):
    name: str
    url: str
    dev_fraction: float = Field(gt=0, lt=1)
    top_k: int = Field(ge=10)


class GridCfg(_Strict):
    fixed_sizes: list[int]
    fixed_overlaps: list[float]
    structured_sizes: list[int]
    semantic_pcts: list[float]
    header_variants: bool


class QualifierCfg(_Strict):
    validation_sample: int = Field(ge=10)
    max_words: int = Field(ge=3)


class FinalCfg(_Strict):
    max_tokens_whole_doc: int = Field(ge=64)


class ProcessingConfig(_Strict):
    corpus_snapshot: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    retriever: RetrieverCfg
    benchmark: BenchmarkCfg
    grid: GridCfg
    budgets: list[int]
    primary_metric: str
    qualifiers: QualifierCfg
    final: FinalCfg


def load_processing_config(path: str | Path | None = None) -> ProcessingConfig:
    p = resolve(path) if path else project_root() / "configs" / "processing.yaml"
    with p.open(encoding="utf-8") as fh:
        return ProcessingConfig.model_validate(yaml.safe_load(fh))
