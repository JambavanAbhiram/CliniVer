"""Typed model for ``configs/service.yaml`` (the local demo / expert-review service; runtime settings only)."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from cliniver.utils.paths import project_root, resolve


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ReviewCfg(_Strict):
    sample_size: int = Field(ge=1)
    verdicts_file: str
    baseline_verdicts_file: str
    annotations_file: str


class ServiceConfig(_Strict):
    snapshot: str
    seed: int
    host: str
    port: int = Field(ge=1024, le=65535)
    llm_model: str
    configuration_source: str
    evidence_top_k: int = Field(ge=1, le=10)
    generation_passages: int = Field(ge=1, le=10)
    review: ReviewCfg
    max_answer_chars: int = Field(ge=200)


def load_service_config(path: str | Path | None = None) -> ServiceConfig:
    p = resolve(path) if path else project_root() / "configs" / "service.yaml"
    cfg = ServiceConfig.model_validate(yaml.safe_load(p.read_text(encoding="utf-8")))
    if cfg.host not in ("127.0.0.1", "localhost"):
        raise ValueError("the research demo binds to localhost only")
    return cfg
