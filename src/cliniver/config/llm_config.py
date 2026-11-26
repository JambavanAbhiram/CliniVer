"""Typed models for ``configs/llm.yaml`` (runtime + pinned models) and ``configs/claims.yaml`` (NB03 experiment)."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from cliniver.utils.paths import project_root, resolve


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RuntimeCfg(_Strict):
    name: str
    release: str
    dir: str
    sha256: dict[str, str]
    host: str
    port: int
    ctx_size: int = Field(ge=1024)
    gpu_layers: int
    vram_budget_gb: float
    extra_args: list[str] = Field(default_factory=list)


class ModelCfg(_Strict):
    repo: str
    revision: str = Field(min_length=40, max_length=40)
    file: str
    license: str
    role: str


class LLMConfig(_Strict):
    runtime: RuntimeCfg
    models_dir: str
    models: dict[str, ModelCfg]

    def model_path(self, key: str) -> Path:
        return project_root() / self.models_dir / key / self.models[key].file


class GenerationCfg(_Strict):
    generators: list[str]
    temperature: float
    top_p: float
    max_tokens: int
    system_prompt: str


class ExtractionCfg(_Strict):
    candidates: list[str]
    dev_size: int
    min_json_validity: float
    selection_metric: str
    nli_model: str
    nli_revision: str = Field(min_length=40, max_length=40)
    entail_threshold: float


class ClaimsConfig(_Strict):
    snapshot: str
    seed: int
    datasets: dict[str, dict]
    generation: GenerationCfg
    extraction: ExtractionCfg
    h2: dict[str, float]
    nli_spot_check: dict[str, int]


def load_llm_config(path: str | Path | None = None) -> LLMConfig:
    p = resolve(path) if path else project_root() / "configs" / "llm.yaml"
    return LLMConfig.model_validate(yaml.safe_load(p.read_text(encoding="utf-8")))


def load_claims_config(path: str | Path | None = None) -> ClaimsConfig:
    p = resolve(path) if path else project_root() / "configs" / "claims.yaml"
    return ClaimsConfig.model_validate(yaml.safe_load(p.read_text(encoding="utf-8")))
