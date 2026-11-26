"""Typed, validated access to ``configs/*.yaml``.

Only ``project.yaml`` exists in Phase 1. Component configs (data, models, retrieval,
evaluation) get their own models here when the phase that needs them begins.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

from cliniver.utils.paths import project_root, resolve

Device = Literal["auto", "cuda", "cpu"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]


class _Strict(BaseModel):
    # Unknown keys are an error: a typo in a config must not silently change an experiment.
    model_config = ConfigDict(extra="forbid", frozen=True)


class ProjectInfo(_Strict):
    name: str
    version: str
    phase: int = Field(ge=0, le=12)


class Reproducibility(_Strict):
    seed: int = Field(ge=0)
    deterministic_torch: bool = False


class Paths(_Strict):
    data_raw: Path
    data_interim: Path
    data_processed: Path
    data_benchmark: Path
    data_synthetic: Path
    artifacts: Path
    experiments: Path
    environment_reports: Path
    logs: Path
    reports: Path

    def absolute(self, name: str) -> Path:
        """Absolute path for a configured entry, e.g. ``paths.absolute("data_raw")``."""
        if name not in type(self).model_fields:
            raise KeyError(f"Unknown path entry: {name!r}")
        return resolve(getattr(self, name))


class Hardware(_Strict):
    device: Device = "auto"
    vram_budget_gb: float = Field(gt=0)
    one_model_on_gpu: bool = True


class Logging(_Strict):
    level: LogLevel = "INFO"
    to_file: bool = True


class Domain(_Strict):
    name: str
    conditions: list[str] = Field(min_length=1)


class ProjectConfig(_Strict):
    project: ProjectInfo
    reproducibility: Reproducibility
    paths: Paths
    hardware: Hardware
    logging: Logging
    domain: Domain


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} must contain a YAML mapping at top level")
    return data


def _apply_env_overrides(raw: dict[str, Any]) -> dict[str, Any]:
    """Environment variables override YAML for runtime-only settings (never for experiment params).

    Only non-empty values override. Applied overrides are logged by callers via
    ``ProjectConfig`` values, and overrides are validated like any YAML value.
    """
    if device := os.environ.get("CLINIVER_DEVICE", "").strip():
        raw.setdefault("hardware", {})["device"] = device.lower()
    if level := os.environ.get("CLINIVER_LOG_LEVEL", "").strip():
        raw.setdefault("logging", {})["level"] = level.upper()
    return raw


def load_project_config(path: str | Path | None = None, *, use_dotenv: bool = True) -> ProjectConfig:
    """Load and validate ``configs/project.yaml`` (or an explicit path).

    ``use_dotenv=True`` first loads ``.env`` (without overriding variables already set in the
    process). Tests pass ``use_dotenv=False`` so they do not depend on a developer's ``.env``.
    """
    if use_dotenv:
        load_dotenv(project_root() / ".env", override=False)
    cfg_path = resolve(path) if path else project_root() / "configs" / "project.yaml"
    return ProjectConfig.model_validate(_apply_env_overrides(_read_yaml(cfg_path)))


@lru_cache(maxsize=1)
def get_project_config() -> ProjectConfig:
    """Cached project config for normal use. Tests should call ``load_project_config`` directly."""
    return load_project_config()
