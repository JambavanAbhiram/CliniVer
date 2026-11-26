from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from cliniver.config import load_project_config
from cliniver.utils.paths import project_root


def _write(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "project.yaml"
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    return p


@pytest.fixture
def base_config() -> dict:
    with (project_root() / "configs" / "project.yaml").open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def test_repository_config_is_valid():
    cfg = load_project_config()
    assert cfg.project.name == "CliniVer"
    assert cfg.reproducibility.seed >= 0
    assert cfg.domain.conditions, "domain scope (ADR-0001) must not be empty"


def test_paths_resolve_inside_project_root():
    cfg = load_project_config()
    raw = cfg.paths.absolute("data_raw")
    assert raw.is_absolute()
    assert project_root() in raw.parents


def test_unknown_path_entry_rejected():
    with pytest.raises(KeyError):
        load_project_config().paths.absolute("not_a_path")


def test_unknown_key_rejected(tmp_path, base_config):
    base_config["reproducibility"]["sead"] = 1  # typo must fail loudly
    with pytest.raises(ValidationError):
        load_project_config(_write(tmp_path, base_config))


def test_invalid_device_rejected(tmp_path, base_config):
    base_config["hardware"]["device"] = "tpu"
    with pytest.raises(ValidationError):
        load_project_config(_write(tmp_path, base_config))


def test_env_override_device(tmp_path, base_config, monkeypatch):
    monkeypatch.setenv("CLINIVER_DEVICE", "CPU")
    cfg = load_project_config(_write(tmp_path, base_config))
    assert cfg.hardware.device == "cpu"


def test_empty_env_value_does_not_override(tmp_path, base_config, monkeypatch):
    # Regression: an example/default value in .env must not mask the YAML setting.
    base_config["hardware"]["device"] = "cpu"
    monkeypatch.setenv("CLINIVER_DEVICE", "  ")
    assert load_project_config(_write(tmp_path, base_config)).hardware.device == "cpu"


def test_invalid_env_override_rejected(tmp_path, base_config, monkeypatch):
    monkeypatch.setenv("CLINIVER_DEVICE", "tpu")
    with pytest.raises(ValidationError):
        load_project_config(_write(tmp_path, base_config))


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_project_config(tmp_path / "nope.yaml")


def test_config_is_immutable():
    cfg = load_project_config()
    with pytest.raises(ValidationError):
        cfg.reproducibility.seed = 7  # type: ignore[misc]


def test_processing_config_loads_and_is_strict(tmp_path):
    import yaml
    from cliniver.config.processing_config import load_processing_config
    cfg = load_processing_config()
    assert cfg.primary_metric.startswith("hit@") and len(cfg.retriever.dense_revision) == 40
    raw = yaml.safe_load((Path(__file__).parents[2] / "configs" / "processing.yaml").read_text(encoding="utf-8"))
    raw["unexpected"] = 1
    p = tmp_path / "p.yaml"; p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(Exception):
        load_processing_config(p)
