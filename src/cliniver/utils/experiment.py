"""Experiment records: one folder per experiment under ``artifacts/experiments/<exp_id>/``.

Every result a notebook reports must come from a file written here, so the numbers in
``docs/experiment_log.md`` can always be traced to code output rather than hand-typing.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from cliniver.utils.hardware import collect_hardware_info, package_versions
from cliniver.utils.paths import project_root


def _git_state() -> dict[str, Any]:
    try:
        root = project_root()
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True)
        return {"commit": head.stdout.strip() or None, "dirty": bool(dirty.stdout.strip())}
    except OSError:
        return {"commit": None, "dirty": None}


def _jsonable(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (datetime,)):
        return obj.isoformat()
    if type(obj).__module__ == "numpy" and hasattr(obj, "item"):   # numpy scalars (bool_, int64, float32 …) → Python values
        return obj.item()
    raise TypeError(f"Not JSON serialisable: {type(obj)}")


@dataclass
class Experiment:
    exp_id: str  # EXP-<notebook>-<nnn>
    title: str
    research_question: str
    config: dict[str, Any]
    seed: int | None = None
    packages: list[str] = field(default_factory=list)
    root: Path | None = None

    def __post_init__(self) -> None:
        self.dir = (self.root or project_root() / "artifacts" / "experiments") / self.exp_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.started_at = datetime.now(timezone.utc)
        self.results: dict[str, Any] = {}
        self._write_record(status="running")

    def _write_record(self, status: str) -> None:
        record = {
            "exp_id": self.exp_id,
            "title": self.title,
            "research_question": self.research_question,
            "status": status,
            "started_at": self.started_at.isoformat(timespec="seconds"),
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "seed": self.seed,
            "config": self.config,
            "git": _git_state(),
            "hardware": collect_hardware_info().to_dict(),
            "packages": package_versions(self.packages),
        }
        (self.dir / "record.json").write_text(json.dumps(record, indent=2, default=_jsonable), encoding="utf-8")

    def log_result(self, key: str, value: Any) -> None:
        """Add a scalar / dict result and persist immediately (results survive a crashed run)."""
        self.results[key] = value
        (self.dir / "results.json").write_text(json.dumps(self.results, indent=2, default=_jsonable),
                                               encoding="utf-8")

    def save_table(self, name: str, df: pd.DataFrame) -> Path:
        path = self.dir / f"{name}.csv"
        df.to_csv(path, index=True)
        return path

    def save_figure(self, name: str, fig: Any) -> Path:
        path = self.dir / f"{name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        return path

    def finish(self) -> None:
        self._write_record(status="completed")
