"""Read/write normalized records as JSONL (one record per line; human-inspectable, diff-friendly)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def write_jsonl(records: Iterable[BaseModel], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(r.model_dump_json() + "\n")
            n += 1
    return n


def read_jsonl(path: Path, model: type[T]) -> list[T]:
    with path.open(encoding="utf-8") as fh:
        return [model.model_validate_json(line) for line in fh if line.strip()]
