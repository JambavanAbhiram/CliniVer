"""Project-root discovery so code behaves the same from notebooks, scripts and tests."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

_ROOT_MARKER = "pyproject.toml"


@lru_cache(maxsize=1)
def project_root() -> Path:
    """Return the CliniVer project root.

    Resolution order:
      1. ``CLINIVER_ROOT`` environment variable, if set.
      2. The nearest ancestor of this file that contains ``pyproject.toml``.
      3. The nearest ancestor of the current working directory with that marker
         (covers notebooks opened from ``notebooks/``).
    """
    env_root = os.environ.get("CLINIVER_ROOT")
    if env_root:
        root = Path(env_root).expanduser().resolve()
        if not (root / _ROOT_MARKER).is_file():
            raise FileNotFoundError(f"CLINIVER_ROOT={root} does not contain {_ROOT_MARKER}")
        return root

    for start in (Path(__file__).resolve(), Path.cwd().resolve()):
        for candidate in (start, *start.parents):
            if (candidate / _ROOT_MARKER).is_file() and (candidate / "src" / "cliniver").is_dir():
                return candidate
    raise FileNotFoundError(
        "Could not locate the CliniVer project root (no pyproject.toml with src/cliniver found). "
        "Set CLINIVER_ROOT explicitly."
    )


def resolve(path: str | Path) -> Path:
    """Resolve a project-relative path to an absolute path (absolute paths pass through)."""
    p = Path(path)
    return p if p.is_absolute() else project_root() / p
