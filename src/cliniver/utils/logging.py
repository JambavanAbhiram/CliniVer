"""Consistent logging for notebooks, scripts and (later) the API."""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"
_CONFIGURED = False


def setup_logging(level: str = "INFO", log_dir: Path | None = None, run_name: str = "cliniver") -> Path | None:
    """Configure the ``cliniver`` logger once per process.

    Returns the log-file path when file logging is enabled, else ``None``. Calling this
    again (e.g. re-running a notebook cell) only updates the level; it never duplicates handlers.
    """
    global _CONFIGURED
    logger = logging.getLogger("cliniver")
    logger.setLevel(level.upper())
    if _CONFIGURED:
        return None

    logger.propagate = False
    formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    log_path = None
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{run_name}_{datetime.now():%Y%m%d_%H%M%S}.log"
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    _CONFIGURED = True
    return log_path


def get_logger(name: str) -> logging.Logger:
    """Child logger under the ``cliniver`` namespace, e.g. ``get_logger(__name__)``."""
    return logging.getLogger(name if name.startswith("cliniver") else f"cliniver.{name}")
