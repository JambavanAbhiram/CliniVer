"""Seeding helpers. Every experiment must call ``set_seed`` and log the seed it used."""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int, deterministic_torch: bool = False) -> int:
    """Seed Python, NumPy and (if installed) PyTorch. Returns the seed for logging.

    ``deterministic_torch=True`` requests deterministic GPU kernels. This is slower and some
    operations will raise an error instead of running non-deterministically.
    """
    if seed < 0:
        raise ValueError("seed must be non-negative")
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)  # affects subprocesses only; documented for completeness

    try:
        import torch
    except ImportError:
        return seed

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        # Required by cuBLAS for deterministic matmuls on CUDA >= 10.2.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
    return seed
