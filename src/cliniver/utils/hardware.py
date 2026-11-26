"""Hardware and software introspection.

Used by ``scripts/validate_environment.py`` and recorded with every experiment, so that
any reported latency or memory figure can be traced to the machine that produced it.
"""

from __future__ import annotations

import platform
import sys
from dataclasses import asdict, dataclass, field
from importlib import metadata
from typing import Any

import psutil


@dataclass(frozen=True)
class GPUInfo:
    index: int
    name: str
    compute_capability: str
    total_vram_gb: float


@dataclass(frozen=True)
class HardwareInfo:
    python_version: str
    platform: str
    processor: str
    cpu_logical_cores: int
    ram_total_gb: float
    torch_version: str | None
    torch_cuda_build: str | None
    cuda_available: bool
    cudnn_version: int | None
    gpus: list[GPUInfo] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def collect_hardware_info() -> HardwareInfo:
    """Snapshot of the current machine. Never raises if PyTorch is missing."""
    torch_version = torch_cuda = cudnn = None
    cuda_available = False
    gpus: list[GPUInfo] = []
    try:
        import torch

        torch_version = torch.__version__
        torch_cuda = torch.version.cuda
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            cudnn = torch.backends.cudnn.version()
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gpus.append(
                    GPUInfo(
                        index=i,
                        name=props.name,
                        compute_capability=f"{props.major}.{props.minor}",
                        total_vram_gb=round(props.total_memory / 1024**3, 2),
                    )
                )
    except ImportError:
        pass

    return HardwareInfo(
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        processor=platform.processor(),
        cpu_logical_cores=psutil.cpu_count(logical=True) or 0,
        ram_total_gb=round(psutil.virtual_memory().total / 1024**3, 2),
        torch_version=torch_version,
        torch_cuda_build=torch_cuda,
        cuda_available=cuda_available,
        cudnn_version=cudnn,
        gpus=gpus,
    )


def resolve_device(preference: str = "auto") -> str:
    """Map a config preference (auto|cuda|cpu) to an actual torch device string.

    ``cuda`` requested but unavailable is an error rather than a silent CPU fallback,
    because it would invalidate latency measurements without anyone noticing.
    """
    preference = preference.lower()
    if preference not in {"auto", "cuda", "cpu"}:
        raise ValueError(f"Unknown device preference: {preference!r}")
    if preference == "cpu":
        return "cpu"
    try:
        import torch

        has_cuda = torch.cuda.is_available()
    except ImportError:
        has_cuda = False
    if preference == "cuda" and not has_cuda:
        raise RuntimeError("Device 'cuda' requested but CUDA is not available")
    return "cuda" if has_cuda else "cpu"


def package_versions(names: list[str]) -> dict[str, str | None]:
    """Installed version for each distribution name (``None`` if not installed)."""
    out: dict[str, str | None] = {}
    for name in names:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out
