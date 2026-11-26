"""Validate the CliniVer environment and write a JSON report.

Run with the venv interpreter:

    .venv\\Scripts\\python.exe scripts\\validate_environment.py            # Windows
    .venv/bin/python scripts/validate_environment.py --expect-device cpu

Every check reports PASS / WARN / FAIL. Exit code is 1 if any check FAILs.
The report goes to ``artifacts/environment/validation_<timestamp>.json`` and records the
facts behind the claim "this environment works".
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Check:
    name: str
    status: str = "PASS"  # PASS | WARN | FAIL
    detail: str = ""
    data: dict[str, Any] = field(default_factory=dict)


def _requirement_names(req_file: Path) -> list[str]:
    names = []
    for line in req_file.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        names.append(re.split(r"[<>=!~\[; ]", line, maxsplit=1)[0])
    return names


# ---------------------------------------------------------------- checks
def check_python() -> Check:
    v = sys.version_info
    c = Check("python_version", data={"version": sys.version.split()[0], "executable": sys.executable})
    if (v.major, v.minor) == (3, 12):
        c.detail = "Python 3.12 (validated version)"
    elif (v.major, v.minor) == (3, 11):
        c.status, c.detail = "WARN", "Python 3.11 is the fallback; 3.12 is the validated version"
    else:
        c.status, c.detail = "FAIL", "Unsupported Python version (need 3.12, or 3.11 as fallback)"
    if ".venv" not in sys.executable:
        c.status = "WARN" if c.status == "PASS" else c.status
        c.detail += "; not running inside the project .venv"
    return c


def check_packages() -> Check:
    from importlib import metadata

    names = ["torch", *_requirement_names(ROOT / "requirements.txt")]
    versions, missing = {}, []
    for n in names:
        try:
            versions[n] = metadata.version(n)
        except metadata.PackageNotFoundError:
            missing.append(n)
    try:
        versions["cliniver"] = metadata.version("cliniver")
    except metadata.PackageNotFoundError:
        missing.append("cliniver (editable install)")
    c = Check("packages", data={"versions": versions, "missing": missing})
    if missing:
        c.status, c.detail = "FAIL", f"Missing: {', '.join(missing)}"
    else:
        c.detail = f"{len(versions)} packages installed"
    return c


def check_project_config() -> Check:
    from cliniver.config import load_project_config

    cfg = load_project_config()
    return Check("project_config", detail=f"configs/project.yaml valid (phase {cfg.project.phase}, seed {cfg.reproducibility.seed})",
                 data={"seed": cfg.reproducibility.seed, "device_pref": cfg.hardware.device})


def check_torch_cuda(expect_device: str) -> Check:
    import torch

    data: dict[str, Any] = {
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    c = Check("torch_cuda", data=data)
    if not torch.cuda.is_available():
        if expect_device == "cuda":
            c.status, c.detail = "FAIL", "CUDA expected but torch.cuda.is_available() is False"
        else:
            c.status, c.detail = "WARN", "Running on CPU (CUDA not available)"
        return c

    props = torch.cuda.get_device_properties(0)
    capability = f"sm_{props.major}{props.minor}"
    arch_list = torch.cuda.get_arch_list()
    data.update({
        "gpu": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "total_vram_gb": round(props.total_memory / 1024**3, 2),
        "compiled_arch_list": arch_list,
        "cudnn": torch.backends.cudnn.version(),
        "bf16_supported": torch.cuda.is_bf16_supported(),
    })
    # The GPU's architecture must be in the compiled list, otherwise kernels fail at runtime.
    compatible = capability in arch_list or any(
        a.startswith("compute_") and int(a.split("_")[1]) <= props.major * 10 + props.minor for a in arch_list
    )
    if not compatible:
        c.status, c.detail = "FAIL", f"{capability} not in this PyTorch build's arch list {arch_list}"
    else:
        c.detail = f"{props.name} ({capability}), CUDA build {torch.version.cuda}"
    return c


def check_gpu_compute() -> Check:
    """Real kernels on the GPU: correctness against CPU, half precision, and an allocation test."""
    import torch

    if not torch.cuda.is_available():
        return Check("gpu_compute", status="WARN", detail="Skipped: no CUDA device")

    g = torch.Generator().manual_seed(0)
    a = torch.randn(1024, 1024, generator=g)
    b = torch.randn(1024, 1024, generator=g)
    ref = a @ b
    gpu = (a.cuda() @ b.cuda()).cpu()
    # TF32 may be enabled for matmuls, so compare with a relative tolerance.
    max_rel_err = ((gpu - ref).abs().max() / ref.abs().max()).item()

    half_ok = True
    for dtype in (torch.float16, torch.bfloat16):
        try:
            x = torch.randn(512, 512, device="cuda", dtype=dtype)
            _ = (x @ x).sum().item()
        except RuntimeError:
            half_ok = False

    # Throughput probe (informational; this measurement is not used as a project result)
    n = 4096
    x = torch.randn(n, n, device="cuda", dtype=torch.float16)
    torch.cuda.synchronize()
    for _ in range(3):
        x @ x
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    iters = 20
    for _ in range(iters):
        x @ x
    torch.cuda.synchronize()
    tflops = 2 * n**3 * iters / (time.perf_counter() - t0) / 1e12

    # Allocation test: 2 GiB block, then release.
    free_before, total = torch.cuda.mem_get_info()
    block = torch.empty(2 * 1024**3, dtype=torch.uint8, device="cuda")
    peak = torch.cuda.max_memory_allocated()
    del block, x
    torch.cuda.empty_cache()

    c = Check("gpu_compute", data={
        "fp32_matmul_max_rel_err_vs_cpu": max_rel_err,
        "fp16_bf16_matmul_ok": half_ok,
        "fp16_matmul_tflops_measured": round(tflops, 1),
        "vram_free_gb_before_alloc": round(free_before / 1024**3, 2),
        "vram_total_gb": round(total / 1024**3, 2),
        "alloc_2gib_ok": True,
        "peak_allocated_gb": round(peak / 1024**3, 2),
    })
    if max_rel_err > 1e-2 or not half_ok:
        c.status, c.detail = "FAIL", "GPU results differ from CPU, or half-precision kernels failed"
    else:
        c.detail = f"matmul correct (rel err {max_rel_err:.1e}), fp16/bf16 OK, 2 GiB alloc OK"
    return c


def check_seeding() -> Check:
    import torch

    from cliniver.utils.seed import set_seed

    set_seed(123)
    a = torch.rand(5).tolist()
    set_seed(123)
    b = torch.rand(5).tolist()
    ok = a == b
    return Check("seeding", status="PASS" if ok else "FAIL",
                 detail="set_seed reproduces identical random draws" if ok else "Seeding not reproducible")


def check_disk() -> Check:
    usage = shutil.disk_usage(ROOT)
    free_gb = usage.free / 1024**3
    c = Check("disk_space", data={"free_gb": round(free_gb, 1)})
    # Expected need: corpus + indexes + several GGUF/HF models (~30-60 GB).
    if free_gb < 30:
        c.status, c.detail = "FAIL", f"Only {free_gb:.0f} GB free; at least 30 GB needed"
    elif free_gb < 80:
        c.status, c.detail = "WARN", f"{free_gb:.0f} GB free; 80+ GB recommended"
    else:
        c.detail = f"{free_gb:.0f} GB free"
    return c


def check_env_file() -> Check:
    import os

    from dotenv import load_dotenv

    env = ROOT / ".env"
    if not env.exists():
        return Check("env_file", status="WARN", detail=".env missing (copy .env.example). Needed before Notebook 01")
    load_dotenv(env, override=False)
    # NCBI credentials are optional: without a key, PubMed runs in keyless mode (paced at 2 req/s,
    # below NCBI's 3 req/s keyless limit; see EXP-01-002). Report the mode, do not warn.
    has_key = bool(os.environ.get("NCBI_API_KEY"))
    return Check("env_file", detail=".env present; NCBI mode: " + ("API key" if has_key else "keyless (supported)"),
                 data={"ncbi_api_key": has_key, "ncbi_email": bool(os.environ.get("NCBI_EMAIL"))})


# ---------------------------------------------------------------- driver
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--expect-device", choices=["auto", "cuda", "cpu"], default="auto")
    args = parser.parse_args()

    checks: list[Callable[[], Check]] = [
        check_python, check_packages, check_project_config,
        lambda: check_torch_cuda(args.expect_device), check_gpu_compute,
        check_seeding, check_disk, check_env_file,
    ]
    results: list[Check] = []
    for fn in checks:
        try:
            results.append(fn())
        except Exception as exc:  # a crashing check is a FAIL, never a silent skip
            name = getattr(fn, "__name__", "check")
            results.append(Check(name, status="FAIL", detail=f"{type(exc).__name__}: {exc}",
                                 data={"traceback": traceback.format_exc()}))

    hardware: dict[str, Any] = {}
    try:
        from cliniver.utils.hardware import collect_hardware_info

        hardware = collect_hardware_info().to_dict()
    except Exception as exc:
        hardware = {"error": repr(exc)}

    width = max(len(r.name) for r in results)
    print("\nCliniVer environment validation")
    print("-" * 72)
    for r in results:
        print(f"{r.status:<5} {r.name:<{width}}  {r.detail}")
    print("-" * 72)

    overall = "FAIL" if any(r.status == "FAIL" for r in results) else (
        "WARN" if any(r.status == "WARN" for r in results) else "PASS")
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "overall": overall,
        "hardware": hardware,
        "checks": [r.__dict__ for r in results],
    }
    out_dir = ROOT / "artifacts" / "environment"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"validation_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Overall: {overall}   report: {out.relative_to(ROOT)}")
    return 1 if overall == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main())
