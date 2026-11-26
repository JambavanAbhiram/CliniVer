"""Create and populate the CliniVer virtual environment (Windows / Linux / macOS).

Run with the *system* Python 3.12, from the project root:

    py -3.12 scripts/setup_environment.py            # Windows, auto-detect GPU
    python3.12 scripts/setup_environment.py --device cpu

Steps (each is idempotent, so it is safe to re-run):
  1. Check the interpreter version.
  2. Create ``.venv`` if it does not exist.
  3. Upgrade pip.
  4. Install PyTorch (CUDA or CPU build) from the official PyTorch index.
  5. Install ``requirements.txt``.
  6. Install the ``cliniver`` package in editable mode.
  7. Register a Jupyter kernel named ``cliniver``.
  8. Create ``.env`` from ``.env.example`` if missing.
  9. Run ``scripts/validate_environment.py``.

Standard library only: it must run before any dependency exists.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / ".venv"

# Pinned in one place. Rationale: docs/decisions/ADR-0007-python-and-pytorch-stack.md
TORCH_VERSION = "2.14.1"
TORCH_CUDA_TAG = "cu130"  # CUDA 13.0 wheels; supports Blackwell (sm_120). Needs driver >= CUDA 13.0.
TORCH_INDEX = "https://download.pytorch.org/whl/{tag}"
SUPPORTED_PYTHON = {(3, 11), (3, 12)}
PREFERRED_PYTHON = (3, 12)


def venv_python() -> Path:
    return VENV / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(map(str, cmd))}", flush=True)
    subprocess.run([str(c) for c in cmd], check=True, cwd=ROOT)


def has_nvidia_gpu() -> bool:
    smi = shutil.which("nvidia-smi")
    if not smi:
        return False
    try:
        out = subprocess.run([smi, "-L"], capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return out.returncode == 0 and "GPU" in out.stdout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                        help="Which PyTorch build to install (default: auto-detect an NVIDIA GPU).")
    parser.add_argument("--skip-validate", action="store_true", help="Do not run the validation script at the end.")
    args = parser.parse_args()

    # 1. Interpreter check
    version = sys.version_info[:2]
    if version not in SUPPORTED_PYTHON:
        print(f"ERROR: Python {version[0]}.{version[1]} is not supported. Use 3.12 (preferred) or 3.11.")
        return 1
    if version != PREFERRED_PYTHON:
        print(f"WARNING: Python {version[0]}.{version[1]} works, but the validated version is 3.12.")

    # 2. venv
    if not venv_python().exists():
        run([sys.executable, "-m", "venv", VENV])
    py = venv_python()

    # 3. pip
    run([py, "-m", "pip", "install", "--upgrade", "pip"])

    # 4. PyTorch
    device = args.device
    if device == "auto":
        device = "cuda" if has_nvidia_gpu() else "cpu"
    tag = TORCH_CUDA_TAG if device == "cuda" else "cpu"
    run([py, "-m", "pip", "install", f"torch=={TORCH_VERSION}", "--index-url", TORCH_INDEX.format(tag=tag)])

    # 5-6. Requirements and the package itself
    run([py, "-m", "pip", "install", "-r", ROOT / "requirements.txt"])
    run([py, "-m", "pip", "install", "-e", ROOT])

    # 7. Jupyter kernel (used by VS Code and Jupyter)
    run([py, "-m", "ipykernel", "install", "--user", "--name", "cliniver",
         "--display-name", "CliniVer (Python 3.12)"])

    # 8. .env
    env_file, env_example = ROOT / ".env", ROOT / ".env.example"
    if not env_file.exists() and env_example.exists():
        shutil.copy(env_example, env_file)
        print("\nCreated .env from .env.example. Add your NCBI_API_KEY and NCBI_EMAIL before running Notebook 01.")

    # 9. Validation
    if not args.skip_validate:
        run([py, ROOT / "scripts" / "validate_environment.py", "--expect-device", device])
    print("\nEnvironment setup complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
