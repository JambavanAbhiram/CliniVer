"""Download the pinned GGUF models listed in configs/llm.yaml.

Plain HTTP with Range requests: resumable into ONE fixed partial file (``<file>.part``), with a read timeout
and automatic retries, then verified against the SHA-256 that Hugging Face publishes for the LFS object.
(huggingface_hub 2.x names partial files randomly per process, so an interrupted download restarted from zero;
on this network the Xet backend also stalled. Both observed 2026-11-20.)

Usage: .venv\\Scripts\\python.exe scripts\\download_models.py [model_key ...]
"""
import hashlib
import sys
import time
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
cfg = yaml.safe_load((ROOT / "configs" / "llm.yaml").read_text(encoding="utf-8"))
keys = sys.argv[1:] or list(cfg["models"])
CHUNK = 1 << 20


def expected_sha256(repo: str, revision: str, file: str) -> str:
    j = requests.get(f"https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true", timeout=60).json()
    return next(s["lfs"]["sha256"] for s in j["siblings"] if s["rfilename"] == file)


def download(url: str, part: Path, max_retries: int = 200) -> None:
    for attempt in range(max_retries):
        have = part.stat().st_size if part.exists() else 0
        try:
            with requests.get(url, headers={"Range": f"bytes={have}-"}, stream=True, timeout=(30, 30)) as r:
                if r.status_code == 416:  # nothing left to fetch
                    return
                r.raise_for_status()
                if have and r.status_code != 206:
                    raise RuntimeError("server ignored the Range request; refusing to append")
                total = have + int(r.headers.get("Content-Length", 0))
                t0, got = time.time(), 0
                with part.open("ab") as fh:
                    for b in r.iter_content(CHUNK):
                        fh.write(b); got += len(b)
                        if got % (256 * CHUNK) < CHUNK:
                            print(f"  {time.strftime('%H:%M:%S')} {(have + got) / 1e9:.2f}/{total / 1e9:.2f} GB "
                                  f"({got / 1e6 / max(1e-6, time.time() - t0):.1f} MB/s)", flush=True)
            if part.stat().st_size >= total:
                return
        except (requests.RequestException, RuntimeError) as exc:
            print(f"  retry {attempt + 1}: {type(exc).__name__}: {str(exc)[:120]}", flush=True)
            time.sleep(min(30, 2 + attempt))
    raise RuntimeError(f"download failed after {max_retries} attempts")


for k in keys:
    m = cfg["models"][k]
    dest = ROOT / cfg["models_dir"] / k / m["file"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    sha_file = dest.parent / (dest.name + ".sha256")
    if dest.exists() and sha_file.exists():
        print(f"{k}: already present"); continue
    want = expected_sha256(m["repo"], m["revision"], m["file"])
    url = f"https://huggingface.co/{m['repo']}/resolve/{m['revision']}/{m['file']}"
    part = dest.with_name(dest.name + ".part")
    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] {k}: {url} (resume from {part.stat().st_size / 1e9 if part.exists() else 0:.2f} GB)", flush=True)
    download(url, part)
    h = hashlib.sha256()
    with part.open("rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    if h.hexdigest() != want:
        raise SystemExit(f"{k}: SHA-256 mismatch (got {h.hexdigest()}, expected {want}); partial file kept for inspection")
    part.rename(dest)
    sha_file.write_text(h.hexdigest() + "\n", encoding="utf-8")
    print(f"[{time.strftime('%H:%M:%S')}] {k}: done {dest.stat().st_size / 1e9:.2f} GB in {time.time() - t0:.0f}s "
          f"sha256={h.hexdigest()[:16]} (verified against Hugging Face)", flush=True)
