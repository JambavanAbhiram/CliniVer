"""Local LLM runtime: a managed llama.cpp ``llama-server`` process and a logging chat client.

Every call is logged (JSONL) with model, quantization file, runtime release, parameters, prompt hash, seed,
timestamp, latency and token counts (research_foundation §9.3). One large model on the GPU at a time:
``LlamaServer`` is a context manager that starts the server, waits for health, samples GPU memory, and stops
it on exit.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


def gpu_memory_used_mb() -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10)
        return int(out.stdout.strip().splitlines()[0])
    except Exception:  # noqa: BLE001 — no GPU / no nvidia-smi: report None, never fail a run
        return None


@dataclass
class LlamaServer:
    binary_dir: Path
    model_path: Path
    model_key: str
    release: str
    host: str = "127.0.0.1"
    port: int = 8089
    ctx_size: int = 8192
    gpu_layers: int = 999
    parallel: int = 1
    extra_args: list[str] = field(default_factory=list)
    log_path: Path | None = None
    startup_timeout_s: float = 300

    def __post_init__(self) -> None:
        self.proc: subprocess.Popen | None = None
        self.baseline_mb: int | None = None
        self.peak_mb: int | None = None
        self._stop = threading.Event()
        self._sampler: threading.Thread | None = None
        self.load_s: float | None = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def _sample(self) -> None:
        while not self._stop.wait(0.5):
            m = gpu_memory_used_mb()
            if m is not None:
                self.peak_mb = max(self.peak_mb or 0, m)

    def __enter__(self) -> "LlamaServer":
        exe = self.binary_dir / "llama-server.exe"
        if not exe.exists():
            exe = self.binary_dir / "llama-server"
        args = [str(exe), "-m", str(self.model_path), "--host", self.host, "--port", str(self.port),
                "-c", str(self.ctx_size), "-ngl", str(self.gpu_layers), "-np", str(self.parallel),
                "--jinja", "--no-webui", *self.extra_args]
        self.baseline_mb = gpu_memory_used_mb()
        log = (self.log_path.open("w", encoding="utf-8") if self.log_path else subprocess.DEVNULL)
        t0 = time.perf_counter()
        self.proc = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT)
        self._sampler = threading.Thread(target=self._sample, daemon=True)
        self._sampler.start()
        while True:
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited with code {self.proc.returncode} (see {self.log_path})")
            try:
                if requests.get(self.url + "/health", timeout=2).status_code == 200:
                    break
            except requests.RequestException:
                pass
            if time.perf_counter() - t0 > self.startup_timeout_s:
                self.__exit__(None, None, None)
                raise TimeoutError("llama-server did not become healthy")
            time.sleep(0.5)
        self.load_s = time.perf_counter() - t0
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if self._sampler:
            self._sampler.join(timeout=2)

    def stats(self) -> dict[str, Any]:
        return {"model": self.model_key, "load_s": round(self.load_s or 0, 1), "gpu_baseline_mb": self.baseline_mb,
                "gpu_peak_mb": self.peak_mb,
                "gpu_peak_minus_baseline_mb": (self.peak_mb - self.baseline_mb) if self.peak_mb and self.baseline_mb is not None else None}


def prompt_hash(messages: list[dict], extra: Any = None) -> str:
    return hashlib.sha256(json.dumps([messages, extra], sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


def call_key(model_file: str, release: str, messages: list[dict], params: dict) -> str:
    return hashlib.sha256(json.dumps([model_file, release, messages, params], sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class ChatClient:
    """OpenAI-compatible chat client for llama-server, logging every call to a JSONL file.

    The log doubles as a **replay cache**: a call with the same model file, runtime release, messages and
    parameters (incl. seed and schema) is answered from the log without contacting the server, so a notebook
    can be re-executed exactly without regenerating (like the raw-data archive of NB01). ``server`` may be None
    in pure replay mode; a cache miss then raises.
    """

    def __init__(self, server: LlamaServer | None, log_file: Path, *, model_key: str, model_file: str, release: str,
                 timeout_s: float = 600) -> None:
        self.server, self.log_file, self.model_file, self.timeout_s = server, log_file, model_file, timeout_s
        self.model_key, self.release = model_key, release
        log_file.parent.mkdir(parents=True, exist_ok=True)
        self.cache: dict[str, dict] = {}
        if log_file.exists():
            for line in log_file.open(encoding="utf-8"):
                rec = json.loads(line)
                if "key" in rec:
                    self.cache[rec["key"]] = rec
        self.n_live = self.n_cached = 0

    def chat(self, messages: list[dict], *, temperature: float = 0.0, max_tokens: int = 512, seed: int = 42,
             json_schema: dict | None = None, enable_thinking: bool = False, tag: str = "", top_p: float = 1.0) -> dict:
        params = {"temperature": temperature, "top_p": top_p, "max_tokens": max_tokens, "seed": seed,
                  "json_schema": json_schema, "enable_thinking": enable_thinking}
        key = call_key(self.model_file, self.release, messages, params)
        if key in self.cache:
            self.n_cached += 1
            return {**self.cache[key], "cached": True}
        if self.server is None:
            raise LookupError(f"replay mode: no cached call for {self.model_key} / {tag}")
        body: dict[str, Any] = {"messages": messages, "temperature": temperature, "max_tokens": max_tokens, "seed": seed,
                                "top_p": top_p, "chat_template_kwargs": {"enable_thinking": enable_thinking}}
        if json_schema is not None:
            body["response_format"] = {"type": "json_schema", "json_schema": {"name": "output", "schema": json_schema, "strict": True}}
        t0 = time.perf_counter()
        r = requests.post(self.server.url + "/v1/chat/completions", json=body, timeout=self.timeout_s)
        latency = time.perf_counter() - t0
        r.raise_for_status()
        j = r.json()
        text = j["choices"][0]["message"].get("content") or ""
        usage, timings = j.get("usage", {}), j.get("timings", {})
        rec = {"key": key, "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "tag": tag, "model": self.model_key,
               "model_file": self.model_file, "runtime": f"llama.cpp {self.release}",
               "params": {"temperature": temperature, "top_p": top_p, "max_tokens": max_tokens, "seed": seed,
                          "json_schema": json_schema is not None, "enable_thinking": enable_thinking},
               "prompt_hash": prompt_hash(messages, json_schema), "messages": messages, "output": text,
               "finish_reason": j["choices"][0].get("finish_reason"), "latency_s": round(latency, 3),
               "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
               "tokens_per_s": round(timings.get("predicted_per_second", 0) or 0, 2),
               "draft_n": timings.get("draft_n"), "draft_accepted": timings.get("draft_n_accepted")}   # speculative decoding stats (None without it)
        with self.log_file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.cache[key] = rec
        self.n_live += 1
        return {**rec, "cached": False}


class ManagedClient:
    """Context manager: a ChatClient that answers from the replay cache and starts ``llama-server`` only on the
    first cache miss (one model on the GPU at a time; the server is stopped on exit). ``stats`` reports load time
    and GPU memory of the live server, or ``None`` when everything was replayed."""

    def __init__(self, llm_cfg, key: str, log_file: Path, server_log: Path | None = None, extra_args_add: list[str] | None = None,
                 **server_kw) -> None:
        self.cfg, self.key, self.server_kw, self.server_log = llm_cfg, key, server_kw, server_log
        self.extra_args_add = list(extra_args_add or [])   # e.g. speculative-decoding flags (EXP-08-003); outputs are cached under the same key
        self.client = ChatClient(None, log_file, model_key=key, model_file=llm_cfg.models[key].file,
                                 release=llm_cfg.runtime.release)
        self.server: LlamaServer | None = None

    def _start(self) -> None:
        rt = self.cfg.runtime
        from cliniver.utils.paths import project_root
        self.server = LlamaServer(project_root() / rt.dir, self.cfg.model_path(self.key), self.key, rt.release, rt.host,
                                  rt.port, rt.ctx_size, rt.gpu_layers, extra_args=list(rt.extra_args) + self.extra_args_add,
                                  log_path=self.server_log, **self.server_kw).__enter__()
        self.client.server = self.server

    def start(self) -> "ManagedClient":
        """Start the server even if every call is cached: hardware measurements (load time, GPU memory) must come
        from a live server, never be lost in a replay (fixed 2026-11-21 after EXP-03-001's replay erased them)."""
        if self.server is None:
            self._start()
        return self

    def chat(self, messages, **kw):
        try:
            return self.client.chat(messages, **kw)
        except LookupError:
            self._start()
            return self.client.chat(messages, **kw)

    def __enter__(self) -> "ManagedClient":
        return self

    def __exit__(self, *exc) -> None:
        if self.server:
            self.server.__exit__(*exc)

    @property
    def stats(self) -> dict | None:
        return self.server.stats() if self.server else None
