"""Polite, auditable HTTP access to public APIs.

* **Rate limiting** per client (minimum interval between requests), set from each source's policy.
* **Retries** with exponential backoff on 429 / 5xx / connection errors, honouring ``Retry-After``.
* **Raw archive**: every successful response body is stored unchanged with a metadata sidecar
  (URL, redacted params, status, timestamp, SHA-256), so normalized data can always be traced
  and re-derived from the original bytes.
"""

from __future__ import annotations

import hashlib
import json
import re
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, TypeVar

import requests

from cliniver import __version__
from cliniver.utils.logging import get_logger

log = get_logger(__name__)
T = TypeVar("T")
R = TypeVar("R")

USER_AGENT = f"CliniVer/{__version__} (academic research prototype; non-commercial)"
_SECRET_PARAMS = {"api_key", "email", "apikey", "key", "token"}
_RETRY_STATUS = {429, 500, 502, 503, 504}


class FetchError(RuntimeError):
    """Raised when a request still fails after all retries (never swallowed silently)."""

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


NOT_FOUND = b'{"__cliniver_not_found__": true}'  # archived marker for an HTTP 404 answer


class DeadlineExceeded(requests.RequestException):
    """A request exceeded its total wall-clock deadline (retryable)."""


class CacheMismatchError(RuntimeError):
    """An archived response exists under this name but was fetched with different request parameters."""


@dataclass
class RateLimitedClient:
    """Thread-safe client for ONE provider.

    Concurrency model: several threads may share a client. ``_wait_turn`` spaces request
    *start times* by ``min_interval_s`` across all threads (the provider's rate limit is never
    exceeded), while up to ``max_in_flight`` requests can be awaiting responses at once, so
    network latency overlaps instead of adding to the interval.
    """

    name: str
    min_interval_s: float  # e.g. 0.4 → at most 2.5 requests/second
    max_retries: int = 4
    backoff_base_s: float = 2.0
    timeout_s: float = 60.0  # per socket read (requests semantics): resets whenever ANY byte arrives
    max_in_flight: int = 4
    # Hard wall-clock limit for one request including the body. Needed because the per-read timeout
    # never fires on a server that trickles bytes: on 2026-11-20 two PubMed efetch connections hung
    # this way and blocked both in-flight slots for > 5 min with no error raised.
    total_timeout_s: float = 240.0

    def __post_init__(self) -> None:
        self._local = threading.local()  # requests.Session is not guaranteed thread-safe: one per thread
        self._last = 0.0
        self._lock = threading.Lock()
        self._slots = threading.BoundedSemaphore(self.max_in_flight)
        self.n_requests = 0
        self.n_retries = 0
        self.n_cache_hits = 0  # responses served from the raw archive (no network call)

    def count_cache_hit(self) -> None:
        with self._lock:
            self.n_cache_hits += 1

    @property
    def _session(self) -> requests.Session:
        if not hasattr(self._local, "session"):
            s = requests.Session()
            s.headers["User-Agent"] = USER_AGENT
            self._local.session = s
        return self._local.session

    def _wait_turn(self) -> None:
        # Holding the lock while sleeping is intentional: threads queue for the next start slot.
        with self._lock:
            delay = self.min_interval_s - (time.monotonic() - self._last)
            if delay > 0:
                time.sleep(delay)
            self._last = time.monotonic()
            self.n_requests += 1

    def _fetch(self, url: str, params: dict[str, Any] | None) -> requests.Response:
        """One HTTP GET with a hard total deadline.

        The body is streamed. A watchdog timer closes the response when ``total_timeout_s`` elapses,
        which unblocks a read stuck on a trickling connection; the deadline is also checked between
        chunks. Either way the caller sees ``DeadlineExceeded`` (a RequestException → retried).
        """
        start = time.monotonic()
        resp = self._session.get(url, params=params, timeout=(min(15.0, self.timeout_s), self.timeout_s), stream=True)
        aborted = threading.Event()

        def abort() -> None:
            # resp.close() would BLOCK here on Windows (it waits for the reader's buffer lock: found
            # by test_total_deadline_aborts_trickling_server). Shutting the socket down does not take
            # that lock and makes the blocked recv() return immediately.
            aborted.set()
            sock = getattr(getattr(resp.raw, "_connection", None), "sock", None)
            if sock is None:
                try:
                    sock = resp.raw._fp.fp.raw._sock
                except AttributeError:
                    sock = None
            if sock is not None:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

        watchdog = threading.Timer(self.total_timeout_s, abort)
        watchdog.daemon = True
        watchdog.start()
        deadline_msg = f"total deadline {self.total_timeout_s:.0f}s exceeded"
        try:
            chunks = []
            for chunk in resp.iter_content(chunk_size=16384):
                chunks.append(chunk)
                if time.monotonic() - start > self.total_timeout_s:
                    raise DeadlineExceeded(deadline_msg)
            # A shutdown socket can look like a normal end of body (no Content-Length): never accept
            # a body the watchdog touched, since it may be silently truncated.
            if aborted.is_set():
                raise DeadlineExceeded(deadline_msg)
            resp._content = b"".join(chunks)  # body fully read: .content / .text work as usual
        except DeadlineExceeded:
            raise
        except (requests.RequestException, OSError, AttributeError, ValueError) as exc:
            if aborted.is_set():
                raise DeadlineExceeded(deadline_msg) from exc
            if isinstance(exc, requests.RequestException):
                raise
            raise requests.ConnectionError(f"read failed: {exc!r}") from exc
        finally:
            watchdog.cancel()
        return resp

    def get(self, url: str, params: dict[str, Any] | None = None) -> requests.Response:
        last_err: Exception | None = None
        attempts = 0
        for attempt in range(self.max_retries + 1):
            attempts = attempt + 1
            resp: requests.Response | None = None
            status: int | None = None
            try:
                with self._slots:
                    self._wait_turn()
                    resp = self._fetch(url, params)
            except requests.RequestException as exc:
                last_err = exc
            else:
                if resp.status_code < 400:
                    return resp
                status = resp.status_code
                last_err = FetchError(f"HTTP {status}: {resp.text[:300]}", status=status)
                if status not in _RETRY_STATUS:
                    break  # 4xx other than 429: retrying will not help
            if attempt < self.max_retries:
                with self._lock:
                    self.n_retries += 1
                wait = self.backoff_base_s * 2**attempt
                if status == 429 and resp is not None and resp.headers.get("Retry-After", "").isdigit():
                    wait = max(wait, float(resp.headers["Retry-After"]))
                log.warning("%s: attempt %d failed (%s); retrying in %.1fs", self.name, attempt + 1, last_err, wait)
                time.sleep(wait)
        raise FetchError(f"{self.name}: GET {url} failed after {attempts} attempt(s): {last_err}",
                         status=getattr(last_err, "status", None))


def redact(params: dict[str, Any] | None) -> dict[str, Any]:
    """Drop credentials and contact details before anything is written to disk."""
    return {k: ("<redacted>" if k.lower() in _SECRET_PARAMS else v) for k, v in (params or {}).items()}


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)[:150]


class RawStore:
    """Stores raw API responses under ``data/raw/<source>/<snapshot>/``."""

    def __init__(self, root: Path, source: str, snapshot: str) -> None:
        self.dir = root / source / snapshot
        self.dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, name: str, ext: str) -> Path:
        return self.dir / f"{_safe_name(name)}.{ext}"

    def exists(self, name: str, ext: str) -> bool:
        return self.path_for(name, ext).is_file()

    def read(self, name: str, ext: str) -> bytes:
        return self.path_for(name, ext).read_bytes()

    def save(self, name: str, ext: str, content: bytes, *, url: str, params: dict[str, Any] | None,
             status: int) -> Path:
        path = self.path_for(name, ext)
        path.write_bytes(content)
        meta = {
            "url": url,
            "params": redact(params),
            "status": status,
            "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "sha256": hashlib.sha256(content).hexdigest(),
            "bytes": len(content),
            "client": USER_AGENT,
        }
        path.with_suffix(path.suffix + ".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return path


def _comparable(params: dict[str, Any] | None) -> dict[str, Any]:
    """Request parameters as archived (JSON round-trip), without credentials/contact details."""
    return json.loads(json.dumps({k: v for k, v in (params or {}).items() if k.lower() not in _SECRET_PARAMS}))


def _verify_cached_request(store: RawStore, name: str, ext: str, url: str, params: dict[str, Any] | None) -> None:
    """Raise if the archived response under ``name`` was fetched with a different URL or parameters.

    The archive is keyed by name; without this check a changed query (e.g. a new page size) would be
    silently answered with the old response.
    """
    meta_path = store.path_for(name, ext).with_suffix(store.path_for(name, ext).suffix + ".meta.json")
    if not meta_path.is_file():
        raise CacheMismatchError(f"{meta_path.name} missing: cannot verify archived response '{name}'")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    stored = {k: v for k, v in meta.get("params", {}).items() if k.lower() not in _SECRET_PARAMS}
    if meta.get("url") != url or stored != _comparable(params):
        raise CacheMismatchError(
            f"Archived '{name}' was fetched with different request: stored url={meta.get('url')} params={stored} "
            f"vs current url={url} params={_comparable(params)}. Use a new cache name or snapshot.")


def parallel_map(fn: Callable[[T], R], items: Iterable[T], max_workers: int) -> list[R]:
    """Apply ``fn`` to every item concurrently; results come back in input order.

    The first exception is re-raised after all submitted work finishes (no silent partial result).
    Rate limiting is enforced by the shared ``RateLimitedClient``, not by ``max_workers``.
    """
    items = list(items)
    if max_workers <= 1 or len(items) <= 1:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(fn, items))


def cached_get(client: RateLimitedClient, store: RawStore, name: str, ext: str, url: str,
               params: dict[str, Any] | None = None, *, refresh: bool = False,
               not_found_ok: bool = False, not_found_status: tuple[int, ...] = (404,),
               not_found_text: str | None = None) -> bytes | None:
    """Return the raw body for (url, params), from the archive if already fetched in this snapshot.

    Re-running a notebook therefore re-uses the same bytes (reproducible) and does not
    re-hit the API. ``refresh=True`` forces a new download.

    ``not_found_ok=True``: an HTTP 404 is a valid *answer* ("no such record"). It is archived as a
    NOT_FOUND marker and returned as ``None``, so a replay reproduces the absence too.
    ``not_found_status`` / ``not_found_text`` extend this to providers that signal "nothing available"
    differently (e.g. PubTator: HTTP 400 "Could not retrieve publications"). The text must match, so a
    genuinely malformed request with the same status still raises.
    """
    if not refresh and store.exists(name, ext):
        _verify_cached_request(store, name, ext, url, params)
        client.count_cache_hit()
        body = store.read(name, ext)
        return None if body == NOT_FOUND else body
    try:
        resp = client.get(url, params=params)
    except FetchError as exc:
        if not_found_ok and exc.status in not_found_status and (not_found_text is None or not_found_text in str(exc)):
            store.save(name, ext, NOT_FOUND, url=url, params=params, status=exc.status)
            return None
        raise
    store.save(name, ext, resp.content, url=url, params=params, status=resp.status_code)
    return resp.content
