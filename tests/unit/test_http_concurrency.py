"""The rate limiter must hold under concurrency: request starts are spaced by min_interval_s
across threads, while response latency overlaps (the reason for parallelism)."""

import threading
import time

from cliniver.ingestion.http import RateLimitedClient, parallel_map

LATENCY = 0.15
INTERVAL = 0.05


class _FakeResponse:
    status_code = 200
    headers: dict = {}
    text = ""
    content = b"{}"

    def iter_content(self, chunk_size=1):
        yield self.content

    def close(self):
        pass


class _FakeSession:
    def __init__(self, starts: list[float], lock: threading.Lock) -> None:
        self.starts, self.lock = starts, lock

    def get(self, url, params=None, timeout=None, stream=False):
        with self.lock:
            self.starts.append(time.monotonic())
        time.sleep(LATENCY)  # simulated network latency
        return _FakeResponse()


def _make(max_in_flight: int, starts: list[float]) -> RateLimitedClient:
    Sub = type("Sub", (RateLimitedClient,), {})
    fake = _FakeSession(starts, threading.Lock())
    Sub._session = property(lambda self: fake)
    return Sub("fake", INTERVAL, max_in_flight=max_in_flight)


def test_parallel_requests_respect_rate_limit():
    starts: list[float] = []
    c = _make(4, starts)
    parallel_map(lambda i: c.get("http://x"), range(12), max_workers=4)
    starts.sort()
    gaps = [b - a for a, b in zip(starts, starts[1:])]
    assert len(starts) == 12 and c.n_requests == 12
    assert min(gaps) >= INTERVAL * 0.9, f"rate limit violated: min gap {min(gaps):.3f}s"


def test_parallelism_overlaps_latency():
    seq_starts: list[float] = []
    seq = _make(1, seq_starts)
    t0 = time.perf_counter()
    for _ in range(8):
        seq.get("http://x")
    t_seq = time.perf_counter() - t0

    par_starts: list[float] = []
    par = _make(4, par_starts)
    t0 = time.perf_counter()
    parallel_map(lambda i: par.get("http://x"), range(8), max_workers=4)
    t_par = time.perf_counter() - t0
    # Sequential ≈ 8 × (interval + latency); parallel ≈ bounded by the interval spacing plus one latency.
    assert t_par < 0.6 * t_seq, f"no speedup: sequential {t_seq:.2f}s vs parallel {t_par:.2f}s"


def test_parallel_map_preserves_order_and_raises():
    assert parallel_map(lambda x: x * 2, range(20), 5) == [x * 2 for x in range(20)]
    try:
        parallel_map(lambda x: 1 / (x - 3), range(6), 3)
    except ZeroDivisionError:
        pass
    else:
        raise AssertionError("exception from a worker must propagate")


def test_not_found_is_archived_and_replayed(tmp_path):
    """Regression (EXP-01-001 replay): a 404 'no such record' answer must be archived so that a
    replay of the snapshot reproduces the absence without a network call."""
    from cliniver.ingestion.http import FetchError, RawStore, cached_get

    class Client404:
        n = 0
        def get(self, url, params=None):
            Client404.n += 1
            raise FetchError("HTTP 404: not found", status=404)
        def count_cache_hit(self):
            pass

    store, c = RawStore(tmp_path, "src", "snap"), Client404()
    assert cached_get(c, store, "x", "json", "http://x", not_found_ok=True) is None
    assert cached_get(c, store, "x", "json", "http://x", not_found_ok=True) is None
    assert Client404.n == 1  # second call served from the archive
    try:
        cached_get(c, store, "y", "json", "http://y")  # without not_found_ok, a 404 is an error
    except FetchError:
        pass
    else:
        raise AssertionError("404 must raise when not_found_ok is False")


def test_not_found_text_must_match(tmp_path):
    """PubTator-style 'nothing available' (HTTP 400 + specific text) is archived as NOT_FOUND,
    but a 400 with any other body (e.g. a malformed request) still raises."""
    from cliniver.ingestion.http import FetchError, RawStore, cached_get

    class Client400:
        def __init__(self, body): self.body = body
        def get(self, url, params=None): raise FetchError(f"HTTP 400: {self.body}", status=400)
        def count_cache_hit(self): pass

    store = RawStore(tmp_path, "src", "snap")
    kw = dict(not_found_ok=True, not_found_status=(400, 404), not_found_text="Could not retrieve publications")
    ok = Client400('{"detail":"Could not retrieve publications"}')
    assert cached_get(ok, store, "a", "json", "http://x", **kw) is None
    bad = Client400('{"detail":"Invalid parameter"}')
    try:
        cached_get(bad, store, "b", "json", "http://x", **kw)
    except FetchError:
        pass
    else:
        raise AssertionError("a 400 that is not 'nothing available' must raise")


def _serve(handler_cls):
    import http.server
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}/"


def test_total_deadline_aborts_trickling_server():
    """Regression (2026-11-20): a server trickling bytes never trips the per-read timeout.
    The total deadline must abort it (here: read timeout 1.0 s, deadline 1.5 s, a byte every 0.4 s)."""
    import http.server

    from cliniver.ingestion.http import FetchError

    class Trickle(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200); self.send_header("Content-Length", "1000"); self.end_headers()
            try:
                for _ in range(1000):
                    self.wfile.write(b"x"); self.wfile.flush(); time.sleep(0.4)
            except OSError:
                pass  # client aborted, as intended
        def log_message(self, *a):
            pass

    srv, url = _serve(Trickle)
    try:
        c = RateLimitedClient("trickle", 0.01, max_retries=0, timeout_s=1.0, total_timeout_s=1.5)
        t0 = time.perf_counter()
        try:
            c.get(url)
        except FetchError as exc:
            assert "deadline" in str(exc)
        else:
            raise AssertionError("trickling response must be aborted")
        assert time.perf_counter() - t0 < 4.0, "deadline did not unblock the read"
    finally:
        srv.shutdown()


def test_normal_response_unaffected_by_deadline():
    import http.server

    class Ok(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = b'{"ok": true}' * 1000
            self.send_response(200); self.send_header("Content-Length", str(len(body))); self.end_headers()
            self.wfile.write(body)
        def log_message(self, *a):
            pass

    srv, url = _serve(Ok)
    try:
        r = RateLimitedClient("ok", 0.01, total_timeout_s=5).get(url)
        assert r.content == b'{"ok": true}' * 1000  # streamed body reassembled completely and exactly
    finally:
        srv.shutdown()


def test_cache_rejects_changed_request(tmp_path):
    from cliniver.ingestion.http import CacheMismatchError, RawStore, cached_get

    class Client:
        n = 0
        def get(self, url, params=None):
            Client.n += 1
            return type("R", (), {"content": b"{}", "status_code": 200})()
        def count_cache_hit(self):
            pass

    store, c = RawStore(tmp_path, "src", "snap"), Client()
    cached_get(c, store, "q", "json", "http://x", {"pageSize": 20})
    assert cached_get(c, store, "q", "json", "http://x", {"pageSize": 20}) == b"{}" and Client.n == 1
    # credentials are not part of the comparison (setting an NCBI email later must not invalidate the archive)
    cached_get(c, store, "q", "json", "http://x", {"pageSize": 20, "email": "a@b.c"})
    try:
        cached_get(c, store, "q", "json", "http://x", {"pageSize": 10})
    except CacheMismatchError:
        pass
    else:
        raise AssertionError("a changed parameter must not be answered from the old archive entry")
