"""Execute a notebook in place with the cliniver kernel; save outputs even if a cell fails.

Exit status is 1 on ANY failure (cell error, cell timeout, kernel death). An earlier version caught only
CellExecutionError and printed status=0 after a cell timeout (2026-11-20).
"""
import sys
import time
import traceback
from pathlib import Path

import nbformat
from nbclient import NotebookClient

def keep_awake(on: bool) -> None:
    """Windows: keep the system out of (Modern) Standby while a notebook runs. EXP-09-002's first full run stalled for
    1.5 h when the laptop entered Modern Standby on an idle timeout mid-run (2026-11-22 14:18). Display may still
    turn off."""
    if sys.platform != "win32":
        return
    import ctypes
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0))


keep_awake(True)
path = Path(sys.argv[1]).resolve()
nb = nbformat.read(path, as_version=4)
# No per-cell timeout: data-heavy cells legitimately run long; hangs are prevented at the HTTP layer
# (total request deadline) and the run is watched externally.
client = NotebookClient(nb, kernel_name="cliniver", timeout=None,
                        resources={"metadata": {"path": str(path.parent)}})
t0 = time.perf_counter()
status = 0
try:
    client.execute()
except BaseException as exc:  # noqa: BLE001 — report every failure, never mask it
    status = 1
    print(f"RUN FAILED: {type(exc).__name__}: {str(exc)[-3000:]}")
    traceback.print_exc()
finally:
    nbformat.write(nb, path)
    errors = sum(1 for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", []) if o.output_type == "error")
    if errors:
        status = 1
    print(f"saved {path} after {time.perf_counter() - t0:.0f}s, status={status}, cells_with_errors={errors}")
    keep_awake(False)
sys.exit(status)
