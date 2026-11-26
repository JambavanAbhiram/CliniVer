import logging
import random

import numpy as np
import pytest

from cliniver.utils import (
    collect_hardware_info,
    get_logger,
    package_versions,
    resolve_device,
    set_seed,
    setup_logging,
)

torch = pytest.importorskip("torch")


def test_set_seed_reproducible():
    set_seed(7)
    first = (random.random(), np.random.rand(), torch.rand(3).tolist())
    set_seed(7)
    second = (random.random(), np.random.rand(), torch.rand(3).tolist())
    assert first == second


def test_set_seed_rejects_negative():
    with pytest.raises(ValueError):
        set_seed(-1)


def test_resolve_device():
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("auto") in {"cpu", "cuda"}
    with pytest.raises(ValueError):
        resolve_device("tpu")


def test_resolve_device_cuda_unavailable_is_error(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError):
        resolve_device("cuda")


def test_hardware_info_fields():
    info = collect_hardware_info()
    assert info.python_version.startswith("3.")
    assert info.ram_total_gb > 0
    assert info.torch_version is not None
    d = info.to_dict()
    assert {"gpus", "cuda_available", "platform"} <= d.keys()


def test_package_versions_handles_missing():
    v = package_versions(["pydantic", "definitely-not-a-package-xyz"])
    assert v["pydantic"] is not None
    assert v["definitely-not-a-package-xyz"] is None


def test_logging_no_duplicate_handlers(tmp_path):
    setup_logging("INFO", log_dir=tmp_path, run_name="t")
    n = len(logging.getLogger("cliniver").handlers)
    setup_logging("DEBUG", log_dir=tmp_path, run_name="t")
    assert len(logging.getLogger("cliniver").handlers) == n
    assert get_logger("x").name == "cliniver.x"


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_matmul_matches_cpu():
    a, b = torch.randn(256, 256), torch.randn(256, 256)
    assert torch.allclose((a.cuda() @ b.cuda()).cpu(), a @ b, rtol=1e-2, atol=1e-2)
