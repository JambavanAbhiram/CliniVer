import pytest

import cliniver.config.settings as settings


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    """Tests must not depend on the developer's ``.env`` or shell environment.

    Removes CliniVer runtime overrides and disables ``.env`` loading, so every test sees
    exactly ``configs/project.yaml`` plus whatever the test itself sets.
    """
    for var in ("CLINIVER_DEVICE", "CLINIVER_LOG_LEVEL", "CLINIVER_ROOT"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(settings, "load_dotenv", lambda *a, **k: False)
    yield
