"""Start the CliniVer Streamlit interface on localhost (settings from configs/service.yaml).

    .venv\\Scripts\\python.exe scripts\\run_app.py
"""

import subprocess
import sys
from pathlib import Path

from cliniver.config.service_config import load_service_config

ROOT = Path(__file__).resolve().parents[1]
cfg = load_service_config()
app = ROOT / "src" / "cliniver" / "service" / "app.py"
cmd = [sys.executable, "-m", "streamlit", "run", str(app),
       "--server.address", cfg.host, "--server.port", str(cfg.port), "--server.headless", "true",
       "--browser.gatherUsageStats", "false", "--server.fileWatcherType", "none", "--server.runOnSave", "false"]
print(f"CliniVer on http://{cfg.host}:{cfg.port}  (Ctrl+C to stop)", flush=True)
sys.exit(subprocess.call(cmd, cwd=ROOT))
