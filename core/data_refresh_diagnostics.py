from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Dict

from common.helpers import read_json, write_json

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
DATA_REFRESH_DIAGNOSTICS_DIR = LOGS_DIR / "data_refresh_diagnostics"
SCHEDULER_PID_FILE = LOGS_DIR / "scheduler.pid"


def _sanitize_run_id(run_id: str) -> str:
    clean = str(run_id or "latest").strip() or "latest"
    for ch in ('\\', '/', ':', '*', '?', '"', '<', '>', '|'):
        clean = clean.replace(ch, "_")
    return clean


def data_refresh_diagnostics_path(run_id: str = "latest") -> Path:
    return DATA_REFRESH_DIAGNOSTICS_DIR / f"{_sanitize_run_id(run_id)}.json"


def persist_data_refresh_diagnostics(payload: Dict[str, Any], run_id: str = "latest") -> None:
    DATA_REFRESH_DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
    run_path = data_refresh_diagnostics_path(run_id)
    write_json(payload, run_path)
    if str(run_id or "latest") != "latest":
        write_json(payload, data_refresh_diagnostics_path("latest"))


def load_data_refresh_diagnostics(run_id: str = "latest") -> Dict[str, Any]:
    path = data_refresh_diagnostics_path(run_id)
    if not path.exists():
        return {}
    try:
        payload = read_json(path)
    except Exception as exc:
        logger.debug("load_data_refresh_diagnostics(%s) failed: %s", run_id, exc)
        return {}
    return dict(payload) if isinstance(payload, dict) else {}


def _pid_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        try:
            proc = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            output = (proc.stdout or "").strip()
            return bool(output) and "No tasks are running" not in output
        except Exception:
            return False
    try:
        os.kill(pid, 0)
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def is_external_scheduler_daemon_active(current_pid: int | None = None) -> bool:
    current = int(current_pid or os.getpid())
    try:
        raw = SCHEDULER_PID_FILE.read_text(encoding="utf-8").strip()
        daemon_pid = int(raw)
    except Exception:
        return False
    if daemon_pid <= 0 or daemon_pid == current:
        return False
    return _pid_exists(daemon_pid)
