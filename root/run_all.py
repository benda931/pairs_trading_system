# -*- coding: utf-8 -*-
"""
all_run.py — נקודת כניסה מאוחדת למערכת
======================================

מטרות:
- להפעיל את ה־Dashboard (Streamlit) בצורה נקייה.
- להפעיל שירות מסחר (אם קיים קובץ מתאים) כתהליך נפרד.
- לנהל את מחזור החיים של התהליכים: אם אחד מת → נסגור את שניהם.
- לספק מנגנון Authentication פשוט דרך משתנה סביבה (לא בסיסמה קשיחה בקוד).

שימוש:
    python all_run.py

הערות:
- אם לא מוגדרת סיסמה במשתנה סביבה, ההרצה תמשיך כרגיל (mode פיתוח).
- שירות המסחר יופעל רק אם נמצא אחד הקבצים:
    root/trading_service.py
    root/live_trading.py
    system_trading_pairs_full.py        (לצורך תאימות לאחור)
"""

from __future__ import annotations

import getpass
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, List


# ===================== Path Setup =====================

PROJECT_ROOT = Path(__file__).resolve().parent
ROOT_DIR = PROJECT_ROOT / "root"
DASHBOARD_FILE = ROOT_DIR / "dashboard.py"

TRADING_CANDIDATES = [
    ROOT_DIR / "trading_service.py",
    ROOT_DIR / "live_trading.py",
    PROJECT_ROOT / "system_trading_pairs_full.py",  # תאימות לגיבוי הישן
]


# ===================== Logging =====================

def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] [all_run] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


logger = logging.getLogger("all_run")


# ===================== Auth =====================

PASSWORD_ENV_VAR = "PAIRS_TRADING_LAUNCH_PASSWORD"


def get_expected_password() -> Optional[str]:
    """
    קורא סיסמה ממשתנה סביבה.
    אם לא מוגדר — מחזיר None (mode פיתוח, בלי מנגנון סיסמה).
    """
    value = os.getenv(PASSWORD_ENV_VAR)
    return value.strip() if value else None


def authenticate() -> bool:
    """
    מנגנון אימות פשוט:
    - אם יש סיסמה במשתנה סביבה → מבקש מהמשתמש להזין.
    - אם אין סיסמה → מדלג (mode פיתוח).
    """
    expected = get_expected_password()
    if not expected:
        logger.warning(
            "No password configured in env (%s). "
            "Running in development mode WITHOUT auth.",
            PASSWORD_ENV_VAR,
        )
        return True

    try:
        entered = getpass.getpass("Enter launch password: ").strip()
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to read password from input: %s", exc)
        return False

    if entered == expected:
        logger.info("Authentication successful.")
        return True

    logger.error("Authentication FAILED.")
    return False


# ===================== Helpers =====================

def _check_paths() -> bool:
    """
    בודק שקובץ ה־Dashboard קיים.
    לא מחייב שירות מסחר — זה אופציונלי.
    """
    ok = True

    if not DASHBOARD_FILE.exists():
        logger.error("Dashboard file not found: %s", DASHBOARD_FILE)
        ok = False
    else:
        logger.info("Dashboard entrypoint: %s", DASHBOARD_FILE)

    trading_entry = find_trading_entrypoint()
    if trading_entry:
        logger.info("Trading service entrypoint: %s", trading_entry)
    else:
        logger.warning(
            "No trading service entrypoint found. "
            "Dashboard will run WITHOUT live trading service."
        )

    return ok


def find_trading_entrypoint() -> Optional[Path]:
    """
    מחפש קובץ שנראה כמו נקודת כניסה לשירות מסחר.
    הקובץ אינו חובה — אם לא נמצא, פשוט לא נריץ שירות מסחר.
    """
    for candidate in TRADING_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def _build_streamlit_command() -> List[str]:
    """
    בונה את פקודת ההפעלה של Streamlit, מבוסס על Python הנוכחי.
    """
    return [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(DASHBOARD_FILE),
    ]


def _build_trading_command(entrypoint: Path) -> List[str]:
    """
    בונה את פקודת ההפעלה לשירות המסחר.
    """
    return [sys.executable, str(entrypoint)]


# ===================== Process Management =====================

def _terminate_process(proc: Optional[subprocess.Popen]) -> None:
    if proc is None:
        return

    if proc.poll() is not None:
        return  # כבר מת

    logger.info("Terminating process pid=%s ...", proc.pid)
    try:
        proc.terminate()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Error in terminate(): %s", exc)

    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        logger.warning("Process did not exit, killing pid=%s ...", proc.pid)
        try:
            proc.kill()
        except Exception as exc:  # noqa: BLE001
            logger.error("Error in kill(): %s", exc)


def run() -> None:
    """
    הפונקציה הראשית:
    - אימות (אם צריך)
    - בדיקת קבצים
    - הרצת Dashboard + שירות מסחר (אם קיים)
    - ניהול מחזור חיים (אם אחד מת → נסגור את השאר)
    """
    _setup_logging()

    logger.info("Starting unified launcher from %s", PROJECT_ROOT)

    if not authenticate():
        logger.error("Authentication failed. Exiting.")
        sys.exit(1)

    if not _check_paths():
        logger.error("Path check failed. Exiting.")
        sys.exit(1)

    trading_entry = find_trading_entrypoint()
    dashboard_cmd = _build_streamlit_command()

    trading_proc: Optional[subprocess.Popen] = None
    dashboard_proc: Optional[subprocess.Popen] = None

    try:
        # 1) שירות מסחר (אופציונלי)
        if trading_entry is not None:
            trading_cmd = _build_trading_command(trading_entry)
            logger.info("Launching trading service: %s", " ".join(trading_cmd))
            trading_proc = subprocess.Popen(trading_cmd)  # noqa: S603, S607
        else:
            logger.info("No trading service to launch.")

        # 2) Dashboard
        logger.info("Launching Streamlit dashboard: %s", " ".join(dashboard_cmd))
        dashboard_proc = subprocess.Popen(dashboard_cmd)  # noqa: S603, S607

        # 3) לולאה לניטור שני התהליכים
        logger.info("Launcher is now monitoring child processes (Ctrl+C to stop).")
        while True:
            time.sleep(2)

            if dashboard_proc and dashboard_proc.poll() is not None:
                logger.warning(
                    "Dashboard process exited with code %s. "
                    "Shutting down trading service (if running).",
                    dashboard_proc.returncode,
                )
                break

            if trading_proc and trading_proc.poll() is not None:
                logger.warning(
                    "Trading service exited with code %s. "
                    "Shutting down dashboard.",
                    trading_proc.returncode,
                )
                break

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. Shutting down children...")
    finally:
        _terminate_process(dashboard_proc)
        _terminate_process(trading_proc)
        logger.info("Launcher finished. Goodbye.")


# ===================== Entrypoint =====================

if __name__ == "__main__":
    # Windows friendly: ensure Ctrl+C עובד יפה
    if hasattr(signal, "SIGINT"):
        signal.signal(signal.SIGINT, signal.default_int_handler)

    run()
