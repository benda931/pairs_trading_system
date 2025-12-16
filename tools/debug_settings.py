# -*- coding: utf-8 -*-
"""
tools/debug_settings.py — HF-grade Settings & Environment Diagnostics
=====================================================================

סקрипט דיאגנוסטיקה מקצועי למערכת ה-Pairs Trading שלך.

תפקידים:
---------
1. AppContext & Settings
   - בודק איך AppContext נטען בפועל.
   - מציג את כל ההגדרות הקריטיות:
        • env/profile
        • ib_enable / ib_host / ib_port / ib_mode / ib_client_id / ib_account
        • sql_store_url / sql_table_prefix / sql_read_only
        • data_dir / logs_dir / backtest_start/end וכו'.

2. SqlStore diagnostics
   - פותח SqlStore.from_settings(...)
   - מציג:
        • engine_url / dialect / default_env / read_only
        • רשימת טבלאות קיימות
        • Row count לטבלאות מפתח:
            dq_pairs, bt_runs, prices, risk_state, risk_timeline,
            dashboard_snapshots, kv_store
        • מזהה אם חסרות טבלאות קריטיות.

3. IBKR diagnostics (אופציונלי)
   - אם ib_enable=True:
        • מנסה להתחבר ל-IBKR דרך ibkr_connection (אם קיים).
        • בודק ib_insync, מצב חיבור, מספר פוזיציות, מספר הזמנות פתוחות (אם אפשר).

4. Environment & Paths
   - Python version, sys.executable, cwd.
   - האם רץ בתוך Streamlit (רק לוג אזהרה, אין דרישה ל-Context).

שימוש:
------
    python tools/debug_settings.py
    python tools/debug_settings.py --verbose
    python tools/debug_settings.py --no-ib --no-sql

הערות:
------
- הסקריפט לא תלוי ב-Streamlit, אך אם AppContext משתמש ב-streamlit.session_state
  הוא עלול להוציא אזהרות — זה תקין ב-"bare mode".
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

# === Ensure project root is on sys.path ===
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ===== Logger setup =====

logger = logging.getLogger("debug_settings")
if not logger.handlers:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-8s [debug_settings] %(message)s")
    )
    logger.addHandler(h)
logger.setLevel(logging.INFO)


# ===== Helpers =====

def _safe_get(obj: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(obj, name, default)
    except Exception:
        return default


def _serialize_settings(settings: Any) -> Dict[str, Any]:
    """
    ניסיון להוציא dict מ-settings בצורה סבירה:
    - אם יש as_dict() → משתמש.
    - אם יש __dict__ → משתמש.
    - אחרת: אוסף שדות בסיסיים באופן ידני.
    """
    if settings is None:
        return {}

    try:
        if hasattr(settings, "as_dict"):
            d = settings.as_dict()  # type: ignore[call-arg]
            if isinstance(d, dict):
                return d
    except Exception:
        pass

    try:
        d = getattr(settings, "__dict__", {})
        if isinstance(d, dict):
            return dict(d)
    except Exception:
        pass

    # fallback — ננסה לשאוב רק שדות ידועים
    keys = [
        "env",
        "profile",
        "ib_enable",
        "ib_host",
        "ib_port",
        "ib_client_id",
        "ib_account",
        "sql_store_url",
        "sql_table_prefix",
        "sql_read_only",
        "data_dir",
        "logs_dir",
        "backtest_start",
        "backtest_end",
    ]
    out: Dict[str, Any] = {}
    for k in keys:
        out[k] = _safe_get(settings, k, None)
    return out


def _print_header(title: str) -> None:
    logger.info("")
    logger.info("=" * 80)
    logger.info(">>> %s", title)
    logger.info("=" * 80)


# ===== Main diagnostics sections =====

def diagnose_app_context(verbose: bool = False) -> Dict[str, Any]:
    """
    מאבחן AppContext & Settings.
    מחזיר dict עם מידע שניתן להעביר לחלקים אחרים.
    """
    _print_header("AppContext & Settings")

    try:
        from core.app_context import AppContext
    except Exception as e:
        logger.error("Failed to import core.app_context.AppContext: %s", e)
        return {}

    try:
        ctx = AppContext.get_global()
    except Exception as e:
        logger.error("AppContext.get_global() failed: %s", e)
        return {}

    settings = getattr(ctx, "settings", None)
    if settings is None:
        logger.error("AppContext.settings is None — cannot proceed.")
        return {}

    logger.info("Settings type: %s", type(settings))

    sdict = _serialize_settings(settings)

    # env/profile
    env = sdict.get("env") or _safe_get(settings, "env", None)
    profile = sdict.get("profile") or _safe_get(settings, "profile", None)
    logger.info("env=%r, profile=%r", env, profile)

    # IB-related
    logger.info("--- IBKR settings ---")
    for key in ("ib_enable", "ib_mode", "ib_host", "ib_port", "ib_client_id", "ib_account"):
        logger.info("%-16s = %r", key, sdict.get(key, _safe_get(settings, key, None)))

    # SQL-related
    logger.info("--- SqlStore settings ---")
    for key in ("sql_store_url", "sql_table_prefix", "sql_read_only"):
        logger.info("%-16s = %r", key, sdict.get(key, _safe_get(settings, key, None)))

    # Paths
    logger.info("--- Paths / dirs ---")
    for key in ("data_dir", "logs_dir", "studies_dir", "exports_dir", "snapshots_dir"):
        logger.info("%-16s = %r", key, sdict.get(key, _safe_get(settings, key, None)))

    # אם יש config_path או דומה — נדפיס
    for attr in ("config_path", "config_file", "source_path", "file_path"):
        val = _safe_get(settings, attr, None)
        if val:
            logger.info("%-16s = %r", attr, val)

    if verbose:
        logger.info("--- Full settings dict (truncated to JSON-safe) ---")
        try:
            json_safe = {}
            for k, v in sdict.items():
                if isinstance(v, (str, int, float, bool)) or v is None:
                    json_safe[k] = v
                elif isinstance(v, (list, dict)):
                    json_safe[k] = v
                else:
                    json_safe[k] = str(v)
            logger.info(json.dumps(json_safe, indent=2, ensure_ascii=False))
        except Exception as e:
            logger.warning("Failed to dump settings as JSON: %s", e)

    return {"ctx": ctx, "settings": settings, "settings_dict": sdict}


def diagnose_sql_store(settings: Any, verbose: bool = False) -> None:
    """
    מאבחן SqlStore:
    - engine_url / dialect / env
    - רשימת טבלאות
    - row counts לטבלאות מפתח.
    """
    _print_header("SqlStore Diagnostics")

    try:
        from core.sql_store import SqlStore
    except Exception as e:
        logger.error("Failed to import core.sql_store.SqlStore: %s", e)
        return

    try:
        store = SqlStore.from_settings(settings)
    except Exception as e:
        logger.error("SqlStore.from_settings() failed: %s", e)
        return

    info = store.get_engine_info()
    logger.info("SqlStore engine_url: %s", info.get("engine_url"))
    logger.info("SqlStore dialect   : %s", info.get("dialect"))
    logger.info("SqlStore env       : %s", info.get("default_env"))
    logger.info("SqlStore read_only : %s", info.get("read_only"))

    tables = store.list_tables()
    logger.info("Tables (%d): %s", len(tables), ", ".join(tables) if tables else "(none)")

    key_tables = [
        "dq_pairs",
        "bt_runs",
        "prices",
        "risk_state",
        "risk_timeline",
        "dashboard_snapshots",
        "kv_store",
    ]

    for short_name in key_tables:
        tbl = store._tbl(short_name)
        if tbl not in tables:
            logger.warning("Table %-24s NOT FOUND", tbl)
            continue

        try:
            q = f"SELECT COUNT(*) AS n FROM {tbl}"
            df = store.raw_query(q)
            n = int(df["n"].iloc[0]) if not df.empty else 0
        except Exception as e:
            logger.warning("Failed to count rows in %s: %s", tbl, e)
            n = -1

        logger.info("Table %-24s rows = %d", tbl, n)

    if verbose and "prices" in [store._tbl(t.replace(store.table_prefix, "")) for t in tables]:
        try:
            df_sample = store.raw_query(
                f"SELECT symbol, MIN(date) AS first_date, MAX(date) AS last_date, COUNT(*) AS n "
                f"FROM {store._tbl('prices')} GROUP BY symbol ORDER BY symbol LIMIT 20"
            )
            if not df_sample.empty:
                logger.info("--- Price coverage sample (first 20 symbols) ---")
                logger.info(df_sample.to_string(index=False))
        except Exception as e:
            logger.warning("Failed to query prices coverage: %s", e)


def diagnose_ib_connection(settings: Any) -> None:
    """
    מנסה לבדוק חיבור ל-IBKR אם ib_enable=True:

    - מייבא root.ibkr_connection אם קיים.
    - משתמש ב-get_ib_instance / ib_connection_status אם קיימים.
    - מדפיס מצב חיבור, מספר פוזיציות, מספר הזמנות פתוחות.
    """
    _print_header("IBKR Diagnostics")

    ib_enable = _safe_get(settings, "ib_enable", False)
    if not ib_enable:
        logger.warning("ib_enable is False in settings — IBKR is logically disabled.")
        return

    # לנסות לטעון ib_insync
    try:
        import ib_insync  # type: ignore
        logger.info("ib_insync version: %s", getattr(ib_insync, "__version__", "unknown"))
    except Exception as e:
        logger.error("ib_insync import failed: %s", e)
        logger.error("Cannot diagnose IBKR without ib_insync installed.")
        return

    # לנסות לחבר דרך root.ibkr_connection
    try:
        ibkr_module = importlib.import_module("root.ibkr_connection")
    except Exception as e:
        logger.error("Failed to import root.ibkr_connection: %s", e)
        return

    get_ib_instance = getattr(ibkr_module, "get_ib_instance", None)
    ib_connection_status = getattr(ibkr_module, "ib_connection_status", None)

    if not callable(get_ib_instance):
        logger.error("root.ibkr_connection.get_ib_instance is not callable or missing.")
        return

    logger.info("Connecting to IBKR via root.ibkr_connection.get_ib_instance(...)")
    try:
        ib = get_ib_instance(
            readonly=True,
            use_singleton=False,
            profile=_safe_get(settings, "ib_mode", None),
            settings=getattr(settings, "as_dict", lambda: settings)(),
        )
    except Exception as e:
        logger.error("get_ib_instance(...) raised error: %s", e)
        return

    if ib is None:
        logger.error("get_ib_instance returned None — no IB connection available.")
        return

    try:
        status = ib_connection_status(ib) if callable(ib_connection_status) else {}
    except Exception:
        status = {}

    logger.info("IB connection status: %r", status)

    try:
        connected = ib.isConnected()
    except Exception:
        connected = None

    logger.info("ib.isConnected() = %r", connected)

    # פוזיציות
    try:
        positions = ib.positions()
        logger.info("IB positions: %d", len(positions))
    except Exception as e:
        logger.warning("ib.positions() failed: %s", e)

    # הזמנות פתוחות
    try:
        open_orders = ib.openOrders()
        logger.info("IB openOrders: %d", len(open_orders))
    except Exception as e:
        logger.warning("ib.openOrders() failed: %s", e)

    try:
        ib.disconnect()
    except Exception:
        pass


def diagnose_environment() -> None:
    """
    מציג מידע כללי על סביבת הריצה (Python, cwd, sys.path וכו').
    """
    _print_header("Environment & Python")

    logger.info("Python version: %s", sys.version.replace("\n", " "))
    logger.info("Executable     : %s", sys.executable)
    logger.info("CWD            : %s", os.getcwd())

    # לבדוק אם רצים בתוך streamlit (בקירוב)
    is_streamlit = "streamlit" in sys.argv[0].lower() or any("streamlit" in p for p in sys.argv)
    logger.info("Running under Streamlit: %s", is_streamlit)

    # קצת path info
    root = Path(__file__).resolve().parent.parent
    logger.info("Project root   : %s", root)
    if str(root) not in sys.path:
        logger.info("Project root is NOT in sys.path (this is OK for most cases).")


# ===== Main =====

def main() -> None:
    parser = argparse.ArgumentParser(
        description="HF-grade diagnostics for AppContext, Settings, SqlStore and IBKR"
    )
    parser.add_argument(
        "--no-sql",
        action="store_true",
        help="Skip SqlStore diagnostics",
    )
    parser.add_argument(
        "--no-ib",
        action="store_true",
        help="Skip IBKR diagnostics",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show verbose information (settings JSON, price coverage, etc.)",
    )

    args = parser.parse_args()

    diagnose_environment()

    ctx_info = diagnose_app_context(verbose=args.verbose)
    settings = ctx_info.get("settings")

    if settings is None:
        logger.error("No settings available — skipping SqlStore/IB diagnostics.")
        return

    if not args.no_sql:
        diagnose_sql_store(settings, verbose=args.verbose)

    if not args.no_ib:
        diagnose_ib_connection(settings)


if __name__ == "__main__":
    main()
