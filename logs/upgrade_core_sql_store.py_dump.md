### Summary of changes

- Added schema ensure helpers for core tables (`pairs_universe`, `trades`, `config_snapshots`, `metrics`) and a generic `_ensure_index` utility; wired them into `SqlStore.__init__` (non–read-only only).
- Hardened the `prices` schema by adding a `freq` column and creating practical indices on `(symbol, date)` and `(symbol, env, date)` when possible.
- Extended `save_price_history` / `load_price_history` to support intraday vs EOD partitioning via a `freq` column, optional history-retention policy (`max_history_days`), and optional date-gap anomaly logging.
- Added a `save_config_snapshot` API plus `_ensure_config_snapshots_schema` to persist run/config/optimizer metadata (including `git_rev`) alongside artifacts.
- Introduced lightweight health/ops helpers: `ping()`, `get_table_row_counts()`, and `get_health_snapshot()` for use by web dashboards and background jobs.
- Implemented a small internal `_log_price_gaps` helper for structured gap/anomaly logging during price ingestion.
- Kept all existing public APIs intact, only adding optional keyword parameters and new methods to avoid breaking callers.
- Minor style/indent fixes and consistent error logging/last_error updates.

---

```python
# -*- coding: utf-8 -*-
"""
core/sql_store.py — SQL Persistence Layer (HF-grade, Fund-Level, v2)
=====================================================================

שכבת Persist מרכזית לכל המודולים במערכת:

תומך ב:
    - Data Quality (symbols / pairs)
    - Signals (universe signals + summaries)
    - Risk (risk_state / risk_timeline)
    - Experiments & Backtests (optional hooks)
    - Context snapshots (ctx, run metadata)
    - Dashboard snapshots (DashboardSnapshot)
    - Key-Value JSON store (user prefs, views, misc dashboards)
    - Price History (EOD / OHLCV) ממקור אמת אחד לטווח הארוך.

מבנה הקובץ (v2):
----------------
חלק 1/3 — Core Infra & Engine Wiring (הקובץ הזה):
    * Header, imports, קבועים (PROJECT_ROOT / LOGS_DIR).
    * SqlStore.__init__ עם יצירת Engine חכמה (DuckDB/SQLite/Postgres).
    * from_settings מקצועי עם:
        - תמיכה ב־AppContext.settings, dict, env vars.
        - ברירת מחדל: DuckDB ב-logs/pairs_trading_<env>.duckdb.
    * פונקציות עזר:
        - _now_utc_iso
        - _tbl (ניהול table_prefix)
        - list_tables / get_engine_info / describe_engine
        - raw_query / read_table
        - _ensure_* schema:
            kv_store, dashboard_snapshots, prices.
        - _ensure_writable + get_last_error.

חלק 2/3 — Data Persistence Layers (ייבנה בהמשך):
    * Data Quality / Signals / Smart Scan / Fair Value / Fundamentals / Metrics.
    * Universe loaders מתקדמים (latest_only, profile/env aware).
    * Backtest metrics flattening, experiment_runs וכו'.

חלק 3/3 — Risk / Snapshots / KV / Prices API מתקדם (ייבנה בהמשך):
    * RiskState / RiskLimits persist.
    * Context snapshots, DashboardSnapshot typed + generic.
    * KV JSON store (prefs/views/other) עם ניהול גרסאות.
    * Price history API מורחב (rollups, sanity checks, health summaries).

עקרונות:
    1. מחלקת SqlStore אחת שמנהלת Engine + קונבנציית טבלאות.
    2. כל שמירה מקבלת:
        - run_id (אופציונלי)
        - section/profile/env (labels)
        - ts_utc אחיד ב-UTC.
    3. API ידידותי ל־pandas / dicts / מודולים אחרים ב-core.
    4. מתאים ל-SQLAlchemy engines: SQLite / DuckDB / Postgres / וכו'.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, List, Mapping, Sequence

import json
import logging
import os

import pandas as pd
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.engine import Engine

from core.data_quality import (
    data_quality_symbols_to_sql_ready,
    data_quality_pairs_to_sql_ready,
)
from core.risk_engine import RiskState, RiskLimits
from core.signals_engine import summarize_universe_signals
from core.dashboard_models import DashboardSnapshot

logger = logging.getLogger(__name__)

JSONDict = Dict[str, Any]

# שורש הפרויקט (בהנחה שהקובץ תחת core/)
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
LOGS_DIR: Path = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)


def _now_utc_iso() -> str:
    """
    מחזיר timestamp ב-UTC בפורמט ISO סטנדרטי עם 'Z' (ללא timezone offset).

    זהו format אחיד לכל ה-ts_utc במערכת.
    """
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_json_dumps(obj: Any) -> str:
    """
    המרה בטוחה ל-JSON עם fallback ל-str.

    שימושי ל-payloadים מורכבים (metrics_json, config_json וכו').
    """
    try:
        return json.dumps(obj, default=str)
    except TypeError:
        return json.dumps(str(obj))


class SqlStore:
    """
    SqlStore — שכבת Persist ברמת קרן.

    מאפיינים:
    ----------
    - engine_url: מחרוזת SQLAlchemy, למשל:
        * "sqlite:///pairs_trading.db"
        * "duckdb:///C:/.../logs/pairs_trading_dev.duckdb"
        * "postgresql+psycopg2://user:pass@host:port/dbname"

    - table_prefix: קידומת לטבלאות (למשל "pt_" לקרן שלך)
    - default_env: סימון environment ("dev" / "paper" / "live")
    - read_only: מצב "קריאה בלבד" (ל-Health / Dashboard)

    תוספות v2 (חלק 1):
    -------------------
    * from_settings חכם:
        - קורא מה־env vars, attributes, config dict.
        - ברירת מחדל: DuckDB ב-logs/pairs_trading_<env>.duckdb.
    * describe_engine / get_engine_info להנדסת מערכת ו-AI Agents.
    * _ensure_prices_schema כדי למנוע שגיאות "table does not exist" עבור prices.
    * last_error מנוהל ברמת מחלקה (לשימוש בדשבורד / SystemHealthSnapshot).
    """

    def __init__(
        self,
        engine_url: str,
        *,
        echo: bool = False,
        table_prefix: str = "",
        default_env: str = "default",
        read_only: bool = False,
    ) -> None:
        self.engine_url = engine_url
        self.table_prefix = table_prefix.strip()
        self.default_env = default_env.strip() or "default"
        self.read_only: bool = bool(read_only)

        # יצירת Engine (HF-grade: future=True + echo לפי settings)
        self.engine: Engine = create_engine(
            engine_url,
            echo=echo,
            future=True,
        )

        self._dialect = self.engine.dialect.name.lower()
        self._last_error: Optional[str] = None  # שגיאה אחרונה (לבריאות מערכת)
        logger.info(
            "SqlStore initialized (url=%s, dialect=%s, prefix=%s, env=%s, read_only=%s)",
            self.engine_url,
            self._dialect,
            self.table_prefix,
            self.default_env,
            self.read_only,
        )

        # סכמות בסיסיות — חשוב שיהיו קיימות לפני שימוש
        # במצב read_only *לא* מריצים שום migrate/CREATE TABLE כדי להימנע מקונפליקטים
        if not self.read_only:
            try:
                self._ensure_dashboard_schema()
            except Exception as e:
                logger.warning("Failed to ensure dashboard_snapshots schema: %s", e)

            try:
                self._ensure_kv_schema()
            except Exception as e:
                logger.warning("Failed to ensure kv_store schema: %s", e)

            try:
                self._ensure_prices_schema()
            except Exception as e:
                logger.warning("Failed to ensure prices schema: %s", e)

            # core tables for live trading / universes / configs / metrics / trades
            try:
                self._ensure_pairs_universe_schema()
            except Exception as e:
                logger.warning("Failed to ensure pairs_universe schema: %s", e)

            try:
                self._ensure_config_snapshots_schema()
            except Exception as e:
                logger.warning("Failed to ensure config_snapshots schema: %s", e)

            try:
                self._ensure_metrics_schema()
            except Exception as e:
                logger.warning("Failed to ensure metrics schema: %s", e)

            try:
                self._ensure_trades_schema()
            except Exception as e:
                logger.warning("Failed to ensure trades schema: %s", e)
        else:
            logger.info("SqlStore in read_only=True — skipping schema ensure/_migrate calls.")

    # ------------------------------------------------------------------
    # Factory מ-App settings (חיבור לדשבורד / AppContext)
    # ------------------------------------------------------------------

    @classmethod
    def from_settings(
        cls,
        settings: Any,
        *,
        env: Optional[str] = None,
        table_prefix: str = "",
        echo: Optional[bool] = None,
        read_only: bool = False,
    ) -> "SqlStore":
        """
        Factory מקצועי שיודע להקים SqlStore מתוך AppContext.settings או dict.

        סדר החיפוש ל-engine_url (מותאם ל-config v2.2.0 + תאימות לאחור):

        1. Environment variables (אם מוגדרים):
           - SQL_STORE_URL
           - PAIRS_SQL_STORE_URL

        2. attributes על settings (אם זה אובייקט):
           - settings.sql_store.engine_url / settings.sql_store.url
           - settings.engine_url
           - settings.sql_store_url / settings.SQL_STORE_URL
           - settings.db_url / settings.DB_URL
           - settings.sqlalchemy_url / settings.SQLALCHEMY_URL

        3. settings.config או settings עצמו אם הוא Mapping:
           לפי סדר:
             - config["sql_store"]["engine_url"] / ["url"]
             - config["data"]["sql_store"]["engine_url"] / ["url"]
             - config["paths"]["sql_store_url"]
             - config["paths"]["duckdb_cache_path"]  → duckdb:///<path>
             - config["data"]["duckdb_cache"]["path"] → duckdb:///<path>
             - config["engine_url"]
             - וגם המפתחות הישנים ברוט:
               "sql_store_url", "SQL_STORE_URL", "db_url", "DB_URL",
               "sqlalchemy_url", "SQLALCHEMY_URL"

        4. Fallback ברמת קרן (legacy בלבד אם אין שום רמז מהקונפיג):
           - DuckDB תחת logs/pairs_trading_<env>.duckdb

        בנוסף:
        - env_name נגזר מ:
            env (arg explict) → settings.env/ENV/profile/PROFILE →
            config["environment"]["default_env"] → "dev"
        - table_prefix:
            אם לא הועבר:
              SQL_TABLE_PREFIX / PAIRS_SQL_TABLE_PREFIX (ENV)
              settings.sql_table_prefix
              config["sql_store"]["prefix"]
        - echo:
            arg → settings.sql_echo → ENV["SQL_ECHO"] (1/true/on) → False
        - read_only:
            arg → settings.sql_read_only → config["sql_store"]["read_only"]
            → ENV["SQL_READ_ONLY"]/["PAIRS_SQL_READ_ONLY"] → False
        """

        # ===== 0) להוציא config כ-Mapping (settings או settings.config) =====
        # אם settings הוא dict / Mapping → זה ה-config.
        if isinstance(settings, Mapping):
            cfg: Mapping[str, Any] = settings
        else:
            cfg = getattr(settings, "config", {}) or {}
            if not isinstance(cfg, Mapping):
                cfg = {}

        # כלי עזר קטן לשאיבה בטוחה מ-nested dict
       