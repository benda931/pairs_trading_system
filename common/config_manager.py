# -*- coding: utf-8 -*-
"""
config_manager.py – Load, save, list, and validate dashboard configurations.
============================================================================

תפקיד הקובץ
-----------
ניהול מלא של קובצי הקונפיגורציה של הדשבורד:

- טעינה + ולידציה עם Pydantic (תמיכה גם ב־v1 וגם ב־v2).
- שמירת קונפיג ראשי (`config.json`) + פרופילים גרסתיים בתיקיית `configs/`.
- התעלמות ממפתחות לא מוכרים (forward-compatible) **תוך שימור שלהם בקובץ**.
- שדרוג אוטומטי של קונפיגים ישנים לסכמה החדשה (הוספת שדות ברירת מחדל וכו').

API ציבורי
-----------
* load_config()              – טוען קונפיג כ־dict (ויוצר ברירת מחדל אם חסר/שבור).
* load_config_model()        – כמו load_config אבל מחזיר DashboardConfig (אובייקט Pydantic).
* load_raw_config()          – טוען את ה־JSON כמו שהוא, ללא ולידציה.
* save_config()              – שומר קונפיג ראשי (עם ולידציה ברירת מחדל).
* save_config_profile()      – שומר snapshot בתיקיית `configs/` עם timestamp.
* list_configs()             – מחזיר רשימת קובצי פרופילים בתיקיית `configs/`.
* ensure_config_dir()        – יוצר את תיקיית `configs/` אם חסרה.
* upgrade_config_dict()      – מריץ ולידציה + שדרוג dict של קונפיג (ללא I/O דיסק).
* load_default_config()      – wrapper נוח ל־load_config(config.json).

הערות חשובות
-------------
- כל מפתח שאינו חלק מ־DashboardConfig **נשמר** בקובץ ולא נזרק.
- אם הקובץ לא JSON תקין או לא עובר ולידציה – נכתב קובץ חדש עם ברירת מחדל.
- תומך בתצורה עשירה: UI, ביצועים, Backtest, Optimizer, ML, IBKR, Risk וכו'.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Dict, List

from common.json_safe import make_json_safe, json_default as _json_default

# -------------------------------------------------------------
# Pydantic compatibility layer (v1 & v2)
# -------------------------------------------------------------
try:
    # Pydantic v2 (preferred)
    from pydantic import BaseModel, Field, ValidationError
    from pydantic import field_validator, ConfigDict  # type: ignore

    PYDANTIC_V2 = True
except ImportError:  # pragma: no cover - fallback to v1
    from pydantic import BaseModel, Field, ValidationError, validator  # type: ignore

    PYDANTIC_V2 = False

    # התאמת API של v2 ל־v1 – field_validator מתלבש על validator
    def field_validator(*fields: str, **kwargs):  # type: ignore
        return validator(*fields, **kwargs)  # type: ignore

    class ConfigDict(dict):  # type: ignore
        """Dummy replacement so type checkers לא יתעצבנו."""
        pass


logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.NullHandler())


# -------------------------------------------------------------
# Paths & constants
# -------------------------------------------------------------
# אפשרות לשלוט בשורש הקונפיג דרך משתנה סביבה – נוח למעבר בין מכונה מקומית/שרת.
ENV_CONFIG_ROOT = "PAIRS_TRADING_CONFIG_ROOT"

if ENV_CONFIG_ROOT in os.environ:
    PROJECT_ROOT = Path(os.environ[ENV_CONFIG_ROOT]).expanduser().resolve()
else:
    PROJECT_ROOT = Path(__file__).resolve().parent

CONFIG_PATH = PROJECT_ROOT / "config.json"   # main config
CONFIG_DIR = PROJECT_ROOT / "configs"        # versioned profiles (snapshots)


# -------------------------------------------------------------
# Pydantic model – DashboardConfig (HF-grade)
# -------------------------------------------------------------
class DashboardConfig(BaseModel):
    """
    סכמה מרכזית לקונפיג הדשבורד.

    שדות רבים מכסים:
        - UI & UX
        - ביצועים / cache / heavy panels
        - Backtest defaults
        - Optimizer defaults
        - ML / SHAP / GPU toggles
        - חיבור IBKR בסיסי
        - Risk limits ברמת פורטפוליו
    """

    # ---- Metadata ----
    version: str = Field(
        default="1.0.0",
        description="Dashboard config schema version (helps with migrations).",
    )
    last_updated: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        description="Last modification timestamp (UTC, ISO-8601).",
    )

    # ---- Storage / SqlStore (optional, for convenience) ----
    engine_url: str | None = Field(
        default=None,
        description="Primary engine URL for SqlStore (e.g. duckdb:///C:/.../cache.duckdb).",
    )
    sql_store_url: str | None = Field(
        default=None,
        description="Legacy alias for engine_url; kept for backward compatibility.",
    )

    # ---- UI / UX ----
    theme: str = Field(
        default="light",
        description="UI theme (light/dark/auto).",
    )
    accent: str = Field(
        default="Blue",
        description="Accent colour display name.",
    )
    language: str = Field(
        default="he",
        description="Default UI language code, e.g. 'he' or 'en'.",
    )
    compact_layout: bool = Field(
        default=False,
        description="Use compact layout (tighter spacing, more info per screen).",
    )
    default_tab: str = Field(
        default="Dashboard",
        description="Tab name to open when the app starts.",
    )
    show_experimental_tabs: bool = Field(
        default=False,
        description="Show experimental / beta tabs in the UI.",
    )

    # ---- Performance / Caching ----
    enable_heavy_panels: bool = Field(
        default=False,
        description="Allow heavy ML / SHAP / macro panels to run.",
    )
    max_rows_per_table: int = Field(
        default=5_000,
        ge=500,
        le=1_000_000,
        description="Safety cap for rows in big tables (performance guardrail).",
    )
    cache_ttl_minutes: int = Field(
        default=30,
        ge=1,
        le=24 * 60,
        description="TTL for in-memory caches used by the dashboard.",
    )

    # ---- Directories (relative to PROJECT_ROOT unless absolute) ----
    data_dir: str = Field(
        default="data",
        description="Directory for cached price data / pre-processed datasets.",
    )
    logs_dir: str = Field(
        default="logs",
        description="Directory for application logs.",
    )
    studies_dir: str = Field(
        default="studies",
        description="Directory for Optuna studies / research outputs.",
    )

    # ---- Backtest defaults ----
    backtest_start: str = Field(
        default="2020-01-01",
        description="Default start date for backtests (YYYY-MM-DD).",
    )
    backtest_end: str = Field(
        default="2024-12-31",
        description="Default end date for backtests (YYYY-MM-DD).",
    )
    default_frequency: str = Field(
        default="D",
        description="Default resampling frequency (e.g. 'D', 'H', 'W').",
    )
    benchmark_symbol: str = Field(
        default="SPY",
        description="Default benchmark for backtest comparison.",
    )

    # ---- Optimizer defaults ----
    max_trials: int = Field(
        default=100,
        ge=10,
        le=5_000,
        description="Default Optuna trials for optimization tab.",
    )
    n_jobs: int = Field(
        default=1,
        ge=-1,
        le=128,
        description="Number of parallel jobs for optimization (-1 = all cores).",
    )
    sampler: str = Field(
        default="tpe",
        description="Optuna sampler name (e.g. 'tpe', 'cmaes').",
    )
    pruner: str = Field(
        default="median",
        description="Optuna pruner name (e.g. 'median', 'successive_halving').",
    )

    # ---- ML / Analysis ----
    enable_ml_analysis: bool = Field(
        default=True,
        description="Enable ML analysis (XGBoost/PCA/clustering panels).",
    )
    enable_shap: bool = Field(
        default=True,
        description="Compute SHAP when `shap` is installed.",
    )
    enable_xgboost: bool = Field(
        default=True,
        description="Use XGBoost when available.",
    )
    enable_gpu: bool = Field(
        default=False,
        description="Allow GPU usage for XGBoost/other models when supported.",
    )

    # ---- IBKR / Live Trading (basic) ----
    ib_enable: bool = Field(
        default=False,
        description="Enable IBKR integration in the UI.",
    )
    ib_host: str = Field(
        default="127.0.0.1",
        description="IBKR host (usually 127.0.0.1).",
    )
    ib_port: int = Field(
        default=7497,
        ge=1,
        le=65_535,
        description="IBKR API port (paper/live).",
    )
    ib_client_id: int = Field(
        default=1,
        ge=1,
        le=32,
        description="Client ID for IBKR sessions.",
    )

    # ---- Risk defaults (global) ----
    max_gross_exposure: float = Field(
        default=1_000_000.0,
        ge=0.0,
        description="Max gross exposure (notional) for the fund.",
    )
    max_leverage: float = Field(
        default=5.0,
        ge=0.1,
        le=20.0,
        description="Max portfolio leverage.",
    )
    per_pair_risk_budget: float = Field(
        default=0.01,
        ge=0.0001,
        le=0.1,
        description="Typical risk budget per pair (fraction of equity).",
    )

    # ---- Pydantic configuration: ignore unknown keys ----
    if PYDANTIC_V2:
        model_config: ConfigDict = ConfigDict(
            extra="ignore",
        )
    else:  # pragma: no cover - v1 style config
        class Config:  # type: ignore
            extra = "ignore"

    # ---- Validators ----
    @field_validator("accent", mode="before")
    def _accent_title(cls, v: Any) -> str:
        """Ensure accent is title-cased."""
        if v is None:
            return "Blue"
        v_str = str(v)
        return v_str.title()

    @field_validator("theme", mode="before")
    def _theme_normalize(cls, v: Any) -> str:
        """Normalize theme value to canonical tokens."""
        if v is None:
            return "light"
        v_str = str(v).lower().strip()
        if v_str in {"dark", "darkmode", "dark_mode"}:
            return "dark"
        if v_str in {"auto", "system"}:
            return "auto"
        return "light"

    @field_validator("language", mode="before")
    def _language_lower(cls, v: Any) -> str:
        """Normalize language code to lower-case (e.g. 'he', 'en')."""
        if v is None:
            return "he"
        return str(v).lower()

    @field_validator("backtest_start", "backtest_end", mode="before")
    def _date_sanity(cls, v: Any) -> str:
        """
        Validate that the date string לפחות בפורמט YYYY-MM-DD.
        לא מרים חריגה על כל טעות קטנה, אבל כן מנסה לשמור על פורמט תקין.
        """
        if v is None:
            return "2020-01-01"
        v_str = str(v).strip()
        # פשוט – רק לוודא אורך וצורת שנה-חודש-יום
        try:
            datetime.strptime(v_str, "%Y-%m-%d")
        except ValueError:
            # במקרה של ערך חסר/שבור – ניפול לברירת מחדל
            return "2020-01-01"
        return v_str


# -------------------------------------------------------------
# Internal helpers
# -------------------------------------------------------------
def _model_to_dict(cfg: DashboardConfig) -> Dict[str, Any]:
    """Convert model to plain dict, Pydantic v1/v2 safe."""
    if PYDANTIC_V2:
        return cfg.model_dump()
    return cfg.dict()


def _json_dumps(data: Dict[str, Any]) -> str:
    """Serialize config dict with json_safe + fallback encoder."""
    safe = make_json_safe(data)
    return json.dumps(
        safe,
        default=_json_default,
        ensure_ascii=False,
        indent=2,
    )


def _write_json(data: Dict[str, Any], path: Path) -> None:
    path.write_text(_json_dumps(data), encoding="utf-8")
    try:
        rel = path.relative_to(PROJECT_ROOT)
    except ValueError:
        rel = path
    logger.info("Config saved to %s", rel)


def ensure_config_dir() -> None:
    """Create `configs/` folder if it does not exist."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def list_configs() -> List[str]:
    """Return list of JSON profile filenames under `configs/`."""
    ensure_config_dir()
    return sorted(p.name for p in CONFIG_DIR.glob("*.json"))


# -------------------------------------------------------------
# Upgrade / validation helpers (no I/O)
# -------------------------------------------------------------
def upgrade_config_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run validation + schema upgrade on a raw config dict.

    - שומר על כל המפתחות הלא מוכרים (forward-compatible).
    - מוסיף שדות ברירת מחדל שחסרים לפי DashboardConfig.
    - מחזיר dict עם הערכים אחרי ולידציה (הערכים התקינים גוברים על המקור).
    """
    cfg_obj = DashboardConfig(**data)
    validated = _model_to_dict(cfg_obj)

    merged: Dict[str, Any] = dict(data)  # copy raw
    merged.update(validated)             # validated values override raw
    return merged


# -------------------------------------------------------------
# Core load / save
# -------------------------------------------------------------
def _default_config() -> DashboardConfig:
    """Return a fresh default DashboardConfig object."""
    return DashboardConfig()


def load_raw_config(path: str | Path | None = None) -> Dict[str, Any]:
    """
    Load config JSON from disk **without** Pydantic validation.

    שימושי לדיבאגר/מיגרציות, אבל לרוב עדיף להשתמש ב־load_config().
    """
    cfg_path = Path(path) if path else CONFIG_PATH
    if not cfg_path.exists():
        return {}
    try:
        return json.loads(cfg_path.read_text(encoding="utf-8"))
    except JSONDecodeError as exc:
        logger.warning("Config file %s is not valid JSON: %s", cfg_path, exc)
        return {}


def load_config(path: str | Path | None = None) -> Dict[str, Any]:
    """
    Load config file into dict. Creates default if missing or invalid.

    - אם הקובץ לא קיים → נכתב קובץ ברירת מחדל ומוחזר dict ברירת מחדל.
    - אם JSON שבור או ולידציה נכשלת → נכתב קובץ ברירת מחדל.
    - אם הכול תקין → dict משודרג (כולל שדות חדשים) נשמר ומוחזר.
    """
    cfg_path = Path(path) if path else CONFIG_PATH

    if not cfg_path.exists():
        logger.info("Main config missing – creating default at %s", cfg_path)
        cfg_obj = _default_config()
        data = _model_to_dict(cfg_obj)
        _write_json(data, cfg_path)
        return data

    # שלב 1: ניסיון לטעון JSON גולמי
    try:
        raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    except JSONDecodeError as exc:
        logger.warning(
            "Config file %s is not valid JSON (%s) – resetting to default.",
            cfg_path,
            exc,
        )
        cfg_obj = _default_config()
        data = _model_to_dict(cfg_obj)
        _write_json(data, cfg_path)
        return data

    # שלב 2: ולידציה + שדרוג
    try:
        upgraded = upgrade_config_dict(raw)
    except ValidationError as exc:
        logger.warning(
            "Invalid config at %s – resetting to default: %s",
            cfg_path,
            exc.errors(),
        )
        cfg_obj = _default_config()
        data = _model_to_dict(cfg_obj)
        _write_json(data, cfg_path)
        return data

    # שלב 3: עדכון שדה last_updated (כשהקונפיג שורד ולידציה)
    upgraded["last_updated"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # רושמים חזרה לדיסק (כולל שדות חדשים/מנורמלים)
    _write_json(upgraded, cfg_path)
    return upgraded


def load_config_model(path: str | Path | None = None) -> DashboardConfig:
    """
    Load config and return DashboardConfig object במקום dict.

    שימושי ללוגיקה פנימית שצריכה סוגים חזקים (IDE, type checking).
    """
    data = load_config(path)
    # כאן המידע כבר עבר upgrade_config_dict, אז ולידציה אמורה להיות טריוויאלית
    return DashboardConfig(**data)


def save_config(
    data: Dict[str, Any],
    path: str | Path | None = None,
    validate: bool = True,
) -> None:
    """
    Overwrite main config (or custom path) with *data*.

    Parameters
    ----------
    data : Dict[str, Any]
        קונפיג כ־dict. אפשר להעביר dict חלקי – השדות החסרים ימולאו בברירת מחדל.
    path : str | Path | None
        נתיב שמירה. ברירת מחדל: CONFIG_PATH.
    validate : bool
        אם True, מריץ upgrade_config_dict כדי לוודא שהקונפיג תקין לפני השמירה.
    """
    cfg_path = Path(path) if path else CONFIG_PATH

    if validate:
        try:
            data = upgrade_config_dict(data)
        except ValidationError as exc:
            logger.error("Failed to validate config before save: %s", exc.errors())
            raise

    _write_json(data, cfg_path)


# -------------------------------------------------------------
# Versioned profile helpers
# -------------------------------------------------------------
def save_config_profile(
    data: Dict[str, Any],
    profile: str | None = None,
    validate: bool = True,
) -> str:
    """
    Save *data* under `configs/` with timestamp-based filename.

    Returns
    -------
    str
        The filename created (relative, not full path).
    """
    ensure_config_dir()

    if validate:
        try:
            data = upgrade_config_dict(data)
        except ValidationError as exc:
            logger.error(
                "Failed to validate config before saving profile: %s",
                exc.errors(),
            )
            raise

    if profile is None:
        profile = f"config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    path = CONFIG_DIR / profile
    _write_json(data, path)
    return profile


def load_default_config() -> Dict[str, Any]:
    """Convenience wrapper for main config."""
    return load_config(CONFIG_PATH)

# === Simple JSON-based settings loader for CLI scripts ===
from functools import lru_cache

try:
    PROJECT_ROOT  # אם כבר מוגדר למעלה – לא נדרוס
except NameError:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.json"


@lru_cache()
def load_settings(config_path: Path | str = DEFAULT_CONFIG_PATH) -> dict:
    """
    טוען את config.json ומחזיר dict עם כל ההגדרות.

    מיועד במיוחד לסקריפטים כמו ingest_ibkr_prices.py / backtest_pair_from_sql.py
    כדי שיהיה להם API פשוט ואחיד.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"config.json not found at: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # אפשר להוסיף כאן normalization קטן אם תרצה בעתיד
    return data

# -------------------------------------------------------------
# Public exports
# -------------------------------------------------------------
__all__ = [
    "DashboardConfig",
    "load_config",
    "load_config_model",
    "load_raw_config",
    "save_config",
    "save_config_profile",
    "list_configs",
    "ensure_config_dir",
    "upgrade_config_dict",
    "load_default_config",
    "PROJECT_ROOT",
    "CONFIG_PATH",
    "CONFIG_DIR",
]
