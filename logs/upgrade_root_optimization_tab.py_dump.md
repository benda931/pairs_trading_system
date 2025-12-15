Summary of changes
- Added a dedicated stub `persist_opt_run_to_sqlstore` and wired it from `optimize_pair_with_telemetry`, so all optimisation runs have a single hook to integrate with `SqlStore`/Parquet later (currently safe no-op).
- Extended `optimize_pair_with_telemetry` to persist richer artifacts per study: manifest, optional Parquet/CSV of full results, and an `opt_config` snapshot, all stored via the existing DuckDB artifacts table.
- Strengthened environment awareness in `render_optimization_tab`: derive an effective env (`local/dev/paper/live/prod`), surface it clearly and gate heavy optimisation actions in LIVE/PROD via an explicit user confirmation checkbox.
- Added a separate LIVE/PROD safeguard for writing best parameters into the shared `opt_best_params_registry`, controlled by `opt_live_update_params`, so live-linked parameters cannot be updated silently.
- Fixed a bug in `register_best_params_for_pair` where `datetime.now(timezone.utc)` was incorrectly called as a function and enhanced the stored entry with the current env.
- Removed an early call to `_render_replay_best_trial_panel` that referenced `TABLE_HEIGHT` before it was defined and was redundant with a later call.
- Enriched optimisation manifests with snapshots of `opt_run_cfg` and `opt_config`, including env and global seed, improving traceability and replay capability.
- Kept all existing public APIs and UI entrypoints intact, treating the legacy `core.optimizer.run_optimization` flow as legacy (still available) while ensuring the main single/batch/Zoom flows continue to use the shared Optuna/Zoom engine with deterministic seeding.

```python
﻿# -*- coding: utf-8 -*-
"""
optimization_tab.py — Hedge-Fund Grade Optimiser (v3)
=====================================================

חלק 1/15 — תשתית כללית:
- טעינת תלויות סטנדרט + צד שלישי בצורה בטוחה.
- עטיפות cache ל-Streamlit (cache_data / cache_resource).
- טעינת ספריות אופציונליות (plotly, duckdb, sklearn, scipy, statsmodels, optuna).
- זיהוי סביבת הרצה (Streamlit / סקריפט / בדיקות).
- הגדרות numpy/pandas, טיפוסי עזר גלובליים.
- פונקציות עזר:
    sk(prefix)          — יצירת מפתחות ייחודיים ל-Streamlit.
    _norm_name(name)    — נירמול שמות (למשל "CMA-ES" -> "CMAES").
    _is_cmaes(name)     — בדיקה האם סמפלר הוא CMA-ES.
    safe_import(name)   — יבוא מודול עם fallback.
"""

from __future__ import annotations

# =========================
# SECTION 0: Standard imports & project root
# =========================
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Iterable, Mapping, Sequence, Union, TypeAlias

import os
import sys
import json
import logging
import atexit
import time
import math
import itertools
import inspect
from datetime import date, datetime, timezone

import warnings
import argparse

# -------- Project root, path injection --------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

APP_NAME = "Pairs-Trading Optimiser"
MODULE_NAME = "root.optimization_tab"
OPT_TAB_VERSION = "3.0.0"

# =========================
# SECTION 0.1: Core numerics & Streamlit
# =========================
import numpy as np
import pandas as pd
import streamlit as st

from core.dashboard_models import DashboardContext
from core.dashboard_service import DashboardService
# במקום: from core.params import PARAM_SPECS
try:
    from core.params import PARAM_SPECS as _PARAM_SPECS_SRC  # type: ignore
except Exception:
    _PARAM_SPECS_SRC = {}  # type: ignore

# נרמול ל-dict[name -> spec]
if isinstance(_PARAM_SPECS_SRC, dict):
    PARAM_SPECS = _PARAM_SPECS_SRC
else:
    tmp: Dict[str, Any] = {}
    try:
        for spec in _PARAM_SPECS_SRC or []:
            name = getattr(spec, "name", None)
            if not name:
                continue
            tmp[str(name)] = spec
    except Exception:
        pass
    PARAM_SPECS = tmp


from root.dashboard_service_factory import (
    create_dashboard_service,
    build_default_dashboard_context,
)

# אופציונלי: להשתמש ב-number_param_input / number_range_input מהדשבורד
try:
    from root.dashboard import number_param_input, number_range_input  # type: ignore[import]
except Exception:
    number_param_input = None
    number_range_input = None

# הגדרות ברירת מחדל ל-numpy/pandas (נוחות + מעט יציבות)
try:
    np.set_printoptions(
        edgeitems=3,
        linewidth=140,
        floatmode="maxprec_equal",
        suppress=True,
    )
except Exception:
    pass

try:
    # pandas 2+ — copy_on_write משפר ביצועים ומונע chained assignment
    pd.options.mode.copy_on_write = True  # type: ignore[attr-defined]
except Exception:
    pass

try:
    # פחות רעש אזהרות בעת join/merge
    pd.options.mode.chained_assignment = "warn"  # type: ignore[attr-defined]
except Exception:
    pass


# =========================
# SECTION 0.2: Environment detection
# =========================
def _detect_streamlit() -> bool:
    """בדיקה האם אנחנו רצים כחלק מ-Streamlit App."""
    try:
        # streamlit >= 1.20
        return bool(getattr(st, "_is_running_with_streamlit", False))
    except Exception:
        # fallback: בדיקה גסה
        return "streamlit" in sys.argv[0].lower()


RUNNING_IN_STREAMLIT: bool = _detect_streamlit()
RUNNING_TESTS: bool = "pytest" in sys.modules
RUNNING_INTERACTIVE: bool = hasattr(sys, "ps1") or sys.flags.interactive


# =========================
# SECTION 0.3: Streamlit cache shims
# =========================
"""
העטיפות הבאות מבטיחות:
- אם אתה מריץ Streamlit בגרסה חדשה → cache_data / cache_resource.
- אם אתה מריץ סקריפט רגיל / בדיקות → העטיפות הן no-op (אבל שומרות חתימה זהה).
"""

if hasattr(st, "cache_data"):
    cache_data = st.cache_data  # type: ignore[attr-defined]
else:
    def cache_data(*_a, **_k):  # type: ignore[no-redef]
        def deco(fn):
            return fn
        return deco

if hasattr(st, "cache_resource"):
    cache_resource = st.cache_resource  # type: ignore[attr-defined]
else:
    def cache_resource(*_a, **_k):  # type: ignore[no-redef]
        def deco(fn):
            return fn
        return deco

from common.zoom_storage import (
    resolve_zoom_storage,
    build_zoom_study_name,
)

# =========================
# SECTION 0.4: Optional libs (Plotly, DuckDB, sklearn, SciPy, statsmodels, Optuna)
# =========================

# ---- Plotly (גרפים אינטראקטיביים) ----
try:
    import plotly.express as px  # type: ignore
except Exception:
    px = None  # type: ignore

try:
    import plotly.graph_objects as go  # type: ignore
except Exception:
    go = None  # type: ignore

# ---- DuckDB — אחסון מקומי מהיר ללוגים וניסויים ----
try:
    import duckdb  # type: ignore
except Exception:
    duckdb = None  # type: ignore

# ---- sklearn — clustering, PCA, RF וכו' ----
try:
    from sklearn.cluster import KMeans  # type: ignore
except Exception:
    KMeans = None  # type: ignore

try:
    from sklearn.decomposition import PCA  # type: ignore
except Exception:
    PCA = None  # type: ignore

try:
    from sklearn.ensemble import RandomForestRegressor  # type: ignore
except Exception:
    RandomForestRegressor = None  # type: ignore

# ---- SciPy — סטטיסטיקה מתקדמת (PSR/DSR, מבחנים, וכו') ----
try:
    import scipy.stats as sps  # type: ignore
except Exception:
    sps = None  # type: ignore

# ---- statsmodels — ADF, ARMA, Cointegration וכו' ----
try:
    import statsmodels.api as sm  # type: ignore
    from statsmodels.tsa.stattools import adfuller as sm_adfuller  # type: ignore
except Exception:
    sm = None  # type: ignore

    def sm_adfuller(*_a, **_k):
        raise RuntimeError("statsmodels not available")

# ---- Optuna — מנוע האופטימיזציה המרכזי ----
try:
    import optuna
    from optuna.samplers import TPESampler, CmaEsSampler
    from optuna.pruners import MedianPruner
except Exception:
    optuna = None
    # object משמש כ-placeholder — נבדוק בהמשך if X is not object
    TPESampler = CmaEsSampler = MedianPruner = object  # type: ignore


# =========================
# SECTION 0.5: Warnings hygiene
# =========================

# מפחיתים רעש אזהרות של sklearn וכד'
try:
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
except Exception:
    pass

# אזהרות deprecation כלליות (למשל Streamlit, Plotly, DuckDB)
warnings.filterwarnings("ignore", message=r".*is deprecated and will be removed.*")

# ניסיון טעינה זהיר של המנוע
try:
    import core.optimizer as opt  # type: ignore[import]
except Exception:
    opt = None

# =========================
# SECTION 0.6: Global helpers — keys, name normalization, safe_import
# =========================

# Counter גלובלי למפתחות ייחודיים ל-Streamlit
_st_key_counter = itertools.count()


def sk(prefix: str) -> str:
    """
    יצירת מפתח ייחודי ל-Streamlit.

    דוגמה:
        st.button("Run optimisation", key=sk("opt_run"))

    המפתח תמיד יהיה מהצורה:
        "<prefix>-<counter>"
    """
    return f"{prefix}-{next(_st_key_counter)}"


def _norm_name(x: str) -> str:
    """
    נירמול שם סמפלר/פרונר/אלגוריתם.

    דוגמאות:
        "CMA-ES"    -> "CMAES"
        "cmaes"     -> "CMAES"
        " tpe  "    -> "TPE"
        "MedianPruner" -> "MEDIANPRUNER"
    """
    return "".join(ch for ch in str(x).strip() if ch.isalnum()).upper()


def _is_cmaes(name: str) -> bool:
    """
    בדיקה נוחה האם השם מייצג CMA-ES (בכל פורמט כתיבה סביר).

    נשתמש בה בכל מקום שבו צריך לבחור בין TPE/CMA-ES (או לוגיקת fallback).
    """
    return _norm_name(name) == "CMAES"


def safe_import(module_name: str) -> Optional[Any]:
    """
    יבוא מודול בצורה בטוחה, מחזיר את המודול או None.

    שימושי כשנרצה להוסיף בעתיד מודולים אופציונליים (למשל:
    mlflow, wandb, prefect) בלי להפיל את הטאב אם אינם מותקנים.
    """
    try:
        return __import__(module_name)
    except Exception:
        return None


# =========================
# SECTION 0.7: Base logger (קונפיג מלא בחלק 2)
# =========================

# חשוב: לא עושים basicConfig כאן כדי לא לשבור Dashboards אחרים.
logger = logging.getLogger("OptTab")
if not logger.handlers:
    # נרשם handler בסיסי רק אם לא הוגדר כבר ע"י main/dashboard
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(_handler)
# רמת לוגיקת ברירת מחדל — תעודכן מאוחר יותר לפי OptSettings
logger.setLevel(logging.INFO)

"""
חלק 2/15 — מודולים פנימיים, OptSettings, לוגינג, עזרי גרסאות.
"""
# =========================
# Helpers
# =========================

def _get_service_and_ctx() -> Tuple[DashboardService, DashboardContext]:
    """
    יוצר DashboardService + DashboardContext בצורה מרוכזת.

    זה מונע ממך להתעסק ב-`service` גלובלי ומעלים את
    ה-"service is not defined".
    """
    service = create_dashboard_service()
    ctx = build_default_dashboard_context()
    return service, ctx

def _run_core_optimization(
    config: Mapping[str, Any],
    *,
    n_trials: int,
    study_name: str = "dashboard_opt",
    candidates: Optional[Union[pd.DataFrame, Sequence[Mapping[str, Any]]]] = None,
) -> Dict[str, Any]:
    """
    עטיפה "HF-grade" סביב run_optimization + helperים של core.optimizer.

    מחזיר dict עם:
        df, summary, ranked, pareto, tagged, best_params

    הערה:
    -----
    זהו מסלול Legacy שעושה שימוש ב-core.optimizer.run_optimization.
    למסלולים החדשים (single/batch/Zoom) מומלץ להשתמש ב:
        optimize_pair_with_telemetry / optimize_pairs_batch / api_zoom_campaign_for_pair
    אשר כולם עוברים דרך run_optuna_for_pair ו-Zoom storage.
    """

    if run_optimization is None:
        st.error("⚠️ run_optimization מ-core.optimizer לא נטען – בדוק את core/optimizer.py.")
        return {}

    # 1) הרצת האופטימיזציה
    df = run_optimization(  # type: ignore[call-arg]
        candidates=candidates,
        config=config,
        n_trials=int(n_trials),
        study_name=str(study_name),
    )

    # 2) סיכום תוצאות
    summary: Dict[str, Any] = {}
    if summarize_results is not None:
        try:
            summary = summarize_results(df)  # type: ignore[call-arg]
        except Exception:
            summary = {}

    # 3) דירוג (rank) לפי score
    if rank_trials is not None:
        try:
            ranked = rank_trials(df, by="score", ascending=False, top=None)  # type: ignore[call-arg]
        except Exception:
            ranked = df.copy()
    else:
        ranked = df.copy()

    # 4) Pareto front
    if compute_pareto_front is not None:
        try:
            pareto = compute_pareto_front(df)  # type: ignore[call-arg]
        except Exception:
            pareto = pd.DataFrame()
    else:
        pareto = pd.DataFrame()

    # 5) סימון top-k כ-is_best
    if tag_best_trials is not None:
        try:
            tagged = tag_best_trials(df, score_col="score", top_k=20)  # type: ignore[call-arg]
        except Exception:
            tagged = df.copy()
    else:
        tagged = df.copy()

    # 6) פרמטרים הכי טובים לפי Sharpe / score
    best_params_sharpe: Dict[str, Any] = {}
    best_params_score: Dict[str, Any] = {}
    if extract_best_params is not None:
        try:
            best_params_sharpe = extract_best_params(df, metric="sharpe")  # type: ignore[call-arg]
        except Exception:
            pass
        try:
            best_params_score = extract_best_params(df, metric="score")  # type: ignore[call-arg]
        except Exception:
            pass

    return {
        "df": df,
        "summary": summary,
        "ranked": ranked,
        "pareto": pareto,
        "tagged": tagged,
        "best_params": {
            "by_sharpe": best_params_sharpe,
            "by_score": best_params_score,
        },
    }

# =========================
# SECTION 1: Core paths & dirs
# =========================

DATA_DIR = PROJECT_ROOT / "data"
STUDIES_DIR = PROJECT_ROOT / "studies"
LOGS_DIR = PROJECT_ROOT / "logs"

for _p in (DATA_DIR, STUDIES_DIR, LOGS_DIR):
    try:
        _p.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


# =========================
# SECTION 1.1: Internal core modules (safe imports)
# =========================
"""
כל המודולים הפנימיים נטענים עם try/except כדי שהטאב לא יקרוס אם אחד מהם חסר.
במקום זה אנחנו שומרים stub / None ונותנים הודעת בריאות ב-System Health.
"""

# Backtester
try:
    from core.optimization_backtester import Backtester  # type: ignore
except Exception as _e_bt:
    Backtester = None  # type: ignore
    logger.warning("Backtester unavailable (core.optimization_backtester): %s", _e_bt)


# Metrics: normalize + weighted score
try:
    from core.metrics import normalize_metrics, compute_weighted_score  # type: ignore
except Exception as _e_metrics:
    normalize_metrics = None  # type: ignore
    compute_weighted_score = None  # type: ignore
    logger.warning("metrics module unavailable (core.metrics): %s", _e_metrics)

# Param distributions (Optuna ranges)
try:
    from core.distributions import get_param_distributions  # type: ignore
except Exception as _e_dist:
    get_param_distributions = None  # type: ignore
    logger.warning("distributions module unavailable (core.distributions): %s", _e_dist)

# Risk parity helper (optional) — normalise param vectors
try:
    from core.risk_parity import apply_risk_parity_to_params  # type: ignore
except Exception as _e_rp:
    apply_risk_parity_to_params = None  # type: ignore
    logger.info("risk_parity helper unavailable (core.risk_parity): %s", _e_rp)

# Analysis helpers: SHAP, PCA, Clusters (optional)
try:
    from core.analysis_helpers import (  # type: ignore
        compute_shap_importance_df,
        compute_pca_transform,
        compute_clusters,
    )
except Exception as _e_ah:
    compute_shap_importance_df = None  # type: ignore
    compute_pca_transform = None  # type: ignore
    compute_clusters = None  # type: ignore
    logger.info("analysis_helpers unavailable (core.analysis_helpers): %s", _e_ah)

# Feature selection (optional)
try:
    from core.feature_selection import select_features  # type: ignore
except Exception as _e_fs:
    select_features = None  # type: ignore
    logger.info("feature_selection unavailable (core.feature_selection): %s", _e_fs)

# Legacy CORE_PARAM_SPECS (אם עדיין משתמשים בו איפשהו)
try:
    from core.params import PARAM_SPECS as CORE_PARAM_SPECS  # type: ignore
except Exception:
    CORE_PARAM_SPECS = None  # type: ignore

# Param ranges & RangeManager החדש
try:
    from core.ranges import DEFAULT_PARAM_RANGES, RangeManager  # type: ignore
except Exception as _e_rng:
    DEFAULT_PARAM_RANGES = {}  # type: ignore
    RangeManager = None  # type: ignore
    logger.info("ranges module unavailable (core.ranges): %s", _e_rng)


# Optimizer helpers (כולל run_optimization)
try:
    from core.optimizer import (  # type: ignore
        run_optimization,
        summarize_results,
        extract_best_params,
        rank_trials,
        compute_pareto_front,
        tag_best_trials,
    )
except Exception:
    run_optimization = None
    summarize_results = None
    extract_best_params = None
    rank_trials = None
    compute_pareto_front = None
    tag_best_trials = None

# Meta-optimizer (clusters-level ensemble scoring)
try:
    from core.meta_optimizer import meta_optimize  # type: ignore
except Exception as _e_meta:
    meta_optimize = None  # type: ignore
    logger.info("meta_optimizer unavailable (core.meta_optimizer): %s", _e_meta)

# ML analysis integration (טאב ML שיודע לעבוד על opt_df)
try:
    from core.ml_analysis import render_ml_for_optuna_session  # type: ignore
except Exception as _e_ml:
    logger.info("ml_analysis unavailable (core.ml_analysis): %s", _e_ml)

    def render_ml_for_optuna_session(*_a, **_k):
        st.caption("ML analysis module (core.ml_analysis) not available.")

# SqlStore – קריאה מה-DuckDB (trials + prices)
try:
    from core.sql_store import SqlStore  # type: ignore
except Exception as _e_sql:
    SqlStore = None  # type: ignore
    logger.info("SqlStore unavailable (core.sql_store): %s", _e_sql)


def persist_opt_run_to_sqlstore(
    pair: str,
    df: pd.DataFrame,
    manifest: Dict[str, Any],
) -> None:
    """
    TODO(Omri): Wire optimisation persistence into SqlStore/Parquet.

    Intended usage:
    ----------------
    - Persist high-level optimisation runs (pair, study_id, best params/score, env, config)
      into the central SqlStore so research/backtest/paper/live all see the same history.
    - Current implementation is a no-op to keep call sites stable until the
      write-API in core.sql_store is finalised.
    """
    # Intentionally a no-op for now.
    return


# =========================
# SECTION 1.2: Settings via Pydantic
# =========================
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class OptSettings(BaseSettings):
    """
    OptSettings — קונפיגורציה מרכזית לטאב האופטימיזציה.

    נטען מתוך:
    - משתני סביבה (אוטומטי, עם prefix OPT_)
    - opt_config.yml / opt_config.json (אם קיים)
    - ערכי ברירת מחדל (fallback)

    שדות עיקריים:
    - env               — פרופיל ריצה ("local", "dev", "prod"...)
    - log_level         — DEBUG/INFO/WARNING/ERROR/CRITICAL
    - data_dir          — תיקיית דאטה
    - studies_dir       — תיקיית ניסויי Optuna / DuckDB
    - slack_webhook     — להודעות תוצאה/שגיאה (אופציונלי)
    - telegram_token    — טלגרם (אופציונלי)
    - telegram_chat_id  — טלגרם (אופציונלי)
    - duck_threads      — מספר ת׳רדים ל-DuckDB
    - duck_memory_limit_mb — הגבלת זיכרון ל-DuckDB (0 = default)
    """

    env: str = "local"
    log_level: str = "INFO"

    data_dir: Path = DATA_DIR
    studies_dir: Path = STUDIES_DIR

    slack_webhook: Optional[str] = None
    telegram_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None

    # DuckDB / performance knobs
    duck_threads: int = int(os.environ.get("OPT_DUCK_THREADS", "4"))
    duck_memory_limit_mb: int = int(os.environ.get("OPT_DUCK_MEM_MB", "0"))  # 0 = DuckDB default

    # Optional feature flags for heavy analysis
    enable_heavy_panels: bool = True
    enable_wandb: bool = False
    enable_mlflow: bool = False

    model_config = SettingsConfigDict(
        env_prefix="OPT_",      # למשל: OPT_LOG_LEVEL, OPT_DUCK_THREADS
        case_sensitive=False,
    )

    @field_validator("log_level")
    @classmethod
    def valid_level(cls, v: str) -> str:
        lvl = (v or "INFO").upper()
        if lvl not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError("Invalid log_level")
        return lvl

    @field_validator("data_dir", "studies_dir")
    @classmethod
    def ensure_dir(cls, v: Path) -> Path:
        try:
            v.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        return v


def _load_settings_from_file(cfg_path: Path) -> Dict[str, Any]:
    """
    טעינת קונפיג מקובץ (YAML/JSON). מטרה: חיצוניות קונפיגורציה.

    סדר העדיפויות:
    1. אם הקובץ לא קיים → {}.
    2. אם סיומת .yml/.yaml → שימוש ב-yaml.safe_load (אם מותקן).
    3. אחרת → JSON רגיל.
    """
    if not cfg_path.exists():
        return {}
    try:
        if cfg_path.suffix.lower() in {".yml", ".yaml"}:
            try:
                import yaml  # type: ignore
                data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            except Exception:
                data = {}
        else:
            data = json.loads(cfg_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {}
        return data
    except Exception as e:
        logger.warning("Failed loading settings from %s: %s", cfg_path, e)
        return {}


def load_settings(cfg_path: Optional[Path] = None) -> OptSettings:
    """
    טוען OptSettings לפי הסדר:
    - קובץ opt_config.yml / opt_config.json (אם cfg_path=None).
    - משתני סביבה עם prefix OPT_.
    - ערכי ברירת מחדל שהוגדרו במחלקה.
    """
    if cfg_path is None:
        # ראשון בעדיפות: YAML, אם קיים. שנית: JSON.
        yaml_path = PROJECT_ROOT / "opt_config.yml"
        json_path = PROJECT_ROOT / "opt_config.json"
        if yaml_path.exists():
            cfg_path = yaml_path
        elif json_path.exists():
            cfg_path = json_path

    file_data: Dict[str, Any] = {}
    if cfg_path is not None:
        file_data = _load_settings_from_file(cfg_path)

    # OptSettings תדע לשלב file_data עם משתני סביבה וערכי ברירת מחדל
    try:
        settings = OptSettings(**file_data)
    except Exception as e:
        logger.warning("OptSettings init failed with file data (%s), using defaults env-only: %s", cfg_path, e)
        settings = OptSettings()

    return settings


SETTINGS = load_settings()


# =========================
# SECTION 1.3: Logging configuration (סופי)
# =========================
"""
בשלב זה יש לנו SETTINGS.log_level → מעדכנים logger.
אנחנו לא קוראים ל-basicConfig פעם נוספת כדי לא להרוס לוגרים אחרים.
"""

def _configure_logger_from_settings() -> None:
    lvl_name = SETTINGS.log_level.upper()
    try:
        lvl = getattr(logging, lvl_name, logging.INFO)
    except Exception:
        lvl = logging.INFO
    logger.setLevel(lvl)
    try:
        logger.info("OptSettings loaded: env=%s, log_level=%s, data_dir=%s, studies_dir=%s",
                    SETTINGS.env, SETTINGS.log_level, SETTINGS.data_dir, SETTINGS.studies_dir)
    except Exception:
        pass


_configure_logger_from_settings()


# =========================
# Optimization metrics → session_state (Home / Tabs אחרים)
# =========================

def _safe_float_or_none_opt(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if not (v == v):  # NaN
            return None
        return v
    except Exception:
        return None


def push_optimization_metrics_to_ctx(
    *,
    best_params: Dict[str, Any],
    best_score: Any,
    n_trials: int,
    objective_name: str = "sharpe",
    pair_id: Optional[str] = None,
    profile: Optional[str] = None,
    mode: str = "single",  # "single" / "batch"
    ctx_key: str = "optimization_metrics",
) -> None:
    """
    שומר מדדי אופטימיזציה רלוונטיים ל-Home / Tabs אחרים ב-session_state[ctx_key].

    דוגמה למה נשמר:
        {
          "objective": "sharpe",
          "best_score": 1.42,
          "n_trials": 200,
          "pair_id": "XLY-XLP",
          "profile": "defensive",
          "mode": "single",
          "best_params": {...}
        }

    כך ה-Home / Risk / Agents יכולים לדעת:
    - מה הייתה ריצת האופטימיזציה האחרונה,
    - מה הסקור הכי טוב,
    - על איזה זוג,
    - ובאיזה פרופיל.
    """
    score = _safe_float_or_none_opt(best_score)
    metrics: Dict[str, Any] = {
        "objective": str(objective_name),
        "best_score": score,
       