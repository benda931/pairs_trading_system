# -*- coding: utf-8 -*-
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

def _ensure_prices_for_pair_before_opt(
    sym1: str,
    sym2: str,
    *,
    start_date: Any,
    end_date: Any,
) -> None:
    """
    וידוא מחירים ב-SqlStore לפני אופטימיזציה לזוג נתון (HF-grade, best-effort בלבד).

    זרימה:
    -------
    1. קורא AppContext.global → settings.
    2. בונה SqlStore.from_settings(env=settings.env).
    3. בונה IBDataIngestor.from_settings(..., connect=False).
    4. מריץ ingest_pair באופן אינקרמנטלי ל־[start_date, end_date].
    5. במקרה של כשל — מדווח ללוג וממשיך, בלי להפיל את הטאב.

    הערות:
    -------
    - לא זורק Exceptions החוצה – רק כותב ללוג.
    - מניח שקובץ core/ib_data_ingestor.py קיים כמו שהגדרת.
    """
    if SqlStore is None:
        logger.info(
            "OptTab.ensure_prices_for_pair_before_opt: SqlStore is None — skipping for %s-%s.",
            sym1,
            sym2,
        )
        return

    # 1) המרת תאריכים
    try:
        if start_date is None or end_date is None:
            logger.info(
                "OptTab.ensure_prices_for_pair_before_opt: missing start/end dates for %s-%s, skipping.",
                sym1,
                sym2,
            )
            return
        start_d = pd.to_datetime(start_date).date()
        end_d = pd.to_datetime(end_date).date()
    except Exception as e:
        logger.info(
            "OptTab.ensure_prices_for_pair_before_opt: failed to parse dates for %s-%s (%r,%r): %s",
            sym1,
            sym2,
            start_date,
            end_date,
            e,
        )
        return

    if start_d > end_d:
        start_d, end_d = end_d, start_d

    # 2) יבוא עצלן של AppContext + IBDataIngestor
    try:
        from core.app_context import AppContext  # type: ignore
        from core.ib_data_ingestor import IBDataIngestor  # type: ignore
    except Exception as e:
        logger.info(
            "OptTab.ensure_prices_for_pair_before_opt: failed to import AppContext/IBDataIngestor (%s), skipping.",
            e,
        )
        return

    # 3) settings + env
    try:
        app_ctx = AppContext.get_global()
        settings = app_ctx.settings
    except Exception as e:
        logger.info(
            "OptTab.ensure_prices_for_pair_before_opt: AppContext.get_global failed (%s), skipping.",
            e,
        )
        return

    try:
        env_str = str(getattr(settings, "env", getattr(SETTINGS, "env", "dev")) or "dev")
    except Exception:
        env_str = "dev"

    # 4) יצירת SqlStore + IBDataIngestor
    try:
        store = SqlStore.from_settings(settings, env=env_str)
    except Exception as e:
        logger.info(
            "OptTab.ensure_prices_for_pair_before_opt: SqlStore.from_settings failed for env=%s (%s), skipping.",
            env_str,
            e,
        )
        return

    try:
        ingestor = IBDataIngestor.from_settings(
            settings=settings,
            store=store,
            env=env_str,
            connect=False,  # ensure_connected יקרה בתוך ingest_symbol
        )
    except Exception as e:
        logger.info(
            "OptTab.ensure_prices_for_pair_before_opt: IBDataIngestor.from_settings failed (%s), skipping.",
            e,
        )
        return

    pair_label = f"{sym1}-{sym2}"

    # 5) קריאה אינקרמנטלית ל-IB → SqlStore.prices
    try:
        ingestor.ingest_pair(
            pair_label,
            start_date=start_d,
            end_date=end_d,
            bar_size=None,       # ישתמש ב־default מה-settings
            what_to_show=None,   # כנ"ל
            incremental=True,    # חשוב: משתמש ב-load_price_history בתוך ingestor
        )
        logger.info(
            "OptTab.ensure_prices_for_pair_before_opt: ensured SqlStore prices for %s in [%s, %s] (env=%s).",
            pair_label,
            start_d,
            end_d,
            env_str,
        )
    except Exception as e:
        logger.info(
            "OptTab.ensure_prices_for_pair_before_opt: ingest_pair failed for %s (%s) — continuing without IB ingest.",
            pair_label,
            e,
        )

def persist_opt_run_to_sqlstore(
    pair: str,
    df: pd.DataFrame,
    manifest: Dict[str, Any],
) -> None:
    """
    Persist a completed optimisation run into SqlStore as "live params" for a pair.

    Responsibilities:
    -----------------
    1. Robustly select a single "best" parameter row from df:
       - Primary: highest Score
       - Secondary (if exists): DSR, wf_robust_penalty, Sharpe, lowest Drawdown
    2. Derive env / profile / run_id / study_id / opt_run_cfg from manifest + session.
    3. Enforce LIVE/PROD safety:
       - In env ∈ {live, prod} → write only if `opt_live_update_params=True`.
    4. Call SqlStore.save_opt_best_params with a clean, JSON-safe payload.

    Notes:
    ------
    - If SqlStore is unavailable or read_only=True → returns quietly.
    - If df is empty or has no usable Score → still stores params from first row,
      but score=None (so we do not "fake" performance).
    """

    # 0) SqlStore not available at all
    if SqlStore is None:
        logger.debug("persist_opt_run_to_sqlstore: SqlStore is None — skipping for pair=%s.", pair)
        return

    # 1) df guard
    if df is None or df.empty:
        logger.debug("persist_opt_run_to_sqlstore: empty df for pair=%s — skipping.", pair)
        return

    df_loc = df.copy()

    # 2) Best row selection (HF-grade but deterministic)
    best_params: Dict[str, Any]
    best_score: Optional[float]

    if "Score" not in df_loc.columns:
        # No Score at all – take first row, but do NOT invent a score.
        row = df_loc.iloc[0]
        best_params = _extract_params_from_opt_row(row)
        best_score = None
        logger.debug(
            "persist_opt_run_to_sqlstore: no Score column for pair=%s, using first row only.",
            pair,
        )
    else:
        score_series = pd.to_numeric(df_loc["Score"], errors="coerce")
        df_loc = df_loc.loc[score_series.notna()].copy()

        if df_loc.empty:
            # Score exists but all NaN → fall back to first row of original df
            row = df.iloc[0]
            best_params = _extract_params_from_opt_row(row)
            best_score = None
            logger.debug(
                "persist_opt_run_to_sqlstore: Score column all-NaN for pair=%s, "
                "falling back to first row.",
                pair,
            )
        else:
            # Build a ranking using multiple signals if they exist
            df_rank = df_loc.copy()
            df_rank["_score"] = pd.to_numeric(df_rank["Score"], errors="coerce")

            if "DSR" in df_rank.columns:
                df_rank["_dsr"] = pd.to_numeric(df_rank["DSR"], errors="coerce")
            else:
                df_rank["_dsr"] = np.nan

            if "wf_robust_penalty" in df_rank.columns:
                df_rank["_wf"] = pd.to_numeric(df_rank["wf_robust_penalty"], errors="coerce")
            else:
                df_rank["_wf"] = np.nan

            if "Sharpe" in df_rank.columns:
                df_rank["_sharpe"] = pd.to_numeric(df_rank["Sharpe"], errors="coerce")
            else:
                df_rank["_sharpe"] = np.nan

            if "Drawdown" in df_rank.columns:
                dd_tmp = pd.to_numeric(df_rank["Drawdown"], errors="coerce")
                df_rank["_dd"] = dd_tmp.abs()  # lower is better
            else:
                df_rank["_dd"] = np.nan

            # Top 20% by Score as a soft pre-filter
            try:
                q80 = df_rank["_score"].quantile(0.80)
                df_top = df_rank[df_rank["_score"] >= q80].copy()
                if df_top.empty:
                    df_top = df_rank.sort_values("_score", ascending=False).head(
                        max(5, int(len(df_rank) * 0.2))
                    )
            except Exception:
                df_top = df_rank.sort_values("_score", ascending=False).head(
                    max(5, int(len(df_rank) * 0.2))
                )

            # Extra robustness filters (if available)
            if df_top["_dsr"].notna().any():
                mask_dsr = df_top["_dsr"] >= 1.0
                if mask_dsr.any():
                    df_top = df_top.loc[mask_dsr].copy()

            if df_top["_wf"].notna().any():
                mask_wf = df_top["_wf"] >= 0.7
                if mask_wf.any():
                    df_top = df_top.loc[mask_wf].copy()

            if df_top.empty:
                df_top = df_rank.sort_values("_score", ascending=False).head(
                    max(5, int(len(df_rank) * 0.2))
                )

            # Final ordering: Score ↓, DSR ↓, wf_robust_penalty ↓, Sharpe ↓, Drawdown ↑ (less)
            df_top = df_top.sort_values(
                by=["_score", "_dsr", "_wf", "_sharpe", "_dd"],
                ascending=[False, False, False, False, True],
            )

            row = df_top.iloc[0]
            best_params = _extract_params_from_opt_row(row)
            try:
                best_score = float(pd.to_numeric(df_top["_score"], errors="coerce").max())
            except Exception:
                best_score = None

    # 3) env / profile / identifiers from manifest + session_state
    try:
        env_str = str(manifest.get("env") or getattr(SETTINGS, "env", "local") or "local").lower()
    except Exception:
        env_str = "local"

    try:
        profile = str(manifest.get("profile") or st.session_state.get("opt_profile", "default"))
    except Exception:
        profile = "default"

    # Study ID (from manifest, if any)
    study_id = manifest.get("study_id")
    try:
        study_id_int: Optional[int] = int(study_id) if study_id is not None else None
    except Exception:
        study_id_int = None

    # Run ID (manifest → session_state)
    try:
        run_id_val = (
            manifest.get("run_id")
            or st.session_state.get("opt_run_id")
            or st.session_state.get("run_id")
        )
        run_id: Optional[str] = str(run_id_val) if run_id_val is not None else None
    except Exception:
        run_id = None

    # opt_run_cfg snapshot (for reproducibility)
    try:
        opt_run_cfg = st.session_state.get("opt_run_cfg", {}) or {}
    except Exception:
        opt_run_cfg = {}

    # 4) LIVE/PROD safety – do not update registry/SqlStore unless explicitly allowed
    try:
        allow_live_update = bool(st.session_state.get("opt_live_update_params", False))
    except Exception:
        allow_live_update = False

    if env_str in {"live", "prod"} and not allow_live_update:
        logger.info(
            "persist_opt_run_to_sqlstore: skipping SqlStore write for pair=%s in env=%s "
            "(opt_live_update_params=False).",
            pair,
            env_str,
        )
        return

    # 5) JSON-safe blobs (manifest & cfg) — for long-term auditing
    try:
        manifest_json = json.dumps(make_json_safe(manifest or {}), ensure_ascii=False)
    except Exception:
        manifest_json = json.dumps(manifest or {}, ensure_ascii=False)

    try:
        opt_run_cfg_json = json.dumps(make_json_safe(opt_run_cfg or {}), ensure_ascii=False)
    except Exception:
        opt_run_cfg_json = json.dumps(opt_run_cfg or {}, ensure_ascii=False)

    # 6) Open SqlStore in write mode
    try:
        env_settings: Dict[str, Any] = {}
        if "SQL_STORE_URL" in os.environ:
            env_settings["engine_url"] = os.environ["SQL_STORE_URL"]

        store = SqlStore.from_settings(env_settings, read_only=False)
    except Exception as e:
        logger.debug(
            "persist_opt_run_to_sqlstore: SqlStore.from_settings failed for pair=%s: %s",
            pair,
            e,
        )
        return

    if getattr(store, "read_only", False):
        logger.info(
            "persist_opt_run_to_sqlstore: SqlStore is read_only=True, "
            "skipping write for pair=%s (env=%s).",
            pair,
            env_str,
        )
        return

    # 7) Final write into SqlStore
    try:
        store.save_opt_best_params(
            pair=str(pair),
            env=env_str,
            profile=profile,
            params=best_params,
            score=_safe_float_or_none_opt(best_score),
            source="optimization_tab",
            study_id=study_id_int,
            run_id=run_id,
            manifest_json=manifest_json,
            opt_run_cfg_json=opt_run_cfg_json,
        )
        logger.info(
            "persist_opt_run_to_sqlstore: saved best params for pair=%s env=%s profile=%s "
            "(study_id=%s, run_id=%s, score=%s).",
            pair,
            env_str,
            profile,
            study_id_int,
            run_id,
            f"{best_score:.4f}" if isinstance(best_score, (int, float)) else "n/a",
        )
    except Exception as e:
        logger.warning(
            "persist_opt_run_to_sqlstore: save_opt_best_params failed for pair=%s: %s",
            pair,
            e,
        )

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
        "n_trials": int(n_trials),
        "pair_id": pair_id,
        "profile": profile,
        "mode": mode,
        "best_params": best_params or {},
    }

    # Timestamp לריצה האחרונה (UTC + Z)
    try:
        dt = datetime.now(timezone.utc)
        metrics["last_run_utc"] = dt.isoformat(timespec="seconds").replace("+00:00", "Z")
    except Exception:
        metrics["last_run_utc"] = None


    st.session_state[ctx_key] = metrics
    st.session_state["last_optimization_time"] = metrics["last_run_utc"]

# =========================
# Helper 1 — Header לטאב
# =========================

def render_optimization_header(ctx: DashboardContext,
                               service: DashboardService) -> None:
    st.markdown("### ⚙️ Optimization – אופטימיזציית פרמטרים")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"**Environment:** `{ctx.env}`")
        st.markdown(f"**Profile:** `{ctx.profile}`")

    with col2:
        st.markdown(f"**Date Range:** `{ctx.start_date} → {ctx.end_date}`")

    with col3:
        # סטטיסטיקת universe בסיסית
        try:
            uni_stats = service.get_universe_stats_for_opt(ctx)
            st.markdown(f"**Pairs in universe:** {uni_stats.n_pairs}")
            st.markdown(f"**Symbols:** {uni_stats.n_symbols}")
        except Exception:
            st.markdown("**Pairs in universe:** N/A")

    with col4:
        # סיכום ריצה אחרונה אם קיימת
        try:
            last_run = service.get_last_opt_run_summary(ctx)
        except Exception:
            last_run = None

        if last_run is not None:
            st.markdown("**Last run score:**")
            st.metric("Score",
                      f"{last_run.score:.3f}",
                      delta=f"{last_run.sharpe:.2f} Sharpe")
        else:
            st.info("No optimization runs yet in this context.")

def _render_optimization_snapshot_panel() -> None:
    """
    פאנל Snapshot קטן בראש הטאב:

    מציג:
    - Objective (Sharpe / Sortino / ...)
    - Best score
    - #Trials
    - Pair / Profile / Mode (single/batch)
    - זמן ריצה אחרון
    """
    metrics = st.session_state.get("optimization_metrics", {}) or {}
    if not metrics:
        st.caption("No optimisation snapshot yet – run an optimisation to populate.")
        return

    objective = metrics.get("objective", "sharpe")
    best_score = metrics.get("best_score")
    n_trials = metrics.get("n_trials")
    pair_id = metrics.get("pair_id") or "universe / batch"
    profile = metrics.get("profile") or st.session_state.get("opt_profile", "default")
    mode = metrics.get("mode", "single")
    last_run = metrics.get("last_run_utc")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Objective", objective)
        st.caption(f"Mode: `{mode}`")

    with c2:
        st.metric(
            "Best score",
            f"{best_score:.3f}" if isinstance(best_score, (int, float)) else "N/A",
        )
        st.caption(f"Pair: `{pair_id}`")

    with c3:
        st.metric(
            "#Trials",
            f"{int(n_trials)}" if isinstance(n_trials, (int, float)) else "N/A",
        )
        if last_run:
            st.caption(f"Last run: `{last_run}`")

    with c4:
        st.metric("Profile", profile)
        st.caption("Source: optimisation_tab")

# =========================
# Helper 2 — Preset Bar
# =========================

def render_opt_presets(prefix: str = "opt") -> dict:
    st.markdown("#### Presets")

    col1, col2, col3, col4 = st.columns(4)

    preset_key = f"{prefix}_preset"
    if preset_key not in st.session_state:
        st.session_state[preset_key] = "custom"

    def _set_preset(name: str) -> None:
        st.session_state[preset_key] = name

    with col1:
        if st.button("💨 Smoke", key=f"{prefix}_preset_smoke"):
            _set_preset("smoke")

    with col2:
        if st.button("🔬 Deep Research", key=f"{prefix}_preset_deep"):
            _set_preset("deep")

    with col3:
        if st.button("🌪 Tail Focus", key=f"{prefix}_preset_tail"):
            _set_preset("tail")

    with col4:
        if st.button("⚡ Fast Debug", key=f"{prefix}_preset_fast"):
            _set_preset("fast")

    return {"preset": st.session_state[preset_key]}

# =========================
# Helper 3 — מספרים במקום סליידר לפי ParamSpec
# =========================

def render_param_control(prefix: str, name: str) -> float:
    """
    קונטרול לפרמטר אחד:
    - מבוסס על PARAM_SPECS
    - בלי slider, רק number_input
    """
    spec = PARAM_SPECS[name]
    label = f"{spec.name} [{spec.lo} – {spec.hi}]"
    tags = ", ".join(spec.tags or [])
    help_text = f"Default: {spec.default} | tags: {tags}"

    value = st.number_input(
        label,
        min_value=float(spec.lo),
        max_value=float(spec.hi),
        value=float(spec.default),
        step=float(spec.step or 0.1),
        key=f"{prefix}_{name}",
        help=help_text,
    )
    return float(value)

# =========================
# Helper 4 — היסטוריית ריצות
# =========================

def render_opt_run_history(service: DashboardService,
                           ctx: DashboardContext,
                           limit: int = 15) -> None:
    st.markdown("#### 🕒 Optimization Run History")

    try:
        runs_df = service.fetch_opt_runs(ctx, limit=limit)
    except Exception:
        runs_df = None

    if runs_df is None or runs_df.empty:
        st.info("אין עדיין ריצות אופטימיזציה בהקשר הזה.")
        return

    st.dataframe(runs_df, width = "stretch")
    st.caption("טיפ: בהמשך אפשר להוסיף כפתור 'Load' לכל run ולשחזר קונפיג.")

# =========================
# SECTION 1.4: Version helper (גרסת ספריות)
# =========================

def _safe_version(name: str) -> str:
    """
    מחזיר __version__ של מודול, או "missing" / "n/a" אם לא זמין.
    שימושי ל-manifest, Telemetry, ודוחות.
    """
    try:
        m = __import__(name)
        return getattr(m, "__version__", "n/a")
    except Exception:
        return "missing"

"""
חלק 3/15 — DuckDB storage & helpers (Hedge-Fund Grade)
======================================================

מה כלול כאן:

1. בחירת נתיב DuckDB חכמה:
   - OPT_CACHE_PATH (ENV) > LOCALAPPDATA > PROJECT_ROOT/cache.duckdb

2. פתיחת חיבור עם:
   - PRAGMA threads / memory_limit לפי SETTINGS
   - retry על קובץ נעול
   - fallback ל-:memory: עם אזהרה

3. get_duck / get_ro_duck + Proxy DUCK:
   - DUCK.sql("SELECT ...") וכו' נוח לשימוש

4. סכימה מלאה:
   - טבלת studies (study-level metadata + reproducibility)
   - טבלת trials  (trial-level parameters & performance)
   - טבלת artifacts (blobs עבור pareto/report/manifest)

5. אינדקסים:
   - studies(pair, created_at), studies(study_id)
   - trials(study_id, trial_no), trials(pair, score)

6. פונקציות API:
   - save_trials_to_duck(...)           → study_id
   - list_pairs_in_db()                 → רשימת זוגות שנשמרו
   - list_studies_for_pair(pair)        → מטא-דאטה על studies
   - load_trials_from_duck(study_id)    → DataFrame שטוח (params + perf + Score)
   - save_artifact_to_duck(study_id, kind, bytes)
   - load_artifacts_for_study(study_id)
   - get_latest_study_id_for_pair(pair)
   - delete_study_from_duck(study_id)   → ניקוי studies+trials+artifacts

כל ה-CRUD ממומש "best-effort" ולא מפיל את הטאב.
"""

# =========================
# SECTION 2: DuckDB path & connection
# =========================

def _default_db_path() -> Path:
    """
    בוחר נתיב ברירת מחדל ל-DuckDB *לצורכי Optuna בלבד*.

    סדר עדיפות:
    1. ENV: OPT_CACHE_PATH
    2. Windows LOCALAPPDATA\\pairs_trading_system\\optuna_cache.duckdb
    3. PROJECT_ROOT / "optuna_cache.duckdb"

    חשוב:
    -----
    SqlStore משתמש בקובץ cache.duckdb (דאטה/מחירים).
    כאן אנחנו משתמשים בקובץ נפרד לאופטימיזציה כדי
    להימנע מקונפליקטים בקונפיגורציית DuckDB.
    """
    # 1) ENV override
    env_path = os.environ.get("OPT_CACHE_PATH")
    if env_path:
        p = Path(env_path)
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        return p

    # 2) LOCALAPPDATA (בעיקר ב-Windows)
    local = os.environ.get("LOCALAPPDATA")
    if local:
        p = Path(local) / "pairs_trading_system" / "optuna_cache.duckdb"
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        return p

    # 3) fallback — שורש הפרויקט
    p = PROJECT_ROOT / "optuna_cache.duckdb"
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return p



DB_PATH: Path = _default_db_path()
logger.info("OptTab using DuckDB path: %s", DB_PATH)


def _open_duck(db_path: Path, retries: int = 3, delay: float = 0.5):
    """
    פותח חיבור ל-DuckDB עם מעט retries:

    - אם DuckDB לא מותקן → RuntimeError.
    - אם קובץ נעול/בעייתי → retry עד retries פעמים עם delay.
    - אם עדיין נכשל → fallback ל-:memory: + אזהרה.
    """
    if duckdb is None:
        raise RuntimeError("DuckDB not installed. Run: pip install duckdb")

    last_err: Optional[Exception] = None
    for _ in range(int(retries)):
        try:
            conn = duckdb.connect(str(db_path))
            # טיונינג בסיסי
            try:
                threads = int(getattr(SETTINGS, "duck_threads", 4))
                conn.execute(f"PRAGMA threads={threads}")
                mem_mb = int(getattr(SETTINGS, "duck_memory_limit_mb", 0))
                if mem_mb > 0:
                    conn.execute(f"PRAGMA memory_limit='{mem_mb}MB'")
            except Exception:
                pass
            atexit.register(conn.close)
            return conn
        except Exception as e:
            last_err = e
            time.sleep(delay)

    # fallback: in-memory
    try:
        mem = duckdb.connect(":memory:")
        try:
            threads = int(getattr(SETTINGS, "duck_threads", 4))
            mem.execute(f"PRAGMA threads={threads}")
        except Exception:
            pass
        atexit.register(mem.close)
        logger.warning("Using in-memory DuckDB fallback due to: %s", last_err)
        return mem
    except Exception:
        if last_err:
            raise last_err
        raise


@cache_resource(show_spinner=False)
def get_duck():
    """חיבור כתיבה/קריאה ל-DuckDB (cached resource)."""
    return _open_duck(DB_PATH)


def get_ro_duck():
    """
    חיבור "read-only" — אם אפשר.
    אם קובץ לא נגיש → fallback ל-:memory: עם אזהרה.
    """
    if duckdb is None:
        raise RuntimeError("DuckDB not installed.")
    try:
        conn = duckdb.connect(str(DB_PATH), read_only=True)
        try:
            threads = int(getattr(SETTINGS, "duck_threads", 4))
            conn.execute(f"PRAGMA threads={threads}")
        except Exception:
            pass
        atexit.register(conn.close)
        return conn
    except Exception:
        mem = duckdb.connect(":memory:")
        try:
            threads = int(getattr(SETTINGS, "duck_threads", 4))
            mem.execute(f"PRAGMA threads={threads}")
        except Exception:
            pass
        atexit.register(mem.close)
        logger.warning("Using in-memory DuckDB read-only fallback.")
        return mem


class _DuckProxy:
    """
    Proxy קטן שנותן תחושה של חיבור DuckDB גלובלי:

        DUCK.sql("SELECT ...")

    מאחורי הקלעים משתמש ב-get_duck() כך שהחיבור מנוהל דרך cache_resource.
    """

    _conn = None

    def _ensure(self):
        if self._conn is None:
            self._conn = get_duck()

    def __getattr__(self, name: str):
        self._ensure()
        return getattr(self._conn, name)


DUCK = _DuckProxy()


# =========================
# SECTION 2.1: Schema & indexes
# =========================

def _ensure_duck_schema() -> None:
    """
    Ensure DuckDB schema for optimisation + telemetry is present and up-to-date.

    Design goals
    ------------
    1) Idempotent: safe to call repeatedly.
    2) Forward compatible: uses ADD COLUMN IF NOT EXISTS for older DBs.
    3) Transactional: creates/patches schema in a single BEGIN/COMMIT when possible.
    4) Explicit: defines intended columns and indexes for fund-grade reproducibility.

    Tables
    ------
    schema_meta:
        - Single-row metadata about schema version and last upgrade timestamp.

    studies:
        - One row per study/run (pair-level experiment metadata).

    trials:
        - One row per trial (params/perf JSON + scores + timings + state).

    artifacts:
        - Binary/JSON/CSV payloads tied to a study (reports, pareto sets, manifests).

    param_ranges:
        - Learned per-pair parameter ranges (lo/hi/step) saved after "learn ranges".

    Notes
    -----
    - DuckDB supports constraints, but in practice many users rely on best-effort
      constraints; we keep uniqueness via indexes and app-level discipline.
    - If get_duck() falls back to :memory:, schema will be created there too, but
      it won't persist. Fix connection configuration issues separately if that happens.
    """
    if duckdb is None:
        # DuckDB is optional, do not crash the app/tab/CLI on missing dependency.
        logger.warning("DuckDB not available; schema init skipped.")
        return

    try:
        conn = get_duck()
    except Exception as e:
        logger.warning("DuckDB schema init failed: get_duck() not available (%s)", e)
        return

    # Keep schema version here (single source of truth for this module)
    SCHEMA_VERSION = "1.2.0"

    try:
        conn.execute("BEGIN")

        # -----------------------------
        # 0) Schema meta (versioning)
        # -----------------------------
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_meta (
                key         VARCHAR PRIMARY KEY,
                value       VARCHAR,
                created_at  TIMESTAMP DEFAULT now(),
                updated_at  TIMESTAMP DEFAULT now()
            );
            """
        )

        # Upsert-like behavior
        # (DuckDB supports INSERT OR REPLACE)
        conn.execute(
            """
            INSERT OR REPLACE INTO schema_meta (key, value, created_at, updated_at)
            VALUES ('schema_version', ?, coalesce((SELECT created_at FROM schema_meta WHERE key='schema_version'), now()), now());
            """,
            [SCHEMA_VERSION],
        )

        # -----------------------------
        # 1) studies
        # -----------------------------
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS studies (
                study_id         BIGINT,
                pair             VARCHAR,
                created_at       TIMESTAMP DEFAULT now(),

                sampler          VARCHAR,
                pruner           VARCHAR,
                n_trials         INTEGER,
                timeout_sec      INTEGER,

                weights_json     VARCHAR,
                profile_json     VARCHAR,
                notes            VARCHAR,

                direction        VARCHAR,
                mode             VARCHAR,

                code_sha         VARCHAR,
                paramspace_hash  VARCHAR,
                sampler_params   VARCHAR,
                storage          VARCHAR,

                seed             INTEGER,
                started_at       TIMESTAMP,
                finished_at      TIMESTAMP,
                duration_sec     DOUBLE,

                optuna_version   VARCHAR,
                app_version      VARCHAR
            );
            """
        )

        # Ensure older DBs get any missing columns (safe no-ops on new DBs)
        for col_def in [
            "study_id         BIGINT",
            "pair             VARCHAR",
            "created_at       TIMESTAMP",
            "sampler          VARCHAR",
            "pruner           VARCHAR",
            "n_trials         INTEGER",
            "timeout_sec      INTEGER",
            "weights_json     VARCHAR",
            "profile_json     VARCHAR",
            "notes            VARCHAR",
            "direction        VARCHAR",
            "mode             VARCHAR",
            "code_sha         VARCHAR",
            "paramspace_hash  VARCHAR",
            "sampler_params   VARCHAR",
            "storage          VARCHAR",
            "seed             INTEGER",
            "started_at       TIMESTAMP",
            "finished_at      TIMESTAMP",
            "duration_sec     DOUBLE",
            "optuna_version   VARCHAR",
            "app_version      VARCHAR",
        ]:
            conn.execute(f"ALTER TABLE studies ADD COLUMN IF NOT EXISTS {col_def};")

        # -----------------------------
        # 2) trials
        # -----------------------------
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS trials (
                study_id           BIGINT,
                trial_no           INTEGER,
                pair               VARCHAR,
                created_at         TIMESTAMP DEFAULT now(),

                params_json        VARCHAR,
                perf_json          VARCHAR,

                score              DOUBLE,
                score_raw          DOUBLE,
                score_norm_json    VARCHAR,

                state              VARCHAR,
                error              VARCHAR,

                datetime_start     TIMESTAMP,
                datetime_complete  TIMESTAMP,
                duration_sec       DOUBLE,

                params_hash        VARCHAR
            );
            """
        )

        for col_def in [
            "study_id           BIGINT",
            "trial_no           INTEGER",
            "pair               VARCHAR",
            "created_at         TIMESTAMP",
            "params_json        VARCHAR",
            "perf_json          VARCHAR",
            "score              DOUBLE",
            "score_raw          DOUBLE",
            "score_norm_json    VARCHAR",
            "state              VARCHAR",
            "error              VARCHAR",
            "datetime_start     TIMESTAMP",
            "datetime_complete  TIMESTAMP",
            "duration_sec       DOUBLE",
            "params_hash        VARCHAR",
        ]:
            conn.execute(f"ALTER TABLE trials ADD COLUMN IF NOT EXISTS {col_def};")

        # -----------------------------
        # 3) artifacts
        # -----------------------------
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS artifacts (
                study_id     BIGINT,
                kind         VARCHAR,
                created_at   TIMESTAMP DEFAULT now(),
                payload      BLOB
            );
            """
        )

        for col_def in [
            "study_id   BIGINT",
            "kind       VARCHAR",
            "created_at TIMESTAMP",
            "payload    BLOB",
        ]:
            conn.execute(f"ALTER TABLE artifacts ADD COLUMN IF NOT EXISTS {col_def};")

        # -----------------------------
        # 4) param_ranges (learned)
        # -----------------------------
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS param_ranges (
                pair        VARCHAR,
                env         VARCHAR,
                profile     VARCHAR,
                param       VARCHAR,
                lo          DOUBLE,
                hi          DOUBLE,
                step        DOUBLE,
                created_at  TIMESTAMP DEFAULT now()
            );
            """
        )

        for col_def in [
            "pair        VARCHAR",
            "env         VARCHAR",
            "profile     VARCHAR",
            "param       VARCHAR",
            "lo          DOUBLE",
            "hi          DOUBLE",
            "step        DOUBLE",
            "created_at  TIMESTAMP",
        ]:
            conn.execute(f"ALTER TABLE param_ranges ADD COLUMN IF NOT EXISTS {col_def};")

        # -----------------------------
        # 5) Indexes (best-effort)
        # -----------------------------
        # studies
        conn.execute("CREATE INDEX IF NOT EXISTS idx_studies_id        ON studies(study_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_studies_pair_time ON studies(pair, created_at);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_studies_mode      ON studies(mode);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_studies_profile   ON studies(profile_json);")

        # trials
        conn.execute("CREATE INDEX IF NOT EXISTS idx_trials_study_no   ON trials(study_id, trial_no);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_trials_pair_score ON trials(pair, score);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_trials_state      ON trials(state);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_trials_params_hash ON trials(params_hash);")

        # artifacts
        conn.execute("CREATE INDEX IF NOT EXISTS idx_artifacts_study_kind ON artifacts(study_id, kind, created_at);")

        # param_ranges
        conn.execute("CREATE INDEX IF NOT EXISTS idx_param_ranges_pair_env_profile ON param_ranges(pair, env, profile, created_at);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_param_ranges_param ON param_ranges(param);")

        conn.execute("COMMIT")

    except Exception as e:
        logger.warning("DuckDB schema init failed: %s", e, exc_info=True)
        try:
            conn.execute("ROLLBACK")
        except Exception:
            pass


def save_param_ranges_for_pair_in_duck(
    pair: str,
    ranges: Dict[str, ParamRange],
    *,
    env: Optional[str] = None,
    profile: Optional[str] = None,
) -> None:
    """
    שומר param ranges per pair ב-DuckDB (optuna_cache.duckdb).

    מבנה הטבלה:
        param_ranges(pair, env, profile, param, lo, hi, step, created_at)

    הלוגיקה:
    --------
    - מוחק רשומות קודמות לאותו pair/env/profile.
    - מכניס רשומה אחת לכל פרמטר.
    """
    if not pair or not ranges:
        return

    env_str = (env or getattr(SETTINGS, "env", "local") or "local").lower()
    profile_str = (profile or st.session_state.get("opt_profile", "default") or "default")

    try:
        _ensure_duck_schema()
    except Exception as e:
        logger.warning("save_param_ranges_for_pair_in_duck: _ensure_duck_schema failed: %s", e)

    try:
        conn = get_duck()
    except Exception as e:
        logger.warning("save_param_ranges_for_pair_in_duck: get_duck failed: %s", e)
        return

    rows: List[Tuple[Any, ...]] = []
    now = datetime.now(timezone.utc)

    for name, tpl in ranges.items():
        try:
            lo, hi, step = tpl
            rows.append(
                (
                    str(pair),
                    str(env_str),
                    str(profile_str),
                    str(name),
                    float(lo),
                    float(hi),
                    None if step is None else float(step),
                    now,
                )
            )
        except Exception:
            continue

    if not rows:
        return

    try:
        conn.execute("BEGIN")
        conn.execute(
            """
            DELETE FROM param_ranges
            WHERE pair = ? AND env = ? AND profile = ?
            """,
            [str(pair), env_str, profile_str],
        )
        conn.executemany(
            """
            INSERT INTO param_ranges (
                pair, env, profile, param, lo, hi, step, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.execute("COMMIT")
        logger.info(
            "Saved %d param ranges for pair=%s env=%s profile=%s into param_ranges",
            len(rows),
            pair,
            env_str,
            profile_str,
        )
    except Exception as e:
        logger.warning("save_param_ranges_for_pair_in_duck failed: %s", e)
        try:
            conn.execute("ROLLBACK")
        except Exception:
            pass


def load_param_ranges_for_pair_from_duck(
    pair: str,
    *,
    env: Optional[str] = None,
    profile: Optional[str] = None,
) -> Dict[str, ParamRange]:
    """
    טוען את הטווחים האחרונים עבור pair/env/profile מ-param_ranges.

    אם אין רשומות → מחזיר {}.
    """
    if duckdb is None:
        return {}

    env_str = (env or getattr(SETTINGS, "env", "local") or "local").lower()
    profile_str = (profile or st.session_state.get("opt_profile", "default") or "default")

    try:
        conn = get_ro_duck()
    except Exception as e:
        logger.warning("load_param_ranges_for_pair_from_duck: get_ro_duck failed: %s", e)
        return {}

    try:
        q = """
        SELECT param, lo, hi, step
        FROM param_ranges
        WHERE pair = ? AND env = ? AND profile = ?
        ORDER BY created_at DESC
        """
        df = conn.execute(q, [str(pair), env_str, profile_str]).df()
        if df.empty:
            return {}
        out: Dict[str, ParamRange] = {}
        for _, r in df.iterrows():
            try:
                name = str(r["param"])
                lo = float(r["lo"])
                hi = float(r["hi"])
                step_val = r.get("step")
                step = float(step_val) if step_val is not None and str(step_val) != "nan" else None
                if hi <= lo:
                    hi = lo + 1e-9
                if name not in out:
                    out[name] = (lo, hi, step)
            except Exception:
                continue
        return out
    except Exception as e:
        logger.warning("load_param_ranges_for_pair_from_duck failed: %s", e)
        return {}

# =========================
# SECTION 2.2: JSON & numeric helpers עבור DB
# =========================

def make_json_safe(obj: Any) -> Any:
    """
    Deep-convert objects (Path, Timestamp, numpy types, וכו') לצורות JSON-ידידותיות.

    נשתמש בזה לפני כתיבה ל-param_json / perf_json / artifacts.
    """
    from pathlib import Path as _Path
    from datetime import date as _date, datetime as _dt
    import numpy as _np
    import pandas as _pd

    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, (_Path, _pd.Timestamp, _date, _dt)):
        return str(obj)
    if isinstance(obj, (_np.integer, _np.floating)):
        return obj.item()
    return obj


def _coerce_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    מנסה להמיר עמודות אובייקט ל-numeric כשהגיוני, ולהעלים inf/NaN חריגים.
    שימושי לפני ייצוא / שמירה.
    """
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_object_dtype(out[c]):
            try:
                out[c] = pd.to_numeric(out[c], errors="ignore")
            except Exception:
                pass
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


# =========================
# SECTION 2.3: Studies/trials API
# =========================

def save_trials_to_duck(
    df: pd.DataFrame,
    pair: str,
    sampler: str,
    n_trials: int,
    timeout_sec: int,
    weights: Dict[str, float],
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Optional[int]:
    """
    שמירת df של תוצאות אופטימיזציה (params + metrics) לטבלת DuckDB.

    (גרסת HF): בנוסף לעמודות הבסיס, שומרת גם:
        - created_at
        - state="COMPLETE"
        - score_raw
        - score_norm_json (אם יגיע בעתיד ב-extra_meta)
        - error (כרגע ריק)
        - params_hash (sha256 על params_json)
    """
    if df is None or df.empty:
        logger.warning("save_trials_to_duck: received empty DataFrame, skipping save.")
        return None

    conn = None
    try:
        _ensure_duck_schema()
        conn = get_duck()

        # === בניית study_id ו-meta בסיסי ===
        study_id = int(time.time() * 1000)

        weights_json = json.dumps(make_json_safe(weights), ensure_ascii=False)
        meta = extra_meta or {}

        code_sha = meta.get("code_sha") or os.getenv("GIT_REV", "")
        paramspace_hash = meta.get("paramspace_hash")
        optuna_version = meta.get("optuna_version") or _safe_version("optuna")
        app_version = meta.get("app_version") or OPT_TAB_VERSION

        direction = meta.get("direction", "maximize")
        pruner = meta.get("pruner", "")
        sampler_params = meta.get("sampler_params", "")
        storage = meta.get("storage", "")

        try:
            seed_default = int(st.session_state.get("global_seed", 0))
        except Exception:
            seed_default = 0
        seed = int(meta.get("seed", seed_default))

        profile_obj = meta.get("profile", {})
        profile_json = json.dumps(make_json_safe(profile_obj), ensure_ascii=False)

        notes = str(meta.get("notes", ""))
        duration_sec = float(meta.get("duration_sec", 0.0))

        logger.info(
            "save_trials_to_duck: study_id=%s pair=%s sampler=%s n_trials=%s timeout=%ss",
            study_id,
            pair,
            sampler,
            n_trials,
            timeout_sec,
        )

        # === INSERT study-level row ===
        conn.execute(
            """
            INSERT INTO studies (
                study_id, pair, sampler, n_trials, timeout_sec, weights_json,
                code_sha, paramspace_hash, optuna_version, app_version,
                direction, pruner, sampler_params, storage, seed,
                profile_json, notes, started_at, finished_at, duration_sec
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, now(), now(), ?)
            """,
            [
                study_id,
                pair,
                sampler,
                int(n_trials),
                int(timeout_sec),
                weights_json,
                code_sha,
                paramspace_hash,
                optuna_version,
                app_version,
                direction,
                pruner,
                sampler_params,
                storage,
                seed,
                profile_json,
                notes,
                duration_sec,
            ],
        )

        # === הכנת רשומות עבור trials ===
        df_local = df.reset_index(drop=True)

        metric_cols_known = {"Sharpe", "Profit", "Drawdown", "Score"}
        metric_cols = [c for c in df_local.columns if c in metric_cols_known]
        param_cols = [c for c in df_local.columns if c not in metric_cols]

        if not metric_cols:
            logger.warning(
                "save_trials_to_duck: no recognized metric columns found; "
                "only params_json will be meaningful."
            )

        # ניקח נורמליזציה גלובלית אם נרצה בעתיד
        score_norm_global = meta.get("score_norm") or meta.get("norm") or {}
        score_norm_json_default = json.dumps(make_json_safe(score_norm_global), ensure_ascii=False)

        # הערכת duration ממוצעת פר trial אם קיים duration_sec ברמת study
        per_trial_duration: Optional[float] = None
        if duration_sec > 0 and len(df_local) > 0:
            per_trial_duration = duration_sec / float(len(df_local))

        records: List[Tuple[Any, ...]] = []

        for i, row in df_local.iterrows():
            # פרמטרים
            params_dict: Dict[str, Any] = {}
            for k in param_cols:
                v = row.get(k)
                if pd.api.types.is_numeric_dtype(df_local[k]):
                    try:
                        params_dict[k] = float(v)
                    except Exception:
                        params_dict[k] = None
                else:
                    params_dict[k] = None if pd.isna(v) else str(v)

            params_json = json.dumps(make_json_safe(params_dict), ensure_ascii=False)

            # מטריקות ביצועים
            perf_dict: Dict[str, Any] = {}
            for k in metric_cols:
                val = row.get(k)
                try:
                    perf_dict[k] = float(val) if pd.notna(val) else None
                except Exception:
                    perf_dict[k] = None

            perf_json = json.dumps(make_json_safe(perf_dict), ensure_ascii=False)

            score_val_raw = row.get("Score", float("nan"))
            try:
                score_val = float(score_val_raw)
            except Exception:
                score_val = float("nan")

            # === שדות HF חדשים ===
            created_at = datetime.now(timezone.utc)
            state = "COMPLETE"  # כל trials ששמרנו הגיעו מסשן שהסתיים בהצלחה
            datetime_start = None
            datetime_complete = None
            this_duration = per_trial_duration

            score_raw = score_val
            score_norm_json = score_norm_json_default
            error_msg = ""

            params_hash = hashlib.sha256(
                params_json.encode("utf-8")
            ).hexdigest()[:16]

            records.append(
                (
                    study_id,                 # study_id
                    int(i),                   # trial_no
                    pair,                     # pair
                    params_json,              # params_json
                    perf_json,                # perf_json
                    score_val,                # score
                    created_at,               # created_at
                    state,                    # state
                    datetime_start,           # datetime_start
                    datetime_complete,        # datetime_complete
                    this_duration,            # duration_sec
                    score_raw,                # score_raw
                    score_norm_json,          # score_norm_json
                    error_msg,                # error
                    params_hash,              # params_hash
                )
            )

        if not records:
            logger.warning("save_trials_to_duck: no trial records to insert (records empty).")
            return study_id

        conn.execute("BEGIN")
        conn.executemany(
            """
            INSERT INTO trials (
                study_id,
                trial_no,
                pair,
                params_json,
                perf_json,
                score,
                created_at,
                state,
                datetime_start,
                datetime_complete,
                duration_sec,
                score_raw,
                score_norm_json,
                error,
                params_hash
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            records,
        )
        conn.execute("COMMIT")

        logger.info(
            "save_trials_to_duck: successfully saved %s trials for study_id=%s",
            len(records),
            study_id,
        )

        return study_id

    except Exception as e:
        logger.warning("save_trials_to_duck failed: %s", e, exc_info=True)
        try:
            if conn is not None:
                conn.execute("ROLLBACK")
        except Exception:
            pass
        return None

def _render_feature_selection_summary(
    opt_results: pd.DataFrame,
    config: Dict[str, Any],
) -> None:
    """
    מציג סיכום Feature Selection על תוצאות האופטימיזציה:
    - איזה metric נבחר.
    - כמה פרמטרים נשארו אחרי פילטר.
    - טבלת trials נבחרים + rank/zscore.
    """

    if opt_results is None or opt_results.empty:
        st.info("אין תוצאות אופטימיזציה להצגת Feature Selection.")
        return

    with st.expander("🧬 Feature Selection – פרמטרים חזקים באמת", expanded=False):
        fs_df = select_features(opt_results, config)

        if fs_df.empty:
            st.caption("ה-Feature Selection החזיר טבלה ריקה (כנראה אין metric מתאים או מעט מדי תוצאות).")
            return

        # ננסה לזהות metric מתוך התוצאה (המנוע כבר סימן בעמודות)
        metric_candidates = ["score", "hf_score", "classic_score", "sharpe", "Sharpe", "return"]
        metric_col = next((c for c in metric_candidates if c in fs_df.columns), None)

        if metric_col:
            st.markdown(f"**Metric מרכזי לבחירה:** `{metric_col}`")
        st.caption(f"נבחרו {len(fs_df)} trials ו-{len(fs_df.columns)} עמודות (כולל metricים ופרמטרים).")

        # הצגת הטבלה עם rank / zscore אם קיימים
        st.dataframe(fs_df, use_container_width=True)

        # תובנות מהירות: איזה פרמטרים הופיעו
        param_cols = [
            c
            for c in fs_df.columns
            if c
            not in {
                metric_col,
                "metric_rank",
                "metric_zscore",
                "score",
                "hf_score",
                "classic_score",
                "sharpe",
                "Sharpe",
                "return",
                "drawdown",
                "Drawdown",
                "ES_95",
            }
        ]
        if param_cols:
            st.markdown("**פרמטרים שנשארו אחרי Feature Selection:**")
            st.code(", ".join(param_cols))
        else:
            st.caption("לא נשארו עמודות פרמטרים אחרי הסינון (נשארו רק מטריקות).")

def list_pairs_in_db(limit: int = 200) -> List[str]:
    """
    מחזיר רשימת pairs שונים שנשמרו ב-studies (לעזר ב-UI).
    """
    try:
        conn = get_ro_duck()
        df = conn.execute(
            "SELECT DISTINCT pair FROM studies WHERE pair IS NOT NULL ORDER BY pair LIMIT ?",
            [int(limit)],
        ).df()
        return sorted(df["pair"].dropna().astype(str).tolist())
    except Exception as e:
        logger.debug("list_pairs_in_db failed: %s", e)
        return []


def list_studies_for_pair(pair: str, limit: int = 30) -> pd.DataFrame:
    """
    מחזיר DataFrame של studies עבור pair נתון.

    כולל מידע בסיסי (sampler, n_trials, seed) +
    timestamps לצורך בחירת run להשוואה.
    """
    try:
        conn = get_ro_duck()
        q = """
        SELECT study_id, created_at, sampler, n_trials, timeout_sec,
               code_sha, direction, seed, duration_sec
        FROM studies
        WHERE pair = ?
        ORDER BY created_at DESC
        LIMIT ?
        """
        return conn.execute(q, [pair, int(limit)]).df()
    except Exception as e:
        logger.debug("list_studies_for_pair failed: %s", e)
        return pd.DataFrame()


def load_trials_from_duck(study_id: int) -> pd.DataFrame:
    """
    טוען trials עבור study_id ופורס את ה-jsons ל-DataFrame שטוח:
    - params_json → עמודות פרמטרים
    - perf_json   → Sharpe/Profit/Drawdown וכו'
    - score       → Score (עמודה אחת)

    מחזיר df נקי מספרים כאשר אפשר.
    """
    try:
        conn = get_ro_duck()
        q = """
        SELECT trial_no, params_json, perf_json, score
        FROM trials
        WHERE study_id = ?
        ORDER BY trial_no
        """
        df = conn.execute(q, [int(study_id)]).df()
        if df.empty:
            return df

        params_df = df["params_json"].apply(json.loads).apply(pd.Series)
        perf_df = df["perf_json"].apply(json.loads).apply(pd.Series)

        out = pd.concat(
            [
                params_df,
                perf_df,
                df[["score"]].rename(columns={"score": "Score"}),
            ],
            axis=1,
        )
        return _coerce_numeric_df(out)
    except Exception as e:
        logger.warning("load_trials_from_duck failed: %s", e)
        return pd.DataFrame()


# =========================
# SECTION 2.4: Artifacts API (reports / pareto / manifest)
# =========================

def save_artifact_to_duck(study_id: int, kind: str, payload: bytes) -> None:
    """
    שומר artifact (BLOB) בטבלת artifacts.

    kind יכול להיות:
        - "pareto_csv"
        - "report_md"
        - "manifest_json"
        - "analytics_csv"
    """
    if duckdb is None:
        return
    try:
        _ensure_duck_schema()
        conn = get_duck()
        conn.execute(
            """
            INSERT INTO artifacts (study_id, kind, payload)
            VALUES (?, ?, ?)
            """,
            [int(study_id), str(kind), payload],
        )
    except Exception as e:
        logger.warning("save_artifact_to_duck failed: %s", e)

def load_artifacts_for_study(study_id: int, kind: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    טען artifacts עבור study_id. אם kind לא None → סנן לפי kind.

    מחזיר רשימה של dicts:
        {
          "study_id": int,
          "kind": str,
          "created_at": Timestamp,
          "payload": bytes
        }
    """
    if duckdb is None:
        return []
    try:
        conn = get_ro_duck()
        if kind is None:
            q = "SELECT study_id, kind, created_at, payload FROM artifacts WHERE study_id = ? ORDER BY created_at"
            df = conn.execute(q, [int(study_id)]).df()
        else:
            q = "SELECT study_id, kind, created_at, payload FROM artifacts WHERE study_id = ? AND kind = ? ORDER BY created_at"
            df = conn.execute(q, [int(study_id), str(kind)]).df()
        if df.empty:
            return []
        out: List[Dict[str, Any]] = []
        for _, row in df.iterrows():
            out.append(
                {
                    "study_id": int(row["study_id"]),
                    "kind": str(row["kind"]),
                    "created_at": row["created_at"],
                    "payload": row["payload"],
                }
            )
        return out
    except Exception as e:
        logger.warning("load_artifacts_for_study failed: %s", e)
        return []


# =========================
# SECTION 2.5: Study management utilities
# =========================

def get_latest_study_id_for_pair(pair: str) -> Optional[int]:
    """מחזיר study_id האחרון עבור pair נתון (או None)."""
    try:
        conn = get_ro_duck()
        df = conn.execute(
            "SELECT study_id FROM studies WHERE pair = ? ORDER BY created_at DESC LIMIT 1",
            [str(pair)],
        ).df()
        if df.empty:
            return None
        return int(df["study_id"].iloc[0])
    except Exception as e:
        logger.debug("get_latest_study_id_for_pair failed: %s", e)
        return None


def delete_study_from_duck(study_id: int) -> None:
    """
    מוחק study+trials+artifacts עבור study_id נתון.

    שימושי לניקוי ניסויים כושלים / מיותרים.
    """
    if duckdb is None:
        return
    try:
        conn = get_duck()
        conn.execute("BEGIN")
        conn.execute("DELETE FROM artifacts WHERE study_id = ?", [int(study_id)])
        conn.execute("DELETE FROM trials WHERE study_id = ?", [int(study_id)])
        conn.execute("DELETE FROM studies WHERE study_id = ?", [int(study_id)])
        conn.execute("COMMIT")
    except Exception as e:
        logger.warning("delete_study_from_duck failed: %s", e)
        try:
            conn.execute("ROLLBACK")  # type: ignore[name-defined]
        except Exception:
            pass

"""
חלק 4/15 — Param ranges, metric engine & risk mapping
=====================================================

מה יש כאן:

1. ParamRange + גישה חכמה ל-PARAM_SPECS:
   - get_param_specs_view(...)      → רשימת ParamSpec מסוננת לפי tags.
   - get_default_param_ranges(...)  → טווחי פרמטרים ל-UI/Optuna (עם overrides).

2. מנוע מטריקות:
   - METRIC_KEYS                    → מיפוי אחיד של שמות מטריקות.
   - extract_metrics(perf)          → dict ביצועים → מבנה אחיד.
   - _norm_fallback(perf)           → נרמול לממד [0,1].
   - _score_fallback(norm, weights) → חישוב Score משוקלל.
   - compute_score(perf, weights)   → API מרכזי שמנסה core.metrics, אחרת fallback.

3. רמות מתקדמות לטווחים:
   - shrink_ranges_around_center(...)  → כיווץ טווחים סביב params טובים.
   - ranges_from_dataset(...)          → יצירת טווחים מדאטה (Top-K וכו’).

4. עזרי Backtester:
   - _apply_param_mapping(params, mapping)  → UI → Backtester.
   - _sanitize_bt_kwargs(bt_kwargs)         → התאמה לחתימה של Backtester.
   - get_session_risk_kwargs()             → slippage/fees/risk מתוך session_state.
"""

# =========================
# SECTION 3: Param ranges & ParamSpec integration
# =========================

# נניח מחלקה core.params.ParamSpec כבר בתמונה דרך CORE_PARAM_SPECS
# ו-helperים כמו filter_by_tags, random_sample_from_specs, clamp_params_to_specs וכו'.

try:
    from core.params import (  # type: ignore
        filter_by_tags as params_filter_by_tags,
        random_sample_from_specs as params_random_sample,
        clamp_params_to_specs as params_clamp_to_specs,
        build_distributions as params_build_distributions,
        # 👇 החדשים
        score_params_dict as params_score_params_dict,
        build_param_importance_table as params_build_importance_table,
    )
except Exception:
    params_filter_by_tags = None  # type: ignore
    params_random_sample = None  # type: ignore
    params_clamp_to_specs = None  # type: ignore
    params_build_distributions = None  # type: ignore

    # 👇 אם core.params עוד לא מכיל את זה, נשאיר None
    params_score_params_dict = None  # type: ignore
    params_build_importance_table = None  # type: ignore



# ParamRange מחזיק: (low, high, step | None)
ParamRange: TypeAlias = Tuple[float, float, Optional[float]]


def get_param_specs_view(
    *,
    tags_include: Optional[List[str]] = None,
    tags_exclude: Optional[List[str]] = None,
) -> List[Any]:
    """
    מחזיר רשימת ParamSpec לפי tags.

    tags_include / tags_exclude:
        - include=["signal", "volatility"] → רק פרמטרים עם tags אלה.
        - exclude=["fair_value"]          → מסיר פרמטרים הייחודיים ל-FV.

    אם CORE_PARAM_SPECS לא קיים → מחזיר [].
    """
    specs = None
    if CORE_PARAM_SPECS is not None:
        try:
            specs = list(CORE_PARAM_SPECS)
        except Exception:
            specs = None

    if not specs:
        return []

    # filter_by_tags אם קיים
    if params_filter_by_tags is not None and (tags_include or tags_exclude):
        try:
            return params_filter_by_tags(
                specs=specs,
                include=tags_include,
                exclude=tags_exclude,
            )
        except Exception as e:
            logger.warning("params_filter_by_tags failed, using full specs: %s", e)
            return specs

    return specs


def _merge_ranges(
    base: Dict[str, ParamRange],
    overrides: Dict[str, ParamRange] | None = None,
) -> Dict[str, ParamRange]:
    """מאחד שני dict של טווחים (overrides גובר על base)."""
    if not overrides:
        return base
    merged = dict(base)
    for k, v in overrides.items():
        merged[k] = v
    return merged


def get_default_param_ranges(
    *,
    tags_include: Optional[List[str]] = None,
    tags_exclude: Optional[List[str]] = None,
    profile: str = "default",
) -> Dict[str, ParamRange]:
    """
    נקודת החיבור בין הטאב לבין core.params.PARAM_SPECS.

    סדר עדיפויות:
    1. ParamSpec (CORE_PARAM_SPECS) → טווחים "תיאורטיים" (lo/hi/step).
       - אפשר לסנן לפי tags_include / tags_exclude.
       - כרגע דולגנו על פרמטרים קטגוריאליים (choices).

    2. DEFAULT_PARAM_RANGES → overrides ידניים/legacy אם קיימים.

    3. התאמה לפי profile:
       - "defensive": כיווץ טווחים לפעילות שמרנית יותר.
       - "aggressive": הרחבת טווחים לפריסה רחבה/חקר אגרסיבי.

    4. fallback מינימלי אם הכל חסר.

    החתימה הזו מאפשרת לנו:
    - לטעון פרופיל שונה לדשבורד (env / config).
    - לשמור את הטאב גמיש מול הרחבות עתידיות.
    """
    # 1) ParamSpec base ranges
    ranges: Dict[str, ParamRange] = {}
    specs_view = get_param_specs_view(
        tags_include=tags_include,
        tags_exclude=tags_exclude,
    )

    if specs_view:
        logger.info("Building default ranges from PARAM_SPECS (%d specs)", len(specs_view))
        for p in specs_view:
            try:
                is_cat = getattr(p, "is_categorical", False)
                lo = getattr(p, "lo", None)
                hi = getattr(p, "hi", None)
                step = getattr(p, "step", None)
                if is_cat:
                    # categorical לא נכנס כרגע ל-ParamRange (יתועדף דרך distributions)
                    continue
                if lo is None or hi is None:
                    continue
                lo_f = float(lo)
                hi_f = float(hi)
                step_f: Optional[float] = float(step) if step is not None else None
                if hi_f <= lo_f:
                    hi_f = lo_f + 1e-9
                ranges[p.name] = (lo_f, hi_f, step_f)
            except Exception:
                continue

    # 2) DEFAULT_PARAM_RANGES (legacy overrides)
    try:
        if DEFAULT_PARAM_RANGES:
            legacy: Dict[str, ParamRange] = {}
            for name, tpl in DEFAULT_PARAM_RANGES.items():  # type: ignore[assignment]
                try:
                    lo, hi, step = tpl
                    legacy[name] = (
                        float(lo),
                        float(hi),
                        float(step) if step is not None else None,
                    )
                except Exception:
                    continue
            if legacy:
                logger.info("Merging DEFAULT_PARAM_RANGES (%d overrides)", len(legacy))
                ranges = _merge_ranges(ranges, legacy)
    except Exception:
        pass

    # 3) Profile-based adjustment (defensive / aggressive)
    #    הרעיון: לא לשנות ערכים, רק להצר/להרחיב את מרחב החיפוש.
    prof = str(profile).strip().lower()
    if ranges and prof in {"defensive", "aggressive"}:
        factor = 0.5 if prof == "defensive" else 1.5
        adj: Dict[str, ParamRange] = {}
        for name, (lo, hi, step) in ranges.items():
            mid = (lo + hi) / 2.0
            span = (hi - lo) * factor / 2.0
            lo2, hi2 = mid - span, mid + span
            if hi2 <= lo2:
                hi2 = lo2 + 1e-9
            adj[name] = (lo2, hi2, step)
        ranges = adj

    # 4) fallback אם אין שום מקור
    if not ranges:
        logger.warning("No param specs found; using minimal fallback ranges.")
        ranges = {
            "lookback": (20.0, 120.0, 5.0),
            "z_entry":  (1.0, 3.0, 0.1),
            "z_exit":   (0.2, 2.0, 0.1),
        }

    return ranges


def shrink_ranges_around_center(
    ranges: Dict[str, ParamRange],
    center_params: Dict[str, Any],
    *,
    radius_factor: float = 0.3,
) -> Dict[str, ParamRange]:
    """
    כיווץ טווחים סביב params קיימים (למשל best_params).

    radius_factor:
        - 0.3 → חלון חדש בגודל 30% מהטווח המקורי סביב הערך הנוכחי.
        - שימושי ל-"local search" אחרי run רחב.

    עובד רק על פרמטרים שנמצאים גם ב-ranges וגם ב-center_params.
    """
    out: Dict[str, ParamRange] = dict(ranges)
    rf = float(max(0.01, min(radius_factor, 5.0)))
    for name, (lo, hi, step) in ranges.items():
        if name not in center_params:
            continue
        try:
            center_val = float(center_params[name])
        except Exception:
            continue
        span_orig = max(hi - lo, 1e-9)
        span_new = span_orig * rf
        lo2 = center_val - span_new / 2.0
        hi2 = center_val + span_new / 2.0
        if hi2 <= lo2:
            hi2 = lo2 + 1e-9
        out[name] = (lo2, hi2, step)
    return out


def ranges_from_dataset(
    df: pd.DataFrame,
    *,
    params: Optional[List[str]] = None,
    q_low: float = 0.10,
    q_high: float = 0.90,
) -> Dict[str, ParamRange]:
    """
    יצירת ranges מתוך DataFrame של תוצאות (למשל opt_df / batch_df).

    - אם params=None → נשתמש בכל העמודות המספריות *שאינן* מטריקות (Sharpe/Profit/DD/Score).
    - q_low, q_high → quantiles שמגדירים את הטווח (למשל 10%–90%).

    זה מאפשר:
        - ללמוד טווחים חדשים מתוך Top-K תוצאות.
        - לייצר preset של active_ranges מתוך opt_df.
    """
    if df is None or df.empty:
        return {}

    df_num = df.select_dtypes(include=[np.number]).copy()
    if df_num.empty:
        return {}

    metric_cols = [c for c in df_num.columns if METRIC_KEYS.get(str(c).lower())]
    candidate_cols = [c for c in df_num.columns if c not in metric_cols]

    cols = params or candidate_cols
    if not cols:
        return {}

    q_low = float(max(0.0, min(q_low, 0.49)))
    q_high = float(max(q_low + 0.01, min(q_high, 0.99)))

    rngs: Dict[str, ParamRange] = {}
    for c in cols:
        if c not in df_num.columns:
            continue
        s = pd.to_numeric(df_num[c], errors="coerce").dropna()
        if s.empty:
            continue
        lo = float(s.quantile(q_low))
        hi = float(s.quantile(q_high))
        if hi <= lo:
            hi = lo + 1e-9
        rngs[str(c)] = (lo, hi, None)

    return rngs


# =========================
# SECTION 3.2: Metric engine — mapping, normalization, scoring
# =========================

METRIC_KEYS: Dict[str, str] = {
    "sharpe": "Sharpe",
    "sharpe_ratio": "Sharpe",
    "sortino": "Sortino",
    "sortino_ratio": "Sortino",
    "profit": "Profit",
    "pnl": "Profit",
    "pnl_usd": "Profit",
    "ret": "Profit",
    "return": "Profit",
    "drawdown": "Drawdown",
    "maxdd": "Drawdown",
    "max_drawdown": "Drawdown",
    "calmar": "Calmar",
    "calmar_ratio": "Calmar",
    "winrate": "WinRate",
    "win_rate": "WinRate",
    "hitrate": "WinRate",
}

# מיפוי בין שמות objective מה-UI לבין שמות המטריקות הפנימיות
OBJECTIVE_TO_METRIC: Dict[str, str] = {
    "SHARPE": "Sharpe",
    "SHARPE_RATIO": "Sharpe",
    "SORTINO": "Sortino",
    "SORTINO_RATIO": "Sortino",
    "CALMAR": "Calmar",
    "CALMAR_RATIO": "Calmar",
    "RETURN": "Profit",
    "RET": "Profit",
    "PROFIT": "Profit",
    "TAILRISK": "Drawdown",
    "DRAWDOWN": "Drawdown",
    "MAXDD": "Drawdown",
    "MAX_DRAWDOWN": "Drawdown",
}

def _apply_objective_context_to_weights(
    base_weights: Dict[str, float],
) -> Dict[str, float]:
    """
    מעדכן את משקלי המטריקות לפי הקשר ה-UI של הטאב:

    - primary_objective / secondary_objective + secondary_weight
    - scenario_profile ("Neutral"/"Risk-On"/"Risk-Off"/"Crisis")
    - scenario_tail_weight (כמה לתת דגש ל-TailRisk/Drawdown)

    הערה:
    -----
    הפונקציה **לא** משנה את ה-sign של המטריקות (זה נשאר ב-compute_score),
    אלא רק את המשקולות היחסיות שלהן.
    """
    # נתחיל מעותק כדי לא ללכלך את המקור
    w: Dict[str, float] = {str(k): float(v) for k, v in (base_weights or {}).items()}

    # נוודא שכל המטריקות העיקריות מופיעות (גם אם 0) כדי לא לקבל KeyError
    for k in ("Sharpe", "Sortino", "Calmar", "Profit", "Drawdown", "WinRate"):
        w.setdefault(k, 0.0)

    # ננסה לקרוא הקשר מה-session (opt_run_cfg נבנה ב-render_optimization_tab)
    ctx_cfg = st.session_state.get("opt_run_cfg", {}) or {}
    prim_raw = str(ctx_cfg.get("primary_objective", "") or "").upper()
    sec_raw = str(ctx_cfg.get("secondary_objective", "") or "").upper()
    sec_weight_cfg = float(ctx_cfg.get("secondary_objective_weight", 0.0) or 0.0)

    primary_metric = OBJECTIVE_TO_METRIC.get(prim_raw)
    secondary_metric = OBJECTIVE_TO_METRIC.get(sec_raw) if sec_raw else None

    # -------------------------
    # א. primary / secondary objective
    # -------------------------
    if primary_metric is not None:
        # נשמור על סכום משקולות = 1 מבחינת |w_i| (פנימית ל-scoring)
        # נבנה מטרה: primary מקבל לפחות 60%, secondary (אם יש) לפי secondary_weight,
        # ושאר המטריקות יקבלו את השאר (אם נשאר).
        sec_w = max(0.0, min(sec_weight_cfg, 0.8)) if secondary_metric else 0.0
        prim_w_target = max(0.6, 1.0 - sec_w)  # לפחות 60% ל-primary
        other_w_total = max(0.0, 1.0 - prim_w_target - sec_w)

        # אם אין משקל ל-"אחרים" — נשאיר אותם קרובים לאפס
        # נשתמש במשקולות המקוריות כדי לחלק את ה-"שארית"
        w_orig_abs = {k: abs(v) for k, v in w.items()}
        # ננקה primary/secondary מתוך "אחרים"
        for k in (primary_metric, secondary_metric):
            if k is not None:
                w_orig_abs.pop(k, None)
        denom = sum(w_orig_abs.values()) or 1.0

        new_w: Dict[str, float] = {k: 0.0 for k in w.keys()}
        new_w[primary_metric] = prim_w_target
        if secondary_metric is not None:
            new_w[secondary_metric] = sec_w

        for k, vabs in w_orig_abs.items():
            share = vabs / denom
            new_w[k] = other_w_total * share

        w = new_w

    # -------------------------
    # ב. Scenario profile / tail-weight
    # -------------------------
    scen = str(ctx_cfg.get("scenario_profile", "Neutral") or "").strip()
    scen = scen.lower()
    tail_w = float(ctx_cfg.get("scenario_tail_weight", 0.0) or 0.0)
    tail_w = max(0.0, min(tail_w, 1.0))

    if tail_w > 0.0:
        # נמשוך את המשקל לכיוון Drawdown ב-Risk-Off/Crisis
        if scen in ("risk-off", "crisis"):
            # מגבירים Drawdown, מורידים Profit
            w["Drawdown"] = w.get("Drawdown", 0.0) * (1.0 + tail_w)
            w["Profit"] = w.get("Profit", 0.0) * (1.0 - 0.5 * tail_w)
            w["Sharpe"] = w.get("Sharpe", 0.0) * (1.0 - 0.3 * tail_w)
        elif scen in ("risk-on",):
            # להפך – יותר דגש על Profit/Sharpe, פחות על DD
            w["Drawdown"] = w.get("Drawdown", 0.0) * (1.0 - 0.5 * tail_w)
            w["Profit"] = w.get("Profit", 0.0) * (1.0 + tail_w)
            w["Sharpe"] = w.get("Sharpe", 0.0) * (1.0 + 0.5 * tail_w)
        else:
            # Neutral → אין שינוי מיוחד
            pass

    # נורמליזצה סופית כך שסכום |w_i| = 1 (מונע התפוצצות)
    z = sum(abs(v) for v in w.values()) or 1.0
    w_norm = {k: float(v) / z for k, v in w.items()}

    return w_norm

def extract_metrics(perf: Dict[str, Any]) -> Dict[str, float]:
    """
    מתאם dict ביצועים למבנה אחיד עם מפתחות סטנדרטיים:

        {
          "Sharpe":  float,
          "Sortino": float,
          "Calmar":  float,
          "Profit":  float,
          "Drawdown": float,
          "WinRate": float,
        }

    שדות שלא קיימים יקבלו 0.0 (כדי למנוע KeyError).
    """
    out: Dict[str, float] = {
        "Sharpe": 0.0,
        "Sortino": 0.0,
        "Calmar": 0.0,
        "Profit": 0.0,
        "Drawdown": 0.0,
        "WinRate": 0.0,
    }
    for k_raw, v in (perf or {}).items():
        key_norm = METRIC_KEYS.get(str(k_raw).strip().lower())
        if not key_norm:
            continue
        try:
            out[key_norm] = float(v)
        except Exception:
            # נשאיר ערך default (0.0)
            pass
    return out

def _infer_sample_size_from_perf(perf: Dict[str, Any]) -> int:
    """
    ניסיון אינטליגנטי להוציא גודל מדגם (T) מתוך מילון perf:

    מחפש שדות אופייניים:
        - Trades / trades / n_trades
        - n_obs / n_samples / T

    ואם לא מוצא – נופל לברירת מחדל:
        - opt_dsr_default_t מתוך session_state (אם קיים)
        - אחרת 250
    """
    candidate_keys = ("Trades", "trades", "n_trades", "n_obs", "n_samples", "T")

    for key in candidate_keys:
        if key in perf and perf[key] is not None:
            try:
                t = int(perf[key])
                if t > 0:
                    return t
            except Exception:
                continue

    # fallback: אפשר לשלוט בזה מה־UI
    try:
        t_default = int(st.session_state.get("opt_dsr_default_t", 250))
    except Exception:
        t_default = 250

    return max(1, t_default)

def _compute_structural_penalties(
    metrics: Dict[str, float],
    perf: Dict[str, Any],
) -> Tuple[Dict[str, float], float]:
    """
    מחשב ענישה מבנית (structural penalties) על סמך:

    - Sharpe      → אסור שיהיה <= 0, ומעדיפים Sharpe≥1.
    - Drawdown    → Penalize חזק כשה־DD גדול.
    - גודל מדגם T → מעט טריידים → פחות אמינות.
    - DSR / p_overfit → נלחמים ב-overfitting.

    מחזיר:
        (penalty_info_dict, total_penalty_factor)

    כאשר total_penalty_factor מוכפל ב-Score הגולמי.
    """

    # --- קריאת המטריקות הבסיסיות ---
    sh = float(metrics.get("Sharpe", 0.0))
    dd_raw = float(metrics.get("Drawdown", 0.0))
    dd = abs(dd_raw)

    # גודל מדגם משוער (טריידים / תצפיות)
    T = _infer_sample_size_from_perf(perf)

    # הערכת מספר אסטרטגיות שנבדקו (multiple testing) — נלקח מה־session אם קיים
    try:
        n_strategies = int(st.session_state.get("opt_n_trials", 200))
    except Exception:
        n_strategies = 200
    n_strategies = max(1, n_strategies)

    # --- DSR / p_overfit ---
    try:
        if sh > 0.0:
            dsr, p_eff = deflated_sharpe_ratio(
                sharpe=sh,
                t=T,
                n_strategies=n_strategies,
                skew=0.0,
                kurt=3.0,
                two_sided=False,
                use_student_t=True,
                max_strategies=2000,
            )
        else:
            # Sharpe שלילי → אין אלפא, אין טעם ב־DSR
            dsr, p_eff = sh, 1.0
    except Exception:
        dsr, p_eff = sh, 1.0

    # --- ענישה לפי Sharpe ---
    if sh <= 0.0:
        sharpe_penalty = 0.05  # לפסול כמעט לגמרי אסטרטגיות עם Sharpe<=0
    elif sh < 0.5:
        sharpe_penalty = 0.4
    elif sh < 1.0:
        sharpe_penalty = 0.7
    elif sh < 1.5:
        sharpe_penalty = 0.9
    else:
        sharpe_penalty = 1.0

    # --- ענישה לפי Drawdown (נניח ש־DD בסקאלה 0–1) ---
    if dd <= 0.10:
        dd_penalty = 1.0
    elif dd <= 0.20:
        dd_penalty = 0.9
    elif dd <= 0.30:
        dd_penalty = 0.75
    elif dd <= 0.40:
        dd_penalty = 0.5
    else:
        dd_penalty = 0.25  # DD קיצוני → עונש כבד

    # --- ענישה לפי מספר טריידים / תצפיות ---
    try:
        min_trades = int(st.session_state.get("opt_min_trades", 30))
    except Exception:
        min_trades = 30

    if T <= 0:
        trades_penalty = 0.1
    elif T < min_trades:
        trades_penalty = max(0.1, T / float(min_trades))
    else:
        trades_penalty = 1.0

    # --- ענישה לפי DSR / p_overfit ---
    # dsr ≈ כמה ה-Sharpe "שווה" אחרי תיקון למולטיפל-טסטינג
    if dsr <= 0.0:
        dsr_penalty = 0.1
    elif dsr < 0.5:
        dsr_penalty = 0.4
    elif dsr < 1.0:
        dsr_penalty = 0.7
    elif dsr < 1.5:
        dsr_penalty = 0.9
    else:
        dsr_penalty = 1.0

    # p_overfit גבוה → סיכוי גבוה שהביצוע הוא רעש
    # p≈1 → penalty≈0.3, p≈0 → penalty≈1
    p_overfit_penalty = float(max(0.3, 1.0 - 0.7 * p_eff))

    # --- חיבור הכל יחד ---
    penalty_components: Dict[str, float] = {
        "Sharpe_penalty": sharpe_penalty,
        "Drawdown_penalty": dd_penalty,
        "Trades_penalty": trades_penalty,
        "DSR_penalty": dsr_penalty,
        "p_overfit_penalty": p_overfit_penalty,
        "DSR": float(dsr),
        "p_overfit": float(p_eff),
        "T_effective": float(T),
    }

    total_penalty = (
        sharpe_penalty
        * dd_penalty
        * trades_penalty
        * dsr_penalty
        * p_overfit_penalty
    )

    # לא נאפשר penalty>1 (שלא יחזק סקור חלש), רק מחליש/משאיר
    total_penalty = float(min(1.0, max(0.0, total_penalty)))

    return penalty_components, total_penalty

def _extract_params_from_opt_row(row: pd.Series) -> Dict[str, Any]:
    """
    מקבל שורה מ-opt_df (תוצאת אופטימיזציה) ומחזיר dict של פרמטרים בלבד,
    בלי המטריקות (Sharpe/Score/Drawdown וכו') ועמודות עזר (Pair/study_id/...).

    משתמש ב-METRIC_KEYS כדי לסנן מטריקות.
    """
    metric_like = {
        c
        for c in row.index
        if METRIC_KEYS.get(str(c).lower())
        or str(c) in {"Score", "Pair", "study_id", "trial_no"}
    }

    params: Dict[str, Any] = {}
    for name in row.index:
        if name in metric_like:
            continue
        val = row[name]
        # אם זה NaN – לא נכניס
        try:
            if pd.isna(val):
                continue
        except Exception:
            pass
        params[str(name)] = val
    return params

def _norm_fallback(perf: Dict[str, Any]) -> Dict[str, float]:
    """
    Normalization fallback כאשר core.metrics.normalize_metrics לא זמין.

    רעיון:
    - Sharpe:   tanh(Sharpe / 3) במיפוי ל-[0,1].
    - Sortino:  tanh(Sortino / 3) ל-[0,1].
    - Calmar:   tanh(Calmar / 3) ל-[0,1].
    - Profit:   tanh(Profit / 1e4) ל-[0,1].
    - Drawdown: 1 - clamp(DD, 0, 1).
    - WinRate:  WinRate% / 100.

    זה נותן "פרופיל" נורמלי יחיד שניתן להשוות בין ניסויים.
    """
    m = extract_metrics(perf)

    sharpe = float(m.get("Sharpe", 0.0))
    sortino = float(m.get("Sortino", 0.0))
    calmar = float(m.get("Calmar", 0.0))
    profit = float(m.get("Profit", 0.0))
    dd = float(m.get("Drawdown", 0.0))
    winrate = float(m.get("WinRate", 0.0))

    s = 0.5 + math.tanh(sharpe / 3.0) / 2.0
    so = 0.5 + math.tanh(sortino / 3.0) / 2.0
    ca = 0.5 + math.tanh(calmar / 3.0) / 2.0
    p = 0.5 + math.tanh(profit / 1e4) / 2.0
    d = 1.0 - min(max(dd, 0.0), 1.0)
    w = max(0.0, min(winrate / 100.0, 1.0))

    return {
        "Sharpe": s,
        "Sortino": so,
        "Calmar": ca,
        "Profit": p,
        "Drawdown": d,
        "WinRate": w,
    }


def _score_fallback(norm: Dict[str, float], weights: Dict[str, float]) -> float:
    """
    Weighted score fallback כאשר core.metrics.compute_weighted_score לא זמין.

    משתמש במפת weights חופשית; מנרמל כך שסכום הערכים |w_i| = 1.
    """
    # Normalise weights
    w_raw = {k: float(v) for k, v in (weights or {}).items()}
    z = sum(abs(v) for v in w_raw.values()) or 1.0
    w = {k: v / z for k, v in w_raw.items()}

    score = 0.0
    for k, v in norm.items():
        score += w.get(k, 0.0) * float(v)
    return float(score)


def compute_score(
    perf: Dict[str, Any],
    weights: Dict[str, float],
) -> Tuple[Dict[str, float], float]:
    """
    מנגנון ניקוד מרכזי — גרסה ברמת קרן גידור:

    שכבה 1: נירמול מטריקות
    -----------------------
    - אם core.metrics.normalize_metrics / compute_weighted_score זמינים:
        משתמש בהם (זוהי שכבת הנירמול/score "הרשמית").
    - אחרת:
        fallback ל-_norm_fallback + _score_fallback (tanh וכו').

    שכבה 2: הקשר טאב (Objective / Scenario)
    ---------------------------------------
    - _apply_objective_context_to_weights:
        primary_objective / secondary_objective / scenario_profile / tail_weight
        → משנה את משקלי המטריקות (Sharpe/Profit/DD/...) לפני השיקלול.

    שכבה 3: ענישה מבנית (Structural Penalties)
    -------------------------------------------
    על בסיס:
    - Sharpe (רצוי > 1).
    - Drawdown (רצוי נמוך, < 0.2–0.3).
    - מספר טריידים/תצפיות T (רצוי ≥ 30).
    - Deflated Sharpe Ratio (DSR) + p_overfit.

    התוצאה:
    --------
    מחזיר:
        norm_ext, final_score

    כאשר:
    - norm_ext כולל:
        * המפתחות הרגילים ("Sharpe", "Profit", "Drawdown", ... מנורמלים)
        * Score_raw (לפני ענישה)
        * DSR, p_overfit
        * penalty_total + penalty_components
    - final_score:
        Score_raw * penalty_total
    """

    # 1) Shelחית metrics נקיות מה-Backtester
    perf_clean = extract_metrics(perf)

    # 2) התאמת משקולות לפי Objective/Scenario של הטאב
    eff_weights = _apply_objective_context_to_weights(weights or {})

    # 3) נירמול + Score גולמי (שכבה כמותית "טהורה")
    norm: Dict[str, float]
    base_score: float

    if callable(normalize_metrics) and callable(compute_weighted_score):
        try:
            norm = normalize_metrics(perf_clean)  # type: ignore[arg-type]
            base_score = float(compute_weighted_score(norm, eff_weights))  # type: ignore[arg-type]
        except Exception as e:
            logger.warning("compute_score: core.metrics failed, using fallback: %s", e)
            norm = _norm_fallback(perf_clean)
            base_score = _score_fallback(norm, eff_weights)
    else:
        norm = _norm_fallback(perf_clean)
        base_score = _score_fallback(norm, eff_weights)

    # 4) ענישה מבנית (Sharpe / DD / Trades / DSR / p_overfit)
    try:
        penalty_info, penalty_total = _compute_structural_penalties(perf_clean, perf)
    except Exception as e:
        logger.debug("compute_score: structural penalties failed, using penalty_total=1.0: %s", e)
        penalty_info = {}
        penalty_total = 1.0

    final_score = float(base_score) * float(penalty_total)

    # הגנות בסיסיות: אם יצא NaN/Inf → דופקים בחריפות (לא נותנים לשבור Optuna)
    try:
        if not math.isfinite(final_score):
            final_score = -1e12
    except Exception:
        final_score = -1e12

    # 5) הרחבת norm במידע נוסף (DSR / penalty / Score_raw)
    norm_ext: Dict[str, float] = dict(norm)

    # Score_raw – הסקור לפני ענישה
    norm_ext["Score_raw"] = float(base_score)

    # DSR / p_overfit / T_effective מתוך penalty_info (אם קיימים)
    for key in ("DSR", "p_overfit", "T_effective"):
        if key in penalty_info:
            try:
                norm_ext[key] = float(penalty_info[key])  # type: ignore[assignment]
            except Exception:
                continue

    # ליניאריזציה של רכיבי ענישה
    try:
        penalty_components = {
            k: float(v)
            for k, v in penalty_info.items()
            if k not in {"DSR", "p_overfit", "T_effective"}
        }
    except Exception:
        penalty_components = {}

    norm_ext["penalty_total"] = float(penalty_total)
    # ננרמל מעט לגודל מוגבל (לא חובה, אבל נוח)
    for k, v in list(penalty_components.items()):
        if not math.isfinite(v):
            penalty_components[k] = 0.0

    # אפשר לשמור גם כמילון מלא (שימושי ב-user_attrs / דוחות)
    norm_ext["penalty_Sharpe"] = float(penalty_components.get("Sharpe_penalty", 1.0))
    norm_ext["penalty_Drawdown"] = float(penalty_components.get("Drawdown_penalty", 1.0))
    norm_ext["penalty_Trades"] = float(penalty_components.get("Trades_penalty", 1.0))
    norm_ext["penalty_DSR"] = float(penalty_components.get("DSR_penalty", 1.0))
    norm_ext["penalty_p_overfit"] = float(penalty_components.get("p_overfit_penalty", 1.0))

    return norm_ext, final_score

# =========================
# SECTION 3.3: Mapping & Backtester kwargs
# =========================

def _apply_param_mapping(params: Dict[str, Any], mapping: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """
    ממפה dict של פרמטרים לפי mapping (UI → Backtester).

    mapping לדוגמה:
        {"z_entry": "z_open", "z_exit": "z_close"}

    params:
        {"z_entry": 2.0, "z_exit": 0.5, "lookback": 60}

    תוצאה:
        {"z_open": 2.0, "z_close": 0.5, "lookback": 60}
    """
    if not mapping:
        return params
    return {mapping.get(k, k): v for k, v in params.items()}


def _sanitize_bt_kwargs(bt_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    מסנן kwargs כך שיישארו רק פרמטרים שמופיעים בחתימה של Backtester.__init__.

    הערה חשובה:
    -----------
    - כאשר רצים בתוך Streamlit (RUNNING_IN_STREAMLIT=True) — נשמור על הסינון
      כדי לא להפיל את ה-UI על TypeError.
    - כאשר רצים ב-CLI / סקריפט (RUNNING_IN_STREAMLIT=False) — לא נסנן בכלל,
      כדי לא לזרוק את רוב הפרמטרים שהאופטימיזציה מדגמת.
    """
    # CLI / סקריפטים → בלי סינון, כדי ש-Backtester יקבל את כל הפרמטרים
    if not RUNNING_IN_STREAMLIT:
        logger.info(
            "OptTab._sanitize_bt_kwargs (CLI mode): skipping sanitization, passing %d keys.",
            len(bt_kwargs),
        )
        return bt_kwargs

    # מצב Streamlit רגיל – נשמור על הסינון כמו קודם
    try:
        if Backtester is None:
            return bt_kwargs
        sig = inspect.signature(Backtester.__init__)
        valid_names = {
            p.name
            for p in sig.parameters.values()
            if p.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        }
        valid_names.discard("self")
        cleaned = {k: v for k, v in bt_kwargs.items() if k in valid_names}
        logger.info(
            "OptTab._sanitize_bt_kwargs kept keys (Streamlit): %s",
            sorted(cleaned.keys()),
        )
        return cleaned
    except Exception as e:
        logger.debug("OptTab._sanitize_bt_kwargs failed, returning raw kwargs: %s", e)
        return bt_kwargs





def get_session_risk_kwargs() -> Dict[str, Any]:
    try:
        slippage = float(st.session_state.get("opt_slippage_bps", 2.0))
        fees = float(st.session_state.get("opt_fees_bps", 1.0))
        return {
            "slippage_bps": slippage,
            "commission_bps": fees,   # 👈 זה נכנס ל-Backtester
            "rebalance_days": int(st.session_state.get("opt_rebalance_days", 5)),
        }
    except Exception:
        return {}


"""
חלק 5/15 — Sampling, Backtest wrapper, Objective & Optuna runner
=================================================================

מרכיבי החלק:

1. paramspace_hash(ranges)       — חתימת מרחב הפרמטרים (לשחזור מלא).
2. _make_trial_rng(trial)        — RNG דטרמיניסטי לכל Trial לפי global_seed.
3. _sample_params(...)           — דגימת פרמטרים (Optuna / ParamSpec / fallback).
4. run_backtest(...)             — עטיפה אחידה ל-Backtester עם טיפול בשגיאות.
5. _build_multi_objective_spec() — הגדרת מטרות Multi-Objective (metrics + directions).
6. _objective_factory(...)       — מפעל objective יחיד (Single או Multi).
7. run_optuna_for_pair(...)      — מנוע אופטימיזציה מלא (Optuna / simulated fallback).
"""

import hashlib


# =========================
# SECTION 4: Param space hash (reproducibility)
# =========================

def paramspace_hash(ranges: Dict[str, ParamRange]) -> str:
    """
    מחזיר hash יציב של מרחב הפרמטרים (ranges).

    - מסדר את הטווחים לפי שם פרמטר.
    - הופך ל-list [(name, lo, hi, step), ...].
    - serializes כ-JSON ואז sha1.

    שימוש:
        hash_str = paramspace_hash(ranges)
    """
    payload = []
    for name, (lo, hi, step) in sorted(ranges.items(), key=lambda kv: kv[0]):
        payload.append(
            (
                str(name),
                float(lo),
                float(hi),
                None if step is None else float(step),
            )
        )
    raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()  # noqa: S324


# =========================
# SECTION 4.1: RNG per trial & param sampling (עם ParamSpec integration)
# =========================

def _make_trial_rng(trial: Any | None) -> "np.random.Generator":
    """
    יוצר RNG דטרמיניסטי לכל trial:

    - אם יש trial.number → seed = global_seed ^ trial.number
    - אם אין trial → seed = global_seed ^ internal_counter

    כך:
    - דטרמיניזם מלא כשה-seed קבוע.
    - לא תלוי ב-time.time() (שגורם לחוסר יציבות בין ריצות).
    """
    base_seed = int(st.session_state.get("global_seed", 1337))
    if trial is not None and hasattr(trial, "number"):
        mix = int(getattr(trial, "number", 0))
    else:
        key = "_opt_internal_trial_counter"
        cnt = int(st.session_state.get(key, 0))
        st.session_state[key] = cnt + 1
        mix = cnt
    return np.random.default_rng(base_seed ^ mix)

# =========================
# Main render function
# =========================

def _sample_params(
    trial: Any,
    ranges: Dict[str, ParamRange],
    rng: "np.random.Generator | None" = None,
    param_specs: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """
    דגימת וקטור פרמטרים אחד (Optuna / ParamSpec / fallback),
    עם שכבת ייצוב אופציונלית בעזרת apply_risk_parity_to_params.
    """
    if rng is None:
        rng = _make_trial_rng(trial)

    params: Dict[str, Any] = {}

    # --- 1) Optuna path (single/multi) ---
    if optuna is not None and trial is not None and hasattr(trial, "suggest_float"):
        for name, (lo, hi, step) in ranges.items():
            lo = float(lo)
            hi = float(hi)
            if hi <= lo:
                hi = lo + 1e-9
            if step is None:
                params[name] = trial.suggest_float(name, lo, hi)
            else:
                params[name] = trial.suggest_float(name, lo, hi, step=float(step))

    # --- 2) ParamSpec-based sampling (כשאין Optuna) ---
    elif trial is None and param_specs and params_random_sample is not None:
        sample: Dict[str, Any] = {}
        try:
            # חתימה בסיסית
            sample = params_random_sample(param_specs)  # type: ignore[call-arg]
        except TypeError:
            try:
                sample = params_random_sample(specs=param_specs, rng=rng)  # type: ignore[call-arg]
            except Exception as e:
                logger.warning("params_random_sample failed, falling back to ranges: %s", e)
                sample = {}
        except Exception as e:
            logger.warning("params_random_sample failed, falling back to ranges: %s", e)
            sample = {}

        if isinstance(sample, dict) and sample:
            params = {str(k): v for k, v in sample.items()}

    # --- 3) Fallback: uniform over ranges (עם RNG) ---
    if not params:
        for name, (lo, hi, step) in ranges.items():
            lo = float(lo)
            hi = float(hi)
            if hi <= lo:
                hi = lo + 1e-9
            if step is None:
                params[name] = float(rng.uniform(lo, hi))
            else:
                grid = np.arange(lo, hi + 1e-9, float(step))
                if grid.size == 0:
                    params[name] = lo
                else:
                    idx = int(rng.integers(0, len(grid)))
                    params[name] = float(grid[idx])

    # --- 4) אופציונלי: ייצוב בעזרת risk-parity על הפרמטרים עצמם ---
    use_param_rp = bool(st.session_state.get("opt_use_param_rp", False))
    if use_param_rp and callable(globals().get("apply_risk_parity_to_params")):
        try:
            params = apply_risk_parity_to_params(  # type: ignore[operator]
                params,
                mode="inverse_deviation",   # ברירת מחדל הגיונית
                min_weight=0.0,
                exclude_keys=None,
                return_weights=False,
            )
        except Exception as e:  # לא מפיל – רק מדווח בלוג
            logger.debug("apply_risk_parity_to_params failed, using raw params: %s", e)

    return params


# =========================
# SECTION 4.2: Backtest wrapper (resilient)
# =========================

def run_backtest(sym1: str, sym2: str, params: Dict[str, Any]) -> Dict[str, float]:
    """
    עטיפה פשוטה ל-Backtester, עם קצת הגנות:

    - מוציאים z_entry / z_exit / lookback / notional / slippage_bps / transaction_cost_per_trade מתוך params.
    - לא מעבירים את אותם פרמטרים פעמיים (לא דרך **params וגם דרך ארגומנטים מפורשים).
    - ב-CLI / מחוץ ל-Streamlit זה יעבוד נקי בלי session_state.
    - במקרה של שגיאה → penalty ברור (Sharpe<0, Profit<0, DD=1.0) ולא NameError.
    """

    # סימולציה אם אין Backtester (ל-dev בלבד)
    if Backtester is None:
        seed = int(st.session_state.get("global_seed", 1337)) if hasattr(st, "session_state") else 1337
        rng = np.random.default_rng(seed)
        sharpe = float(rng.normal(0.5, 0.4))
        profit = float(rng.normal(2000.0, 1000.0))
        dd = float(abs(rng.normal(0.15, 0.05)))
        return {"Sharpe": sharpe, "Profit": profit, "Drawdown": dd}

    # נתחיל מהעתק שניתן ללכלך
    p: Dict[str, Any] = dict(params or {})

    # --- שליפת פרמטרים בסיסיים מתוך params (וגם שמות חלופיים) ---
    try:
        z_entry = float(
            p.pop("z_entry", p.pop("z_open", 1.0))
        )
        z_exit = float(
            p.pop("z_exit", p.pop("z_close", 0.5))
        )
        lookback = int(
            p.pop("lookback", p.pop("rolling_window", 60))
        )
    except Exception as e:
        logger.warning("run_backtest: failed to extract z_entry/z_exit/lookback from params=%s (%s)", params, e)
        # fallback קשיח – עדיף על NameError
        z_entry = 1.0
        z_exit = 0.5
        lookback = 60

    notional = float(p.pop("notional", 50_000.0))
    slippage_bps_param = float(p.pop("slippage_bps", 2.0))
    transaction_cost = float(p.pop("transaction_cost_per_trade", 3.0))

    # --- Risk kwargs מה-session (אם יש Streamlit) ---
    risk_kwargs: Dict[str, Any] = {}
    try:
        risk_kwargs = dict(get_session_risk_kwargs())
    except Exception:
        risk_kwargs = {}

    # לוודא שלא מעבירים פעמיים
    for key in ("slippage_bps", "transaction_cost_per_trade", "notional"):
        risk_kwargs.pop(key, None)

    # בשלב הזה אנחנו לא מעבירים את p ל-Backtester, כדי לא לקבל TypeError על פרמטרים לא מוכרים.
    # אם בעתיד תרצה למפות עוד פרמטרים → תוסיף אותם כאן מפורשות.

    try:
        bt = Backtester(
            symbol_a=sym1,
            symbol_b=sym2,
            z_entry=z_entry,
            z_exit=z_exit,
            lookback=lookback,
            notional=notional,
            slippage_bps=slippage_bps_param,
            transaction_cost_per_trade=transaction_cost,
            **risk_kwargs,
        )
        perf = bt.run()
        return extract_metrics(perf)
    except Exception as e:
        # חשוב: לא להתייחס ל-z_entry/z_exit פה בשם (למנוע NameError), רק להדפיס params המקוריים
        logger.warning(
            "Backtest error for %s-%s with params=%s: %s",
            sym1,
            sym2,
            params,
            e,
        )
        # penalty profile קבוע (רע מאוד אבל לא NaN)
        return {"Sharpe": -5.0, "Profit": -1e6, "Drawdown": 1.0}


# =========================
# SECTION 4.3: Multi-Objective spec helper
# =========================

def _build_multi_objective_spec(
    objective_metrics: Optional[List[str]] = None,
) -> Tuple[List[str], List[str]]:
    """
    בונה (metrics, directions) עבור Multi-Objective Optuna.

    לדוגמה:
        metrics    = ["Sharpe", "Profit", "Drawdown"]
        directions = ["maximize", "maximize", "minimize"]

    אם objective_metrics=None:
        ברירת מחדל: ["Sharpe", "Profit", "Drawdown"].
    """
    if not objective_metrics:
        objective_metrics = ["Sharpe", "Profit", "Drawdown"]

    metrics_clean: List[str] = []
    directions: List[str] = []

    for m in objective_metrics:
        mk = str(m).strip()
        if not mk:
            continue
        # נקבע כיוון דיפולט: DD → minimize, אחרת maximize
        low = mk.lower()
        if low in ("drawdown", "maxdd", "max_drawdown"):
            dir_m = "minimize"
            canonical = "Drawdown"
        elif low in ("sharpe", "sharpe_ratio"):
            dir_m = "maximize"
            canonical = "Sharpe"
        elif low in ("profit", "pnl", "pnl_usd", "return", "ret"):
            dir_m = "maximize"
            canonical = "Profit"
        else:
            # פרמטרים אחרים — ברירת מחדל maximize
            dir_m = "maximize"
            canonical = mk
        metrics_clean.append(canonical)
        directions.append(dir_m)

    return metrics_clean, directions


# =========================
# SECTION 4.4: Objective factory (Single/Multi)
# =========================

def _objective_factory(
    sym1: str,
    sym2: str,
    ranges: Dict[str, ParamRange],
    weights: Dict[str, float],
    *,
    param_mapping: Optional[Dict[str, str]] = None,
    direction: str = "maximize",
    profile: str = "default",
    mode: str = "single",  # "single" או "multi"
    objective_metrics: Optional[List[str]] = None,
    param_specs: Optional[List[Any]] = None,
) -> Any:
    """
    בונה פונקציית objective עבור Optuna study.

    מצב "single":
    -------------
        - דומה ל-Single Objective: מחזיר Score יחיד.
        - direction: "maximize"/"minimize" → הופך סימן ב-return.

    מצב "multi":
    ------------
        - מחזיר vector (tuple) של מטרות, למשל:
            (Sharpe, Profit, Drawdown)
        - הכיוונים (maximize/minimize) נבחרים ב-Optuna דרך directions=[...],
          לפי _build_multi_objective_spec(objective_metrics).

        - עדיין מחושב Score משוקלל אחד לצרכי דיאגנוסטיקה/דוחות.

    כל Trial שומר ב-user_attrs:
        - perf, params, norm, profile, mode, direction, metrics, duration_sec, וכו'.
    """
    mode = str(mode).lower().strip()
    is_multi = mode.startswith("multi")
    direction = str(direction).lower().strip()
    is_minimize = direction.startswith("min")

    metrics_for_multi, directions_multi = _build_multi_objective_spec(objective_metrics) if is_multi else ([], [])

    def _objective(trial: "optuna.trial.Trial") -> Any:  # type: ignore[name-defined]
        t_start = time.time()
        rng = _make_trial_rng(trial)

        # --------------------------------------------------
        # 0) קריאת הקשר Walk-Forward מה-session (אם קיים)
        # --------------------------------------------------
        wf_cfg = st.session_state.get("opt_run_cfg", {}) or {}
        wf_use = bool(wf_cfg.get("wf_use", False))
        wf_folds = int(wf_cfg.get("wf_folds", 0) or 0)
        robust_min_folds = int(wf_cfg.get("robust_min_folds", max(1, wf_folds))) if wf_folds else 0
        robust_min_sharpe = float(wf_cfg.get("robust_min_sharpe", 0.3))
        # תאריכים מתוך ctx (שמור בדשבורד)
        ctx_dict = st.session_state.get("ctx", {}) or {}
        sd_ctx = ctx_dict.get("start_date")
        ed_ctx = ctx_dict.get("end_date")

        # ננסה להמיר ל-date (אם זה Timestamp/str וכד')
        from datetime import date as _date_type  # local import כדי לא לשבור מקום אחר
        try:
            if sd_ctx is not None and not isinstance(sd_ctx, _date_type):
                sd_ctx = pd.to_datetime(sd_ctx).date()
            if ed_ctx is not None and not isinstance(ed_ctx, _date_type):
                ed_ctx = pd.to_datetime(ed_ctx).date()
        except Exception:
            sd_ctx = ed_ctx = None

        use_wf_this_trial = (
            wf_use
            and wf_folds >= 2
            and Backtester is not None
            and sd_ctx is not None
            and ed_ctx is not None
        )

        # --------------------------------------------------
        # 1) דגימת params
        # --------------------------------------------------
        params = _sample_params(trial, ranges, rng=rng, param_specs=param_specs)

        # מאפשר override mapping שונה לזה שב-session
        if param_mapping is not None:
            params_mapped = _apply_param_mapping(params, param_mapping)
        else:
            params_mapped = params

        # --------------------------------------------------
        # 2) Backtest / Walk-Forward
        # --------------------------------------------------
        used_wf = False
        wf_summary: Dict[str, Any] = {}

        if use_wf_this_trial:
            try:
                # מריץ WF מלא לפרמטרים הנוכחיים
                wf_df = _run_walkforward_for_params(
                    sym1,
                    sym2,
                    params_mapped,
                    start_date=sd_ctx,    # type: ignore[arg-type]
                    end_date=ed_ctx,      # type: ignore[arg-type]
                    n_splits=int(wf_folds),
                    weights=weights,
                )
                if wf_df is not None and not wf_df.empty:
                    used_wf = True

                    # אגרגציה בסיסית: ממוצע Sharpe/Profit, Drawdown הכי גרוע
                    sh_series = pd.to_numeric(wf_df.get("Sharpe", pd.Series(index=wf_df.index)), errors="coerce")
                    pf_series = pd.to_numeric(wf_df.get("Profit", pd.Series(index=wf_df.index)), errors="coerce")
                    dd_series = pd.to_numeric(wf_df.get("Drawdown", pd.Series(index=wf_df.index)), errors="coerce").abs()

                    perf_wf = {
                        "Sharpe": float(sh_series.mean(skipna=True)),
                        "Profit": float(pf_series.mean(skipna=True)),
                        "Drawdown": float(dd_series.max(skipna=True)),
                    }

                    # מדד רובסטיות: כמה folds עוברים מסך Sharpe
                    ok_mask = sh_series >= float(robust_min_sharpe)
                    n_ok = int(ok_mask.sum())
                    total_folds = int(wf_df.shape[0])
                    if total_folds <= 0:
                        robustness_penalty = 0.0
                    else:
                        if n_ok >= robust_min_folds:
                            robustness_penalty = 1.0
                        else:
                            # penalty לינארי פשוט – לא פחות מ-0.2
                            robustness_penalty = max(0.2, n_ok / float(total_folds))

                    # Score לפי perf המאוחד + penalty לרובסטיות
                    norm, score = compute_score(perf_wf, weights)
                    score *= float(robustness_penalty)

                    perf = perf_wf
                    wf_summary = {
                        "wf_folds": total_folds,
                        "wf_n_ok": n_ok,
                        "wf_min_sharpe": float(sh_series.min(skipna=True)),
                        "wf_mean_sharpe": float(sh_series.mean(skipna=True)),
                        "wf_robust_penalty": float(robustness_penalty),
                    }
                else:
                    # WF החזיר ריק – ניפול ל-backtest רגיל
                    used_wf = False
            except Exception as e:
                logger.debug("Walk-Forward inside objective failed; falling back to single-window: %s", e)
                used_wf = False

        if not used_wf:
            # Backtest רגיל על כל הטווח
            perf = run_backtest(sym1, sym2, params_mapped)
            norm, score = compute_score(perf, weights)

        # --------------------------------------------------
        # 3) final_score לסינגל-אובג'קטיב
        # --------------------------------------------------
        if not is_multi:
            final_score = float(-score if is_minimize else score)
        else:
            final_score = float(score)  # רק לדיאגנוסטיקה ב-user_attrs

        t_end = time.time()

        # --------------------------------------------------
        # 4) Multi-objective vector (אם יש)
        # --------------------------------------------------
        if is_multi:
            m = extract_metrics(perf)
            vec = [float(m.get(k, 0.0)) for k in metrics_for_multi]

            # אם השתמשנו ב-WF – נבלע את ה-robust_penalty גם על הווקטור עצמו
            if used_wf and wf_summary.get("wf_robust_penalty") is not None:
                rp = float(wf_summary["wf_robust_penalty"])
                vec = [rp * v for v in vec]

            objective_value: Any = tuple(vec)
        else:
            objective_value = final_score

        # --------------------------------------------------
        # 5) user_attrs (גם עבור Multi וגם Single)
        # --------------------------------------------------
        trial_meta = {
            "perf": perf,
            "params": params,
            "norm": norm,
            "profile": profile,
            "direction": direction,
            "mode": "multi" if is_multi else "single",
            "metrics": metrics_for_multi if is_multi else ["Score"],
            "multi_directions": directions_multi if is_multi else [direction],
            # זמנים timezone-aware ב-UTC
            "datetime_start": datetime.fromtimestamp(t_start, tz=timezone.utc).isoformat(),
            "datetime_complete": datetime.fromtimestamp(t_end, tz=timezone.utc).isoformat(),
            "duration_sec": float(t_end - t_start),
            "score_single": float(score),
            "wf_used": bool(used_wf),
            **wf_summary,
        }

        try:
            for k, v in trial_meta.items():
                trial.set_user_attr(k, v)
        except Exception:
            pass

        # 6) Report for pruners (רק על Score יחיד, אם יש רלוונטיות)
        try:
            trial.report(float(score), step=0)
        except Exception:
            pass

        return objective_value

    return _objective


# =========================
# SECTION 4.5: Optuna runner (Single/Multi, ParamSpec-aware, fallback)
# =========================

def run_optuna_for_pair(
    sym1: str,
    sym2: str,
    *,
    ranges: Dict[str, ParamRange],
    weights: Dict[str, float],
    n_trials: int,
    timeout_min: int,
    direction: str,
    sampler_name: str,
    pruner_name: str,
    param_mapping: Optional[Dict[str, str]],
    profile: str,
    multi_objective: bool,
    objective_metrics: Optional[List[str]],
    param_specs: Optional[List[Any]] = None,
    study_name: Optional[str] = None,
    storage_url: Optional[str] = None,
    zoom_meta: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Optional[int]]:

    """
    מנוע אופטימיזציה עבור זוג אחד.

    פרמטרים מרכזיים:
    ------------------
    sym1, sym2            — הסימבולים של הזוג.
    ranges                — טווחי פרמטרים (lo, hi, step).
    weights               — משקולות מטריקות ל-Score (Sharpe/Profit/DD וכו').
    n_trials              — מספר ניסיונות מקסימלי.
    timeout_min           — מגבלת זמן (בדקות).
    direction             — "maximize"/"minimize" (ל-Single Objective).
    sampler_name          — "TPE" או "CMAES" (או כל מחרוזת; CMA-ES מזוהה דרך _is_cmaes).
    pruner_name           — "median" או "none".
    param_mapping         — UI → Backtester mapping (לא חובה).
    profile               — פרופיל ריצה (defensive/aggressive/...).
    multi_objective       — האם להפעיל Multi-Objective.
    objective_metrics     — רשימת מטריקות ל-MO (למשל ["Sharpe","Profit","Drawdown"]).
    param_specs           — רשימת ParamSpec מלאה (לשימוש ב-_sample_params fallback).

    התנהגות:
    ----------
    אם Optuna זמין:
        - Single Objective:
            create_study(direction="maximize") — sign flipping לפי direction.
        - Multi-Objective:
            create_study(directions=[...]) לפי objective_metrics.

    אם Optuna לא זמין:
        - Simulated search (עם ParamSpec אם אפשר), עדיין שומר ל-DuckDB.

    מחזיר:
        df_sorted, study_id
    """
    n_trials = int(max(1, n_trials))
    timeout_sec = max(0, int(timeout_min * 60))
    direction = str(direction).lower().strip()
    sym1 = str(sym1).strip()
    sym2 = str(sym2).strip()
    pair_label = f"{sym1}-{sym2}"

    is_minimize = direction.startswith("min")
    mode = "multi" if multi_objective else "single"
    metrics_mo, directions_mo = _build_multi_objective_spec(objective_metrics) if multi_objective else ([], [])

    logger.info(
        "Starting optimisation for %s with n_trials=%d timeout=%ds sampler=%s pruner=%s profile=%s mode=%s",
        pair_label,
        n_trials,
        timeout_sec,
        sampler_name,
        pruner_name,
        profile,
        mode,
    )

    # Hash של מרחב הפרמטרים לצורך שחזור
    pspace_hash = paramspace_hash(ranges)

    # ====== מצב ללא Optuna (fallback) ======
    if optuna is None:
        logger.warning("Optuna not installed; running simulated optimisation for %s", pair_label)
        rng = np.random.default_rng(int(st.session_state.get("global_seed", 1337)))
        records: List[Dict[str, Any]] = []

        for _ in range(max(n_trials, 10)):
            params = _sample_params(trial=None, ranges=ranges, rng=rng, param_specs=param_specs)
            perf = run_backtest(sym1, sym2, params)
            norm, score = compute_score(perf, weights)

            row: Dict[str, Any] = {**params, **perf, "Score": score}

            # 🔴 NEW — ParamScore לפי core.params
            if params_score_params_dict is not None:
                try:
                    ps = params_score_params_dict(params)
                    row["ParamScore"] = float(ps.get("total_score", 0.0))
                    row["ParamScore_raw"] = float(ps.get("raw_score", 0.0))
                    row["ParamScore_weight"] = float(ps.get("total_weight", 0.0))
                except Exception as e:
                    logger.debug("ParamScore (simulated) failed: %s", e)

            records.append(row)


        df = pd.DataFrame.from_records(records)
        study_id = save_trials_to_duck(
            df,
            pair_label,
            sampler="SIMULATED",
            n_trials=n_trials,
            timeout_sec=timeout_sec,
            weights=weights,
            extra_meta={
                "paramspace_hash": pspace_hash,
                "direction": direction,
                "profile": profile,
                "mode": mode,
                "app_version": OPT_TAB_VERSION,
                "optuna_version": _safe_version("optuna"),
            },
        )
        df_sorted = df.sort_values("Score", ascending=is_minimize)
        return df_sorted, study_id

    # ====== Optuna path ======
    # Sampler
    if _is_cmaes(sampler_name) and CmaEsSampler is not object:
        sampler_obj = CmaEsSampler(seed=int(st.session_state.get("global_seed", 1337)))
    else:
        sampler_obj = TPESampler(seed=int(st.session_state.get("global_seed", 1337))) if TPESampler is not object else None

    # Pruner
    if str(pruner_name).lower().startswith("median") and MedianPruner is not object:
        pruner_obj = MedianPruner(n_startup_trials=min(10, max(n_trials // 5, 1)))
    else:
        pruner_obj = None

    # Study creation: single vs multi
    # ---------------------------------------------------
    # 1) כיוון (single / multi objective)
    if multi_objective:
        directions = directions_mo or ["maximize"]
        study_kwargs: Dict[str, Any] = {
            "directions": directions,
        }
    else:
        study_kwargs = {
            "direction": "maximize",  # sign flipping נעשה ב-objective לפי direction
        }

    if sampler_obj is not None:
        study_kwargs["sampler"] = sampler_obj
    if pruner_obj is not None:
        study_kwargs["pruner"] = pruner_obj

    # 2) בחירת storage URL:
    #    - אם פונקציה חיצונית (zoom) העבירה storage_url → זה שולט.
    #    - אחרת נשתמש ב-OPTUNA_STORAGE מהסביבה (אם קיים).
    #    - ואם גם זה לא קיים → נשתמש בזום-storage הדיפולטי (shared Optuna DB).
    if storage_url:
        effective_storage_url = storage_url
    else:
        env_storage_url = os.getenv("OPTUNA_STORAGE", "")
        if env_storage_url:
            effective_storage_url = env_storage_url
        else:
            try:
                zoom_cfg = resolve_zoom_storage(PROJECT_ROOT)
                effective_storage_url = zoom_cfg.storage_url or ""
            except Exception:
                effective_storage_url = ""

    # 3) בחירת study_name:
    #    - אם Zoom נתן לנו study_name → משתמשים בו (zoom::<PAIR>::stageX).
    #    - אחרת נבנה שם דיפולטי.
    if study_name is None:
        # pair_label כבר קיים למעלה בפונקציה (למשל "BITO-BKCH")
        base_name = f"{pair_label}-{profile}"
        if effective_storage_url:
            # עם טיימסטמפ כדי להימנע מהתנגשויות
            study_name = f"{base_name}-{int(time.time())}"
        else:
            # כשאין storage חיצוני, השם פחות קריטי
            study_name = base_name

    # 4) יצירת ה-Study
    if effective_storage_url:
        study = optuna.create_study(
            study_name=study_name,
            storage=effective_storage_url,
            load_if_exists=True,  # חשוב ל-Zoom: לאחד stages אם צריך
            **study_kwargs,  # type: ignore[arg-type]
        )
    else:
        study = optuna.create_study(**study_kwargs)  # type: ignore[arg-type]

    # 5) meta נוח ל-export / debug
    try:
        study.set_user_attr("sym1", sym1)
        study.set_user_attr("sym2", sym2)
        study.set_user_attr("pair_label", pair_label)
        study.set_user_attr("profile", profile)
        study.set_user_attr("mode", mode)
        if zoom_meta:
            for k, v in zoom_meta.items():
                # לא נדרוס ערכים קיימים אם כבר יש כאלה
                if k not in study.user_attrs:
                    study.set_user_attr(k, v)
    except Exception:
        # לא קריטי להפיל ריצה על meta
        logger.debug("run_optuna_for_pair: failed to set user_attrs on study", exc_info=True)

    # Objective
    objective = _objective_factory(
        sym1,
        sym2,
        ranges,
        weights,
        param_mapping=param_mapping,
        direction=direction,
        profile=profile,
        mode=mode,
        objective_metrics=metrics_mo,
        param_specs=param_specs,
    )

    # Run
    t0 = time.time()
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout_sec if timeout_sec > 0 else None,
    )
    duration_sec = time.time() - t0

    # איסוף תוצאות
    rows: List[Dict[str, Any]] = []
    for t in study.trials:
        ua = getattr(t, "user_attrs", {}) or {}
        perf = ua.get("perf", {})
        params = ua.get("params", {})
        norm = ua.get("norm", {})
        score_single = ua.get("score_single", None)

        row: Dict[str, Any] = {**params, **perf}
        if score_single is not None:
            row["Score"] = float(score_single)
        else:
            row["Score"] = float(t.value) if not multi_objective else float("nan")

        # 🔴 NEW — ParamScore לכל סט פרמטרים
        if params_score_params_dict is not None and isinstance(params, dict):
            try:
                ps = params_score_params_dict(params)
                row["ParamScore"] = float(ps.get("total_score", 0.0))
                row["ParamScore_raw"] = float(ps.get("raw_score", 0.0))
                row["ParamScore_weight"] = float(ps.get("total_weight", 0.0))
            except Exception as e:
                logger.debug("ParamScore (Optuna) failed: %s", e)

        for k, v in (norm or {}).items():
            row[f"norm_{k}"] = v

        dur = ua.get("duration_sec")
        if dur is not None:
            row["trial_duration_sec"] = float(dur)

        rows.append(row)

    df = pd.DataFrame(rows)

    # שמירה ל-DuckDB עם metadata עשיר
    try:
        study_id = save_trials_to_duck(
            df,
            pair_label,
            sampler=str(sampler_name),
            n_trials=n_trials,
            timeout_sec=timeout_sec,
            weights=weights,
            extra_meta={
                "paramspace_hash": pspace_hash,
                "direction": direction,
                "profile": profile,
                "mode": mode,
                "duration_sec": duration_sec,
                "optuna_version": _safe_version("optuna"),
                "app_version": OPT_TAB_VERSION,
                "sampler_params": str(sampler_obj),
                "storage": effective_storage_url,
            },
        )

    except Exception as e:
        logger.warning("save_trials_to_duck from Optuna study failed: %s", e)
        study_id = None

    # למיון UI (גם אם Multi) נשתמש ב-Score יחיד (score_single)
    df_sorted = df.sort_values("Score", ascending=is_minimize)
    logger.info(
        "Finished optimisation for %s: trials=%d, duration=%.1fs, study_id=%s, mode=%s",
        pair_label,
        len(df),
        duration_sec,
        study_id,
        mode,
    )
    return df_sorted, study_id

"""
חלק 6/15 — Notifications, Telemetry & Batch Orchestrator
=========================================================

מה כלול כאן:

1. _notify_slack(msg) / _notify_telegram(msg) / _notify_all(msg)
   - שימוש ב-SETTINGS.slack_webhook / SETTINGS.telegram_token / SETTINGS.telegram_chat_id
   - הודעות best-effort בלבד (לא מפילות את הטאב).

2. _build_opt_manifest(...) — manifest JSON עבור study יחיד:
   - pair, n_trials, profile, mode, directions, paramspace_hash,
     best_score, rows, timestamps, גרסאות וכו'.

3. optimize_pair_with_telemetry(...)
   - עטיפה מעל run_optuna_for_pair:
       * לוגים מסודרים.
       * שמירת manifest כ-artifact ב-DuckDB.
       * שליחת Notification (אופציונלי).
       * החזרת (df, study_id, manifest_dict).

4. optimize_pairs_batch(...)
   - Batch Orchestrator:
       * מקבל רשימת pairs.
       * מריץ אופטימיזציה על כל זוג (עם אותו ranges/weights או שונים).
       * מציג progress (דרך Streamlit) + summary טבלאי.
       * שומר df מאוחד ב-session_state["opt_df_batch"].
"""

# =========================
# SECTION 5: Notifications (Slack / Telegram)
# =========================

def _notify_slack(msg: str) -> None:
    """
    Notification ל-Slack (best-effort בלבד).

    משתמש ב-SETTINGS.slack_webhook (אם קיים). אם לא → עושה כלום.

    הערה: נועד רק לדברים "גדולים" (סיום study / כשל חמור), לא לספאם.
    """
    url = getattr(SETTINGS, "slack_webhook", None)
    if not url:
        return
    try:
        import urllib.request
        body = json.dumps({"text": msg}, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=5)  # nosec
    except Exception as e:
        logger.debug("Slack notification failed: %s", e)


def _notify_telegram(msg: str) -> None:
    """
    Notification לטלגרם (best-effort).

    משתמש ב-SETTINGS.telegram_token / SETTINGS.telegram_chat_id.
    אם חלק מהשניים חסר → עושה כלום.
    """
    tok = getattr(SETTINGS, "telegram_token", None)
    chat = getattr(SETTINGS, "telegram_chat_id", None)
    if not tok or not chat:
        return
    try:
        import urllib.parse
        import urllib.request

        api = f"https://api.telegram.org/bot{tok}/sendMessage"
        data = urllib.parse.urlencode(
            {
                "chat_id": chat,
                "text": msg,
                "disable_web_page_preview": 1,
            }
        ).encode("utf-8")
        urllib.request.urlopen(api, data=data, timeout=5)  # nosec
    except Exception as e:
        logger.debug("Telegram notification failed: %s", e)


def _notify_all(msg: str) -> None:
    """
    שולח הודעה ל-Slack וטלגרם (אם מוגדרים).
    """
    _notify_slack(msg)
    _notify_telegram(msg)


# =========================
# SECTION 5.1: Manifest builder for a single optimisation run
# =========================

def _build_opt_manifest(
    pair: str,
    df: pd.DataFrame,
    study_id: Optional[int],
    *,
    ranges: Dict[str, ParamRange],
    weights: Dict[str, float],
    profile: str,
    mode: str,
    direction: str,
    paramspace_hash_value: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    בונה manifest JSON עבור study יחיד.

    כולל:
        - pair, study_id, ts, app/env info
        - מרחב פרמטרים (hash + טווחים)
        - משקולות מטריקות
        - stats: rows, best_score, avg_sharpe, max_profit, min_dd
        - metadata חופשי מ-extra
    """
    dt = datetime.now(timezone.utc)
    now = dt.isoformat().replace("+00:00", "Z")
    m: Dict[str, Any] = {
        "pair": pair,
        "study_id": study_id,
        "timestamp": now,
        "app_version": OPT_TAB_VERSION,
        "env": getattr(SETTINGS, "env", "local"),
        "direction": direction,
        "mode": mode,
        "profile": profile,
        "paramspace_hash": paramspace_hash_value or paramspace_hash(ranges),
        "ranges": make_json_safe(ranges),
        "weights": make_json_safe(weights),
        "rows": int(df.shape[0]) if df is not None else 0,
        "columns": list(map(str, df.columns)) if df is not None else [],
        "metrics": {},
    }

    # KPIs מהירים
    try:
        if df is not None and not df.empty:
            m["metrics"]["best_score"] = float(pd.to_numeric(df.get("Score", pd.Series()), errors="coerce").max())
            if "Sharpe" in df.columns:
                m["metrics"]["avg_sharpe"] = float(pd.to_numeric(df["Sharpe"], errors="coerce").mean())
            if "Profit" in df.columns:
                m["metrics"]["max_profit"] = float(pd.to_numeric(df["Profit"], errors="coerce").max())
            if "Drawdown" in df.columns:
                m["metrics"]["min_drawdown"] = float(pd.to_numeric(df["Drawdown"], errors="coerce").min())
    except Exception as e:
        logger.debug("Manifest metrics build failed: %s", e)

    if extra:
        try:
            m.update(extra)
        except Exception:
            pass

    return m


# =========================
# SECTION 5.2: Single-pair optimisation with telemetry & artifact
# =========================

def optimize_pair_with_telemetry(
    sym1: str,
    sym2: str,
    *,
    ranges: Dict[str, ParamRange],
    weights: Dict[str, float],
    n_trials: int = 50,
    timeout_min: int = 10,
    direction: str = "maximize",
    sampler_name: str = "TPE",
    pruner_name: str = "median",
    param_mapping: Optional[Dict[str, str]] = None,
    profile: str = "default",
    multi_objective: bool = False,
    objective_metrics: Optional[List[str]] = None,
    param_specs: Optional[List[Any]] = None,
    notify: bool = True,
) -> Tuple[pd.DataFrame, Optional[int], Dict[str, Any]]:
    """
    Wrapper מעל run_optuna_for_pair עם Telemetry ו-artifacts:

    - מריץ run_optuna_for_pair עם כל הפרמטרים.
    - בונה manifest (dict) עם פרטים ו-KPI-ים.
    - שומר את manifest כ-artifact ב-DuckDB (kind="manifest_json").
    - שולח Notification ל-Slack/Telegram (אם notify=True והוגדרו webhooks).
    - מחזיר (df_sorted, study_id, manifest_dict).
    """
    pair_label = f"{sym1}-{sym2}"
    mode = "multi" if multi_objective else "single"

    # 1) הרצת האופטימיזציה בפועל
    df_sorted, study_id = run_optuna_for_pair(
        sym1,
        sym2,
        ranges=ranges,
        weights=weights,
        n_trials=n_trials,
        timeout_min=timeout_min,
        direction=direction,
        sampler_name=sampler_name,
        pruner_name=pruner_name,
        param_mapping=param_mapping,
        profile=profile,
        multi_objective=multi_objective,
        objective_metrics=objective_metrics,
        param_specs=param_specs,
    )

    # 2) בניית manifest (מורחב)
    opt_run_cfg_snapshot = st.session_state.get("opt_run_cfg", {}) or {}
    opt_config_snapshot = st.session_state.get("opt_config", {}) or {}

    manifest = _build_opt_manifest(
        pair=pair_label,
        df=df_sorted,
        study_id=study_id,
        ranges=ranges,
        weights=weights,
        profile=profile,
        mode=mode,
        direction=direction,
        paramspace_hash_value=None,
        extra={
            "sampler_name": sampler_name,
            "pruner_name": pruner_name,
            "n_trials": int(n_trials),
            "timeout_min": int(timeout_min),
            "multi_objective": bool(multi_objective),
            "objective_metrics": objective_metrics or [],
            # תוספות רפרודוקציה:
            "env": getattr(SETTINGS, "env", "local"),
            "global_seed": st.session_state.get("global_seed"),
            "opt_run_cfg": make_json_safe(opt_run_cfg_snapshot),
            "opt_config": make_json_safe(opt_config_snapshot),
        },
    )

    # 3) שמירה כ-artifact ב-DuckDB (manifest + full results)
    try:
        if study_id is not None:
            payload_manifest = json.dumps(
                make_json_safe(manifest),
                ensure_ascii=False,
                indent=2,
            ).encode("utf-8")
            save_artifact_to_duck(int(study_id), kind="manifest_json", payload=payload_manifest)

            # ארטיפקט עם כל תוצאות האופטימיזציה (CSV)
            if df_sorted is not None and not df_sorted.empty:
                payload_csv = df_sorted.to_csv(index=False).encode("utf-8")
                save_artifact_to_duck(int(study_id), kind="results_csv", payload=payload_csv)

            # אופציונלי: snapshot של opt_config כ-JSON נפרד
            if opt_config_snapshot:
                payload_cfg = json.dumps(
                    make_json_safe(opt_config_snapshot),
                    ensure_ascii=False,
                    indent=2,
                ).encode("utf-8")
                save_artifact_to_duck(int(study_id), kind="opt_config_json", payload=payload_cfg)
    except Exception as e:
        logger.warning("Saving optimisation artifacts failed for %s: %s", pair_label, e)

    # 3.5) Hook ל-SqlStore (כרגע no-op, אבל API יציב)
    try:
        persist_opt_run_to_sqlstore(pair_label, df_sorted, manifest)
    except Exception as e:
        logger.debug("persist_opt_run_to_sqlstore failed for %s: %s", pair_label, e)

    # 4) Notifications (best-effort בלבד)
    if notify and study_id is not None:
        try:
            best_score = manifest.get("metrics", {}).get("best_score", None)
            msg = f"✅ Optimisation done for {pair_label} (study_id={study_id}, mode={mode}, profile={profile})"
            if best_score is not None:
                msg += f" — best Score={best_score:.3f}"
            _notify_all(msg)
        except Exception:
            pass

    return df_sorted, study_id, manifest


# =========================
# SECTION 5.3: Batch Orchestrator (multi-pair)
# =========================

def optimize_pairs_batch(
    pairs: List[Tuple[str, str]],
    *,
    ranges: Optional[Dict[str, ParamRange]] = None,
    weights: Optional[Dict[str, float]] = None,
    n_trials: int = 50,
    timeout_min: int = 10,
    direction: str = "maximize",
    sampler_name: str = "TPE",
    pruner_name: str = "median",
    profile: str = "default",
    multi_objective: bool = False,
    objective_metrics: Optional[List[str]] = None,
    param_specs: Optional[List[Any]] = None,
    notify_each: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """
    Batch Orchestrator — מריץ אופטימיזציה על מספר זוגות ברצף.

    pairs:
        רשימה של (sym1, sym2) — לדוגמה:
            [("XLY","XLP"), ("QQQ","IWM"), ...]

    ranges, weights:
        אם לא נמסרים → יילקחו מ-get_default_param_ranges() ומברירת המחדל של ה-sidebar.

    התנהגות:
    ----------
    - עבור כל זוג:
        * מריץ optimize_pair_with_telemetry(...).
        * שומר את df_results עם עמודת Pair (label).
        * שומר את manifest במילון per-pair.
    - מציג progress דרך Streamlit (אם רצים בתוך הטאב).
    - מחזיר:
        * df_all — איחוד כל התוצאות, עם עמודת "Pair".
        * manifests — dict: pair_label → manifest_dict.

    כמו כן:
    - שומר df_all ב-st.session_state["opt_df_batch"] לצורך המשך ניתוח.
    """
    if not pairs:
        return pd.DataFrame(), {}

    # הכנת ranges/weights אם לא נמסרו
    if ranges is None:
        ranges = get_default_param_ranges(profile=profile)
    if weights is None:
        weights = st.session_state.get("loaded_weights_eff", {"Sharpe": 0.5, "Profit": 0.3, "Drawdown": 0.2})

    progress_bar = None
    status_placeholder = None
    try:
        progress_bar = st.progress(0.0)
        status_placeholder = st.empty()
    except Exception:
        # אם לא רצים בתוך Streamlit — מתעלמים
        pass

    manifests: Dict[str, Dict[str, Any]] = {}
    results: List[pd.DataFrame] = []

    total = len(pairs)
    for idx, (sym1, sym2) in enumerate(pairs, start=1):
        pair_label = f"{sym1}-{sym2}"

        if status_placeholder is not None:
            status_placeholder.info(
                f"Running optimisation {idx}/{total} for pair {pair_label} "
                f"(trials={n_trials}, timeout={timeout_min}m, profile={profile})"
            )

        try:
            df_pair, study_id, manifest = optimize_pair_with_telemetry(
                sym1,
                sym2,
                ranges=ranges,
                weights=weights,
                n_trials=n_trials,
                timeout_min=timeout_min,
                direction=direction,
                sampler_name=sampler_name,
                pruner_name=pruner_name,
                param_mapping=st.session_state.get("opt_param_mapping"),
                profile=profile,
                multi_objective=multi_objective,
                objective_metrics=objective_metrics,
                param_specs=param_specs,
                notify=notify_each,
            )
            if df_pair is not None and not df_pair.empty:
                df_pair = df_pair.copy()
                df_pair["Pair"] = pair_label
                if study_id is not None:
                    df_pair["study_id"] = int(study_id)
                results.append(df_pair)
            manifests[pair_label] = manifest
        except Exception as e:
            logger.warning("Batch optimisation failed for %s: %s", pair_label, e)
            dt = datetime.now(timezone.utc)
            manifests[pair_label] = {
                "pair": pair_label,
                "error": str(e),
                "timestamp": dt.isoformat(timespec="seconds").replace("+00:00", "Z"),
            }


        if progress_bar is not None:
            try:
                progress_bar.progress(idx / float(total))
            except Exception:
                pass

    # איחוד תוצאות
    if results:
        df_all = pd.concat(results, ignore_index=True)
    else:
        df_all = pd.DataFrame()

    # שמירה ב-session_state כדי שטאבים אחרים (Overview / Analytics) יוכלו להשתמש
    st.session_state["opt_df_batch"] = df_all

    return df_all, manifests
"""
חלק 7/15 — UI Entrypoint, Sidebar Config & Core Results
=======================================================

מה כלול כאן:

1. _render_profile_sidebar() — פרופיל/קונפיג לוגי (log level, data_dir, studies_dir).
2. _sidebar_core_config() — טווחי פרמטרים + משקולות מטריקות ל-scoring.
3. render_optimization_tab(ctx: dict | None = None, **ctrl_opt) — טאב האופטימיזציה עצמו:
   - כותרת + System Health.
   - Sidebar: טווחים, משקולות, Run Controls (Single/Multi objective).
   - Run כלי: Single pair + Batch.
   - תצוגת תוצאות בסיסית (KPIs, Top-N).
   - קריאה לפאנלים מתקדמים (שיגיעו בחלקים 8–15) דרך wrappers.
"""

# =========================
# SECTION 6: Profile/sidebar controls for settings
# =========================

def _render_profile_sidebar() -> Optional["OptSettings"]:
    """
    פרופיל/קונפיג לתשתית האופטימיזציה דרך ה-sidebar.

    מאפשר:
    - לשנות log_level (DEBUG/INFO/...).
    - לשנות data_dir / studies_dir.
    - להציג קצת מידע סביבת הרצה.
    """
    with st.sidebar.expander("Profile / Settings", expanded=False):
        lvl = st.selectbox(
            "Log level",
            ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            index=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"].index(SETTINGS.log_level),
            key="opt_prof_log_level",
        )
        data_dir_str = st.text_input(
            "Data dir",
            value=str(SETTINGS.data_dir),
            key="opt_prof_data_dir",
        )
        studies_dir_str = st.text_input(
            "Studies dir",
            value=str(SETTINGS.studies_dir),
            key="opt_prof_studies_dir",
        )

        st.caption(f"Current env: {SETTINGS.env!r} | DB Path: {DB_PATH}")

        allow_live_update = st.checkbox(
            "Allow LIVE/PROD to update best_params registry",
            value=bool(st.session_state.get("opt_live_update_params", False)),
            key="opt_live_update_params",
            help=(
                "כאשר פעיל, גם בסביבת LIVE/PROD מותר לעדכן את האופטימיזציה האחרונה "
                "ל־opt_best_params_registry. השאר כבוי אם אתה רוצה שהפרמטרים בלייב יהיו סטטיים."
            ),
        )

        apply = st.button("Apply profile", key="opt_prof_apply")
        if apply:
            p_data = Path(data_dir_str)
            p_studies = Path(studies_dir_str)
            try:
                p_data.mkdir(parents=True, exist_ok=True)
                p_studies.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            # Important: משתמשים באותה מחלקת OptSettings כדי לשמור מבנה
            from typing import cast
            new_settings = cast(
                "OptSettings",
                type(SETTINGS)(
                    env=SETTINGS.env,
                    log_level=lvl,
                    data_dir=p_data,
                    studies_dir=p_studies,
                    slack_webhook=getattr(SETTINGS, "slack_webhook", None),
                    telegram_token=getattr(SETTINGS, "telegram_token", None),
                    telegram_chat_id=getattr(SETTINGS, "telegram_chat_id", None),
                    duck_threads=getattr(SETTINGS, "duck_threads", 4),
                    duck_memory_limit_mb=getattr(SETTINGS, "duck_memory_limit_mb", 0),
                    enable_heavy_panels=getattr(SETTINGS, "enable_heavy_panels", True),
                    enable_wandb=getattr(SETTINGS, "enable_wandb", False),
                    enable_mlflow=getattr(SETTINGS, "enable_mlflow", False),
                ),
            )
            return new_settings
    return None


# =========================
# SECTION 6.1: Sidebar core config (ranges, weights, profile)
# =========================

def _sidebar_core_config() -> Dict[str, Any]:
    """
    מחזיר הגדרות בסיס ל-Optimisation Tab:

    - ranges: dict[str, ParamRange]
    - weights: dict[str, float] (Sharpe/Profit/DD/...)
    - profile: str ("default"/"defensive"/"aggressive"/...)

    נשמרים גם ב-session_state:
        - active_ranges
        - loaded_weights_eff
        - opt_profile
    """
    global SETTINGS
    
    with st.sidebar.expander("Optimization Config", expanded=True):
        # בחירת פרופיל (משפיע על get_default_param_ranges)
        # אם opt_profile כבר קיים ב-session → משתמשים בו,
        # אחרת נגזור ברירת מחדל לפי מצב המאקרו/ריסק (defense/offense).
        profile_default = st.session_state.get("opt_profile", None)

        if not profile_default:
            # ננסה לקרוא פרופיל ריסק מהמאקרו / הום
            risk_prof = (
                st.session_state.get("risk_profile_from_macro")  # למשל "defense"/"offense"/"balanced"
                or st.session_state.get("home_risk_focus")       # "balanced"/"defense"/"offense"
            )
            if isinstance(risk_prof, str):
                rp = risk_prof.lower()
                if rp in ("defense", "defensive", "risk_off", "stress", "crisis"):
                    profile_default = "defensive"
                elif rp in ("offense", "offensive", "risk_on"):
                    profile_default = "aggressive"
                else:
                    profile_default = "default"
            else:
                profile_default = "default"

        profile = st.selectbox(
            "Profile",
            ["default", "defensive", "aggressive"],
            index=["default", "defensive", "aggressive"].index(str(profile_default)),
            key="opt_profile_select",
        )
        st.session_state["opt_profile"] = profile

        # טווחי ברירת מחדל מהמערכת
        base_ranges = get_default_param_ranges(profile=profile)

        # active_ranges (ניתן לעדכן בהמשך דרך פאנלים מתקדמים)
        active_ranges = st.session_state.get("active_ranges")
        if not active_ranges:
            st.session_state["active_ranges"] = {
                name: (float(lo), float(hi), step)
                for name, (lo, hi, step) in base_ranges.items()
            }
            active_ranges = st.session_state["active_ranges"]

        ranges: Dict[str, ParamRange] = active_ranges

        # משקולות מטריקות — נשמרות ב-session כ-loaded_weights_eff
        try:
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            w_sh = c1.number_input("W Sharpe", value=0.4, step=0.05, key="opt_w_sharpe")
            w_pf = c2.number_input("W Profit", value=0.3, step=0.05, key="opt_w_profit")
            w_dd = c3.number_input("W Drawdown", value=0.2, step=0.05, key="opt_w_dd")
            w_sr = c4.number_input("W Sortino", value=0.05, step=0.05, key="opt_w_sortino")
            w_ca = c5.number_input("W Calmar", value=0.05, step=0.05, key="opt_w_calmar")
            w_wr = c6.number_input("W WinRate", value=0.0, step=0.05, key="opt_w_winrate")
            weights = {
                "Sharpe": float(w_sh),
                "Profit": float(w_pf),
                "Drawdown": float(w_dd),
                "Sortino": float(w_sr),
                "Calmar": float(w_ca),
                "WinRate": float(w_wr),
            }
        except Exception:
            weights = {"Sharpe": 0.5, "Profit": 0.3, "Drawdown": 0.2}

        st.session_state["loaded_weights_eff"] = weights

        # 🔹 אופציה לייצוב דגימת פרמטרים בעזרת risk-parity helper
        use_param_rp = st.checkbox(
            "נרמול דגימת פרמטרים (Risk-Parity helper)",
            value=bool(st.session_state.get("opt_use_param_rp", False)),
            key="opt_use_param_rp",
            help=(
                "כאשר פעיל – אחרי דגימת הפרמטרים תופעל התאמה של apply_risk_parity_to_params "
                "כדי למנוע פרמטר אחד דומיננטי מספרית."
            ),
        )

    return {
        "ranges": ranges,
        "weights": weights,
        "profile": profile,
    }



# =========================
# SECTION 6.2: Main UI entrypoint — render_optimization_tab
# =========================

def render_optimization_tab(
    ctx: dict | None = None,
    **ctrl_opt,
) -> None:
    """
    Main UI entrypoint לטאב האופטימיזציה – גרסה מורחבת ברמת קרן גידור.

    ctx מגיע מה-dashboard (AppContext.asdict), ויכול להכיל:
        - start_date, end_date
        - capital, pairs, config, controls, seed, run_id, profile, section

    ctrl_opt מגיע מטופס ה-controls בטאב הראשי (dashboard Tab 8),
    וכולל בין השאר:
        - n_trials, timeout_min, direction
        - param_ranges (טווחים לכל פרמטר)
        - experiment_name, opt_mode, meta_optimization_enabled וכו'.
    """

    # Run-control flags – תמיד יהיו מוגדרים כדי למנוע UnboundLocalError
    dry_run: bool = False
    run_single: bool = False
    run_batch: bool = False

    # ==== 0) Service + base context ====
    service = create_dashboard_service()
    base_ctx = build_default_dashboard_context()

    # ==== 0.0) Environment awareness (local/dev/paper/live/prod) ====
    effective_env = str(getattr(SETTINGS, "env", "local") or "local").lower()
    is_live_env = effective_env in {"live", "prod"}

    st.caption(f"Effective optimisation env: `{effective_env}`")

    # ==== 0.0) Focus-mode toggle (לפני הכל) ====
    st.sidebar.checkbox(
        "Focus mode (hide heavy analytics)",
        value=bool(st.session_state.get("opt_focus_mode", False)),
        key="opt_focus_mode",
        help="כאשר זה מופעל – פאנלים כבדים (Analytics/ML וכו') נשארים סגורים כברירת מחדל.",
    )
    FOCUS = bool(st.session_state.get("opt_focus_mode", False))

    # ==== 0.1) Header יפה למעלה ====
    render_optimization_header(base_ctx, service)
    st.markdown("---")

    # ==== 0.2) Presets bar לפני כל הקונטרולים ====
    preset_info = render_opt_presets(prefix="opt")
    current_preset = preset_info["preset"]

    if current_preset == "smoke":
        default_n_trials = 30
        default_timeout_min = 3
    elif current_preset == "deep":
        default_n_trials = 500
        default_timeout_min = 60
    elif current_preset == "tail":
        default_n_trials = 300
        default_timeout_min = 45
    elif current_preset == "fast":
        default_n_trials = 10
        default_timeout_min = 2
    else:  # custom
        default_n_trials = 200
        default_timeout_min = 30

    st.markdown("---")

    # ==== 0.3) Signal / Risk מספריים פשוטים מלמעלה (quality-of-life) ====
    with st.expander("⚙️ Quick signal / risk knobs (ParamSpec-based)", expanded=False):
        st.markdown("#### Signal parameters (Simple)")
        try:
            z_open = render_param_control("opt_sig", "z_open")
            z_close = render_param_control("opt_sig", "z_close")
        except Exception:
            # fallback במקרה ש-PARAM_SPECS לא כולל אותם
            z_open = st.number_input("Z-open threshold", -5.0, 5.0, 1.5, 0.1, key="opt_z_open_simple")
            z_close = st.number_input("Z-close threshold", -5.0, 5.0, 0.5, 0.1, key="opt_z_close_simple")

        st.markdown("#### Risk parameters")
        try:
            max_exposure = render_param_control("opt_risk", "max_exposure_per_trade")
        except Exception:
            max_exposure = st.number_input(
                "Max exposure per trade",
                min_value=0.0,
                max_value=1.0,
                value=0.1,
                step=0.01,
                key="opt_max_exposure_simple",
            )
        try:
            max_gross = render_param_control("opt_risk", "max_gross_exposure")
        except Exception:
            max_gross = st.number_input(
                "Max gross exposure",
                min_value=0.0,
                max_value=5.0,
                value=1.0,
                step=0.1,
                key="opt_max_gross_simple",
            )

        st.caption(
            "הערכים כאן הם יותר 'knobs ידניים' להשראה; האופטימיזציה בפועל משתמשת ב־param_ranges "
            "ובמרחב הפרמטרים המלא."
        )

    # ==== 0.4) היסטוריית ריצות בסוף הטאב (נרוץ אחר כך שוב) ====
    # נשאיר את הקריאה בסוף הפונקציה, אחרי שיש df, כדי שלא יסתיר דברים.
    # (שמנו כאן רק הערה – הקריאה האמיתית נשארת למטה.)

    # ==== 0.5) ctx ו-seed/run_id מתוך session/ctx חיצוני ====
    if ctx is None:
        ctx = st.session_state.get("ctx", {}) or {}
    if not isinstance(ctx, dict):
        ctx = {}

    start_date = ctx.get("start_date", getattr(base_ctx, "start_date", None))
    end_date = ctx.get("end_date", getattr(base_ctx, "end_date", None))

    seed_val = int(ctx.get("seed", st.session_state.get("global_seed", 1337)))
    st.session_state["global_seed"] = seed_val
    run_id = ctx.get("run_id") or f"opt-{seed_val}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    st.session_state["opt_run_id"] = run_id

    # ==== 0.6) רמזים מטאב ניתוח זוג (opt_pair_status) ==== 
    opt_pair_status = st.session_state.get("opt_pair_status")
    if isinstance(opt_pair_status, dict):
        pair_label = str(opt_pair_status.get("pair") or "")
        if pair_label and not ctrl_opt.get("sym1") and not ctrl_opt.get("sym_x"):
            sx = sy = None
            if "-" in pair_label:
                parts = pair_label.split("-", 1)
                sx, sy = parts[0].strip(), parts[1].strip()
            if sx and sy:
                ctrl_opt.setdefault("sym1", sx)
                ctrl_opt.setdefault("sym2", sy)

        hint = opt_pair_status.get("opt_hint", {}) or {}
        if hint:
            if hint.get("primary_objective") and "primary_objective" not in ctrl_opt:
                ctrl_opt["primary_objective"] = hint["primary_objective"]
            if hint.get("profile") and "profile" not in ctrl_opt:
                st.session_state.setdefault("opt_profile", hint["profile"])
            if hint.get("scenario_profile") and "scenario_profile" not in ctrl_opt:
                ctrl_opt["scenario_profile"] = hint["scenario_profile"]
            if hint.get("scenario_tail_weight") is not None and "scenario_tail_weight" not in ctrl_opt:
                ctrl_opt["scenario_tail_weight"] = float(hint["scenario_tail_weight"])
            if hint.get("wf_use") is not None and "wf_use" not in ctrl_opt:
                ctrl_opt["wf_use"] = bool(hint["wf_use"])

    # ==== 1) Title + Settings Sidebar ====
    st.title("⚙️ Pairs-Trading Optimiser — Hedge-Fund Grade")

    # Snapshot מהריצה האחרונה (אם יש)
    _render_optimization_snapshot_panel()
    st.markdown("---")

    new_settings = _render_profile_sidebar()
    if new_settings is not None:
        SETTINGS = new_settings
        _configure_logger_from_settings()
        st.success("Profile applied; settings updated for this session.")

    # ==== 2) System Health ==== (משאירים כמו שהיה)
    essentials = {
        "Backtester": Backtester is not None,
        "Optuna": (optuna is not None) and (TPESampler is not object),
        "CMA-ES": (CmaEsSampler is not object),
        "MedianPruner": (MedianPruner is not object),
        "Distributions": get_param_distributions is not None,
        "Metrics (normalize & score)": (normalize_metrics is not None and compute_weighted_score is not None),
    }
    with st.expander("🩺 System Health", expanded=not all(essentials.values())):
        for k, v in essentials.items():
            st.write(("✅" if v else "❌"), k)
        if not essentials["Backtester"]:
            st.markdown("**Backtester missing** – ודא ש-`core.optimization_backtester` importable.")
        if not essentials["Optuna"]:
            st.markdown("**Optuna not ready** – `pip install optuna` ודא ש-TPESampler זמין.")
        if not essentials["Distributions"]:
            st.markdown("**Missing distributions** – מצופה `core.distributions.get_param_distributions(ranges)`.")  
        if not essentials["Metrics (normalize & score)"]:
            st.markdown("משתמש ב-fallback פנימי למדדים/score.")

    # ==== 3) Core Config (ranges, weights, profile) ====
    cfg_core = _sidebar_core_config()
    ranges: Dict[str, ParamRange] = cfg_core["ranges"]
    weights: Dict[str, float] = cfg_core["weights"]
    profile: str = cfg_core["profile"]

    param_ranges_from_tab = ctrl_opt.get("param_ranges")
    if isinstance(param_ranges_from_tab, dict):
        for name, rng in param_ranges_from_tab.items():
            if name in ranges:
                ranges[name] = rng

    # ---- Defaults from ctrl_opt / ctx ----
    primary_objective = str(ctrl_opt.get("primary_objective", "Sharpe"))
    secondary_objective = str(ctrl_opt.get("secondary_objective", ""))
    secondary_weight = float(ctrl_opt.get("secondary_objective_weight", 0.0))

    direction = str(ctrl_opt.get("direction", "maximize"))
    multi_objective = bool(ctrl_opt.get("multi_objective", False))
    objective_metrics = list(ctrl_opt.get("objective_metrics") or [])

    sampler_name = str(ctrl_opt.get("sampler_name", "TPE"))
    pruner_name = str(ctrl_opt.get("pruner_name", "median"))
    max_concurrent_trials = int(ctrl_opt.get("max_concurrent_trials", 4))

    wf_use = bool(ctrl_opt.get("wf_use", True))
    wf_folds = int(ctrl_opt.get("wf_folds", 3))
    wf_warmup = int(ctrl_opt.get("wf_warmup_days", 60))
    robust_min_folds = int(ctrl_opt.get("robust_min_folds", max(1, wf_folds - 1)))
    robust_min_sharpe = float(ctrl_opt.get("robust_min_sharpe", 0.3))

    scenario_profile = str(ctrl_opt.get("scenario_profile", "Neutral"))
    scenario_tail_weight = float(ctrl_opt.get("scenario_tail_weight", 0.3))
    fresh_weight = float(ctrl_opt.get("fresh_weight", 0.5))

    meta_enabled = bool(ctrl_opt.get("meta_optimization_enabled", False))
    experiment_name = str(ctrl_opt.get("experiment_name", ""))

    # ==== 4) Sidebar Run Controls (מורחב) ====
    with st.sidebar.expander("Run Controls", expanded=True):
        # Experiment name חדש
        experiment_name = st.text_input(
            "Experiment name (optional)",
            value=str(ctrl_opt.get("experiment_name", "")),
            key="opt_experiment_name",
        )

        # Symbols / mode
        default_sym1 = ctrl_opt.get("sym1") or ctrl_opt.get("sym_x") or ctx.get("sym1") or "XLY"
        default_sym2 = ctrl_opt.get("sym2") or ctrl_opt.get("sym_y") or ctx.get("sym2") or "XLP"
        sym1 = st.text_input("Symbol A", value=str(default_sym1), key="opt_sym1")
        sym2 = st.text_input("Symbol B", value=str(default_sym2), key="opt_sym2")

        opt_mode = str(ctrl_opt.get("opt_mode", "Universe / Top Pairs"))
        opt_mode = st.radio(
            "Target Mode",
            ["Universe / Top Pairs", "זוג נבחר בלבד"],
            index=0 if opt_mode == "Universe / Top Pairs" else 1,
            key="opt_target_mode",
        )

        # Trials & timeout
        n_trials = int(
            st.number_input(
                "n_trials (סך ניסויים ל-Optuna)",
                min_value=10,
                max_value=10000,
                value=int(ctrl_opt.get("n_trials", default_n_trials)),
                step=10,
                key="opt_n_trials",
            )
        )
        timeout_min = int(
            st.number_input(
                "Timeout (דקות)",
                min_value=1,
                max_value=720,
                value=int(ctrl_opt.get("timeout_min", default_timeout_min)),
                step=1,
                key="opt_timeout_min",
            )
        )

    # =========================
    # Run optimization section
    # =========================
    st.markdown("#### Run optimization")

    # כאן אתה צריך לוודא שיש לך config מוכן ל-run_optimization
    opt_config = st.session_state.get("opt_config")

    if opt_config is None:
        st.warning(
            "לא נמצא config לאופטימיזציה (st.session_state['opt_config']).  \n"
            "תבנה dict שמתאים ל-run_optimization ותשמור אותו ב-session_state['opt_config'], "
            "או תחבר לכאן פונקציה שבונה את ה-config מהשירות/קונטקסט."
        )
    elif run_optimization is None:
        st.error("⚠️ run_optimization מ-core.optimizer לא זמין (import נכשל).")
    else:
        # יש גם config וגם run_optimization זמין → אפשר להריץ
        if st.button("🚀 Run optimization", key="opt_run_btn"):
            with st.spinner("מריץ Optuna אופטימיזציה..."):
                res = _run_core_optimization(
                    config=opt_config,
                    n_trials=int(n_trials),
                    study_name="dashboard_opt",
                    candidates=None,
                )

            if not res:
                st.warning("האופטימיזציה לא החזירה תוצאות.")
            else:
                df = res["df"]
                summary = res["summary"]
                ranked = res["ranked"]
                pareto = res["pareto"]
                tagged = res["tagged"]
                best_params = res["best_params"]

                st.success("האופטימיזציה הסתיימה ✅")

                # כמה כרטיסי KPI
                st.markdown("#### 📊 Optimization summary")
                col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                rows = float(summary.get("rows", float(len(df))))
                best_sharpe = summary.get("best_sharpe")
                avg_sharpe = summary.get("avg_sharpe")
                best_score = summary.get("best_score")

                with col_s1:
                    st.metric("Trials", f"{int(rows)}")
                with col_s2:
                    if best_sharpe is not None:
                        st.metric("Best Sharpe", f"{best_sharpe:.3f}")
                with col_s3:
                    if avg_sharpe is not None:
                        st.metric("Avg Sharpe", f"{avg_sharpe:.3f}")
                with col_s4:
                    if best_score is not None:
                        st.metric("Best score", f"{best_score:.3f}")

                # טבלת top-tagged
                st.markdown("#### 🏅 Top tagged trials")
                st.dataframe(tagged.head(50), width = "stretch")

                # Pareto
                if isinstance(pareto, pd.DataFrame) and not pareto.empty:
                    st.markdown("#### 🎯 Pareto front")
                    st.dataframe(pareto, width = "stretch")

                # best params
                st.markdown("#### ⭐ Best parameter sets")
                col_b1, col_b2 = st.columns(2)
                with col_b1:
                    st.markdown("**Best by Sharpe**")
                    st.json(best_params.get("by_sharpe") or {})
                with col_b2:
                    st.markdown("**Best by score**")
                    st.json(best_params.get("by_score") or {})
                _render_feature_selection_summary(df, opt_config)

        # Objectives
        primary_objective = st.selectbox(
            "Primary Objective",
            ["Sharpe", "Sortino", "Calmar", "Return", "TailRisk"],
            index=["Sharpe", "Sortino", "Calmar", "Return", "TailRisk"].index(
                str(ctrl_opt.get("primary_objective", "Sharpe"))
            ),
            key="opt_primary_obj",
        )
        secondary_objective = st.selectbox(
            "Secondary Objective (optional)",
            ["None", "Sharpe", "Sortino", "Calmar", "Return", "TailRisk"],
            index=0,
            key="opt_secondary_obj",
        )
        secondary_weight = float(
            st.number_input(
                "משקל Secondary Objective",
                min_value=0.0,
                max_value=1.0,
                value=float(ctrl_opt.get("secondary_objective_weight", 0.0)),
                step=0.05,
                key="opt_secondary_weight",
            )
        )
        if secondary_objective == "None":
            secondary_objective = ""

        direction = st.selectbox(
            "Score direction",
            ["maximize", "minimize"],
            index=["maximize", "minimize"].index(str(ctrl_opt.get("direction", "maximize"))),
            key="opt_direction",
        )

        # Multi-objective flag (לעתיד)
        multi_objective = st.checkbox(
            "Multi-Objective mode (Optuna)",
            value=bool(ctrl_opt.get("multi_objective", False)),
            key="opt_multi_objective",
        )
        if multi_objective:
            objective_metrics = st.multiselect(
                "Metrics למצב Multi-Objective",
                ["Sharpe", "Profit", "Drawdown", "Sortino", "Calmar", "WinRate"],
                default=ctrl_opt.get("objective_metrics") or ["Sharpe", "Profit", "Drawdown"],
                key="opt_mo_metrics",
            )
        else:
            objective_metrics = []

        # Sampler / Pruner
        sampler_name = st.selectbox(
            "Sampler",
            ["TPE", "CMAES"],
            index=["TPE", "CMAES"].index(str(ctrl_opt.get("sampler_name", "TPE")).upper()),
            key="opt_sampler",
        )
        pruner_name = st.selectbox(
            "Pruner",
            ["median", "none"],
            index=["median", "none"].index(str(ctrl_opt.get("pruner_name", "median")).lower()),
            key="opt_pruner",
        )
        max_concurrent_trials = int(
            st.number_input(
                "Max concurrent trials (hint)",
                min_value=1,
                max_value=64,
                value=int(ctrl_opt.get("max_concurrent_trials", 4)),
                step=1,
                key="opt_max_concurrent",
            )
        )

        # WF & Robustness
        wf_use = st.checkbox(
            "הפעל Walk-Forward per trial (אם נתמך ב-Backtester)",
            value=bool(ctrl_opt.get("wf_use", True)),
            key="opt_wf_use",
        )
        wf_folds = int(
            st.number_input(
                "WF folds (אם מופעל)",
                min_value=1,
                max_value=12,
                value=int(ctrl_opt.get("wf_folds", 3)),
                step=1,
                key="opt_wf_folds",
            )
        )
        wf_warmup = int(
            st.number_input(
                "WF warmup days",
                min_value=0,
                max_value=365,
                value=int(ctrl_opt.get("wf_warmup_days", 60)),
                step=5,
                key="opt_wf_warmup",
            )
        )
        robust_min_folds = int(
            st.number_input(
                "מינ' folds עם ביצועים סבירים",
                min_value=1,
                max_value=wf_folds,
                value=int(ctrl_opt.get("robust_min_folds", max(1, wf_folds - 1))),
                step=1,
                key="opt_robust_min_folds",
            )
        )
        robust_min_sharpe = float(
            st.number_input(
                "Sharpe מינימלי per fold (Robustness floor)",
                min_value=-1.0,
                max_value=3.0,
                value=float(ctrl_opt.get("robust_min_sharpe", 0.3)),
                step=0.05,
                key="opt_robust_min_sharpe",
            )
        )

        # Scenario-aware
        scenario_options = ["Neutral", "Risk-On", "Risk-Off", "Crisis"]
        scenario_default = str(ctrl_opt.get("scenario_profile", "Neutral"))
        if scenario_default not in scenario_options:
            scenario_default = "Neutral"

        scenario_profile = st.selectbox(
            "Scenario Profile (לוגי, ל-analysis & reports)",
            scenario_options,
            index=scenario_options.index(scenario_default),
            key="opt_scenario_profile",
        )
        scenario_tail_weight = float(
            st.number_input(
                "משקל Tail-Risk בתרחיש (0–1)",
                min_value=0.0,
                max_value=1.0,
                value=float(ctrl_opt.get("scenario_tail_weight", 0.3)),
                step=0.05,
                key="opt_scenario_tail_weight",
            )
        )

        # Fresh vs warm data weight (לניתוח עתידי)
        fresh_weight = float(
            st.number_input(
                "משקל לתקופה האחרונה בתוצאה (0–1, לניתוח עתידי)",
                min_value=0.0,
                max_value=1.0,
                value=float(ctrl_opt.get("fresh_weight", 0.5)),
                step=0.05,
                key="opt_fresh_weight",
            )
        )

        # Batch mode
        batch_mode = st.checkbox(
            "Run batch over multiple pairs",
            value=bool(ctrl_opt.get("batch_mode", False)),
            key="opt_batch_mode",
        )
        batch_pairs_raw = ""
        if batch_mode:
            st.caption("Format: lines of `SYM1-SYM2`, e.g.:\nXLY-XLP\nQQQ-IWM")
            default_batch = "\n".join(
                list(st.session_state.get("opt_batch_pairs", [])) or ["XLY-XLP", "QQQ-IWM"]
            )
            batch_pairs_raw = st.text_area(
                "Batch pairs",
                value=default_batch,
                key="opt_batch_pairs_text",
            )
            st.session_state["opt_batch_pairs"] = [
                ln.strip() for ln in batch_pairs_raw.splitlines() if ln.strip()
            ]

        # Meta-optimization flag
        meta_enabled = bool(ctrl_opt.get("meta_optimization_enabled", False))
        meta_enabled = st.checkbox(
            "Meta-Optimization (לוגי, לניתוח ול-step הבא)",
            value=meta_enabled,
            key="opt_meta_enabled",
        )

        # SqlStore/IBKR ingestion לפני אופטימיזציה
        ensure_prices_via_ib = st.checkbox(
            "Ensure SqlStore prices via IBKR before optimisation",
            value=bool(ctrl_opt.get("ensure_prices_via_ib", True)),
            key="opt_ensure_prices_ib",
            help=(
                "כאשר פעיל – לפני כל ריצת אופטימיזציה (single/batch) הטאב ינסה להשלים מחירים חסרים "
                "ב-SqlStore עבור הזוגות המתאימים דרך IBKR (באופן אינקרמנטלי). "
                "במקרה של כשל – הריצה תמשיך כרגיל, רק עם לוג."
            ),
        )

        # === Run complexity estimator (רעיון חדש) ===
        approx_pairs = 1 if not batch_mode else max(1, len(st.session_state.get("opt_batch_pairs", [])))
        approx_wf_factor = wf_folds if wf_use and wf_folds > 1 else 1
        complexity_units = n_trials * approx_pairs * approx_wf_factor
        st.caption(
            f"🔍 Estimated complexity units ≈ `{complexity_units:,}` "
            f"(~ n_trials × pairs × WF_folds)."
        )
        if complexity_units > 50_000:
            st.warning("הריצה הזו כבדה יחסית – שקול להוריד WF folds או n_trials.")

        # ==== Action buttons: dry-run / single / batch ====
        dry_run = st.button("🔎 Dry-run config (no optimisation)", key="opt_dry_run")

        allow_live_opt = True
        if is_live_env:
            allow_live_opt = st.checkbox(
                "Allow heavy optimisation in LIVE/PROD env",
                value=bool(st.session_state.get("opt_allow_live_opt", False)),
                key="opt_allow_live_opt",
                help=(
                    "בטיחות: בסביבת LIVE/PROD הרצה של Optuna/Batch יכולה להיות כבדה מאוד. "
                    "סמן רק אם אתה בטוח שאתה רוצה להריץ את זה על סביבת live."
                ),
            )

        run_single_raw = st.button("▶ Run (single pair)", type="primary", key="opt_run_single")
        run_batch_raw = st.button("🧪 Run batch", key="opt_run_batch") if batch_mode else False

        # החלת gating: ב-LIVE/PROD בלי checkbox → לא מריצים
        run_single = bool(run_single_raw and (allow_live_opt or not is_live_env))
        run_batch = bool(run_batch_raw and (allow_live_opt or not is_live_env))


    # ==== 5) שמירת opt_run_cfg ל-session (שימוש ע"י scoring, WF וכו') ====
    # וידוא ברירות מחדל לכל המשתנים – גם אם ה-UI של ה-Objectives/WF לא רץ
    # (למשל כאשר opt_config is None או run_optimization is None).

    direction = locals().get("direction", str(ctrl_opt.get("direction", "maximize")))
    multi_objective = locals().get("multi_objective", bool(ctrl_opt.get("multi_objective", False)))
    objective_metrics = locals().get(
        "objective_metrics",
        list(ctrl_opt.get("objective_metrics") or []),
    )

    primary_objective = locals().get(
        "primary_objective",
        str(ctrl_opt.get("primary_objective", "Sharpe")),
    )
    secondary_objective = locals().get(
        "secondary_objective",
        str(ctrl_opt.get("secondary_objective", "")),
    )
    secondary_weight = locals().get(
        "secondary_weight",
        float(ctrl_opt.get("secondary_objective_weight", 0.0)),
    )

    sampler_name = locals().get(
        "sampler_name",
        str(ctrl_opt.get("sampler_name", "TPE")),
    )
    pruner_name = locals().get(
        "pruner_name",
        str(ctrl_opt.get("pruner_name", "median")),
    )
    max_concurrent_trials = locals().get(
        "max_concurrent_trials",
        int(ctrl_opt.get("max_concurrent_trials", 4)),
    )

    wf_use = locals().get("wf_use", bool(ctrl_opt.get("wf_use", True)))
    wf_folds = locals().get("wf_folds", int(ctrl_opt.get("wf_folds", 3)))
    wf_warmup = locals().get("wf_warmup", int(ctrl_opt.get("wf_warmup_days", 60)))
    robust_min_folds = locals().get(
        "robust_min_folds",
        int(ctrl_opt.get("robust_min_folds", max(1, wf_folds - 1))),
    )
    robust_min_sharpe = locals().get(
        "robust_min_sharpe",
        float(ctrl_opt.get("robust_min_sharpe", 0.3)),
    )

    scenario_profile = locals().get(
        "scenario_profile",
        str(ctrl_opt.get("scenario_profile", "Neutral")),
    )
    scenario_tail_weight = locals().get(
        "scenario_tail_weight",
        float(ctrl_opt.get("scenario_tail_weight", 0.3)),
    )
    fresh_weight = locals().get(
        "fresh_weight",
        float(ctrl_opt.get("fresh_weight", 0.5)),
    )

    meta_enabled = locals().get(
        "meta_enabled",
        bool(ctrl_opt.get("meta_optimization_enabled", False)),
    )
    experiment_name = locals().get(
        "experiment_name",
        str(ctrl_opt.get("experiment_name", "")),
    )

    opt_run_cfg = {
        "run_id": run_id,
        "seed": seed_val,
        "sym1": sym1,
        "sym2": sym2,
        "opt_mode": opt_mode,
        "n_trials": n_trials,
        "timeout_min": timeout_min,
        "direction": direction,
        "primary_objective": primary_objective,
        "secondary_objective": secondary_objective,
        "secondary_weight": secondary_weight,
        "multi_objective": multi_objective,
        "objective_metrics": objective_metrics,
        "sampler_name": sampler_name,
        "pruner_name": pruner_name,
        "max_concurrent_trials": max_concurrent_trials,
        "wf_use": wf_use,
        "wf_folds": wf_folds,
        "wf_warmup": wf_warmup,
        "robust_min_folds": robust_min_folds,
        "robust_min_sharpe": robust_min_sharpe,
        "scenario_profile": scenario_profile,
        "scenario_tail_weight": scenario_tail_weight,
        "fresh_weight": fresh_weight,
        "meta_optimization_enabled": meta_enabled,
        "experiment_name": experiment_name,
        "ensure_prices_via_ib": bool(
            locals().get("ensure_prices_via_ib", ctrl_opt.get("ensure_prices_via_ib", True))
        ),
    }

    st.session_state["opt_run_cfg"] = opt_run_cfg


    # ==== 5.1 Dry-run config – sanity check בלבד ====

    if dry_run:
        with st.expander("Dry-run config summary", expanded=True):
            st.json(opt_run_cfg)
            issues = []
            if n_trials < 20:
                issues.append("n_trials נמוך מ-20 – זה יותר Smoke-test מאשר אופטימיזציה.")
            if batch_mode and not st.session_state.get("opt_batch_pairs"):
                issues.append("Batch mode פעיל אבל אין אף pair ברשימה.")
            if wf_use and wf_folds < 2:
                issues.append("WF מופעל אבל WF_folds < 2 – אין משמעות אמיתית ל-WF.")
            if not ranges:
                issues.append("active_ranges ריק – אין מרחב פרמטרים לאופטימיזציה.")
            if not weights:
                issues.append("weights ריק – scoring לא מוגדר היטב.")
            if issues:
                st.warning("🔎 Findings:")
                st.write("\n".join(f"• {msg}" for msg in issues))
            else:
                st.success("No obvious issues detected in config.")
        # dry-run לא מריץ כלום – נמשיך הלאה כדי לראות תוצאות קודמות אם יש.

    # ==== 6) Run single / batch ==== 
    res_container = st.container()
    df_single = None
    df_batch = None

    if is_live_env and (run_single or run_batch) and not st.session_state.get("opt_allow_live_opt", False):
        st.warning("Optuna/Batch optimisation in LIVE/PROD env חסומה עד שתסמן את ה-checkbox ב־Run Controls.")

    # Single pair optimisation
    if run_single:
        st.session_state["_opt_last_single_pair"] = f"{sym1}-{sym2}"
        with st.spinner(f"Optimising {sym1}-{sym2}…"):
            df_single, study_id, manifest = optimize_pair_with_telemetry(
                sym1,
                sym2,
                ranges=ranges,
                weights=weights,
                n_trials=n_trials,
                timeout_min=timeout_min,
                direction=direction,
                sampler_name=sampler_name,
                pruner_name=pruner_name,
                param_mapping=st.session_state.get("opt_param_mapping"),
                profile=profile,
                multi_objective=multi_objective,
                objective_metrics=objective_metrics,
                param_specs=get_param_specs_view(),
                notify=True,
            )

            st.session_state["opt_df"] = df_single
            st.session_state["opt_last_manifest"] = manifest
            st.session_state["opt_last_study_id"] = study_id

            # --- רישום best params לזוג הנוכחי (HF-grade) ---
            # קודם כל נבחר live-candidate מתוך df_single בעזרת DSR/WF אם קיים
            best_params, best_score = select_live_candidate_from_opt_df(df_single)

            # אם ה-manifest מכיל best_params/best_score מפורש – ניתן לו עדיפות
            if isinstance(manifest, dict):
                try:
                    man_best_params = manifest.get("best_params") or {}
                    if man_best_params:
                        best_params = man_best_params
                except Exception:
                    pass
                try:
                    man_best_score = manifest.get("metrics", {}).get("best_score")
                    if man_best_score is not None:
                        best_score = float(man_best_score)
                except Exception:
                    pass

            pair_label = f"{sym1}-{sym2}"

            # רישום ל-registry ברמת המערכת (עם הגנות LIVE/PROD)
            register_best_params_for_pair(
                pair_label,
                best_params,
                score=best_score,
                profile=profile,
            )


            # ננסה להוציא best_params / best_score מתוך manifest אם יש
            if isinstance(manifest, dict):
                try:
                    best_params = manifest.get("best_params") or {}
                except Exception:
                    best_params = {}
                try:
                    best_score = manifest.get("metrics", {}).get("best_score")
                except Exception:
                    best_score = None

            # fallback: מה-DataFrame עצמו
            if (not best_params) and df_single is not None and not df_single.empty:
                try:
                    # נבחר שורה עם Score הכי גבוה (אם קיים)
                    if "Score" in df_single.columns:
                        row = df_single.sort_values("Score", ascending=False).iloc[0]
                    else:
                        row = df_single.iloc[0]
                    # פרמטרים = כל מה שלא מטריקות
                    metric_like = {
                        c for c in df_single.columns
                        if METRIC_KEYS.get(str(c).lower()) or c in ("Score", "Pair", "study_id")
                    }
                    best_params = {c: row[c] for c in df_single.columns if c not in metric_like}
                    if "Score" in df_single.columns:
                        best_score = float(pd.to_numeric(df_single["Score"], errors="coerce").max())
                except Exception:
                    best_params = best_params or {}
                    best_score = best_score

            pair_label = f"{sym1}-{sym2}"
            register_best_params_for_pair(
                pair_label,
                best_params,
                score=best_score,
                profile=profile,
            )

            # --- Snapshot למערכת כולה (Home / Risk / Agents) ---
            # ננסה להוציא best_score ו-best_params מה-manifest; אם אין, נ fallback ל-DataFrame
            best_score = None
            best_params = {}

            if isinstance(manifest, dict):
                try:
                    best_score = manifest.get("metrics", {}).get("best_score")
                except Exception:
                    best_score = None

            if best_score is None and df_single is not None and not df_single.empty and "Score" in df_single.columns:
                try:
                    best_score = float(pd.to_numeric(df_single["Score"], errors="coerce").max())
                except Exception:
                    best_score = None

            try:
                # אם הוספת בעתיד "best_params" ל-manifest – ננסה להשתמש בהם
                if isinstance(manifest, dict) and "best_params" in manifest:
                    best_params = manifest.get("best_params") or {}
            except Exception:
                best_params = {}

            # --- Snapshot למערכת כולה (Home / Risk / Agents) ---
            push_optimization_metrics_to_ctx(
                best_params=best_params,
                best_score=best_score,
                n_trials=n_trials,
                objective_name=str(primary_objective),
                pair_id=f"{sym1}-{sym2}",
                profile=profile,
                mode="single",
            )


            history = st.session_state.get("opt_runs_timeline", [])
            if not isinstance(history, list):
                history = []
            history.append(
                {
                    "ts": datetime.now().isoformat(timespec="seconds"),
                    "run_id": run_id,
                    "type": "single",
                    "pair": f"{sym1}-{sym2}",
                    "n_trials": n_trials,
                    "timeout_min": timeout_min,
                    "profile": profile,
                    "primary_objective": primary_objective,
                    "experiment_name": experiment_name,
                }
            )
            st.session_state["opt_runs_timeline"] = history[-50:]

        with res_container:
            if df_single is not None and not df_single.empty:
                st.success(f"Completed optimisation for {sym1}-{sym2}. study_id={study_id or 'n/a'}")
            else:
                st.warning("No results produced (single pair).")

    # Batch optimisation
    if run_batch and batch_mode:
        batch_list: List[Tuple[str, str]] = []
        for ln in st.session_state.get("opt_batch_pairs", []):
            ln = ln.strip()
            if not ln:
                continue
            sep_used: Optional[str] = None
            for sep in ("|", "/", "\\", ":", "-"):
                if sep in ln:
                    sep_used = sep
                    break
            if not sep_used:
                continue
            a, b = ln.split(sep_used, 1)
            a, b = a.strip(), b.strip()
            if a and b:
                batch_list.append((a, b))

        # אם אין בכלל זוגות אחרי הפרסינג – אין מה להריץ
        if not batch_list:
            st.warning("Batch mode פעיל אבל לא הוגדר אף pair תקין ברשימה.")
        else:
            # וידוא מחירים לכל זוג ב-Batch (Best-effort בלבד, לפני Optuna)
            # משתמש ב־ensure_prices_via_ib מה־Run Controls (checkbox)
            ensure_flag = bool(
                locals().get(
                    "ensure_prices_via_ib",
                    ctrl_opt.get("ensure_prices_via_ib", True),
                )
            )
            if ensure_flag:
                for sx, sy in batch_list:
                    try:
                        _ensure_prices_for_pair_before_opt(
                            sx,
                            sy,
                            start_date=start_date,
                            end_date=end_date,
                        )
                    except Exception as e:
                        logger.debug(
                            "OptTab: _ensure_prices_for_pair_before_opt failed for batch pair %s-%s: %s",
                            sx,
                            sy,
                            e,
                        )

            with st.spinner(f"Running batch optimisation for {len(batch_list)} pairs…"):
                df_batch, manifests = optimize_pairs_batch(
                    batch_list,
                    ranges=ranges,
                    weights=weights,
                    n_trials=n_trials,
                    timeout_min=timeout_min,
                    direction=direction,
                    sampler_name=sampler_name,
                    pruner_name=pruner_name,
                    profile=profile,
                    multi_objective=multi_objective,
                    objective_metrics=objective_metrics,
                    param_specs=get_param_specs_view(),
                    notify_each=False,
                )
                st.session_state["opt_df_batch"] = df_batch
                st.session_state["opt_batch_manifests"] = manifests

                # --- רישום best params לכל זוג מתוך Batch (HF-grade live candidate) ---
                if df_batch is not None and not df_batch.empty and "Pair" in df_batch.columns:
                    try:
                        for pair_label, sub in df_batch.groupby("Pair"):
                            # 1) בחירה חכמה מתוך ה-sub באמצעות select_live_candidate_from_opt_df
                            best_params, best_score = select_live_candidate_from_opt_df(sub)

                            # 2) אם יש manifest ספציפי לזוג – נותן לו זכות veto
                            man = manifests.get(pair_label) if isinstance(manifests, dict) else None
                            if isinstance(man, dict):
                                try:
                                    man_best_params = man.get("best_params") or {}
                                    if man_best_params:
                                        best_params = man_best_params
                                except Exception:
                                    pass
                                try:
                                    man_best_score = man.get("metrics", {}).get("best_score")
                                    if man_best_score is not None:
                                        best_score = float(man_best_score)
                                except Exception:
                                    pass

                            register_best_params_for_pair(
                                str(pair_label),
                                best_params,
                                score=best_score,
                                profile=profile,
                            )
                    except Exception as e:
                        logger.debug("register_best_params_for_pair from batch failed: %s", e)

                # --- Snapshot גם לריצת Batch ---
                best_score_batch: Optional[float] = None
                if df_batch is not None and not df_batch.empty and "Score" in df_batch.columns:
                    try:
                        best_score_batch = float(pd.to_numeric(df_batch["Score"], errors="coerce").max())
                    except Exception:
                        best_score_batch = None

                push_optimization_metrics_to_ctx(
                    best_params={},   # ב-Batch פחות הגיוני לשמור סט אחד, זה יותר "overview"
                    best_score=best_score_batch,
                    n_trials=n_trials,
                    objective_name=str(primary_objective),
                    pair_id="batch",
                    profile=profile,
                    mode="batch",
                )

                history = st.session_state.get("opt_runs_timeline", [])
                if not isinstance(history, list):
                    history = []
                history.append(
                    {
                        "ts": datetime.now().isoformat(timespec="seconds"),
                        "run_id": run_id,
                        "type": "batch",
                        "n_pairs": len(batch_list),
                        "n_trials": n_trials,
                        "timeout_min": timeout_min,
                        "profile": profile,
                        "primary_objective": primary_objective,
                        "experiment_name": experiment_name,
                    }
                )
                st.session_state["opt_runs_timeline"] = history[-50:]

            with res_container:
                if df_batch is not None and not df_batch.empty:
                    st.success(f"Batch finished — {len(df_batch)} rows across {len(batch_list)} pairs.")
                else:
                    st.warning("Batch produced no rows (check logs).")

    # ==== 7) Use last results if no fresh run ====
    df_all = None
    if df_batch is not None and not df_batch.empty:
        df_all = df_batch
    else:
        df_all = st.session_state.get("opt_df", pd.DataFrame())

    if df_all is None or df_all.empty:
        st.info("Run an optimisation (single or batch) to see results.")
        # עדיין אפשר לראות היסטוריה
        st.markdown("---")
        render_opt_run_history(service, base_ctx)
        return

    df = df_all.copy()

    # =========================
    # Build optimization config
    # =========================

    st.markdown("#### Data source")

    price_csv = st.text_input(
        "Price CSV path (leg A)",
        value="data/AAPL.csv",
        key="opt_price_csv",
        help="קובץ CSV עם מחירי Close של leg A (ניתן להחליף ל-preloaded series בעתיד).",
    )
    hedge_csv = st.text_input(
        "Hedge CSV path (leg B)",
        value="data/MSFT.csv",
        key="opt_hedge_csv",
        help="קובץ CSV עם מחירי Close של leg B.",
    )

    # כאן אתה יכול בעתיד להחליף את ה-CSV ב-series מהמערכת (SqlStore/IBKR וכו')
    data_cfg: Dict[str, Any] = {
        "price_csv": price_csv,
        "hedge_csv": hedge_csv,
    }

    # טווחי פרמטרים – היום פשוט מה-UI, מחר מ-PARAM_SPECS
    ranges_cfg: Dict[str, Any] = {
        "z_open": (float(z_open) - 1.0, float(z_open) + 1.0),
        "z_close": (float(z_close) - 1.0, float(z_close) + 1.0),
        "max_exposure_per_trade": (0.0, float(max_exposure)),
    }

    # Signal generator – כאן אתה מחבר לפונקציה האמיתית שלך
    try:
        from common.signal_generator import generate_signals_for_backtest  # דוגמה – תעדכן לשם האמיתי
        signal_generator = generate_signals_for_backtest
    except Exception:
        signal_generator = None

    signals_cfg: Dict[str, Any] = {
        "generator": signal_generator,
    }

    # backtest_params – כאן אתה שם כל מה שה-Backtester שלך צריך
    backtest_params: Dict[str, Any] = {
        "start_date": getattr(base_ctx, "start_date", None),
        "end_date": getattr(base_ctx, "end_date", None),
        # אפשר להוסיף כאן risk_profile, fees, slippage וכו'
    }

    # הגדרות Optuna + score
    optuna_cfg: Dict[str, Any] = {
        "direction": "maximize",
        "sampler": "TPE",
        "pruner": "median",
        "timeout_sec": int(timeout_min * 60),
        "n_trials": int(n_trials),
        "multi_objective": False,
    }

    score_cfg: Dict[str, Any] = {
        "mode": "hybrid",      # classic / hf / hybrid
        "hybrid_alpha": 0.7,   # משקל HF מול classic
    }

    opt_config: Dict[str, Any] = {
        "data": data_cfg,
        "ranges": ranges_cfg,
        "signals": signals_cfg,
        "backtest_params": backtest_params,
        "optuna": optuna_cfg,
        "score": score_cfg,
        # אופציונלי:
        # "seed": 42,
        # "seed_from_candidates": True,
    }
    # נשמור את הקונפיג ל-session כך שהכפתור למעלה יעבוד עם הגרסה העדכנית
    st.session_state["opt_config"] = opt_config

    # ==== 8) Quick KPIs ====
    st.subheader("📊 Quick KPIs")
    try:
        c1, c2, c3, c4, c5, c6 = st.columns(6)

        if "Score" in df.columns:
            best_score = pd.to_numeric(df["Score"], errors="coerce").max()
            if pd.notna(best_score):
                c1.metric("Best Score", f"{best_score:.3f}")

        if "Sharpe" in df.columns:
            avg_sharpe = pd.to_numeric(df["Sharpe"], errors="coerce").mean()
            if pd.notna(avg_sharpe):
                c2.metric("Avg Sharpe", f"{avg_sharpe:.3f}")

        if "Profit" in df.columns:
            max_profit = pd.to_numeric(df["Profit"], errors="coerce").max()
            if pd.notna(max_profit):
                c3.metric("Max Profit", f"{max_profit:,.0f}")

        if "Drawdown" in df.columns:
            min_dd = pd.to_numeric(df["Drawdown"], errors="coerce").min()
            if pd.notna(min_dd):
                c4.metric("Min Drawdown", f"{min_dd:.3f}")

        c5.metric("Total trials", f"{len(df):,}")

        if "ParamScore" in df.columns:
            best_param_score = pd.to_numeric(df["ParamScore"], errors="coerce").max()
            if pd.notna(best_param_score):
                c6.metric("Best ParamScore", f"{best_param_score:.3f}")
    except Exception:
        pass


    # ==== 8.1 Validation Snapshot (health of opt_df) ====
    try:
        report = validate_opt_df(df)
        if report:
            ok = bool(report.get("ok", False))
            issues = report.get("issues", []) or []
            summary = report.get("summary", {}) or {}
            if ok and not issues:
                st.success("Validation: opt_df looks healthy (Score/Sharpe/DD present).")
            else:
                with st.expander("⚠️ Validation issues in optimisation results", expanded=False):
                    st.write("Detected issues:")
                    for msg in issues:
                        st.write(f"- {msg}")
                    st.caption("Summary:")
                    st.json(summary)
    except Exception as e:
        logger.debug("validate_opt_df failed: %s", e)

    # ==== 9) Top-N view + basic exports ====
    st.subheader("🏅 Top 20 Results (basic view)")

    sort_by = st.selectbox(
        "Sort Top-20 by",
        ["Score", "ParamScore"],
        index=0 if "ParamScore" not in df.columns else 0,
        key="opt_top_sort_by",
    )

    score_col = sort_by if sort_by in df.columns else ("Score" if "Score" in df.columns else None)

    if score_col:
        top = df.sort_values(score_col, ascending=False).head(20)
    else:
        top = df.head(20)

    table_height = int(st.session_state.get("_opt_table_height", 420))
    st.dataframe(top, width="stretch", height=table_height)

    try:
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button(
            "💾 Download Top-20 (CSV)",
            data=top.to_csv(index=False).encode("utf-8"),
            file_name=f"opt_top20_{ts_str}.csv",
            mime="text/csv",
            key="opt_dl_top20_csv",
        )
        st.download_button(
            "💾 Download ALL results (CSV)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=f"opt_all_results_{ts_str}.csv",
            mime="text/csv",
            key="opt_dl_all_csv",
        )
    except Exception:
        pass

        # ==== 9.1 Validation Snapshot (using validate_opt_df) ====
    try:
        report = validate_opt_df(df)
        if not report.get("ok", False):
            with st.expander("⚠️ Validation issues (opt_df)", expanded=False):
                st.write("Issues detected in optimisation results:")
                for msg in report.get("issues", []):
                    st.write(f"- {msg}")
                st.json(report.get("summary", {}))
    except Exception:
        # לא מפיל את הטאב בגלל ולידציה
        pass

    # ==== 10) Advanced Analytics & Ops Hub ====
    st.subheader("🔬 Analytics & Ops Hub — HF-grade view")

    # גובה טבלאות דיפולטי (אפשר לשלוט דרך session_state אם תרצה)
    TABLE_HEIGHT = int(st.session_state.get("_opt_table_height", 420))

    # 10.1 Analytics Hub (Pareto / DSR / Walk-Forward / וכו')
    try:
        _render_optimization_analytics_section(TABLE_HEIGHT, df, sym1, sym2)
    except Exception as e:
        st.caption(f"Analytics Hub failed: {e}")

    # 10.2 Operations / Ranges / Presets / Datasets / Reports
    try:
        _render_optimization_operations_sections(TABLE_HEIGHT, df, sym1, sym2)
    except Exception as e:
        st.caption(f"Operations sections failed: {e}")

    # 10.3 Monitoring / Telemetry / Studies timeline
    try:
        _render_optimization_monitoring_sections(TABLE_HEIGHT, df, sym1, sym2)
    except Exception as e:
        st.caption(f"Monitoring sections failed: {e}")

    # 10.x Replay Best Trial (חיבור ישיר בין Optuna ↔ Backtester אמיתי)
    try:
        _render_replay_best_trial_panel(TABLE_HEIGHT, df, sym1, sym2)
    except Exception as e:
        st.caption(f"Replay-best-trial panel failed: {e}")

    # 10.4 Portfolio Constructor Pro (multi-pair sleeve)
    try:
        _render_optimization_portfolio_section(TABLE_HEIGHT, df, sym1, sym2)
    except Exception as e:
        st.caption(f"Portfolio section failed: {e}")

    # 10.5 Governance & Reproducibility Manifest
    try:
        _render_reproducibility_governance_section(TABLE_HEIGHT, df, sym1, sym2)
    except Exception as e:
        st.caption(f"Governance / reproducibility section failed: {e}")

    # 10.6 Macro & Factor overlay (scenario analysis)
    try:
        _render_optimization_macro_factor_section(TABLE_HEIGHT, df, sym1, sym2)
    except Exception as e:
        st.caption(f"Macro/factor overlay section failed: {e}")

    # 10.7 Audit pack (settings + ranges + presets + snapshot)
    try:
        _render_optimization_audit_pack_section(TABLE_HEIGHT, df, sym1, sym2)
    except Exception as e:
        st.caption(f"Audit pack section failed: {e}")

    # 10.8 Finalize (Dev tools, Error console, i18n, snapshot)
    try:
        _render_optimization_finalize(TABLE_HEIGHT, df, sym1, sym2)
    except Exception as e:
        st.caption(f"Finalize section failed: {e}")

# =========================
# SECTION 7: Pareto helper
# =========================
from datetime import date as _date_type

def _compute_pareto_df(
    df: pd.DataFrame,
    *,
    score_col: str = "Score",
    sharpe_col: str = "Sharpe",
    dd_col: str = "Drawdown",
    max_points: Optional[int] = None,
    min_improvement: float = 0.0,
) -> pd.DataFrame:
    """
    מחשב Pareto Frontier על בסיס:
        - maximize Score
        - maximize Sharpe
        - minimize Drawdown

    שדרוגים:
    --------
    1. משתמש ב-core.optimizer.compute_pareto_front אם זמין.
    2. ניקוי נתונים:
       - המרת העמודות ל-numeric.
       - הסרת NaN/Inf לפני החישוב.
    3. תמיכה ב-min_improvement:
       - אם min_improvement > 0 → נקודה תיחשב דומיננטית רק אם
         יש שיפור משמעותי לפחות בממד אחד (Score/Sharpe/DD).
    4. תמיכה ב-max_points:
       - אם frontier גדול מאוד → מגביל ל-top-N לפי Score (כדי לא להציף את ה-UI).

    החזרה:
    -------
    DataFrame של נקודות non-dominated (שומר את כל העמודות המקוריות).
    """

    # ===== 0) בדיקות בסיס =====
    if df is None or df.empty:
        return pd.DataFrame()

    cols_needed = [score_col, sharpe_col, dd_col]
    for c in cols_needed:
        if c not in df.columns:
            logger.debug("Pareto: column '%s' not found in df; returning empty.", c)
            return pd.DataFrame()

    # ===== 1) core.optimizer.compute_pareto_front אם זמין =====
    if callable(compute_pareto_front):
        try:
            pareto = compute_pareto_front(
                df,
                score_col=score_col,
                sharpe_col=sharpe_col,
                dd_col=dd_col,
            )
            # אפשר עדיין להגביל max_points אם ביקשו
            if max_points is not None and max_points > 0 and not pareto.empty:
                pareto = pareto.sort_values(score_col, ascending=False).head(int(max_points))
            return pareto
        except Exception as e:
            logger.warning("compute_pareto_front failed, using manual Pareto: %s", e)

    # ===== 2) ניקוי נתונים ידני =====
    # נוודא שהעמודות מספריות ונפטר מ-NaN/Inf
    d = df.copy()
    for c in cols_needed:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.replace([np.inf, -np.inf], np.nan)
    d_sub = d[cols_needed].dropna().copy()

    if d_sub.empty:
        return pd.DataFrame()

    # נשמור אינדקסים כדי לחזור ל-df המקורי
    idx_valid = d_sub.index
    vals = d_sub[cols_needed].to_numpy(dtype=float)
    n = vals.shape[0]

    # ===== 3) Manual Pareto O(N^2) עם tol =====
    mask = np.ones(n, dtype=bool)
    tol = float(min_improvement) if min_improvement and min_improvement > 0 else 0.0

    # כדי להשתמש ב-min_improvement בצורה נורמלית, נאפיין deltas:
    # Score / Sharpe: גבוה יותר טוב.
    # Drawdown: נמוך יותר טוב.
    for i in range(n):
        if not mask[i]:
            continue

        s_i, sh_i, dd_i = vals[i]

        # הפרשים מול כל השאר
        delta_score = vals[:, 0] - s_i
        delta_sharpe = vals[:, 1] - sh_i
        delta_dd = dd_i - vals[:, 2]  # חיובי אם האחרים עם DD נמוך יותר (טוב יותר)

        # תנאי "לא פחות טוב" בכל הממדים (בהתחשב ב-tol)
        cond = (
            (delta_score >= -tol) &
            (delta_sharpe >= -tol) &
            (delta_dd >= -tol)
        )
        # ותנאי "טוב יותר משמעותית" לפחות באחד (בהתחשב ב-tol)
        strict = (
            (delta_score > tol) |
            (delta_sharpe > tol) |
            (delta_dd > tol)
        )

        dominated_by_any = np.any(cond & strict)
        if dominated_by_any:
            mask[i] = False

    pareto_idx = idx_valid[mask]
    pareto = df.loc[pareto_idx].copy()

    # ===== 4) הגבלת מספר נקודות אם ביקשו =====
    if max_points is not None and max_points > 0 and not pareto.empty:
        try:
            pareto = pareto.sort_values(score_col, ascending=False).head(int(max_points))
        except Exception:
            pareto = pareto.head(int(max_points))

    return pareto

# =========================
# SECTION 7.1: Deflated Sharpe Ratio (DSR, approx)
# =========================

def deflated_sharpe_ratio(
    sharpe: float,
    t: int,
    n_strategies: int,
    *,
    skew: float = 0.0,
    kurt: float = 3.0,
    two_sided: bool = False,
    use_student_t: bool = True,
    max_strategies: Optional[int] = None,
) -> Tuple[float, float]:
    """
    הערכת Deflated Sharpe Ratio בקירוב (Bailey & López de Prado style, slightly enhanced).

    Inputs
    ------
    sharpe : float
        Sharpe ratio raw.
    t : int
        Effective sample size (מספר טריידים / תצפיות).
    n_strategies : int
        מספר האסטרטגיות שנבדקו (למולטיפל-טסטינג).
    skew : float, optional
        Skewness של התשואות (אם ידוע). משפיע מעט על התיקון.
    kurt : float, optional
        Kurtosis של התשואות (אם ידוע). kurt≈3 זה נורמלי.
    two_sided : bool, optional
        אם True → מבחן דו־צדדי (|Z|), אחרת חד־צדדי (Z>0).
    use_student_t : bool, optional
        אם True ו-SciPy זמינה → מחשבים p_single לפי Student-t (df=t-1) במקום Normal.
    max_strategies : int | None, optional
        מגבלה על n_strategies האפקטיבי. אם None → משתמשים בערך המקורי.

    Outputs
    -------
    (dsr, p_eff) : Tuple[float, float]
        dsr   — Sharpe "מתוקן" (z-score יעיל אחרי תיקון למולטיפל-טסטינג).
        p_eff — effective p-value של Overfitting (ככל שהוא קטן יותר, טוב יותר).

    רעיון בסיסי (עם שדרוגים)
    -------------------------
    1. מחשבים t-stat ≈ Sharpe * sqrt(T).
       - אפשרות להשתמש ב-Student-t במקום Normal.
       - תיקון קטן ל-skew/kurt דרך Edgeworth-style factor.
    2. p_single ≈ P(Z > t_stat) או P(|Z| > |t_stat|) (חד/דו־צדדי).
    3. p_eff ≈ 1 - (1 - p_single)^n_strategies, מחושב עם log1p כדי לשמור יציבות נומרית.
    4. dsr ≈ Φ^{-1}(1 - p_eff), או student-t הפוך אם בחרנו Student-t.

    הערות:
    -------
    - אם Sharpe <= 0 → נניח שאין "אלפא" אמיתית, מחזירים (Sharpe, 1.0).
    - אם אין SciPy, נשתמש ב-approximation נורמלית בלבד (erf).
    """

    # --- 0) Sanitization & guards ---
    try:
        sharpe = float(sharpe)
        t = int(t)
        n_strategies = int(n_strategies)
    except Exception:
        return 0.0, 1.0

    t = max(1, t)
    if t <= 1:
        # מעט מדי תצפיות כדי לעשות משהו חכם
        return sharpe, 1.0

    if n_strategies <= 1:
        n_strategies = 1
    if max_strategies is not None and max_strategies > 0:
        n_strategies = min(n_strategies, int(max_strategies))

    # Sharpe שלילי / אפסי: אין מה "להגן" עליו — מחזירים אותו כמות שהוא
    if sharpe <= 0:
        return sharpe, 1.0

    # --- 1) t-stat base ---
    t_stat = sharpe * math.sqrt(float(t))

    # תיקון קטן ל-skew/kurt (Edgeworth-style; optional, רך מאוד)
    try:
        # עבור T גדול, התיקון קטן; עבור T קטן – קצת משמעותי יותר
        inv_sqrt_t = 1.0 / math.sqrt(float(t))
        # זה לא נוסחה מדויקת מ־Bailey, אבל correction קטן הגיוני:
        # factor ≈ 1 + skew/(6*sqrt(T)) - (kurt-3)/(24*T)
        fac = 1.0 + (skew * inv_sqrt_t) / 6.0 - (kurt - 3.0) / (24.0 * float(t))
        fac = max(0.5, min(1.5, fac))  # לא נותנים לזה להשתגע
        t_stat *= fac
    except Exception:
        pass

    # --- 2) p_single (חד/דו-צדדי) ---
    def _norm_sf(x: float) -> float:
        """Survival function של Normal(0,1) ללא SciPy."""
        # 1 - CDF(x) = 0.5 * (1 - erf(x / sqrt(2)))
        return 0.5 * (1.0 - math.erf(x / math.sqrt(2.0)))

    if sps is not None and use_student_t:
        # ננסה Student-t אם אפשר
        try:
            df_t = max(1, t - 1)
            if two_sided:
                # P(|T| > |t_stat|)
                p_single = float(2.0 * sps.t.sf(abs(t_stat), df=df_t))
            else:
                p_single = float(sps.t.sf(t_stat, df=df_t))
        except Exception:
            # fallback לנורמל
            if two_sided:
                p_single = 2.0 * _norm_sf(abs(t_stat))
            else:
                p_single = _norm_sf(t_stat)
    else:
        # Normal approx בלבד
        if two_sided:
            p_single = 2.0 * _norm_sf(abs(t_stat))
        else:
            p_single = _norm_sf(t_stat)

    # הגנת קצה
    p_single = float(min(max(p_single, 1e-16), 1.0 - 1e-16))

    # --- 3) multiple testing: p_eff ---
    # p_eff = 1 - (1 - p_single)^n_strategies
    # נחשב בצורה יציבה עם log1p
    try:
        log_one_minus = math.log1p(-p_single)  # ~ -p_single עבור p קטן
        p_eff = 1.0 - math.exp(n_strategies * log_one_minus)
    except Exception:
        # fallback פשוט
        p_eff = 1.0 - (1.0 - p_single) ** n_strategies

    p_eff = float(min(max(p_eff, 1e-16), 1.0 - 1e-16))

    # --- 4) inverse CDF (z-score מייצג DSR) ---
    if sps is not None:
        try:
            # ב-DSR בדרך כלל משתמשים בנורמל הפוך, גם אם השתמשנו ב-t ל-p_single.
            z = float(sps.norm.ppf(1.0 - p_eff))
        except Exception:
            z = sharpe
    else:
        # approx inverse CDF לנורמל אם אין SciPy
        # נשתמש ב-approximation סטנדרטית של Probit (Abramowitz & Stegun style)
        # אם זה נופל, נחזור ל-Sharpe.
        try:
            # p_eff קטן → 1 - p_eff קרוב ל-1 → נרצה quantile גבוה
            p = 1.0 - p_eff
            # approximation: https://en.wikipedia.org/wiki/Probit#Approximations
            # נשתמש בגרסה פשוטה
            a1, a2, a3 = -39.6968302866538, 220.946098424521, -275.928510446969
            a4, a5, a6 = 138.357751867269, -30.6647980661472, 2.50662827745924
            b1, b2, b3 = -54.4760987982241, 161.585836858041, -155.698979859887
            b4, b5 = 66.8013118877197, -13.2806815528857
            c1, c2, c3 = -0.00778489400243029, -0.322396458041136, -2.40075827716184
            c4, c5 = -2.54973253934373, 4.37466414146497
            c6, c7 = 2.93816398269878, 0.00778469570904146
            d1, d2, d3 = 0.00778469570904146, 0.32246712907004, 2.445134137143
            d4 = 3.75440866190742

            # break-points
            plow = 0.02425
            phigh = 1 - plow

            if p < plow:
                q = math.sqrt(-2 * math.log(p))
                z = (
                    (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
                    / ((((d1 * q + d2) * q + d3) * q + d4) * q + 1)
                )
            elif p > phigh:
                q = math.sqrt(-2 * math.log(1 - p))
                z = -(
                    (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
                    / ((((d1 * q + d2) * q + d3) * q + d4) * q + 1)
                )
            else:
                q = p - 0.5
                r = q * q
                z = (
                    (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q
                ) / (
                    ((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1
                )
            z = float(z)
        except Exception:
            z = sharpe

    # guard: דואגים שלא "נרוויח" DSR גבוה יותר מאשר Sharpe בגלוי במדידות קצרות מאוד
    if t < 30 and z > sharpe:
        z = sharpe

    return z, p_eff


# =========================
# SECTION 7.2: Walk-Forward helper (Backtester with explicit dates)
# =========================

def _run_backtest_with_dates(
    sym1: str,
    sym2: str,
    params: Dict[str, Any],
    start_date: _date_type,
    end_date: _date_type,
) -> Dict[str, float]:
    """
    מריץ Backtester על חלון זמן ספציפי (Walk-Forward).

    דורש ש-Backtester.__init__ יודע לקבל start_date/end_date, אחרת
    _sanitize_bt_kwargs יסנן אותם החוצה ואז ה-Backtester ירוץ על ברירת המחדל שלו.

    מחזיר dict ביצועים מנורמל:
        {"Sharpe", "Profit", "Drawdown", ...}

    On error → penalty profile (Sharpe<0, Profit<0, DD=1).

    שדרוגים:
    --------
    1. תומך גם ב-date וגם ב-datetime/str כ-start_date/end_date (המרה בטוחה).
    2. משתמש במיפוי opt_param_mapping מה-session (UI→Backtester).
    3. מוסיף risk_kwargs מ-get_session_risk_kwargs בצורה בטוחה.
    4. מנקה kwargs דרך _sanitize_bt_kwargs כדי להתאים לחתימת Backtester.
    5. לוג מפורט יותר (DEBUG + WARNING).
    6. הסימולציה (כשאין Backtester) לוקחת seed גלובלי ויכולה לקבל penalty פרופיל מה-session.
    7. מבטיח שהפלט תמיד כולל מפתחות "Sharpe"/"Profit"/"Drawdown" (גם אם ה-Backtester מחזיר פורמט אחר).
    8. מגביל ערכים קיצוניים (למשל DD מטורף) לכדי טווח הגיוני.
    """

    # ==== 0) Fallback לסימולציה אם אין Backtester ====
    if Backtester is None:
        seed = int(st.session_state.get("global_seed", 1337))
        rng = np.random.default_rng(seed)
        # אפשר לאפשר פרופיל penalty מה-session (אופציונלי)
        penalty_cfg = st.session_state.get("wf_penalty_profile", {}) or {}
        mean_sh = float(penalty_cfg.get("mean_sharpe", 1.0))
        std_sh = float(penalty_cfg.get("std_sharpe", 0.4))
        mean_p = float(penalty_cfg.get("mean_profit", 3000.0))
        std_p = float(penalty_cfg.get("std_profit", 800.0))
        mean_dd = float(penalty_cfg.get("mean_dd", 0.1))
        std_dd = float(penalty_cfg.get("std_dd", 0.05))

        sharpe = float(rng.normal(mean_sh, std_sh))
        profit = float(rng.normal(mean_p, std_p))
        dd = float(abs(rng.normal(mean_dd, std_dd)))
        return {"Sharpe": sharpe, "Profit": profit, "Drawdown": dd}

    # ==== 1) נרמול תאריכים (תומך ב-date/datetime/str) ====
    def _to_date(val: Any) -> _date_type:
        if isinstance(val, _date_type):
            return val
        try:
            return pd.to_datetime(val).date()
        except Exception:
            raise ValueError(f"Invalid date value: {val!r}")

    try:
        sd = _to_date(start_date)
        ed = _to_date(end_date)
    except Exception as e:
        logger.warning(
            "Walk-Forward dates invalid for %s-%s (%r→%r): %s",
            sym1,
            sym2,
            start_date,
            end_date,
            e,
        )
        return {"Sharpe": -5.0, "Profit": -1e6, "Drawdown": 1.0}

    if sd > ed:
        # אם start אחרי end → נהפוך
        sd, ed = ed, sd

    # ==== 2) מיפוי פרמטרים UI→Backtester ====
    mapping = st.session_state.get("opt_param_mapping", {}) or {}
    params_mapped = _apply_param_mapping(params or {}, mapping)

    bt_kwargs = dict(params_mapped)

    # 👇 השמות הנכונים לפי BacktestConfig / Backtester
    bt_kwargs["start"] = sd
    bt_kwargs["end"] = ed

    # 🔹 חשוב: להגיד למנוע שזה מגיע מ-SqlStore, לא מ-AUTO/YF
    bt_kwargs.setdefault("data_source", "SQL")

    bt_kwargs = _sanitize_bt_kwargs(bt_kwargs)

    # Risk kwargs
    try:
        bt_kwargs.update(get_session_risk_kwargs())
    except Exception:
        pass

    # ==== 4) הרצת Backtester עם טיפול בשגיאות ====
    try:
        bt = Backtester(symbol_a=sym1, symbol_b=sym2, **bt_kwargs)
        perf = bt.run()
        # perf יכול להיות dict/Series/אובייקט – extract_metrics מנרמל
        metrics = extract_metrics(perf)
        # קליפינג בסיסי כדי למנוע ערכים בלתי סבירים
        sh = float(metrics.get("Sharpe", 0.0))
        p = float(metrics.get("Profit", 0.0))
        dd = float(metrics.get("Drawdown", 0.0))
        # למשל DD גדול מ-10 כנראה non-sense (תלוי סקאלה, אבל עדיף לא לשבור)
        if abs(dd) > 10.0:
            dd = float(np.sign(dd) * 10.0)
        out = {"Sharpe": sh, "Profit": p, "Drawdown": dd}
        logger.debug(
            "WF segment %s-%s (%s→%s) -> Sharpe=%.3f, Profit=%.1f, DD=%.3f",
            sym1,
            sym2,
            sd,
            ed,
            out["Sharpe"],
            out["Profit"],
            out["Drawdown"],
        )
        return out
    except Exception as e:
        logger.warning(
            "Walk-Forward backtest error for %s-%s (%s→%s): %s",
            sym1,
            sym2,
            sd,
            ed,
            e,
        )
        return {"Sharpe": -5.0, "Profit": -1e6, "Drawdown": 1.0}

def _run_walkforward_for_params(
    sym1: str,
    sym2: str,
    params: Dict[str, Any],
    *,
    start_date: _date_type,
    end_date: _date_type,
    n_splits: int = 4,
    weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    מריץ Walk-Forward על פרמטרים נתונים לאורך n_splits חלונות זמן רציפים.

    החזרה: DataFrame עם שורות לכל חלון:
        - split_id
        - segment_label (אופציונלי, למשל "1/4")
        - start_date, end_date
        - n_days
        - Sharpe, Profit, Drawdown
        - Score (לפי compute_score)
        - DSR (Deflated Sharpe Ratio, אם אפשר)
        - p_overfit (p_eff לפי DSR)

    שדרוגים:
    --------
    1. שימוש ב-min_segment_days מתוך session_state (opt_wf_min_days) במקום 10 קבוע.
    2. התאמת n_splits אפקטיבי לגודל החלון (לא מריץ WF על חלונות גרועים).
    3. חישוב n_days לכל חלון, לטובת דיאגנוסטיקה.
    4. שימוש ב-deflated_sharpe_ratio (אם זמין) כדי לתת DSR ו-p_overfit לכל חלון.
    5. שימוש ב-weights מה-session (loaded_weights_eff) אם לא הועברו מפורשות.
    6. הגנה על empty / מעט מדי ימים.
    7. החזרת DataFrame מסודר, מתאים לגרפים/דוחות.
    """

    # ==== 0) נרמול n_splits ו-min_segment_days ====
    if n_splits < 1:
        n_splits = 1

    try:
        idx = pd.date_range(start=start_date, end=end_date, freq="D")
    except Exception as e:
        logger.warning("WF date_range failed for %s-%s: %s", sym1, sym2, e)
        return pd.DataFrame()

    if idx.empty:
        return pd.DataFrame()

    # אפשר להגדיר מינימום ימים per segment מה-session
    min_seg_days = int(st.session_state.get("opt_wf_min_days", 10))
    min_seg_days = max(5, min_seg_days)

    # מתאימים את מספר החלונות כך שיהיו לפחות min_seg_days לכל חלון
    max_splits_reasonable = max(1, len(idx) // min_seg_days)
    n_splits_eff = max(1, min(int(n_splits), max_splits_reasonable))

    if len(idx) < n_splits_eff * min_seg_days:
        # מעט מדי ימים לחלוקה סבירה
        logger.debug(
            "WF: not enough days for %d splits of %d days for %s-%s (len=%d)",
            n_splits_eff,
            min_seg_days,
            sym1,
            sym2,
            len(idx),
        )
        return pd.DataFrame()

    try:
        splits = np.array_split(idx, n_splits_eff)
    except Exception as e:
        logger.warning("WF: np.array_split failed for %s-%s: %s", sym1, sym2, e)
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    use_weights = weights or st.session_state.get(
        "loaded_weights_eff",
        {"Sharpe": 0.5, "Profit": 0.3, "Drawdown": 0.2},
    )

    # ==== 1) לולאה על החלונות ====
    for i, s in enumerate(splits, start=1):
        if s.size == 0:
            continue
        sd = s[0].date()
        ed = s[-1].date()
        n_days = int((s[-1] - s[0]).days + 1)

        # להריץ backtest על הסגמנט
        perf = _run_backtest_with_dates(sym1, sym2, params, sd, ed)

        # compute_score משתמש ב-extract_metrics פנימית, אבל כבר יש לנו perf dict
        norm, score = compute_score(perf, use_weights)

        # Deflated Sharpe Ratio per segment (approx)
        sh = float(perf.get("Sharpe") or 0.0)
        # t אפקטיבי: אפשר להשתמש ב-n_days או ערך מה-session (opt_wf_t_per_day)
        t_per_day = float(st.session_state.get("opt_wf_t_per_day", 1.0))
        t_eff = max(1, int(round(n_days * t_per_day)))
        n_strats = n_splits_eff  # מספר החלונות כאומדן למספר אסטרטגיות שנבדקו

        try:
            dsr, p_eff = deflated_sharpe_ratio(
                sh,
                t_eff,
                n_strats,
                skew=0.0,
                kurt=3.0,
                two_sided=False,
                use_student_t=True,
                max_strategies=1000,
            )
        except Exception:
            dsr, p_eff = sh, 1.0

        row = {
            "split_id": i,
            "segment_label": f"{i}/{n_splits_eff}",
            "start_date": sd,
            "end_date": ed,
            "n_days": n_days,
            "Sharpe": perf.get("Sharpe"),
            "Profit": perf.get("Profit"),
            "Drawdown": perf.get("Drawdown"),
            "Score": score,
            "DSR": dsr,
            "p_overfit": p_eff,
        }
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)

def _opt_resolve_pair_with_reverse_for_replay(
    store: Any, 
    pair: str,
    *,
    limit: int = 1,
    only_complete: bool = False,
) -> Tuple[str, pd.DataFrame]:
    """
    מנסה למצוא trials עבור pair כמו שהוא, ואם אין – עבור ההיפוך.

    מחזיר:
        pair_used – הכיוון שבו באמת נמצא trial
        df_best   – DataFrame מ-get_best_trials_for_pair
    """
    sym_a, sym_b = pair.split("-")

    df_direct = store.get_best_trials_for_pair(
        pair,
        limit=limit,
        only_complete=only_complete,
    )
    if not df_direct.empty:
        logger.info(
            "ReplayTab: using direct pair=%s (found %s trials, top score=%s)",
            pair,
            len(df_direct),
            float(df_direct["score"].max()) if "score" in df_direct.columns else None,
        )
        return pair, df_direct

    rev_pair = f"{sym_b}-{sym_a}"
    df_rev = store.get_best_trials_for_pair(
        rev_pair,
        limit=limit,
        only_complete=only_complete,
    )
    if not df_rev.empty:
        logger.info(
            "ReplayTab: direct pair=%s empty, reverse=%s has %s trials (top score=%s)",
            pair,
            rev_pair,
            len(df_rev),
            float(df_rev["score"].max()) if "score" in df_rev.columns else None,
        )
        st.info(
            f"לא נמצאו trials עבור `{pair}`, "
            f"אבל כן נמצאו עבור `{rev_pair}` – נשתמש בהם."
        )
        return rev_pair, df_rev

    st.warning(
        f"⚠️ לא נמצאו trials ב-DuckDB עבור `{pair}` או עבור הכיוון ההפוך."
    )
    return pair, df_direct

# =========================
# SECTION 7.3: Main Analytics Hub for the Optimisation tab
# =========================

def _render_optimization_analytics_section(TABLE_HEIGHT: int, df: pd.DataFrame, sym1: str, sym2: str) -> None:
    """
    Analytics Hub — חלק "מחקר" של טאב האופטימיזציה.

    כולל:
    - Pareto Frontier (Score/Sharpe/DD) + גרפים + artifact ל-DuckDB.
    - DSR & Overfitting diagnostics (Top-K אסטרטגיות).
    - Walk-Forward Analysis & Robustness (per strategy) + artifact.

    df: opt_df / opt_df_batch עבור זוג אחד (או Batch במידת הצורך).
    """
    if df is None or df.empty:
        st.caption("Analytics: no results yet.")
        return

    pair_label = f"{sym1}-{sym2}"
    FOCUS = bool(st.session_state.get("opt_focus_mode", False))

    # ------------------ 8.1 Pareto Frontier ------------------
    with st.expander(
        "1️⃣ Pareto Frontier — Score / Sharpe / Drawdown",
        expanded=not FOCUS,
    ):
        try:
            score_col = "Score" if "Score" in df.columns else None
            sharpe_col = "Sharpe" if "Sharpe" in df.columns else None
            dd_col = "Drawdown" if "Drawdown" in df.columns else None
            if not (score_col and sharpe_col and dd_col):
                st.caption("Need Score / Sharpe / Drawdown columns for Pareto.")
            else:
                # סף שיפור מינימלי ודילול נקודות (רעיון חדש)
                col_ctrl1, col_ctrl2 = st.columns(2)
                with col_ctrl1:
                    min_improvement = st.number_input(
                        "Sensitivity (min improvement, Δ)",
                        min_value=0.0,
                        max_value=0.5,
                        value=0.01,
                        step=0.01,
                        key=sk("pareto_min_impr"),
                        help="Δ מינימלי כדי להחשיב נקודה כדומיננטית משמעותית; 0 → ללא סף.",
                    )
                with col_ctrl2:
                    max_points = st.number_input(
                        "Max Pareto points (UI subset)",
                        min_value=10,
                        max_value=500,
                        value=100,
                        step=10,
                        key=sk("pareto_max_points"),
                        help="לצורך UI בלבד – הסף לא משפיע על חישובים פנימיים אחרים.",
                    )

                pareto_df = _compute_pareto_df(
                    df,
                    score_col=score_col,
                    sharpe_col=sharpe_col,
                    dd_col=dd_col,
                    max_points=int(max_points),
                    min_improvement=float(min_improvement),
                )
                if pareto_df.empty:
                    st.caption("No Pareto frontier (or insufficient data).")
                else:
                    st.caption(
                        f"Non-dominated points: {len(pareto_df)} / {len(df)} "
                        f"({len(pareto_df) / max(1, len(df)):.1%})"
                    )
                    st.dataframe(
                        pareto_df.head(50),
                        width="stretch",
                        height=min(TABLE_HEIGHT, 420),
                    )

                    # סטטיסטיקה מהירה על Pareto (רעיון חדש)
                    try:
                        sh_series = pd.to_numeric(pareto_df[sharpe_col], errors="coerce")
                        sc_series = pd.to_numeric(pareto_df[score_col], errors="coerce")
                        dd_series = pd.to_numeric(pareto_df[dd_col], errors="coerce").abs()
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Pareto Sharpe (median)", f"{sh_series.median():.2f}")
                        c2.metric("Pareto Score (median)", f"{sc_series.median():.3f}")
                        c3.metric("Pareto DD (median)", f"{dd_series.median():.3f}")
                    except Exception:
                        pass

                    # Scatter DD vs Sharpe בצבע Score
                    try:
                        if px is not None:
                            fig_p = px.scatter(
                                pareto_df,
                                x=dd_col,
                                y=sharpe_col,
                                color=score_col,
                                title="Pareto Frontier — Sharpe vs Drawdown (colored by Score)",
                                hover_data=pareto_df.columns,
                            )
                            st.plotly_chart(fig_p, width = "stretch")
                    except Exception:
                        pass

                    # 3D scatter: Score / Sharpe / DD
                    try:
                        if px is not None:
                            fig3d = px.scatter_3d(
                                pareto_df,
                                x=dd_col,
                                y=sharpe_col,
                                z=score_col,
                                color=score_col,
                                title="Pareto Frontier — 3D View (Score / Sharpe / DD)",
                                hover_data=pareto_df.columns,
                            )
                            st.plotly_chart(fig3d, width = "stretch")
                    except Exception:
                        pass

                    # Artifact ל-DuckDB
                    try:
                        last_study_id = st.session_state.get("opt_last_study_id")
                        if last_study_id is not None:
                            payload = pareto_df.to_csv(index=False).encode("utf-8")
                            save_artifact_to_duck(int(last_study_id), kind="pareto_csv", payload=payload)
                    except Exception as e:
                        logger.debug("Saving Pareto artifact failed for %s: %s", pair_label, e)

                    # הורדה ל-CSV
                    st.download_button(
                        "Download Pareto set (CSV)",
                        data=pareto_df.to_csv(index=False).encode("utf-8"),
                        file_name=f"pareto_frontier_{pair_label}.csv",
                        mime="text/csv",
                        key=sk("dl_pareto_frontier"),
                    )
        except Exception as e:
            st.caption(f"Pareto analytics failed: {e}")

    # ------------------ 8.2 Deflated Sharpe & Overfitting ------------------
    with st.expander(
        "2️⃣ Deflated Sharpe Ratio (DSR) & Overfitting Diagnostics",
        expanded=not FOCUS,
    ):
        try:
            score_col = "Score" if "Score" in df.columns else None
            sharpe_col = "Sharpe" if "Sharpe" in df.columns else None
            if sharpe_col is None:
                st.caption("No Sharpe column found; cannot compute DSR.")
            else:
                # N overall strategies (אפשר לשנות ידנית, רעיון חדש)
                n_strategies_total = len(df)

                c_cfg1, c_cfg2, c_cfg3 = st.columns(3)
                with c_cfg1:
                    k = st.slider(
                        "Top-K strategies",
                        min_value=5,
                        max_value=min(200, len(df)),
                        value=min(20, len(df)),
                        step=5,
                        key=sk("dsr_topk"),
                    )
                with c_cfg2:
                    n_strats_mode = st.selectbox(
                        "Multiple-testing N:",
                        ["Use total rows", "Manual"],
                        index=0,
                        key=sk("dsr_n_mode"),
                    )
                with c_cfg3:
                    use_student_t = st.checkbox(
                        "Use Student-t (if available)",
                        value=True,
                        key=sk("dsr_use_t"),
                    )

                if n_strats_mode == "Manual":
                    n_strategies_total = int(
                        st.number_input(
                            "n_strategies (manual)",
                            min_value=1,
                            max_value=100_000,
                            value=len(df),
                            step=1,
                            key=sk("dsr_n_manual"),
                        )
                    )

                if score_col is not None:
                    top_df = df.sort_values(score_col, ascending=False).head(int(k)).copy()
                else:
                    top_df = df.sort_values(sharpe_col, ascending=False).head(int(k)).copy()

                # הערכת T (sample size)
                trade_col = None
                for cand in ("Trades", "trades", "n_trades"):
                    if cand in df.columns:
                        trade_col = cand
                        break
                if trade_col:
                    n_trades_vec = pd.to_numeric(top_df[trade_col], errors="coerce").fillna(100.0)
                else:
                    n_trades_vec = pd.Series(100.0, index=top_df.index)

                rows_dsr: List[Dict[str, Any]] = []
                for idx_row, row in top_df.iterrows():
                    sh = float(pd.to_numeric(row[sharpe_col], errors="coerce"))
                    t_eff = int(max(1, n_trades_vec.loc[idx_row]))
                    dsr, p_eff = deflated_sharpe_ratio(
                        sh,
                        t=t_eff,
                        n_strategies=n_strategies_total,
                        skew=0.0,
                        kurt=3.0,
                        two_sided=False,
                        use_student_t=use_student_t,
                        max_strategies=1000,
                    )

                    # Classification לפי DSR/p_overfit (רעיון חדש)
                    if dsr >= 2.0 and p_eff < 0.01:
                        label = "Strong"
                    elif dsr >= 1.0 and p_eff < 0.05:
                        label = "OK"
                    elif dsr >= 0.5 and p_eff < 0.10:
                        label = "Weak"
                    else:
                        label = "Noise"

                    rows_dsr.append(
                        {
                            "Sharpe": sh,
                            "Trades": t_eff,
                            "DSR": dsr,
                            "p_overfit": p_eff,
                            "Class": label,
                        }
                    )

                dsr_df = pd.DataFrame(rows_dsr)
                if dsr_df.empty:
                    st.caption("No DSR entries (empty top-K).")
                else:
                    st.dataframe(
                        dsr_df,
                        width="stretch",
                        height=min(TABLE_HEIGHT, 420),
                    )

                    # מדד סיכום: כמה חזקים / OK / Weak / Noise
                    class_counts = dsr_df["Class"].value_counts()
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Strong", f"{class_counts.get('Strong', 0)}")
                    c2.metric("OK", f"{class_counts.get('OK', 0)}")
                    c3.metric("Noise/Weak", f"{class_counts.get('Weak', 0) + class_counts.get('Noise', 0)}")

                    # threshold
                    thr = st.number_input(
                        "DSR threshold (Strong/OK)",
                        0.0,
                        5.0,
                        1.0,
                        0.1,
                        key=sk("dsr_thr"),
                    )
                    n_pass = int((dsr_df["DSR"] >= thr).sum())
                    st.caption(f"{n_pass}/{len(dsr_df)} strategies have DSR ≥ {thr:.2f}")

                    # Histogram ל-DSR
                    try:
                        if px is not None and not dsr_df.empty:
                            fig_dsr = px.histogram(
                                dsr_df,
                                x="DSR",
                                nbins=30,
                                title="DSR Distribution (Top-K)",
                            )
                            st.plotly_chart(fig_dsr, width = "stretch")
                    except Exception:
                        pass

                    # Scatter DSR vs Sharpe (רעיון חדש)
                    try:
                        if px is not None:
                            fig_corr = px.scatter(
                                dsr_df,
                                x="Sharpe",
                                y="DSR",
                                color="Class",
                                title="Sharpe vs DSR (Top-K)",
                            )
                            st.plotly_chart(fig_corr, width = "stretch")
                    except Exception:
                        pass

                    # Artifact ל-DuckDB
                    try:
                        last_study_id = st.session_state.get("opt_last_study_id")
                        if last_study_id is not None:
                            payload = dsr_df.to_csv(index=False).encode("utf-8")
                            save_artifact_to_duck(int(last_study_id), kind="dsr_csv", payload=payload)
                    except Exception as e:
                        logger.debug("Saving DSR artifact failed for %s: %s", pair_label, e)

                    # הורדה ל-CSV
                    st.download_button(
                        "Download DSR table (CSV)",
                        data=dsr_df.to_csv(index=False).encode("utf-8"),
                        file_name=f"dsr_{pair_label}.csv",
                        mime="text/csv",
                        key=sk("dsr_dl_csv"),
                    )

        except Exception as e:
            st.caption(f"DSR diagnostics failed: {e}")

    # ------------------ 8.3 Walk-Forward & Robustness ------------------
    with st.expander(
        "3️⃣ Walk-Forward & Robustness (per strategy)",
        expanded=not FOCUS,
    ):
        try:
            if Backtester is None:
                st.caption("Backtester not available — cannot run Walk-Forward.")
            else:
                ctx_dict = st.session_state.get("ctx", {}) or {}
                sd_ctx = ctx_dict.get("start_date")
                ed_ctx = ctx_dict.get("end_date")
                if not (sd_ctx and ed_ctx):
                    st.caption("No start/end dates found in ctx; Walk-Forward not available.")
                else:
                    # המרה ל-date
                    if not isinstance(sd_ctx, _date_type):
                        sd_ctx = pd.to_datetime(sd_ctx).date()
                    if not isinstance(ed_ctx, _date_type):
                        ed_ctx = pd.to_datetime(ed_ctx).date()

                    n_splits = int(
                        st.number_input(
                            "Number of splits (walk-forward)",
                            min_value=2,
                            max_value=12,
                            value=4,
                            step=1,
                            key=sk("wf_n_splits"),
                        )
                    )
                    topk = st.slider(
                        "Top-K strategies to test (by Score)",
                        min_value=1,
                        max_value=min(20, len(df)),
                        value=min(5, len(df)),
                        step=1,
                        key=sk("wf_topk"),
                    )

                    # בחירת שורות ל-WF
                    if "Score" in df.columns:
                        df_top = df.sort_values("Score", ascending=False).head(int(topk)).copy()
                    else:
                        df_top = df.head(int(topk)).copy()

                    # עמודות פרמטרים = כל מה שלא מטריקות
                    metric_like = set(
                        c for c in df.columns
                        if METRIC_KEYS.get(str(c).lower()) or str(c) in ("Score", "Pair", "study_id")
                    )
                    param_cols = [
                        c for c in df.columns
                        if c not in metric_like and pd.api.types.is_numeric_dtype(df[c])
                    ]
                    if not param_cols:
                        st.caption("No parameter columns detected for Walk-Forward.")
                    else:
                        wf_results: List[pd.DataFrame] = []
                        for i, (_, row) in enumerate(df_top.iterrows(), start=1):
                            params = {c: row[c] for c in param_cols}
                            wf_df = _run_walkforward_for_params(
                                sym1,
                                sym2,
                                params,
                                start_date=sd_ctx,
                                end_date=ed_ctx,
                                n_splits=n_splits,
                            )
                            if wf_df.empty:
                                continue
                            wf_df["strategy_rank"] = i
                            wf_results.append(wf_df)

                        if not wf_results:
                            st.caption("No Walk-Forward results (insufficient data or Backtester limitations).")
                        else:
                            wf_all = pd.concat(wf_results, ignore_index=True)
                            st.dataframe(
                                wf_all,
                                width="stretch",
                                height=min(TABLE_HEIGHT, 420),
                            )

                            # Summary KPIs per strategy rank
                            try:
                                grp = wf_all.groupby("strategy_rank").agg(
                                    Score_mean=("Score", "mean"),
                                    Score_std=("Score", "std"),
                                    Score_min=("Score", "min"),
                                    Score_max=("Score", "max"),
                                    DSR_mean=("DSR", "mean"),
                                    p_overfit_mean=("p_overfit", "mean"),
                                )
                                st.caption("Walk-Forward Score/DSR stats per strategy rank:")
                                st.dataframe(
                                    grp,
                                    width="stretch",
                                    height=min(TABLE_HEIGHT, 300),
                                )
                            except Exception:
                                pass

                            # Heatmap: Score by split × rank
                            try:
                                if px is not None:
                                    pivot = wf_all.pivot_table(
                                        index="split_id",
                                        columns="strategy_rank",
                                        values="Score",
                                        aggfunc="mean",
                                    )
                                    fig_hm = px.imshow(
                                        pivot,
                                        aspect="auto",
                                        origin="lower",
                                        labels=dict(
                                            x="strategy_rank",
                                            y="split_id",
                                            color="Score",
                                        ),
                                        title="Walk-Forward Score heatmap (split × strategy_rank)",
                                    )
                                    st.plotly_chart(fig_hm, width = "stretch")
                            except Exception:
                                pass

                            # Artifact ל-DuckDB
                            try:
                                last_study_id = st.session_state.get("opt_last_study_id")
                                if last_study_id is not None:
                                    payload = wf_all.to_csv(index=False).encode("utf-8")
                                    save_artifact_to_duck(int(last_study_id), kind="walkforward_csv", payload=payload)
                            except Exception as e:
                                logger.debug("Saving Walk-Forward artifact failed for %s: %s", pair_label, e)

                            # הורדה ל-CSV
                            st.download_button(
                                "Download Walk-Forward results (CSV)",
                                data=wf_all.to_csv(index=False).encode("utf-8"),
                                file_name=f"walkforward_{pair_label}.csv",
                                mime="text/csv",
                                key=sk("wf_dl_csv"),
                            )

                            # המלצה מילולית קטנה (רעיון חדש)
                            try:
                                dsr_mean = float(wf_all["DSR"].mean())
                                sh_mean = float(wf_all["Sharpe"].mean())
                                if dsr_mean >= 1.5 and sh_mean >= 1.0:
                                    rec = "Walk-Forward נראה חזק ומייצב את האסטרטגיה."
                                elif dsr_mean >= 1.0:
                                    rec = "Walk-Forward בינוני – כדאי לבדוק עוד חלונות / פרופילים."
                                else:
                                    rec = "Walk-Forward חלש – קיים סיכון Overfit משמעותי."
                                st.caption(f"📌 WF summary: {rec}")
                            except Exception:
                                pass

        except Exception as e:
            st.caption(f"Walk-Forward analytics failed: {e}")



def _render_replay_best_trial_panel(TABLE_HEIGHT: int, df: pd.DataFrame, sym1: str, sym2: str) -> None:
    """
    פאנל קטן בתוך טאב האופטימיזציה שעושה:

    1. בחירת pair + טווח תאריכים.
    2. שליפת best trial מ-DuckDB (SqlStore.trials).
    3. הרצת Backtester אמיתי (data_source="SQL") עם אותם params.
    4. הצגת ביצועים אמיתיים + params עיקריים.
    """
    with st.expander("🎯 Replay best trial על הדאטה האמיתי (SqlStore + IBKR)", expanded=False):
        # הגנות בסיסיות
        if Backtester is None:
            st.warning("Backtester (OptimizationBacktester) לא נטען – אי אפשר להריץ backtest.")
            return

        # ===== 1) בחירת זוג + טווח תאריכים =====
        default_pair = f"{sym1}-{sym2}" if sym1 and sym2 else ""
        pair = st.text_input(
            "Pair (פורמט SYM1-SYM2, למשל XLY-XLP / BITO-BKCH)",
            value=default_pair,
            key="replay_pair_input",
        ).strip().upper()

        if "-" not in pair:
            st.info("הכנס pair בפורמט 'SYM1-SYM2' כדי להתחיל.")
            return

        c1, c2, c3 = st.columns(3)
        use_dates = c1.checkbox("להשתמש בטווח תאריכים ידני", value=True, key="replay_use_dates")

        start_date = end_date = None
        if use_dates:
            ctx = st.session_state.get("ctx", {}) or {}
            ctx_start = ctx.get("start_date")
            ctx_end = ctx.get("end_date")

            if isinstance(ctx_start, (datetime, date)):
                default_start = ctx_start
            else:
                default_start = date(2010, 1, 1)

            if isinstance(ctx_end, (datetime, date)):
                default_end = ctx_end
            else:
                default_end = date.today()

            start_date = c2.date_input(
                "Start date",
                value=default_start,
                key="replay_start_date",
            )
            end_date = c3.date_input(
                "End date",
                value=default_end,
                key="replay_end_date",
            )

        only_complete = st.checkbox(
            "רק trials במצב COMPLETE (אם state נשמר)",
            value=False,
            key="replay_only_complete",
        )
        limit = st.number_input(
            "כמה best trials להביא (ניקח את הראשון בלבד להרצה):",
            min_value=1,
            max_value=20,
            value=1,
            step=1,
            key="replay_limit",
        )

        # ===== 2) כפתור הרצה =====
        run_btn = st.button("🚀 Run Replay Best Trial", key="replay_run_btn")

        if not run_btn:
            return

        # ===== 3) SqlStore read_only =====
        try:
            env_settings: Dict[str, Any] = {}
            # אם יש SQL_STORE_URL – SqlStore.from_settings כבר יודע להשתמש
            if "SQL_STORE_URL" in os.environ:
                env_settings["engine_url"] = os.environ["SQL_STORE_URL"]

            store = SqlStore.from_settings(env_settings, read_only=True)
        except Exception as e:
            st.error(f"SqlStore.from_settings נכשל: {e}")
            logger.exception("ReplayTab: SqlStore init failed")
            return

        # ===== 4) להביא best trial (כולל reverse אם צריך) =====
        pair_used, df_best = _opt_resolve_pair_with_reverse_for_replay(
            store=store,
            pair=pair,
            limit=int(limit),
            only_complete=only_complete,
        )

        if df_best.empty:
            st.warning(f"⚠️ אין רשומות ב-trials עבור `{pair}` (וגם לא עבור הכיוון ההפוך).")
            return

        row = df_best.iloc[0]
        best_params: Dict[str, Any] = row.get("params") or {}
        best_perf_meta: Dict[str, Any] = row.get("perf") or {}

        # ===== 5) הצגת ה-trial שנבחר =====
        st.markdown("##### ✅ Best trial שנבחר מה-DuckDB")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Pair used", pair_used)
        c2.metric("Study ID", f"{int(row['study_id'])}")
        c3.metric("Trial no.", f"{int(row['trial_no'])}")
        c4.metric(
            "Score",
            f"{float(row['score']):.6f}" if "score" in row and row["score"] is not None else "N/A",
        )

        st.caption(
            f"Sampler={row.get('sampler')} | n_trials={row.get('n_trials')} | "
            f"timeout_sec={row.get('timeout_sec')} | state={row.get('state')}"
        )

        # פרמטרים עיקריים
        if best_params:
            with st.expander("🔍 פרמטרים עיקריים מה-trial (preview)", expanded=True):
                keys_of_interest = [
                    "z_entry",
                    "z_exit",
                    "lookback",
                    "half_life",
                    "mean_reversion_speed",
                    "hurst",
                    "hedge_ratio",
                    "ADF_pval",
                    "adf_tstat",
                    "beta_OLS",
                    "beta_kalman_vol",
                    "spread_mean",
                    "spread_std",
                ]
                rows_preview = []
                for k in keys_of_interest:
                    if k in best_params:
                        rows_preview.append({"param": k, "value": best_params.get(k)})
                if rows_preview:
                    st.table(pd.DataFrame(rows_preview))
                else:
                    st.write("אין פרמטרים עיקריים מוכרים – עדיין אפשר להריץ backtest מלא.")

        # ===== 6) להריץ Backtester אמיתי עם data_source="SQL" =====
        st.markdown("##### 📈 מריץ Backtest אמיתי על מחירים מ-SqlStore (IBKR/duckdb)")

        sym_a, sym_b = pair_used.split("-")

        try:
            bt = Backtester(
                symbol_a=sym_a,
                symbol_b=sym_b,
                start=start_date,
                end=end_date,
                data_source="SQL",          # 🔥 מקור מחירים: SqlStore (IBKR/duckdb)
                **best_params,              # כל הפרמטרים מה-Optuna
            )
        except TypeError as te:
            st.error(f"יצירת Backtester נכשלה (TypeError): {te}")
            st.caption("בדוק אם אחד השמות ב-best_params מתנגש עם פרמטר קונסטרקטור.")
            logger.exception("ReplayTab: Backtester init failed")
            return
        except Exception as e:
            st.error(f"יצירת Backtester נכשלה: {e}")
            logger.exception("ReplayTab: Backtester init failed")
            return

        with st.spinner("מריץ pipeline מקצועי (run) על הדאטה האמיתי..."):
            try:
                perf_dict = bt.run()  # אצלך run() כבר זה ה־professional+
            except Exception as e:
                st.error(f"❌ Professional backtest failed: {e}")
                logger.exception("ReplayTab: professional backtest failed")
                return

        # ===== 7) הצגת התוצאות האמיתיות =====
        st.markdown("#### ✅ Backtest performance (אמיתי, מ-SqlStore/IBKR)")

        if isinstance(perf_dict, dict):
            perf_df = pd.DataFrame(
                [{"metric": k, "value": perf_dict[k]} for k in sorted(perf_dict.keys())]
            )
            st.table(perf_df)
        else:
            st.write(perf_dict)

        # השוואה לפורמט מה-Optuna אם קיים perf_meta
        if best_perf_meta:
            st.markdown("#### 📊 Perf meta מה-Optuna (אם נשמר ב-perf_json)")
            meta_df = pd.DataFrame(
                [{"metric": k, "value": best_perf_meta[k]} for k in sorted(best_perf_meta.keys())]
            )
            st.table(meta_df)

def _render_replay_from_optdf_panel(TABLE_HEIGHT: int, df: pd.DataFrame, sym1: str, sym2: str) -> None:
    """
    פאנל שני: עושה Replay ל-*שורה נבחרת* מתוך opt_df (לאו דווקא מה-DuckDB).

    לוגיקה:
    --------
    1. בוחרים row (אינדקס) מתוך df.
    2. מחלצים ממנו פרמטרים עם _extract_params_from_opt_row.
    3. מריצים Backtester אמיתי עם data_source="SQL" על אותו pair.
    4. מציגים ביצועים אמיתיים.
    """
    with st.expander("🎛 Replay על שורה נבחרת מ־opt_df (אותם פרמטרים על דאטה אמיתי)", expanded=False):
        if Backtester is None:
            st.warning("Backtester (OptimizationBacktester) לא נטען – אי אפשר להריץ backtest.")
            return

        if df is None or df.empty:
            st.info("אין opt_df טעון כרגע – תריץ אופטימיזציה קודם.")
            return

        # Pair ברירת מחדל מהטאב (sym1-sym2)
        default_pair = f"{sym1}-{sym2}" if sym1 and sym2 else ""
        pair = st.text_input(
            "Pair (SYM1-SYM2) לשימוש בריפליי",
            value=default_pair,
            key="replay_opt_pair_input",
        ).strip().upper()

        if "-" not in pair:
            st.info("הכנס pair בפורמט 'SYM1-SYM2' כדי להתחיל.")
            return

        # בחירת אינדקס מה-DataFrame
        indices = list(df.index)
        if not indices:
            st.info("אין אינדקסים ב-opt_df.")
            return

        idx_selected = st.selectbox(
            "בחר שורה (index) מתוך opt_df להרצה",
            indices,
            key="replay_opt_row_index",
        )

        # טווח תאריכים
        c1, c2, c3 = st.columns(3)
        use_dates = c1.checkbox("להשתמש בטווח תאריכים ידני", value=True, key="replay_opt_use_dates")

        start_date = end_date = None
        if use_dates:
            ctx = st.session_state.get("ctx", {}) or {}
            ctx_start = ctx.get("start_date")
            ctx_end = ctx.get("end_date")

            if isinstance(ctx_start, (datetime, date)):
                default_start = ctx_start
            else:
                default_start = date(2010, 1, 1)

            if isinstance(ctx_end, (datetime, date)):
                default_end = ctx_end
            else:
                default_end = date.today()

            start_date = c2.date_input(
                "Start date",
                value=default_start,
                key="replay_opt_start_date",
            )
            end_date = c3.date_input(
                "End date",
                value=default_end,
                key="replay_opt_end_date",
            )

        run_btn = st.button("🚀 Run Replay for this row", key="replay_opt_run_btn")

        if not run_btn:
            return

        row = df.loc[idx_selected]
        params = _extract_params_from_opt_row(row)

        # להצגה: פרמטרים עיקריים (אבל הפעם מתוך ה־row)
        with st.expander("🔍 פרמטרים שנמשכו מהשורה", expanded=True):
            preview_rows = [{"param": k, "value": params[k]} for k in sorted(params.keys())]
            st.table(pd.DataFrame(preview_rows))

        sym_a, sym_b = pair.split("-")

        # בניית ה-Backtester
        try:
            bt = Backtester(
                symbol_a=sym_a,
                symbol_b=sym_b,
                start=start_date,
                end=end_date,
                data_source="SQL",      # מחירים אמיתיים מה-SqlStore/IBKR
                **params,               # כל הפרמטרים מהשורה
            )
        except TypeError as te:
            st.error(f"יצירת Backtester נכשלה (TypeError): {te}")
            st.caption("נראה שאחד השדות ב-row מתנגש עם פרמטר קונסטרקטור (למשל 'symbol_a' וכו').")
            logger.exception("Replay from opt_df: Backtester init failed")
            return
        except Exception as e:
            st.error(f"יצירת Backtester נכשלה: {e}")
            logger.exception("Replay from opt_df: Backtester init failed")
            return

        # הרצה
        with st.spinner(f"מריץ Backtest על {pair} עם פרמטרים מהשורה {idx_selected}..."):
            try:
                perf_dict = bt.run()
            except Exception as e:
                st.error(f"❌ Professional backtest failed: {e}")
                logger.exception("Replay from opt_df: professional backtest failed")
                return

        st.markdown("#### ✅ ביצועים אמיתיים (מ-SqlStore/IBKR) עבור השורה שנבחרה")
        if isinstance(perf_dict, dict):
            perf_df = pd.DataFrame(
                [{"metric": k, "value": perf_dict[k]} for k in sorted(perf_dict.keys())]
            )
            st.table(perf_df)
        else:
            st.write(perf_dict)
           
"""
חלק 9/15 — Advanced Analytics & Labs (Pro)
==========================================

Panels מתקדמים מעל opt_df:
- Feature Importance & SHAP
- Clusters & PCA
- Risk & Costs Lab
- Data Quality & Coverage
- Feature Lab (Scaling / Polynomial / Interactions)
- ML Pro — Surrogate Model & Validation
- Anomaly Detection (z-score outliers)
- Param Sweep & Surrogate Suggestions
- Actionable Recommendations

הכל נקרא דרך:
    _render_optimization_extra_sections(TABLE_HEIGHT, df, sym1, sym2)
מהטאב הראשי (חלק 7).
"""

def _render_optimization_extra_sections(TABLE_HEIGHT: int, df: pd.DataFrame, sym1: str, sym2: str) -> None:
    if df is None or df.empty:
        st.caption("Extra analytics: no data in opt_df yet.")
        return

    # זיהוי עמודות פרמטרים לעומת מטריקות
    metric_cols = [c for c in df.columns if METRIC_KEYS.get(str(c).lower()) or str(c) in ("Score",)]
    param_cols = [
        c for c in df.columns
        if c not in metric_cols and pd.api.types.is_numeric_dtype(df[c])
    ]

    if not param_cols:
        st.caption("No numeric parameter columns detected for advanced analytics.")
        return
    skip_heavy = not bool(getattr(SETTINGS, "enable_heavy_panels", True))
    heavy_hint = "Heavy analysis skipped by default (enable via settings or activate here)."


    # ============================================================
    # 9.0 Meta-Optimization Ensemble Scoring (per opt_df)
    # ============================================================
    with st.expander("🧮 Meta-Optimization Ensemble Scoring (per opt_df)", expanded=False):
        if meta_optimize is None:
            st.caption("meta_optimizer module (core.meta_optimizer) not available – cannot run meta-scoring.")
        else:
            try:
                # טווחים פעילים (מגדירים אילו עמודות הם פרמטרים)
                ranges_cfg = st.session_state.get("active_ranges", {})
                if not ranges_cfg:
                    st.caption("No active_ranges in session – Meta-Optimization uses them to know which columns are parameters.")
                else:
                    # משקולות מה-sidebar (Sharpe/Profit/DD/Sortino/Calmar/WinRate)
                    weights_eff = st.session_state.get(
                        "loaded_weights_eff",
                        {"Sharpe": 0.4, "Profit": 0.3, "Drawdown": 0.2, "Sortino": 0.05, "Calmar": 0.05},
                    )

                    # בניית spec של מטריקות לפי שמות העמודות ב-df (האותיות בדיוק כמו בעמודות)
                    metrics_spec: Dict[str, Dict[str, Any]] = {}
                    for name, w in weights_eff.items():
                        if not w:
                            continue
                        col_name = str(name)
                        # אם העמודה לא קיימת ב-df – לא נכלול
                        if col_name not in df.columns:
                            continue
                        # Drawdown → נמוך עדיף; השאר → גבוה עדיף
                        higher_is_better = False if col_name in {"Drawdown", "MaxDrawdown"} else True
                        metrics_spec[col_name] = {
                            "weight": float(w),
                            "higher_is_better": higher_is_better,
                        }

                    if not metrics_spec:
                        st.caption("No matching metric columns found for weights – check that Sharpe/Profit/DD/etc exist in opt_df.")
                    else:
                        meta_config = {
                            "ranges": ranges_cfg,
                            "meta_optimize": {
                                "metrics": metrics_spec,
                            },
                        }

                        if st.button("▶ Compute meta-score & best parameters", key=sk("meta_run")):
                            with st.spinner("Running meta_optimize on current opt_df..."):
                                meta_res = meta_optimize(df, meta_config)  # type: ignore[arg-type]

                            best_params = meta_res.get("best_params", {})
                            feature_importance = meta_res.get("feature_importance", pd.DataFrame())
                            all_scores = meta_res.get("all_scores", df)
                            top_candidates = meta_res.get("top_candidates", all_scores.head(20))

                            st.subheader("🏆 Meta-level Best Parameters")
                            if best_params:
                                st.json(best_params)
                            else:
                                st.caption("meta_optimize did not return best_params (check ranges & metrics).")

                            st.subheader("📊 Meta Top Candidates")
                            st.dataframe(
                                top_candidates,
                                width="stretch",
                                height=min(TABLE_HEIGHT, 320),
                            )

                            if not feature_importance.empty:
                                st.subheader("🧬 Meta Feature Importance (|corr(param, meta_score)|)")
                                st.dataframe(
                                    feature_importance,
                                    width="stretch",
                                    height=min(TABLE_HEIGHT, 260),
                                )
                                try:
                                    if px is not None:
                                        fig_fi = px.bar(
                                            feature_importance.head(25),
                                            x="feature",
                                            y="importance",
                                            title="Meta Feature Importance",
                                        )
                                        st.plotly_chart(fig_fi, width="stretch")
                                except Exception:
                                    pass

                                # כפתור: כיווץ טווחים סביב best_params
                                if best_params:
                                    if st.button("⚙ Shrink active_ranges around meta best_params", key=sk("meta_shrink")):
                                        try:
                                            new_ranges = shrink_ranges_around_center(
                                                ranges_cfg,
                                                best_params,
                                                radius_factor=0.4,
                                            )
                                            st.session_state["active_ranges"] = new_ranges
                                            st.success("active_ranges updated (shrunk around meta best_params).")
                                        except Exception as e:
                                            st.warning(f"Failed to shrink ranges: {e}")
                            else:
                                st.caption("No feature_importance from meta_optimize (maybe no ranges or no variance in parameters).")

            except Exception as e:
                st.caption(f"Meta-Optimization panel failed: {e}")

    # ============================================================
    # 9.1 Feature Importance & SHAP (Parameter → Score)
    # ============================================================
    with st.expander("🧠 Feature Importance & SHAP (parameters → Score)", expanded=False):
        try:
            cols = list(param_cols)
            if not cols:
                st.caption("No parameter columns for feature importance.")
            else:
                df_clean = df.copy()
                y = pd.to_numeric(df_clean.get("Score", pd.Series(index=df.index, dtype=float)), errors="coerce")
                X = df_clean[cols].replace([np.inf, -np.inf], np.nan).dropna()
                common_idx = X.index.intersection(y.dropna().index)
                X = X.loc[common_idx]
                y = y.loc[common_idx]

                if X.empty or y.empty:
                    st.caption("Insufficient data for feature importance.")
                else:
                    do_heavy = st.checkbox(
                        "Compute SHAP / heavy feature importance",
                        value=not skip_heavy,
                        key=sk("fi_heavy"),
                    )
                    if not do_heavy:
                        st.caption(heavy_hint)
                    else:
                        imp_series = None

                        # SHAP דרך core.analysis_helpers אם קיים
                        if callable(compute_shap_importance_df):
                            try:
                                imp_series = compute_shap_importance_df(X, y)  # type: ignore[arg-type]
                            except Exception as e:
                                st.caption(f"Internal SHAP failed: {e}")

                        # fallback: RandomForest feature_importances_
                        if imp_series is None and RandomForestRegressor is not None:
                            try:
                                rf = RandomForestRegressor(
                                    n_estimators=400,
                                    random_state=int(st.session_state.get("global_seed", 1337)),
                                    n_jobs=-1,
                                )
                                rf.fit(X, y)
                                imp_series = pd.Series(
                                    rf.feature_importances_, index=X.columns, name="importance"
                                )
                            except Exception as e:
                                st.caption(f"RandomForest feature importance failed: {e}")

                        if imp_series is not None and not imp_series.empty:
                            imp_sorted = imp_series.sort_values(ascending=False)
                            st.dataframe(
                                imp_sorted.to_frame("importance"),
                                width="stretch",
                                height=min(TABLE_HEIGHT, 420),
                            )
                            try:
                                if px is not None:
                                    fig_imp = px.bar(
                                        imp_sorted.reset_index(),
                                        x="index",
                                        y="importance",
                                        title="Feature Importance (higher → more impact on Score)",
                                    )
                                    st.plotly_chart(fig_imp, width="stretch")
                            except Exception:
                                pass

                            # הורדה ל-CSV
                            st.download_button(
                                "Download feature_importance.csv",
                                data=imp_sorted.to_frame("importance").to_csv(index=True).encode("utf-8"),
                                file_name=f"feature_importance_{sym1}-{sym2}.csv",
                                mime="text/csv",
                                key=sk("fi_dl_csv"),
                            )
                        else:
                            st.caption("Could not compute feature importance.")
        except Exception as e:
            st.caption(f"Feature importance panel failed: {e}")

    # ============================================================
    # 9.2 Clusters & PCA View
    # ============================================================
    with st.expander("🧬 Clusters & PCA View (parameter space)", expanded=False):
        try:
            cols = list(param_cols)
            if len(cols) < 2:
                st.caption("Need at least 2 numeric parameter columns for clustering/PCA.")
            else:
                X = df[cols].select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).dropna()
                if X.empty:
                    st.caption("Parameter matrix is empty after cleaning.")
                else:
                    method = st.selectbox(
                        "Clustering method",
                        ["KMeans", "Internal compute_clusters (if available)"],
                        index=0,
                        key=sk("cl_method"),
                    )
                    use_internal = (
                        "Internal" in method
                        and callable(compute_clusters)
                        and callable(compute_pca_transform)
                    )

                    labels = None
                    pca2 = None

                    if use_internal:
                        try:
                            labels = compute_clusters(X)  # type: ignore[arg-type]
                            pca2 = compute_pca_transform(X, n_components=2)  # type: ignore[arg-type]
                        except Exception as e:
                            st.caption(f"Internal cluster/PCA failed: {e}")
                            use_internal = False

                    if not use_internal:
                        if KMeans is None or PCA is None:
                            st.warning("sklearn KMeans/PCA not available.")
                        else:
                            k = st.slider("Number of clusters (k)", 2, 10, 4, key=sk("cl_kmeans"))
                            # תאימות עם גרסאות sklearn שונות לגבי n_init
                            try:
                                km = KMeans(n_clusters=int(k), n_init="auto", random_state=42).fit(X.values)
                            except TypeError:
                                km = KMeans(n_clusters=int(k), n_init=10, random_state=42).fit(X.values)
                            labels = km.labels_
                            pca2 = PCA(n_components=2, random_state=42).fit_transform(X.values)

                    if labels is not None and pca2 is not None:
                        dvis = pd.DataFrame(
                            {"pc1": pca2[:, 0], "pc2": pca2[:, 1], "cluster": pd.Series(labels).astype(int)}
                        )
                        st.dataframe(
                            dvis.head(50),
                            width="stretch",
                            height=min(TABLE_HEIGHT, 360),
                        )
                        try:
                            if px is not None:
                                fig = px.scatter(
                                    dvis,
                                    x="pc1",
                                    y="pc2",
                                    color=dvis["cluster"].astype(str),
                                    title="Parameter Clusters (PCA 2D)",
                                )
                                st.plotly_chart(fig, use_container_width=True)
                        except Exception:
                            pass

                        # הצעת טווחים חדשים לפי קלאסטר נבחר
                        cl_sel = st.selectbox(
                            "Cluster for range suggestion",
                            sorted(set(pd.Series(labels).astype(int))),
                            key=sk("cl_range"),
                        )
                        if st.button("Suggest ranges from selected cluster", key=sk("cl_suggest_ranges")):
                            try:
                                mask = pd.Series(labels) == int(cl_sel)
                                sub = X.loc[mask]
                                rngs = ranges_from_dataset(sub, params=list(cols), q_low=0.10, q_high=0.90)
                                if rngs:
                                    st.session_state["active_ranges"] = rngs
                                    st.success(f"Active ranges updated from cluster {cl_sel} sample ({len(rngs)} params).")
                                else:
                                    st.info("No ranges derived from cluster (maybe cluster too small?).")
                            except Exception as e:
                                st.warning(f"Range suggestion failed: {e}")
        except Exception as e:
            st.caption(f"Cluster/PCA panel failed: {e}")

    # ============================================================
    # 9.3 Risk & Costs Lab
    # =========================

    with st.expander("💸 Risk & Costs Lab (slippage / fees / stress)", expanded=False):
        try:
            if df.empty:
                st.caption("No results available for Risk & Costs lab.")
            else:
                c1, c2, c3 = st.columns(3)
                slp = c1.number_input(
                    "Slippage (bps)",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(st.session_state.get("opt_slippage_bps", 2.0)),
                    step=0.5,
                    key=sk("rc_slp"),
                )
                fee = c2.number_input(
                    "Fees (bps)",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(st.session_state.get("opt_fees_bps", 1.0)),
                    step=0.5,
                    key=sk("rc_fee"),
                )
                stress = c3.slider(
                    "DD stress factor (×)",
                    min_value=0.5,
                    max_value=2.0,
                    value=1.2,
                    step=0.05,
                    key=sk("rc_stress"),
                )

                dfx = df.copy()
                if "Profit" in dfx.columns:
                    notional_proxy = 10.0  # ניתן לשפר בהתאם לגודל עסקאות
                    dfx["Profit_net"] = (
                        pd.to_numeric(dfx["Profit"], errors="coerce")
                        - (slp + fee) * notional_proxy
                    )
                if "Drawdown" in dfx.columns:
                    dfx["Drawdown_stress"] = pd.to_numeric(dfx["Drawdown"], errors="coerce") * float(stress)

                st.dataframe(
                    dfx.head(50),
                    width="stretch",
                    height=min(TABLE_HEIGHT, 420),
                )

                st.download_button(
                    "Download risk_costs_lab.csv",
                    data=dfx.to_csv(index=False).encode("utf-8"),
                    file_name=f"risk_costs_lab_{sym1}-{sym2}.csv",
                    mime="text/csv",
                    key=sk("rc_lab_dl"),
                )
        except Exception as e:
            st.caption(f"Risk & Costs lab failed: {e}")


    # =========================================================
    # 9.4 Data Quality & Coverage
    # =========================

    with st.expander("🔍 Data Quality & Parameter Coverage", expanded=False):
        try:
            nan_counts = df.isna().sum().sort_values(ascending=False)
            st.caption("Missingness per column (top):")
            st.dataframe(
                nan_counts.head(40).to_frame("missing"),
                width="stretch",
                height=min(TABLE_HEIGHT, 300),
            )

            param_cov = df[param_cols].notna().mean().sort_values(ascending=False)
            st.caption("Parameter coverage (fraction non-NaN):")
            st.dataframe(
                param_cov.to_frame("coverage"),
                width="stretch",
                height=min(TABLE_HEIGHT, 300),
            )

            met_cols = [c for c in ("Sharpe", "Profit", "Drawdown", "Score") if c in df.columns]
            if met_cols:
                desc = df[met_cols].describe(
                    percentiles=[0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99]
                )
                st.caption("Metric distributions (extended percentiles):")
                st.dataframe(
                    desc,
                    width="stretch",
                    height=min(TABLE_HEIGHT, 420),
                )
        except Exception as e:
            st.caption(f"Data-quality panel failed: {e}")

    # =========================================================
    # 9.5 Feature Lab (Scaling / Polynomial / Interactions)
    # =========================

    with st.expander("🧪 Feature Lab (scaling / polynomial / interactions)", expanded=False):
        try:
            if df.empty:
                st.caption("No data for Feature Lab.")
            else:
                feat_cols = param_cols
                X = df[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

                scaler_opt = st.selectbox(
                    "Scaler",
                    ["None", "Standard", "MinMax"],
                    index=0,
                    key=sk("fe_scaler"),
                )
                poly_deg = st.slider("Polynomial degree", 1, 5, 1, key=sk("fe_poly"))
                only_inter = st.checkbox("Interactions only", value=False, key=sk("fe_only_inter"))

                try:
                    from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
                except Exception:
                    st.caption("scikit-learn not available for Feature Lab.")
                    raise

                Z = X.copy()
                if scaler_opt == "Standard":
                    Z = pd.DataFrame(
                        StandardScaler().fit_transform(Z),
                        columns=feat_cols,
                        index=Z.index,
                    )
                elif scaler_opt == "MinMax":
                    Z = pd.DataFrame(
                        MinMaxScaler().fit_transform(Z),
                        columns=feat_cols,
                        index=Z.index,
                    )

                if poly_deg > 1:
                    pf = PolynomialFeatures(
                        degree=int(poly_deg),
                        include_bias=False,
                        interaction_only=bool(only_inter),
                    )
                    Zp = pf.fit_transform(Z)
                    cols_new = list(pf.get_feature_names_out(feat_cols))
                    Z = pd.DataFrame(Zp, columns=cols_new, index=Z.index)

                st.dataframe(
                    Z.head(30),
                    width="stretch",
                    height=min(TABLE_HEIGHT, 380),
                )

                c1, c2 = st.columns(2)
                if c1.button("Save to dataset registry (features_lab)", key=sk("fe_save_ds")):
                    store = st.session_state.get("_opt_datasets", {})
                    store["features_lab"] = Z.copy()
                    st.session_state["_opt_datasets"] = store
                    st.success("Saved 'features_lab' to dataset registry.")
                if c2.button("Set as active opt_df", key=sk("fe_set_active")):
                    st.session_state["opt_df"] = Z.copy()
                    st.success(f"opt_df ← features_lab ({len(Z)} rows)")
        except Exception as e:
            st.caption(f"Feature Lab panel failed: {e}")

    # =========================================================
    # 9.6 ML Pro — Surrogate Model & Validation
    # =========================

    with st.expander("🤖 ML Pro — Surrogate Model & Validation (Score ~ params)", expanded=False):
        try:
            if df is None or df.empty or "Score" not in df.columns:
                st.caption("Need 'Score' column and parameter features for ML.")
            else:
                feat_cols = [c for c in param_cols]
                X = df[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
                y = pd.to_numeric(df["Score"], errors="coerce").fillna(0.0)

                # Imports פנימיים של sklearn
                try:
                    from sklearn.model_selection import train_test_split
                    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
                    from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
                    from sklearn.ensemble import RandomForestRegressor as RFReg, GradientBoostingRegressor
                    from sklearn.inspection import permutation_importance
                except Exception:
                    st.caption("scikit-learn (modeling) not available.")
                    raise

                model_name = st.selectbox(
                    "Model",
                    ["Linear", "Ridge", "Lasso", "ElasticNet", "RandomForest", "GradientBoosting"],
                    key=sk("ml_model"),
                )
                test_size = st.slider("Test size", 0.1, 0.5, 0.2, 0.05, key=sk("ml_test"))
                random_state = st.number_input("Random state", 0, 100000, 42, 1, key=sk("ml_seed"))

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=float(test_size), random_state=int(random_state)
                )

                if model_name == "Linear":
                    model = LinearRegression()
                elif model_name == "Ridge":
                    model = Ridge(random_state=random_state)
                elif model_name == "Lasso":
                    model = Lasso(random_state=random_state)
                elif model_name == "ElasticNet":
                    model = ElasticNet(random_state=random_state)
                elif model_name == "RandomForest":
                    model = RFReg(n_estimators=400, random_state=random_state, n_jobs=-1)
                else:
                    model = GradientBoostingRegressor(random_state=random_state)

                model.fit(X_train, y_train)
                pred = model.predict(X_test)

                # Metrics
                try:
                    r2 = float(r2_score(y_test, pred))
                    mae = float(mean_absolute_error(y_test, pred))
                    rmse = float(mean_squared_error(y_test, pred, squared=False))
                except Exception:
                    r2 = mae = rmse = float("nan")

                # Spearman correlation
                def _safe_spearman(a, b):
                    try:
                        import scipy.stats as _sps
                        return float(_sps.spearmanr(a, b, nan_policy="omit").correlation)
                    except Exception:
                        return float("nan")

                rho = _safe_spearman(y_test, pred)
                st.write(
                    {
                        "R2": round(r2, 4),
                        "MAE": round(mae, 4),
                        "RMSE": round(rmse, 4),
                        "SpearmanR": round(rho, 4),
                    }
                )

                # Permutation importance
                try:
                    pi = permutation_importance(
                        model,
                        X_test,
                        y_test,
                        n_repeats=16,
                        random_state=int(random_state),
                    )
                    imp_df = (
                        pd.DataFrame({"feature": feat_cols, "importance": pi.importances_mean})
                        .sort_values("importance", ascending=False)
                    )
                    st.subheader("Permutation Importance")
                    st.dataframe(
                        imp_df,
                        width="stretch",
                        height=min(TABLE_HEIGHT, 360),
                    )
                except Exception:
                    pass

                # Predictions & residuals להורדה
                try:
                    out = pd.DataFrame({"y_true": y_test.values, "y_pred": pred})
                    out["resid"] = out["y_true"] - out["y_pred"]
                    st.download_button(
                        "Download predictions.csv",
                        data=out.to_csv(index=False).encode("utf-8"),
                        file_name=f"surrogate_predictions_{sym1}-{sym2}.csv",
                        mime="text/csv",
                        key=sk("ml_pred_dl"),
                    )
                except Exception:
                    pass
        except Exception as e:
            st.caption(f"ML Pro panel failed: {e}")

    # =========================================================
    # SECTION 9.7: Anomaly Detection (z-score) & Param Sweep
    # =========================

    with st.expander("⚠️ Anomaly Detection (z-score) & Param Sweep", expanded=False):
        try:
            # --- Anomaly Detection ---
            cols_num = [
                c for c in df.columns
                if pd.api.types.is_numeric_dtype(df[c]) and c not in ("trial_no",)
            ]
            if cols_num:
                zdf = df[cols_num].apply(
                    lambda s: (s - s.mean()) / (s.std(ddof=0) if s.std(ddof=0) else 1.0)
                )
                thr = st.slider("z-score threshold", 2.0, 6.0, 3.0, 0.1, key=sk("an_th"))
                mask = (zdf.abs() > float(thr)).any(axis=1)
                outl = df.loc[mask]
                st.write({"outliers": int(mask.sum())})
                st.dataframe(
                    outl.head(50),
                    width="stretch",
                    height=min(TABLE_HEIGHT, 300),
                )
                if mask.any():
                    st.download_button(
                        "Download outliers.csv",
                        data=outl.to_csv(index=False).encode("utf-8"),
                        file_name=f"outliers_{sym1}-{sym2}.csv",
                        mime="text/csv",
                        key=sk("an_dl"),
                    )
            else:
                st.caption("No numeric columns for anomaly detection.")

            st.markdown("---")

            # --- Param Sweep Generator + Surrogate Suggestions ---
            ranges = st.session_state.get("active_ranges") or {}
            if not ranges:
                st.caption("No active_ranges in session — set them in the sidebar first.")
                return

            max_pts = st.number_input(
                "Max sweep points", min_value=10, max_value=5000, value=200, step=10, key=sk("sw_max")
            )
            mode_sw = st.selectbox("Sweep mode", ["grid", "random"], index=0, key=sk("sw_mode"))
            keys = list(ranges.keys())
            if not keys:
                st.caption("active_ranges is empty.")
                return

            # אנחנו ניקח רק חלק מהפרמטרים כדי לא להיתקע (למשל עד 8)
            keys = keys[:8]

            candidates: List[Dict[str, float]] = []
            if mode_sw == "grid":
                per_dim = max(2, int(round(max_pts ** (1 / max(1, len(keys))))))
                vals: List[np.ndarray] = []
                for k in keys:
                    lo, hi, step = ranges[k]
                    vals.append(np.linspace(float(lo), float(hi), num=per_dim))
                for tup in itertools.product(*vals):
                    candidates.append({k: float(v) for k, v in zip(keys, tup)})
            else:
                rng = np.random.default_rng(int(st.session_state.get("global_seed", 1337)))
                for _ in range(int(max_pts)):
                    row: Dict[str, float] = {}
                    for k in keys:
                        lo, hi, step = ranges[k]
                        row[k] = float(rng.uniform(float(lo), float(hi)))
                    candidates.append(row)

            dsw = pd.DataFrame(candidates)
            st.caption(f"Generated {len(dsw)} sweep points on {len(keys)} parameters.")
            st.dataframe(
                dsw.head(30),
                width="stretch",
                height=min(TABLE_HEIGHT, 280),
            )

            # Surrogate suggestion: נשתמש ב-RF אם אפשר
            if RandomForestRegressor is not None and "Score" in df.columns:
                if st.button("Score sweep via surrogate RF & suggest Top-K", key=sk("sw_score_btn")):
                    try:
                        X = df[param_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
                        y = pd.to_numeric(df["Score"], errors="coerce").fillna(0.0)
                    except Exception as e:
                        st.caption(f"Data prep for surrogate failed: {e}")
                        return
                    try:
                        rf = RandomForestRegressor(
                            n_estimators=600,
                            random_state=int(st.session_state.get("global_seed", 1337)),
                            n_jobs=-1,
                        )
                        rf.fit(X, y)
                        X_sw = dsw[param_cols].reindex(columns=param_cols).replace([np.inf, -np.inf], np.nan).fillna(0.0)
                        preds = rf.predict(X_sw)
                        dsw_scored = dsw.copy()
                        dsw_scored["score_pred"] = preds
                        topk = st.number_input("Top-K suggestions", 5, 100, 20, 5, key=sk("sw_sugg_k"))
                        top_sugg = dsw_scored.sort_values("score_pred", ascending=False).head(int(topk))
                        st.dataframe(
                            top_sugg,
                            width="stretch",
                            height=min(TABLE_HEIGHT, 320),
                        )
                        if st.button("Use Top-K suggestions to tighten ranges", key=sk("sw_use_topk")):
                            try:
                                rngs = ranges_from_dataset(
                                    top_sugg,
                                    params=list(param_cols),
                                    q_low=0.1,
                                    q_high=0.9,
                                )
                                if rngs:
                                    st.session_state["active_ranges"] = rngs
                                    st.success(f"Active ranges tightened using Top-{topk} surrogate suggestions.")
                                else:
                                    st.caption("No ranges derived from suggestions.")
                            except Exception as e:
                                st.caption(f"Failed to apply suggestions: {e}")
                        st.download_button(
                            "Download sweep_scored.csv",
                            data=dsw_scored.to_csv(index=False).encode("utf-8"),
                            file_name=f"sweep_scored_{sym1}-{sym2}.csv",
                            mime="text/csv",
                            key=sk("sw_scored_dl"),
                        )
                    except Exception as e:
                        st.caption(f"Surrogate scoring failed: {e}")
        except Exception as e:
            st.caption(f"Anomaly/Sweep panel failed: {e}")

    # =========================================================
    # SECTION 9.8: Actionable Recommendations
    # =========================

    with st.expander("📌 Actionable Recommendations", expanded=False):
        try:
            rules: List[str] = []
            # מבוסס על df הנוכחי
            if "Score" in df.columns:
                q90 = float(pd.to_numeric(df["Score"], errors="coerce").quantile(0.90))
                rules.append(f"Focus on trials with Score ≥ {q90:.3f} (top decile).")
            if "Sharpe" in df.columns:
                sh_med = float(pd.to_numeric(df["Sharpe"], errors="coerce").median())
                rules.append(f"Sharpe median ≈ {sh_med:.2f}; prefer param zones yielding Sharpe above median.")
            if "Drawdown" in df.columns:
                dd_75 = float(pd.to_numeric(df["Drawdown"], errors="coerce").quantile(0.75))
                rules.append(f"Target Drawdown ≤ {dd_75:.2f} (≤ 75th percentile) for production.")
            if param_cols:
                rules.append(
                    "Use cluster-based ranges for stable regions, then shrink ranges around best local solutions."
                )
            if not rules:
                st.caption("Not enough metrics to derive recommendations.")
            else:
                st.write("\n".join(f"• {r}" for r in rules))
        except Exception:
            pass
"""
חלק 10/15 — Operations & Power Tools
====================================

פאנלים לניהול ותפעול מקצועי מעל opt_df:

1. Parameter Ranges — Pro Editor:
   - עריכה כטבלה (st.data_editor), כיווץ/הרחבה אוניברסלית.
   - ולידציות (lo<hi, מספר bins הגיוני, step לא קיצוני).
   - עדכון active_ranges לריצות הבאות.

2. Preset Profiles:
   - שמירת פרופילים בשם (weights / ranges / mapping / risk / UI).
   - יצוא/יבוא presets כ-JSON.
   - יישור לאותה שפה של SETTINGS + session_state.

3. Dataset Registry:
   - שמירת DataFrames שונים בשם (features, signals, sweep וכו').
   - בחירה / הצגה / יצוא ל-CSV.
   - גזירת active_ranges מדאטה (quantiles).
   - בחירת dataset כ-opt_df נוכחי.

4. Warm-Start & Seeding:
   - גזירת active_ranges מדאטה או sweep.csv / sweep דינמי.
   - סימון הרצה חדשה עם ה-ranges המעודכנים.

5. Study Browser & Merge (DuckDB):
   - עיון ב-studies קיימים לזוג הנוכחי (מ־DuckDB).
   - טעינת trials של study למצב opt_df.
   - מיזוג מספר studies לקובץ אחד (merged_studies.csv).

6. Report Builder & Checklist:
   - בניית report markdown + JSON ל-top-K.
   - בדיקת כשירות (יש Score/Sharpe/DD? כמה שורות?).
"""

def _render_optimization_operations_sections(TABLE_HEIGHT: int, df: pd.DataFrame, sym1: str, sym2: str) -> None:
    FOCUS = bool(st.session_state.get("opt_focus_mode", False))

    # נשתמש בזה כמה פעמים
    ranges = st.session_state.get("active_ranges") or {}
    weights_eff = st.session_state.get("loaded_weights_eff", {"Sharpe": 0.5, "Profit": 0.3, "Drawdown": 0.2})

    # =========================================================
    # 9.x ML Analysis מודולרי מתוך core.ml_analysis
    # =========================================================
    with st.expander("🤖 ML Analysis (core.ml_analysis)", expanded=False):
        try:
            # משתמש ב-st.session_state["opt_df"] כברירת מחדל
            render_ml_for_optuna_session("opt_df")
        except Exception as e:
            st.caption(f"ML analysis panel failed: {e}")

    # =========================================================
    # 10.1 Parameter Ranges — Pro Editor
    # =========================================================
    with st.expander("⚙️ Parameter Ranges — Pro Editor", expanded=not FOCUS):
        try:
            def _ranges_to_df(_ranges: Dict[str, ParamRange]) -> pd.DataFrame:
                rows: List[Dict[str, Any]] = []
                for k, tpl in (_ranges or {}).items():
                    lo, hi, step = tpl
                    rows.append(
                        {
                            "param": str(k),
                            "low": float(lo),
                            "high": float(hi),
                            "step": None if step is None else float(step),
                        }
                    )
                return (
                    pd.DataFrame(rows, columns=["param", "low", "high", "step"])
                    .sort_values("param")
                    .reset_index(drop=True)
                )

            def _df_to_ranges(df_in: pd.DataFrame) -> Dict[str, ParamRange]:
                out: Dict[str, ParamRange] = {}
                for _, r in df_in.iterrows():
                    try:
                        name = str(r["param"]).strip()
                        if not name:
                            continue
                        lo = float(r["low"])
                        hi = float(r["high"])
                        step_val = r.get("step")
                        step = float(step_val) if step_val is not None and str(step_val) != "nan" else None
                        if hi <= lo:
                            hi = lo + 1e-9
                        out[name] = (lo, hi, step)
                    except Exception:
                        continue
                return out

            df_rng = _ranges_to_df(ranges)
            st.caption("Edit ranges below. Use Apply to persist into active_ranges.")
            df_edit = st.data_editor(
                df_rng,
                width="stretch",
                height=TABLE_HEIGHT,
                num_rows="dynamic",
                key=sk("rng_editor"),
            )

            c1, c2, c3, c4 = st.columns(4)
            do_apply = c1.button("Apply to active_ranges", key=sk("rng_apply"))
            widen = c2.button("Widen all by +10%", key=sk("rng_widen"))
            narrow = c3.button("Narrow all by −10%", key=sk("rng_narrow"))
            reset_base = c4.button("Reset to defaults", key=sk("rng_reset"))

            # כיווץ/הרחבה
            if widen or narrow:
                try:
                    fac = 1.1 if widen else 0.9
                    df_scaled = df_edit.copy()
                    df_scaled["low"] = pd.to_numeric(df_scaled["low"], errors="coerce")
                    df_scaled["high"] = pd.to_numeric(df_scaled["high"], errors="coerce")
                    mid = (df_scaled["low"] + df_scaled["high"]) / 2.0
                    span = (df_scaled["high"] - df_scaled["low"]) * fac / 2.0
                    df_scaled["low"], df_scaled["high"] = mid - span, mid + span
                    st.session_state["_rng_scaled_preview"] = df_scaled
                    st.dataframe(
                        df_scaled.head(30),
                        width="stretch",
                        height=min(TABLE_HEIGHT, 260),
                    )
                    st.caption("Scaled preview — click 'Apply to active_ranges' to persist.")
                    df_edit = df_scaled
                except Exception as e:
                    st.warning(f"Scaling failed: {e}")

            # Reset לערכי ברירת מחדל
            if reset_base:
                try:
                    base = get_default_param_ranges(profile=str(SETTINGS.env or "default"))
                    st.session_state["active_ranges"] = base
                    st.success("active_ranges reset to get_default_param_ranges().")
                    df_rng = _ranges_to_df(base)
                    st.dataframe(df_rng, width="stretch", height=min(TABLE_HEIGHT, 260))
                except Exception as e:
                    st.warning(f"Reset failed: {e}")

            # Apply
            if do_apply:
                try:
                    new_ranges = _df_to_ranges(df_edit)
                    issues: List[str] = []
                    for k, (lo, hi, step) in new_ranges.items():
                        if not (lo < hi):
                            issues.append(f"{k}: low >= high (fixed to hi = low+1e-9 internally)")
                        span = hi - lo
                        if step is not None and step > 0:
                            bins = span / step
                            if bins < 3:
                                issues.append(f"{k}: step too large (bins < 3)")
                            if bins > 2e5:
                                issues.append(f"{k}: step too small (bins > 200k)")
                    st.session_state["active_ranges"] = new_ranges
                    if issues:
                        st.warning("Some potential issues:\n" + "\n".join(issues))
                    st.success(f"Updated active_ranges with {len(new_ranges)} parameters.")
                except Exception as e:
                    st.error(f"Apply failed: {e}")
        except Exception as e:
            st.caption(f"Ranges editor failed: {e}")

    # =========================================================
    # 10.2 Preset Profiles (weights / ranges / mapping / risk / UI)
    # =========================

    with st.expander("📦 Preset Profiles (Ranges / Weights / Mapping / Risk / UI)", expanded=False):
        try:
            presets: Dict[str, Any] = st.session_state.get("_opt_presets", {})
            preset_name = st.text_input("Preset name", value="default", key=sk("pr_name"))

            # Snapshot נוכחי
            dt = datetime.now(timezone.utc)
            current_profile = {
                "timestamp": dt.isoformat(timespec="seconds").replace("+00:00", "Z"),
                "weights": weights_eff,
                "ranges": st.session_state.get("active_ranges", {}),
                "mapping": st.session_state.get("opt_param_mapping", {}),
                "risk": get_session_risk_kwargs(),
                "ui": {
                    "theme": st.session_state.get("opt_theme", "auto"),
                    "compact": bool(st.session_state.get("opt_ui_compact", False)),
                    "high_contrast": bool(st.session_state.get("opt_ui_high_contrast", False)),
                    "focus_mode": bool(st.session_state.get("opt_focus_mode", False)),
                },
            }

            c1, c2, c3 = st.columns(3)

            if c1.button("💾 Save/Update preset", key=sk("pr_save")):
                presets[preset_name] = current_profile
                st.session_state["_opt_presets"] = presets
                st.success(f"Preset '{preset_name}' saved/updated in session.")

            if c2.button("⬇️ Export preset JSON", key=sk("pr_export")):
                if preset_name in presets:
                    payload = presets[preset_name]
                    st.download_button(
                        f"Download {preset_name}.json",
                        data=json.dumps(make_json_safe(payload), ensure_ascii=False, indent=2).encode("utf-8"),
                        file_name=f"{preset_name}.json",
                        mime="application/json",
                        key=sk("pr_dl_json"),
                    )
                else:
                    st.warning("Preset not found in session.")

            up = c3.file_uploader("⬆️ Import preset (.json)", type=["json"], key=sk("pr_upl"))
            if up is not None:
                try:
                    obj = json.loads(up.read().decode("utf-8"))
                    if isinstance(obj, dict):
                        st.session_state["_opt_presets"] = {**presets, preset_name: obj}
                        # אפשר גם ליישם מיידית:
                        if "weights" in obj and isinstance(obj["weights"], dict):
                            st.session_state["loaded_weights_eff"] = obj["weights"]
                        if "ranges" in obj and isinstance(obj["ranges"], dict):
                            st.session_state["active_ranges"] = obj["ranges"]
                        if "mapping" in obj and isinstance(obj["mapping"], dict):
                            st.session_state["opt_param_mapping"] = obj["mapping"]
                        if "risk" in obj and isinstance(obj["risk"], dict):
                            st.session_state["opt_slippage_bps"] = float(obj["risk"].get("slippage_bps", 2.0))
                            st.session_state["opt_fees_bps"] = float(obj["risk"].get("fees_bps", 1.0))
                            st.session_state["opt_max_positions"] = int(obj["risk"].get("max_positions", 5))
                            st.session_state["opt_rebalance_days"] = int(obj["risk"].get("rebalance_days", 5))
                        if "ui" in obj and isinstance(obj["ui"], dict):
                            st.session_state["opt_ui_compact"] = bool(obj["ui"].get("compact", False))
                            st.session_state["opt_ui_high_contrast"] = bool(obj["ui"].get("high_contrast", False))
                            st.session_state["opt_focus_mode"] = bool(obj["ui"].get("focus_mode", False))
                        st.success(f"Imported preset into '{preset_name}' and applied settings to session.")
                    else:
                        st.warning("Invalid preset JSON format.")
                except Exception as e:
                    st.error(f"Invalid preset JSON: {e}")

            # רשימת פרופילים קיימים
            if presets:
                st.caption("Existing presets in session:")
                st.json({k: {"timestamp": v.get("timestamp")} for k, v in presets.items()})
        except Exception as e:
            st.caption(f"Preset profiles panel failed: {e}")

    # =========================================================
    # 10.3 Dataset Registry (multi-dataset management)
    # =========================

    with st.expander("🗃️ Dataset Registry (multi-dataset store)", expanded=False):
        try:
            store: Dict[str, pd.DataFrame] = st.session_state.get("_opt_datasets", {})
            ds_name = st.text_input("Dataset name", value="features_v1", key=sk("ds_name"))

            c1, c2, c3, c4 = st.columns(4)

            if c1.button("Save current opt_df → registry", key=sk("ds_save_curr")):
                dcur = st.session_state.get("opt_df", pd.DataFrame())
                if dcur is None or dcur.empty:
                    st.warning("Current opt_df is empty.")
                else:
                    store[ds_name] = dcur.copy()
                    st.session_state["_opt_datasets"] = store
                    st.success(f"Saved current opt_df as '{ds_name}'.")

            if c2.button("Save last feature_lap (features_lab)", key=sk("ds_save_feat")):
                dfeat = st.session_state.get("features_lab", st.session_state.get("features_lab", pd.DataFrame()))
                if dfeat is None or dfeat.empty:
                    st.warning("No features_lab dataset found.")
                else:
                    store["features_lab"] = dfeat.copy()
                    st.session_state["_opt_datasets"] = store
                    st.success("Saved 'features_lab' to registry.")

            if c3.button("Set selected as active opt_df", key=sk("ds_set_active_btn")):
                # נבחר בהמשך אחרי בחירת dataset בתיבה
                st.session_state["_ds_set_active_requested"] = True

            if c4.button("Export selected dataset (CSV)", key=sk("ds_export_btn")):
                st.session_state["_ds_export_requested"] = True

            keys = sorted(store.keys())
            if keys:
                ds_sel = st.selectbox("Available datasets", keys, key=sk("ds_sel"))
                dfx = store.get(ds_sel, pd.DataFrame())
                st.dataframe(dfx.head(50), width="stretch", height=TABLE_HEIGHT)

                # derive ranges from dataset
                d1, d2, d3 = st.columns(3)
                if d1.button("Derive ranges from dataset (10–90%)", key=sk("ds_ranges")):
                    try:
                        rngs = ranges_from_dataset(dfx)
                        if rngs:
                            st.session_state["active_ranges"] = rngs
                            st.success(f"Derived {len(rngs)} ranges from dataset '{ds_sel}'.")
                        else:
                            st.info("No numeric columns or not enough data to derive ranges.")
                    except Exception as e:
                        st.warning(f"Derive ranges failed: {e}")

                if st.session_state.get("_ds_set_active_requested"):
                    st.session_state["_ds_set_active_requested"] = False
                    if not dfx.empty:
                        st.session_state["opt_df"] = dfx.copy()
                        st.success(f"opt_df ← dataset '{ds_sel}' ({len(dfx)} rows).")

                if st.session_state.get("_ds_export_requested"):
                    st.session_state["_ds_export_requested"] = False
                    if not dfx.empty:
                        st.download_button(
                            f"Download {ds_sel}.csv",
                            data=dfx.to_csv(index=False).encode("utf-8"),
                            file_name=f"{ds_sel}.csv",
                            mime="text/csv",
                            key=sk("ds_dl_csv"),
                        )
            else:
                st.caption("No datasets stored yet. Use the buttons above to add some.")
        except Exception as e:
            st.caption(f"Dataset registry panel failed: {e}")

    # =========================================================
    # 10.4 Warm-Start & Seeding Tools
    # =========================

    with st.expander("🔥 Warm-Start & Seeding Tools", expanded=False):
        try:
            # Seed from active opt_df
            if df is not None and not df.empty:
                c1, c2 = st.columns(2)
                if c1.button("Derive ranges from current opt_df (top 20%)", key=sk("ws_from_opt")):
                    try:
                        if "Score" in df.columns:
                            top_chunk = df.sort_values("Score", ascending=False).head(max(10, len(df) // 5))
                        else:
                            top_chunk = df.head(max(10, len(df) // 5))
                        rngs = ranges_from_dataset(top_chunk)
                        if rngs:
                            st.session_state["active_ranges"] = rngs
                            st.success(f"Derived {len(rngs)} ranges from current opt_df (top slice).")
                        else:
                            st.info("No ranges derived from opt_df (maybe not enough numeric columns).")
                    except Exception as e:
                        st.warning(f"Derive from opt_df failed: {e}")

                # Seed from uploaded sweep.csv
                up_sw = c2.file_uploader("Upload sweep.csv for seeding (columns = param names)", type=["csv"], key=sk("ws_upl"))
                if up_sw is not None:
                    try:
                        dsw = pd.read_csv(up_sw)
                        st.dataframe(dsw.head(30), width="stretch", height=min(TABLE_HEIGHT, 260))
                        if st.button("Use sweep.csv to set active_ranges (min/max per param)", key=sk("ws_use_sw")):
                            rngs: Dict[str, ParamRange] = {}
                            for c in dsw.columns:
                                if pd.api.types.is_numeric_dtype(dsw[c]):
                                    lo, hi = float(dsw[c].min()), float(dsw[c].max())
                                    if hi <= lo:
                                        hi = lo + 1e-9
                                    rngs[str(c)] = (lo, hi, None)
                            if rngs:
                                st.session_state["active_ranges"] = rngs
                                st.success(f"active_ranges set from sweep.csv ({len(rngs)} params).")
                            else:
                                st.info("No numeric columns in sweep.csv.")
                    except Exception as e:
                        st.warning(f"sweep.csv parse failed: {e}")
        except Exception as e:
            st.caption(f"Warm-start tools failed: {e}")

    # =========================================================
    # 10.5 Study Browser & Merge (DuckDB)
    # =========================

    with st.expander("📚 Study Browser & Merge (DuckDB)", expanded=False):
        try:
            if duckdb is None:
                st.caption("DuckDB not available — install duckdb to use study browser.")
            else:
                pair_label = f"{sym1}-{sym2}"
                st.caption(f"Studies for pair: **{pair_label}**")

                df_st = list_studies_for_pair(pair_label, limit=50)
                if df_st.empty:
                    st.caption("No studies found in DuckDB for this pair yet.")
                else:
                    st.dataframe(df_st, width="stretch", height=min(TABLE_HEIGHT, 260))
                    st.caption("Select a study to load its trials into opt_df or merge multiple studies.")

                    study_ids = df_st["study_id"].astype(int).tolist()
                    sel_id = st.selectbox("Select study_id", study_ids, key=sk("sb_sel"))
                    c1, c2, c3 = st.columns(3)

                    if c1.button("Load selected study trials → opt_df", key=sk("sb_load")):
                        try:
                            df_tr = load_trials_from_duck(int(sel_id))
                            if not df_tr.empty:
                                st.session_state["opt_df"] = df_tr.copy()
                                st.success(f"Loaded {len(df_tr)} trials from study_id={sel_id} into opt_df.")
                            else:
                                st.info("Selected study has no trials.")
                        except Exception as e:
                            st.warning(f"Load trials failed: {e}")

                    if c2.button("Delete selected study", key=sk("sb_del")):
                        try:
                            delete_study_from_duck(int(sel_id))
                            st.success(f"Deleted study_id={sel_id} from DuckDB.")
                        except Exception as e:
                            st.warning(f"Delete failed: {e}")

                    # Merge selected studies
                    sel_ids_multi = st.multiselect("Studies to merge", study_ids, max_selections=10, key=sk("sb_merge_sel"))
                    if c3.button("Merge selected studies", key=sk("sb_merge_btn")):
                        try:
                            all_parts = []
                            for sid in sel_ids_multi:
                                dft = load_trials_from_duck(int(sid))
                                if not dft.empty:
                                    dft["_study_id"] = int(sid)
                                    all_parts.append(dft)
                            if all_parts:
                                df_all = pd.concat(all_parts, ignore_index=True)
                                st.session_state["opt_df"] = df_all.copy()
                                st.success(f"Merged {len(sel_ids_multi)} studies → {len(df_all)} rows in opt_df.")
                                st.dataframe(df_all.head(100), width="stretch", height=min(TABLE_HEIGHT, 260))
                                st.download_button(
                                    "Download merged_studies.csv",
                                    data=df_all.to_csv(index=False).encode("utf-8"),
                                    file_name=f"merged_studies_{pair_label}.csv",
                                    mime="text/csv",
                                    key=sk("sb_merge_dl"),
                                )
                            else:
                                st.info("No trials found in selected studies.")
                        except Exception as e:
                            st.caption(f"Merge failed: {e}")
        except Exception as e:
            st.caption(f"Study browser panel failed: {e}")

    # =========================================================
    # 10.6 Report Builder & Checklist
    # =========================
    dt = datetime.now(timezone.utc)
    ts = dt.isoformat(timespec="seconds").replace("+00:00", "Z")
    with st.expander("📝 Report Builder & Post-Run Checklist", expanded=False):
        try:
            title = st.text_input(
                "Report title",
                value=f"Optimization Report — {sym1}-{sym2}",
                key=sk("rpt_title"),
            )
            notes = st.text_area(
                "Notes (markdown)",
                value="",
                key=sk("rpt_notes"),
                height=120,
            )
            topn = st.slider(
                "Top-N rows to include",
                5,
                200,
                20,
                5,
                key=sk("rpt_topn"),
            )

            if st.button("Build report", key=sk("rpt_build")):
                try:
                    dfx = df.copy()
                    if "Score" in dfx.columns:
                        dfx = dfx.sort_values("Score", ascending=False).head(int(topn))
                    else:
                        dfx = dfx.head(int(topn))

                    # KPIs קטנים
                    kpi: Dict[str, Any] = {}
                    try:
                        if "Sharpe" in df.columns:
                            kpi["best_sharpe"] = float(pd.to_numeric(df["Sharpe"], errors="coerce").max())
                            kpi["avg_sharpe"] = float(pd.to_numeric(df["Sharpe"], errors="coerce").mean())
                        if "Score" in df.columns:
                            kpi["best_score"] = float(pd.to_numeric(df["Score"], errors="coerce").max())
                            kpi["avg_score"] = float(pd.to_numeric(df["Score"], errors="coerce").mean())
                        if "Profit" in df.columns:
                            kpi["max_profit"] = float(pd.to_numeric(df["Profit"], errors="coerce").max())
                        if "Drawdown" in df.columns:
                            kpi["min_dd"] = float(pd.to_numeric(df["Drawdown"], errors="coerce").min())
                    except Exception:
                        pass

                    # Markdown
                    lines = [
                        f"# {title}",
                        "",
                        f"Pair: **{sym1} / {sym2}**",
                        f"Rows (top-N): {len(dfx)}",
                        f"Generated at: {ts}",
                        "",
                        "## Notes",
                        notes or "(none)",
                        "",
                        "## KPIs",
                        "```json",
                        json.dumps(kpi, ensure_ascii=False, indent=2),
                        "```",
                        "",
                        "## Top-N rows (CSV)",
                        "```csv",
                        dfx.to_csv(index=False),
                        "```",
                    ]
                    md_text = "\n".join(lines)
                    st.markdown(md_text)
                    st.download_button(
                        "Download report.md",
                        data=md_text.encode("utf-8"),
                        file_name="optimization_report.md",
                        mime="text/markdown",
                        key=sk("rpt_dl_md"),
                    )
                    st.download_button(
                        "Download topN.csv",
                        data=dfx.to_csv(index=False).encode("utf-8"),
                        file_name="optimization_topN.csv",
                        mime="text/csv",
                        key=sk("rpt_dl_csv"),
                    )

                    # JSON payload (למערכות אחרות / audit)
                    payload = {
                        "title": title,
                        "pair": f"{sym1}-{sym2}",
                        "generated_at": ts,
                        "kpi": kpi,
                        "top_rows": dfx.to_dict(orient="records"),
                        "notes": notes,
                        "weights": weights_eff,
                        "ranges": ranges,
                    }
                    st.download_button(
                        "Download report.json",
                        data=json.dumps(make_json_safe(payload), ensure_ascii=False, indent=2).encode("utf-8"),
                        file_name="optimization_report.json",
                        mime="application/json",
                        key=sk("rpt_dl_json"),
                    )

                    # Post-run checklist
                    items: List[str] = []
                    try:
                        if df is None or df.empty:
                            items.append("✗ opt_df is empty.")
                        else:
                            items.append(f"✓ Rows: {len(df)}")
                        if "Score" not in df.columns:
                            items.append("✗ Missing 'Score' column.")
                        if "Sharpe" not in df.columns:
                            items.append("⚠ Missing 'Sharpe' column.")
                        if "Drawdown" not in df.columns:
                            items.append("⚠ Missing 'Drawdown' column.")
                        if not ranges:
                            items.append("⚠ active_ranges is empty.")
                        if not weights_eff:
                            items.append("⚠ scoring weights not set.")
                    except Exception:
                        pass

                    st.markdown("### Post-Run Checklist")
                    if items:
                        st.write("\n".join(items))
                    else:
                        st.write("✓ No obvious issues detected.")

                except Exception as _e:
                    st.caption(f"Report builder failed: {_e}")
        except Exception as e:
            st.caption(f"Report/checklist panel failed: {e}")
"""
חלק 11/15 — Monitoring, Run History, Manifest Hub & Batch Portfolio View (Pro+)
===============================================================================

לוח בקרה מעל אופטימיזציה:
- Run History: היסטוריית studies ב-DuckDB, KPIs, השוואת שני ריצות.
- Manifest & Artifacts Hub: צפייה, הורדה והשוואת manifests/CSV artifacts.
- Batch Portfolio View: מבט across pairs על opt_df_batch (כמעט כמו פורטפוליו).
- Telemetry מתקדם: גודל df, מידע על DB, גרסאות ספריות.
"""

def _render_optimization_monitoring_sections(TABLE_HEIGHT: int, df: pd.DataFrame, sym1: str, sym2: str) -> None:
    FOCUS = bool(st.session_state.get("opt_focus_mode", False))
    pair_label = f"{sym1}-{sym2}"

    # =========================================================
    # 11.1 Run History & Studies Timeline (DuckDB)
    # =========================
    with st.expander("📊 Run History & Studies Timeline (DuckDB)", expanded=not FOCUS):
        try:
            if duckdb is None:
                st.caption("DuckDB not available — cannot show run history.")
            else:
                df_st = list_studies_for_pair(pair_label, limit=200)
                if df_st.empty:
                    st.caption("No studies found in DuckDB for this pair yet.")
                else:
                    # אפשרות לסנן לפי sampler/profile/direction
                    sampler_filter = st.selectbox(
                        "Filter by sampler",
                        ["(all)"] + sorted(df_st["sampler"].dropna().astype(str).unique().tolist()),
                        key=sk("mon_samp"),
                    )
                    dir_filter = st.selectbox(
                        "Filter by direction",
                        ["(all)", "maximize", "minimize"],
                        key=sk("mon_dir"),
                    )

                    df_hist = df_st.copy()
                    if sampler_filter != "(all)":
                        df_hist = df_hist[df_hist["sampler"] == sampler_filter]
                    if dir_filter != "(all)" and "direction" in df_hist.columns:
                        df_hist = df_hist[df_hist["direction"] == dir_filter]

                    df_hist = df_hist.sort_values("created_at", ascending=True)
                    st.dataframe(
                        df_hist,
                        width="stretch",
                        height=min(TABLE_HEIGHT, 260),
                    )

                    # ננסה להוציא best_score/best_sharpe לכל study
                    scores: List[float] = []
                    sharpes: List[float] = []
                    times: List[Any] = []
                    ids: List[int] = []
                    profiles: List[str] = []

                    for _, row in df_hist.iterrows():
                        sid = int(row["study_id"])
                        t_created = row.get("created_at")
                        prof = str(row.get("profile_json", "") or row.get("profile", "") or "")
                        profiles.append(prof)
                        try:
                            dft = load_trials_from_duck(sid)
                            if not dft.empty:
                                if "Score" in dft.columns:
                                    scores.append(float(pd.to_numeric(dft["Score"], errors="coerce").max()))
                                else:
                                    scores.append(float("nan"))
                                if "Sharpe" in dft.columns:
                                    sharpes.append(float(pd.to_numeric(dft["Sharpe"], errors="coerce").max()))
                                else:
                                    sharpes.append(float("nan"))
                            else:
                                scores.append(float("nan"))
                                sharpes.append(float("nan"))
                        except Exception:
                            scores.append(float("nan"))
                            sharpes.append(float("nan"))
                        times.append(t_created)
                        ids.append(sid)

                    hist_df = pd.DataFrame(
                        {
                            "study_id": ids,
                            "created_at": times,
                            "best_score": scores,
                            "best_sharpe": sharpes,
                            "profile": profiles,
                        }
                    ).dropna(subset=["created_at"])

                    if not hist_df.empty and px is not None:
                        try:
                            fig_hist = px.line(
                                hist_df,
                                x="created_at",
                                y="best_score",
                                color=hist_df["profile"].astype(str),
                                markers=True,
                                title=f"Best Score over studies — {pair_label}",
                                hover_data=["study_id", "best_sharpe"],
                            )
                            st.plotly_chart(fig_hist, width="stretch")
                        except Exception:
                            st.dataframe(
                                hist_df,
                                width="stretch",
                                height=min(TABLE_HEIGHT, 260),
                            )

                    # השוואת שתי ריצות (A vs B)
                    st.markdown("### Compare two studies (A vs B)")
                    if not hist_df.empty:
                        ids_for_compare = hist_df["study_id"].astype(int).tolist()
                        c1, c2 = st.columns(2)
                        sid_a = c1.selectbox("Study A", ids_for_compare, key=sk("cmp_a"))
                        sid_b = c2.selectbox("Study B", ids_for_compare, key=sk("cmp_b"))

                        if sid_a and sid_b and sid_a != sid_b:
                            try:
                                dfa = load_trials_from_duck(int(sid_a))
                                dfb = load_trials_from_duck(int(sid_b))
                                if "Score" in dfa.columns and "Score" in dfb.columns:
                                    a_best = float(pd.to_numeric(dfa["Score"], errors="coerce").max())
                                    b_best = float(pd.to_numeric(dfb["Score"], errors="coerce").max())
                                else:
                                    a_best = b_best = float("nan")
                                st.write(
                                    {
                                        "study_a": int(sid_a),
                                        "best_score_a": a_best,
                                        "n_tr_a": len(dfa),
                                        "study_b": int(sid_b),
                                        "best_score_b": b_best,
                                        "n_tr_b": len(dfb),
                                    }
                                )
                            except Exception as e:
                                st.caption(f"Compare failed: {e}")
        except Exception as e:
            st.caption(f"Run history panel failed: {e}")

    # =========================================================
    # 11.2 Manifest & Artifacts Hub (with comparison)
    # =========================
    with st.expander("📜 Manifest & Artifacts Hub (view & compare)", expanded=False):
        try:
            if duckdb is None:
                st.caption("DuckDB not available — cannot show manifests.")
            else:
                df_st_all = list_studies_for_pair(pair_label, limit=50)
                sid_options = df_st_all["study_id"].astype(int).tolist() if not df_st_all.empty else []
                default_sid = st.session_state.get("opt_last_study_id") or (sid_options[-1] if sid_options else None)

                c1, c2 = st.columns(2)
                sid_main = c1.selectbox(
                    "Primary study_id",
                    sid_options or [default_sid],
                    index=(sid_options.index(default_sid) if default_sid in sid_options else 0) if sid_options else 0,
                    key=sk("man_sid_main"),
                ) if default_sid is not None else None

                sid_ref = c2.selectbox(
                    "Reference study_id (for diff)",
                    [None] + sid_options,
                    index=0,
                    key=sk("man_sid_ref"),
                ) if sid_options else None

                if sid_main is None:
                    st.caption("No primary study selected.")
                else:
                    artifacts_main = load_artifacts_for_study(int(sid_main))
                    if not artifacts_main:
                        st.caption("No artifacts stored for primary study.")
                    else:
                        kinds_main = sorted(set(a["kind"] for a in artifacts_main))
                        kind_sel = st.selectbox("Artifact kind (primary)", kinds_main, key=sk("man_kind_main"))
                        filtered_main = [a for a in artifacts_main if a["kind"] == kind_sel]
                        art_main = filtered_main[-1] if filtered_main else None

                        st.write(f"Primary: study_id={sid_main}, kind={kind_sel}")
                        if art_main is not None:
                            payload_main = art_main["payload"]
                            # נסה להציג
                            if kind_sel.endswith("_json") or kind_sel == "manifest_json":
                                try:
                                    obj = json.loads(payload_main.decode("utf-8"))
                                    st.json(obj)
                                except Exception as e:
                                    st.caption(f"Failed to decode JSON artifact: {e}")
                            elif kind_sel.endswith("_csv"):
                                try:
                                    df_art = pd.read_csv(pd.io.common.BytesIO(payload_main))  # type: ignore[arg-type]
                                except Exception:
                                    try:
                                        df_art = pd.read_csv(pd.io.common.StringIO(payload_main.decode("utf-8")))  # type: ignore[arg-type]
                                    except Exception as e2:
                                        st.caption(f"Failed to decode CSV artifact: {e2}")
                                        df_art = pd.DataFrame()
                                if not df_art.empty:
                                    st.dataframe(
                                        df_art.head(50),
                                        width="stretch",
                                        height=min(TABLE_HEIGHT, 260),
                                    )
                            st.download_button(
                                "Download primary artifact",
                                data=payload_main,
                                file_name=f"artifact_{sid_main}_{kind_sel}.bin",
                                key=sk("art_dl_main"),
                            )

                        # השוואה מול study_ref (אם נבחר)
                        if sid_ref is not None:
                            st.markdown("### Compare manifests (primary vs reference)")
                            artifacts_ref = load_artifacts_for_study(int(sid_ref))
                            artifacts_ref_kind = [a for a in artifacts_ref if a["kind"] == kind_sel]
                            art_ref = artifacts_ref_kind[-1] if artifacts_ref_kind else None
                            if art_ref is None:
                                st.caption(f"No artifact of kind={kind_sel} in reference study={sid_ref}.")
                            else:
                                try:
                                    # נסה להשוות כ-JSON (אם Manifest)
                                    obj_main = json.loads(payload_main.decode("utf-8")) if kind_sel.endswith("_json") else None
                                    obj_ref = json.loads(art_ref["payload"].decode("utf-8")) if kind_sel.endswith("_json") else None
                                except Exception:
                                    obj_main = obj_ref = None

                                if obj_main is not None and obj_ref is not None:
                                    # diff בסיסי: מפתחות שונים/ערכים שונים
                                    keys_all = sorted(set(obj_main.keys()) | set(obj_ref.keys()))
                                    diff_rows = []
                                    for k in keys_all:
                                        v1 = obj_main.get(k, "(missing)")
                                        v2 = obj_ref.get(k, "(missing)")
                                        if v1 != v2:
                                            diff_rows.append({"key": k, "primary": v1, "reference": v2})
                                    if diff_rows:
                                        st.dataframe(
                                            pd.DataFrame(diff_rows),
                                            width="stretch",
                                            height=min(TABLE_HEIGHT, 260),
                                        )
                                    else:
                                        st.caption("No differences in top-level manifest keys.")
                                else:
                                    st.caption("Diff is only implemented for JSON manifests.")
        except Exception as e:
            st.caption(f"Manifest & artifacts panel failed: {e}")

    # =========================================================
    # 11.3 Batch Portfolio View (opt_df_batch) — משודרג
    # =========================
    with st.expander("📂 Batch Portfolio View (opt_df_batch)", expanded=False):
        try:
            df_batch = st.session_state.get("opt_df_batch", pd.DataFrame())
            if df_batch is None or df_batch.empty:
                st.caption("No batch results loaded (run batch mode to populate opt_df_batch).")
            else:
                if "Pair" not in df_batch.columns:
                    st.caption("Batch dataframe has no 'Pair' column; cannot build portfolio view.")
                else:
                    st.caption(f"Batch results: {len(df_batch)} rows across {df_batch['Pair'].nunique()} pairs.")
                    # KPIs לכל זוג
                    stats_cols = [c for c in ("Score", "Sharpe", "Profit", "Drawdown") if c in df_batch.columns]
                    grp_pairs = df_batch.groupby("Pair")[stats_cols]
                    agg = grp_pairs.agg(["mean", "max", "min"]).reset_index()
                    agg.columns = ["Pair"] + [f"{c}_{m}" for c, m in itertools.product(stats_cols, ("mean", "max", "min"))]
                    st.dataframe(
                        agg,
                        width="stretch",
                        height=min(TABLE_HEIGHT, 260),
                    )

                    pairs_available = sorted(df_batch["Pair"].unique().tolist())
                    sel_pairs = st.multiselect(
                        "Select pairs for portfolio sleeve",
                        pairs_available,
                        default=pairs_available[: min(5, len(pairs_available))],
                        key=sk("pf_pairs"),
                    )
                    if sel_pairs:
                        df_sel = df_batch[df_batch["Pair"].isin(sel_pairs)].copy()

                        # מציגים צירוף אחד per pair (Score הגבוה ביותר)
                        rep_rows = []
                        for p in sel_pairs:
                            sub = df_sel[df_sel["Pair"] == p]
                            if "Score" in sub.columns:
                                sub = sub.sort_values("Score", ascending=False)
                            rep_rows.append(sub.head(1))
                        rep_df = pd.concat(rep_rows, ignore_index=True)

                        mode_weights = st.selectbox(
                            "Weighting mode",
                            ["Equal weights", "Score-proportional", "Sharpe/DD heuristic"],
                            index=0,
                            key=sk("pf_mode"),
                        )

                        if mode_weights.startswith("Equal"):
                            w_vec = np.ones(len(rep_df))
                        elif mode_weights.startswith("Score"):
                            w_vec = pd.to_numeric(rep_df.get("Score", pd.Series(index=rep_df.index)), errors="coerce").fillna(1.0).to_numpy()
                        else:
                            # Sharpe/DD heuristic ~ Sharpe / (1+DD)
                            sh = pd.to_numeric(rep_df.get("Sharpe", pd.Series(index=rep_df.index, dtype=float)), errors="coerce").fillna(0.0)
                            dd = pd.to_numeric(rep_df.get("Drawdown", pd.Series(index=rep_df.index, dtype=float)), errors="coerce").fillna(0.1)
                            w_vec = (sh / (1.0 + dd)).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
                            if not np.any(np.abs(w_vec) > 0):
                                w_vec = np.ones(len(rep_df))

                        w_sum = np.sum(np.abs(w_vec)) or 1.0
                        w = w_vec / w_sum
                        rep_df["weight"] = w

                        st.caption("Representative portfolio of selected pairs (one strategy per pair):")
                        st.dataframe(
                            rep_df[["Pair"] + [c for c in ("Score", "Sharpe", "Drawdown") if c in rep_df.columns] + ["weight"]],
                            width="stretch",
                            height=min(TABLE_HEIGHT, 280),
                        )

                        # approx portfolio KPIs
                        try:
                            if "Score" in rep_df.columns:
                                port_score = float((pd.to_numeric(rep_df["Score"], errors="coerce") * rep_df["weight"]).sum())
                            else:
                                port_score = float("nan")
                            if "Sharpe" in rep_df.columns:
                                port_sharpe = float((pd.to_numeric(rep_df["Sharpe"], errors="coerce") * rep_df["weight"]).sum())
                            else:
                                port_sharpe = float("nan")
                            if "Drawdown" in rep_df.columns:
                                port_dd = float((pd.to_numeric(rep_df["Drawdown"], errors="coerce") * np.abs(rep_df["weight"])).sum())
                            else:
                                port_dd = float("nan")
                        except Exception:
                            port_score = port_sharpe = port_dd = float("nan")

                        st.write(
                            {
                                "Portfolio Score (approx)": port_score,
                                "Portfolio Sharpe (approx)": port_sharpe,
                                "Portfolio DD (weighted sum)": port_dd,
                            }
                        )

                        st.download_button(
                            "Download portfolio_representation.csv",
                            data=rep_df.to_csv(index=False).encode("utf-8"),
                            file_name=f"portfolio_rep_{pair_label}.csv",
                            mime="text/csv",
                            key=sk("pf_rep_dl"),
                        )
        except Exception as e:
            st.caption(f"Batch portfolio panel failed: {e}")

    # =========================================================
    # 11.4 Telemetry & Environment Snapshot
    # =========================
    with st.expander("📡 Telemetry & Environment Snapshot", expanded=False):
        try:
            info: Dict[str, Any] = {
                "rows": int(df.shape[0]) if df is not None else 0,
                "cols": int(df.shape[1]) if df is not None else 0,
            }
            try:
                mem = int(df.memory_usage(deep=True).sum()) if df is not None else 0
                info["bytes"] = mem
            except Exception:
                pass

            # DB info (אם קיים DuckDB)
            if duckdb is not None:
                try:
                    conn = get_ro_duck()
                    c_st = conn.execute("SELECT COUNT(*) FROM studies").fetchone()[0]
                    c_tr = conn.execute("SELECT COUNT(*) FROM trials").fetchone()[0]
                    info["duckdb_studies"] = int(c_st)
                    info["duckdb_trials"] = int(c_tr)
                except Exception:
                    pass
                try:
                    if DB_PATH.exists():
                        info["duckdb_file_bytes"] = int(DB_PATH.stat().st_size)
                except Exception:
                    pass

            # גרסאות ספריות מרכזיות
            versions = {
                "python": sys.version.split()[0],
                "pandas": _safe_version("pandas"),
                "numpy": _safe_version("numpy"),
                "duckdb": _safe_version("duckdb"),
                "optuna": _safe_version("optuna"),
                "sklearn": _safe_version("sklearn"),
                "streamlit": _safe_version("streamlit"),
            }
            info["versions"] = versions

            st.json(info)

            # Rolling Score monitoring
            if df is not None and "Score" in df.columns:
                try:
                    win = st.slider("Rolling monitoring window (rows)", 5, 400, 60, key=sk("mon_win"))
                    s = pd.to_numeric(df["Score"], errors="coerce").reset_index(drop=True)
                    mon = s.rolling(win, min_periods=max(5, win // 3)).mean()
                    mon_df = pd.DataFrame({"idx": mon.index, "Score_roll": mon.values})
                    if px is not None:
                        fig_mon = px.line(mon_df, x="idx", y="Score_roll", title="Rolling mean Score (monitoring)")
                        st.plotly_chart(fig_mon, width="stretch")
                    else:
                        st.dataframe(mon_df.tail(50), width="stretch", height=min(TABLE_HEIGHT, 260))
                except Exception:
                    pass
        except Exception as e:
            st.caption(f"Telemetry panel failed: {e}")
"""
חלק 12/15 — Portfolio Constructor Pro & Governance / Reproducibility Hub
=======================================================================

החלק הזה מוסיף שני פאנלים:

1. _render_optimization_portfolio_section(...)
   - בונה פורטפוליו של זוגות מתוך opt_df_batch:
     * בחירת pairs, בחירת אסטרטגיה מובילה לכל זוג (Best Score).
     * מספר שיטות משקלות (Equal / Score / Sharpe/DD / Risk-parity).
     * חישוב KPIs לתיק (Score / Sharpe / Drawdown בקירוב).
     * חישוב משקלי דולר (בהינתן Budget) ומגבלות כמו max_weight_per_pair.

2. _render_reproducibility_governance_section(...)
   - מרכז רפרודוקציה ו-Governance:
     * manifest אחרון (opt_last_manifest) + context (ranges/weights/env/versions).
     * הצגה במסך + הורדה כ-JSON.
     * שמירה כ-artifact ב-DuckDB (אם study_id זמין).
"""

def _render_optimization_portfolio_section(
    TABLE_HEIGHT: int,
    df: pd.DataFrame,
    sym1: str,
    sym2: str,
) -> None:
    FOCUS = bool(st.session_state.get("opt_focus_mode", False))

    with st.expander("🏦 Portfolio Constructor Pro (multi-pair sleeve)", expanded=not FOCUS):
        try:
            df_batch = st.session_state.get("opt_df_batch", pd.DataFrame())
            if df_batch is None or df_batch.empty:
                st.caption("No batch results (opt_df_batch) — run batch mode in the main tab to populate.")
                return

            if "Pair" not in df_batch.columns:
                st.caption("Batch dataframe has no 'Pair' column; cannot build portfolio.")
                return

            st.caption(
                f"Batch results: {len(df_batch)} rows across "
                f"{df_batch['Pair'].nunique()} pairs."
            )

            # =====================
            # הגדרות פורטפוליו גלובליות
            # =====================
            c1, c2, c3 = st.columns(3)
            budget = c1.number_input(
                "Budget per sleeve ($)",
                min_value=1_000.0,
                max_value=100_000_000.0,
                value=100_000.0,
                step=5_000.0,
                key=sk("pf_budget"),
            )
            max_weight = c2.slider(
                "Max weight per pair",
                min_value=0.05,
                max_value=1.0,
                value=0.3,
                step=0.05,
                key=sk("pf_max_weight"),
            )
            target_pairs = c3.slider(
                "Target #pairs in sleeve",
                min_value=1,
                max_value=min(20, int(df_batch["Pair"].nunique())),
                value=min(5, int(df_batch["Pair"].nunique())),
                step=1,
                key=sk("pf_target_pairs"),
            )

            # טבלת KPIs לפי זוג
            stats_cols = [c for c in ("Score", "Sharpe", "Profit", "Drawdown") if c in df_batch.columns]
            grp_pairs = df_batch.groupby("Pair")
            agg = grp_pairs[stats_cols].agg(["mean", "max", "min"]).reset_index()
            agg.columns = ["Pair"] + [f"{c}_{m}" for c, m in itertools.product(stats_cols, ("mean", "max", "min"))]
            st.dataframe(
                agg,
                width="stretch",
                height=min(TABLE_HEIGHT, 260),
            )

            # בחירת זוגות
            pairs_available = sorted(df_batch["Pair"].unique().tolist())
            default_sel = pairs_available[: int(target_pairs)]
            sel_pairs = st.multiselect(
                "Select pairs for portfolio sleeve",
                pairs_available,
                default=default_sel,
                key=sk("pf_pairs_pro"),
            )
            if not sel_pairs:
                st.caption("Select at least one pair to build a portfolio.")
                return

            # =====================
            # בניית אסטרטגיה מייצגת לכל זוג
            # =====================
            df_sel = df_batch[df_batch["Pair"].isin(sel_pairs)].copy()
            rep_rows = []
            for p in sel_pairs:
                sub = df_sel[df_sel["Pair"] == p]
                if "Score" in sub.columns:
                    sub = sub.sort_values("Score", ascending=False)
                rep_rows.append(sub.head(1))
            rep_df = pd.concat(rep_rows, ignore_index=True)

            st.caption("Representative strategy per pair (Top per Score):")
            cols_show = ["Pair"] + [c for c in ("Score", "Sharpe", "Drawdown", "Profit") if c in rep_df.columns]
            st.dataframe(
                rep_df[cols_show],
                width="stretch",
                height=min(TABLE_HEIGHT, 260),
            )

            # =====================
            # שיטות משקלות
            # =====================
            mode_weights = st.selectbox(
                "Weighting mode",
                ["Equal weights", "Score-proportional", "Sharpe/DD heuristic", "Risk-parity (1/vol)"],
                index=0,
                key=sk("pf_mode_pro"),
            )

            n_pairs = len(rep_df)
            # Score/Sharpe/DD/Profit series
            scr = pd.to_numeric(rep_df.get("Score", pd.Series(index=rep_df.index)), errors="coerce").fillna(0.0)
            sh = pd.to_numeric(rep_df.get("Sharpe", pd.Series(index=rep_df.index)), errors="coerce").fillna(0.0)
            dd = pd.to_numeric(rep_df.get("Drawdown", pd.Series(index=rep_df.index)), errors="coerce").fillna(0.1)
            # Estimate vol as Sharpe denominator proxy if אין עמודת Vol:
            vol_guess = sh.copy()
            vol_guess[vol_guess == 0] = 1.0
            vol = 1.0 / vol_guess.replace(0.0, np.nan).abs().fillna(1.0)  # בערך הפוך ל-Sharpe

            if mode_weights.startswith("Equal"):
                w_vec = np.ones(n_pairs)
            elif mode_weights.startswith("Score"):
                w_vec = scr.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
                if not np.any(np.abs(w_vec) > 0):
                    w_vec = np.ones(n_pairs)
            elif mode_weights.startswith("Sharpe/DD"):
                w_vec = (sh / (1.0 + dd)).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
                if not np.any(np.abs(w_vec) > 0):
                    w_vec = np.ones(n_pairs)
            else:  # Risk-parity (1/vol)
                w_vec = (1.0 / vol).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
                if not np.any(np.abs(w_vec) > 0):
                    w_vec = np.ones(n_pairs)

            # נורמליזציה לגבולות
            w_vec = np.clip(w_vec, 0.0, None)
            if not np.any(w_vec > 0):
                w_vec = np.ones(n_pairs)
            w = w_vec / (np.sum(w_vec) or 1.0)

            # enforce max_weight
            if max_weight < 1.0:
                w = np.minimum(w, max_weight)
                w = w / (np.sum(w) or 1.0)

            rep_df = rep_df.copy()
            rep_df["weight"] = w

            # =====================
            # חישוב משקלי דולר ו-KPIs
            # =====================
            rep_df["dollars"] = rep_df["weight"] * float(budget)

            try:
                if "Score" in rep_df.columns:
                    port_score = float((scr * rep_df["weight"]).sum())
                else:
                    port_score = float("nan")
                if "Sharpe" in rep_df.columns:
                    port_sharpe = float((sh * rep_df["weight"]).sum())
                else:
                    port_sharpe = float("nan")
                if "Drawdown" in rep_df.columns:
                    port_dd = float((dd.abs() * np.abs(rep_df["weight"])).sum())
                else:
                    port_dd = float("nan")
            except Exception:
                port_score = port_sharpe = port_dd = float("nan")

            st.caption("Portfolio sleeve (final weights & dollar allocation):")
            cols_port = ["Pair", "weight", "dollars"] + [c for c in ("Score", "Sharpe", "Drawdown", "Profit") if c in rep_df.columns]
            st.dataframe(
                rep_df[cols_port],
                width="stretch",
                height=min(TABLE_HEIGHT, 280),
            )

            st.write(
                {
                    "Budget": budget,
                    "Portfolio Score (approx)": port_score,
                    "Portfolio Sharpe (approx)": port_sharpe,
                    "Portfolio DD (weighted sum, approx)": port_dd,
                }
            )

            # =====================
            # שמירה כ-artifact + הורדות
            # =====================
            payload = rep_df[cols_port].copy()
            ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

            # הורדה ל-CSV
            st.download_button(
                "Download portfolio_representation.csv",
                data=payload.to_csv(index=False).encode("utf-8"),
                file_name=f"portfolio_representation_{len(sel_pairs)}pairs_{ts}.csv",
                mime="text/csv",
                key=sk("pf_dl_csv_pro"),
            )

            # הורדה כ-JSON (למשל לשליחת הפרופיל לסוכן ביצוע)
            portfolio_json = {
                "timestamp": ts,
                "pairs": payload["Pair"].tolist(),
                "weights": payload["weight"].tolist(),
                "dollars": payload["dollars"].tolist(),
                "budget": float(budget),
                "approx_kpi": {
                    "score": port_score,
                    "sharpe": port_sharpe,
                    "drawdown": port_dd,
                },
            }
            st.download_button(
                "Download portfolio_json.json",
                data=json.dumps(make_json_safe(portfolio_json), ensure_ascii=False, indent=2).encode("utf-8"),
                file_name=f"portfolio_json_{ts}.json",
                mime="application/json",
                key=sk("pf_dl_json_pro"),
            )

            # שמירה כ-artifact ב-DuckDB (אם יש study_id אחרון)
            last_study_id = st.session_state.get("opt_last_study_id")
            if duckdb is not None and last_study_id is not None:
                try:
                    payload_bytes = json.dumps(make_json_safe(portfolio_json), ensure_ascii=False, indent=2).encode("utf-8")
                    save_artifact_to_duck(int(last_study_id), kind="portfolio_json", payload=payload_bytes)
                    st.caption(f"Portfolio artifact saved for study_id={last_study_id}.")
                except Exception as e:
                    st.caption(f"Portfolio artifact save failed: {e}")

        except Exception as e:
            st.caption(f"Portfolio constructor panel failed: {e}")


def _render_reproducibility_governance_section(
    TABLE_HEIGHT: int,
    df: pd.DataFrame,
    sym1: str,
    sym2: str,
) -> None:
    """
    Governance & Reproducibility Hub:

    - מציג manifest אחרון (opt_last_manifest) אם יש.
    - מרחיב אותו עם:
        * paramspace_hash (אם אפשר לחשב).
        * active_ranges, loaded_weights_eff.
        * env_info (גרסאות, DB path, env).
        * opt_df summary (rows/cols/columns).
    - מאפשר הורדה כ-JSON.
    - שומר כ-artifact של "repro_manifest_json" ב-DuckDB אם יש study_id.
    """
    FOCUS = bool(st.session_state.get("opt_focus_mode", False))
    pair_label = f"{sym1}-{sym2}"

    with st.expander("🛡 Governance & Reproducibility Manifest (Pro)", expanded=not FOCUS):
        try:
            last_manifest = st.session_state.get("opt_last_manifest") or {}
            last_study_id = st.session_state.get("opt_last_study_id")
            ranges = st.session_state.get("active_ranges", {})
            weights = st.session_state.get("loaded_weights_eff", {"Sharpe": 0.5, "Profit": 0.3, "Drawdown": 0.2})

            # paramspace hash
            try:
                p_hash = last_manifest.get("paramspace_hash") or paramspace_hash(ranges)
            except Exception:
                p_hash = None

            # env_info (גרסאות, DB, env)
            env_info = {
                "env": getattr(SETTINGS, "env", "local"),
                "app_version": OPT_TAB_VERSION,
                "duckdb_path": str(DB_PATH),
                "python": sys.version.split()[0],
                "pandas": _safe_version("pandas"),
                "numpy": _safe_version("numpy"),
                "duckdb": _safe_version("duckdb"),
                "optuna": _safe_version("optuna"),
                "sklearn": _safe_version("sklearn"),
                "streamlit": _safe_version("streamlit"),
            }

            # opt_df summary
            df_summary = {}
            try:
                if df is not None and not df.empty:
                    df_summary = {
                        "rows": int(df.shape[0]),
                        "cols": int(df.shape[1]),
                        "columns": list(map(str, df.columns)),
                    }
            except Exception:
                df_summary = {}

            # Repro manifest מלא
            repro_manifest: Dict[str, Any] = {
                "pair": pair_label,
                "study_id": last_study_id,
                "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "run_id": st.session_state.get("opt_run_id") or st.session_state.get("run_id"),
                "paramspace_hash": p_hash,
                "active_ranges": make_json_safe(ranges),
                "weights": make_json_safe(weights),
                "env_info": env_info,
                "opt_df_summary": df_summary,
                "last_manifest_core": last_manifest,
            }

            st.json(repro_manifest)

            # הורדה
            ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            st.download_button(
                "Download reproducibility_manifest.json",
                data=json.dumps(make_json_safe(repro_manifest), ensure_ascii=False, indent=2).encode("utf-8"),
                file_name=f"repro_manifest_{pair_label}_{ts}.json",
                mime="application/json",
                key=sk("gov_dl_manifest_pro"),
            )

            # שמירה כ-artifact
            if duckdb is not None and last_study_id is not None:
                try:
                    payload_bytes = json.dumps(make_json_safe(repro_manifest), ensure_ascii=False, indent=2).encode("utf-8")
                    save_artifact_to_duck(int(last_study_id), kind="repro_manifest_json", payload=payload_bytes)
                    st.caption(f"Repro manifest artifact saved for study_id={last_study_id}.")
                except Exception as e:
                    st.caption(f"Repro manifest artifact save failed: {e}")

        except Exception as e:
            st.caption(f"Governance/reproducibility panel failed: {e}")
"""
חלק 13/15 — Dev Tools Pro, Error Console Pro, i18n & Finalize Chain
===================================================================

החלק הזה מרכז את כל מה שמפתח/מנהל צריך בסוף הטאב:

- Dev Tools:
  * חתימת Backtester.
  * מפת session_state["opt_*"].
  * SETTINGS וצורתם.
  * active_ranges / weights / mapping / risk_kwargs.
  * תקינות df (Score/Sharpe/DD וכו').

- Error Console:
  * הצגת opt_logs (אם אתה מזין לשם).
  * סינון לפי substring.
  * מחיקה.

- i18n / RTL:
  * RTL לטאב האופטימיזציה.
  * שפת תוויות.
  * העדפות פורמט מספר/תאריך (נשמרות ל-session לשימוש עתידי).

- Finalize:
  * קורא ל-Dev Tools, Error Console, i18n.
  * קורא לפאנל Portfolio Pro וחלון Governance/Repro (חלק 12).
  * יוצר opt_last_snapshot מסודר ל-dashboard ולשאר הטאבים.
"""


def _render_optimization_dev_tools(TABLE_HEIGHT: int, df: pd.DataFrame, sym1: str, sym2: str) -> None:
    """Developer-facing helpers: חתימות, session-keys, SETTINGS וכו'."""
    FOCUS = bool(st.session_state.get("opt_focus_mode", False))
    pair_label = f"{sym1}-{sym2}"

    with st.expander("🧰 Developer Mode — Diagnostics (OptTab)", expanded=False):
        try:
            # Backtester signature
            st.markdown("**Backtester signature**")
            try:
                if Backtester is None:
                    st.caption("Backtester not available.")
                else:
                    sig = inspect.signature(Backtester.__init__)
                    st.code(str(sig), language="text")
            except Exception as e:
                st.caption(f"Signature unavailable: {e}")

            # OptSettings summary
            st.markdown("**OptSettings summary**")
            try:
                settings_view = getattr(SETTINGS, "model_dump", None)
                if callable(settings_view):
                    st.json(SETTINGS.model_dump())
                else:
                    st.json(
                        {
                            "env": SETTINGS.env,
                            "log_level": SETTINGS.log_level,
                            "data_dir": str(SETTINGS.data_dir),
                            "studies_dir": str(SETTINGS.studies_dir),
                        }
                    )
            except Exception:
                st.caption("Unable to display SETTINGS (OptSettings).")

            # DuckDB info (path + existence)
            st.markdown("**DuckDB info**")
            try:
                info_db = {"DB_PATH": str(DB_PATH)}
                if DB_PATH.exists():
                    info_db["file_size_bytes"] = int(DB_PATH.stat().st_size)
                else:
                    info_db["file_size_bytes"] = None
                st.json(info_db)
            except Exception:
                st.caption("Cannot display DuckDB info.")

            # Session keys (opt_*)
            st.markdown("**Session keys (opt_*)**")
            try:
                keys = sorted(
                    [k for k in st.session_state.keys() if str(k).startswith("opt_")]
                )
                if keys:
                    st.code("\n".join(keys), language="text")
                else:
                    st.caption("(no opt_* keys yet)")
            except Exception:
                st.caption("Unable to list session_state keys.")

            # Param mapping (UI → Backtester)
            st.markdown("**Param mapping (UI → Backtester)**")
            try:
                mp = st.session_state.get("opt_param_mapping", {})
                st.json(mp or {})
            except Exception:
                st.caption("Mapping not available.")

            # Active ranges / weights / risk
            st.markdown("**Active ranges (head)**")
            try:
                rng = st.session_state.get("active_ranges", {})
                if rng:
                    preview = dict(list(rng.items())[:12])
                    st.json(preview)
                else:
                    st.caption("active_ranges is empty.")
            except Exception:
                st.caption("Cannot display active_ranges.")

            st.markdown("**Loaded scoring weights**")
            try:
                st.json(
                    st.session_state.get(
                        "loaded_weights_eff",
                        {"Sharpe": 0.5, "Profit": 0.3, "Drawdown": 0.2},
                    )
                )
            except Exception:
                st.caption("Cannot display weights.")

            st.markdown("**Risk kwargs (slippage/fees/max_positions/rebalance)**")
            try:
                st.json(get_session_risk_kwargs())
            except Exception:
                st.caption("Cannot display risk kwargs.")

            # df State summary
            st.markdown("**opt_df state summary**")
            try:
                summary = {
                    "pair": pair_label,
                    "rows": int(df.shape[0]) if df is not None else 0,
                    "cols": int(df.shape[1]) if df is not None else 0,
                    "has_Score": bool("Score" in df.columns) if df is not None else False,
                    "has_Sharpe": bool("Sharpe" in df.columns) if df is not None else False,
                    "has_Profit": bool("Profit" in df.columns) if df is not None else False,
                    "has_Drawdown": bool("Drawdown" in df.columns) if df is not None else False,
                }
                st.json(summary)
            except Exception:
                st.caption("Cannot summarise opt_df.")
        except Exception as e:
            st.caption(f"Dev tools failed: {e}")


def _render_optimization_error_console() -> None:
    """Error console dedicated to OptTab (opt_logs)."""
    with st.expander("❗ Error Console — Optimisation Tab", expanded=False):
        try:
            buf = st.session_state.get("opt_logs", [])
            if not buf:
                st.caption("No errors logged yet (opt_logs is empty).")
                return

            c1, c2 = st.columns([3, 1])
            q = c1.text_input("Filter (contains)", value="", key=sk("opt_err_filter"))
            show_n = int(
                c2.number_input(
                    "Last N lines",
                    min_value=50,
                    max_value=1000,
                    value=500,
                    step=50,
                    key=sk("opt_err_n"),
                )
            )
            lines = [ln for ln in buf if (q in ln)] if q else buf
            st.code("\n".join(lines[-show_n:]), language="text")

            c3, c4 = st.columns(2)
            if c3.button("Clear opt_logs", key=sk("opt_err_clear")):
                st.session_state["opt_logs"] = []
                st.success("opt_logs cleared.")
            if c4.button("Copy error buffer to clipboard (browser)", key=sk("opt_err_copy")):
                # Streamlit לא נותן גישה ישירה ל-clipboard, אבל
                # אם תפתח כלי dev תוכלו להעתיק מהקוד המוצג.
                st.info("Copy from the code block above (clipboard support is browser-dependent).")
        except Exception as e:
            st.caption(f"Error console failed: {e}")


def _render_optimization_i18n_controls() -> None:
    """i18n/RTL toggles for the optimisation tab."""
    with st.expander("🌐 i18n / RTL — Optimisation Tab", expanded=False):
        try:
            rtl = st.checkbox(
                "Enable RTL layout for this tab",
                value=bool(st.session_state.get("opt_rtl", False)),
                key=sk("opt_i18n_rtl"),
            )
            st.session_state["opt_rtl"] = rtl

            lang = st.selectbox(
                "Language for labels (tab-local)",
                ["auto", "he", "en"],
                index=["auto", "he", "en"].index(str(st.session_state.get("opt_lang", "auto"))),
                key=sk("opt_i18n_lang"),
            )
            st.session_state["opt_lang"] = lang

            # פורמט מספר/תאריך (הגדרות עתידיות, נשמרות ב-session למימוש אחיד)
            st.markdown("**Formatting preferences (stored for future use)**")
            c1, c2 = st.columns(2)
            num_fmt = c1.selectbox(
                "Number format",
                ["default", "2dp", "4dp", "scientific"],
                index=0,
                key=sk("opt_fmt_num"),
            )
            date_fmt = c2.selectbox(
                "Date format",
                ["YYYY-MM-DD", "DD/MM/YYYY", "MM/DD/YYYY"],
                index=0,
                key=sk("opt_fmt_date"),
            )
            st.session_state["opt_num_fmt"] = num_fmt
            st.session_state["opt_date_fmt"] = date_fmt

            st.caption(
                "Note: these preferences are stored in session_state and can be "
                "used by future versions of the tab/UI to render numbers/dates accordingly."
            )
        except Exception as e:
            st.caption(f"i18n controls failed: {e}")


def _render_optimization_finalize(
    TABLE_HEIGHT: int,
    df: pd.DataFrame,
    sym1: str,
    sym2: str,
) -> None:
    """
    Final wrapper שנקרא בסוף render_optimization_tab:

    - Dev Tools Pro (דיאגנוסטיקה).
    - Error Console Pro.
    - i18n / RTL controls.
    - Portfolio Constructor Pro (חלק 12).
    - Governance / Reproducibility Hub (חלק 12).
    - Snapshot ל-session (opt_last_snapshot) לטובת dashboard/טאבים אחרים.
    """
    # Dev tools
    try:
        _render_optimization_dev_tools(TABLE_HEIGHT, df, sym1, sym2)
    except Exception as e:
        st.caption(f"Dev tools failed: {e}")

    # Error console
    try:
        _render_optimization_error_console()
    except Exception as e:
        st.caption(f"Error console failed: {e}")

    # i18n/RTL controls
    try:
        _render_optimization_i18n_controls()
    except Exception as e:
        st.caption(f"i18n controls failed: {e}")

    # Snapshot קצר ל-session
    try:
        snap = {
            "pair": f"{sym1}-{sym2}",
            "rows": int(df.shape[0]) if df is not None else 0,
            "cols": int(df.shape[1]) if df is not None else 0,
            "has_Score": bool(df is not None and "Score" in df.columns),
            "best_score": float(
                pd.to_numeric(df.get("Score", pd.Series(index=df.index)), errors="coerce").max()
            ) if df is not None and "Score" in df.columns and not df.empty else None,
            "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        st.session_state["opt_last_snapshot"] = snap
    except Exception:
        pass
"""
חלק 14/15 — Macro & Factor Overlay + Audit Pack
===============================================

פאנלים:

1. _render_optimization_macro_factor_section:
   - עבודה עם CSV פקטורים וחשיפות.
   - ניתוח תרחיש shocks לפי פקטורים.
   - Impact table + גרף + הורדה.

2. _render_optimization_audit_pack_section:
   - איסוף כל פרופיל האופטימיזציה לחבילה אחת.
   - הורדה כ-JSON.
   - שמירה כ-artifact ב-DuckDB (audit_pack_json) אם יש study_id.
"""


def _render_optimization_macro_factor_section(
    TABLE_HEIGHT: int,
    df: pd.DataFrame,
    sym1: str,
    sym2: str,
) -> None:
    """
    Macro / Factor Overlay:
    -----------------------
    מאפשר:
    - העלאת קובץ CSV עם פקטורי מאקרו/פקטורים (Date, Factor1, Factor2, ...).
    - (אופציונלי) העלאת CSV עם חשיפות פקטוריות לכל Pair:
        Pair, Factor1, Factor2, ...
    - הגדרת shocks לכל פקטור (למשל +1 std, -2 std).
    - חישוב Scenario Impact לכל זוג לפי אינווקטור exposures · shocks (dot product).

    הערה: מדובר ב-layer אנליטי מעל התוצאות — לא נוגעים ב-backtester.
    """
    FOCUS = bool(st.session_state.get("opt_focus_mode", False))

    with st.expander("🌍 Macro & Factor Overlay (scenario analysis)", expanded=not FOCUS):
        try:
            # =====================
            # Factor time-series CSV
            # =====================
            st.markdown("#### 1. Factor returns time-series (optional)")
            fac_upl = st.file_uploader(
                "Upload factor CSV (Date + Factor1, Factor2, ...)",
                type=["csv"],
                key=sk("fac_ts_upl"),
            )

            df_factors = pd.DataFrame()
            if fac_upl is not None:
                try:
                    df_factors = pd.read_csv(fac_upl)
                    # ננרמל עמודת Date אם קיימת
                    if "Date" in df_factors.columns:
                        df_factors["Date"] = pd.to_datetime(df_factors["Date"], errors="coerce")
                    st.dataframe(
                        df_factors.head(20),
                        width="stretch",
                        height=min(TABLE_HEIGHT, 260),
                    )
                except Exception as e:
                    st.caption(f"Failed to parse factor CSV: {e}")

            # =====================
            # Factor Exposure per Pair
            # =====================
            st.markdown("#### 2. Factor exposures per pair")
            st.caption(
                "Upload exposures CSV with columns: `Pair` (matching opt_df_batch) and factor columns "
                "(same names as factor file or arbitrary names)."
            )

            df_exposures = st.session_state.get("_opt_factor_exposures", pd.DataFrame())
            exp_upl = st.file_uploader(
                "Upload exposures CSV (Pair, Factor1, Factor2, ...)",
                type=["csv"],
                key=sk("fac_exp_upl"),
            )

            if exp_upl is not None:
                try:
                    df_exposures = pd.read_csv(exp_upl)
                    st.session_state["_opt_factor_exposures"] = df_exposures.copy()
                except Exception as e:
                    st.caption(f"Failed to parse exposures CSV: {e}")

            if df_exposures is None or df_exposures.empty:
                st.caption(
                    "No exposures loaded. You can still define ad-hoc exposures for a single pair below."
                )

            # רשימת פקטורים מתוך exposures או factor CSV
            factor_cols: List[str] = []
            if not df_exposures.empty:
                factor_cols = [c for c in df_exposures.columns if c != "Pair"]
            elif not df_factors.empty:
                factor_cols = [
                    c for c in df_factors.columns if c != "Date" and pd.api.types.is_numeric_dtype(df_factors[c])
                ]

            factor_cols = sorted(factor_cols)
            if not factor_cols:
                st.caption("No factor names inferred yet (upload exposures/factors).")
                # עדיין נמשיך לתת אפשרות ad-hoc עבור זוג אחד
            else:
                st.caption(f"Detected factors: {', '.join(factor_cols)}")

            # =====================
            # Single-pair ad-hoc exposures
            # =====================
            st.markdown("#### 3. Ad-hoc exposures for current pair")
            exp_current: Dict[str, float] = {}
            if factor_cols:
                for fc in factor_cols:
                    val = st.number_input(
                        f"Exposure of {sym1}-{sym2} to {fc}",
                        min_value=-10.0,
                        max_value=10.0,
                        value=0.0,
                        step=0.1,
                        key=sk(f"fac_exp_{fc}"),
                    )
                    exp_current[fc] = float(val)

            # =====================
            # Shocks UI
            # =====================
            st.markdown("#### 4. Scenario shocks per factor")
            shocks: Dict[str, float] = {}
            if factor_cols:
                for fc in factor_cols:
                    val = st.number_input(
                        f"Shock for {fc} (e.g. +1 = +1σ move)",
                        min_value=-10.0,
                        max_value=10.0,
                        value=0.0,
                        step=0.25,
                        key=sk(f"fac_shock_{fc}"),
                    )
                    shocks[fc] = float(val)
            else:
                st.caption("Define factors to enable scenario shocks.")

            # =====================
            # Impact table (multi-pair) from opt_df_batch if possible
            # =====================
            df_batch = st.session_state.get("opt_df_batch", pd.DataFrame())
            if df_batch is None or df_batch.empty or "Pair" not in df_batch.columns or not factor_cols:
                st.caption("Impact table requires opt_df_batch with 'Pair' and factor definitions.")
            else:
                st.markdown("#### 5. Scenario impact table (multi-pair)")

                # Build exposures matrix
                # Priority: df_exposures; fallback: par exposures for current pair only.
                df_exp = df_exposures.copy()
                if not df_exp.empty:
                    if "Pair" not in df_exp.columns:
                        st.caption("Exposures CSV missing 'Pair' column — cannot join.")
                        df_exp = pd.DataFrame()
                    else:
                        # ensure only relevant cols
                        df_exp = df_exp[["Pair"] + [c for c in factor_cols if c in df_exp.columns]]
                else:
                    # ad-hoc exposures only for current pair
                    df_exp = pd.DataFrame({"Pair": [f"{sym1}-{sym2}"]})
                    for fc in factor_cols:
                        df_exp[fc] = float(exp_current.get(fc, 0.0))

                # join batch pairs with exposures
                pairs_in_batch = sorted(df_batch["Pair"].unique().tolist())
                df_pairs = pd.DataFrame({"Pair": pairs_in_batch})
                df_join = df_pairs.merge(df_exp, on="Pair", how="left").fillna(0.0)

                # compute impact = sum(exposure * shock) per pair
                impacts: List[float] = []
                for _, row in df_join.iterrows():
                    imp = 0.0
                    for fc in factor_cols:
                        imp += float(row.get(fc, 0.0)) * float(shocks.get(fc, 0.0))
                    impacts.append(float(imp))
                df_join["ScenarioImpact"] = impacts

                # sort by | impact | desc
                df_join = df_join.sort_values("ScenarioImpact", ascending=False)
                st.dataframe(
                    df_join,
                    width="stretch",
                    height=min(TABLE_HEIGHT, 360),
                )

                # bar chart
                try:
                    if px is not None:
                        fig_imp = px.bar(
                            df_join,
                            x="Pair",
                            y="ScenarioImpact",
                            title="Scenario Impact per pair (factor shock · exposure)",
                        )
                        st.plotly_chart(fig_imp, width="stretch")
                except Exception:
                    pass

                # הורדה ל-CSV
                st.download_button(
                    "Download scenario_impact.csv",
                    data=df_join.to_csv(index=False).encode("utf-8"),
                    file_name="scenario_impact.csv",
                    mime="text/csv",
                    key=sk("fac_imp_dl"),
                )
        except Exception as e:
            st.caption(f"Macro & factor overlay panel failed: {e}")


def _render_optimization_audit_pack_section(
    TABLE_HEIGHT: int,
    df: pd.DataFrame,
    sym1: str,
    sym2: str,
) -> None:
    """
    Audit & Profiles Pack:
    ----------------------
    אוסף חבילת JSON אחת שמרכזת:

    - SETTINGS (OptSettings).
    - active_ranges, loaded_weights_eff, opt_param_mapping, risk kwargs.
    - presets (_opt_presets) — רק metadata (לא כל תוכן אם כבד).
    - datasets (_opt_datasets) — רשימת שמות בלבד.
    - opt_last_snapshot, opt_last_manifest (אם קיימים).

    מאפשר:
    - הורדה של audit_pack.json.
    - שמירה כ-artifact ב-DuckDB (audit_pack_json) אם יש study_id.
    """
    FOCUS = bool(st.session_state.get("opt_focus_mode", False))
    pair_label = f"{sym1}-{sym2}"

    with st.expander("📁 Audit & Profiles Pack (JSON)", expanded=False):
        try:
            # SETTINGS
            try:
                settings_view = getattr(SETTINGS, "model_dump", None)
                settings_dict = SETTINGS.model_dump() if callable(settings_view) else {
                    "env": SETTINGS.env,
                    "log_level": SETTINGS.log_level,
                    "data_dir": str(SETTINGS.data_dir),
                    "studies_dir": str(SETTINGS.studies_dir),
                }
            except Exception:
                settings_dict = {}

            # active_ranges, weights, mapping, risk_kwargs
            ranges = st.session_state.get("active_ranges", {})
            weights = st.session_state.get("loaded_weights_eff", {"Sharpe": 0.5, "Profit": 0.3, "Drawdown": 0.2})
            mapping = st.session_state.get("opt_param_mapping", {})
            risk_kwargs = get_session_risk_kwargs()

            # presets & datasets
            presets = st.session_state.get("_opt_presets", {})
            datasets = st.session_state.get("_opt_datasets", {})
            dataset_names = sorted(datasets.keys()) if isinstance(datasets, dict) else []

            # last snapshot & manifest
            last_snapshot = st.session_state.get("opt_last_snapshot", {})
            last_manifest = st.session_state.get("opt_last_manifest", {})

            audit_pack: Dict[str, Any] = {
                "pair": pair_label,
                "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "run_id": st.session_state.get("opt_run_id") or st.session_state.get("run_id"),
                "settings": settings_dict,
                "ranges": make_json_safe(ranges),
                "weights": make_json_safe(weights),
                "mapping": make_json_safe(mapping),
                "risk_kwargs": make_json_safe(risk_kwargs),
                "presets_metadata": {
                    name: {"timestamp": p.get("timestamp")} for name, p in (presets or {}).items()
                },
                "datasets_available": dataset_names,
                "last_snapshot": last_snapshot,
                "last_manifest": last_manifest,
            }

            st.json(audit_pack)

            ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            st.download_button(
                "Download audit_pack.json",
                data=json.dumps(make_json_safe(audit_pack), ensure_ascii=False, indent=2).encode("utf-8"),
                file_name=f"audit_pack_{pair_label}_{ts}.json",
                mime="application/json",
                key=sk("audit_dl_json"),
            )

            # שמירה כ-artifact אם אפשר
            last_study_id = st.session_state.get("opt_last_study_id")
            if duckdb is not None and last_study_id is not None:
                try:
                    payload_bytes = json.dumps(make_json_safe(audit_pack), ensure_ascii=False, indent=2).encode("utf-8")
                    save_artifact_to_duck(int(last_study_id), kind="audit_pack_json", payload=payload_bytes)
                    st.caption(f"Audit pack artifact saved for study_id={last_study_id}.")
                except Exception as e:
                    st.caption(f"Audit pack artifact save failed: {e}")

        except Exception as e:
            st.caption(f"Audit & Profiles pack panel failed: {e}")
"""
חלק 15/15 — Validation & Public API Hooks
=========================================

החלק הזה הופך את optimization_tab גם ל-"ספריית ליבה" עם API חיצוני:

1. validate_opt_df(df) -> Dict[str, Any]
   - בודק האם df של תוצאות אופטימיזציה "בריא":
     * האם קיימים Score/Sharpe/Drawdown?
     * האם יש מספיק שורות?
     * האם יש outliers קיצוניים?

2. api_optimize_pair(...)
   - מאפשר להריץ אופטימיזציה על זוג (sym1, sym2) מתוך קוד (CLI/סוכן AI),
     בלי UI ובלי Streamlit – רק DataFrame ו-metadata.

3. api_load_latest_study(pair: str) -> Tuple[pd.DataFrame, Dict[str, Any]]
   - מחזיר את df התוצאות של ה-study האחרון עבור pair ממסד DuckDB + manifest קטן.

4. api_get_portfolio_from_batch(...)
   - מפיק פורטפוליו בסיסי מתוך df_batch (בדומה לפאנל ה-Portfolio),
     אך ברמת API (ללא Streamlit) — מחזיר DataFrame עם משקלים.

הפונקציות הללו לא תלויות ב-Streamlit UI; הן מנצלות את המנוע שבנינו.
"""

# =========================
# 15.1 Validation helper
# =========================

def validate_opt_df(df: pd.DataFrame) -> Dict[str, Any]:
    """
    רץ על df של אופטימיזציה ומחזיר דו"ח קטן:

    החזרה:
        {
            "ok": bool,
            "issues": List[str],
            "summary": {
                "rows": int,
                "cols": int,
                "has_Score": bool,
                "has_Sharpe": bool,
                "has_Drawdown": bool,
                "score_max": float|None,
                "score_min": float|None,
                "sharpe_max": float|None,
                ...
            }
        }

    שימוש:
        report = validate_opt_df(df)
        if not report["ok"]: ...  # להימנע מהעלאה לפרודקשן
    """
    issues: List[str] = []
    summary: Dict[str, Any] = {
        "rows": 0,
        "cols": 0,
        "has_Score": False,
        "has_Sharpe": False,
        "has_Drawdown": False,
        "score_max": None,
        "score_min": None,
        "sharpe_max": None,
        "sharpe_min": None,
        "dd_max": None,
        "dd_min": None,
    }

    if df is None or df.empty:
        issues.append("opt_df is empty.")
        return {"ok": False, "issues": issues, "summary": summary}

    summary["rows"] = int(df.shape[0])
    summary["cols"] = int(df.shape[1])

    # Score
    if "Score" in df.columns:
        summary["has_Score"] = True
        try:
            sc = pd.to_numeric(df["Score"], errors="coerce")
            summary["score_max"] = float(sc.max())
            summary["score_min"] = float(sc.min())
            if sc.isna().all():
                issues.append("Score column contains only NaN.")
        except Exception:
            issues.append("Failed to parse Score column numerically.")
    else:
        issues.append("Missing 'Score' column.")

    # Sharpe
    if "Sharpe" in df.columns:
        summary["has_Sharpe"] = True
        try:
            sh = pd.to_numeric(df["Sharpe"], errors="coerce")
            summary["sharpe_max"] = float(sh.max())
            summary["sharpe_min"] = float(sh.min())
        except Exception:
            issues.append("Failed to parse Sharpe column numerically.")
    else:
        issues.append("Missing 'Sharpe' column (may still be ok for dev runs).")

    # Drawdown
    if "Drawdown" in df.columns:
        summary["has_Drawdown"] = True
        try:
            dd = pd.to_numeric(df["Drawdown"], errors="coerce").abs()
            summary["dd_max"] = float(dd.max())
            summary["dd_min"] = float(dd.min())
            # אזהרה אם Drawdown > 2 במונחים יחסיים
            if dd.max() > 2.0:
                issues.append("Some Drawdown values > 2.0 (check normalization).")
        except Exception:
            issues.append("Failed to parse Drawdown column numerically.")
    else:
        issues.append("Missing 'Drawdown' column (risk view incomplete).")

    ok = len(issues) == 0
    return {"ok": ok, "issues": issues, "summary": summary}

def derive_param_ranges_for_pair_from_opt_df(
    df: pd.DataFrame,
    pair: str,
    *,
    profile: str = "default",
    env: Optional[str] = None,
    elite_frac: float = 0.2,
    sharpe_min: float = 0.0,
    dd_max: float = 0.3,
    min_rows: int = 20,
) -> Dict[str, ParamRange]:
    """
    מפיק טווחי פרמטרים "אליטיים" עבור pair מתוך opt_df של Optuna/Zoom.

    לוגיקה:
    --------
    1. בוחר top-X% לפי Score (או לפי Profit אם Score חסר).
    2. מסנן לפי איכות:
       - Sharpe > sharpe_min
       - Drawdown < dd_max
    3. מחשב quantiles (10–90%) לחבורה מצומצמת של פרמטרים:
         z_entry, z_exit, lookback, hedge_ratio, half_life, notional
    4. מרחיב קצת (×1.2 סביב המרכז) כדי לא להתקבע מדי.

    מחזיר dict:
        name -> (lo, hi, step=None)
    """
    if df is None or df.empty:
        return {}

    df_loc = df.copy()

    # 1) Score or Profit as ordering
    if "Score" in df_loc.columns:
        base = pd.to_numeric(df_loc["Score"], errors="coerce")
    elif "Profit" in df_loc.columns:
        base = pd.to_numeric(df_loc["Profit"], errors="coerce")
    else:
        # אין שום מדד → נחזיר כלום
        return {}

    df_loc = df_loc.loc[base.notna()].copy()
    if df_loc.shape[0] < max(5, min_rows):
        return {}

    elite_frac = float(max(0.01, min(elite_frac, 0.9)))
    q = base.quantile(1.0 - elite_frac)
    elite = df_loc[base >= q].copy()
    if elite.shape[0] < min_rows:
        elite = df_loc.sort_values(base.name, ascending=False).head(max(min_rows, int(len(df_loc) * elite_frac)))

    # 2) סינון לפי Sharpe / Drawdown אם קיימים
    if "Sharpe" in elite.columns:
        sh = pd.to_numeric(elite["Sharpe"], errors="coerce")
        elite = elite.loc[sh > float(sharpe_min)].copy()
    if "Drawdown" in elite.columns:
        dd = pd.to_numeric(elite["Drawdown"], errors="coerce").abs()
        elite = elite.loc[dd < float(dd_max)].copy()

    if elite.shape[0] < max(5, min_rows // 2):
        # אם נשאר מעט מדי – לא נלמד טווחים
        return {}

    # 3) פרמטרים חשובים בלבד
    core_params = [
        "z_entry",
        "z_exit",
        "lookback",
        "hedge_ratio",
        "half_life",
        "notional",
    ]
    ranges: Dict[str, ParamRange] = {}

    for name in core_params:
        if name not in elite.columns:
            continue
        s = pd.to_numeric(elite[name], errors="coerce").dropna()
        if s.empty:
            continue

        q_lo = float(s.quantile(0.10))
        q_hi = float(s.quantile(0.90))
        if q_hi <= q_lo:
            q_hi = q_lo + 1e-9

        # הרחבה קטנה (×1.2 סביב המרכז)
        mid = 0.5 * (q_lo + q_hi)
        span = (q_hi - q_lo) * 1.2 / 2.0
        lo2 = mid - span
        hi2 = mid + span
        if hi2 <= lo2:
            hi2 = lo2 + 1e-9

        ranges[name] = (float(lo2), float(hi2), None)

    return ranges


def derive_and_save_param_ranges_for_pair(
    df: pd.DataFrame,
    pair: str,
    *,
    profile: str = "default",
    env: Optional[str] = None,
) -> Dict[str, ParamRange]:
    """
    עטיפה נוחה:
    - מפיק טווחים מ-opt_df עבור pair.
    - אם יש טווחים → שומר אותם ב-param_ranges (DuckDB).
    - מחזיר את הטווחים.
    """
    env_str = env or getattr(SETTINGS, "env", "local") or "local"
    ranges = derive_param_ranges_for_pair_from_opt_df(
        df,
        pair,
        profile=profile,
        env=env_str,
        elite_frac=0.2,
        sharpe_min=0.0,
        dd_max=0.3,
        min_rows=20,
    )
    if not ranges:
        logger.info("derive_and_save_param_ranges_for_pair: no ranges learned for %s", pair)
        return {}

    save_param_ranges_for_pair_in_duck(pair, ranges, env=env_str, profile=profile)
    return ranges

def select_live_candidate_from_opt_df(
    df: pd.DataFrame,
    *,
    min_dsr: float = 1.0,
    min_wf_penalty: float = 0.7,
    top_frac: float = 0.2,
) -> Tuple[Dict[str, Any], Optional[float]]:
    """
    בוחר "Live-ready" candidate אחד מתוך df של אופטימיזציה, בצורה יציבה ומקצועית.

    הלוגיקה:
    --------
    1. אם df ריק → מחזיר ({}, None).
    2. אם אין עמודת 'Score' → לוקח את השורה הראשונה כ-fallback.
    3. אחרת:
       א. מסנן ל-top X% לפי Score (ברירת מחדל 20%).
       ב. אם קיימת עמודת DSR → מסנן ל-DSR >= min_dsr.
       ג. אם קיימת עמודת wf_robust_penalty → מסנן ל-wf_robust_penalty >= min_wf_penalty.
       ד. אם אחרי כל הסינונים הסדרה ריקה → חוזר ל-top לפי Score בלבד.
       ה. מתוך הסט הנותר בוחר את השורה עם Score הגבוה ביותר.

    מחזיר:
        best_params_dict, best_score
    """
    if df is None or df.empty:
        return {}, None

    # אם אין Score – fallback גס
    if "Score" not in df.columns:
        row = df.iloc[0]
        params = _extract_params_from_opt_row(row)
        return params, None

    # --- Top X% לפי Score ---
    sc = pd.to_numeric(df["Score"], errors="coerce")
    df_loc = df.loc[sc.notna()].copy()
    if df_loc.empty:
        return {}, None

    top_frac = float(max(0.01, min(top_frac, 1.0)))
    q = sc.quantile(1.0 - top_frac)
    df_top = df_loc[sc >= q].copy()
    if df_top.empty:
        df_top = df_loc.sort_values("Score", ascending=False).head(max(5, int(len(df_loc) * top_frac)))

    # --- סינון לפי DSR אם קיים ---
    if "DSR" in df_top.columns and min_dsr is not None:
        dsr_vals = pd.to_numeric(df_top["DSR"], errors="coerce")
        mask_dsr = dsr_vals >= float(min_dsr)
        if mask_dsr.any():
            df_top = df_top.loc[mask_dsr].copy()

    # --- סינון לפי WF robustness אם קיים ---
    if "wf_robust_penalty" in df_top.columns and min_wf_penalty is not None:
        wf_vals = pd.to_numeric(df_top["wf_robust_penalty"], errors="coerce")
        mask_wf = wf_vals >= float(min_wf_penalty)
        if mask_wf.any():
            df_top = df_top.loc[mask_wf].copy()

    # אם אחרי הסינונים ריק – חוזרים לטופ לפי Score
    if df_top.empty:
        df_top = df_loc.sort_values("Score", ascending=False).head(max(5, int(len(df_loc) * top_frac)))

    # --- בחירת מועמד סופי: מקס Score ---
    df_top = df_top.sort_values("Score", ascending=False)
    row_best = df_top.iloc[0]
    best_params = _extract_params_from_opt_row(row_best)

    try:
        best_score = float(pd.to_numeric(df_top["Score"], errors="coerce").max())
    except Exception:
        best_score = None

    return best_params, best_score

def register_best_params_for_pair(
    pair_label: str,
    best_params: Dict[str, Any],
    *,
    score: Any = None,
    profile: Optional[str] = None,
) -> None:
    """
    רישום סט פרמטרים "הכי טובים" ל-Zוג ב-Session Registry גלובלי:

        st.session_state["opt_best_params_registry"] = {
            "XLY-XLP": {
                "params": {...},
                "score": 1.42,
                "profile": "defensive",
                "updated_at": "2025-12-03T17:21:00Z",
            },
            ...
        }

    המטרה:
    - Backtest / Execution / Agents יוכלו לשלוף במהירות:
        best = st.session_state["opt_best_params_registry"].get("XLY-XLP")
    """
    if not pair_label:
        return

    # 👇 הגנת LIVE/PROD — לא לעדכן registry בלי opt_live_update_params
    try:
        env_str = str(getattr(SETTINGS, "env", "local") or "local").lower()
    except Exception:
        env_str = "local"

    try:
        allow_live_update = bool(st.session_state.get("opt_live_update_params", False))
    except Exception:
        allow_live_update = False

    if env_str in {"live", "prod"} and not allow_live_update:
        logger.info(
            "register_best_params_for_pair: skipping registry update for %s in env=%s "
            "(opt_live_update_params=False)",
            pair_label,
            env_str,
        )
        return

    try:
        reg = st.session_state.get("opt_best_params_registry", {})
        if not isinstance(reg, dict):
            reg = {}
    except Exception:
        reg = {}

    entry: Dict[str, Any] = {
        "params": dict(best_params or {}),
        "score": _safe_float_or_none_opt(score),
        "profile": profile,
        "env": env_str,
    }
    try:
        dt = datetime.now(timezone.utc)
        entry["updated_at"] = dt.isoformat(timespec="seconds").replace("+00:00", "Z")
    except Exception:
        entry["updated_at"] = None

    reg[str(pair_label)] = entry
    st.session_state["opt_best_params_registry"] = reg


# =========================
# 15.2 API — single-pair optimisation
# =========================

def api_optimize_pair(
    sym1: str,
    sym2: str,
    *,
    ranges: Optional[Dict[str, ParamRange]] = None,
    weights: Optional[Dict[str, float]] = None,
    n_trials: int = 50,
    timeout_min: int = 10,
    direction: str = "maximize",
    sampler_name: str = "TPE",
    pruner_name: str = "median",
    profile: str = "default",
    multi_objective: bool = False,
    objective_metrics: Optional[List[str]] = None,
    param_mapping: Optional[Dict[str, str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    API חיצוני להרצת אופטימיזציה על זוג (ללא UI, מתאים לסקריפטים/Agents).

    שימוש לדוגמה (CLI/סקריפט):

        from root.optimization_tab import api_optimize_pair

        df_res, meta = api_optimize_pair(
            "XLY", "XLP",
            n_trials=200,
            timeout_min=10,
            profile="defensive",
        )

    החזרה:
        df_sorted, metadata dict:
            df_sorted — DataFrame תוצאות ממויין לפי Score (או כפי ש-run_optuna_for_pair מחזיר).

            meta ≈ {
              "pair": "XLY-XLP",
              "study_id": int|None,
              "n_trials": int,
              "timeout_min": int,
              "direction": str,
              "profile": str,
              "multi_objective": bool,
              "objective_metrics": [...],
              "paramspace_hash": str|None,
              "weights_used": {...},
              "ranges_used": {...},   # מצומצם לפי שם → (lo,hi,step)
              "status": "ok" | "error",
              "error_message": str|None,
              "duration_sec": float,
              "best_score": float|None,
              "global_seed": int|None,
            }

    הערות:
    -------
    - הפונקציה משתמשת במנוע run_optuna_for_pair (כולל DuckDB) כפי שנבנה בטאב.
    - אפשר להעביר ranges/weights מותאמים, או לתת לו לבנות לבד.
    - במקרה של Exception, הפונקציה לא מפילה את הסקריפט אלא מחזירה df ריק ו-meta עם status="error".
    """
    t0 = time.time()
    status = "ok"
    error_message: Optional[str] = None

    # ---- 1) Normalise pair labels ----
    sym1 = str(sym1).strip()
    sym2 = str(sym2).strip()
    pair_label = f"{sym1}-{sym2}"

    # ---- 2) Ranges: build / validate / augment ----
    try:
        if ranges is None:
            # full default ranges לפי פרופיל
            ranges = get_default_param_ranges(profile=profile)
        else:
            # אם הגיע dict ריק או קטן מדי → נרחיב מה-defaults
            base = get_default_param_ranges(profile=profile)
            if not ranges:
                ranges = base
            else:
                # נוודא שכל פרמטר חשוב מופיע (לפחות מה-defaults)
                missing = [k for k in base.keys() if k not in ranges]
                for name in missing:
                    ranges[name] = base[name]
    except Exception as e:
        logger.warning("api_optimize_pair: get_default_param_ranges failed for %s: %s", pair_label, e)
        ranges = ranges or {}
        if not ranges:
            # fallback מינימלי
            ranges = {
                "z_entry": (1.0, 3.0, 0.1),
                "z_exit": (0.1, 2.0, 0.1),
            }

    # paramspace hash (אם יש פונקציה)
    try:
        paramspace_hash_value = paramspace_hash(ranges)  # type: ignore[arg-type]
    except Exception:
        paramspace_hash_value = None

    # ---- 3) Weights: build / normalise ----
    if weights is None:
        weights = {
            "Sharpe": 0.4,
            "Profit": 0.3,
            "Drawdown": 0.2,
            "Sortino": 0.05,
            "Calmar": 0.05,
        }

    # נעשה נורמליזציה רכה לסכום |weights|
    try:
        w_raw = {k: float(v) for k, v in (weights or {}).items()}
        z = sum(abs(v) for v in w_raw.values())
        if z <= 0:
            raise ValueError("zero norm weights")
        weights_norm = {k: v / z for k, v in w_raw.items()}
    except Exception:
        logger.warning("api_optimize_pair: invalid weights, using fallback defaults.")
        weights_norm = {
            "Sharpe": 0.5,
            "Profit": 0.3,
            "Drawdown": 0.2,
        }

    # ---- 4) Seed info (for meta) ----
    global_seed = None
    try:
        global_seed = int(st.session_state.get("global_seed", 1337))
    except Exception:
        global_seed = None

    # ---- 5) Run optimisation via run_optuna_for_pair ----
    try:
        df_sorted, study_id = run_optuna_for_pair(
            sym1,
            sym2,
            ranges=ranges,
            weights=weights_norm,
            n_trials=n_trials,
            timeout_min=timeout_min,
            direction=direction,
            sampler_name=sampler_name,
            pruner_name=pruner_name,
            param_mapping=param_mapping,
            profile=profile,
            multi_objective=multi_objective,
            objective_metrics=objective_metrics,
            param_specs=get_param_specs_view(),
        )
    except Exception as e:
        logger.warning("api_optimize_pair: run_optuna_for_pair failed for %s: %s", pair_label, e)
        status = "error"
        error_message = str(e)
        df_sorted = pd.DataFrame()
        study_id = None

    # ---- 6) Duration & basic summary ----
    duration_sec = float(time.time() - t0)

    if df_sorted is not None and not df_sorted.empty and "Score" in df_sorted.columns:
        try:
            best_score = float(pd.to_numeric(df_sorted["Score"], errors="coerce").max())
        except Exception:
            best_score = None
    else:
        best_score = None

    # ---- 7) Build lightweight ranges summary for meta ----
    ranges_summary: Dict[str, Any] = {}
    try:
        for name, tpl in (ranges or {}).items():
            lo, hi, step = tpl
            ranges_summary[str(name)] = {
                "lo": float(lo),
                "hi": float(hi),
                "step": float(step) if step is not None else None,
            }
    except Exception:
        ranges_summary = {}

    meta: Dict[str, Any] = {
        "pair": pair_label,
        "study_id": study_id,
        "n_trials": int(n_trials),
        "timeout_min": int(timeout_min),
        "direction": str(direction),
        "profile": str(profile),
        "multi_objective": bool(multi_objective),
        "objective_metrics": objective_metrics or [],
        "paramspace_hash": paramspace_hash_value,
        "weights_used": weights_norm,
        "ranges_used": ranges_summary,
        "status": status,
        "error_message": error_message,
        "duration_sec": duration_sec,
        "best_score": best_score,
        "global_seed": global_seed,
    }

    # לוג קצר לסקריפטים / agents
    try:
        logger.info(
            "api_optimize_pair[%s]: status=%s, study_id=%s, best_score=%s, duration=%.1fs",
            pair_label,
            status,
            study_id,
            f"{best_score:.3f}" if best_score is not None else "n/a",
            duration_sec,
        )
    except Exception:
        pass

    return df_sorted, meta

def api_optimize_pairs_batch(
    pairs: List[Tuple[str, str]],
    *,
    ranges: Optional[Dict[str, ParamRange]] = None,
    weights: Optional[Dict[str, float]] = None,
    n_trials: int = 50,
    timeout_min: int = 10,
    direction: str = "maximize",
    sampler_name: str = "TPE",
    pruner_name: str = "median",
    profile: str = "default",
    multi_objective: bool = False,
    objective_metrics: Optional[List[str]] = None,
    param_mapping: Optional[Dict[str, str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    API Batch להרצת אופטימיזציה על מספר זוגות (ללא UI ישיר).

    pairs:
        רשימת זוגות (sym1, sym2), לדוגמה:
            [("XLY","XLP"), ("QQQ","IWM")]

    החזרה:
        df_all, meta

        df_all — DataFrame עם כל התוצאות מכל הזוגות, בפורמט דומה ל-opt_df_batch:
            - כולל עמודת "Pair"
            - אם run_optuna_for_pair/optimize_pairs_batch שומרים study_id → תהיה גם עמודת "study_id"

        meta — dict מטא-דאטה, לדוגמה:
            {
              "status": "ok" | "error",
              "error_message": str|None,
              "duration_sec": float,
              "pairs": ["XLY-XLP", "QQQ-IWM"],
              "n_pairs": int,
              "n_trials": int,
              "timeout_min": int,
              "direction": str,
              "profile": str,
              "multi_objective": bool,
              "objective_metrics": [...],
              "paramspace_hash": str|None,
              "weights_used": {...},
              "per_pair": {
                  "XLY-XLP": {
                      "pair": "XLY-XLP",
                      "study_ids": [...],
                      "best_score": float|None,
                      "best_sharpe": float|None,
                      "rows": int,
                  },
                  ...
              },
            }
    """
    t0 = time.time()
    status = "ok"
    error_message: Optional[str] = None

    # ---- 0) Empty pairs guard ----
    if not pairs:
        return pd.DataFrame(), {
            "status": "empty",
            "error_message": None,
            "duration_sec": 0.0,
            "pairs": [],
            "n_pairs": 0,
            "n_trials": int(n_trials),
            "timeout_min": int(timeout_min),
            "direction": str(direction),
            "profile": str(profile),
            "multi_objective": bool(multi_objective),
            "objective_metrics": objective_metrics or [],
            "paramspace_hash": None,
            "weights_used": {},
            "per_pair": {},
        }

    # ---- 1) Normalise labels ----
    pairs_norm: List[Tuple[str, str]] = []
    pair_labels: List[str] = []
    for a, b in pairs:
        sa = str(a).strip()
        sb = str(b).strip()
        pairs_norm.append((sa, sb))
        pair_labels.append(f"{sa}-{sb}")

    # ---- 2) Ranges: build/augment once (משותף לכל הזוגות) ----
    try:
        if ranges is None:
            ranges = get_default_param_ranges(profile=profile)
        else:
            base = get_default_param_ranges(profile=profile)
            if not ranges:
                ranges = base
            else:
                missing = [k for k in base.keys() if k not in ranges]
                for name in missing:
                    ranges[name] = base[name]
    except Exception as e:
        logger.warning("api_optimize_pairs_batch: get_default_param_ranges failed: %s", e)
        ranges = ranges or {}
        if not ranges:
            ranges = {
                "z_entry": (1.0, 3.0, 0.1),
                "z_exit": (0.1, 2.0, 0.1),
            }

    try:
        paramspace_hash_value = paramspace_hash(ranges)  # type: ignore[arg-type]
    except Exception:
        paramspace_hash_value = None

    # ---- 3) Weights: build / normalise ----
    if weights is None:
        weights = {
            "Sharpe": 0.4,
            "Profit": 0.3,
            "Drawdown": 0.2,
            "Sortino": 0.05,
            "Calmar": 0.05,
        }

    try:
        w_raw = {k: float(v) for k, v in (weights or {}).items()}
        z = sum(abs(v) for v in w_raw.values())
        if z <= 0:
            raise ValueError("zero norm weights")
        weights_norm = {k: v / z for k, v in w_raw.items()}
    except Exception:
        logger.warning("api_optimize_pairs_batch: invalid weights, using fallback defaults.")
        weights_norm = {
            "Sharpe": 0.5,
            "Profit": 0.3,
            "Drawdown": 0.2,
        }

    # ---- 4) Run batch optimisation (מבוסס על optimize_pairs_batch) ----
    try:
        df_all, manifests = optimize_pairs_batch(
            pairs_norm,
            ranges=ranges,
            weights=weights_norm,
            n_trials=n_trials,
            timeout_min=timeout_min,
            direction=direction,
            sampler_name=sampler_name,
            pruner_name=pruner_name,
            profile=profile,
            multi_objective=multi_objective,
            objective_metrics=objective_metrics,
            param_specs=get_param_specs_view(),
            notify_each=False,
        )
    except Exception as e:
        logger.warning("api_optimize_pairs_batch: optimise_pairs_batch failed: %s", e)
        status = "error"
        error_message = str(e)
        df_all = pd.DataFrame()
        manifests = {}

    duration_sec = float(time.time() - t0)

    # ---- 5) Build per-pair meta ----
    per_pair_meta: Dict[str, Dict[str, Any]] = {}
    if df_all is not None and not df_all.empty and "Pair" in df_all.columns:
        # נוודא טיפוס
        df_all = df_all.copy()
        df_all["Pair"] = df_all["Pair"].astype(str)

        for pl in sorted(set(df_all["Pair"])):
            sub = df_all[df_all["Pair"] == pl].copy()
            if sub.empty:
                per_pair_meta[pl] = {
                    "pair": pl,
                    "study_ids": [],
                    "best_score": None,
                    "best_sharpe": None,
                    "rows": 0,
                }
                continue
            # study_ids אם יש
            study_ids: List[int] = []
            if "study_id" in sub.columns:
                try:
                    study_ids = sorted(set(int(x) for x in sub["study_id"].dropna().tolist()))
                except Exception:
                    study_ids = []

            # best_score / best_sharpe
            best_score = None
            best_sharpe = None
            if "Score" in sub.columns:
                try:
                    best_score = float(pd.to_numeric(sub["Score"], errors="coerce").max())
                except Exception:
                    best_score = None
            if "Sharpe" in sub.columns:
                try:
                    best_sharpe = float(pd.to_numeric(sub["Sharpe"], errors="coerce").max())
                except Exception:
                    best_sharpe = None

            per_pair_meta[pl] = {
                "pair": pl,
                "study_ids": study_ids,
                "best_score": best_score,
                "best_sharpe": best_sharpe,
                "rows": int(len(sub)),
                # אפשר להוסיף כאן גם manifest הייעודי אם רוצים
                "manifest": manifests.get(pl),
            }
    else:
        # אין Pair בעמודות – ננסה עדיין לשמור משהו ב-per_pair_meta, אך רק עם labels
        for pl in pair_labels:
            per_pair_meta[pl] = {
                "pair": pl,
                "study_ids": [],
                "best_score": None,
                "best_sharpe": None,
                "rows": 0,
                "manifest": manifests.get(pl),
            }

    meta: Dict[str, Any] = {
        "status": status,
        "error_message": error_message,
        "duration_sec": duration_sec,
        "pairs": pair_labels,
        "n_pairs": len(pair_labels),
        "n_trials": int(n_trials),
        "timeout_min": int(timeout_min),
        "direction": str(direction),
        "profile": str(profile),
        "multi_objective": bool(multi_objective),
        "objective_metrics": objective_metrics or [],
        "paramspace_hash": paramspace_hash_value,
        "weights_used": weights_norm,
        "per_pair": per_pair_meta,
    }

    try:
        logger.info(
            "api_optimize_pairs_batch: status=%s, pairs=%d, duration=%.1fs",
            status,
            len(pair_labels),
            duration_sec,
        )
    except Exception:
        pass

    return df_all, meta



# =========================
# 15.3 API — load latest study from DuckDB
# =========================

def api_load_latest_study(pair: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    טוען את ה-study האחרון עבור pair נתון מתוך DuckDB:

        df, meta = api_load_latest_study("XLY-XLP")

    החזרה:
        df_trials, meta dict:
            {
              "study_id": int|None,
              "created_at": Timestamp|None,
              "sampler": str|None,
              "n_trials": int|None,
              "timeout_sec": int|None,
            }

    אם אין DuckDB או אין records — מחזיר df ריק ו-meta ריק/מאוד חלקי.
    """
    if duckdb is None:
        return pd.DataFrame(), {}

    sid_latest = get_latest_study_id_for_pair(pair)
    if sid_latest is None:
        return pd.DataFrame(), {}

    # ננסה להביא גם פרטים מתוך טבלת studies
    try:
        conn = get_ro_duck()
        q = """
        SELECT study_id, pair, created_at, sampler, n_trials, timeout_sec
        FROM studies
        WHERE study_id = ?
        """
        df_meta = conn.execute(q, [int(sid_latest)]).df()
        meta_raw = df_meta.iloc[0].to_dict() if not df_meta.empty else {}
    except Exception:
        meta_raw = {"study_id": sid_latest, "pair": pair}

    # טעינת trials עצמם
    df_trials = load_trials_from_duck(int(sid_latest))
    return df_trials, meta_raw


# =========================
# 15.4 API — portfolio from batch (no UI)
# =========================

def api_get_portfolio_from_batch(
    df_batch: pd.DataFrame,
    *,
    budget: float = 100_000.0,
    mode_weights: str = "Equal weights",
    max_weight: float = 0.3,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    API לבניית פורטפוליו בסיסי מתוך Batch (df_batch), ללא UI.

    df_batch חייב לכלול:
        - "Pair"
        - מומלץ: "Score", "Sharpe", "Drawdown"

    החזרה:
        df_portfolio, kpis_dict

        df_portfolio כולל:
            Pair, Score?, Sharpe?, Drawdown?, weight, dollars

        kpis_dict כולל:
            "Budget", "PortfolioScore", "PortfolioSharpe", "PortfolioDD"
    """
    if df_batch is None or df_batch.empty or "Pair" not in df_batch.columns:
        return pd.DataFrame(), {}

    pairs_available = sorted(df_batch["Pair"].unique().tolist())
    # Strategy representative per pair: ראשונה לפי Score (אם קיים)
    rep_rows = []
    for p in pairs_available:
        sub = df_batch[df_batch["Pair"] == p]
        if "Score" in sub.columns:
            sub = sub.sort_values("Score", ascending=False)
        rep_rows.append(sub.head(1))
    rep_df = pd.concat(rep_rows, ignore_index=True)

    # metric series
    scr = pd.to_numeric(rep_df.get("Score", pd.Series(index=rep_df.index)), errors="coerce").fillna(0.0)
    sh = pd.to_numeric(rep_df.get("Sharpe", pd.Series(index=rep_df.index)), errors="coerce").fillna(0.0)
    dd = pd.to_numeric(rep_df.get("Drawdown", pd.Series(index=rep_df.index)), errors="coerce").fillna(0.1)

    n_pairs = len(rep_df)
    if mode_weights.startswith("Equal"):
        w_vec = np.ones(n_pairs)
    elif mode_weights.startswith("Score"):
        w_vec = scr.to_numpy()
        if not np.any(np.abs(w_vec) > 0):
            w_vec = np.ones(n_pairs)
    elif mode_weights.startswith("Sharpe/DD"):
        w_vec = (sh / (1.0 + dd)).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
        if not np.any(np.abs(w_vec) > 0):
            w_vec = np.ones(n_pairs)
    else:  # Risk-parity
        vol_guess = sh.copy()
        vol_guess[vol_guess == 0] = 1.0
        vol = 1.0 / vol_guess.replace(0.0, np.nan).abs().fillna(1.0)
        w_vec = (1.0 / vol).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
        if not np.any(np.abs(w_vec) > 0):
            w_vec = np.ones(n_pairs)

    w_vec = np.clip(w_vec, 0.0, None)
    if not np.any(w_vec > 0):
        w_vec = np.ones(n_pairs)
    w = w_vec / (np.sum(w_vec) or 1.0)

    if max_weight < 1.0:
        w = np.minimum(w, max_weight)
        w = w / (np.sum(w) or 1.0)

    rep_df = rep_df.copy()
    rep_df["weight"] = w
    rep_df["dollars"] = rep_df["weight"] * float(budget)

    # KPIs
    try:
        if "Score" in rep_df.columns:
            port_score = float((scr * rep_df["weight"]).sum())
        else:
            port_score = float("nan")
        if "Sharpe" in rep_df.columns:
            port_sharpe = float((sh * rep_df["weight"]).sum())
        else:
            port_sharpe = float("nan")
        if "Drawdown" in rep_df.columns:
            port_dd = float((dd.abs() * np.abs(rep_df["weight"])).sum())
        else:
            port_dd = float("nan")
    except Exception:
        port_score = port_sharpe = port_dd = float("nan")

    kpis = {
        "Budget": float(budget),
        "PortfolioScore": port_score,
        "PortfolioSharpe": port_sharpe,
        "PortfolioDD": port_dd,
    }

    # נשאיר את מבנה העמודות דומה למה שהצגנו ב-UI
    cols_port = ["Pair", "weight", "dollars"] + [c for c in ("Score", "Sharpe", "Drawdown", "Profit") if c in rep_df.columns]
    return rep_df[cols_port], kpis

# =========================
# 15.5 Zoom Campaign (adaptive param ranges, HF-grade)
# =========================

def _select_elite_trials_for_zoom(
    df: pd.DataFrame,
    *,
    score_col: str = "Score",
    elite_frac: float = 0.2,
    min_rows: int = 20,
    dsr_min: Optional[float] = None,
    dsr_col: str = "DSR",
    wf_min: Optional[float] = None,
    wf_col: str = "wf_robust_penalty",
) -> pd.DataFrame:
    """
    בחירת "Elite set" של ניסויים לצורך Zoom על מרחב הפרמטרים.

    לוגיקה:
    -------
    1. Top-X% לפי Score (elite_frac).
    2. (אופציונלי) סינון לפי DSR >= dsr_min אם יש עמודת DSR.
    3. (אופציונלי) סינון לפי wf_robust_penalty >= wf_min אם יש עמודה כזו.

    אם הכל נמחק → fallback ל-top-k לפי Score בלבד.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    if score_col not in df.columns:
        logger.debug("_select_elite_trials_for_zoom: %r missing, returning full df", score_col)
        return df.copy()

    df_loc = df.copy()
    sc = pd.to_numeric(df_loc[score_col], errors="coerce")
    df_loc = df_loc.loc[sc.notna()].copy()

    if df_loc.shape[0] < max(5, min_rows):
        # מעט מדי תצפיות → נחזיר הכל, אין טעם להחמיר
        return df_loc

    elite_frac = float(max(0.01, min(elite_frac, 0.9)))
    q = sc.quantile(1.0 - elite_frac)
    elite = df_loc[sc >= q].copy()

    # סינון DSR
    if dsr_min is not None and dsr_col in elite.columns:
        dsr_vals = pd.to_numeric(elite[dsr_col], errors="coerce")
        elite = elite.loc[dsr_vals >= float(dsr_min)].copy()

    # סינון WF robustness
    if wf_min is not None and wf_col in elite.columns:
        wf_vals = pd.to_numeric(elite[wf_col], errors="coerce")
        elite = elite.loc[wf_vals >= float(wf_min)].copy()

    # אם הכל נמחק – fallback לטופ-K לפי Score
    if elite.empty:
        k = min(max(10, int(df_loc.shape[0] * elite_frac)), df_loc.shape[0])
        elite = df_loc.sort_values(score_col, ascending=False).head(k).copy()

    return elite


def _build_zoomed_ranges_from_elite(
    base_ranges: Dict[str, ParamRange],
    current_ranges: Dict[str, ParamRange],
    elite_df: pd.DataFrame,
    *,
    zoom_factor: float = 0.6,
    q_low: float = 0.15,
    q_high: float = 0.85,
    min_width_frac: float = 0.05,
) -> Dict[str, ParamRange]:
    """
    בונה טווחי פרמטרים חדשים (zoomed ranges) מתוך Elite DF.

    רעיון:
    ------
    - לומדים quantile-ranges לכל פרמטר מתוך elite_df (q_low–q_high).
    - span חדש = max(span_elite, span_current * zoom_factor, min_width_rel).
    - מרכז = מרכז ה-Elite.
    - קליפינג לטווח העל (base_ranges), כדי לא "לברוח" מהגדרות המערכת.

    zoom_factor:
        1.0  → כמעט לא מכווץ לעומת הטווח הנוכחי.
        0.5  → span חדש ≈ חצי מהטווח הנוכחי (אבל לא פחות מה-Elite).

    min_width_frac:
        רוחב מינימלי = min_width_frac * span_base.
    """
    if elite_df is None or elite_df.empty:
        return dict(current_ranges)

    df_num = elite_df.select_dtypes(include=[np.number])
    if df_num.empty:
        return dict(current_ranges)

    candidate_params = [c for c in df_num.columns if c in current_ranges]
    if not candidate_params:
        return dict(current_ranges)

    zoom_factor = float(max(0.05, min(zoom_factor, 1.0)))
    q_low = float(max(0.0, min(q_low, 0.49)))
    q_high = float(max(q_low + 0.01, min(q_high, 0.99)))
    min_width_frac = float(max(0.0, min(min_width_frac, 0.5)))

    elite_ranges = ranges_from_dataset(elite_df, params=candidate_params, q_low=q_low, q_high=q_high)
    new_ranges: Dict[str, ParamRange] = dict(current_ranges)

    for name, (cur_lo, cur_hi, cur_step) in current_ranges.items():
        base_lo, base_hi, base_step = base_ranges.get(name, (cur_lo, cur_hi, cur_step))

        base_span = float(max(base_hi - base_lo, 1e-9))
        cur_span = float(max(cur_hi - cur_lo, 1e-9))

        if name not in elite_ranges:
            # אין מידע חדש → נשאיר את הטווח הנוכחי
            new_ranges[name] = (float(cur_lo), float(cur_hi), cur_step)
            continue

        elite_lo, elite_hi, _ = elite_ranges[name]
        elite_span = float(max(elite_hi - elite_lo, 1e-9))
        center = 0.5 * (elite_lo + elite_hi)

        # span חכם: לוקח בחשבון גם את הטווח הנוכחי וגם את אליטת התוצאות
        span_from_current = cur_span * zoom_factor
        span_candidate = max(elite_span, span_from_current)

        # מינימום רוחב יחסי לטווח הבסיסי
        min_width = base_span * min_width_frac if min_width_frac > 0 else 0.0
        span_candidate = max(span_candidate, min_width)

        lo_new = center - 0.5 * span_candidate
        hi_new = center + 0.5 * span_candidate

        # קליפינג לטווח העל (הגדרות המערכת)
        lo_new = max(float(lo_new), float(base_lo))
        hi_new = min(float(hi_new), float(base_hi))
        if hi_new <= lo_new:
            hi_new = lo_new + 1e-9

        step_new = cur_step if cur_step is not None else base_step
        new_ranges[name] = (float(lo_new), float(hi_new), step_new)

    return new_ranges


def api_zoom_campaign_for_pair(
    sym1: str,
    sym2: str,
    *,
    base_ranges: Optional[Dict[str, ParamRange]] = None,
    weights: Optional[Dict[str, float]] = None,
    n_stages: int = 3,
    n_trials_per_stage: int = 50,
    timeout_min: int = 10,
    direction: str = "maximize",
    sampler_name: str = "TPE",
    pruner_name: str = "median",
    profile: str = "default",
    multi_objective: bool = False,
    objective_metrics: Optional[List[str]] = None,
    cleanup_strategy: str = "keep_last",  # "keep_last" / "keep_all"
    elite_frac: float = 0.2,
    dsr_min: Optional[float] = None,
    wf_min: Optional[float] = None,
    min_range_width_frac: float = 0.05,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Zoom Campaign API — אופטימיזציה רב-שלבית ברמת קרן גידור.

    בכל Stage:
    ----------
    1. מריץ run_optuna_for_pair על טווחי הפרמטרים הנוכחיים (current_ranges).
    2. בוחר Elite set:
       - Top Score (elite_frac).
       - אופציונלי: DSR >= dsr_min, WF >= wf_min.
    3. בונה טווחים חדשים (zoomed ranges) סביב אזורי הביצועים הטובים.
    4. מחליט האם לעצור מוקדם (early stop) לפי שיפור ב-Score/העדר Elite.

    cleanup_strategy:
        "keep_last" — מוחק את ה-studies הקודמים ב-DuckDB ומשאיר רק את האחרון.
        "keep_all"  — לא מוחק כלום.

    החזרה:
    -------
    df_final: DataFrame של השלב האחרון (או מה-DuckDB אם קיים study_id).
    meta:     dict מטא-דאטה על כל הקמפיין.
    """
    t0 = time.time()
    status = "ok"
    error_message: Optional[str] = None

    sym1 = str(sym1).strip()
    sym2 = str(sym2).strip()
    pair_label = f"{sym1}-{sym2}"

    # 🔹 כאן אנחנו מגדירים את ה-storage הרשמי של ה-Zoom
    zoom_cfg = resolve_zoom_storage(PROJECT_ROOT)

    n_stages = max(1, int(n_stages))
    n_trials_per_stage = max(1, int(n_trials_per_stage))
    timeout_min = max(1, int(timeout_min))

    # --- בסיס טווחים ---
    try:
        base_ranges_full = base_ranges or get_default_param_ranges(profile=profile)
    except Exception as e:
        logger.warning("api_zoom_campaign_for_pair: get_default_param_ranges failed for %s: %s", pair_label, e)
        base_ranges_full = base_ranges or {
            "z_entry": (1.0, 3.0, 0.1),
            "z_exit": (0.1, 2.0, 0.1),
        }

    current_ranges: Dict[str, ParamRange] = dict(base_ranges_full)

    # --- משקולות ---
    if weights is None:
        weights = {"Sharpe": 0.4, "Profit": 0.3, "Drawdown": 0.2}
    try:
        w_raw = {k: float(v) for k, v in (weights or {}).items()}
        z = sum(abs(v) for v in w_raw.values()) or 1.0
        eff_weights = {k: v / z for k, v in w_raw.items()}
    except Exception:
        logger.warning("api_zoom_campaign_for_pair: invalid weights, using fallback.")
        eff_weights = {"Sharpe": 0.5, "Profit": 0.3, "Drawdown": 0.2}

    # --- zoom factor schedule ---
    if n_stages == 1:
        zoom_schedule = [0.7]
    else:
        start_z, end_z = 0.8, 0.5
        zoom_schedule = [
            float(start_z + (end_z - start_z) * s / max(1, n_stages - 1))
            for s in range(n_stages)
        ]

    # --- hash התחלתי ---
    try:
        paramspace_hash_initial = paramspace_hash(current_ranges)
    except Exception:
        paramspace_hash_initial = None

    stages_meta: List[Dict[str, Any]] = []
    best_score_overall: Optional[float] = None
    prev_best_score: Optional[float] = None
    study_ids_history: List[int] = []

    df_last_stage = pd.DataFrame()

    # =========================
    # לולאת Zoom stages
    # =========================
    for stage_idx in range(n_stages):
        zoom_factor = float(zoom_schedule[stage_idx])
        logger.info(
            "ZoomCampaign[%s] stage %d/%d — zoom_factor=%.3f, n_trials=%d, timeout_min=%d",
            pair_label,
            stage_idx,
            n_stages - 1,
            zoom_factor,
            n_trials_per_stage,
            timeout_min,
        )

        # 🔹 שם study קבוע לזוג+סטייג' (zoom::<PAIR>::stageX)
        study_name = build_zoom_study_name(
            pair_label,
            cfg=zoom_cfg,
            stage=stage_idx,
        )

        logger.debug(
            "ZoomCampaign[%s] stage %d: study_name=%s | storage_url=%s",
            pair_label,
            stage_idx,
            study_name,
            zoom_cfg.storage_url,
        )

        try:
            df_stage, study_id = run_optuna_for_pair(
                sym1,
                sym2,
                ranges=current_ranges,
                weights=eff_weights,
                n_trials=n_trials_per_stage,
                timeout_min=timeout_min,
                direction=direction,
                sampler_name=sampler_name,
                pruner_name=pruner_name,
                param_mapping=None,
                profile=profile,
                multi_objective=multi_objective,
                objective_metrics=objective_metrics,
                param_specs=get_param_specs_view(),
                # 🔽 אלה הקריטיים ל-Zoom storage:
                study_name=study_name,
                storage_url=zoom_cfg.storage_url,
                zoom_meta={
                    "pair": pair_label,
                    "zoom_stage": stage_idx,
                    "profile": profile,
                },
            )
        except Exception as e:
            logger.warning("ZoomCampaign[%s] stage %d failed: %s", pair_label, stage_idx, e)
            status = "error"
            error_message = str(e)
            df_stage = pd.DataFrame()
            study_id = None

        df_last_stage = df_stage

        if study_id is not None:
            study_ids_history.append(int(study_id))

        report = validate_opt_df(df_stage)
        elite_df = _select_elite_trials_for_zoom(
            df_stage,
            score_col="Score",
            elite_frac=elite_frac,
            min_rows=max(10, n_trials_per_stage // 5),
            dsr_min=dsr_min,
            dsr_col="DSR",
            wf_min=wf_min,
            wf_col="wf_robust_penalty",
        )

        # best_score בשלב
        if "Score" in df_stage.columns and not df_stage.empty:
            try:
                best_score_stage = float(pd.to_numeric(df_stage["Score"], errors="coerce").max())
            except Exception:
                best_score_stage = None
        else:
            best_score_stage = None

        if best_score_stage is not None:
            if best_score_overall is None or best_score_stage > best_score_overall:
                best_score_overall = best_score_stage

        stages_meta.append(
            {
                "stage": stage_idx,
                "study_id": study_id,
                "study_name": study_name,              # 🔽 מוסיפים למטא
                "storage_url": zoom_cfg.storage_url,   # 🔽 מוסיפים למטא
                "zoom_factor": zoom_factor,
                "ranges": {
                    k: {"lo": float(v[0]), "hi": float(v[1]), "step": v[2]}
                    for k, v in current_ranges.items()
                },
                "validation": report,
                "elite_rows": int(elite_df.shape[0]) if elite_df is not None else 0,
                "best_score": best_score_stage,
            }
        )

        # --- עצירה 1: אין Elite ---
        if elite_df is None or elite_df.empty:
            logger.info("ZoomCampaign[%s] stage %d: no elite set -> stopping.", pair_label, stage_idx)
            if status == "ok":
                status = "no_elite"
            break

        # --- עצירה 2: אין שיפור משמעותי ב-Score ---
        if prev_best_score is not None and best_score_stage is not None and stage_idx >= 1:
            if (best_score_stage - prev_best_score) < 1e-4:
                logger.info(
                    "ZoomCampaign[%s] stage %d: best_score_stage≈prev -> early stop.",
                    pair_label,
                    stage_idx,
                )
                prev_best_score = best_score_stage
                break

        prev_best_score = best_score_stage

        # --- שלב אחרון – אין צורך לעדכן טווחים ---
        if stage_idx == n_stages - 1:
            break

        # --- עדכון טווחים לפי Elite ---
        current_ranges = _build_zoomed_ranges_from_elite(
            base_ranges_full,
            current_ranges,
            elite_df,
            zoom_factor=zoom_factor,
            q_low=0.15,
            q_high=0.85,
            min_width_frac=min_range_width_frac,
        )

        # --- Cleanup ב-DuckDB לפי strategy ---
        if cleanup_strategy == "keep_last" and duckdb is not None:
            if len(study_ids_history) > 1:
                to_delete = study_ids_history[:-1]
                for sid in to_delete:
                    try:
                        delete_study_from_duck(int(sid))
                    except Exception as e:
                        logger.debug(
                            "ZoomCampaign[%s] delete_study_from_duck(%s) failed: %s",
                            pair_label,
                            sid,
                            e,
                        )
                study_ids_history = study_ids_history[-1:]

    # =========================
    # df סופי
    # =========================
    df_final = df_last_stage
    if duckdb is not None and study_ids_history:
        try:
            df_final = load_trials_from_duck(int(study_ids_history[-1]))
        except Exception:
            pass

    # hash סופי
    try:
        paramspace_hash_final = paramspace_hash(current_ranges)
    except Exception:
        paramspace_hash_final = None

    duration_sec = float(time.time() - t0)

    meta: Dict[str, Any] = {
        "pair": pair_label,
        "stages": stages_meta,
        "final_stage": stages_meta[-1]["stage"] if stages_meta else None,
        "final_best_score": best_score_overall,
        "status": status,
        "error_message": error_message,
        "duration_sec": duration_sec,
        "paramspace_hash_initial": paramspace_hash_initial,
        "paramspace_hash_final": paramspace_hash_final,
    }

    try:
        logger.info(
            "ZoomCampaign[%s]: status=%s, stages=%d, final_best_score=%s, duration=%.1fs",
            pair_label,
            status,
            len(stages_meta),
            f"{best_score_overall:.4f}" if best_score_overall is not None else "n/a",
            duration_sec,
        )
    except Exception:
        pass

    return df_final, meta

# =========================
# CLI entrypoint (HF-grade)
# =========================

def _build_cli_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m root.optimization_tab",
        description="HF-grade optimisation / zoom CLI (reproducible, auditable).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)

    # -----------------------
    # Common CLI arguments
    # -----------------------
    def _add_common_pair_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--pair", type=str, required=True, help="Pair label: SYM1-SYM2 (e.g. XLP-XLY).")
        sp.add_argument("--profile", type=str, default="default", help="Profile: default/defensive/aggressive.")
        sp.add_argument("--seed", type=int, default=1337, help="Global seed (reproducibility).")
        sp.add_argument("--sampler", type=str, default="TPE", help="Sampler: TPE/CMAES (and more if you add).")
        sp.add_argument("--pruner", type=str, default="median", help="Pruner: median/none.")
        sp.add_argument("--direction", type=str, default="maximize", choices=["maximize", "minimize"], help="Score direction.")
        sp.add_argument("--storage", type=str, default="", help="Optuna storage URL override. Example: sqlite:///data/zoom_studies.db")
        sp.add_argument("--study-prefix", type=str, default="zoom", help="Prefix for study naming (used if study-name not provided).")
        sp.add_argument("--study-name", type=str, default="", help="Explicit Optuna study_name override.")
        sp.add_argument("--export-csv", type=str, default="", help="Export final df to CSV path.")
        sp.add_argument("--export-json", type=str, default="", help="Export metadata/manifest to JSON path.")
        sp.add_argument("--print-best", action="store_true", help="Print best trial summary to stdout.")

    # -----------------------
    # zoom-campaign
    # -----------------------
    z = sub.add_parser("zoom-campaign", help="Multi-stage zoom campaign for one pair.")
    _add_common_pair_args(z)
    z.add_argument("--n-trials", type=int, default=80, help="Trials per stage.")
    z.add_argument("--timeout-sec", type=int, default=900, help="Timeout per stage (seconds).")
    z.add_argument("--zoom-stages", type=int, default=3, help="Number of stages.")
    z.add_argument("--cleanup", type=str, default="keep_last", choices=["keep_last", "keep_all"], help="DuckDB cleanup strategy (if used).")

    # Robustness knobs
    z.add_argument("--elite-frac", type=float, default=0.2, help="Elite fraction used to tighten ranges each stage.")
    z.add_argument("--min-range-width-frac", type=float, default=0.05, help="Min relative width vs base ranges (avoid over-shrinking).")
    z.add_argument("--dsr-min", type=float, default=None, help="Optional DSR threshold for elite filtering.")
    z.add_argument("--wf-min", type=float, default=None, help="Optional WF robustness threshold for elite filtering.")

    # Repeats (statistical reliability)
    z.add_argument("--repeats", type=int, default=1, help="Repeat the entire campaign N times with different seeds (seed+i).")

    # -----------------------
    # single (alias)
    # -----------------------
    s = sub.add_parser("single", help="Alias for zoom-campaign (default 1 stage).")
    _add_common_pair_args(s)
    s.add_argument("--n-trials", type=int, default=80, help="Trials for the single stage.")
    s.add_argument("--timeout-sec", type=int, default=900, help="Timeout (seconds).")
    s.add_argument("--zoom-stages", type=int, default=1, help="Number of stages (alias default=1).")
    s.add_argument("--elite-frac", type=float, default=0.2, help="Elite fraction (kept for compatibility).")
    s.add_argument("--min-range-width-frac", type=float, default=0.05, help="Min relative width vs base ranges.")
    s.add_argument("--dsr-min", type=float, default=None, help="Optional DSR threshold.")
    s.add_argument("--wf-min", type=float, default=None, help="Optional WF threshold.")
    s.add_argument("--repeats", type=int, default=1, help="Repeat N times with different seeds (seed+i).")

    # -----------------------
    # run-pair (one-shot)
    # -----------------------
    rp = sub.add_parser("run-pair", help="Run a single optimisation (no zoom logic) and persist.")
    _add_common_pair_args(rp)
    rp.add_argument("--n-trials", type=int, default=200, help="Trials.")
    rp.add_argument("--timeout-sec", type=int, default=1200, help="Timeout (seconds).")

    # -----------------------
    # best (inspect best trial from Optuna storage)
    # -----------------------
    b = sub.add_parser("best", help="Show best trial for a pair study from Optuna storage (sqlite).")
    b.add_argument("--pair", type=str, required=True, help="Pair label: SYM1-SYM2.")
    b.add_argument("--storage", type=str, default="", help="Optuna storage URL. If empty -> resolve_zoom_storage().")
    b.add_argument("--study-name", type=str, default="", help="Study name override. If empty -> zoom::<PAIR>::stage0")
    b.add_argument("--top", type=int, default=20, help="Show top-N trials by score.")
    b.add_argument("--order-by", type=str, default="score", help="Ordering column: score / value / etc.")

    # -----------------------
    # export (export study trials from Optuna sqlite)
    # -----------------------
    ex = sub.add_parser("export", help="Export trials from Optuna study to CSV.")
    ex.add_argument("--pair", type=str, required=True, help="Pair label: SYM1-SYM2.")
    ex.add_argument("--storage", type=str, default="", help="Optuna storage URL. If empty -> resolve_zoom_storage().")
    ex.add_argument("--study-name", type=str, default="", help="Study name override. If empty -> zoom::<PAIR>::stage0")
    ex.add_argument("--out", type=str, required=True, help="Output CSV path.")
    ex.add_argument("--top", type=int, default=0, help="If >0, export top-N only by score.")

    # -----------------------
    # validate (run robustness validation on best trial)
    # -----------------------
    v = sub.add_parser("validate", help="Validate best params on real data (replay-style) for a given study.")
    v.add_argument("--pair", type=str, required=True, help="Pair label: SYM1-SYM2.")
    v.add_argument("--storage", type=str, default="", help="Optuna storage URL. If empty -> resolve_zoom_storage().")
    v.add_argument("--study-name", type=str, default="", help="Study name override. If empty -> zoom::<PAIR>::stage0")
    v.add_argument("--top", type=int, default=1, help="Validate top-N best trials (default=1).")
    v.add_argument("--start", type=str, default="", help="Optional start date YYYY-MM-DD.")
    v.add_argument("--end", type=str, default="", help="Optional end date YYYY-MM-DD.")
    v.add_argument("--data-source", type=str, default="SQL", help="Backtester data_source (SQL recommended).")

    # -----------------------
    # clear-study (delete Optuna study from storage)
    # -----------------------
    cs = sub.add_parser("clear-study", help="Delete an Optuna study from storage (dangerous).")
    cs.add_argument("--storage", type=str, required=True, help="Optuna storage URL.")
    cs.add_argument("--study-name", type=str, required=True, help="Study name to delete.")
    cs.add_argument("--yes", action="store_true", help="Confirm deletion.")

    return p


def _resolve_optuna_storage_for_cli(storage_override: str) -> str:
    if storage_override:
        return str(storage_override)
    try:
        zoom_cfg = resolve_zoom_storage(PROJECT_ROOT)
        if getattr(zoom_cfg, "storage_url", None):
            return str(zoom_cfg.storage_url)
    except Exception:
        pass
    return ""


def _default_zoom_study_name(pair_label: str) -> str:
    # Consistent with your earlier usage in logs
    return f"zoom::{pair_label}::stage0"


def _write_text_file(path: str, text: str) -> None:
    if not path:
        return
    pth = Path(path)
    pth.parent.mkdir(parents=True, exist_ok=True)
    pth.write_text(text, encoding="utf-8")


def _write_csv_file(path: str, df: "pd.DataFrame") -> None:
    if not path:
        return
    pth = Path(path)
    pth.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(pth, index=False)


def _print_best_from_df(df: "pd.DataFrame") -> None:
    if df is None or df.empty:
        print("No rows.")
        return
    if "score" in df.columns:
        best_row = df.sort_values("score", ascending=False).iloc[0]
    elif "Score" in df.columns:
        best_row = df.sort_values("Score", ascending=False).iloc[0]
    else:
        best_row = df.iloc[0]
    cols = [c for c in ("trial_no", "score", "Score", "Sharpe", "Profit", "Drawdown") if c in df.columns]
    print("BEST ROW (preview):")
    for c in cols:
        print(f"  {c}: {best_row.get(c)}")


def _main_cli(argv: Optional[List[str]] = None) -> None:
    parser = _build_cli_parser()
    args = parser.parse_args(argv)

    # -----------------------
    # Helper: parse pair
    # -----------------------
    def _split_pair(pair_str: str) -> Tuple[str, str, str]:
        pair_str = str(pair_str).strip().upper()
        if "-" not in pair_str:
            raise SystemExit(f"Invalid --pair format: {pair_str!r}, expected SYM1-SYM2")
        a, b = [p.strip() for p in pair_str.split("-", 1)]
        return a, b, f"{a}-{b}"

    # -----------------------
    # Commands
    # -----------------------
    if args.command in {"zoom-campaign", "single"}:
        sym1, sym2, pair_label = _split_pair(args.pair)

        # Make CLI deterministic even without Streamlit
        try:
            st.session_state["global_seed"] = int(args.seed)  # type: ignore[attr-defined]
        except Exception:
            pass

        storage_url = _resolve_optuna_storage_for_cli(getattr(args, "storage", ""))

        # stage settings
        n_trials = int(getattr(args, "n_trials"))
        timeout_sec = int(getattr(args, "timeout_sec"))
        timeout_min = max(1, int(timeout_sec // 60)) if timeout_sec > 0 else 10
        n_stages = int(getattr(args, "zoom_stages"))
        profile = str(getattr(args, "profile"))

        elite_frac = float(getattr(args, "elite_frac", 0.2))
        dsr_min = getattr(args, "dsr_min", None)
        wf_min = getattr(args, "wf_min", None)
        min_w = float(getattr(args, "min_range_width_frac", 0.05))
        cleanup = str(getattr(args, "cleanup", "keep_last"))

        repeats = int(max(1, getattr(args, "repeats", 1)))

        last_df = None
        meta_all: Dict[str, Any] = {"pair": pair_label, "repeats": repeats, "runs": []}

        for i in range(repeats):
            seed_i = int(args.seed) + int(i)
            try:
                st.session_state["global_seed"] = seed_i  # type: ignore[attr-defined]
            except Exception:
                pass

            logger.info(
                "CLI %s run %d/%d for %s | sampler=%s | trials/stage=%d | timeout=%ds | stages=%d | profile=%s | seed=%d",
                args.command,
                i + 1,
                repeats,
                pair_label,
                args.sampler,
                n_trials,
                timeout_sec,
                n_stages,
                profile,
                seed_i,
            )

            # Study name strategy:
            # - if user provided explicit study-name => use it
            # - else: use zoom storage naming convention
            study_name_override = str(getattr(args, "study_name", "") or "").strip()
            if study_name_override:
                study_name = study_name_override
            else:
                study_name = _default_zoom_study_name(pair_label)

            df_final, meta = api_zoom_campaign_for_pair(
                sym1,
                sym2,
                base_ranges=None,
                weights=None,
                n_stages=n_stages,
                n_trials_per_stage=n_trials,
                timeout_min=timeout_min,
                direction=str(args.direction),
                sampler_name=str(args.sampler),
                pruner_name=str(args.pruner),
                profile=profile,
                multi_objective=False,
                objective_metrics=None,
                cleanup_strategy=cleanup,
                elite_frac=elite_frac,
                dsr_min=dsr_min,
                wf_min=wf_min,
                min_range_width_frac=min_w,
            )

            # Attach storage/study info for auditing (even if API already stores it)
            meta_run = {
                "seed": seed_i,
                "storage_url": storage_url,
                "study_name": study_name,
                "meta": meta,
            }
            meta_all["runs"].append(meta_run)
            last_df = df_final

        # Print summary
        status = meta_all["runs"][-1]["meta"].get("status") if meta_all.get("runs") else None
        best = meta_all["runs"][-1]["meta"].get("final_best_score") if meta_all.get("runs") else None

        print("=== Zoom Campaign Summary ===")
        print(f"Pair:        {pair_label}")
        print(f"Status:      {status}")
        print(f"Best score:  {best}")
        print(f"Repeats:     {repeats}")
        if storage_url:
            print(f"Storage URL: {storage_url}")

        # Optional prints/exports
        if getattr(args, "print_best", False) and last_df is not None:
            try:
                _print_best_from_df(last_df)
            except Exception:
                pass

        if getattr(args, "export_csv", "") and last_df is not None:
            try:
                _write_csv_file(str(args.export_csv), last_df)
                print(f"Saved CSV: {args.export_csv}")
            except Exception as e:
                print(f"Failed saving CSV: {e}")

        if getattr(args, "export_json", ""):
            try:
                _write_text_file(str(args.export_json), json.dumps(make_json_safe(meta_all), ensure_ascii=False, indent=2))
                print(f"Saved JSON: {args.export_json}")
            except Exception as e:
                print(f"Failed saving JSON: {e}")

        return

    # -----------------------
    # run-pair (no zoom logic)
    # -----------------------
    if args.command == "run-pair":
        sym1, sym2, pair_label = _split_pair(args.pair)
        try:
            st.session_state["global_seed"] = int(args.seed)  # type: ignore[attr-defined]
        except Exception:
            pass

        timeout_sec = int(getattr(args, "timeout_sec"))
        timeout_min = max(1, int(timeout_sec // 60)) if timeout_sec > 0 else 10

        df_sorted, meta = api_optimize_pair(
            sym1,
            sym2,
            n_trials=int(getattr(args, "n_trials")),
            timeout_min=int(timeout_min),
            direction=str(getattr(args, "direction")),
            sampler_name=str(getattr(args, "sampler")),
            pruner_name=str(getattr(args, "pruner")),
            profile=str(getattr(args, "profile")),
            multi_objective=False,
            objective_metrics=None,
            param_mapping=None,
        )

        print("=== Run Pair Summary ===")
        print(f"Pair:   {pair_label}")
        print(f"Rows:   {len(df_sorted) if df_sorted is not None else 0}")
        print(f"Meta:   {meta}")

        if getattr(args, "export_csv", "") and df_sorted is not None:
            _write_csv_file(str(args.export_csv), df_sorted)
            print(f"Saved CSV: {args.export_csv}")

        if getattr(args, "export_json", ""):
            _write_text_file(str(args.export_json), json.dumps(make_json_safe(meta), ensure_ascii=False, indent=2))
            print(f"Saved JSON: {args.export_json}")

        return

    # -----------------------
    # best / export / validate / clear-study
    # -----------------------
    if args.command in {"best", "export", "validate", "clear-study"}:
        # All these require optuna
        if optuna is None:
            raise SystemExit("Optuna is not installed. Install it: pip install optuna")

        if args.command == "clear-study":
            if not args.yes:
                raise SystemExit("Refusing to delete study without --yes")
            storage_url = str(args.storage)
            study_name = str(args.study_name)
            optuna.delete_study(study_name=study_name, storage=storage_url)
            print(f"Deleted study: {study_name} from {storage_url}")
            return

        pair = str(getattr(args, "pair")).strip().upper()
        storage_url = _resolve_optuna_storage_for_cli(getattr(args, "storage", ""))
        study_name = str(getattr(args, "study_name", "")).strip() or _default_zoom_study_name(pair)

        if not storage_url:
            raise SystemExit("No storage URL available (pass --storage or configure zoom_storage).")

        study = optuna.load_study(study_name=study_name, storage=storage_url)

        # Flatten trials to a dataframe-like list of dicts
        rows = []
        for t in study.trials:
            ua = dict(getattr(t, "user_attrs", {}) or {})
            perf = ua.get("perf", {}) or {}
            params = ua.get("params", {}) or {}
            row = {"trial_no": int(t.number), "score": float(ua.get("score_single", t.value if t.value is not None else float("nan")))}
            for k, v in params.items():
                row[str(k)] = v
            for k, v in perf.items():
                row[str(k)] = v
            rows.append(row)

        df_view = pd.DataFrame(rows) if rows else pd.DataFrame()

        if args.command == "best":
            top = int(getattr(args, "top", 20))
            if not df_view.empty and "score" in df_view.columns:
                df_view = df_view.sort_values("score", ascending=False).head(top)
            print(f"Storage : {storage_url}")
            print(f"Study   : {study_name}")
            print("")
            if df_view.empty:
                print("No trials found.")
            else:
                _print_best_from_df(df_view)
                print("")
                print(df_view.head(top).to_string(index=False))
            return

        if args.command == "export":
            out = str(getattr(args, "out"))
            top = int(getattr(args, "top", 0))
            if top > 0 and not df_view.empty and "score" in df_view.columns:
                df_out = df_view.sort_values("score", ascending=False).head(top)
            else:
                df_out = df_view
            _write_csv_file(out, df_out)
            print(f"Exported {len(df_out)} rows to: {out}")
            return

        if args.command == "validate":
            if Backtester is None:
                raise SystemExit("Backtester is not available; cannot validate.")
            topn = int(getattr(args, "top", 1))
            df_rank = df_view.sort_values("score", ascending=False).head(topn) if (not df_view.empty and "score" in df_view.columns) else df_view.head(topn)

            # Optional dates
            sd = str(getattr(args, "start", "")).strip()
            ed = str(getattr(args, "end", "")).strip()
            start = pd.to_datetime(sd).date() if sd else None
            end = pd.to_datetime(ed).date() if ed else None

            data_source = str(getattr(args, "data_source", "SQL"))

            print("=== Validation ===")
            print(f"Pair: {pair}")
            print(f"Study: {study_name}")
            print(f"Top-N: {topn}")
            print(f"Data source: {data_source}")
            print("")

            for _, row in df_rank.iterrows():
                params = {k: row[k] for k in df_rank.columns if k not in {"trial_no", "score"} and k not in {"Sharpe", "Profit", "Drawdown"}}
                sym1, sym2 = pair.split("-", 1)

                try:
                    bt = Backtester(
                        symbol_a=sym1,
                        symbol_b=sym2,
                        start=start,
                        end=end,
                        data_source=data_source,
                        **params,
                    )
                    perf = bt.run()
                except Exception as e:
                    print(f"Trial {row.get('trial_no')} validate failed: {e}")
                    continue

                print(f"Trial {row.get('trial_no')} | score={row.get('score')}")
                if isinstance(perf, dict):
                    for k in sorted(perf.keys()):
                        print(f"  {k}: {perf[k]}")
                else:
                    print(perf)
                print("")

            return

    parser.print_help()


if __name__ == "__main__":
    _main_cli()
