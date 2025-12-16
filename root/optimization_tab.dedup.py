from __future__ import annotations
"""
root/optimization_tab.py — Pairs Optimiser (V10 skeleton)
Minimal, clean, Pydantic v2–compatible skeleton we will extend incrementally.
"""

# =========================
# SECTION 0: Imports
# =========================
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import os
import json
import logging
import atexit, time

try:
    import duckdb  # type: ignore[reportMissingImports]
except Exception:
    duckdb = None  # type: ignore
import numpy as np
import pandas as pd
import streamlit as st
import warnings

# Streamlit caching shims (short & typed)
if hasattr(st, "cache_data"):
    cache_data = st.cache_data  # type: ignore[attr-defined]
else:
    def cache_data(*_a, **_k):  # fallback no-op
        def deco(fn):
            return fn
        return deco

if hasattr(st, "cache_resource"):
    cache_resource = st.cache_resource  # type: ignore[attr-defined]
else:
    def cache_resource(*_a, **_k):  # fallback no-op
        def deco(fn):
            return fn
        return deco
from typing import Callable, Mapping, Set, Iterable, Literal, Annotated
from functools import lru_cache
from datetime import datetime, timezone
import math

# Scientific/Stats (optional)
try:
    import scipy.stats as sps  # type: ignore
except Exception:
    sps = None  # type: ignore

try:
    import statsmodels.api as sm  # type: ignore
    from statsmodels.tsa.stattools import adfuller as sm_adfuller  # type: ignore
except Exception:
    sm = None  # type: ignore
    def sm_adfuller(*_a, **_k):
        raise RuntimeError("statsmodels not available")

# Timezone helpers (optional)
try:
    import pytz  # type: ignore
except Exception:
    pytz = None  # type: ignore

try:
    from tzlocal import get_localzone  # type: ignore
except Exception:
    def get_localzone():
        return timezone.utc

# Extra Plotly API (optional)
try:
    import plotly.graph_objects as go  # type: ignore
except Exception:
    go = None  # type: ignore

# Quiet down noisy warnings from optional libs
try:
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
except Exception:
    pass
warnings.filterwarnings("ignore", message=r".*is deprecated and will be removed.*")

# Streamlit unique key helper (prevents DuplicateElementKey)
import itertools
_st_key_counter = itertools.count()

def sk(prefix: str) -> str:
    return f"{prefix}-{next(_st_key_counter)}"

# Optional visuals & ML (used later in the file; safe to miss at runtime)
try:
    import plotly.express as px  # type: ignore
except Exception:
    px = None  # type: ignore

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

# Introspection for Backtester kwargs sanitation (used later)
import inspect

# Pydantic v2
from pydantic import BaseModel, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# (Optional) optuna will be wired later
try:
    import optuna
    from optuna.samplers import TPESampler, CmaEsSampler
    from optuna.pruners import MedianPruner
except Exception:  # keep the file importable even without optuna
    optuna = None
    TPESampler = CmaEsSampler = MedianPruner = object  # type: ignore

# =========================
# SECTION 0.1: Internal Modules (lazy / safe imports)
# =========================
# Core backtest & scoring
try:
    from core.optimization_backtester import Backtester
except Exception:  # keep UI working even if core isn't ready
    Backtester = None  # type: ignore

try:
    from core.metrics import normalize_metrics, compute_weighted_score
except Exception:
    normalize_metrics = None  # type: ignore
    compute_weighted_score = None  # type: ignore

try:
    from core.distributions import get_param_distributions
except Exception:
    get_param_distributions = None  # type: ignore

try:
    from core.risk_parity import apply_risk_parity
except Exception:
    apply_risk_parity = None  # type: ignore

# Analysis / Viz helpers
try:
    from core.analysis_helpers import (
        compute_shap_importance_df,
        compute_pca_transform,
        compute_clusters,
    )
except Exception:
    compute_shap_importance_df = None  # type: ignore
    compute_pca_transform = None  # type: ignore
    compute_clusters = None  # type: ignore

try:
    from core.ml_analysis import render_ml_analysis
except Exception:
    render_ml_analysis = None  # type: ignore

# AutoML & Meta-Optimization
try:
    from common.automl_tools import run_automl_summary
except Exception:
    run_automl_summary = None  # type: ignore

try:
    from core.meta_optimization import meta_optimization_sampling
except Exception:
    meta_optimization_sampling = None  # type: ignore

# =========================
# SECTION 0.2: Additional Internal Modules (optional, used when available)
# =========================
# common/* helpers
try:
    from common.data_loader import load_prices as common_load_prices  # type: ignore
except Exception:
    common_load_prices = None  # type: ignore

try:
    from common.feature_engineering import build_features  # type: ignore
except Exception:
    build_features = None  # type: ignore

try:
    from common.advanced_metrics import compute_advanced_metrics  # type: ignore
except Exception:
    compute_advanced_metrics = None  # type: ignore

try:
    from common.signal_generator import generate_signal_candidates  # type: ignore
except Exception:
    generate_signal_candidates = None  # type: ignore

try:
    from common.stat_tests import adf_test  # type: ignore
except Exception:
    adf_test = None  # type: ignore

try:
    from common.helpers import to_timeframe  # type: ignore
except Exception:
    to_timeframe = None  # type: ignore

try:
    from common.matrix_helpers import winsorize  # type: ignore
except Exception:
    winsorize = None  # type: ignore

# core/* optional analytics/selection/recommendation
try:
    from core.feature_selection import select_features  # type: ignore
except Exception:
    select_features = None  # type: ignore

try:
    from core.clustering import cluster_pairs  # type: ignore
except Exception:
    cluster_pairs = None  # type: ignore

try:
    from core.analytics import summarize_results  # type: ignore
except Exception:
    summarize_results = None  # type: ignore

try:
    from core.pair_recommender import recommend_pairs  # type: ignore
except Exception:
    recommend_pairs = None  # type: ignore

try:
    from core.params import PARAM_SPECS as CORE_PARAM_SPECS  # type: ignore
except Exception:
    CORE_PARAM_SPECS = None  # type: ignore

try:
    from core.ranges import get_default_ranges as core_get_default_ranges  # type: ignore
except Exception:
    core_get_default_ranges = None  # type: ignore


# =========================
# SECTION 1: Constants & Paths
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
STUDIES_DIR = PROJECT_ROOT / "studies"
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# =========================
# SECTION 2: Settings (Pydantic v2)
# =========================
class OptSettings(BaseSettings):
    env: str = "local"
    log_level: str = "INFO"
    data_dir: Path = DATA_DIR
    studies_dir: Path = STUDIES_DIR

    slack_webhook: Optional[str] = None
    telegram_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None

    model_config = SettingsConfigDict(env_prefix="OPT_", case_sensitive=False)

    @field_validator("log_level")
    @classmethod
    def valid_level(cls, v: str) -> str:
        lvl = (v or "INFO").upper()
        if lvl not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError("Invalid log_level")
        return lvl


def load_settings(cfg_path: Optional[Path] = None) -> OptSettings:
    """Load YAML/JSON config if provided; otherwise environment defaults."""
    cfg_path = cfg_path or (PROJECT_ROOT / "opt_config.yml")
    data: Dict[str, Any] = {}
    if cfg_path.exists():
        if cfg_path.suffix in {".yml", ".yaml"}:
            try:
                import yaml  # deferred import
                data = yaml.safe_load(cfg_path.read_text()) or {}
            except Exception:
                data = {}
        else:
            data = json.loads(cfg_path.read_text())
    return OptSettings(**data)

SETTINGS = load_settings()

# =========================
# SECTION 3: Logging
# =========================
logging.basicConfig(
    level=SETTINGS.log_level,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("OptTab")
logger.info("Settings loaded → %s", SETTINGS.model_dump())

# =========================
# SECTION 4: Storage / DuckDB (safe, lazy, single-connection)
# =========================
def _default_db_path() -> Path:
    """בחר נתיב לקובץ cache.duckdb שמקטין סיכוי לנעילות:
    1) אם הוגדר OPT_CACHE_PATH – נכבד אותו.
    2) אחרת נעדיף %LOCALAPPDATA%/pairs_trading_system/cache.duckdb (לא מסונכרן ע"י OneDrive).
    3) נפילה חזרה ל-project root.
    """
    env_path = os.environ.get("OPT_CACHE_PATH")
    if env_path:
        p = Path(env_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    local = os.environ.get("LOCALAPPDATA")
    if local:
        p = Path(local) / "pairs_trading_system" / "cache.duckdb"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    p = PROJECT_ROOT / "cache.duckdb"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p



DB_PATH = _default_db_path()

# ===== DEBUG: הדפסות כדי לוודא שהקובץ הנכון נטען ושנתיב ה-DB אינו תחת OneDrive =====
logger.info("OPT TAB LOADED FROM: %s", __file__)
logger.info("USING DB PATH: %s", DB_PATH)
if "OneDrive" in str(DB_PATH):
    logger.warning("DB path appears under OneDrive — consider OPT_CACHE_PATH or LOCALAPPDATA.")

try:
    # הצג גם ב-UI של Streamlit (אם רץ בתוך האפליקציה)
    if st.sidebar:
        with st.sidebar.expander("⚙ Debug: Optimization Tab"):
            st.write("**Loaded from:**", __file__)
            st.write("**DB Path:**", str(DB_PATH))
except Exception:
    pass


def _open_duck(db_path: Path, retries: int = 3, delay: float = 0.5):
    """Open a writable DuckDB connection with light retry + cleanup.
    Applies PRAGMA once per connection and registers close on exit.
    """
    if duckdb is None:
        raise RuntimeError("DuckDB is not installed. Run `pip install duckdb` to enable local storage.")
    last_err = None
    for i in range(retries):
        try:
            conn = duckdb.connect(str(db_path))  # read_only=False by default
            conn.execute("PRAGMA threads=4")
            atexit.register(conn.close)
            return conn
        except duckdb.IOException as e:
            last_err = e
            if "File is already open" in str(e) and i < retries - 1:
                time.sleep(delay)
                continue
            break
    # אופציונלי: אם נעול לחלוטין, נfallback לזיכרון כדי לא לקרוס (עדיף לעבוד עם read-only למעלה במקרים של דוחות)
    try:
        mem = duckdb.connect(":memory:")
        mem.execute("PRAGMA threads=4")
        atexit.register(mem.close)
        return mem
    except Exception:
        if last_err:
            raise last_err
        raise duckdb.IOException("Failed to open DuckDB connection and fallback.")


# Streamlit-aware cached factory (לא פותח חיבור ברמת import)
if hasattr(st, "cache_resource"):
    @cache_resource(show_spinner=False)
    def get_duck():
        return _open_duck(DB_PATH)
else:
    _DUCK_SINGLETON = None
    def get_duck():
        global _DUCK_SINGLETON
        if _DUCK_SINGLETON is None:
            _DUCK_SINGLETON = _open_duck(DB_PATH)
        return _DUCK_SINGLETON


def get_ro_duck():
    """חיבור קריאה-בלבד – בטוח לשימוש מקבילי לדוחות/טבלאות."""
    if duckdb is None:
        raise RuntimeError("DuckDB is not installed. Run `pip install duckdb` to enable local storage.")
    try:
        conn = duckdb.connect(str(DB_PATH), read_only=True)
        conn.execute("PRAGMA threads=4")
        atexit.register(conn.close)
        return conn
    except duckdb.IOException:
        # אם גם read_only נעול, נחזור לזיכרון לקריאה
        mem = duckdb.connect(":memory:")
        mem.execute("PRAGMA threads=4")
        atexit.register(mem.close)
        return mem


class _DuckProxy:
    """פרוקסי שמאפשר קוד קיים בסגנון DUCK.execute(...), אבל פותח חיבור רק בשימוש ראשון."""
    _conn = None
    def _ensure(self):
        if self._conn is None:
            self._conn = get_duck()
    def __getattr__(self, name):
        self._ensure()
        return getattr(self._conn, name)


# תאימות לאחור: קוד שקורא DUCK.execute(...) ימשיך לעבוד, בלי לפתוח חיבור בזמן import
DUCK = _DuckProxy()

# =========================
# SECTION 4.1: DuckDB Schema & IO Helpers
# =========================

def _ensure_duck_schema() -> None:
    """Create studies/trials tables if missing (idempotent)."""
    try:
        conn = get_duck()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS studies (
                study_id      BIGINT,
                pair          VARCHAR,
                created_at    TIMESTAMP DEFAULT now(),
                sampler       VARCHAR,
                n_trials      INTEGER,
                timeout_sec   INTEGER,
                weights_json  VARCHAR
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS trials (
                study_id    BIGINT,
                trial_no    INTEGER,
                pair        VARCHAR,
                params_json VARCHAR,
                perf_json   VARCHAR,
                score       DOUBLE,
                created_at  TIMESTAMP DEFAULT now()
            );
            """
        )
    except Exception as e:
        logger.warning("duck schema init failed: %s", e)


def make_json_safe(obj: Any) -> Any:
    """Ensure an object is JSON-serializable for Streamlit display and DB."""
    try:
        json.dumps(obj)
        return obj
    except Exception:
        if isinstance(obj, dict):
            return {str(k): make_json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [make_json_safe(x) for x in obj]
        return str(obj)


def save_trials_to_duck(df: pd.DataFrame, pair: str, sampler: str, n_trials: int, timeout_sec: int, weights: Dict[str, float]) -> Optional[int]:
    """Persist a completed optimization run: one row in studies + many rows in trials. Returns study_id."""
    if df is None or df.empty:
        return None
    try:
        _ensure_duck_schema()
        conn = get_duck()
        # study id = unix ms
        study_id = int(time.time() * 1000)
        weights_json = json.dumps(weights)
        conn.execute(
            "INSERT INTO studies (study_id, pair, sampler, n_trials, timeout_sec, weights_json) VALUES (?, ?, ?, ?, ?, ?)",
            [study_id, pair, sampler, int(n_trials), int(timeout_sec), weights_json],
        )
        # trials
        records = []
        param_cols = [c for c in df.columns if c not in ("Sharpe","Profit","Drawdown","Score")]
        for i, row in df.reset_index(drop=True).iterrows():
            params: Dict[str, Any] = {}
            for k in param_cols:
                v = row.get(k)
                if pd.api.types.is_numeric_dtype(df[k]):
                    try:
                        params[k] = float(v)
                    except Exception:
                        params[k] = None
                else:
                    params[k] = None if pd.isna(v) else str(v)
            perf = {k: row[k] for k in ("Sharpe","Profit","Drawdown") if k in df.columns}
            records.append((
                study_id,
                int(i),
                pair,
                json.dumps(make_json_safe(params)),
                json.dumps(make_json_safe(perf)),
                float(row.get("Score", float("nan"))),
            ))
        conn.executemany(
            "INSERT INTO trials (study_id, trial_no, pair, params_json, perf_json, score) VALUES (?, ?, ?, ?, ?, ?)",
            records,
        )
        return study_id
    except Exception as e:
        logger.warning("save_trials_to_duck failed: %s", e)
        return None


def list_pairs_in_db(limit: int = 200) -> List[str]:
    try:
        conn = get_ro_duck()
        df = conn.execute("SELECT DISTINCT pair FROM studies ORDER BY pair LIMIT ?", [int(limit)]).df()
        return sorted(df["pair"].dropna().astype(str).tolist())
    except Exception:
        return []


def list_studies_for_pair(pair: str, limit: int = 30) -> pd.DataFrame:
    try:
        conn = get_ro_duck()
        q = "SELECT study_id, created_at, sampler, n_trials, timeout_sec FROM studies WHERE pair = ? ORDER BY created_at DESC LIMIT ?"
        return conn.execute(q, [pair, int(limit)]).df()
    except Exception as e:
        logger.debug("list_studies_for_pair failed: %s", e)
        return pd.DataFrame()


def load_trials_from_duck(study_id: int) -> pd.DataFrame:
    try:
        conn = get_ro_duck()
        q = "SELECT trial_no, params_json, perf_json, score FROM trials WHERE study_id = ? ORDER BY trial_no"
        df = conn.execute(q, [int(study_id)]).df()
        if df.empty:
            return df
        # expand json columns
        params_df = df["params_json"].apply(json.loads).apply(pd.Series)
        perf_df = df["perf_json"].apply(json.loads).apply(pd.Series)
        out = pd.concat([params_df, perf_df, df[["score"]].rename(columns={"score":"Score"})], axis=1)
        return out
    except Exception as e:
        logger.warning("load_trials_from_duck failed: %s", e)
    return pd.DataFrame()

# =========================
# SECTION 4.2: Metrics fallback (if core.metrics is missing)
# =========================

def _norm_fallback(perf: Dict[str, float]) -> Dict[str, float]:
    # simple min-max like normalization with safe guards
    sharpe = float(perf.get("Sharpe", 0.0))
    profit = float(perf.get("Profit", 0.0))
    dd = float(perf.get("Drawdown", 0.0))
    # scale roughly into 0..1 ranges
    s = 0.5 + np.tanh(sharpe / 3.0) / 2.0
    p = 0.5 + np.tanh(profit / 1e4) / 2.0
    d = 1.0 - min(max(dd, 0.0), 1.0)  # expect drawdown fraction
    return {"Sharpe": s, "Profit": p, "Drawdown": d}


def _score_fallback(norm: Dict[str, float], weights: Dict[str, float]) -> float:
    w = {k: float(v) for k, v in weights.items()}
    z = sum(abs(v) for v in w.values()) or 1.0
    w = {k: v / z for k, v in w.items()}
    return float(sum(w.get(k, 0.0) * float(norm.get(k, 0.0)) for k in ("Sharpe","Profit","Drawdown")))


# =========================
# SECTION 6.5: Helper Types & Utilities (no duplication)
# =========================

from typing import TypeAlias

ParamRange: TypeAlias = Tuple[float, float, Optional[float]]  # (low, high, optional step)


def get_default_param_ranges() -> Dict[str, ParamRange]:
    """Resolve parameter ranges from core when available, else provide safe fallbacks.
    Priority:
        1) core.ranges.get_default_ranges()
        2) core.params.PARAM_SPECS  →  {name: (low, high, step?)}
        3) Minimal defaults
    """
    if core_get_default_ranges is not None:
        try:
            out = core_get_default_ranges()
            if isinstance(out, dict):
                return out  # expected format
        except Exception as e:
            logger.warning("get_default_param_ranges: core_get_default_ranges failed: %s", e)

    if CORE_PARAM_SPECS is not None and isinstance(CORE_PARAM_SPECS, dict):
        out2: Dict[str, ParamRange] = {}
        for name, spec in CORE_PARAM_SPECS.items():
            try:
                lo = float(spec.get("low", 0.0))
                hi = float(spec.get("high", 1.0))
                step = spec.get("step")
                stepf = float(step) if step is not None else None
                if hi <= lo:
                    hi = lo + 1.0
                out2[name] = (lo, hi, stepf)
            except Exception as e:
                logger.debug("PARAM_SPECS entry skipped (%s): %s", name, e)
        if out2:
            return out2

    # minimal fallback
    return {"lookback": (20.0, 120.0, 5.0), "z_open": (1.0, 3.0, 0.1), "z_close": (0.2, 2.0, 0.1)}


def run_backtest(sym1: str, sym2: str, params: Dict[str, Any]) -> Dict[str, float]:
    """Thin wrapper around core.optimization_backtester.Backtester, returning perf metrics."""
    if Backtester is None:
        raise RuntimeError("Backtester unavailable (core.optimization_backtester.Backtester not found)")
    bt = Backtester(symbol_a=sym1, symbol_b=sym2, **params)  # adjust constructor if needed
    res = bt.run()
    return {
        "Sharpe": float(res.get("Sharpe", 0.0)),
        "Profit": float(res.get("Profit", 0.0)),
        "Drawdown": float(res.get("Drawdown", 0.0)),
    }


def _render_profile_sidebar() -> Optional[OptSettings]:
    try:
        with st.sidebar.expander("Profile / Settings", expanded=False):
            lvl = st.selectbox(
                "Log level",
                ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                index=["DEBUG","INFO","WARNING","ERROR","CRITICAL"].index(SETTINGS.log_level),
                key="opt_prof_log_level",
            )
            data_dir = st.text_input("Data dir", str(SETTINGS.data_dir), key="opt_prof_data_dir")
            studies_dir = st.text_input("Studies dir", str(SETTINGS.studies_dir), key="opt_prof_studies_dir")
            if st.button("Apply profile", key="opt_prof_apply"):
                p_data = Path(data_dir)
                p_studies = Path(studies_dir)
                p_data.mkdir(parents=True, exist_ok=True)
                p_studies.mkdir(parents=True, exist_ok=True)
                return OptSettings(env=SETTINGS.env, log_level=lvl, data_dir=p_data, studies_dir=p_studies)
    except Exception:
        return None
    return None


def _sidebar_common() -> Dict[str, Any]:
    """Return shared sidebar configuration: parameter ranges & metric weights (editable)."""
    with st.sidebar.expander("Optimization Config", expanded=True):
        ranges = get_default_param_ranges()
        try:
            c1, c2, c3 = st.columns(3)
            w_sh = c1.number_input("W Sharpe", value=0.5, step=0.1, key="opt_w_sharpe")
            w_pf = c2.number_input("W Profit", value=0.3, step=0.1, key="opt_w_profit")
            w_dd = c3.number_input("W Drawdown", value=0.2, step=0.1, key="opt_w_dd")
            weights = {"Sharpe": float(w_sh), "Profit": float(w_pf), "Drawdown": float(w_dd)}
        except Exception:
            weights = {"Sharpe": 0.5, "Profit": 0.3, "Drawdown": 0.2}
    return {"ranges": ranges, "weights": weights}


def _recommend_params(df: pd.DataFrame, param_cols: List[str]):
    try:
        from sklearn.ensemble import RandomForestRegressor
        if not param_cols:
            return []
        X = df[param_cols].select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if "Score" not in df.columns or len(X) < 10:
            return []
        y = df["Score"].astype(float)
        rf = RandomForestRegressor(n_estimators=200, random_state=42)
        rf.fit(X, y)
        return sorted(zip(param_cols, rf.feature_importances_), key=lambda x: x[1], reverse=True)
    except Exception:
        return []


def _suggest_ranges_from_df(df: pd.DataFrame, param_cols: List[str], q_low: float = 0.1, q_high: float = 0.9) -> Dict[str, ParamRange]:
    out: Dict[str, ParamRange] = {}
    if df is None or df.empty or not param_cols:
        return out
    try:
        top = df.sort_values("Score", ascending=False).head(50)
        for c in param_cols:
            if not pd.api.types.is_numeric_dtype(top[c]):
                continue
            lo = float(top[c].quantile(q_low))
            hi = float(top[c].quantile(q_high))
            if hi <= lo:
                hi = lo + 1e-9
            out[c] = (lo, hi, None)
    except Exception as e:
        logger.debug("_suggest_ranges_from_df: failed: %s", e)
        return out
    return out


def _suggest_ranges_from_cluster(df: pd.DataFrame, param_cols: List[str], labels: pd.Series, cluster_id: int, width: float = 0.5) -> Dict[str, ParamRange]:
    out: Dict[str, ParamRange] = {}
    try:
        mask = (labels.astype(int) == int(cluster_id))
        sub = df.loc[mask, param_cols].select_dtypes(include=[np.number]).dropna()
        if sub.empty:
            return out
        mu = sub.mean()
        sd = sub.std(ddof=0).replace(0.0, np.nan).fillna(1e-9)
        for p in param_cols:
            m = float(mu.get(p, 0.0))
            s = float(sd.get(p, 1e-9)) * float(width)
            lo, hi = m - s, m + s
            if hi <= lo:
                hi = lo + 1e-9
            out[p] = (lo, hi, None)
    except Exception as e:
        logger.debug("_suggest_ranges_from_cluster failed: %s", e)
        return out
    return out

# =========================
# SECTION 7: UI Entrypoint (full)
# =========================

def render_optimization_tab() -> None:
    st.title("⚙️ Pairs‑Trading Optimiser — Pro")

    # Sidebar profile
    new_settings = _render_profile_sidebar()
    if new_settings is not None:
        global SETTINGS
        SETTINGS = new_settings
        logger.info("Profile applied via sidebar → %s", SETTINGS.model_dump())

    # Health essentials
    essentials = {
        "Backtester": Backtester is not None,
        "optuna": (optuna is not None) and (TPESampler is not object) and (CmaEsSampler is not object) and (MedianPruner is not object),
        "get_param_distributions": get_param_distributions is not None,
        "metrics (normalize & score)": (normalize_metrics is not None and compute_weighted_score is not None),
    }
    with st.expander("System Health", expanded=not all(essentials.values())):
        for k, v in essentials.items():
            st.write(("✅" if v else "❌"), k)
        if not essentials["Backtester"]:
            st.markdown("**Backtester missing**: ensure `core.optimization_backtester` is importable.")
        if not essentials["optuna"]:
            st.markdown("**Install Optuna**: `pip install optuna`")
        if not essentials["get_param_distributions"]:
            st.markdown("**Missing distributions**: function `core.distributions.get_param_distributions(ranges)` expected.")
        has_metrics = (normalize_metrics is not None) and (compute_weighted_score is not None)
        st.write(("✅" if has_metrics else "❌"), "metrics (normalize & score)")
        if not has_metrics:
            st.markdown("Using internal fallbacks for metrics.")

    # Load pairs (placeholder)
    pairs = [{"symbols": ["AAPL", "MSFT"]}]
    pair_names = [f"{p['symbols'][0]}-{p['symbols'][1]}" for p in pairs]
    selected = st.selectbox("Select Pair", pair_names, key=sk("pair"))
    sym1, sym2 = selected.split("-")

    sidebar_vals = _sidebar_common()

    # Configuration snapshot
    with st.expander("Configuration Snapshot", expanded=False):
        ranges_default = sidebar_vals["ranges"]
        use_active = st.toggle("Use Active Ranges (from Visual Dashboard)", value=False, key=sk("use_active_ranges"))
        ranges = st.session_state.get("active_ranges") if use_active and isinstance(st.session_state.get("active_ranges"), dict) and st.session_state.get("active_ranges") else ranges_default
        weights = sidebar_vals["weights"]
        st.write("**Pair:**", f"{sym1}-{sym2}")
        st.write("**Param count:**", len(ranges))
        st.write("**Weights:**", weights)
        # Normalize/Effective weights
        available_metrics = {"Sharpe", "Profit", "Drawdown"}
        weights_eff = {m: float(weights.get(m, 0.0)) for m in available_metrics}
        normalize_w = st.toggle("Normalize weights to sum=1", value=True, key=sk("normalize_weights"))
        if normalize_w:
            _z = sum(abs(v) for v in weights_eff.values()) or 1.0
            weights_eff = {k: v / _z for k, v in weights_eff.items()}
        st.write("**Effective Weights:**", weights_eff)
        # Preview first few ranges
        if ranges:
            preview_items = list(ranges.items())[:8]
            st.json(make_json_safe({k: v for k, v in preview_items}))

    # Controls
    sampler = st.selectbox("Sampler", ["TPE", "CMA-ES"], index=0, key=sk("sampler"))
    timeout = st.number_input("Timeout (sec)", min_value=0, max_value=7200, value=600, step=30, key=sk("timeout"))
    n_trials = st.number_input("Trials", min_value=10, max_value=5000, value=200, step=10, key=sk("n_trials"))

    # --- Safe defaults (from session_state/globals) ---
    try:
        if 'ranges' not in locals() or not isinstance(ranges, dict):
            ranges = st.session_state.get('active_ranges') or {}
        if 'weights_eff' not in locals() or not isinstance(weights_eff, dict):
            weights_eff = st.session_state.get('loaded_weights_eff') or {'Sharpe': 0.5, 'Profit': 0.3, 'Drawdown': 0.2}
        if 'essentials' not in locals() or not isinstance(essentials, dict):
            essentials = {
                'Backtester': Backtester is not None,
                'optuna': optuna is not None,
                'get_param_distributions': get_param_distributions is not None,
            }
        if 'sampler' not in locals() or not isinstance(sampler, str):
            sampler = st.session_state.get('opt_sampler_main') or 'TPE'
        if 'timeout' not in locals() or not isinstance(timeout, (int, float)):
            timeout = int(st.session_state.get('opt_timeout_main') or 600)
        if 'n_trials' not in locals() or not isinstance(n_trials, (int, float)):
            n_trials = int(st.session_state.get('opt_n_trials_main') or 200)
        if 'sym1' not in locals() or 'sym2' not in locals():
            pair_val = st.session_state.get('opt_pair_select_main') or st.session_state.get('opt_pair_select') or 'AAPL-MSFT'
            try:
                sym1, sym2 = str(pair_val).split('-')
            except Exception:
                sym1, sym2 = 'AAPL', 'MSFT'
    except Exception:
        ranges = {} ; weights_eff = {'Sharpe':0.5,'Profit':0.3,'Drawdown':0.2}
        essentials = {'Backtester': Backtester is not None, 'optuna': optuna is not None, 'get_param_distributions': get_param_distributions is not None}
        sampler='TPE'; timeout=600; n_trials=200; sym1='AAPL'; sym2='MSFT'

    # Ensure symbol exists (satisfy Pylance/static analysis) and call if wired
    if 'render_opt_ui_actions_block' not in globals():
        def render_opt_ui_actions_block(*_args, **_kwargs):  # placeholder to satisfy linters
            with st.expander("Actions (placeholder)", expanded=False):
                st.info("`render_opt_ui_actions_block` placeholder: define/import the real implementation to enable actions.")

    if callable(render_opt_ui_actions_block):
        render_opt_ui_actions_block(ranges, weights_eff, essentials, sampler, n_trials, timeout, sym1, sym2)
    else:
        with st.expander("Actions (not wired)", expanded=False):
            st.info("`render_opt_ui_actions_block` is not callable. Define/import it or paste its implementation.")
            st.json(make_json_safe({
                "pair": f"{sym1}-{sym2}",
                "have_optuna": bool(optuna is not None),
                "have_backtester": bool(Backtester is not None),
                "n_ranges": len(ranges) if isinstance(ranges, dict) else 0,
                "weights": weights_eff,
            }))



def _open_duck(db_path: Path, retries: int = 3, delay: float = 0.5):
    """Open a writable DuckDB connection with light retry + cleanup.
    Applies PRAGMA once per connection and registers close on exit.
    """
    last_err = None
    for i in range(retries):
        try:
            conn = duckdb.connect(str(db_path))  # read_only=False by default
            conn.execute("PRAGMA threads=4")
            atexit.register(conn.close)
            return conn
        except duckdb.IOException as e:
            last_err = e
            if "File is already open" in str(e) and i < retries - 1:
                time.sleep(delay)
                continue
            break
    # אופציונלי: אם נעול לחלוטין, נfallback לזיכרון כדי לא לקרוס (עדיף לעבוד עם read-only למעלה במקרים של דוחות)
    try:
        mem = duckdb.connect(":memory:")
        mem.execute("PRAGMA threads=4")
        atexit.register(mem.close)
        return mem
    except Exception:
        if last_err:
            raise last_err
        raise duckdb.IOException("Failed to open DuckDB connection and fallback.")


# Streamlit-aware cached factory (לא פותח חיבור ברמת import)
if hasattr(st, "cache_resource"):
    @st.cache_resource(show_spinner=False)
    def get_duck():
        return _open_duck(DB_PATH)
else:
    _DUCK_SINGLETON = None
    def get_duck():
        global _DUCK_SINGLETON
        if _DUCK_SINGLETON is None:
            _DUCK_SINGLETON = _open_duck(DB_PATH)
        return _DUCK_SINGLETON


def _build_objective(sym1: str, sym2: str, ranges: Dict[str, Tuple[float, float, Optional[float]]], weights: Dict[str, float]):
    """Create an Optuna objective using real Backtester, scoring via metrics or fallbacks."""
    def objective(trial: "optuna.trial.Trial") -> float:  # type: ignore
        # sample params
        params: Dict[str, Any] = {}
        for name, (lo, hi, step) in ranges.items():
            if step is None:
                params[name] = trial.suggest_float(name, float(lo), float(hi))
            else:
                params[name] = trial.suggest_float(name, float(lo), float(hi), step=float(step))
        # run backtest
        if Backtester is None:
            raise RuntimeError("Backtester module is not available")
        # apply UI→Backtester param mapping & sanitize
        try:
            import streamlit as st
            mapping = st.session_state.get("opt_param_mapping")
        except Exception:
            mapping = None
        params_mapped = _apply_param_mapping(params, mapping)
        params_mapped = _sanitize_bt_kwargs(params_mapped)
        _log(f"BT kwargs (objective): {params_mapped}", "INFO")
        bt = Backtester(symbol_a=sym1, symbol_b=sym2, **params_mapped)  # adjust ctor if needed
        perf = bt.run()  # expected dict with Sharpe/Profit/Drawdown
        sharpe = float(perf.get("Sharpe", 0.0))
        profit = float(perf.get("Profit", 0.0))
        dd = float(perf.get("Drawdown", 0.0))
        perf_map = {"Sharpe": sharpe, "Profit": profit, "Drawdown": dd}
        nm = normalize_metrics(perf_map) if callable(normalize_metrics) else _norm_fallback(perf_map)
        score = compute_weighted_score(nm, weights) if callable(compute_weighted_score) else _score_fallback(nm, weights)
        trial.set_user_attr("perf", perf_map)
        trial.set_user_attr("params", params)
        return float(score)
    return objective


def _optuna_optimize(sym1: str, sym2: str, ranges: Dict[str, Tuple[float, float, Optional[float]]], weights: Dict[str, float], sampler_name: str, n_trials: int, timeout: int) -> pd.DataFrame:
    """Run a real Optuna study when core & optuna are available. Returns results DataFrame."""
    if optuna is None or Backtester is None or get_param_distributions is None:
        raise RuntimeError("Real optimization unavailable (missing optuna/core)")

    # Build distributions (prefer central function if it transforms ranges)
    try:
        distributions = get_param_distributions(ranges)
        # When distributions object is returned, we still pass raw ranges to sampling to keep UI understandable.
        _ = distributions  # noqa: F841 (for linters)
    except Exception:
        # fall back to raw ranges
        pass

    if sampler_name.upper() == "CMA-ES" and CmaEsSampler is not object:
        sampler = CmaEsSampler()
    else:
        sampler = TPESampler() if TPESampler is not object else None

    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(_build_objective(sym1, sym2, ranges, weights), n_trials=int(n_trials), timeout=int(timeout) if timeout else None)

    # Collect trials into DataFrame
    rows: List[Dict[str, Any]] = []
    for t in study.trials:
        params = dict(t.user_attrs.get("params", {}))
        perf = dict(t.user_attrs.get("perf", {}))
        row = {**params, **perf, "Score": t.value}
        rows.append(row)
    return pd.DataFrame(rows)


# ---- (keep UI code below; we'll refactor placement in the next step) ----

def _validate_ranges(ranges: Dict[str, Tuple[float, float, Optional[float]]]) -> Dict[str, Tuple[float, float, Optional[float]]]:
    """Ensure each (lo, hi, step) is numeric and lo < hi; auto-fix trivial issues."""
    fixed: Dict[str, Tuple[float, float, Optional[float]]] = {}
    for k, v in ranges.items():
        try:
            lo, hi, step = (float(v[0]), float(v[1]), (float(v[2]) if v[2] is not None else None))
            if hi <= lo:
                hi = lo + (step if step is not None else 1e-9)
            fixed[k] = (lo, hi, step)
        except Exception:
            # drop invalid entries silently
            continue
    return fixed


def _apply_param_mapping(params: Dict[str, Any], mapping: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Map UI param names -> Backtester kwarg names. If mapping is None/empty, return as-is."""
    if not mapping:
        return params
    out: Dict[str, Any] = {}
    for k, v in params.items():
        out[mapping.get(k, k)] = v
    return out


def _normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    z = sum(abs(float(v)) for v in weights.values()) or 1.0
    return {k: float(v) / z for k, v in weights.items()}


def _log(msg: str, level: str = "INFO") -> None:
    """Append a message to session logs and to python logger."""
    import streamlit as st
    key = "opt_logs"
    st.session_state.setdefault(key, [])
    st.session_state[key].append(f"[{level}] {msg}")
    try:
        if level == "ERROR":
            logger.warning(msg)
        elif level == "WARN":
            logger.warning(msg)
        else:
            logger.info(msg)
    except Exception:
        pass


def _get_logs() -> str:
    import streamlit as st
    logs = st.session_state.get("opt_logs", [])
    return "\n".join(logs)


def _validate_for_optuna(ranges: Dict[str, Tuple[float, float, Optional[float]]], n_trials: int, timeout: int) -> None:
    """Raise ValueError with a friendly message if inputs are not valid for Optuna."""
    if n_trials <= 0:
        raise ValueError("n_trials must be > 0")
    for name, tpl in ranges.items():
        if not isinstance(tpl, tuple) or len(tpl) < 2:
            raise ValueError(f"Range for '{name}' is malformed")
        lo, hi = float(tpl[0]), float(tpl[1])
        if not (lo < hi):
            raise ValueError(f"Range for '{name}' must satisfy lo < hi (got {lo} >= {hi})")
    if timeout < 0:
        raise ValueError("timeout must be >= 0")


def _bt_known_kwargs() -> Optional[set]:
    try:
        import inspect
        if Backtester is None:
            return None
        sig = inspect.signature(Backtester.__init__)
        names = {p.name for p in sig.parameters.values() if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)}
        names.discard('self')
        return names
    except Exception:
        return None


def _sanitize_bt_kwargs(bt_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    names = _bt_known_kwargs()
    if not names:
        return bt_kwargs
    unknown = [k for k in bt_kwargs.keys() if k not in names]
    if unknown:
        _log(f"Unknown Backtester kwargs ignored: {unknown}", "WARN")
    return {k: v for k, v in bt_kwargs.items() if k in names}

# --- Data Quality helpers ---
def _dq_coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce all numeric-like columns and standardize infinities to NaN."""
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_object_dtype(out[c]):
            try:
                out[c] = pd.to_numeric(out[c], errors="ignore")
            except Exception:
                pass
    out = out.replace([np.inf, -np.inf], np.nan)
    return out

def _dq_hash_params_row(row: pd.Series, exclude: Iterable[str]) -> tuple:
    """Hash a row's parameter set to help identify duplicates."""
    keys = sorted([k for k in row.index if k not in exclude])
    return tuple((k, row[k]) for k in keys)

def _dq_basic_clean(df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """Return (clean_df, report). Light-touch: coerce numeric, drop exact-duplicate rows, optional dedup by params."""
    report: Dict[str, Any] = {}
    if df is None or df.empty:
        report["empty"] = True
        return df, report
    exclude_cols = {"Sharpe", "Profit", "Drawdown", "Score"}
    before = len(df)
    df2 = _dq_coerce_numeric(df)
    # exact duplicate rows
    dup_exact = int(df2.duplicated().sum())
    df2 = df2.drop_duplicates()
    # optional param-set duplicate detection (same params but different score/perf)
    try:
        param_cols = [c for c in df2.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df2[c])]
        if param_cols:
            hashes = df2.apply(lambda r: _dq_hash_params_row(r[param_cols], set()), axis=1)
            dup_param_mask = hashes.duplicated()
            dup_param = int(dup_param_mask.sum())
        else:
            dup_param = 0
    except Exception:
        dup_param = 0
    after = len(df2)
    report.update({
        "rows_before": before,
        "rows_after": after,
        "exact_duplicates_removed": dup_exact,
        "paramset_duplicates_detected": dup_param,
        "nan_fraction": float(df2.isna().mean().mean()) if after else 0.0,
    })
    return df2, report

    # Quick Optuna Smoke (placeholder)
    if st.button("Quick Optuna Smoke (3 trials)", key="opt_btn_smoke", disabled=not all(essentials.values())):
        try:
            df_smoke = pd.DataFrame()
            if optuna is not None and get_param_distributions is not None:
                df_smoke = pd.DataFrame([
                    {
                        **{k: (lo + hi) / 2.0 for k, (lo, hi, _s) in ranges.items()},
                        "Sharpe": np.random.normal(1.0, 0.3),
                        "Profit": np.random.normal(2000, 500),
                        "Drawdown": np.random.uniform(0.05, 0.25),
                        "Score": np.random.uniform(0.2, 0.9),
                    }
                    for _ in range(3)
                ])
            st.success("Optuna smoke ran (placeholder)")
            if not df_smoke.empty:
                st.dataframe(df_smoke, width="stretch")
        except Exception as e:
            st.error(f"Optuna smoke failed: {e}")

    # Run Optimization (prefers real Optuna+Backtester; falls back to simulated)
    if st.button("Run Optimization", key="opt_btn_run", disabled=not any(essentials.values())):
        try:
            if optuna is not None and Backtester is not None and get_param_distributions is not None:
                try:
                    _validate_for_optuna(ranges, int(n_trials), int(timeout))
                except Exception as ve:
                    _log(f"Optuna guard: {ve}", "ERROR")
                    st.error(f"Validation failed: {ve}")
                    raise
                df = _optuna_optimize(sym1, sym2, ranges, weights_eff, sampler, int(n_trials), int(timeout))
                st.success("Optimization finished (real Optuna)")
            else:
                rng = np.random.default_rng(42)
                rows = []
                for _ in range(int(n_trials)):
                    row = {k: rng.uniform(lo, hi) for k, (lo, hi, _s) in ranges.items()}
                    # apply mapping for simulated kwargs too (consistency)
                    row = _apply_param_mapping(row, st.session_state.get("opt_param_mapping"))
                    row.update({
                        "Sharpe": rng.normal(1.0, 0.4),
                        "Profit": rng.normal(2500, 800),
                        "Drawdown": rng.uniform(0.05, 0.3),
                    })
                    nm = normalize_metrics(row) if callable(normalize_metrics) else _norm_fallback(row)
                    row["Score"] = (
                        compute_weighted_score(nm, weights_eff)
                        if callable(compute_weighted_score)
                        else _score_fallback(nm, weights_eff)
                    )
                    rows.append(row)
                df = pd.DataFrame(rows)
                st.success("Optimization finished (simulated)")

            st.session_state["opt_df"] = df
            
            # --- Advanced Metrics enrichment (optional) ---
            try:
                if compute_advanced_metrics is not None:
                    df_adv = compute_advanced_metrics(st.session_state["opt_df"])  # returns df with extra columns
                    if isinstance(df_adv, pd.DataFrame) and not df_adv.empty:
                        st.session_state["opt_df"] = df_adv
                        df = df_adv
                        _log("Advanced metrics added to dataframe", "INFO")
            except Exception as _e:
                _log(f"Advanced metrics failed: {_e}", "WARN")

            # --- Risk-Parity (optional) ---
            try:
                if apply_risk_parity is not None:
                    rp_weights = apply_risk_parity(df)
                    with st.expander("Risk-Parity Weights", expanded=False):
                        st.json(make_json_safe(rp_weights))
                        st.download_button(
                            "Download risk_parity_weights.json",
                            data=json.dumps(make_json_safe(rp_weights), ensure_ascii=False, indent=2).encode("utf-8"),
                            file_name="risk_parity_weights.json",
                            mime="application/json",
                            key="opt_dl_rp_json",
                        )
            except Exception as _e:
                _log(f"Risk parity failed: {_e}", "WARN")

            # Save to DuckDB
            try:
                pair_key = f"{sym1}-{sym2}"
                study_id = save_trials_to_duck(df, pair_key, sampler, int(n_trials), int(timeout), weights_eff)
                if study_id:
                    st.caption(f"Saved run to DuckDB (study_id={study_id})")
                    st.session_state["last_study_id"] = study_id
            except Exception as ex:
                st.info(f"Save to DuckDB skipped: {ex}")
        except Exception as e:
            st.error(f"Optimization failed: {e}")

    # ----- Visual Dashboard -----
    st.subheader("Visual Dashboard")

    # Sidebar utilities
    with st.sidebar:
        # Param Mapping UI
        with st.expander("Param Mapping (UI → Backtester)", expanded=False):
            # Default mapping (only if none set yet)
            try:
                if not st.session_state.get("opt_param_mapping"):
                    st.session_state["opt_param_mapping"] = {
                        "lookback": "rolling_window",
                        "z_open": "entry_z",
                        "z_close": "exit_z",
                    }
            except Exception:
                pass
            mapping_json = st.text_area(
                "Mapping (JSON)",
                value=(
                    json.dumps(st.session_state.get("opt_param_mapping", {}), ensure_ascii=False, indent=2)
                    if st.session_state.get("opt_param_mapping") else "{}"
                ),
                key="opt_map_json",
                height=160,
                help=(
                    "Example:\n"
                    "{\n"
                    '  "lookback": "rolling_window",\n'
                    '  "z_open": "entry_z",\n'
                    '  "z_close": "exit_z"\n'
                    "}"
                ),
            )
            colm1, colm2 = st.columns([1,1])
            if colm1.button("Apply Mapping", key="opt_btn_apply_map"):
                try:
                    st.session_state["opt_param_mapping"] = json.loads(mapping_json) if mapping_json.strip() else {}
                    st.success("Mapping applied")
                except Exception as e:
                    st.error(f"Invalid JSON: {e}")
            if colm2.button("Clear Mapping", key="opt_btn_clear_map"):
                st.session_state["opt_param_mapping"] = {}
                st.success("Mapping cleared")

        # Load from DuckDB
        with st.expander("Load past study (DuckDB)", expanded=False):
            pairs_db = list_pairs_in_db()
            sel_pair = st.selectbox("Pair", pairs_db, key="opt_db_pair") if pairs_db else None
            if sel_pair:
                df_studies = list_studies_for_pair(sel_pair)
                if df_studies.empty:
                    st.caption("No studies yet for selected pair")
                else:
                    options = [
                        f"{int(r.study_id)} | {str(r.created_at)} | {r.sampler} {int(r.n_trials)}t/{int(r.timeout_sec)}s"
                        for _, r in df_studies.iterrows()
                    ]
                    idx = st.selectbox(
                        "Study",
                        list(range(len(options))),
                        format_func=lambda i: options[i],
                        key="opt_db_study_idx",
                    )
                    if st.button("Load study", key="opt_db_load_btn"):
                        sid = int(df_studies.iloc[int(idx)].study_id)
                        df_loaded = load_trials_from_duck(sid)
                        if df_loaded is None or df_loaded.empty:
                            st.warning("Selected study contains no trials")
                        else:
                            st.session_state["opt_df"] = df_loaded
                            st.success(f"Loaded study {sid} → session")
        # Export current DF → DuckDB as new study
        with st.expander("Export current results to DuckDB", expanded=False):
            df_curr = st.session_state.get("opt_df", pd.DataFrame())
            if df_curr is None or df_curr.empty:
                st.caption("No results in session.")
            else:
                sampler_used = st.text_input("Sampler label", value="manual", key="opt_db_export_sampler")
                n_trials_used = st.number_input(
                    "n_trials (label)", min_value=1, value=int(len(df_curr)), step=1, key="opt_db_export_trials"
                )
                timeout_used = st.number_input("timeout_sec (label)", min_value=0, value=0, step=10, key="opt_db_export_timeout")
                weights_used = weights_eff
                if st.button("Export session DF to DuckDB", key="opt_db_export_btn"):
                    try:
                        sid2 = save_trials_to_duck(
                            df_curr, f"{sym1}-{sym2}", sampler_used, int(n_trials_used), int(timeout_used), weights_used
                        )
                        if sid2:
                            st.success(f"Saved current DF to DuckDB (study_id={sid2})")
                            st.session_state["last_study_id"] = sid2
                        else:
                            st.warning("Export failed or produced no study_id.")
                    except Exception as e:
                        st.error(f"Export failed: {e}")
        # Settings Snapshot (export/import)
        with st.expander("Settings Snapshot", expanded=False):
            try:
                snapshot = {
                    "pair": f"{sym1}-{sym2}",
                    "timestamp": pd.Timestamp.now().isoformat(),
                    "ranges": make_json_safe(ranges),
                    "weights": make_json_safe(weights_eff),
                    "mapping": make_json_safe(st.session_state.get("opt_param_mapping", {})),
                    "normalize_weights": bool(st.session_state.get("opt_normalize_weights", True)),
                }
                c1, c2 = st.columns(2)
                with c1:
                    st.download_button(
                        "Download settings_snapshot.json",
                        data=json.dumps(snapshot, ensure_ascii=False, indent=2).encode("utf-8"),
                        file_name="settings_snapshot.json",
                        mime="application/json",
                        key="opt_dl_settings_snapshot",
                        width="stretch",
                    )
                    if st.button("Apply Active (ranges/weights/mapping)", key="opt_btn_apply_active", width="stretch"):
                        try:
                            # Ranges
                            st.session_state["active_ranges"] = snapshot["ranges"]
                            # Weights → mirror into both possible UI spaces
                            w = snapshot.get("weights", {}) or {}
                            for k_ui in ("w_sharpe", "opt_w_sharpe_main"):
                                st.session_state[k_ui] = float(w.get("Sharpe", 0.5))
                            for k_ui in ("w_profit", "opt_w_profit_main"):
                                st.session_state[k_ui] = float(w.get("Profit", 0.3))
                            for k_ui in ("w_dd", "opt_w_dd_main"):
                                st.session_state[k_ui] = float(w.get("Drawdown", 0.2))
                            # Normalize flag
                            st.session_state["normalize_weights"] = bool(snapshot.get("normalize_weights", True))
                            st.session_state["opt_normalize_weights_main"] = bool(snapshot.get("normalize_weights", True))
                            # Mapping
                            if isinstance(snapshot.get("mapping"), dict):
                                st.session_state["opt_param_mapping"] = snapshot["mapping"]
                            st.success("Applied active ranges/weights/mapping from snapshot → UI/session.")
                        except Exception as _e:
                            st.error(f"Apply Active failed: {_e}")
                with c2:
                    uploaded = st.file_uploader("Load snapshot (.json)", type=["json"], key="opt_snapshot_upl")
                    if uploaded is not None:
                        try:
                            snap = json.loads(uploaded.read().decode("utf-8"))
                            if isinstance(snap.get("ranges"), dict):
                                st.session_state["active_ranges"] = snap["ranges"]
                            if isinstance(snap.get("weights"), dict):
                                st.session_state["loaded_weights_eff"] = snap["weights"]
                            if isinstance(snap.get("mapping"), dict):
                                st.session_state["opt_param_mapping"] = snap["mapping"]
                            st.success("Snapshot loaded → active ranges/mapping (weights in 'loaded_weights_eff').")
                        except Exception as e:
                            st.error(f"Invalid snapshot: {e}")
            except Exception as _e:
                st.info(f"Settings snapshot unavailable: {_e}")

        # AutoML Summary (optional)
        with st.expander("AutoML Summary", expanded=False):
            df_curr = st.session_state.get("opt_df", pd.DataFrame())
            if run_automl_summary is None:
                st.caption("AutoML module not available.")
            elif df_curr is None or df_curr.empty:
                st.caption("No results in session.")
            else:
                try:
                    automl_summary = run_automl_summary(df_curr)
                    if isinstance(automl_summary, pd.DataFrame):
                        st.dataframe(automl_summary, width="stretch")
                        st.download_button(
                            "Download automl_summary.csv",
                            data=automl_summary.to_csv(index=False).encode("utf-8"),
                            file_name="automl_summary.csv",
                            mime="text/csv",
                            key="opt_dl_automl_csv",
                        )
                    else:
                        st.json(make_json_safe(automl_summary))
                except Exception as _e:
                    st.error(f"AutoML summary failed: {_e}")
                    _log(f"AutoML summary failed: {_e}", "ERROR")

        # Download study metadata (JSON)
        with st.expander("Download study metadata (JSON)", expanded=False):
            try:
                meta = {
                    "pair": f"{sym1}-{sym2}",
                    "timestamp": pd.Timestamp.now().isoformat(),
                    "last_study_id": int(st.session_state.get("last_study_id", 0)) if st.session_state.get("last_study_id") else None,
                    "rows_in_df": int(st.session_state.get("opt_df", pd.DataFrame()).shape[0]),
                    "normalize_weights": bool(st.session_state.get("normalize_weights", True)),
                }
                st.download_button(
                    "Download metadata.json",
                    data=json.dumps(meta, ensure_ascii=False, indent=2).encode("utf-8"),
                    file_name="metadata.json",
                    mime="application/json",
                    key="opt_dl_meta_json",
                )
            except Exception:
                st.caption("No metadata available.")

        # Study Profile (full export)
        with st.expander("Study Profile (full export)", expanded=False):
            try:
                # KPIs (safe)
                kpi = {}
                try:
                    if "Sharpe" in df.columns:
                        kpi["avg_sharpe"] = float(pd.to_numeric(df["Sharpe"], errors="coerce").mean())
                    if "Score" in df.columns:
                        kpi["best_score"] = float(pd.to_numeric(df["Score"], errors="coerce").max())
                    if "Profit" in df.columns:
                        kpi["max_profit"] = float(pd.to_numeric(df["Profit"], errors="coerce").max())
                    if "Drawdown" in df.columns:
                        kpi["min_drawdown"] = float(pd.to_numeric(df["Drawdown"], errors="coerce").min())
                except Exception:
                    pass
                # Data‑quality quick report (no mutation)
                dq_df, dq_report = _dq_basic_clean(df)
                profile = {
                    "pair": f"{sym1}-{sym2}",
                    "timestamp": pd.Timestamp.now().isoformat(),
                    "sampler": str(sampler),
                    "n_trials": int(n_trials),
                    "timeout_sec": int(timeout),
                    "weights": make_json_safe(weights_eff),
                    "ranges": make_json_safe(ranges),
                    "mapping": make_json_safe(st.session_state.get("opt_param_mapping", {})),
                    "last_study_id": int(st.session_state.get("last_study_id", 0)) if st.session_state.get("last_study_id") else None,
                    "rows": int(df.shape[0]),
                    "cols": int(df.shape[1]),
                    "kpi": kpi,
                    "data_quality": make_json_safe(dq_report),
                }
                # JSON export
                st.download_button(
                    "Download study_profile.json",
                    data=json.dumps(profile, ensure_ascii=False, indent=2).encode("utf-8"),
                    file_name="study_profile.json",
                    mime="application/json",
                    key="opt_dl_study_profile_json",
                )
                # Markdown export (compact)
                md = [
                    f"# Study Profile — {sym1}-{sym2}",
                    f"**Timestamp:** {profile['timestamp']}",
                    f"**Sampler:** {profile['sampler']}  |  **Trials:** {profile['n_trials']}  |  **Timeout:** {profile['timeout_sec']}s",
                    f"**Rows:** {profile['rows']}  |  **Cols:** {profile['cols']}",
                    "\n## KPIs:",
                    json.dumps(profile["kpi"], ensure_ascii=False, indent=2),
                    "\n## Weights:",
                    json.dumps(profile["weights"], ensure_ascii=False, indent=2),
                    "\n## Ranges (head):",
                    json.dumps(dict(list(profile["ranges"].items())[:10]), ensure_ascii=False, indent=2),
                    "\n## Param Mapping:",
                    json.dumps(profile["mapping"], ensure_ascii=False, indent=2),
                    "\n## Data Quality:",
                    json.dumps(profile["data_quality"], ensure_ascii=False, indent=2),
                ]
                st.download_button(
                    "Download study_profile.md",
                    data="\n\n".join(md).encode("utf-8"),
                    file_name="study_profile.md",
                    mime="text/markdown",
                    key="opt_dl_study_profile_md",
                )
            except Exception as _e:
                st.info(f"Study profile unavailable: {_e}")

    # Results frame
    df = st.session_state.get("opt_df", pd.DataFrame())
    if df.empty:
        st.info("Run an optimization first.")
        return

    # Data Quality — inspect & (optionally) clean
    with st.expander("Data Quality", expanded=False):
        c1, c2, c3 = st.columns(3)
        show_report = c1.checkbox("Show report", value=True, key="opt_dq_show_report")
        dedup_params = c2.checkbox("Dedup by parameter set", value=False, key="opt_dq_dedup_params")
        apply_clean = c3.checkbox("Apply cleaning to session", value=False, key="opt_dq_apply")

        # always coerce/compute report on a copy
        df_clean, report = _dq_basic_clean(df)
        if show_report:
            st.json(make_json_safe(report))
        if dedup_params:
            try:
                exclude_cols = {"Sharpe", "Profit", "Drawdown", "Score"}
                param_cols = [c for c in df_clean.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df_clean[c])]
                if param_cols:
                    hashes = df_clean.apply(lambda r: _dq_hash_params_row(r[param_cols], set()), axis=1)
                    df_clean = df_clean.loc[~hashes.duplicated()].copy()
            except Exception as _e:
                _log(f"DQ param-dedup failed: {_e}", "WARN")
        st.dataframe(df_clean.head(50), width="stretch")
        if apply_clean:
            st.session_state["opt_df"] = df_clean
            df = df_clean
            st.success("Applied data-quality cleaning to session results")

    # Filter & Inspect
    with st.expander("Filter & Inspect", expanded=False):
        min_score = st.slider("Min Score", 0.0, 1.0, 0.0, 0.01, key="opt_filter_min_score") if "Score" in df.columns else 0.0
        view_cols = [c for c in df.columns if c != "Score"]
        default_cols = view_cols[: min(8, len(view_cols))]
        show_cols = st.multiselect("Columns to view", view_cols, default=default_cols, key="opt_filter_cols")
        dff = df[df["Score"] >= float(min_score)].copy() if "Score" in df.columns else df.copy()
        try:
            if show_cols:
                extra = ["Score"] if "Score" in dff.columns and "Score" not in show_cols else []
                st.dataframe(dff[show_cols + extra], width="stretch")
            if "Score" in df.columns and not df["Score"].isna().all():
                best_idx = int(df["Score"].idxmax())
                st.markdown("**Best Trial (by Score)**")
                st.json(make_json_safe(df.loc[best_idx].to_dict()))
        except Exception:
            st.dataframe(dff, width="stretch")

    # Summary
    with st.expander("Summary", expanded=False):
        try:
            rows_n, cols_n = int(df.shape[0]), int(df.shape[1])
            nan_pct = float(df.isna().mean().mean()) if rows_n and cols_n else 0.0
            st.write({"rows": rows_n, "cols": cols_n, "avg_nan_pct": round(nan_pct, 4)})
        except Exception:
            pass

    # Feature Importance (SHAP)
    with st.expander("Feature Importance (SHAP)", expanded=False):
        if compute_shap_importance_df is None:
            st.info("SHAP analysis unavailable: core.analysis_helpers.compute_shap_importance_df not found.")
        else:
            try:
                param_cols = [
                    c for c in df.columns if c not in ("Sharpe", "Profit", "Drawdown", "Score") and pd.api.types.is_numeric_dtype(df[c])
                ]
                if not param_cols:
                    st.info("No numeric parameter columns found for SHAP analysis.")
                else:
                    # Optional feature selection to stabilize SHAP
                    try:
                        if select_features is not None:
                            selected = select_features(df[param_cols], df["Score"])  # list or boolean mask
                            if isinstance(selected, list):
                                param_cols = [c for c in param_cols if c in set(selected)]
                            else:
                                param_cols = list(pd.Index(param_cols)[selected])
                            _log(f"Feature selection applied for SHAP: {len(param_cols)} features", "INFO")
                    except Exception as _e:
                        _log(f"Feature selection skipped: {_e}", "WARN")
                    imp = compute_shap_importance_df(df[param_cols], df["Score"])  # type: ignore[arg-type]
                    try:
                        st.bar_chart(imp.sort_values(ascending=False))
                    except Exception:
                        st.write(imp)
                    try:
                        topk = int(st.number_input("Top-K to export", min_value=5, max_value=50, value=20, step=1, key="opt_shap_topk"))
                        imp_sorted = imp.sort_values(ascending=False)
                        top_series = imp_sorted.head(topk)
                        shap_csv = top_series.reset_index(); shap_csv.columns = ["parameter", "importance"]
                        st.download_button(
                            "Download Top-K SHAP (CSV)",
                            data=shap_csv.to_csv(index=False).encode("utf-8"),
                            file_name="topk_shap.csv",
                            mime="text/csv",
                            key="opt_dl_shap_csv",
                        )
                    except Exception:
                        pass
            except Exception as e:
                st.info(f"SHAP failed: {e}")

    # Cluster View
    with st.expander("Cluster View (KMeans + PCA)", expanded=False):
        try:
            param_cols = [
                c for c in df.columns if c not in ("Sharpe", "Profit", "Drawdown", "Score") and pd.api.types.is_numeric_dtype(df[c])
            ]
            if len(param_cols) >= 2:
                # Prefer internal helpers if available
                use_internal = (compute_pca_transform is not None) and (compute_clusters is not None)
                if use_internal:
                    X_int = df[param_cols].select_dtypes(include=[np.number]).fillna(0.0)
                    labels = compute_clusters(X_int)
                    pca2 = compute_pca_transform(X_int, n_components=2)
                    dvis = pd.DataFrame({"pc1": pca2[:, 0], "pc2": pca2[:, 1], "cluster": labels})
                    import plotly.express as px
                    fig = px.scatter(dvis, x="pc1", y="pc2", color=pd.Series(labels).astype(str), title="Clusters (internal) + PCA(2)")
                    st.plotly_chart(fig, width="stretch")
                    cl_sel = st.selectbox("Cluster for range suggestion", sorted(set(pd.Series(labels).astype(int))), key="opt_cl_sel")
                    sugg = _suggest_ranges_from_cluster(df, param_cols, pd.Series(labels), int(cl_sel))
                else:
                    from sklearn.cluster import KMeans
                    from sklearn.decomposition import PCA
                    k = st.slider("Clusters (k)", 2, 10, 4, key="opt_cl_k")
                    X = df[param_cols].select_dtypes(include=[np.number]).fillna(0.0).values
                    km = KMeans(n_clusters=int(k), n_init="auto", random_state=42).fit(X)
                    pca2 = PCA(n_components=2, random_state=42).fit_transform(X)
                    dvis = pd.DataFrame({"pc1": pca2[:, 0], "pc2": pca2[:, 1], "cluster": km.labels_})
                    import plotly.express as px
                    fig = px.scatter(dvis, x="pc1", y="pc2", color=dvis["cluster"].astype(str), title="KMeans clusters (PCA 2D)")
                    st.plotly_chart(fig, width="stretch")
                    cl_sel = st.selectbox("Cluster for range suggestion", sorted(set(km.labels_)), key="opt_cl_sel")
                    sugg = _suggest_ranges_from_cluster(df, param_cols, pd.Series(km.labels_), int(cl_sel))
                if sugg:
                    st.json(make_json_safe(sugg))
                    if st.button("Use suggested ranges as active", key="opt_btn_use_sugg"):
                        st.session_state["active_ranges"] = sugg
                        st.success("Active ranges updated from cluster suggestion")
            else:
                st.caption("Need ≥2 numeric parameter columns for clustering.")
        except Exception as e:
            st.info(f"Cluster view unavailable: {e}")

    # KPIs
    try:
        c1, c2, c3, c4 = st.columns(4)
        if "Sharpe" in df.columns:
            c1.metric("Avg Sharpe", f"{pd.to_numeric(df['Sharpe'], errors='coerce').mean():.2f}")
        if "Score" in df.columns:
            c2.metric("Best Score", f"{pd.to_numeric(df['Score'], errors='coerce').max():.3f}")
        if "Profit" in df.columns:
            c3.metric("Max Profit", f"{pd.to_numeric(df['Profit'], errors='coerce').max():.0f}")
        if "Drawdown" in df.columns:
            c4.metric("Min Drawdown", f"{pd.to_numeric(df['Drawdown'], errors='coerce').min():.3f}")
    except Exception:
        pass

    # Top 20 & exports
    st.subheader("Top 20 Results")
    top = df.sort_values("Score", ascending=False).head(20)
    st.dataframe(top, width="stretch")
    try:
        ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        top_csv = top.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Top-20 (CSV)", data=top_csv, file_name=f"top20_{sym1}-{sym2}_{ts}.csv", mime="text/csv", key="opt_dl_top20_csv"
        )
        csv_all = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download ALL results (CSV)", data=csv_all, file_name="optimization_results.csv", mime="text/csv", key="opt_dl_all_csv"
        )
    except Exception:
        pass

    with st.expander("Use Top-K Quantile Ranges as Active", expanded=False):
        try:
            k = st.number_input("Top-K", min_value=5, max_value=200, value=50, step=5, key="opt_topk_k")
            param_cols = [c for c in df.columns if c not in ("Sharpe","Profit","Drawdown","Score") and pd.api.types.is_numeric_dtype(df[c])]
            if not param_cols:
                st.caption("No numeric parameter columns.")
            else:
                topk_df = df.sort_values("Score", ascending=False).head(int(k))
                sugg = _suggest_ranges_from_df(topk_df, param_cols)
                if sugg:
                    st.json(make_json_safe(sugg))
                    if st.button("Activate these ranges", key="opt_btn_use_topk"):
                        st.session_state["active_ranges"] = sugg
                        st.success("Active ranges updated from Top-K quantiles")
        except Exception as e:
            st.info(f"Top-K suggestion unavailable: {e}")

    # Heatmap
    with st.expander("Correlation Heatmap", expanded=False):
        try:
            import plotly.express as px
            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            subset = num_cols[:25]
            if subset:
                corr = df[subset].corr()
                fig_hm = px.imshow(corr, text_auto=False, title="Correlation Heatmap (subset)")
                st.plotly_chart(fig_hm, width="stretch")
            else:
                st.caption("No numeric columns to plot.")
        except Exception as e:
            st.info(f"Heatmap unavailable: {e}")

    # Contour surface
    with st.expander("Contour Surface (mean Score)", expanded=False):
        try:
            import plotly.express as px
            param_cols = [
                c for c in df.columns if c not in ("Sharpe","Profit","Drawdown","Score") and pd.api.types.is_numeric_dtype(df[c])
            ]
            if len(param_cols) < 2:
                st.caption("Need at least two numeric parameter columns.")
            else:
                x_param = st.selectbox("X Parameter", param_cols, key="opt_contour_x")
                y_param = st.selectbox("Y Parameter", param_cols, key="opt_contour_y")
                bins = st.slider("Grid bins", 10, 60, 25, key="opt_contour_bins")
                dfx = df[[x_param, y_param, "Score"]].dropna().copy()
                if len(dfx) > 0:
                    dfx["x_bin"] = pd.cut(dfx[x_param], bins=bins)
                    dfx["y_bin"] = pd.cut(dfx[y_param], bins=bins)
                    pivot = dfx.groupby(["x_bin", "y_bin"])['Score'].mean().unstack()
                    xs = [interval.mid for interval in pivot.index.categories] if hasattr(pivot.index, 'categories') else list(range(len(pivot.index)))
                    ys = [interval.mid for interval in pivot.columns.categories] if hasattr(pivot.columns, 'categories') else list(range(len(pivot.columns)))
                    fig_cont = px.imshow(
                        pivot.values,
                        x=ys,
                        y=xs,
                        origin='lower',
                        labels={'x': y_param, 'y': x_param, 'color': 'Mean Score'},
                        title=f"Score Surface: {x_param} vs {y_param}",
                    )
                    st.plotly_chart(fig_cont, width="stretch")
                    try:
                        pivot_n = dfx.groupby(["x_bin", "y_bin"])['Score'].size().unstack()
                        total_pts = int(len(dfx))
                        mean_per_cell = float(pivot_n.values.mean()) if pivot_n.size > 0 else 0.0
                        st.caption(f"Points: {total_pts} | Avg points per cell: {mean_per_cell:.1f}")
                    except Exception:
                        pass
                else:
                    st.caption("Not enough data for contour surface.")
        except Exception as e:
            st.info(f"Contour surface unavailable: {e}")

    # Pareto
    if {"Profit", "Drawdown", "Sharpe", "Score"}.issubset(df.columns):
        try:
            import plotly.express as px
            fig_pf = px.scatter(
                df,
                x="Profit",
                y="Drawdown",
                size="Sharpe",
                color="Score",
                hover_data=[c for c in df.columns if c not in ("Profit", "Drawdown", "Sharpe", "Score")][:10],
                title="Pareto Front",
            )
            st.plotly_chart(fig_pf, width="stretch")
        except Exception as e:
            st.warning(f"Pareto plot unavailable: {e}")

    # SHAP
    with st.expander("Feature Importance (SHAP)", expanded=False):
        if compute_shap_importance_df is None:
            st.info("SHAP analysis unavailable: core.analysis_helpers.compute_shap_importance_df not found.")
        else:
            try:
                param_cols = [
                    c for c in df.columns if c not in ("Sharpe", "Profit", "Drawdown", "Score") and pd.api.types.is_numeric_dtype(df[c])
                ]
                if not param_cols:
                    st.info("No numeric parameter columns found for SHAP analysis.")
                else:
                    imp = compute_shap_importance_df(df[param_cols], df["Score"])  # type: ignore[arg-type]
                    try:
                        st.bar_chart(imp.sort_values(ascending=False))
                    except Exception:
                        st.write(imp)
                    try:
                        topk = int(st.number_input("Top-K to export", min_value=5, max_value=50, value=20, step=1, key="opt_shap_topk"))
                        imp_sorted = imp.sort_values(ascending=False)
                        top_series = imp_sorted.head(topk)
                        shap_csv = top_series.reset_index(); shap_csv.columns = ["parameter", "importance"]
                        st.download_button(
                            "Download Top-K SHAP (CSV)",
                            data=shap_csv.to_csv(index=False).encode("utf-8"),
                            file_name="topk_shap.csv",
                            mime="text/csv",
                            key="opt_dl_shap_csv",
                        )
                    except Exception:
                        pass
            except Exception as e:
                st.info(f"SHAP failed: {e}")

    # Distributions
    with st.expander("Distributions", expanded=False):
        try:
            if "Sharpe" in df.columns:
                st.markdown("**Sharpe**"); st.bar_chart(df[["Sharpe"]])
            if "Score" in df.columns:
                st.markdown("**Score**"); st.bar_chart(df[["Score"]])
        except Exception:
            pass

    # Cluster view
    with st.expander("Cluster View (KMeans + PCA)", expanded=False):
        try:
            from sklearn.cluster import KMeans
            from sklearn.decomposition import PCA
            param_cols = [
                c for c in df.columns if c not in ("Sharpe", "Profit", "Drawdown", "Score") and pd.api.types.is_numeric_dtype(df[c])
            ]
            if len(param_cols) >= 2:
                k = st.slider("Clusters (k)", 2, 10, 4, key="opt_cl_k")
                X = df[param_cols].select_dtypes(include=[np.number]).fillna(0.0).values
                km = KMeans(n_clusters=int(k), n_init="auto", random_state=42).fit(X)
                pca2 = PCA(n_components=2, random_state=42).fit_transform(X)
                dvis = pd.DataFrame({"pc1": pca2[:, 0], "pc2": pca2[:, 1], "cluster": km.labels_})
                import plotly.express as px
                fig = px.scatter(dvis, x="pc1", y="pc2", color=dvis["cluster"].astype(str), title="KMeans clusters (PCA 2D)")
                st.plotly_chart(fig, width="stretch")
                # Suggest ranges for a selected cluster
                cl_sel = st.selectbox("Cluster for range suggestion", sorted(set(km.labels_)), key="opt_cl_sel")
                sugg = _suggest_ranges_from_cluster(df, param_cols, pd.Series(km.labels_), int(cl_sel))
                if sugg:
                    st.json(make_json_safe(sugg))
                    if st.button("Use suggested ranges as active", key="opt_btn_use_sugg"):
                        st.session_state["active_ranges"] = sugg
                        st.success("Active ranges updated from cluster suggestion")
            else:
                st.caption("Need ≥2 numeric parameter columns for clustering.")
        except Exception as e:
            st.info(f"Cluster view unavailable: {e}")

    st.caption("End of Optimization Tab UI.")

    # Logs panel
    with st.expander("Logs", expanded=False):
        st.code(_get_logs() or "(no logs yet)", language="text")

# =========================
# SECTION 4.1: DuckDB Schema & IO Helpers
# =========================

