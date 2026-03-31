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

import duckdb
import numpy as np
import pandas as pd
import streamlit as st

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
    2) אחרת נעדיף %LOCALAPPDATA%/pairs_trading_system/cache.duckdb (לא מסונכרן ע\"י OneDrive).
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
    # fallback
    p = PROJECT_ROOT / "cache.duckdb"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

DB_PATH = _default_db_path()

# ===== DEBUG: מי הקובץ ומה הנתיב?
print("OPT TAB LOADED FROM:", __file__)
print("USING DB PATH:", DB_PATH)
if "OneDrive" in str(DB_PATH):
    print("[WARN] DB path appears under OneDrive — consider OPT_CACHE_PATH or LOCALAPPDATA.")
try:
    if st.sidebar:
        with st.sidebar.expander("⚙ Debug: Optimization Tab"):
            st.write("**Loaded from:**", __file__)
            st.write("**DB Path:**", str(DB_PATH))
except Exception:
    pass

def _open_duck(db_path: Path, retries: int = 3, delay: float = 0.5):
    import time, atexit
    # Use module-level duckdb import (avoids unresolved local import in some linters/environments)
    global duckdb
    last_err = None
    for i in range(retries):
        try:
            conn = duckdb.connect(str(db_path))  # writeable
            conn.execute("PRAGMA threads=4")
            atexit.register(conn.close)
            return conn
        except duckdb.IOException as e:
            last_err = e
            if "File is already open" in str(e) and i < retries - 1:
                time.sleep(delay)
                continue
            break
    try:
        mem = duckdb.connect(":memory:")
        mem.execute("PRAGMA threads=4")
        atexit.register(mem.close)
        return mem
    except Exception:
        if last_err:
            raise last_err
        raise duckdb.IOException("Failed to open DuckDB connection and fallback.")

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

def get_ro_duck():
    import duckdb, atexit
    try:
        conn = duckdb.connect(str(DB_PATH), read_only=True)
        conn.execute("PRAGMA threads=4")
        atexit.register(conn.close)
        return conn
    except duckdb.IOException:
        mem = duckdb.connect(":memory:")
        mem.execute("PRAGMA threads=4")
        atexit.register(mem.close)
        return mem

class _DuckProxy:
    _conn = None
    def _ensure(self):
        if self._conn is None:
            self._conn = get_duck()
    def __getattr__(self, name):
        self._ensure()
        return getattr(self._conn, name)

DUCK = _DuckProxy()

# =========================
# SECTION 5: Domain Models
# =========================
class PairConfig(BaseModel):
    symbols: List[str]

    @model_validator(mode="after")
    def check_two_symbols(self) -> "PairConfig":
        if not isinstance(self.symbols, list) or len(self.symbols) != 2:
            raise ValueError("Each pair must contain exactly two symbols")
        return self


# =========================
# SECTION 6: Data Access Helpers (minimal)
# =========================
_pairs_cache: Optional[List[Dict[str, Any]]] = None

def load_pairs(refresh: bool = False) -> List[Dict[str, Any]]:
    global _pairs_cache
    if _pairs_cache is not None and not refresh:
        return _pairs_cache

    pairs_path_json = SETTINGS.data_dir / "pairs.json"
    pairs_path_yaml = SETTINGS.data_dir / "pairs.yml"

    pairs: List[Dict[str, Any]] = []
    try:
        if pairs_path_json.exists():
            pairs = json.loads(pairs_path_json.read_text(encoding="utf-8"))
        elif pairs_path_yaml.exists():
            import yaml  # deferred
            pairs = yaml.safe_load(pairs_path_yaml.read_text(encoding="utf-8")) or []
        else:
            pairs = [  # demo defaults
                {"symbols": ["AAPL", "MSFT"]},
                {"symbols": ["GOOG", "META"]},
            ]
    except Exception as e:
        logger.warning("Failed loading pairs; using demo. %s", e)
        pairs = [{"symbols": ["AAPL", "MSFT"]}]

    # validate (soft)
    ok: List[Dict[str, Any]] = []
    for p in pairs:
        try:
            PairConfig(**p)
            ok.append(p)
        except Exception as ve:
            logger.warning("Skipping invalid pair %s: %s", p, ve)
    _pairs_cache = ok
    return ok


# =========================
# SECTION 7: Parameters & Backtest API (stubs)
# =========================
ParamRange = Tuple[float, float, float]

def get_default_param_ranges() -> Dict[str, ParamRange]:
    """Router-only: require ranges from core.ranges or core.params; raise if unavailable."""
    if core_get_default_ranges is not None:
        r = core_get_default_ranges()
        out: Dict[str, ParamRange] = {}
        for k, v in r.items():
            lo, hi = float(v[0]), float(v[1])
            step = float(v[2]) if len(v) > 2 else 0.0
            out[str(k)] = (lo, hi, step)
        return out
    if CORE_PARAM_SPECS is not None:
        out: Dict[str, ParamRange] = {}
        for spec in CORE_PARAM_SPECS:
            name, lo, hi, step = spec[0], float(spec[1]), float(spec[2]), float(spec[3])
            out[str(name)] = (lo, hi, step)
        return out
    raise RuntimeError("Parameter ranges unavailable: provide core.ranges.get_default_ranges or core.params.PARAM_SPECS")


def _result_to_dict(result: Any) -> Dict[str, float]:
    """Convert backtester result to canonical metrics dict with key mapping.
    Accepts dict/object; supports common aliases → {Sharpe, Profit, Drawdown}.
    """
    def canonicalize(d: Dict[str, Any]) -> Dict[str, float]:
        # build lower-case map
        lower = {str(k).lower(): v for k, v in d.items()}
        out: Dict[str, float] = {}
        # alias map
        aliases = {
            "sharpe": "Sharpe",
            "sharpe_ratio": "Sharpe",
            "sr": "Sharpe",
            "profit": "Profit",
            "pnl": "Profit",
            "total_pnl": "Profit",
            "net_pnl": "Profit",
            "drawdown": "Drawdown",
            "max_drawdown": "Drawdown",
            "max_dd": "Drawdown",
            "dd": "Drawdown",
        }
        # try direct canonical first
        for k in ("Sharpe", "Profit", "Drawdown"):
            if k in d:
                try:
                    out[k] = float(d[k])
                except Exception:
                    pass
        # fill from aliases
        for lk, canon in aliases.items():
            if canon not in out and lk in lower:
                try:
                    out[canon] = float(lower[lk])
                except Exception:
                    pass
        return out

    # direct dict
    if isinstance(result, dict):
        return canonicalize(result)
    # object with to-dict methods
    for meth in ("model_dump", "to_dict", "dict"):
        if hasattr(result, meth):
            try:
                d = getattr(result, meth)()
                if isinstance(d, dict):
                    return canonicalize(d)
            except Exception:
                pass
    # attribute fallback
    attr_map = {
        "Sharpe": ["Sharpe", "sharpe", "sharpe_ratio", "sr"],
        "Profit": ["Profit", "profit", "pnl", "total_pnl", "net_pnl"],
        "Drawdown": ["Drawdown", "drawdown", "max_drawdown", "max_dd", "dd"],
    }
    out: Dict[str, float] = {}
    for canon, names in attr_map.items():
        for name in names:
            if hasattr(result, name):
                try:
                    out[canon] = float(getattr(result, name))
                    break
                except Exception:
                    continue
    return out


def run_backtest(sym1: str, sym2: str, params: Dict[str, Any]) -> Dict[str, float]:
    """Router-only: delegate to core Backtester; no internal fallback."""
    if Backtester is None:
        raise RuntimeError("Backtester unavailable: please provide core.optimization_backtester.Backtester")
    bt = Backtester(sym1, sym2, **params)
    res = bt.run()
    perf = _result_to_dict(res)
    if not perf:
        raise RuntimeError("Backtester returned empty/unsupported result")
    return perf


# =========================
# SECTION 8: Streamlit UI — Optimiser Tab (initial minimal)
# =========================

def _render_profile_sidebar() -> Optional[OptSettings]:
    """Load/preview a JSON/YAML profile and export current settings. Returns new settings if applied."""
    st.sidebar.subheader("Profile")
    up = st.sidebar.file_uploader("Upload profile (JSON/YAML)", type=["json","yml","yaml"])
    loaded: Optional[Dict[str, Any]] = None
    if up is not None:
        try:
            content = up.read()
            if up.name.endswith((".yml",".yaml")):
                import yaml  # deferred
                loaded = yaml.safe_load(content) or {}
            else:
                loaded = json.loads(content.decode("utf-8"))
            st.sidebar.json(loaded)
        except Exception as e:
            st.sidebar.error(f"Failed to parse profile: {e}")
    apply = st.sidebar.button("Apply Profile", disabled=loaded is None)
    if apply and loaded is not None:
        try:
            return OptSettings(**loaded)
        except Exception as e:
            st.sidebar.error(f"Invalid profile: {e}")

    # Export current profile (as JSON+YAML)
    export_dict = SETTINGS.model_dump()
    export_json = json.dumps(export_dict, indent=2, ensure_ascii=False)
    try:
        import yaml  # deferred
        export_yaml = yaml.safe_dump(export_dict, sort_keys=False, allow_unicode=True)
    except Exception:
        export_yaml = None
    st.sidebar.download_button("Export Profile (JSON)", data=export_json, file_name="profile.json", mime="application/json")
    if export_yaml is not None:
        st.sidebar.download_button("Export Profile (YAML)", data=export_yaml, file_name="profile.yaml", mime="application/x-yaml")
    return None


def _recommend_params(df: pd.DataFrame, sel_params: List[str], n_top: int = 20) -> List[Tuple[str, float]]:
    """RandomForest importances over top-N rows. Returns list[(param, importance)]."""
    try:
        from sklearn.ensemble import RandomForestRegressor
    except Exception:
        return []
    if df.empty:
        return []
    top = df.sort_values("Score", ascending=False).head(int(n_top)).copy()
    cols = [c for c in sel_params if c in top.columns]
    if not cols:
        return []
    X = top[cols]
    y = top["Score"]
    try:
        rf = RandomForestRegressor(random_state=42)
        rf.fit(X, y)
        imps = {c: float(v) for c, v in zip(cols, rf.feature_importances_)}
        return sorted(imps.items(), key=lambda kv: kv[1], reverse=True)
    except Exception:
        return []


def _export_top(df: pd.DataFrame, filename_stub: str = "top20") -> None:
    """Offer CSV/Excel/Markdown export buttons for a DataFrame."""
    if df.empty:
        return
    # CSV
    csv = df.to_csv(index=False)
    st.download_button("⬇️ CSV", csv, file_name=f"{filename_stub}.csv")
    # Excel
    from io import BytesIO
    buf = BytesIO()
    try:
        df.to_excel(buf, index=False)
        buf.seek(0)
        st.download_button(
            "⬇️ Excel",
            buf.getvalue(),
            file_name=f"{filename_stub}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    except Exception:
        pass
    # Markdown
    md_lines = [
        "|" + "|".join(map(str, df.columns)) + "|",
        "|" + "|".join(["---"] * len(df.columns)) + "|",
    ]
    for row in df.values.tolist():
        md_lines.append("|" + "|".join(map(str, row)) + "|")
    md = "\n".join(md_lines)
    st.download_button("⬇️ Markdown", md, file_name=f"{filename_stub}.md")


def _sidebar_common() -> Dict[str, Any]:
    st.sidebar.subheader("Metric Weights")
    w_sharpe = st.sidebar.slider("Sharpe", 0.0, 1.0, 0.5, 0.05)
    w_profit = st.sidebar.slider("Profit", 0.0, 1.0, 0.5, 0.05)
    w_dd = st.sidebar.slider("Drawdown", 0.0, 1.0, 0.5, 0.05)

    st.sidebar.subheader("Hyperparameter Ranges")
    ranges = get_default_param_ranges()
    edited: Dict[str, ParamRange] = {}
    for k, (lo, hi, step) in ranges.items():
        ui_step = float(step) if (step is not None and float(step) > 0) else max((float(hi) - float(lo)) / 100.0, 0.001)
        lo_n = st.sidebar.number_input(f"{k} min", value=float(lo), step=float(ui_step))
        hi_n = st.sidebar.number_input(f"{k} max", value=float(hi), step=float(ui_step))
        edited[k] = (float(lo_n), float(hi_n), float(ui_step))

    return {
        "weights": {"Sharpe": w_sharpe, "Profit": w_profit, "Drawdown": w_dd},
        "ranges": edited,
    }



def _optuna_optimize(sym1: str, sym2: str, ranges: Dict[str, ParamRange], weights: Dict[str, float], n_trials: int, timeout: int, sampler_name: str = "TPE") -> pd.DataFrame:
    """Run Optuna optimization (strict router). Returns DataFrame of completed trials with params+perf+Score."""
    if optuna is None or isinstance(TPESampler, object):
        raise RuntimeError("Optuna unavailable: install optuna to run optimization")
    if get_param_distributions is None:
        raise RuntimeError("Distributions unavailable: provide core.distributions.get_param_distributions")
    if normalize_metrics is None or compute_weighted_score is None:
        raise RuntimeError("Metrics unavailable: provide core.metrics.normalize_metrics & compute_weighted_score")

    # Build distributions
    dists = get_param_distributions(ranges)
    from optuna.distributions import FloatDistribution, IntDistribution

    sampler = TPESampler() if sampler_name.upper().startswith("TPE") else CmaEsSampler()

    def suggest_param(trial: "optuna.Trial", name: str, dist: Any) -> float:
        if isinstance(dist, IntDistribution):
            return float(trial.suggest_int(name, int(dist.low), int(dist.high), step=int(dist.step) if dist.step else 1))
        if isinstance(dist, FloatDistribution):
            if dist.step:
                return float(trial.suggest_float(name, float(dist.low), float(dist.high), step=float(dist.step)))
            return float(trial.suggest_float(name, float(dist.low), float(dist.high)))
        # tuple fallback
        if isinstance(dist, tuple) and len(dist) >= 2:
            lo, hi = float(dist[0]), float(dist[1])
            step = float(dist[2]) if len(dist) > 2 else None
            if step and step > 0:
                return float(trial.suggest_float(name, lo, hi, step=step))
            return float(trial.suggest_float(name, lo, hi))
        return float(trial.suggest_float(name, 0.0, 1.0))

    def objective(trial: "optuna.Trial") -> float:
        params: Dict[str, Any] = {k: suggest_param(trial, k, d) for k, d in dists.items()}
        params = apply_risk_parity(params) if apply_risk_parity is not None else params
        perf = run_backtest(sym1, sym2, params)
        norm = normalize_metrics(perf)
        score = compute_weighted_score(norm, weights)
        trial.set_user_attr("perf", perf)
        return float(score)

    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=MedianPruner())
    study.optimize(objective, n_trials=int(n_trials), timeout=int(timeout) if timeout else None)

    recs: List[Dict[str, Any]] = []
    for t in study.trials:
        if t.state.name == "COMPLETE":
            perf = t.user_attrs.get("perf", {})
            recs.append({**t.params, **perf, "Score": t.value})
    return pd.DataFrame(recs)


def render_optimization_tab() -> None:
    st.title("⚙️ Pairs‑Trading Optimiser — Skeleton")

    # Sidebar profile
    new_settings = _render_profile_sidebar()
    if new_settings is not None:
        global SETTINGS
        SETTINGS = new_settings
        logger.info("Profile applied via sidebar → %s", SETTINGS.model_dump())

    section = st.sidebar.radio("בחר אזור:", ["Optimiser", "Visual Dashboard"], index=0)

    pairs = load_pairs()
    if not pairs:
        st.error("No pairs configured")
        return

    pair_names = [f"{p['symbols'][0]}-{p['symbols'][1]}" for p in pairs]
    selected = st.selectbox("Select Pair", pair_names)
    sym1, sym2 = selected.split("-")

    sidebar_vals = _sidebar_common()

    st.subheader("Run")

    # Health Check (router essentials)
    essentials = {
        "Backtester": Backtester is not None,
        "optuna": (optuna is not None) and (TPESampler is not None) and (CmaEsSampler is not None) and (MedianPruner is not None),
        "get_param_distributions": get_param_distributions is not None,
        "metrics (normalize & score)": (normalize_metrics is not None and compute_weighted_score is not None),
    }
    try:
        _ = get_default_param_ranges()
        essentials["param ranges"] = True
    except Exception:
        essentials["param ranges"] = False

    ok = all(essentials.values())
    with st.expander("System Health", expanded=not ok):
        for k, v in essentials.items():
            st.write(("✅" if v else "❌"), k)
        # Show ranges source info
        rng_src = "core.ranges.get_default_ranges" if core_get_default_ranges is not None else ("core.params.PARAM_SPECS" if CORE_PARAM_SPECS is not None else "—")
        st.write("**Ranges source:**", rng_src)
        if not ok:
            st.warning("Fix the red items above to enable Run Optimization.")

    # Optuna controls
    sampler = st.selectbox("Sampler", ["TPE", "CMA-ES"], index=0)
    timeout = st.number_input("Timeout (sec)", min_value=0, max_value=7200, value=600, step=30)
    n_trials = st.number_input("Trials", min_value=10, max_value=5000, value=300, step=10)

    # Configuration snapshot & sanity check
    with st.expander("Configuration Snapshot", expanded=False):
        ranges = sidebar_vals["ranges"]
        weights = sidebar_vals["weights"]
        st.write("**Pair:**", f"{sym1}-{sym2}")
        st.write("**Sampler:**", sampler, "| **Trials:**", int(n_trials), "| **Timeout:**", int(timeout), "sec")
        st.write("**Param count:**", len(ranges))
        st.write("**Weights:**", weights)
        # show first 8 ranges
        if ranges:
            preview_items = list(ranges.items())[:8]
            st.json({k: v for k, v in preview_items})

    if st.button("Run Sanity Check (1 backtest)"):
        try:
            ranges = sidebar_vals["ranges"]
            mid_params = {k: (lo + hi) / 2.0 for k, (lo, hi, _step) in ranges.items()}
            perf = run_backtest(sym1, sym2, mid_params)
            st.success("Sanity backtest ran successfully")
            st.json({"params": mid_params, "perf": perf})
        except Exception as e:
            st.error(f"Sanity check failed: {e}")

    # Quick Optuna Smoke (few trials)
    if st.button("Quick Optuna Smoke (3 trials)", disabled=not ok):
        try:
            ranges = sidebar_vals["ranges"]
            weights = sidebar_vals["weights"]
            df_smoke = _optuna_optimize(sym1, sym2, ranges, weights, n_trials=3, timeout=min(int(timeout), 60), sampler_name=sampler)
            st.success("Optuna smoke ran successfully")
            st.dataframe(df_smoke.sort_values("Score", ascending=False).head(5), width="stretch")
        except Exception as e:
            st.error(f"Optuna smoke failed: {e}")

    if st.button("Run Optimization", disabled=not ok):
        try:
            ranges = sidebar_vals["ranges"]
            weights = sidebar_vals["weights"]
            df = _optuna_optimize(sym1, sym2, ranges, weights, int(n_trials), int(timeout), sampler_name=sampler)
            st.session_state["opt_df"] = df
            st.success("Optimization finished")
            if not df.empty and summarize_results is not None:
                with st.expander("Summary (post-optimization)", expanded=False):
                    summary = summarize_results(df)
                    st.dataframe(summary, width="stretch") if isinstance(summary, pd.DataFrame) else st.write(summary)
        except Exception as e:
            st.error(f"Optimization failed: {e}")

    # Meta-Optimization UI (optional)
    if section == "Optimiser" and meta_optimization_sampling is not None:
        with st.expander("Meta-Optimization", expanded=False):
            st.caption("Refine parameter ranges via outer/inner optimisation rounds")
            n_outer = st.number_input("Outer rounds", min_value=1, value=3, step=1, key="meta_outer")
            n_inner = st.number_input("Inner trials per round", min_value=10, value=50, step=10, key="meta_inner")
            sampler_meta = st.selectbox("Sampler", ["TPE", "CMA-ES"], index=0, key="meta_sampler")
            shrink = st.slider("Shrink factor", 0.05, 0.5, 0.2, 0.05, key="meta_shrink")
            if st.button("Run Meta-Optimization", key="run_meta_opt"):
                try:
                    base = get_default_param_ranges()
                    ws = sidebar_vals.get("weights", {"Sharpe":0.5,"Profit":0.3,"Drawdown":0.2})
                    result = meta_optimization_sampling(
                        base_ranges=base,
                        pair=(sym1, sym2),
                        n_outer=int(n_outer),
                        n_inner=int(n_inner),
                        sampler_name=sampler_meta,
                        weights=ws,
                        shrink_factor=float(shrink),
                    )
                    st.json(result)
                except Exception as e:
                    st.warning(f"Meta-optimization failed: {e}")

    # Utility: data/feature preview
    def _preview_data_features(sym1: str, sym2: str) -> None:
        with st.expander("Preview Data / Features", expanded=False):
            if common_load_prices is None:
                st.info("common.data_loader.load_prices not available")
                return
            try:
                s1 = common_load_prices(sym1)
                s2 = common_load_prices(sym2)
                st.markdown("**Raw Prices (head)**")
                st.dataframe(s1.head(10), width="stretch")
                st.dataframe(s2.head(10), width="stretch")
                # optional resample
                if to_timeframe is not None:
                    try:
                        s1 = to_timeframe(s1, "1D")
                        s2 = to_timeframe(s2, "1D")
                    except Exception:
                        pass
                # simple pair DF
                try:
                    df_pair = pd.DataFrame({f"{sym1}_close": s1["Close"], f"{sym2}_close": s2["Close"]}).dropna()
                except Exception:
                    # fallback to any last column names
                    c1 = s1.columns[-1]
                    c2 = s2.columns[-1]
                    df_pair = pd.DataFrame({f"{sym1}_{c1}": s1[c1], f"{sym2}_{c2}": s2[c2]}).dropna()
                if build_features is not None:
                    try:
                        feats = build_features(df_pair)
                        st.markdown("**Engineered Features (head)**")
                        st.dataframe(feats.head(10), width="stretch")
                    except Exception as e:
                        st.info(f"Feature build failed: {e}")
                # ADF test on simple spread
                try:
                    spread = df_pair.iloc[:,0] - df_pair.iloc[:,1]
                    if winsorize is not None:
                        try:
                            spread = winsorize(spread, limits=(0.01,0.01))
                        except Exception:
                            pass
                    if adf_test is not None:
                        res = adf_test(spread)
                        st.markdown("**ADF Test on spread**")
                        st.write(res)
                except Exception:
                    pass
            except Exception as e:
                st.warning(f"Preview failed: {e}")

    # ----- Visual Dashboard -----
    if section == "Visual Dashboard":
        # Optional quick actions in sidebar
        with st.sidebar:
            # Save Results to studies/
            if st.button("Save Results to studies/"):
                try:
                    df_current = st.session_state.get("opt_df", pd.DataFrame())
                    if df_current.empty:
                        st.warning("No results to save")
                    else:
                        SETTINGS.studies_dir.mkdir(parents=True, exist_ok=True)
                        ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                        out = SETTINGS.studies_dir / f"study_{ts}.csv"
                        df_current.to_csv(out, index=False)
                        st.success(f"Saved: {out.name}")
                except Exception as e:
                    st.error(f"Save failed: {e}")

            # Load Results from CSV
            up_csv = st.file_uploader("Load results CSV", type=["csv"], key="load_results_csv")
            if up_csv is not None:
                try:
                    df_loaded = pd.read_csv(up_csv)
                    st.session_state["opt_df"] = df_loaded
                    st.success("Results loaded into session")
                except Exception as e:
                    st.error(f"Load failed: {e}")

            # Recommend Pairs (if available)
            if recommend_pairs is not None and st.button("Recommend Pairs"):
                try:
                    rec = recommend_pairs()  # depends on your implementation signature
                    st.markdown("**Recommended Pairs:**")
                    if isinstance(rec, pd.DataFrame):
                        st.dataframe(rec.head(20), width="stretch")
                    else:
                        st.write(rec)
                except Exception as e:
                    st.warning(f"Pair recommendation unavailable: {e}")

        df = st.session_state.get("opt_df", pd.DataFrame())
        if df.empty:
            st.info("Run an optimization first.")
        else:
            st.subheader("Top 20 Results")
            top = df.sort_values("Score", ascending=False).head(20)
            st.dataframe(top, width="stretch")
            _export_top(top, filename_stub="top20")

            # Correlation Heatmap (parameters + Score)
            try:
                import plotly.express as px
                num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
                # avoid overly wide heatmap; cap to first 25 columns for performance
                subset = num_cols[:25]
                if subset:
                    corr = df[subset].corr()
                    fig_hm = px.imshow(corr, text_auto=False, title="Correlation Heatmap (subset)")
                    st.plotly_chart(fig_hm, width="stretch")
            except Exception as e:
                st.info(f"Heatmap unavailable: {e}")

            # Contour Surface: pick two params to visualize mean Score
            with st.expander("Contour Surface (mean Score)", expanded=False):
                param_cols = [c for c in df.columns if c not in ("Sharpe","Profit","Drawdown","Score") and pd.api.types.is_numeric_dtype(df[c])]
                if len(param_cols) < 2:
                    st.info("Need at least two numeric parameters.")
                else:
                    x_param = st.selectbox("X Parameter", param_cols, key="contour_x")
                    y_param = st.selectbox("Y Parameter", param_cols, key="contour_y")
                    n_grid = st.slider("Grid bins", 10, 60, 25, key="contour_bins")
                    dfx = df[[x_param, y_param, "Score"]].dropna().copy()
                    try:
                        dfx["x_bin"] = pd.cut(dfx[x_param], bins=n_grid)
                        dfx["y_bin"] = pd.cut(dfx[y_param], bins=n_grid)
                        pivot = dfx.groupby(["x_bin","y_bin"])['Score'].mean().unstack()
                        xs = [interval.mid for interval in pivot.index.categories]
                        ys = [interval.mid for interval in pivot.columns.categories]
                        fig_cont = px.imshow(pivot.values, x=ys, y=xs, origin='lower', labels={'x': y_param, 'y': x_param, 'color': 'Mean Score'}, title=f"Score Surface: {x_param} vs {y_param}")
                        st.plotly_chart(fig_cont, width="stretch")
                    except Exception as e:
                        st.info(f"Contour plot unavailable: {e}")

            # Pareto Front (Profit vs Drawdown, size=Sharpe, color=Score)
            if {"Profit", "Drawdown", "Sharpe", "Score"}.issubset(df.columns):
                try:
                    import plotly.express as px
                    fig_pf = px.scatter(
                        df,
                        x="Profit",
                        y="Drawdown",
                        size="Sharpe",
                        color="Score",
                        title="Pareto Front",
                    )
                    st.plotly_chart(fig_pf, width="stretch")
                except Exception as e:
                    st.warning(f"Pareto plot unavailable: {e}")

            # Sharpe Distribution
            if "Sharpe" in df.columns:
                st.subheader("Sharpe Distribution")
                try:
                    st.bar_chart(df["Sharpe"])  # simple histogram alternative
                except Exception:
                    st.dataframe(df[["Sharpe"]].describe())

            # Feature Importance (SHAP)
            with st.expander("Feature Importance (SHAP)", expanded=False):
                if compute_shap_importance_df is None:
                    st.info("SHAP analysis unavailable: core.analysis_helpers.compute_shap_importance_df not found.")
                else:
                    try:
                        # candidate parameter columns (exclude performance metrics)
                        sel_params = [c for c in df.columns if c not in ("Sharpe","Profit","Drawdown","Score") and pd.api.types.is_numeric_dtype(df[c])]
                        if not sel_params:
                            st.info("No numeric parameter columns found for SHAP analysis.")
                        else:
                            # Compute importance series (expected: pd.Series indexed by param)
                            imp = compute_shap_importance_df(df[sel_params], df["Score"])  # type: ignore[arg-type]
                            # Show bar chart of importances
                            try:
                                st.bar_chart(imp.sort_values(ascending=False))
                            except Exception:
                                st.write(imp)

                            # Build Top-K SHAP table with distribution hints
                            top_k = st.slider("Top-K features", 5, min(30, len(sel_params)), min(20, len(sel_params)))
                            # sort desc and select
                            imp_sorted = imp.sort_values(ascending=False)
                            top_names = list(imp_sorted.head(top_k).index)

                            # percentiles over full results
                            def pct(s: pd.Series, q: float) -> float:
                                try:
                                    return float(np.nanpercentile(s.to_numpy(dtype=float, copy=False), q))
                                except Exception:
                                    return float("nan")

                            # also compute on high-Score slice (top decile)
                            try:
                                thresh = np.nanpercentile(df["Score"].to_numpy(dtype=float, copy=False), 90.0)
                                df_high = df[df["Score"] >= thresh]
                            except Exception:
                                df_high = pd.DataFrame()

                            rows = []
                            for name in top_names:
                                col = df[name]
                                p10 = pct(col, 10)
                                p50 = pct(col, 50)
                                p90 = pct(col, 90)
                                if not df_high.empty:
                                    p10_h = pct(df_high[name], 10)
                                    p50_h = pct(df_high[name], 50)
                                    p90_h = pct(df_high[name], 90)
                                else:
                                    p10_h = p50_h = p90_h = float("nan")
                                rows.append({
                                    "feature": name,
                                    "importance": float(imp_sorted.loc[name]),
                                    "p10": p10, "p50": p50, "p90": p90,
                                    "p10_top": p10_h, "p50_top": p50_h, "p90_top": p90_h,
                                })
                            df_shap_table = pd.DataFrame(rows)
                            st.subheader("Top-K SHAP features")
                            st.dataframe(df_shap_table, width="stretch")
                            _export_top(df_shap_table, filename_stub="top_shap_features")

                            # Optional: Suggested ranges from high-score slice
                            with st.expander("Suggested Ranges (from high-score slice)", expanded=False):
                                if df_high.empty:
                                    st.info("Not enough high-score samples to derive suggested ranges.")
                                else:
                                    # propose using [p10_top, p90_top] as refined range hints
                                    df_ranges = df_shap_table[["feature","p10_top","p50_top","p90_top"]].rename(columns={"p10_top":"suggest_min","p90_top":"suggest_max","p50_top":"median_top"})
                                    st.dataframe(df_ranges, width="stretch")
                                    _export_top(df_ranges, filename_stub="suggested_ranges_from_shap")
                    except Exception as e:
                        st.info(f"SHAP importance unavailable: {e}")

            # ML Analysis
            with st.expander("ML Analysis", expanded=False):
                if render_ml_analysis is None:
                    st.info("ML analysis unavailable: core.ml_analysis.render_ml_analysis not found.")
                else:
                    try:
                        sel_params = [c for c in df.columns if c not in ("Sharpe","Profit","Drawdown","Score")]
                        render_ml_analysis(df, sel_params)  # type: ignore[misc]
                    except Exception as e:
                        st.info(f"ML analysis failed: {e}")

            # Recommendations (sidebar button)
            with st.sidebar:
                if st.button("Recommend Params"):
                    sel_params = [c for c in df.columns if c not in ("Sharpe","Profit","Drawdown","Score")]
                    recs = _recommend_params(df, sel_params)
                    if not recs:
                        st.warning("Recommendations unavailable (need sklearn & non-empty results)")
                    else:
                        st.markdown("**Parameter Importances (RF):**")
                        for name, imp in recs[:15]:
                            st.write(f"{name}: {imp:.3f}")
            
            # Optional: summarize results using core.analytics if available
            if summarize_results is not None:
                try:
                    st.subheader("Summary")
                    summary = summarize_results(df)
                    if isinstance(summary, pd.DataFrame):
                        st.dataframe(summary, width="stretch")
                    else:
                        st.write(summary)
                except Exception as e:
                    st.info(f"Summary unavailable: {e}")

            # AutoML Insights (if available)
            if run_automl_summary is not None:
                with st.expander("AutoML Insights", expanded=False):
                    try:
                        sel_params = [c for c in df.columns if c not in ("Sharpe","Profit","Drawdown","Score")]
                        insights = run_automl_summary(df, sel_params)
                        if isinstance(insights, pd.DataFrame):
                            st.dataframe(insights, width="stretch")
                        else:
                            st.json(insights)
                    except Exception as e:
                        st.info(f"AutoML summary unavailable: {e}")

            # Data/feature preview
            _preview_data_features(sym1, sym2)


# =========================
# SECTION 9: __main__
# =========================
if __name__ == "__main__":
    render_optimization_tab()

