# -*- coding: utf-8 -*-
"""
backtest.py — Professional Pairs Backtester (Streamlit-ready, v3 HF-grade)
===========================================================================

Role in the system
------------------
This module is the *core backtest engine* for your pairs-trading system.
It is designed to be:

- **Engine-first**: clean API (`run_backtest`, `run_backtest_with_vol`,
  `objective_backtest`) that other modules (optimizer, dashboard, agents)
  can call.
- **HF-grade**: rich risk controls, volatility targeting, statistical gates,
  and structured outputs that plug into analytics / reporting / DuckDB.
- **Streamlit-ready**: `render_backtest_tab()` builds a full UI on top.
- **Extensible**: supports new strategies, risk profiles, and execution
  models without breaking existing callers.

Key concepts (high level)
-------------------------
1. Strategy layer  (Strategy enum)
   - Today: Z-score mean reversion on pairs.
   - Future: Fair-value / FV-engine, intraday variants, portfolio sleeves.

2. Config layer    (BacktestConfig, ExecutionConfig, RiskConfig)
   - Separates *what* the strategy does (z_entry / z_exit / lookback / gates)
     from *how* it is executed (slippage / costs / bar lag / VT) and
     *how* risk is controlled (max bars, z_stop, run DD stop, limits).

3. Result layer    (Trade, BacktestResult)
   - Trades table: every backtest stores a rich trade ledger with reasons,
     VT notional, and risk flags.
   - Metrics: HF-style KPIs + “extended metrics” that can be segmented
     by regime/timeframe in later parts.

4. Validation layer (BacktestParams / Pydantic)
   - Optional but powerful. Keeps API sane even when called from Optuna,
     agents, or external scripts.

This is **Part 1/6**:
- Imports + optional deps
- Logging setup
- Core enums
- Core dataclasses (ExecutionConfig, RiskConfig, BacktestConfig, Trade,
  BacktestResult)
- BacktestParams (Pydantic validation)

Parts 2–6 will build on this:
- Part 2: Engine math, VT override, core run_backtest().
- Part 3: Optuna objective, synthetic-data helpers, tests.
- Part 4: HF-grade metrics, segment analytics, MC/regime hooks.
- Part 5: Streamlit tab UI (render_backtest_tab) with all the extras.
- Part 6: Integration helpers (DuckDB logging, CLI, agents hooks).
"""
from __future__ import annotations

# ============================================================================
# Imports & Optional Dependencies
# ============================================================================

import json 
import logging
from dataclasses import dataclass, asdict, field
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd
import hashlib
from pathlib import Path
# ---- Optional UI / plotting deps (graceful degradation) --------------------
try:
    import streamlit as st  # type: ignore
except Exception:  # pragma: no cover
    st = None  # type: ignore

# ---- Optional link to optimisation best-params registry --------------------

def _get_opt_best_params_entry(pair_label: str) -> Optional[Dict[str, Any]]:
    """
    מחזיר רשומה מתוך opt_best_params_registry עבור זוג מסוים, אם קיימת.

    ציפייה:
        st.session_state["opt_best_params_registry"] = {
            "XLY-XLP": {
                "params": {...},
                "score": float|None,
                "profile": "defensive"/"...",
                "updated_at": "2025-12-03T17:21:00Z",
            },
            ...
        }

    מחזיר את ה-entry עצמו (dict) או None אם אין.
    """
    if st is None:
        return None
    try:
        reg = st.session_state.get("opt_best_params_registry", {})
        if not isinstance(reg, dict):
            return None
        entry = reg.get(str(pair_label))
        if isinstance(entry, dict):
            return entry
        return None
    except Exception:
        return None

try:
    import plotly.graph_objects as go  # type: ignore
except Exception:  # pragma: no cover
    go = None  # type: ignore

# ---- Optional Pydantic v2 (validation layer) --------------------------------
try:
    from pydantic import BaseModel, Field, field_validator  # type: ignore
except Exception:  # pragma: no cover
    BaseModel = object  # type: ignore

    def Field(*_a, **_k):  # type: ignore
        return None

    def field_validator(*_a, **_k):  # type: ignore
        def _wrap(fn):
            return fn
        return _wrap

# ---- Optional storage / stats deps -----------------------------------------
try:
    import duckdb  # type: ignore
except Exception:  # pragma: no cover
    duckdb = None  # type: ignore

try:
    from statsmodels.tsa.stattools import coint  # type: ignore
except Exception:  # pragma: no cover
    coint = None  # type: ignore

# ---- Project-level imports (expected in your repo) --------------------------
try:
    # Historical price loader — must return DataFrame with DatetimeIndex + 'close'
    from common.data_loader import load_price_data  # type: ignore
except Exception:  # pragma: no cover
    load_price_data = None  # type: ignore

try:
    from volatility import calculate_atr  # type: ignore
except Exception:  # pragma: no cover
    calculate_atr = None  # type: ignore

try:
    from common.utils import (  # type: ignore
        calculate_zscore,                 # (s1, s2, lookback) -> zscore series
        calculate_historical_volatility,  # optional
        get_implied_volatility,           # optional
        calculate_beta,                   # (s1, s2, lookback) -> beta series
        evaluate_edge,                    # (z, corr) -> edge score
    )
except Exception:
    # Fallbacks keep the engine usable if utils module is missing.
    def calculate_zscore(s1: pd.Series, s2: pd.Series, lookback: int) -> pd.Series:
        spread = s1 - s2
        m = spread.rolling(lookback).mean()
        sd = spread.rolling(lookback).std(ddof=0)
        return (spread - m) / sd

    def calculate_beta(s1: pd.Series, s2: pd.Series, lookback: int) -> pd.Series:
        cov = s1.rolling(lookback).cov(s2)
        var = s2.rolling(lookback).var(ddof=0)
        return cov / (var.replace(0, np.nan))

    def evaluate_edge(z: pd.Series | float, corr: pd.Series | float) -> pd.Series:
        z_s = pd.Series(z)
        c_s = pd.Series(corr)
        # Simple edge proxy: |z| * max(corr, 0)
        return (z_s.abs() * np.maximum(0.0, c_s)).astype(float)

    def calculate_historical_volatility(*_a, **_k):  # pragma: no cover
        return None

    def get_implied_volatility(*_a, **_k):  # pragma: no cover
        return None

from common.sql_price_loader import (
    init_sql_store_from_config,
    load_pair_with_spread,
)

def _calculate_beta_safe(
    s1: pd.Series,
    s2: pd.Series,
    lookback: int,
) -> pd.Series:
    """
    Wrapper אחיד ל-calculate_beta:

    תומך בחתימות שונות:
    1. calculate_beta(s1, s2, lookback)
    2. calculate_beta(spread, lookback)
    3. calculate_beta(spread)

    ולא משנה מה הפונקציה מחזירה (scalar / array / Series) –
    תמיד מחזיר Series מיושר לאינדקס של s1.
    """
    index = s1.index
    spread = s1 - s2

    def _to_series(res: Any) -> pd.Series:
        """המרה קשיחה לכל צורה ל-Series באורך index."""
        if isinstance(res, pd.Series):
            return pd.to_numeric(res.reindex(index), errors="coerce").astype(float)

        if isinstance(res, (list, tuple, np.ndarray)):
            arr = np.asarray(res, dtype=float)
            if arr.shape[0] == len(index):
                return pd.Series(arr, index=index)
            # אם האורך לא תואם – נזרוק ונעבור לנסיון הבא
            raise ValueError("beta array length mismatch")

        # scalar → נעשה ממנו קו ישר על כל הסדרה
        if np.isscalar(res):
            return pd.Series(float(res), index=index)

        raise TypeError(f"Unsupported beta return type: {type(res)}")

    # ניסיון 1: API ישן (s1, s2, lookback)
    try:
        res = calculate_beta(s1, s2, lookback)  # type: ignore[arg-type]
        return _to_series(res)
    except TypeError:
        pass
    except Exception:
        pass

    # ניסיון 2: API חדש (spread, lookback)
    try:
        res = calculate_beta(spread, lookback)  # type: ignore[arg-type]
        return _to_series(res)
    except TypeError:
        pass
    except Exception:
        pass

    # ניסיון 3: API (spread) בלבד
    try:
        res = calculate_beta(spread)  # type: ignore[arg-type]
        return _to_series(res)
    except Exception:
        # fallback נאיבי לגמרי – כדי *לא* ליפול
        cov = s1.rolling(lookback).cov(s2)
        var = s2.rolling(lookback).var(ddof=0).replace(0, np.nan)
        beta = (cov / var).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return beta.reindex(index)

# ============================================================================
# Logging
# ============================================================================

logger = logging.getLogger("backtest")

if not logger.handlers:
    # Respect external log configuration if already set; otherwise minimal handler.
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | backtest | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)

# Allow LOG_LEVEL env override via basicConfig if nothing else configured.
if logger.level == logging.NOTSET:
    logging.basicConfig(level=logging.INFO)

# ---- Adapter: safe beta computation (handles different calculate_beta signatures) ----

def _compute_beta_series(
    p1: pd.Series,
    p2: pd.Series,
    lookback: int,
) -> pd.Series:
    """
    Adapter סביב calculate_beta מהמערכת שלך, כדי להתמודד עם חתימות שונות.

    Priority (ניסיון לפי סדר):
    1. calculate_beta(p1, p2, lookback)
    2. calculate_beta(p1, p2)
    3. calculate_beta(spread, lookback)   # אם הפונקציה בנויה על spread בלבד
    4. fallback: חישוב beta קלאסי: cov(p1,p2) / var(p2)

    בכל מקרה מחזיר pd.Series באורך האינדקס של p1.
    """
    # אם יש calculate_beta אמיתי (לא fallback שלנו)
    if "calculate_beta" in globals() and callable(calculate_beta):  # type: ignore[name-defined]
        # 1) הניסיון ה"טבעי": p1, p2, lookback
        try:
            beta = calculate_beta(p1, p2, lookback)  # type: ignore[call-arg]
            if isinstance(beta, pd.Series):
                return beta.reindex(p1.index)
        except TypeError:
            pass
        except Exception:
            pass

        # 2) אולי החתימה היא (p1, p2)
        try:
            beta = calculate_beta(p1, p2)  # type: ignore[call-arg]
            if isinstance(beta, pd.Series):
                return beta.reindex(p1.index)
        except TypeError:
            pass
        except Exception:
            pass

        # 3) אולי החתימה היא (spread, window)
        try:
            spread_tmp = p1 - p2
            beta = calculate_beta(spread_tmp, lookback)  # type: ignore[call-arg]
            if isinstance(beta, pd.Series):
                return beta.reindex(p1.index)
        except TypeError:
            pass
        except Exception:
            pass

    # 4) Fallback: חישוב ידני, יציב
    cov = p1.rolling(lookback).cov(p2)
    var = p2.rolling(lookback).var(ddof=0)
    beta_fallback = cov / var.replace(0, np.nan)
    return beta_fallback

# ============================================================================
# Enums & Core Types
# ============================================================================


class Strategy(Enum):
    """
    Strategy family identifier.

    We keep it broad so the same module can serve:
    - PAIRS_ZSCORE        : classic Z-score mean-reversion.
    - PAIRS_FAIR_VALUE    : fair-value / FV-engine driven spreads.
    - PAIRS_INTRADAY      : intraday mean-reversion variants.
    - PORTFOLIO_RP        : portfolio sleeve (notional allocation only).
    """

    PAIRS_ZSCORE = "pairs_zscore"
    PAIRS_FAIR_VALUE = "pairs_fair_value"
    PAIRS_INTRADAY = "pairs_intraday"
    PORTFOLIO_RP = "portfolio_rp"


class SlippageMode(str, Enum):
    BPS = "bps"
    ATR_FRAC = "atr_frac"


class TradeSide(str, Enum):
    SHORT_LONG = "Short-Long"  # short X, long Y
    LONG_SHORT = "Long-Short"  # long X, short Y
    FLAT = "Flat"


class ExitReason(str, Enum):
    NORMAL = "normal"
    Z_EXIT = "z_exit"
    Z_STOP = "z_stop"
    MAX_BARS = "max_bars"
    EDGE_DROP = "edge_drop"
    ATR_EXIT = "atr_exit"
    REVERSAL = "reversal"
    RUN_DD_STOP = "run_dd_stop"
    KILL_SWITCH = "kill_switch"
    OTHER = "other"


# ============================================================================
# Config Dataclasses
# ============================================================================


@dataclass
class ExecutionConfig:
    """
    Execution model and cost controls.

    Attributes
    ----------
    bar_lag :
        How many bars after the signal the fill is assumed (1 = next bar).
    slippage_mode :
        "bps" (basis points on notional) or "atr_frac" (fraction of ATR per leg).
    slippage_bps :
        Used when slippage_mode == "bps". Applied on notional * 4 legs (2 entry, 2 exit).
    slippage_atr_frac :
        Used when slippage_mode == "atr_frac". Multiplied by ATR * notional * 4 legs.
    transaction_cost_per_trade :
        Fixed round-trip cost per trade (commission, fees, borrow, etc.).
    notional :
        Base notional for sizing (before volatility targeting).
    target_annual_vol :
        Optional *volatility targeting* knob: if > 0, the engine will scale notional
        so that annualised vol of the pair sleeve attempts to match this target.
    notional_max :
        Cap for volatility-targeted notional (per trade).
    """

    bar_lag: int = 1
    slippage_mode: SlippageMode = SlippageMode.BPS
    slippage_bps: float = 0.0
    slippage_atr_frac: float = 0.0
    transaction_cost_per_trade: float = 0.0
    notional: float = 1.0
    target_annual_vol: float = 0.0
    notional_max: float = 0.0


@dataclass
class RiskConfig:
    """
    Risk controls applied per trade and per run.

    Attributes
    ----------
    max_bars_held :
        Hard cap on hold duration; if None → disabled.
    z_stop :
        Exit if |Z| moves against us and exceeds this threshold.
    run_dd_stop_pct :
        Stop opening new trades if *run* drawdown (from peak) exceeds this fraction
        of peak PnL (e.g. 0.20 = 20% DD).
    max_open_trades :
        Optional cap on concurrent open trades per pair (future-use hook).
    kill_switch_pnl :
        Optional absolute PnL threshold (negative) where the engine stops trading.
    """

    max_bars_held: Optional[int] = None
    z_stop: Optional[float] = None
    run_dd_stop_pct: Optional[float] = None
    max_open_trades: Optional[int] = None
    kill_switch_pnl: Optional[float] = None


@dataclass
class BacktestConfig:
    """
    BacktestConfig bundles strategy settings, execution model and risk controls.

    Strategy (pairs Z-score mean-reversion)
    ---------------------------------------
    z_entry / z_exit :
        Entry/exit thresholds for |Z|.
    lookback :
        Rolling window for beta, spread mean/std, and Z.
    atr_window :
        ATR window when ATR-based filters or slippage are used.

    Filters & statistical gates
    ---------------------------
    corr_min :
        Minimum rolling correlation to allow entry.
    beta_range :
        Acceptable beta range (min, max) for entries.
    atr_max :
        Maximum ATR for entry (filters out extreme-vol regimes).
    edge_min :
        Minimum edge score for entries (evaluate_edge).
    atr_exit_max / edge_exit_min :
        Exit conditions based on ATR / edge.

    coint_pmax :
        Rolling cointegration p-value upper bound (if statsmodels available).
    hl_window :
        Window for OU half-life estimation.
    half_life_max :
        Maximum acceptable half-life in bars.

    execution & risk :
        Nested ExecutionConfig and RiskConfig (see their docstrings).
    """

    # Core Z-score strategy knobs
    z_entry: float = 2.0
    z_exit: float = 0.5
    lookback: int = 30
    atr_window: int = 14

    # Filters / entry & exit gates
    corr_min: Optional[float] = None
    beta_range: Optional[Tuple[float, float]] = None
    atr_max: Optional[float] = None
    edge_min: Optional[float] = None
    atr_exit_max: Optional[float] = None
    edge_exit_min: Optional[float] = None

    # Advanced statistical gates
    coint_pmax: Optional[float] = None
    hl_window: Optional[int] = None
    half_life_max: Optional[float] = None

    # Backwards-compat cost knobs (kept for older callers; engine uses exec)
    transaction_cost_per_trade: float = 0.0
    slippage_bps: float = 0.0
    notional: float = 1.0

    # Execution & risk sub-configs
    exec: ExecutionConfig = field(default_factory=ExecutionConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)

    # Warm-up bars before entries (None => auto based on windows)
    warmup_bars: Optional[int] = None


# ============================================================================
# Result Structures
# ============================================================================


@dataclass
class Trade:
    """
    Single trade record for a pairs Z-score strategy.

    Fields are intentionally rich so that later analytics / ML / agents can
    consume them without touching the engine.

    Mandatory core fields
    ---------------------
    entry_time / exit_time :
        Timestamps of trade.
    side :
        SHORT_LONG (short X, long Y) or LONG_SHORT (long X, short Y).
    entry_idx / exit_idx :
        Integer indices into the main price frame (for fast slicing).
    entry_z / exit_z :
        Z at entry/exit.
    entry_beta :
        Beta (hedge ratio) at entry.
    entry_corr :
        Rolling correlation at entry.
    entry_spread :
        Spread value at entry.
    pnl :
        Realised PnL in *notional currency*.

    HF-grade additions
    ------------------
    atr_at_entry :
        ATR at entry (if available).
    edge_at_entry :
        Edge score at entry (evaluate_edge).
    bars_held :
        Holding period length in bars.
    vt_notional :
        Actual notional used after volatility targeting (if enabled).
    fees_slippage :
        Monetary cost due to costs + slippage for this trade.
    exit_reason :
        Structured code (ExitReason enum) describing why we exited.
    risk_flags :
        Optional list of strings like ["z_stop", "dd_breach"] for post-hoc audits.
    regime_label :
        Optional textual regime tag at entry ("LVHC", "HVHC", etc.).
    """

    entry_time: pd.Timestamp
    exit_time: Optional[pd.Timestamp]
    side: str  # keep as str for backward compat; use TradeSide for new code
    entry_idx: int
    exit_idx: Optional[int]

    entry_z: float
    exit_z: Optional[float]

    entry_beta: float
    entry_corr: float
    entry_spread: float

    atr_at_entry: Optional[float]
    edge_at_entry: Optional[float]

    pnl: float
    bars_held: int

    # HF-grade extra fields (safe defaults for backward compatibility)
    vt_notional: Optional[float] = None
    fees_slippage: Optional[float] = None
    exit_reason: Optional[str] = None
    risk_flags: List[str] = field(default_factory=list)
    regime_label: Optional[str] = None


@dataclass
class BacktestResult:
    """
    Container for a full backtest run.

    strategy :
        Strategy enum (e.g. Strategy.PAIRS_ZSCORE).
    symbols :
        (sym_x, sym_y).
    window :
        (start_datetime, end_datetime) of the data used.
    config :
        Snapshot of BacktestConfig (and nested ExecutionConfig/RiskConfig) as dict.
    metrics :
        Flat dict of numeric KPIs (Sharpe, Profit, Drawdown, WinRate, etc.).
    trades :
        DataFrame of Trade rows (converted via asdict per trade).

    Extended fields (for HF / reporting)
    ------------------------------------
    run_id :
        Optional run identifier (e.g. ctx.run_id from dashboard).
    scenario_name :
        Optional tag such as "base", "stress_high_vol", etc.
    regime_summary :
        Optional dict with regime-level aggregates (DD, Sharpe per regime, etc.)
        — filled in later parts of the module.
    """

    strategy: Strategy
    symbols: Tuple[str, str]
    window: Tuple[datetime, datetime]
    config: Dict[str, Any]
    metrics: Dict[str, float]
    trades: pd.DataFrame

    run_id: Optional[str] = None
    scenario_name: Optional[str] = None
    regime_summary: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable representation (for DuckDB / API / reports)."""
        return {
            "strategy": self.strategy.value,
            "symbols": list(self.symbols),
            "window": (self.window[0].isoformat(), self.window[1].isoformat()),
            "config": self.config,
            "metrics": self.metrics,
            "trades": self.trades.to_dict(orient="records"),
            "run_id": self.run_id,
            "scenario_name": self.scenario_name,
            "regime_summary": self.regime_summary,
        }

    @property
    def equity_curve(self) -> pd.Series:
        """Cumulative PnL series derived from trades (by exit_time or index)."""
        if self.trades is None or self.trades.empty or "pnl" not in self.trades.columns:
            return pd.Series(dtype=float)
        df = self.trades.copy()
        if "exit_time" in df.columns:
            idx = pd.to_datetime(df["exit_time"].fillna(df["entry_time"]), errors="coerce")
        else:
            idx = pd.RangeIndex(start=1, stop=len(df) + 1, step=1)
        pnl = pd.to_numeric(df["pnl"], errors="coerce").fillna(0.0)
        eq = pnl.cumsum()
        eq.index = idx
        return eq.sort_index()


# ============================================================================
# Pydantic BacktestParams — Validation Layer (Optional but Recommended)
# ============================================================================


class BacktestParams(BaseModel):  # type: ignore[misc]
    """
    High-level, validated parameter bundle for a single backtest run.

    This is mainly used for:
    - UI → engine (render_backtest_tab → run_backtest)
    - Agents / external scripts → engine
    - Optuna search spaces (objective_backtest)

    Notes
    -----
    - All fields have *safe defaults* so partial dicts are fine.
    - If Pydantic is not installed, this class degrades to a simple object and
      the engine falls back to "best effort" without raising.
    """

    sym_x: str
    sym_y: str

    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None

    # Core Z-score strategy knobs
    z_entry: float = Field(default=2.0, ge=0.1, le=10)
    z_exit: float = Field(default=0.5, ge=0.05, le=10)
    lookback: int = Field(default=30, ge=10, le=252)
    atr_window: int = Field(default=14, ge=5, le=100)

    # Filters / gates
    corr_min: Optional[float] = Field(default=None)
    beta_range: Optional[Tuple[float, float]] = None
    edge_min: Optional[float] = None
    atr_max: Optional[float] = None
    edge_exit_min: Optional[float] = None
    atr_exit_max: Optional[float] = None

    # Advanced statistical gates
    coint_pmax: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    hl_window: Optional[int] = Field(default=None, ge=10, le=252)
    half_life_max: Optional[float] = Field(default=None, ge=1.0, le=2000)

    # Execution & risk
    notional: Optional[float] = Field(default=1.0, gt=0)
    slippage_mode: Optional[Literal["bps", "atr_frac"]] = "bps"
    slippage_bps: Optional[float] = Field(default=0.0, ge=0, le=1000)
    slippage_atr_frac: Optional[float] = Field(default=0.0, ge=0.0, le=1.0)
    transaction_cost_per_trade: Optional[float] = Field(default=0.0, ge=0.0, le=1000)
    bar_lag: Optional[int] = Field(default=1, ge=0, le=10)
    max_bars_held: Optional[int] = Field(default=None, ge=1, le=5000)
    z_stop: Optional[float] = Field(default=None, ge=0.1, le=10)
    run_dd_stop_pct: Optional[float] = Field(default=None, ge=0.01, le=1.0)

    # Volatility targeting (explicit API; also mirrored via ExecutionConfig)
    target_annual_vol: Optional[float] = Field(default=0.0, ge=0.0, le=2.0)
    notional_max: Optional[float] = Field(default=0.0, ge=0.0)

    # Book-keeping / tagging
    scenario_name: Optional[str] = Field(default=None, description="Scenario label (e.g. 'base', 'stress_vol_up').")
    run_id: Optional[str] = Field(default=None, description="External run identifier (e.g. dashboard ctx.run_id).")

    @field_validator("z_exit")
    def _z_exit_lt_entry(cls, v, info):  # type: ignore[override]
        """
        Ensure z_exit < z_entry when both provided.
        """
        try:
            z_entry = info.data.get("z_entry", 2.0)
            if v >= z_entry:
                raise ValueError("z_exit must be < z_entry")
        except Exception:
            # if we fail to read z_entry, don't block; engine will still be safe
            return v
        return v
    
Number = Union[int, float]

# ============================================================================
# Part 2/6 — Core Engine, Risk Logic & Volatility Targeting
# ============================================================================

# ---------- Helpers: Slippage & Costs --------------------------------------


def _compute_slippage_cost(
    exec_cfg: ExecutionConfig,
    atr_value: Optional[float],
    notional: float,
    *,
    legs: int = 4,
) -> float:
    """
    Compute monetary slippage for a *round-trip* trade on all legs.

    Modes
    -----
    - BPS:
        cost = slippage_bps / 1e4 * notional * legs
    - ATR_FRAC:
        cost = slippage_atr_frac * ATR * notional * legs

    Notes
    -----
    - legs=4 → 2 legs (X,Y) בכניסה + 2 legs ביציאה.
    - אם ATR לא זמין במצב ATR_FRAC → נופלים ל־0 (לא נזרוק שגיאה).
    """
    mode = getattr(exec_cfg, "slippage_mode", SlippageMode.BPS)
    if mode == SlippageMode.ATR_FRAC:
        if atr_value is None or not np.isfinite(atr_value):
            return 0.0
        return float(exec_cfg.slippage_atr_frac * float(atr_value) * float(notional) * float(legs))

    # BPS מוד – בסיסי
    return float((exec_cfg.slippage_bps / 1e4) * float(notional) * float(legs))


# ---------- Helpers: Rolling Cointegration & Half-Life ---------------------


def _rolling_coint_p(s1: pd.Series, s2: pd.Series, window: Optional[int]) -> pd.Series:
    """
    Rolling Engle-Granger cointegration p-value.

    אם statsmodels / coint לא זמינים או window קטן – מחזיר NaN.

    משמש כ-gate לכניסות:
        coint_pmax → נבדק מול הערך האחרון של הסדרה.
    """
    if coint is None or window is None or window < 10:
        return pd.Series(np.nan, index=s1.index)

    pvals: List[float] = []
    idx = s1.index

    for i in range(len(s1)):
        if i < window:
            pvals.append(np.nan)
            continue
        try:
            y1 = s1.iloc[i - window: i]
            y2 = s2.iloc[i - window: i]
            if y1.isna().any() or y2.isna().any():
                pvals.append(np.nan)
                continue
            p = float(coint(y1.values, y2.values, trend="c")[1])
        except Exception:
            p = np.nan
        pvals.append(p)

    return pd.Series(pvals, index=idx)


def _rolling_half_life(spread: pd.Series, window: Optional[int]) -> pd.Series:
    """
    Rolling OU half-life via AR(1):

        x_t = a + b * x_{t-1} + eps,  hl = -ln(2) / ln(b)

    - אם b<=0 או b>=1 → נחשב כ-NaN (לא Mean-Reverting תקין).
    - משמש כ-gate לכניסות (half_life_max).
    """
    if window is None or window < 10:
        return pd.Series(np.nan, index=spread.index)

    x = spread.astype(float)
    hl: List[float] = []

    for i in range(len(x)):
        if i < window:
            hl.append(np.nan)
            continue
        try:
            y = x.iloc[i - window + 1 : i + 1].values
            xlag = np.roll(y, 1)[1:]
            y = y[1:]
            if len(y) < 3:
                hl.append(np.nan)
                continue
            # OLS slope b
            cov = np.cov(xlag, y, ddof=0)
            var_x = np.var(xlag) + 1e-12
            b = cov[0, 1] / var_x
            if b <= 0.0 or b >= 1.0:
                hl.append(np.nan)
            else:
                hl.append(float(-np.log(2.0) / np.log(b)))
        except Exception:
            hl.append(np.nan)

    return pd.Series(hl, index=spread.index)



# ---------- Volatility Targeting Helpers -----------------------------------


def _vt_get_params(cfg: BacktestConfig) -> Tuple[Optional[float], Optional[float]]:
    """
    Extract volatility-targeting knobs from cfg + globals + Streamlit session.

    Priority:
        1. cfg.exec.target_annual_vol / cfg.exec.notional_max
        2. CLI globals (_VT_TGT, _VT_CAP)
        3. Streamlit session_state["bt_tgtvol"] / ["bt_ncap"]
    """
    tgt = getattr(cfg.exec, "target_annual_vol", 0.0)
    cap = getattr(cfg.exec, "notional_max", 0.0)

    # CLI global overrides (run_backtest_with_vol / CLI)
    try:
        if ("_VT_TGT" in globals()) and (tgt is None or tgt == 0):
            tv = float(globals().get("_VT_TGT"))
            if tv > 0:
                tgt = tv
        if ("_VT_CAP" in globals()) and (cap is None or cap == 0):
            nc = float(globals().get("_VT_CAP"))
            if nc > 0:
                cap = nc
    except Exception:
        pass

    # Streamlit overrides (if running inside dashboard)
    try:
        if st is not None:
            # קרא מה־widget החדש bt_tgtvol_input
            if (tgt is None or tgt == 0) and "bt_tgtvol_input" in st.session_state:
                tv = float(st.session_state.get("bt_tgtvol_input", 0.0))
                if tv > 0:
                    tgt = tv
            # קרא מה־widget החדש bt_ncap_input
            if (cap is None or cap == 0) and "bt_ncap_input" in st.session_state:
                nc = float(st.session_state.get("bt_ncap_input", 0.0))
                if nc > 0:
                    cap = nc
    except Exception:
        pass

    tgt = float(tgt) if tgt and tgt > 0 else None
    cap = float(cap) if cap and cap > 0 else None
    return tgt, cap


def _vt_realized_vol(
    df: pd.DataFrame,
    entry_idx: int,
    beta0: float,
    window: int,
) -> Optional[float]:
    """
    Estimate annualised realised vol of the pair spread before entry:

        r_pair_t = 1 * ret1 + (-beta0) * ret2

    Used to compute local notional:
        local_n ≈ base_notional * (target_annual_vol / vol).
    """
    try:
        start = max(0, entry_idx - window)
        r = (1.0 * df["ret1"].iloc[start:entry_idx] + (-beta0) * df["ret2"].iloc[start:entry_idx]).dropna()
        if len(r) < max(20, window // 2):
            return None
        vol = float(r.std(ddof=0) * np.sqrt(252.0))
        return vol if vol > 0 else None
    except Exception:
        return None


# ---------- Trade PnL with Volatility Targeting ----------------------------


def _compute_trade_pnl_and_cost(
    df: pd.DataFrame,
    i0: int,
    i1: int,
    side: str,
    cfg: BacktestConfig,
) -> Tuple[float, float, float]:
    """
    מחשב PnL לטרייד יחיד + notional אחרי Volatility Targeting + עלויות כוללות.

    Returns
    -------
    (net_pnl, vt_notional, total_cost)

    net_pnl     : PnL אחרי עלויות (slippage + transaction_cost_per_trade)
    vt_notional : הנוטיונל האפקטיבי בו השתמשנו בפועל
    total_cost  : סכום עלויות (slippage + transaction_cost)
    """
    # 1) ולידציה בסיסית לאינדקסים
    if i0 < 0 or i1 <= i0 or i1 >= len(df):
        return 0.0, 0.0, 0.0

    bar_lag = max(0, int(getattr(cfg.exec, "bar_lag", 1)))
    i_exec_start = min(len(df) - 1, i0 + bar_lag)
    if i_exec_start >= i1:
        return 0.0, 0.0, 0.0

    seg = df.iloc[i_exec_start : i1 + 1]
    if seg.empty:
        return 0.0, 0.0, 0.0

    # 2) Beta + צד הפוזיציה → משקלי הרגליים
    beta0 = _safe_float(df.iloc[i0].get("beta", 0.0), 0.0)

    if side == TradeSide.SHORT_LONG.value:
        # Short X, Long Y
        w1, w2 = -1.0, +beta0
    elif side == TradeSide.LONG_SHORT.value:
        # Long X, Short Y
        w1, w2 = +1.0, -beta0
    else:
        return 0.0, 0.0, 0.0

    # 3) Volatility Targeting
    base_notional = float(getattr(cfg.exec, "notional", 1.0))
    local_notional = base_notional

    tgt_vol, cap_notional = _vt_get_params(cfg)
    if tgt_vol is not None and tgt_vol > 0:
        window = max(20, int(getattr(cfg, "lookback", 30)))
        realized_vol = _vt_realized_vol(df, i0, beta0, window)
        if realized_vol is not None and realized_vol > 0:
            local_notional = base_notional * (tgt_vol / realized_vol)
            if cap_notional is not None and cap_notional > 0:
                local_notional = min(local_notional, cap_notional)
            local_notional = max(0.0, float(local_notional))

    # 4) Gross PnL (סכום תשואות יומיות על שתי הרגליים)
    ret1 = pd.to_numeric(seg["ret1"], errors="coerce").fillna(0.0)
    ret2 = pd.to_numeric(seg["ret2"], errors="coerce").fillna(0.0)
    gross_pnl = float(local_notional * (w1 * ret1 + w2 * ret2).sum())

    # 5) עלויות: Slippage + Transaction Cost
    atr_val = None
    try:
        if "atr" in df.columns:
            atr_val = _safe_opt_float(df.iloc[i0].get("atr", np.nan))
    except Exception:
        atr_val = None

    slippage_cost = float(_compute_slippage_cost(cfg.exec, atr_val, local_notional, legs=4))
    fixed_cost = float(getattr(cfg.exec, "transaction_cost_per_trade", 0.0) or 0.0)
    total_cost = slippage_cost + fixed_cost

    net_pnl = gross_pnl - total_cost

    # 6) לוג DEBUG (אופציונלי)
    if logger.isEnabledFor(logging.DEBUG):
        try:
            entry_row = df.iloc[i_exec_start]
            exit_row = df.iloc[i1]
            logger.debug(
                "[TRADE] %s-%s | side=%s | entry=%s p1=%.4f p2=%.4f | "
                "exit=%s p1=%.4f p2=%.4f | beta=%.4f | gross=%.2f | "
                "slip=%.2f | cost=%.2f | net=%.2f",
                cfg.__dict__.get("sym_x", "X"),
                cfg.__dict__.get("sym_y", "Y"),
                side,
                getattr(entry_row, "name", ""),
                _safe_float(entry_row.get("p1", np.nan), np.nan),
                _safe_float(entry_row.get("p2", np.nan), np.nan),
                getattr(exit_row, "name", ""),
                _safe_float(exit_row.get("p1", np.nan), np.nan),
                _safe_float(exit_row.get("p2", np.nan), np.nan),
                beta0,
                gross_pnl,
                slippage_cost,
                fixed_cost,
                net_pnl,
            )
        except Exception:
            pass

    return float(net_pnl), float(local_notional), float(total_cost)

# ============================================================================
# Core Backtest Engine
# ============================================================================


def run_backtest_with_vol(
    *,
    target_annual_vol: Optional[float] = None,
    notional_max: Optional[float] = None,
    **kwargs: Any,
) -> BacktestResult:
    """
    Convenience wrapper to run backtest while explicitly setting VT knobs.

    Usage
    -----
        res = run_backtest_with_vol(
            target_annual_vol=0.15,
            notional_max=500_000,
            sym_x="XLY",
            sym_y="XLP",
            ...
        )

    Internally:
    -----------
    - Sets globals `_VT_TGT` / `_VT_CAP` temporarily (used by _vt_get_params).
    - Calls run_backtest(**kwargs).
    - Cleans up globals afterwards.
    """
    if target_annual_vol is not None and target_annual_vol > 0:
        globals()["_VT_TGT"] = float(target_annual_vol)
    if notional_max is not None and notional_max > 0:
        globals()["_VT_CAP"] = float(notional_max)

    try:
        return run_backtest(**kwargs)  # type: ignore[arg-type]
    finally:
        if "_VT_TGT" in globals():
            del globals()["_VT_TGT"]
        if "_VT_CAP" in globals():
            del globals()["_VT_CAP"]

# =======================================================================
# Part 3/6 — Signal Frame, Gating Engine & Trade State Helpers (Pro)
# =======================================================================
"""
חלק 3 — שכבת "מנוע" לפני הלולאה עצמה:

מה כלול כאן:
------------
1. Signal Frame Builder:
   - _build_price_frame(...)    → הבאת דאטה גולמי (p1/p2), התאמת אינדקסים.
   - _compute_signal_frame(...) → חישוב beta, spread, z, corr, ATR, coint_p, half_life, edge.
   - שמירה על עמודות עזר (ret1/ret2, rolling_vol, warmup_mask וכו').

2. Gating Engine (כניסות/יציאות):
   - Entry/Exit context dataclasses (EntryContext / ExitContext).
   - _entry_filters_ok(...)     → בדיקת gates לכניסה (corr, beta, ATR, edge, HL, coint).
   - _exit_filters_ok(...)      → בדיקת תנאי יציאה (Z, reversal, ATR, edge_drop).
   - _build_entry_side(...)     → קביעה האם Short-Long או Long-Short.
   - _update_run_risk_state(...)→ מעקב אחרי run-level DD stop.

3. Trade state helpers:
   - TradeState dataclass       → מייצג פוזיציה פתוחה (entry index, z, side וכו').
   - _init_trade_state(...)     → יצירת TradeState חדש.
   - _close_trade_state(...)    → סגירת TradeState → Trade dataclass.

החלק הזה לא מריץ Backtest בעצמו — הוא מספק API נקי
שחלק 4 ישתמש בו כדי לכתוב לולאה קריאה וקצרה.
"""

from dataclasses import dataclass, field

# -----------------------------------------------------------------------
# 3.1 — Signal frame construction
# -----------------------------------------------------------------------

def _build_price_frame(
    sym_x: str,
    sym_y: str,
    start_date: datetime,
    end_date: datetime,
) -> pd.DataFrame:
    """
    טוען מחירי close לשני הסימבולים ומחזיר DataFrame מאוחד ברמת HF-grade.

    **סדר עדיפויות מקצועי למקור דאטה**:
    -----------------------------------
    1. SqlStore (DuckDB, IBKR → prices)  ← מקור אמת של המערכת.
    2. Fallback ל-load_price_data (למשל Yahoo) אם אין דאטה ב-SqlStore
       או אם יש שגיאה ב-SqlStore.

    Columns (בסיסיים):
        p1, p2       — מחירי הסגירה לשני הנכסים (מיושר לפי union index).
        ret1, ret2   — תשואות יומיות פשוטות (pct_change), עם 0.0 בשורה הראשונה.

    Columns (מורחבים):
        log_ret1, log_ret2 — תשואות לוג יומיות (ln(1+ret)).
    """

    if start_date >= end_date:
        raise ValueError(
            f"Invalid backtest window: start_date ({start_date}) "
            f"must be earlier than end_date ({end_date})."
        )

    # -------------------------------------------------
    # 1) Helper פנימי שטוען *סימבול אחד* עם עדיפות ל-SqlStore
    # -------------------------------------------------
    def _load_one(sym: str) -> pd.DataFrame:
        """
        Loader HF-grade:

        1. ניסיון ראשון: SqlStore (IBKR → DuckDB).
        2. אם אין דאטה / שגיאה → fallback ל-load_price_data.
        """
        # ===== 1.1 SqlStore (HF-grade, מקור אמת) =====
        try:
            project_root = Path(__file__).resolve().parent.parent
            store = init_sql_store_from_config(project_root, env="dev")

            df_sql = store.load_price_history(sym, env="dev")
            if df_sql is not None and not df_sql.empty:
                df_sql = df_sql.copy()
                df_sql.index = pd.to_datetime(df_sql.index)

                mask = (df_sql.index >= start_date) & (df_sql.index <= end_date)
                df_sql = df_sql.loc[mask]

                if not df_sql.empty:
                    logger.info(
                        "Using SqlStore price history for %s (%s → %s, rows=%d)",
                        sym,
                        df_sql.index.min(),
                        df_sql.index.max(),
                        len(df_sql),
                    )
                    return df_sql
                else:
                    logger.info(
                        "SqlStore has no rows for %s in range %s → %s — falling back to loader.",
                        sym,
                        start_date,
                        end_date,
                    )
        except Exception:
            logger.debug(
                "SqlStore price load failed for %s — falling back to load_price_data.",
                sym,
                exc_info=True,
            )

        # ===== 1.2 fallback: load_price_data (למשל Yahoo) =====
        if load_price_data is None:
            raise RuntimeError(
                f"Cannot load prices for {sym}: SqlStore failed/empty and "
                "common.data_loader.load_price_data is not available."
            )

        try:
            # משתמשים בחתימה הפשוטה (symbol, start_date, end_date) כמו שהוגדר בחלק 3.
            df_loader = load_price_data(sym, start_date, end_date)
            logger.info(
                "Using fallback loader price data for %s (rows=%s)",
                sym,
                len(df_loader) if df_loader is not None else 0,
            )
            return df_loader
        except TypeError as exc:
            raise TypeError(
                "load_price_data() signature is not compatible with the expected "
                "positional form (symbol, start_date, end_date)."
            ) from exc

    # -------------------------------------------------
    # 2) טוענים X/Y דרך ה-helper המאוחד
    # -------------------------------------------------
    df_x = _load_one(sym_x)
    df_y = _load_one(sym_y)

    if df_x is None or df_x.empty:
        raise ValueError(f"No price data for {sym_x} in backtest window {start_date} → {end_date}.")
    if df_y is None or df_y.empty:
        raise ValueError(f"No price data for {sym_y} in backtest window {start_date} → {end_date}.")

    # Normalize index to DatetimeIndex
    df_x = df_x.copy()
    df_y = df_y.copy()
    df_x.index = pd.to_datetime(df_x.index)
    df_y.index = pd.to_datetime(df_y.index)

    if not isinstance(df_x.index, pd.DatetimeIndex):
        raise TypeError(f"Expected DatetimeIndex for {sym_x}, got {type(df_x.index)}.")
    if not isinstance(df_y.index, pd.DatetimeIndex):
        raise TypeError(f"Expected DatetimeIndex for {sym_y}, got {type(df_y.index)}.")

    # הסרת כפילויות ו-Sort
    df_x = df_x[~df_x.index.duplicated(keep="last")].sort_index()
    df_y = df_y[~df_y.index.duplicated(keep="last")].sort_index()

    # Build unified frame (union + forward-fill)
    df = pd.DataFrame(index=df_x.index.union(df_y.index)).sort_index()

    # ננסה קודם 'close' מ-SqlStore; אם אין – ניקח את העמודה הראשונה כמחיר
    def _extract_close(df_one: pd.DataFrame, sym: str) -> pd.Series:
        if "close" in df_one.columns:
            return df_one["close"]
        # אם בטעות הפורמט שונה (לא אמור לקרות ב-SqlStore), לפחות לא נתרסק:
        return df_one.iloc[:, 0]

    try:
        p1 = _extract_close(df_x, sym_x).reindex(df.index).ffill()
        p2 = _extract_close(df_y, sym_y).reindex(df.index).ffill()
    except KeyError as exc:
        raise KeyError(
            f"Expected 'close' column in loader output for {sym_x}/{sym_y}. "
            f"Got columns: {list(df_x.columns)} (for {sym_x}), {list(df_y.columns)} (for {sym_y})"
        ) from exc

    df["p1"] = pd.to_numeric(p1, errors="coerce")
    df["p2"] = pd.to_numeric(p2, errors="coerce")
    df = df.dropna(subset=["p1", "p2"])

    if len(df) < 2:
        raise ValueError(
            f"Not enough overlapping data points for {sym_x}/{sym_y} "
            f"in window {start_date} → {end_date} (len={len(df)})."
        )

    # Daily simple returns (for PnL & risk)
    df["ret1"] = df["p1"].pct_change().fillna(0.0)
    df["ret2"] = df["p2"].pct_change().fillna(0.0)

    # Daily log returns (optional but שימושי מאוד לניתוח)
    df["log_ret1"] = np.log1p(df["ret1"])
    df["log_ret2"] = np.log1p(df["ret2"])

    return df


def _compute_signal_frame(
    base: pd.DataFrame,
    cfg: BacktestConfig,
) -> pd.DataFrame:
    """
    מקבל בסיס מחירים (p1/p2/ret1/ret2) ומוסיף:

        beta        — rolling hedge ratio (dynamic)
        spread      — ספרד נבחר מתוך כמה מודלים (auto model selection)
        spread_mean — ממוצע מתגלגל.
        spread_std  — סטיית תקן מתגלגלת.
        z           — Z-score.
        corr        — rolling correlation.
        atr         — ATR על leg X (אם פונקציה זמינה).
        coint_p     — rolling cointegration p-value (אם statsmodels קיימת).
        half_life   — rolling OU half-life של spread.
        edge        — proxy לאיכות האדג' (from evaluate_edge).
        vol_roll    — rolling volatility של spread (daily).
        warmup_ok   — בוליאני: האם הנקודה אחרי warmup מינימלי.

    החלק “נובל” כאן הוא בחירת מודל spread/Z בצורה אוטומטית ורובוסטית לכל זוג:
    dynamic beta, static beta, log-ratio – לפי איכות ה־Z בפועל.
    """
    df = base.copy()

    # --- 0) sanity על lookback ---
    lookback = int(cfg.lookback)
    if lookback < 10:
        raise ValueError("lookback must be >= 10")

    # כדי להימנע מהפתעות בטיפוסים
    p1 = pd.to_numeric(df["p1"], errors="coerce").astype(float)
    p2 = pd.to_numeric(df["p2"], errors="coerce").astype(float)

    # --- 1) Dynamic beta (HF-grade) ---
    beta_dyn = _calculate_beta_safe(p1, p2, lookback).clip(-10, 10)

    # helper פנימי לחישוב mean/std/z ל-spread מסוים
    def _build_z_from_spread(spread: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        spread = pd.to_numeric(spread, errors="coerce").astype(float)
        m = spread.rolling(lookback).mean()
        std = spread.rolling(lookback).std(ddof=0)
        std = std.replace([np.inf, -np.inf], np.nan)
        std = std.where(std != 0.0, np.nan)
        z = (spread - m) / std
        z = z.replace([np.inf, -np.inf], np.nan)
        return spread, m, std, z

    # helper פנימי להערכת איכות Z של מודל
    def _score_z(z: pd.Series) -> Tuple[float, float, float]:
        z_clean = pd.to_numeric(z, errors="coerce").replace([np.inf, -np.inf], np.nan)
        if z_clean.empty:
            return 0.0, 0.0, 0.0
        valid_frac = float(z_clean.notna().mean())
        if z_clean.notna().sum() <= 1:
            return valid_frac, 0.0, 0.0
        z_std = float(z_clean.std(ddof=0))
        if not np.isfinite(z_std):
            z_std = 0.0
        score = valid_frac * z_std
        return valid_frac, z_std, score

    candidates: Dict[str, Dict[str, Any]] = {}

    # --- 2) Candidate #1: dynamic beta spread ---
    spread_dyn = p1 - beta_dyn * p2
    s1, m1, std1, z1 = _build_z_from_spread(spread_dyn)
    vf1, zstd1, score1 = _score_z(z1)
    candidates["beta_dynamic"] = {
        "spread": s1,
        "mean": m1,
        "std": std1,
        "z": z1,
        "beta": beta_dyn,
        "valid_frac": vf1,
        "z_std": zstd1,
        "score": score1,
    }

    # --- 3) Candidate #2: static beta (k = median ratio) ---
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (p1 / p2).replace([np.inf, -np.inf], np.nan)
    if ratio.notna().any():
        k_static = float(ratio.median(skipna=True))
    else:
        k_static = 1.0
    beta_static = pd.Series(k_static, index=df.index)
    spread_static = p1 - beta_static * p2
    s2, m2, std2, z2 = _build_z_from_spread(spread_static)
    vf2, zstd2, score2 = _score_z(z2)
    candidates["beta_static"] = {
        "spread": s2,
        "mean": m2,
        "std": std2,
        "z": z2,
        "beta": beta_static,
        "valid_frac": vf2,
        "z_std": zstd2,
        "score": score2,
    }

    # --- 4) Candidate #3: log-ratio spread ---
    # מאוד יציב סטטיסטית, טוב כ-fallback וגם כמודל בפני עצמו.
    p1_pos = p1.where(p1 > 0).ffill()
    p2_pos = p2.where(p2 > 0).ffill()
    with np.errstate(divide="ignore", invalid="ignore"):
        spread_log = (np.log(p1_pos) - np.log(p2_pos)).replace([np.inf, -np.inf], np.nan)
    s3, m3, std3, z3 = _build_z_from_spread(spread_log)
    vf3, zstd3, score3 = _score_z(z3)
    candidates["log_ratio"] = {
        "spread": s3,
        "mean": m3,
        "std": std3,
        "z": z3,
        "beta": beta_dyn,  # מבחינת גידור בפועל נמשיך להשתמש ב-beta הדינמי
        "valid_frac": vf3,
        "z_std": zstd3,
        "score": score3,
    }

    # --- 5) בחירת המודל המנצח (אפשרי future override דרך cfg.spread_model) ---
    requested_model = getattr(cfg, "spread_model", "auto") or "auto"
    best_name = None
    best_score = -1.0
    best_vf = 0.0
    best_zstd = 0.0

    # אם המשתמש ביקש מודל ספציפי (log_ratio / beta_dynamic / beta_static) – ננסה קודם אותו
    preferred_order: List[str]
    if requested_model in candidates and requested_model != "auto":
        preferred_order = [requested_model] + [k for k in candidates.keys() if k != requested_model]
    else:
        # סדר עדיפות כללי: dynamic → log_ratio → static
        preferred_order = ["beta_dynamic", "log_ratio", "beta_static"]

    for name in preferred_order:
        c = candidates.get(name)
        if not c:
            continue
        vf = float(c["valid_frac"])
        zstd = float(c["z_std"])
        sc = float(c["score"])
        # מסנן מודלים דגנרטיביים: מעט מאוד ברים תקינים או Z בלי פיזור
        if vf < 0.10 or zstd <= 1e-6:
            continue
        if sc > best_score:
            best_name = name
            best_score = sc
            best_vf = vf
            best_zstd = zstd

    # אם אף מודל לא עבר את הסף → fallback מסודר
    if best_name is None:
        # קודם ננסה log_ratio (הכי חזק סטטיסטית), אם גם שם הכל גרוע – dynamic
        if candidates["log_ratio"]["valid_frac"] > 0.0:
            best_name = "log_ratio"
            best_vf = float(candidates["log_ratio"]["valid_frac"])
            best_zstd = float(candidates["log_ratio"]["z_std"])
        else:
            best_name = "beta_dynamic"
            best_vf = float(candidates["beta_dynamic"]["valid_frac"])
            best_zstd = float(candidates["beta_dynamic"]["z_std"])

    chosen = candidates[best_name]

    logger.info(
        "[DEBUG signal_frame] spread_model=%s | valid_frac=%.3f | z_std=%.4f | scores={dyn=%.4f, static=%.4f, log=%.4f}",
        best_name,
        best_vf,
        best_zstd,
        float(candidates["beta_dynamic"]["score"]),
        float(candidates["beta_static"]["score"]),
        float(candidates["log_ratio"]["score"]),
    )

    # נרשום את המודל הנבחר ל-DataFrame
    df["beta"] = beta_dyn  # hedge ratio דינמי לשימוש ב-PnL
    df["spread"] = chosen["spread"]
    df["spread_mean"] = chosen["mean"]
    df["spread_std"] = chosen["std"]
    df["z"] = chosen["z"]

    # --- 6) corr (על מחירי close רגילים) ---
    df["corr"] = p1.rolling(lookback).corr(p2).fillna(0.0)

    # --- 7) ATR (optional, על p1 בלבד אם אין OHLC מלא) ---
    if calculate_atr is not None:
        try:
            tmp = pd.DataFrame(
                {"high": p1, "low": p1, "close": p1},
                index=df.index,
            )
            atr_series = calculate_atr(tmp, window=cfg.atr_window).reindex(df.index)
            df["atr"] = pd.to_numeric(atr_series, errors="coerce").ffill()
        except Exception:
            df["atr"] = np.nan
    else:
        df["atr"] = np.nan

    # --- 8) Rolling cointegration p & OU half-life על ה-spread **הנבחר** ---
    hl_window = int(cfg.hl_window or cfg.lookback)
    df["coint_p"] = _rolling_coint_p(p1, p2, hl_window)
    df["half_life"] = _rolling_half_life(df["spread"], hl_window)

    # --- 9) Edge proxy & realized vol על ה-spread הנבחר ---
    df["edge"] = evaluate_edge(df["z"], df["corr"])

    try:
        rv_win = max(20, min(lookback, 60))
        sp_ret = df["spread"].pct_change().replace([np.inf, -np.inf], np.nan)
        df["vol_roll"] = (
            sp_ret.rolling(rv_win)
            .std(ddof=0)
            .mul(np.sqrt(252.0))
            .replace([np.inf, -np.inf], np.nan)
        )
    except Exception:
        df["vol_roll"] = np.nan

    # --- 10) Warmup mask (שימוש בכל ה-windowים הרלוונטיים) ---
    auto_warm = max(cfg.lookback, cfg.atr_window, hl_window) * 2
    warmup_bars = getattr(cfg, "warmup_bars", None)
    if warmup_bars is None or warmup_bars <= 0:
        warmup_bars = auto_warm
    warmup_bars = int(max(lookback + 1, warmup_bars))

    idx = df.index
    mask = pd.Series(False, index=idx)
    if len(idx) > warmup_bars:
        mask.iloc[warmup_bars:] = True
    df["warmup_ok"] = mask

    return df


def _debug_signal_profile(
    df: pd.DataFrame,
    sym_x: str,
    sym_y: str,
    cfg: BacktestConfig,
    *,
    label: str = "",
) -> None:
    """
    DEBUG מקצועי: מדפיס פרופיל של הסיגנל עבור זוג ו־config נתון.

    מדפיס:
    - סטטיסטיקות של z / spread / corr.
    - כמה ברים עומדים בסף |z| >= z_entry.
    - כמה ברים עוברים את כל ה-gates יחד.
    """
    try:
        z = pd.to_numeric(df.get("z"), errors="coerce")
        sp = pd.to_numeric(df.get("spread"), errors="coerce")
        cr = pd.to_numeric(df.get("corr"), errors="coerce")
        atr = pd.to_numeric(df.get("atr"), errors="coerce")
        edge = pd.to_numeric(df.get("edge"), errors="coerce")
        coint_p = pd.to_numeric(df.get("coint_p"), errors="coerce")
        hl = pd.to_numeric(df.get("half_life"), errors="coerce")

        z_abs = z.abs()
        n = len(df)

        def _q(s: pd.Series, q: float) -> float:
            s2 = s.dropna()
            if s2.empty:
                return float("nan")
            return float(s2.quantile(q))

        z_entry = float(cfg.z_entry)

        # בסיס: פרופיל Z
        logger.info(
            "[DEBUG %s] %s-%s | N=%d | z[min, q25, q50, q75, max] = [%.3f, %.3f, %.3f, %.3f, %.3f] | count(|z|>=z_entry=%.3f) = %d",
            label,
            sym_x,
            sym_y,
            n,
            _q(z, 0.0),
            _q(z, 0.25),
            _q(z, 0.5),
            _q(z, 0.75),
            _q(z, 1.0),
            z_entry,
            int((z_abs >= z_entry).sum()),
        )

        # פרופיל ספרד וקורלציה
        logger.info(
            "[DEBUG %s] spread[mean=%.6f, std=%.6f] | corr[mean=%.3f, min=%.3f, max=%.3f]",
            label,
            float(sp.mean()) if not sp.empty else float("nan"),
            float(sp.std(ddof=0)) if not sp.empty else float("nan"),
            float(cr.mean()) if not cr.empty else float("nan"),
            float(cr.min()) if not cr.empty else float("nan"),
            float(cr.max()) if not cr.empty else float("nan"),
        )

        # נבדוק כמה ברים עוברים את ה-gates לפי cfg (בערך כמו _entry_filters_ok, אבל וקטורי)
        mask = pd.Series(True, index=df.index)

        if cfg.edge_min is not None:
            mask = mask & (edge >= cfg.edge_min)

        if cfg.atr_max is not None and cfg.atr_max > 0:
            mask = mask & (atr <= cfg.atr_max)

        if cfg.corr_min is not None:
            mask = mask & (cr >= cfg.corr_min)

        if cfg.beta_range is not None:
            beta = pd.to_numeric(df.get("beta"), errors="coerce")
            lo_raw, hi_raw = cfg.beta_range
            # נרמול טווח הבטא כדי למנוע lo>hi
            lo, hi = sorted((float(lo_raw), float(hi_raw)))
            mask = mask & (beta >= lo) & (beta <= hi)

        if cfg.coint_pmax is not None:
            mask = mask & (coint_p <= cfg.coint_pmax)

        if cfg.half_life_max is not None:
            mask = mask & (hl <= cfg.half_life_max)

        # כמה ברים עוברים גם את ה-gates וגם את סף ה-Z
        mask_all = mask & (z_abs >= z_entry)

        logger.info(
            "[DEBUG %s] gates summary: pass_gates=%d, pass_gates_and_Z=%d",
            label,
            int(mask.sum()),
            int(mask_all.sum()),
        )

    except Exception as e:
        logger.exception("[DEBUG %s] failed to print signal profile: %s", label, e)

def _safe_float(value: Any, default: float = 0.0) -> float:
    """
    ממיר ערך ל-float בצורה בטוחה:
    - אם אי אפשר להמיר → מחזיר default.
    - אם יוצא NaN / ±inf → מחזיר default.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(v):
        return default
    return v


def _safe_opt_float(value: Any, *, default: float | None = None) -> float | None:
    """
    כמו _safe_float אבל מחזיר None אם אין ערך תקין (מתאים לשדות אופציונליים:
    ATR, edge, coint_p, half_life וכו').
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(v):
        return default
    return v
# -----------------------------------------------------------------------
# 3.2 — Entry / Exit gating engine
# -----------------------------------------------------------------------

@dataclass
class EntryContext:
    """קונטקסט של קנדל בעת שוקלים כניסה לפוזיציה."""
    z: float
    corr: float
    beta: float
    spread: float
    atr: Optional[float]
    edge: Optional[float]
    coint_p: Optional[float]
    half_life: Optional[float]


@dataclass
class ExitContext:
    """קונטקסט של קנדל בעת שוקלים יציאה מפוזיציה."""
    z: float
    prev_z: float
    corr: float
    atr: Optional[float]
    edge: Optional[float]
    bars_held: int
    entry_z: float


@dataclass
class RunRiskState:
    """מעקב אחרי drawdown ברמת הריצה כולה (לגייטים גלובליים)."""
    cum_pnl: float = 0.0
    peak_pnl: float = 0.0
    allow_new_entries: bool = True
    last_dd: float = 0.0  # אחוז/absolute DD האחרון


def _entry_filters_ok(ctx: EntryContext, cfg: BacktestConfig) -> bool:
    """
    בדיקת gates לכניסה:
    - edge_min: quality של האדג'.
    - atr_max: הגבלה על תנודתיות מיידית.
    - corr_min: מינימום קורלציה rolling.
    - beta_range: טווח קביל ל-beta.
    - coint_pmax: p-value מקסימלי ל-coint (אם זמין).
    - half_life_max: HL מקסימלי (אם זמין).
    """
    # Edge
    if cfg.edge_min is not None:
        if ctx.edge is None or not np.isfinite(ctx.edge) or ctx.edge < cfg.edge_min:
            return False

    # ATR
    if cfg.atr_max is not None and cfg.atr_max > 0:
        if ctx.atr is not None and np.isfinite(ctx.atr) and ctx.atr > cfg.atr_max:
            return False

    # Corr
    if cfg.corr_min is not None:
        if not np.isfinite(ctx.corr) or ctx.corr < cfg.corr_min:
            return False

    # Beta
    if cfg.beta_range is not None:
        lo_raw, hi_raw = cfg.beta_range
        lo, hi = sorted((float(lo_raw), float(hi_raw)))
        if not np.isfinite(ctx.beta) or ctx.beta < lo or ctx.beta > hi:
            return False


    # Cointegration
    if cfg.coint_pmax is not None:
        if ctx.coint_p is not None and np.isfinite(ctx.coint_p) and ctx.coint_p > cfg.coint_pmax:
            return False

    # Half-life
    if cfg.half_life_max is not None:
        if ctx.half_life is not None and np.isfinite(ctx.half_life) and ctx.half_life > cfg.half_life_max:
            return False

    return True


def _exit_filters_ok(
    ctx: ExitContext,
    cfg: BacktestConfig,
    exit_conditions: List[str],
) -> bool:
    """
    בדיקת תנאי יציאה לפי רשימת exit_conditions:

    exit_conditions נתמך:
        - "Z-Score"     : יציאה כש-|Z| <= z_exit.
        - "Reversal Z"  : יציאה כאשר סימן Z מתהפך.
        - "ATR"         : יציאה אם ATR חורג מ-atr_exit_max.
        - "Edge Drop"   : יציאה אם edge יורד מתחת ל-edge_exit_min.
    תנאי Risk נוספים:
        - z_stop (cfg.risk.z_stop)        : hard stop ב-|Z| גבוה נגדנו.
        - max_bars_held (cfg.risk.max_bars_held).
    """
    # תנאי בסיסי: Exit ברמת Z (mean-reversion התחולל)
    if "Z-Score" in exit_conditions:
        if abs(ctx.z) <= cfg.z_exit:
            return True

    # Reversal של Z (חציית 0 לכיוון השני)
    if "Reversal Z" in exit_conditions:
        if np.sign(ctx.z) != np.sign(ctx.prev_z):
            return True

    # ATR גבוה מידי בזמן הפוזיציה
    if "ATR" in exit_conditions and cfg.atr_exit_max is not None and cfg.atr_exit_max > 0:
        if ctx.atr is not None and np.isfinite(ctx.atr) and ctx.atr > cfg.atr_exit_max:
            return True

    # Edge Drop: האדג' נעלם
    if "Edge Drop" in exit_conditions and cfg.edge_exit_min is not None:
        if ctx.edge is not None and np.isfinite(ctx.edge) and ctx.edge < cfg.edge_exit_min:
            return True

    # --- Risk rules ---
    # Z-Stop (hard stop נגדנו)
    if cfg.risk.z_stop is not None and cfg.risk.z_stop > 0:
        if abs(ctx.z) >= cfg.risk.z_stop and np.sign(ctx.z) == np.sign(ctx.entry_z):
            return True

    # Max bars held
    if cfg.risk.max_bars_held is not None and cfg.risk.max_bars_held > 0:
        if ctx.bars_held >= cfg.risk.max_bars_held:
            return True

    return False


def _build_entry_side(z_value: float) -> str:
    """
    קובע את סוג הפוזיציה:
        z>0 → Short-Long  (short X, long Y, spread יקר מדי)
        z<0 → Long-Short (long X, short Y, spread זול מדי)
    """
    if z_value >= 0:
        return "Short-Long"
    return "Long-Short"


def _update_run_risk_state(
    state: RunRiskState,
    realized_pnl: float,
    cfg: BacktestConfig,
) -> RunRiskState:
    """
    מעדכן מצב ריצה (RunRiskState) אחרי סגירת טרייד:
    - cum_pnl      : סכום PnL מצטבר.
    - peak_pnl     : שיא ה-equity.
    - last_dd      : drawdown אחרון (ביחס לשיא).
    - allow_new_entries : הופך ל-False אם run_dd_stop_pct נחצה.
    """
    state.cum_pnl += float(realized_pnl)
    state.peak_pnl = max(state.peak_pnl, state.cum_pnl)
    if state.peak_pnl > 0:
        dd_abs = state.peak_pnl - state.cum_pnl
        state.last_dd = float(dd_abs / state.peak_pnl)
    else:
        state.last_dd = 0.0

    if cfg.risk.run_dd_stop_pct is not None and cfg.risk.run_dd_stop_pct > 0:
        if state.last_dd >= cfg.risk.run_dd_stop_pct:
            state.allow_new_entries = False

    return state


# -----------------------------------------------------------------------
# 3.3 — Trade state helpers (לשימוש בלולאה בחלק 4)
# -----------------------------------------------------------------------

@dataclass
class TradeState:
    """מצב פוזיציה פתוחה בתוך הלולאה."""
    in_trade: bool = False
    side: str = ""
    entry_idx: int = -1
    entry_time: Optional[pd.Timestamp] = None
    entry_z: float = np.nan
    entry_beta: float = 0.0
    entry_corr: float = 0.0
    entry_spread: float = 0.0
    atr_at_entry: Optional[float] = None
    edge_at_entry: Optional[float] = None


def _init_trade_state(
    idx: int,
    ts: pd.Timestamp,
    row: pd.Series,
) -> TradeState:
    """יוצר TradeState חדש בזמן כניסה לפוזיציה, עם המרות בטוחות לכל המספרים."""
    # צד הפוזיציה לפי סימן ה-Z הנוכחי
    z_val = _safe_float(getattr(row, "z", np.nan), 0.0)
    side = _build_entry_side(z_val)

    # שימוש גורף ב-_safe_float / _safe_opt_float כדי לא להתפוצץ על טיפוסים "מוזרים"
    beta_val = _safe_float(getattr(row, "beta", np.nan), 0.0)
    corr_val = _safe_float(getattr(row, "corr", np.nan), 0.0)
    spread_val = _safe_float(getattr(row, "spread", np.nan), 0.0)

    atr_val = _safe_opt_float(getattr(row, "atr", np.nan)) if "atr" in row.index else None
    edge_val = _safe_opt_float(getattr(row, "edge", np.nan)) if "edge" in row.index else None

    return TradeState(
        in_trade=True,
        side=side,
        entry_idx=idx,
        entry_time=ts,
        entry_z=z_val,
        entry_beta=beta_val,
        entry_corr=corr_val,
        entry_spread=spread_val,
        atr_at_entry=atr_val,
        edge_at_entry=edge_val,
    )



def _close_trade_state(
    df: pd.DataFrame,
    cfg: BacktestConfig,
    state: TradeState,
    exit_idx: int,
) -> Trade:
    """
    סוגר TradeState ויוצר אובייקט Trade כולל חישוב P&L:

    PnL מחושב ע"י פונקציית _compute_trade_pnl_and_cost,
    שמשלבת Volatility Targeting + עלויות/סליפג'.
    """
    if not state.in_trade or state.entry_idx < 0:
        raise RuntimeError("Cannot close non-active TradeState")

    entry_idx = state.entry_idx
    bars_held = exit_idx - entry_idx
    row_exit = df.iloc[exit_idx]

    # שימוש באותו מנוע PnL כמו בחלק 2 (VT + עלויות)
    pnl, vt_notional, fees = _compute_trade_pnl_and_cost(
        df=df,
        i0=entry_idx,
        i1=exit_idx,
        side=state.side,
        cfg=cfg,
    )

    trade = Trade(
        entry_time=state.entry_time or df.index[entry_idx],
        exit_time=row_exit.name,
        side=state.side,
        entry_idx=entry_idx,
        exit_idx=exit_idx,
        entry_z=state.entry_z,
        exit_z=float(row_exit.z),
        entry_beta=state.entry_beta,
        entry_corr=state.entry_corr,
        entry_spread=state.entry_spread,
        atr_at_entry=state.atr_at_entry,
        edge_at_entry=state.edge_at_entry,
        pnl=float(pnl),
        bars_held=int(bars_held),
        vt_notional=float(vt_notional) if vt_notional is not None else None,
        fees_slippage=float(fees),
        exit_reason=None,
        risk_flags=[],
        regime_label=None,
    )
    return trade

# =======================================================================
# Part 4/6 — Core Backtest Loop & Result Builder (Ultra Pro)
# =======================================================================
"""
חלק 4 — מנוע הבקטסט עצמו:

מה כלול:
--------
1. נירמול כל הפרמטרים דרך BacktestParams (Pydantic) — *שימוש בכל שדה*:
   - z_entry / z_exit / lookback / atr_window
   - corr_min / beta_range / edge_min / atr_max
   - edge_exit_min / atr_exit_max
   - coint_pmax / hl_window / half_life_max
   - notional / slippage_mode / slippage_bps / slippage_atr_frac
   - transaction_cost_per_trade / bar_lag / max_bars_held / z_stop / run_dd_stop_pct

2. בניית BacktestConfig + ExecutionConfig + RiskConfig מלאים מהפרמטרים.

3. בניית signal frame עשיר:
   - p1/p2, ret1/ret2
   - beta, spread, spread_mean, spread_std, z, corr
   - atr, coint_p, half_life, edge, vol_roll, warmup_ok

4. לולאת backtest נקייה המבוססת על:
   - TradeState (in_trade, entry_idx, side, entry_z, ...).
   - RunRiskState (cum_pnl, peak_pnl, DD, allow_new_entries).
   - EntryContext / ExitContext + _entry_filters_ok / _exit_filters_ok.
   - שימוש בתנאי entry_conditions / exit_conditions (Z, Edge, ATR, Corr, Beta, Reversal).

5. בניית BacktestResult:
   - trades_df מלא עם כל השדות (entry/exit, bars_held, z, beta, edge וכו').
   - metrics מורחבים: Sharpe, Profit, Drawdown, WinRate, ProfitFactor, Calmar,
     Sortino, Avg/Median PnL, AvgBarsHeld, ExposurePct, MaxConsecLosses, Skew/Kurtosis.

6. התאמה מלאה ל־run_backtest_with_vol / objective_backtest / CLI:
   - החתימה של run_backtest נשארת זהה.
   - Volatility Targeting מתבצע דרך _trade_pnl (חלק 1/2), על בסיס state ב-cfg.exec
     וה־session_state (bt_tgtvol / bt_ncap) אם קיים.
"""

# -----------------------------------------------------------------------
# 4.1 — Metrics helper
# -----------------------------------------------------------------------

def _compute_backtest_metrics(
    trades_df: pd.DataFrame,
    total_bars: int,
) -> Dict[str, float]:
    """חישוב מדדי ביצועים מורחבים על בסיס טבלת הטריידים."""
    if trades_df is None or trades_df.empty:
        return {
            "Sharpe": 0.0,
            "Profit": 0.0,
            "Drawdown": 0.0,
            "WinRate": 0.0,
            "ProfitFactor": float("nan"),
            "Calmar": 0.0,
            "Trades": 0,
            "AvgPnL": 0.0,
            "MedianPnL": 0.0,
            "Sortino": 0.0,
            "AvgBarsHeld": 0.0,
            "ExposurePct": 0.0,
            "Skew": 0.0,
            "Kurtosis": 0.0,
            "MaxConsecLosses": 0.0,
        }

    pnls = trades_df["pnl"].to_numpy(dtype=float)
    profit = float(pnls.sum()) if pnls.size else 0.0

    # Equity / DD
    cum = np.cumsum(pnls) if pnls.size else np.array([])
    peak = np.maximum.accumulate(cum) if pnls.size else np.array([])
    drawdowns = (peak - cum) if pnls.size else np.array([])
    max_dd = float(drawdowns.max()) if drawdowns.size else 0.0

    # Sharpe (per trade → מנורמל ל-252)
    if pnls.size > 1 and pnls.std() > 0:
        sharpe = float((pnls.mean() / pnls.std()) * np.sqrt(252.0))
    else:
        sharpe = 0.0

    # WinRate / ProfitFactor
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    win_rate = float(len(wins) / len(pnls)) if pnls.size else 0.0
    gains = float(wins.sum()) if wins.size else 0.0
    loss_sum = float(-losses.sum()) if losses.size else 0.0
    profit_factor = float(gains / loss_sum) if loss_sum > 0 else float("nan")

    # Calmar (approx: mean per trade → annualized / MaxDD)
    if max_dd > 0 and pnls.size:
        calmar = float((pnls.mean() * 252.0) / max_dd)
    else:
        calmar = 0.0

    trades_count = int(len(trades_df))
    avg_pnl = float(pnls.mean()) if pnls.size else 0.0
    median_pnl = float(np.median(pnls)) if pnls.size else 0.0

    # Sortino: downside סטייה בלבד
    downside = pnls[pnls < 0]
    downside_std = float(downside.std(ddof=0)) if downside.size > 0 else 0.0
    sortino = float((avg_pnl * np.sqrt(252.0)) / downside_std) if downside_std > 0 else 0.0

    # Exposure
    avg_bars = float(trades_df["bars_held"].mean()) if trades_count else 0.0
    exposure_pct = float(trades_df["bars_held"].sum() / total_bars) if (trades_count and total_bars > 0) else 0.0

    # Skew / Kurtosis (על PnL per trade)
    if pnls.size > 1:
        m = pnls.mean()
        s = pnls.std(ddof=0)
        if s > 0:
            skew = float(((pnls - m) ** 3).mean() / (s ** 3))
            kurt = float(((pnls - m) ** 4).mean() / (s ** 4))
        else:
            skew = 0.0
            kurt = 0.0
    else:
        skew = 0.0
        kurt = 0.0

    # Max consecutive losses
    max_consec_losses = 0
    cur_losses = 0
    for v in pnls:
        if v < 0:
            cur_losses += 1
            max_consec_losses = max(max_consec_losses, cur_losses)
        else:
            cur_losses = 0

    return {
        "Sharpe": sharpe,
        "Profit": profit,
        "Drawdown": max_dd,
        "WinRate": win_rate,
        "ProfitFactor": profit_factor,
        "Calmar": calmar,
        "Trades": trades_count,
        "AvgPnL": avg_pnl,
        "MedianPnL": median_pnl,
        "Sortino": sortino,
        "AvgBarsHeld": avg_bars,
        "ExposurePct": exposure_pct,
        "Skew": skew,
        "Kurtosis": kurt,
        "MaxConsecLosses": float(max_consec_losses),
    }


# -----------------------------------------------------------------------
# 4.2 — Core engine: run_backtest (uses ALL BacktestParams fields)
# -----------------------------------------------------------------------

def run_backtest(
    sym_x: str,
    sym_y: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    entry_conditions: Optional[List[str]] = None,
    exit_conditions: Optional[List[str]] = None,
    z_entry: float = 2.0,
    z_exit: float = 0.5,
    edge_min: Optional[float] = None,
    atr_max: Optional[float] = None,
    corr_min: Optional[float] = None,
    beta_range: Optional[Tuple[float, float]] = None,
    edge_exit_min: Optional[float] = None,
    atr_exit_max: Optional[float] = None,
    lookback: int = 30,
    atr_window: int = 14,
    # advanced gates
    coint_pmax: Optional[float] = None,
    hl_window: Optional[int] = None,
    half_life_max: Optional[float] = None,
    # execution & risk (optional; defaults in BacktestConfig)
    notional: Optional[float] = None,
    slippage_bps: Optional[float] = None,
    slippage_mode: Optional[Literal["bps", "atr_frac"]] = None,
    slippage_atr_frac: Optional[float] = None,
    transaction_cost_per_trade: Optional[float] = None,
    bar_lag: Optional[int] = None,
    max_bars_held: Optional[int] = None,
    z_stop: Optional[float] = None,
    run_dd_stop_pct: Optional[float] = None,
) -> BacktestResult:
    """
    מנוע הבקטסט המרכזי — מחזיר BacktestResult.

    שים לב:
    -------
    * כל הפרמטרים של BacktestParams משוקללים כאן:
      - בניית cfg (BacktestConfig + ExecutionConfig + RiskConfig).
      - לוגיקת כניסה/יציאה / gates / DD stop / warmup.
    * פונקציית _trade_pnl (מהחלקים הקודמים) כבר תומכת ב-Volatility Targeting
      לפי cfg.exec.target_annual_vol / cfg.exec.notional_max וה-session_state.
    """
    logger.info("Starting backtest %s-%s", sym_x, sym_y)

    # ---- 1) Normalize dates ----
    end_date = end_date or datetime.today()
    start_date = start_date or (end_date - timedelta(days=365))

    # ---- 2) Pydantic validation (BacktestParams) ----
    #     כל השדות עוברים דרך Pydantic → מייצב את הקלט ומוודא ranges.
    try:
        params = BacktestParams(
            sym_x=sym_x,
            sym_y=sym_y,
            start_date=start_date,
            end_date=end_date,
            z_entry=z_entry,
            z_exit=z_exit,
            lookback=lookback,
            atr_window=atr_window,
            corr_min=corr_min,
            beta_range=beta_range,
            edge_min=edge_min,
            atr_max=atr_max,
            edge_exit_min=edge_exit_min,
            atr_exit_max=atr_exit_max,
            coint_pmax=coint_pmax,
            hl_window=hl_window,
            half_life_max=half_life_max,
            notional=notional if notional is not None else 1.0,
            slippage_mode=slippage_mode if slippage_mode is not None else "bps",
            slippage_bps=slippage_bps if slippage_bps is not None else 0.0,
            slippage_atr_frac=slippage_atr_frac if slippage_atr_frac is not None else 0.0,
            transaction_cost_per_trade=transaction_cost_per_trade if transaction_cost_per_trade is not None else 0.0,
            bar_lag=bar_lag if bar_lag is not None else 1,
            max_bars_held=max_bars_held,
            z_stop=z_stop,
            run_dd_stop_pct=run_dd_stop_pct,
        )  # type: ignore[call-arg]

        # Pull normalized values back out (מבטיח שימוש *בכל* שדות params)
        sym_x = params.sym_x
        sym_y = params.sym_y
        start_date = params.start_date or start_date
        end_date = params.end_date or end_date
        z_entry = float(params.z_entry)
        z_exit = float(params.z_exit)
        lookback = int(params.lookback)
        atr_window = int(params.atr_window)

        corr_min = params.corr_min
        beta_range = params.beta_range
        edge_min = params.edge_min
        atr_max = params.atr_max
        edge_exit_min = params.edge_exit_min
        atr_exit_max = params.atr_exit_max

        coint_pmax = params.coint_pmax
        hl_window = params.hl_window
        half_life_max = params.half_life_max

        notional = float(params.notional) if params.notional is not None else None
        slippage_mode = params.slippage_mode
        slippage_bps = float(params.slippage_bps) if params.slippage_bps is not None else None
        slippage_atr_frac = float(params.slippage_atr_frac) if params.slippage_atr_frac is not None else None
        transaction_cost_per_trade = float(params.transaction_cost_per_trade) if params.transaction_cost_per_trade is not None else None
        bar_lag = int(params.bar_lag) if params.bar_lag is not None else None
        max_bars_held = params.max_bars_held
        z_stop = params.z_stop
        run_dd_stop_pct = params.run_dd_stop_pct
    except Exception as e:
        # אם Pydantic לא קיים או validation נכשל — נמשיך עם הערכים הגולמיים
        logger.debug("BacktestParams validation skipped/failed: %s", e)

    # ---- 3) Build BacktestConfig (core + exec + risk) ----
    cfg = BacktestConfig(
        z_entry=z_entry,
        z_exit=z_exit,
        lookback=lookback,
        atr_window=atr_window,
        edge_min=edge_min,
        atr_max=atr_max,
        corr_min=corr_min,
        beta_range=beta_range,
        edge_exit_min=edge_exit_min,
        atr_exit_max=atr_exit_max,
        coint_pmax=coint_pmax,
        hl_window=hl_window,
        half_life_max=half_life_max,
    )

    # Execution & Risk from params (כל שדות BacktestParams נכנסים ל-cfg.exec / cfg.risk)
    if notional is not None:
        cfg.exec.notional = float(notional)
    if slippage_bps is not None:
        cfg.exec.slippage_bps = float(slippage_bps)
    if slippage_mode is not None:
        cfg.exec.slippage_mode = slippage_mode
    if slippage_atr_frac is not None:
        cfg.exec.slippage_atr_frac = float(slippage_atr_frac)
    if transaction_cost_per_trade is not None:
        cfg.exec.transaction_cost_per_trade = float(transaction_cost_per_trade)
    if bar_lag is not None:
        cfg.exec.bar_lag = int(bar_lag)

    if max_bars_held is not None and max_bars_held > 0:
        cfg.risk.max_bars_held = int(max_bars_held)
    if z_stop is not None and z_stop > 0:
        cfg.risk.z_stop = float(z_stop)
    if run_dd_stop_pct is not None and run_dd_stop_pct > 0:
        cfg.risk.run_dd_stop_pct = float(run_dd_stop_pct)

    # Warmup bars override (מה-UI, אם רלוונטי)
    try:
        if st is not None:
            w_override = int(st.session_state.get("bt_warmup", 0))
            if w_override > 0:
                setattr(cfg, "warmup_bars", w_override)
    except Exception:
        pass

    # ---- 4) Build signal frame (prices + indicators) ----
    base = _build_price_frame(sym_x, sym_y, start_date, end_date)
    df = _compute_signal_frame(base, cfg)
    total_bars = int(len(df))

    if df.empty:
        raise ValueError("Signal frame is empty after loading data and computing indicators.")

    # DEBUG: פרופיל סיגנל עבור הזוג
    _debug_signal_profile(df, sym_x, sym_y, cfg, label="run_backtest")

    # ---- 5) Prepare loop state ----
    entry_conditions = entry_conditions or ["Z-Score"]
    exit_conditions = exit_conditions or ["Z-Score"]

    trade_state = TradeState()
    run_state = RunRiskState()
    trades: List[Trade] = []

    # ---- 6) Main backtest loop ----
    # לולאה על כל bar. נשתמש ב-warmup_ok כדי לא להיכנס מוקדם מדי.
    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        ts = row.name  # Timestamp

        # נתעלם ממקומות שבהם אין z תקין
        try:
            if row.z is None or np.isnan(row.z) or not row.warmup_ok:
                continue
        except Exception:
            continue

        # ENTRY: רק אם אין פוזיציה, וה-run_state מאפשר כניסות חדשות
        if not trade_state.in_trade and run_state.allow_new_entries:
            # האם תנאי "טריגר" של Z מתקיים?
            trigger_z = ("Z-Score" not in entry_conditions) or (abs(row.z) >= cfg.z_entry)

            # בונים EntryContext לכל הגייטים – עם המרות בטוחות
            ectx = EntryContext(
                z=_safe_float(row.z, 0.0),
                corr=_safe_float(getattr(row, "corr", np.nan), 0.0),
                beta=_safe_float(getattr(row, "beta", np.nan), 0.0),
                spread=_safe_float(getattr(row, "spread", np.nan), 0.0),
                atr=_safe_opt_float(row.atr) if "atr" in row.index else None,
                edge=_safe_opt_float(row.edge) if "edge" in row.index else None,
                coint_p=_safe_opt_float(row.coint_p) if "coint_p" in row.index else None,
                half_life=_safe_opt_float(row.half_life) if "half_life" in row.index else None,
            )

            ok_filters = _entry_filters_ok(ectx, cfg)
            ok_entry = trigger_z and ok_filters

            # Plugin hook לפני כניסה (אם קיים)
            try:
                from plugins import before_entry_hook  # type: ignore
            except Exception:
                before_entry_hook = None  # type: ignore

            if before_entry_hook is not None:
                try:
                    veto = before_entry_hook(row=row, cfg=cfg, df=df.iloc[:i])
                    if veto is False:
                        ok_entry = False
                except Exception:
                    pass

            if ok_entry:
                trade_state = _init_trade_state(i, ts, row)
                continue  # נעבור לבר הבא

        # EXIT: אם יש פוזיציה פתוחה
        if trade_state.in_trade and trade_state.entry_idx >= 0:
            bars_held = i - trade_state.entry_idx

            xctx = ExitContext(
                z=_safe_float(row.z, 0.0),
                prev_z=_safe_float(getattr(prev, "z", row.z), _safe_float(row.z, 0.0)),
                corr=_safe_float(getattr(row, "corr", np.nan), 0.0),
                atr=_safe_opt_float(row.atr) if "atr" in row.index else None,
                edge=_safe_opt_float(row.edge) if "edge" in row.index else None,
                bars_held=bars_held,
                entry_z=_safe_float(trade_state.entry_z, 0.0),
            )

            should_exit = _exit_filters_ok(xctx, cfg, exit_conditions)

            if should_exit:
                trade = _close_trade_state(df, cfg, trade_state, exit_idx=i)
                trades.append(trade)
                trade_state = TradeState()  # reset

                # עדכון מצב DD ברמת run
                run_state = _update_run_risk_state(run_state, trade.pnl, cfg)

                # Plugin hook אחרי יציאה (אם קיים)
                try:
                    from plugins import after_exit_hook  # type: ignore
                except Exception:
                    after_exit_hook = None  # type: ignore
                if after_exit_hook is not None:
                    try:
                        # נשלח את קטע הדאטה של הטרייד לצורך דיאגנוסטיקה
                        after_exit_hook(trade=trade, cfg=cfg, df=df.iloc[trade.entry_idx : i + 1])
                    except Exception:
                        pass

    # ---- 7) Trades DataFrame ----
    trades_df = (
        pd.DataFrame([asdict(t) for t in trades])
        if trades
        else pd.DataFrame(
            columns=[
                "entry_time",
                "exit_time",
                "side",
                "entry_idx",
                "exit_idx",
                "entry_z",
                "exit_z",
                "entry_beta",
                "entry_corr",
                "entry_spread",
                "atr_at_entry",
                "edge_at_entry",
                "pnl",
                "bars_held",
            ]
        )
    )

    # ---- 8) Metrics ----
    metrics = _compute_backtest_metrics(trades_df, total_bars=total_bars)

    # ---- 9) Build BacktestResult ----
    result = BacktestResult(
        strategy=Strategy.PAIRS_ZSCORE,
        symbols=(sym_x, sym_y),
        window=(start_date, end_date),
        config={
            "lookback": lookback,
            "atr_window": atr_window,
            "z_entry": z_entry,
            "z_exit": z_exit,
            "edge_min": edge_min,
            "atr_max": atr_max,
            "corr_min": corr_min,
            "beta_range": beta_range,
            "edge_exit_min": edge_exit_min,
            "atr_exit_max": atr_exit_max,
            "coint_pmax": coint_pmax,
            "hl_window": hl_window,
            "half_life_max": half_life_max,
            "exec": asdict(cfg.exec),
            "risk": asdict(cfg.risk),
            "entry_conditions": entry_conditions,
            "exit_conditions": exit_conditions,
        },
        metrics=metrics,
        trades=trades_df,
    )

    logger.info(
        "Backtest finished %s-%s: Trades=%d, Sharpe=%.2f, Profit=%.2f, MaxDD=%.2f",
        sym_x,
        sym_y,
        int(metrics.get("Trades", 0)),
        float(metrics.get("Sharpe", 0.0)),
        float(metrics.get("Profit", 0.0)),
        float(metrics.get("Drawdown", 0.0)),
    )
    return result

# =======================================================================
# Part 5/6 — Streamlit Backtest Tab & Visual Analytics (Ultra Pro)
# =======================================================================

def _kpi_block(result: BacktestResult) -> None:
    """
    מציג בלוק KPIs מורחב עבור BacktestResult.

    כולל:
    - Sharpe / Sortino / Calmar / ProfitFactor / MaxDD / Trades.
    - WinRate, Avg/Median PnL, Exposure, MaxConsecLosses, Skew/Kurtosis.
    - סיכום טקסטואלי קצר ל"מצב האסטרטגיה".
    """
    if st is None:
        return

    m = result.metrics or {}

    sharpe = float(m.get("Sharpe", 0.0))
    sortino = float(m.get("Sortino", 0.0))
    profit = float(m.get("Profit", 0.0))
    dd = float(m.get("Drawdown", 0.0))
    win_rate = float(m.get("WinRate", 0.0)) * 100.0
    pf = m.get("ProfitFactor", float("nan"))
    calmar = float(m.get("Calmar", 0.0))
    trades = int(m.get("Trades", 0))
    avg_pnl = float(m.get("AvgPnL", 0.0))
    med_pnl = float(m.get("MedianPnL", 0.0))
    exposure = float(m.get("ExposurePct", 0.0)) * 100.0
    max_consec_losses = float(m.get("MaxConsecLosses", 0.0))
    skew = float(m.get("Skew", 0.0))
    kurt = float(m.get("Kurtosis", 0.0))

    # שורה ראשית
    c1, c2, c3 = st.columns(3)
    c1.metric("Sharpe", f"{sharpe:.2f}")
    pf_str = "—" if pf is None or (isinstance(pf, float) and np.isnan(pf)) else f"{float(pf):.2f}"
    c2.metric("Profit Factor", pf_str)
    c3.metric("Calmar", f"{calmar:.2f}")

    # שורה שנייה
    c4, c5, c6 = st.columns(3)
    c4.metric("Total Profit", f"{profit:,.2f}")
    c5.metric("Max Drawdown", f"{dd:,.2f}")
    c6.metric("Win Rate", f"{win_rate:.1f}%")

    # שורה שלישית
    c7, c8, c9 = st.columns(3)
    c7.metric("Trades", f"{trades}")
    c8.metric("Avg PnL / trade", f"{avg_pnl:.4f}")
    c9.metric("Median PnL", f"{med_pnl:.4f}")

    # שורה רביעית
    c10, c11, c12 = st.columns(3)
    c10.metric("Sortino", f"{sortino:.2f}")
    c11.metric("Exposure % (bars)", f"{exposure:.1f}%")
    c12.metric("Max Consecutive Losses", f"{max_consec_losses:.0f}")

    with st.expander("🔎 Distribution & Shape (Skew/Kurtosis)", expanded=False):
        d1, d2 = st.columns(2)
        d1.metric("Skew (PnL)", f"{skew:.2f}")
        d2.metric("Kurtosis (PnL)", f"{kurt:.2f}")

    # סיכום טקסטואלי קצר
    comments: List[str] = []
    comments.append(f"• Sharpe ≈ {sharpe:.2f}, Sortino ≈ {sortino:.2f}, ProfitFactor ≈ {pf_str}.")
    comments.append(f"• Trades={trades}, WinRate≈{win_rate:.1f}%, Exposure≈{exposure:.1f}% מהבר־ים.")
    if dd > 0:
        comments.append(f"• יחס PnL/DD (approx) ≈ {profit/dd:.2f}.")
    if max_consec_losses >= 5:
        comments.append("• רצף הפסדים מקסימלי גבוה יחסית — יש לשקול התאמת גודל פוזיציה או פילטרים.")
    if sharpe < 0:
        comments.append("• Sharpe שלילי — האסטרטגיה במצב לא טוב על חלון הזמן הזה.")
    elif sharpe < 1:
        comments.append("• Sharpe נמוך מ-1 — צריך עוד שיפור/פילטרים או זיווגים איכותיים יותר.")
    elif sharpe < 2:
        comments.append("• Sharpe בין 1 ל-2 — סביר, אבל עדיין לא ברמת קרן גידור מובילה.")
    else:
        comments.append("• Sharpe מעל 2 — ביצועים חזקים יחסית, כדאי לבדוק יציבות על חלונות אחרים.")

    st.markdown("**📌 סיכום מילולי:**")
    st.markdown("\n".join(comments))


def _trades_visuals(trades_df: pd.DataFrame, title_suffix: str = "") -> None:
    """
    ויזואליזציה מורחבת לטריידים:
    - Equity Curve + Drawdown Curve
    - פאי Win/Loss
    - היסטוגרמת PnL per Trade
    - Rolling Sharpe על רצף הטריידים (w=20)
    - Heatmap של PnL לפי BarsHeld × Side
    """
    if trades_df is None or trades_df.empty or go is None or st is None:
        return

    df = trades_df.copy()
    df["exit_time_eff"] = df["exit_time"].fillna(df["entry_time"])

    # ==== 1. Equity + Drawdown ====
    pnl = pd.to_numeric(df["pnl"], errors="coerce").fillna(0.0)
    eq = pnl.cumsum()
    peak = np.maximum.accumulate(eq.to_numpy())
    dd = peak - eq.to_numpy()

    fig_ec = go.Figure()
    fig_ec.add_trace(
        go.Scatter(
            x=df["exit_time_eff"],
            y=eq,
            mode="lines+markers",
            name="Equity",
        )
    )
    fig_ec.add_trace(
        go.Scatter(
            x=df["exit_time_eff"],
            y=-dd,
            mode="lines",
            name="Drawdown (neg)",
            line=dict(dash="dot"),
        )
    )
    fig_ec.update_layout(
        title=f"Equity & Drawdown {title_suffix}",
        xaxis_title="Time",
        yaxis_title="PnL / Drawdown",
    )
    st.plotly_chart(fig_ec, use_container_width=True)

    # ==== 2. Win/Loss pie ====
    outcome = np.where(pnl > 0, "Win", np.where(pnl < 0, "Loss", "Flat"))
    vc = pd.Series(outcome).value_counts()
    fig_pie = go.Figure(
        [go.Pie(labels=vc.index.tolist(), values=vc.values.tolist(), hole=0.4)]
    )
    fig_pie.update_layout(title="Trade Outcomes (Win/Loss/Flat)")
    st.plotly_chart(fig_pie, use_container_width=True)

    # ==== 3. Histogram of trade PnL ====
    fig_hist = go.Figure(
        [
            go.Histogram(
                x=pnl.values,
                nbinsx=40,
                name="PnL per Trade",
            )
        ]
    )
    fig_hist.update_layout(
        title="PnL per Trade Distribution",
        xaxis_title="PnL per Trade",
        yaxis_title="Count",
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # ==== 4. Rolling Sharpe over trade sequence (w=20) ====
    try:
        w = 20
        seq = pnl
        rs = seq.rolling(w).apply(
            lambda x: (x.mean() / (x.std(ddof=0) if x.std(ddof=0) > 0 else np.nan))
            * np.sqrt(252),
            raw=False,
        )
        fig_rs = go.Figure(
            [
                go.Scatter(
                    x=df["exit_time_eff"],
                    y=rs,
                    mode="lines",
                    name=f"Rolling Sharpe (w={w})",
                )
            ]
        )
        fig_rs.update_layout(
            title="Rolling Sharpe (by trade sequence)",
            xaxis_title="Time",
            yaxis_title="Sharpe (approx)",
        )
        st.plotly_chart(fig_rs, use_container_width=True)
    except Exception:
        pass

    # ==== 5. Heatmap: PnL vs BarsHeld × Side ====
    try:
        df_heat = df.copy()
        df_heat["bars_held"] = pd.to_numeric(df_heat["bars_held"], errors="coerce").fillna(0.0)
        df_heat["side"] = df_heat["side"].astype(str)
        df_heat["PnL"] = pnl

        # Discretize bars_held
        df_heat["bars_bin"] = pd.qcut(
            df_heat["bars_held"],
            q=min(10, max(1, df_heat["bars_held"].nunique())),
            duplicates="drop",
        ).astype(str)

        pivot = (
            df_heat.pivot_table(
                values="PnL",
                index="side",
                columns="bars_bin",
                aggfunc="mean",
            )
            .fillna(0.0)
            .sort_index(axis=1)
        )
        if not pivot.empty:
            fig_hm = go.Figure(
                data=go.Heatmap(
                    z=pivot.values,
                    x=pivot.columns.tolist(),
                    y=pivot.index.tolist(),
                    colorbar=dict(title="Avg PnL"),
                )
            )
            fig_hm.update_layout(
                title="Average PnL by Side × Holding Period (bins)",
                xaxis_title="Bars Held (bin)",
                yaxis_title="Side",
            )
            st.plotly_chart(fig_hm, use_container_width=True)
    except Exception:
        pass

# =======================================================================
# DuckDB Logging Helper (Part 6-lite)
# =======================================================================

# ===================== Advanced Analytics: MC, Walk-Forward, Sensitivity, History =====================

def _run_monte_carlo_from_trades(
    trades_df: pd.DataFrame,
    *,
    n_scenarios: int = 1000,
    horizon_trades: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Monte-Carlo על רצף הטריידים:
    - דוגם עם החזרה PnL per trade ובונה מסלולי PnL סינתטיים.
    - מחזיר:
        summary_df: סטטיסטיקות על כל המסלולים (PnL, MaxDD, Sharpe).
        paths_df:   טבלת מסלולים (index=step, columns=scenario_id, values=cum_pnl).
    """
    if trades_df is None or trades_df.empty or "pnl" not in trades_df.columns:
        return pd.DataFrame(), pd.DataFrame()

    pnl = pd.to_numeric(trades_df["pnl"], errors="coerce").dropna().to_numpy()
    if pnl.size == 0:
        return pd.DataFrame(), pd.DataFrame()

    n_trades = pnl.size
    horizon = int(horizon_trades) if horizon_trades is not None else n_trades

    rng = np.random.default_rng(42)
    all_paths = np.empty((horizon, n_scenarios), dtype=float)
    all_paths[:] = np.nan

    max_dds = []
    sharpes = []
    finals = []

    for j in range(n_scenarios):
        idx = rng.integers(0, n_trades, size=horizon)
        path = pnl[idx]
        cum = np.cumsum(path)
        all_paths[:, j] = cum

        finals.append(float(cum[-1]))
        peak = np.maximum.accumulate(cum)
        dd = peak - cum
        max_dd = float(dd.max()) if dd.size else 0.0
        max_dds.append(max_dd)

        if path.std(ddof=0) > 0 and path.size > 1:
            sh = float(path.mean() / path.std(ddof=0) * np.sqrt(252))
        else:
            sh = 0.0
        sharpes.append(sh)

    summary_df = pd.DataFrame(
        {
            "scenario": np.arange(n_scenarios),
            "final_pnl": finals,
            "max_dd": max_dds,
            "sharpe": sharpes,
        }
    )
    paths_idx = np.arange(1, horizon + 1)
    paths_df = pd.DataFrame(all_paths, index=paths_idx)
    return summary_df, paths_df


def _run_walkforward_grid(
    sym_x: str,
    sym_y: str,
    base_kwargs: Dict[str, Any],
    *,
    start_date: datetime,
    end_date: datetime,
    n_splits: int = 4,
) -> pd.DataFrame:
    """
    Walk-Forward Backtest:
    - מחלק את טווח התאריכים ל-n_splits חלונות.
    - מריץ backtest עבור כל חלון (עם אותם פרמטרים).
    - מחזיר DataFrame עם metrics לכל חלון: Sharpe, Profit, MaxDD, Trades.
    """
    if n_splits < 1:
        n_splits = 1

    try:
        dates = pd.date_range(start=start_date, end=end_date, freq="D")
    except Exception:
        return pd.DataFrame()

    if len(dates) < n_splits * 30:
        # מעט מדי ימים לחלוקה משמעותית
        return pd.DataFrame()

    chunks = np.array_split(dates, n_splits)
    rows: List[Dict[str, Any]] = []

    for i, ch in enumerate(chunks, start=1):
        if len(ch) < 10:
            continue
        sd = ch[0].to_pydatetime()
        ed = ch[-1].to_pydatetime()

        try:
            res = run_backtest(
                sym_x=sym_x,
                sym_y=sym_y,
                start_date=sd,
                end_date=ed,
                **base_kwargs,
            )
        except Exception as e:
            logger.warning("Walk-forward segment %d failed: %s", i, e)
            continue

        m = res.metrics or {}
        rows.append(
            {
                "segment": i,
                "start": sd.date().isoformat(),
                "end": ed.date().isoformat(),
                "Sharpe": float(m.get("Sharpe", 0.0)),
                "Profit": float(m.get("Profit", 0.0)),
                "Drawdown": float(m.get("Drawdown", 0.0)),
                "Trades": int(m.get("Trades", 0)),
            }
        )

    return pd.DataFrame(rows)


def _run_param_sensitivity_grid(
    sym_x: str,
    sym_y: str,
    base_kwargs: Dict[str, Any],
    *,
    start_date: datetime,
    end_date: datetime,
    z_entry_grid: List[float],
    z_exit_grid: List[float],
) -> pd.DataFrame:
    """
    רגישות פרמטרים לזוג אחד:
    - סריקה על גבי רשת (z_entry × z_exit).
    - מריץ backtest לכל נקודה (יחסית coarse כדי לא להיות כבד מדי).
    - מחזיר טבלה עם Sharpe/Profit לכל קומבינציה.
    """
    rows: List[Dict[str, Any]] = []
    for ze in z_entry_grid:
        for zx in z_exit_grid:
            if zx >= ze:
                continue
            try:
                res = run_backtest(
                    sym_x=sym_x,
                    sym_y=sym_y,
                    start_date=start_date,
                    end_date=end_date,
                    z_entry=float(ze),
                    z_exit=float(zx),
                    **base_kwargs,
                )
            except Exception as e:
                logger.warning("Param sweep (z_entry=%.2f, z_exit=%.2f) failed: %s", ze, zx, e)
                continue

            m = res.metrics or {}
            rows.append(
                {
                    "z_entry": float(ze),
                    "z_exit": float(zx),
                    "Sharpe": float(m.get("Sharpe", 0.0)),
                    "Profit": float(m.get("Profit", 0.0)),
                    "Drawdown": float(m.get("Drawdown", 0.0)),
                    "Trades": int(m.get("Trades", 0)),
                }
            )

    return pd.DataFrame(rows)


def _render_backtest_advanced_panels(
    result: BacktestResult,
    *,
    sym_x: str,
    sym_y: str,
    start_date: datetime,
    end_date: datetime,
    base_kwargs: Dict[str, Any],
) -> None:
    """
    פאנלים מתקדמים לבקטסט:
    - Monte-Carlo על PnL per trade.
    - Walk-Forward לפי חלונות זמן.
    - רגישות פרמטרים (z_entry/z_exit) עם Heatmap של Sharpe.
    - השוואה לריצות קודמות (History בתוך session_state).
    """
    if st is None or go is None:
        return

    trades_df = result.trades if isinstance(result.trades, pd.DataFrame) else pd.DataFrame()

    # ===== 1) Monte-Carlo Stress Test =====
    with st.expander("🧪 Monte-Carlo Stress Test על טריידים", expanded=False):
        if trades_df.empty:
            st.caption("אין טריידים לביצוע Monte-Carlo (אין מה לסמלץ).")
        else:
            col_mc1, col_mc2 = st.columns(2)
            with col_mc1:
                n_scen = st.number_input(
                    "מספר תרחישים (ניסויי Monte-Carlo)",
                    min_value=100,
                    max_value=5000,
                    value=1000,
                    step=100,
                    key="bt_mc_n",
                )
            with col_mc2:
                horiz = st.number_input(
                    "אופק טריידים לסימולציה (0 = #טריידים בפועל)",
                    min_value=0,
                    max_value=5000,
                    value=0,
                    step=10,
                    key="bt_mc_h",
                )
            horizon_trades = int(horiz) if horiz > 0 else None

            if st.button("🚀 הרץ Monte-Carlo", key="bt_mc_run"):
                with st.spinner("מריץ Monte-Carlo על רצף הטריידים…"):
                    mc_summary, mc_paths = _run_monte_carlo_from_trades(
                        trades_df,
                        n_scenarios=int(n_scen),
                        horizon_trades=horizon_trades,
                    )

                if mc_summary.empty:
                    st.info("Monte-Carlo לא החזיר תוצאה (אולי מעט מדי טריידים).")
                else:
                    st.markdown("**סטטיסטיקות על תרחישי Monte-Carlo**")
                    st.dataframe(mc_summary.describe().T, width="stretch", height=260)

                    # התפלגות PnL סופי
                    fig_mc = go.Figure()
                    fig_mc.add_trace(
                        go.Histogram(
                            x=mc_summary["final_pnl"],
                            nbinsx=40,
                            name="Final PnL",
                        )
                    )
                    fig_mc.update_layout(
                        title="התפלגות PnL סופי בתרחישי Monte-Carlo",
                        xaxis_title="Final PnL",
                        yaxis_title="Count",
                    )
                    st.plotly_chart(fig_mc, use_container_width=True)

                    # מסלולים נבחרים
                    try:
                        sample_cols = mc_paths.columns[: min(20, mc_paths.shape[1])]
                        fig_paths = go.Figure()
                        for col in sample_cols:
                            fig_paths.add_trace(
                                go.Scatter(
                                    x=mc_paths.index,
                                    y=mc_paths[col],
                                    mode="lines",
                                    line=dict(width=1),
                                    showlegend=False,
                                )
                            )
                        fig_paths.update_layout(
                            title="מסלולי PnL במונחי טריידים (מדגם תרחישים)",
                            xaxis_title="Trade index in scenario",
                            yaxis_title="Cumulative PnL",
                        )
                        st.plotly_chart(fig_paths, use_container_width=True)
                    except Exception:
                        pass

    # ===== 2) Walk-Forward Backtest =====
    with st.expander("🧭 Walk-Forward Analysis (חלונות זמן)", expanded=False):
        col_wf1, col_wf2 = st.columns(2)
        with col_wf1:
            n_splits = st.slider(
                "כמה חלונות זמן?",
                min_value=2,
                max_value=12,
                value=4,
                step=1,
                key="bt_wf_splits",
            )
        with col_wf2:
            wf_mode = st.selectbox(
                "מה למדוד?",
                ["Sharpe", "Profit", "Drawdown", "Trades"],
                index=0,
                key="bt_wf_metric",
            )

        if st.button("🚀 הרץ Walk-Forward", key="bt_wf_run"):
            with st.spinner("מריץ Walk-Forward Backtests…"):
                wf_df = _run_walkforward_grid(
                    sym_x=sym_x,
                    sym_y=sym_y,
                    base_kwargs=base_kwargs,
                    start_date=start_date,
                    end_date=end_date,
                    n_splits=int(n_splits),
                )

            if wf_df.empty:
                st.info("Walk-Forward לא החזיר נתונים (בדוק טווח תאריכים / פרמטרים).")
            else:
                st.dataframe(wf_df, width="stretch", height=260)
                metric_col = wf_mode
                if metric_col in wf_df.columns:
                    fig_wf = go.Figure()
                    fig_wf.add_trace(
                        go.Bar(
                            x=wf_df["segment"].astype(str),
                            y=wf_df[metric_col],
                            name=metric_col,
                        )
                    )
                    fig_wf.update_layout(
                        title=f"{metric_col} לפי חלון Walk-Forward",
                        xaxis_title="Segment",
                        yaxis_title=metric_col,
                    )
                    st.plotly_chart(fig_wf, use_container_width=True)

    # ===== 3) Param Sensitivity (z_entry / z_exit) =====
    with st.expander("📐 Param Sensitivity (z_entry / z_exit Heatmap)", expanded=False):
        col_ps1, col_ps2 = st.columns(2)
        with col_ps1:
            z_entry_min = st.number_input("z_entry min", 0.5, 5.0, value=1.0, step=0.1, key="bt_ps_ze_min")
            z_entry_max = st.number_input("z_entry max", 0.5, 5.0, value=3.0, step=0.1, key="bt_ps_ze_max")
            grid_e = st.slider("מספר נקודות z_entry", 3, 10, 5, key="bt_ps_ze_n")
        with col_ps2:
            z_exit_min = st.number_input("z_exit min", 0.1, 4.0, value=0.2, step=0.1, key="bt_ps_zx_min")
            z_exit_max = st.number_input("z_exit max", 0.1, 4.0, value=1.0, step=0.1, key="bt_ps_zx_max")
            grid_x = st.slider("מספר נקודות z_exit", 3, 10, 4, key="bt_ps_zx_n")

        if st.button("🚀 הרץ רגישות פרמטרים", key="bt_ps_run"):
            z_entry_grid = np.linspace(z_entry_min, z_entry_max, int(grid_e)).tolist()
            z_exit_grid = np.linspace(z_exit_min, z_exit_max, int(grid_x)).tolist()

            with st.spinner("מריץ backtests לרגישות פרמטרים…"):
                sens_df = _run_param_sensitivity_grid(
                    sym_x=sym_x,
                    sym_y=sym_y,
                    base_kwargs=base_kwargs,
                    start_date=start_date,
                    end_date=end_date,
                    z_entry_grid=z_entry_grid,
                    z_exit_grid=z_exit_grid,
                )

            if sens_df.empty:
                st.info("לא התקבלו תוצאות לרגישות פרמטרים (בדוק טווחים/פרמטרים).")
            else:
                st.dataframe(sens_df, width="stretch", height=260)

                # Heatmap של Sharpe
                try:
                    import plotly.express as px  # type: ignore

                    pivot = sens_df.pivot_table(
                        index="z_exit",
                        columns="z_entry",
                        values="Sharpe",
                        aggfunc="mean",
                    )
                    fig_hm = px.imshow(
                        pivot.sort_index(ascending=True),
                        aspect="auto",
                        origin="lower",
                        labels=dict(x="z_entry", y="z_exit", color="Sharpe"),
                        title="Heatmap של Sharpe לפי z_entry / z_exit",
                    )
                    st.plotly_chart(fig_hm, use_container_width=True)
                except Exception as e:
                    st.caption(f"Heatmap plotting failed: {e}")

    # ===== 4) Run History & Comparison (Session-level) =====
    with st.expander("🧾 Run History & Comparison", expanded=False):
        # נשמור היסטוריה ב-session
        history: List[Dict[str, Any]] = st.session_state.get("bt_run_history", [])
        snap = {
            "ts": datetime.now(timezone.utc)().isoformat(),
            "sym_x": sym_x,
            "sym_y": sym_y,
            "Sharpe": float(result.metrics.get("Sharpe", 0.0)),
            "Profit": float(result.metrics.get("Profit", 0.0)),
            "Drawdown": float(result.metrics.get("Drawdown", 0.0)),
            "Trades": int(result.metrics.get("Trades", 0)),
        }
        history.append(snap)
        st.session_state["bt_run_history"] = history[-50:]  # נשמור רק 50 אחרונות

        hist_df = pd.DataFrame(st.session_state["bt_run_history"])
        if hist_df.empty:
            st.caption("אין עדיין היסטוריית ריצות (זו הראשונה).")
        else:
            st.dataframe(hist_df.tail(20), width="stretch", height=260)

            if len(hist_df) >= 2:
                last = hist_df.iloc[-1]
                prev = hist_df.iloc[-2]
                d_sh = float(last["Sharpe"] - prev["Sharpe"])
                d_p = float(last["Profit"] - prev["Profit"])
                d_dd = float(last["Drawdown"] - prev["Drawdown"])

                c1, c2, c3 = st.columns(3)
                c1.metric("Δ Sharpe (Last vs Prev)", f"{d_sh:+.2f}")
                c2.metric("Δ Profit", f"{d_p:+.0f}")
                c3.metric("Δ Max DD", f"{d_dd:+.2f}")

            # הורדה של היסטוריה
            try:
                st.download_button(
                    "💾 הורד היסטוריית Backtest (CSV)",
                    data=hist_df.to_csv(index=False).encode("utf-8"),
                    file_name="backtest_run_history.csv",
                    mime="text/csv",
                    key="bt_hist_dl",
                )
            except Exception:
                pass

# -----------------------------------------------------------------------
# 5.1 — Result panels: trades table, exports, history, audit
# -----------------------------------------------------------------------

def _build_bt_run_snapshot(result: BacktestResult) -> Dict[str, Any]:
    """יוצר snapshot קטן לריצה, לשימוש ב־history / audit."""
    return {
        "ts_utc": datetime.now(timezone.utc)().isoformat() + "Z",
        "symbols": list(result.symbols),
        "window": (
            result.window[0].isoformat() if result.window[0] else None,
            result.window[1].isoformat() if result.window[1] else None,
        ),
        "metrics": dict(result.metrics),
        "config": dict(result.config),
    }

def push_backtest_metrics_to_ctx(
    result: BacktestResult,
    ctx_key: str = "backtest_metrics",
) -> None:
    """
    שומר מדדי בקטסט רלוונטיים ב-session_state[ctx_key]
    כדי שטאבים אחרים (Home / Risk / Agents) יוכלו להשתמש בהם.

    מבנה לדוגמה:
        {
          "pair": "XLY-XLP",
          "sharpe": 1.42,
          "profit": 12345.0,
          "max_dd": 3456.0,
          "win_rate": 0.61,
          "trades": 53,
          "exposure_pct": 0.37,
          "window": "2019-01-01 → 2024-01-01",
          "last_run_utc": "2025-12-03T18:05:00Z",
        }
    """
    if st is None:
        return
    try:
        m = result.metrics or {}

        sharpe = float(m.get("Sharpe", 0.0))
        profit = float(m.get("Profit", 0.0))
        max_dd = float(m.get("Drawdown", 0.0))
        win_rate = float(m.get("WinRate", 0.0))
        trades = int(m.get("Trades", 0))
        exposure_pct = float(m.get("ExposurePct", 0.0))

        window_str = None
        try:
            w0, w1 = result.window
            if w0 and w1:
                window_str = f"{w0.date().isoformat()} → {w1.date().isoformat()}"
        except Exception:
            window_str = None

        payload: Dict[str, Any] = {
            "pair": f"{result.symbols[0]}-{result.symbols[1]}",
            "sharpe": sharpe,
            "profit": profit,
            "max_dd": max_dd,
            "win_rate": win_rate,
            "trades": trades,
            "exposure_pct": exposure_pct,
            "window": window_str,
        }
        try:
            payload["last_run_utc"] = datetime.now(timezone.utc)().isoformat(timespec="seconds") + "Z"
        except Exception:
            payload["last_run_utc"] = None

        st.session_state[ctx_key] = payload
    except Exception:
        # לא מפילים את הטאב בגלל שמירת מטא
        logger.debug("push_backtest_metrics_to_ctx failed", exc_info=True)

def _render_backtest_result_panels(
    result: BacktestResult,
    db_path: str = "",
    trades_table: str = "bt_trades",
    runs_table: str = "bt_runs",
) -> None:
    """
    מציג את התוצאה של הבקטסט:
    - KPIs
    - טבלת טריידים (עם בחירת מספר שורות)
    - ויזואליזציות מתקדמות
    - הורדות (CSV/JSON + Audit JSON)
    - לוגינג ל-DuckDB (אם path קיים)
    - השוואה לריצה קודמת מתוך history
    """
    if st is None:
        return

    _kpi_block(result)

    trades = result.trades.copy()
    if trades.empty:
        st.info("לא נוצרו שום טריידים עבור הפרמטרים האלה.")
        return

    # ===== Trades table =====
    st.subheader("📋 Trades")

    # בחירה כמה שורות להציג
    max_rows = st.slider(
        "מספר שורות להצגה",
        min_value=20,
        max_value=min(2000, len(trades)),
        value=min(200, len(trades)),
        step=20,
        key="bt_trades_rows",
    )
    # אפשר לבחור לסדר לפי גודל PnL
    sort_by_pnl = st.selectbox(
        "מיון טבלת טריידים",
        options=["Chronological", "PnL desc", "PnL asc"],
        index=0,
        key="bt_trades_sort",
    )
    df_view = trades.copy()
    if "pnl" in df_view.columns:
        if sort_by_pnl == "PnL desc":
            df_view = df_view.sort_values("pnl", ascending=False)
        elif sort_by_pnl == "PnL asc":
            df_view = df_view.sort_values("pnl", ascending=True)
    df_view = df_view.head(max_rows)

    # סטייליזציה קלה ל-PnL
    try:
        styled = df_view.style.applymap(
            lambda v: (
                "background-color:#d1ffd1" if isinstance(v, (int, float)) and v > 0
                else ("background-color:#ffd1d1" if isinstance(v, (int, float)) and v < 0 else "")
            ),
            subset=["pnl"],
        )  # type: ignore[arg-type]
        st.dataframe(styled, use_container_width=True)
    except Exception:
        st.dataframe(df_view, use_container_width=True)

    # ===== Visuals =====
    if go is not None:
        st.subheader("📊 Visual Analytics")
        _trades_visuals(trades, title_suffix=f"{result.symbols[0]}-{result.symbols[1]}")

    # ===== Downloads & Audit =====
    st.subheader("📦 Export & Audit")

    col_dl1, col_dl2, col_dl3 = st.columns(3)
    with col_dl1:
        csv = trades.to_csv(index=False).encode("utf-8")
        st.download_button(
            "💾 Download Trades (CSV)",
            data=csv,
            file_name=f"{result.symbols[0]}_{result.symbols[1]}_backtest_trades.csv",
            mime="text/csv",
            key="bt_dl_trades_csv",
        )
    with col_dl2:
        json_bytes = json.dumps(result.to_dict(), default=str, ensure_ascii=False, indent=2).encode("utf-8")
        st.download_button(
            "🧾 Download Result (JSON)",
            data=json_bytes,
            file_name=f"{result.symbols[0]}_{result.symbols[1]}_backtest_result.json",
            mime="application/json",
            key="bt_dl_result_json",
        )
    with col_dl3:
        snapshot = _build_bt_run_snapshot(result)
        audit_bytes = json.dumps(snapshot, default=str, ensure_ascii=False, indent=2).encode("utf-8")
        st.download_button(
            "🛡 Download Audit Snapshot (JSON)",
            data=audit_bytes,
            file_name=f"{result.symbols[0]}_{result.symbols[1]}_backtest_audit.json",
            mime="application/json",
            key="bt_dl_audit_json",
        )

    # ===== Logging to DuckDB =====
    if db_path:
        try:
            log_result_to_duckdb(result, db_path, trades_table=trades_table, runs_table=runs_table)
            st.success(f"נשמר ל-DuckDB ({db_path})")
        except Exception as e:
            st.warning(f"⚠️ DuckDB logging failed: {e}")

    # ===== Run history in session (for comparison) =====
    try:
        hist = st.session_state.get("bt_run_history", [])
        if not isinstance(hist, list):
            hist = []
        snap = _build_bt_run_snapshot(result)
        hist.append(snap)
        st.session_state["bt_run_history"] = hist
    except Exception:
        pass

    with st.expander("⏱ Run History (session) & Comparison", expanded=False):
        hist = st.session_state.get("bt_run_history", [])
        if not hist:
            st.caption("אין עדיין היסטוריית ריצות (session).")
        else:
            # נציג עד 10 ריצות אחרונות
            hist_tail = hist[-10:]
            df_hist = pd.DataFrame(
                [
                    {
                        "ts_utc": h.get("ts_utc"),
                        "sym_pair": "-".join(h.get("symbols") or []),
                        "Sharpe": h.get("metrics", {}).get("Sharpe"),
                        "Profit": h.get("metrics", {}).get("Profit"),
                        "MaxDD": h.get("metrics", {}).get("Drawdown"),
                        "Trades": h.get("metrics", {}).get("Trades"),
                    }
                    for h in hist_tail
                ]
            )
            st.dataframe(df_hist, use_container_width=True)

            if len(hist_tail) >= 2:
                last = hist_tail[-1]
                prev = hist_tail[-2]
                st.markdown("**השוואה בין הריצה האחרונה לקודמת:**")
                def _safe(v): 
                    try:
                        return float(v)
                    except Exception:
                        return float("nan")
                sh_last = _safe(last["metrics"]["Sharpe"])
                sh_prev = _safe(prev["metrics"]["Sharpe"])
                pf_last = _safe(last["metrics"]["Profit"])
                pf_prev = _safe(prev["metrics"]["Profit"])
                dd_last = _safe(last["metrics"]["Drawdown"])
                dd_prev = _safe(prev["metrics"]["Drawdown"])

                st.write(
                    {
                        "ΔSharpe": (sh_last - sh_prev) if np.isfinite(sh_last) and np.isfinite(sh_prev) else None,
                        "ΔProfit": (pf_last - pf_prev) if np.isfinite(pf_last) and np.isfinite(pf_prev) else None,
                        "ΔMaxDD": (dd_last - dd_prev) if np.isfinite(dd_last) and np.isfinite(dd_prev) else None,
                    }
                )


# -----------------------------------------------------------------------
# 5.2 — Streamlit UI: render_backtest_tab
# -----------------------------------------------------------------------

def render_backtest_tab(opt_params: Optional[Dict[str, Any]] = None) -> None:
    """
    Backtest Tab v3.6 — HF-grade, UX משופר, חיבור מלא לדשבורד + שדרוגים מתקדמים
    ============================================================================
    - כולל את כל הפיצ'רים של v3.5 +:
      1) חיווי חכם לזוג (ctx + selected_pair + היסטוריה).
      2) Recent pairs dropdown (מהיסטוריית ריצות).
      3) Date Presets (Custom / 1Y / 3Y / 5Y / YTD / 5Y Max).
      4) Risk Budget מקצועי (Portfolio equity + %).
      5) Performance Targets (Sharpe / Max DD) לריסק פוסטר.
      6) Config Presets לפי שם (save/load).
      7) Debug mode עם Raw Config.
      8) Pre-run Quality Checks לפני ריצה.
    """
    if st is None:
        raise RuntimeError("Streamlit is not installed in this environment.")

    st.header("🔁 Backtest — Pair Z-Score Mean Reversion (Pro)")

    # ===== 0. Merge incoming opt_params with session_state["opt_params"] =====
    base_params: Dict[str, Any] = {}
    if isinstance(st.session_state.get("opt_params"), dict):
        base_params.update(st.session_state.get("opt_params") or {})
    if isinstance(opt_params, dict):
        base_params.update(opt_params)

    # ===== 0.1 Pull context from dashboard ctx (if exists) =====
    ctx = st.session_state.get("ctx", {}) or {}
    ctx_start = ctx.get("start_date")
    ctx_end = ctx.get("end_date")

    # ===== 0.2 selected_pair → default sym_x/sym_y (כולל ctx) =====
    selected_pair_label = (
        st.session_state.get("selected_pair")
        or ctx.get("pair")
        or ctx.get("selected_pair")
    )

    def _split_pair(label: str) -> Tuple[Optional[str], Optional[str]]:
        if not isinstance(label, str):
            return None, None
        for sep in ("|", "/", "\\", ":", "-"):
            if sep in label:
                a, b = label.split(sep, 1)
                a, b = a.strip(), b.strip()
                if a and b:
                    return a, b
        return None, None

    # ברירות מחדל לזוג עבודה, כולל selected_pair והקשר
    sym_x_default = str(base_params.get("sym_x", "") or "").strip().upper()
    sym_y_default = str(base_params.get("sym_y", "") or "").strip().upper()

    if (not sym_x_default or not sym_y_default) and selected_pair_label:
        sx, sy = _split_pair(selected_pair_label)
        if sx and sy:
            sym_x_default = sx.upper()
            sym_y_default = sy.upper()

    if not sym_x_default:
        sym_x_default = "SPY"
    if not sym_y_default:
        sym_y_default = "QQQ"

    # נעדכן ב-base_params כדי שיירשם בהיסטוריה וכו'
    base_params.setdefault("sym_x", sym_x_default)
    base_params.setdefault("sym_y", sym_y_default)

    # ===== 0.3 היסטוריית קונפיגים לצורך Recent Pairs & Presets =====
    cfg_history: List[Dict[str, Any]] = st.session_state.get("bt_config_history", [])
    recent_pairs: List[str] = []
    for cfg in reversed(cfg_history[-50:]):  # עד 50 אחורה, לוקחים ייחודיים
        sx = str(cfg.get("sym_x") or "").strip().upper()
        sy = str(cfg.get("sym_y") or "").strip().upper()
        if sx and sy:
            p = f"{sx}-{sy}"
            if p not in recent_pairs:
                recent_pairs.append(p)
        if len(recent_pairs) >= 10:
            break

    # ===== 0.4 פונקציה להחזרת ערך ברירת מחדל בהיגיון משותף =====
    def _profile_defaults(p: str) -> Dict[str, Any]:
        p = p.lower().strip()
        if p == "defensive":
            return {
                "z_entry": 2.0,
                "z_exit": 0.7,
                "z_stop": 3.0,
                "run_dd_stop_pct": 0.15,
                "half_life_max": 300.0,
                "target_annual_vol": 0.08,
            }
        if p == "aggressive":
            return {
                "z_entry": 1.5,
                "z_exit": 0.3,
                "z_stop": 4.0,
                "run_dd_stop_pct": 0.30,
                "half_life_max": 800.0,
                "target_annual_vol": 0.18,
            }
        # default
        return {
            "z_entry": 2.0,
            "z_exit": 0.5,
            "z_stop": 3.5,
            "run_dd_stop_pct": 0.20,
            "half_life_max": 500.0,
            "target_annual_vol": 0.12,
        }

    def _get(name: str, fallback: Any) -> Any:
        # קודם מה-base_params, אחר כך מהפרופיל, ואחר כך fallback
        if name in base_params and base_params[name] not in (None, ""):
            return base_params[name]
        prof = str(base_params.get("bt_profile", "default"))
        prof_def = _profile_defaults(prof)
        if name in prof_def:
            return prof_def[name]
        return fallback

    # ===== 1. Layout =====
    col_left, col_right = st.columns([1.25, 1.75])

    # =======================================================================
    # LEFT — Parameters & Config
    # =======================================================================
    with col_left:
        st.subheader("⚙️ הגדרת בקטסט")

        # ---- 1.0 Pair context & Recent Pairs (חדש) ----
        st.caption(
            f"Pair context: ctx_pair='{ctx.get('pair', '-')}', selected_pair='{selected_pair_label or '-'}'"
        )


        if recent_pairs:
            recent_choice = st.selectbox(
                "Recent pairs (מהריצות האחרונות)",
                options=["(לא להשתמש)", *recent_pairs],
                index=0,
                key="bt_recent_pair",
                help="ניתן לבחור זוג אחרון מריצות קודמות כדי למלא את השדות אוטומטית.",
            )
            if recent_choice != "(לא להשתמש)":
                sx, sy = _split_pair(recent_choice)
                if sx and sy:
                    sym_x_default = sx.upper()
                    sym_y_default = sy.upper()
                    # נסנכרן גם ל-session_state.selected_pair
                    st.session_state["selected_pair"] = recent_choice

        # ---- 1.1 Risk Profile ----
        profile = st.selectbox(
            "Risk Profile",
            options=["default", "defensive", "aggressive"],
            index=["default", "defensive", "aggressive"].index(
                str(base_params.get("bt_profile", "default"))
            ),
            key="bt_profile",
            help="סט ברירות מחדל ל-Z, עצירות, DD וכו'. עדיין ניתן לערוך הכל ידנית.",
        )

        # מחשבים פרופיל שוב אחרי בחירת המשתמש
        prof_def = _profile_defaults(profile)

        # ---- 1.2 Run Mode Preset (Smoke/Baseline/Deep WF) ----
        st.markdown("**Run Mode Preset**")
        run_mode = st.radio(
            "Run Mode (Presets)",
            options=["Baseline", "Smoke Test", "Deep WF Prep"],
            index=0,
            key="bt_run_mode",
            help=(
                "Baseline — חלון מלא, הגדרות רגילות.\n"
                "Smoke Test — חלון קצר ומהיר לבדיקה.\n"
                "Deep WF Prep — פרמטרים מותאמים לניתוח Walk-Forward עמוק."
            ),
        )

        # ---- 1.3 Symbols ----
        c1, c2 = st.columns(2)
        with c1:
            sym_x = st.text_input(
                "Symbol X",
                value=sym_x_default,
                key="bt_sym_x",
            )
        with c2:
            sym_y = st.text_input(
                "Symbol Y",
                value=sym_y_default,
                key="bt_sym_y",
            )

        sym_x = (sym_x or "").strip().upper()
        sym_y = (sym_y or "").strip().upper()
        pair_label = f"{sym_x}-{sym_y}"

        # ---- 1.3.5 Link לאופטימיזציה (מועבר לכאן, אחרי שיש sym_x/sym_y) ----
        opt_entry = _get_opt_best_params_entry(pair_label)

        if opt_entry:
            score_txt = opt_entry.get("score")
            prof_txt = opt_entry.get("profile")
            updated = opt_entry.get("updated_at")
            st.markdown(f"✅ Found optimised parameters for **{pair_label}**")
            if score_txt is not None:
                st.caption(f"Best Score (opt): `{score_txt}`")
            if prof_txt:
                st.caption(f"Opt profile: `{prof_txt}`")
            if updated:
                st.caption(f"Updated at (opt): `{updated}`")

            use_opt_params = st.checkbox(
                "Use last optimised parameters for this pair",
                value=True,
                key="bt_use_opt_params",
                help="כאשר מסומן, הבקטסט יעדכן את הפרמטרים לפי האופטימיזציה האחרונה לזוג הזה.",
            )
            st.session_state["bt_use_opt_params_flag"] = bool(use_opt_params)
        else:
            st.session_state["bt_use_opt_params_flag"] = False
            st.caption(
                f"ℹ️ No optimised params found for `{pair_label}` in opt_best_params_registry."
            )

        # ---- 1.4 Dates + Date Presets ----
        start_default = (datetime.today() - timedelta(days=365 * 3)).date()
        end_default = datetime.today().date()

        # קונטקסט מה-ctx אם קיים
        if ctx_start is not None:
            try:
                start_default = pd.to_datetime(ctx_start).date()
            except Exception:
                pass
        if ctx_end is not None:
            try:
                end_default = pd.to_datetime(ctx_end).date()
            except Exception:
                pass

        date_preset = st.selectbox(
            "Date preset",
            options=["Custom", "1Y", "3Y", "5Y", "YTD", "5Y Max"],
            index=1 if base_params.get("date_preset") in (None, "", "3Y") else 0,
            key="bt_date_preset",
            help="חלונות זמן מהירים לבקטסט. Custom מאפשר לבחור היסטוריה חופשית.",
        )

        today = datetime.today().date()
        if date_preset == "Custom":
            start_val = _get("start_date", start_default)
            end_val = _get("end_date", end_default)
        else:
            end_val = today
            if date_preset == "1Y":
                start_val = today - timedelta(days=365)
            elif date_preset == "3Y":
                start_val = today - timedelta(days=365 * 3)
            elif date_preset == "5Y":
                start_val = today - timedelta(days=365 * 5)
            elif date_preset == "YTD":
                start_val = date(today.year, 1, 1)
            elif date_preset == "5Y Max":
                start_val = today - timedelta(days=365 * 5)
            else:
                start_val = start_default

        c3, c4 = st.columns(2)
        with c3:
            start_date = st.date_input(
                "Start",
                value=start_val,
                key="bt_start",
            )
        with c4:
            end_date = st.date_input(
                "End",
                value=end_val,
                key="bt_end",
            )

        # Smoke Test → קיצור חלון אקטיבי
        if run_mode == "Smoke Test":
            try:
                end_dt = datetime.combine(end_date, datetime.min.time())
                start_date = (end_dt - timedelta(days=180)).date()
            except Exception:
                pass

        # ---- 1.5 Core params ----
        st.markdown("**Core Parameters**")
        c5, c6 = st.columns(2)
        with c5:
            lookback = int(
                st.number_input(
                    "Lookback (bars)",
                    min_value=10,
                    max_value=252,
                    value=int(_get("lookback", 30)),
                    step=1,
                    key="bt_lookback",
                )
            )
            atr_window = int(
                st.number_input(
                    "ATR Window",
                    min_value=5,
                    max_value=100,
                    value=int(_get("atr_window", 14)),
                    step=1,
                    key="bt_atrwin",
                )
            )
        with c6:
            z_entry = float(
                st.number_input(
                    "Z Entry",
                    min_value=0.5,
                    max_value=5.0,
                    value=float(_get("z_entry", 2.0)),
                    step=0.1,
                    key="bt_zentry",
                )
            )
            z_exit = float(
                st.number_input(
                    "Z Exit",
                    min_value=0.1,
                    max_value=min(4.0, z_entry - 0.05),
                    value=float(_get("z_exit", 0.5)),
                    step=0.1,
                    key="bt_zexit",
                )
            )

        # ---- 1.6 Execution & Risk (כולל Risk Budget חדש) ----
        with st.expander("🚀 Execution & Risk", expanded=False):
            c7, c8, c9 = st.columns(3)
            with c7:
                notional = float(
                    st.number_input(
                        "Base Notional per Trade",
                        min_value=0.1,
                        max_value=1_000_000.0,
                        value=float(_get("notional", 10_000.0)),
                        step=1_000.0,
                        key="bt_notional",
                    )
                )
            with c8:
                bar_lag = int(
                    st.number_input(
                        "Bar Lag",
                        min_value=0,
                        max_value=10,
                        value=int(_get("bar_lag", 1)),
                        step=1,
                        key="bt_barlag",
                    )
                )
            with c9:
                max_bars_held = int(
                    st.number_input(
                        "Max Bars Held (0=off)",
                        min_value=0,
                        max_value=5000,
                        value=int(_get("max_bars_held", 0)),
                        step=1,
                        key="bt_maxbars",
                    )
                )

            c10, c11, c12 = st.columns(3)
            with c10:
                z_stop = float(
                    st.number_input(
                        "Z Stop (abs, 0=off)",
                        min_value=0.0,
                        max_value=10.0,
                        value=float(_get("z_stop", 0.0)),
                        step=0.1,
                        key="bt_zstop",
                    )
                )
            with c11:
                run_dd_stop_pct = float(
                    st.number_input(
                        "Run DD Stop % (0=off)",
                        min_value=0.0,
                        max_value=1.0,
                        value=float(_get("run_dd_stop_pct", 0.0)),
                        step=0.01,
                        key="bt_rundd",
                    )
                )
            with c12:
                slippage_mode = st.selectbox(
                    "Slippage Mode",
                    ["bps", "atr_frac"],
                    index=0 if _get("slippage_mode", "bps") == "bps" else 1,
                    key="bt_slipmode",
                )

            c13, c14, c15 = st.columns(3)
            with c13:
                slip_bps = float(
                    st.number_input(
                        "Slippage (bps)",
                        min_value=0.0,
                        max_value=500.0,
                        value=float(_get("slippage_bps", 0.0)),
                        step=0.5,
                        key="bt_slip",
                    )
                )
            with c14:
                slip_atr_frac = float(
                    st.number_input(
                        "Slippage ATR frac",
                        min_value=0.0,
                        max_value=1.0,
                        value=float(_get("slippage_atr_frac", 0.0)),
                        step=0.01,
                        key="bt_slipatr",
                    )
                )
            with c15:
                tc = float(
                    st.number_input(
                        "Transaction Cost per Trade",
                        min_value=0.0,
                        max_value=1_000.0,
                        value=float(_get("transaction_cost_per_trade", 0.0)),
                        step=0.1,
                        key="bt_tc",
                    )
                )

            c16, c17 = st.columns(2)
            with c16:
                target_ann_vol = float(
                    st.number_input(
                        "Target Annual Vol (0=off)",
                        min_value=0.0,
                        max_value=2.0,
                        value=float(_get("target_annual_vol", 0.0)),
                        step=0.01,
                        key="bt_tgtvol_input",
                    )
                )
            with c17:
                max_notional = float(
                    st.number_input(
                        "Max Notional (0=off)",
                        min_value=0.0,
                        max_value=10_000_000.0,
                        value=float(_get("notional_max", 0.0)),
                        step=10_000.0,
                        key="bt_ncap_input",
                    )
                )

            # --- Risk Budget (חדש) ---
            cRB1, cRB2 = st.columns(2)
            with cRB1:
                portfolio_equity_est = float(
                    st.number_input(
                        "Portfolio Equity (estimate)",
                        min_value=1_000.0,
                        max_value=100_000_000.0,
                        value=float(_get("portfolio_equity_est", 100_000.0)),
                        step=10_000.0,
                        key="bt_portfolio_equity_est",
                        help="שווי משוער של ההון עליו הסטרטגיה יושבת. משמש רק להערכת ריסק.",
                    )
                )
            with cRB2:
                risk_budget_pct = float(
                    st.slider(
                        "Risk Budget % (strategy slice)",
                        min_value=0.0,
                        max_value=25.0,
                        value=float(_get("risk_budget_pct", 5.0)),
                        step=0.5,
                        key="bt_risk_budget_pct",
                        help="כמה אחוז מתוך ההון הכולל מוקצה ל-setup הזה. אינפורמטיבי בלבד.",
                    )
                )

            try:
                risk_budget_abs = portfolio_equity_est * (risk_budget_pct / 100.0)
                approx_trades_slice = risk_budget_abs / max(notional, 1e-9)
                st.caption(
                    f"📐 Approx strategy budget ≈ {risk_budget_abs:,.0f} (≈ {approx_trades_slice:,.1f} בסיסי trades של {notional:,.0f})."
                )
            except Exception:
                st.caption("📐 Risk budget estimation unavailable (בעיה בחישוב).")

            warmup_bars = int(
                st.number_input(
                    "Warm-up bars (0=auto)",
                    min_value=0,
                    max_value=10000,
                    value=int(_get("warmup_bars", 0)),
                    step=10,
                    key="bt_warmup",
                )
            )

        # ---- 1.7 Statistical Gates ----
        with st.expander("📐 Statistical Gates (Cointegration / HL / Corr / Edge)", expanded=False):
            cS1, cS2, cS3 = st.columns(3)
            with cS1:
                coint_pmax = float(
                    st.number_input(
                        "Max Cointegration p (0=off)",
                        min_value=0.0,
                        max_value=1.0,
                        value=float(_get("coint_pmax", 0.0)),
                        step=0.01,
                        key="bt_cointp",
                    )
                )
            with cS2:
                hl_window = int(
                    st.number_input(
                        "HL Window",
                        min_value=10,
                        max_value=252,
                        value=int(_get("hl_window", lookback)),
                        step=1,
                        key="bt_hlwin",
                    )
                )
            with cS3:
                half_life_max = float(
                    st.number_input(
                        "Max Half-Life (bars, 0=off)",
                        min_value=0.0,
                        max_value=2000.0,
                        value=float(_get("half_life_max", 0.0)),
                        step=1.0,
                        key="bt_hlmax",
                    )
                )

            cS4, cS5, cS6 = st.columns(3)
            with cS4:
                corr_min = float(
                    st.number_input(
                        "Min Corr (entry, -1..1, 0=off)",
                        min_value=-1.0,
                        max_value=1.0,
                        value=float(_get("corr_min", 0.0)),
                        step=0.05,
                        key="bt_corr_min",
                    )
                )
            with cS5:
                beta_range = st.slider(
                    "Beta Range (0=disabled if unused)",
                    min_value=0.0,
                    max_value=3.0,
                    value=tuple(_get("beta_range", (0.5, 1.5))),
                    step=0.05,
                    key="bt_beta",
                )
            with cS6:
                edge_min = float(
                    st.number_input(
                        "Min Edge (entry, 0=off)",
                        min_value=0.0,
                        max_value=5.0,
                        value=float(_get("edge_min", 0.0)),
                        step=0.05,
                        key="bt_edge_min",
                    )
                )

            cS7, cS8 = st.columns(2)
            with cS7:
                atr_max = float(
                    st.number_input(
                        "Max ATR (entry, 0=off)",
                        min_value=0.0,
                        max_value=100.0,
                        value=float(_get("atr_max", 0.0)),
                        step=0.1,
                        key="bt_atrmax",
                    )
                )
            with cS8:
                atr_exit_max = float(
                    st.number_input(
                        "Max ATR (exit, 0=off)",
                        min_value=0.0,
                        max_value=100.0,
                        value=float(_get("atr_exit_max", 0.0)),
                        step=0.1,
                        key="bt_atr_exit",
                    )
                )

            edge_exit_min = float(
                st.number_input(
                    "Min Edge (exit drop, 0=off)",
                    min_value=0.0,
                    max_value=5.0,
                    value=float(_get("edge_exit_min", 0.0)),
                    step=0.05,
                    key="bt_edge_exit",
                )
            )

        # ---- 1.8 Logging (DuckDB) ----
        with st.expander("🪵 Logging (DuckDB)", expanded=False):
            cL1, cL2, cL3 = st.columns(3)
            with cL1:
                db_path = st.text_input(
                    "DB Path (ריק = ללא לוגינג)",
                    value=str(_get("db_path", "")),
                    key="bt_db",
                )
            with cL2:
                trades_table = st.text_input(
                    "Trades Table",
                    value=str(_get("trades_table", "bt_trades")),
                    key="bt_tbl_t",
                )
            with cL3:
                runs_table = st.text_input(
                    "Runs Table",
                    value=str(_get("runs_table", "bt_runs")),
                    key="bt_tbl_r",
                )

        # ---- 1.9 Entry / Exit conditions ----
        st.markdown("**Entry / Exit Conditions**")
        entry_conditions = st.multiselect(
            "Entry Conditions",
            ["Z-Score", "Edge", "ATR", "Correlation", "Beta"],
            default=_get("entry_conditions", ["Z-Score"]),
            key="bt_entry_cond",
        )
        exit_conditions = st.multiselect(
            "Exit Conditions",
            ["Z-Score", "Reversal Z", "ATR", "Edge Drop"],
            default=_get("exit_conditions", ["Z-Score"]),
            key="bt_exit_cond",
        )

        # Scenario label (for reports)
        scenario_name = st.text_input(
            "Scenario Label (optional)",
            value=str(base_params.get("scenario_name", "")),
            key="bt_scenario_label",
            help="לתייג ריצה, לדוגמה 'base', 'stress_vol_up', 'post-fomc' וכו'.",
        )

        # ---- 1.10 Performance Targets (חדש) ----
        cPT1, cPT2 = st.columns(2)
        with cPT1:
            target_sharpe = float(
                st.number_input(
                    "Target Sharpe (informal)",
                    min_value=0.0,
                    max_value=5.0,
                    value=float(_get("target_sharpe", 1.5)),
                    step=0.1,
                    key="bt_target_sharpe",
                    help="יעד Sharpe אינפורמטיבי בלבד, משמש לחיווי בריסק פוסטר בסוף.",
                )
            )
        with cPT2:
            max_dd_tolerance = float(
                st.number_input(
                    "Max acceptable DD %",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(_get("max_dd_tolerance", 0.25)),
                    step=0.05,
                    key="bt_max_dd_tol",
                    help="כמה דרואודאון יחסי (מרבי) מקובל עליך לריצה הזו (אינפורמטיבי בלבד).",
                )
            )

        # ---- 1.11 Config tools: history + export + presets + debug ----
        st.markdown("**Config Tools**")
        cH1, cH2, cH3 = st.columns(3)
        run_button = cH1.button("🚀 Run Backtest", key="bt_run")
        load_last_cfg = cH2.button("Load last config", key="bt_load_last_cfg")
        export_cfg = cH3.button("Export config JSON", key="bt_export_cfg")

        presets: Dict[str, Dict[str, Any]] = st.session_state.get("bt_presets", {})
        preset_name = st.text_input(
            "Preset name",
            value=str(base_params.get("preset_name", "")),
            key="bt_preset_name",
            help="ניתן לשמור/לטעון קונפיג לפי שם (למשל 'SPY-QQQ swing', 'defensive_3Y').",
        )
        cP1, cP2 = st.columns(2)
        save_preset_btn = cP1.button("💾 Save preset", key="bt_save_preset_btn")
        load_preset_btn = cP2.button("📂 Load preset", key="bt_load_preset_btn")

        debug_mode = st.checkbox(
            "Debug mode (show raw config)",
            value=bool(base_params.get("debug_mode", False)),
            key="bt_debug_mode",
        )

    # =======================================================================
    # RIGHT — Results, Context, Advanced Panels
    # =======================================================================
    with col_right:
        st.subheader("📊 Backtest Result")

        # ---- 2.1 Build current config dict (for hash/history/export) ----
        current_config = {
            "sym_x": sym_x,
            "sym_y": sym_y,
            "start_date": start_date,
            "end_date": end_date,
            "lookback": lookback,
            "atr_window": atr_window,
            "z_entry": z_entry,
            "z_exit": z_exit,
            "notional": notional,
            "bar_lag": bar_lag,
            "max_bars_held": max_bars_held,
            "z_stop": z_stop,
            "run_dd_stop_pct": run_dd_stop_pct,
            "slippage_mode": slippage_mode,
            "slippage_bps": slip_bps,
            "slippage_atr_frac": slip_atr_frac,
            "transaction_cost_per_trade": tc,
            "target_annual_vol": target_ann_vol,
            "notional_max": max_notional,
            "warmup_bars": warmup_bars,
            "coint_pmax": coint_pmax,
            "hl_window": hl_window,
            "half_life_max": half_life_max,
            "corr_min": corr_min,
            "beta_range": beta_range,
            "edge_min": edge_min,
            "atr_max": atr_max,
            "atr_exit_max": atr_exit_max,
            "edge_exit_min": edge_exit_min,
            "db_path": db_path,
            "trades_table": trades_table,
            "runs_table": runs_table,
            "entry_conditions": entry_conditions,
            "exit_conditions": exit_conditions,
            "bt_profile": profile,
            "run_mode": run_mode,
            "scenario_name": scenario_name,
            "portfolio_equity_est": portfolio_equity_est,
            "risk_budget_pct": risk_budget_pct,
            "date_preset": date_preset,
            "target_sharpe": target_sharpe,
            "max_dd_tolerance": max_dd_tolerance,
            "preset_name": preset_name,
            "debug_mode": debug_mode,
        }

        # שמירת קונפיג אחרון ל-tabs אחרים / Agents
        st.session_state["bt_last_config"] = current_config

        # Presets save/load
        if save_preset_btn and preset_name:
            presets[preset_name] = current_config.copy()
            st.session_state["bt_presets"] = presets
            st.success(f"Preset '{preset_name}' saved.")
        if load_preset_btn and preset_name and preset_name in presets:
            st.session_state["opt_params"] = presets[preset_name].copy()
            st.success(f"Preset '{preset_name}' loaded into opt_params — בצע Rerun לראות אותו בטופס.")

        # אם המשתמש בחר להשתמש בפרמטרים מאופטימיזציה → נעדכן את הקונפיג
        try:
            use_opt_params_flag = bool(st.session_state.get("bt_use_opt_params_flag", False))
        except Exception:
            use_opt_params_flag = False

        if use_opt_params_flag:
            opt_entry_for_pair = _get_opt_best_params_entry(pair_label)
            if opt_entry_for_pair:
                best_params = opt_entry_for_pair.get("params") or {}
                if isinstance(best_params, dict) and best_params:
                    for k, v in best_params.items():
                        if k in current_config:
                            current_config[k] = v
                    current_config["_used_opt_params"] = True
            else:
                current_config["_used_opt_params"] = False
        else:
            current_config["_used_opt_params"] = False

        # Config hash קטן לשימוש ב-Audit / Logs
        try:
            raw_cfg = json.dumps(
                {k: str(v) for k, v in current_config.items()},
                sort_keys=True,
                ensure_ascii=False,
            )
            cfg_hash = hashlib.sha256(raw_cfg.encode("utf-8")).hexdigest()[:12]
        except Exception:
            cfg_hash = "n/a"

        # ---- 2.2 Context Summary ----
        with st.expander("📋 Context Summary (read-only)", expanded=False):
            ctx_view = {
                "pair": f"{sym_x}-{sym_y}",
                "dates": f"{start_date} → {end_date}",
                "profile": profile,
                "run_mode": run_mode,
                "scenario": scenario_name or "(none)",
                "lookback": lookback,
                "z_entry": z_entry,
                "z_exit": z_exit,
                "notional": notional,
                "z_stop": z_stop if z_stop > 0 else None,
                "run_dd_stop_pct": run_dd_stop_pct if run_dd_stop_pct > 0 else None,
                "coint_pmax": coint_pmax if coint_pmax > 0 else None,
                "half_life_max": half_life_max if half_life_max > 0 else None,
                "corr_min": corr_min if corr_min != 0 else None,
                "edge_min": edge_min if edge_min > 0 else None,
                "atr_max": atr_max if atr_max > 0 else None,
                "target_annual_vol": target_ann_vol if target_ann_vol > 0 else None,
                "notional_max": max_notional if max_notional > 0 else None,
                "portfolio_equity_est": portfolio_equity_est,
                "risk_budget_pct": risk_budget_pct,
                "target_sharpe": target_sharpe,
                "max_dd_tolerance": max_dd_tolerance,
                "config_hash": cfg_hash,
                "used_opt_params": bool(current_config.get("_used_opt_params", False)),
            }
            st.json(ctx_view)

        # Debug raw config
        if debug_mode:
            with st.expander("🧪 Raw config (debug)", expanded=False):
                st.json(current_config)

        # ---- 2.3 Config history tools ----
        cfg_history = st.session_state.get("bt_config_history", [])
        if load_last_cfg and cfg_history:
            last_cfg = cfg_history[-1]
            st.session_state["opt_params"] = last_cfg
            st.success("Last config loaded into opt_params — בצע רענון (Rerun) כדי לראות אותו בטופס.")

        if export_cfg:
            cfg_json = json.dumps(
                current_config,
                default=str,
                ensure_ascii=False,
                indent=2,
            ).encode("utf-8")
            st.download_button(
                "Download config.json",
                data=cfg_json,
                file_name="backtest_config.json",
                mime="application/json",
                key="bt_dl_cfg_json",
            )

        # ---- 2.4 Run Backtest ----
        result: Optional[BacktestResult] = None
        base_kwargs_for_panels: Dict[str, Any] = {}

        if run_button:
            try:
                # שמירת opt_params מעודכנים
                st.session_state["opt_params"] = current_config

                # שמירת config בהיסטוריה (בצורה קומפקטית)
                cfg_history.append(current_config)
                st.session_state["bt_config_history"] = cfg_history[-30:]  # עד 30 אחרונות

                # Pre-run QC (לא חוסם, רק מזהיר)
                qc_warnings: List[str] = []
                if z_exit >= z_entry:
                    qc_warnings.append("Z Exit גדול או שווה ל-Z Entry — לא הגיוני ל-mean-reversion.")
                if start_date >= end_date:
                    qc_warnings.append("Start date מאוחר או שווה ל-End date.")
                try:
                    span_days = (datetime.combine(end_date, datetime.min.time()) -
                                 datetime.combine(start_date, datetime.min.time())).days
                    if span_days < 60:
                        qc_warnings.append(f"חלון זמן קצר מאוד ({span_days} ימים).")
                except Exception:
                    pass
                if corr_min > 0.9:
                    qc_warnings.append("Min Corr > 0.9 — ייתכן שאתה מסנן יותר מדי זוגות.")
                if half_life_max and half_life_max < lookback:
                    qc_warnings.append("Half-Life Max קטן מה-lookback — כדאי לבדוק שההיגיון מתאים.")

                if qc_warnings:
                    st.warning("Pre-run checks:\n- " + "\n- ".join(qc_warnings))

                with st.spinner("🏃 Running backtest…"):
                    result = run_backtest_with_vol(
                        target_annual_vol=target_ann_vol if target_ann_vol > 0 else None,
                        notional_max=max_notional if max_notional > 0 else None,
                        sym_x=sym_x,
                        sym_y=sym_y,
                        start_date=datetime.combine(start_date, datetime.min.time()),
                        end_date=datetime.combine(end_date, datetime.min.time()),
                        entry_conditions=entry_conditions,
                        exit_conditions=exit_conditions,
                        z_entry=z_entry,
                        z_exit=z_exit,
                        edge_min=edge_min if edge_min > 0 else None,
                        atr_max=atr_max if atr_max > 0 else None,
                        corr_min=corr_min if corr_min != 0 else None,
                        beta_range=beta_range,
                        edge_exit_min=edge_exit_min if edge_exit_min > 0 else None,
                        atr_exit_max=atr_exit_max if atr_exit_max > 0 else None,
                        lookback=lookback,
                        atr_window=atr_window,
                        coint_pmax=coint_pmax if coint_pmax > 0 else None,
                        hl_window=hl_window if hl_window > 0 else None,
                        half_life_max=half_life_max if half_life_max > 0 else None,
                        notional=notional,
                        slippage_bps=slip_bps,
                        slippage_mode=slippage_mode,
                        slippage_atr_frac=slip_atr_frac,
                        transaction_cost_per_trade=tc,
                        bar_lag=bar_lag,
                        max_bars_held=max_bars_held if max_bars_held > 0 else None,
                        z_stop=z_stop if z_stop > 0 else None,
                        run_dd_stop_pct=run_dd_stop_pct if run_dd_stop_pct > 0 else None,
                    )

                if result is not None:
                    # שימור ב-session
                    st.session_state["bt_last_result"] = result

                    base_kwargs_for_panels = {
                        "entry_conditions": entry_conditions,
                        "exit_conditions": exit_conditions,
                        "z_entry": z_entry,
                        "z_exit": z_exit,
                        "edge_min": edge_min if edge_min > 0 else None,
                        "atr_max": atr_max if atr_max > 0 else None,
                        "corr_min": corr_min if corr_min != 0 else None,
                        "beta_range": beta_range,
                        "edge_exit_min": edge_exit_min if edge_exit_min > 0 else None,
                        "atr_exit_max": atr_exit_max if atr_exit_max > 0 else None,
                        "lookback": lookback,
                        "atr_window": atr_window,
                        "coint_pmax": coint_pmax if coint_pmax > 0 else None,
                        "hl_window": hl_window if hl_window > 0 else None,
                        "half_life_max": half_life_max if half_life_max > 0 else None,
                        "notional": notional,
                        "slippage_bps": slip_bps,
                        "slippage_mode": slippage_mode,
                        "slippage_atr_frac": slip_atr_frac,
                        "transaction_cost_per_trade": tc,
                        "bar_lag": bar_lag,
                        "max_bars_held": max_bars_held if max_bars_held > 0 else None,
                        "z_stop": z_stop if z_stop > 0 else None,
                        "run_dd_stop_pct": run_dd_stop_pct if run_dd_stop_pct > 0 else None,
                    }

                    _render_backtest_result_panels(
                        result,
                        db_path=db_path,
                        trades_table=trades_table,
                        runs_table=runs_table,
                    )

                    try:
                        _render_backtest_advanced_panels(
                            result,
                            sym_x=sym_x,
                            sym_y=sym_y,
                            start_date=datetime.combine(start_date, datetime.min.time()),
                            end_date=datetime.combine(end_date, datetime.min.time()),
                            base_kwargs=base_kwargs_for_panels,
                        )
                    except Exception as e:
                        st.caption(f"Advanced panels failed (non-fatal): {e}")

                    # Risk posture משודרג לפי היעדים
                    try:
                        m = result.metrics or {}
                        sh = float(m.get("Sharpe", 0.0))
                        dd_val = float(m.get("Drawdown", 0.0))
                        risk_note = []

                        if sh < 0:
                            risk_note.append("❌ Strategy losing money on this window.")
                        elif sh < 1:
                            risk_note.append("⚠ Sharpe < 1 — marginal edge.")
                        else:
                            risk_note.append("✅ Sharpe looks acceptable.")

                        if target_sharpe > 0:
                            if sh >= target_sharpe:
                                risk_note.append(
                                    f"🎯 Sharpe {sh:.2f} עומד מעל היעד ({target_sharpe:.2f})."
                                )
                            else:
                                risk_note.append(
                                    f"ℹ Sharpe {sh:.2f} מתחת ליעד ({target_sharpe:.2f})."
                                )

                        if dd_val > 0:
                            risk_note.append(f"Max DD ≈ {dd_val:,.2f}.")
                            if max_dd_tolerance > 0:
                                if dd_val <= max_dd_tolerance:
                                    risk_note.append("📉 DD בתוך טווח הסבילות שהוגדר.")
                                else:
                                    risk_note.append("⚠ DD גבוה מהסבילות שהוגדרה לריצה הזו.")

                        st.markdown("**Risk Posture (informal):**")
                        st.write(" ".join(risk_note))
                    except Exception:
                        pass

                    # --- דחיפת מדדי בקטסט ל-ctx הגלובלי (Home / Risk / Agents) ---
                    try:
                        push_backtest_metrics_to_ctx(result)
                    except Exception:
                        logger.debug("push_backtest_metrics_to_ctx failed (non-fatal)", exc_info=True)

            except Exception as e:
                st.error(f"Backtest error: {e}")
                logger.exception("Backtest tab error", exc_info=e)
        else:
            # אם יש תוצאה קודמת ב-session — נציג אותה (Quality-of-life UX)
            result_prev = st.session_state.get("bt_last_result")
            if isinstance(result_prev, BacktestResult):
                st.caption("מוצגת הריצה האחרונה (bt_last_result מה-session).")
                _render_backtest_result_panels(
                    result_prev,
                    db_path=_get("db_path", ""),
                    trades_table=_get("trades_table", "bt_trades"),
                    runs_table=_get("runs_table", "bt_runs"),
                )

                # ננסה גם Advanced Panels על הריצה האחרונה (אם יש לנו start/end ב-ctx)
                try:
                    sd_prev = ctx_start or start_default
                    ed_prev = ctx_end or end_default
                    sd_prev_dt = datetime.combine(pd.to_datetime(sd_prev).date(), datetime.min.time())
                    ed_prev_dt = datetime.combine(pd.to_datetime(ed_prev).date(), datetime.min.time())

                    base_kwargs_for_panels = {
                        "entry_conditions": base_params.get("entry_conditions", ["Z-Score"]),
                        "exit_conditions": base_params.get("exit_conditions", ["Z-Score"]),
                        "z_entry": float(base_params.get("z_entry", 2.0)),
                        "z_exit": float(base_params.get("z_exit", 0.5)),
                        "edge_min": base_params.get("edge_min"),
                        "atr_max": base_params.get("atr_max"),
                        "corr_min": base_params.get("corr_min"),
                        "beta_range": base_params.get("beta_range"),
                        "edge_exit_min": base_params.get("edge_exit_min"),
                        "atr_exit_max": base_params.get("atr_exit_max"),
                        "lookback": int(base_params.get("lookback", 30)),
                        "atr_window": int(base_params.get("atr_window", 14)),
                        "coint_pmax": base_params.get("coint_pmax"),
                        "hl_window": base_params.get("hl_window"),
                        "half_life_max": base_params.get("half_life_max"),
                        "notional": base_params.get("notional", 10_000.0),
                        "slippage_bps": base_params.get("slippage_bps", 0.0),
                        "slippage_mode": base_params.get("slippage_mode", "bps"),
                        "slippage_atr_frac": base_params.get("slippage_atr_frac", 0.0),
                        "transaction_cost_per_trade": base_params.get("transaction_cost_per_trade", 0.0),
                        "bar_lag": base_params.get("bar_lag", 1),
                        "max_bars_held": base_params.get("max_bars_held"),
                        "z_stop": base_params.get("z_stop"),
                        "run_dd_stop_pct": base_params.get("run_dd_stop_pct"),
                    }

                    _render_backtest_advanced_panels(
                        result_prev,
                        sym_x=result_prev.symbols[0],
                        sym_y=result_prev.symbols[1],
                        start_date=sd_prev_dt,
                        end_date=ed_prev_dt,
                        base_kwargs=base_kwargs_for_panels,
                    )
                except Exception:
                    pass
            else:
                st.info("הגדר פרמטרים בצד שמאל ולחץ 🚀 Run Backtest כדי לראות תוצאות.")

# =======================================================================
# Part 6/6 — Integrations: DuckDB Logging, Programmatic API & CLI
# =======================================================================

def log_result_to_duckdb(
    result: BacktestResult,
    db_path: str,
    trades_table: str = "bt_trades",
    runs_table: str = "bt_runs",
    run_id: Optional[str] = None,
) -> None:
    """
    Persist trades and run metadata to DuckDB. No-op אם duckdb לא מותקן.

    Parameters
    ----------
    result : BacktestResult
        תוצאת הבקטסט (כולל trades ו-metrics).
    db_path : str
        נתיב לקובץ DuckDB (למשל 'data/backtests.duckdb').
    trades_table : str
        שם טבלת הטריידים (ברירת מחדל: 'bt_trades').
    runs_table : str
        שם טבלת הריצות (ברירת מחדל: 'bt_runs').
    run_id : str | None
        מזהה ריצה ייחודי; אם None ניצור כזה לפי timestamp.

    Notes
    -----
    - אם הטבלאות לא קיימות — ניצור אותן לפי הסכמה של הריצה הראשונה.
    - השדות metrics/config נשמרים כ-JSON כדי לאפשר הרחבה גמישה.
    """
    if duckdb is None:
        logger.warning("duckdb not available — skipping DuckDB logging")
        return

    try:
        con = duckdb.connect(db_path)
    except Exception as e:
        logger.warning("Failed to connect DuckDB at %s: %s", db_path, e)
        return

    try:
        rid = run_id or result.run_id or f"run_{datetime.now(timezone.utc)().isoformat()}"

        # --- Runs table: שורה אחת לכל ריצה ---
        run_row = pd.DataFrame(
            [
                {
                    "run_id": rid,
                    "ts": datetime.now(timezone.utc)(),
                    "strategy": result.strategy.value,
                    "sym_x": result.symbols[0],
                    "sym_y": result.symbols[1],
                    "start": result.window[0],
                    "end": result.window[1],
                    "metrics_json": json.dumps(result.metrics, ensure_ascii=False),
                    "config_json": json.dumps(result.config, default=str, ensure_ascii=False),
                }
            ]
        )

        con.execute(
            f"CREATE TABLE IF NOT EXISTS {runs_table} AS SELECT * FROM run_row LIMIT 0"
        )
        con.register("run_row", run_row)
        con.execute(f"INSERT INTO {runs_table} SELECT * FROM run_row")

        # --- Trades table: הרבה שורות לכל ריצה ---
        trades = result.trades.copy()
        trades["run_id"] = rid
        # לוודא שעכשיו יש Index טורי ולא MultiIndex מוזר
        trades = trades.reset_index(drop=True)

        con.execute(
            f"CREATE TABLE IF NOT EXISTS {trades_table} AS SELECT * FROM trades LIMIT 0"
        )
        con.register("trades", trades)
        con.execute(f"INSERT INTO {trades_table} SELECT * FROM trades")

    except Exception as e:
        logger.warning("DuckDB logging failed: %s", e)
    finally:
        try:
            con.close()
        except Exception:
            pass


# -------------------------------------------------------------------
# Programmatic API — convenient wrappers for agents / scripts
# -------------------------------------------------------------------

def api_run_backtest(params: Dict[str, Any]) -> BacktestResult:
    """
    API פרוגרמטי נוח להפעלת בקטסט מתוך סקריפט / Agent.

    Parameters
    ----------
    params : dict
        מילון שמייצג BacktestParams. אפשר חלקי — חסרים יקבלו ברירות מחדל.

        דוגמאות-מפתח שימושיים:
        - 'sym_x', 'sym_y' (חובה)
        - 'start_date', 'end_date' (datetime או str ISO)
        - 'z_entry', 'z_exit', 'lookback', 'atr_window'
        - 'corr_min', 'beta_range', 'edge_min', 'atr_max'
        - 'edge_exit_min', 'atr_exit_max'
        - 'coint_pmax', 'hl_window', 'half_life_max'
        - exec & risk: 'notional', 'slippage_bps', 'slippage_mode',
          'slippage_atr_frac', 'transaction_cost_per_trade',
          'bar_lag', 'max_bars_held', 'z_stop', 'run_dd_stop_pct'
        - VT: 'target_annual_vol', 'notional_max'

    Returns
    -------
    BacktestResult
        תוצאת הבקטסט המלאה לשימוש חיצוני.
    """
    # נוודא שזוג הסימבולים קיים
    if "sym_x" not in params or "sym_y" not in params:
        raise ValueError("api_run_backtest: 'sym_x' ו-'sym_y' הם שדות חובה.")

    # המרה רכה של start/end במידה והגיעו כמחרוזות
    def _maybe_parse_dt(val: Any) -> Optional[datetime]:
        if val is None:
            return None
        if isinstance(val, datetime):
            return val
        try:
            return datetime.fromisoformat(str(val))
        except Exception:
            return None

    start_date = _maybe_parse_dt(params.get("start_date"))
    end_date = _maybe_parse_dt(params.get("end_date"))

    # Volatility targeting מפורש, אם הועבר
    tvol = params.get("target_annual_vol")
    ncap = params.get("notional_max")

    # מפרידים פרמטרי VT משאר kwargs
    kwargs = dict(params)
    kwargs.pop("target_annual_vol", None)
    kwargs.pop("notional_max", None)
    kwargs["start_date"] = start_date
    kwargs["end_date"] = end_date

    # אם יש VT → נשתמש ב-run_backtest_with_vol; אחרת run_backtest רגיל
    if tvol or ncap:
        return run_backtest_with_vol(
            target_annual_vol=float(tvol) if tvol else None,
            notional_max=float(ncap) if ncap else None,
            **kwargs,  # type: ignore[arg-type]
        )
    return run_backtest(**kwargs)  # type: ignore[arg-type]


def api_backtest_to_frames(params: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Wrapper שמחזיר (trades_df, metrics_dict) בצורה נוחה לניתוח מהיר.

    Example
    -------
        trades, metrics = api_backtest_to_frames({
            "sym_x": "XLY",
            "sym_y": "XLP",
            "lookback": 60,
            "z_entry": 2.0,
            "z_exit": 0.5,
        })
    """
    res = api_run_backtest(params)
    return res.trades.copy(), dict(res.metrics)


# -------------------------------------------------------------------
# CLI Entrypoint — run backtests from terminal
# -------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pairs Z-Score Backtester (HF-grade)")

    # Required symbols
    parser.add_argument("sym_x", type=str, help="First leg symbol (X)")
    parser.add_argument("sym_y", type=str, help="Second leg symbol (Y)")

    # Dates
    parser.add_argument("--start", type=str, default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD)")

    # Core
    parser.add_argument("--lookback", type=int, default=30)
    parser.add_argument("--atr_window", type=int, default=14)
    parser.add_argument("--z_entry", type=float, default=2.0)
    parser.add_argument("--z_exit", type=float, default=0.5)

    # Filters / gates
    parser.add_argument("--corr_min", type=float, default=None)
    parser.add_argument("--beta_min", type=float, default=None)
    parser.add_argument("--beta_max", type=float, default=None)
    parser.add_argument("--edge_min", type=float, default=None)
    parser.add_argument("--atr_max", type=float, default=None)
    parser.add_argument("--edge_exit_min", type=float, default=None)
    parser.add_argument("--atr_exit_max", type=float, default=None)
    parser.add_argument("--coint_pmax", type=float, default=None)
    parser.add_argument("--hl_window", type=int, default=None)
    parser.add_argument("--half_life_max", type=float, default=None)

    # Exec & risk
    parser.add_argument("--notional", type=float, default=10_000.0)
    parser.add_argument("--slippage_bps", type=float, default=0.0)
    parser.add_argument("--slippage_mode", type=str, choices=["bps", "atr_frac"], default="bps")
    parser.add_argument("--slippage_atr_frac", type=float, default=0.0)
    parser.add_argument("--transaction_cost_per_trade", type=float, default=0.0)
    parser.add_argument("--bar_lag", type=int, default=1)
    parser.add_argument("--max_bars_held", type=int, default=0)
    parser.add_argument("--z_stop", type=float, default=0.0)
    parser.add_argument("--run_dd_stop_pct", type=float, default=0.0)

    # Volatility targeting (CLI)
    parser.add_argument("--target_annual_vol", type=float, default=None)
    parser.add_argument("--notional_max", type=float, default=None)

    # Output
    parser.add_argument("--json_out", type=str, default="", help="Path to save JSON result")
    parser.add_argument("--db_path", type=str, default="", help="Optional DuckDB path for logging")
    parser.add_argument("--trades_table", type=str, default="bt_trades")
    parser.add_argument("--runs_table", type=str, default="bt_runs")

    args = parser.parse_args()

    # Parse dates
    sd = datetime.fromisoformat(args.start) if args.start else None
    ed = datetime.fromisoformat(args.end) if args.end else None

    # Build params dict
    params_cli: Dict[str, Any] = {
        "sym_x": args.sym_x,
        "sym_y": args.sym_y,
        "start_date": sd,
        "end_date": ed,
        "lookback": args.lookback,
        "atr_window": args.atr_window,
        "z_entry": args.z_entry,
        "z_exit": args.z_exit,
        "corr_min": args.corr_min,
        "beta_range": (
            (args.beta_min, args.beta_max)
            if args.beta_min is not None and args.beta_max is not None
            else None
        ),
        "edge_min": args.edge_min,
        "atr_max": args.atr_max,
        "edge_exit_min": args.edge_exit_min,
        "atr_exit_max": args.atr_exit_max,
        "coint_pmax": args.coint_pmax,
        "hl_window": args.hl_window,
        "half_life_max": args.half_life_max,
        "notional": args.notional,
        "slippage_bps": args.slippage_bps,
        "slippage_mode": args.slippage_mode,
        "slippage_atr_frac": args.slippage_atr_frac,
        "transaction_cost_per_trade": args.transaction_cost_per_trade,
        "bar_lag": args.bar_lag,
        "max_bars_held": args.max_bars_held if args.max_bars_held > 0 else None,
        "z_stop": args.z_stop if args.z_stop > 0 else None,
        "run_dd_stop_pct": args.run_dd_stop_pct if args.run_dd_stop_pct > 0 else None,
        "target_annual_vol": args.target_annual_vol,
        "notional_max": args.notional_max,
    }

    # Run via API wrapper (כולל VT אם הוגדר)
    res = api_run_backtest(params_cli)

    # לוגינג ל-DuckDB אם ביקשו
    if args.db_path:
        try:
            log_result_to_duckdb(
                res,
                args.db_path,
                trades_table=args.trades_table,
                runs_table=args.runs_table,
            )
        except Exception as e:
            logger.warning("DuckDB logging from CLI failed: %s", e)

    # Output JSON (stdout + אופציונלי לקובץ)
    payload = json.dumps(res.to_dict(), default=str, ensure_ascii=False)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            f.write(payload)
    print(payload)
