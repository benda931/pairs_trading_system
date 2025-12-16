# -*- coding: utf-8 -*-
"""
core/optimization_backtester.py — Hedge-Fund Grade Pairs Backtester (Part 1/6)
==============================================================================

Purpose
-------
This module provides the *execution engine* for a single pairs-trading
backtest, designed to be driven by the optimization tab (Optuna / meta-opt).

The overall design (Parts 1–6)
------------------------------
1. (THIS PART)  Core models, config, result schema, and backtester skeleton.
2. Data loading layer: price history, spread construction, sanity checks.
3. Signal & position engine: entries/exits, position sizing, transaction costs.
4. Risk & metrics layer: PnL curve, drawdowns, Sharpe/Sortino/Calmar, exposures.
5. Advanced overlays: volatility/beta normalization, scenario engine, stress tests.
6. Integration & helpers: convenience wrappers, debug tools, aliases for UI.

Key invariants
--------------
- Public entrypoint: `OptimizationBacktester.run() -> dict[str, float]`
  returning at least: {"Sharpe", "Profit", "Drawdown"}.
- All extra metrics live in BacktestResult.stats and are *optional* for the UI.
- The class is parameter-driven: any **params passed from the optimization tab
  are stored and later used (lookback, z_open, z_close, stop_z, hr_method, etc.).

NOTE (for now)
--------------
This Part 1 contains a *safe placeholder* implementation of `run()` so that:
- Imports from root/optimization_tab.py **do not break**.
- The dashboard can start up even before Parts 2–6 are pasted.
Later parts will replace the internals of `run()` with a full production
backtest pipeline, without changing its public signature.
"""

from __future__ import annotations

# =========================
# SECTION 0: Imports & Type Aliases
# =========================

from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, TypeAlias

import logging
import math
import os

import numpy as np
import pandas as pd

# Optional internal loader (will be used in later parts)
try:
    from common.utils import load_price_data as _load_price_data  # type: ignore[attr-defined]
except Exception:
    _load_price_data = None  # type: ignore[assignment]

try:
    from core.sql_store import SqlStore  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - optional
    SqlStore = None  # type: ignore[assignment]

try:
    # פונקציה שתכתוב ב-core.ib_data_ingestor ותהיה אחראית להביא דאטה מ-IBKR
    from core.ib_data_ingestor import (
        load_pair_prices_for_backtest as _ib_load_pair_prices_for_backtest,  # type: ignore[attr-defined]
    )
except Exception:  # pragma: no cover - optional
    _ib_load_pair_prices_for_backtest = None  # type: ignore[assignment]

# ---- Logging setup (can be overridden by dashboard/settings) ----
logger = logging.getLogger("OptimizationBacktester")
if not logger.handlers:
    logging.basicConfig(
        level=getattr(logging, os.environ.get("OPT_BT_LOG_LEVEL", "INFO").upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

# ---- Type aliases ----
NDArray: TypeAlias = np.ndarray
SeriesLike: TypeAlias = pd.Series | NDArray
FrameLike: TypeAlias = pd.DataFrame

# =========================
# SECTION 1: Utility helpers (base math & sanity)
# =========================

def _safe_annualization_factor(periods_per_year: float | int) -> float:
    """Return a safe annualization factor (sqrt for returns-based ratios)."""
    try:
        p = float(periods_per_year)
        return math.sqrt(p) if p > 0 else 1.0
    except Exception:
        return 1.0


def _safe_div(num: float, den: float, default: float = 0.0) -> float:
    """Robust division helper used by risk ratios."""
    try:
        if den == 0 or np.isnan(den):
            return default
        return float(num) / float(den)
    except Exception:
        return default


def _ensure_date(d: Any) -> Optional[date]:
    """Normalize various date-ish inputs to a `date`, or None."""
    if d is None:
        return None
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    try:
        return datetime.fromisoformat(str(d)).date()
    except Exception:
        return None


# =========================
# SECTION 2: Backtest Config Model
# =========================

@dataclass
class BacktestConfig:
    """
    Configuration for a *single* pair backtest.

    This is intentionally rich so the engine can support multiple "styles"
    (mean-reversion, stat-arb, beta-hedged overlay, volatility-scaled, etc.)
    without changing the public interface.

    Core identifiers
    ----------------
    symbol_a / symbol_b : tickers of the pair legs.

    Time window
    -----------
    start / end : inclusive date range; if None → inferred from data.
    min_history_days : minimal required number of calendar days.

    Capital & risk
    --------------
    initial_capital      : starting equity (base currency).
    max_gross_exposure   : hard cap on |long| + |short|.
    target_gross_leverage: optional leverage target (gross / equity).
    risk_free_rate       : annualized RF for excess-return ratios.

    Microstructure & costs
    ----------------------
    commission_bps : per-side commission (basis points).
    slippage_bps   : assumed slippage (basis points).
    lot_size       : min trade size (for rounding / lotting).

    Neutrality / structure
    ----------------------
    dollar_neutral : maintain dollar neutrality by design.
    beta_neutral   : maintain beta neutrality (future extension).
    rebalance_days : optional rebalance frequency (for baskets).

    Engine / frequency
    ------------------
    bar_freq       : base frequency ("D", "H", etc.).
    data_source    : logical name of the market data source.
    """

    # ---- identity ----
    symbol_a: str
    symbol_b: str

    # ---- time ----
    start: Optional[date] = None
    end: Optional[date] = None
    min_history_days: int = 120

    # ---- capital & risk ----
    initial_capital: float = float(os.environ.get("OPT_BT_INIT_CAPITAL", 100_000))
    max_gross_exposure: float = float(os.environ.get("OPT_BT_MAX_GROSS", 300_000))
    target_gross_leverage: float = 1.0
    risk_free_rate: float = 0.02  # 2% RF as default

    # ---- costs ----
    commission_bps: float = 1.0
    slippage_bps: float = 2.0
    lot_size: float = 1.0

    # ---- structure ----
    dollar_neutral: bool = True
    beta_neutral: bool = False
    rebalance_days: int = 0  # 0 = "no periodic rebalance overlay"

    # ---- engine ----
    bar_freq: str = "D"
    data_source: str = "AUTO"  # "IB", "YF", "PARQUET", etc.

    # ---- extensibility ----
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Normalize dates
        self.start = _ensure_date(self.start)
        self.end = _ensure_date(self.end)

        if self.start and self.end and self.end < self.start:
            raise ValueError(f"BacktestConfig: end < start ({self.start} → {self.end})")

        if self.initial_capital <= 0:
            raise ValueError("BacktestConfig: initial_capital must be > 0")

        if self.max_gross_exposure <= 0:
            raise ValueError("BacktestConfig: max_gross_exposure must be > 0")

        if self.min_history_days <= 0:
            self.min_history_days = 60  # reasonable floor

        if not self.symbol_a or not self.symbol_b:
            raise ValueError("BacktestConfig: symbol_a and symbol_b must be non-empty")

    def to_dict(self) -> Dict[str, Any]:
        """Export to plain dict (dates → ISO)."""
        d = asdict(self)
        for key in ("start", "end"):
            v = d.get(key)
            if isinstance(v, (date, datetime)):
                d[key] = v.isoformat()
        return d


# SECTION 3: Backtest Result Model
# =========================

@dataclass
class BacktestResult:
    """
    Normalized, hedge-fund grade backtest result for a single pair.

    Required core metrics (used by the optimization tab)
    ----------------------------------------------------
    sharpe        : annualized Sharpe ratio of equity curve.
    profit        : final PnL in currency terms.
    max_drawdown  : max peak-to-trough drawdown as *positive* fraction (0–1).

    Extended metrics (optional but highly recommended)
    --------------------------------------------------
    sortino       : Sortino ratio (downside-vol based).
    calmar        : Calmar ratio (CAGR / |max_drawdown|).
    ulcer_index   : Ulcer index (drawdown severity).
    total_return  : Total return over the backtest horizon.

    Trade/behaviour metrics
    -----------------------
    win_rate      : % of winning trades (0–1).
    n_trades      : # of completed trades.
    avg_trade_pnl : average PnL per trade (currency).
    exposure      : average gross exposure / capital (0–X).
    turnover      : average notional traded / capital.

    Tail-risk metrics
    -----------------
    var_95        : 95% historical Value-at-Risk (one-period loss, >0 = loss).
    es_95         : 95% historical Expected Shortfall (average tail loss).

    Additional data
    ---------------
    equity_curve   : index-aligned equity series.
    returns        : aligned returns series (for metrics / WF).
    window_metrics : list of per-window metrics for WF / stability.
    regime_stats   : per-regime aggregated stats (if available).
    stats          : arbitrary structured metrics (JSON-safe).
    """

    # ---- core ----
    sharpe: float
    profit: float
    max_drawdown: float  # positive fraction (e.g. 0.23 for -23%)

    # ---- extended risk/return ----
    sortino: float = 0.0
    calmar: float = 0.0
    ulcer_index: float = 0.0
    total_return: float = 0.0

    # ---- tail risk ----
    var_95: float = 0.0
    es_95: float = 0.0

    # ---- trade stats ----
    win_rate: float = 0.0
    n_trades: int = 0
    avg_trade_pnl: float = 0.0

    # ---- exposure / turnover ----
    exposure: float = 0.0
    turnover: float = 0.0

    # ---- detailed outputs ----
    equity_curve: Optional[pd.Series] = None
    returns: Optional[pd.Series] = None
    window_metrics: List[Dict[str, Any]] = field(default_factory=list)
    regime_stats: Dict[str, Any] = field(default_factory=dict)
    stats: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Normalize NaNs / infinities into safe values for scalar metrics
        scalar_fields = (
            "sharpe",
            "profit",
            "max_drawdown",
            "sortino",
            "calmar",
            "ulcer_index",
            "total_return",
            "var_95",
            "es_95",
            "win_rate",
            "avg_trade_pnl",
            "exposure",
            "turnover",
        )
        for fld in scalar_fields:
            val = getattr(self, fld)
            if val is None or (
                isinstance(val, (float, int))
                and (math.isnan(val) or math.isinf(val))
            ):
                setattr(self, fld, 0.0)

        # Drawdown is stored as positive fraction (0–1)
        if self.max_drawdown < 0:
            self.max_drawdown = abs(self.max_drawdown)

        # Clamp basic probabilities / ratios to sane bounds
        self.win_rate = float(min(max(self.win_rate, 0.0), 1.0))
        if self.n_trades < 0:
            self.n_trades = 0

        # risk / tail metrics must be non-negative
        self.exposure = max(self.exposure, 0.0)
        self.turnover = max(self.turnover, 0.0)
        self.ulcer_index = max(self.ulcer_index, 0.0)
        self.var_95 = max(self.var_95, 0.0)
        self.es_95 = max(self.es_95, 0.0)

        # Ensure container attributes are well-formed
        if self.window_metrics is None:
            self.window_metrics = []
        if self.regime_stats is None:
            self.regime_stats = {}
        if self.stats is None:
            self.stats = {}

    # ---- minimal interface expected by optimization_tab ----
    def to_perf_dict(self) -> Dict[str, float]:
        """
        The minimal dict required by the optimization UI.
        """
        return {
            "Sharpe": float(self.sharpe),
            "Profit": float(self.profit),
            "Drawdown": float(self.max_drawdown),
        }

    # ---- richer export for logs / storage ----
    def to_full_dict(self) -> Dict[str, Any]:
        """Full export, including extended stats and equity / returns."""
        out: Dict[str, Any] = {
            "Sharpe": float(self.sharpe),
            "Profit": float(self.profit),
            "Drawdown": float(self.max_drawdown),
            "Sortino": float(self.sortino),
            "Calmar": float(self.calmar),
            "UlcerIndex": float(self.ulcer_index),
            "TotalReturn": float(self.total_return),
            "WinRate": float(self.win_rate),
            "Trades": int(self.n_trades),
            "AvgTradePnL": float(self.avg_trade_pnl),
            "Exposure": float(self.exposure),
            "Turnover": float(self.turnover),
            "VaR_95": float(self.var_95),
            "ES_95": float(self.es_95),
            "stats": dict(self.stats),
        }

        # optional rich outputs
        if isinstance(self.equity_curve, pd.Series):
            out["equity_index"] = self.equity_curve.index.astype(str).tolist()
            out["equity_values"] = self.equity_curve.astype(float).tolist()

        if isinstance(self.returns, pd.Series):
            out["returns_index"] = self.returns.index.astype(str).tolist()
            out["returns_values"] = self.returns.astype(float).tolist()

        if self.window_metrics:
            out["window_metrics"] = list(self.window_metrics)
        if self.regime_stats:
            out["regime_stats"] = dict(self.regime_stats)

        return out

# =========================
# WF helpers: window metrics + stability
# =========================

def build_window_metrics_from_returns(
    returns: Optional[pd.Series],
    n_windows: int = 4,
) -> List[Dict[str, Any]]:
    """
    Slice a returns series into n_windows chronological chunks and compute
    per-window performance metrics for WF / stability analysis.

    Each window metric dict contains:
        - start_date, end_date
        - sharpe, max_drawdown, total_return
        - n_obs

    Assumes daily returns; Sharpe is annualized with sqrt(252).
    """
    if returns is None:
        return []
    if not isinstance(returns, pd.Series) or returns.empty:
        return []

    # Ensure sorted by time
    returns = returns.sort_index()
    n = len(returns)
    if n_windows <= 0 or n < 2:
        return []

    window_metrics: List[Dict[str, Any]] = []
    # Simple equal-size slicing by index
    for i in range(n_windows):
        start_idx = int(i * n / n_windows)
        end_idx = int((i + 1) * n / n_windows)
        if end_idx <= start_idx:
            continue
        r_win = returns.iloc[start_idx:end_idx]
        if r_win.empty:
            continue

        start_date = r_win.index[0]
        end_date = r_win.index[-1]
        mu = float(r_win.mean())
        sigma = float(r_win.std(ddof=1)) if len(r_win) > 1 else 0.0

        # Annualized Sharpe (assuming daily data)
        if sigma > 0:
            sharpe = (mu / sigma) * math.sqrt(252.0)
        else:
            sharpe = 0.0

        # Equity curve and max drawdown inside this window
        eq = (1.0 + r_win).cumprod()
        peak = eq.cummax()
        dd = (eq / peak) - 1.0
        max_dd = float(dd.min())  # negative
        max_dd_pos = abs(max_dd)  # store as positive fraction

        total_return = float(eq.iloc[-1] - 1.0)

        window_metrics.append(
            {
                "start_date": start_date,
                "end_date": end_date,
                "sharpe": sharpe,
                "max_drawdown": max_dd_pos,
                "total_return": total_return,
                "n_obs": int(len(r_win)),
            }
        )

    return window_metrics

def compute_wf_stability(result: BacktestResult) -> Dict[str, float]:
    """
    Compute walk-forward stability metrics from BacktestResult.window_metrics.

    Outputs:
        - wf_sharpe_mean
        - wf_sharpe_std
        - wf_stability_score  (higher = more stable)
        - wf_max_dd_mean
        - wf_max_dd_worst
        - wf_n_windows
    """
    wms = result.window_metrics or []
    if len(wms) == 0:
        # Fallback: treat whole-period metrics as one window
        return {
            "wf_sharpe_mean": float(result.sharpe),
            "wf_sharpe_std": 0.0,
            "wf_stability_score": float(result.sharpe),
            "wf_max_dd_mean": float(result.max_drawdown),
            "wf_max_dd_worst": float(result.max_drawdown),
            "wf_n_windows": 1.0,
        }

    sharpe_vals: List[float] = []
    dd_vals: List[float] = []

    for w in wms:
        s = w.get("sharpe")
        d = w.get("max_drawdown")
        if s is None or d is None:
            continue
        sharpe_vals.append(float(s))
        dd_vals.append(float(d))

    if len(sharpe_vals) == 0:
        # No valid window Sharpe, fallback to whole-period
        return {
            "wf_sharpe_mean": float(result.sharpe),
            "wf_sharpe_std": 0.0,
            "wf_stability_score": float(result.sharpe),
            "wf_max_dd_mean": float(result.max_drawdown),
            "wf_max_dd_worst": float(result.max_drawdown),
            "wf_n_windows": float(len(wms)),
        }

    sharpe_arr = np.asarray(sharpe_vals, dtype=float)
    dd_arr = np.asarray(dd_vals, dtype=float)

    wf_sharpe_mean = float(sharpe_arr.mean())
    wf_sharpe_std = float(sharpe_arr.std(ddof=1)) if len(sharpe_arr) > 1 else 0.0
    wf_max_dd_mean = float(dd_arr.mean())
    wf_max_dd_worst = float(dd_arr.max())  # drawdown stored as positive

    # Simple stability heuristic:
    #   high mean Sharpe, low std → high score
    wf_stability_score = wf_sharpe_mean / (1.0 + wf_sharpe_std)
    if not np.isfinite(wf_stability_score):
        wf_stability_score = 0.0

    metrics = {
        "wf_sharpe_mean": wf_sharpe_mean,
        "wf_sharpe_std": max(wf_sharpe_std, 0.0),
        "wf_stability_score": wf_stability_score,
        "wf_max_dd_mean": wf_max_dd_mean,
        "wf_max_dd_worst": wf_max_dd_worst,
        "wf_n_windows": float(len(sharpe_vals)),
    }

    # אופציונלי: לשמור את זה בתוך result.stats לשימוש עתידי / לוגים
    try:
        result.stats.update(metrics)
    except Exception:
        # לא נכשיל בגלל stats
        pass

    return metrics

# =========================
# HF-grade objective: config + scoring
# =========================

@dataclass
class HFObjectiveConfig:
    """
    High-level configuration for the hedge-fund-grade optimization objective.

    Thresholds are expressed in *fractions* of equity unless stated otherwise:
        - drawdown 0.20 → 20% peak-to-trough
        - ES/VaR 0.03 → 3% daily loss (approx)
    """

    # --- risk thresholds (soft / hard) ---
    dd_soft: float = 0.15   # above this, start penalizing
    dd_hard: float = 0.35   # above this, trial is effectively rejected

    es_soft: float = 0.03   # 3% tail loss (daily)
    es_hard: float = 0.06   # 6% tail loss → reject

    # --- trading behavior constraints ---
    min_trades: int = 40         # below this → heavy penalty
    target_trades: int = 150     # sweet spot
    max_turnover: float = 10.0   # notional traded / capital
    max_exposure: float = 3.0    # average gross exposure

    # --- weights for different components ---
    w_sharpe: float = 1.0            # whole-period Sharpe
    w_wf_stability: float = 0.8      # WF stability score
    w_dd_profile: float = 0.8        # drawdown penalties
    w_tail: float = 0.7              # ES/VaR penalties
    w_behavior: float = 0.4          # trades / turnover / exposure

    # --- numerical knobs ---
    hard_reject_score: float = -1e6  # score for "invalid" param sets


def compute_hf_objective(
    result: BacktestResult,
    cfg: HFObjectiveConfig | None = None,
) -> float:
    """
    Compute a single scalar score suitable for Optuna, combining:

        1. Whole-period Sharpe.
        2. WF stability (mean / std of window Sharpe).
        3. Drawdown profile (max + worst WF window).
        4. Tail risk (ES / VaR).
        5. Trading behavior (trades, turnover, exposure).

    Higher is better.
    """
    if cfg is None:
        cfg = HFObjectiveConfig()

    # ---- 1) WF stability & robust Sharpe ----
    wf = compute_wf_stability(result)
    wf_sharpe_mean = wf["wf_sharpe_mean"]
    wf_sharpe_std = wf["wf_sharpe_std"]
    wf_stability_score = wf["wf_stability_score"]
    wf_max_dd_worst = wf["wf_max_dd_worst"]

    # robust Sharpe: לא נותנים ל-Sharpe של חלון אחד להשתולל
    robust_sharpe = float(min(result.sharpe, wf_sharpe_mean))

    # ---- 2) Drawdown profile ----
    dd_overall = float(result.max_drawdown)
    dd_effective = max(dd_overall, wf_max_dd_worst)

    # soft penalty 0..1 בין dd_soft ל-dd_hard
    if dd_effective <= cfg.dd_soft:
        dd_soft_penalty = 0.0
    elif dd_effective >= cfg.dd_hard:
        dd_soft_penalty = 1.0
    else:
        dd_soft_penalty = (dd_effective - cfg.dd_soft) / max(
            1e-6, cfg.dd_hard - cfg.dd_soft
        )

    # hard reject (trial לא ראוי לחיים)
    if dd_effective >= cfg.dd_hard * 1.1:
        return cfg.hard_reject_score

    # ---- 3) Tail risk: ES / VaR ----
    es = float(result.es_95)
    var = float(result.var_95)
    tail_scale = max(es, var)

    if tail_scale <= cfg.es_soft:
        tail_penalty = 0.0
    elif tail_scale >= cfg.es_hard:
        tail_penalty = 1.0
    else:
        tail_penalty = (tail_scale - cfg.es_soft) / max(
            1e-6, cfg.es_hard - cfg.es_soft
        )

    if tail_scale >= cfg.es_hard * 1.2:
        return cfg.hard_reject_score

    # ---- 4) Trading behavior (trades, turnover, exposure) ----
    n_trades = int(result.n_trades)
    turnover = float(result.turnover)
    exposure = float(result.exposure)

    # trades penalty: מעט מדי טריידים = overfit / illiquid
    if n_trades <= 0:
        trades_penalty = 1.0  # הכי גרוע
    elif n_trades < cfg.min_trades:
        trades_penalty = 1.0
    elif n_trades < cfg.target_trades:
        # penalty יורד לינארית כשהטריידים מתקרבים ליעד
        trades_penalty = (cfg.target_trades - n_trades) / max(
            1.0, cfg.target_trades - cfg.min_trades
        )
    else:
        trades_penalty = 0.0

    # turnover penalty: יותר מדי סיבובים
    turnover_penalty = 0.0
    if turnover > cfg.max_turnover:
        turnover_penalty = min(1.0, (turnover - cfg.max_turnover) / cfg.max_turnover)

    # exposure penalty: מינוף גבוה מדי
    exposure_penalty = 0.0
    if exposure > cfg.max_exposure:
        exposure_penalty = min(
            1.0, (exposure - cfg.max_exposure) / max(1e-6, cfg.max_exposure)
        )

    behavior_penalty = 0.5 * trades_penalty + 0.25 * turnover_penalty + 0.25 * exposure_penalty

    # אם אין כמעט טריידים → reject חזק
    if n_trades < max(5, cfg.min_trades // 4):
        return cfg.hard_reject_score

    # ---- 5) בונים ציון סופי ----
    # בסיס: שילוב בין Sharpe לבין WF stability:
    base_score = (
        cfg.w_sharpe * robust_sharpe
        + cfg.w_wf_stability * wf_stability_score
    )

    # מורידים נקודות על פרופיל drawdown
    dd_score_adj = -cfg.w_dd_profile * dd_soft_penalty * (1.0 + dd_effective * 4.0)

    # מורידים נקודות על Tail
    tail_score_adj = -cfg.w_tail * tail_penalty * (1.0 + tail_scale * 20.0)

    # מורידים נקודות על התנהגות (turnover / exposure / #trades)
    behavior_score_adj = -cfg.w_behavior * behavior_penalty * 3.0

    final_score = base_score + dd_score_adj + tail_score_adj + behavior_score_adj

    # Safety: NaN/inf → reject
    if not np.isfinite(final_score):
        return cfg.hard_reject_score

    # אפשר לשמור את המטריקות בתוך result.stats (לא חובה אבל שימושי ל־Insights)
    try:
        result.stats.update(
            {
                "hf_base_score": float(base_score),
                "hf_final_score": float(final_score),
                "hf_dd_soft_penalty": float(dd_soft_penalty),
                "hf_tail_penalty": float(tail_penalty),
                "hf_behavior_penalty": float(behavior_penalty),
                "hf_wf_sharpe_mean": float(wf_sharpe_mean),
                "hf_wf_sharpe_std": float(wf_sharpe_std),
            }
        )
    except Exception:
        pass

    return float(final_score)

# =========================
# SECTION 4: Core Backtester Skeleton
# =========================

class OptimizationBacktester:
    """
    Core backtester for a single pair, designed to be driven by Optuna.

    Typical usage from the optimization tab
    ---------------------------------------
    bt = OptimizationBacktester(
        symbol_a="XLY",
        symbol_b="XLP",
        lookback=60,
        z_open=2.0,
        z_close=0.5,
        stop_z=3.5,
        take_z=0.0,
        hr_method="ols",
        ...
    )
    perf_dict = bt.run()   # {"Sharpe": ..., "Profit": ..., "Drawdown": ...}

    Design notes
    ------------
    - `config` holds *engine-level* settings (capital, RF, costs, freq, etc.).
    - `params` holds *strategy-level* knobs (lookback, thresholds, filters, ...).
    - Later parts will implement the full pipeline:
      data → spread → signals → trades → equity → metrics.
    """

    def __init__(
        self,
        symbol_a: str,
        symbol_b: str,
        *,
        start: Optional[date] = None,
        end: Optional[date] = None,
        initial_capital: float = float(os.environ.get("OPT_BT_INIT_CAPITAL", 100_000)),
        max_gross_exposure: float = float(os.environ.get("OPT_BT_MAX_GROSS", 300_000)),
        risk_free_rate: float = 0.02,
        commission_bps: float = 1.0,
        slippage_bps: float = 2.0,
        dollar_neutral: bool = True,
        beta_neutral: bool = False,
        bar_freq: str = "D",
        data_source: str = "AUTO",
        min_history_days: int = 120,
        rebalance_days: int = 0,
        **params: Any,
    ) -> None:
        """
        Build a backtester instance for a single pair.

        Any additional keyword arguments (**params) are treated as
        *strategy parameters* and will be used later for the signal
        and risk layer (lookback, z_open, z_close, filters, etc.).
        """
        self.config = BacktestConfig(
            symbol_a=symbol_a,
            symbol_b=symbol_b,
            start=start,
            end=end,
            min_history_days=int(min_history_days),
            initial_capital=float(initial_capital),
            max_gross_exposure=float(max_gross_exposure),
            risk_free_rate=float(risk_free_rate),
            commission_bps=float(commission_bps),
            slippage_bps=float(slippage_bps),
            dollar_neutral=bool(dollar_neutral),
            beta_neutral=bool(beta_neutral),
            bar_freq=str(bar_freq),
            data_source=str(data_source),
            rebalance_days=int(rebalance_days),
            extra={},  # may be used by advanced overlays later
        )
        self.params: Dict[str, Any] = dict(params)

        logger.debug(
            "OptimizationBacktester init | cfg=%s | params=%s",
            self.config.to_dict(),
            self.params,
        )

    # ---------- convenience constructors ----------

    @classmethod
    def from_params(cls, symbol_a: str, symbol_b: str, params: Mapping[str, Any]) -> "OptimizationBacktester":
        """
        Convenience constructor for Optuna / grid-search code:
        takes a flat params dict and forwards it as **params.
        """
        kwargs = dict(params)
        # Allow overrides in params for config-level fields if present:
        for key in list(kwargs.keys()):
            if key in {
                "start", "end", "initial_capital", "max_gross_exposure",
                "risk_free_rate", "commission_bps", "slippage_bps",
                "dollar_neutral", "beta_neutral", "bar_freq",
                "data_source", "min_history_days", "rebalance_days",
            }:
                # they will be passed explicitly in __init__
                pass
        return cls(symbol_a=symbol_a, symbol_b=symbol_b, **kwargs)

    # -----------------------------------------
    # NOTE: The *real* implementation of run()
    # will be added in Parts 2–4.
    # -----------------------------------------
    def run(self) -> Dict[str, float]:
        """
        Execute the backtest and return the minimal perf dict.

        ⚠️ CURRENTLY (Part 1/6):
        This is a SAFE PLACEHOLDER so that:
        - the dashboard can import and display System Health,
        - you can finish pasting Parts 2–6 without breaking anything.

        Once Parts 2–6 are pasted, this method will:
        - load price data
        - build the spread
        - generate signals & positions
        - simulate PnL and equity
        - compute full BacktestResult
        and return `result.to_perf_dict()`.
        """
        logger.error(
            "OptimizationBacktester.run() called while only Part 1/6 of "
            "core/optimization_backtester.py is present. "
            "Paste Parts 2–6 for a full production backtest."
        )

        # Minimal zero-risk placeholder so the UI doesn't explode:
        placeholder = BacktestResult(
            sharpe=0.0,
            profit=0.0,
            max_drawdown=0.0,
            sortino=0.0,
            calmar=0.0,
            win_rate=0.0,
            n_trades=0,
            avg_trade_pnl=0.0,
            exposure=0.0,
            turnover=0.0,
            equity_curve=None,
            stats={
                "placeholder": True,
                "reason": "Backtester not fully implemented (only Part 1/6 present).",
                "config": self.config.to_dict(),
                "params": dict(self.params),
            },
        )
        return placeholder.to_perf_dict()


# Alias for existing code that imports `Backtester`
Backtester = OptimizationBacktester

# =========================
# PART 2/6: Data & Spread Engine (Hedge-Fund Grade)
# =========================




def _infer_periods_per_year(freq: str) -> float:
    """
    Infer approximate periods-per-year for annualization based on bar frequency.

    "D"  → ~252 (trading days)
    "H"  → ~252*6 (6 hours/day as rough intraday)
    anything else → 252 as a safe default.
    """
    f = (freq or "D").upper()
    if f in {"D", "1D"}:
        return 252.0
    if f in {"H", "1H"}:
        return 252.0 * 6.0
    if f in {"W", "1W"}:
        return 52.0
    if f in {"M", "1M"}:
        return 12.0
    # default: trading-day like
    return 252.0


def _clip_window_by_config(df: pd.DataFrame, cfg: BacktestConfig) -> pd.DataFrame:
    """
    Clip a price DataFrame by (cfg.start, cfg.end) if provided, and
    ensure it's index-sorted.

    שדרוג חשוב:
    -----------
    • אם קיימת עמודת 'date' (או 'Date') → נשתמש בה בתור האינדקס הזמני,
      במקום לנסות להמיר את index המספרי ל-datetime (0→1970-01-01 וכו').
    """
    if df.empty:
        return df

    # קודם כל נעדיף להשתמש בעמודת date אם קיימת
    if not isinstance(df.index, pd.DatetimeIndex):
        date_col = None
        for cand in ("date", "Date", "ts", "ts_utc", "datetime"):
            if cand in df.columns:
                date_col = cand
                break

        if date_col is not None:
            try:
                idx = pd.to_datetime(df[date_col])
                df = df.copy()
                df.index = idx
            except Exception:
                # אם המרה נכשלה – ניפול בחזרה ללוגיקה הישנה
                pass

    # אחרי הטיפול ב-date column, נמשיך כמו קודם
    df = df.sort_index()

    if isinstance(df.index, pd.DatetimeIndex):
        if cfg.start is not None:
            df = df[df.index.date >= cfg.start]
        if cfg.end is not None:
            df = df[df.index.date <= cfg.end]
    else:
        # Best-effort: try to treat index as datetime (fallback בלבד)
        try:
            idx = pd.to_datetime(df.index)
            df = df.set_index(idx)
            if cfg.start is not None:
                df = df[df.index.date >= cfg.start]
            if cfg.end is not None:
                df = df[df.index.date <= cfg.end]
        except Exception:
            # leave as-is if we cannot parse
            pass

    return df


def _select_price_series(prices: pd.DataFrame, symbol: str) -> pd.Series:
    """
    Extract a 1D price series for a single symbol from a generic 'prices' DataFrame.

    תומך בשתי צורות עיקריות:
    1. LONG format:
         columns: ['symbol', <datetime_col>, 'close'/ 'price'/ ...]
    2. WIDE format:
         index: datetime, columns: one column per symbol (BITO, BKCH, ...)

    מחזיר Series עם אינדקס זמן וערכי מחיר (float).
    """

    # ===== Case 0: יש עמודת symbol וה־index כבר DatetimeIndex =====
    # זה בדיוק המצב אחרי שאנחנו מעדכנים את index ל-date ב-SqlStore.
    if "symbol" in prices.columns and isinstance(prices.index, pd.DatetimeIndex):
        # ננחש את עמודת המחיר
        price_candidates = ["close", "adj_close", "price", "px", "last", "value"]
        price_col = next((c for c in price_candidates if c in prices.columns), None)
        if price_col is None:
            # fallback: numeric column כלשהו שאינו symbol
            numeric_cols = prices.select_dtypes("number").columns.tolist()
            numeric_cols = [c for c in numeric_cols if c not in {"symbol"}]
            if not numeric_cols:
                raise ValueError(
                    "Could not infer price column in prices frame with DatetimeIndex; "
                    f"numeric={prices.select_dtypes('number').columns.tolist()}"
                )
            price_col = numeric_cols[0]

        sub = prices.loc[prices["symbol"] == symbol].copy()
        if sub.empty:
            raise ValueError(f"No rows in prices for symbol={symbol}")

        sub = sub.dropna(subset=[price_col])
        sub = sub.sort_index()  # לפי ה־DatetimeIndex

        s = sub[price_col].astype(float)
        s.name = symbol
        return s

    # ===== Case 1: LONG format with 'symbol' column (index לא בהכרח זמן) =====
    if "symbol" in prices.columns:
        # ננחש את עמודת הזמן
        dt_candidates = ["ts_utc", "ts", "date", "datetime", "asof", "time"]
        dt_col = next((c for c in dt_candidates if c in prices.columns), None)
        if dt_col is None:
            raise ValueError(
                f"prices frame has 'symbol' column but no datetime column; "
                f"columns={list(prices.columns)}"
            )

        # ננחש את עמודת המחיר
        price_candidates = ["close", "adj_close", "price", "px", "last", "value"]
        price_col = next((c for c in price_candidates if c in prices.columns), None)
        if price_col is None:
            numeric_cols = prices.select_dtypes("number").columns.tolist()
            numeric_cols = [c for c in numeric_cols if c not in {dt_col, "symbol"}]
            if not numeric_cols:
                raise ValueError(
                    "Could not infer price column in prices frame; "
                    f"numeric={prices.select_dtypes('number').columns.tolist()}"
                )
            price_col = numeric_cols[0]

        sub = prices.loc[prices["symbol"] == symbol].copy()
        if sub.empty:
            raise ValueError(f"No rows in prices for symbol={symbol}")

        sub = sub.dropna(subset=[price_col])
        sub = sub.sort_values(dt_col)

        # כאן לא תהיה התנגשות בין date כ-index לבין date כעמודה,
        # כי אנחנו משתמשים בעמודת הזמן dt_col מפורשות.
        s = sub.set_index(dt_col)[price_col].astype(float)
        s.name = symbol
        return s

    # ===== Case 2: WIDE format: symbol as a direct column =====
    if symbol in prices.columns:
        s = prices[symbol].dropna()
        if isinstance(s, pd.DataFrame):
            price_candidates = [
                c
                for c in s.columns
                if str(c).lower() in ("close", "price", "px", "last", "adj_close")
            ]
            if price_candidates:
                s = s[price_candidates[0]]
            else:
                s = s.iloc[:, 0]
        s = s.sort_index()
        s.name = symbol
        return s

    # ===== Nothing matched =====
    raise ValueError(
        f"Cannot find symbol {symbol} in prices frame; "
        f"columns={list(prices.columns)[:10]}..."
    )



def _load_pair_prices_via_utils(cfg: BacktestConfig) -> Optional[pd.DataFrame]:
    """
    Try to load pair prices via common.utils.load_price_data if available.

    We *intentionally* try a few common signatures to make this robust across
    minor refactors of common/utils.py
    """
    if _load_price_data is None:
        return None

    sym_a, sym_b = cfg.symbol_a, cfg.symbol_b
    start, end, freq = cfg.start, cfg.end, cfg.bar_freq

    # candidate call patterns
    call_specs = [
        # (callable, description)
        (lambda: _load_price_data([sym_a, sym_b], start=start, end=end, freq=freq), "load_price_data([a,b], start,end,freq)"),
        (lambda: _load_price_data(symbols=[sym_a, sym_b], start=start, end=end, freq=freq), "load_price_data(symbols=[a,b], start,end,freq)"),
        (lambda: _load_price_data(sym_a, sym_b, start=start, end=end, freq=freq), "load_price_data(a,b,start,end,freq)"),
    ]

    last_err: Optional[Exception] = None
    for fn, desc in call_specs:
        try:
            df = fn()
            if isinstance(df, pd.DataFrame) and not df.empty:
                logger.debug("Loaded prices via utils.%s", desc)
                return df
        except Exception as e:  # pragma: no cover - defensive
            last_err = e
            continue

    if last_err is not None:
        logger.debug("load_price_data via utils failed: %s", last_err)

    return None



def _load_pair_prices_via_sql_store(cfg: BacktestConfig) -> pd.DataFrame:
    """
    Load pair prices *מ-SqlStore* (DuckDB) עבור שני הסימבולים:

    מצופה שטבלת prices ב-SqlStore תהיה בפורמט "LONG":
        symbol | date | open | high | low | close | volume | ...

    מחזיר:
        DataFrame עם עמודת 'symbol' + 'date' + שאר העמודות,
        כששני הסימבולים (symbol_a, symbol_b) בפנים.
    """
    if SqlStore is None:
        raise RuntimeError("SqlStore is not available (import failed).")

    # SqlStore.from_settings יקח את SQL_STORE_URL מה־env אם לא העברת engine_url
    store = SqlStore.from_settings({}, read_only=True)

    symbols = [cfg.symbol_a, cfg.symbol_b]
    dfs: list[pd.DataFrame] = []

    for sym in symbols:
        try:
            df_sym = store.load_price_history(sym)  # זו הפונקציה שקיימת אצלך ב-SqlStore
        except Exception as exc:
            logger.warning("load_price_history(%s) from SqlStore failed: %s", sym, exc)
            continue

        if df_sym is None or df_sym.empty:
            logger.warning("SqlStore.load_price_history(%s) returned empty DataFrame", sym)
            continue

        df_sym = df_sym.copy()

        # לוודא שיש עמודת 'symbol'
        if "symbol" not in df_sym.columns:
            df_sym["symbol"] = sym

        # לוודא שיש 'date' כעמודת זמן
        if "date" not in df_sym.columns:
            if isinstance(df_sym.index, pd.DatetimeIndex):
                df_sym = df_sym.reset_index().rename(columns={"index": "date"})
            else:
                # fallback: ניקח את האינדקס כעמודת תאריך
                df_sym["date"] = pd.to_datetime(df_sym.index)

        # נשמור רק עמודות רלוונטיות (אם קיימות)
        keep_cols = [c for c in df_sym.columns
                     if c in {"symbol", "date", "open", "high", "low", "close", "volume",
                              "adj_close", "ts_utc", "env", "run_id", "section"}]
        if not keep_cols:
            # אם אין שום התאמה – נשאיר הכל
            keep_cols = df_sym.columns.tolist()

        df_sym = df_sym[keep_cols]
        dfs.append(df_sym)

    if not dfs:
        raise ValueError(
            f"SqlStore has no price history for symbols {symbols} "
            f"(table 'prices' empty or missing rows)."
        )

    prices = pd.concat(dfs, axis=0, ignore_index=True)
    # ניקוי NaN בסיסי
    prices = prices.dropna(subset=["date", "symbol"])
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.sort_values(["symbol", "date"]).reset_index(drop=True)

    return prices

def _load_pair_prices_via_ibkr(cfg: BacktestConfig) -> pd.DataFrame:
    """
    Load pair prices via Interactive Brokers (IBKR) לצורכי Backtest.

    העיקרון:
    --------
    - הפונקציה כאן היא רק "hook":
      היא קוראת לפונקציה שתהיה ב-core.ib_data_ingestor:
          load_pair_prices_for_backtest(...)
    - אם הפונקציה שם לא קיימת / לא זמינה → נזרוק RuntimeError,
      וה-layer העליון (_load_pair_prices) כבר יתפוס ויפול חזרה ל-SqlStore / utils.

    מצופה ש-load_pair_prices_for_backtest תחזיר DataFrame בפורמט:
        index: DatetimeIndex
        columns: לפחות שני price series לשני הסימבולים
                 (או wide עם שמות עמודות = סימבולים, או LONG עם 'symbol' וכו')
    """
    if _ib_load_pair_prices_for_backtest is None:
        raise RuntimeError(
            "IBKR backtest loader is not available. "
            "Define core.ib_data_ingestor.load_pair_prices_for_backtest first."
        )

    sym_a, sym_b = cfg.symbol_a, cfg.symbol_b

    # נעביר את אותם פרמטרים ל-IBKR loader, בצורה generic.
    # אתה יכול להתאים את החתימה בפועל ב-core.ib_data_ingestor.
    df = _ib_load_pair_prices_for_backtest(
        symbols=[sym_a, sym_b],
        start=cfg.start,
        end=cfg.end,
        freq=cfg.bar_freq,
        data_source=str(cfg.data_source or "IBKR"),
    )

    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError(
            f"IBKR loader returned empty/non-DataFrame for pair {sym_a}-{sym_b}"
        )

    return df

def _load_pair_prices(cfg: BacktestConfig) -> pd.DataFrame:
    """
    Unified entrypoint for loading prices for a pair (HF-grade).

    Strategy / Priority
    -------------------
    Controlled by `cfg.data_source` (case-insensitive):

    - None / "AUTO" / "HYBRID":
        1) SqlStore (אם זמין)
        2) common.utils.load_price_data (internal utils)
        3) IBKR (direct API fallback)

    - "SQL" / "SQL_STORE" / "DUCKDB":
        1) SqlStore
        2) utils.load_price_data
        3) IBKR

    - "UTILS" / "INTERNAL":
        1) utils.load_price_data
        2) SqlStore
        3) IBKR

    - "IB" / "IBKR" / "INTERACTIVE_BROKERS":
        1) IBKR
        2) SqlStore
        3) utils.load_price_data

    Notes
    -----
    - All loaders MUST return a pd.DataFrame with a Date/Datetime index and columns
      for both symbols (e.g. ["A", "B"] or "close_a"/"close_b").
    - `_clip_window_by_config` is applied after each successful load attempt.
    - Raises ValueError if all loaders fail or result in an empty frame.
    """

    ds = (cfg.data_source or "AUTO").upper()

    sym_a = getattr(cfg, "symbol_a", None) or getattr(cfg, "sym1", "?")
    sym_b = getattr(cfg, "symbol_b", None) or getattr(cfg, "sym2", "?")

    def _try_sql_store() -> Optional[pd.DataFrame]:
        if SqlStore is None:
            logger.debug("SqlStore is None — skipping SqlStore loader for %s-%s", sym_a, sym_b)
            return None
        try:
            df_sql = _load_pair_prices_via_sql_store(cfg)
            if isinstance(df_sql, pd.DataFrame) and not df_sql.empty:
                df_sql = _clip_window_by_config(df_sql, cfg)
                if not df_sql.empty:
                    logger.info(
                        "Loaded pair prices via SqlStore for %s-%s (rows=%d, data_source=%s)",
                        sym_a,
                        sym_b,
                        len(df_sql),
                        ds,
                    )
                    return df_sql
                else:
                    logger.warning(
                        "SqlStore returned data for %s-%s but empty after clipping.",
                        sym_a,
                        sym_b,
                    )
            else:
                logger.debug("SqlStore returned empty/non-DataFrame for %s-%s", sym_a, sym_b)
        except Exception as e:
            logger.warning(
                "SqlStore price load failed for %s-%s: %s — falling back.",
                sym_a,
                sym_b,
                e,
            )
        return None

    def _try_utils() -> Optional[pd.DataFrame]:
        try:
            df_utils = _load_pair_prices_via_utils(cfg)
            if isinstance(df_utils, pd.DataFrame) and not df_utils.empty:
                df_utils = _clip_window_by_config(df_utils, cfg)
                if not df_utils.empty:
                    logger.info(
                        "Loaded pair prices via internal utils for %s-%s (rows=%d)",
                        sym_a,
                        sym_b,
                        len(df_utils),
                    )
                    return df_utils
                else:
                    logger.warning(
                        "Internal utils returned data for %s-%s but empty after clipping.",
                        sym_a,
                        sym_b,
                    )
            else:
                logger.debug("Internal utils returned empty/non-DataFrame for %s-%s", sym_a, sym_b)
        except Exception as e:
            logger.warning(
                "Internal utils price load failed for %s-%s: %s — falling back.",
                sym_a,
                sym_b,
                e,
            )
        return None

    def _try_ibkr() -> Optional[pd.DataFrame]:
        """
        Loader via Interactive Brokers.

        מצופה ש-_load_pair_prices_via_ibkr:
        - יחזיר DataFrame עם אינדקס זמן ועמודות לשני הסימבולים.
        - יטפל ב-connection / contracts / pacing בתוך הפונקציה.
        """
        try:
            df_ib = _load_pair_prices_via_ibkr(cfg)
            if isinstance(df_ib, pd.DataFrame) and not df_ib.empty:
                df_ib = _clip_window_by_config(df_ib, cfg)
                if not df_ib.empty:
                    logger.info(
                        "Loaded pair prices via IBKR for %s-%s (rows=%d)",
                        sym_a,
                        sym_b,
                        len(df_ib),
                    )
                    return df_ib
                else:
                    logger.warning(
                        "IBKR returned data for %s-%s but empty after clipping.",
                        sym_a,
                        sym_b,
                    )
            else:
                logger.debug("IBKR loader returned empty/non-DataFrame for %s-%s", sym_a, sym_b)
        except Exception as e:
            logger.warning(
                "IBKR price load failed for %s-%s: %s — falling back.",
                sym_a,
                sym_b,
                e,
            )
        return None

    # סדרי עדיפויות לפי data_source
    if ds in {"SQL", "SQL_STORE", "DUCKDB"}:
        loaders = [_try_sql_store, _try_utils, _try_ibkr]
    elif ds in {"UTILS", "INTERNAL"}:
        loaders = [_try_utils, _try_sql_store, _try_ibkr]
    elif ds in {"IB", "IBKR", "INTERACTIVE_BROKERS"}:
        loaders = [_try_ibkr, _try_sql_store, _try_utils]
    else:  # AUTO / HYBRID / כל דבר אחר
        loaders = [_try_sql_store, _try_utils, _try_ibkr]

    last_err_msg = None
    for loader in loaders:
        df = loader()
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df

    # אם הגענו לכאן – כל הנסיונות נכשלו / יצאו ריקים
    msg = (
        f"Price data for pair {sym_a}-{sym_b} is empty after trying data_source={ds} "
        f"(SqlStore / utils / IBKR)."
    )
    logger.error(msg)
    raise ValueError(msg)



def _build_spread_frame(cfg: BacktestConfig, prices: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a generic pair price frame into a canonical spread DataFrame:

        index: datetime
        columns:
            px_a       - price of symbol_a
            px_b       - price of symbol_b
            spread     - px_a - px_b * hedge_ratio  (initially hedge_ratio=1)
            spread_ret - percentage change of spread (for risk stats)

    Later parts may override hedge_ratio using OLS / Kalman etc.
    """
    sym_a, sym_b = cfg.symbol_a, cfg.symbol_b

    # Try extract price series for both legs
    try:
        px_a = _select_price_series(prices, sym_a)
        px_b = _select_price_series(prices, sym_b)
    except Exception as e:
        raise ValueError(f"Failed to extract prices for pair {sym_a}-{sym_b}: {e}") from e

    df = pd.DataFrame({"px_a": px_a, "px_b": px_b}).dropna(how="any")
    df = df.sort_index()

    if df.empty:
        raise ValueError(f"No overlapping data for pair {sym_a}-{sym_b} after alignment.")

    # Basic sanity: enough history?
    if isinstance(df.index, pd.DatetimeIndex) and cfg.min_history_days > 0:
        span_days = (df.index[-1] - df.index[0]).days
        if span_days < cfg.min_history_days:
            raise ValueError(
                f"Insufficient history for pair {sym_a}-{sym_b}: "
                f"{span_days} days < min_history_days={cfg.min_history_days}"
            )

    # For now use hedge_ratio=1.0 — later parts will integrate OLS/Kalman, etc.
    hedge_ratio = 1.0
    spread = df["px_a"] - hedge_ratio * df["px_b"]
    df["spread"] = spread
    df["spread_ret"] = df["spread"].pct_change().fillna(0.0)

    return df


def _bt_prepare_data(self: "OptimizationBacktester") -> pd.DataFrame:
    """
    Instance-level helper: load prices for the current pair and build
    the canonical spread DataFrame. The result is cached on `self._data`
    for re-use by later pipeline stages (signals, trades, metrics).

    Raises descriptive ValueError / RuntimeError if data is missing or invalid.
    """
    cfg = self.config

    # Load raw prices (via utils or yfinance)
    prices = _load_pair_prices(cfg)

    # Build canonical spread frame
    spread_df = _build_spread_frame(cfg, prices)

    # Cache on instance for further steps
    self._data = spread_df  # type: ignore[attr-defined]

    logger.debug(
        "Prepared data for pair %s-%s | rows=%d | columns=%s",
        cfg.symbol_a,
        cfg.symbol_b,
        len(spread_df),
        list(spread_df.columns),
    )

    return spread_df


# Attach data-layer helpers as methods on OptimizationBacktester
OptimizationBacktester._prepare_data = _bt_prepare_data  # type: ignore[attr-defined]
OptimizationBacktester._load_pair_prices = lambda self: _load_pair_prices(self.config)  # type: ignore[attr-defined]
OptimizationBacktester._build_spread_frame = lambda self, prices: _build_spread_frame(self.config, prices)  # type: ignore[attr-defined]
OptimizationBacktester._infer_periods_per_year = staticmethod(_infer_periods_per_year)  # type: ignore[attr-defined]

# =========================
# PART 3/6: Signal & PnL Engine (Hedge-Fund Grade)
# =========================

def _bt_extract_strategy_params(self: "OptimizationBacktester") -> Dict[str, Any]:
    """
    Extract and normalize strategy-level parameters from self.params.

    We keep this function opinionated but robust:
    - If a parameter is missing, use a reasonable professional default.
    - All numeric parameters are coerced to float/int safely.
    """
    p = dict(self.params)

    def _f(name: str, default: float) -> float:
        try:
            return float(p.get(name, default))
        except Exception:
            return float(default)

    def _i(name: str, default: int) -> int:
        try:
            return int(float(p.get(name, default)))
        except Exception:
            return int(default)

    params = {
        "lookback": _i("lookback", 60),
        "z_open": _f("z_open", 2.0),
        "z_close": _f("z_close", 0.5),
        "stop_z": _f("stop_z", 4.0),
        "take_z": _f("take_z", 0.0),  # 0 → disabled
        "max_holding_days": _i("max_holding_days", 30),
        "min_spread_vol": _f("min_spread_vol", 1e-6),
        "max_position_abs": _f("max_position_abs", 1.0),  # abs(pos) ∈ [0,1]
        "hr_method": str(p.get("hr_method", "1to1")).lower(),  # stub for future
    }

    # Basic guards
    if params["lookback"] < 10:
        params["lookback"] = 10
    if params["max_holding_days"] <= 0:
        params["max_holding_days"] = 30
    if params["max_position_abs"] <= 0:
        params["max_position_abs"] = 1.0

    return params


def _bt_build_signal_frame(
    self: "OptimizationBacktester",
    data: pd.DataFrame,
    params: Dict[str, Any],
) -> pd.DataFrame:
    """
    Build a canonical signal frame from the spread data:

    Columns:
        px_a, px_b       - leg prices (from _prepare_data)
        spread           - raw spread
        spread_ret       - spread returns
        mu               - rolling mean of spread
        sigma            - rolling std of spread
        zscore           - (spread - mu) / sigma
        pos              - spread position: +1 (long), -1 (short), 0 (flat)
        w_a, w_b         - portfolio weights on each leg (proportional)
        trade_id         - integer ID for each completed/open trade
        holding_days     - days in current trade
    """
    cfg = self.config
    df = data.copy()

    lb = int(params["lookback"])
    if lb >= len(df):
        raise ValueError(f"lookback={lb} >= data length={len(df)}")

    # Rolling mean / std of spread
    df["mu"] = df["spread"].rolling(lb, min_periods=max(5, lb // 2)).mean()
    df["sigma"] = df["spread"].rolling(lb, min_periods=max(5, lb // 2)).std(ddof=0)
    df["sigma"] = df["sigma"].replace(0.0, np.nan)

    # Z-score
    df["zscore"] = (df["spread"] - df["mu"]) / df["sigma"]
    df["zscore"] = df["zscore"].replace([np.inf, -np.inf], np.nan)

    # Mask: only trade when sigma סבירה
    df["valid"] = df["sigma"] >= float(params["min_spread_vol"])

    # Init position & trade info
    pos = np.zeros(len(df), dtype=float)
    trade_id = np.full(len(df), fill_value=-1, dtype=int)
    hold_days = np.zeros(len(df), dtype=int)

    # Weights per leg (will fill later)
    w_a = np.zeros(len(df), dtype=float)
    w_b = np.zeros(len(df), dtype=float)

    z_open = float(params["z_open"])
    z_close = float(params["z_close"])
    stop_z = float(params["stop_z"])
    take_z = float(params["take_z"])
    max_holding = int(params["max_holding_days"])
    max_abs_pos = float(params["max_position_abs"])

    # Defensive: z_close should be smaller than z_open
    if abs(z_close) > abs(z_open):
        z_close = z_open * 0.5

    current_pos = 0.0
    current_trade_id = -1
    current_hold = 0
    entry_equity_anchor = 1.0  # used for take_z logic later (Part 4/5)

    idx = df.index

    # We start after lookback window
    start_idx = lb

    for i in range(start_idx, len(df)):
        z = df["zscore"].iloc[i]
        valid = bool(df["valid"].iloc[i])

        prev_pos = current_pos

        # If not valid (too low sigma) → prefer to flatten
        if not valid or np.isnan(z):
            current_pos = 0.0
        else:
            # Long spread: buy A, sell B
            # Short spread: sell A, buy B
            # z > 0 → spread מעל הממוצע → short spread
            # z < 0 → spread מתחת לממוצע → long spread
            if current_pos == 0.0:
                # Entry rules
                if z <= -z_open:
                    current_pos = +max_abs_pos  # long spread
                    current_trade_id += 1
                    current_hold = 0
                    entry_equity_anchor = 1.0
                elif z >= z_open:
                    current_pos = -max_abs_pos  # short spread
                    current_trade_id += 1
                    current_hold = 0
                    entry_equity_anchor = 1.0
            else:
                # We are in a trade → check exit conditions
                current_hold += 1

                # For long spread: expect z to mean-revert upwards towards 0
                if current_pos > 0:
                    exit_revert = (z >= -z_close)
                    exit_stop = (z <= -abs(stop_z))
                    # take_z לפי Z יעד (אופציונלי) – אם מוגדר כערך חיובי
                    exit_take = (take_z > 0.0 and z >= -abs(take_z))
                else:
                    # For short spread: expect z to revert down towards 0
                    exit_revert = (z <= z_close)
                    exit_stop = (z >= abs(stop_z))
                    exit_take = (take_z > 0.0 and z <= abs(take_z))

                exit_time = (current_hold >= max_holding)

                if exit_revert or exit_stop or exit_take or exit_time:
                    current_pos = 0.0

        # Set arrays
        pos[i] = current_pos
        hold_days[i] = current_hold if current_pos != 0.0 else 0
        trade_id[i] = current_trade_id if current_pos != 0.0 else -1

        # Position sizing in *weights*: symmetric dollar-neutral
        # Long spread: +w on A, -w on B
        if current_pos > 0:
            w_a[i] = +current_pos * 0.5
            w_b[i] = -current_pos * 0.5
        elif current_pos < 0:
            w_a[i] = current_pos * 0.5
            w_b[i] = -current_pos * 0.5
        else:
            w_a[i] = 0.0
            w_b[i] = 0.0

    df["pos"] = pos
    df["trade_id"] = trade_id
    df["holding_days"] = hold_days
    df["w_a"] = w_a
    df["w_b"] = w_b

    # Basic metadata
    df["symbol_a"] = cfg.symbol_a
    df["symbol_b"] = cfg.symbol_b

    return df


def _bt_simulate_equity(
    self: "OptimizationBacktester",
    sig_df: pd.DataFrame,
    params: Dict[str, Any],
) -> BacktestResult:
    """
    Given a signal frame (with positions & weights), simulate the equity curve
    and compute professional risk/return metrics.

    The simulation is *weights-based*:
    - w_a, w_b are portfolio weights (fraction of capital) on each leg.
    - PnL_t = equity_{t-1} * (w_a * ret_a + w_b * ret_b) - costs.
    """
    cfg = self.config
    df = sig_df.copy()

    # Sanity: require enough rows and some active trading
    if len(df) < max(40, int(params["lookback"]) + 5):
        # Not enough data → return neutral result
        return BacktestResult(
            sharpe=0.0,
            profit=0.0,
            max_drawdown=0.0,
            sortino=0.0,
            calmar=0.0,
            win_rate=0.0,
            n_trades=0,
            avg_trade_pnl=0.0,
            exposure=0.0,
            turnover=0.0,
            equity_curve=None,
            stats={
                "reason": "insufficient_rows",
                "rows": len(df),
            },
        )

    # Compute leg returns
    df["ret_a"] = df["px_a"].pct_change().fillna(0.0)
    df["ret_b"] = df["px_b"].pct_change().fillna(0.0)

    # Prepare arrays for equity simulation
    n = len(df)
    equity = np.zeros(n, dtype=float)
    equity[0] = cfg.initial_capital

    pnl = np.zeros(n, dtype=float)
    costs = np.zeros(n, dtype=float)

    # Transaction cost model (simple but realistic enough)
    cost_bps = float(cfg.commission_bps) + float(cfg.slippage_bps)

    w_a = df["w_a"].to_numpy(dtype=float)
    w_b = df["w_b"].to_numpy(dtype=float)

    # For turnover & trade stats
    turnover_notional = 0.0
    trade_pnls: List[float] = []
    trade_equity_start: Optional[float] = None
    trade_last_id: int = -1

    for i in range(1, n):
        prev_eq = equity[i - 1]
        if prev_eq <= 0:
            prev_eq = cfg.initial_capital

        # Portfolio return before costs
        r_port = w_a[i - 1] * df["ret_a"].iloc[i] + w_b[i - 1] * df["ret_b"].iloc[i]
        pnl_raw = prev_eq * r_port

        # Turnover & costs (when weights change)
        dw_a = abs(w_a[i] - w_a[i - 1])
        dw_b = abs(w_b[i] - w_b[i - 1])
        notional = prev_eq * (dw_a + dw_b)
        turnover_notional += notional

        trade_cost = notional * (cost_bps / 10_000.0)
        costs[i] = trade_cost

        pnl[i] = pnl_raw - trade_cost
        equity[i] = prev_eq + pnl[i]

        # Trade PnL tracking by trade_id
        tid = int(df["trade_id"].iloc[i])
        if tid >= 0:
            # entering a new trade
            if tid != trade_last_id:
                # close old trade if existed
                if trade_last_id >= 0 and trade_equity_start is not None:
                    trade_pnls.append(equity[i - 1] - trade_equity_start)
                trade_equity_start = prev_eq
                trade_last_id = tid
        else:
            # no active trade at this row → consider closing last trade
            if trade_last_id >= 0 and trade_equity_start is not None:
                trade_pnls.append(prev_eq - trade_equity_start)
                trade_last_id = -1
                trade_equity_start = None

    # If a trade is open at the end, close it at final equity
    if trade_last_id >= 0 and trade_equity_start is not None:
        trade_pnls.append(equity[-1] - trade_equity_start)

    equity_series = pd.Series(equity, index=df.index, name="equity")

    # Profit
    profit = float(equity_series.iloc[-1] - equity_series.iloc[0])

    # Returns for risk metrics
    rets = equity_series.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # === Tail risk: VaR / ES (95%) על התשואות היומיות ===
    try:
        if len(rets) > 0:
            q05 = float(np.quantile(rets, 0.05))
            # losses חיוביים (כמה % אפשר להפסיד)
            var_95 = -q05
            tail = rets[rets <= q05]
            es_95 = -float(tail.mean()) if len(tail) > 0 else var_95
        else:
            var_95 = 0.0
            es_95 = 0.0
    except Exception:
        var_95 = 0.0
        es_95 = 0.0

    # === WF: חלוקת התשואות לחלונות ובניית window_metrics ===
    try:
        window_metrics = build_window_metrics_from_returns(rets, n_windows=4)
    except Exception:
        window_metrics = []

    periods_per_year = self._infer_periods_per_year(cfg.bar_freq)
    rf = float(cfg.risk_free_rate)

    if len(rets) > 5:
        mean_ret = float(rets.mean())
        vol = float(rets.std(ddof=0))
        ann_ret = mean_ret * periods_per_year
        ann_vol = vol * math.sqrt(periods_per_year) if vol > 0 else 0.0

        excess = ann_ret - rf
        sharpe = _safe_div(excess, ann_vol, default=0.0)

        # Sortino
        downside = rets.copy()
        downside = downside[downside < 0.0]
        ds_vol = float(downside.std(ddof=0)) if len(downside) > 0 else 0.0
        ds_ann = ds_vol * math.sqrt(periods_per_year) if ds_vol > 0 else 0.0
        sortino = _safe_div(excess, ds_ann, default=0.0)
    else:
        sharpe = 0.0
        sortino = 0.0

    # Max Drawdown & Calmar (based on equity)
    roll_max = equity_series.cummax()
    dd = (roll_max - equity_series) / roll_max
    dd = dd.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    max_dd = float(dd.max())

    # CAGR approx for Calmar
    try:
        days = (equity_series.index[-1] - equity_series.index[0]).days or 1
        years = days / 365.25
    except Exception:
        years = len(equity_series) / periods_per_year if periods_per_year > 0 else 1.0

    if years <= 0:
        calmar = 0.0
    else:
        cagr = (equity_series.iloc[-1] / max(equity_series.iloc[0], 1e-9)) ** (1.0 / years) - 1.0
        calmar = _safe_div(cagr, max_dd, default=0.0) if max_dd > 0 else 0.0

    # Trade stats
    n_trades = len(trade_pnls)
    if n_trades > 0:
        wins = sum(1 for x in trade_pnls if x > 0)
        win_rate = wins / float(n_trades)
        avg_trade_pnl = float(np.mean(trade_pnls))
    else:
        win_rate = 0.0
        avg_trade_pnl = 0.0

    # Exposure & Turnover
    exposure = float(np.mean(np.abs(df["pos"].to_numpy(dtype=float))))
    turnover = float(_safe_div(turnover_notional, cfg.initial_capital * max(len(df), 1), default=0.0))

    stats = {
        "periods_per_year": periods_per_year,
        "rf": rf,
        "n_rows": len(df),
        "n_trades": n_trades,
        "trade_pnls": trade_pnls,
        "avg_daily_return": float(rets.mean()),
        "vol_daily_return": float(rets.std(ddof=0)),
        "total_costs": float(costs.sum()),
    }

    result = BacktestResult(
        sharpe=sharpe,
        profit=profit,
        max_drawdown=max_dd,
        sortino=sortino,
        calmar=calmar,
        win_rate=win_rate,
        n_trades=n_trades,
        avg_trade_pnl=avg_trade_pnl,
        exposure=exposure,
        turnover=turnover,
        var_95=var_95,
        es_95=es_95,
        equity_curve=equity_series,
        returns=rets,
        window_metrics=window_metrics,
        stats=stats,
    )
    return result


def _bt_run(self: "OptimizationBacktester") -> Dict[str, float]:
    """
    Full run implementation (v1):

        1. Prepare data (prices + spread) via _prepare_data()
        2. Extract strategy params from self.params
        3. Build signal frame (z-score, positions, trade IDs)
        4. Simulate equity & compute metrics
        5. Return {Sharpe, Profit, Drawdown} for the optimization tab

    Any exception is caught and converted into a neutral BacktestResult with
    stats['reason'] set, so that Optuna can continue exploring without
    blowing up the whole study.
    """
    try:
        data = self._prepare_data()  # from Part 2
        params = _bt_extract_strategy_params(self)
        sig_df = _bt_build_signal_frame(self, data, params)
        result = _bt_simulate_equity(self, sig_df, params)
        return result.to_perf_dict()
    except Exception as e:
        logger.warning("Backtest run failed for %s-%s: %s",
                       self.config.symbol_a, self.config.symbol_b, e)
        # Fallback neutral result
        fallback = BacktestResult(
            sharpe=0.0,
            profit=0.0,
            max_drawdown=0.0,
            sortino=0.0,
            calmar=0.0,
            win_rate=0.0,
            n_trades=0,
            avg_trade_pnl=0.0,
            exposure=0.0,
            turnover=0.0,
            equity_curve=None,
            stats={
                "reason": "exception",
                "error": str(e),
                "pair": f"{self.config.symbol_a}-{self.config.symbol_b}",
                "config": self.config.to_dict(),
                "params": dict(self.params),
            },
        )
        return fallback.to_perf_dict()


# Attach Part 3 logic to OptimizationBacktester (override placeholder run)
OptimizationBacktester._extract_strategy_params = _bt_extract_strategy_params  # type: ignore[attr-defined]
OptimizationBacktester._build_signals = _bt_build_signal_frame  # type: ignore[attr-defined]
OptimizationBacktester._simulate_equity = _bt_simulate_equity  # type: ignore[attr-defined]
OptimizationBacktester.run = _bt_run  # type: ignore[assignment]

# =========================
# PART 4/6: Advanced Hedge-Ratio, Vol Targeting & Regimes (Pro)
# =========================

def _bt_compute_global_ols_beta(x: pd.Series, y: pd.Series) -> Tuple[float, float, float]:
    """
    Global OLS: y ≈ alpha + beta * x

    Returns
    -------
    beta  : float
    alpha : float
    r2    : float
    """
    xv = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    yv = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(xv) & np.isfinite(yv)
    xv = xv[mask]
    yv = yv[mask]

    if len(xv) < 10:
        return 1.0, 0.0, 0.0

    mx = float(xv.mean())
    my = float(yv.mean())
    cov_xy = float(np.mean(xv * yv) - mx * my)
    var_x = float(np.mean(xv * xv) - mx * mx)
    if var_x <= 0:
        return 1.0, 0.0, 0.0

    beta = cov_xy / var_x
    alpha = my - beta * mx

    # R²
    y_hat = alpha + beta * xv
    ss_res = float(np.sum((yv - y_hat) ** 2))
    ss_tot = float(np.sum((yv - my) ** 2)) or 1.0
    r2 = max(0.0, 1.0 - ss_res / ss_tot)

    return float(beta), float(alpha), float(r2)


def _bt_apply_hedge_ratio_pro(
    self: "OptimizationBacktester",
    data: pd.DataFrame,
    params: Dict[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply a *professional* hedge-ratio model to the pair and rebuild the spread.

    Supported hr_method:
        - "1to1" / "equal" / "naive"
        - "ols"          → global β + α, with R² diagnostics
        - "rolling_ols"  → dynamic β_t (window-based), clipped & smoothed

    כל המתודות מייצרות:
        columns:
            hedge_ratio : β (קבוע או דינמי)
            spread      : px_a - β * px_b
            spread_ret  : התשואה של הספרד
    """
    df = data.copy()
    if df.empty:
        raise ValueError("Cannot apply hedge ratio on empty data.")

    px_a = pd.to_numeric(df["px_a"], errors="coerce").astype(float)
    px_b = pd.to_numeric(df["px_b"], errors="coerce").astype(float)

    method = str(self.params.get("hr_method", params.get("hr_method", "1to1"))).lower().strip()
    hr_stats: Dict[str, Any] = {
        "method": method,
        "dynamic": False,
        "window": None,
        "beta_const": None,
        "alpha_const": None,
        "r2_global": None,
    }

    # Default: 1-to-1
    hr_series = pd.Series(1.0, index=df.index, dtype=float)
    beta_const = 1.0
    alpha_const = 0.0
    r2_global = 0.0

    # ---- 1) naive / equal ----
    if method in ("1to1", "equal", "naive", ""):
        beta_const = 1.0
        alpha_const = 0.0
        hr_series[:] = beta_const

    # ---- 2) global OLS ----
    elif method == "ols":
        beta_const, alpha_const, r2_global = _bt_compute_global_ols_beta(px_b, px_a)
        # הגבלת β לערכים סבירים
        beta_const = float(np.clip(beta_const, -50.0, 50.0))
        hr_series[:] = beta_const

    # ---- 3) rolling OLS (דינמי) ----
    elif method in ("rolling_ols", "dynamic_ols"):
        win = int(self.params.get("hr_lookback", params.get("lookback", 60)))
        win = max(20, min(win, len(df)))
        hr_stats["dynamic"] = True
        hr_stats["window"] = win

        # ראשית β גלובלי, כדי לספק fallback
        beta_const, alpha_const, r2_global = _bt_compute_global_ols_beta(px_b, px_a)
        beta_const = float(np.clip(beta_const, -50.0, 50.0))

        x = px_b
        y = px_a
        mx = x.rolling(win, min_periods=win // 2).mean()
        my = y.rolling(win, min_periods=win // 2).mean()
        mxy = (x * y).rolling(win, min_periods=win // 2).mean()
        mx2 = (x * x).rolling(win, min_periods=win // 2).mean()

        var_x = mx2 - mx**2
        cov_xy = mxy - mx * my

        beta = cov_xy / var_x.replace(0.0, np.nan)
        beta = beta.replace([np.inf, -np.inf], np.nan)
        # fallback ל־β גלובלי
        beta = beta.fillna(beta_const)
        # קליפה לערכים ריאליים
        beta = beta.clip(lower=-50.0, upper=50.0)
        # עדינה: smooth קל
        beta = beta.ewm(span=max(5, win // 4), adjust=False).mean()

        hr_series = beta.astype(float)

    else:
        # unknown method → fallback ל־1:1
        method = "1to1"
        beta_const = 1.0
        alpha_const = 0.0
        hr_series[:] = beta_const

    # בונה ספרד על בסיס β
    df["hedge_ratio"] = hr_series
    spread = px_a - hr_series * px_b
    df["spread"] = spread
    df["spread_ret"] = spread.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)

    hr_stats.update(
        {
            "method": method,
            "beta_const": float(beta_const),
            "alpha_const": float(alpha_const),
            "r2_global": float(r2_global),
            "beta_mean": float(hr_series.mean()),
            "beta_std": float(hr_series.std(ddof=0)),
            "beta_min": float(hr_series.min()),
            "beta_max": float(hr_series.max()),
        }
    )

    return df, hr_stats


def _bt_apply_vol_targeting_pro(
    self: "OptimizationBacktester",
    sig_df: pd.DataFrame,
    params: Dict[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Volatility targeting overlay על המשקולות (w_a, w_b):

    target_vol_ann (בפרמטרים) מייצג:
        סטיית תקן שנתית רצויה לפורטפוליו.

    האלגוריתם:
        1. מחשב תשואות ריאליות של הפורטפוליו לפי w_a, w_b ו־ret_a, ret_b.
        2. מעריך סטיית תקן יומית אמיתית בפוזיציות פעילות.
        3. גוזר מקדם scale = target_daily / realized_daily.
        4. מקליפ את scale לטווח [0.1, 5.0] כדי למנוע לברג' קיצוני.
        5. מכפיל את w_a, w_b ב־scale.
    """
    df = sig_df.copy()
    vt_stats: Dict[str, Any] = {
        "applied": False,
        "scale": 1.0,
        "target_vol_ann": None,
        "realized_vol_daily": None,
        "realized_vol_portfolio_daily": None,
        "sample_size": 0,
    }

    target_vol_ann = float(self.params.get("target_vol_ann", params.get("target_vol_ann", 0.0)))
    if target_vol_ann <= 0.0:
        return df, vt_stats  # overlay כבוי

    if not {"px_a", "px_b", "w_a", "w_b"}.issubset(df.columns):
        return df, vt_stats

    # תשואות הלגים
    df["ret_a"] = df["px_a"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["ret_b"] = df["px_b"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)

    w_a = df["w_a"].astype(float).to_numpy()
    w_b = df["w_b"].astype(float).to_numpy()
    ret_a = df["ret_a"].to_numpy(dtype=float)
    ret_b = df["ret_b"].to_numpy(dtype=float)

    # תשואת פורטפוליו ריאלית (לפני קוסט)
    port_ret = w_a * ret_a + w_b * ret_b
    series_port_ret = pd.Series(port_ret, index=df.index)

    active_mask = df["pos"].astype(float).to_numpy() != 0.0
    rets_active = series_port_ret[active_mask].dropna()
    vt_stats["sample_size"] = int(len(rets_active))

    if len(rets_active) < 20:
        return df, vt_stats

    realized_daily = float(rets_active.std(ddof=0))
    vt_stats["realized_vol_portfolio_daily"] = realized_daily

    # אם אין תנודתיות – אין מה לסקל
    if realized_daily <= 0:
        return df, vt_stats

    periods_per_year = self._infer_periods_per_year(self.config.bar_freq)
    target_daily = float(target_vol_ann) / math.sqrt(periods_per_year) if periods_per_year > 0 else target_vol_ann

    scale = target_daily / realized_daily
    scale = float(np.clip(scale, 0.1, 5.0))  # הגבלת לברג'
    vt_stats["target_vol_ann"] = float(target_vol_ann)
    vt_stats["realized_vol_daily"] = realized_daily
    vt_stats["scale"] = scale
    vt_stats["applied"] = True

    df["w_a"] = df["w_a"].astype(float) * scale
    df["w_b"] = df["w_b"].astype(float) * scale

    return df, vt_stats


def _bt_tag_regimes_pro(
    self: "OptimizationBacktester",
    sig_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Tag regimes based on:
        - Volatility (sigma of spread): "low", "mid", "high"
        - Z-score state: "over", "neutral", "under"
        - Combined regime: "<vol>-<zstate>" (e.g., "high-over")

    Stats שמורים:
        - shares_by_vol
        - shares_by_zstate
        - shares_by_combined
    """
    df = sig_df.copy()
    reg_stats: Dict[str, Any] = {
        "applied": False,
        "shares_by_vol": {},
        "shares_by_zstate": {},
        "shares_by_combined": {},
    }

    if "sigma" not in df.columns or "zscore" not in df.columns:
        return df, reg_stats

    vol = pd.to_numeric(df["sigma"], errors="coerce")
    z = pd.to_numeric(df["zscore"], errors="coerce")

    if vol.dropna().empty or z.dropna().empty:
        return df, reg_stats

    # Vol regimes לפי quantiles
    q_low = float(vol.quantile(0.33))
    q_high = float(vol.quantile(0.67))

    def _vol_reg(v: float) -> str:
        if np.isnan(v):
            return "unknown"
        if v <= q_low:
            return "low"
        if v >= q_high:
            return "high"
        return "mid"

    # Z-state regimes לפי thresholds
    z_th = float(self.params.get("regime_z_threshold", 1.0))

    def _z_state(zz: float) -> str:
        if np.isnan(zz):
            return "unknown"
        if zz >= z_th:
            return "over"     # spread "rich" / מעל הממוצע
        if zz <= -z_th:
            return "under"    # spread "cheap" / מתחת לממוצע
        return "neutral"

    df["vol_regime"] = vol.map(_vol_reg)
    df["z_state"] = z.map(_z_state)
    df["combined_regime"] = df["vol_regime"].astype(str) + "-" + df["z_state"].astype(str)

    # סטטיסטיקות
    reg_stats["applied"] = True
    reg_stats["shares_by_vol"] = (
        df["vol_regime"].value_counts(normalize=True).to_dict()
    )
    reg_stats["shares_by_zstate"] = (
        df["z_state"].value_counts(normalize=True).to_dict()
    )
    reg_stats["shares_by_combined"] = (
        df["combined_regime"].value_counts(normalize=True).to_dict()
    )

    return df, reg_stats


def _bt_run_professional(self: "OptimizationBacktester") -> Dict[str, float]:
    """
    Professional run() pipeline:

        1. Load & prepare base spread data          → _prepare_data()
        2. Extract normalized strategy parameters   → _extract_strategy_params()
        3. Apply advanced hedge-ratio model         → _bt_apply_hedge_ratio_pro()
        4. Build signal frame (z, pos, trades)      → _build_signals()
        5. Tag volatility/Z regimes (diagnostic)    → _bt_tag_regimes_pro()
        6. Apply volatility targeting overlay       → _bt_apply_vol_targeting_pro()
        7. Simulate equity & compute metrics        → _simulate_equity()
        8. Enrich BacktestResult.stats with overlays & regimes
        9. Cache last frames/result on self for debugging/analytics.

    מחזיר:
        dict { "Sharpe", "Profit", "Drawdown" }
    כמו שהטאב מצפה.
    """
    try:
        # 1–2: base data + params
        data = self._prepare_data()
        params = self._extract_strategy_params()

        # 3: hedge-ratio
        data_hr, hr_stats = _bt_apply_hedge_ratio_pro(self, data, params)

        # 4: signals (incl. pos, w_a, w_b, trade_id, holding_days)
        sig_df = self._build_signals(data_hr, params)

        # 5: regimes
        sig_df, reg_stats = _bt_tag_regimes_pro(self, sig_df)

        # 6: vol targeting overlay
        sig_df_vt, vt_stats = _bt_apply_vol_targeting_pro(self, sig_df, params)

        # 7: simulation
        result = self._simulate_equity(sig_df_vt, params)

        # 8: enrich stats
        overlays = result.stats.get("overlays", {})
        overlays["hedge_ratio"] = hr_stats
        overlays["regimes"] = reg_stats
        overlays["vol_targeting"] = vt_stats
        result.stats["overlays"] = overlays

        # 9: cache last frames / result on instance
        try:
            self._last_data = data_hr          # type: ignore[attr-defined]
            self._last_signals = sig_df_vt     # type: ignore[attr-defined]
            self._last_result = result         # type: ignore[attr-defined]
        except Exception:
            pass

        return result.to_perf_dict()

    except Exception as e:
        logger.warning(
            "Professional backtest run failed for %s-%s: %s",
            self.config.symbol_a,
            self.config.symbol_b,
            e,
        )
        fallback = BacktestResult(
            sharpe=0.0,
            profit=0.0,
            max_drawdown=0.0,
            sortino=0.0,
            calmar=0.0,
            win_rate=0.0,
            n_trades=0,
            avg_trade_pnl=0.0,
            exposure=0.0,
            turnover=0.0,
            equity_curve=None,
            stats={
                "reason": "exception_professional_run",
                "error": str(e),
                "pair": f"{self.config.symbol_a}-{self.config.symbol_b}",
                "config": self.config.to_dict(),
                "params": dict(self.params),
            },
        )
        try:
            self._last_result = fallback  # type: ignore[attr-defined]
        except Exception:
            pass
        return fallback.to_perf_dict()


# Attach Part 4 Pro logic to OptimizationBacktester (override previous run/overlays)
OptimizationBacktester._apply_hedge_ratio = _bt_apply_hedge_ratio_pro  # type: ignore[attr-defined]
OptimizationBacktester._apply_vol_targeting = _bt_apply_vol_targeting_pro  # type: ignore[attr-defined]
OptimizationBacktester._tag_regimes = _bt_tag_regimes_pro  # type: ignore[attr-defined]
OptimizationBacktester.run = _bt_run_professional  # type: ignore[assignment]

# =========================
# PART 5/6: Stationarity, Risk Profile & Scenario Engine (Ultra Pro)
# =========================

# Optional: statsmodels for cointegration & stationarity tests
try:
    from statsmodels.tsa.stattools import coint as _sm_coint  # type: ignore[attr-defined]
    from statsmodels.tsa.stattools import adfuller as _sm_adf  # type: ignore[attr-defined]
    from statsmodels.tsa.stattools import kpss as _sm_kpss  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - optional dependency
    _sm_coint = None  # type: ignore[assignment]
    _sm_adf = None  # type: ignore[assignment]
    _sm_kpss = None  # type: ignore[assignment]


# ---------- 5.1 Stationarity helpers ----------

def _bt_safe_adf(x: pd.Series) -> Dict[str, Any]:
    """Run ADF test safely; return structured dict or 'unavailable'."""
    if _sm_adf is None:
        return {"available": False, "reason": "statsmodels_adf_missing"}

    arr = pd.to_numeric(x, errors="coerce").dropna().to_numpy(dtype=float)
    if len(arr) < 30:
        return {"available": False, "reason": "too_short", "n": len(arr)}

    try:
        stat, pval, *_rest = _sm_adf(arr, autolag="AIC")
        crit = _rest[2] if len(_rest) >= 3 else {}
        return {
            "available": True,
            "stat": float(stat),
            "pvalue": float(pval),
            "crit": {k: float(v) for k, v in crit.items()},
            "n": int(len(arr)),
        }
    except Exception as e:  # pragma: no cover - defensive
        return {"available": False, "reason": f"error: {e}"}


def _bt_safe_kpss(x: pd.Series) -> Dict[str, Any]:
    """Run KPSS test safely; return structured dict or 'unavailable'."""
    if _sm_kpss is None:
        return {"available": False, "reason": "statsmodels_kpss_missing"}

    arr = pd.to_numeric(x, errors="coerce").dropna().to_numpy(dtype=float)
    if len(arr) < 30:
        return {"available": False, "reason": "too_short", "n": len(arr)}

    try:
        stat, pval, lags, crit = _sm_kpss(arr, regression="c", nlags="auto")
        return {
            "available": True,
            "stat": float(stat),
            "pvalue": float(pval),
            "lags": int(lags),
            "crit": {k: float(v) for k, v in crit.items()},
            "n": int(len(arr)),
        }
    except Exception as e:  # pragma: no cover
        return {"available": False, "reason": f"error: {e}"}


def _bt_safe_coint(x: pd.Series, y: pd.Series) -> Dict[str, Any]:
    """Run Engle-Granger cointegration test safely."""
    if _sm_coint is None:
        return {"available": False, "reason": "statsmodels_coint_missing"}

    x_arr = pd.to_numeric(x, errors="coerce").dropna().to_numpy(dtype=float)
    y_arr = pd.to_numeric(y, errors="coerce").dropna().to_numpy(dtype=float)

    n = min(len(x_arr), len(y_arr))
    if n < 30:
        return {"available": False, "reason": "too_short", "n": n}

    x_arr = x_arr[-n:]
    y_arr = y_arr[-n:]

    try:
        stat, pval, crit = _sm_coint(x_arr, y_arr)
        return {
            "available": True,
            "stat": float(stat),
            "pvalue": float(pval),
            "crit": {k: float(v) for k, v in crit.items()},
            "n": int(n),
        }
    except Exception as e:  # pragma: no cover
        return {"available": False, "reason": f"error: {e}"}


def _bt_stationarity_and_cointegration(
    self: "OptimizationBacktester",
    data: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Compute stationarity & cointegration diagnostics for the pair:

        - ADF & KPSS על px_a, px_b
        - ADF & KPSS על הספרד
        - Engle-Granger coint test על (px_a, px_b)

    התוצאה נשמרת כחלק מ־stats["advanced"]["stationarity"].
    """
    stats: Dict[str, Any] = {
        "px_a_adf": {},
        "px_b_adf": {},
        "spread_adf": {},
        "spread_kpss": {},
        "coint": {},
    }

    if data.empty:
        stats["reason"] = "empty_data"
        return stats

    px_a = pd.to_numeric(data["px_a"], errors="coerce")
    px_b = pd.to_numeric(data["px_b"], errors="coerce")
    spread = pd.to_numeric(data.get("spread", px_a - px_b), errors="coerce")

    stats["px_a_adf"] = _bt_safe_adf(px_a)
    stats["px_b_adf"] = _bt_safe_adf(px_b)
    stats["spread_adf"] = _bt_safe_adf(spread)
    stats["spread_kpss"] = _bt_safe_kpss(spread)
    stats["coint"] = _bt_safe_coint(px_a, px_b)

    return stats


# ---------- 5.2 Risk profile from equity & returns ----------

def _bt_compute_risk_profile(
    self: "OptimizationBacktester",
    equity: pd.Series,
) -> Dict[str, Any]:
    """
    Compute an extended risk profile from the equity curve:

        - daily returns distribution
        - VaR / ES (95%, 99%)
        - up / down capture (מעין סימולציה פנימית)
        - path stats: longest DD, time-under-water
    """
    if not isinstance(equity, pd.Series) or equity.empty:
        return {"reason": "no_equity"}

    eq = pd.to_numeric(equity, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if eq.empty:
        return {"reason": "no_equity_valid"}

    rets = eq.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if rets.empty:
        return {"reason": "no_returns"}

    # Basic distribution stats
    mu = float(rets.mean())
    sigma = float(rets.std(ddof=0))
    skew = float(rets.skew())
    kurt = float(rets.kurtosis())

    # VaR / ES (Parametric + Historical)
    def _var_es(series: pd.Series, alpha: float) -> Dict[str, float]:
        s = series.dropna()
        if s.empty:
            return {"var_hist": 0.0, "es_hist": 0.0, "var_norm": 0.0, "es_norm": 0.0}

        # Historical (left tail)
        q = float(s.quantile(alpha))
        es = float(s[s <= q].mean()) if (s <= q).any() else q

        # Parametric Normal
        mu_loc = float(s.mean())
        sig_loc = float(s.std(ddof=0)) or 1e-9
        from scipy.stats import norm as _norm  # type: ignore[attr-defined]

        z = float(_norm.ppf(alpha))
        var_n = mu_loc + z * sig_loc
        # ES (for normal) ≈ mu - σ * φ(z)/α  (לזנב שמאל)
        es_n = float(mu_loc - sig_loc * _norm.pdf(z) / alpha)

        return {
            "var_hist": q,
            "es_hist": es,
            "var_norm": var_n,
            "es_norm": es_n,
        }

    try:
        rp_95 = _var_es(rets, 0.05)
        rp_99 = _var_es(rets, 0.01)
    except Exception:  # אם scipy חסר וכו' — רק Historical
        rp_95 = _var_es(rets, 0.05)  # type: ignore[arg-type]
        rp_99 = _var_es(rets, 0.01)  # type: ignore[arg-type]

    # Drawdown path stats
    roll_max = eq.cummax()
    dd = (roll_max - eq) / roll_max
    dd = dd.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    max_dd = float(dd.max())
    under_water = dd > 0.0
    # longest run underwater
    longest_under = 0
    curr = 0
    for flag in under_water:
        if flag:
            curr += 1
            longest_under = max(longest_under, curr)
        else:
            curr = 0

    profile: Dict[str, Any] = {
        "mu_daily": mu,
        "sigma_daily": sigma,
        "skew": skew,
        "kurtosis": kurt,
        "max_drawdown": max_dd,
        "underwater_longest_obs": int(longest_under),
        "VaR_95": rp_95,
        "VaR_99": rp_99,
        "n_obs": int(len(rets)),
    }
    return profile


# ---------- 5.3 Scenario engines (cost & stress) ----------

def _bt_run_cost_scenarios(
    self: "OptimizationBacktester",
    sig_df: pd.DataFrame,
    params: Dict[str, Any],
) -> Dict[str, Dict[str, float]]:
    """
    Cost-sensitivity scenarios:

        - cost_0_5x : half costs (comm+slip)
        - cost_1_0x : base
        - cost_2_0x : double costs

    השיטה מסלימה את config.commission_bps / slippage_bps זמנית,
    מריצה self._simulate_equity על אותו sig_df, ואז מחזירה אותם לערכים המקוריים.
    """
    cfg = self.config
    scenarios: Dict[str, Dict[str, float]] = {}

    base_comm = float(cfg.commission_bps)
    base_slip = float(cfg.slippage_bps)

    def _sim_with_cost(mult: float) -> Dict[str, float]:
        cfg.commission_bps = base_comm * mult
        cfg.slippage_bps = base_slip * mult
        try:
            res = self._simulate_equity(sig_df, params)
            return res.to_perf_dict()
        finally:
            cfg.commission_bps = base_comm
            cfg.slippage_bps = base_slip

    for mult, name in [(0.5, "cost_0_5x"), (1.0, "cost_1_0x"), (2.0, "cost_2_0x")]:
        try:
            scenarios[name] = _sim_with_cost(mult)
        except Exception as e:  # pragma: no cover
            logger.debug("Cost scenario %s failed: %s", name, e)
            scenarios[name] = {
                "Sharpe": 0.0,
                "Profit": 0.0,
                "Drawdown": 0.0,
            }

    return scenarios


def _bt_run_stress_scenarios(
    self: "OptimizationBacktester",
    equity: pd.Series,
) -> Dict[str, Dict[str, Any]]:
    """
    Stress scenarios על בסיס עקומת ההון שכבר חישבנו:

        - shock_eq_-10 : הורדת ההון ב־10% מיידית ובדיקת DD חדש.
        - shock_eq_-20 : כמו 20%.
        - dd_150pct    : הגדלת כל drawdown פי 1.5 (סימולציה ישירה על DD).

    לא מריצים backtest מחדש — עובדים על equity curve ומחזירים:
        - max_drawdown
        - equity_final
        - pseudo_VaR_95 (על הסנריו)
    """
    scenarios: Dict[str, Dict[str, Any]] = {}

    if not isinstance(equity, pd.Series) or equity.empty:
        return scenarios

    eq = pd.to_numeric(equity, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if eq.empty:
        return scenarios

    def _eq_stats(eq_s: pd.Series) -> Dict[str, Any]:
        roll_max = eq_s.cummax()
        dd = (roll_max - eq_s) / roll_max
        dd = dd.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        max_dd = float(dd.max())
        rets = eq_s.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        if rets.empty:
            var_95 = 0.0
        else:
            var_95 = float(rets.quantile(0.05))
        return {
            "equity_final": float(eq_s.iloc[-1]),
            "max_drawdown": max_dd,
            "pseudo_VaR_95": var_95,
        }

    # shock -10% / -20% בתחילת הסדרה
    for shock, name in [(0.10, "shock_eq_-10"), (0.20, "shock_eq_-20")]:
        try:
            eq_s = eq.copy()
            eq_s.iloc[0] = eq_s.iloc[0] * (1.0 - shock)
            scenarios[name] = _eq_stats(eq_s)
        except Exception as e:  # pragma: no cover
            logger.debug("Stress scenario %s failed: %s", name, e)

    # drawdown 150% (החמרת DD)
    try:
        roll_max = eq.cummax()
        dd = (roll_max - eq) / roll_max
        dd = dd.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        # נבנה "equity" סינטטי שמייצג DD מחומר ב־150%
        max_lvl = eq.iloc[0]
        eq_stressed = []
        for d in dd:
            # equity_t ≈ max_lvl * (1 - min(d * 1.5, 0.99))
            eq_stressed.append(max_lvl * (1.0 - min(d * 1.5, 0.99)))
            max_lvl = max(max_lvl, eq_stressed[-1])
        eq_stressed = pd.Series(eq_stressed, index=eq.index)
        scenarios["dd_150pct"] = _eq_stats(eq_stressed)
    except Exception as e:  # pragma: no cover
        logger.debug("Stress scenario dd_150pct failed: %s", e)

    return scenarios


# ---------- 5.4 Wrap professional run with full diagnostics ----------

# Save original professional run from Part 4
_BT_ORIG_RUN_PROFESSIONAL = _bt_run_professional  # type: ignore[name-defined]


def _bt_run_professional_plus(self: "OptimizationBacktester") -> Dict[str, float]:
    """
    Wrapper על `_bt_run_professional`:

        1. מריץ את ה־pipeline המקצועי (Part 4).
        2. לוקח data + signals + result שהוטמנו על self (בחלק 4).
        3. מחשב:
            - stationarity & cointegration
            - risk_profile (VaR/ES, DD, skew/kurtosis)
            - cost_scenarios (אם enable_scenarios=True או force_cost_scenarios=True)
            - stress_scenarios על עקומת ההון
        4. מכניס הכל ל־result.stats["advanced"].
        5. שומר _last_result המעודכן.

    מחזיר:
        dict {"Sharpe", "Profit", "Drawdown"} — כמו תמיד.
    """
    perf = _BT_ORIG_RUN_PROFESSIONAL(self)  # הריצה המקצועית הבסיסית

    try:
        data = getattr(self, "_last_data", None)
        sig_df = getattr(self, "_last_signals", None)
        result = getattr(self, "_last_result", None)

        if result is None:
            return perf

        advanced: Dict[str, Any] = result.stats.get("advanced", {})

        # Stationarity & cointegration
        if isinstance(data, pd.DataFrame) and not data.empty:
            advanced["stationarity"] = _bt_stationarity_and_cointegration(self, data)

        # Risk profile from equity
        if isinstance(result.equity_curve, pd.Series) and not result.equity_curve.empty:
            advanced["risk_profile"] = _bt_compute_risk_profile(self, result.equity_curve)

        # Cost scenarios (אופציונלי)
        enable_scenarios = bool(
            self.params.get("enable_scenarios", False)
            or self.params.get("force_cost_scenarios", False)
        )
        if enable_scenarios and isinstance(sig_df, pd.DataFrame) and not sig_df.empty:
            try:
                scen = _bt_run_cost_scenarios(self, sig_df, self._extract_strategy_params())
                advanced["cost_scenarios"] = scen
            except Exception as e:
                advanced["cost_scenarios_error"] = str(e)

        # Stress scenarios על עקומת ההון
        if isinstance(result.equity_curve, pd.Series) and not result.equity_curve.empty:
            try:
                stress = _bt_run_stress_scenarios(self, result.equity_curve)
                advanced["stress_scenarios"] = stress
            except Exception as e:
                advanced["stress_scenarios_error"] = str(e)

        result.stats["advanced"] = advanced
        try:
            self._last_result = result  # type: ignore[attr-defined]
        except Exception:
            pass

    except Exception as e:  # pragma: no cover - לא להכשיל אופטימיזציה על דיאגנוסטיקה
        logger.debug("Professional+ diagnostics failed (ignored): %s", e)

    return perf


# Override .run with the extended professional+ pipeline
OptimizationBacktester.run = _bt_run_professional_plus  # type: ignore[assignment]

# =========================
# PART 6/6: Debug Snapshots, Export & Convenience Helper
# =========================

def _bt_debug_snapshot(
    self: "OptimizationBacktester",
    head: int = 50,
) -> Dict[str, Any]:
    """
    מחזיר snapshot קומפקטי של ה־backtest האחרון:

        - config & params
        - data_head (אחרי HR)
        - signals_head (אחרי vol targeting)
        - perf & stats
        - summary של regimes אם קיימים

    שימוש:
        bt = OptimizationBacktester("XLY","XLP", ...)
        bt.run()
        snap = bt.debug_snapshot()
    """
    cfg_dict = self.config.to_dict()
    params = dict(self.params)

    data = getattr(self, "_last_data", None)
    sig_df = getattr(self, "_last_signals", None)
    result = getattr(self, "_last_result", None)

    snap: Dict[str, Any] = {
        "pair": f"{self.config.symbol_a}-{self.config.symbol_b}",
        "config": cfg_dict,
        "params": params,
    }

    # Data head
    try:
        if isinstance(data, pd.DataFrame) and not data.empty:
            snap["data_head"] = data.head(head).to_dict(orient="list")
        else:
            snap["data_head"] = None
    except Exception:
        snap["data_head"] = None

    # Signals head
    try:
        if isinstance(sig_df, pd.DataFrame) and not sig_df.empty:
            snap["signals_head"] = sig_df.head(head).to_dict(orient="list")
        else:
            snap["signals_head"] = None
    except Exception:
        snap["signals_head"] = None

    # Regimes summary אם יש
    try:
        if isinstance(sig_df, pd.DataFrame) and not sig_df.empty and "combined_regime" in sig_df.columns:
            snap["regimes_summary"] = (
                sig_df["combined_regime"]
                .value_counts(normalize=True)
                .to_dict()
            )
        else:
            snap["regimes_summary"] = None
    except Exception:
        snap["regimes_summary"] = None

    # Perf & stats
    if result is not None:
        snap["perf"] = result.to_perf_dict()
        snap["stats"] = result.stats
    else:
        snap["perf"] = None
        snap["stats"] = {}

    return snap


def _bt_to_dict(self: "OptimizationBacktester") -> Dict[str, Any]:
    """
    Serialize the backtester + last result into JSON-safe dict —
    נוח לשמירה ב־DuckDB / קבצי לוג / S3.
    """
    out: Dict[str, Any] = {
        "pair": f"{self.config.symbol_a}-{self.config.symbol_b}",
        "config": self.config.to_dict(),
        "params": dict(self.params),
    }
    result = getattr(self, "_last_result", None)
    if result is not None:
        out["perf"] = result.to_perf_dict()
        out["stats"] = result.stats
    else:
        out["perf"] = None
        out["stats"] = {}
    return out


def _bt_export_frames(
    self: "OptimizationBacktester",
    folder: str | Path,
    prefix: Optional[str] = None,
    limit_rows: int = 50_000,
) -> Dict[str, str]:
    """
    Export latest data/signals/equity to CSV (או Parquet אם תרצה להרחיב):

        - <prefix>_data.csv
        - <prefix>_signals.csv
        - <prefix>_equity.csv

    מחזיר dict עם הנתיבים שנשמרו בפועל.
    """
    base = Path(folder).expanduser().resolve()
    base.mkdir(parents=True, exist_ok=True)
    if not prefix:
        prefix = f"{self.config.symbol_a}-{self.config.symbol_b}"

    paths: Dict[str, str] = {}

    data = getattr(self, "_last_data", None)
    sig_df = getattr(self, "_last_signals", None)
    result = getattr(self, "_last_result", None)

    try:
        if isinstance(data, pd.DataFrame) and not data.empty:
            p = base / f"{prefix}_data.csv"
            data.head(limit_rows).to_csv(p, index=True)
            paths["data"] = str(p)
    except Exception as e:
        logger.debug("export data failed: %s", e)

    try:
        if isinstance(sig_df, pd.DataFrame) and not sig_df.empty:
            p = base / f"{prefix}_signals.csv"
            sig_df.head(limit_rows).to_csv(p, index=True)
            paths["signals"] = str(p)
    except Exception as e:
        logger.debug("export signals failed: %s", e)

    try:
        if result is not None and isinstance(result.equity_curve, pd.Series):
            p = base / f"{prefix}_equity.csv"
            result.equity_curve.to_csv(p, index=True, header=True)
            paths["equity"] = str(p)
    except Exception as e:
        logger.debug("export equity failed: %s", e)

    return paths


def run_backtest_pair(
    symbol_a: str,
    symbol_b: str,
    **params: Any,
) -> Dict[str, float]:
    """
    Convenience function לשימוש חיצוני:

        perf = run_backtest_pair(
            "XLY", "XLP",
            lookback=60,
            z_open=2.0,
            z_close=0.5,
            hr_method="rolling_ols",
            target_vol_ann=0.20,
            enable_scenarios=True,
            ...
        )

    מחזיר:
        {"Sharpe": ..., "Profit": ..., "Drawdown": ...}
    """
    bt = OptimizationBacktester(symbol_a=symbol_a, symbol_b=symbol_b, **params)
    return bt.run()


# Attach helpers to the class
OptimizationBacktester.debug_snapshot = _bt_debug_snapshot  # type: ignore[attr-defined]
OptimizationBacktester.to_dict = _bt_to_dict  # type: ignore[attr-defined]
OptimizationBacktester.export_frames = _bt_export_frames  # type: ignore[attr-defined]
