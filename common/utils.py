# -*- coding: utf-8 -*-
"""
common/utils.py — unified quant utilities (v10, HF-grade)
==========================================================

תפקיד המודול
------------
מודול Utilities מרכזי ברמת קרן גידור, ללא תלות ב-Yahoo Finance:

1. טיפוסי עזר ו-helpers כלליים:
   - ensure_series, normalize_symbol, metric helpers.

2. שכבת Market Data *אגרסיבית נגד Yahoo*:
   - load_price_data / load_price_panel / load_price_data_range / get_close_series_range
   - כרגע: **לא משתמשים ב-yfinance בכלל**.
   - כל הפונקציות הללו זורקות RuntimeError עם הודעה ברורה:
     "Yahoo/yfinance backend disabled — use IBKR/SqlStore instead".
   - החיבור ל-IBKR/SqlStore ייעשה דרך data-loader ייעודי (core/sql_store, ib_data_ingestor וכו’).

3. סטטיסטיקה בסיסית:
   - Z-score, Half-life, Correlation, Volatility Z-score, Historical Vol.

4. Risk & Performance:
   - log/simple returns, drawdown path, max drawdown,
   - rolling Sharpe & Sortino.

5. Hedge & Spread:
   - OLS hedge ratio, Kalman hedge ratio (עם fallback ל-OLS),
   - spread computation, basic_pair_metrics.

6. Risk Parity:
   - risk_parity_weights על בסיס inverse-vol (future-ready ל-eigendecomposition עמוק יותר).

7. Statistical Tests:
   - pearson_pvalue, adf_pvalue, johansen_trace.

8. Pair quality scoring:
   - evaluate_pair_quality (DataFrame או מטריקות),
   - evaluate_edge (zscore + correlation → score [0,1]).

9. JSON export:
   - metrics_to_json.

Backwards compatibility
-----------------------
שמות ציבוריים שנשמרים:
- ensure_series, load_price_data, calculate_zscore, calculate_half_life,
  calculate_correlation, calculate_volatility_zscore, calculate_historical_volatility,
  get_implied_volatility, calculate_beta, hurst_exponent, ols_hedge_ratio,
  kalman_hedge_ratio, risk_parity_weights, pearson_pvalue, adf_pvalue,
  johansen_trace, evaluate_pair_quality, evaluate_edge.

הערה חשובה
-----------
פונקציות טעינת מחירים *לא עושות כלום כרגע חוץ מלהגיד לך בבירור*:
"לא משתמשים ב-Yahoo. חבר אותי ל-IBKR/SqlStore".
כך אתה מוגן מפני שימוש לא מודע ב-yfinance, ותוכל לחבר מקור דאטה יחיד ונקי.
"""

from __future__ import annotations

import ast
import functools
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, overload

import numpy as np
import pandas as pd
from numpy.linalg import eigh
from scipy.stats import pearsonr
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from common.json_safe import make_json_safe, json_default as _json_default

# -----------------------------------------------------------------------------
# Logger
# -----------------------------------------------------------------------------

logger = logging.getLogger(__name__)
if not logger.handlers:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)

# -----------------------------------------------------------------------------
# Type aliases
# -----------------------------------------------------------------------------

NumericLikeSeries = Union[pd.Series, Sequence[float], np.ndarray]

# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------

def ensure_series(x: NumericLikeSeries, name: str = "value") -> pd.Series:
    """Ensure x is returned as a pandas Series."""
    return x if isinstance(x, pd.Series) else pd.Series(x, name=name)


# -----------------------------------------------------------------------------
# Symbol parsing (no data source here)
# -----------------------------------------------------------------------------

def _parse_symbol_input(symbols: Union[str, Sequence[str]]) -> List[str]:
    """Robustly parse symbol input into a clean list of tickers.

    Handles:
    - "XLY"
    - ["XLY", "XLC"] / ("XLY", "XLC") / Index / ndarray
    - "['XLY', 'XLC']"
    - "{'SYMBOLS': ['XLY', 'XLC']}" / "{'symbols': [...]}" (stringified dict)
    - dict-like objects with keys SYMBOLS / symbols / sym_x,sym_y.
    """
    # dict-like passed directly
    if isinstance(symbols, dict):
        if "symbols" in symbols:
            return _parse_symbol_input(symbols["symbols"])
        if "SYMBOLS" in symbols:
            return _parse_symbol_input(symbols["SYMBOLS"])
        if {"sym_x", "sym_y"}.issubset(symbols):
            return [str(symbols["sym_x"]).strip(), str(symbols["sym_y"]).strip()]
        # fall back to all values
        out: List[str] = []
        for v in symbols.values():
            out.extend(_parse_symbol_input(v))
        # dedupe
        seen: set[str] = set()
        uniq: List[str] = []
        for s in out:
            s2 = str(s).strip()
            if s2 and s2 not in seen:
                seen.add(s2)
                uniq.append(s2)
        return uniq

    # list / array / index etc.
    if isinstance(symbols, (list, tuple, set, np.ndarray, pd.Index)):
        return [str(s).strip() for s in symbols if str(s).strip()]

    s = str(symbols).strip()
    if not s:
        return []

    # Try ast.literal_eval for stringified lists / dicts
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, (list, tuple, set)):
            return [str(v).strip() for v in parsed if str(v).strip()]
        if isinstance(parsed, dict):
            if "SYMBOLS" in parsed:
                v = parsed["SYMBOLS"]
                if isinstance(v, (list, tuple, set)):
                    return [str(x).strip() for x in v if str(x).strip()]
                return [str(v).strip()]
            if "symbols" in parsed:
                v = parsed["symbols"]
                if isinstance(v, (list, tuple, set)):
                    return [str(x).strip() for x in v if str(x).strip()]
                return [str(v).strip()]
            if {"sym_x", "sym_y"}.issubset(parsed):
                return [str(parsed["sym_x"]).strip(), str(parsed["sym_y"]).strip()]
    except Exception:
        # Fallback to simple string heuristics
        pass

    cleaned = (
        s.replace("[", "")
        .replace("]", "")
        .replace("{", "")
        .replace("}", "")
        .replace("'", "")
        .replace('"', "")
    )

    if "," in cleaned:
        parts = cleaned.split(",")
    else:
        parts = cleaned.split()

    return [p.strip() for p in parts if p.strip()]


def normalize_symbol(symbol: Union[str, Sequence[str], Dict[str, Any]]) -> str:
    """Normalize arbitrary symbol input to a single clean ticker string.

    Thin wrapper around `_parse_symbol_input`, convenient when
    you know you want exactly one ticker.
    """
    tickers = _parse_symbol_input(symbol)
    if not tickers:
        raise ValueError(f"normalize_symbol: empty/invalid symbol input: {symbol!r}")
    if len(tickers) > 1:
        logger.debug(
            "normalize_symbol: got multiple %s, using first (%s)", tickers, tickers[0]
        )
    return tickers[0]


# -----------------------------------------------------------------------------
# Market Data loaders — HF-grade, NO YAHOO
# -----------------------------------------------------------------------------
#
# רעיון:
#   - כאן נמצאות פונקציות העזר שהקוד שלך כבר משתמש בהן (load_price_data וכו’),
#     אבל *הן לא משתמשות ב-yfinance* יותר.
#   - במקום זה, הן זורקות RuntimeError ברור, כדי שתדע לחבר אותן ל-IBKR/SqlStore.
#   - כשתרצה, אפשר יהיה להחליף את השורות הזורקות בקוד שקורא ל-SqlStore / IBKR.
#
# אסטרטגיה:
#   - מחזיקים את ה-API הקיים כדי לא לשבור imports.
#   - מגנים על המערכת מפני שימוש לא מודע ב-Yahoo.
# -----------------------------------------------------------------------------

@functools.lru_cache(maxsize=128)
def load_price_data(
    symbol: Union[str, Sequence[str]],
    period: str = "6mo",
    *,
    auto_adjust: bool = False,
    **backend_kwargs: Any,
) -> pd.DataFrame:
    """
    HF-grade market data loader (period-based) — *backend disabled*.

    כרגע:
    -----
    - הפונקציה הזו לא קוראת לשום מקור דאטה אמיתי.
    - היא רק מזריקה RuntimeError ברור:
        "Yahoo/yfinance backend disabled. Use IBKR/SqlStore instead."

    איך להשתמש בעתיד:
    ------------------
    - בצע ingestion מ-IBKR לתוך SqlStore (DuckDB/SQLite).
    - כתוב פונקציה נפרדת (למשל core.data_loader.load_price_history_from_store)
      שמחזירה מחירי OHLCV מתוך SqlStore.
    - קרא לפונקציה הזו במקום ה-RuntimeError.
    """
    ticker = normalize_symbol(symbol)
    raise RuntimeError(
        f"load_price_data: data backend disabled for symbol={ticker!r}. "
        "This function must be wired to IBKR/SqlStore (no Yahoo)."
    )


def load_price_panel(
    symbols: Sequence[Union[str, Sequence[str]]],
    period: str = "6mo",
    *,
    auto_adjust: bool = False,
    **backend_kwargs: Any,
) -> pd.DataFrame:
    """
    HF-grade universe loader (panel) — *backend disabled*.

    אמור להחזיר מטריצת מחירי close לכל יקום הסימבולים.

    כרגע:
    -----
    - מזריק RuntimeError.
    - משמש כנקודת חיבור עתידית ל-SqlStore / IBKR universe.
    """
    raise RuntimeError(
        "load_price_panel: data backend disabled. "
        "Wire this function to IBKR/SqlStore (e.g. prices table → close matrix)."
    )


def load_price_data_range(
    symbol: Union[str, Sequence[str], Dict[str, Any]],
    start: Union[str, pd.Timestamp],
    end: Union[str, pd.Timestamp],
    *,
    auto_adjust: bool = False,
    **backend_kwargs: Any,
) -> pd.DataFrame:
    """
    HF-grade range-based loader — *backend disabled*.

    אמור לטעון OHLCV עבור symbol בין [start, end] ממקור אמת (IBKR/SqlStore).

    כרגע:
    -----
    - מזריק RuntimeError ברור כדי לא לאפשר שימוש ב-Yahoo.
    """
    ticker = normalize_symbol(symbol)
    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)
    raise RuntimeError(
        f"load_price_data_range: data backend disabled for symbol={ticker!r} "
        f"(start={start_ts}, end={end_ts}). "
        "Connect this function to IBKR/SqlStore (no Yahoo)."
    )


def get_close_series_range(
    symbol: Union[str, Sequence[str], Dict[str, Any]],
    start: Union[str, pd.Timestamp],
    end: Union[str, pd.Timestamp],
    *,
    auto_adjust: bool = False,
    **backend_kwargs: Any,
) -> pd.Series:
    """
    Convenience wrapper returning only the `close` series between [start, end].

    כרגע:
    -----
    - פשוט קורא ל-load_price_data_range (שזורק RuntimeError).
    - ברגע שתחבר את load_price_data_range ל-IBKR/SqlStore, הפונקציה הזו
      תתחיל לעבוד באופן שקוף.
    """
    df = load_price_data_range(symbol, start, end, auto_adjust=auto_adjust, **backend_kwargs)
    if df.empty:
        return pd.Series(name=normalize_symbol(symbol), dtype=float)
    s = df["close"].astype(float)
    s.name = normalize_symbol(symbol)
    return s


# -----------------------------------------------------------------------------
# Core statistics
# -----------------------------------------------------------------------------

def calculate_zscore(x: pd.Series, y: pd.Series, window: int = 20) -> float:
    """Z-score של spread בין שתי סדרות מחירים."""
    spread = x - y
    mu = spread.rolling(window).mean()
    sd = spread.rolling(window).std()
    z = (spread - mu) / sd
    return float(z.iloc[-1]) if len(z) and not np.isnan(z.iloc[-1]) else 0.0


def calculate_half_life(spread: pd.Series) -> float:
    """Half-life של mean reversion עבור spread."""
    lagged = spread.shift(1).dropna()
    delta = spread.diff().dropna()
    if len(lagged) == 0 or len(delta) == 0:
        return float("nan")
    beta, _ = np.polyfit(lagged, delta, 1)
    return max(min(-np.log(2) / beta, 1000), 0) if beta else np.nan


def calculate_correlation(x: pd.Series, y: pd.Series) -> float:
    """קורלציה פשוטה בין שתי סדרות."""
    return float(x.corr(y))


def calculate_volatility_zscore(series: pd.Series, window: int = 20) -> float:
    """Z-score של סטיית תקן רולינג על תשואות."""
    vol = series.pct_change().rolling(window).std()
    mu = vol.rolling(window).mean()
    sd = vol.rolling(window).std()
    if len(vol) == 0 or len(mu) == 0 or len(sd) == 0:
        return 0.0
    if sd.iloc[-1] == 0 or np.isnan(sd.iloc[-1]):
        return 0.0
    return float((vol.iloc[-1] - mu.iloc[-1]) / sd.iloc[-1])


def calculate_historical_volatility(series: pd.Series) -> float:
    """Historical volatility שנתי על בסיס log-returns."""
    lr = np.log(series / series.shift(1)).dropna()
    return float(lr.std() * np.sqrt(252))


# -----------------------------------------------------------------------------
# Return & drawdown utilities (HF-grade)
# -----------------------------------------------------------------------------

def to_log_returns(price: pd.Series) -> pd.Series:
    """
    Convert a price series into log-returns.

    log_return_t = log(p_t / p_{t-1})
    """
    s = ensure_series(price, name="price")
    lr = np.log(s / s.shift(1))
    return lr.dropna().rename("log_return")


def to_simple_returns(price: pd.Series) -> pd.Series:
    """
    Convert a price series into simple returns.

    r_t = p_t / p_{t-1} - 1
    """
    s = ensure_series(price, name="price")
    r = s / s.shift(1) - 1.0
    return r.dropna().rename("return")


def compute_drawdown(equity: pd.Series) -> pd.DataFrame:
    """
    Compute drawdown path for an equity curve.

    Returns DataFrame with:
    - equity: original equity curve
    - peak: running maximum
    - drawdown: equity - peak
    - drawdown_pct: (equity / peak - 1)
    """
    eq = ensure_series(equity, name="equity").astype(float)
    if eq.empty:
        return pd.DataFrame(
            columns=["equity", "peak", "drawdown", "drawdown_pct"],
            dtype=float,
        )
    peak = eq.cummax()
    drawdown = eq - peak
    drawdown_pct = (eq / peak) - 1.0
    out = pd.DataFrame(
        {
            "equity": eq,
            "peak": peak,
            "drawdown": drawdown,
            "drawdown_pct": drawdown_pct,
        }
    )
    return out


def max_drawdown(equity: pd.Series) -> float:
    """
    Maximum drawdown (in percentage terms) for an equity curve.

    Returns minimum of drawdown_pct (typically a negative number).
    """
    dd = compute_drawdown(equity)
    return float(dd["drawdown_pct"].min()) if not dd.empty else 0.0


def rolling_sharpe(
    returns: pd.Series,
    window: int = 63,
    *,
    risk_free: float = 0.0,
    annualization: int = 252,
) -> pd.Series:
    """
    Rolling Sharpe ratio of a return series.
    """
    r = ensure_series(returns, name="return").astype(float)
    excess = r - risk_free
    mu = excess.rolling(window).mean()
    sd = excess.rolling(window).std()
    sharpe = (mu / sd) * np.sqrt(annualization)
    sharpe = sharpe.replace([np.inf, -np.inf], np.nan)
    return sharpe.rename("rolling_sharpe")


def rolling_sortino(
    returns: pd.Series,
    window: int = 63,
    *,
    risk_free: float = 0.0,
    annualization: int = 252,
) -> pd.Series:
    """
    Rolling Sortino ratio of a return series
    (downside deviation in the denominator).
    """
    r = ensure_series(returns, name="return").astype(float)
    excess = r - risk_free
    downside = excess.copy()
    downside[downside > 0] = 0.0
    downside_std = downside.rolling(window).std()
    mu = excess.rolling(window).mean()
    sortino = (mu / downside_std) * np.sqrt(annualization)
    sortino = sortino.replace([np.inf, -np.inf], np.nan)
    return sortino.rename("rolling_sortino")


# -----------------------------------------------------------------------------
# Placeholder – implied volatility (to maintain backward-compat)
# -----------------------------------------------------------------------------

def get_implied_volatility(ticker: str) -> float:
    """Stub: real IV fetch requires external API; returns NaN for now."""
    logger.debug("get_implied_volatility stub called for %s", ticker)
    return float("nan")


def calculate_beta(prices: pd.Series, benchmark: Optional[pd.Series] = None) -> float:
    """Simple beta estimate of a price series vs benchmark (log-returns)."""
    if benchmark is None:
        benchmark = prices
    lr = np.log(prices / prices.shift(1)).dropna()
    br = np.log(benchmark / benchmark.shift(1)).dropna()
    if len(lr) == 0 or len(br) == 0:
        return float("nan")
    # align
    idx = lr.index.intersection(br.index)
    lr = lr.loc[idx]
    br = br.loc[idx]
    if len(idx) == 0:
        return float("nan")
    cov = np.cov(lr, br)[0, 1]
    var = np.var(br)
    return float(cov / var) if var else np.nan


# -----------------------------------------------------------------------------
# Professional extensions: Hurst, Hedge, Spread, Pair metrics
# -----------------------------------------------------------------------------

def hurst_exponent(series: pd.Series, min_lag: int = 2, max_lag: int = 20) -> float:
    """Estimate Hurst exponent with a simple R/S-like method."""
    s = ensure_series(series, name="series").dropna()
    if len(s) < max_lag + 2:
        return float("nan")
    lags = range(min_lag, max_lag)
    tau = []
    for lag in lags:
        diff = s.diff(lag).dropna()
        if diff.empty:
            continue
        tau.append(np.sqrt(np.std(diff)))
    if len(tau) < 2:
        return float("nan")
    poly = np.polyfit(np.log(list(lags)[: len(tau)]), np.log(tau), 1)
    return float(poly[0] * 2.0)


def ols_hedge_ratio(x: pd.Series, y: pd.Series) -> Tuple[float, pd.Series]:
    """Static OLS hedge ratio between x and y."""
    sx = ensure_series(x, name="x").astype(float)
    sy = ensure_series(y, name="y").astype(float)
    idx = sx.index.intersection(sy.index)
    sx = sx.loc[idx]
    sy = sy.loc[idx]
    if len(sx) == 0:
        return 0.0, sy - sx
    model = OLS(sy, add_constant(sx)).fit()
    hedge = model.params.iloc[1]
    resid = model.resid
    return float(hedge), resid


def kalman_hedge_ratio(x: pd.Series, y: pd.Series) -> Tuple[float, pd.Series]:
    """Dynamic hedge ratio using a simple Kalman filter.

    Falls back to OLS hedge ratio if pykalman is not installed.
    """
    sx = ensure_series(x, name="x").astype(float)
    sy = ensure_series(y, name="y").astype(float)
    idx = sx.index.intersection(sy.index)
    sx = sx.loc[idx]
    sy = sy.loc[idx]
    if len(sx) == 0:
        return 0.0, sy - sx

    try:
        from pykalman import KalmanFilter  # type: ignore
    except ImportError:  # graceful fallback
        logger.warning("pykalman missing – falling back to OLS hedge ratio")
        return ols_hedge_ratio(sx, sy)

    delta = 1e-5
    trans_cov = delta / (1 - delta) * np.eye(2)
    obs_mat = np.vstack([sx.values, np.ones(len(sx))]).T[:, np.newaxis]
    kf = KalmanFilter(
        transition_matrices=np.eye(2),
        observation_matrices=obs_mat,
        initial_state_mean=np.zeros(2),
        transition_covariance=trans_cov,
        observation_covariance=1.0,
    )
    state_means, _ = kf.filter(sy.values)
    hedge = state_means[:, 0]
    resid = sy - hedge * sx
    return float(hedge[-1]), resid


def compute_spread(
    x: pd.Series,
    y: pd.Series,
    *,
    log: bool = True,
    name: str = "spread",
) -> pd.Series:
    sx = ensure_series(x, name="x").astype(float)
    sy = ensure_series(y, name="y").astype(float)
    idx = sx.index.intersection(sy.index)
    sx = sx.loc[idx]
    sy = sy.loc[idx]

    # הגנה מפני מחירי 0/NaN ללוג
    if log:
        sx = sx.replace(0, np.nan)
        sy = sy.replace(0, np.nan)
        spread = np.log(sx) - np.log(sy)
    else:
        spread = sx - sy

    spread = spread.replace([np.inf, -np.inf], np.nan)
    spread.name = name
    return spread



def basic_pair_metrics(
    x: pd.Series,
    y: pd.Series,
    *,
    z_window: int = 20,
) -> Dict[str, float]:
    """Compute a small set of basic pair metrics using common utils only.

    Returns a dict with keys like:
    - corr, half_life, zscore, vol_z, hurst, adf_pvalue, hedge
    """
    sx = ensure_series(x, name="x").astype(float)
    sy = ensure_series(y, name="y").astype(float)
    idx = sx.index.intersection(sy.index)
    sx = sx.loc[idx]
    sy = sy.loc[idx]

    if len(sx) < 30:
        return {
            "corr": float("nan"),
            "half_life": float("nan"),
            "zscore": float("nan"),
            "vol_z": float("nan"),
            "hurst": float("nan"),
            "adf_pvalue": float("nan"),
            "hedge": float("nan"),
        }

    corr = calculate_correlation(np.log(sx), np.log(sy))
    hedge, resid = ols_hedge_ratio(np.log(sx), np.log(sy))
    hl = calculate_half_life(resid)
    spread = compute_spread(sx, sy, log=True)
    zscore = calculate_zscore(sx, sy, window=z_window)
    vol_z = calculate_volatility_zscore((sx + sy) / 2.0, window=z_window)
    hurst_val = hurst_exponent(resid)
    adf_p = adf_pvalue(resid)

    return {
        "corr": corr,
        "half_life": hl,
        "zscore": zscore,
        "vol_z": vol_z,
        "hurst": hurst_val,
        "adf_pvalue": adf_p,
        "hedge": hedge,
    }


def risk_parity_weights(assets: pd.DataFrame) -> pd.Series:
    """
    Compute naive risk-parity weights based on inverse volatility.

    Eigendecomposition is kept for future HF-style extensions.
    """
    if assets is None or assets.empty:
        return pd.Series(dtype=float)
    rets = assets.pct_change().dropna()
    if rets.empty:
        return pd.Series(1.0 / len(assets.columns), index=assets.columns)
    cov = rets.cov().values
    if cov.size == 0:
        return pd.Series(1.0 / len(assets.columns), index=assets.columns)
    _eigvals, _eigvecs = eigh(cov)  # reserved for future diagnostics
    inv_vol = 1 / np.sqrt(np.diag(cov))
    w = inv_vol / inv_vol.sum()
    return pd.Series(w, index=assets.columns)


# -----------------------------------------------------------------------------
# Statistical tests
# -----------------------------------------------------------------------------

def pearson_pvalue(x: pd.Series, y: pd.Series) -> Tuple[float, float]:
    """Pearson correlation + p-value."""
    sx = ensure_series(x, name="x").dropna()
    sy = ensure_series(y, name="y").dropna()
    idx = sx.index.intersection(sy.index)
    sx = sx.loc[idx]
    sy = sy.loc[idx]
    if len(sx) < 3:
        return float("nan"), float("nan")
    r, p = pearsonr(sx.values, sy.values)
    return float(r), float(p)


def adf_pvalue(series: pd.Series) -> float:
    """ADF p-value for stationarity."""
    s = ensure_series(series, name="series").dropna()
    if len(s) < 10:
        return float("nan")
    return float(adfuller(s)[1])


def johansen_trace(x: pd.Series, y: pd.Series) -> float:
    """Johansen trace statistic for cointegration of two series."""
    sx = ensure_series(x, name="x").dropna()
    sy = ensure_series(y, name="y").dropna()
    df = pd.concat([sx, sy], axis=1).dropna()
    if df.shape[0] < 20:
        return float("nan")
    res = coint_johansen(df, 0, 1)
    return float(res.lr1[0])


# -----------------------------------------------------------------------------
# Pair quality scoring
# -----------------------------------------------------------------------------

_THRESHOLDS: Dict[str, float] = {
    "corr": 0.7,
    "p_value": 0.05,
    "half_life": 20,
    "vol_z": 1.5,
    "adf_p": 0.05,
    "joh_trace": 10.0,
    "hurst": 0.5,
}


def _score(metrics: Dict[str, float], cfg: Dict[str, Any]) -> int:
    thr = {**_THRESHOLDS, **cfg.get("thresholds", {})}
    score = 0
    score += int(abs(metrics.get("corr", 0.0)) > thr["corr"])
    score += int(metrics.get("p_value", 1.0) < thr["p_value"])
    score += int(metrics.get("half_life", 1e9) < thr["half_life"])
    score += int(abs(metrics.get("vol_z", 1e9)) < thr["vol_z"])
    score += int(metrics.get("adf_p", 1.0) < thr["adf_p"])
    score += int(metrics.get("joh_trace", 0.0) > thr["joh_trace"])
    score += int(metrics.get("hurst", 1.0) < thr["hurst"])
    return score


def _evaluate_from_df(
    df: pd.DataFrame, config: Dict[str, Any] | None = None
) -> Dict[str, float]:
    s1, s2 = df.iloc[:, 0], df.iloc[:, 1]
    corr, p_val = pearson_pvalue(s1, s2)
    hedge, resid = ols_hedge_ratio(s1, s2)
    metrics: Dict[str, float] = {
        "corr": corr,
        "p_value": p_val,
        "half_life": calculate_half_life(resid),
        "vol_z": calculate_volatility_zscore((s1 + s2) / 2),
        "adf_p": adf_pvalue(resid),
        "joh_trace": johansen_trace(s1, s2),
        "hurst": hurst_exponent(resid),
        "hedge": hedge,
    }
    if config is not None:
        metrics["score"] = _score(metrics, config)
    return metrics


@overload
def evaluate_pair_quality(
    df: pd.DataFrame, config: Dict[str, Any] | None = None
) -> Dict[str, float]:
    ...


@overload
def evaluate_pair_quality(
    corr: float,
    p_value: float,
    half_life: float,
    vol_z: float,
    config: Dict[str, Any] | None = None,
) -> Union[int, Dict[str, float]]:
    ...


def evaluate_pair_quality(*args, **kwargs):  # type: ignore
    """
    Flexible wrapper.

    Accepts:
    1. DataFrame [, config]
    2. corr, p_value, half_life, vol_z [, config]
    3. corr, p_value, half_life, vol_z, config (legacy 5-tuple)
    """
    if not args:
        raise ValueError("evaluate_pair_quality: no arguments provided")

    if isinstance(args[0], pd.DataFrame):
        return _evaluate_from_df(args[0], kwargs.get("config"))

    if len(args) == 5:
        corr, p_value, half_life, vol_z, cfg = args  # type: ignore
    elif len(args) == 4:
        corr, p_value, half_life, vol_z = args  # type: ignore
        cfg = kwargs.get("config", {})
    else:
        raise ValueError("evaluate_pair_quality: unsupported argument pattern")

    base = {
        "corr": float(corr),
        "p_value": float(p_value),
        "half_life": float(half_life),
        "vol_z": float(vol_z),
        "adf_p": 0.0,
        "joh_trace": 0.0,
        "hurst": 0.0,
    }
    return _score(base, cfg)


# -----------------------------------------------------------------------------
# Edge evaluation
# -----------------------------------------------------------------------------

Number = Union[int, float, np.number]


def evaluate_edge(
    zscore: Union[Number, pd.Series, np.ndarray],
    correlation: Union[Number, pd.Series, np.ndarray],
    z_threshold: float = 0.5,
    corr_threshold: float = 0.2,
):
    """
    מחשב "edge" גם לסקאלר וגם לוקטור (Series / ndarray).

    Scalar:
        evaluate_edge(2.0, 0.8) -> float

    Vector:
        evaluate_edge(df["z"], df["corr"]) -> pd.Series עם index של df.

    לוגיקה בסיסית:
    1. אם |z| קטן מסף או הקורלציה חלשה → edge = 0.
    2. אחרת → edge גדל ככל שה-|z| גדול מעל הסף, מוכפל בקורלציה.
    3. שומרים על סימן ה־z (כדי לקודד כיוון mean-reversion).
    """

    # ---- נבדוק אם מדובר בוקטורים (Series / ndarray) או בסקאלרים ----
    is_z_vec = isinstance(zscore, (pd.Series, np.ndarray))
    is_c_vec = isinstance(correlation, (pd.Series, np.ndarray))

    # ======================
    # מקרה סקאלרי "אמיתי"
    # ======================
    if not is_z_vec and not is_c_vec:
        z_val = float(zscore)
        c_val = float(correlation)

        # אין מספיק סטייה / קורלציה → אין edge
        if abs(z_val) < z_threshold or c_val < corr_threshold:
            return 0.0

        excess_z = abs(z_val) - float(z_threshold)
        edge = excess_z * c_val

        # שומר על סימן z (אם z<0 → edge שלילי)
        if z_val < 0:
            edge = -edge

        return float(edge)

    # ======================
    # מקרה וקטורי (Series / ndarray)
    # ======================

    # אם שניהם Series – ניישר אינדקסים
    if isinstance(zscore, pd.Series) and isinstance(correlation, pd.Series):
        zs, cs = zscore.align(correlation)
        z_arr = zs.to_numpy(dtype=float)
        c_arr = cs.to_numpy(dtype=float)
        index = zs.index
    else:
        z_arr = np.asarray(zscore, dtype=float)
        c_arr = np.asarray(correlation, dtype=float)
        index = getattr(zscore, "index", None) or getattr(correlation, "index", None)

    z_th = float(z_threshold)
    c_th = float(corr_threshold)

    # מסכה: רק איפה שיש מספיק סטייה וקורלציה
    mask = (np.abs(z_arr) >= z_th) & (c_arr >= c_th)

    edge = np.zeros_like(z_arr, dtype=float)
    excess_z = np.abs(z_arr) - z_th

    edge[mask] = excess_z[mask] * c_arr[mask]

    # שימור סימן ה־z (פוזיציית mean-reversion)
    sign = np.sign(z_arr)
    edge = edge * sign

    # אם הקלט היה Series → נחזיר Series עם אותו index, אחרת numpy array
    if index is not None:
        return pd.Series(edge, index=index, name="edge")

    return edge

# -----------------------------------------------------------------------------
# JSON-safe export helpers
# -----------------------------------------------------------------------------

def metrics_to_json(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Convert metrics dict into a JSON-safe structure."""
    return make_json_safe(metrics)


# -----------------------------------------------------------------------------
# Exports
# -----------------------------------------------------------------------------

__all__ = (
    "NumericLikeSeries",
    "ensure_series",
    "_parse_symbol_input",
    "normalize_symbol",
    "load_price_data",
    "load_price_data_range",
    "get_close_series_range",
    "load_price_panel",
    "calculate_zscore",
    "calculate_half_life",
    "calculate_correlation",
    "calculate_volatility_zscore",
    "calculate_historical_volatility",
    "to_log_returns",
    "to_simple_returns",
    "compute_drawdown",
    "max_drawdown",
    "rolling_sharpe",
    "rolling_sortino",
    "get_implied_volatility",
    "calculate_beta",
    "hurst_exponent",
    "ols_hedge_ratio",
    "kalman_hedge_ratio",
    "compute_spread",
    "basic_pair_metrics",
    "risk_parity_weights",
    "pearson_pvalue",
    "adf_pvalue",
    "johansen_trace",
    "evaluate_pair_quality",
    "evaluate_edge",
    "metrics_to_json",
)
