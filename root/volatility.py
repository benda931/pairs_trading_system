# -*- coding: utf-8 -*-
"""
volatility.py — Robust volatility helpers (ATR & friends)
--------------------------------------------------------
Hedge‑fund–grade utilities for computing True Range and ATR, with:
- Case‑insensitive OHLC handling (High/Low/Close or high/low/close)
- Multiple smoothing methods: Wilder (RMA), SMA, EMA
- Safe defaults, type hints, and helpful errors
"""
from __future__ import annotations

import logging
from typing import Iterable, Optional, Tuple, Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(name)s: %(message)s')

# ------------------------------
# Internal helpers
# ------------------------------
_DEF_HIGH = ("High", "high", "HIGH")
_DEF_LOW  = ("Low", "low", "LOW")
_DEF_CLOSE= ("Close", "close", "CLOSE", "Adj Close", "adj_close", "adj close")
_DEF_OPEN = ("Open", "open", "OPEN")



def _resolve_col(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    cols = {str(c).strip(): c for c in df.columns}
    lower = {k.lower(): v for k, v in cols.items()}
    for name in candidates:
        key = str(name).strip()
        if key in cols:
            return cols[key]
        if key.lower() in lower:
            return lower[key.lower()]
    return None


def _get_open(df: pd.DataFrame, open_col: Optional[str] = None) -> pd.Series:
    col = open_col or _resolve_col(df, _DEF_OPEN)
    if col is None:
        cand = df.get('Adj Open', df.get('adj_open'))
        if cand is not None:
            return pd.to_numeric(cand, errors="coerce")
        num = df.select_dtypes(include="number")
        if not num.empty:
            return pd.to_numeric(num.iloc[:, 0], errors="coerce")
        raise KeyError("Missing required Open column")
    return pd.to_numeric(df[col], errors="coerce")


def _get_ohlc(df: pd.DataFrame,
              high: Optional[str] = None,
              low: Optional[str] = None,
              close: Optional[str] = None) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Return (high, low, close) series, resolving common column aliases.

    Raises
    ------
    KeyError
        If required columns are not found.
    """
    hcol = high or _resolve_col(df, _DEF_HIGH)
    lcol = low  or _resolve_col(df, _DEF_LOW)
    ccol = close or _resolve_col(df, _DEF_CLOSE)
    missing = [n for n, c in [("high", hcol), ("low", lcol), ("close", ccol)] if c is None]
    if missing:
        raise KeyError(f"Missing required OHLC columns: {', '.join(missing)}")
    h = pd.to_numeric(df[hcol], errors="coerce")
    l = pd.to_numeric(df[lcol], errors="coerce")
    c = pd.to_numeric(df[ccol], errors="coerce")
    return h, l, c


def true_range(high: pd.Series, low: pd.Series, prev_close: pd.Series) -> pd.Series:
    """Compute True Range = max( H-L, |H-Pc|, |L-Pc| ).

    Inputs are coerced to float and aligned by index.
    """
    h = pd.to_numeric(high, errors="coerce")
    l = pd.to_numeric(low, errors="coerce")
    pc = pd.to_numeric(prev_close, errors="coerce")
    tr1 = h - l
    tr2 = (h - pc).abs()
    tr3 = (l - pc).abs()
    out = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    out.name = "TR"
    return out


def _rma(x: pd.Series, window: int) -> pd.Series:
    """Wilder's RMA smoothing (a.k.a. EMA with alpha=1/window)."""
    alpha = 1.0 / float(window)
    return x.ewm(alpha=alpha, adjust=False, min_periods=window).mean()


def _ema(x: pd.Series, window: int) -> pd.Series:
    return x.ewm(span=window, adjust=False, min_periods=window).mean()


def _sma(x: pd.Series, window: int) -> pd.Series:
    return x.rolling(window=window, min_periods=window).mean()


# ------------------------------
# Public API
# ------------------------------
Method = Literal["wilder", "sma", "ema"]

def calculate_atr(
    df: pd.DataFrame,
    window: int = 14,
    *,
    method: Method = "wilder",
    high_col: Optional[str] = None,
    low_col: Optional[str] = None,
    close_col: Optional[str] = None,
    name: Optional[str] = None,
) -> pd.Series:
    """Average True Range with robust OHLC resolution and multiple methods.

    Parameters
    ----------
    df : DataFrame
        Must contain High/Low/Close columns (case‑insensitive). Supports 'Adj Close'.
    window : int, default 14
        Lookback window.
    method : {'wilder','sma','ema'}, default 'wilder'
        Smoothing method: Wilder (RMA), simple moving average, or exponential.
    high_col, low_col, close_col : str, optional
        Explicit column names if auto‑detection should be overridden.
    name : str, optional
        Custom output name; defaults to 'ATR_{method}_{window}'.

    Returns
    -------
    pd.Series
        ATR aligned to *df* index.
    """
    if window <= 0:
        raise ValueError("window must be positive")

    h, l, c = _get_ohlc(df, high_col, low_col, close_col)
    pc = c.shift(1)
    tr = true_range(h, l, pc)

    method = method.lower()
    if method == "wilder":
        atr = _rma(tr, window)
    elif method == "ema":
        atr = _ema(tr, window)
    elif method == "sma":
        atr = _sma(tr, window)
    else:
        raise ValueError("method must be one of {'wilder','sma','ema'}")

    atr.name = name or f"ATR_{method}_{window}"
    return atr


# ------------------------------
# Advanced volatility estimators
# ------------------------------

def parkinson_vol(df: pd.DataFrame, window: int = 20,
                   *, high_col: Optional[str] = None,
                   low_col: Optional[str] = None,
                   trading_days: int = 252) -> pd.Series:
    """Parkinson (1980) high–low estimator (annualised).

    σ² = (1 / (4 ln 2)) * mean( (ln(H/L))² ) over window
    """
    if window <= 0:
        raise ValueError("window must be positive")
    h, l, _ = _get_ohlc(df, high_col, low_col, None)
    hl = (np.log(h) - np.log(l)) ** 2
    coef = 1.0 / (4.0 * np.log(2.0))
    rv = (coef * hl.rolling(window, min_periods=window).mean()).pow(0.5)
    return rv * np.sqrt(trading_days)

def garman_klass_vol(df: pd.DataFrame, window: int = 20,
                     *, high_col: Optional[str] = None,
                     low_col: Optional[str] = None,
                     close_col: Optional[str] = None,
                     open_col: Optional[str] = None,
                     trading_days: int = 252) -> pd.Series:
    """Garman–Klass (1980) estimator (annualised). Requires Open/High/Low/Close.

    σ² = 0.5(ln(H/L))² - (2ln2 - 1)(ln(C/O))²
    """
    # Resolve columns (tolerant: try common names)
    if window <= 0:
        raise ValueError("window must be positive")
    h, l, c = _get_ohlc(df, high_col, low_col, close_col)
    o = _get_open(df, open_col)
    x = (np.log(h) - np.log(l)) ** 2
    y = (np.log(c) - np.log(o)) ** 2
    var = 0.5 * x.rolling(window, min_periods=window).mean() - (2*np.log(2)-1.0) * y.rolling(window, min_periods=window).mean()
    vol = var.clip(lower=0).pow(0.5) * np.sqrt(trading_days)
    return vol

def rogers_satchell_vol(df: pd.DataFrame, window: int = 20,
                        *, high_col: Optional[str] = None,
                        low_col: Optional[str] = None,
                        close_col: Optional[str] = None,
                        open_col: Optional[str] = None,
                        trading_days: int = 252) -> pd.Series:
    """Rogers–Satchell (1991) open–close robust estimator (annualised)."""
    h, l, c = _get_ohlc(df, high_col, low_col, close_col)
    o = _get_open(df, open_col)
    term = (np.log(h) - np.log(c)) * (np.log(h) - np.log(o)) + (np.log(l) - np.log(c)) * (np.log(l) - np.log(o))
    var = term.rolling(window, min_periods=window).mean().clip(lower=0)
    return var.pow(0.5) * np.sqrt(trading_days)

def yang_zhang_vol(df: pd.DataFrame, window: int = 20,
                    *, high_col: Optional[str] = None,
                    low_col: Optional[str] = None,
                    close_col: Optional[str] = None,
                    open_col: Optional[str] = None,
                    trading_days: int = 252) -> pd.Series:
    """Yang–Zhang (2000) estimator (annualised)."""
    # Resolve
    if window <= 0:
        raise ValueError("window must be positive")
    h, l, c = _get_ohlc(df, high_col, low_col, close_col)
    o = _get_open(df, open_col)
    # Components
    co = np.log(c) - np.log(o)
    oo = np.log(o) - np.log(o.shift(1))
    hl = (np.log(h) - np.log(l)) ** 2
    k = 0.34 / (1.34 + (window + 1)/(window - 1)) if window > 1 else 0.34
    sigma2_o = oo.rolling(window, min_periods=window).var()
    sigma2_c = co.rolling(window, min_periods=window).var()
    sigma2_rs = hl.rolling(window, min_periods=window).mean() - (8*np.log(2)-2) * sigma2_c
    var = sigma2_o + k * sigma2_c + (1 - k) * sigma2_rs
    return var.clip(lower=0).pow(0.5) * np.sqrt(trading_days)

def realized_vol_from_close(close: pd.Series, window: int = 20, trading_days: int = 252) -> pd.Series:
    """Realised volatility from close-to-close returns (annualised)."""
    r = pd.to_numeric(close, errors='coerce').pct_change()
    vol = r.rolling(window, min_periods=window).std() * np.sqrt(trading_days)
    return vol.rename(f"RV_{window}")

def atr_percent(df: pd.DataFrame, window: int = 14, *, method: Method = "wilder",
                high_col: Optional[str] = None, low_col: Optional[str] = None,
                close_col: Optional[str] = None) -> pd.Series:
    """ATR as percent of price (Close)."""
    atr = calculate_atr(df, window, method=method, high_col=high_col, low_col=low_col, close_col=close_col)
    _, _, c = _get_ohlc(df, high_col, low_col, close_col)
    c_safe = c.replace(0, np.nan)
    pct = (atr / c_safe).rename(f"ATR%_{method}_{window}")
    return pct


def volatility_regime(df: pd.DataFrame, *, estimator: Literal["yz","rs","pk","gk","rv"] = "yz",
                      window: int = 20, low: float = 0.10, high: float = 0.30,
                      close_col: Optional[str] = None) -> pd.Series:
    """Return per-date regime label {'low','mid','high'} based on an annualised vol estimator.

    Thresholds are absolute (e.g., 0.10=10% annual). Choose estimator: yz/rs/pk/gk/rv.
    """
    if estimator == "yz":
        vol = yang_zhang_vol(df, window)
    elif estimator == "rs":
        vol = rogers_satchell_vol(df, window)
    elif estimator == "pk":
        vol = parkinson_vol(df, window)
    elif estimator == "gk":
        vol = garman_klass_vol(df, window)
    elif estimator == "rv":
        _, _, c = _get_ohlc(df, None, None, close_col)
        vol = realized_vol_from_close(c, window)
    else:
        raise ValueError("estimator must be one of {'yz','rs','pk','gk','rv'}")
    reg = pd.Series(index=vol.index, dtype=object)
    reg.loc[vol < low] = 'low'
    reg.loc[(vol >= low) & (vol < high)] = 'mid'
    reg.loc[vol >= high] = 'high'
    return reg.rename(f"REG_{estimator}_{window}")


def is_spike_day(df: pd.DataFrame, *, window: int = 14, threshold: float = 0.04,
                 method: Method = "wilder",
                 high_col: Optional[str] = None, low_col: Optional[str] = None,
                 close_col: Optional[str] = None) -> pd.Series:
    """Return boolean Series where ATR% > threshold (e.g., 0.04 = 4%)."""
    atrp = atr_percent(df, window, method=method, high_col=high_col, low_col=low_col, close_col=close_col)
    return (atrp > threshold).rename(f"SPIKE_{int(threshold*100)}bps")

__all__ = [
    "calculate_atr",
    "true_range",
    "parkinson_vol",
    "garman_klass_vol",
    "rogers_satchell_vol",
    "yang_zhang_vol",
    "realized_vol_from_close",
    "atr_percent",
    "volatility_regime",
    "is_spike_day",
]
