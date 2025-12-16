# -*- coding: utf-8 -*-
"""
root/backtest_logic.py — HF-grade Pairs Backtest Wrapper & Vol Overview (Part 1/3)
===================================================================================

תפקיד הקובץ במערכת
-------------------
זהו שכבת *wrapper* מעל מנוע הבקטסט הראשי (`core.backtest`) + מודול התנודתיות,
עם שתי מטרות:

1. Backtest מהיר לטאב הדשבורד (Tab 3):
   - הרצת סימולציה לזוג בודד על בסיס Z-score / ספרד, עם קונפיג גמיש.
   - הצגת KPI-ים, גרפים, טבלת טריידים, ו-logs.

2. Volatility Overview:
   - ניתוח תנודתיות מרובה-אומדים (Yang–Zhang, RS, Parkinson, Realized) לכל נכס.
   - זיהוי משטרי תנודתיות (vol regimes) ו-Spike days.

המודול בנוי כך:
----------------
Part 1/3 (החלק הזה)
    - Imports & logging
    - Public API (__all__)
    - BacktestConfig (Pydantic-friendly) + `coerce`
    - `get_param_distributions` לאופטימיזציה
    - `_compute_kpis_from_pnl` (KPI helper מוכלל)

Part 2/3
    - מנוע backtest מקומי (run_backtest) עם:
        * זיהוי data gaps
        * book-keeping מסודר לטריידים
        * שילוב Volatility Targeting / ATR% / Macro Mult
    - חיבור ל-DuckDB (append אופציונלי)

Part 3/3
    - `render_volatility_overview` לטאב ייעודי/פאנל.
    - helper-ים נוספים לנוחות (השוואת סימולציות קודמות, וכו').
"""

from __future__ import annotations

# ============================================================================
# Imports & optional deps
# ============================================================================

from typing import Any, Dict, List, Optional, Tuple, Union, Sequence

import logging
import os

import numpy as np
import pandas as pd

try:
    import streamlit as st  # type: ignore
except Exception:  # pragma: no cover
    st = None  # type: ignore

try:
    import plotly.graph_objects as go  # type: ignore
except Exception:  # pragma: no cover
    go = None  # type: ignore

# yfinance משמש רק אם נרצה fallback לנתונים חיצוניים
try:
    import yfinance as yf  # type: ignore
except Exception:  # pragma: no cover
    yf = None  # type: ignore

# volatility + utils פנימיים
from volatility import (  # type: ignore
    calculate_atr,
    atr_percent,
    parkinson_vol,
    garman_klass_vol,
    rogers_satchell_vol,
    yang_zhang_vol,
    realized_vol_from_close,
    volatility_regime,
    is_spike_day,
)
from common.utils import (  # type: ignore
    get_implied_volatility,
    load_price_data,
    calculate_historical_volatility,
    calculate_beta,
)

# נסה להתחבר למנוע הבקטסט הראשי – לא חובה (המנוע המקומי עדיין זמין)
try:
    from root.backtest import (
        BacktestResult as CoreBacktestResult,
        run_backtest_with_vol as core_run_backtest_with_vol,
    )
except Exception:  # pragma: no cover
    CoreBacktestResult = None  # type: ignore
    core_run_backtest_with_vol = None  # type: ignore

# ============================================================================
# Logging
# ============================================================================

logger = logging.getLogger("backtest_logic")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | backtest_logic | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)
# לא נוגעים ב-basicConfig כאן – המערכת הראשית כבר מטפלת בזה.

# ============================================================================
# Public API (נשמרת התאימות לקוד הקיים)
# ============================================================================

__all__ = [
    "BacktestConfig",
    "run_backtest",               # ימומש בחלק 2/3
    "_compute_kpis_from_pnl",
    "render_volatility_overview", # ימומש בחלק 3/3
    "get_param_distributions",
]

# ============================================================================
# BacktestConfig (Pydantic v2-friendly, with safe fallback)
# ============================================================================

try:
    from pydantic import BaseModel, Field  # type: ignore
except Exception:  # pragma: no cover

    class BaseModel:  # type: ignore[no-redef]
        """Fallback dummy BaseModel – מאפשר API דומה גם בלי Pydantic."""
        def __init__(self, **data: Any) -> None:
            for k, v in data.items():
                setattr(self, k, v)

    def Field(default: Any = None, **_: Any) -> Any:  # type: ignore[no-redef]
        return default


class BacktestConfig(BaseModel):  # type: ignore[misc]
    """
    Hedge-fund-grade config for the *simple* pairs backtest wrapper.

    זהו קונפיג ייעודי לטאב Backtest (root), נפרד מה-BacktestConfig של core.backtest,
    אבל עם שדות דומים כדי שיהיה קל למפות ביניהם.

    Fields בסיסיים
    ---------------
    rolling_window :
        חלון לחישוב z-score (על spread).
    atr_window :
        חלון ATR לשימוש ב-normalisation ושליטה בריסק.
    capital :
        הון הבסיס לסימולציה (USD).
    fee_rate :
        שיעור עמלה פר-leg (commission), כאחוז מהנוטיונל (0.0005 = 5bps).
    slippage_bps :
        סליפג' בבסיס נקודות (bps) פר round-trip (משוקלל בפונקציה).

    Risk / PnL Gates
    ----------------
    stop_loss_pct :
        Stop-loss מנורמל להון (כהפרש מצטבר) – 0.05 = 5% מההון.
    take_profit_pct :
        Take-profit מנורמל להון – 0.10 = 10%.
    risk_free :
        ריבית חסרת סיכון לשימוש ב־Sharpe / Sortino.

    Volatility & Exposure
    ---------------------
    vol_estimator :
        Estimator בסיסי למשטר תנודתיות: 'yz' / 'rs' / 'pk' / 'gk' / 'rv'.
    vol_target :
        טארגט תנודתיות שנתית (0 = כבוי) לצורך scaling.
    max_exposure_per_trade :
        cap לנוטיונל פר טרייד כאחוז מההון (1.0 = 100% מההון).
    use_atr_percent :
        האם להשתמש ב-ATR% לנורמליזציה בין נכסים.

    תאריכים & Logging
    ------------------
    start_date / end_date :
        פילטר טווח תאריכים (ISO string, או יעברו לדאטה כ-pd.to_datetime).
    duckdb_path :
        אם לא None – נכתוב trades גם ל-DuckDB (טבלת backtests).
    """

    rolling_window: int = Field(
        20, ge=5, le=252, description="Window for z-score rolling stats"
    )
    atr_window: int = Field(
        14, ge=5, le=252, description="ATR lookback"
    )
    capital: float = Field(
        50_000.0, gt=0
    )
    fee_rate: float = Field(
        0.0005, ge=0.0, le=0.02, description="Per-leg commission rate"
    )
    slippage_bps: float = Field(
        1.0, ge=0.0, le=50.0, description="Round-trip slippage in bps"
    )
    stop_loss_pct: float = Field(
        0.05, ge=0.0, le=1.0
    )
    take_profit_pct: float = Field(
        0.10, ge=0.0, le=1.0
    )
    risk_free: float = Field(
        0.01, ge=-0.05, le=0.2
    )

    # Advanced volatility / exposure knobs
    vol_estimator: str = Field(
        "yz", description="Vol estimator for regime/sizing: yz/rs/pk/gk/rv"
    )
    vol_target: float = Field(
        0.0, ge=0.0, le=1.0,
        description="Annual target vol for position scaling (0=off)"
    )
    max_exposure_per_trade: float = Field(
        1.0, ge=0.0, le=1.0,
        description="Cap as fraction of capital per trade"
    )
    use_atr_percent: bool = Field(
        True, description="Use ATR% for cross-asset normalisation"
    )

    # Date filters (optional)
    start_date: Optional[str] = Field(
        None, description="ISO start date filter (YYYY-MM-DD)"
    )
    end_date: Optional[str] = Field(
        None, description="ISO end date filter (YYYY-MM-DD)"
    )

    # Logging / persistence
    duckdb_path: Optional[str] = Field(
        None, description="If set, append trades to DuckDB at this path"
    )

    @classmethod
    def coerce(cls, cfg: Optional[Dict[str, Any]]) -> "BacktestConfig":
        """
        Robust coercion from dict → BacktestConfig.

        - מנסה להשתמש ב־Pydantic v2 (`model_validate`) אם קיים.
        - אחרת ממפה שדות ידנית ומחיל ברירות מחדל בטוחות.
        """
        data = dict(cfg or {})
        try:
            # Pydantic v2 style
            if hasattr(cls, "model_validate"):
                return cls.model_validate(data)  # type: ignore[attr-defined]
        except Exception:
            logger.debug("BacktestConfig.model_validate failed; falling back to manual coerce.", exc_info=True)

        # Manual coercion (works גם בלי Pydantic)
        def _get(name: str, default: Any) -> Any:
            return data.get(name, default)

        return cls(  # type: ignore[call-arg]
            rolling_window=int(_get("rolling_window", 20)),
            atr_window=int(_get("atr_window", 14)),
            capital=float(_get("capital", 50_000.0)),
            fee_rate=float(_get("fee_rate", 0.0005)),
            slippage_bps=float(_get("slippage_bps", 1.0)),
            stop_loss_pct=float(_get("stop_loss_pct", 0.05)),
            take_profit_pct=float(_get("take_profit_pct", 0.10)),
            risk_free=float(_get("risk_free", 0.01)),
            vol_estimator=str(_get("vol_estimator", "yz")),
            vol_target=float(_get("vol_target", 0.0)),
            max_exposure_per_trade=float(_get("max_exposure_per_trade", 1.0)),
            use_atr_percent=bool(_get("use_atr_percent", True)),
            start_date=_get("start_date", None),
            end_date=_get("end_date", None),
            duckdb_path=_get("duckdb_path", None),
        )


# ============================================================================
# Parameter Distributions for Optimisation (Optuna-friendly)
# ============================================================================

def get_param_distributions(optuna_style: bool = False) -> Dict[str, Any]:
    """
    מחזיר מרחב פרמטרים לאופטימיזציה על הבקטסט הזה.

    אם optuna_style == False:
        - מחזיר bounds פשוטים: {name: (low, high)}.

    אם optuna_style == True ואפשר לייבא optuna:
        - מחזיר dict של פונקציות שנקראות עם trial ומחזירות sample,
          למשל: dist["z_entry"](trial) → float.
    """
    space: Dict[str, Tuple[float, float]] = {
        "rolling_window": (5, 60),
        "atr_window": (5, 60),
        "z_entry": (0.8, 3.5),
        "z_exit": (0.05, 1.5),
        "fee_rate": (0.0, 0.001),
        "slippage_bps": (0.0, 10.0),
        "vol_target": (0.0, 0.40),
        "max_exposure_per_trade": (0.1, 1.0),
    }

    if not optuna_style:
        return space

    try:
        import optuna  # type: ignore  # noqa: F401

        return {
            "rolling_window": lambda t: t.suggest_int("rolling_window", 5, 60),
            "atr_window": lambda t: t.suggest_int("atr_window", 5, 60),
            "z_entry": lambda t: t.suggest_float("z_entry", 0.8, 3.5),
            "z_exit": lambda t: t.suggest_float("z_exit", 0.05, 1.5),
            "fee_rate": lambda t: t.suggest_float("fee_rate", 0.0, 0.001),
            "slippage_bps": lambda t: t.suggest_int("slippage_bps", 0, 10),
            "vol_target": lambda t: t.suggest_float("vol_target", 0.0, 0.40),
            "max_exposure_per_trade": lambda t: t.suggest_float(
                "max_exposure_per_trade", 0.1, 1.0
            ),
        }
    except Exception:
        # אם Optuna לא זמין – נחזיר את ה-bounds הפשוטים
        return space


# ============================================================================
# Small KPI helper (pure, reusable)
# ============================================================================

Number = Union[int, float]


def _compute_kpis_from_pnl(
    pnl: pd.Series,
    capital: float,
    risk_free: float,
) -> Dict[str, Optional[float]]:
    """
    חישוב KPIs עיקריים מרצף PnL (USD) של טריידים, בהנחת capital קבוע.

    מחשב:
        - win_rate (%)
        - max_dd_usd
        - cagr
        - sharpe
        - sortino
        - calmar
        - es95 (Expected Shortfall, 5%)
    """
    pnl = pd.Series(pnl, dtype=float).dropna()
    if pnl.empty:
        return {
            "win_rate": 0.0,
            "max_dd_usd": 0.0,
            "cagr": 0.0,
            "sharpe": None,
            "sortino": None,
            "calmar": None,
            "es95": None,
        }

    cum = pnl.cumsum()
    win_rate = float((pnl > 0).mean() * 100.0)
    max_dd = float((cum.cummax() - cum).max()) if len(cum) else 0.0

    equity = capital + cum
    days = max(1, len(pnl))
    cagr = float((equity.iloc[-1] / capital) ** (365.0 / days) - 1.0) if len(equity) else 0.0

    rets = pnl / capital
    ann_mu = float(rets.mean() * 252.0) if len(rets) else 0.0
    ann_vol = float(rets.std(ddof=1) * np.sqrt(252.0)) if len(rets) > 1 else 0.0

    sharpe = (ann_mu - risk_free) / ann_vol if ann_vol > 0 else np.nan
    downside_std = (
        float(rets[rets < 0].std(ddof=1) * np.sqrt(252.0))
        if (rets < 0).any()
        else np.nan
    )
    sortino = (ann_mu - risk_free) / downside_std if downside_std > 0 else np.nan
    calmar = cagr / (max_dd / capital) if max_dd > 0 else np.nan

    # ES95 (Expected Shortfall)
    try:
        q = float(pnl.quantile(0.05))
        es = float(pnl[pnl <= q].mean())
    except Exception:
        es = np.nan

    def _clean(x: float) -> Optional[float]:
        return float(x) if np.isfinite(x) else None

    return {
        "win_rate": win_rate,
        "max_dd_usd": max_dd,
        "cagr": cagr,
        "sharpe": _clean(sharpe),
        "sortino": _clean(sortino),
        "calmar": _clean(calmar),
        "es95": _clean(es),
    }

# ============================================================================
# Part 2/3 — Local Pairs Backtest Engine Wrapper (Tab Backtest)
# ============================================================================

Number = Union[int, float]


def _safe_last(series: pd.Series, default: float = 0.0) -> float:
    """Return last non-NaN value from a series, or default."""
    try:
        return float(pd.to_numeric(series, errors="coerce").dropna().iloc[-1])
    except Exception:
        return float(default)


def _load_pair_data(
    bt_x: str,
    bt_y: str,
    cfg: BacktestConfig,
) -> Optional[pd.DataFrame]:
    """
    טוען ומנקה דאטה לשני הסימבולים:

    Returns DataFrame עם:
        date, close_x, close_y  (מיושר לפי inner join).
    """
    df_x = load_price_data(bt_x)
    df_y = load_price_data(bt_y)

    if df_x is None or df_y is None or df_x.empty or df_y.empty:
        if st is not None:
            st.warning("לא נמצאו נתונים עבור אחד הסימבולים.")
        logger.warning("Empty data for %s or %s", bt_x, bt_y)
        return None

    # normalize columns to lower and ensure 'date'
    df_x = df_x.copy()
    df_y = df_y.copy()
    df_x.columns = [str(c).lower() for c in df_x.columns]
    df_y.columns = [str(c).lower() for c in df_y.columns]

    if "date" not in df_x.columns:
        df_x = df_x.reset_index().rename(columns={"index": "date"})
    if "date" not in df_y.columns:
        df_y = df_y.reset_index().rename(columns={"index": "date"})

    df_x["date"] = pd.to_datetime(df_x["date"], errors="coerce")
    df_y["date"] = pd.to_datetime(df_y["date"], errors="coerce")

    # Optional date filters from cfg
    if cfg.start_date:
        try:
            sd = pd.to_datetime(cfg.start_date)
            df_x = df_x[df_x["date"] >= sd]
            df_y = df_y[df_y["date"] >= sd]
        except Exception:
            logger.debug("start_date filter failed", exc_info=True)
    if cfg.end_date:
        try:
            ed = pd.to_datetime(cfg.end_date)
            df_x = df_x[df_x["date"] <= ed]
            df_y = df_y[df_y["date"] <= ed]
        except Exception:
            logger.debug("end_date filter failed", exc_info=True)

    if df_x.empty or df_y.empty:
        if st is not None:
            st.warning("אין נתונים אחרי סינון טווחי התאריכים.")
        return None

    # ensure 'close' numeric
    if "close" not in df_x.columns:
        df_x["close"] = pd.to_numeric(
            df_x.select_dtypes(include="number").iloc[:, 0], errors="coerce"
        )
    else:
        df_x["close"] = pd.to_numeric(df_x["close"], errors="coerce")

    if "close" not in df_y.columns:
        df_y["close"] = pd.to_numeric(
            df_y.select_dtypes(include="number").iloc[:, 0], errors="coerce"
        )
    else:
        df_y["close"] = pd.to_numeric(df_y["close"], errors="coerce")

    df = (
        pd.merge(
            df_x[["date", "close"]],
            df_y[["date", "close"]],
            on="date",
            how="inner",
            suffixes=("_x", "_y"),
        )
        .dropna()
        .sort_values("date")
        .reset_index(drop=True)
    )

    if df.empty:
        if st is not None:
            st.warning("אין חפיפה בין הסדרות לטווח הנתון.")
        return None

    return df


def _compute_leg_vol_stats(
    bt_x: str,
    bt_y: str,
    df_x: pd.DataFrame,
    df_y: pd.DataFrame,
    aw: int,
    use_atr_percent: bool,
) -> Dict[str, Any]:
    """
    מחשב סטטיסטיקות תנודתיות/ATR/IV/HV/β לכל leg, ומחזיר dict:

        {
          "atr_x", "atr_y",
          "iv_x", "iv_y",
          "hv_x", "hv_y",
          "beta_x", "beta_y",
          "adj_vol_x", "adj_vol_y",
        }
    """
    out: Dict[str, Any] = {
        "atr_x": np.nan,
        "atr_y": np.nan,
        "iv_x": np.nan,
        "iv_y": np.nan,
        "hv_x": np.nan,
        "hv_y": np.nan,
        "beta_x": np.nan,
        "beta_y": np.nan,
        "adj_vol_x": np.nan,
        "adj_vol_y": np.nan,
    }

    # Build OHLC-like frames from whatever columns we have
    def _make_ohlc(df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        cols = {c.lower(): c for c in d.columns}
        H = d[cols.get("high", list(d.columns)[0])] if cols else d.iloc[:, 0]
        L = d[cols.get("low", list(d.columns)[0])] if cols else d.iloc[:, 0]
        C = d[cols.get("close", list(d.columns)[-1])] if cols else d.iloc[:, -1]
        O = d[cols.get("open", list(d.columns)[0])] if cols else d.iloc[:, 0]
        return pd.DataFrame({"High": H, "Low": L, "Close": C, "Open": O})

    ohlc_x = _make_ohlc(df_x)
    ohlc_y = _make_ohlc(df_y)

    # ATR (leg X/Y) – uses volatility.calculate_atr
    try:
        out["atr_x"] = _safe_last(calculate_atr(ohlc_x, aw), 0.0)  # type: ignore[arg-type]
    except Exception:
        out["atr_x"] = 0.0
    try:
        out["atr_y"] = _safe_last(calculate_atr(ohlc_y, aw), 0.0)  # type: ignore[arg-type]
    except Exception:
        out["atr_y"] = 0.0

    # IV / HV / β – safe fallbacks
    try:
        out["iv_x"] = float(get_implied_volatility(bt_x))
    except Exception:
        out["iv_x"] = np.nan
    try:
        out["iv_y"] = float(get_implied_volatility(bt_y))
    except Exception:
        out["iv_y"] = np.nan
    try:
        out["hv_x"] = float(calculate_historical_volatility(df_x["close"]))  # type: ignore[arg-type]
    except Exception:
        out["hv_x"] = np.nan
    try:
        out["hv_y"] = float(calculate_historical_volatility(df_y["close"]))  # type: ignore[arg-type]
    except Exception:
        out["hv_y"] = np.nan
    try:
        out["beta_x"] = float(calculate_beta(df_x["close"]))  # type: ignore[arg-type]
    except Exception:
        out["beta_x"] = np.nan
    try:
        out["beta_y"] = float(calculate_beta(df_y["close"]))  # type: ignore[arg-type]
    except Exception:
        out["beta_y"] = np.nan

    # Adj vol blend (defensive)
    if use_atr_percent:
        try:
            atrp_x = atr_percent(ohlc_x, aw)  # type: ignore[arg-type]
            atrp_y = atr_percent(ohlc_y, aw)  # type: ignore[arg-type]
            out["adj_vol_x"] = _safe_last(atrp_x, 0.0)
            out["adj_vol_y"] = _safe_last(atrp_y, 0.0)
        except Exception:
            out["adj_vol_x"] = _safe_last(out["atr_x"], 0.0)
            out["adj_vol_y"] = _safe_last(out["atr_y"], 0.0)
    else:
        # blend של ATR + IV + HV + β
        avx = 0.4 * _safe_last(out["atr_x"]) + 0.3 * (out["iv_x"] if np.isfinite(out["iv_x"]) else 0.0)
        avx += 0.2 * (out["hv_x"] if np.isfinite(out["hv_x"]) else 0.0)
        avx += 0.1 * (out["beta_x"] if np.isfinite(out["beta_x"]) else 0.0)

        avy = 0.4 * _safe_last(out["atr_y"]) + 0.3 * (out["iv_y"] if np.isfinite(out["iv_y"]) else 0.0)
        avy += 0.2 * (out["hv_y"] if np.isfinite(out["hv_y"]) else 0.0)
        avy += 0.1 * (out["beta_y"] if np.isfinite(out["beta_y"]) else 0.0)

        out["adj_vol_x"] = float(avx)
        out["adj_vol_y"] = float(avy)

    return out


def _compute_position_sizes(
    capital: float,
    vol_stats: Dict[str, Any],
    cfg: BacktestConfig,
) -> Tuple[float, float]:
    """
    מחשב position_size_x / position_size_y על בסיס:
    - ATR% / vol blend בין הנכסים
    - vol_target (אם >0)
    - macro_mult מה-session (אם קיים)
    - max_exposure_per_trade כ-cap סופי
    """
    adj_vol_x = float(vol_stats.get("adj_vol_x", 0.0))
    adj_vol_y = float(vol_stats.get("adj_vol_y", 0.0))

    if adj_vol_x in (0.0, np.inf, -np.inf) or not np.isfinite(adj_vol_x):
        ratio = 1.0
    else:
        ratio = float(adj_vol_y / adj_vol_x)
        if not np.isfinite(ratio) or ratio <= 0:
            ratio = 1.0
    ratio = float(max(0.1, min(ratio, 10.0)))

    pos_x = capital * ratio / (1.0 + ratio)
    pos_y = capital / (1.0 + ratio)

    # vol_target scaling (using chosen estimator later; here רק scale כולל)
    # הערך המדויק של vol_estimator יחושב בלוגיקת run_backtest עצמה.
    # כאן רק אופציה ל-scale מראש אם cfg.vol_target > 0 (כ-fallback).
    if cfg.vol_target > 0:
        # scale כללי שמוחל בהמשך אחרי חישוב last_vol
        # (בפועל, run_backtest מחשב last_vol ו-scale יותר מדויק).
        pass

    # Macro multiplier מה-session (אם קיים)
    macro_mult = 1.0
    try:
        if st is not None:
            macro_mult = float(st.session_state.get("macro_mult", 1.0))
    except Exception:
        macro_mult = 1.0

    pos_x *= macro_mult
    pos_y *= macro_mult

    # Cap notional per trade
    cap_notional = capital * float(cfg.max_exposure_per_trade)
    pos_x = float(min(pos_x, cap_notional))
    pos_y = float(min(pos_y, cap_notional))

    return pos_x, pos_y


def run_backtest(
    bt_x: str,
    bt_y: str,
    z_entry: float,
    z_exit: float,
    config: Optional[Dict[str, Any]],
):
    """
    Pairs backtest with robust I/O, clean Hebrew UI, and safe fallbacks.

    Args
    ----
    bt_x, bt_y :
        סימבולים/טיקרים של שני הנכסים.
    z_entry :
        סף לפתיחת פוזיציה (|z| > z_entry).
    z_exit :
        סף לסגירת פוזיציה (|z| < z_exit).
    config :
        dict של קונפיג (או None) – יועבר ל-BacktestConfig.coerce.

    Returns
    -------
    dict או None:
        {
          "trades": DataFrame,
          "kpis": {...},
          "curves": {"cum_pnl": Series, "drawdown": Series},
        }
    """
    if st is None:
        raise RuntimeError("run_backtest requires Streamlit environment (st is None).")

    try:
        # ===== 1) Config & defaults =====
        cfg_obj = BacktestConfig.coerce(config)
        rw = int(cfg_obj.rolling_window)
        aw = int(cfg_obj.atr_window)
        capital = float(cfg_obj.capital)
        fee_rate = float(cfg_obj.fee_rate)
        slip_bps = float(cfg_obj.slippage_bps)
        stop_loss_pct = float(cfg_obj.stop_loss_pct)
        take_profit_pct = float(cfg_obj.take_profit_pct)
        risk_free = float(cfg_obj.risk_free)

        stop_loss_abs = capital * stop_loss_pct
        take_profit_abs = capital * take_profit_pct

        # ===== 2) Load & normalize prices =====
        df_pair = _load_pair_data(bt_x, bt_y, cfg_obj)
        if df_pair is None or df_pair.empty:
            return {"trades": [], "kpis": {}, "curves": {}}

        # נקרא שוב את df_x/df_y המקוריים לצורך ATR/vol (עם כל העמודות)
        df_x_full = load_price_data(bt_x)
        df_y_full = load_price_data(bt_y)
        if df_x_full is None or df_y_full is None or df_x_full.empty or df_y_full.empty:
            st.warning("לא נמצאו נתוני OHLC מלאים לחישובי תנודתיות, משתמש ב-fallbacks.")
            df_x_full = df_pair.rename(columns={"close_x": "close"})
            df_y_full = df_pair.rename(columns={"close_y": "close"})

        # ===== 3) Indicators (spread / z, ATR, vol stats) =====
        spread = df_pair["close_x"] - df_pair["close_y"]
        roll_mean = spread.rolling(rw, min_periods=max(5, rw // 2)).mean()
        roll_std = spread.rolling(rw, min_periods=max(5, rw // 2)).std()
        z = (spread - roll_mean) / (roll_std + 1e-9)

        vol_stats = _compute_leg_vol_stats(bt_x, bt_y, df_x_full, df_y_full, aw, cfg_obj.use_atr_percent)
        atr_x = vol_stats.get("atr_x", 0.0)

        # ===== 4) Position sizing =====
        position_size_x, position_size_y = _compute_position_sizes(
            capital=capital,
            vol_stats=vol_stats,
            cfg=cfg_obj,
        )

        # vol_target לפי estimator שבחרת (YZ/RS/PK/GK/RV) – scale נוסף
        try:
            est = cfg_obj.vol_estimator.lower()
            ohlc_for_est = df_x_full.copy()
            cols = {c.lower(): c for c in ohlc_for_est.columns}
            H = ohlc_for_est[cols.get("high", list(ohlc_for_est.columns)[0])]
            L = ohlc_for_est[cols.get("low", list(ohlc_for_est.columns)[0])]
            C = ohlc_for_est[cols.get("close", list(ohlc_for_est.columns)[-1])]
            O = ohlc_for_est[cols.get("open", list(ohlc_for_est.columns)[0])]
            ohlc = pd.DataFrame({"High": H, "Low": L, "Close": C, "Open": O})

            if est == "yz":
                vol_series = yang_zhang_vol(ohlc, 20)  # type: ignore[arg-type]
            elif est == "rs":
                vol_series = rogers_satchell_vol(ohlc, 20)  # type: ignore[arg-type]
            elif est == "pk":
                vol_series = parkinson_vol(ohlc, 20)  # type: ignore[arg-type]
            elif est == "gk":
                vol_series = garman_klass_vol(ohlc, 20)  # type: ignore[arg-type]
            else:
                vol_series = realized_vol_from_close(ohlc["Close"], 20)  # type: ignore[arg-type]

            last_vol = float(vol_series.dropna().iloc[-1]) if not vol_series.empty else 0.0
            if last_vol and cfg_obj.vol_target > 0:
                scale = float(min(2.0, max(0.25, cfg_obj.vol_target / last_vol)))
                position_size_x *= scale
                position_size_y *= scale
        except Exception:
            logger.debug("vol_target scaling failed", exc_info=True)

        # Cap שוב אם scale הגדיל מעבר למותר
        cap_notional = capital * cfg_obj.max_exposure_per_trade
        position_size_x = float(min(position_size_x, cap_notional))
        position_size_y = float(min(position_size_y, cap_notional))

        # ===== 5) Build backtest frame =====
        df_bt = (
            pd.DataFrame(
                {
                    "date": df_pair["date"],
                    "z": z,
                    "spread": spread,
                    "px": df_pair["close_x"],
                    "py": df_pair["close_y"],
                }
            )
            .dropna()
            .reset_index(drop=True)
        )

        if df_bt.empty:
            st.warning("לאחר חישוב spread/Z אין מספיק נתונים לסימולציה.")
            return {"trades": [], "kpis": {}, "curves": {}}

        # round-trip costs
        roundtrip_fee = fee_rate * (position_size_x + position_size_y) * 2.0
        roundtrip_slip = (slip_bps / 10_000.0) * (position_size_x + position_size_y) * 2.0

        # ===== 6) Trading loop =====
        in_trade = False
        entry_px = entry_py = None
        entry_date = None
        direction = 0  # +1 = long spread (long X short Y), -1 = short spread (short X long Y)

        pnl_list: List[float] = []
        trades: List[Dict[str, Any]] = []
        drawdowns: List[float] = []
        peak = 0.0

        wins = losses = 0

        for i in range(len(df_bt)):
            row = df_bt.iloc[i]

            if not in_trade:
                # Entry condition: |Z| > z_entry
                if abs(row["z"]) > float(z_entry):
                    in_trade = True
                    entry_date = row["date"]
                    entry_px = float(row["px"])
                    entry_py = float(row["py"])
                    # z>0: spread גבוה → short X long Y  → direction=-1 (פר הגדרה למטה)
                    direction = -1 if row["z"] > 0 else 1
            else:
                # Evaluate exit conditions
                exit_signal = False
                exit_reason = "z_exit"

                if abs(row["z"]) < float(z_exit):
                    exit_signal = True
                    exit_reason = "z_exit"
                else:
                    # Check SL/TP on unrealized PnL
                    tmp_profit = direction * (
                        (entry_px - float(row["px"])) * position_size_x
                        + (float(row["py"]) - entry_py) * position_size_y
                    )
                    if tmp_profit <= -stop_loss_abs:
                        exit_signal = True
                        exit_reason = "stop_loss"
                    elif tmp_profit >= take_profit_abs:
                        exit_signal = True
                        exit_reason = "take_profit"

                if exit_signal:
                    exit_date = row["date"]
                    exit_px = float(row["px"])
                    exit_py = float(row["py"])

                    gross = direction * (
                        (entry_px - exit_px) * position_size_x
                        + (exit_py - entry_py) * position_size_y
                    )
                    net = gross - roundtrip_fee - roundtrip_slip
                    profit = float(net)

                    pnl_list.append(profit)
                    cur_equity = sum(pnl_list)
                    peak = max(peak, cur_equity)
                    drawdowns.append(peak - cur_equity)

                    if profit > 0:
                        wins += 1
                    else:
                        losses += 1

                    trades.append(
                        {
                            "entry": entry_date,
                            "exit": exit_date,
                            "z_entry": float(z_entry),
                            "z_exit": float(z_exit),
                            "profit_usd": round(profit, 2),
                            "notional_x": round(position_size_x, 2),
                            "notional_y": round(position_size_y, 2),
                            "atr_x": atr_x if np.isfinite(atr_x) else None,
                            "iv_x": vol_stats.get("iv_x")
                            if np.isfinite(vol_stats.get("iv_x", np.nan))
                            else None,
                            "hv_x": vol_stats.get("hv_x")
                            if np.isfinite(vol_stats.get("hv_x", np.nan))
                            else None,
                            "beta_x": vol_stats.get("beta_x")
                            if np.isfinite(vol_stats.get("beta_x", np.nan))
                            else None,
                            "exit_reason": exit_reason,
                        }
                    )
                    in_trade = False

        # ===== 7) Reporting & UI =====
        if not pnl_list:
            st.warning("לא בוצעו עסקאות בטווח הנתון.")
            return {"trades": [], "kpis": {}, "curves": {}}

        pnl_series = pd.Series(pnl_list, dtype=float)
        kpis = _compute_kpis_from_pnl(pnl_series, capital, risk_free)
        cum_pnl = pnl_series.cumsum()
        equity = capital + cum_pnl

        days_bt = max(1, len(df_bt))
        cagr = float((equity.iloc[-1] / capital) ** (365.0 / days_bt) - 1.0) if len(equity) else 0.0

        st.success(
            f"בוצעו {len(pnl_series)} עסקאות | רווח כולל: ${pnl_series.sum():.2f} | "
            f"ממוצע לעסקה: ${pnl_series.mean():.2f} | "
            f"שיעור הצלחה: {kpis['win_rate']:.1f}% | "
            f"מקס' ירידה: ${kpis['max_dd_usd']:.2f} | "
            f"Sharpe: {kpis['sharpe'] if kpis['sharpe'] is not None else 'N/A'} | "
            f"Sortino: {kpis['sortino'] if kpis['sortino'] is not None else 'N/A'} | "
            f"CAGR: {cagr:.2%} | "
            f"Calmar: {kpis['calmar'] if kpis['calmar'] is not None else 'N/A'} | "
            f"ES95: {kpis['es95'] if kpis['es95'] is not None else 'N/A'}"
        )

        # ==== Charts ====
        if go is not None:
            # Cum PnL
            fig_eq = go.Figure(
                data=go.Scatter(y=cum_pnl, mode="lines+markers", name="Cumulative PnL")
            )
            fig_eq.update_layout(title="Cumulative PnL (Trades Sequence)")
            st.plotly_chart(fig_eq, width = "stretch")

            # PnL vs Time (by trade exit)
            df_trades = pd.DataFrame(trades)
            df_trades["exit"] = pd.to_datetime(df_trades["exit"], errors="coerce")
            df_trades = df_trades.sort_values("exit")
            fig_pnl = go.Figure(
                data=go.Scatter(
                    x=df_trades["exit"],
                    y=df_trades["profit_usd"].cumsum(),
                    mode="lines+markers",
                    name="PnL vs Time",
                )
            )
            fig_pnl.update_layout(title="PnL vs Time")
            st.plotly_chart(fig_pnl, width = "stretch")

            # Drawdown path (per trade)
            fig_dd = go.Figure(
                data=go.Scatter(
                    y=pd.Series(drawdowns),
                    mode="lines+markers",
                    name="Drawdown",
                )
            )
            fig_dd.update_layout(title="Drawdown (per trade sequence)")
            st.plotly_chart(fig_dd, width = "stretch")

        # ==== Trades table & stats ====
        df_trades = pd.DataFrame(trades)
        df_trades["duration_days"] = (
            pd.to_datetime(df_trades["exit"]) - pd.to_datetime(df_trades["entry"])
        ).dt.days
        avg_duration = float(df_trades["duration_days"].mean()) if len(df_trades) else 0.0
        std_pnl = float(df_trades["profit_usd"].std()) if len(df_trades) else 0.0

        st.markdown(
            f"**משך ממוצע לעסקה:** {avg_duration:.1f} ימים | "
            f"**סטיית תקן PnL לעסקה:** ${std_pnl:.2f}"
        )

        df_display = df_trades.rename(
            columns={
                "entry": "כניסה",
                "exit": "יציאה",
                "z_entry": "Z כניסה",
                "z_exit": "Z יציאה",
                "profit_usd": "רווח ($)",
                "notional_x": "כמות X",
                "notional_y": "כמות Y",
                "atr_x": "ATR X",
                "iv_x": "IV X",
                "hv_x": "HV X",
                "beta_x": "Beta X",
                "exit_reason": "סיבת יציאה",
            }
        )
        st.dataframe(df_display, width = "stretch")

        # ==== Downloads & logs ====
        csv_bytes = df_trades.to_csv(index=False).encode("utf-8")
        st.download_button(
            "הורד טבלת עסקאות (CSV)",
            data=csv_bytes,
            file_name=f"{bt_x}_{bt_y}_trades.csv",
            mime="text/csv",
            key="bt_trades_csv",
        )

        os.makedirs("logs", exist_ok=True)
        log_path = os.path.join("logs", f"{bt_x}_{bt_y}_log.csv")
        df_trades.to_csv(log_path, index=False)
        st.info(f"נשמר לוג אחרון אל: {log_path}")

        # Optional: append to DuckDB if configured
        if cfg_obj.duckdb_path:
            try:
                import duckdb  # type: ignore

                con = duckdb.connect(cfg_obj.duckdb_path)
                df_to_write = df_trades.assign(pair=f"{bt_x}_{bt_y}")
                con.execute(
                    "CREATE TABLE IF NOT EXISTS backtests AS SELECT * FROM df_to_write LIMIT 0"
                ).close()
                con.register("df_to_write", df_to_write)
                con.execute("INSERT INTO backtests SELECT * FROM df_to_write").close()
                con.close()
                st.success(f"נשמרה גם רשומה ל-DuckDB: {cfg_obj.duckdb_path}")
            except Exception as e:
                st.warning(f"DuckDB append failed: {e}")

        # השוואה לרווחים קודמים מקבצי logs
        try:
            all_logs: List[pd.DataFrame] = []
            for file in os.listdir("logs"):
                if file.endswith("_log.csv") and file != os.path.basename(log_path):
                    df_old = pd.read_csv(os.path.join("logs", file))
                    df_old["pair"] = file.replace("_log.csv", "")
                    all_logs.append(df_old)
            if all_logs:
                df_all = pd.concat(
                    all_logs + [df_trades.assign(pair=f"{bt_x}_{bt_y}")]
                ).reset_index(drop=True)
                pivot = (
                    df_all.groupby("pair")["profit_usd"]
                    .sum()
                    .sort_values(ascending=False)
                )
                st.markdown("### השוואה לרווחים קודמים (Logs קודמים):")
                st.bar_chart(pivot)
        except Exception as e:
            st.warning(f"שגיאה בקריאת סימולציות קודמות: {e}")

        st.markdown("---")
        st.subheader("סימולציות ביצוע עסקאות (אוטומציה — מושבת)")

        enable_trading = st.toggle(
            "אשר סימולציית שליחת פקודות (לוג בלבד, ללא ברוקר אמיתי)",
            value=False,
        )
        if enable_trading:
            for trade in trades:
                st.markdown(
                    f"טיקט: LONG/SHORT spread {bt_x}-{bt_y} | Notional≈${capital:,.0f}"
                )
            st.success("סימולציית ביצוע הסתיימה (ללא חיבור לברוקר).")

        return {
            "trades": df_trades,
            "kpis": {
                "total_pnl": float(pnl_series.sum()),
                "avg_pnl": float(pnl_series.mean()),
                **kpis,
            },
            "curves": {
                "cum_pnl": cum_pnl,
                "drawdown": pd.Series(drawdowns),
            },
        }

    except Exception as e:
        logger.exception("Backtest simulation failed", exc_info=e)
        st.error(f"שגיאה בסימולציה: {e}")
        return None

# ============================================================================
# Part 3/3 — Volatility Overview Panel (HF-grade) & Helpers
# ============================================================================

def _build_ohlc_from_loader(df: pd.DataFrame) -> pd.DataFrame:
    """
    Helper קטן שמוציא מסגרת OHLC מתוך מה שה-loader מחזיר.

    מנרמל:
        - High / Low / Close / Open
        - אם חסר משהו → משתמש בעמודה המספרית הראשונה/האחרונה.
    """
    d = df.copy()
    d.columns = [str(c).lower() for c in d.columns]
    cols = {c.lower(): c for c in d.columns}

    def _pick(name: str, fallback_idx: int) -> pd.Series:
        if name in cols:
            return pd.to_numeric(d[cols[name]], errors="coerce")
        return pd.to_numeric(d.iloc[:, fallback_idx], errors="coerce")

    if d.empty:
        return pd.DataFrame(columns=["High", "Low", "Close", "Open"])

    H = _pick("high", 0)
    L = _pick("low", 0)
    C = _pick("close", min(len(d.columns) - 1, 0))
    O = _pick("open", 0)

    return pd.DataFrame({"High": H, "Low": L, "Close": C, "Open": O}, index=d.index)


def render_volatility_overview(
    pairs: Sequence[str],
    start: Optional[str] = None,
    end: Optional[str] = None,
    window: int = 20,
) -> Optional[pd.DataFrame]:
    """
    Volatility Overview – טאב/פאנל תנודתיות ברמת קרן גידור.

    Features
    --------
    - מחשב לכל נכס:
        * Yang–Zhang (YZ), Rogers–Satchell (RS), Parkinson (PK),
          Garman–Klass (GK), Realized Vol (RV).
        * ATR% (אם אפשר), Volatility Regime, Spike flag.
        * Close אחרון, מספר ימי דאטה אפקטיבי.
    - מציג:
        * KPIs גלובליים (מדיאנים של מדדי תנודתיות, #Spikes, חלוקת Regimes).
        * טבלת Summary עם אפשרות סינון/חיפוש.
        * בחירה של נכס אחד ל־Drilldown גרפי על סדרות התנודתיות.
    - מחזיר את DataFrame ה-summary להמשך שימוש במודולים אחרים.

    Notes
    -----
    - אם Streamlit (st) אינו זמין → נחזיר רק DataFrame (ללא UI).
    - אם load_price_data מחזיר אינדקס שאינו DatetimeIndex,
      לא נכפה סינון תאריכים (רק נחזיר את הערכים האחרונים).
    """
    if not pairs:
        if st is not None:
            st.info("בחר לפחות נכס אחד להצגה.")
        return None

    summary_rows: List[Dict[str, Any]] = []

    for sym in pairs:
        try:
            df_raw = load_price_data(sym)
            if df_raw is None or df_raw.empty:
                logger.warning("No data for symbol %s in volatility overview", sym)
                continue

            df = df_raw.copy()
            # ננסה לסנן לפי start/end רק אם האינדקס הוא DatetimeIndex
            try:
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index, errors="coerce")
            except Exception:
                pass

            if isinstance(df.index, pd.DatetimeIndex):
                if start:
                    try:
                        df = df[df.index >= pd.to_datetime(start)]
                    except Exception:
                        pass
                if end:
                    try:
                        df = df[df.index <= pd.to_datetime(end)]
                    except Exception:
                        pass

            if df.empty:
                continue

            ohlc = _build_ohlc_from_loader(df)

            # Vol estimators
            yz_val = rs_val = pk_val = gk_val = rv_val = np.nan
            try:
                yz_val = float(yang_zhang_vol(ohlc, window).dropna().iloc[-1])  # type: ignore[arg-type]
            except Exception:
                pass
            try:
                rs_val = float(rogers_satchell_vol(ohlc, window).dropna().iloc[-1])  # type: ignore[arg-type]
            except Exception:
                pass
            try:
                pk_val = float(parkinson_vol(ohlc, window).dropna().iloc[-1])  # type: ignore[arg-type]
            except Exception:
                pass
            try:
                gk_val = float(garman_klass_vol(ohlc, window).dropna().iloc[-1])  # type: ignore[arg-type]
            except Exception:
                pass
            try:
                rv_val = float(realized_vol_from_close(ohlc["Close"], window).dropna().iloc[-1])  # type: ignore[arg-type]
            except Exception:
                pass

            # ATR% (אם אפשר)
            atrp_val = np.nan
            try:
                atrp_series = atr_percent(ohlc, window)  # type: ignore[arg-type]
                atrp_val = float(atrp_series.dropna().iloc[-1]) if not atrp_series.empty else np.nan
            except Exception:
                pass

            # Regime & spike flags
            reg_val = None
            spike_flag = False
            try:
                reg_series = volatility_regime(ohlc, estimator="yz", window=window)  # type: ignore[arg-type]
                reg_val = reg_series.dropna().iloc[-1] if not reg_series.empty else None
            except Exception:
                pass
            try:
                spike_series = is_spike_day(ohlc, window=14, threshold=0.04)  # type: ignore[arg-type]
                spike_flag = bool(spike_series.dropna().iloc[-1]) if not spike_series.empty else False
            except Exception:
                pass

            # Last close & effective sample size
            try:
                last_close = float(ohlc["Close"].dropna().iloc[-1])
            except Exception:
                last_close = np.nan
            n_obs = int(ohlc["Close"].dropna().shape[0])

            summary_rows.append(
                {
                    "Symbol": sym,
                    f"YZ({window})": yz_val,
                    f"RS({window})": rs_val,
                    f"PK({window})": pk_val,
                    f"GK({window})": gk_val,
                    f"RV({window})": rv_val,
                    "ATR%": atrp_val,
                    "LastClose": last_close,
                    "N_obs": n_obs,
                    "Regime": reg_val,
                    "Spike>4%": spike_flag,
                }
            )
        except Exception as e:
            logger.warning("Volatility overview failed for %s: %s", sym, e)
            if st is not None:
                st.warning(f"{sym}: {e}")

    if not summary_rows:
        if st is not None:
            st.info("אין נתוני תנודתיות להצגה עבור הנכסים שנבחרו.")
        return None

    df_sum = pd.DataFrame(summary_rows)

    # אם אין Streamlit (שימוש כספרייה בלבד) → נחזיר את הטבלה ונצא
    if st is None:
        return df_sum

    # ===== UI Layer (Streamlit) =====
    st.header("📈 Volatility Overview — Multi-Estimator Panel")

    # KPIs על פני כל universe
    c1, c2, c3, c4, c5 = st.columns(5)
    yz_col = f"YZ({window})"
    rs_col = f"RS({window})"
    pk_col = f"PK({window})"
    rv_col = f"RV({window})"

    c1.metric(
        f"Median {yz_col}",
        f"{np.nanmedian(df_sum[yz_col]):.2%}" if yz_col in df_sum else "N/A",
    )
    c2.metric(
        f"Median {rs_col}",
        f"{np.nanmedian(df_sum[rs_col]):.2%}" if rs_col in df_sum else "N/A",
    )
    c3.metric(
        f"Median {pk_col}",
        f"{np.nanmedian(df_sum[pk_col]):.2%}" if pk_col in df_sum else "N/A",
    )
    c4.metric(
        f"Median {rv_col}",
        f"{np.nanmedian(df_sum[rv_col]):.2%}" if rv_col in df_sum else "N/A",
    )
    c5.metric(
        "Spikes today",
        f"{int(df_sum['Spike>4%'].sum())}",
    )

    # פילטרים מהירים על הטבלה
    with st.expander("🔎 פילטרים מתקדמים לתנודתיות", expanded=False):
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            min_obs = st.number_input(
                "Minimum observations",
                min_value=10,
                max_value=int(df_sum["N_obs"].max()),
                value=window * 2,
                step=10,
                key="vol_min_obs",
            )
        with col_f2:
            regime_filter = st.multiselect(
                "Regime filter",
                options=sorted(df_sum["Regime"].dropna().astype(str).unique().tolist()),
                default=[],
                key="vol_reg_filter",
            )
        with col_f3:
            spike_only = st.checkbox("הצג רק נכסים עם Spike>4%", value=False, key="vol_spike_only")

        df_view = df_sum.copy()
        df_view = df_view[df_view["N_obs"] >= int(min_obs)]

        if regime_filter:
            df_view = df_view[df_view["Regime"].astype(str).isin(regime_filter)]
        if spike_only:
            df_view = df_view[df_view["Spike>4%"]]

    st.subheader("📋 Summary Table (filtered)")
    st.dataframe(df_view, width = "stretch")

    # הורדת CSV
    try:
        csv_bytes = df_view.to_csv(index=False).encode("utf-8")
        st.download_button(
            "💾 הורד טבלת תנודתיות (CSV)",
            data=csv_bytes,
            file_name=f"volatility_overview_{window}.csv",
            mime="text/csv",
            key="vol_overview_csv",
        )
    except Exception:
        pass

    # Drilldown על סימבול אחד – הצגת גרפי Vol לאורך זמן
    st.subheader("🔬 Drilldown — Volatility Time Series לנכס בודד")
    sym_choices = df_view["Symbol"].tolist()
    if sym_choices:
        sel_sym = st.selectbox("בחר סימבול ל-Drilldown", sym_choices, key="vol_drill_sym")
        try:
            df_raw = load_price_data(sel_sym)
            if df_raw is not None and not df_raw.empty:
                ohlc = _build_ohlc_from_loader(df_raw)
                # נחשב סדרות vol לאורך זמן
                try:
                    yz_series = yang_zhang_vol(ohlc, window)  # type: ignore[arg-type]
                except Exception:
                    yz_series = pd.Series(dtype=float)
                try:
                    rv_series = realized_vol_from_close(ohlc["Close"], window)  # type: ignore[arg-type]
                except Exception:
                    rv_series = pd.Series(dtype=float)

                if go is not None and (not yz_series.empty or not rv_series.empty):
                    fig_vol = go.Figure()
                    if not yz_series.empty:
                        fig_vol.add_trace(
                            go.Scatter(
                                x=yz_series.index,
                                y=yz_series.values,
                                mode="lines",
                                name=f"YZ({window})",
                            )
                        )
                    if not rv_series.empty:
                        fig_vol.add_trace(
                            go.Scatter(
                                x=rv_series.index,
                                y=rv_series.values,
                                mode="lines",
                                name=f"RV({window})",
                            )
                        )
                    fig_vol.update_layout(
                        title=f"Volatility Time Series — {sel_sym}",
                        xaxis_title="Date",
                        yaxis_title="Annualised Vol",
                    )
                    st.plotly_chart(fig_vol, width = "stretch")
                else:
                    st.info("לא ניתן היה לבנות גרף תנודתיות (סדרות ריקות).")
            else:
                st.info("אין מספיק דאטה ל-Drilldown עבור הסימבול שנבחר.")
        except Exception as e:
            st.warning(f"Drilldown נכשל עבור {sel_sym}: {e}")

    return df_sum
