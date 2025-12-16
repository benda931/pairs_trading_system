# -*- coding: utf-8 -*-
"""
common/signal_generator.py — Hedge-Fund-Grade Signal Engine
===========================================================

מודול יצירת סיגנלים מקצועי למערכת המסחר הזוגי שלך.

תומך ב־:
    - Z-score / Spread Z-score
    - Bollinger Bands
    - CUSUM
    - RSI
    - Rolling Correlation
    - Rolling Cointegration (p-value)
    - Rolling ADF
    - Aggregation / Consensus של סיגנלים

Features
--------
- Pydantic-based configuration לכל משפחת סיגנלים (קונפיג ניתן לשמירה כ-JSON).
- לוגים מקצועיים עם Logger מודולרי.
- Caching פנימי לסיגנלים (כדי לא לחשב שוב ושוב על אותן סדרות).
- SignalGenerator מרכזי שתומך בהוקס לפני/אחרי (callbacks ל-alerts/backtest).
- API נקי, מפורש, וניתן להרחבה לטכניקות נוספות (MACD, ATR, וכו').

שימוש בסיסי
-----------
    from common.signal_generator import SignalGenerator

    cfg = SignalGenerator.default_config()
    cfg["zscore"]["window"] = 30

    sg = SignalGenerator(cfg)
    price_x = ...  # pd.Series
    price_y = ...  # pd.Series

    signals = sg.generate(price_x, series2=price_y)
    consensus = sg.aggregate_signals(signals)
"""

from __future__ import annotations

import logging
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, validator

from common.json_safe import make_json_safe, json_default as _json_default


# ============================================================================
# Logger setup
# ============================================================================

logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)


# ============================================================================
# Type aliases & callback hooks
# ============================================================================

PriceSeries = pd.Series
SignalFrame = pd.DataFrame

# callback שמקבל DataFrame עם סיגנלים (למשל לשליחת Alert / ל-Backtest)
SignalCallback = Callable[[pd.DataFrame], None]
# הוקים לפני/אחרי – יותר גמישים (אם תרצה לעבוד על price בלבד)
PreHook = Callable[[PriceSeries, Optional[PriceSeries]], None]
PostHook = Callable[[pd.DataFrame], None]


# ============================================================================
# Caching infrastructure (ללא lru_cache על Series unhashable)
# ============================================================================

_SIGNAL_CACHES: Dict[str, Dict[Any, pd.DataFrame]] = {}


def _make_hashable(arg: Any) -> Any:
    """
    הפיכה גסה של פרמטרים ל-key עבור cache.

    - pd.Series → (id, name, length)
    - Pydantic BaseModel → (class_name, json)
    - list/dict/set → repr
    - אחרת → הערך עצמו
    """
    if isinstance(arg, pd.Series):
        return ("Series", id(arg), arg.name, len(arg))
    if isinstance(arg, BaseModel):
        try:
            return (arg.__class__.__name__, arg.json())
        except Exception:
            return (arg.__class__.__name__, repr(arg.dict()))
    if isinstance(arg, (list, dict, set, tuple)):
        return repr(arg)
    return arg


def signal_cache(func: Callable[..., pd.DataFrame]) -> Callable[..., pd.DataFrame]:
    """
    Decorator: Cache תוצאות של פונקציות סיגנל (לפי args/kwargs).

    לא משתמשים ישירות ב-lru_cache על Series (unhashable),
    אלא ב־dict פנימי פר-פונקציה עם key גמיש.
    """
    cache: Dict[Any, pd.DataFrame] = _SIGNAL_CACHES.setdefault(func.__name__, {})

    @wraps(func)
    def wrapper(*args, **kwargs) -> pd.DataFrame:
        key_args = tuple(_make_hashable(a) for a in args)
        key_kwargs = tuple(sorted((k, _make_hashable(v)) for k, v in kwargs.items()))
        key = (key_args, key_kwargs)

        if key in cache:
            return cache[key]

        out = func(*args, **kwargs)
        if not isinstance(out, pd.DataFrame):
            raise TypeError(f"{func.__name__} must return a pandas.DataFrame, got {type(out)}")
        cache[key] = out
        return out

    # מאפשר ל-SignalGenerator.clear_caches לנקות
    wrapper._signal_cache = cache  # type: ignore[attr-defined]
    return wrapper


def clear_all_signal_caches() -> None:
    """Clear all internal signal caches (useful לפני Backtest גדול)."""
    for cache in _SIGNAL_CACHES.values():
        cache.clear()
    logger.info("All signal caches cleared.")


# ============================================================================
# Pydantic Configuration Models
# ============================================================================

class ZScoreConfig(BaseModel):
    """
    Z-score signal configuration.

    Attributes
    ----------
    window : int
        Rolling window length.
    entry_threshold : float
        Absolute z-score threshold to open a position.
    exit_threshold : float
        Absolute z-score threshold to close a position.
    enabled : bool
        If False – לא מחשב בכלל את הסיגנל הזה.
    """
    window: int = 20
    entry_threshold: float = 2.0
    exit_threshold: float = 1.0
    enabled: bool = True

    @validator("window")
    def window_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("window must be >= 1")
        return v


class BollingerConfig(BaseModel):
    """
    Bollinger Bands signal configuration.
    """
    window: int = 20
    num_std: float = 2.0
    enabled: bool = True

    @validator("window")
    def window_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("window must be >= 1")
        return v


class CUSUMConfig(BaseModel):
    """
    CUSUM signal configuration.

    threshold : float
        Trigger level for positive/negative CUSUM.
    drift : float
        Drift adjustment per step.
    """
    threshold: float = 0.5
    drift: float = 0.0
    enabled: bool = True


class RSIConfig(BaseModel):
    """
    RSI-based signal configuration.
    """
    window: int = 14
    lower: float = 30.0
    upper: float = 70.0
    enabled: bool = True

    @validator("window")
    def window_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("window must be >= 1")
        return v


class CorrelationConfig(BaseModel):
    """
    Rolling correlation signal configuration.

    entry_threshold : float
        לעיתים תפרש כ-"correlation מתחת לזה מייצר signal" (או מעל).
    exit_threshold : float
        רמת קורלציה שנחשבת "חזרה לנורמלי".
    """
    window: int = 20
    entry_threshold: float = 0.8
    exit_threshold: float = 0.9
    enabled: bool = True

    @validator("window")
    def window_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("window must be >= 1")
        return v


class CointegrationConfig(BaseModel):
    """
    Rolling cointegration (Engle-Granger/Johansen) p-value-based signal.

    entry_pval : float
        p-value threshold to consider cointegration "valid".
    exit_pval : float
        p-value threshold above which cointegration "fails".
    """
    window: int = 30
    entry_pval: float = 0.05
    exit_pval: float = 0.1
    enabled: bool = True

    @validator("window")
    def window_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("window must be >= 1")
        return v


class ADFConfig(BaseModel):
    """
    Rolling ADF p-value-based mean-reversion test configuration.
    """
    window: int = 30
    entry_pval: float = 0.05
    exit_pval: float = 0.1
    enabled: bool = True

    @validator("window")
    def window_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("window must be >= 1")
        return v


class SpreadConfig(BaseModel):
    """
    Spread Z-score configuration for two legs.

    hedge_ratio : float
        Fixed hedge ratio (למשל beta) לשימוש בחישוב spread.
    window : int
        Rolling window לזיהוי mean-reversion.
    entry_threshold, exit_threshold : float
        Thresholds לזיהוי כניסה/יציאה.
    """
    hedge_ratio: float = 1.0
    window: int = 20
    entry_threshold: float = 2.0
    exit_threshold: float = 1.0
    enabled: bool = True

    @validator("window")
    def window_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("window must be >= 1")
        return v


# ============================================================================
# Internal Utilities
# ============================================================================

def _validate_series(series: pd.Series, name: str = "series") -> None:
    """
    Validate that object is a pandas.Series ולא ריק לגמרי.

    Parameters
    ----------
    series : pd.Series
    name : str
        Logical name used for logging / error messages.
    """
    if not isinstance(series, pd.Series):
        logger.error("%s must be a pandas Series, got %s", name, type(series))
        raise TypeError(f"{name} must be a pandas Series")
    if series.empty:
        logger.warning("Empty series provided to %s", name)
    if not series.index.is_monotonic_increasing:
        logger.debug("%s index is not sorted – recommended to sort before passing.", name)

# ============================================================================
# Signal Functions (Z-score, Bollinger, CUSUM, RSI, Corr, Cointegration, ADF, Spread)
# ============================================================================

@signal_cache
def zscore_signals(price: PriceSeries, cfg: ZScoreConfig) -> SignalFrame:
    """
    Generate Z-score based signals.

    Logic
    -----
    - z = rolling Z-score of price.
    - Long entry  (+1) when z crosses below -entry_threshold.
    - Short entry (-1) when z crosses above +entry_threshold.
    - Exit (0) when there was an open position and |z| < exit_threshold.

    Columns
    -------
    zscore : float
        Rolling Z-score.
    entry  : int
        +1 = long entry, -1 = short entry, 0 = no new entry.
    exit   : int
        1 = exit to flat, 0 = otherwise.
    """
    if not cfg.enabled:
        return pd.DataFrame(index=price.index, columns=["zscore", "entry", "exit"]).fillna(0.0)

    _validate_series(price, "price")
    from common.feature_engineering import zscore as fe_zscore

    z = fe_zscore(price, cfg.window)

    z_shift = z.shift(1)
    long_entry = ((z < -cfg.entry_threshold) & (z_shift >= -cfg.entry_threshold)).astype(int)
    short_entry = ((z > cfg.entry_threshold) & (z_shift <= cfg.entry_threshold)).astype(int)
    entry = long_entry - short_entry  # +1 long, -1 short, 0 none

    prev_pos = entry.replace(0, np.nan).ffill().shift(1).fillna(0)
    exit_mask = (prev_pos != 0) & (z.abs() < cfg.exit_threshold)
    exit_ = exit_mask.astype(int)

    df = pd.DataFrame(
        {"zscore": z, "entry": entry, "exit": exit_},
        index=price.index,
    )
    return df


@signal_cache
def bollinger_signals(price: PriceSeries, cfg: BollingerConfig) -> SignalFrame:
    """
    Generate signals based on Bollinger Bands.

    Logic
    -----
    - Long entry  (+1) when price crosses below lower band.
    - Short entry (-1) when price crosses above upper band.
    - Exit when price crosses back into the band region.

    Columns
    -------
    upper, lower : float
        Bollinger upper/lower bands.
    entry        : int
        +1 = long, -1 = short, 0 = no new entry.
    exit         : int
        1 = exit to flat, 0 = otherwise.
    """
    if not cfg.enabled:
        return pd.DataFrame(
            index=price.index,
            columns=["upper", "lower", "entry", "exit"],
        ).fillna(0.0)

    _validate_series(price, "price")
    from common.feature_engineering import bollinger_bands

    bands = bollinger_bands(price, cfg.window, cfg.num_std)
    lower, upper = bands["bb_lower"], bands["bb_upper"]

    prev_price = price.shift(1)
    long_entry = ((price < lower) & (prev_price >= lower)).astype(int)
    short_entry = ((price > upper) & (prev_price <= upper)).astype(int)
    entry = long_entry - short_entry

    prev_pos = entry.replace(0, np.nan).ffill().shift(1).fillna(0)
    # Exit if previously long and price >= lower OR previously short and price <= upper
    exit_long = (prev_pos == 1) & (price >= lower)
    exit_short = (prev_pos == -1) & (price <= upper)
    exit_ = (exit_long | exit_short).astype(int)

    df = pd.DataFrame(
        {"upper": upper, "lower": lower, "entry": entry, "exit": exit_},
        index=price.index,
    )
    return df


@signal_cache
def cusum_signals(price: PriceSeries, cfg: CUSUMConfig) -> SignalFrame:
    """
    Generate signals based on CUSUM thresholds.

    Logic
    -----
    Uses core.feature_engineering.cusum to compute cusum_pos / cusum_neg.

    - Long entry  (+1) when cusum_pos exceeds threshold.
    - Short entry (-1) when cusum_neg < -threshold.
    - Exit when corresponding cusum reverts below |threshold|.
    """
    if not cfg.enabled:
        return pd.DataFrame(
            index=price.index,
            columns=["cusum_pos", "cusum_neg", "entry", "exit"],
        ).fillna(0.0)

    _validate_series(price, "price")
    from common.feature_engineering import cusum

    csum = cusum(price, cfg.threshold, cfg.drift)
    cp, cn = csum["cusum_pos"], csum["cusum_neg"]

    long_entry = (cp > cfg.threshold).astype(int)
    short_entry = (cn < -cfg.threshold).astype(int)
    entry = long_entry - short_entry

    prev_pos = entry.replace(0, np.nan).ffill().shift(1).fillna(0)
    exit_long = (prev_pos == 1) & (cp <= cfg.threshold)
    exit_short = (prev_pos == -1) & (cn >= -cfg.threshold)
    exit_ = (exit_long | exit_short).astype(int)

    df = csum.copy()
    df["entry"], df["exit"] = entry, exit_
    return df


@signal_cache
def rsi_signals(price: PriceSeries, cfg: RSIConfig) -> SignalFrame:
    """
    Generate signals based on RSI extremes.

    Logic
    -----
    - Long entry  (+1) when RSI crosses below lower bound.
    - Short entry (-1) when RSI crosses above upper bound.
    - Exit when RSI returns inside [lower, upper].

    Columns
    -------
    rsi   : float
    entry : int (+1/-1/0)
    exit  : int (1/0)
    """
    if not cfg.enabled:
        return pd.DataFrame(index=price.index, columns=["rsi", "entry", "exit"]).fillna(0.0)

    _validate_series(price, "price")
    from common.feature_engineering import rsi as fe_rsi

    r = fe_rsi(price, cfg.window)

    r_prev = r.shift(1)
    long_entry = ((r < cfg.lower) & (r_prev >= cfg.lower)).astype(int)
    short_entry = ((r > cfg.upper) & (r_prev <= cfg.upper)).astype(int)
    entry = long_entry - short_entry

    prev_pos = entry.replace(0, np.nan).ffill().shift(1).fillna(0)
    exit_ = ((prev_pos != 0) & (r.between(cfg.lower, cfg.upper))).astype(int)

    df = pd.DataFrame({"rsi": r, "entry": entry, "exit": exit_}, index=price.index)
    return df


@signal_cache
def correlation_signals(
    price1: PriceSeries,
    price2: PriceSeries,
    cfg: CorrelationConfig,
) -> SignalFrame:
    """
    Generate signals based on rolling correlation between two series.

    דוגמה אופיינית:
      - Entry/alert כש-corr יורד מתחת ל-entry_threshold (קשר היסטורי נחלש).
      - Exit כש-corr חוזר מעל exit_threshold.

    Columns
    -------
    corr  : float
    entry : int (+1 = divergence alert, 0 otherwise)
    exit  : int (1 = back to normal, 0 otherwise)
    """
    if not cfg.enabled:
        return pd.DataFrame(index=price1.index, columns=["corr", "entry", "exit"]).fillna(0.0)

    _validate_series(price1, "series1")
    _validate_series(price2, "series2")

    from common.feature_engineering import rolling_correlation

    corr = rolling_correlation(price1, price2, cfg.window)
    corr_prev = corr.shift(1)

    # Entry when correlation drops below entry_threshold from above
    entry = ((corr < cfg.entry_threshold) & (corr_prev >= cfg.entry_threshold)).astype(int)

    # Exit when correlation climbs back above exit_threshold
    prev_state = entry.replace(0, np.nan).ffill().shift(1).fillna(0)
    exit_ = ((prev_state == 1) & (corr >= cfg.exit_threshold)).astype(int)

    df = pd.DataFrame({"corr": corr, "entry": entry, "exit": exit_}, index=price1.index)
    return df


@signal_cache
def rolling_cointegration_signals(
    price1: PriceSeries,
    price2: PriceSeries,
    cfg: CointegrationConfig,
) -> SignalFrame:
    """
    Generate signals based on rolling cointegration p-values.

    מימוש ישיר עם statsmodels.tsa.stattools.coint על חלונות מתגלגלים.

    Logic
    -----
    - p_value_t = מבחן קו-אינטגרציה על חלון [t-window+1, t].
    - Entry = 1 כאשר p_value < entry_pval (קשר קו-אינטגרטיבי "חזק").
    - Exit  = 1 כאשר p_value > exit_pval.

    Columns
    -------
    p_value : float
    entry   : int (0/1)
    exit    : int (0/1)
    """
    if not cfg.enabled:
        return pd.DataFrame(index=price1.index, columns=["p_value", "entry", "exit"]).fillna(0.0)

    _validate_series(price1, "series1")
    _validate_series(price2, "series2")

    try:
        from statsmodels.tsa.stattools import coint
    except ImportError as exc:
        logger.error("rolling_cointegration_signals requires statsmodels (coint): %s", exc)
        return pd.DataFrame(index=price1.index, columns=["p_value", "entry", "exit"]).fillna(np.nan)

    n = len(price1)
    if n < cfg.window:
        return pd.DataFrame(index=price1.index, columns=["p_value", "entry", "exit"]).fillna(np.nan)

    pvals: List[float] = [np.nan] * n
    for i in range(cfg.window - 1, n):
        s1 = price1.iloc[i - cfg.window + 1 : i + 1]
        s2 = price2.iloc[i - cfg.window + 1 : i + 1]
        try:
            stat, pval, _ = coint(s1, s2)
            pvals[i] = float(pval)
        except Exception as exc:
            logger.debug("coint failed at i=%d: %s", i, exc)
            pvals[i] = np.nan

    p_series = pd.Series(pvals, index=price1.index)
    entry = (p_series < cfg.entry_pval).astype(int)
    exit_ = (p_series > cfg.exit_pval).astype(int)

    df = pd.DataFrame({"p_value": p_series, "entry": entry, "exit": exit_}, index=price1.index)
    return df


@signal_cache
def adf_signals(price: PriceSeries, cfg: ADFConfig) -> SignalFrame:
    """
    Generate signals based on rolling ADF p-values.

    מימוש ישיר עם statsmodels.tsa.stattools.adfuller על חלון מתגלגל.

    Logic
    -----
    - בוחנים ADF על כל חלון [t-window+1, t].
    - p_value_t = p-value של המבחן.
    - Entry = 1 כאשר p_value < entry_pval → spread "stationary" (מתאים ל-MR).
    - Exit  = 1 כאשר p_value > exit_pval.

    Columns
    -------
    p_value : float
    entry   : int (0/1)
    exit    : int (0/1)
    """
    if not cfg.enabled:
        return pd.DataFrame(index=price.index, columns=["p_value", "entry", "exit"]).fillna(0.0)

    _validate_series(price, "price")

    try:
        from statsmodels.tsa.stattools import adfuller
    except ImportError as exc:
        logger.error("adf_signals requires statsmodels (adfuller): %s", exc)
        return pd.DataFrame(index=price.index, columns=["p_value", "entry", "exit"]).fillna(np.nan)

    n = len(price)
    if n < cfg.window:
        return pd.DataFrame(index=price.index, columns=["p_value", "entry", "exit"]).fillna(np.nan)

    pvals: List[float] = [np.nan] * n
    for i in range(cfg.window - 1, n):
        window_series = price.iloc[i - cfg.window + 1 : i + 1].dropna()
        # צריך לפחות כמה נקודות שונות כדי שהמבחן יעבוד
        if window_series.nunique() < 3:
            pvals[i] = np.nan
            continue
        try:
            res = adfuller(window_series, autolag="AIC")
            pvals[i] = float(res[1])  # res[1] הוא p-value
        except Exception as exc:
            logger.debug("adfuller failed at i=%d: %s", i, exc)
            pvals[i] = np.nan

    p = pd.Series(pvals, index=price.index)
    entry = (p < cfg.entry_pval).astype(int)
    exit_ = (p > cfg.exit_pval).astype(int)

    df = pd.DataFrame({"p_value": p, "entry": entry, "exit": exit_}, index=price.index)
    return df


@signal_cache
def spread_signals(
    price1: PriceSeries,
    price2: PriceSeries,
    cfg: SpreadConfig,
) -> SignalFrame:
    """
    Generate signals based on spread Z-score between two legs.

    Logic
    -----
    - spread_t = X_t - hedge_ratio * Y_t
    - z = rolling z-score of spread.
    - Long entry  (+1) כאשר spread נמוך מדי (z < -entry_threshold).
    - Short entry (-1) כאשר spread גבוה מדי (z > +entry_threshold).
    - Exit כאשר |z| < exit_threshold ויש פוזיציה פתוחה.

    Columns
    -------
    spread : float
    zscore : float
    entry  : int (+1/-1/0)
    exit   : int (1/0)
    """
    if not cfg.enabled:
        return pd.DataFrame(index=price1.index, columns=["spread", "zscore", "entry", "exit"]).fillna(0.0)

    _validate_series(price1, "series1")
    _validate_series(price2, "series2")

    from common.feature_engineering import spread as fe_spread, zscore as fe_zscore

    sp = fe_spread(price1, price2, cfg.hedge_ratio)
    zs = fe_zscore(sp, cfg.window)

    zs_prev = zs.shift(1)
    long_entry = ((zs < -cfg.entry_threshold) & (zs_prev >= -cfg.entry_threshold)).astype(int)
    short_entry = ((zs > cfg.entry_threshold) & (zs_prev <= cfg.entry_threshold)).astype(int)
    entry = long_entry - short_entry

    prev_pos = entry.replace(0, np.nan).ffill().shift(1).fillna(0)
    exit_ = ((prev_pos != 0) & (zs.abs() < cfg.exit_threshold)).astype(int)

    df = pd.DataFrame(
        {"spread": sp, "zscore": zs, "entry": entry, "exit": exit_},
        index=price1.index,
    )
    return df

# ============================================================================
# Signal Generator Orchestrator
# ============================================================================

class SignalGenerator:
    """
    SignalGenerator — מנוע סיגנלים מרוכז ברמת קרן גידור.

    אחראי על:
        - ניהול קונפיג לכל משפחת סיגנלים (Zscore/Bollinger/CUSUM/RSI/Corr/ADf/Coint/Spread).
        - הפעלת כל פונקציות הסיגנלים על סדרות המחיר.
        - קריאות ל-hooks לפני/אחרי (למשל alerts, logging מיוחד).
        - יצירת DataFrame מרוכז עם כל הסיגנלים.

    קונבנציית שמות
    --------------
    לכל signal family, העמודות יורצו עם prefix:

        zscore_signals → zscore_zscore, zscore_entry, zscore_exit
        bollinger_signals → bollinger_upper, bollinger_lower, bollinger_entry, bollinger_exit
        rsi_signals → rsi_rsi, rsi_entry, rsi_exit
        ...

    aggregation מתבצע על כל העמודות שמסתיימות ב:
        *_entry, *_exit.

    Callback / Hooks
    ----------------
    - callback(df): נקרא אחרי יצירת כל הסיגנלים (df = כל הסיגנלים).
    - before_hooks: רשימת פונקציות (price, series2) → None, רצות לפני generate().
    - after_hooks: רשימת פונקציות (df) → None, רצות אחרי generate() וגם אחרי aggregate.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        callback: SignalCallback | None = None,
        *,
        before_hooks: List[PreHook] | None = None,
        after_hooks: List[PostHook] | None = None,
    ) -> None:
        self.callback = callback
        self.before_hooks: List[PreHook] = before_hooks or []
        self.after_hooks: List[PostHook] = after_hooks or []

        # נטען את הקונפיגים (אם key חסר, פיידנטיק ישתמש ב-defaults)
        self.zcfg = ZScoreConfig(**config.get("zscore", {}))
        self.bcfg = BollingerConfig(**config.get("bollinger", {}))
        self.ccfg = CUSUMConfig(**config.get("cusum", {}))
        self.rcfg = RSIConfig(**config.get("rsi", {}))
        self.cor_cfg = CorrelationConfig(**config.get("correlation", {}))
        self.coint_cfg = CointegrationConfig(**config.get("rolling_cointegration", {}))
        self.adf_cfg = ADFConfig(**config.get("adf", {}))
        self.spread_cfg = SpreadConfig(**config.get("spread", {}))

    # ----------------------------------------------------------------------
    # Core generation
    # ----------------------------------------------------------------------
    def generate(
        self,
        price: PriceSeries,
        series2: PriceSeries | None = None,
    ) -> SignalFrame:
        """
        Generate all configured signals.

        Parameters
        ----------
        price : pd.Series
            מחיר ה-leg הראשון (למשל X).
        series2 : pd.Series or None, default None
            מחיר ה-leg השני (Y). חובה עבור correlation/cointegration/spread.

        Returns
        -------
        pd.DataFrame
            כל עמודות הסיגנלים, עם prefix לפי סוג הסיגנל.
        """
        _validate_series(price, "price")

        # Hooks לפני – למשל לוגיקה/איסוף דאטה
        for hook in self.before_hooks:
            try:
                hook(price, series2)
            except Exception as exc:
                logger.warning("before_hook %r failed: %s", hook, exc)

        frames: List[pd.DataFrame] = []

        # Z-score
        if self.zcfg.enabled:
            try:
                df_z = zscore_signals(price, self.zcfg).add_prefix("zscore_")
                frames.append(df_z)
            except Exception as exc:
                logger.error("zscore_signals failed: %s", exc)

        # Bollinger
        if self.bcfg.enabled:
            try:
                df_b = bollinger_signals(price, self.bcfg).add_prefix("bollinger_")
                frames.append(df_b)
            except Exception as exc:
                logger.error("bollinger_signals failed: %s", exc)

        # CUSUM
        if self.ccfg.enabled:
            try:
                df_c = cusum_signals(price, self.ccfg).add_prefix("cusum_")
                frames.append(df_c)
            except Exception as exc:
                logger.error("cusum_signals failed: %s", exc)

        # RSI
        if self.rcfg.enabled:
            try:
                df_r = rsi_signals(price, self.rcfg).add_prefix("rsi_")
                frames.append(df_r)
            except Exception as exc:
                logger.error("rsi_signals failed: %s", exc)

        # שני-legים נדרשים לסיגנלים זוגיים
        if series2 is not None:
            _validate_series(series2, "series2")

            # Correlation
            if self.cor_cfg.enabled:
                try:
                    df_corr = correlation_signals(price, series2, self.cor_cfg).add_prefix("corr_")
                    frames.append(df_corr)
                except Exception as exc:
                    logger.error("correlation_signals failed: %s", exc)

            # Rolling Cointegration
            if self.coint_cfg.enabled:
                try:
                    df_coint = rolling_cointegration_signals(price, series2, self.coint_cfg).add_prefix("coint_")
                    frames.append(df_coint)
                except Exception as exc:
                    logger.error("rolling_cointegration_signals failed: %s", exc)

            # Spread Z-score
            if self.spread_cfg.enabled:
                try:
                    df_spread = spread_signals(price, series2, self.spread_cfg).add_prefix("spread_")
                    frames.append(df_spread)
                except Exception as exc:
                    logger.error("spread_signals failed: %s", exc)
        else:
            if any(
                cfg.enabled
                for cfg in (self.cor_cfg, self.coint_cfg, self.spread_cfg)
            ):
                logger.warning(
                    "series2 is None but correlation/cointegration/spread configs are enabled – "
                    "these signals will be skipped."
                )

        # ADF (על price בלבד)
        if self.adf_cfg.enabled:
            try:
                df_adf = adf_signals(price, self.adf_cfg).add_prefix("adf_")
                frames.append(df_adf)
            except Exception as exc:
                logger.error("adf_signals failed: %s", exc)

        if not frames:
            logger.warning("No signals generated (all configs disabled?).")
            df = pd.DataFrame(index=price.index)
        else:
            df = pd.concat(frames, axis=1)
            # entry/exit NaNs → 0, כדי שיהיה קל לעבוד עם זה
            for col in df.columns:
                if col.endswith("entry") or col.endswith("exit"):
                    df[col] = df[col].fillna(0).astype(int)

        # Callback אחרי יצירת כל הסיגנלים
        if self.callback is not None:
            try:
                self.callback(df)
            except Exception as exc:
                logger.warning("callback %r failed: %s", self.callback, exc)

        # Hooks אחרי generate (לפני aggregation)
        for hook in self.after_hooks:
            try:
                hook(df)
            except Exception as exc:
                logger.warning("after_hook %r failed (post-generate): %s", hook, exc)

        return df

    # ----------------------------------------------------------------------
    # Aggregation / Consensus
    # ----------------------------------------------------------------------
    def aggregate_signals(self, df: SignalFrame) -> SignalFrame:
        """
        Aggregate signals into consensus entry/exit.

        Rules
        -----
        - לוקחים רק עמודות שמסתיימות ב- '_entry' ו '_exit'.
        - entry:
            * אוספים כמה סיגנלים נותנים long (+1) וכמה short (-1).
            * threshold = majority (>= floor(n_signals/2)+1).
            * אם long_votes >= threshold → consensus_entry = +1
            * elif short_votes >= threshold → consensus_entry = -1
            * else → 0
        - exit:
            * exit_cols אמורים להיות 0/1.
            * consensus_exit = 1 אם מספר ה-exits >= threshold.

        Returns
        -------
        pd.DataFrame עם העמודות:
            entry : +1/-1/0
            exit  : 1/0
        """
        entry_cols = [c for c in df.columns if c.endswith("entry")]
        exit_cols = [c for c in df.columns if c.endswith("exit")]

        if not entry_cols or not exit_cols:
            logger.warning("aggregate_signals: no entry/exit columns found.")
            return pd.DataFrame(
                {"entry": pd.Series(0, index=df.index), "exit": pd.Series(0, index=df.index)}
            )

        n_signals = len(entry_cols)
        threshold = max(1, n_signals // 2 + 1)

        entry_block = df[entry_cols].fillna(0)
        exit_block = df[exit_cols].fillna(0)

        long_votes = (entry_block > 0).sum(axis=1)
        short_votes = (entry_block < 0).sum(axis=1)

        consensus_entry = np.where(
            long_votes >= threshold,
            1,
            np.where(short_votes >= threshold, -1, 0),
        )

        exit_votes = exit_block.sum(axis=1)
        consensus_exit = (exit_votes >= threshold).astype(int)

        consensus = pd.DataFrame(
            {"entry": consensus_entry, "exit": consensus_exit},
            index=df.index,
        )

        # Hooks אחרי aggregation
        for hook in self.after_hooks:
            try:
                hook(consensus)
            except Exception as exc:
                logger.warning("after_hook %r failed (post-aggregate): %s", hook, exc)

        return consensus

    # ----------------------------------------------------------------------
    # Cache management & default config
    # ----------------------------------------------------------------------
    @staticmethod
    def clear_caches() -> None:
        """Clear all signal caches (wrapper around clear_all_signal_caches)."""
        clear_all_signal_caches()

    @staticmethod
    def default_config() -> Dict[str, Any]:
        """
        Return a default config dict suitable ל-YAML/JSON or UI-editing.

        Example structure
        -----------------
        {
          "zscore": {...},
          "bollinger": {...},
          ...
        }
        """
        return {
            "zscore": ZScoreConfig().dict(),
            "bollinger": BollingerConfig().dict(),
            "cusum": CUSUMConfig().dict(),
            "rsi": RSIConfig().dict(),
            "correlation": CorrelationConfig().dict(),
            "rolling_cointegration": CointegrationConfig().dict(),
            "adf": ADFConfig().dict(),
            "spread": SpreadConfig().dict(),
        }

    @staticmethod
    def to_json_safe_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        הפיכת קונפיג (שיכול להכיל Pydantic models) ל-JSON-safe dict.

        שימושי אם אתה רוצה לשמור config ל־config_signals.json.
        """
        out: Dict[str, Any] = {}
        for key, value in config.items():
            if isinstance(value, BaseModel):
                out[key] = value.dict()
            elif isinstance(value, dict):
                out[key] = {k: (v.dict() if isinstance(v, BaseModel) else v) for k, v in value.items()}
            else:
                out[key] = value
        return make_json_safe(out, default=_json_default)  # type: ignore[arg-type]


# ============================================================================
# Candidate Generator for Optimisation of Signal Parameters
# ============================================================================

def generate_signal_candidates(config: Dict[str, Any], max_samples: int = 200) -> pd.DataFrame:
    """
    Generate candidate signal parameter configurations for optimisation.

    Parameters
    ----------
    config : dict
        configuration dict with "ranges" key, e.g.:

        {
            "zscore_window": [10, 60],
            "zscore_entry_threshold": [1.0, 3.0],
            "zscore_exit_threshold": [0.5, 2.0],
            "rsi_window": [7, 30],
            ...
        }

        כל value יכול להיות:
            * [min, max] → נבנה גריד של עד 5 נקודות.
            * רשימה של ערכים ספציפיים → נשתמש כפי שהוא.
            * scalar → נעטוף כ-singleton list.
    max_samples : int, default 200
        maximum samples to generate if the full grid is too large.

    Returns
    -------
    pd.DataFrame
        Each row is a parameter configuration.
    """
    import itertools
    import random as _rnd

    ranges = config.get("ranges", {})
    grid_params: Dict[str, List[Any]] = {}

    for key, rng in ranges.items():
        # [min, max]
        if isinstance(rng, list) and len(rng) == 2:
            lo, hi = rng
            if isinstance(lo, int) and isinstance(hi, int):
                step = max((hi - lo) // 4, 1)
                grid_params[key] = list(range(lo, hi + 1, step))
            else:
                # floats – linspace 5 נקודות
                grid_params[key] = [round(lo + i * (hi - lo) / 4, 6) for i in range(5)]
        elif isinstance(rng, list):
            grid_params[key] = list(rng)
        else:
            grid_params[key] = [rng]

    combos = list(itertools.product(*grid_params.values()))
    _rnd.shuffle(combos)

    if len(combos) > max_samples:
        combos = combos[:max_samples]

    cols = list(grid_params.keys())
    return pd.DataFrame(combos, columns=cols)


# ============================================================================
# Public API
# ============================================================================

__all__ = [
    # Config models
    "ZScoreConfig",
    "BollingerConfig",
    "CUSUMConfig",
    "RSIConfig",
    "CorrelationConfig",
    "CointegrationConfig",
    "ADFConfig",
    "SpreadConfig",
    # Core signal functions
    "zscore_signals",
    "bollinger_signals",
    "cusum_signals",
    "rsi_signals",
    "correlation_signals",
    "rolling_cointegration_signals",
    "adf_signals",
    "spread_signals",
    # Orchestrator
    "SignalGenerator",
    # Cache helpers
    "clear_all_signal_caches",
    # Optimisation helper
    "generate_signal_candidates",
]
