# -*- coding: utf-8 -*-
"""
core/feature_engineering.py
===========================

ארגז כלים מקצועי ליצירת פיצ'רים מסדרות זמן ברמת קרן גידור:

- Scaling & Normalization: MinMax, Robust, Z-Score.
- Rolling statistics: mean, std, var, skew, kurt, autocorr.
- Momentum & returns: momentum, ROC, log-returns.
- Technical-style indicators: RSI, Bollinger Bands, entropy, Hurst.
- Spread & ratio features לזוגות: spread, volatility-adjusted spread, z-scores.
- Drawdown & clustering: drawdown, volatility clustering, Shannon entropy.
- ACF / PACF features (כולל rolling & plotting).
- Feature selection: correlation-based selection, target correlation.
- Batch helpers: apply_feature_to_df, generate_param_grid, combine_features.

הקובץ הזה *לא* תלוי ב-Streamlit בטאב עצמו – אפשר להוסיף caching ברמת ה-UI.
"""

from __future__ import annotations

import functools
import warnings
from itertools import product
from typing import Any, Callable, Dict, List, Sequence

import numpy as np
import pandas as pd
import pandas.api.types as ptypes


# =============================================================================
# Decorators & Validation
# =============================================================================

def validate_series(func: Callable) -> Callable:
    """
    Decorator: validate that input is a numeric pandas.Series with sorted index.

    - ממיין את האינדקס אם הוא לא מונוטוני.
    - זורק TypeError אם הטיפוס / dtype לא מתאימים.
    """

    @functools.wraps(func)
    def wrapper(series: pd.Series, *args, **kwargs):
        if not isinstance(series, pd.Series):
            raise TypeError(f"{func.__name__}: input must be pandas.Series, got {type(series)}")
        if not ptypes.is_numeric_dtype(series.dtype):
            raise TypeError(f"{func.__name__}: series dtype must be numeric, got {series.dtype}")
        if not series.index.is_monotonic_increasing:
            series = series.sort_index()
        return func(series, *args, **kwargs)

    return wrapper


def require_window(func: Callable) -> Callable:
    """
    Decorator: ensure rolling/window functions return NaNs when series shorter than window.

    אם כמות הערכים הלא-NaN קטנה מגודל החלון – מוחזרת סדרה מלאה NaN + אזהרה.
    """

    @functools.wraps(func)
    def wrapper(series: pd.Series, window: int, *args, **kwargs):
        valid_len = series.dropna().size
        if valid_len < window:
            warnings.warn(
                f"{func.__name__}: series length {valid_len} < window {window}, returning NaNs.",
                RuntimeWarning,
            )
            return pd.Series(np.nan, index=series.index)
        return func(series, window, *args, **kwargs)

    return wrapper


def handle_missing(func: Callable) -> Callable:
    """
    Decorator: handle missing values in series.

    שימוש:
        @handle_missing
        def f(series, ..., fill_method=None):
            ...

    Parameters
    ----------
    fill_method : {"ffill","bfill","drop",None}
        - "ffill" – מילוי קדימה.
        - "bfill" – מילוי אחורה.
        - "drop"  – הסרת NaN לפני החישוב.
        - None    – לא נוגעים ב-NaN.
    """

    @functools.wraps(func)
    def wrapper(series: pd.Series, *args, fill_method=None, **kwargs):
        if fill_method == "ffill":
            series = series.ffill()
        elif fill_method == "bfill":
            series = series.bfill()
        elif fill_method == "drop":
            series = series.dropna()
        return func(series, *args, **kwargs)

    return wrapper


# =============================================================================
# Scaling & Normalization
# =============================================================================

@validate_series
def minmax_scale(series: pd.Series, feature_range: tuple[float, float] = (0.0, 1.0)) -> pd.Series:
    """
    Min-Max normalization: scale series to [min_val, max_val].

    Parameters
    ----------
    series : pd.Series
    feature_range : tuple[float, float], default (0, 1)

    Returns
    -------
    pd.Series
    """
    min_val, max_val = feature_range
    data_min = series.min()
    data_max = series.max()
    denom = data_max - data_min
    if denom == 0:
        # כל הערכים זהים – נחזיר את האמצע של הטווח
        return pd.Series((min_val + max_val) / 2.0, index=series.index)
    scale = (series - data_min) / denom
    return scale * (max_val - min_val) + min_val


@validate_series
def robust_scale(series: pd.Series, quantile_range: tuple[float, float] = (25.0, 75.0)) -> pd.Series:
    """
    Robust scaling using median and IQR.

    Parameters
    ----------
    series : pd.Series
    quantile_range : (lower_percentile, upper_percentile), default (25, 75)

    Returns
    -------
    pd.Series
    """
    lower, upper = quantile_range
    q_low = series.quantile(lower / 100.0)
    q_high = series.quantile(upper / 100.0)
    iqr = q_high - q_low
    if iqr == 0:
        return pd.Series(0.0, index=series.index)
    return (series - series.median()) / iqr


@validate_series
def zscore_full(series: pd.Series, eps: float = 1e-12) -> pd.Series:
    """
    Z-score על כל הסדרה (לא rolling).

    z_t = (x_t - mean(x)) / std(x)
    """
    mean = series.mean()
    std = series.std(ddof=0)
    if std < eps:
        return pd.Series(0.0, index=series.index)
    return (series - mean) / std


# =============================================================================
# Rolling statistics & Z-Scores
# =============================================================================

@handle_missing
@validate_series
@require_window
def rolling_mean(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


@handle_missing
@validate_series
@require_window
def rolling_std(series: pd.Series, window: int, ddof: int = 0) -> pd.Series:
    return series.rolling(window=window, min_periods=window).std(ddof=ddof)


@handle_missing
@validate_series
@require_window
def rolling_var(series: pd.Series, window: int, ddof: int = 0) -> pd.Series:
    return series.rolling(window=window, min_periods=window).var(ddof=ddof)


@handle_missing
@validate_series
@require_window
def rolling_skewness(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).skew()


@handle_missing
@validate_series
@require_window
def rolling_kurtosis(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).kurt()


@handle_missing
@validate_series
@require_window
def zscore(series: pd.Series, window: int, eps: float = 1e-12) -> pd.Series:
    """
    Rolling Z-Score: (x_t - mean_t(window)) / std_t(window).

    Parameters
    ----------
    series : pd.Series
    window : int
    eps : float, default 1e-12
        Guard against division by zero.

    Returns
    -------
    pd.Series
    """
    m = rolling_mean(series, window)
    s = rolling_std(series, window)
    s_safe = s.mask(s.abs() < eps, np.nan)
    return (series - m) / s_safe


# =============================================================================
# Returns, Momentum & RSI
# =============================================================================

@handle_missing
@validate_series
def momentum(series: pd.Series, period: int = 5) -> pd.Series:
    """
    Simple momentum: x_t - x_{t-period}
    """
    return series.diff(period)


@handle_missing
@validate_series
def rate_of_change(series: pd.Series, period: int = 5) -> pd.Series:
    """
    Rate of Change: (x_t / x_{t-period}) - 1
    """
    base = series.shift(period)
    return series.div(base).sub(1.0)


@handle_missing
@validate_series
def log_return(series: pd.Series, period: int = 1) -> pd.Series:
    """
    Log return: ln(x_t / x_{t-period}).
    """
    return np.log(series / series.shift(period))


@handle_missing
@validate_series
def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """
    Relative Strength Index (RSI) over a rolling window.

    Implementation:
        - delta = diff(series)
        - Gains / losses
        - Rolling mean of gains & losses
        - RSI = 100 - 100 / (1 + RS)
    """
    delta = series.diff(1)
    gain = delta.where(delta > 0.0, 0.0)
    loss = -delta.where(delta < 0.0, 0.0)

    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


# =============================================================================
# Spread / Ratio / Rolling correlation & beta
# =============================================================================

@validate_series
def spread(series1: pd.Series, series2: pd.Series, hedge_ratio: float | pd.Series = 1.0) -> pd.Series:
    """
    Spread בין שתי סדרות: S_t = X_t - h_t * Y_t.

    hedge_ratio יכול להיות:
        - float קבוע.
        - pd.Series (למשל beta משתנה בזמן).
    """
    if isinstance(hedge_ratio, pd.Series):
        hedge_ratio = hedge_ratio.reindex(series1.index).fillna(method="ffill")
    return series1 - hedge_ratio * series2


@handle_missing
@validate_series
@require_window
def rolling_correlation(series1: pd.Series, series2: pd.Series, window: int) -> pd.Series:
    """
    Rolling Pearson correlation בין שתי סדרות.
    """
    aligned = series1.align(series2, join="inner")
    s1, s2 = aligned
    return s1.rolling(window=window, min_periods=window).corr(s2)


@handle_missing
@validate_series
@require_window
def rolling_beta(series1: pd.Series, series2: pd.Series, window: int) -> pd.Series:
    """
    Rolling beta של series1 ביחס ל-series2:

        beta_t = Cov(X,Y)_t / Var(Y)_t
    """
    aligned = series1.align(series2, join="inner")
    s1, s2 = aligned
    cov = s1.rolling(window=window, min_periods=window).cov(s2)
    var = s2.rolling(window=window, min_periods=window).var()
    return cov / var.replace(0, np.nan)


@validate_series
def ratio(series1: pd.Series, series2: pd.Series) -> pd.Series:
    """
    Simple ratio of two series: X_t / Y_t.
    """
    aligned = series1.align(series2, join="inner")
    s1, s2 = aligned
    return s1 / s2.replace(0, np.nan)


def volatility_adjusted_spread(
    series1: pd.Series,
    series2: pd.Series,
    hedge_ratio: float | pd.Series,
    window: int,
) -> pd.Series:
    """
    Spread divided by its rolling volatility.

    VA_t = Spread_t / std(Spread)_t(window)
    """
    sp = spread(series1, series2, hedge_ratio)
    vol = rolling_std(sp, window)
    return sp / vol.replace(0, np.nan)


def spread_zscore(
    series1: pd.Series,
    series2: pd.Series,
    hedge_ratio: float | pd.Series,
    window: int,
) -> pd.Series:
    """
    Z-score of the spread feature over a rolling window.
    """
    sp = spread(series1, series2, hedge_ratio)
    return zscore(sp, window)


def rolling_beta_zscore(
    series1: pd.Series,
    series2: pd.Series,
    window_beta: int,
    window_zscore: int,
) -> pd.Series:
    """
    Z-score of rolling beta feature:
        1) Compute rolling beta over window_beta.
        2) Compute rolling Z-score over window_zscore.
    """
    beta_series = rolling_beta(series1, series2, window_beta)
    return zscore(beta_series, window_zscore)


# =============================================================================
# EMA & derived features
# =============================================================================

@validate_series
def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average (EMA)."""
    return series.ewm(span=span, adjust=False).mean()


def ema_momentum(series: pd.Series, period: int = 5, span: int = 5) -> pd.Series:
    """
    EMA of the momentum feature: apply momentum then EMA smoothing.
    """
    return ema(momentum(series, period), span)


def ema_zscore(series: pd.Series, window: int, span: int) -> pd.Series:
    """
    EMA of rolling Z-score: first rolling z-score, then EMA smoothing.
    """
    return ema(zscore(series, window), span)


def time_decay_feature(feature_series: pd.Series, span: int) -> pd.Series:
    """
    Apply exponential time decay smoothing to any feature series.
    """
    return feature_series.ewm(span=span, adjust=False).mean()


# =============================================================================
# CUSUM, Drawdown & Volatility clustering
# =============================================================================

@validate_series
def cusum(series: pd.Series, threshold: float, drift: float = 0.0) -> pd.DataFrame:
    """
    Cumulative Sum (CUSUM) control chart.

    Parameters
    ----------
    series : pd.Series
    threshold : float
        Trigger threshold.
    drift : float, default 0.0
        Drift adjustment.

    Returns
    -------
    pd.DataFrame with columns ['cusum_pos', 'cusum_neg'].
    """
    cumsum_pos = np.zeros(len(series))
    cumsum_neg = np.zeros(len(series))

    values = series.values
    for i in range(1, len(values)):
        change = values[i] - values[i - 1] - drift
        cumsum_pos[i] = max(0.0, cumsum_pos[i - 1] + change)
        cumsum_neg[i] = min(0.0, cumsum_neg[i - 1] + change)
        if cumsum_pos[i] < threshold:
            cumsum_pos[i] = 0.0
        if abs(cumsum_neg[i]) < threshold:
            cumsum_neg[i] = 0.0

    return pd.DataFrame(
        {"cusum_pos": cumsum_pos, "cusum_neg": cumsum_neg},
        index=series.index,
    )


@handle_missing
@validate_series
def drawdown(series: pd.Series) -> pd.Series:
    """
    Relative drawdown from peak: (P_t - max_{<=t} P) / max_{<=t} P.
    """
    peak = series.cummax()
    return (series - peak) / peak.replace(0, np.nan)


@validate_series
def volatility_clustering(series: pd.Series, window: int) -> pd.Series:
    """
    מדד פשוט ל-clustering של תנודתיות:
        - מחשבים תשואות (pct_change).
        - מעלים בריבוע (squared returns).
        - לוקחים ממוצע מתגלגל.

    מחזיר סדרת mean(squared returns) על פני חלון.
    """
    returns = series.pct_change().dropna()
    sq_returns = returns.pow(2)
    return sq_returns.rolling(window=window, min_periods=window).mean()


@validate_series
def shannon_entropy(series: pd.Series, window: int, bins: int = 10) -> pd.Series:
    """
    Shannon entropy על תשואות לוגיות בחלון זז.

    Parameters
    ----------
    series : pd.Series
    window : int
    bins : int, default 10

    Returns
    -------
    pd.Series
    """
    from scipy.stats import entropy

    def _ent(x: pd.Series) -> float:
        vals = x.dropna().values
        if len(vals) == 0:
            return np.nan
        p, _ = np.histogram(vals, bins=bins, density=True)
        p = p[p > 0]
        if p.size == 0:
            return np.nan
        return float(entropy(p))

    log_ret = np.log(series / series.shift(1))
    return log_ret.rolling(window=window, min_periods=window).apply(_ent, raw=False)


@validate_series
def rolling_hurst(series: pd.Series, window: int, max_lag: int = 20) -> pd.Series:
    """
    Rolling Hurst exponent בערכי חלון.

    Parameters
    ----------
    series : pd.Series
    window : int
    max_lag : int, default 20

    Returns
    -------
    pd.Series
    """

    def _hurst_calc(x: np.ndarray) -> float:
        s = pd.Series(x)
        lags = range(2, max_lag)
        tau = [np.std(s.diff(lag).dropna()) for lag in lags]
        if any(t <= 0 for t in tau):
            return np.nan
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return float(poly[0] * 2.0)

    return series.rolling(window=window, min_periods=window).apply(_hurst_calc, raw=True)


# =============================================================================
# Bollinger Bands
# =============================================================================

@validate_series
@require_window
def bollinger_bands(series: pd.Series, window: int, num_std: float = 2.0) -> pd.DataFrame:
    """
    מחושב Bollinger Bands: moving average ו-upper/lower bands בסביבת מכפילי סטיית תקן.

    מחזיר DataFrame עם עמודות:
        ['bb_mean', 'bb_upper', 'bb_lower'].
    """
    mb = series.rolling(window=window, min_periods=window).mean()
    sd = series.rolling(window=window, min_periods=window).std()
    upper = mb + num_std * sd
    lower = mb - num_std * sd
    return pd.DataFrame(
        {"bb_mean": mb, "bb_upper": upper, "bb_lower": lower},
        index=series.index,
    )


def bollinger_bandwidth(series: pd.Series, window: int, num_std: float = 2.0) -> pd.Series:
    """
    Bollinger Bandwidth: (upper - lower) / middle band.
    """
    bands = bollinger_bands(series, window, num_std)
    return (bands["bb_upper"] - bands["bb_lower"]) / bands["bb_mean"].replace(0, np.nan)


def bollinger_percent_b(series: pd.Series, window: int, num_std: float = 2.0) -> pd.Series:
    """
    %B של Bollinger Bands: (series - lower) / (upper - lower).
    """
    bands = bollinger_bands(series, window, num_std)
    denom = (bands["bb_upper"] - bands["bb_lower"]).replace(0, np.nan)
    return (series - bands["bb_lower"]) / denom


# =============================================================================
# Autocorrelation & ACF / PACF features
# =============================================================================

@validate_series
def autocorrelation(series: pd.Series, lag: int) -> float:
    """Autocorrelation at a specific lag."""
    return float(series.autocorr(lag))


@validate_series
def rolling_autocorrelation(series: pd.Series, window: int, lag: int, step: int = 1) -> pd.Series:
    """
    Rolling autocorrelation בסדרות משנה של אורך window (קפיצות של step).
    """
    results: list[float] = []
    n = len(series)
    for start in range(0, n - window + 1, step):
        end = start + window
        win = series.iloc[start:end]
        if len(win.dropna()) > lag:
            val = float(win.autocorr(lag))
        else:
            val = np.nan
        results.append(val)
    idx = series.index[window - 1 :: step]
    return pd.Series(results, index=idx)


@validate_series
def pacf_feature(series: pd.Series, nlags: int) -> pd.Series:
    """Partial Autocorrelation Function values up to nlags."""
    from statsmodels.tsa.stattools import pacf

    vals = pacf(series.dropna(), nlags=nlags)
    return pd.Series(vals, index=range(len(vals)))


@validate_series
def rolling_pacf(series: pd.Series, window: int, nlags: int, step: int = 1) -> pd.DataFrame:
    """
    Rolling PACF: לכל חלון מחשבים PACF עד nlags.

    מחזיר DataFrame:
        index = תאריך סוף החלון
        columns = pacf_0 ... pacf_nlags
    """
    from statsmodels.tsa.stattools import pacf

    records: list[list[float]] = []
    indices: list[pd.Timestamp] = []
    n = len(series)

    for start in range(0, n - window + 1, step):
        end = start + window
        win = series.iloc[start:end]
        if len(win.dropna()) < nlags + 1:
            vals = [np.nan] * (nlags + 1)
        else:
            vals = pacf(win.dropna(), nlags=nlags)
        records.append(list(vals))
        indices.append(series.index[end - 1])

    cols = [f"pacf_{i}" for i in range(nlags + 1)]
    return pd.DataFrame(records, index=indices, columns=cols)


@validate_series
def plot_pacf_feature(series: pd.Series, nlags: int = 20):
    """
    מציג גרף PACF עבור הסדרה עד מספר lags נתון.
    """
    from statsmodels.graphics.tsaplots import plot_pacf
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    plot_pacf(series.dropna(), ax=ax, lags=nlags, method="ywm")
    ax.set_title("Partial Autocorrelation (PACF)")
    plt.tight_layout()
    return fig


@validate_series
def acf_feature(series: pd.Series, nlags: int = 20) -> pd.Series:
    """Autocorrelation Function values up to nlags."""
    from statsmodels.tsa.stattools import acf

    vals = acf(series.dropna(), nlags=nlags, fft=True)
    return pd.Series(vals, index=range(len(vals)))


@validate_series
def rolling_acf(series: pd.Series, window: int, nlags: int, step: int = 1) -> pd.DataFrame:
    """
    Rolling ACF: לכל חלון מחשבים ACF עד nlags.

    מחזיר DataFrame:
        index = תאריך סוף החלון
        columns = acf_0 ... acf_nlags
    """
    from statsmodels.tsa.stattools import acf

    records: list[list[float]] = []
    indices: list[pd.Timestamp] = []
    n = len(series)

    for start in range(0, n - window + 1, step):
        end = start + window
        win = series.iloc[start:end].dropna()
        if len(win) < nlags + 1:
            vals = [np.nan] * (nlags + 1)
        else:
            vals = acf(win, nlags=nlags, fft=True)
        records.append(list(vals))
        indices.append(series.index[end - 1])

    cols = [f"acf_{i}" for i in range(nlags + 1)]
    return pd.DataFrame(records, index=indices, columns=cols)


@validate_series
def plot_acf_feature(series: pd.Series, nlags: int = 20):
    """
    מציג גרף ACF עבור הסדרה עד מספר lags נתון.
    """
    from statsmodels.graphics.tsaplots import plot_acf
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    plot_acf(series.dropna(), ax=ax, lags=nlags)
    ax.set_title("Autocorrelation (ACF)")
    plt.tight_layout()
    return fig


# =============================================================================
# Feature Selection Helpers
# =============================================================================

def select_features_by_correlation(df: pd.DataFrame, threshold: float = 0.9) -> List[str]:
    """
    בחירת תכונות לפי מתאם ביניהן – מסיר עמודות "עודפות" בקורלציה גבוהה.

    Parameters
    ----------
    df : pd.DataFrame
        Features only (בלי target).
    threshold : float, default 0.9
        If |corr| > threshold → אחת מהעמודות תוסר.

    Returns
    -------
    list[str]
        רשימת עמודות שנבחרו (ללא המיותרות).
    """
    corr = df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return [col for col in df.columns if col not in to_drop]


def select_features_by_target_correlation(
    df: pd.DataFrame,
    target: pd.Series,
    threshold: float = 0.1,
) -> List[str]:
    """
    בחירת תכונות לפי הקורלציה שלהן עם ה-target.

    Parameters
    ----------
    df : pd.DataFrame
        Features.
    target : pd.Series
        Target (aligned index).
    threshold : float, default 0.1
        מינימום |corr| כדי להיכנס.

    Returns
    -------
    list[str]
        Features עם |corr| > threshold.
    """
    target = target.reindex(df.index)
    corrs = df.apply(lambda col: col.corr(target))
    return corrs[abs(corrs) > threshold].index.tolist()


# =============================================================================
# Batch-processing / Vectorization Helpers
# =============================================================================

def apply_feature_to_df(df: pd.DataFrame, func: Callable, **kwargs) -> pd.DataFrame:
    """
    מפעיל פונקציית feature על כל עמודה ב-DataFrame באופן וקטורי.

    func(series, **kwargs) → pd.Series
    """
    return pd.concat({col: func(df[col], **kwargs) for col in df.columns}, axis=1)


def generate_param_grid(param_dict: Dict[str, Sequence[Any]]) -> List[Dict[str, Any]]:
    """
    יוצר רשימת פרמטרים (grid) מ-dict שבו כל ערך הוא רשימת אפשרויות.

    Example
    -------
    param_dict = {"window": [20, 60], "span": [10, 30]}
    → [{"window":20,"span":10}, {"window":20,"span":30}, ...]
    """
    keys = list(param_dict.keys())
    values = [param_dict[k] for k in keys]
    return [dict(zip(keys, comb)) for comb in product(*values)]


def combine_features(
    df: pd.DataFrame,
    func: Callable[[pd.Series, pd.Series], pd.Series],
    pairs: List[tuple[str, str]],
) -> pd.DataFrame:
    """
    Applies a binary function across specified feature pairs.

    Parameters
    ----------
    df : pd.DataFrame
    func : callable
        func(col1_series, col2_series) -> pd.Series
    pairs : list of (col1, col2)

    Returns
    -------
    pd.DataFrame
        New columns named f"{col1}_{col2}".
    """
    result: Dict[str, pd.Series] = {}
    for col1, col2 in pairs:
        if col1 in df.columns and col2 in df.columns:
            result[f"{col1}_{col2}"] = func(df[col1], df[col2])
    return pd.DataFrame(result, index=df.index)


# =============================================================================
# Public exports
# =============================================================================

# =============================================================================
# High-level feature builders (לשימוש ישיר בפייפליין / דשבורד)
# =============================================================================

def build_univariate_features(
    series: pd.Series,
    *,
    base_name: str = "x",
    windows: Dict[str, int] | None = None,
) -> pd.DataFrame:
    """
    בניית מטריצת פיצ'רים לסדרה בודדת ברמת קרן גידור.

    Parameters
    ----------
    series : pd.Series
        Price / level series.
    base_name : str, default "x"
        Prefix לשמות העמודות (למשל "x", "y", "spread").
    windows : dict[str, int] or None
        חלונות לחישובים מתגלגלים. אם None, יוגדר ברירת מחדל טובה.

        דוגמה:
            {
                "short": 20,
                "medium": 60,
                "long": 120,
            }

    Returns
    -------
    pd.DataFrame
        Index = כמו series, עמודות = פיצ'רים שונים.
    """
    if windows is None:
        windows = {"short": 20, "medium": 60, "long": 120}

    s = series.astype(float)
    feats: Dict[str, pd.Series] = {}

    # Scaling / returns
    feats[f"{base_name}_logret_1"] = log_return(s, period=1)
    feats[f"{base_name}_logret_5"] = log_return(s, period=5)
    feats[f"{base_name}_ret_1"] = rate_of_change(s, period=1)
    feats[f"{base_name}_ret_5"] = rate_of_change(s, period=5)

    # Momentum
    feats[f"{base_name}_mom_5"] = momentum(s, period=5)
    feats[f"{base_name}_mom_20"] = momentum(s, period=20)

    # RSI
    feats[f"{base_name}_rsi_14"] = rsi(s, window=14)

    # Rolling stats על כמה חלונות
    for name, w in windows.items():
        feats[f"{base_name}_z_{name}"] = zscore(s, w)
        feats[f"{base_name}_vol_{name}"] = rolling_std(s, w)
        feats[f"{base_name}_skew_{name}"] = rolling_skewness(s, w)
        feats[f"{base_name}_kurt_{name}"] = rolling_kurtosis(s, w)

    # Bollinger
    bb = bollinger_bands(s, window=windows["medium"])
    for col in bb.columns:
        feats[f"{base_name}_{col}"] = bb[col]
    feats[f"{base_name}_bb_width"] = bollinger_bandwidth(
        s, window=windows["medium"]
    )
    feats[f"{base_name}_bb_percent_b"] = bollinger_percent_b(
        s, window=windows["medium"]
    )

    # Entropy / Hurst / clustering – פיצ'רים "איטיים" יותר
    feats[f"{base_name}_entropy_{windows['long']}"] = shannon_entropy(
        s, window=windows["long"]
    )
    feats[f"{base_name}_hurst_{windows['long']}"] = rolling_hurst(
        s, window=windows["long"]
    )
    feats[f"{base_name}_vol_cluster_{windows['medium']}"] = volatility_clustering(
        s, window=windows["medium"]
    )

    # Drawdown
    feats[f"{base_name}_drawdown"] = drawdown(s)

    # מחזירים DataFrame אחד עם alignment מלא על האינדקס
    return pd.DataFrame(feats, index=series.index)

def build_pair_feature_matrix(
    x: pd.Series,
    y: pd.Series,
    *,
    hedge_ratio: float | pd.Series = 1.0,
    x_name: str = "x",
    y_name: str = "y",
    spread_name: str = "spread",
    windows: Dict[str, int] | None = None,
) -> pd.DataFrame:
    """
    בניית מטריצת פיצ'רים מלאה לזוג (pair) ברמת קרן גידור.

    Parameters
    ----------
    x, y : pd.Series
        Price series לשני ה-legs של הזוג (reindexed & aligned internally).
    hedge_ratio : float or pd.Series, default 1.0
        יחס גידור (קבוע או משתנה), משמש לחישוב ה-spread.
    x_name, y_name : str, default "x","y"
        Prefixes לעמודות של כל leg.
    spread_name : str, default "spread"
        Prefix לעמודות פיצ'רים של ה-spread.
    windows : dict[str,int] or None
        חלונות עבור rolling metrics. אם None, מוגדר:
            {"short":20, "medium":60, "long":120}.

    Returns
    -------
    pd.DataFrame
        פיצ'רים משולבים לזוג – מתאים לפייפליין ML/Backtest.
    """
    if windows is None:
        windows = {"short": 20, "medium": 60, "long": 120}

    # יישור אינדקסים
    x, y = x.align(y, join="inner")
    x = x.astype(float)
    y = y.astype(float)

    # פיצ'רים לכל leg בנפרד
    fx = build_univariate_features(x, base_name=x_name, windows=windows)
    fy = build_univariate_features(y, base_name=y_name, windows=windows)

    # spread & ratio
    sp = spread(x, y, hedge_ratio)
    fr_spread = build_univariate_features(sp, base_name=spread_name, windows=windows)

    # Spread-specific metrics (Z, vol-adjusted, beta-related)
    feats: Dict[str, pd.Series] = {}
    feats[f"{spread_name}_raw"] = sp
    feats[f"{spread_name}_z_short"] = spread_zscore(x, y, hedge_ratio, windows["short"])
    feats[f"{spread_name}_z_medium"] = spread_zscore(x, y, hedge_ratio, windows["medium"])
    feats[f"{spread_name}_z_long"] = spread_zscore(x, y, hedge_ratio, windows["long"])

    feats[f"{spread_name}_va_short"] = volatility_adjusted_spread(x, y, hedge_ratio, windows["short"])
    feats[f"{spread_name}_va_medium"] = volatility_adjusted_spread(x, y, hedge_ratio, windows["medium"])

    # Rolling correlation & beta בין x ל-y
    feats["xy_corr_short"] = rolling_correlation(x, y, windows["short"])
    feats["xy_corr_medium"] = rolling_correlation(x, y, windows["medium"])
    feats["xy_corr_long"] = rolling_correlation(x, y, windows["long"])

    feats["xy_beta_short"] = rolling_beta(x, y, windows["short"])
    feats["xy_beta_medium"] = rolling_beta(x, y, windows["medium"])
    feats["xy_beta_long"] = rolling_beta(x, y, windows["long"])

    df_spread_extra = pd.DataFrame(feats, index=x.index)

    # Ratio features
    r = ratio(x, y)
    fr_ratio = build_univariate_features(r, base_name="ratio", windows=windows)

    # Combine הכל למטריצה אחת
    features = pd.concat(
        [fx, fy, fr_spread, df_spread_extra, fr_ratio],
        axis=1,
    )

    return features


__all__ = [
    # decorators
    "validate_series",
    "require_window",
    "handle_missing",
    # scaling
    "minmax_scale",
    "robust_scale",
    "zscore_full",
    # rolling stats
    "rolling_mean",
    "rolling_std",
    "rolling_var",
    "rolling_skewness",
    "rolling_kurtosis",
    "zscore",
    # returns & momentum
    "momentum",
    "rate_of_change",
    "log_return",
    "rsi",
    # spread / ratio / beta
    "spread",
    "rolling_correlation",
    "rolling_beta",
    "ratio",
    "volatility_adjusted_spread",
    "spread_zscore",
    "rolling_beta_zscore",
    # EMA
    "ema",
    "ema_momentum",
    "ema_zscore",
    "time_decay_feature",
    # cusum / drawdown / clustering
    "cusum",
    "drawdown",
    "volatility_clustering",
    "shannon_entropy",
    "rolling_hurst",
    # Bollinger
    "bollinger_bands",
    "bollinger_bandwidth",
    "bollinger_percent_b",
    # ACF / PACF
    "autocorrelation",
    "rolling_autocorrelation",
    "pacf_feature",
    "rolling_pacf",
    "plot_pacf_feature",
    "acf_feature",
    "rolling_acf",
    "plot_acf_feature",
    # feature selection
    "select_features_by_correlation",
    "select_features_by_target_correlation",
    # batch helpers
    "apply_feature_to_df",
    "generate_param_grid",
    "combine_features",
    "build_univariate_features",
    "build_pair_feature_matrix",
]

