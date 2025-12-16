# -*- coding: utf-8 -*-
"""
advanced_metrics.py — Advanced statistical metrics
==================================================

Hedge-fund grade toolbox for matrix-series analytics:

* Distance-based dependence (distance covariance / correlation).
* Dynamic Time Warping (DTW) for misaligned series.
* Mahalanobis / distance-based risk diagnostics.
* Kalman filtering & state-space smoothing (1D + multivariate).
* Spectral tools (FFT, band-power, coherence, multitaper, entropy).
* Cointegration / VAR / Granger causality (panel diagnostics).
* Shrinkage covariances & graphical models (Graphical Lasso).
* Network-based factor structure (correlation/partial-corr networks).
* Tail risk, microstructure, liquidity & drawdown analytics.
* Macro regime / news sentiment helpers for overlay / filters.

Design goals
------------
- Clean integration with `common.matrix_helpers` abstractions (SeriesND, NDArray).
- Safe handling of optional dependencies (SciPy, statsmodels, arch, sklearn, mne, etc.).
- Graceful degradation: פונקציה זורקת ImportError *רק* כשבאמת משתמשים בה.
- JSON-safe outputs עבור dashboard / API / report-generation.
- CPU-first with optional GPU acceleration (CuPy) שקוף למעלה.
"""

from __future__ import annotations

from common.json_safe import make_json_safe, json_default as _json_default

# ---------------------------------------------------------------------------
# Typing & standard libs
# ---------------------------------------------------------------------------
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Mapping,
    MutableMapping,
    Optional,
    Sequence as _SequenceAlias,
)

# Typing compatibility for Sequence across Python versions
try:  # pragma: no cover - Py3.9+
    from collections.abc import Sequence
except Exception:  # pragma: no cover - very old environments
    Sequence = _SequenceAlias  # type: ignore[assignment]

import importlib
import logging
import time
from functools import wraps

import numpy as np
import pandas as pd

from numpy.linalg import inv  # used in partial / graphical correlations

# ייבוא תשתית מטריצות מהמערכת שלך
from common.matrix_helpers import (
    ensure_matrix_series,
    NDArray,
    SeriesND,
    apply_matrix_series_parallel,
)

__version__ = "0.7.0"



# ---------------------------------------------------------------------------
# GPU / CPU backend selector
# ---------------------------------------------------------------------------
try:  # pragma: no cover - optional GPU acceleration
    import cupy as xp  # type: ignore[import]

    _GPU_ENABLED: bool = True
    _BACKEND_NAME: str = "cupy"
except Exception:  # CPU-only fallback
    xp = np  # type: ignore[assignment]
    _GPU_ENABLED = False
    _BACKEND_NAME = "numpy"


def get_backend_info() -> Dict[str, Any]:
    """
    Return a small dict describing the numerical backend.

    Useful לדשבורד/דיאגנוסטיקה (למשל תחת טאב System Info).
    """
    return {
        "gpu_enabled": bool(_GPU_ENABLED),
        "backend": _BACKEND_NAME,
        "xp_module": xp.__name__,  # type: ignore[attr-defined]
    }


# ---------------------------------------------------------------------------
# Optional dependencies loader (ללא MissingImports מ-Pylance)
# ---------------------------------------------------------------------------

def _optional_import(mod: str, attr: str | None = None) -> Any:
    """
    Load optional dependency safely by name.

    - No static `from X import Y` → Pylance לא צועק MissingImports.
    - אם attr=None: מחזיר את המודול עצמו.
    - אם attr!=None: מחזיר את ה־attribute (או None אם נכשל).
    """
    try:
        module = importlib.import_module(mod)
        return getattr(module, attr) if attr else module
    except Exception:
        return None


# ---- distance / statistics -------------------------------------------------
fastdtw = _optional_import("fastdtw", "fastdtw")
scipy_dcor = _optional_import("scipy.stats", "distance_correlation")
scipy_mahal = _optional_import("scipy.spatial.distance", "mahalanobis")
_cdist = _optional_import("scipy.spatial.distance", "cdist")

# ---- time-series & state-space ---------------------------------------------
KalmanFilter = _optional_import("pykalman", "KalmanFilter")
coint = _optional_import("statsmodels.tsa.stattools", "coint")
grangercausalitytests = _optional_import(
    "statsmodels.tsa.stattools",
    "grangercausalitytests",
)
coint_johansen = _optional_import(
    "statsmodels.tsa.vector_ar.vecm",
    "coint_johansen",
)
VAR = _optional_import("statsmodels.tsa.api", "VAR")

# ---- GARCH / multivariate volatility ---------------------------------------
_arch_model_for_vol = _optional_import("arch", "arch_model")
arch_model = _arch_model_for_vol  # alias for clarity
ConstantConditionalCorrelation = _optional_import(
    "arch.multivariate",
    "ConstantConditionalCorrelation",
)

# ---- sklearn (PCA, CCA, shrinkage, graphical lasso) ------------------------
PCA = _optional_import("sklearn.decomposition", "PCA")
CCA = _optional_import("sklearn.cross_decomposition", "CCA")
LedoitWolf = _optional_import("sklearn.covariance", "LedoitWolf")
OAS = _optional_import("sklearn.covariance", "OAS")
GraphicalLasso = _optional_import("sklearn.covariance", "GraphicalLasso")

# ---- spectral / neuroscience -----------------------------------------------
psd_array_multitaper = _optional_import(
    "mne.time_frequency",
    "psd_array_multitaper",
)

# ---- signal processing -----------------------------------------------------
coherence = _optional_import("scipy.signal", "coherence")

# ---- networks --------------------------------------------------------------
nx = _optional_import("networkx")


# ---------------------------------------------------------------------------
# Logging & timing utilities
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    logger.addHandler(_handler)

# ברירת מחדל: INFO (במערכת חיה תוכל להוריד ל-WARNING)
logger.setLevel(logging.INFO)


def set_advanced_metrics_log_level(level: int | str) -> None:
    """
    Update log level for this module only.

    Parameters
    ----------
    level : int or str
        כמו ב-logging (e.g., logging.DEBUG, "INFO", "WARNING"...).
    """
    if isinstance(level, str):
        level = logging.getLevelName(level.upper())  # type: ignore[assignment]
    logger.setLevel(level)  # type: ignore[arg-type]


def timeit(fn: Callable) -> Callable:
    """
    Decorator that logs runtime at DEBUG level.

    שימושי מאוד ל-profiler קליל סביב heavy metrics (DCC, VAR, distance matrices).
    """

    @wraps(fn)
    def _wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        logger.debug("%s executed in %.2f ms", fn.__name__, elapsed_ms)
        return out

    return _wrapper


# ---------------------------------------------------------------------------
# JSON-safe helpers (ל-reporting / UI / API)
# ---------------------------------------------------------------------------

def to_json_safe(obj: Any) -> Any:
    """
    Convert nested metrics to a JSON-serialisable structure.

    עטיפה נוחה מעל common.json_safe.make_json_safe כדי שלא תצטרך לזכור
    את הפרמטרים בכל מקום.

    * שומר np.ndarray כ-list.
    * שומר pd.Series/Frame כ-dictים.
    * משאיר float/int/str כמו שהם.
    """
    try:
        return make_json_safe(obj, default=_json_default)  # type: ignore[arg-type]
    except TypeError:
        # תאימות לגרסאות ישנות של make_json_safe שלא מקבלות default
        return make_json_safe(obj)


def require_dependency(name: str, obj: Any) -> None:
    """
    Raise a clear ImportError אם פונקציה דורשת חבילה אופציונלית חסרה.

    שימוש בתוך פונקציות למשל:
        require_dependency("fastdtw", fastdtw)

    כדי לקבל הודעה ברורה וסטנדרטית.
    """
    if obj is None:
        raise ImportError(
            f"{name} is required for this function but is not installed. "
            f"Install it via `pip install {name}` and retry."
        )


# ---------------------------------------------------------------------------
# Public API (שמות הפונקציות שעליהן תבנה המערכת / הדשבורד)
# ---------------------------------------------------------------------------
__all__ = [
    # Backend / helpers
    "get_backend_info",
    "set_advanced_metrics_log_level",
    "to_json_safe",
    "require_dependency",
    # Distance / dependence
    "dynamic_time_warping",
    "distance_correlation",
    "distance_covariance_matrix",
    "distance_correlation_matrix",
    "mahalanobis_distance",
    "rolling_mahalanobis_distance",
    "rolling_distance_correlation",
    # Spectral (FFT & friends)
    "spectral_analysis",
    "rolling_spectral_analysis",
    "cross_spectral_density",
    "spectral_entropy",
    "band_power",
    "rolling_band_power",
    "spectral_coherence",
    "coherence_band_score",
    "multitaper_spectrum",
    # Kalman / state-space
    "kalman_filter",
    "kalman_spread_smoother",
    # Cointegration / VAR / causality
    "cointegration_test",
    "cointegration_panel_summary",
    "granger_causality",
    "granger_causality_matrix",
    "var_diagnostics",
    # Correlation structures & networks
    "partial_correlation_matrix",
    "graphical_lasso_precision",
    "graphical_lasso_partial_corr",
    "correlation_network_centrality",
    "dynamic_conditional_correlation",
    "dynamic_conditional_correlation_summary",
    "canonical_correlation_analysis",
    "shrinkage_covariance",
    "oas_covariance",
    "ridge_shrinkage_covariance",
    "ewma_oas_covariance",
    # Tail / entropy
    "tail_dependence",
    "sample_entropy",
    "hurst_exponent",
    # Microstructure & liquidity
    "order_flow_imbalance",
    "bid_ask_spread_pct",
    "amihud_illiq",
    "intraday_vol_ratio",
    # Risk diagnostics
    "autocorr_lag1",
    "drawdown_half_life",
    "garch_volatility",
    "pca_residual_vol",
    # Market & macro
    "beta_market_dynamic",
    "beta_market_multi",
    "news_sentiment_score",
    "macro_regime_classifier",
]

# ===========================================================================
# Distance-based measures (DTW, distance cov/corr, Mahalanobis)
# ===========================================================================
#
# הנחות מהחלק הראשון:
# - np, pd, xp, _GPU_ENABLED
# - fastdtw, scipy_dcor, scipy_mahal, _cdist
# - ensure_matrix_series, SeriesND, NDArray
# - apply_matrix_series_parallel, rolling_matrix
# - timeit, require_dependency
# ===========================================================================


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_two_columns(mat: NDArray, fn_name: str) -> np.ndarray:
    """
    Ensure matrix has at least two columns; raise a clear error otherwise.
    """
    arr = np.asarray(mat, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{fn_name}: expected 2-D array, got shape {arr.shape}")
    if arr.shape[1] < 2:
        raise ValueError(
            f"{fn_name}: expected at least 2 columns, got {arr.shape[1]}"
        )
    return arr


def _dropna_rows(x: np.ndarray, y: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Drop rows with NaN from x (and y if provided) בצורה מסונכרנת.
    """
    x = np.asarray(x, dtype=float)
    if y is None:
        mask = np.isfinite(x).all(axis=-1) if x.ndim > 1 else np.isfinite(x)
        return x[mask], None
    y = np.asarray(y, dtype=float)
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same length for NaN filtering")
    mask = np.isfinite(x) & np.isfinite(y)
    if x.ndim > 1:
        mask = np.isfinite(x).all(axis=-1) & np.isfinite(y).all(axis=-1)
    return x[mask], y[mask]


def _double_center_distance(x: np.ndarray) -> np.ndarray:
    """
    Build a double-centered distance matrix for vector x.

    משמש גם לדיסטנס-קובאריאנס וגם לדיסטנס-קורלציה.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n == 0:
        return np.zeros((0, 0), dtype=float)
    A = np.abs(x[:, None] - x[None, :])
    A -= A.mean(axis=0, keepdims=True)
    A -= A.mean(axis=1, keepdims=True)
    A += A.mean()
    return A


# ===========================================================================
# Dynamic Time Warping (DTW)
# ===========================================================================

def _dtw_dp(x: np.ndarray, y: np.ndarray) -> float:
    """
    Exact dynamic programming DTW (O(n²)) – fallback כשאין fastdtw.

    NB: בהינתן n גדול (אלפי נקודות), כדאי לשקול מראש שימוש ב-fastdtw.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n, m = x.size, y.size
    if n == 0 or m == 0:
        return float("nan")
    dp = np.full((n + 1, m + 1), np.inf, dtype=float)
    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        xi = x[i - 1]
        for j in range(1, m + 1):
            cost = abs(xi - y[j - 1])
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(dp[n, m])


@timeit
def dynamic_time_warping(
    series: SeriesND,
    *,
    use_fastdtw: bool = True,
    dropna: bool = True,
) -> pd.Series:
    """
    Compute Dynamic Time Warping (DTW) distance between the first two columns
    of each matrix in the series.

    Parameters
    ----------
    series : SeriesND
        Series of matrices (T × k). Only the first two columns are used.
    use_fastdtw : bool, default True
        If True and fastdtw is available, use it (O(n) approx). Otherwise falls
        back to exact dynamic-programming (O(n²)).
    dropna : bool, default True
        If True, rows containing NaN in either series are dropped before DTW.

    Returns
    -------
    pd.Series
        Index = series.index, values = float DTW distance per item.

    Notes
    -----
    * אם אתה רוצה לכפות fastdtw (ולקבל ImportError אם איננו), אפשר לבדוק:
        require_dependency("fastdtw", fastdtw)
      לפני הקריאה לפונקציה.
    """
    series = ensure_matrix_series(series, "dynamic_time_warping")

    def _dtw_single(mat: NDArray) -> float:
        arr = _ensure_two_columns(mat, "dynamic_time_warping")
        x, y = arr[:, 0], arr[:, 1]
        if dropna:
            x, y = _dropna_rows(x, y)
            if x.size == 0 or y.size == 0:
                return float("nan")
        if use_fastdtw and fastdtw is not None:
            try:
                dist, _ = fastdtw(x, y)  # type: ignore[call-arg]
                return float(dist)
            except Exception as exc:
                logger.warning("dynamic_time_warping: fastdtw failed – %s", exc)
                return _dtw_dp(x, y)
        return _dtw_dp(x, y)

    return series.apply(_dtw_single)


def dynamic_time_warping_matrix(
    series: SeriesND,
    *,
    use_fastdtw: bool = True,
    dropna: bool = True,
) -> pd.Series:
    """
    Compute pairwise DTW distance לכל זוג עמודות במטריצה.

    עבור כל מטריצה T×p:
      - אחזיר מטריצה p×p שבה (i,j) = DTW(col_i, col_j)

    Parameters
    ----------
    series : SeriesND
    use_fastdtw : bool, default True
    dropna : bool, default True

    Returns
    -------
    pd.Series
        Index = series.index, values = ndarray (p×p) עם מרחקי DTW.
    """
    series = ensure_matrix_series(series, "dynamic_time_warping_matrix")

    def _dtw_mat(mat: NDArray) -> NDArray:
        arr = np.asarray(mat, dtype=float)
        p = arr.shape[1]
        out = np.zeros((p, p), dtype=float)
        for i in range(p):
            xi = arr[:, i]
            if dropna:
                xi, _ = _dropna_rows(xi)
            for j in range(i, p):
                yj = arr[:, j]
                if dropna:
                    yj, _ = _dropna_rows(yj)
                if xi.size == 0 or yj.size == 0:
                    d = float("nan")
                else:
                    if use_fastdtw and fastdtw is not None:
                        try:
                            d, _ = fastdtw(xi, yj)  # type: ignore[call-arg]
                            d = float(d)
                        except Exception:
                            d = _dtw_dp(xi, yj)
                    else:
                        d = _dtw_dp(xi, yj)
                out[i, j] = out[j, i] = d
        return out

    return series.apply(_dtw_mat)


# ===========================================================================
# Distance Correlation בין שתי סדרות (scalar)
# ===========================================================================

def _distance_correlation_manual(x: np.ndarray, y: np.ndarray) -> float:
    """מימוש מפורש של distance correlation לשני וקטורים."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.size
    if n < 2:
        return float("nan")

    A = _double_center_distance(x)
    B = _double_center_distance(y)

    dcov = np.sqrt((A * B).sum() / (n * n))
    dvarx = np.sqrt((A * A).sum() / (n * n))
    dvary = np.sqrt((B * B).sum() / (n * n))
    if dvarx <= 0 or dvary <= 0:
        return 0.0
    return float(dcov / np.sqrt(dvarx * dvary))


@timeit
def distance_correlation(
    series: SeriesND,
    *,
    dropna: bool = True,
    prefer_scipy: bool = True,
) -> pd.Series:
    """
    Compute distance correlation between the first two columns of each matrix.

    Parameters
    ----------
    series : SeriesND
        Series of matrices (T × k). Only columns 0 and 1 are used.
    dropna : bool, default True
        Drop rows with NaN before computing.
    prefer_scipy : bool, default True
        If True and scipy.stats.distance_correlation is available, use it.
        Otherwise use our manual implementation.

    Returns
    -------
    pd.Series
        Index = series.index, values = distance correlation in [0, 1].
    """
    series = ensure_matrix_series(series, "distance_correlation")

    def _dcor(mat: NDArray) -> float:
        arr = _ensure_two_columns(mat, "distance_correlation")
        x, y = arr[:, 0], arr[:, 1]
        if dropna:
            x, y = _dropna_rows(x, y)
            if x.size < 2:
                return float("nan")

        if prefer_scipy and scipy_dcor is not None:
            try:
                return float(scipy_dcor(x, y))  # type: ignore[misc]
            except Exception as exc:
                logger.warning("distance_correlation: scipy_dcor failed – %s", exc)
                return _distance_correlation_manual(x, y)

        return _distance_correlation_manual(x, y)

    return series.apply(_dcor)


# ===========================================================================
# Distance covariance / correlation matrices (p×p)
# ===========================================================================

@timeit
def distance_covariance_matrix(
    series: SeriesND,
    debiased: bool = False,
    dropna: bool = True,
) -> pd.Series:
    """
    Compute a p×p distance covariance matrix עבור כל T×p מטריצה בסדרה.

    Parameters
    ----------
    series : SeriesND
        Series of 2-D arrays with shape (T, p).
    debiased : bool, default False
        If True, apply the sample bias correction (n/(n-1))**2.
    dropna : bool, default True
        If True, drop rows with NaN in any column לפני החישוב.

    Returns
    -------
    pd.Series
        Index = series.index, value = ndarray (p×p) של distance covariances.
    """
    series = ensure_matrix_series(series, "distance_covariance_matrix")

    def _dcov(mat: NDArray) -> NDArray:
        mat = np.asarray(mat, dtype=float)
        if dropna:
            mask = np.isfinite(mat).all(axis=1)
            mat = mat[mask]
        n, p = mat.shape
        if n < 2 or p < 1:
            return np.full((p, p), np.nan, dtype=float)

        # double-centered distance לכל משתנה
        D = np.empty((p, n, n), dtype=float)
        for k in range(p):
            D[k] = _double_center_distance(mat[:, k])

        # בניית מטריצת covariance על בסיס D
        M = np.zeros((p, p), dtype=float)
        base = float(n * n)
        for i in range(p):
            Ai = D[i]
            for j in range(p):
                cov = float((Ai * D[j]).sum()) / base
                if debiased and n > 1:
                    cov *= (n / (n - 1)) ** 2
                M[i, j] = cov
        return M

    return series.apply(_dcov)


@timeit
def distance_correlation_matrix(
    series: SeriesND,
    debiased: bool = False,
    adjusted: bool = False,
    dropna: bool = True,
) -> pd.Series:
    """
    Compute a p×p distance correlation matrix עבור כל T×p מטריצה בסדרה.

    Parameters
    ----------
    series : SeriesND
        Series of 2-D arrays (T, p).
    debiased : bool, default False
        If True, distance_covariance_matrix משתמש בגרסה המדואבת.
    adjusted : bool, default False
        If True, adjust variances by factor (n/(n-1)).
    dropna : bool, default True
        Forwarded to distance_covariance_matrix.

    Returns
    -------
    pd.Series
        Index = series.index, value = ndarray (p×p) של distance correlations.
    """
    series = ensure_matrix_series(series, "distance_correlation_matrix")
    dcov_series = distance_covariance_matrix(series, debiased=debiased, dropna=dropna)
    sizes = series.apply(lambda mat: int(np.asarray(mat).shape[0]))

    def _dcor_from_cov(idx: Any) -> NDArray:
        cov_mat = np.asarray(dcov_series.loc[idx], dtype=float)
        p = cov_mat.shape[0]
        var = np.diag(cov_mat).astype(float)
        n = sizes.loc[idx]
        if adjusted and n > 1:
            var *= (n / (n - 1))
        C = np.zeros((p, p), dtype=float)
        for i in range(p):
            for j in range(p):
                denom = np.sqrt(var[i] * var[j])
                C[i, j] = cov_mat[i, j] / denom if denom > 0 else 0.0
        return C

    return pd.Series({idx: _dcor_from_cov(idx) for idx in series.index})


# ===========================================================================
# Mahalanobis distance matrices
# ===========================================================================

@timeit
def mahalanobis_distance(
    series: SeriesND,
    *,
    use_gpu: bool | None = None,
    dropna: bool = True,
) -> pd.Series:
    """
    Vectorised Mahalanobis distance matrix לכל מטריצה בסדרה.

    Parameters
    ----------
    series : SeriesND
        Series of (T×p) matrices – מדד מרחק בין השורות (observations).
    use_gpu : bool or None, default None
        * True  – נסה GPU (cupy) אם זמין.
        * False – כפייה על CPU בלבד.
        * None  – GPU אם זמין ואין SciPy.cdist, אחרת CPU.
    dropna : bool, default True
        Drop rows with NaN לפני החישוב.

    Returns
    -------
    pd.Series
        Index = series.index, value = ndarray (T×T) של מרחקי Mahalanobis.

    Notes
    -----
    * אם SciPy זמין (cdist) – משתמשים בו בגרסת CPU לטובת מהירות.
    * אם Σ סינגולרית – מוחזרים NaNs.
    """
    series = ensure_matrix_series(series, "mahalanobis_distance")

    # החלטת backend
    if use_gpu is None:
        _use_gpu = _GPU_ENABLED and _cdist is None
    else:
        _use_gpu = bool(use_gpu) and _GPU_ENABLED

    def _mahal_cpu(mat: NDArray) -> NDArray:
        arr = np.asarray(mat, dtype=float)
        if dropna:
            mask = np.isfinite(arr).all(axis=1)
            arr = arr[mask]
        n = arr.shape[0]
        if n < 2:
            return np.full((n, n), np.nan, dtype=float)

        cov = np.cov(arr, rowvar=False)
        try:
            VI = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            return np.full((n, n), np.nan, dtype=float)

        if _cdist is not None:
            return _cdist(arr, arr, metric="mahalanobis", VI=VI)  # type: ignore[misc]

        out = np.zeros((n, n), dtype=float)
        for i in range(n):
            xi = arr[i]
            for j in range(i, n):
                xj = arr[j]
                if scipy_mahal is not None:
                    d = float(scipy_mahal(xi, xj, VI))  # type: ignore[misc]
                else:
                    diff = xi - xj
                    d = float(np.sqrt(diff @ VI @ diff))
                out[i, j] = out[j, i] = d
        return out

    def _mahal_gpu(mat: NDArray) -> NDArray:
        arr = xp.asarray(mat, dtype=xp.float64)  # type: ignore[attr-defined]
        if dropna:
            mask = xp.isfinite(arr).all(axis=1)  # type: ignore[attr-defined]
            arr = arr[mask]
        n = arr.shape[0]
        if n < 2:
            return xp.full((n, n), xp.nan)  # type: ignore[attr-defined]
        cov = xp.cov(arr, rowvar=False)  # type: ignore[attr-defined]
        try:
            VI = xp.linalg.inv(cov)  # type: ignore[attr-defined]
        except Exception:
            return xp.full((n, n), xp.nan)  # type: ignore[attr-defined]
        diffs = arr[:, None, :] - arr[None, :, :]
        left = xp.einsum("...k,kl->...l", diffs, VI)  # type: ignore[attr-defined]
        dists = xp.sqrt(xp.einsum("...k,...k->...", left, diffs))  # type: ignore[attr-defined]
        return dists

    func = _mahal_gpu if _use_gpu else _mahal_cpu
    return series.apply(func)


def rolling_mahalanobis_distance(
    series: SeriesND,
    window: int,
    *,
    dropna: bool = True,
) -> pd.Series:
    """
    Rolling Mahalanobis distance matrix מבוסס חלון של מטריצות עבר.

    עבור כל אינדקס t:
        1. לוקחים את המטריצות [t-window+1 ... t]
        2. עושים ממוצע על ציר "window" → מטריצה אחת (T×p)
        3. מחשבים עליה Mahalanobis distance בין השורות.

    Parameters
    ----------
    series : SeriesND
    window : int
        מספר האיברים (מטריצות) בכל חלון.
    dropna : bool, default True
        Forwarded ל-mahalanobis_distance.

    Returns
    -------
    pd.Series
        Index = series.index, value = ndarray (T×T) או NaN אם אין מספיק היסטוריה.
    """
    series = ensure_matrix_series(series, "rolling_mahalanobis_distance")
    mats = [np.asarray(m, dtype=float) for m in series]
    idx_list = list(series.index)

    results: list[Any] = []
    for i, _idx in enumerate(idx_list):
        if i + 1 < window:
            results.append(np.nan)
            continue
        tensor = np.stack(mats[i + 1 - window : i + 1], axis=2)  # (T×p×window)
        mat_avg = tensor.mean(axis=2)
        # משתמשים במימוש CPU (יציב/פשוט)
        dists = mahalanobis_distance(
            pd.Series([mat_avg], index=[0]),
            use_gpu=False,
            dropna=dropna,
        ).iloc[0]
        results.append(dists)

    return pd.Series(results, index=series.index)


def rolling_distance_correlation(
    series: SeriesND,
    window: int,
    *,
    dropna: bool = True,
) -> pd.Series:
    """
    Rolling distance correlation בין שתי הסדרות הראשונות.

    עבור כל אינדקס t:
        - לוקחים את המטריצות האחרונות בחלון [t-window+1 ... t]
        - עושים ממוצע על הציר השלישי → מטריצה אחת (T×p)
        - מחשבים distance_correlation על העמודות 0 ו-1.

    Parameters
    ----------
    series : SeriesND
    window : int
    dropna : bool, default True

    Returns
    -------
    pd.Series
        Index = series.index, value = float distance correlation (או NaN).
    """
    series = ensure_matrix_series(series, "rolling_distance_correlation")
    mats = [np.asarray(m, dtype=float) for m in series]
    idx_list = list(series.index)

    results: list[float] = []

    for i, _idx in enumerate(idx_list):
        if i + 1 < window:
            results.append(float("nan"))
            continue
        tensor = np.stack(mats[i + 1 - window : i + 1], axis=2)  # (T×p×window)
        mat_avg = tensor.mean(axis=2)
        try:
            val = distance_correlation(
                pd.Series([mat_avg], index=[0]),
                dropna=dropna,
            ).iloc[0]
            results.append(float(val))
        except Exception:
            results.append(float("nan"))

    return pd.Series(results, index=series.index)
# ===========================================================================
# (סוף החלק הראשון של advanced_metrics.py)
# ===========================================================================

# ===========================================================================
# Spectral analysis (FFT, cross-spectrum, entropy, band-power, coherence)
# Kalman filtering & state-space smoothing
# ===========================================================================


# ---------------------------------------------------------------------------
# Internal spectral helpers
# ---------------------------------------------------------------------------

def _move_time_axis(arr: np.ndarray, axis: int) -> np.ndarray:
    """
    Move time axis to front (axis=0) for consistent FFT operations.
    """
    arr = np.asarray(arr, dtype=float)
    if axis == 0:
        return arr
    return np.moveaxis(arr, axis, 0)


def _safe_normalise_power(power: np.ndarray, axis: int) -> np.ndarray:
    """
    Normalise power along given axis, guarding against division by zero.
    """
    denom = power.sum(axis=axis, keepdims=True)
    denom = np.where(denom == 0, 1.0, denom)
    return power / denom


# ===========================================================================
# Basic FFT spectral analysis
# ===========================================================================

@timeit
def spectral_analysis(
    series: SeriesND,
    axis: int = 0,
    power_only: bool = True,
    normalise: bool = False,
    detrend: bool = False,
) -> pd.Series:
    """
    Compute FFT-based spectrum for each matrix in the series.

    Parameters
    ----------
    series : SeriesND
        Series of 2-D arrays (T × k).
    axis : int, default 0
        Axis along which to apply FFT (typically time axis).
    power_only : bool, default True
        If True, return power spectrum |FFT|**2; otherwise return complex FFT.
    normalise : bool, default False
        If True and power_only=True, normalise power so it sums to 1
        across the FFT axis (useful for spectral entropy).
    detrend : bool, default False
        If True, subtract mean along time axis לפני FFT.

    Returns
    -------
    pd.Series
        Each element is an ndarray with the same shape as input matrix (for
        complex FFT) או עם אותה צורה עבור power.
    """
    series = ensure_matrix_series(series, "spectral_analysis")

    def _fft(mat: NDArray) -> NDArray:
        arr = _move_time_axis(mat, axis)
        if detrend:
            arr = arr - arr.mean(axis=0, keepdims=True)
        spectrum = np.fft.fft(arr, axis=0)
        if not power_only:
            # מחזירים חזרה את ציר הזמן למקומו המקורי
            return np.moveaxis(spectrum, 0, axis)
        power = np.abs(spectrum) ** 2
        if normalise:
            power = _safe_normalise_power(power, axis=0)
        return np.moveaxis(power, 0, axis)

    return apply_matrix_series_parallel(series, _fft)


@timeit
def rolling_spectral_analysis(
    series: SeriesND,
    window: int,
    axis: int = 0,
    power_only: bool = True,
    normalise: bool = False,
    detrend: bool = False,
) -> pd.Series:
    """
    Compute spectral analysis over a rolling window of matrices.

    Uses `rolling_matrix` helper to construct a 3-D tensor (rows, cols, window)
    and applies FFT to the window-averaged slice.

    Parameters
    ----------
    series : SeriesND
        Series of 2-D arrays (T × k).
    window : int
        Number of matrices in each rolling block.
    axis : int, default 0
        Axis along which to apply FFT on the averaged slice.
    power_only : bool, default True
        Return power spectrum |FFT|**2 if True, else complex FFT.
    normalise : bool, default False
        Normalise power spectrum within the FFT axis (if power_only=True).
    detrend : bool, default False
        If True, subtract mean along time axis before FFT in each window.

    Returns
    -------
    pd.Series
        Index = series.index, value = ndarray with FFT result for each window.
    """
    series = ensure_matrix_series(series, "rolling_spectral_analysis")

    def _fft_tensor(tensor: NDArray) -> NDArray:
        # tensor: (rows, cols, window)
        mean_slice = xp.mean(tensor, axis=2)  # type: ignore[attr-defined]
        arr = np.asarray(mean_slice, dtype=float)
        arr = _move_time_axis(arr, axis)
        if detrend:
            arr = arr - arr.mean(axis=0, keepdims=True)
        spec = np.fft.fft(arr, axis=0)
        if not power_only:
            return np.moveaxis(spec, 0, axis)
        power = np.abs(spec) ** 2
        if normalise:
            power = _safe_normalise_power(power, axis=0)
        return np.moveaxis(power, 0, axis)

    return rolling_matrix(series, window, fn=_fft_tensor)


# ===========================================================================
# Cross-spectral density & spectral entropy
# ===========================================================================

@timeit
def cross_spectral_density(
    series: SeriesND,
    axis: int = 0,
    normalise: bool = True,
) -> pd.Series:
    """
    Compute cross-spectral density matrix for each matrix in the series.

    For each matrix X (T × k):

        S(f) = F(X)^H · F(X) / T

    where F(X) is the FFT along the time axis.

    Parameters
    ----------
    series : SeriesND
        Series of 2-D arrays (T × k).
    axis : int, default 0
        Axis interpreted as time for the FFT.
    normalise : bool, default True
        If True, divide by T to get average cross-power.

    Returns
    -------
    pd.Series
        Each element is an ndarray of shape (freqs, k, k) with complex values.
    """
    series = ensure_matrix_series(series, "cross_spectral_density")

    def _csd(mat: NDArray) -> NDArray:
        arr = _move_time_axis(mat, axis)
        T, k = arr.shape
        if T == 0 or k == 0:
            return np.zeros((0, k, k), dtype=complex)
        F = np.fft.fft(arr, axis=0)  # (T × k)
        S = np.empty((T, k, k), dtype=complex)
        scale = float(T) if normalise and T > 0 else 1.0
        for i in range(T):
            v = F[i : i + 1, :]  # shape (1, k)
            S[i] = (np.conjugate(v.T) @ v) / scale
        return S

    return series.apply(_csd)


@timeit
def spectral_entropy(
    series: SeriesND,
    axis: int = 0,
    normalised: bool = True,
    log_base: float = np.e,
) -> pd.Series:
    """
    Compute spectral entropy עבור כל מטריצה (per column), based on power spectrum.

    Steps:
    1. Compute power spectrum for each column via FFT.
    2. Normalise power within the FFT axis.
    3. Compute Shannon entropy of the resulting distribution.

    Parameters
    ----------
    series : SeriesND
        Series of 2-D arrays (T × k).
    axis : int, default 0
        Time axis for FFT.
    normalised : bool, default True
        If True, divide by log(#freqs) to get entropy in [0, 1].
    log_base : float, default np.e
        Base of logarithm (e for nats, 2 for bits).

    Returns
    -------
    pd.Series
        Each element is a 1-D array (k,) of spectral entropy per column.
    """
    series = ensure_matrix_series(series, "spectral_entropy")

    def _entropy(mat: NDArray) -> NDArray:
        arr = _move_time_axis(mat, axis)
        T, k = arr.shape
        if T == 0 or k == 0:
            return np.full(k, np.nan, dtype=float)
        F = np.fft.fft(arr, axis=0)
        power = np.abs(F) ** 2  # (freq × k)
        power = _safe_normalise_power(power, axis=0)
        eps = 1e-12
        log_p = np.log(power + eps) / np.log(log_base)
        H = -(power * log_p).sum(axis=0)  # entropy per column
        if normalised and T > 1:
            H = H / np.log(T) * np.log(log_base)
        return H.astype(float)

    return series.apply(_entropy)


# ===========================================================================
# Band power & rolling band power
# ===========================================================================

@timeit
def band_power(
    series: SeriesND,
    fs: float,
    bands: Mapping[str, tuple[float, float]],
    axis: int = 0,
    relative: bool = False,
) -> pd.DataFrame:
    """
    Compute band-limited power עבור כל band וכל עמודה.

    Typical usage (daily data, for example):

        bands = {
            "low": (0.0, 0.1),
            "mid": (0.1, 0.3),
            "high": (0.3, 0.5),
        }

    Parameters
    ----------
    series : SeriesND
        Series of 2-D arrays (T × k).
    fs : float
        Sampling frequency.
    bands : mapping name -> (f_low, f_high)
        Frequency bands (in same units as fs).
    axis : int, default 0
        Time axis for FFT.
    relative : bool, default False
        If True, divide band power by total power per column.

    Returns
    -------
    pd.DataFrame
        Index = series.index.
        Columns = MultiIndex (band_name, column_idx).
    """
    series = ensure_matrix_series(series, "band_power")

    records: Dict[Any, Dict[tuple[str, int], float]] = {}

    for idx, mat in series.items():
        arr = _move_time_axis(mat, axis)  # (T × k)
        T, k = arr.shape
        if T == 0 or k == 0:
            records[idx] = {}
            continue

        freqs = np.fft.fftfreq(T, d=1.0 / fs)
        F = np.fft.fft(arr, axis=0)
        power = np.abs(F) ** 2  # (freq × k)

        total_power = power.sum(axis=0)
        rec: Dict[tuple[str, int], float] = {}

        for band_name, (f_lo, f_hi) in bands.items():
            mask = (freqs >= f_lo) & (freqs <= f_hi)
            if not mask.any():
                for col in range(k):
                    rec[(band_name, col)] = float("nan")
                continue
            band_power_vals = power[mask].sum(axis=0)
            for col in range(k):
                val = float(band_power_vals[col])
                if relative:
                    denom = float(total_power[col]) if total_power[col] != 0 else np.nan
                    val = val / denom if denom == denom else np.nan  # guard NaN
                rec[(band_name, col)] = val
        records[idx] = rec

    if not records:
        return pd.DataFrame()

    col_keys = sorted({k for rec in records.values() for k in rec.keys()})
    cols = pd.MultiIndex.from_tuples(col_keys, names=["band", "column"])
    data = []
    idx_list = []
    for idx, rec in records.items():
        row = [rec.get(col, float("nan")) for col in col_keys]
        data.append(row)
        idx_list.append(idx)
    return pd.DataFrame(data, index=idx_list, columns=cols)


@timeit
def rolling_band_power(
    series: SeriesND,
    fs: float,
    bands: Mapping[str, tuple[float, float]],
    window: int,
    axis: int = 0,
    relative: bool = False,
) -> pd.DataFrame:
    """
    Rolling variant של band_power:

    עבור כל אינדקס t:
      - לוקחים את המטריצות [t-window+1 ... t]
      - עושים ממוצע ביניהן
      - מחשבים band_power על המטריצה הממוצעת.

    Parameters
    ----------
    series : SeriesND
    fs : float
    bands : mapping band_name -> (f_low, f_high)
    window : int
    axis : int, default 0
    relative : bool, default False

    Returns
    -------
    pd.DataFrame
        Index = series.index, columns = MultiIndex (band, column_idx).
    """
    series = ensure_matrix_series(series, "rolling_band_power")
    mats = [np.asarray(m, dtype=float) for m in series]
    idx_list = list(series.index)

    records: Dict[Any, Dict[tuple[str, int], float]] = {}

    for i, idx in enumerate(idx_list):
        if i + 1 < window:
            continue
        tensor = np.stack(mats[i + 1 - window : i + 1], axis=2)  # (T × k × window)
        avg_mat = tensor.mean(axis=2)
        tmp_series = pd.Series([avg_mat], index=[0])
        tmp_bp = band_power(
            tmp_series,
            fs=fs,
            bands=bands,
            axis=axis,
            relative=relative,
        )
        # tmp_bp index = [0]
        records[idx] = {
            (b, c): float(tmp_bp.loc[0, (b, c)]) for (b, c) in tmp_bp.columns
        }

    if not records:
        return pd.DataFrame(index=idx_list)

    col_keys = sorted({k for rec in records.values() for k in rec.keys()})
    cols = pd.MultiIndex.from_tuples(col_keys, names=["band", "column"])
    data = []
    out_idx = []
    for idx in idx_list:
        rec = records.get(idx, {})
        row = [rec.get(col, float("nan")) for col in col_keys]
        data.append(row)
        out_idx.append(idx)
    return pd.DataFrame(data, index=out_idx, columns=cols)


# ===========================================================================
# Coherence (FFT-based dependency)
# ===========================================================================

@timeit
def spectral_coherence(
    series: SeriesND,
    fs: float = 1.0,
    nperseg: Optional[int] = None,
) -> pd.Series:
    """
    Compute magnitude-squared coherence between the first two columns.

    Parameters
    ----------
    series : SeriesND
        Series of 2-D arrays with at least two columns.
    fs : float, default 1.0
        Sampling frequency.
    nperseg : int or None, default None
        Segment length for the FFT (forwarded to scipy.signal.coherence).

    Returns
    -------
    pd.Series
        Each element is a tuple (f, Cxy) where f are frequencies and
        Cxy is the coherence array.
    """
    require_dependency("scipy.signal", coherence)
    series = ensure_matrix_series(series, "spectral_coherence")

    results: Dict[Any, Any] = {}
    for idx, mat in series.items():
        arr = np.asarray(mat, dtype=float)
        if arr.shape[1] < 2:
            results[idx] = (np.array([]), np.array([]))
            continue
        x = arr[:, 0]
        y = arr[:, 1]
        try:
            f, Cxy = coherence(x, y, fs=fs, nperseg=nperseg or None)  # type: ignore[misc]
        except Exception as exc:
            logger.warning("spectral_coherence failed for %s: %s", idx, exc)
            f, Cxy = np.array([]), np.array([])
        results[idx] = (f, Cxy)

    return pd.Series(results)


@timeit
def coherence_band_score(
    series: SeriesND,
    fs: float,
    band: tuple[float, float],
    nperseg: Optional[int] = None,
    agg: str = "mean",
) -> pd.Series:
    """
    Average coherence בתוך חלון תדרים מסוים בין שתי הסדרות הראשונות.

    Parameters
    ----------
    series : SeriesND
        Series of 2-D arrays with at least two columns.
    fs : float
        Sampling frequency.
    band : (f_low, f_high)
        Frequency band of interest.
    nperseg : int or None
        Segment length for the FFT.
    agg : {"mean","median","max"}, default "mean"
        Aggregation of coherence בתוך ה-band.

    Returns
    -------
    pd.Series
        Average coherence in the band per index (scalar).
    """
    fC = spectral_coherence(series, fs=fs, nperseg=nperseg)
    f_low, f_high = band
    out: Dict[Any, float] = {}

    for idx, val in fC.items():
        f, Cxy = val
        if f.size == 0:
            out[idx] = float("nan")
            continue
        mask = (f >= f_low) & (f <= f_high)
        if not mask.any():
            out[idx] = float("nan")
            continue
        vals = Cxy[mask]
        if agg == "mean":
            out[idx] = float(np.nanmean(vals))
        elif agg == "median":
            out[idx] = float(np.nanmedian(vals))
        elif agg == "max":
            out[idx] = float(np.nanmax(vals))
        else:
            raise ValueError(f"Unknown agg: {agg!r}")
    return pd.Series(out)


# ===========================================================================
# Multitaper spectrum (MNE)
# ===========================================================================

@timeit
def multitaper_spectrum(
    series: SeriesND,
    sfreq: float = 1.0,
    bandwidth: float = 4.0,
    adaptive: bool = True,
) -> pd.Series:
    """
    Estimate power spectral density per matrix using the Multitaper method.

    Requires `mne.time_frequency.psd_array_multitaper`.

    Returns
    -------
    pd.Series
        Each element is (freqs, psd) for the averaged column-wise series.
    """
    require_dependency("mne.time_frequency", psd_array_multitaper)
    series = ensure_matrix_series(series, "multitaper_spectrum")

    def _mt(mat: NDArray):
        data = np.asarray(mat, dtype=float).mean(axis=1)
        psd, freqs = psd_array_multitaper(
            data,
            sfreq=sfreq,
            bandwidth=bandwidth,
            adaptive=adaptive,
            normalization="full",
            verbose=False,
        )
        return freqs, psd

    return series.apply(_mt)


# ===========================================================================
# Kalman filtering (state-space utilities)
# ===========================================================================


@timeit
def kalman_filter(
    series: SeriesND,
    transition_matrices: Any | None = None,
    observation_matrices: Any | None = None,
    transition_covariance: Any | float = 1.0,
    observation_covariance: Any | float = 1.0,
    initial_state_mean: Any | None = None,
    initial_state_covariance: Any | float = 1.0,
    n_dim_state: int | None = None,
    n_dim_obs: int | None = None,
) -> pd.Series:
    """
    Apply a configurable Kalman filter לכל מטריצה בסדרה.

    Each matrix is treated כמטריצת תצפיות (T × k) עבור k סדרות תצפית.

    Parameters
    ----------
    series : SeriesND
        Series of 2-D arrays (T × k).
    transition_matrices, observation_matrices, ...
        If None, reasonable defaults are chosen (identity dynamics
        and direct observation).
    n_dim_state : int or None
        Dimension of the hidden state. If None, defaults to k.
    n_dim_obs : int or None
        Observation dimension. If None, defaults to k.

    Returns
    -------
    pd.Series
        Each element is an ndarray of filtered state means (T × n_dim_state).
    """
    require_dependency("pykalman", KalmanFilter)
    series = ensure_matrix_series(series, "kalman_filter")

    def _build_kf(k_obs: int) -> Any:
        n_state = n_dim_state or k_obs
        n_obs = n_dim_obs or k_obs

        # Default: identity state + direct observation
        F = transition_matrices
        H = observation_matrices
        if F is None:
            F = np.eye(n_state)
        if H is None:
            H = np.zeros((n_obs, n_state))
            H[:, : min(n_obs, n_state)] = np.eye(min(n_obs, n_state))

        Q = transition_covariance
        R = observation_covariance
        if np.isscalar(Q):
            Q = float(Q) * np.eye(n_state)
        if np.isscalar(R):
            R = float(R) * np.eye(n_obs)

        init_mean = initial_state_mean
        if init_mean is None:
            init_mean = np.zeros(n_state)

        init_cov = initial_state_covariance
        if np.isscalar(init_cov):
            init_cov = float(init_cov) * np.eye(n_state)

        return KalmanFilter(
            transition_matrices=F,
            observation_matrices=H,
            transition_covariance=Q,
            observation_covariance=R,
            initial_state_mean=init_mean,
            initial_state_covariance=init_cov,
        )

    cache: Dict[int, Any] = {}

    def _smooth(mat: NDArray) -> NDArray:
        arr = np.asarray(mat, dtype=float)
        T, k = arr.shape
        if k not in cache:
            cache[k] = _build_kf(k)
        kf = cache[k]
        filtered_state_means, _ = kf.smooth(arr)
        return filtered_state_means

    return series.apply(_smooth)


@timeit
def kalman_spread_smoother(
    spread: pd.Series,
    process_var: float = 1e-5,
    obs_var: float = 1e-2,
) -> pd.DataFrame:
    """
    1-D Kalman smoother לספרד בודד (שימוש ישיר בטאבים של pair-analysis).

    Parameters
    ----------
    spread : pd.Series
        Observed spread series.
    process_var : float, default 1e-5
        Variance of state noise.
    obs_var : float, default 1e-2
        Variance of observation noise.

    Returns
    -------
    pd.DataFrame
        Columns: 'filtered', 'smoothed', 'variance'.
    """
    require_dependency("pykalman", KalmanFilter)
    x = spread.astype(float).values.reshape(-1, 1)
    if x.size == 0:
        return pd.DataFrame(
            {"filtered": [], "smoothed": [], "variance": []},
            index=spread.index,
        )

    kf = KalmanFilter(
        transition_matrices=np.array([[1.0]]),
        observation_matrices=np.array([[1.0]]),
        transition_covariance=np.array([[process_var]]),
        observation_covariance=np.array([[obs_var]]),
        initial_state_mean=x[0],
        initial_state_covariance=np.array([[obs_var]]),
    )
    filtered_state_means, filtered_state_cov = kf.filter(x)
    smoothed_state_means, smoothed_state_cov = kf.smooth(x)

    return pd.DataFrame(
        {
            "filtered": filtered_state_means.ravel(),
            "smoothed": smoothed_state_means.ravel(),
            "variance": smoothed_state_cov.reshape(-1),
        },
        index=spread.index,
    )
# ===========================================================================
# Cointegration, causality & VAR diagnostics
# ===========================================================================


@timeit
def cointegration_test(
    series: SeriesND,
    method: str = "engle",
    *,
    trend: str = "c",
    maxlag: int | None = None,
    autolag: str = "aic",
    det_order: int = 0,
    k_ar_diff: int = 1,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Test cointegration between the first two columns or the full matrix.

    Parameters
    ----------
    series : SeriesND
        Series of 2-D arrays.
    method : {"engle", "johansen"}, default "engle"
        * "engle"    – Engle-Granger two-step test on first two columns.
        * "johansen" – Johansen trace test on the full matrix.
    trend : {"c","ct","ctt","nc"}, default "c"
        Trend assumption for Engle-Granger (see statsmodels coint docs).
    maxlag : int or None, default None
        Max lag for Engle-Granger ADF regression (None → let statsmodels choose).
    autolag : {"aic","bic","t-stat",None}, default "aic"
        Lag selection criterion for Engle-Granger.
    det_order : int, default 0
        Deterministic term order for Johansen (0=const, -1=no det. term, וכו').
    k_ar_diff : int, default 1
        Number of lagged differences for Johansen.
    alpha : float, default 0.05
        Significance level for 'is_cointegrated' flag (Engle-Granger).

    Returns
    -------
    pd.DataFrame
        For method="engle":
            columns = {coint_score, p_value, is_cointegrated, alpha}
        For method="johansen":
            columns = {trace_stats, crit_vals_90, crit_vals_95, crit_vals_99,
                       rank_90, rank_95, rank_99}
    """
    series = ensure_matrix_series(series, "cointegration_test")
    results: dict[Any, dict[str, Any]] = {}

    for idx, mat in series.items():
        arr = np.asarray(mat, dtype=float)
        try:
            if method == "engle":
                require_dependency("statsmodels.tsa.stattools.coint", coint)
                if arr.shape[1] < 2:
                    raise ValueError("cointegration_test(engle): need at least 2 columns")
                y0 = arr[:, 0]
                y1 = arr[:, 1]
                score, pvalue, _ = coint(  # type: ignore[misc]
                    y0,
                    y1,
                    trend=trend,
                    maxlag=maxlag,
                    autolag=autolag,
                )
                res: dict[str, Any] = {
                    "coint_score": float(score),
                    "p_value": float(pvalue),
                    "alpha": float(alpha),
                    "is_cointegrated": bool(pvalue < alpha),
                }
            else:
                require_dependency("statsmodels.tsa.vector_ar.vecm.coint_johansen", coint_johansen)
                if arr.shape[1] < 2:
                    raise ValueError("cointegration_test(johansen): need at least 2 columns")
                res_j = coint_johansen(arr, det_order=det_order, k_ar_diff=k_ar_diff)  # type: ignore[misc]
                trace_stats = np.asarray(res_j.lr1, dtype=float)
                crit = np.asarray(res_j.cvt, dtype=float)  # (r, 3): 90,95,99
                rank_90 = int((trace_stats > crit[:, 0]).sum())
                rank_95 = int((trace_stats > crit[:, 1]).sum())
                rank_99 = int((trace_stats > crit[:, 2]).sum())
                res = {
                    "trace_stats": trace_stats.tolist(),
                    "crit_vals_90": crit[:, 0].tolist(),
                    "crit_vals_95": crit[:, 1].tolist(),
                    "crit_vals_99": crit[:, 2].tolist(),
                    "rank_90": rank_90,
                    "rank_95": rank_95,
                    "rank_99": rank_99,
                }
        except Exception as exc:
            res = {"error": str(exc)}
        results[idx] = res

    return pd.DataFrame.from_dict(results, orient="index")


def cointegration_panel_summary(
    series: SeriesND,
    alpha: float = 0.05,
    method: str = "engle",
) -> pd.DataFrame:
    """
    Convenience wrapper עבור cointegration_test – מחזיר טבלת סיכום.

    מתאים במיוחד לפאנל של זוגות (כל אינדקס = זוג):

    * method="engle":
        - עמודות: {coint_score, p_value, is_cointegrated, alpha}
    * method="johansen":
        - rank_90, rank_95, rank_99 + אינדיקציה לטווח 95%.

    Parameters
    ----------
    series : SeriesND
    alpha : float, default 0.05
    method : {"engle", "johansen"}, default "engle"
    """
    raw = cointegration_test(series, method=method, alpha=alpha)

    if method == "engle":
        cols = [c for c in ["coint_score", "p_value", "is_cointegrated"] if c in raw.columns]
        out = raw[cols].copy()
        out["alpha"] = alpha
        return out

    cols = [c for c in ["rank_90", "rank_95", "rank_99"] if c in raw.columns]
    if not cols:
        return raw
    out = raw[cols].copy()
    out["has_cointegration_95"] = raw.get("rank_95", 0).fillna(0) > 0
    return out


# ===========================================================================
# Granger causality
# ===========================================================================

@timeit
def granger_causality(
    series: SeriesND,
    maxlag: int = 5,
    verbose: bool = False,
    alpha: float | None = None,
) -> pd.Series:
    """
    Granger causality tests between the first two columns of each matrix.

    Parameters
    ----------
    series : SeriesND
    maxlag : int, default 5
        Maximum lag to test.
    verbose : bool, default False
        Forwarded to statsmodels.grangercausalitytests.
    alpha : float or None, default None
        If set, מוסיף שדה 'reject' ללג (p_value < alpha).

    Returns
    -------
    pd.Series
        כל איבר הוא dict: lag -> {stat, p_value, reject?} או {"error": ...}.
    """
    require_dependency("statsmodels.tsa.stattools.grangercausalitytests", grangercausalitytests)
    series = ensure_matrix_series(series, "granger_causality")
    results: dict[Any, Any] = {}

    for idx, mat in series.items():
        arr = np.asarray(mat, dtype=float)
        out: dict[Any, Any] = {}
        try:
            if arr.shape[1] < 2:
                raise ValueError("granger_causality: need at least 2 columns")
            # לפי statsmodels: העמודה הראשונה היא ה-dependent
            data = np.column_stack([arr[:, 0], arr[:, 1]])
            tests = grangercausalitytests(data, maxlag=maxlag, verbose=verbose)  # type: ignore[misc]
            for lag, res in tests.items():
                stat, pval = res[0]["ssr_chi2test"][:2]
                item: dict[str, Any] = {
                    "stat": float(stat),
                    "p_value": float(pval),
                }
                if alpha is not None:
                    item["reject"] = bool(pval < alpha)
                out[int(lag)] = item
        except Exception as exc:
            out = {"error": str(exc)}
        results[idx] = out

    return pd.Series(results)


@timeit
def granger_causality_matrix(
    series: SeriesND,
    maxlag: int = 5,
    alpha: float = 0.05,
) -> pd.Series:
    """
    מטריצה כיוונית של Granger לכל זוג עמודות (i → j) בכל מטריצה.

    עבור כל אינדקס בסדרה:
      * מחשבים עבור כל זוג i≠j את המבחן "i גורם ל-j"
      * שומרים את ה-min p-value על פני הלגים
      * מחזירים dict: {(i, j): {"p_value", "reject"}}

    Parameters
    ----------
    series : SeriesND
    maxlag : int, default 5
    alpha : float, default 0.05

    Returns
    -------
    pd.Series
        כל איבר הוא dict מ-(i,j) -> {p_value, reject} או {"error": ...}.
    """
    require_dependency("statsmodels.tsa.stattools.grangercausalitytests", grangercausalitytests)
    series = ensure_matrix_series(series, "granger_causality_matrix")
    out_series: dict[Any, Any] = {}

    for idx, mat in series.items():
        arr = np.asarray(mat, dtype=float)
        k = arr.shape[1]
        res_idx: dict[Any, Any] = {}
        try:
            if k < 2:
                raise ValueError("granger_causality_matrix: need at least 2 columns")

            for i in range(k):
                for j in range(k):
                    if i == j:
                        continue
                    # לפי convention של statsmodels:
                    # data[:, 0] = dependent (j), data[:, 1] = predictor (i)
                    data = np.column_stack([arr[:, j], arr[:, i]])
                    tests = grangercausalitytests(data, maxlag=maxlag, verbose=False)  # type: ignore[misc]
                    pvals = [tests[lag][0]["ssr_chi2test"][1] for lag in tests]
                    min_p = float(min(pvals))
                    res_idx[(i, j)] = {
                        "p_value": min_p,
                        "reject": bool(min_p < alpha),
                    }
        except Exception as exc:
            res_idx = {"error": str(exc)}
        out_series[idx] = res_idx

    return pd.Series(out_series)


# ===========================================================================
# VAR diagnostics
# ===========================================================================

@timeit
def var_diagnostics(
    series: SeriesND,
    maxlags: int = 5,
    ic: str = "aic",
) -> pd.DataFrame:
    """
    Fit a VAR model לכל מטריצה (T×k) ולהחזיר מדדי איכות + יציבות.

    Parameters
    ----------
    series : SeriesND
    maxlags : int, default 5
        Maximum lag order to consider.
    ic : {"aic","bic","hqic"}, default "aic"
        Information criterion לשימוש בבחירת lag אופציונלית.

    Returns
    -------
    pd.DataFrame
        Columns:
            - aic, bic, hqic
            - selected_lag (עפ"י ic)
            - is_stable (כל השורשים < 1)
            - max_root, min_root
            - cond_number (של Σ_hat)
            - error (אם הניסיון נכשל)
    """
    require_dependency("statsmodels.tsa.api.VAR", VAR)
    series = ensure_matrix_series(series, "var_diagnostics")
    results: dict[Any, dict[str, Any]] = {}

    for idx, mat in series.items():
        arr = np.asarray(mat, dtype=float)
        res_idx: dict[str, Any] = {}
        try:
            model = VAR(arr)  # type: ignore[misc]
            res = model.fit(maxlags=maxlags, ic=ic)
            roots = np.asarray(res.roots, dtype=float)
            aic_val = float(res.aic)
            bic_val = float(res.bic)
            hqic_val = float(res.hqic)
            selected_lag = int(res.k_ar)

            is_stable = bool(np.all(np.abs(roots) < 1.0))
            max_root = float(np.abs(roots).max()) if roots.size else np.nan
            min_root = float(np.abs(roots).min()) if roots.size else np.nan

            try:
                sigma_u = np.asarray(res.sigma_u, dtype=float)
                cond = float(np.linalg.cond(sigma_u))
            except Exception:
                cond = float("nan")

            res_idx.update(
                {
                    "aic": aic_val,
                    "bic": bic_val,
                    "hqic": hqic_val,
                    "selected_lag": selected_lag,
                    "is_stable": is_stable,
                    "max_root": max_root,
                    "min_root": min_root,
                    "cond_number": cond,
                }
            )
        except Exception as exc:
            res_idx = {"error": str(exc)}
        results[idx] = res_idx

    return pd.DataFrame.from_dict(results, orient="index")


# ===========================================================================
# Correlation structures (partial, graphical, networks)
# ===========================================================================

@timeit
def partial_correlation_matrix(series: SeriesND) -> pd.Series:
    """
    Compute partial correlation matrix עבור כל מטריצה בסדרה.

    ρ_{ij|rest} = -Σ^{-1}_{ij} / sqrt(Σ^{-1}_{ii} * Σ^{-1}_{jj})
    """
    series = ensure_matrix_series(series, "partial_correlation_matrix")

    def _pcm(mat: NDArray) -> NDArray:
        arr = np.asarray(mat, dtype=float)
        p = arr.shape[1]
        if p < 2:
            return np.full((p, p), np.nan, dtype=float)
        try:
            cov = np.cov(arr, rowvar=False)
            prec = inv(cov)
            d = np.sqrt(np.diag(prec))
            pcm = -prec / np.outer(d, d)
            np.fill_diagonal(pcm, 1.0)
            return pcm
        except Exception:
            return np.full((p, p), np.nan, dtype=float)

    return apply_matrix_series_parallel(series, _pcm)


@timeit
def graphical_lasso_precision(
    series: SeriesND,
    alpha: float = 0.01,
) -> pd.Series:
    """
    Regularised precision matrix (Graphical Lasso) לכל מטריצה.

    דורש sklearn.covariance.GraphicalLasso.

    Parameters
    ----------
    series : SeriesND
    alpha : float, default 0.01
        Regularisation strength.

    Returns
    -------
    pd.Series
        כל איבר הוא precision matrix (p×p).
    """
    require_dependency("sklearn.covariance.GraphicalLasso", GraphicalLasso)
    series = ensure_matrix_series(series, "graphical_lasso_precision")

    def _gl(mat: NDArray) -> NDArray:
        arr = np.asarray(mat, dtype=float)
        model = GraphicalLasso(alpha=alpha, max_iter=200)  # type: ignore[misc]
        model.fit(arr)
        return np.asarray(model.precision_, dtype=float)

    return series.apply(_gl)


@timeit
def graphical_lasso_partial_corr(
    series: SeriesND,
    alpha: float = 0.01,
) -> pd.Series:
    """
    Partial correlation לפי precision מ-Graphical Lasso.

    ρ_{ij|rest} = -Θ_{ij} / sqrt(Θ_{ii} Θ_{jj}), כש-Θ = precision.

    Parameters
    ----------
    series : SeriesND
    alpha : float, default 0.01

    Returns
    -------
    pd.Series
        כל איבר הוא מטריצת partial correlation (p×p).
    """
    prec_series = graphical_lasso_precision(series, alpha=alpha)

    def _to_pcor(prec: NDArray) -> NDArray:
        prec = np.asarray(prec, dtype=float)
        d = np.sqrt(np.diag(prec))
        pcm = -prec / np.outer(d, d)
        np.fill_diagonal(pcm, 1.0)
        return pcm

    return prec_series.apply(_to_pcor)


@timeit
def correlation_network_centrality(
    series: SeriesND,
    *,
    threshold: float = 0.3,
    use_partial: bool = False,
    centrality: str = "degree",
) -> pd.Series:
    """
    מדדי centrality ברשת קורלציות/partial-corr עבור כל מטריצה.

    Parameters
    ----------
    series : SeriesND
    threshold : float, default 0.3
        קשרים עם |corr| < threshold לא יכנסו לרשת.
    use_partial : bool, default False
        אם True – מבוסס על partial correlation (unregularised).
        אם False – מבוסס על Pearson correlation.
    centrality : {"degree","eigenvector","betweenness"}, default "degree"

    Returns
    -------
    pd.Series
        כל איבר הוא pd.Series של centrality per node (column index).
    """
    require_dependency("networkx", nx)
    series = ensure_matrix_series(series, "correlation_network_centrality")

    if use_partial:
        base_mats = partial_correlation_matrix(series)
    else:
        def _corr(mat: NDArray) -> NDArray:
            arr = np.asarray(mat, dtype=float)
            return np.corrcoef(arr, rowvar=False)

        base_mats = apply_matrix_series_parallel(series, _corr)

    results: dict[Any, pd.Series] = {}

    for idx, C in base_mats.items():
        C = np.asarray(C, dtype=float)
        p = C.shape[0]
        if p == 0:
            results[idx] = pd.Series(dtype=float)
            continue

        G = nx.Graph()
        G.add_nodes_from(range(p))
        for i in range(p):
            for j in range(i + 1, p):
                w = float(C[i, j])
                if np.isnan(w) or abs(w) < threshold:
                    continue
                G.add_edge(i, j, weight=w)

        if centrality == "degree":
            cen_dict = nx.degree_centrality(G)
        elif centrality == "eigenvector":
            try:
                cen_dict = nx.eigenvector_centrality_numpy(G, weight="weight")
            except Exception:
                cen_dict = {n: 0.0 for n in G.nodes()}
        elif centrality == "betweenness":
            cen_dict = nx.betweenness_centrality(G, weight="weight", normalized=True)
        else:
            raise ValueError(f"Unknown centrality metric: {centrality!r}")

        cen_series = pd.Series({i: float(cen_dict.get(i, 0.0)) for i in range(p)})
        results[idx] = cen_series

    return pd.Series(results)


# ===========================================================================
# DCC-GARCH & dynamic correlation summary
# ===========================================================================

@timeit
def dynamic_conditional_correlation(
    series: SeriesND,
    p: int = 1,
    q: int = 1,
) -> pd.Series:
    """
    Estimate time-varying correlation matrices using a (C)DCC-GARCH-like model.

    Implementation uses `arch.multivariate.ConstantConditionalCorrelation`
    when available. Returns pd.Series:
        index -> ndarray of shape (T, k, k) with evolving correlation matrices.
    """
    require_dependency("arch.multivariate.ConstantConditionalCorrelation", ConstantConditionalCorrelation)
    series = ensure_matrix_series(series, "dynamic_conditional_correlation")
    results: dict[Any, Any] = {}

    for idx, mat in series.items():
        arr = np.asarray(mat, dtype=float)
        try:
            dcc = ConstantConditionalCorrelation(  # type: ignore[misc]
                mean=None,
                vol="Garch",
                p=p,
                q=q,
            )
            res = dcc.fit(arr, disp="off")
            dyn_corr = res.dynamic_corr
        except Exception as exc:
            logger.warning("dynamic_conditional_correlation failed for %s: %s", idx, exc)
            dyn_corr = None
        results[idx] = dyn_corr

    return pd.Series(results)


@timeit
def dynamic_conditional_correlation_summary(
    series: SeriesND,
    p: int = 1,
    q: int = 1,
) -> pd.DataFrame:
    """
    Summary statistics מה-DCC:

    עבור כל אינדקס:
      * mean_corr      – ממוצע קורלציה על פני זמן וכל הזוגות.
      * median_corr    – חציון.
      * corr_vol       – סטיית תקן של הקורלציה (volatility of correlation).

    Useful לדשבורד סיכון/תנודתיות קשר בין נכסים.
    """
    dyn = dynamic_conditional_correlation(series, p=p, q=q)
    records: dict[Any, dict[str, float]] = {}

    for idx, arr in dyn.items():
        try:
            if arr is None:
                raise ValueError("dynamic_corr is None")
            vals = arr[np.triu_indices(arr.shape[1], k=1)]
            vals = np.asarray(vals, dtype=float)
            records[idx] = {
                "mean_corr": float(np.nanmean(vals)),
                "median_corr": float(np.nanmedian(vals)),
                "corr_vol": float(np.nanstd(vals, ddof=0)),
            }
        except Exception as exc:
            records[idx] = {
                "mean_corr": np.nan,
                "median_corr": np.nan,
                "corr_vol": np.nan,
                "error": str(exc),
            }

    return pd.DataFrame.from_dict(records, orient="index")


# ===========================================================================
# Canonical Correlation Analysis (CCA)
# ===========================================================================

@timeit
def canonical_correlation_analysis(
    series_x: SeriesND,
    series_y: SeriesND,
    n_components: int = 2,
) -> pd.DataFrame:
    """
    Canonical Correlation Analysis בין שני פאנלים של מטריצות.

    Parameters
    ----------
    series_x : SeriesND
        Series of 2-D arrays (T × p_x) עבור dataset X.
    series_y : SeriesND
        Series of 2-D arrays (T × p_y) עבור dataset Y.
    n_components : int, default 2
        Number of canonical components to compute.

    Returns
    -------
    pd.DataFrame
        Index = series indices.
        Columns = corr_1, ..., corr_n (canonical correlations per index).
    """
    require_dependency("sklearn.cross_decomposition.CCA", CCA)

    sx = ensure_matrix_series(series_x, "canonical_correlation_analysis_x")
    sy = ensure_matrix_series(series_y, "canonical_correlation_analysis_y")

    if not sx.index.equals(sy.index):
        raise ValueError(
            "canonical_correlation_analysis: series_x and series_y must have matching indices"
        )

    results: dict[Any, list[float]] = {}
    for idx in sx.index:
        X = np.asarray(sx.loc[idx], dtype=float)
        Y = np.asarray(sy.loc[idx], dtype=float)
        try:
            k = min(n_components, X.shape[1], Y.shape[1])
            if k < 1:
                raise ValueError("Not enough columns for CCA")
            cca = CCA(n_components=k)  # type: ignore[misc]
            X_c, Y_c = cca.fit_transform(X, Y)
            corrs = [float(np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1]) for i in range(k)]
        except Exception:
            corrs = [float("nan")] * n_components
        if len(corrs) < n_components:
            corrs = corrs + [float("nan")] * (n_components - len(corrs))
        results[idx] = corrs

    cols = [f"corr_{i + 1}" for i in range(n_components)]
    return pd.DataFrame.from_dict(results, orient="index", columns=cols)


# ===========================================================================
# Optional heavy dependencies for this block (all optional & guarded)
# ===========================================================================

# sklearn.covariance (LedoitWolf, OAS)
try:
    from sklearn.covariance import LedoitWolf as _LedoitWolf  # type: ignore[import]
    from sklearn.covariance import OAS as _OAS  # type: ignore[import]
except ImportError:
    _LedoitWolf = None  # type: ignore[assignment]
    _OAS = None  # type: ignore[assignment]

# sklearn.decomposition.PCA
try:
    from sklearn.decomposition import PCA as _PCA  # type: ignore[import]
except ImportError:
    _PCA = None  # type: ignore[assignment]

# arch for GARCH
try:
    from arch import arch_model as _arch_model  # type: ignore[import]
except ImportError:
    _arch_model = None  # type: ignore[assignment]

# NLTK VADER sentiment
try:
    import nltk  # type: ignore[import]
    from nltk.sentiment import SentimentIntensityAnalyzer as _SIA  # type: ignore[import]
except ImportError:
    nltk = None  # type: ignore[assignment]
    _SIA = None  # type: ignore[assignment]

# Hidden Markov Models for macro regimes
try:
    from hmmlearn.hmm import GaussianHMM as _GaussianHMM  # type: ignore[import]
except ImportError:
    _GaussianHMM = None  # type: ignore[assignment]


# ===========================================================================
# Shrinkage covariances
# ===========================================================================

@timeit
def shrinkage_covariance(series: SeriesND) -> SeriesND:
    """
    Ledoit–Wolf shrinkage covariance לכל מטריצה בסדרה.

    * אם scikit-learn לא מותקנת – ייזרק ImportError ברור.
    """
    if _LedoitWolf is None:
        raise ImportError(
            "shrinkage_covariance requires scikit-learn (sklearn.covariance.LedoitWolf). "
            "Install via `pip install scikit-learn`."
        )

    series = ensure_matrix_series(series, "shrinkage_covariance")

    def _lw(mat: NDArray) -> NDArray:
        arr = np.asarray(mat, dtype=float)
        model = _LedoitWolf()  # type: ignore[call-arg]
        model.fit(arr)
        return np.asarray(model.covariance_, dtype=float)

    return series.apply(_lw)


@timeit
def oas_covariance(series: SeriesND) -> SeriesND:
    """
    Oracle Approximating Shrinkage (OAS) covariance לכל מטריצה.

    * מותאם לדאטה high-dimensional (p גדול, T קטן).
    """
    if _OAS is None:
        raise ImportError(
            "oas_covariance requires scikit-learn (sklearn.covariance.OAS). "
            "Install via `pip install scikit-learn`."
        )

    series = ensure_matrix_series(series, "oas_covariance")

    def _oas(mat: NDArray) -> NDArray:
        arr = np.asarray(mat, dtype=float)
        model = _OAS(store_precision=False, assume_centered=False)  # type: ignore[call-arg]
        model.fit(arr)
        return np.asarray(model.covariance_, dtype=float)

    return series.apply(_oas)


@timeit
def ridge_shrinkage_covariance(series: SeriesND, alpha: float = 0.1) -> SeriesND:
    """
    Ridge shrinkage covariance: Σ + α I.

    מתאים כ-regularisation פשוט לפני PCA / inversion.
    """
    series = ensure_matrix_series(series, "ridge_shrinkage_covariance")

    def _ridge(mat: NDArray) -> NDArray:
        arr = xp.asarray(mat, dtype=xp.float64)  # type: ignore[attr-defined]
        cov = xp.cov(arr, rowvar=False)  # type: ignore[attr-defined]
        I = xp.eye(cov.shape[0], dtype=cov.dtype)  # type: ignore[attr-defined]
        return cov + alpha * I

    return series.apply(_ridge)

@timeit
def ewma_covariance(series: SeriesND, span: int = 60) -> SeriesND:
    """
    EWMA covariance לכל מטריצה בסדרה.

    נותן מטריצת קווריאנס אחת (p×p) לכל איבר בסדרה, עם משקל אקספוננציאלי על הזמן.
    זה תחליף מקומי ל-ewma_covariance שהייתה פעם ב-matrix_helpers.
    """
    series = ensure_matrix_series(series, "ewma_covariance")

    def _ewma_single(mat: NDArray) -> NDArray:
        arr = np.asarray(mat, dtype=float)
        if arr.ndim != 2:
            raise ValueError(f"ewma_covariance: expected 2-D array, got shape {arr.shape}")
        n, p = arr.shape
        if n == 0:
            return np.full((p, p), np.nan, dtype=float)

        # משקלים אקספוננציאליים (EWMA סטנדרטי, כמו ב-pandas ewm)
        alpha = 2.0 / (span + 1.0)
        # משקל ליום t (0=ישן ביותר, n-1=חדש ביותר)
        w = (1.0 - alpha) ** np.arange(n)
        w = w[::-1]  # יותר משקל לימים האחרונים
        w = w / w.sum()

        # ממוצע משוקלל
        mu = (w[:, None] * arr).sum(axis=0)
        X = arr - mu

        # Σ = X^T diag(w) X
        W = w[:, None]
        cov = (X * W).T @ X
        return cov

    return series.apply(_ewma_single)

@timeit
def ewma_oas_covariance(series: SeriesND, span: int = 60) -> SeriesND:
    """
    EWMA covariance ואז OAS shrinkage – שילוב של time-decay + small-sample correction.
    """
    if _OAS is None:
        raise ImportError(
            "ewma_oas_covariance requires scikit-learn (sklearn.covariance.OAS). "
            "Install via `pip install scikit-learn`."
        )

    ewma = ewma_covariance(series, span)  # type: ignore[misc]
    return oas_covariance(ewma)


# ===========================================================================
# Tail dependence & entropy
# ===========================================================================

@timeit
def tail_dependence(
    df: pd.DataFrame,
    u: float = 0.95,
    cols=None,
) -> pd.Series:
    """
    Estimate upper and lower tail dependence coefficients between two columns.

    For columns [X, Y], λ_U ≈ P(Y > F_Y^{-1}(u) | X > F_X^{-1}(u)),
    approximated at quantile level *u*.
    """
    if df.shape[1] < 2:
        return pd.Series({"tail_dep_upper": np.nan, "tail_dep_lower": np.nan})

    if cols is None:
        c0, c1 = 0, 1
    else:
        if len(cols) != 2:
            raise ValueError("tail_dependence: cols must be a sequence of length 2")
        c0, c1 = cols

    x = df.iloc[:, c0].dropna()
    y = df.iloc[:, c1].dropna()
    joined = pd.concat([x, y], axis=1, join="inner").dropna()
    if joined.empty:
        return pd.Series({"tail_dep_upper": np.nan, "tail_dep_lower": np.nan})

    x = joined.iloc[:, 0]
    y = joined.iloc[:, 1]

    xu = x.quantile(u)
    yu = y.quantile(u)
    xl = x.quantile(1 - u)
    yl = y.quantile(1 - u)

    mask_u = x > xu
    mask_l = x < xl

    upper = (
        ((mask_u & (y > yu)).sum() / mask_u.sum())
        if mask_u.sum() > 0
        else np.nan
    )
    lower = (
        ((mask_l & (y < yl)).sum() / mask_l.sum())
        if mask_l.sum() > 0
        else np.nan
    )

    return pd.Series(
        {
            "tail_dep_upper": float(upper),
            "tail_dep_lower": float(lower),
        },
        index=["tail_dep_upper", "tail_dep_lower"],
    )


@timeit
def sample_entropy(
    series: SeriesND,
    m: int = 2,
    r: float = 0.2,
    average_across_columns: bool = True,
) -> pd.Series:
    """
    Estimate sample entropy עבור כל מטריצה בסדרה.
    """
    from math import log

    series = ensure_matrix_series(series, "sample_entropy")

    def _sampen_1d(ts: np.ndarray) -> float:
        ts = np.asarray(ts, dtype=float)
        N = ts.size
        if N <= m + 1:
            return float("nan")
        tol = r * np.std(ts)
        if tol == 0:
            return float("nan")

        def _phi(m_dim: int) -> int:
            count = 0
            for i in range(N - m_dim):
                template = ts[i : i + m_dim]
                for j in range(i + 1, N - m_dim + 1):
                    if np.max(np.abs(template - ts[j : j + m_dim])) <= tol:
                        count += 1
            return count

        B = _phi(m)
        A = _phi(m + 1)
        if B <= 0 or A <= 0:
            return float("nan")
        return float(-log(A / B))

    def _sampen_mat(mat: NDArray) -> float:
        arr = np.asarray(mat, dtype=float)
        if arr.ndim != 2:
            return float("nan")
        if average_across_columns:
            ts = arr.mean(axis=1)
        else:
            ts = arr[:, 0]
        return _sampen_1d(ts)

    return apply_matrix_series_parallel(series, _sampen_mat)


@timeit
def hurst_exponent(series: SeriesND) -> pd.Series:
    """
    Estimate Hurst exponent (R/S method) עבור כל מטריצה, על ממוצע העמודות.

    H > 0.5 → persistence / trending.
    H < 0.5 → mean-reversion.
    H ≈ 0.5 → random walk-like.
    """
    series = ensure_matrix_series(series, "hurst_exponent")

    def _hurst(mat: NDArray) -> float:
        ts = xp.mean(xp.asarray(mat, dtype=xp.float64), axis=1)  # type: ignore[attr-defined]
        N = ts.shape[0]
        if N < 2:
            return float("nan")
        Y = xp.cumsum(ts - ts.mean())  # type: ignore[attr-defined]
        R = float(Y.max() - Y.min())
        S = float(ts.std())
        if S <= 0:
            return float("nan")
        return float(np.log(R / S) / np.log(N))

    return series.apply(_hurst)


# ===========================================================================
# Microstructure & liquidity metrics
# ===========================================================================

@timeit
def order_flow_imbalance(
    df_l1: pd.DataFrame,
    window: int = 2,
) -> pd.Series:
    """
    Order Flow Imbalance (OFI): absolute sum of net order sizes over a window,
    normalised by mid-volume.

    Expects columns:
        - 'bid_size'
        - 'ask_size'
    """
    required_cols = {"bid_size", "ask_size"}
    if not required_cols.issubset(df_l1.columns):
        raise ValueError(
            f"order_flow_imbalance: expected columns {required_cols}, "
            f"got {set(df_l1.columns)}"
        )

    bids = df_l1["bid_size"].astype(float)
    asks = df_l1["ask_size"].astype(float)

    delta = (asks - bids).rolling(window).sum().abs()
    denom = (asks + bids) / 2.0
    denom = denom.replace(0, np.nan)
    return delta / denom


@timeit
def bid_ask_spread_pct(df_l1: pd.DataFrame) -> pd.Series:
    """
    Percentage bid–ask spread relative to mid-price.

    Expects columns:
        - 'bid'
        - 'ask'
    """
    required_cols = {"bid", "ask"}
    if not required_cols.issubset(df_l1.columns):
        raise ValueError(
            f"bid_ask_spread_pct: expected columns {required_cols}, "
            f"got {set(df_l1.columns)}"
        )

    bid = df_l1["bid"].astype(float)
    ask = df_l1["ask"].astype(float)
    mid = (bid + ask) / 2.0
    mid = mid.replace(0, np.nan)
    return (ask - bid) / mid


@timeit
def amihud_illiq(
    close: pd.Series,
    volume: pd.Series,
    window: int = 20,
) -> pd.Series:
    """
    Compute Amihud (2002) illiquidity over a rolling window.

    Amihud illiquidity ≈ |ΔP| / Volume (scaled here by previous close).
    """
    close = close.astype(float)
    volume = volume.astype(float)

    ret = close.diff().abs() / close.shift(1)
    mean_ret = ret.rolling(window, min_periods=1).mean()
    mean_vol = volume.rolling(window, min_periods=1).mean().replace(0, np.nan)
    return mean_ret / mean_vol


@timeit
def intraday_vol_ratio(
    min_ret: pd.Series,
    daily_ret: pd.Series,
    window: int = 20,
    clip_min: float = 0.2,
    clip_max: float = 3.0,
) -> pd.Series:
    """
    Ratio σ_intra / σ_daily over a rolling window.
    """
    min_ret = min_ret.astype(float)
    daily_ret = daily_ret.astype(float)

    sigma_intra = min_ret.rolling(window).std(ddof=0)
    sigma_daily = daily_ret.rolling(window).std(ddof=0).replace(0, np.nan)
    ratio = sigma_intra / sigma_daily
    return ratio.clip(clip_min, clip_max)


# ===========================================================================
# Risk & statistical diagnostics
# ===========================================================================

@timeit
def autocorr_lag1(series: pd.Series, window: int = 60) -> pd.Series:
    """
    Lag-1 autocorrelation computed over a rolling window.

    Clipped to [-1, +1] so extreme NaNs do not propagate.

    Parameters
    ----------
    series : pd.Series
        Time series (returns, spreads וכדומה).
    window : int, default 60
        Rolling window length.

    Returns
    -------
    pd.Series
        Lag-1 autocorrelation per timestamp.
    """
    series = series.astype(float)

    def _ac(x: pd.Series) -> float:
        try:
            return float(x.autocorr(lag=1))
        except Exception:
            return float("nan")

    return series.rolling(window).apply(_ac, raw=False).clip(-1.0, 1.0)


@timeit
def drawdown_half_life(equity: pd.Series, max_lookback: int = 60) -> pd.Series:
    """
    Estimate the half-life (bars) of drawdowns in an equity curve.

    For each date t:
        - מחשבים drawdown_t = 1 - equity_t / max_{<=t}(equity)
        - מריצים אחורה עד שה-drawdown ירד ל-50% מהערך הנוכחי
        - מחזירים את מספר הברים שנדרשו או NaN אם לא חזר.

    Parameters
    ----------
    equity : pd.Series
        Equity curve (PnL מצטבר / NAV).
    max_lookback : int, default 60
        Maximum bars to look back when searching for half-life.

    Returns
    -------
    pd.Series
        Half-life in bars per timestamp (capped by max_lookback).
    """
    equity = equity.astype(float)
    dd = 1.0 - equity / equity.cummax()

    def _half_life(vec: pd.Series) -> float:
        d = float(vec.iloc[-1])
        if not np.isfinite(d) or d <= 0:
            return float("nan")
        target = d / 2.0
        cnt = 0
        for v in reversed(vec.iloc[:-1]):
            cnt += 1
            if cnt > max_lookback:
                break
            if v <= target:
                return float(cnt)
        return float("nan")

    return dd.expanding().apply(_half_life, raw=False).clip(1, max_lookback)


# ===========================================================================
# GARCH Conditional Volatility (with EWMA fallback)
# ===========================================================================

@timeit
def garch_volatility(
    returns,
    p: int = 1,
    q: int = 1,
    span_ewma: int = 20,
) -> pd.Series:
    """
    Estimate conditional volatility with a GARCH(p,q) model, preserving length.

    Parameters
    ----------
    returns : array-like or pd.Series
        1-D return series (decimal, not percent).
    p, q : int, default 1
        GARCH(p, q) orders.
    span_ewma : int, default 20
        EWMA span for fallback when `arch` is unavailable or fails.

    Returns
    -------
    pd.Series
        Conditional volatility series, same index/length as input.
    """
    ret_ser = pd.Series(returns, copy=False).astype(float)

    if _arch_model is not None:
        try:
            am = _arch_model(ret_ser * 100.0, vol="Garch", p=p, q=q, rescale=False)  # type: ignore[call-arg]
            res = am.fit(disp="off")
            vol = res.conditional_volatility / 100.0
            vol.index = ret_ser.index
            return vol.reindex_like(ret_ser)
        except Exception as exc:
            logger.warning("garch_volatility: arch_model failed – %s", exc)

    # EWMA fallback ensures same length
    return ret_ser.ewm(span=span_ewma, adjust=False).std()


# ===========================================================================
# PCA residual volatility
# ===========================================================================

@timeit
def pca_residual_vol(
    df: pd.DataFrame,
    n_components: int = 3,
    window: int = 60,
    min_obs: int = 10,
) -> pd.Series:
    """
    Residual volatility after removing n_components PCA factors.

    לכל חלון מתגלגל:
        1. מריצים PCA על X (T×k).
        2. משחזרים X_hat מהקומפוננטות הראשיות.
        3. מחשבים את סטיית התקן של השאריות (X - X_hat).

    Parameters
    ----------
    df : pd.DataFrame
        Panel of time series (rows=time, columns=assets/factors).
    n_components : int, default 3
        Max number of principal components to remove.
    window : int, default 60
        Rolling window length (bars).
    min_obs : int, default 10
        Minimum observations required in a window to compute PCA.

    Returns
    -------
    pd.Series
        Residual volatility per timestamp (clipped to [0.01, 1.0]).
    """
    if _PCA is None:
        logger.warning("pca_residual_vol: sklearn.decomposition.PCA not available")
        return pd.Series(np.nan, index=df.index)

    df = df.astype(float)
    pca_cache: dict[int, Any] = {}

    def _window_resid(x: pd.DataFrame) -> float:
        X = x.values.astype(float)
        T, k = X.shape
        if T < min_obs or k < 2:
            return float("nan")
        k_eff = min(n_components, k - 1)
        if k_eff not in pca_cache:
            pca_cache[k_eff] = _PCA(n_components=k_eff)  # type: ignore[call-arg]
        pca = pca_cache[k_eff]
        try:
            pca.fit(X)
            X_hat = pca.inverse_transform(pca.transform(X))
            resid = X - X_hat
            return float(np.std(resid, ddof=0))
        except Exception as exc:
            logger.warning("pca_residual_vol: PCA failed – %s", exc)
            return float("nan")

    return df.rolling(window).apply(_window_resid, raw=False).clip(0.01, 1.0)


# ===========================================================================
# Market betas
# ===========================================================================

@timeit
def beta_market_dynamic(
    asset_ret: pd.Series,
    market_ret: pd.Series,
    window: int = 60,
) -> pd.Series:
    """
    Rolling beta of asset_ret against market_ret over a window.
    """
    asset_ret = asset_ret.astype(float)
    market_ret = market_ret.astype(float)

    num = asset_ret.rolling(window).cov(market_ret)
    den = market_ret.rolling(window).var(ddof=1).replace(0, np.nan)
    return num / den


@timeit
def beta_market_multi(
    asset_ret: pd.Series,
    market_ret: pd.Series,
    windows=None,
) -> pd.DataFrame:
    """
    Compute rolling beta for multiple window lengths in one call.

    Parameters
    ----------
    asset_ret : pd.Series
    market_ret : pd.Series
    windows : iterable of int or None
        If None → uses (20, 60, 120).

    Returns
    -------
    pd.DataFrame
        Columns = ['beta_w{w}'] for each window.
    """
    if windows is None:
        windows = (20, 60, 120)

    dfs: dict[str, pd.Series] = {}
    for w in windows:
        dfs[f"beta_w{w}"] = beta_market_dynamic(asset_ret, market_ret, window=int(w))
    return pd.DataFrame(dfs)


# ===========================================================================
# News sentiment (headline-level)
# ===========================================================================

@timeit
def news_sentiment_score(
    headlines: pd.Series,
    cache=None,
) -> pd.Series:
    """
    Compute headline sentiment in [-1, 1] using VADER (fallback neutral).

    Design:
    -------
    * אם NLTK+VADER זמינים → מחשבים ציון 'compound'.
    * אם חסר משהו → מחזירים 0.0 (ניטרלי) לכל הכותרות.
    """
    if cache is None:
        cache = {}

    # אם אין מודול / אנלייזר – מחזירים ניטרלי
    if nltk is None or _SIA is None:
        logger.warning("news_sentiment_score: NLTK/VADER unavailable – returning neutral=0.0")
        return pd.Series(0.0, index=headlines.index)

    sia = cache.get("sia")
    if sia is None:
        try:
            # לוודא שה-lexicon קיים
            try:
                nltk.data.find("sentiment/vader_lexicon.zip")  # type: ignore[attr-defined]
            except LookupError:
                nltk.download("vader_lexicon", quiet=True)  # type: ignore[attr-defined]
            sia = _SIA()
            cache["sia"] = sia
        except Exception as exc:
            logger.warning("news_sentiment_score: failed to init VADER – %s", exc)
            return pd.Series(0.0, index=headlines.index)

    scores = headlines.fillna("").astype(str).apply(
        lambda x: sia.polarity_scores(x)["compound"],  # type: ignore[call-arg]
    )
    return scores.clip(-1, 1)


# ===========================================================================
# Macro regime classifier
# ===========================================================================

@timeit
def macro_regime_classifier(
    df_macro: pd.DataFrame,
    model=None,
) -> pd.Series:
    """
    Classify macro regime per row into structured labels.

    Priority:
    1. אם model לא None ו-callable – ננסה להשתמש בו.
    2. אם hmmlearn זמינה – HMM דו-מצבי על gdp_nowcast & cpi_surprise.
    3. אחרת – heuristic rule-engine פשוט.

    Expected columns (minimum):
        gdp_nowcast, cpi_surprise, unemployment
    Optional:
        ism_pmi, vix, credit_spread וכו'.

    Output labels:
        {"growth", "inflation", "stagflation", "recession", "neutral"}
    """
    df_macro = df_macro.copy()

    # 1) מודל חיצוני של המשתמש (למשל CatBoost/LightGBM וכו')
    if model is not None and callable(model):
        try:
            preds = model(df_macro)
            if isinstance(preds, pd.Series):
                return preds
            return pd.Series(preds, index=df_macro.index)
        except Exception as exc:
            logger.warning("macro_regime_classifier: custom model failed – %s", exc)

    # 2) Hidden Markov – אם hmmlearn זמינה ויש עמודות מתאימות
    if _GaussianHMM is not None and {
        "gdp_nowcast",
        "cpi_surprise",
    }.issubset(df_macro.columns):
        try:
            features = (
                df_macro[["gdp_nowcast", "cpi_surprise"]]
                .astype(float)
                .fillna(0.0)
                .values
            )
            if features.shape[0] >= 10:
                hmm = _GaussianHMM(n_components=2, covariance_type="full", n_iter=100)  # type: ignore[call-arg]
                hmm.fit(features)
                hidden_states = hmm.predict(features)
                means = hmm.means_[:, 0]  # רכיב GDP
                hi_state = int(np.argmax(means))
                lo_state = int(np.argmin(means))
                state_map = {hi_state: "growth", lo_state: "recession"}
                labels = [state_map.get(int(s), "neutral") for s in hidden_states]
                return pd.Series(labels, index=df_macro.index)
        except Exception as exc:
            logger.warning("macro_regime_classifier: HMM path failed – %s", exc)

    # 3) Heuristic rule-engine
    def _classify(row: pd.Series) -> str:
        gdp = float(row.get("gdp_nowcast", 0.0))
        cpi = float(row.get("cpi_surprise", 0.0))
        unemp = float(row.get("unemployment", 0.0))
        pmi = float(row.get("ism_pmi", 50.0))

        growth = (gdp > 0) and (cpi <= 0.5) and (pmi >= 50)
        inflation = (cpi > 0.5) and (gdp >= 0)
        recession = (gdp < 0) and (unemp > 5)
        stagflation = recession and (cpi > 0.5)

        if growth:
            return "growth"
        if stagflation:
            return "stagflation"
        if inflation and not recession:
            return "inflation"
        if recession:
            return "recession"
        return "neutral"

    return df_macro.apply(_classify, axis=1)

def rolling_matrix(
    series: SeriesND,
    window: int,
    *,
    fn: Callable[[NDArray], NDArray] | None = None,
) -> pd.Series:
    """
    Rolling helper ל-SeriesND של מטריצות.

    לכל אינדקס t:
      - אם t+1 < window → מחזירים NaN (אין מספיק היסטוריה).
      - אחרת:
          * נערום את המטריצות [t-window+1 ... t] ל-tensor בגודל
            (rows, cols, window)
          * אם fn לא None → נחזיר fn(tensor)
          * אחרת → נחזיר את ה-tensor עצמו.

    זה בדיוק ה-API שהפונקציות:
      - rolling_spectral_analysis
      - rolling_band_power

    מצפות לקבל.
    """
    # ודא שהסדרה בפורמט של "matrix series"
    series = ensure_matrix_series(series, "rolling_matrix")

    # ממירים את כל המטריצות ל-numpy arrays
    mats = [np.asarray(m, dtype=float) for m in series]
    idx_list = list(series.index)

    results: list[NDArray | float] = []
    for i, _idx in enumerate(idx_list):
        if i + 1 < window:
            # פחות מ־window מטריצות → אין חלון מלא
            results.append(np.nan)
            continue

        # tensor בגודל (rows, cols, window)
        tensor = np.stack(mats[i + 1 - window : i + 1], axis=2)

        if fn is not None:
            results.append(fn(tensor))
        else:
            results.append(tensor)

    return pd.Series(results, index=series.index)
