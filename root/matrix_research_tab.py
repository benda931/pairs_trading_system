# -*- coding: utf-8 -*-
"""
matrix_research_tab.py — 🔬 Matrix / Correlation Research (HF-grade)
====================================================================

High-end research tab for matrix research (correlation / covariance / returns),
designed to be integrated into `root/dashboard.py`:

    from root.matrix_research_tab import render_matrix_research_tab

Objectives
----------
- Provide hedge-fund–grade research tools over matrix data:
  * Correlation / covariance matrices.
  * Returns panels (T×N).
  * Generic tensors (N-D) with graceful degradation.
- Integrate deeply with:
  * `common.matrix_helpers`   — backend, PCA, rolling, diagnostics, reshape.
  * `common.advanced_metrics` — DTW, distance correlation, cointegration, VAR...
  * `common.helpers.summarize_series` — rich descriptive statistics.
- Stay robust even when some dependencies are missing:
  * The tab should still "run", just with fewer capabilities.

File structure
--------------
1. Core imports, type aliases, dataclasses & basic inference utilities.   <-- כאן
2. Numerical helpers (summaries, PCA, advanced metrics, series tools).
3. Streamlit sections (render_* functions without orchestration).
4. Top-level orchestrator: `render_matrix_research_tab`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import combinations
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd
import streamlit as st

# Optional anomaly detection engine
try:
    from core import anomaly_detection as anom  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    anom = None  # type: ignore[assignment]

# Optional plotting backend (Plotly)
try:
    import plotly.express as px  # type: ignore[import]
except Exception:  # pragma: no cover
    px = None  # type: ignore[assignment]

# Optional SciPy for hierarchical clustering of correlation matrices
try:
    from scipy.cluster.hierarchy import linkage, leaves_list  # type: ignore[import]
    from scipy.spatial.distance import squareform  # type: ignore[import]
except Exception:  # pragma: no cover
    linkage = None  # type: ignore[assignment]
    leaves_list = None  # type: ignore[assignment]
    squareform = None  # type: ignore[assignment]

# Optional AppContext / Risk Engine integration
try:
    from core.app_context import (  # או core.app_context אם זה הנתיב אצלך
        AppContext,
        get_current_ctx,
        ctx_to_action_playbook,
        compute_live_readiness,
        register_experiment_run,
    )
except Exception:  # pragma: no cover
    AppContext = None  # type: ignore[assignment]

    def get_current_ctx() -> Any:  # type: ignore[no-redef]
        return None

    def ctx_to_action_playbook(
        ctx: Any, *, kpis: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:  # type: ignore[no-redef]
        return {}

    def compute_live_readiness(ctx: Any) -> Dict[str, Any]:  # type: ignore[no-redef]
        return {}

    def register_experiment_run(  # type: ignore[no-redef]
        ctx: Any,
        *,
        kpis: Optional[Dict[str, Any]] = None,
        extra_meta: Optional[Dict[str, Any]] = None,
        reward: Optional[float] = None,
        auto_reward: bool = True,
    ) -> None:
        return

logger = logging.getLogger(__name__)

# Public API
__all__ = ["render_matrix_research_tab"]


# ===================== Type aliases & basic config =====================

MatrixShape = Tuple[int, ...]
MatrixName = str

# Sanity limits for UI previews (to avoid exploding the browser)
MAX_PREVIEW_ROWS: int = 200
MAX_PREVIEW_COLS: int = 50
MAX_FLAT_PREVIEW: int = 50_000  # max number of elements to show fully


# ===================== matrix_helpers — import with robust fallback =====================

# המטרה: אם common.matrix_helpers קיים, נייבא ממנו NDArray, SeriesND וכו'.
# אם הוא לא קיים, נגדיר NDArray כ-Ani מתוך typing (לא כמשתנה רגיל!), כדי שפיילאנס לא יתלונן.

try:
    from common.matrix_helpers import (  # type: ignore[attr-defined]
        NDArray,       # בד"כ מוגדר שם כ-TypeAlias
        SeriesND,
        RollingPolicy,
        backend_info,
        ensure_matrix_series,
        # rolling
        rolling_mean,
        rolling_std,
        rolling_cov,
        rolling_corr,
        rolling_zscore,
        rolling_quantile,
        rolling_ewm_mean,
        rolling_ewm_cov,
        # stats / PCA
        stat_cov,
        stat_corr,
        stat_center,
        stat_scale,
        stat_std,
        pca_decompose,
        pca_scree,
        pca_components_df,
        pca_scores_df,
        # diagnostics
        is_symmetric as mh_is_symmetric,
        is_psd as mh_is_psd,
        condition_number,
        nearest_psd,
        # reshape / flatten
        reshape_safe,
        flatten_row,
        flatten_series_table,
    )
except Exception:  # pragma: no cover
    # 💡 טריק נגד Pylance:
    # במקום NDArray = Any (שגורם ל-"Variable not allowed in type expression"),
    # אנחנו עושים: from typing import Any as NDArray — זה טיפוס ולא "משתנה".
    from typing import Any as NDArray  # type: ignore[assignment]

    SeriesND = pd.Series  # type: ignore[assignment]
    RollingPolicy = object  # רק placeholder; לא נשתמש בו כטיפוס מהותי.

    def backend_info() -> Dict[str, Any]:  # type: ignore[no-redef]
        """Fallback backend info when matrix_helpers is not available."""
        return {"backend": "unknown", "device": "cpu", "gpu": False}

    def ensure_matrix_series(obj: Any, *_: Any, **__: Any) -> SeriesND:  # type: ignore[no-redef]
        """Minimal fallback: wrap an iterable as Series[object]."""
        if isinstance(obj, pd.Series):
            return obj
        return pd.Series(list(obj), dtype="object")  # type: ignore[arg-type]

    # Rolling helpers (raise → חוסר backend יהיה שקוף למעלה ב-UI)
    def rolling_mean(*_: Any, **__: Any) -> SeriesND:
        raise NotImplementedError("matrix_helpers.rolling_mean is not available")

    def rolling_std(*_: Any, **__: Any) -> SeriesND:
        raise NotImplementedError("matrix_helpers.rolling_std is not available")

    def rolling_cov(*_: Any, **__: Any) -> SeriesND:
        raise NotImplementedError("matrix_helpers.rolling_cov is not available")

    def rolling_corr(*_: Any, **__: Any) -> SeriesND:
        raise NotImplementedError("matrix_helpers.rolling_corr is not available")

    def rolling_zscore(*_: Any, **__: Any) -> SeriesND:
        raise NotImplementedError("matrix_helpers.rolling_zscore is not available")

    def rolling_quantile(*_: Any, **__: Any) -> SeriesND:
        raise NotImplementedError("matrix_helpers.rolling_quantile is not available")

    def rolling_ewm_mean(*_: Any, **__: Any) -> SeriesND:
        raise NotImplementedError("matrix_helpers.rolling_ewm_mean is not available")

    def rolling_ewm_cov(*_: Any, **__: Any) -> SeriesND:
        raise NotImplementedError("matrix_helpers.rolling_ewm_cov is not available")

    # Stats / PCA
    def stat_cov(*_: Any, **__: Any) -> NDArray:
        raise NotImplementedError("matrix_helpers.stat_cov is not available")

    def stat_corr(*_: Any, **__: Any) -> NDArray:
        raise NotImplementedError("matrix_helpers.stat_corr is not available")

    def stat_center(*_: Any, **__: Any) -> NDArray:
        raise NotImplementedError("matrix_helpers.stat_center is not available")

    def stat_scale(*_: Any, **__: Any) -> NDArray:
        raise NotImplementedError("matrix_helpers.stat_scale is not available")

    def stat_std(*_: Any, **__: Any) -> NDArray:
        raise NotImplementedError("matrix_helpers.stat_std is not available")

    def pca_decompose(*_: Any, **__: Any) -> Dict[str, Any]:
        raise NotImplementedError("matrix_helpers.pca_decompose is not available")

    def pca_scree(*_: Any, **__: Any) -> pd.DataFrame:
        raise NotImplementedError("matrix_helpers.pca_scree is not available")

    def pca_components_df(*_: Any, **__: Any) -> pd.DataFrame:
        raise NotImplementedError("matrix_helpers.pca_components_df is not available")

    def pca_scores_df(*_: Any, **__: Any) -> pd.DataFrame:
        raise NotImplementedError("matrix_helpers.pca_scores_df is not available")

    # Diagnostics
    def mh_is_symmetric(*_: Any, **__: Any) -> bool:
        return False

    def mh_is_psd(*_: Any, **__: Any) -> bool:
        return False

    def condition_number(*_: Any, **__: Any) -> float:
        return float("nan")

    def nearest_psd(mat: NDArray, *_: Any, **__: Any) -> NDArray:
        return np.asarray(mat)

    # Reshape / flatten fallbacks
    def reshape_safe(mat: NDArray, *_: Any, **__: Any) -> NDArray:
        return np.asarray(mat)

    def flatten_row(series: SeriesND, *_: Any, **__: Any) -> SeriesND:
        return series

    def flatten_series_table(series: SeriesND, *_: Any, **__: Any) -> pd.DataFrame:
        return pd.DataFrame()


# ===================== summarize_series (helpers) =====================

try:
    from common.helpers import summarize_series  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    summarize_series = None  # type: ignore[assignment]

# Optional navigation helper from dashboard (cross-tab flows)
try:
    # בקוד האמיתי הקובץ חי תחת root/, לכן ה-import הזה תקין
    from root.dashboard import set_nav_target  # type: ignore[import]
except Exception:  # pragma: no cover
    set_nav_target = None  # type: ignore[assignment]

# ===================== advanced_metrics (DTW, cointegration, VAR, וכו') =====================

try:
    from common import advanced_metrics as adv  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    adv = None  # type: ignore[assignment]


# ===================== Data classes =====================


@dataclass
class MatrixDescriptor:
    """
    Lightweight description of a single matrix entry inside an .npz file.

    Attributes
    ----------
    name:
        Key inside the npz archive.
    shape:
        Full shape of the underlying ndarray.
    ndim:
        Number of dimensions.
    inferred_kind:
        Heuristic label:
        'correlation' / 'covariance' / 'returns-2d' / 'returns-1d' / 'square-generic' / 'generic-...'.
    symmetric:
        Whether the matrix is numerically symmetric (for 2D square matrices).
    notes:
        Free-form notes (e.g., "expected symmetric but matrix appears asymmetric").
    """

    name: MatrixName
    shape: MatrixShape
    ndim: int
    inferred_kind: str
    symmetric: bool
    notes: str = ""


@dataclass
class ShapeGroup:
    """
    Group of matrices that share the same shape.

    This is the natural representation for "matrix series" (e.g.,
    daily correlation snapshots with identical dimensions), which can
    then be analyzed along the time dimension.
    """

    shape: MatrixShape
    names: List[MatrixName]


# ===================== Inference utilities (no UI) =====================


def _infer_matrix_kind(mat: np.ndarray) -> str:
    """
    Try to infer matrix kind based on shape and basic value properties.

    Rules
    -----
    - 1D:
        → 'returns-1d'
    - 2D square:
        * diag≈1 and entries in [-1,1] → 'correlation'
        * diag>0 → 'covariance'
        * otherwise → 'square-generic'
    - 2D non-square:
        → 'returns-2d'
    - ndim != 1,2:
        → 'generic-{ndim}d'
    """
    mat = np.asarray(mat)

    if mat.ndim == 1:
        return "returns-1d"
    if mat.ndim != 2:
        return f"generic-{mat.ndim}d"

    n, m = mat.shape
    if n == m:
        diag = np.diag(mat)
        diag = diag[np.isfinite(diag)]
        if diag.size == 0:
            return "square-generic"

        if np.allclose(diag, 1.0, atol=1e-3) and np.nanmin(mat) >= -1.05 and np.nanmax(
            mat
        ) <= 1.05:
            return "correlation"

        if np.nanmin(diag) > 0:
            return "covariance"

        return "square-generic"

    # Non-square 2D: most often a T×N returns panel.
    return "returns-2d"


def _guess_asset_labels(
    npz_obj: Mapping[str, np.ndarray],
    key: str,
    mat: np.ndarray,
) -> List[str]:
    """
    Try to infer asset labels (column names) from companion arrays inside the npz.

    Heuristic:
    - Look for 1D arrays whose length matches mat.shape[1] under keys:
      'symbols', 'tickers', 'assets',
      f'{key}_symbols', f'{key}_tickers', f'{key}_assets'.
    - If nothing matches → fallback: ['col_0', 'col_1', ...].
    """
    mat = np.asarray(mat)

    if mat.ndim != 2:
        return [f"col_{i}" for i in range(mat.size)]

    n_cols = mat.shape[1]
    candidates = [
        "symbols",
        "tickers",
        "assets",
        f"{key}_symbols",
        f"{key}_tickers",
        f"{key}_assets",
    ]

    for ck in candidates:
        if ck in npz_obj:
            arr = np.asarray(npz_obj[ck])
            if arr.ndim == 1 and arr.shape[0] == n_cols:
                return [str(x) for x in arr]

    return [f"col_{i}" for i in range(n_cols)]


def _build_descriptors(npz_obj: Mapping[str, np.ndarray]) -> List[MatrixDescriptor]:
    """
    Build MatrixDescriptor objects for all ndarray entries in an npz archive.

    Non-ndarray entries (e.g. pickled Python objects) are ignored.
    """
    descs: List[MatrixDescriptor] = []

    for name in npz_obj.files:
        arr = np.asarray(npz_obj[name])
        if not isinstance(arr, np.ndarray):
            continue

        kind = _infer_matrix_kind(arr)
        symmetric = bool(
            arr.ndim == 2 and arr.shape[0] == arr.shape[1] and np.allclose(arr, arr.T)
        )

        notes: List[str] = []
        if kind in ("correlation", "covariance") and not symmetric:
            notes.append("expected symmetric but matrix appears asymmetric")

        descs.append(
            MatrixDescriptor(
                name=name,
                shape=tuple(arr.shape),
                ndim=arr.ndim,
                inferred_kind=kind,
                symmetric=symmetric,
                notes="; ".join(notes),
            )
        )

    return descs


def _group_by_shape(descriptors: Sequence[MatrixDescriptor]) -> List[ShapeGroup]:
    """
    Group matrices by shape so we can treat each group as a potential
    "matrix series" (e.g. time-indexed correlation snapshots).
    """
    groups: Dict[MatrixShape, List[MatrixName]] = {}
    for d in descriptors:
        groups.setdefault(d.shape, []).append(d.name)

    return [
        ShapeGroup(shape=shape, names=sorted(names))
        for shape, names in sorted(groups.items(), key=lambda kv: kv[0])
    ]
# ===================== Summaries & advanced_metrics wrappers =====================


def _safe_summarize_series(series: pd.Series, label: str) -> Optional[Dict[str, Any]]:
    """
    עטיפה בטוחה ל-common.helpers.summarize_series.

    התנהגות:
    ----------
    - אם summarize_series לא קיים → מחזיר None.
    - ניסיון ראשון: חתימה עשירה (series, name=..., advanced=True).
    - ניסיון שני: חתימה בסיסית (series).
    - במקרה של כשל → מחזיר None, כותב אזהרה ל-logger.
    """
    if summarize_series is None:
        return None

    try:
        return summarize_series(series, name=label, advanced=True)  # type: ignore[call-arg]
    except TypeError:
        try:
            return summarize_series(series)  # type: ignore[call-arg]
        except Exception as exc:  # pragma: no cover
            logger.warning("summarize_series(simple) failed for %s: %s", label, exc)
            return None
    except Exception as exc:  # pragma: no cover
        logger.warning("summarize_series failed for %s: %s", label, exc)
        return None


def _safe_adv_call(fn_name: str, *args: Any, **kwargs: Any) -> Optional[Any]:
    """
    קריאה בטוחה לפונקציה מתוך common.advanced_metrics.

    - אם המודול adv לא קיים / הפונקציה לא קיימת / יש כשל → מחזיר None.
    - מיועד לעטוף קריאות כמו:
        _safe_adv_call("dynamic_time_warping", series)
        _safe_adv_call("partial_correlation_matrix", series)
    """
    if adv is None:
        return None

    fn = getattr(adv, fn_name, None)
    if fn is None:
        return None

    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # pragma: no cover
        logger.warning("advanced_metrics.%s failed: %s", fn_name, exc)
        return None

def _safe_anom_call(fn_name: str, *args: Any, **kwargs: Any) -> Optional[Any]:
    """
    קריאה בטוחה לפונקציה מתוך core.anomaly_detection.

    - אם המודול anom לא קיים / הפונקציה לא קיימת / יש כשל → מחזיר None.
    - מיועד לקריאות כמו:
        _safe_anom_call("detect_corr_anomalies", corr_df)
        _safe_anom_call("detect_return_anomalies", returns_df)
    """
    if anom is None:
        return None

    fn = getattr(anom, fn_name, None)
    if fn is None:
        return None

    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # pragma: no cover
        logger.warning("anomaly_detection.%s failed: %s", fn_name, exc)
        return None

# ===================== Matrix ↔ DataFrame helpers =====================


def _matrix_to_df(
    mat: np.ndarray,
    labels: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    המרת מטריצה ל-DataFrame:

    - 1D → עמודה בודדת בשם 'values'.
    - 2D → DataFrame עם עמודות labels אם גודל מתאים, אחרת col_0, col_1...
    - N-D (N != 1,2) → flatten לוקטור values.
    """
    mat = np.asarray(mat)

    if mat.ndim == 1:
        return pd.DataFrame({"values": mat})

    if mat.ndim != 2:
        flat = mat.reshape(-1)
        return pd.DataFrame({"values": flat})

    n_rows, n_cols = mat.shape
    if labels is None or len(labels) != n_cols:
        labels = [f"col_{i}" for i in range(n_cols)]

    return pd.DataFrame(mat, columns=list(labels))


def _limit_df_for_preview(df: pd.DataFrame) -> pd.DataFrame:
    """
    מגביל DataFrame לתצוגה (rows/cols) לפי קונפיג גלובלי, בלי לשנות את ה-DataFrame המקורי.
    """
    n_rows, n_cols = df.shape
    rows = min(n_rows, MAX_PREVIEW_ROWS)
    cols = min(n_cols, MAX_PREVIEW_COLS)
    return df.iloc[:rows, :cols]


def _flatten_for_summary(mat: np.ndarray) -> pd.Series:
    """
    הופך מטריצה לסדרה שטוחה לצורך summarize_series, עם חיתוך אם יש יותר מדי איברים.
    """
    mat = np.asarray(mat)
    flat = mat.reshape(-1)
    if flat.size > MAX_FLAT_PREVIEW:
        flat = flat[:MAX_FLAT_PREVIEW]
    return pd.Series(flat, name="matrix_values")


# ===================== Basic stats & diagnostics =====================


def _matrix_basic_stats(mat: np.ndarray) -> Dict[str, Any]:
    """
    סטטיסטיקות בסיסיות על המטריצה (בלי שימוש ב-matrix_helpers):

    - shape / ndim
    - min / max / mean / std (על כל האיברים)
    - נוכחות NaN/Inf
    """
    mat = np.asarray(mat, dtype=float)
    flat = mat.reshape(-1)

    finite = np.isfinite(flat)
    if not finite.any():
        return {
            "n_elements": int(flat.size),
            "n_finite": 0,
            "min": np.nan,
            "max": np.nan,
            "mean": np.nan,
            "std": np.nan,
            "has_nan": True,
            "has_inf": True,
        }

    vals = flat[finite]
    return {
        "n_elements": int(flat.size),
        "n_finite": int(finite.sum()),
        "min": float(vals.min()),
        "max": float(vals.max()),
        "mean": float(vals.mean()),
        "std": float(vals.std(ddof=1) if vals.size > 1 else 0.0),
        "has_nan": bool(np.isnan(flat).any()),
        "has_inf": bool(np.isinf(flat).any()),
    }


def _matrix_diagnostics(
    mat: np.ndarray,
    inferred_kind: str,
) -> Dict[str, Any]:
    """
    אבחונים רחבים למטריצה (סימטריה, PSD, מספר-תנאי, backend):

    - symmetric / psd (אם פונקציות mtx_helpers זמינות).
    - condition_number (אם קיים; אחרת NaN).
    - backend_info (CPU/GPU וכו').
    """
    mat = np.asarray(mat, dtype=float)
    diag: Dict[str, Any] = {
        "inferred_kind": inferred_kind,
        "shape": tuple(mat.shape),
        "ndim": mat.ndim,
    }

    # סימטריה + PSD דרך matrix_helpers אם אפשר
    try:
        diag["is_symmetric_exact"] = bool(mh_is_symmetric(mat))  # type: ignore[arg-type]
    except Exception:
        diag["is_symmetric_exact"] = bool(
            mat.ndim == 2 and mat.shape[0] == mat.shape[1] and np.allclose(mat, mat.T)
        )

    try:
        diag["is_psd"] = bool(mh_is_psd(mat))  # type: ignore[arg-type]
    except Exception:
        diag["is_psd"] = None

    # מספר תנאי
    try:
        diag["condition_number"] = float(condition_number(mat))  # type: ignore[arg-type]
    except Exception:
        diag["condition_number"] = np.nan

    # מידע backend
    try:
        diag["backend_info"] = backend_info()
    except Exception:
        diag["backend_info"] = {"backend": "unknown"}

    # סטטיסטיקות בסיסיות
    diag.update(_matrix_basic_stats(mat))

    return diag


# ===================== PCA helpers (single matrix) =====================


def _wrap_single_matrix_series(
    mat: np.ndarray,
    index_label: str = "snapshot",
) -> SeriesND:
    """
    עוטף מטריצה בודדת ל- SeriesND עם אינדקס אחד, לשימוש בפונקציות
    שעובדות על SeriesND (כמו pca_decompose / advanced_metrics).

    index_label משמש כשם ה-"תצפית" (למשל תאריך / שם snapshot).
    """
    ser = pd.Series([np.asarray(mat)], index=[index_label])  # type: ignore[arg-type]
    return ensure_matrix_series(ser, "single_matrix")  # type: ignore[no-any-return]


def _run_pca_on_matrix(
    df: pd.DataFrame,
    *,
    n_components: Optional[int] = None,
    center: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    PCA למטריצה בודדת (T×N):

    אם matrix_helpers.pca_decompose זמין:
        - משתמשים בו כדי לקבל:
            components (loadings),
            explained_var,
            explained_ratio.
    אחרת:
        - PCA ידני דרך cov + eig.

    מחזיר:
        {
            "components": DataFrame,
            "explained_var": DataFrame,
            "explained_ratio": DataFrame,
        }
    עם אינדקס יחיד (שם ה-snapshot).
    """
    mat = df.values.astype(float)
    series = _wrap_single_matrix_series(mat, index_label="matrix")

    # ניסיון ראשון – דרך matrix_helpers
    try:
        pca_res = pca_decompose(series, n_components=n_components, center=center)  # type: ignore[call-arg]
        components = pca_res.get("components", pd.DataFrame())
        explained_var = pca_res.get("explained_var", pd.DataFrame())
        explained_ratio = pca_res.get("explained_ratio", pd.DataFrame())
        return {
            "components": components,
            "explained_var": explained_var,
            "explained_ratio": explained_ratio,
        }
    except Exception as exc:
        logger.warning("pca_decompose failed, falling back to manual PCA: %s", exc)

    # Fallback: PCA ידני
    if mat.ndim != 2 or mat.shape[1] < 2:
        return {
            "components": pd.DataFrame(),
            "explained_var": pd.DataFrame(),
            "explained_ratio": pd.DataFrame(),
        }

    X = mat.copy()
    if center:
        X = X - np.nanmean(X, axis=0, keepdims=True)
    X = np.nan_to_num(X, nan=0.0)

    # Covariance (columns as variables)
    cov = np.cov(X, rowvar=False)
    try:
        eigvals, eigvecs = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return {
            "components": pd.DataFrame(),
            "explained_var": pd.DataFrame(),
            "explained_ratio": pd.DataFrame(),
        }

    # מיין לפי ערכים עצמיים מהגדול לקטן
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    if n_components is not None and n_components < eigvals.size:
        eigvals = eigvals[:n_components]
        eigvecs = eigvecs[:, :n_components]

    total_var = float(eigvals.sum()) or 1.0
    ratio = eigvals / total_var

    comp_cols = [f"PC{j+1}" for j in range(eigvecs.shape[1])]
    # components: rows=features, cols=PCs
    components = pd.DataFrame(eigvecs, index=df.columns, columns=comp_cols)
    explained_var = pd.DataFrame([eigvals], index=["matrix"], columns=comp_cols)
    explained_ratio = pd.DataFrame([ratio], index=["matrix"], columns=comp_cols)

    return {
        "components": components,
        "explained_var": explained_var,
        "explained_ratio": explained_ratio,
    }


# ===================== Series-of-matrices helpers =====================


def _build_matrix_series_from_group(
    npz_obj: Mapping[str, np.ndarray],
    group: "ShapeGroup",
) -> SeriesND:
    """
    בונה SeriesND של מטריצות מתוך ShapeGroup:

    - index = שמות המטריצות בתוך הקובץ (למשל timestamps / שמות snapshots).
    - values = המטריצות עצמן (np.ndarray).
    - ensure_matrix_series דואג להומוגניות 2D + ולידציה.
    """
    mats: List[NDArray] = []
    for name in group.names:
        arr = np.asarray(npz_obj[name])
        mats.append(arr)

    ser = pd.Series(mats, index=group.names, dtype="object")  # type: ignore[arg-type]
    return ensure_matrix_series(ser, f"shape={group.shape}")  # type: ignore[no-any-return]


def _rolling_corr_strength(
    series: SeriesND,
    window: int,
) -> Optional[pd.Series]:
    """
    דוגמה לניתוח סדרת מטריצות: "חוזק קורלציה" מתגלגל.

    - משתמש ב-rolling_corr מתוך matrix_helpers אם זמין.
    - מחזיר סדרה של scalar per snapshot (למשל ממוצע |corr| מחוץ לדיאגונל).
    """
    try:
        corr_series = rolling_corr(series, window=window)  # type: ignore[call-arg]
    except Exception as exc:
        logger.warning("rolling_corr failed: %s", exc)
        return None

    def _strength(mat: NDArray) -> float:
        mat = np.asarray(mat, dtype=float)
        if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
            return float("nan")
        # מתעלמים מהדיאגונל
        mask = ~np.eye(mat.shape[0], dtype=bool)
        vals = np.abs(mat[mask])
        if vals.size == 0:
            return float("nan")
        return float(np.nanmean(vals))

    try:
        return corr_series.apply(_strength)  # type: ignore[no-any-return]
    except Exception as exc:
        logger.warning("rolling_corr_strength apply failed: %s", exc)
        return None


# ===================== Advanced metrics helpers (pair & matrix level) =====================


def _build_pair_series(df: pd.DataFrame, col_a: str, col_b: str) -> SeriesND:
    """
    בונה SeriesND עם snapshot אחד של זוג נכסים (2 עמודות):

    - מטריצת בסיס: shape = (T, 2).
    - index = ['pair'].
    - מתאים ל-dynamic_time_warping / distance_correlation / mahalanobis וכו'.
    """
    sub = df[[col_a, col_b]].dropna()
    mat = sub.values.astype(float)
    ser = pd.Series([mat], index=["pair"], dtype="object")  # type: ignore[arg-type]
    return ensure_matrix_series(ser, "pair_series")  # type: ignore[no-any-return]


def _compute_pair_advanced_metrics(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
) -> Dict[str, Any]:
    """
    מחשב מדדים מתקדמים לזוג נכסים (אם הפונקציות זמינות ב-advanced_metrics):

    - DTW distance (dynamic_time_warping)
    - Distance correlation (distance_correlation)
    - Mahalanobis distance (summary) (mahalanobis_distance)
    - Hurst exponent (hurst_exponent)
    """
    results: Dict[str, Any] = {}
    if adv is None:
        return results

    series = _build_pair_series(df, col_a, col_b)

    # DTW distance
    dtw_series = _safe_adv_call("dynamic_time_warping", series)
    if isinstance(dtw_series, pd.Series) and not dtw_series.empty:
        results["dtw_distance"] = float(dtw_series.iloc[-1])

    # Distance correlation
    dcor_series = _safe_adv_call("distance_correlation", series)
    if isinstance(dcor_series, pd.Series) and not dcor_series.empty:
        results["distance_correlation"] = float(dcor_series.iloc[-1])

    # Mahalanobis distance matrix -> סיכום
    mahal_series = _safe_adv_call("mahalanobis_distance", series)
    if isinstance(mahal_series, pd.Series) and not mahal_series.empty:
        mat = np.asarray(mahal_series.iloc[-1], dtype=float)
        if mat.size > 0:
            # ממוצע מרחקים מתוך מטריצת מרחקים
            results["mahalanobis_avg"] = float(np.nanmean(mat))

    # Hurst exponent (על כל סוכן בנפרד, ממוצע)
    hurst_series = _safe_adv_call("hurst_exponent", series)
    if isinstance(hurst_series, pd.Series) and not hurst_series.empty:
        vals = np.asarray(hurst_series.iloc[-1], dtype=float)
        if vals.size > 0:
            results["hurst_mean"] = float(np.nanmean(vals))

    return results


def _compute_matrix_advanced_matrices(
    df: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """
    מדדים מטריציוניים מתקדמים (אם זמינים):

    - partial_correlation_matrix → Mat[k×k]
    - distance_covariance_matrix  → Mat[k×k]
    - distance_correlation_matrix → Mat[k×k]

    מחזיר dict של DataFrame כדי שיהיה נוח להציג אותם ב-UI.
    """
    out: Dict[str, pd.DataFrame] = {}
    if adv is None:
        return out

    mat = df.values.astype(float)
    series = _wrap_single_matrix_series(mat, index_label="matrix")

    # Partial correlation
    pcm_series = _safe_adv_call("partial_correlation_matrix", series)
    if isinstance(pcm_series, pd.Series) and not pcm_series.empty:
        pcm = np.asarray(pcm_series.iloc[-1], dtype=float)
        out["partial_corr"] = pd.DataFrame(pcm, index=df.columns, columns=df.columns)

    # Distance covariance matrix
    dcov_series = _safe_adv_call("distance_covariance_matrix", series)
    if isinstance(dcov_series, pd.Series) and not dcov_series.empty:
        dcov = np.asarray(dcov_series.iloc[-1], dtype=float)
        out["distance_cov"] = pd.DataFrame(dcov, index=df.columns, columns=df.columns)

    # Distance correlation matrix
    dcor_series = _safe_adv_call("distance_correlation_matrix", series)
    if isinstance(dcor_series, pd.Series) and not dcor_series.empty:
        dcor = np.asarray(dcor_series.iloc[-1], dtype=float)
        out["distance_corr"] = pd.DataFrame(dcor, index=df.columns, columns=df.columns)

    return out

def _render_anomaly_detection_section(df: pd.DataFrame) -> None:
    """
    Anomaly Detection על:
    - מטריצת קורלציה (detect_corr_anomalies)
    - מטריצת תשואות (detect_return_anomalies)

    משתמש ב-core.anomaly_detection אם זמין, עם קריאה בטוחה (_safe_anom_call).
    """
    st.markdown("### 🚨 Anomaly Detection – קורלציות ותשואות חריגות")

    if anom is None:
        st.info("מודול core.anomaly_detection לא זמין בסביבה – דילוג על Anomaly Detection.")
        return

    if df.shape[1] < 2 or df.shape[0] < 5:
        st.info("צריך לפחות שתי עמודות וכמה עשרות תצפיות כדי לזהות אנומליות.")
        return

    # ---- Data coverage / universe snapshot ----
    with st.expander("🧾 Data coverage (מה יש לנו באמת?)", expanded=False):
        try:
            cov_info = []
            for col in df.columns:
                s = df[col].dropna()
                cov_info.append(
                    {
                        "symbol": col,
                        "len": int(len(s)),
                        "start": s.index.min() if len(s) else None,
                        "end": s.index.max() if len(s) else None,
                        "missing_pct": float(100.0 * (1.0 - len(s) / max(len(df), 1))),
                    }
                )
            cov_df = pd.DataFrame(cov_info)
            st.dataframe(cov_df, width="stretch")
        except Exception:
            st.info("לא הצלחתי לחשב Data coverage (בעיה באינדקס/תאריכים).")

    # 1) תשואות – נשתמש ב-pct_change (תלוי מה יש לך ב-npz – לרוב מחירי נכסים)
    returns_df = df.pct_change().dropna(how="all")
    if returns_df.empty:
        st.info("אחרי pct_change אין מספיק נתונים כדי לחשב תשואות.")
        return

    # 2) מטריצת קורלציה
    corr_df = returns_df.corr()
    if corr_df.empty:
        st.info("לא הצלחנו לבנות מטריצת קורלציה מהנתונים.")
        return

    col1, col2 = st.columns(2)

    # ----- אנומליות במטריצת קורלציה -----
    with col1:
        st.subheader("📌 אנומליות במטריצת הקורלציה")
        try:
            corr_anom = _safe_anom_call(
                "detect_corr_anomalies",
                corr_df,
                baseline_corr=None,
                cfg=None,  # אפשר להעביר CorrAnomalyConfig אם תרצה בעתיד
            )
        except Exception:
            corr_anom = None

        if isinstance(corr_anom, pd.DataFrame) and not corr_anom.empty:
            st.caption(f"נמצאו {len(corr_anom)} זוגות עם קורלציה חריגה.")
            st.dataframe(corr_anom, width="stretch", height=260)
            st.download_button(
                "⬇️ הורד אנומליות קורלציה (CSV)",
                corr_anom.to_csv(index=False).encode("utf-8"),
                file_name="matrix_corr_anomalies.csv",
                mime="text/csv",
                key="matrix_corr_anom_csv",
            )
        else:
            st.info("לא זוהו אנומליות משמעותיות בקורלציה (או שהפונקציה לא החזירה תוצאה).")

    # ----- אנומליות בתשואות -----
    with col2:
        st.subheader("📈 אנומליות בתשואות (Returns)")
        try:
            ret_anom = _safe_anom_call(
                "detect_return_anomalies",
                returns_df,
                cfg=None,  # אפשר להעביר ReturnAnomalyConfig בעתיד
            )
        except Exception:
            ret_anom = None

        if isinstance(ret_anom, pd.DataFrame) and not ret_anom.empty:
            st.caption(f"נמצאו {len(ret_anom)} אנומליות בתשואות.")
            st.dataframe(ret_anom.head(300), width="stretch", height=260)
            st.download_button(
                "⬇️ הורד אנומליות תשואות (CSV)",
                ret_anom.to_csv(index=False).encode("utf-8"),
                file_name="matrix_return_anomalies.csv",
                mime="text/csv",
                key="matrix_ret_anom_csv",
            )
        else:
            st.info("לא זוהו אנומליות משמעותיות בתשואות (או שהפונקציה לא החזירה תוצאה).")

# ===================== Streamlit section renderers (building blocks) =====================


def _render_matrix_preview_section(
    name: str,
    mat: np.ndarray,
    labels: Sequence[str],
    desc: MatrixDescriptor,
) -> pd.DataFrame:
    """
    פריוויו ראשוני למטריצה + KPI בסיסי על הנתונים.
    מחזיר את ה-DataFrame המוצג לצורך שימוש בהמשך (PCA/advanced).
    """
    st.subheader(f"📌 Matrix preview — `{name}`")

    stats = _matrix_basic_stats(mat)

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Shape", f"{desc.shape}")
    kpi2.metric("Kind", desc.inferred_kind)
    kpi3.metric("Finite ratio", f"{stats['n_finite'] / max(stats['n_elements'], 1):.2%}")
    kpi4.metric("Std (all values)", f"{stats['std']:.4g}")

    st.write(
        f"**ndim:** {desc.ndim} | "
        f"**symmetric:** {desc.symmetric} | "
        f"**has_nan:** {stats['has_nan']} | "
        f"**has_inf:** {stats['has_inf']}"
    )
    if desc.notes:
        st.info(desc.notes)

    df = _matrix_to_df(mat, labels=labels)
    st.caption("תצוגה חלקית של המטריצה (preview מוגבל):")
    st.dataframe(_limit_df_for_preview(df), width="stretch")

    return df


def _render_matrix_summary_section(
    name: str,
    mat: np.ndarray,
    desc: MatrixDescriptor,
    df: Optional[pd.DataFrame] = None,
) -> None:
    """
    תקציר סטטיסטי ברמת מטריצה וברמת נכס:
    - summarize_series על כל האיברים.
    - summarize_series ברמת נכס (עמודות) אם df זמין.
    - דיאגנוסטיקה מורחבת (PSD, symmetry, condition number, backend).
    """
    st.markdown("### 4️⃣ Summary & Diagnostics")

    flat = _flatten_for_summary(mat)
    summary = _safe_summarize_series(flat, label=f"{name}_values")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📊 summarize_series — all entries")
        if summary is not None:
            try:
                st.json(summary)
            except Exception:
                st.dataframe(pd.DataFrame(summary, index=[0]), width="stretch")
        else:
            st.info("summarize_series לא זמין או נכשל — דילוג על תקציר האיברים.")

        # per-asset summary (אם יש DataFrame עם עמודות)
        if df is not None and df.shape[1] >= 1:
            rows: list[dict[str, Any]] = []
            for col in df.columns:
                s = df[col].dropna()
                res = _safe_summarize_series(s, label=str(col))
                if not res:
                    continue
                row: dict[str, Any] = {"asset": col}
                for k, v in res.items():
                    if isinstance(v, (int, float, np.number, str, bool)):
                        row[str(k)] = v
                rows.append(row)
            if rows:
                st.markdown("#### 🧱 Per-asset summary")
                st.dataframe(pd.DataFrame(rows), width="stretch")

    with col2:
        st.markdown("#### 🩺 Diagnostics")
        diag = _matrix_diagnostics(mat, inferred_kind=desc.inferred_kind)
        st.json(diag)


def _render_matrix_corr_cov_section(df: pd.DataFrame) -> None:
    """
    מציג מטריצות קורלציה/קובאריאנס, עם אפשרות להשתמש ב-matrix_helpers אם זמין.
    """
    st.markdown("### 5️⃣ Correlation / Covariance")

    if df.shape[1] < 2:
        st.info("מטריצה עם עמודה אחת — אין קורלציה/קובאריאנס בין נכסים.")
        return

    use_helpers = st.checkbox(
        "להעדיף חישוב דרך matrix_helpers (stat_corr/stat_cov) אם זמין",
        value=True,
        key="matrix_use_helpers_corr_cov",
    )

    if use_helpers:
        try:
            cov_mat = stat_cov(df.values.astype(float))  # type: ignore[arg-type]
            corr_mat = stat_corr(df.values.astype(float))  # type: ignore[arg-type]
            corr = pd.DataFrame(corr_mat, index=df.columns, columns=df.columns)
            cov = pd.DataFrame(cov_mat, index=df.columns, columns=df.columns)
        except Exception as exc:
            logger.warning("stat_corr/stat_cov failed, falling back to pandas: %s", exc)
            corr = df.corr()
            cov = df.cov()
    else:
        corr = df.corr()
        cov = df.cov()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Correlation matrix")
        if px is not None:
            fig = px.imshow(
                corr,
                text_auto=".2f",
                title="Correlation heatmap",
                aspect="auto",
            )
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(
            corr.style.background_gradient(axis=None),
            width="stretch",
        )

    with col2:
        st.subheader("📉 Covariance matrix")
        if px is not None:
            fig = px.imshow(
                cov,
                text_auto=".2f",
                title="Covariance heatmap",
                aspect="auto",
            )
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(
            cov.style.background_gradient(axis=None),
            width="stretch",
        )

    # Clustered correlation heatmap (hierarchical)
    with st.expander("🧩 Clustered correlation heatmap (hierarchical)", expanded=False):
        _render_clustered_corr_heatmap(corr)

def _render_clustered_corr_heatmap(corr: pd.DataFrame) -> None:
    """
    Clustered correlation heatmap (hierarchical):
    - ממפה את הנכסים לפי hierarchical clustering על distance מהקורלציה.
    """
    st.markdown("#### 🧩 Clustered correlation (hierarchical)")

    if corr.shape[1] < 2:
        st.info("צריך לפחות שני נכסים כדי להציג Clustered correlation.")
        return

    if linkage is None or squareform is None or leaves_list is None:
        st.info("SciPy לא זמין בסביבה – לא ניתן לבצע hierarchical clustering.")
        return

    corr_filled = corr.fillna(0.0)
    # Distance מתוך קורלציה: d = sqrt(0.5 * (1 - corr))
    corr_clip = corr_filled.clip(-1.0, 1.0)
    dist_mat = np.sqrt(0.5 * (1.0 - corr_clip.values))

    try:
        condensed = squareform(dist_mat, checks=False)
        Z = linkage(condensed, method="average")
        order = leaves_list(Z)
    except Exception as exc:  # pragma: no cover
        logger.warning("Clustered corr heatmap failed: %s", exc)
        st.info("לא הצלחנו לחשב Cluster Map לקורלציה (בעיה ב-SciPy או בנתונים).")
        return

    ordered_labels = corr_filled.columns.values[order]
    corr_ord = corr_filled.loc[ordered_labels, ordered_labels]

    if px is not None:
        fig = px.imshow(
            corr_ord,
            text_auto=".2f",
            title="Clustered correlation heatmap (hierarchical)",
            aspect="auto",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        corr_ord.style.background_gradient(axis=None),
        width="stretch",
    )

def _render_matrix_pca_section(df: pd.DataFrame) -> None:
    """
    PCA למטריצה יחידה:
    - בחירת מספר רכיבים + center.
    - טבלת explained variance + cumulative.
    - גרף bar ל-explained variance.
    - טבלת components (loadings).
    """
    st.markdown("### 6️⃣ PCA — Principal Components")

    if df.shape[1] < 2 or df.shape[0] <= df.shape[1]:
        st.info("PCA דורש לפחות שני נכסים ויותר תצפיות ממספר העמודות.")
        return

    col1, col2 = st.columns(2)
    with col1:
        max_comp = min(df.shape[1], 10)
        n_components = st.slider(
            "מספר רכיבי PCA:",
            min_value=1,
            max_value=max_comp,
            value=max_comp,
            step=1,
            key="matrix_pca_n_components",
        )
    with col2:
        center = st.checkbox("Center data לפני PCA", value=True, key="matrix_pca_center")

    res = _run_pca_on_matrix(df, n_components=n_components, center=center)
    components = res.get("components", pd.DataFrame())
    ev = res.get("explained_var", pd.DataFrame())
    er = res.get("explained_ratio", pd.DataFrame())

    if components.empty or er.empty:
        st.info("לא הצלחנו להריץ PCA (או שהנתונים לא מתאימים) — דילוג.")
        return

    st.subheader("🧬 Explained variance")
    scree_df = er.T.copy()
    scree_df.columns = ["ExplainedVarianceRatio"]
    scree_df["Eigenvalue"] = ev.T.iloc[:, 0]
    scree_df["Cumulative"] = scree_df["ExplainedVarianceRatio"].cumsum()
    st.dataframe(scree_df, width="stretch")

    if px is not None:
        fig = px.bar(
            scree_df.reset_index(),
            x="index",
            y="ExplainedVarianceRatio",
            title="PCA — Explained variance ratio",
            labels={"index": "Component"},
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("🧭 Loadings (components)")
    pc_to_show = st.selectbox(
        "בחר רכיב להצגת loadings:",
        options=list(components.columns),
        index=0,
        key="matrix_pca_component_to_show",
    )
    loadings = components[pc_to_show].sort_values(ascending=False).to_frame("loading")
    st.dataframe(loadings, width="stretch")


def _render_pair_advanced_section(df: pd.DataFrame) -> None:
    """
    ניתוח מתקדם לזוג נכסים:
    - בחירת שני נכסים.
    - DTW, distance correlation, tail dependence.
    - אופציה ל-cointegration + Granger על הזוג.
    """
    if df.shape[1] < 2:
        st.info("לעבודה על זוג נכסים יש צורך לפחות בשתי עמודות.")
        return

    st.markdown("### 7️⃣ Advanced pair metrics")

    columns = list(df.columns)
    default_pair = columns[:2]

    cols_select = st.multiselect(
        "בחר שני נכסים לניתוח זוגי:",
        options=columns,
        default=default_pair,
        key="matrix_pair_advanced_cols",
    )

    if len(cols_select) != 2:
        st.info("בחר *בדיוק* שני נכסים (או השאר ריק אם זה לא נדרש).")
        return

    col_a, col_b = cols_select
    pair_df = df[[col_a, col_b]].dropna()

    col1, col2 = st.columns(2)

    # advanced_metrics: DTW + distance corr + Hurst וכו'
    with col1:
        st.markdown("#### 📐 DTW / Distance / Hurst")

        metrics = _compute_pair_advanced_metrics(pair_df, col_a, col_b)
        if metrics:
            for k, v in metrics.items():
                if isinstance(v, (int, float, np.number)):
                    st.metric(k, f"{v:.4f}")
                else:
                    st.metric(k, str(v))
        else:
            st.info("advanced_metrics לא זמין או לא החזיר תוצאות לזוג הזה.")

        # distance_correlation (גרסה scalar)
        if adv is not None:
            try:
                series = _build_pair_series(df, col_a, col_b)
                dcor_series = adv.distance_correlation(series)  # type: ignore[attr-defined]
                if isinstance(dcor_series, pd.Series) and not dcor_series.empty:
                    st.metric(
                        "Distance correlation (matrix)",
                        f"{float(dcor_series.iloc[-1]):.4f}",
                    )
            except Exception as exc:
                logger.warning("distance_correlation failed: %s", exc)

    # Tail & cointegration
    with col2:
        st.markdown("#### 🦊 Tail & cointegration")

        # tail dependence
        try:
            tail = _safe_adv_call("tail_dependence", pair_df, 0.95)
        except Exception:
            tail = None

        if isinstance(tail, pd.Series):
            st.caption("Tail dependence coefficients (u=0.95):")
            st.dataframe(tail.to_frame("value"))
        else:
            st.info("tail_dependence לא זמין / נכשל.")

        # cointegration_test
        try:
            if adv is not None:
                series = _build_pair_series(df, col_a, col_b)
                coint_df = adv.cointegration_test(series)  # type: ignore[attr-defined]
                if isinstance(coint_df, pd.DataFrame) and not coint_df.empty:
                    st.dataframe(coint_df, width="stretch")
        except Exception as exc:
            logger.warning("cointegration_test failed: %s", exc)


def _render_matrix_advanced_section(df: pd.DataFrame) -> None:
    """
    מדדים מטריציוניים מתקדמים:
    - partial correlation matrix
    - distance covariance matrix
    - distance correlation matrix
    """
    st.markdown("### 8️⃣ Advanced matrix metrics")

    if df.shape[1] < 2:
        st.info("מטריצה עם עמודה אחת — אין מדדים מטריציוניים מתקדמים להצגה.")
        return

    matrices = _compute_matrix_advanced_matrices(df)

    col1, col2 = st.columns(2)

    with col1:
        if "partial_corr" in matrices:
            st.subheader("📌 Partial correlation")
            pcm = matrices["partial_corr"]
            if px is not None:
                fig = px.imshow(
                    pcm,
                    text_auto=".2f",
                    title="Partial correlation matrix",
                    aspect="auto",
                )
                st.plotly_chart(fig, use_container_width=True)
            st.dataframe(
                pcm.style.background_gradient(axis=None),
                width="stretch",
            )
        else:
            st.info("partial_correlation_matrix לא זמין מתוך advanced_metrics.")

    with col2:
        if "distance_corr" in matrices:
            st.subheader("📌 Distance correlation")
            dcor = matrices["distance_corr"]
            if px is not None:
                fig = px.imshow(
                    dcor,
                    text_auto=".2f",
                    title="Distance correlation matrix",
                    aspect="auto",
                )
                st.plotly_chart(fig, use_container_width=True)
            st.dataframe(
                dcor.style.background_gradient(axis=None),
                width="stretch",
            )
        else:
            st.info("distance_correlation_matrix לא זמין מתוך advanced_metrics.")

    if "distance_cov" in matrices:
        st.subheader("📌 Distance covariance")
        dcov = matrices["distance_cov"]
        st.dataframe(
            dcov.style.background_gradient(axis=None),
            width="stretch",
        )


def _render_volatility_shrinkage_section(df: pd.DataFrame) -> None:
    """
    Volatility & shrinkage tools:
    - GARCH volatility על נכס בודד.
    - PCA residual volatility.
    - Shrinkage covariance (Ledoit-Wolf / OAS / EWMA+OAS).
    """
    st.markdown("### 9️⃣ Volatility & Shrinkage")

    if adv is None:
        st.info("advanced_metrics לא זמין — דילוג על Volatility & Shrinkage.")
        return

    if df.shape[1] < 1:
        st.info("אין עמודות נתונים לניתוח.")
        return

    cols = list(df.columns)
    col1, col2 = st.columns(2)

    with col1:
        asset = st.selectbox(
            "נכס לניתוח תנודתיות (GARCH):",
            options=cols,
            index=0,
            key="matrix_vol_asset",
        )
        try:
            ret = df[asset].dropna()
            garch = adv.garch_volatility(ret)  # type: ignore[attr-defined]
            if isinstance(garch, pd.Series) and not garch.empty:
                st.line_chart(garch, width="stretch")
        except Exception as exc:
            logger.warning("garch_volatility failed: %s", exc)
            st.info("garch_volatility לא זמין / נכשל.")

    with col2:
        st.caption("PCA residual volatility (factor-removal)")
        try:
            res_vol = adv.pca_residual_vol(df.dropna(), n_components=3, window=60)  # type: ignore[attr-defined]
            if isinstance(res_vol, pd.Series) and not res_vol.empty:
                st.line_chart(res_vol, width="stretch")
        except Exception as exc:
            logger.warning("pca_residual_vol failed: %s", exc)
            st.info("pca_residual_vol לא זמין / נכשל.")

    st.subheader("Shrinkage covariance (Ledoit-Wolf / OAS / EWMA+OAS)")
    try:
        series_single = _wrap_single_matrix_series(df.values, index_label="matrix")
        lw_series = _safe_adv_call("shrinkage_covariance", series_single)
        oas_series = _safe_adv_call("oas_covariance", series_single)
        ewma_oas_series = _safe_adv_call("ewma_oas_covariance", series_single, span=60)

        if isinstance(lw_series, pd.Series) and not lw_series.empty:
            lw = np.asarray(lw_series.iloc[-1], dtype=float)
            st.caption("Ledoit-Wolf shrinked covariance:")
            st.dataframe(
                pd.DataFrame(lw, index=df.columns, columns=df.columns),
                width="stretch",
            )
        if isinstance(oas_series, pd.Series) and not oas_series.empty:
            oas = np.asarray(oas_series.iloc[-1], dtype=float)
            st.caption("OAS covariance:")
            st.dataframe(
                pd.DataFrame(oas, index=df.columns, columns=df.columns),
                width="stretch",
            )
        if isinstance(ewma_oas_series, pd.Series) and not ewma_oas_series.empty:
            eo = np.asarray(ewma_oas_series.iloc[-1], dtype=float)
            st.caption("EWMA+OAS covariance:")
            st.dataframe(
                pd.DataFrame(eo, index=df.columns, columns=df.columns),
                width="stretch",
            )
    except Exception as exc:
        logger.warning("shrinkage covariance section failed: %s", exc)


def _render_matrix_entropy_section(df: pd.DataFrame) -> None:
    """
    Entropy & complexity על המטריצה:
    - sample entropy.
    - Hurst exponent.
    - spectral coherence בין שני נכסים.
    """
    st.markdown("### 🔟 Entropy & Complexity")

    if adv is None:
        st.info("advanced_metrics לא זמין — דילוג על entropy/Hurst.")
        return

    series_single = _wrap_single_matrix_series(df.values, index_label="matrix")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 🔢 Sample entropy")
        samp_ent = _safe_adv_call("sample_entropy", series_single)
        if isinstance(samp_ent, pd.Series) and not samp_ent.empty:
            st.metric("Sample entropy", f"{float(samp_ent.iloc[0]):.4f}")
        else:
            st.info("sample_entropy לא זמין / נכשל.")

    with col2:
        st.markdown("#### 📈 Hurst exponent")
        hurst = _safe_adv_call("hurst_exponent", series_single)
        if isinstance(hurst, pd.Series) and not hurst.empty:
            st.metric("Hurst exponent", f"{float(hurst.iloc[0]):.4f}")
        else:
            st.info("hurst_exponent לא זמין / נכשל.")

    with col3:
        st.markdown("#### 📡 Spectral coherence (first two columns)")
        try:
            coh = _safe_adv_call("spectral_coherence", series_single, fs=1.0, nperseg=None)
        except Exception:
            coh = None
        if isinstance(coh, pd.Series) and not coh.empty:
            arr = np.asarray(coh.iloc[0], dtype=float)
            avg_coh = float(np.nanmean(arr)) if arr.size else np.nan
            st.metric("Avg coherence", f"{avg_coh:.4f}")
        else:
            st.info("spectral_coherence לא זמין / נכשל.")


def _render_series_overview_section(shape_groups: Sequence[ShapeGroup]) -> None:
    """
    מציג חתך על קבוצות הצורה (shape groups) — כמה מטריצות יש בכל shape.
    """
    st.markdown("### 🔁 Matrix-series overview (by shape)")

    rows: list[dict[str, Any]] = []
    for g in shape_groups:
        rows.append(
            {
                "shape": str(g.shape),
                "n_matrices": len(g.names),
                "names_sample": ", ".join(g.names[:5]) + ("..." if len(g.names) > 5 else ""),
            }
        )

    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch")
    else:
        st.info("לא נמצאו קבוצות של מטריצות עם אותה צורה (shape groups).")


def _render_series_analytics_section(
    npz_obj: Mapping[str, np.ndarray],
    group: ShapeGroup,
) -> None:
    """
    אנליטיקה מתקדמת לסדרה של מטריצות (matrix series) עם אותו shape:

    - Rolling correlation-strength.
    - Rolling distance correlation.
    - Rolling Mahalanobis distance (summary).
    - Rolling spectral power.
    - VAR diagnostics.
    - Dynamic conditional correlation (DCC) summary.
    """
    st.markdown("### 🧵 Series analytics (Matrix series)")

    # חייבים לפחות 2 מטריצות בסדרה כדי לעשות ניתוח סדרתי
    if len(group.names) < 2:
        st.info("כדי לבצע אנליזת סדרה צריך לפחות 2 מטריצות בקבוצה.")
        return

    st.write(f"עובדים על shape={group.shape} עם {len(group.names)} מטריצות.")

    # בונים SeriesND של מטריצות מהקבוצה
    series = _build_matrix_series_from_group(npz_obj, group)

    # פרמטרים לחלונות
    col1, col2, col3 = st.columns(3)
    with col1:
        window_corr = st.slider(
            "חלון ל-rolling correlation strength (מספר snapshots):",
            min_value=2,
            max_value=min(len(group.names), 50),
            value=min(5, len(group.names)),
            step=1,
            key=f"series_window_corr_{group.shape}",
        )
    with col2:
        window_adv = st.slider(
            "חלון ל-rolling distance / Mahalanobis:",
            min_value=2,
            max_value=min(len(group.names), 50),
            value=min(10, len(group.names)),
            step=1,
            key=f"series_window_adv_{group.shape}",
        )
    with col3:
        window_spec = st.slider(
            "חלון ל-rolling spectral analysis:",
            min_value=2,
            max_value=min(len(group.names), 50),
            value=min(10, len(group.names)),
            step=1,
            key=f"series_window_spec_{group.shape}",
        )

    # 1) rolling correlation strength
    st.subheader("📈 Rolling correlation strength")
    corr_strength = _rolling_corr_strength(series, window=window_corr)
    if corr_strength is not None and not corr_strength.empty:
        st.line_chart(corr_strength, width="stretch")
    else:
        st.info("rolling_corr_strength לא זמין (יתכן שאין matrix_helpers מלא).")

    # 2) rolling distance correlation
    st.subheader("📉 Rolling distance correlation (average)")
    dcor_series = _safe_adv_call(
        "rolling_distance_correlation",
        series,
        window=window_adv,
        average=True,
    )
    if isinstance(dcor_series, pd.Series) and not dcor_series.empty:
        st.line_chart(dcor_series, width="stretch")
    else:
        st.info("rolling_distance_correlation לא זמין מתוך advanced_metrics.")

    # 3) rolling Mahalanobis distance
    st.subheader("📏 Rolling Mahalanobis distance (summary)")
    maha_series = _safe_adv_call(
        "rolling_mahalanobis_distance",
        series,
        window=window_adv,
    )
    if isinstance(maha_series, pd.Series) and not maha_series.empty:
        avg_maha = maha_series.apply(
            lambda m: float(np.nanmean(np.asarray(m))) if isinstance(m, np.ndarray) else np.nan
        )
        st.line_chart(avg_maha, width="stretch")
    else:
        st.info("rolling_mahalanobis_distance לא זמין מתוך advanced_metrics.")

    # 4) rolling spectral analysis (power only)
    st.subheader("🔊 Rolling spectral power (aggregate)")
    spec_series = _safe_adv_call(
        "rolling_spectral_analysis",
        series,
        window=window_spec,
        axis=0,
        power_only=True,
    )
    if isinstance(spec_series, pd.Series) and not spec_series.empty:
        # כל איבר הוא ספקטרום; נסכם לכל snapshot את סכום הכוח הכולל
        power_series = spec_series.apply(
            lambda spec: float(np.nansum(np.asarray(spec))) if isinstance(spec, np.ndarray) else np.nan
        )
        st.line_chart(power_series, width="stretch")
    else:
        st.info("rolling_spectral_analysis לא זמין מתוך advanced_metrics.")

    # 5) VAR diagnostics
    st.subheader("📚 VAR diagnostics")
    try:
        var_diag = _safe_adv_call("var_diagnostics", series, maxlags=5)
    except Exception:
        var_diag = None

    if isinstance(var_diag, pd.DataFrame) and not var_diag.empty:
        st.dataframe(var_diag, width="stretch")
    else:
        st.info("var_diagnostics לא זמין / נכשל.")

    # 6) Dynamic conditional correlation (DCC)
    st.subheader("🔗 Dynamic conditional correlation (DCC-GARCH)")
    try:
        dcc_series = _safe_adv_call("dynamic_conditional_correlation", series, p=1, q=1)
    except Exception:
        dcc_series = None

    if isinstance(dcc_series, pd.Series) and not dcc_series.empty:
        # כל איבר: מטריצת קורלציה; נסכם את ממוצע |corr| מחוץ לדיאגונל
        def _avg_offdiag(mat: np.ndarray) -> float:
            mat = np.asarray(mat, dtype=float)
            if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
                return np.nan
            mask = ~np.eye(mat.shape[0], dtype=bool)
            vals = np.abs(mat[mask])
            return float(np.nanmean(vals)) if vals.size else np.nan

        dcc_strength = dcc_series.apply(_avg_offdiag)
        st.line_chart(dcc_strength, width="stretch")
    else:
        st.info("dynamic_conditional_correlation לא זמין / נכשל (חסר arch.multivariate?).")

# ===================== Risk & Pair Analytics / AutoML hooks =====================

# ---- ייבוא כלים מהרמה הגלובלית (common.utils / common.helpers / common.automl_tools) ----

try:
    from common.utils import (  # type: ignore[attr-defined]
        basic_pair_metrics,
        evaluate_pair_quality,
        metrics_to_json,
        risk_parity_weights,
    )
except Exception:  # pragma: no cover
    def basic_pair_metrics(
        x: pd.Series,
        y: pd.Series,
        *,
        z_window: int = 20,
    ) -> Dict[str, float]:
        """Fallback basic pair metrics – גרסה מצומצמת."""
        sx = x.astype(float)
        sy = y.astype(float)
        idx = sx.index.intersection(sy.index)
        sx = sx.loc[idx]
        sy = sy.loc[idx]
        if len(sx) < 30:
            return {k: float("nan") for k in (
                "corr", "half_life", "zscore", "vol_z", "hurst", "adf_pvalue", "hedge"
            )}
        corr = sx.corr(sy)
        spread = np.log(sx) - np.log(sy)
        zscore = (spread - spread.mean()) / spread.std(ddof=1)
        return {
            "corr": float(corr),
            "half_life": float("nan"),
            "zscore": float(zscore.iloc[-1]),
            "vol_z": float("nan"),
            "hurst": float("nan"),
            "adf_pvalue": float("nan"),
            "hedge": 1.0,
        }

    def evaluate_pair_quality(*args, **kwargs) -> Dict[str, float]:
        """Fallback – מחזיר dict ריק אם evaluate_pair_quality לא קיים."""
        return {}

    def metrics_to_json(metrics: Dict[str, Any]) -> Dict[str, Any]:
        return dict(metrics)

    def risk_parity_weights(assets: pd.DataFrame) -> pd.Series:
        rets = assets.pct_change().dropna()
        vols = rets.std()
        inv_vol = 1 / vols.replace(0, np.nan)
        w = inv_vol / inv_vol.sum()
        return w.fillna(0.0)


# risk_metrics מגיע מ-common.helpers (שם הוא מוגדר בפועל)
try:
    from common.helpers import risk_metrics  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    def risk_metrics(
        returns: Iterable[float] | np.ndarray,
        alpha: float = 0.05,
        window: int | None = None,
    ) -> Dict[str, float | np.ndarray]:
        """Fallback risk_metrics – מחזיר רק stdev/mean כמדד בסיסי."""
        arr = np.asarray(list(returns), dtype=float)
        if arr.size == 0:
            return {
                "VaR": np.nan,
                "CVaR": np.nan,
                "max_drawdown": np.nan,
                "ulcer_index": np.nan,
            }
        if window is not None and window < arr.size:
            arr = arr[-window:]
        mean = float(np.nanmean(arr))
        std = float(np.nanstd(arr, ddof=1))
        return {
            "VaR": mean - 1.65 * std,
            "CVaR": mean - 2.0 * std,
            "max_drawdown": np.nan,
            "ulcer_index": np.nan,
        }


# AutoML / PyCaret / Explainability — אם זמין
try:
    from common.automl_tools import (  # type: ignore[attr-defined]
        run_pycaret_regression,
        leaderboard_plot,
        trials_to_features,
        ensure_figure as automl_ensure_figure,
    )
except Exception:  # pragma: no cover
    run_pycaret_regression = None  # type: ignore[assignment]
    leaderboard_plot = None  # type: ignore[assignment]
    trials_to_features = None  # type: ignore[assignment]

    def automl_ensure_figure(fig: Any) -> Any:  # type: ignore[no-redef]
        return fig


# ===================== Risk helpers =====================


def _build_equal_weight_portfolio(df: pd.DataFrame, cols: Sequence[str]) -> pd.Series:
    """
    בונה פורטפוליו equal-weight על subset של עמודות (מחירי נכסים).

    מניח שהעמודות הן מחירי נכסים, ומחזיר price series.
    """
    if not cols:
        raise ValueError("אין עמודות שנבחרו לפורטפוליו.")
    sub = df[cols].dropna(how="all")
    if sub.empty:
        raise ValueError("אחרי dropna אין נתונים לפורטפוליו.")
    portfolio = sub.mean(axis=1)
    portfolio.name = "portfolio_price"
    return portfolio


def _build_risk_parity_portfolio(df: pd.DataFrame, cols: Sequence[str]) -> Tuple[pd.Series, pd.Series]:
    """
    בונה פורטפוליו risk-parity פשוט:
    - משקלות risk_parity_weights
    - מחיר פורטפוליו כ-w·P.
    """
    if not cols:
        raise ValueError("אין עמודות שנבחרו לפורטפוליו.")
    sub = df[cols].dropna(how="all")
    if sub.empty:
        raise ValueError("אחרי dropna אין נתונים לפורטפוליו.")

    w = risk_parity_weights(sub)
    # נדרג מחדש על subset בלבד
    w = w.loc[[c for c in cols if c in w.index]].copy()
    w /= w.sum() if w.sum() != 0 else 1.0

    # מחיר פורטפוליו: Σ w_i * P_i
    port = (sub * w.reindex(sub.columns).fillna(0.0)).sum(axis=1)
    port.name = "rp_portfolio_price"
    return port, w


def _compute_risk_contributions(w: pd.Series, cov: np.ndarray, asset_names: Sequence[str]) -> pd.DataFrame:
    """
    מחשב תרומת סיכון (Risk Contribution) של כל נכס לפיזור הפורטפוליו.

    RC_i = w_i * (Σ_j Cov_ij * w_j) / Var(portfolio)
    """
    w_vec = w.reindex(asset_names).fillna(0.0).values.astype(float)
    cov = np.asarray(cov, dtype=float)
    if cov.shape != (len(asset_names), len(asset_names)):
        raise ValueError("צורת cov לא תואמת למספר הנכסים.")

    port_var = float(w_vec @ cov @ w_vec.T)
    if port_var <= 0:
        rc = np.full_like(w_vec, np.nan, dtype=float)
    else:
        # marginal contribution = Cov * w
        mc = cov @ w_vec
        rc = w_vec * mc / port_var

    df_rc = pd.DataFrame(
        {
            "weight": w_vec,
            "risk_contrib": rc,
            "risk_contrib_pct": rc / rc.sum() if np.isfinite(rc).any() else np.nan,
        },
        index=list(asset_names),
    )
    return df_rc


# ===================== Pair quality & risk-parity =====================


def _render_pair_quality_section(df: pd.DataFrame) -> None:
    """
    ניתוח איכות זוג + risk-parity לזוג:
    - בחירת שני נכסים.
    - basic_pair_metrics (corr, half-life, z-score, vol_z, hurst, adf).
    - evaluate_pair_quality (score / diagnostics).
    - risk-parity weights לזוג + תרומת סיכון אם נבנה פורטפוליו.
    """
    st.markdown("### 1️⃣1️⃣ Pair quality & risk-parity (HF-level)")

    if df.shape[1] < 2:
        st.info("צריך לפחות שני נכסים (עמודות) כדי לנתח זוג.")
        return

    cols = list(df.columns)
    default_pair = cols[:2]

    selected = st.multiselect(
        "בחר שני נכסים לזוג (pair):",
        options=cols,
        default=default_pair,
        key="matrix_pair_quality_cols",
    )
    if len(selected) != 2:
        st.info("בחר *בדיוק* שני נכסים.")
        return

    col_a, col_b = selected
    pair_df = df[[col_a, col_b]].dropna()
    if pair_df.empty:
        st.info("אין מספיק נתונים לשני הנכסים שנבחרו.")
        return

    x = pair_df[col_a]
    y = pair_df[col_b]

    # Basic pair metrics מהמנוע המרכזי
    basics = basic_pair_metrics(x, y)
    quality = evaluate_pair_quality(pair_df) or {}
    metrics_row = {**basics, **quality}

    # ---- Cross-tab navigation: open this pair in Pair Analysis tab ----
    pair_name = f"{col_a}/{col_b}"
    nav_col1, nav_col2 = st.columns([2, 1])
    with nav_col1:
        st.caption(f"🔗 זוג נוכחי: **{pair_name}**")
    with nav_col2:
        if set_nav_target is not None and st.button(
            "🔍 עבור לטאב הזוג עם זוג זה",
            key=f"matrix_go_to_pair_{col_a}_{col_b}",
        ):
            try:
                # ה-router בדשבורד כבר יודע למפות 'pair' לטאב הרלוונטי
                set_nav_target("pair", {"pair": pair_name})
                st.success("הגדרתי יעד ניווט לטאב הזוג. עבור לטאב Pair Analysis לראות פרטים.")
            except Exception as exc:
                st.warning(f"לא הצלחתי להגדיר nav_target לטאב הזוג: {exc}")

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Corr (log)", f"{metrics_row.get('corr', float('nan')):.3f}")
    k2.metric("Half-life", f"{metrics_row.get('half_life', float('nan')):.1f}")
    k3.metric("Z-score (last)", f"{metrics_row.get('zscore', float('nan')):.2f}")
    k4.metric("Vol Z", f"{metrics_row.get('vol_z', float('nan')):.2f}")

    st.subheader("📏 Pair metrics (full)")
    st.dataframe(pd.DataFrame([metrics_row]), width="stretch")

    # Risk-parity weights על זוג
    try:
        rp_weights = risk_parity_weights(pair_df)
        rp_weights = rp_weights.loc[[col_a, col_b]]
    except Exception:
        rp_weights = None

    st.subheader("⚖️ Risk-parity weights (pair)")
    if rp_weights is not None and not rp_weights.empty:
        st.dataframe(
            rp_weights.to_frame("weight").style.format("{:.3f}"),
            width="stretch",
        )
    else:
        st.info("risk_parity_weights לא זמין / נכשל.")

def _render_pair_grid_quality_section(df: pd.DataFrame) -> None:
    """
    Pair quality grid: ניתוח המוני של זוגות (multi-pair ranking)
    על subset של נכסים:

    - משתמש ב-basic_pair_metrics + evaluate_pair_quality אם זמין.
    - ממיין זוגות לפי score / corr / edge וכו'.
    """
    st.markdown("### 1️⃣1️⃣b Pair quality grid (multi-pair ranking)")

    if df.shape[1] < 2:
        st.info("צריך לפחות שני נכסים (עמודות) כדי לדרג זוגות.")
        return

    cols = list(df.columns)
    selected = st.multiselect(
        "בחר subset של נכסים לניתוח זוגות (עדיף לא יותר מדי לצורך ביצועים):",
        options=cols,
        default=cols[: min(8, len(cols))],
        key="matrix_pair_grid_cols",
    )

    if len(selected) < 2:
        st.info("בחר לפחות שני נכסים כדי להרכיב זוגות.")
        return

    all_pairs = list(combinations(selected, 2))
    n_pairs = len(all_pairs)
    if n_pairs == 0:
        st.info("לא הצלחנו להרכיב זוגות מתוך הבחירה.")
        return

    max_default = min(100, n_pairs)
    max_pairs = st.slider(
        "מקסימום זוגות לחישוב (לצורך ביצועים):",
        min_value=1,
        max_value=n_pairs,
        value=max_default,
        step=1,
        key="matrix_pair_grid_max_pairs",
    )

    # סאבסמפול אם יש יותר מדי זוגות
    if n_pairs > max_pairs:
        rng = np.random.default_rng(42)
        idx = np.sort(rng.choice(n_pairs, size=max_pairs, replace=False))
        pairs_iter = [all_pairs[i] for i in idx]
    else:
        pairs_iter = all_pairs

    rows: list[dict[str, Any]] = []
    for a, b in pairs_iter:
        pair_df = df[[a, b]].dropna()
        if pair_df.shape[0] < 30:
            continue

        x = pair_df[a]
        y = pair_df[b]

        try:
            basics = basic_pair_metrics(x, y)
        except Exception:  # pragma: no cover
            basics = {}

        try:
            quality = evaluate_pair_quality(pair_df) or {}
        except Exception:  # pragma: no cover
            quality = {}

        merged: dict[str, Any] = {}
        merged.update(basics)
        merged.update(quality)

        row: dict[str, Any] = {
            "pair": f"{a}/{b}",
            "asset_a": a,
            "asset_b": b,
        }

        for k, v in merged.items():
            if isinstance(v, (int, float, np.number)):
                row[str(k)] = float(v)
            else:
                row[str(k)] = v

        rows.append(row)

    if not rows:
        st.info("לא הצלחנו לחשב מדדים לזוגות (מעט מדי נתונים או כשל בחישוב).")
        return

    grid_df = pd.DataFrame(rows)

    sort_candidates = [c for c in ["score", "pair_score", "edge", "corr"] if c in grid_df.columns]
    sort_col: Optional[str] = None
    if sort_candidates:
        sort_col = st.selectbox(
            "בחר עמודת דירוג עיקרית:",
            options=sort_candidates,
            index=0,
            key="matrix_pair_grid_sort",
        )
        ascending = sort_col in ["adf_pvalue"]
        grid_df = grid_df.sort_values(sort_col, ascending=ascending)

    st.dataframe(grid_df, width="stretch")

    csv_data = grid_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ הורד טבלת זוגות (CSV)",
        data=csv_data,
        file_name="matrix_pair_quality_grid.csv",
        mime="text/csv",
        key="matrix_pair_grid_csv",
    )

# ===================== Portfolio returns & risk =====================


def _render_returns_risk_section(df: pd.DataFrame) -> None:
    """
    ניתוח תשואות וסיכון ברמת מטריצה:
    - בחירת subset של נכסים לבניית פורטפוליו.
    - בחירת שיטת בניית פורטפוליו (equal-weight / risk-parity).
    - חישוב simple/log returns.
    - Rolling Sharpe / Sortino.
    - Risk metrics (VaR / CVaR / max drawdown / ulcer index) בשלושה אופקי זמן.
    - Risk contributions לפי cov.
    - שמירת snapshots של תיקים והשוואה ביניהם.
    - Cross-tab nav לטאבים Portfolio / Backtest.
    - חיבור ל-AppContext / Action Playbook.
    """
    st.markdown("### 1️⃣2️⃣ Portfolio returns, risk & contributions")

    if df.shape[1] == 0:
        st.info("אין נכסים לניתוח.")
        return

    cols = list(df.columns)
    selected = st.multiselect(
        "בחר נכסים לפורטפוליו (אם לא תבחר כלום → כל העמודות):",
        options=cols,
        default=cols[: min(6, len(cols))],
        key="matrix_portfolio_cols",
    )
    if not selected:
        selected = cols

    col1, col2, col3 = st.columns(3)
    with col1:
        portfolio_mode = st.selectbox(
            "שיטת בניית פורטפוליו:",
            options=["Equal-weight", "Risk-parity"],
            index=0,
            key="matrix_portfolio_mode",
        )
    with col2:
        rf = st.number_input(
            "Risk-free annual rate (אחוז, לשימוש ב-Sharpe/Sortino):",
            min_value=-5.0,
            max_value=10.0,
            value=0.0,
            step=0.25,
            key="matrix_portfolio_rf",
        ) / 100.0
    with col3:
        annualization = st.selectbox(
            "מספר תקופות בשנה:",
            options=[252, 365, 52, 12],
            index=0,
            format_func=lambda x: f"{x} (trading days/year)" if x == 252 else str(x),
            key="matrix_portfolio_annualization",
        )

    # בניית פורטפוליו
    try:
        if portfolio_mode == "Equal-weight":
            portfolio_price = _build_equal_weight_portfolio(df, selected)
            weights = pd.Series(
                1.0 / len(selected),
                index=selected,
                name="weight",
            )
        else:
            portfolio_price, weights = _build_risk_parity_portfolio(df, selected)
    except Exception as exc:
        logger.warning("Portfolio construction failed: %s", exc)
        st.warning(f"נכשל בבניית פורטפוליו: {exc}")
        return

    st.subheader("💰 Portfolio price curve")
    st.line_chart(portfolio_price, width="stretch")

    # תשואות
    simple_ret = portfolio_price.pct_change().dropna()
    log_ret = np.log(portfolio_price / portfolio_price.shift(1)).dropna()
    simple_ret.name = "simple_return"
    log_ret.name = "log_return"

    c1, c2 = st.columns(2)
    with c1:
        st.caption("Simple returns")
        st.line_chart(simple_ret, width="stretch")
    with c2:
        st.caption("Log returns")
        st.line_chart(log_ret, width="stretch")

    # Rolling Sharpe / Sortino
    window = st.slider(
        "חלון rolling ל-Sharpe/Sortino (מספר תקופות):",
        min_value=20,
        max_value=252,
        value=63,
        step=5,
        key="matrix_rolling_window",
    )

    excess = simple_ret - rf / annualization
    mu = excess.rolling(window).mean()
    sd = excess.rolling(window).std()
    downside = excess.copy()
    downside[downside > 0] = 0.0
    ds = downside.rolling(window).std()

    sharpe = (mu / sd) * np.sqrt(annualization)
    sortino = (mu / ds) * np.sqrt(annualization)
    rs_df = pd.concat(
        [sharpe.rename("rolling_sharpe"), sortino.rename("rolling_sortino")],
        axis=1,
    )
    st.subheader("📊 Rolling Sharpe / Sortino")
    st.line_chart(rs_df, width="stretch")

    # Risk metrics בשלושה אופקים: כל התקופה, 1Y, 3M
    st.subheader("🛡 Risk metrics (VaR / CVaR / Max DD / Ulcer index)")

    horiz = {
        "Full period": None,
        "Last 1Y (~252)": 252,
        "Last 3M (~63)": 63,
    }
    rows: list[dict[str, Any]] = []
    for label, win in horiz.items():
        try:
            rm = risk_metrics(simple_ret.values, alpha=0.05, window=win)
        except Exception as exc:
            logger.warning("risk_metrics failed for horizon %s: %s", label, exc)
            rm = {}
        row = {"horizon": label}
        for k in ("VaR", "CVaR", "max_drawdown", "ulcer_index"):
            v = rm.get(k, np.nan)
            if isinstance(v, np.ndarray):
                v = float(v[~np.isnan(v)][-1]) if np.any(~np.isnan(v)) else np.nan
            row[k] = float(v) if isinstance(v, (int, float, np.number)) else np.nan
        rows.append(row)

    st.dataframe(pd.DataFrame(rows), width="stretch")

    # Risk contributions לפי cov של נכסים
    st.subheader("📡 Risk contributions by asset")
    try:
        sub = df[selected].dropna()
        rets = sub.pct_change().dropna()
        cov = rets.cov().values
        rc_df = _compute_risk_contributions(weights, cov, selected)
        st.dataframe(
            rc_df.style.format(
                {
                    "weight": "{:.3f}",
                    "risk_contrib": "{:.4f}",
                    "risk_contrib_pct": "{:.2%}",
                }
            ),
            width="stretch",
        )
    except Exception as exc:
        logger.warning("risk contribution computation failed: %s", exc)
        st.info("לא הצלחנו לחשב תרומות סיכון – ייתכן שיש מעט מדי נתונים או קובאריאנס לא תקין.")

    # ---- שמירת תיקים והשוואה ביניהם ----
    with st.expander("📊 Save / compare portfolios (Matrix Research)", expanded=False):
        saved: Dict[str, Dict[str, Any]] = st.session_state.get(
            "matrix_saved_portfolios", {}
        )

        default_name = f"matrix_portfolio_{len(saved) + 1}"
        name = st.text_input(
            "שם תיק לשמירה:",
            value=default_name,
            key="matrix_portfolio_save_name",
        )

        if st.button("💾 שמור snapshot של התיק הנוכחי", key="matrix_portfolio_save_btn"):
            # KPIs עיקריים – מחושבים ישירות כאן
            try:
                if len(portfolio_price) > 1:
                    total_pnl = float(
                        portfolio_price.iloc[-1] / portfolio_price.iloc[0] - 1.0
                    )
                else:
                    total_pnl = 0.0
            except Exception:
                total_pnl = 0.0

            try:
                sh_clean = sharpe.dropna()
                last_sharpe_val = float(sh_clean.iloc[-1]) if not sh_clean.empty else float("nan")
            except Exception:
                last_sharpe_val = float("nan")

            var_full_val = np.nan
            cvar_full_val = np.nan
            max_dd_val = np.nan
            for row in rows:
                if row.get("horizon") == "Full period":
                    var_full_val = row.get("VaR", np.nan)
                    cvar_full_val = row.get("CVaR", np.nan)
                    max_dd_val = row.get("max_drawdown", np.nan)
                    break

            snapshot: Dict[str, Any] = {
                "total_pnl": total_pnl,
                "sharpe": last_sharpe_val,
                "max_dd": max_dd_val,
                "var_full": var_full_val,
                "cvar_full": cvar_full_val,
                "n_assets": len(selected),
                "mode": portfolio_mode,
            }

            new_saved = dict(saved)
            new_saved[name] = snapshot
            st.session_state["matrix_saved_portfolios"] = new_saved
            st.success(f"תיק `{name}` נשמר להשוואה.")

        saved = st.session_state.get("matrix_saved_portfolios", {})

        if saved:
            st.markdown("#### 🧾 Saved portfolio snapshots")
            comp_df = (
                pd.DataFrame.from_dict(saved, orient="index")
                .reset_index()
                .rename(columns={"index": "name"})
            )
            st.dataframe(comp_df, width="stretch")

            if st.button("🧹 נקה את כל ה-snapshots", key="matrix_portfolio_clear_saves"):
                st.session_state["matrix_saved_portfolios"] = {}
                st.success("ניקיתי את רשימת התיקים השמורים.")
        else:
            st.caption("אין עדיין תיקי Matrix Research שמורים להשוואה.")

    # ---- Cross-tab navigation: send this portfolio to Portfolio/Fund tab + Backtest ----
    if set_nav_target is not None and weights is not None:
        with st.expander(
            "📂 שלח תיק זה לטאב הפורטפוליו / הקרן / ה-Backtest",
            expanded=False,
        ):
            st.caption("המשקולות מחושבות לפי הבחירה שלך (Equal-weight / Risk-parity וכו').")

            if st.button(
                "🚀 שלח לטאב Portfolio / Fund View",
                key="matrix_send_portfolio_to_fund",
            ):
                try:
                    payload = {
                        "symbols": list(weights.index),
                        "weights": {k: float(v) for k, v in weights.items()},
                        "source": "matrix_research",
                    }
                    set_nav_target("portfolio", payload)
                    st.success(
                        "התיק נשלח כ-nav_target לטאב הפורטפוליו. פתח את הטאב שם לניתוח עמוק."
                    )
                except Exception as exc:
                    st.warning(f"לא הצלחתי להגדיר nav_target לטאב הפורטפוליו: {exc}")

            if st.button(
                "📈 שלח לטאב Backtest (עם המשקולות האלה)",
                key="matrix_send_portfolio_to_backtest",
            ):
                try:
                    bt_payload = {
                        "symbols": list(weights.index),
                        "weights": {k: float(v) for k, v in weights.items()},
                        "source": "matrix_research",
                        "mode": "portfolio",
                    }
                    set_nav_target("backtest", bt_payload)
                    st.success(
                        "שלחתי את התיק כ-nav_target לטאב ה-Backtest. עבור לטאב Backtest כדי להריץ סימולציה."
                    )
                except Exception as exc:
                    st.warning(f"לא הצלחתי להגדיר nav_target לטאב ה-Backtest: {exc}")

    # -------- חיבור ל-AppContext: Action Playbook מבוסס Matrix KPIs --------
    try:
        app_ctx = get_current_ctx()
    except Exception as exc:
        logger.warning("get_current_ctx failed in _render_returns_risk_section: %s", exc)
        app_ctx = None

    if app_ctx is not None and AppContext is not None:
        # KPIs בסיסיים מתוך הפורטפוליו
        try:
            if len(portfolio_price) > 1:
                total_pnl = float(
                    portfolio_price.iloc[-1] / portfolio_price.iloc[0] - 1.0
                )
            else:
                total_pnl = 0.0
        except Exception as exc:
            logger.warning("Failed to compute total_pnl from portfolio_price: %s", exc)
            total_pnl = 0.0

        try:
            last_sharpe = (
                float(sharpe.dropna().iloc[-1]) if not sharpe.dropna().empty else None
            )
        except Exception as exc:
            logger.warning("Failed to extract last_sharpe: %s", exc)
            last_sharpe = None

        try:
            last_dd = None
            for row in rows:
                if row.get("horizon") == "Full period":
                    last_dd = row.get("max_drawdown")
                    break
        except Exception as exc:
            logger.warning("Failed to extract max_drawdown from rows: %s", exc)
            last_dd = None

        # נוסיף גם VaR / CVaR של התקופה המלאה ל-KPIs של הניסוי
        var_full = None
        cvar_full = None
        try:
            for row in rows:
                if row.get("horizon") == "Full period":
                    var_full = row.get("VaR")
                    cvar_full = row.get("CVaR")
                    break
        except Exception as exc:
            logger.warning("Failed to extract VaR/CVaR from rows: %s", exc)

        kpis_for_ctx: Dict[str, Any] = {
            "total_pnl": total_pnl,
            "sharpe": last_sharpe,
            "max_dd": last_dd,
        }

        # KPIs מורחבים יותר לרישום הניסוי
        kpis_for_exp: Dict[str, Any] = dict(kpis_for_ctx)
        kpis_for_exp.update(
            {
                "var_full": var_full,
                "cvar_full": cvar_full,
                "n_assets": len(selected),
                "portfolio_mode": portfolio_mode,
            }
        )

        # רישום הניסוי ב-_experiment_runs
        try:
            register_experiment_run(
                app_ctx,
                kpis=kpis_for_exp,
                extra_meta={
                    "source": "matrix_research_tab",
                    "portfolio_mode": portfolio_mode,
                    "selected_assets": selected,
                    "annualization": annualization,
                },
            )
        except Exception as exc:
            logger.warning("register_experiment_run failed in matrix tab: %s", exc)

        # הפקת Playbook ברמת קרן
        try:
            playbook = ctx_to_action_playbook(app_ctx, kpis=kpis_for_ctx)
        except Exception as exc:
            logger.warning("ctx_to_action_playbook failed in matrix tab: %s", exc)
            playbook = {}

        with st.expander(
            "🧭 Action Playbook (Fund-level, בהתבסס על Matrix Portfolio)",
            expanded=False,
        ):
            if playbook and playbook.get("actions"):
                for act in playbook["actions"]:
                    if not isinstance(act, dict):
                        continue
                    code = act.get("code", "action")
                    text = act.get("text", "")
                    prio = act.get("priority", "medium")
                    prefix = f"[{prio.upper()}] " if prio else ""
                    if text:
                        st.markdown(f"- {prefix}{text}  \n  `({code})`")
                    else:
                        st.markdown(f"- {prefix}`{code}`")
                st.markdown("**Summary (machine-readable):**")
                st.json(playbook.get("summary", {}))
            else:
                st.info("לא הופק Playbook (חסר AppContext או KPIs).")

# ===================== AutoML sandbox =====================


def _render_automl_quick_section(df: pd.DataFrame) -> None:
    """
    Quick AutoML (PyCaret) על מטריצת features שנבחרה מתוך ה-DataFrame:

    - המשתמש בוחר עמודת יעד (target) ועמודות features.
    - אם common.automl_tools + PyCaret זמינים → run_pycaret_regression(df).
    - מציג leaderboard + גרף bar דרך leaderboard_plot אם קיים.
    - מיועד לניתוח cross-sectional / snapshot של מטריצה כ-feature space.
    """
    st.markdown("### 1️⃣3️⃣ AutoML sandbox (PyCaret regression)")

    if run_pycaret_regression is None:
        st.info("AutoML (PyCaret) לא זמין בסביבה זו — common.automl_tools לא importable.")
        return

    if df.shape[1] < 2:
        st.info("צריך לפחות שתי עמודות (target + לפחות feature אחד).")
        return

    cols = list(df.columns)

    col1, col2 = st.columns(2)
    with col1:
        target_col = st.selectbox(
            "בחר עמודת יעד (target) ל-AutoML:",
            options=cols,
            index=len(cols) - 1 if len(cols) > 1 else 0,
            key="matrix_automl_target",
        )
    with col2:
        feature_cols = st.multiselect(
            "בחר עמודות כ-features (אם ריק → כל מה שלא target):",
            options=[c for c in cols if c != target_col],
            default=[c for c in cols if c != target_col][: min(10, len(cols) - 1)],
            key="matrix_automl_features",
        )

    if not feature_cols:
        feature_cols = [c for c in cols if c != target_col]

    data = df[feature_cols + [target_col]].dropna()
    if data.empty:
        st.info("אחרי dropna אין שורות ל-AutoML.")
        return

    st.caption(f"Dataset ל-AutoML: {data.shape[0]} שורות, {len(feature_cols)} features + 1 target.")
    st.dataframe(data.head(50), width="stretch")

    metric = st.selectbox(
        "Metric למיון מודלים:",
        options=["MAE", "RMSE", "R2"],
        index=0,
        key="matrix_automl_metric",
    )

    if not st.button("🚀 Run PyCaret AutoML (quick)", key="matrix_automl_run"):
        return

    with st.spinner("מריץ PyCaret AutoML..."):
        try:
            # לפי automl_tools: run_pycaret_regression(df)
            leaderboard = run_pycaret_regression(
                data.rename(columns={target_col: "target"})
            )  # type: ignore[operator]
        except Exception as exc:
            st.error(f"AutoML נכשל: {exc}")
            return

    st.subheader("🏆 Leaderboard (PyCaret)")
    st.dataframe(leaderboard, width="stretch")

    if leaderboard_plot is not None:
        try:
            fig = leaderboard_plot(leaderboard, metric=metric)  # type: ignore[operator]
            fig = automl_ensure_figure(fig)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as exc:
            st.info(f"leaderboard_plot נכשל: {exc}")

    # אופציונלי: המרה ל-"feature importance table" דרך trials_to_features אם רלוונטי
    if trials_to_features is not None and st.checkbox(
        "להפיק טבלת features/metrics מתוך leaderboard (אם תומך)?",
        value=False,
        key="matrix_automl_features_table_toggle",
    ):
        try:
            feats_df = trials_to_features(leaderboard, drop_nan=True)  # type: ignore[operator]
            st.subheader("🔍 AutoML features / tuning table")
            st.dataframe(feats_df, width="stretch")
        except Exception as exc:
            st.info(f"trials_to_features נכשל: {exc}")
# ===================== Orchestrator (HF-grade) =====================
import io


def render_matrix_research_tab(
    pairs: Optional[List[Any]] = None,
    config: Optional[Dict[str, Any]] = None,
    ctx: Optional[Mapping[str, Any]] = None,
    controls: Optional[Mapping[str, Any]] = None,
) -> None:
    """
    Main Streamlit tab for matrix / correlation research (HF-grade).

    Flow high-level
    ----------------
    1. העלאת קובץ ‎.npz‎ והפקת descriptors לכל המטריצות.
    2. הצגת Inventory כולל אפשרות להוריד CSV + סיכום לפי kind/shape.
    3. בחירת preset / workflow (Pair / Portfolio / Series) מה-Sidebar שמכוון את ה-UI.
    4. טאבים פנימיים:
       a. 📌 Single matrix analytics
          - Preview + summarize_series + diagnostics + KPIs מהירים.
          - Correlation / Covariance (+ Clustered / Advanced דרך פונקציות העזר).
          - PCA & components (עם Light-mode לסקייל גדול).
          - Advanced pair & matrix metrics.
          - Volatility & shrinkage.
          - Entropy & complexity.
          - Pair quality & risk-parity.
          - Portfolio returns & risk (כולל חיבור ל-Portfolio / Backtest / AppContext).
          - AutoML sandbox.
          - Export של דו"ח מטריצה בודדת (CSV/Markdown).
       b. 🧵 Matrix series analytics
          - Overview על shape groups.
          - בחירת shape group (עם שליטה על גודל מקסימלי).
          - Rolling correlation strength.
          - Rolling distance correlation.
          - Rolling Mahalanobis distance.
          - Rolling spectral power.
          - VAR diagnostics.
          - Dynamic conditional correlation (DCC) summary.
    """

    # ------------------------------------------------------------------
    # Sidebar — הקשר גלובלי + Presets + Feature flags
    # ------------------------------------------------------------------
    st.sidebar.markdown("### 🔬 Matrix Research — Controls")

    # מידע על backend (CPU/GPU וכו') אם זמין
    try:
        be_info = backend_info()
    except Exception:
        be_info = {"backend": "unknown"}
    st.sidebar.markdown("**Backend info:**")
    st.sidebar.json(be_info, expanded=False)

    # Presets / Workflows (מכוונים את פתיחת ה-expanders והדגשים)
    analysis_preset = st.sidebar.selectbox(
        "Analysis preset / workflow",
        [
            "Manual / custom",
            "Pair selection workflow",
            "Portfolio risk deep dive",
            "Matrix series & regimes",
        ],
        index=0,
        key="matrix_analysis_preset",
    )

    source = st.sidebar.selectbox(
        "Data source",
        ["Uploaded .npz", "Dashboard universe / prices"],
        index=0,
        key="matrix_data_source",
    )

    # דגלים כלליים (כרגע בעיקר אינפורמטיביים, אפשר להרחיב לשימוש פנימי)
    show_heavy_series = st.sidebar.checkbox(
        "לאפשר חישובי סדרה כבדים (VAR / DCC / spectral)",
        value=True,
        key="matrix_enable_heavy_series",
    )
    show_automl = st.sidebar.checkbox(
        "להציג AutoML sandbox (PyCaret)",
        value=True,
        key="matrix_enable_automl",
    )
    light_mode = st.sidebar.checkbox(
        "Light mode (להימנע מחישובים כבדים על מטריצות ענק)",
        value=False,
        key="matrix_light_mode",
    )

    # דגלים נגזרים מה-preset
    is_pair_flow = analysis_preset == "Pair selection workflow"
    is_portfolio_flow = analysis_preset == "Portfolio risk deep dive"
    is_series_flow = analysis_preset == "Matrix series & regimes"

    # הקשר מה-dashboard (ctx) ברמת UI
    if ctx is not None and isinstance(ctx, Mapping):
        with st.sidebar.expander("📎 Dashboard context (read-only)", expanded=False):
            ctx_items = list(ctx.items())
            preview_ctx = dict(ctx_items[:10])
            st.json(preview_ctx)

    # הצגת AppContext האמיתי (אם קיים) + live readiness ברמת קרן
    try:
        app_ctx = get_current_ctx()
    except Exception:
        app_ctx = None

    if app_ctx is not None and AppContext is not None:
        with st.sidebar.expander("🧠 AppContext / Live readiness", expanded=False):
            try:
                st.json(app_ctx.short_summary())
            except Exception:
                pass

            try:
                live_info = compute_live_readiness(app_ctx)
                st.markdown("**Live readiness snapshot:**")
                st.json(live_info)
            except Exception:
                st.info("לא הצלחנו לחשב live_readiness מתוך AppContext.")

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------
    st.header("🔬 Matrix / Correlation Research")
    st.caption(
        "טאב מחקר מטריצות קורלציה / קובאריאנס / תשואות ברמת קרן גידור — "
        "עם ניתוח מספרי, סיכון, AutoML ו־Workflow-ים מוגדרים מראש."
    )

    if analysis_preset != "Manual / custom":
        preset_msg = {
            "Pair selection workflow": "פוקוס על בחירת זוגות, איכות זוג ורשת של pairs.",
            "Portfolio risk deep dive": "פוקוס על קורלציה, קובאריאנס, PCA וריסק של פורטפוליו.",
            "Matrix series & regimes": "פוקוס על סדרת מטריצות, rolling metrics ומשטרי שוק.",
        }.get(analysis_preset, "")
        if preset_msg:
            st.info(f"Preset פעיל: **{analysis_preset}** — {preset_msg}")

    if source == "Dashboard universe / prices":
        st.info(
            "מצב 'Dashboard universe / prices' מוכן לחיבור ל-Universe/SqlStore, "
            "אבל כרגע הטאב עדיין משתמש בקובץ ‎.npz‎ שמועלה כאן כמקור מטריצות."
        )

    # ------------------------------------------------------------------
    # 1. העלאת קובץ npz
    # ------------------------------------------------------------------
    uploaded_file = st.file_uploader(
        "העלה קובץ ‎.npz‎ המכיל מטריצות (correlation / covariance / returns / custom):",
        type=["npz"],
        key="matrix_npz_uploader_main",
    )

    if uploaded_file is None:
        st.info("⬆️ העלה קובץ ‎.npz‎ כדי להתחיל ניתוח.")
        return

    try:
        npz_obj = np.load(uploaded_file)
    except Exception as exc:
        st.error(f"❌ כשל בטעינת ה-npz: {exc}")
        logger.exception("Failed to load npz in matrix_research_tab")
        return

    if len(npz_obj.files) == 0:
        st.warning("⚠️ הקובץ לא מכיל מערכי numpy (npz.files ריק).")
        return

    # ------------------------------------------------------------------
    # 2. descriptors + shape groups + inventory export + סינון
    # ------------------------------------------------------------------
    descriptors = _build_descriptors(npz_obj)
    if not descriptors:
        st.warning("⚠️ לא נמצאו מטריצות מתאימות ב-npz (אין ndarray entries).")
        return

    desc_df = pd.DataFrame(
        [
            {
                "name": d.name,
                "shape": str(d.shape),
                "ndim": d.ndim,
                "kind": d.inferred_kind,
                "symmetric": d.symmetric,
                "notes": d.notes,
            }
            for d in descriptors
        ]
    ).sort_values("name")

    st.markdown("### 📚 Matrix inventory (all matrices in npz)")

    # --- Filters על ה-inventory ---
    f1, f2, f3 = st.columns([2, 1, 1])
    with f1:
        kind_values = sorted(desc_df["kind"].unique())
        kind_filter = st.multiselect(
            "סינון לפי kind (correlation / covariance / returns וכו'):",
            options=kind_values,
            default=kind_values,
            key="matrix_inventory_kind_filter",
        )
    with f2:
        ndim_min = st.number_input(
            "מינימום ndim:",
            min_value=1,
            max_value=10,
            value=1,
            step=1,
            key="matrix_inventory_ndim_min",
        )
    with f3:
        only_symmetric = st.checkbox(
            "להציג רק מטריצות symmetric",
            value=False,
            key="matrix_inventory_only_symmetric",
        )

    inv_view = desc_df.copy()
    if kind_filter:
        inv_view = inv_view[inv_view["kind"].isin(kind_filter)]
    inv_view = inv_view[inv_view["ndim"] >= ndim_min]
    if only_symmetric:
        inv_view = inv_view[inv_view["symmetric"]]

    st.dataframe(inv_view, width="stretch")

    # Export inventory
    inv_csv = desc_df.to_csv(index=False).encode("utf-8")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button(
            "⬇️ Download inventory (CSV)",
            data=inv_csv,
            file_name="matrix_inventory.csv",
            mime="text/csv",
            key="matrix_inventory_csv",
        )
    with c2:
        st.caption(f"סה\"כ מטריצות בקובץ: **{len(descriptors)}**")
    with c3:
        st.caption(f"מספר מטריצות בתצוגה לאחר סינון: **{len(inv_view)}**")

    # סיכום מהיר של ה-inventory
    with st.expander("ℹ️ Inventory summary (kind / shape breakdown)", expanded=False):
        st.markdown("**By inferred kind:**")
        st.dataframe(desc_df["kind"].value_counts().to_frame("count"), width="stretch")

        st.markdown("**Top shapes (by count):**")
        st.dataframe(
            desc_df["shape"].value_counts().to_frame("count").head(10),
            width="stretch",
        )

        problematic = desc_df[desc_df["notes"].astype(str).str.len() > 0]
        if not problematic.empty:
            st.markdown("**Matrices with diagnostics notes:**")
            st.dataframe(problematic, width="stretch")

    shape_groups = _group_by_shape(descriptors)

    # ------------------------------------------------------------------
    # 3. טאבים עליונים: Single matrix / Series analytics
    # ------------------------------------------------------------------
    tab_single, tab_series = st.tabs(
        [
            "📌 Single matrix analytics",
            "🧵 Matrix series analytics",
        ]
    )

    # ------------------------------------------------------------------
    # טאבו ראשון — Single matrix analytics
    # ------------------------------------------------------------------
    with tab_single:
        st.markdown("## 📌 Single matrix analytics")

        matrix_names = [d.name for d in descriptors]

        # מנגנון לזכירת המטריצה האחרונה שנבחרה בסשן + התאמה ל-preset
        last_selected_key = "matrix_last_selected_name"
        default_idx = 0

        # אם יש בחירה קודמת – היא מנצחת
        if last_selected_key in st.session_state:
            last_name = st.session_state[last_selected_key]
            if last_name in matrix_names:
                default_idx = matrix_names.index(last_name)
        else:
            # בחירת מטריצה ברירת מחדל לפי preset וה-kind
            preferred_order: List[str] = []
            if is_pair_flow:
                preferred_order = ["returns-2d", "returns-1d", "correlation"]
            elif is_portfolio_flow:
                preferred_order = ["correlation", "covariance", "returns-2d"]
            elif is_series_flow:
                preferred_order = ["correlation", "covariance", "returns-2d", "generic-2d"]

            if preferred_order:
                for kind in preferred_order:
                    for d in descriptors:
                        if d.inferred_kind == kind:
                            default_idx = matrix_names.index(d.name)
                            break
                    else:
                        continue
                    break

        selected_name = st.selectbox(
            "בחר מטריצה ראשית לניתוח:",
            options=matrix_names,
            index=default_idx,
            key="matrix_main_select",
        )
        st.session_state[last_selected_key] = selected_name

        desc = next(d for d in descriptors if d.name == selected_name)
        mat = np.asarray(npz_obj[selected_name])

        # תיוג עמודות לפי metadata אם קיים
        labels = _guess_asset_labels(npz_obj, selected_name, mat)

        # --------- 3.1 Preview + core info ---------
        df = _render_matrix_preview_section(
            selected_name,
            mat,
            labels=labels,
            desc=desc,
        )

        # נחשב גם diagnostics כאן לצורך export + KPIs מהירים
        diag = _matrix_diagnostics(mat, inferred_kind=desc.inferred_kind)
        diag_df = pd.DataFrame([diag])

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        try:
            finite_ratio = diag["n_finite"] / max(diag["n_elements"], 1)
        except Exception:
            finite_ratio = float("nan")

        kpi1.metric("Kind", desc.inferred_kind)
        kpi2.metric("Shape", str(desc.shape))
        kpi3.metric("Finite %", f"{finite_ratio:.1%}")
        kpi4.metric("Std (all)", f"{diag.get('std', float('nan')):.4g}")

        # --------- 3.2 תתי-טאבים בתוך Single-matrix ---------
        core_tab, quant_tab, risk_tab, ml_tab = st.tabs(
            [
                "🔧 Core analytics",
                "🧪 Advanced quant",
                "🛡 Risk & Portfolio",
                "🤖 ML / AutoML",
            ]
        )

        # ---------- Core analytics ----------
        with core_tab:
            st.markdown("### 🔧 Core analytics")

            with st.expander("4️⃣ Summary & Diagnostics", expanded=True):
                _render_matrix_summary_section(selected_name, mat, desc, df=df)

            with st.expander(
                "5️⃣ Correlation / Covariance",
                expanded=is_portfolio_flow or is_series_flow or is_pair_flow,
            ):
                _render_matrix_corr_cov_section(df)

            with st.expander(
                "6️⃣ PCA & Components",
                expanded=is_portfolio_flow and not light_mode,
            ):
                if light_mode and df.shape[0] * df.shape[1] > 200_000:
                    st.info(
                        "Light mode פעיל ומספר התאים במטריצה גדול מאוד — "
                        "דילגנו על PCA. בטל Light mode כדי להריץ PCA מלא."
                    )
                else:
                    _render_matrix_pca_section(df)

            # Export של דו"ח בסיסי
            with st.expander("📥 Export single-matrix report", expanded=False):
                # 1) DataFrame של המטריצה
                mat_csv = df.to_csv(index=True).encode("utf-8")
                st.download_button(
                    "⬇️ Download matrix as CSV",
                    data=mat_csv,
                    file_name=f"{selected_name}_matrix.csv",
                    mime="text/csv",
                    key="matrix_single_csv",
                )

                # 2) Diagnostics כ-CSV
                diag_csv = diag_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download diagnostics as CSV",
                    data=diag_csv,
                    file_name=f"{selected_name}_diagnostics.csv",
                    mime="text/csv",
                    key="matrix_diag_csv",
                )

                # 3) Markdown דו"ח טקסטואלי
                md_buf = io.StringIO()
                md_buf.write(f"# Matrix report — `{selected_name}`\n\n")
                md_buf.write(f"- Shape: `{desc.shape}`\n")
                md_buf.write(f"- Kind: `{desc.inferred_kind}`\n")
                md_buf.write(f"- Symmetric: `{desc.symmetric}`\n")
                if desc.notes:
                    md_buf.write(f"- Notes: {desc.notes}\n")
                md_buf.write("\n## Diagnostics\n\n")
                for k, v in diag.items():
                    md_buf.write(f"- **{k}**: {v}\n")
                md_data = md_buf.getvalue().encode("utf-8")
                st.download_button(
                    "⬇️ Download Markdown report",
                    data=md_data,
                    file_name=f"{selected_name}_report.md",
                    mime="text/markdown",
                    key="matrix_md_report",
                )

        # ---------- Advanced quant ----------
        with quant_tab:
            st.markdown("### 🧪 Advanced quantitative analytics")

            # Advanced pair metrics
            with st.expander(
                "7️⃣ Advanced pair metrics (DTW / distance / tail / cointegration)",
                expanded=is_pair_flow,
            ):
                _render_pair_advanced_section(df)

            # Advanced matrix metrics (partial/distance)
            with st.expander(
                "8️⃣ Advanced matrix metrics (partial/distance)",
                expanded=is_series_flow or is_portfolio_flow,
            ):
                _render_matrix_advanced_section(df)

            # Anomaly detection – קורלציות / תשואות חריגות
            with st.expander(
                "🚨 Anomaly detection (corr/returns)",
                expanded=is_series_flow,
            ):
                _render_anomaly_detection_section(df)

            # Volatility & Shrinkage
            with st.expander(
                "9️⃣ Volatility & Shrinkage (GARCH / PCA residual / shrinked cov)",
                expanded=is_series_flow or is_portfolio_flow,
            ):
                _render_volatility_shrinkage_section(df)

            # Entropy & complexity
            with st.expander(
                "🔟 Entropy & Complexity (sample entropy / Hurst / spectral coherence)",
                expanded=is_series_flow,
            ):
                _render_matrix_entropy_section(df)

        # ---------- Risk & Portfolio ----------
        with risk_tab:
            st.markdown("### 🛡 Risk & Portfolio analytics")

            with st.expander(
                "1️⃣1️⃣ Pair quality & risk-parity (pair level)",
                expanded=is_pair_flow,
            ):
                _render_pair_quality_section(df)

            with st.expander(
                "1️⃣1️⃣b Pair quality grid (multi-pair ranking)",
                expanded=is_pair_flow,
            ):
                _render_pair_grid_quality_section(df)

            with st.expander(
                "1️⃣2️⃣ Portfolio returns & risk (Sharpe / Sortino / VaR / RC)",
                expanded=is_portfolio_flow or not is_pair_flow,
            ):
                _render_returns_risk_section(df)

        # ---------- ML / AutoML ----------
        with ml_tab:
            st.markdown("### 🤖 ML / AutoML sandbox")

            if show_automl:
                with st.expander(
                    "1️⃣3️⃣ AutoML sandbox (PyCaret regression)",
                    expanded=is_portfolio_flow,
                ):
                    _render_automl_quick_section(df)
            else:
                st.info(
                    "AutoML sandbox כבוי כרגע (כדי לחסוך משאבים). "
                    "ניתן להפעיל אותו מה-Sidebar."
                )

    # ------------------------------------------------------------------
    # טאבו שני — Matrix series analytics (shape groups)
    # ------------------------------------------------------------------
    with tab_series:
        st.markdown("## 🧵 Matrix series analytics")

        if not shape_groups:
            st.info("לא נמצאו קבוצות של מטריצות עם אותה צורה (shape groups).")
            return

        # Overview על shape groups
        _render_series_overview_section(shape_groups)

        # עובדים רק על groups שיש בהם לפחות 2 מטריצות
        groups_with_2 = [g for g in shape_groups if len(g.names) >= 2]
        if not groups_with_2:
            st.info("אין shape groups עם יותר ממטריצה אחת — אין מה להריץ series analytics.")
            return

        max_n_mats = max(len(g.names) for g in groups_with_2)
        max_group_size = st.slider(
            "מקסימום מספר מטריצות בקבוצת shape לניתוח (לצורך ביצועים):",
            min_value=2,
            max_value=max_n_mats,
            value=min(50, max_n_mats),
            step=1,
            key="matrix_series_max_group_size",
        )

        valid_groups = [
            g for g in groups_with_2 if len(g.names) <= max_group_size
        ]
        if not valid_groups:
            st.info("אחרי הסינון לפי גודל קבוצה לא נשארו shape groups מתאימות.")
            return

        group_labels = [
            f"{g.shape} — {len(g.names)} matrices" for g in valid_groups
        ]

        # זיכרון shape-group אחרון
        last_group_key = "matrix_last_shape_group"
        default_group_idx = 0
        if last_group_key in st.session_state:
            last_label = st.session_state[last_group_key]
            if last_label in group_labels:
                default_group_idx = group_labels.index(last_label)

        selected_group_label = st.selectbox(
            "בחר shape group לניתוח סדרתי:",
            options=group_labels,
            index=default_group_idx,
            key="matrix_series_group_select",
        )
        st.session_state[last_group_key] = selected_group_label

        selected_group = valid_groups[group_labels.index(selected_group_label)]

        st.write(
            f"עובדים על **shape={selected_group.shape}** עם **{len(selected_group.names)}** מטריצות:\n\n"
            f"{', '.join(selected_group.names[:8])}"
            f"{'...' if len(selected_group.names) > 8 else ''}"
        )

        if analysis_preset == "Matrix series & regimes":
            st.info(
                "Preset 'Matrix series & regimes' פעיל — התמקד ב-rolling metrics, VAR, DCC "
                "ותנועת הקורלציה לאורך זמן כדי לזהות משטרי שוק."
            )

        if not show_heavy_series:
            st.info(
                "חישובי סדרה כבדים (VAR / DCC / spectral) כבויים כרגע. "
                "ניתן להפעיל אותם מה-Sidebar (Enable heavy series)."
            )
        else:
            _render_series_analytics_section(npz_obj, selected_group)
