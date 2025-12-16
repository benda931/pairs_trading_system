# -*- coding: utf-8 -*-
"""
core/anomaly_detection.py
=========================
Advanced anomaly-detection utilities for the pairs-trading system.

המטרה:
------
מודול מרכזי ל*זיהוי חריגות* (Anomalies) ברמת:
    - סדרות זמן חד-ממדיות (spread / returns / volatility).
    - סדרות מרובות משתנים (returns panel, feature panel).
    - מטריצות קורלציה / קובאריאנס (Matrix Research).
    - שינויי משטר (Change-points / Regime shifts).

המודול משלב:
    * כלים סטטיסטיים קלאסיים (Z-score / IQR / Mahalanobis / robust stats).
    * מודלים מבוססי Machine Learning (IsolationForest, LOF).
    * זיהוי נקודות שינוי (ruptures).
    * "Ensembles" – שילוב כמה גלאים לסיגנל אחד, עם אפשרות טיונינג.

עקרונות:
--------
* Pure functions – ללא side-effects, קל לבדיקה.
* Graceful degradation – אם תלות חסרה (ruptures / sklearn) → המודול ממשיך לתפקד,
  רק בלי הגלאים המתאימים.
* Vectorised – איפה שאפשר, ניצול וקטוריזציה של NumPy/Pandas.
* Explainable – רוב הפונקציות מחזירות גם *score* וגם *סף* ו/או *דגלים*.

תלויות:
-------
* numpy, pandas
* scikit-learn (IsolationForest, LocalOutlierFactor, EmpiricalCovariance, MinCovDet)
* scipy (zscore helper)
* ruptures (אופציונלי) לזיהוי change-points
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union


import warnings

import numpy as np
import pandas as pd
from scipy.stats import zscore as _zscore
from sklearn.covariance import EmpiricalCovariance, MinCovDet
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# Optional dependency – change-point detection
try:
    import ruptures as rpt  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    rpt = None  # sentinel for availability

# Optional dependency – matplotlib (visual helper only)
try:
    import matplotlib.pyplot as _plt  # type: ignore
except Exception:  # pragma: no cover
    _plt = None  # type: ignore[assignment]


# ============================================================================
# __all__ – נשמור API קיים, ונוסיף יכולות חדשות
# ============================================================================

__all__: List[str] = [
    # Basic rolling detectors
    "rolling_mahalanobis",
    "rolling_zscore",
    "rolling_iqr_score",
    # ML-based scores
    "isolation_forest_score",
    "lof_score",
    # Change-points
    "detect_change_points",
    # Ensemble (legacy)
    "ensemble_anomaly_signal",
    # Visual helper
    "plot_anomaly_summary",
    # ---- New HF-grade utilities ----
    "RollingConfig",
    "EnsembleConfig",
    "CorrAnomalyConfig",
    "ReturnAnomalyConfig",
    "rolling_mad_score",
    "rolling_volatility_spike_score",
    "rolling_ewma_residual_score",
    "ensemble_series_anomalies",
    "detect_corr_anomalies",
    "detect_return_anomalies",
]


# ============================================================================
# 0. Helper utilities & dataclasses
# ============================================================================


def _ensure_frame(data: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
    """Return data as **DataFrame** with original index preserved."""
    return data.to_frame() if isinstance(data, pd.Series) else data


def _ensure_series(series: Union[pd.Series, Iterable[float]]) -> pd.Series:
    """Normalize input to pandas.Series with a simple RangeIndex if needed."""
    if isinstance(series, pd.Series):
        return series
    return pd.Series(list(series), name="series")


@dataclass(frozen=True)
class RollingConfig:
    """Config לפונקציות rolling חד-ממדיות.

    Attributes
    ----------
    window : int
        אורך חלון rolling (מספר תצפיות).
    min_periods : int
        מספר מינימלי של נקודות לפני שמתחילים להחזיר מדד.
    center : bool
        האם למרכז את החלון (כמו pandas.rolling(center=True)).
    """

    window: int = 252
    min_periods: int = 50
    center: bool = False


@dataclass(frozen=True)
class EnsembleConfig:
    """Config ל-Ensemble על סדרה חד-ממדית.

    Attributes
    ----------
    window : int
        חלון rolling לציון Z/IQR.
    z_thresh : float
        סף |Z| לזיהוי חריגה.
    iqr_thresh : float
        סף |IQR-score| (robust Z) לזיהוי חריגה.
    iso_contamination : float
        שיעור חריגות צפוי ל-IsolationForest (0.0–0.5).
    iso_thresh : float
        סף תוצאה מ-IsolationForest (decision_function).
    lof_neighbors : int
        מספר שכנים ל-LOF.
    lof_quantile : float
        Quantile על LOF מתחתיו נחשב אנומלי.
    """

    window: int = 252
    z_thresh: float = 3.0
    iqr_thresh: float = 4.5
    iso_contamination: float = 0.02
    iso_thresh: float = -0.2
    lof_neighbors: int = 30
    lof_quantile: float = 0.05


@dataclass(frozen=True)
class CorrAnomalyConfig:
    """Config לזיהוי אנומליות במטריצות קורלציה."""

    method: str = "zscore"  # 'zscore' / 'delta'
    z_thresh: float = 3.0    # לסף |Z|
    delta_thresh: float = 0.25  # סף |Δcorr| כאשר method='delta'
    baseline_weight: float = 0.5  # לערבוב baseline + matrix נוכחי אם רוצים החלקה


@dataclass(frozen=True)
class ReturnAnomalyConfig:
    """Config לזיהוי אנומליות על מטריצת תשואות (T×N)."""

    axis: str = "time"  # 'time' או 'asset'
    z_thresh: float = 3.5
    iqr_thresh: float = 4.0
    use_iforest: bool = True
    contamination: float = 0.01
    max_features: Optional[int] = None  # max מספר עמודות לתזמון מבחינת ביצועים


# ============================================================================
# 1. Robust distance-based detectors (rolling)
# ============================================================================


def rolling_mahalanobis(
    data: Union[pd.Series, pd.DataFrame],
    window: int = 252,
    min_periods: int = 50,
    robust: bool = True,
    step: int = 1,
) -> pd.Series:
    """Rolling Mahalanobis distance (vectorised-ish, HF-grade).

    Parameters
    ----------
    data        : Series או DataFrame (spread / returns / features).
    window      : אורך חלון (≈ שנה).
    min_periods : כמות נתונים מינימלית להתחלה.
    robust      : שימוש ב-Minimum-Covariance-Determinant (Robust).
    step        : stride (1 = כל נקודה, 5 = כל 5 נקודות).

    Returns
    -------
    pd.Series
        סדרת מרחקים (מרחק גדול → יותר "חריג").
    """
    x = _ensure_frame(data).dropna(how="all")
    n = len(x)
    dist = pd.Series(np.nan, index=x.index, name="mahalanobis")

    if n < min_periods:
        return dist

    # נלך בצורה פשוטה אך גמישה – אפשר לשדרג בעתיד ל-sliding window מלא.
    for end in range(min_periods, n, step):
        start = max(0, end - window)
        window_df = x.iloc[start:end]
        if window_df.shape[0] < min_periods:
            continue
        try:
            if robust:
                est = MinCovDet().fit(window_df)
            else:
                est = EmpiricalCovariance().fit(window_df)
            d_val = est.mahalanobis(x.iloc[[end]])[0]
            dist.iloc[end] = float(d_val)
        except ValueError:
            # Singular matrix – נשאיר NaN
            continue

    return dist


def rolling_zscore(
    series: pd.Series,
    window: int = 252,
) -> pd.Series:
    """Rolling *z-score* (mean/std) – מדד חריגות קלאסי.

    z_t = (x_t - mean_window) / std_window
    מחזיר את ה-Z *האחרון* בכל חלון.
    """
    s = _ensure_series(series).astype(float)
    z = s.rolling(window).apply(
        lambda sub: _zscore(sub.values)[-1] if len(sub) > 1 else np.nan,
        raw=False,
    )
    return z.rename("zscore")


def rolling_iqr_score(
    series: pd.Series,
    window: int = 252,
) -> pd.Series:
    """Rolling IQR score (Robust Z).

    Score_t = (x_t - median) / (IQR / 1.349) ≈ robust z-score
    """
    s = _ensure_series(series).astype(float)

    def _robust_z(sub: pd.Series) -> float:
        if len(sub) < 10:
            return np.nan
        med = sub.median()
        iqr = sub.quantile(0.75) - sub.quantile(0.25)
        if iqr == 0:
            return np.nan
        scale = iqr / 1.349
        return (sub.iloc[-1] - med) / scale

    r = s.rolling(window).apply(_robust_z, raw=False)
    return r.rename("iqr_score")


def rolling_mad_score(
    series: pd.Series,
    window: int = 252,
) -> pd.Series:
    """Rolling MAD-score (Median Absolute Deviation).

    Score = (x_t - median) / (1.4826 * MAD)
    1.4826 ~ scale factor to make MAD comparable to std (normal dist).
    """
    s = _ensure_series(series).astype(float)

    def _mad_z(sub: pd.Series) -> float:
        if len(sub) < 10:
            return np.nan
        med = sub.median()
        mad = (sub - med).abs().median()
        if mad == 0:
            return np.nan
        return (sub.iloc[-1] - med) / (1.4826 * mad)

    r = s.rolling(window).apply(_mad_z, raw=False)
    return r.rename("mad_score")


def rolling_volatility_spike_score(
    series: pd.Series,
    window: int = 252,
) -> pd.Series:
    """מדד "spike" בתנודתיות – כמה הסטייה התקנית האחרונה חריגה מול ההיסטוריה.

    Logic:
    -------
    1. מחשבים rolling std (vol_t).
    2. מחשבים z-score של vol_t עצמו.
    """
    s = _ensure_series(series).astype(float)
    vol = s.rolling(window).std(ddof=1)
    vol_z = rolling_zscore(vol.dropna(), window=window)
    vol_z = vol_z.reindex(s.index)
    return vol_z.rename("vol_spike_score")


def rolling_ewma_residual_score(
    series: pd.Series,
    span: int = 60,
    window_z: int = 252,
) -> pd.Series:
    """Residual Z-score ביחס ל-EWMA (ממוצע אקספוננציאלי).

    Steps:
    -------
    1. EWMA_t = series.ewm(span=span).mean()
    2. residual_t = series_t - EWMA_t
    3. rolling_zscore(residual_t, window_z)
    """
    s = _ensure_series(series).astype(float)
    ewma = s.ewm(span=span, adjust=False).mean()
    residual = (s - ewma).rename("ewma_residual")
    res_z = rolling_zscore(residual, window=window_z)
    return res_z.rename("ewma_resid_z")


# ============================================================================
# 2. ML-based detectors (IsolationForest / LOF)
# ============================================================================


def isolation_forest_score(
    data: Union[pd.Series, pd.DataFrame],
    contamination: float = 0.01,
    n_estimators: int = 200,
    random_state: int = 42,
) -> pd.Series:
    """Isolation-Forest anomaly *score*.

    Notes
    -----
    - ערכים *נמוכים* יותר (יותר שליליים) → יותר אנומלי.
    - זהו decision_function של sklearn (לא labels).
    """
    frame = _ensure_frame(data).dropna(how="any")
    if frame.empty:
        return pd.Series(dtype=float, name="iso_score")

    x = frame.values
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        n_jobs=-1,
        random_state=random_state,
    ).fit(x)
    scores = model.decision_function(x)
    return pd.Series(scores, index=frame.index, name="iso_score")


def lof_score(
    data: Union[pd.Series, pd.DataFrame],
    n_neighbors: int = 20,
) -> pd.Series:
    """Local-Outlier-Factor *negative* score (יותר שלילי → יותר אנומלי)."""
    frame = _ensure_frame(data).dropna(how="any")
    if frame.empty:
        return pd.Series(dtype=float, name="lof_score")

    x = frame.values
    model = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        model.fit(x)
    scores = model.negative_outlier_factor_
    return pd.Series(scores, index=frame.index, name="lof_score")


# ============================================================================
# 3. Change-point detection (ruptures)
# ============================================================================


def detect_change_points(
    series: pd.Series,
    model: str = "rbf",
    penalty: float | None = None,
    n_bkps: int | None = None,
) -> List[int]:
    """Return indices of detected change-points using **ruptures**.

    Either supply *penalty* or *n_bkps* (exact number of breakpoints).
    """
    if rpt is None:
        raise ImportError("`ruptures` not installed – run `pip install ruptures`.")

    s = _ensure_series(series).dropna()
    if s.empty:
        return []

    algo = rpt.Binseg(model=model).fit(s.values)
    if (penalty is None) == (n_bkps is None):
        raise ValueError("Provide exactly one of `penalty` or `n_bkps`.")

    bkps = algo.predict(pen=penalty) if penalty is not None else algo.predict(n_bkps=n_bkps)
    # ruptures מחזיר גם את נקודת הסיום – נסנן החוצה
    bkps = [b for b in bkps if b < len(s)]
    return sorted(bkps)


# ============================================================================
# 4. Ensemble aggregator – HF-grade
# ============================================================================


def ensemble_anomaly_signal(
    series: pd.Series,
    window: int = 252,
    z_thresh: float = 3.0,
    iqr_thresh: float = 4.5,
    iso_thresh: float = -0.2,
    lof_quantile: float = 0.05,
) -> pd.Series:
    """Binary *ensemble* anomaly signal (שומר על API הישן)."""
    cfg = EnsembleConfig(
        window=window,
        z_thresh=z_thresh,
        iqr_thresh=iqr_thresh,
        iso_thresh=iso_thresh,
        lof_quantile=lof_quantile,
    )
    return ensemble_series_anomalies(series, cfg)


def ensemble_series_anomalies(
    series: pd.Series,
    cfg: EnsembleConfig,
) -> pd.Series:
    """Ensemble מורכב ומוסבר יותר לחריגות בסדרה חד-ממדית.

    Logic (per index):
    -------------------
    anomaly_t = (
        |z_t| > cfg.z_thresh
        OR |iqr_t| > cfg.iqr_thresh
        OR iso_t < cfg.iso_thresh
        OR lof_t < LOF_quantile(cfg.lof_quantile)
    )
    """
    s = _ensure_series(series).astype(float)

    # בסיסים
    z = rolling_zscore(s, window=cfg.window)
    iq = rolling_iqr_score(s, window=cfg.window)

    # IsolationForest/LOF – נריץ על כל הדאטה, ואז ניישר אינדקס
    try:
        iso = isolation_forest_score(s, contamination=cfg.iso_contamination)
    except Exception:
        iso = pd.Series(np.nan, index=s.index, name="iso_score")

    try:
        lof = lof_score(s, n_neighbors=cfg.lof_neighbors)
    except Exception:
        lof = pd.Series(np.nan, index=s.index, name="lof_score")

    # ניישר את כל הסדרות על אותו אינדקס
    df = pd.concat([s, z, iq, iso, lof], axis=1)
    # שמות
    df.columns = ["value", "z", "iqr", "iso", "lof"]

    # חישוב סף LOF לפי quantile (יותר שלילי → יותר אנומלי)
    lof_cut = df["lof"].quantile(cfg.lof_quantile) if df["lof"].notna().any() else np.nan

    cond_z = df["z"].abs() > cfg.z_thresh
    cond_iq = df["iqr"].abs() > cfg.iqr_thresh
    cond_iso = df["iso"] < cfg.iso_thresh
    cond_lof = df["lof"] < lof_cut

    signal = (cond_z | cond_iq | cond_iso | cond_lof).astype(bool)
    return signal.rename("ensemble_anomaly")


# ============================================================================
# 5. Correlation & returns anomaly detectors (Matrix Research)
# ============================================================================


def detect_corr_anomalies(
    corr: pd.DataFrame,
    baseline_corr: Optional[pd.DataFrame] = None,
    cfg: Optional[CorrAnomalyConfig] = None,
) -> pd.DataFrame:
    """
    זיהוי אנומליות במטריצת קורלציה (N×N).

    Output:
    -------
    DataFrame עם שורות:
        asset_i, asset_j, corr_ij, baseline_ij, delta, z, is_anomaly

    Notes:
    ------
    - אם baseline_corr לא מועבר, נשתמש רק במטריצה עצמה (לפי z-score על כל off-diagonal).
    - cfg.method:
        * "zscore" – z על כל הערכים off-diagonal.
        * "delta" – |corr - baseline| > delta_thresh.
    """
    if cfg is None:
        cfg = CorrAnomalyConfig()

    if corr is None or corr.empty:
        return pd.DataFrame()

    corr = corr.copy()
    # לוודא סימטריה בסיסית
    if corr.shape[0] != corr.shape[1]:
        raise ValueError("corr must be square (N×N).")

    assets = list(corr.index)
    m = corr.values.astype(float)

    # baseline – אם קיים
    if baseline_corr is not None and not baseline_corr.empty:
        b = baseline_corr.reindex(index=assets, columns=assets).values.astype(float)
    else:
        b = np.full_like(m, np.nan)

    # נבנה טבלת i,j,corr,baseline
    rows: List[Dict[str, Any]] = []
    n = len(assets)
    for i in range(n):
        for j in range(i + 1, n):
            c_ij = m[i, j]
            b_ij = b[i, j]
            rows.append(
                {
                    "asset_i": assets[i],
                    "asset_j": assets[j],
                    "corr": float(c_ij),
                    "baseline_corr": float(b_ij) if np.isfinite(b_ij) else np.nan,
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # delta מול baseline אם יש
    df["delta"] = df["corr"] - df["baseline_corr"]
    # אם אין baseline, delta=corr (לצורך ניתוח שינויים קיצוניים)
    df.loc[df["baseline_corr"].isna(), "delta"] = df.loc[df["baseline_corr"].isna(), "corr"]

    # Z-score על corr או delta (off-diagonal)
    if cfg.method == "zscore":
        z_vals = _zscore(df["corr"].values, nan_policy="omit")
        df["z"] = z_vals
        df["is_anomaly"] = df["z"].abs() > cfg.z_thresh
    else:  # "delta"
        df["z"] = np.nan
        df["is_anomaly"] = df["delta"].abs() > cfg.delta_thresh

    return df.sort_values("is_anomaly", ascending=False).reset_index(drop=True)


def detect_return_anomalies(
    returns: pd.DataFrame,
    cfg: Optional[ReturnAnomalyConfig] = None,
) -> pd.DataFrame:
    """
    זיהוי אנומליות על מטריצת תשואות (T×N).

    Modes:
    -------
    - axis="time":   מזהה timestamps חריגים (across assets).
    - axis="asset":  מזהה נכסים עם התנהגות חריגה (במימד הזמן).

    Output:
    -------
    DataFrame עם שדות כמו:
        idx (index), axis, score_z, score_iqr, iso_score, is_anomaly
    """
    if cfg is None:
        cfg = ReturnAnomalyConfig()

    if returns is None or returns.empty:
        return pd.DataFrame()

    df = returns.copy().astype(float)

    # הגבלת מספר עמודות לצורך ביצועים אם רוצים
    if cfg.max_features is not None and df.shape[1] > cfg.max_features:
        df = df.iloc[:, : cfg.max_features]

    rows: List[Dict[str, Any]] = []

    if cfg.axis == "time":
        # כל שורה – vector של נכסים באותו timestamp
        base = _ensure_frame(df).dropna(how="all")
        if base.empty:
            return pd.DataFrame()

        # Z/IQR על סכום מטורף? עדיף על norm של הווקטור
        norms = np.linalg.norm(base.values, axis=1)
        idx = base.index

        z = rolling_zscore(pd.Series(norms, index=idx), window=min(len(idx), 252))
        iq = rolling_iqr_score(pd.Series(norms, index=idx), window=min(len(idx), 252))

        try:
            iso = isolation_forest_score(base, contamination=cfg.contamination)
        except Exception:
            iso = pd.Series(np.nan, index=base.index, name="iso_score")

        cut_iso = cfg.iso_thresh
        for i in range(len(idx)):
            row = {
                "axis": "time",
                "idx": idx[i],
                "zscore": float(z.iloc[i]) if not np.isnan(z.iloc[i]) else np.nan,
                "iqr_score": float(iq.iloc[i]) if not np.isnan(iq.iloc[i]) else np.nan,
                "iso_score": float(iso.iloc[i]) if not np.isnan(iso.iloc[i]) else np.nan,
            }
            rows.append(row)

        out = pd.DataFrame(rows)
        if out.empty:
            return out

        out["is_anomaly"] = (
            out["zscore"].abs() > cfg.z_thresh
        ) | (
            out["iqr_score"].abs() > cfg.iqr_thresh
        ) | (
            out["iso_score"] < cut_iso
        )

        return out.sort_values("is_anomaly", ascending=False).reset_index(drop=True)

    else:  # axis="asset" – נסתכל על כל נכס בנפרד
        for col in df.columns:
            s = df[col].dropna()
            if len(s) < 10:
                continue
            z = rolling_zscore(s, window=min(len(s), 252))
            iq = rolling_iqr_score(s, window=min(len(s), 252))
            mad_s = rolling_mad_score(s, window=min(len(s), 252))

            for t in s.index:
                row = {
                    "axis": "asset",
                    "asset": col,
                    "idx": t,
                    "zscore": float(z.loc[t]) if t in z.index else np.nan,
                    "iqr_score": float(iq.loc[t]) if t in iq.index else np.nan,
                    "mad_score": float(mad_s.loc[t]) if t in mad_s.index else np.nan,
                }
                rows.append(row)

        out = pd.DataFrame(rows)
        if out.empty:
            return out

        out["is_anomaly"] = (
            out["zscore"].abs() > cfg.z_thresh
        ) | (
            out["iqr_score"].abs() > cfg.iqr_thresh
        ) | (
            out["mad_score"].abs() > cfg.iqr_thresh
        )

        return out.sort_values("is_anomaly", ascending=False).reset_index(drop=True)


# ============================================================================
# 6. Quick visual checker (Matplotlib helper)
# ============================================================================


def plot_anomaly_summary(
    series: pd.Series,
    signal: pd.Series,
    ax=None,
):  # pragma: no cover
    """Quick-and-dirty visual plot – תלוי רק ב-Matplotlib (אם קיים).

    Parameters
    ----------
    series : pd.Series
        הסדרה המקורית.
    signal : pd.Series
        סדרת בוליאנית או אנומלי-סקור → אנו לוקחים True/False לפי >0.5.
    ax : matplotlib Axes, optional
        ציר לפלוט עליו; אם None ניצור פיגר חדש.

    Returns
    -------
    ax : matplotlib.pyplot.Axes
    """
    if _plt is None:
        raise ImportError("matplotlib not available – cannot plot anomaly summary.")

    s = _ensure_series(series)
    sig = signal.copy()
    # אם לא בוליאני – נתייחס כ-score ונמיר לבוליאני לפי 0.5
    if sig.dtype != bool:
        try:
            sig = sig.astype(float) > 0.5
        except Exception:
            sig = sig.astype(bool)

    if ax is None:
        _, ax = _plt.subplots(figsize=(12, 4))

    s.plot(ax=ax, label="series")
    idx_anom = sig[sig].index
    ax.scatter(idx_anom, s.loc[idx_anom], color="red", marker="x", label="anomaly")
    ax.set_title("Anomaly summary")
    ax.legend()
    return ax
