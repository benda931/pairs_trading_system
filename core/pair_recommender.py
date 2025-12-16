# -*- coding: utf-8 -*-
"""
core/pair_recommender.py — Pair Research & Parameter Suggestions (v2)
=====================================================================

מודול מחקר והמלצה לזוג נכסים, ברמת "קרן גידור":

1. analyze_pair(price_a, price_b) → PairDiagnostics
   - מחזיר אובייקט אבחון מפורט עם:
     • Cointegration / VAR / Granger
     • Dynamic & static correlation
     • Tail dependence, entropy, Hurst
     • חצי־חיים של הספרד, סטטיסטיקות בסיסיות
     • ציון איכות לזוג (0–100) + תיוג איכות ('excellent' / 'good' / 'medium' / 'weak').

2. recommend_params(diag) → Dict[str, float]
   - מקבל PairDiagnostics ומחזיר סט פרמטרים מומלצים לזוג:
     • z_entry, z_exit, lookback, stop_z, take_profit_z, max_holding_days
     • hedge_style (ols / beta-tight / conservative)
     • quality_score, quality_label

3. recommend_pair(price_a, price_b) → (diag, rec)
   - עטיפה נוחה שמריצה גם אבחון וגם המלצות פרמטרים.

שימוש אופייני
-------------
    diag = analyze_pair(price_a, price_b)
    rec  = recommend_params(diag)
    diag, rec = recommend_pair(price_a, price_b)

הקובץ נכתב כך שיהיה:
- רובסטי (שגיאות בחישוב מדד → None + warning בלוג, לא קריסה).
- JSON-friendly (PairDiagnostics ניתן להמרה למילון דרך asdict()).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple

import logging
import numpy as np
import pandas as pd

from common.advanced_metrics import (  # type: ignore
    cointegration_test,
    dynamic_conditional_correlation,
    granger_causality,
    partial_correlation_matrix,
    sample_entropy,
    tail_dependence,
    var_diagnostics,
)
from core.metrics import PerformanceMetrics  # type: ignore

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# ======================================================================
# Dataclass – אבחון מלא + ציון איכות
# ======================================================================

@dataclass
class PairDiagnostics:
    """
    אובייקט אבחון לזוג נכסים.

    כל שדה הוא Optional – אם חישוב מסוים נכשל, נשאר None.
    """

    # בסיס
    n_obs: Optional[int] = None
    avg_price_a: Optional[float] = None
    avg_price_b: Optional[float] = None

    # Cointegration
    coint_stat: Optional[float] = None
    coint_pvalue: Optional[float] = None

    # Granger causality: ממוצע p-value + כיוון (a->b / b->a)
    granger_mean_pvalue: Optional[float] = None
    granger_leads: Optional[str] = None  # 'a->b', 'b->a' או None

    # VAR diagnostics
    var_aic: Optional[float] = None
    var_bic: Optional[float] = None
    var_hqic: Optional[float] = None
    var_is_stable: Optional[bool] = None

    # Partial correlation
    partial_corr: Optional[float] = None

    # Dynamic conditional correlation (mean)
    dyn_corr_mean: Optional[float] = None

    # Pearson correlation of returns
    corr: Optional[float] = None

    # Hurst exponent של הספרד
    hurst: Optional[float] = None

    # Tail dependence (upper, lower)
    tail_u: Optional[float] = None
    tail_l: Optional[float] = None

    # Sample entropy של הספרד
    entropy: Optional[float] = None

    # Hedge ratio (OLS beta)
    beta_ols: Optional[float] = None

    # Spread stats
    spread_mean: Optional[float] = None
    spread_std: Optional[float] = None
    spread_half_life: Optional[float] = None  # בימים (אומדן)

    # ציוני איכות כלליים
    quality_score: Optional[float] = None  # 0–100
    quality_label: Optional[str] = None    # 'excellent' / 'good' / 'medium' / 'weak'
    quality_notes: Optional[str] = None    # תיאור חופשי/מקוצר


# ======================================================================
# Helper functions
# ======================================================================

def _align_price_series(price_a: pd.Series, price_b: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """Align two price series on their common index and drop missing values."""
    df = pd.DataFrame({"a": price_a, "b": price_b}).dropna()
    return df["a"], df["b"]


def _compute_beta_ols(price_a: pd.Series, price_b: pd.Series) -> Optional[float]:
    """Compute OLS hedge ratio (beta) of price_a on price_b."""
    try:
        import statsmodels.api as sm  # type: ignore
    except Exception:
        logger.warning("statsmodels not available – beta_ols will be None.")
        return None
    try:
        if len(price_b) < 10:
            return None
        X = sm.add_constant(price_b.values)
        model = sm.OLS(price_a.values, X).fit()
        return float(model.params[1])
    except Exception as e:
        logger.warning(f"_compute_beta_ols failed: {e}")
        return None


def _compute_coint(a: np.ndarray, b: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    """Compute Engle–Granger cointegration statistic and p-value using existing function."""
    try:
        mat = np.column_stack([a, b])
        series = pd.Series([mat])
        res = cointegration_test(series, method="engle")
        stat = float(res.loc[0, "coint_score"])
        pval = float(res.loc[0, "p_value"])
        return stat, pval
    except Exception as e:
        logger.warning(f"cointegration_test failed: {e}")
        return None, None


def _compute_granger(a: np.ndarray, b: np.ndarray, maxlag: int = 5) -> Tuple[Optional[float], Optional[str]]:
    """
    Compute Granger causality p-values across lags and infer lead/lag direction.
    Returns average p-value and a simple direction indicator: 'a->b', 'b->a' or None.
    """
    try:
        mat = np.column_stack([a, b])
        series = pd.Series([mat])
        gc_res = granger_causality(series, maxlag=maxlag, verbose=False)
        stats = gc_res.iloc[0]
        pvals = []
        direction = None

        for lag, val in stats.items():
            if isinstance(val, dict):
                pvals.append(val.get("p_value", np.nan))
        mean_p = float(np.nanmean(pvals)) if pvals else None

        # כיוון: ננסה גם הצד ההפוך
        try:
            mat_swap = np.column_stack([b, a])
            series_swap = pd.Series([mat_swap])
            swap_res = granger_causality(series_swap, maxlag=maxlag, verbose=False)
            stats_ab = stats
            stats_ba = swap_res.iloc[0]

            p_ab = None
            p_ba = None
            # מניחים lag=1 הוא העיקרי
            if isinstance(stats_ab[1], dict):
                p_ab = stats_ab[1].get("p_value")
            if isinstance(stats_ba[1], dict):
                p_ba = stats_ba[1].get("p_value")

            if p_ab is not None and p_ba is not None:
                if p_ab < 0.05 and p_ba >= 0.05:
                    direction = "a->b"
                elif p_ba < 0.05 and p_ab >= 0.05:
                    direction = "b->a"
        except Exception:
            pass

        return mean_p, direction
    except Exception as e:
        logger.warning(f"granger_causality failed: {e}")
        return None, None


def _compute_var_diag(a: np.ndarray, b: np.ndarray, maxlags: int = 5) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[bool]]:
    """Fit VAR model and extract AIC, BIC, HQIC and stability flag."""
    try:
        mat = np.column_stack([a, b])
        series = pd.Series([mat])
        res = var_diagnostics(series, maxlags=maxlags)
        row = res.iloc[0]
        return float(row["aic"]), float(row["bic"]), float(row["hqic"]), bool(row["is_stable"])
    except Exception as e:
        logger.warning(f"var_diagnostics failed: {e}")
        return None, None, None, None


def _compute_partial_corr(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    """Compute partial correlation between two series using existing function."""
    try:
        mat = np.column_stack([a, b])
        series = pd.Series([mat])
        res = partial_correlation_matrix(series)
        pcorr = res.iloc[0]  # NDArray (k×k)
        return float(pcorr[0, 1])
    except Exception as e:
        logger.warning(f"partial_correlation_matrix failed: {e}")
        return None


def _compute_dyn_corr(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    """Compute mean dynamic conditional correlation using DCC-GARCH."""
    try:
        mat = np.column_stack([a, b])
        series = pd.Series([mat])
        res = dynamic_conditional_correlation(series, p=1, q=1)
        dyn = res.iloc[0]  # (T, k, k)
        if dyn is None or not isinstance(dyn, np.ndarray):
            return None
        corr_series = dyn[:, 0, 1]
        return float(np.nanmean(corr_series))
    except Exception as e:
        logger.warning(f"dynamic_conditional_correlation failed: {e}")
        return None


def _compute_hurst(spread: pd.Series) -> Optional[float]:
    """Compute Hurst exponent via PerformanceMetrics on spread returns."""
    try:
        rets = spread.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        if len(rets) < 50:
            return None
        pm = PerformanceMetrics(rets)
        return float(pm.hurst_exponent(max_lag=100))
    except Exception as e:
        logger.warning(f"hurst_exponent failed: {e}")
        return None


def _compute_tail_dep(ra: pd.Series, rb: pd.Series) -> Tuple[Optional[float], Optional[float]]:
    """Compute upper and lower tail dependence using existing function."""
    try:
        df = pd.DataFrame({"x": ra, "y": rb}).dropna()
        if df.empty:
            return None, None
        result = tail_dependence(df, u=0.95)
        if isinstance(result, tuple) and len(result) == 2:
            return float(result[0]), float(result[1])
        return None, None
    except Exception as e:
        logger.warning(f"tail_dependence failed: {e}")
        return None, None


def _compute_entropy(spread: pd.Series) -> Optional[float]:
    """Compute sample entropy using existing function."""
    try:
        mat = spread.dropna().values.reshape(-1, 1)
        if mat.shape[0] < 50:
            return None
        series = pd.Series([mat])
        ent_series = sample_entropy(series, m=2, r=0.2)
        return float(ent_series.iloc[0])
    except Exception as e:
        logger.warning(f"sample_entropy failed: {e}")
        return None


def _compute_half_life(spread: pd.Series) -> Optional[float]:
    """
    אומדן חצי־חיים (mean-reversion half-life) לספרד.

    נשתמש במודל:
        spread_t ≈ α + φ * spread_{t-1} + ε_t

    ואז:
        half_life ≈ -ln(2) / ln(φ)

    אם φ לא הגיוני (<=0 או ≥1) – נחזיר None.
    """
    try:
        s = spread.dropna().astype(float)
        if len(s) < 50:
            return None
        s_lag = s.shift(1).dropna()
        s_curr = s.loc[s_lag.index]
        x = s_lag.values
        y = s_curr.values
        # OLS ל־φ
        x_mean = x.mean()
        y_mean = y.mean()
        cov_xy = np.mean(x * y) - x_mean * y_mean
        var_x = np.mean(x * x) - x_mean * x_mean
        if var_x <= 0:
            return None
        phi = cov_xy / var_x
        if phi <= 0 or phi >= 1:
            return None
        half_life = -np.log(2) / np.log(phi)
        if not np.isfinite(half_life) or half_life <= 0:
            return None
        return float(half_life)
    except Exception as e:
        logger.warning(f"_compute_half_life failed: {e}")
        return None


def _compute_spread_stats(spread: pd.Series) -> Tuple[Optional[float], Optional[float]]:
    try:
        s = spread.dropna().astype(float)
        if s.empty:
            return None, None
        return float(s.mean()), float(s.std(ddof=0))
    except Exception:
        return None, None


def _score_pair_quality(diag: PairDiagnostics) -> Tuple[float, str, str]:
    """
    פונקציה שמייצרת ציון איכות לזוג (0–100) + label + הסבר קצר.

    השיקולים:
    ----------
    - coint_pvalue <= 0.05 → חזק (סטציונריות טובה).
    - dyn_corr_mean / corr גבוהים (>= 0.8).
    - VAR יציב.
    - Hurst נמוך (≈ 0.3–0.4) → mean-reversion טוב.
    - tail_u גדול מדי → מוריד ציון (רעש זנב גבוה).
    - entropy גבוהה מדי → ספרד "אקרעי", פחות predictability.
    """
    score = 50.0
    notes = []

    # Cointegration
    if diag.coint_pvalue is not None:
        if diag.coint_pvalue <= 0.01:
            score += 15
            notes.append("Strong cointegration (p <= 0.01)")
        elif diag.coint_pvalue <= 0.05:
            score += 10
            notes.append("Cointegration (p <= 0.05)")
        elif diag.coint_pvalue <= 0.1:
            score += 5
            notes.append("Weak cointegration (p <= 0.1)")
        else:
            score -= 10
            notes.append("No clear cointegration (p > 0.1)")

    # Dynamic / static corr
    dyn_corr = diag.dyn_corr_mean if diag.dyn_corr_mean is not None else diag.corr
    if dyn_corr is not None:
        if dyn_corr >= 0.9:
            score += 10
            notes.append("Very high correlation ≈ 0.9+")
        elif dyn_corr >= 0.8:
            score += 6
            notes.append("High correlation ≈ 0.8+")
        elif dyn_corr >= 0.6:
            score += 2
            notes.append("Moderate correlation")
        else:
            score -= 5
            notes.append("Low/moderate correlation")

    # VAR stability
    if diag.var_is_stable is False:
        score -= 10
        notes.append("VAR model unstable")

    # Hurst
    if diag.hurst is not None:
        if diag.hurst < 0.35:
            score += 8
            notes.append("Strong mean-reversion (H < 0.35)")
        elif diag.hurst < 0.5:
            score += 4
            notes.append("Mild mean-reversion (H < 0.5)")
        elif diag.hurst > 0.6:
            score -= 6
            notes.append("Trend-like behavior (H > 0.6)")

    # Tail dependence (upper)
    if diag.tail_u is not None:
        if diag.tail_u > 0.3:
            score -= 5
            notes.append("High upper tail dependence (risk of joint spikes)")
        elif diag.tail_u < 0.15:
            score += 2
            notes.append("Low upper tail dependence")

    # Entropy
    if diag.entropy is not None:
        if diag.entropy > 0.8:
            score -= 4
            notes.append("High entropy — noisy/irregular spread")
        elif diag.entropy < 0.5:
            score += 3
            notes.append("Moderate entropy — more structure")

    # Normalization
    score = max(0.0, min(100.0, score))

    if score >= 80:
        label = "excellent"
    elif score >= 65:
        label = "good"
    elif score >= 50:
        label = "medium"
    else:
        label = "weak"

    notes_str = "; ".join(notes) if notes else "No strong diagnostics available."
    return score, label, notes_str


# ======================================================================
# Public API
# ======================================================================

def analyze_pair(price_a: pd.Series, price_b: pd.Series) -> PairDiagnostics:
    """
    חישוב מדדים רבי־מימדיים לזוג נכסים על בסיס פונקציות הליבה ב־core.advanced_metrics.

    - ה־Series שנכנסים לפונקציה אמורים להיות מחירי סגירה (רצוי).
    - הפונקציה דואגת ליישר את הסדרות ולהפיל NaN.
    - במקרה של סדרה קצרה מאוד (פחות מ־50 תצפיות) רוב המדדים יחזרו None.

    Returns:
        PairDiagnostics עם כל המדדים והציונים.
    """
    a, b = _align_price_series(price_a, price_b)
    diag = PairDiagnostics()
    diag.n_obs = int(len(a))

    if len(a) < 30:
        logger.warning("analyze_pair: less than 30 observations – diagnostics will be minimal.")
        diag.avg_price_a = float(a.mean()) if len(a) > 0 else None
        diag.avg_price_b = float(b.mean()) if len(b) > 0 else None
        return diag

    diag.avg_price_a = float(a.mean())
    diag.avg_price_b = float(b.mean())

    arr_a = a.values.astype(float)
    arr_b = b.values.astype(float)

    # Beta OLS
    diag.beta_ols = _compute_beta_ols(a, b)
    beta = diag.beta_ols if diag.beta_ols is not None else 1.0
    spread = a - beta * b

    # Spread stats
    diag.spread_mean, diag.spread_std = _compute_spread_stats(spread)
    diag.spread_half_life = _compute_half_life(spread)

    # Cointegration
    diag.coint_stat, diag.coint_pvalue = _compute_coint(arr_a, arr_b)

    # Granger causality
    diag.granger_mean_pvalue, diag.granger_leads = _compute_granger(arr_a, arr_b, maxlag=5)

    # VAR diagnostics
    diag.var_aic, diag.var_bic, diag.var_hqic, diag.var_is_stable = _compute_var_diag(arr_a, arr_b, maxlags=5)

    # Partial correlation
    diag.partial_corr = _compute_partial_corr(arr_a, arr_b)

    # Dynamic conditional correlation (mean)
    diag.dyn_corr_mean = _compute_dyn_corr(arr_a, arr_b)

    # Pearson correlation of returns
    try:
        diag.corr = float(a.pct_change().corr(b.pct_change()))
    except Exception:
        diag.corr = None

    # Hurst exponent
    diag.hurst = _compute_hurst(spread)

    # Tail dependence (upper, lower) on returns
    ra = a.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    rb = b.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    diag.tail_u, diag.tail_l = _compute_tail_dep(ra, rb)

    # Sample entropy
    diag.entropy = _compute_entropy(spread)

    # Quality scoring
    q_score, q_label, q_notes = _score_pair_quality(diag)
    diag.quality_score = q_score
    diag.quality_label = q_label
    diag.quality_notes = q_notes

    return diag


def recommend_params(diag: PairDiagnostics) -> Dict[str, float]:
    """
    מייצרת המלצות פרמטרים למסחר זוגי על בסיס PairDiagnostics.

    לוגיקה עיקרית:
    ---------------
    - z_entry:
        • בסיס: 2.0
        • זוג איכותי (quality_score גבוה) → z_entry נמוך יותר (1.6–2.0)
        • זוג חלש → z_entry גבוה יותר (2.4–3.0)
        • cointegration חלשה / VAR לא יציב → מעלה z_entry
        • tail_u גבוה (זנב חזק) → מעלה z_entry

    - z_exit:
        • ברירת מחדל ~ z_entry / 4, גבול עליון 0.8.

    - lookback:
        • חצי־חיים קצר (spread_half_life נמוך) → lookback קצר (30–60).
        • חצי־חיים ארוך → lookback ארוך (90–150).
        • ניתן לשלב עם hurst.

    - stop_z / take_profit_z:
        • stop_z מעט מעל z_entry (למשל z_entry + 1.0).
        • take_profit_z: אופציונלי – אם Hurst נמוך וחצי־חיים קצר.

    - max_holding_days:
        • ביחס לחצי־החיים (פי 2–3), או דיפולט 30 אם לא קיים.
    """
    rec: Dict[str, float] = {}

    # --- z_entry base ---
    z_entry = 2.0

    # quality-based tuning
    q = diag.quality_score if diag.quality_score is not None else 50.0
    if q >= 80:
        z_entry -= 0.3  # זוג מעולה → אפשר להיכנס קצת יותר אגרסיבי
    elif q >= 65:
        z_entry -= 0.1
    elif q < 50:
        z_entry += 0.3

    # cointegration / VAR penalties
    if diag.coint_pvalue is None or diag.coint_pvalue > 0.1:
        z_entry += 0.3
    if diag.var_is_stable is False:
        z_entry += 0.3

    # correlation penalties
    dyn_corr = diag.dyn_corr_mean if diag.dyn_corr_mean is not None else diag.corr
    if dyn_corr is not None and dyn_corr < 0.8:
        z_entry += 0.2

    # tail dependence penalty
    if diag.tail_u is not None and diag.tail_u > 0.3:
        z_entry += 0.2

    # clamp range
    z_entry = float(max(1.4, min(3.0, z_entry)))
    rec["z_entry"] = round(z_entry, 2)

    # z_exit ≈ quarter of z_entry, capped
    z_exit = min(0.8, z_entry / 4.0)
    rec["z_exit"] = round(z_exit, 2)

    # --- lookback ---
    # default
    lookback = 90.0

    hl = diag.spread_half_life
    if hl is not None and hl > 0:
        if hl <= 10:
            lookback = 30.0
        elif hl <= 20:
            lookback = 45.0
        elif hl <= 40:
            lookback = 60.0
        elif hl <= 60:
            lookback = 90.0
        else:
            lookback = 120.0
    else:
        # fallback via Hurst
        if diag.hurst is not None:
            if diag.hurst < 0.35:
                lookback = 45.0
            elif diag.hurst < 0.5:
                lookback = 60.0
            else:
                lookback = 120.0

    rec["lookback"] = float(lookback)

    # --- stop_z / take_profit_z ---
    stop_z = z_entry + 1.0
    stop_z = float(min(4.0, stop_z))
    rec["stop_z"] = round(stop_z, 2)

    # take profit: רק אם יש mean-reversion ברור
    take_z: Optional[float] = None
    if diag.hurst is not None and diag.hurst < 0.5 and (hl is not None and hl < 40):
        take_z = max(0.5, z_entry * 0.5)
        take_z = float(min(take_z, 1.5))
    rec["take_profit_z"] = round(take_z, 2) if take_z is not None else 0.0

    # --- max_holding_days ---
    if hl is not None and hl > 0:
        max_holding = int(min(hl * 3.0, 90))
        max_holding = max(max_holding, 10)
    else:
        max_holding = 30
    rec["max_holding_days"] = float(max_holding)

    # --- hedge style ---
    if diag.beta_ols is not None:
        if abs(diag.beta_ols - 1.0) < 0.1:
            hedge_style = "beta-tight"
        elif abs(diag.beta_ols) < 0.3 or abs(diag.beta_ols) > 2.0:
            hedge_style = "conservative"
        else:
            hedge_style = "ols"
    else:
        hedge_style = "naive"

    rec["beta_ols"] = float(diag.beta_ols) if diag.beta_ols is not None else 1.0
    rec["hedge_style"] = hedge_style

    # --- copy diagnostics into rec for reference (flat dict) ---
    for k, v in asdict(diag).items():
        # נשמור את המדדים המקוריים תחת prefix 'diag_' כדי לא לדרוס
        if v is not None and k not in rec:
            # לדוגמה: diag_coint_pvalue, diag_dyn_corr_mean וכו'
            rec[f"diag_{k}"] = float(v) if isinstance(v, (int, float)) else v

    # נוסיף גם את ציון האיכות בצורה גלויה
    if diag.quality_score is not None:
        rec["quality_score"] = float(diag.quality_score)
    if diag.quality_label is not None:
        rec["quality_label"] = str(diag.quality_label)

    return rec


def recommend_pair(price_a: pd.Series, price_b: pd.Series) -> Tuple[PairDiagnostics, Dict[str, float]]:
    """
    עטיפה נוחה:

        diag, rec = recommend_pair(price_a, price_b)

    diag = PairDiagnostics (אבחון מלא)
    rec  = Dict[str, float/str]   (סט פרמטרים מומלצים + המדדים)
    """
    diag = analyze_pair(price_a, price_b)
    rec = recommend_params(diag)
    return diag, rec
