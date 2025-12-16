# -*- coding: utf-8 -*-
"""
common/macro_sensitivity.py — Pair-level macro sensitivity (HF-grade v1)
========================================================================

תפקיד הקובץ:
------------
1. למדוד את רגישות הזוג (spread/returns) לפקטורי מאקרו:
   - ריביות (קצר/ארוך), סלופים (slope_10y_3m, slope_10y_2y וכו').
   - וולאטיליות (VIX, MOVE, term structure של VIX).
   - קרדיט (credit_spread).
   - FX (DXY, USD strength).
   - Risk-on / Risk-off proxy.

2. לתת מדדים:
   - בטאות (beta) לכל פקטור.
   - t-stats / R² בסיסי.
   - ביצועים לפי משטרים (rates_high/rates_low, curve_inverted/neutral, high_vol/low_vol וכו').

3. בניית סיכום טקסטואלי קריא לזוג:
   - "בריביות גבוהות ועקום inverted, הזוג נוטה להיפתח לכיוון ..., התשואה הממוצעת היא ..."

תלות:
------
- common/macro_factors.py: macro_df + build_macro_regime_series / build_macro_snapshot.
- לא תלוי ב-Streamlit; אפשר להשתמש גם במחקר offline.

"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd

try:
    import statsmodels.api as sm  # type: ignore
    _HAS_SM = True
except Exception:
    _HAS_SM = False


# ==========================
# Part 1 — Models & Config
# ==========================


@dataclass
class MacroExposure:
    """חשיפה לפקטור מאקרו יחיד (beta + סטטיסטיקות בסיסיות)."""

    factor: str
    beta: float
    tstat: Optional[float] = None
    pvalue: Optional[float] = None
    r2_partial: Optional[float] = None


@dataclass
class RegimePerformance:
    """ביצועי הזוג תחת משטר מאקרו מסוים."""

    regime_label: str           # e.g. "rates_high", "curve_inverted"
    n_obs: int                  # מספר תצפיות
    mean_ret: float             # תשואה ממוצעת (או שינוי spread)
    median_ret: float           # חציון
    hit_ratio: float            # אחוז ימים חיוביים
    vol: float                  # סטיית תקן
    sharpe: float               # Sharpe יומי (mean / vol) אם vol>0


@dataclass
class PairMacroSensitivity:
    """
    סיכום רגישות מאקרו לזוג אחד.

    exposures      : dict factor -> MacroExposure.
    regime_perf    : dict regime_key -> RegimePerformance.
    overall_score  : float (0-100) — חוסן/מובהקות הקשר למאקרו (גבוה = זווית מאקרו חדה).
    summary_text   : str — סיכום מילולי איכותי להצגה בטאב.
    as_of          : Optional[pd.Timestamp] — תאריך אחרון של הדאטה.
    """

    exposures: Dict[str, MacroExposure]
    regime_perf: Dict[str, RegimePerformance]
    overall_score: float
    summary_text: str
    as_of: Optional[pd.Timestamp]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "exposures": {k: asdict(v) for k, v in self.exposures.items()},
            "regime_perf": {k: asdict(v) for k, v in self.regime_perf.items()},
            "overall_score": self.overall_score,
            "summary_text": self.summary_text,
            "as_of": self.as_of.isoformat() if self.as_of is not None else None,
        }


@dataclass
class PairMacroSensitivityConfig:
    """קונפיג לרגישות מאקרו לזוג.

    factor_cols    : רשימת עמודות מאקרו שישמשו כ-X ברגרסיה.
    ret_method     : "diff" | "pct" | "log" — איך לקבל returns/spread_delta מהסדרה הגולמית.
    min_obs        : מינימום מספר תצפיות לרגרסיה/סטטיסטיקה.
    recent_window  : חלון "חם" (ימים) לחשיפה עדכנית (לצד חשיפה ארוכת טווח).
    min_recent_obs : מינימום תצפיות לחלון החם.
    regime_keys    : אילו משטרים נבדקים (מילון: שם→עמודת משטר ב-regime_df).
    """

    factor_cols: Tuple[str, ...] = (
        "rate_short",
        "rate_long",
        "slope_10y_3m",
        "slope_10y_2y",
        "vix",
        "vix_term_1_0",
        "credit_spread",
        "dxy",
        "risk_on_proxy",
    )
    ret_method: str = "diff"  # "diff" for spread changes; "pct"/"log" for returns
    min_obs: int = 60
    recent_window: int = 252
    min_recent_obs: int = 40
    regime_keys: Dict[str, str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.regime_keys is None:
            # המפתחות כאן יהפכו לעמודות ב-regime_df
            self.regime_keys = {
                "rates": "rates_regime",
                "curve": "curve_regime",
                "vol": "vol_regime",
                "credit": "credit_regime",
                "fx": "fx_regime",
                "risk": "risk_regime",
            }


# ==========================
# Part 2 — Core computation
# ==========================


def _compute_returns(series: pd.Series, method: str) -> pd.Series:
    """המרת סדרת מחירים/Spread לסדרת תשואות/שינויים."""
    s = series.sort_index().astype(float)
    if method == "diff":
        return s.diff()
    if method == "log":
        return np.log(s).diff()
    if method == "pct":
        return s.pct_change()
    return s.diff()


def _align_pair_and_macro(
    pair_series: pd.Series,
    macro_df: pd.DataFrame,
) -> Tuple[pd.Series, pd.DataFrame]:
    """מיישר את סדרת הזוג ואת המאקרו על אינדקס משותף, מוריד NaN.

    מחזיר:
        (y, X) כאשר y=pair_series aligned, X=macro_df aligned.
    """
    y = pair_series.sort_index().astype(float)
    X = macro_df.sort_index()

    idx = y.index.intersection(X.index)
    if len(idx) == 0:
        return y.iloc[0:0], X.iloc[0:0]

    y = y.loc[idx]
    X = X.loc[idx]

    df = pd.concat([y.rename("y"), X], axis=1)
    df = df.dropna(how="any", axis=0)

    y_aligned = df["y"]
    X_aligned = df.drop(columns=["y"])
    return y_aligned, X_aligned


def _drop_constant_and_collinear(X: pd.DataFrame, corr_thresh: float = 0.98) -> pd.DataFrame:
    """מנקה עמודות קבועות/כמעט קבועות ועמודות מאוד מקולינאריות.

    - מוריד עמודות עם סטיית תקן ≈ 0.
    - מוריד עמודות עם מתאם |ρ|>corr_thresh אחת מול השנייה (משאיר נציג אחד).
    """
    if X.empty:
        return X

    Xc = X.copy()
    # הסרה של עמודות עם וריאציה כמעט אפס
    stds = Xc.std()
    keep = [c for c in Xc.columns if float(stds.get(c, 0.0)) > 1e-8]
    Xc = Xc[keep]

    if Xc.shape[1] <= 1:
        return Xc

    # הסרה של מתאם כמעט מלא
    corr = Xc.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > corr_thresh)]
    Xc = Xc.drop(columns=to_drop, errors="ignore")
    return Xc


def _ols_exposures(
    y: pd.Series,
    X: pd.DataFrame,
) -> Dict[str, MacroExposure]:
    """מבצע רגרסיית OLS רב-ממדית ומחזיר MacroExposure לכל factor.

    אם statsmodels זמין → נקבל גם t-stat/p-value; אחרת least-squares נקי.
    בנוסף, מסיר פקטורים קבועים/מקולינאריים כדי להימנע ממטריצות סינגולריות.
    """
    exposures: Dict[str, MacroExposure] = {}
    if y.empty or X.empty or len(y) < 10:
        return exposures

    # ניקוי פקטורים בעייתיים
    X = _drop_constant_and_collinear(X)
    if X.empty:
        return exposures

    cols = list(X.columns)
    X_mat = X.values.astype(float)
    y_vec = y.values.astype(float)

    if _HAS_SM:
        X_design = sm.add_constant(X_mat)
        model = sm.OLS(y_vec, X_design)
        res = model.fit()
        betas = res.params[1:]   # מדלגים על ה-constant
        tvals = res.tvalues[1:]
        pvals = res.pvalues[1:]
        # R² כולל
        # r2 = float(res.rsquared)  # אפשר להחזיר אם תרצה

        for i, name in enumerate(cols):
            exposures[name] = MacroExposure(
                factor=name,
                beta=float(betas[i]),
                tstat=float(tvals[i]),
                pvalue=float(pvals[i]),
                r2_partial=None,  # אפשר להרחיב ל-R² חלקי אם תרצה
            )
    else:
        # fallback פשוט: least squares
        X_design = np.column_stack([np.ones(len(X_mat)), X_mat])
        beta_vec, _, _, _ = np.linalg.lstsq(X_design, y_vec, rcond=None)
        betas = beta_vec[1:]  # בלי ה-constant

        for i, name in enumerate(cols):
            exposures[name] = MacroExposure(
                factor=name,
                beta=float(betas[i]),
                tstat=None,
                pvalue=None,
                r2_partial=None,
            )

    return exposures


def _compute_regime_performance(
    ret_series: pd.Series,
    regime_series: pd.Series,
    prefix: str,
    min_obs: int,
) -> Dict[str, RegimePerformance]:
    """
    מחשב ביצועים של הזוג לפי משטר אחד (עמודה אחת של Regime).

    ret_series   : סדרת תשואות/שינויים (יומית).
    regime_series: סדרת תוויות (למשל "rates_high","rates_low"...).
    prefix       : prefix לשם המפתח (e.g. "rates", "curve").
    """
    perf: Dict[str, RegimePerformance] = {}
    if ret_series.empty or regime_series.empty:
        return perf

    df = pd.DataFrame({"ret": ret_series, "regime": regime_series}).dropna(how="any", axis=0)
    if df.empty:
        return perf

    for regime_label, grp in df.groupby("regime"):
        n = len(grp)
        if n < min_obs / 4:  # לפחות 1/4 מהנמוך הכללי
            continue
        m = float(grp["ret"].mean())
        med = float(grp["ret"].median())
        vol = float(grp["ret"].std())
        hit = float((grp["ret"] > 0).mean()) if n > 0 else 0.0
        sharpe = m / vol if vol > 0 else 0.0
        key = f"{prefix}:{regime_label}"
        perf[key] = RegimePerformance(
            regime_label=str(regime_label),
            n_obs=n,
            mean_ret=m,
            median_ret=med,
            hit_ratio=hit,
            vol=vol,
            sharpe=sharpe,
        )

    return perf


def compute_pair_macro_sensitivity(
    pair_series: pd.Series,
    macro_df: pd.DataFrame,
    regime_df: Optional[pd.DataFrame] = None,
    cfg: Optional[PairMacroSensitivityConfig] = None,
) -> PairMacroSensitivity:
    """
    הפונקציה המרכזית: מחשבת רגישות מאקרו לזוג אחד.

    Parameters
    ----------
    pair_series : Series
        סדרת spread או price/PNL של הזוג (index=DateTimeIndex).
    macro_df : DataFrame
        DataFrame של פקטורי מאקרו (output של load_macro_factors + add_derived_factors).
    regime_df : DataFrame, optional
        DataFrame של משטרים לאורך זמן (output של build_macro_regime_series).
    cfg : PairMacroSensitivityConfig, optional
        קונפיג לחישוב (פקטורים רלוונטיים, שיטת תשואה, וכו').
    """
    cfg = cfg or PairMacroSensitivityConfig()

    # 1) Compute returns / spread deltas
    y_full = _compute_returns(pair_series, cfg.ret_method)

    # 2) Filter macro columns
    cols = [c for c in cfg.factor_cols if c in macro_df.columns]
    X_full = macro_df[cols].copy()
    # 3) Align
    y, X = _align_pair_and_macro(y_full, X_full)

    exposures = _ols_exposures(y, X)
    regime_perf: Dict[str, RegimePerformance] = {}

    # 4) Regime-based performance (אם יש regime_df)
    if regime_df is not None and not regime_df.empty:
        reg_aligned = regime_df.reindex(y.index).dropna(how="all", axis=0)
        # מיישרים שוב y לפי reg_aligned כדי לא לספור תאריכים בלי משטר
        y_reg = y.reindex(reg_aligned.index).dropna()
        for key, col in cfg.regime_keys.items():
            if col not in reg_aligned.columns:
                continue
            perf_k = _compute_regime_performance(
                ret_series=y_reg,
                regime_series=reg_aligned[col],
                prefix=key,
                min_obs=cfg.min_obs,
            )
            regime_perf.update(perf_k)

    # 5) Overall score: קומבינציה של:
    #    - כמה בטאות "משמעותיות" (tstat גדול / abs(beta) גדול)
    #    - האם יש הבדל ביצועים חזק בין משטרים
    overall_score = _compute_overall_score(exposures, regime_perf, min_obs=cfg.min_obs)

    # 6) Summary text in Hebrew
    summary_text = summarize_pair_macro_sensitivity(exposures, regime_perf, overall_score)

    as_of = pair_series.dropna().index.max() if not pair_series.dropna().empty else None

    return PairMacroSensitivity(
        exposures=exposures,
        regime_perf=regime_perf,
        overall_score=overall_score,
        summary_text=summary_text,
        as_of=as_of,
    )


# ==========================
# Part 3 — Scoring & Text
# ==========================


def _compute_overall_score(
    exposures: Dict[str, MacroExposure],
    regime_perf: Dict[str, RegimePerformance],
    min_obs: int,
) -> float:
    """
    מחשב ציון כללי (0-100) לזוג על בסיס:
    - גודל הבטאות (abs(beta)) לפקטורים חשובים (rates, slope, vix, credit, risk_on).
    - מספר משטרים שבהם יש Sharpe "חזק".
    """

    if not exposures and not regime_perf:
        return 0.0

    # משקל לפי פקטורים
    factor_weights: Dict[str, float] = {
        "rate_short": 1.2,
        "rate_long": 1.2,
        "slope_10y_3m": 1.5,
        "slope_10y_2y": 1.5,
        "vix": 1.3,
        "vix_term_1_0": 1.0,
        "credit_spread": 1.1,
        "dxy": 0.8,
        "risk_on_proxy": 1.3,
    }

    beta_score = 0.0
    beta_weight_sum = 0.0
    for name, exp in exposures.items():
        w = factor_weights.get(name, 0.5)
        beta_score += w * min(abs(exp.beta), 3.0)  # מגביל בטאות קיצוניות
        beta_weight_sum += w

    beta_score_norm = beta_score / beta_weight_sum if beta_weight_sum > 0 else 0.0  # ~0..3

    # משטרים עם Sharpe משמעותי
    regime_score = 0.0
    n_good_regimes = 0
    for rp in regime_perf.values():
        if rp.n_obs < max(10, min_obs // 4):
            continue
        if abs(rp.sharpe) >= 0.5:
            n_good_regimes += 1
            regime_score += min(abs(rp.sharpe), 3.0)

    regime_score_norm = regime_score / max(1, n_good_regimes) if n_good_regimes > 0 else 0.0

    # משקל (70% בטאות, 30% משטרים)
    raw = 0.7 * beta_score_norm + 0.3 * regime_score_norm
    # נרצה סולם 0-100; נניח raw ∈ [0,3] → כפול ~25
    score = max(0.0, min(100.0, raw * 25.0))
    return float(score)


def summarize_pair_macro_sensitivity(
    exposures: Dict[str, MacroExposure],
    regime_perf: Dict[str, RegimePerformance],
    overall_score: float,
) -> str:
    """
    מייצר סיכום מילולי קריא של רגישות מאקרו לזוג.

    הרעיון:
    - למצוא את הפקטורים הדומיננטיים (beta גבוה).
    - לבדוק אם הזוג אוהב/שונא:
        * ריביות גבוהות/נמוכות, עקום inverted/steepener, high_vol/low_vol, וכו'.
    - להחזיר פסקה קצרה שאפשר להציג בטאב המאקרו.
    """

    if not exposures and not regime_perf:
        return "לא נמצאה רגישות מאקרו מובהקת לזוג (אין מספיק דאטה או שאין קשר ברור)."

    # פקטורים דומיננטיים
    # ניקח את 3 הפקטורים עם |beta| הגדולה ביותר
    exp_list = list(exposures.values())
    exp_list.sort(key=lambda e: abs(e.beta), reverse=True)
    top_exposures = exp_list[:3]

    lines: List[str] = []

    # שורה ראשונה: ציון כללי
    if overall_score >= 70:
        lines.append(f"רגישות מאקרו חזקה (ציון {overall_score:.0f}/100).")
    elif overall_score >= 40:
        lines.append(f"רגישות מאקרו בינונית (ציון {overall_score:.0f}/100).")
    else:
        lines.append(f"רגישות מאקרו חלשה (ציון {overall_score:.0f}/100).")

    # פירוט לפי פקטורים
    for exp in top_exposures:
        name = exp.factor
        beta = exp.beta
        direction = "עולה" if beta > 0 else "יורדת"
        f_desc = {
            "rate_short": "שינוי בריבית קצרה",
            "rate_long": "שינוי בריבית ארוכה (10Y)",
            "slope_10y_3m": "ה steepness בעקום 10Y–3M",
            "slope_10y_2y": "ה steepness בעקום 10Y–2Y",
            "vix": "רמת ה-VIX",
            "vix_term_1_0": "Term structure של ה-VIX (contango/backwardation)",
            "credit_spread": "מרווחי האשראי",
            "dxy": "עוצמת הדולר (DXY)",
            "risk_on_proxy": "מדד risk-on/risk-off",
        }.get(name, name)

        lines.append(
            f"- {f_desc}: כאשר הפקטור {direction}, הזוג נוטה ל"
            + ("עלות" if beta > 0 else "רדת")  # אפשר לשפר להבדיל long/short, כאן זה high-level
            + f" (beta ≈ {beta:.2f})."
        )

    # Regime hints: איפה הזוג עובד טוב/רע
    # ניקח את שלושת המשטרים עם Sharpe הגבוה/נמוך ביותר
    rp_list = list(regime_perf.values())
    rp_list.sort(key=lambda r: r.sharpe, reverse=True)
    top_regimes = rp_list[:3]
    rp_list.sort(key=lambda r: r.sharpe)
    worst_regimes = rp_list[:2]

    if top_regimes:
        lines.append("הזוג נוטה להציג ביצועים חזקים במצבים הבאים:")
        for rp in top_regimes:
            if rp.n_obs < 10:
                continue
            lines.append(
                f"  • {rp.regime_label}: Sharpe {rp.sharpe:.2f}, ממוצע יומי {rp.mean_ret:.4f}, hit-ratio {rp.hit_ratio:.0%} (n={rp.n_obs})."
            )

    if worst_regimes:
        lines.append("לעומת זאת, ביצועים חלשים נצפו ב:")
        for rp in worst_regimes:
            if rp.n_obs < 10:
                continue
            lines.append(
                f"  • {rp.regime_label}: Sharpe {rp.sharpe:.2f}, ממוצע יומי {rp.mean_ret:.4f}, hit-ratio {rp.hit_ratio:.0%} (n={rp.n_obs})."
            )

    return "\n".join(lines)


def build_exposures_table(exposures: Dict[str, MacroExposure]) -> pd.DataFrame:
    """בונה טבלת חשיפות (betas וכו') נוחה להצגה בטאב המאקרו."""
    if not exposures:
        return pd.DataFrame(columns=["factor","beta","tstat","pvalue"])
    rows = []
    for name, exp in exposures.items():
        rows.append(
            {
                "factor": name,
                "beta": float(exp.beta),
                "tstat": float(exp.tstat) if exp.tstat is not None else np.nan,
                "pvalue": float(exp.pvalue) if exp.pvalue is not None else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("factor").reset_index(drop=True)


def build_regime_performance_table(regime_perf: Dict[str, RegimePerformance]) -> pd.DataFrame:
    """טבלת ביצועים לפי משטרים (Sharpe, hit-ratio וכו')."""
    if not regime_perf:
        return pd.DataFrame(
            columns=["regime_key","regime_label","n_obs","mean_ret","median_ret","hit_ratio","vol","sharpe"]
        )
    rows = []
    for key, rp in regime_perf.items():
        rows.append(
            {
                "regime_key": key,
                "regime_label": rp.regime_label,
                "n_obs": rp.n_obs,
                "mean_ret": rp.mean_ret,
                "median_ret": rp.median_ret,
                "hit_ratio": rp.hit_ratio,
                "vol": rp.vol,
                "sharpe": rp.sharpe,
            }
        )
    return pd.DataFrame(rows).sort_values("regime_key").reset_index(drop=True)


__all__ = [
    "MacroExposure",
    "RegimePerformance",
    "PairMacroSensitivity",
    "PairMacroSensitivityConfig",
    "compute_pair_macro_sensitivity",
    "summarize_pair_macro_sensitivity",
    "build_exposures_table",
    "build_regime_performance_table",
]
