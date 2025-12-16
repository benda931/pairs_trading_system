# -*- coding: utf-8 -*-
"""
core/fair_value_advisor.py — High-level advisor over Fair Value engine outputs
===============================================================================

המטרה:
- לקבל DataFrame של תוצאות FairValueEngine (שורה לכל זוג).
- לגזור ממנו אינדיקטורים גלובליים (quality, concentration, turnover, וכו').
- להחזיר:
    * summary — מדדים מספריים.
    * advice — רשימת עצות פרמטריות לשיפור.

⚠️ חשוב:
- המודול **לא** משנה כלום במערכת. הוא רק קורא DataFrame ומחזיר מילון.
- הוא בנוי להיות עמיד: אם חסרה עמודה מסוימת (dsr_net, rp_weight וכו') הוא פשוט מתעלם ממנה.
- כל ההיגדים הם Heuristics ברמת "קרן גידור" – נועדו לעזור לכוונן universe ופרמטרים, לא חוקים קשיחים.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class AdviceItem:
    """
    פריט עצה בודד:

    severity:
        "info" / "warning" / "critical"
    category:
        "edge", "risk", "turnover", "concentration",
        "stat_quality", "horizon", "robustness", "universe",
        "implementation", "data_quality", וכו'.
    message:
        טקסט קצר וברור לעין.
    rationale:
        הסבר קצת יותר מפורט למה עלינו לעדכן.
    suggested_changes:
        מילון של "פרמטר -> רעיון שינוי" (מחרוזות, לא קוד קשיח).
    """
    id: str
    severity: str
    category: str
    message: str
    rationale: str
    suggested_changes: Dict[str, str]


def _safe_series(df: pd.DataFrame, col: str) -> Optional[pd.Series]:
    """החזרת עמודה אם קיימת, אחרת None."""
    return df[col] if col in df.columns else None


def _percent(x: float) -> float:
    """המרה לאחוזים מעוגלים לשתי ספרות."""
    return float(np.round(100.0 * x, 2))


def _num_stats(series: Optional[pd.Series]) -> Dict[str, Optional[float]]:
    """
    מחשב סטטיסטיקות בסיסיות לסדרה נומרית:
    mean, median, std, p10, p90, min, max.
    אם אין סדרה/נתונים → ערכים None.
    """
    if series is None:
        return {
            "mean": None,
            "median": None,
            "std": None,
            "p10": None,
            "p90": None,
            "min": None,
            "max": None,
        }
    v = pd.to_numeric(series, errors="coerce").dropna()
    if v.empty:
        return {
            "mean": None,
            "median": None,
            "std": None,
            "p10": None,
            "p90": None,
            "min": None,
            "max": None,
        }
    return {
        "mean": float(v.mean()),
        "median": float(v.median()),
        "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
        "p10": float(v.quantile(0.10)),
        "p90": float(v.quantile(0.90)),
        "min": float(v.min()),
        "max": float(v.max()),
    }


def analyze_engine_rows(df_rows: pd.DataFrame) -> Dict[str, Any]:
    """
    ניתוח טבלת FairValueEngine (שורה = זוג) → summary + advice list.

    Parameters
    ----------
    df_rows : pd.DataFrame
        מצופה לכלול חלק מהעמודות:
        - dsr_net, psr_net, sr_net
        - rp_weight, target_pos_units, turnover_est, avg_hold_days
        - zscore, mispricing, net_edge_z, cost_spread_units
        - adf_p, residual_adf_p, is_coint, halflife
        - (אופציונלי) n_trades, liquidity_score, vol_regime, וכו'.

    Returns
    -------
    Dict[str, Any]
        {
          "summary": {...},
          "advice": [AdviceItem-as-dict, ...]
        }
    """
    df = df_rows.copy()
    n_pairs = len(df)
    advice: List[AdviceItem] = []

    if n_pairs == 0:
        return {
            "summary": {
                "n_pairs": 0,
                "note": "No pairs to analyze",
            },
            "advice": [
                asdict(
                    AdviceItem(
                        id="no_pairs",
                        severity="info",
                        category="universe",
                        message="אין זוגות לניתוח.",
                        rationale="DataFrame של FairValueEngine הגיע ריק.",
                        suggested_changes={
                            "universe": "להרחיב universe או להתיר פילטרים מעט (min_edge_z, min_corr)."
                        },
                    )
                )
            ],
        }

    # ------------- מדדים בסיסיים -------------
    dsr = _safe_series(df, "dsr_net")
    psr = _safe_series(df, "psr_net")
    sr = _safe_series(df, "sr_net")

    rp_weight = _safe_series(df, "rp_weight")
    turnover_est = _safe_series(df, "turnover_est")
    avg_hold_days = _safe_series(df, "avg_hold_days")
    net_edge_z = _safe_series(df, "net_edge_z")
    cost_units = _safe_series(df, "cost_spread_units")
    zscore = _safe_series(df, "zscore")

    adf_p = _safe_series(df, "adf_p")
    residual_adf_p = _safe_series(df, "residual_adf_p")
    is_coint = _safe_series(df, "is_coint")
    halflife = _safe_series(df, "halflife")

    n_trades = _safe_series(df, "n_trades") or _safe_series(df, "trades")  # אופציונלי
    liquidity = _safe_series(df, "liquidity_score")  # אופציונלי
    vol_regime = _safe_series(df, "vol_regime") or _safe_series(df, "regime_label")  # אופציונלי

    # ------------- Quality: כמה זוגות באמת טובים? -------------
    good_mask = None
    if dsr is not None:
        good_mask = dsr > 0.0
    elif psr is not None:
        good_mask = psr > 0.0
    elif sr is not None:
        good_mask = sr > 0.0

    if good_mask is not None:
        n_good = int(good_mask.sum())
        frac_good = n_good / n_pairs if n_pairs > 0 else 0.0
    else:
        n_good = None
        frac_good = None

    # ------------- Concentration: כמה ה-universe מרוכז? -------------
    conc_top5 = None
    conc_top10 = None
    w = None
    if rp_weight is not None:
        w = rp_weight.abs().fillna(0.0)
        w_sum = float(w.sum()) or 1.0
        w_sorted = w.sort_values(ascending=False)
        conc_top5 = float(w_sorted.head(5).sum() / w_sum)
        conc_top10 = float(w_sorted.head(10).sum() / w_sum)

    # ------------- Turnover + Edge -------------
    avg_turnover = float(turnover_est.mean()) if turnover_est is not None else None
    avg_hold = float(avg_hold_days.mean()) if avg_hold_days is not None else None
    avg_edge_z = float(net_edge_z.mean()) if net_edge_z is not None else None
    avg_cost_units = float(cost_units.mean()) if cost_units is not None else None

    # ------------- סטטיסטיקה: mean reversion / cointegration -------------
    avg_adf = float(adf_p.mean()) if adf_p is not None else None
    avg_resid_adf = float(residual_adf_p.mean()) if residual_adf_p is not None else None
    coint_rate = None
    if is_coint is not None:
        coint_rate = float((is_coint == True).mean())  # noqa: E712

    avg_halflife = None
    hl_stats: Dict[str, Optional[float]] = {}
    if halflife is not None:
        # מתעלמים מ-NaN ו-Inf
        hl = pd.to_numeric(halflife, errors="coerce").replace([np.inf, -np.inf], np.nan)
        if hl.notna().sum() > 0:
            hl_stats = _num_stats(hl)
            avg_halflife = hl_stats["mean"]

    # ------------- סטטיסטיקות עומק נוספות -------------
    dsr_stats = _num_stats(dsr)
    edge_stats = _num_stats(net_edge_z)
    to_stats = _num_stats(turnover_est)
    hold_stats = _num_stats(avg_hold_days)

    share_neg_edge = None
    share_high_turnover = None
    share_short_hold = None
    share_long_hold = None

    if net_edge_z is not None:
        s = pd.to_numeric(net_edge_z, errors="coerce")
        share_neg_edge = float((s <= 0).mean())

    if turnover_est is not None:
        s = pd.to_numeric(turnover_est, errors="coerce")
        share_high_turnover = float((s > 20).mean())  # למשל > 20 בשנה

    if avg_hold_days is not None:
        s = pd.to_numeric(avg_hold_days, errors="coerce")
        share_short_hold = float((s < 3).mean())
        share_long_hold = float((s > 30).mean())

    # ============================================================
    #  בניית summary (מורחב)
    # ============================================================
    summary: Dict[str, Any] = {
        # בסיס:
        "n_pairs": n_pairs,
        "n_good_pairs": n_good,
        "frac_good_pairs": _percent(frac_good) if frac_good is not None else None,
        "concentration_top5_pct": _percent(conc_top5) if conc_top5 is not None else None,
        "concentration_top10_pct": _percent(conc_top10) if conc_top10 is not None else None,
        "avg_turnover_est": avg_turnover,
        "avg_hold_days": avg_hold,
        "avg_edge_z": avg_edge_z,
        "avg_cost_spread_units": avg_cost_units,
        "avg_adf_p": avg_adf,
        "avg_residual_adf_p": avg_resid_adf,
        "coint_rate_pct": _percent(coint_rate) if coint_rate is not None else None,
        "avg_halflife": avg_halflife,
        # עומק DSR / Edge:
        "dsr_mean": dsr_stats["mean"],
        "dsr_median": dsr_stats["median"],
        "dsr_std": dsr_stats["std"],
        "dsr_p90": dsr_stats["p90"],
        "edge_mean": edge_stats["mean"],
        "edge_median": edge_stats["median"],
        "edge_std": edge_stats["std"],
        "edge_p90": edge_stats["p90"],
        # Turnover / Hold distribution:
        "turnover_mean": to_stats["mean"],
        "turnover_std": to_stats["std"],
        "hold_median": hold_stats["median"],
        "hold_p90": hold_stats["p90"],
        "share_negative_edge_pct": _percent(share_neg_edge) if share_neg_edge is not None else None,
        "share_high_turnover_pct": _percent(share_high_turnover) if share_high_turnover is not None else None,
        "share_short_hold_pct": _percent(share_short_hold) if share_short_hold is not None else None,
        "share_long_hold_pct": _percent(share_long_hold) if share_long_hold is not None else None,
        # Halflife spread:
        "halflife_std": hl_stats.get("std"),
        "halflife_p90": hl_stats.get("p90"),
    }

    # ============================================================
    #  עצות — Heuristics (הרבה יותר עשיר)
    # ============================================================

    # --- 0. Universe size too small / too large ---
    if n_pairs < 10:
        advice.append(
            AdviceItem(
                id="universe_too_small",
                severity="info",
                category="universe",
                message="Universe קטן מדי – קשה לבנות פורטפוליו מפוזר.",
                rationale=(
                    f"רק {n_pairs} זוגות באוניברס. זה מגביל את היכולת לפזר סיכון ולבחור "
                    "את הטובים ביותר מתוך סל רחב."
                ),
                suggested_changes={
                    "universe": "להרחיב universe (סקטורים, מדינות, horizons שונים) ולהפעיל שוב את ה-Engine.",
                    "filters": "להרפות מעט פילטרים ראשוניים (min_corr, min_edge_z) כדי לראות עוד מועמדים.",
                },
            )
        )

    if n_pairs > 500:
        advice.append(
            AdviceItem(
                id="universe_too_large",
                severity="info",
                category="universe",
                message="Universe ענק — עלול להיות כבד תפעולית וחישובית.",
                rationale=(
                    f"קיימים {n_pairs} זוגות באוניברס. זה מוסיף עומס על האופטימיזציה, Backtest, ומערכות הביצוע."
                ),
                suggested_changes={
                    "universe": "להגדיר universe 'ליבה' של זוגות איכותיים (top bucket לפי dsr_net/net_edge_z).",
                    "filters": "להקשיח מעט min_edge_z או איכות סטטיסטית כדי לצמצם כמות.",
                },
            )
        )

    # --- 1. מעט זוגות טובים (edge חיובי נמוך) ---
    if n_good is not None and n_good < max(5, 0.2 * n_pairs):
        advice.append(
            AdviceItem(
                id="low_good_pairs",
                severity="warning",
                category="edge",
                message="רוב הזוגות מייצרים edge נמוך או שלילי.",
                rationale=(
                    f"מתוך {n_pairs} זוגות רק {n_good} מוגדרים כבעלי תשואה מותאמת סיכון חיובית "
                    "(dsr_net/psr_net/sr_net > 0)."
                ),
                suggested_changes={
                    "universe": "לסנן מראש universe לחלונות/ענפים יותר יציבים, "
                                "או לדרוש איכות סטטיסטית גבוהה יותר (cointegration, mean reversion).",
                    "filters": "להעלות סף מינימלי ל-dsr_net/psr_net או ל-net_edge_z בעת בחירת זוגות למסחר.",
                },
            )
        )

    # --- 2. ריכוז משקולות גבוה מדי (Top-5) ---
    if conc_top5 is not None and conc_top5 > 0.6:
        advice.append(
            AdviceItem(
                id="high_concentration_top5",
                severity="warning",
                category="concentration",
                message="הפורטפוליו מרוכז מדי בכמה זוגות בודדים.",
                rationale=(
                    f"חמשת הזוגות הגדולים מחזיקים כ-{_percent(conc_top5)}% מהמשקל (rp_weight). "
                    "זה מגביר סיכון ספציפי לזוג."
                ),
                suggested_changes={
                    "max_rp_weight": "להגביל משקל מרבי לזוג יחיד (למשל 10–15%).",
                    "position_sizing": "ליישם cap על target_pos_units לפי VaR/ES לזוג בודד.",
                    "universe": "להרחיב מספר זוגות איכותיים כדי לפזר משקל.",
                },
            )
        )

    # --- 3. משקל רב על זוגות עם Edge שלילי/אפס ---
    if w is not None and dsr is not None:
        try:
            dsr_num = pd.to_numeric(dsr, errors="coerce")
            mask_bad = dsr_num <= 0
            if mask_bad.any():
                share_w_bad = float(w[mask_bad].sum() / (float(w.sum()) or 1.0))
                if share_w_bad > 0.25:
                    advice.append(
                        AdviceItem(
                            id="weight_on_negative_pairs",
                            severity="warning",
                            category="concentration",
                            message="חלק גדול מהמשקל נופל על זוגות עם Edge לא חיובי.",
                            rationale=(
                                f"כ-{_percent(share_w_bad)}% ממשקל ה-risk-parity יושב על זוגות עם dsr_net<=0. "
                                "זה מגדיל סיכון להפסד מצטבר."
                            ),
                            suggested_changes={
                                "filters": "להוציא או להוריד משקל לזוגות עם dsr_net<=0 או net_edge_z<=0.",
                                "rp_weight": "לחשב משקולות תוך penalize לזוגות עם Edge נמוך/שלילי.",
                            },
                        )
                    )
        except Exception:
            pass

    # --- 4. משקל נמוך מדי על זוגות חזקים ---
    if w is not None and dsr is not None:
        try:
            dsr_num = pd.to_numeric(dsr, errors="coerce")
            mask_strong = dsr_num > 1.0
            if mask_strong.any():
                share_w_strong = float(w[mask_strong].sum() / (float(w.sum()) or 1.0))
                if share_w_strong < 0.3:
                    advice.append(
                        AdviceItem(
                            id="underweight_strong_pairs",
                            severity="info",
                            category="edge",
                            message="זוגות החזקים ביותר מקבלים משקל יחסית נמוך.",
                            rationale=(
                                f"זוגות עם dsr_net>1.0 מחזיקים רק כ-{_percent(share_w_strong)}% מהמשקל. "
                                "ייתכן שכדאי להגדיל משקל לזוגות איכותיים במיוחד."
                            ),
                            suggested_changes={
                                "rp_weight": "להגדיר משקולות פונקציה של Edge (למשל פרופורציונלי ל-net_edge_z).",
                                "sizing": "להוסיף שכבת overweight לזוגות בחמישון העליון של dsr_net.",
                            },
                        )
                    )
        except Exception:
            pass

    # --- 5. Turnover גבוה עם Edge נמוך ---
    if avg_turnover is not None and avg_edge_z is not None:
        if avg_turnover > 15 and avg_edge_z < 0.5:
            advice.append(
                AdviceItem(
                    id="high_turnover_low_edge",
                    severity="warning",
                    category="turnover",
                    message="תחלופה גבוהה ביחס ל-edge — סיכוי לשחיקת עלויות.",
                    rationale=(
                        f"turnover_est ממוצע ≈ {avg_turnover:.1f} בשנה, בעוד net_edge_z ממוצע ≈ {avg_edge_z:.2f}. "
                        "שילוב כזה עלול לגרום לעלויות עסקה לשחוק את הרווח."
                    ),
                    suggested_changes={
                        "min_avg_hold": "להעלות min_avg_hold (לדוגמה מ-3→5 או 5→8 ימים) באופטימיזציה.",
                        "penalty_turnover": "להוסיף penalty_turnover חיובי ב-OptConfig כדי לדחוף לפתרונות איטיים יותר.",
                        "z_in_out": "להרחיב מעט z_out כדי לצאת מאוחר יותר, ולהעלות z_in כדי להיכנס רק בסטיות חזקות יותר.",
                    },
                )
            )

    # --- 6. חלק גדול מאוד של החזקות קצרות מאוד / ארוכות מאוד ---
    if share_short_hold is not None and share_short_hold > 0.5:
        advice.append(
            AdviceItem(
                id="too_many_short_holds",
                severity="warning",
                category="horizon",
                message="חלק גדול מהזוגות נסחרים בטווחי זמן קצרים מאוד.",
                rationale=(
                    f"כ-{_percent(share_short_hold)}% מהזוגות עם avg_hold_days<3. "
                    "זה מגביר רגישות למיקרוסטרוקטורה, עמלות ו-slippage."
                ),
                suggested_changes={
                    "min_avg_hold": "להגדיר constraint מינימלי של מספר ימי החזקה באופטימיזציה.",
                    "z_in_out": "להעלות z_in או להוריד z_out כדי לצמצם מסחר מיותר בסטיות קטנות.",
                },
            )
        )

    if share_long_hold is not None and share_long_hold > 0.5:
        advice.append(
            AdviceItem(
                id="too_many_long_holds",
                severity="info",
                category="horizon",
                message="רוב הזוגות נסחרים בחזקות ארוכות מאוד.",
                rationale=(
                    f"כ-{_percent(share_long_hold)}% מהזוגות עם avg_hold_days>30. "
                    "זה יכול לגרום ל-capital lock-up ולהקטין גמישות אלוקציה."
                ),
                suggested_changes={
                    "window": "להתאים window/secondary_windows לתקופות קצרות יותר כדי לתפוס mean-reversion טקטי.",
                    "portfolio": "להגדיר מגבלת זמן ממוצע לפוזיציה כחלק ממדיניות הסיכון.",
                },
            )
        )

    # --- 7. Edge נמוך ביחס לעלות ---
    if avg_edge_z is not None and avg_cost_units is not None and avg_cost_units > 0:
        ratio = avg_edge_z / avg_cost_units
        if ratio < 2.0:
            advice.append(
                AdviceItem(
                    id="edge_vs_cost_poor",
                    severity="warning",
                    category="edge",
                    message="ה-edge הנקי ביחס לעלויות נמוך.",
                    rationale=(
                        f"net_edge_z ממוצע ≈ {avg_edge_z:.2f}, בעוד cost_spread_units ≈ {avg_cost_units:.2f}, "
                        f"כלומר יחס edge/cost ≈ {ratio:.2f} בלבד."
                    ),
                    suggested_changes={
                        "costs_bps": "לבחון אם אפשר לעדכן הנחות עלויות (costs_bps/slippage_bps) או לעבור לזוגות נזילים יותר.",
                        "filters": "להגדיר פילטר שמוציא זוגות עם net_edge_z נמוך מדי ביחס לעלות אפקטיבית.",
                        "execution": "לשפר אלגוריתמי ביצוע (TWAP/VWAP) כדי להקטין slippage.",
                    },
                )
            )

    # --- 8. שונות גבוהה בעלויות ---
    if cost_units is not None:
        c_stats = _num_stats(cost_units)
        if c_stats["std"] is not None and c_stats["mean"] is not None:
            if c_stats["mean"] > 0 and c_stats["std"] / c_stats["mean"] > 1.0:
                advice.append(
                    AdviceItem(
                        id="costs_high_variability",
                        severity="info",
                        category="implementation",
                        message="עלויות העסקה שונות מאוד בין הזוגות.",
                        rationale=(
                            f"סטיית התקן של cost_spread_units גבוהה יחסית לממוצע. "
                            "זה מרמז על universe שמכיל גם זוגות זולים וגם יקרים מאוד לביצוע."
                        ),
                        suggested_changes={
                            "universe": "לשקול פילוח universe לפי רמת עלות/נזילות ולתכנן אסטרטגיות שונות.",
                            "filters": "להגביל universe לזוגות עם עלות ממוצעת נמוכה לסגנונות מסחר מהירים.",
                        },
                    )
                )

    # --- 9. איכות סטטיסטית נמוכה (mean reversion / cointegration) ---
    if avg_adf is not None and avg_adf > 0.1:
        advice.append(
            AdviceItem(
                id="weak_mean_reversion",
                severity="warning",
                category="stat_quality",
                message="עדות חלשה ל-mean reversion בספרדים.",
                rationale=(
                    f"ADF p-value ממוצע ≈ {avg_adf:.3f} (> 0.1), מה שמרמז שחלק גדול מהזוגות "
                    "לא מראים mean reversion חזק."
                ),
                suggested_changes={
                    "mean_revert_pvalue": "להקטין את הסף (למשל מ-0.1 ל-0.05) בבניית universe.",
                    "window": "להתאים window/secondary_windows כדי ללכוד horizon רלוונטי יותר.",
                },
            )
        )

    if coint_rate is not None and coint_rate < 0.4:
        advice.append(
            AdviceItem(
                id="low_coint_rate",
                severity="info",
                category="stat_quality",
                message="אחוז נמוך מהזוגות עומד בקריטריון קו-אינטגרציה.",
                rationale=(
                    f"רק כ-{_percent(coint_rate)}% מהזוגות מסומנים כ-is_coint=True. "
                    "זה לא בהכרח בעייתי, אבל דורש תשומת לב בבחירה סופית."
                ),
                suggested_changes={
                    "coint_filter": "להחמיר את סף coint_pvalue או לדרוש residual_adf_enabled במצב ON "
                                    "בזוגות שמגיעים למסחר אמיתי.",
                },
            )
        )

    # --- 10. הרבה משקל על זוגות שאינם קו-אינטגרטיביים ---
    if is_coint is not None and w is not None:
        mask_non = (is_coint == False)  # noqa: E712
        if mask_non.any():
            share_w_non = float(w[mask_non].sum() / (float(w.sum()) or 1.0))
            if share_w_non > 0.3:
                advice.append(
                    AdviceItem(
                        id="weight_on_non_coint",
                        severity="warning",
                        category="stat_quality",
                        message="משקל משמעותי מופנה לזוגות שאינם קו-אינטגרטיביים לפי המדדים.",
                        rationale=(
                            f"כ-{_percent(share_w_non)}% מהמשקל ב-universe יושב על זוגות עם is_coint=False. "
                            "זה יכול להגדיל סיכון drift במקום mean reversion."
                        ),
                        suggested_changes={
                            "universe": "להוציא או להפחית משקל לזוגות שאינם עומדים בקריטריון קו-אינטגרציה.",
                            "risk": "לדרוש בטיוב האסטרטגיה שזוגות non-coint יוקצו לפורטפוליו נפרד או יקבלו cap נמוך.",
                        },
                    )
                )

    # --- 11. Halflife קיצוני או פיזור חד ---
    if avg_halflife is not None:
        if avg_halflife < 1.0:
            advice.append(
                AdviceItem(
                    id="halflife_too_short",
                    severity="info",
                    category="horizon",
                    message="רוב הזוגות mean-reverting מהר מדי ביחס לחלונות.",
                    rationale=(
                        f"halflife ממוצע ≈ {avg_halflife:.1f} ימים בלבד. "
                        "זה יכול להעיד שה-window גדול מדי ביחס לדינמיקה בפועל."
                    ),
                    suggested_changes={
                        "window": "לשקול להקטין window או להוסיף secondary_windows קצרים יותר.",
                        "execution": "להבטיח שהמערכת מסוגלת להגיב תפעולית (ביצוע ועמלות) בקצב כזה.",
                    },
                )
            )
        elif avg_halflife > 60:
            advice.append(
                AdviceItem(
                    id="halflife_too_long",
                    severity="info",
                    category="horizon",
                    message="זמני ה-mean reversion ארוכים מאוד.",
                    rationale=(
                        f"halflife ממוצע ≈ {avg_halflife:.1f} ימים. "
                        "הספרדים נסגרים לאט, זה עלול לדרוש סבלנות גבוהה וריסק פרופיל אחר."
                    ),
                    suggested_changes={
                        "window": "לשקול חלונות קצרים יותר או פילטר שמוציא זוגות עם halflife קיצוני.",
                        "z_in_out": "להתאים z_in/z_out לגישה טקטית יותר (כניסה בסטיות חזקות בלבד).",
                    },
                )
            )

    if hl_stats.get("std") is not None and hl_stats.get("mean") is not None:
        std_hl = hl_stats["std"]
        mean_hl = hl_stats["mean"]
        if mean_hl and std_hl and std_hl > mean_hl:
            advice.append(
                AdviceItem(
                    id="halflife_high_dispersion",
                    severity="info",
                    category="horizon",
                    message="פיזור גדול מאוד ב-halflife בין הזוגות.",
                    rationale=(
                        "סטיית התקן של halflife גבוהה מהממוצע – חלק מהזוגות mean-reverting מהר מאוד "
                        "ואחרים מאוד איטיים. זה מקשה על ניהול אחיד של horizon."
                    ),
                    suggested_changes={
                        "segmentation": "לפצל universe ל-buckets לפי halflife ולעצב פרמטרים ייחודיים לכל bucket.",
                        "filters": "להוציא קצוות קיצוניים מאוד (halflife נמוך מאוד/גבוה מאוד) עבור אסטרטגיה ליבה.",
                    },
                )
            )

    # --- 12. Z-Score קיצוני על פני הרבה זוגים ---
    if zscore is not None:
        try:
            z = pd.to_numeric(zscore, errors="coerce")
            extreme = float((z.abs() > 3.5).mean())
            if extreme > 0.3:
                advice.append(
                    AdviceItem(
                        id="zscore_extreme_many",
                        severity="info",
                        category="robustness",
                        message="חלק משמעותי מהזוגות מציגים Z-Score קיצוני.",
                        rationale=(
                            f"כ-{_percent(extreme)}% מהזוגות נמצאים עם |Z|>3.5. "
                            "זה עלול להעיד על תקופת שוק חריגה או על חסרון ברובסטיות האמידה."
                        ),
                        suggested_changes={
                            "winsor": "להפעיל use_winsor/use_winsor_for_z או להקשיח zscore_clip.",
                            "vol_regime": "להפעיל volatility_regime_windows/labels כדי להתאים אסטרטגיה לרג'ים שונים.",
                        },
                    )
                )
        except Exception:
            pass

    # --- 13. Edge distribution — מעט מאוד זוגות עם Edge חזק ---
    if edge_stats["p90"] is not None and edge_stats["median"] is not None:
        if edge_stats["p90"] < 1.0 and edge_stats["median"] <= 0.0:
            advice.append(
                AdviceItem(
                    id="edge_distribution_weak",
                    severity="warning",
                    category="edge",
                    message="התפלגות ה-Edge חלשה – כמעט אין זוגות בעלי Edge חזק.",
                    rationale=(
                        "גם החמישון העליון של net_edge_z נמוך מ-1.0 והחציון קרוב ל-0 או שלילי. "
                        "ייתכן שהסיגנל הנוכחי לא מתאים לשוק/פרמטרים הנוכחיים."
                    ),
                    suggested_changes={
                        "features": "להרחיב או לשנות פיצ'רים ב-Fair Value Engine (גורמים מקרו/סקטור/סטייל).",
                        "strategy": "לשקול וריאציות אסטרטגיות אחרות (momentum / carry) עבור חלק מהזוגות.",
                    },
                )
            )

    # --- 14. שונות גבוהה ב-DSR (פיזור גדול בין זוגות) ---
    if dsr_stats["std"] is not None and dsr_stats["std"] > 0 and dsr_stats["mean"] is not None:
        if abs(dsr_stats["std"]) > 2 * abs(dsr_stats["mean"] or 0):
            advice.append(
                AdviceItem(
                    id="dsr_high_dispersion",
                    severity="info",
                    category="edge",
                    message="פיזור גדול מאוד ב-DSR בין הזוגות.",
                    rationale=(
                        "סטיית התקן של dsr_net גבוהה פי שניים מהממוצע (או יותר). "
                        "זה מרמז שיש מעט זוגות טובים מאוד והרבה בינוניים/חלשים."
                    ),
                    suggested_changes={
                        "universe": "להפריד בין זוגות ליבה (high-DSR) לבין ניסיוניים/לימודיים באלוקציה נמוכה.",
                        "sizing": "להקצות משקל גבוה יותר לזוגות בחמישון העליון של DSR.",
                    },
                )
            )

    # --- 15. יחס PSR / Sharpe — סכנת tail risk ---
    if psr is not None and sr is not None:
        try:
            psr_stats = _num_stats(psr)
            sr_stats = _num_stats(sr)
            if psr_stats["mean"] is not None and sr_stats["mean"] is not None:
                if psr_stats["mean"] < 0.5 * (sr_stats["mean"] or 0):
                    advice.append(
                        AdviceItem(
                            id="psr_weaker_than_sr",
                            severity="info",
                            category="robustness",
                            message="PSR חלש משמעותית מ-Sharpe – ייתכן ש-edge לא יציב.",
                            rationale=(
                                "ה-PSR (Probabilistic Sharpe Ratio) הממוצע נמוך משמעותית מה-Sharpe הממוצע, "
                                "מה שמרמז שהטווחים/פיזור התשואות עשויים לפגוע ביציבות ה-edge."
                            ),
                            suggested_changes={
                                "validation": "לבצע בדיקות יציבות נוספות (sub-samples / walk-forward).",
                                "filters": "להעלות סף PSR מינימלי ולא להסתמך על Sharpe בלבד.",
                            },
                        )
                    )
        except Exception:
            pass

    # --- 16. מעט טריידים אבל Edge גבוה — חשד לאוברפיט ---
    if n_trades is not None and dsr is not None:
        try:
            nt = pd.to_numeric(n_trades, errors="coerce")
            dsr_num = pd.to_numeric(dsr, errors="coerce")
            mask_suspicious = (dsr_num > 2.0) & (nt < 30)
            frac_susp = float(mask_suspicious.mean())
            if frac_susp > 0.2:  # יותר מ-20% מהזוגות
                advice.append(
                    AdviceItem(
                        id="high_edge_low_trades",
                        severity="warning",
                        category="robustness",
                        message="חלק גדול מהזוגות מראים Edge גבוה על מעט טריידים – חשד לאוברפיט.",
                        rationale=(
                            f"כ-{_percent(frac_susp)}% מהזוגות עם dsr_net>2 ו-n_trades<30. "
                            "זה מדגם קטן מדי כדי לבטוח בו כהוכחה ל-edge יציב."
                        ),
                        suggested_changes={
                            "backtest": "להאריך תקופת נתונים או להגדיל מספר טריידים לפני שקובעים פרמטרים סופיים.",
                            "filters": "להחמיר סף מינימלי ל-n_trades עבור זוגים שמגיעים לפרודקשן.",
                        },
                    )
                )
        except Exception:
            pass

    # --- 17. מצב Volatility Regime מעורב (אם קיימת עמודה) ---
    if vol_regime is not None:
        try:
            vc = vol_regime.astype(str).value_counts(normalize=True)
            if len(vc) > 3:
                advice.append(
                    AdviceItem(
                        id="many_vol_regimes",
                        severity="info",
                        category="robustness",
                        message="ה-universe משתרע על פני הרבה משטרי תנודתיות.",
                        rationale=(
                            "זוהו יותר משלושה vol_regime שונים (לפי העמודה vol_regime/regime_label). "
                            "ייתכן שכדאי לעצב פרמטרים ייעודיים לכל משטר ולא להשתמש בסט פרמטרים אחיד."
                        ),
                        suggested_changes={
                            "segmentation": "להריץ אופטימיזציה נפרדת לכל vol_regime / bucket מאקרו.",
                            "risk": "להגביל חשיפה מצטברת למשטרים קיצוניים במיוחד (high-vol, crisis).",
                        },
                    )
                )
        except Exception:
            pass

    # --- 18. מדדי נזילות (אם קיימים) ---
    if liquidity is not None:
        try:
            l_stats = _num_stats(liquidity)
            if l_stats["mean"] is not None and l_stats["mean"] < 0.3:
                advice.append(
                    AdviceItem(
                        id="liquidity_low",
                        severity="warning",
                        category="implementation",
                        message="ציון הנזילות הממוצע של ה-universe נמוך.",
                        rationale=(
                            "liquidity_score ממוצע נמוך מ-0.3 (בסקלה מנורמלת). "
                            "זה מעלה סיכון ל-slippage גבוה וקושי לגודל פוזיציות."
                        ),
                        suggested_changes={
                            "universe": "להוציא זוגות עם נזילות חלשה במיוחד, או להגביל אותם לגודל פוזיציה קטן.",
                            "costs_bps": "לעדכן את הנחות העלויות עבור זוגות עם liquidity_score נמוך.",
                        },
                    )
                )
        except Exception:
            pass

    # --- 19. בדיקת איכות דאטה — חסרים מדדים קריטיים ---
    missing_core = []
    for col in ("dsr_net", "net_edge_z", "halflife", "is_coint"):
        if col not in df.columns:
            missing_core.append(col)
    if missing_core:
        advice.append(
            AdviceItem(
                id="missing_core_metrics",
                severity="info",
                category="data_quality",
                message="חסרים חלק מהמדדים הקריטיים להערכת universe.",
                rationale=(
                    f"העמודות הבאות חסרות: {', '.join(missing_core)}. "
                    "זה לא עוצר את הניתוח, אבל מגביל את עומק ההבנה של איכות הזוגות."
                ),
                suggested_changes={
                    "engine_output": "להרחיב את ה-Engine כך שיחזיר את המדדים האלה לכל זוג.",
                    "reporting": "לעדכן צינור הדיווח (SQL/Parquet/DuckDB) כך שהעמודות יישמרו ונותחו.",
                },
            )
        )

    # --- 20. יותר מדי זוגות 'שוליים' עם Edge כמעט 0 ---
    if net_edge_z is not None:
        try:
            ez = pd.to_numeric(net_edge_z, errors="coerce")
            mask_marginal = (ez > -0.2) & (ez < 0.2)
            frac_marginal = float(mask_marginal.mean())
            if frac_marginal > 0.5:
                advice.append(
                    AdviceItem(
                        id="too_many_marginal_pairs",
                        severity="info",
                        category="edge",
                        message="חלק גדול מה-universe מורכב מזוגות עם Edge גבולי.",
                        rationale=(
                            f"כ-{_percent(frac_marginal)}% מהזוגות עם net_edge_z קרוב ל-0 (בין -0.2 ל-0.2). "
                            "זה הופך קשה להבדיל בין טובים לרעים מבחינת צפייה לרווח."
                        ),
                        suggested_changes={
                            "filters": "להקשיח סף מינימלי ל-net_edge_z כדי לא לבזבז משאבים על זוגות שוליים.",
                            "universe": "לאפשר universe רחב, אך לשמור 'ליבת מסחר' רק לזוגות שעוברים סף Edge גבוה.",
                        },
                    )
                )
        except Exception:
            pass

    # המרה ל-list של dicts
    advice_dicts = [asdict(a) for a in advice]

    return {
        "summary": summary,
        "advice": advice_dicts,
    }
