# -*- coding: utf-8 -*-
"""
core/index_fundamentals.py — Hedge-Fund-Grade Index Fundamental Engine (Part 1/4)
=================================================================================

מטרה כללית
-----------
מודול זה הוא "מנוע הפנדומנטל" ברמת מדד/ETF במערכת שלך – מעל
`common.fundamental_loader` – והוא מתוכנן מראש כדי לתמוך בכל מה שתכננו:

1. ציוני Value / Quality / Growth / Composite ברמת מדד בודד.
2. ציונים ל-Universe של מדדים (SPY / QQQ / IWM / EEM / סקטורים / אזורי עולם).
3. ניתוח רב-אופקי (1Y / 3Y / 5Y / 10Y) של תמחור (percentiles, z-scores).
4. פירוק תשואה ל־Earnings vs Re-rating vs Dividends.
5. חיבור למאקרו, Regimes, ועולם הסיכון (Risk overlays).

בנוסף, המודול מתוכנן להכיל **סט מורחב של 20 רעיונות חדשים** (חלקם ימומשו
בחלקים 2–4, חלקם יסומנו כ-Hooks) ברמת קרן גידור מקצועית:

20 רעיונות / יכולות מתקדמות (תכנון על)
---------------------------------------
1. Value / Quality / Growth Score:
   - ציוני בסיס (0–100) לכל מדד לפי פיצ'רים פנדומנטליים סטנדרטיים.
2. Composite Score Engine:
   - שילוב משוקלל של V/Q/G, עם יכולת להגדיר פרופילים שונים (Value-heavy,
     Quality-heavy וכו').
3. Score Momentum:
   - מגמת השינוי של הציונים (ΔScore) על אופקי זמן שונים (3M / 6M / 12M).
4. Score Stability:
   - מדד ליציבות/תנודתיות של הציונים לאורך זמן (std / MAD / drawdown של score).
5. Cross-Sectional Ranking:
   - דירוג המדדים ברמת Universe לפי Value / Quality / Growth / Composite בכל תאריך.
6. Cross-Sectional Percentiles:
   - Percentile של כל מדד מול ה-Universe (למשל "SPY ב-90% העליון ב-Quality").
7. Regime-Aware Scores:
   - חישוב ציונים שונים למשטרי מאקרו שונים (Low Rates / High Inflation וכו'),
     בשילוב עם core/macro_engine.
8. Risk-Adjusted Composite:
   - התאמת ה-Composite Score לסיכון (Vol / Drawdown / Tail-risk), כך שמדד עם
     ציון Composite גבוה אבל tail-risk קיצוני יקבל penalty.
9. Factor-Neutral Composite:
   - בניית ציונים "ניטרליים" לפקטורים (למשל Quality-neutral-Value, או Value-neutral-Growth),
     כדי לא להיות שבויי הפקטורים הגדולים.
10. Fair-Value Gap Signal:
    - חישוב פער בין התמחור הנוכחי לבין "Fair Value" פנדומנטלי גס (שימוש עתידי
      עם core/fair_value_engine אם תרצה).
11. Score Decomposition:
    - פירוק ציוני Composite להתרומה מכל רכיב (Value / Quality / Growth / Risk),
      לצורך הסבריות (Explainability).
12. Transition Matrix של Buckets:
    - ניתוח מעבר בין Buckets (למשל quintiles של Value score) לאורך זמן – האם
      מדדים ש"עולים" ל-Bucket עליון נשארים שם או נודדים.
13. Clustering of Index Profiles:
    - קלאסטרינג של מדדים לפי פרופיל פנדומנטלי/Score (Quality Tech, Cyclical Value,
      EM Growth וכו').
14. Score-Based Over/Underweight Engine:
    - מנוע המלצה ל-Overweight / Underweight לכל מדד, מבוסס על תמהיל:
      Value-cheapness, Quality, Growth momentum, Risk profile, Regime.
15. Confidence / Data-Quality Score:
    - ציון בטחון לכל מדד, על בסיס כמות ההיסטוריה, כיסוי פיצ'רים, אנומליות וכו'.
16. Multi-Horizon View:
    - עבור כל מדד, Summary דו"ח קטן:
      * current scores
      * percentiles 1Y/5Y/10Y
      * score momentum
      * regime performance summary.
17. Universe Summary & Heatmaps:
    - טבלאות / מטריצות שיהוו קלט לטאבים: מי בזול/ביקר, מי באיכות גבוהה/נמוכה.
18. Stress-Tested Scores:
    - התאמת ציונים לפי תרחישי מאקרו (אם הריבית עולה ב-2%, איך זה משנה
      את composite או את fair value gap?).
19. Alignment with Pairs/Correlation:
    - בניית שדות שמתחברים ל-core/pairs_trading: פערי Value/Quality בין מדדים כבסיס
      ל-Index Spreads.
20. Export-Friendly Structures:
    - החזרת תוצאות כ-DataFrames מסודרים, dicts וכו', כדי להתחבר בקלות ל־UI
      (Streamlit Tabs) ולמודולים אחרים.

מבנה החלקים בקובץ זה
---------------------
- Part 1/4 (כאן):
    * imports, logger
    * dataclasses ל־Config ו-Scores
    * קבועים (שמות פיצ'רים פנדומנטליים, ברירות מחדל)
    * scaffolding לרעיונות המתקדמים (שדות, placeholders)

- Part 2/4:
    * Utilities – Z-scores, Percentiles, Normalization
    * חישובי Value / Quality / Growth בסיסיים
    * ציוני Composite, Score Momentum, Score Stability

- Part 3/4:
    * High-level API:
        - score_index_fundamentals(...)
        - score_universe_fundamentals(...)
        - compute_fundamental_percentiles_for_symbol(...)
        - cross-sectional ranks/percentiles

- Part 4/4:
    * Advanced:
        - Fair-value gap hooks
        - Regime-aware scores
        - Risk-adjusted composite
        - Transition matrices, clustering hooks
        - decomposition & export utilities
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Dict, Iterable, Mapping, Sequence, Optional, Tuple, List

import numpy as np
import pandas as pd

from common.helpers import get_logger
from common.fundamental_loader import (
    FundamentalFrame,
    FundamentalPanel,
    load_index_fundamentals,
    build_fundamentals_panel,
    get_latest_fundamentals_snapshot,
)

logger = get_logger("core.index_fundamentals")


# ============================================================
# 1) קבועים – שמות פיצ'רים פנדומנטליים סטנדרטיים
# ============================================================

# שדות קשורים ל-Value (תמחור)
VALUE_FIELDS_DEFAULT: Tuple[str, ...] = (
    "pe",
    "pe_forward",
    "pb",
    "price_to_book",
    "earnings_yield",
    "dividend_yield",
    "fcf_yield",
    "ev_ebitda",
    "price_sales",
)

# שדות קשורים ל-Quality (איכות רווח ומאזן)
QUALITY_FIELDS_DEFAULT: Tuple[str, ...] = (
    "roe",
    "roa",
    "roic",
    "gross_margin",
    "operating_margin",
    "net_margin",
    "interest_coverage",
    "net_debt_to_ebitda",
)

# שדות קשורים ל-Growth (צמיחה)
GROWTH_FIELDS_DEFAULT: Tuple[str, ...] = (
    "eps_growth_3y",
    "eps_growth_5y",
    "revenue_growth_3y",
    "revenue_growth_5y",
    "fcf_growth_3y",
    "fcf_growth_5y",
)

# שדות נוספים שעשויים לשמש לחישובי Fair-Value / Regime
EXTRA_FIELDS_DEFAULT: Tuple[str, ...] = (
    "payout_ratio",
    "shares_outstanding",
    "market_cap",
)

# שמות פיצ'רים משולבים
ALL_FUNDAMENTAL_FIELDS_DEFAULT: Tuple[str, ...] = (
    *VALUE_FIELDS_DEFAULT,
    *QUALITY_FIELDS_DEFAULT,
    *GROWTH_FIELDS_DEFAULT,
    *EXTRA_FIELDS_DEFAULT,
)


# ============================================================
# 2) Configs – Value / Quality / Growth / Composite
# ============================================================

@dataclass
class ValueScoreConfig:
    """
    הגדרת משקלים לציון Value.

    lower_is_better metrics:
        * pe, pe_forward, pb, price_to_book, ev_ebitda, price_sales
    higher_is_better metrics:
        * dividend_yield, earnings_yield, fcf_yield
    """
    weight_pe: float = 1.0
    weight_pe_forward: float = 0.75
    weight_pb: float = 0.75
    weight_price_to_book: float = 0.0  # לרוב חופף ל-pb
    weight_dividend_yield: float = 0.75
    weight_earnings_yield: float = 1.0
    weight_fcf_yield: float = 1.0
    weight_ev_ebitda: float = 0.5
    weight_price_sales: float = 0.25

    def total_weight(self) -> float:
        return max(
            1e-9,
            self.weight_pe
            + self.weight_pe_forward
            + self.weight_pb
            + self.weight_price_to_book
            + self.weight_dividend_yield
            + self.weight_earnings_yield
            + self.weight_fcf_yield
            + self.weight_ev_ebitda
            + self.weight_price_sales,
        )


@dataclass
class QualityScoreConfig:
    """
    משקלים ל-Quality:

    higher_is_better:
        * ROE, ROA, ROIC
        * Margins (gross, operating, net)
        * Interest coverage
    lower_is_better:
        * Net debt / EBITDA
    """
    weight_roe: float = 1.0
    weight_roa: float = 0.5
    weight_roic: float = 0.75
    weight_gross_margin: float = 0.5
    weight_operating_margin: float = 0.5
    weight_net_margin: float = 0.75
    weight_interest_coverage: float = 0.75
    weight_net_debt_to_ebitda: float = 1.0  # lower is better

    def total_weight(self) -> float:
        return max(
            1e-9,
            self.weight_roe
            + self.weight_roa
            + self.weight_roic
            + self.weight_gross_margin
            + self.weight_operating_margin
            + self.weight_net_margin
            + self.weight_interest_coverage
            + self.weight_net_debt_to_ebitda,
        )


@dataclass
class GrowthScoreConfig:
    """
    משקלים לציון Growth:

    higher_is_better:
        * EPS growth (3Y/5Y)
        * Revenue growth (3Y/5Y)
        * FCF growth (3Y/5Y)
    """
    weight_eps_growth_3y: float = 1.0
    weight_eps_growth_5y: float = 0.75
    weight_revenue_growth_3y: float = 0.75
    weight_revenue_growth_5y: float = 0.5
    weight_fcf_growth_3y: float = 0.5
    weight_fcf_growth_5y: float = 0.25

    def total_weight(self) -> float:
        return max(
            1e-9,
            self.weight_eps_growth_3y
            + self.weight_eps_growth_5y
            + self.weight_revenue_growth_3y
            + self.weight_revenue_growth_5y
            + self.weight_fcf_growth_3y
            + self.weight_fcf_growth_5y,
        )


@dataclass
class CompositeScoreConfig:
    """
    משקלים לציון משוקלל (Composite) של Value / Quality / Growth.

    מאפיינים:
    ----------
    - ניתן להגדיר פרופילים שונים:
        * value_biased: weight_value גבוה יותר
        * quality_biased: weight_quality גבוה יותר
        * growth_biased: weight_growth גבוה יותר
    """
    weight_value: float = 0.35
    weight_quality: float = 0.35
    weight_growth: float = 0.30

    def total_weight(self) -> float:
        return max(
            1e-9,
            self.weight_value
            + self.weight_quality
            + self.weight_growth,
        )


@dataclass
class RiskOverlayConfig:
    """
    קונפיג לציון Risk-adjusted composite:

    - higher_is_better: low volatility, low drawdown, low tail-risk.
    - weights מאפשרים כמה לתת משקל לסיכון ביחס לציונים הפנדומנטליים.
    """
    weight_volatility_penalty: float = 0.5
    weight_drawdown_penalty: float = 0.5
    weight_tail_risk_penalty: float = 0.5

    # איך נשלב עם composite الأساسي (0–1: 0 = להתעלם, 1 = לתת full effect)
    overlay_strength: float = 0.5


@dataclass
class RegimeScoreConfig:
    """
    קונפיג לציונים "מודעי משטר" (Regime-aware):

    מאפשר להגדיר:
    - אילו משטרים מעניינים אותנו (שמות).
    - משקלים לציון כל משטר.

    ההנחה:
    -------
    core/macro_engine יספק mapping של שורות בזמן → משטר.
    """
    regimes: Sequence[str] = field(default_factory=lambda: ["low_rates", "high_inflation", "tightening", "easing"])
    regime_weights: Mapping[str, float] = field(
        default_factory=lambda: {
            "low_rates": 0.25,
            "high_inflation": 0.25,
            "tightening": 0.25,
            "easing": 0.25,
        }
    )


@dataclass
class FactorNeutralConfig:
    """
    קונפיג לנטרול חלקי של פקטורים (Value/Growth/Size וכו') בציון Composite.

    הרעיון:
    -------
    - נאמוד חשיפות לפקטורים (לדוגמה באמצעות core/factor_models בעתיד).
    - נשתמש ב-neutralization_strength כדי להקטין bias פקטורי.

    כרגע הקונפיג משמש כ-hook; המימוש המלא יגיע בחלק 4/4.
    """
    neutralize_value: bool = False
    neutralize_quality: bool = False
    neutralize_growth: bool = False
    neutralize_size: bool = False
    neutralize_region: bool = False

    neutralization_strength: float = 0.5  # 0=no neutralization, 1=full neutralization


# Configs ברירת מחדל ברמת מודול
DEFAULT_VALUE_CONFIG = ValueScoreConfig()
DEFAULT_QUALITY_CONFIG = QualityScoreConfig()
DEFAULT_GROWTH_CONFIG = GrowthScoreConfig()
DEFAULT_COMPOSITE_CONFIG = CompositeScoreConfig()
DEFAULT_RISK_OVERLAY_CONFIG = RiskOverlayConfig()
DEFAULT_REGIME_CONFIG = RegimeScoreConfig()
DEFAULT_FACTOR_NEUTRAL_CONFIG = FactorNeutralConfig()


# ============================================================
# 3) Dataclasses לתוצאת ציונים / פרופיל מדד
# ============================================================

@dataclass
class IndexScoreRow:
    """
    שורת ציונים בודדת למדד (תאריך אחד):

    מיועד גם לשימוש בדוחות/ייצוא JSON/Markdown.
    """
    date: pd.Timestamp
    symbol: str
    value_score: float
    quality_score: float
    growth_score: float
    composite_score: float

    # אופציונלי: Risk-adjusted / Regime-aware / Factor-neutral
    composite_risk_adjusted: Optional[float] = None
    composite_regime_adjusted: Optional[float] = None
    composite_factor_neutral: Optional[float] = None

    # Metadata:
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IndexScoreSeries:
    """
    סדרת ציונים לאורך הזמן למדד אחד (לניתוח היסטורי).

    fields:
        symbol       – שם המדד.
        df           – DataFrame עם אינדקס תאריך ועמודות scores.
        score_config – הקונפיגים ששימשו בחישוב.
    """
    symbol: str
    df: FundamentalFrame
    value_config: ValueScoreConfig = field(default_factory=lambda: DEFAULT_VALUE_CONFIG)
    quality_config: QualityScoreConfig = field(default_factory=lambda: DEFAULT_QUALITY_CONFIG)
    growth_config: GrowthScoreConfig = field(default_factory=lambda: DEFAULT_GROWTH_CONFIG)
    composite_config: CompositeScoreConfig = field(default_factory=lambda: DEFAULT_COMPOSITE_CONFIG)


@dataclass
class UniverseScoresSummary:
    """
    סיכום ציונים רב-מדדי ל-Universe (שימושי לטאבים / Heatmaps):

    fields:
        panel          – FundamentalPanel עם score columns (MultiIndex: date, symbol).
        snapshot_date  – תאריך snapshot אחרון ברירת מחדל.
        rank_columns   – dictionary של {column_name -> DataFrame של ranks/percentiles}.
        coverage_info  – מידע על איזה מדדים נכנסו/נפסלו (למשל בלי דאטה).
    """
    panel: FundamentalPanel
    snapshot_date: Optional[pd.Timestamp] = None
    rank_columns: Dict[str, pd.DataFrame] = field(default_factory=dict)
    coverage_info: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# 4) Hooks לרעיונות המתקדמים (ממשקים בלבד בשלב זה)
# ============================================================

@dataclass
class ScoreEngineConfig:
    """
    קונפיג כללי ל-"מנוע הציונים" ברמת Index:

    כולל:
    - תתי קונפיגים (Value/Quality/Growth/Composite/Risk/Regime/Factor-neutral).
    - הגדרות כלליות ל-Score Momentum / Stability וכו'.
    """
    value: ValueScoreConfig = field(default_factory=lambda: DEFAULT_VALUE_CONFIG)
    quality: QualityScoreConfig = field(default_factory=lambda: DEFAULT_QUALITY_CONFIG)
    growth: GrowthScoreConfig = field(default_factory=lambda: DEFAULT_GROWTH_CONFIG)
    composite: CompositeScoreConfig = field(default_factory=lambda: DEFAULT_COMPOSITE_CONFIG)
    risk_overlay: RiskOverlayConfig = field(default_factory=lambda: DEFAULT_RISK_OVERLAY_CONFIG)
    regime: RegimeScoreConfig = field(default_factory=lambda: DEFAULT_REGIME_CONFIG)
    factor_neutral: FactorNeutralConfig = field(default_factory=lambda: DEFAULT_FACTOR_NEUTRAL_CONFIG)

    # אנדרוגה: הגדרות ל-Score Momentum / Stability
    score_momentum_window_days: int = 180
    score_stability_window_days: int = 365


# החלק הבא (Part 2/4) יוסיף:
#   - פונקציות util לחישוב zscores/percentiles וכו'
#   - פונקציות פנימיות לחישוב Value/Quality/Growth/Composite
#   - פונקציות ל-Score Momentum ו-Score Stability
#   - עדיין בלי API public (זה יבוא ב-Part 3/4)

# ============================================================
# Part 2/4 — Utilities & Core Score Computation (HF-grade)
# ============================================================
"""
בחלק הזה אנחנו מממשים את **ליבת** מנוע הציונים:

1. Utilities:
   - המרת אינדקס לזמן (DatetimeIndex) ויישור סדרות.
   - חישובי Z-score, Percentiles, Rolling changes.
   - טיפול ב-Outliers.

2. חישוב ציונים בסיסיים:
   - Value subscores (פירוט לפי פיצ'רים).
   - Value score משוכלל.
   - Quality score.
   - Growth score.
   - Composite score.

3. הרחבות (20 רעיונות/פרמטרים חדשים – ברמת Utilities ו-Config שימושית):
   חלקם ממומשים כאן כחישובים, חלקם כפרמטרים/Flags/Hookים:
   ----------------------------------------------------------------
   1.  log_transform_numeric        – אפשרות להשתמש בלוגריתם על חלק מהפיצ'רים.
   2.  winsorize_zscore_threshold   – חיתוך קצוות קיצוניים לפני חישוב ציונים.
   3.  min_observations_per_field   – דרישה למינימום תצפיות כדי שמשתנה יחשב בציון.
   4.  min_history_days_for_scores  – דרישת היסטוריה מינימלית למדד.
   5.  allow_negative_value_metrics – האם לאפשר ערכים שליליים בשדות מסוימים.
   6.  score_momentum_window_days   – חלון ימים למומנטום ציונים.
   7.  score_stability_window_days  – חלון ימים לחישוב סטיות/יציבות.
   8.  score_smoothing_span         – EMA smoothing לציונים.
   9.  per_field_weights_override   – משקל שונה לפי שדה ספציפי (dictionary).
   10. weight_cap_per_field         – תקרה למשקל יחסי של שדה בודד בציון.
   11. missing_data_penalty         – ענישה למדדים עם הרבה נתוני NaN.
   12. robustness_mode              – מצב "Robust" עם שימוש ב-Median/MAD במקום mean/std.
   13. transform_to_rank_instead_of_pct – מעבר לדירוג (Rank) במקום אחוזון.
   14. use_cross_sectional_ranks    – אופציה להתחשב ברנק cross-sectional באותו תאריך.
   15. normalize_scores_globally    – נירמול ציונים על פני זמן/Universe.
   16. allow_negative_scores        – האם לאפשר ציון מתחת 0 (למשל אחרי penalties).
   17. floor_score_at_zero          – פרמטר בקרה על חיתוך תחתון ב-0.
   18. ceiling_score_at_100         – פרמטר בקרה על חיתוך עליון ב-100.
   19. debug_fields_track           – רשימת שדות שתרצה לעקוב אחריהם ב-log/df.
   20. per_symbol_overrides         – אפשרות להגדיר overrides פר מדד (future hook).

רוב הפרמטרים הללו ימומשו באמצעות:
- פונקציות utility שמכבדות אותם.
- שימוש ב-ScoreEngineConfig שנבנה ב-Part 1/4.
"""

# ============================================================
# 5) Core utilities – time index, numeric handling, outliers
# ============================================================

def _ensure_datetime_index(df: FundamentalFrame) -> FundamentalFrame:
    """
    מבטיח שאינדקס ה-DataFrame הוא DatetimeIndex אם יש עמודת date/דומה.

    אם כבר DatetimeIndex – מחזיר כמות שהוא (sorted).
    אם לא – ננסה לזהות עמודת תאריך ולהפוך אותה לאינדקס.
    אם נכשל – נשאיר כפי שהוא אך נרשום אזהרה.
    """
    if df is None or df.empty:
        return df

    if isinstance(df.index, pd.DatetimeIndex):
        df = df.sort_index()
        return df

    # ננסה לזהות עמודת תאריך
    date_candidates = ["date", "datetime", "period_end", "report_date", "as_of_date"]
    cols_lower = {c.lower(): c for c in df.columns}
    chosen = None
    for cand in date_candidates:
        if cand in cols_lower:
            chosen = cols_lower[cand]
            break

    if chosen is None:
        logger.warning(
            "Index fundamentals DF has no DatetimeIndex and no obvious date column; "
            "some time-based metrics may be unavailable."
        )
        return df

    df = df.copy()
    try:
        df[chosen] = pd.to_datetime(df[chosen])
        df = df.set_index(chosen)
        df = df.sort_index()
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to convert %r to DatetimeIndex: %s", chosen, exc)

    return df


def _select_numeric(df: FundamentalFrame) -> FundamentalFrame:
    """
    מחזיר תת-DataFrame של עמודות מספריות בלבד.
    """
    if df is None or df.empty:
        return df
    return df.select_dtypes(include=["number", "float", "int"])


def _apply_log_transform(
    df: FundamentalFrame,
    fields: Sequence[str],
) -> FundamentalFrame:
    """
    החלת לוגריתם על שדות מסוימים (log(1+x)) – בעיקר לערכים חיוביים.

    - אם ערך <= -1 → נשאיר NaN (לא הגיוני ללוג).
    - אם עמודה לא קיימת → נתעלם בשקט.
    """
    if df is None or df.empty:
        return df

    df = df.copy()
    for col in fields:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        # עבור ערכים > -1
        valid_mask = s > -1.0
        s_valid = s.where(valid_mask)
        df[col] = np.log1p(s_valid)
    return df


def _winsorize_by_zscore(
    s: pd.Series,
    threshold: float = 5.0,
) -> pd.Series:
    """
    Winsorization דרך Z-score:
    - מחשב Z-score.
    - ערכים עם |z| > threshold → ייחתכו לגבול הקרוב.
    """
    s = pd.to_numeric(s, errors="coerce")
    mask = s.notna()
    if mask.sum() < 3:
        return s

    x = s[mask]
    mean = x.mean()
    std = x.std(ddof=0) or 1e-9
    z = (x - mean) / std

    # חיתוך
    z_clipped = z.clip(-threshold, threshold)
    x_clipped = mean + z_clipped * std

    out = s.copy()
    out.loc[mask] = x_clipped
    return out


def _apply_winsorization_to_df(
    df: FundamentalFrame,
    columns: Sequence[str],
    threshold: float = 5.0,
) -> FundamentalFrame:
    """
    מפעיל winsorization לפי Z-score על רשימת עמודות.
    """
    if df is None or df.empty:
        return df

    df = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        df[col] = _winsorize_by_zscore(df[col], threshold=threshold)
    return df


# ============================================================
# 6) Scoring utilities – z-scores, percentiles, ranks, momentum
# ============================================================

def _safe_series(df: FundamentalFrame, col: str) -> pd.Series:
    """
    מחזיר סדרה מספרית עבור עמודה נתונה, עם טיפול ב-errors כ-NaN.
    """
    if col not in df.columns:
        return pd.Series(dtype=float)
    s = pd.to_numeric(df[col], errors="coerce")
    return s


def _zscore_series(
    s: pd.Series,
    *,
    robust: bool = False,
) -> pd.Series:
    """
    מחזיר Z-score לכל נקודה בסדרה (x - mean) / std.

    - אם robust=True → נשתמש ב-Median + MAD כבסיס.
    - אם std או MAD=0 → נחזיר 0.
    """
    s = pd.to_numeric(s, errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan)
    s = s.dropna()
    if s.empty:
        return pd.Series(dtype=float)

    if robust:
        median = s.median()
        mad = (s - median).abs().median()
        if mad == 0 or np.isnan(mad):
            return pd.Series(np.zeros(len(s)), index=s.index)
        return (s - median) / (1.4826 * mad)  # approx to std
    else:
        mean = s.mean()
        std = s.std(ddof=0)
        if std == 0 or np.isnan(std):
            return pd.Series(np.zeros(len(s)), index=s.index)
        return (s - mean) / std


def _percentile_rank(
    s: pd.Series,
    *,
    robust: bool = False,
) -> pd.Series:
    """
    מחשב percentile rank לכל ערך [0, 100].

    robust:
        - בגרסה הנוכחית אין שינוי לוגי, אבל ניתן להרחיב (למשל ע"י שימוש
          בתצפיות trimmed).
    """
    s = pd.to_numeric(s, errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan)
    ranks = s.rank(pct=True) * 100.0
    return ranks


def _invert_for_value(score: pd.Series) -> pd.Series:
    """
    הופך score (0–100) כך ש-0→100 ו-100→0.
    מיועד למקרים בהם "נמוך יותר טוב" (למשל P/E).
    """
    return 100.0 - score


def _normalize_score(
    s: pd.Series,
    *,
    floor_zero: bool = True,
    ceiling_100: bool = True,
    allow_negative: bool = False,
) -> pd.Series:
    """
    מבטיח ש-score נמצא בטווח הגיוני:

    - אם allow_negative=False → נחתוך תחתון ב-0.
    - אם ceiling_100=True → נחתוך עליון ב-100.
    """
    s = pd.to_numeric(s, errors="coerce")
    if not allow_negative and floor_zero:
        s = s.clip(lower=0.0)
    if ceiling_100:
        s = s.clip(upper=100.0)
    return s


def _compute_series_momentum(
    s: pd.Series,
    window_days: int,
) -> pd.Series:
    """
    מחשב מומנטום של סדרה (diff) על בסיס חלון זמן בימים.

    הגישה:
    -------
    - מניח שהאינדקס הוא DatetimeIndex.
    - עבור כל תאריך t, נמצא את הערך ב-(t - window_days) אם קיים.
    - momentum(t) ≈ s(t) - s(t - window_days).

    הערה:
    ------
    אם אין תצפית בדיוק ב-t-window_days, ניקח את הקרובה ביותר לפני.
    """
    if s.empty or not isinstance(s.index, pd.DatetimeIndex):
        return pd.Series(dtype=float)

    s = s.sort_index()
    result = pd.Series(index=s.index, dtype=float)

    for ts in s.index:
        cutoff = ts - pd.Timedelta(days=window_days)
        # תצפיות לפני cutoff
        past_vals = s[s.index <= cutoff]
        if past_vals.empty:
            result.loc[ts] = np.nan
        else:
            past_val = past_vals.iloc[-1]
            result.loc[ts] = float(s.loc[ts] - past_val)

    return result


def _compute_series_stability(
    s: pd.Series,
    window_days: int,
) -> pd.Series:
    """
    יציבות score – Rolling std על חלון זמן בימים.

    - מניח אינדקס זמן.
    - על כל חלון [t-window_days, t], מחשב std.
    """
    if s.empty or not isinstance(s.index, pd.DatetimeIndex):
        return pd.Series(dtype=float)

    s = s.sort_index()
    result = pd.Series(index=s.index, dtype=float)

    for ts in s.index:
        cutoff = ts - pd.Timedelta(days=window_days)
        window_vals = s[(s.index >= cutoff) & (s.index <= ts)].dropna()
        if window_vals.shape[0] < 2:
            result.loc[ts] = np.nan
        else:
            result.loc[ts] = float(window_vals.std(ddof=0))

    return result


def _apply_score_smoothing(
    s: pd.Series,
    *,
    span: Optional[int] = None,
) -> pd.Series:
    """
    מחליק score בסגנון EMA (Exponential Moving Average) אם span>0.
    """
    if span is None or span <= 0 or s.empty:
        return s
    s = s.sort_index()
    return s.ewm(span=span, adjust=False).mean()


# ============================================================
# 7) Value / Quality / Growth subscores & aggregation
# ============================================================

def _compute_value_subscores(
    df: FundamentalFrame,
    *,
    cfg: ValueScoreConfig,
    log_transform: bool = False,
    winsorize_threshold: Optional[float] = None,
    robust: bool = False,
) -> Dict[str, pd.Series]:
    """
    מחזיר dict של תתי-סקורים לפיצ'רי Value שונים (בסקאלה 0–100).

    פרמטרים:
    ---------
    log_transform :
        אם True → נבצע log(1+x) על שדות כמו pe, pb וכו' לפני חישוב ranks.
    winsorize_threshold :
        אם לא None → נחתוך קצוות על בסיס Z-score לפני חישוב percentile_rank.
    robust :
        אם True → אפשר להשתמש במוד robust בחישובי Z (בעתיד).

    פלט:
    ----
    dict:
        key: שם תת-סקור (למשל "value_pe")
        value: pd.Series של scores 0–100 לאורך הזמן.
    """
    if df is None or df.empty:
        return {}

    df = _ensure_datetime_index(df)

    # נייצר DataFrame עבודה לשדות ה-Value
    value_fields = [f for f in VALUE_FIELDS_DEFAULT if f in df.columns]
    work_df = df.copy()

    # 1) log transform (אם ביקשת)
    if log_transform:
        work_df = _apply_log_transform(work_df, value_fields)

    # 2) winsorization (אם threshold סופק)
    if winsorize_threshold is not None and winsorize_threshold > 0:
        work_df = _apply_winsorization_to_df(
            work_df,
            columns=value_fields,
            threshold=winsorize_threshold,
        )

    subs: Dict[str, pd.Series] = {}

    # lower is better
    lower_better = [
        ("pe", "value_pe", cfg.weight_pe),
        ("pe_forward", "value_pe_forward", cfg.weight_pe_forward),
        ("pb", "value_pb", cfg.weight_pb),
        ("price_to_book", "value_price_to_book", cfg.weight_price_to_book),
        ("ev_ebitda", "value_ev_ebitda", cfg.weight_ev_ebitda),
        ("price_sales", "value_price_sales", cfg.weight_price_sales),
    ]

    for col, key, w in lower_better:
        if w <= 0 or col not in work_df.columns:
            continue
        s = _safe_series(work_df, col)
        if s.dropna().shape[0] < 3:
            continue
        r = _percentile_rank(s, robust=robust)
        subs[key] = _normalize_score(_invert_for_value(r))

    # higher is better
    higher_better = [
        ("dividend_yield", "value_dividend_yield", cfg.weight_dividend_yield),
        ("earnings_yield", "value_earnings_yield", cfg.weight_earnings_yield),
        ("fcf_yield", "value_fcf_yield", cfg.weight_fcf_yield),
    ]

    for col, key, w in higher_better:
        if w <= 0 or col not in work_df.columns:
            continue
        s = _safe_series(work_df, col)
        if s.dropna().shape[0] < 3:
            continue
        r = _percentile_rank(s, robust=robust)
        subs[key] = _normalize_score(r)

    return subs


def _aggregate_value_score(
    subscores: Dict[str, pd.Series],
    cfg: ValueScoreConfig,
    *,
    allow_negative: bool = False,
) -> pd.Series:
    """
    מאגד תתי-סקורים לציון Value אחד.

    value_score(t) = sum(weight_i * subs_i(t)) / total_weight_used

    - אם עבור מדד מסוים אין אף תת-סקור – נחזיר סדרה ריקה.
    """
    if not subscores:
        return pd.Series(dtype=float)

    total_used_weight = 0.0
    agg: Optional[pd.Series] = None

    def _add(key: str, weight: float, current: Optional[pd.Series]) -> Tuple[Optional[pd.Series], float]:
        if weight <= 0 or key not in subscores:
            return current, 0.0
        s = subscores[key]
        if current is None:
            current = weight * s
        else:
            current = current.add(weight * s, fill_value=0.0)
        return current, weight

    for key, w in [
        ("value_pe", cfg.weight_pe),
        ("value_pe_forward", cfg.weight_pe_forward),
        ("value_pb", cfg.weight_pb),
        ("value_price_to_book", cfg.weight_price_to_book),
        ("value_dividend_yield", cfg.weight_dividend_yield),
        ("value_earnings_yield", cfg.weight_earnings_yield),
        ("value_fcf_yield", cfg.weight_fcf_yield),
        ("value_ev_ebitda", cfg.weight_ev_ebitda),
        ("value_price_sales", cfg.weight_price_sales),
    ]:
        agg, used = _add(key, w, agg)
        total_used_weight += used

    if agg is None or total_used_weight <= 0:
        return pd.Series(dtype=float)

    score = agg / total_used_weight
    return _normalize_score(score, allow_negative=allow_negative)


def _compute_quality_score(
    df: FundamentalFrame,
    cfg: QualityScoreConfig,
    *,
    winsorize_threshold: Optional[float] = None,
    robust: bool = False,
) -> pd.Series:
    """
    איכות (Quality) – מיזוג של ROE/ROA/ROIC, margins, interest coverage, leverage.
    """
    if df is None or df.empty:
        return pd.Series(dtype=float)

    df = _ensure_datetime_index(df)
    fields = [f for f in QUALITY_FIELDS_DEFAULT if f in df.columns]
    work_df = df.copy()

    # winsorization אם צריך
    if winsorize_threshold is not None and winsorize_threshold > 0:
        work_df = _apply_winsorization_to_df(work_df, fields, threshold=winsorize_threshold)

    agg: Optional[pd.Series] = None
    total_used_weight = 0.0

    def _add_positive(col: str, weight: float, current: Optional[pd.Series]) -> Tuple[Optional[pd.Series], float]:
        if weight <= 0 or col not in work_df.columns:
            return current, 0.0
        s = _safe_series(work_df, col)
        if s.dropna().shape[0] < 3:
            return current, 0.0
        r = _percentile_rank(s, robust=robust)
        if current is None:
            current = weight * r
        else:
            current = current.add(weight * r, fill_value=0.0)
        return current, weight

    def _add_negative(col: str, weight: float, current: Optional[pd.Series]) -> Tuple[Optional[pd.Series], float]:
        if weight <= 0 or col not in work_df.columns:
            return current, 0.0
        s = _safe_series(work_df, col)
        if s.dropna().shape[0] < 3:
            return current, 0.0
        r = _percentile_rank(s, robust=robust)
        inv = _invert_for_value(r)
        if current is None:
            current = weight * inv
        else:
            current = current.add(weight * inv, fill_value=0.0)
        return current, weight

    # higher is better
    for col, w in [
        ("roe", cfg.weight_roe),
        ("roa", cfg.weight_roa),
        ("roic", cfg.weight_roic),
        ("gross_margin", cfg.weight_gross_margin),
        ("operating_margin", cfg.weight_operating_margin),
        ("net_margin", cfg.weight_net_margin),
        ("interest_coverage", cfg.weight_interest_coverage),
    ]:
        agg, used = _add_positive(col, w, agg)
        total_used_weight += used

    # lower is better: net_debt_to_ebitda
    agg, used = _add_negative("net_debt_to_ebitda", cfg.weight_net_debt_to_ebitda, agg)
    total_used_weight += used

    if agg is None or total_used_weight <= 0:
        return pd.Series(dtype=float)

    score = agg / total_used_weight
    return _normalize_score(score)


def _compute_growth_score(
    df: FundamentalFrame,
    cfg: GrowthScoreConfig,
    *,
    winsorize_threshold: Optional[float] = None,
    robust: bool = False,
) -> pd.Series:
    """
    ציון Growth – מבוסס על קצבי צמיחה (EPS, Revenue, FCF).
    """
    if df is None or df.empty:
        return pd.Series(dtype=float)

    df = _ensure_datetime_index(df)
    fields = [f for f in GROWTH_FIELDS_DEFAULT if f in df.columns]
    work_df = df.copy()

    if winsorize_threshold is not None and winsorize_threshold > 0:
        work_df = _apply_winsorization_to_df(work_df, fields, threshold=winsorize_threshold)

    agg: Optional[pd.Series] = None
    total_used_weight = 0.0

    def _add_positive(col: str, weight: float, current: Optional[pd.Series]) -> Tuple[Optional[pd.Series], float]:
        if weight <= 0 or col not in work_df.columns:
            return current, 0.0
        s = _safe_series(work_df, col)
        if s.dropna().shape[0] < 3:
            return current, 0.0
        r = _percentile_rank(s, robust=robust)
        if current is None:
            current = weight * r
        else:
            current = current.add(weight * r, fill_value=0.0)
        return current, weight

    for col, w in [
        ("eps_growth_3y", cfg.weight_eps_growth_3y),
        ("eps_growth_5y", cfg.weight_eps_growth_5y),
        ("revenue_growth_3y", cfg.weight_revenue_growth_3y),
        ("revenue_growth_5y", cfg.weight_revenue_growth_5y),
        ("fcf_growth_3y", cfg.weight_fcf_growth_3y),
        ("fcf_growth_5y", cfg.weight_fcf_growth_5y),
    ]:
        agg, used = _add_positive(col, w, agg)
        total_used_weight += used

    if agg is None or total_used_weight <= 0:
        return pd.Series(dtype=float)

    score = agg / total_used_weight
    return _normalize_score(score)


def _compute_composite_score(
    value_score: pd.Series,
    quality_score: pd.Series,
    growth_score: pd.Series,
    cfg: CompositeScoreConfig,
    *,
    smoothing_span: Optional[int] = None,
) -> pd.Series:
    """
    ציון משולב – משקל יחסי ל-Value / Quality / Growth.

        composite = (wV * V + wQ * Q + wG * G) / (wV + wQ + wG)

    ניתן להחיל smoothing (EMA) על התוצאה.
    """
    total_w = cfg.total_weight()
    if total_w <= 0:
        return pd.Series(dtype=float)

    # ניישר אינדקסים
    idx = value_score.index.union(quality_score.index).union(growth_score.index)
    v = value_score.reindex(idx)
    q = quality_score.reindex(idx)
    g = growth_score.reindex(idx)

    agg = (
        cfg.weight_value * v.fillna(v.mean())
        + cfg.weight_quality * q.fillna(q.mean())
        + cfg.weight_growth * g.fillna(g.mean())
    )
    composite = agg / total_w
    composite = _normalize_score(composite)

    if smoothing_span:
        composite = _apply_score_smoothing(composite, span=smoothing_span)
        composite = _normalize_score(composite)  # אחרי החלקה

    return composite


# ============================================================
# 8) High-level helper for computing full score set on a DF
# ============================================================

def _compute_scores_for_df(
    df: FundamentalFrame,
    symbol: str,
    engine_cfg: ScoreEngineConfig,
) -> FundamentalFrame:
    """
    פונקציה פנימית שמקבלת DataFrame fundamentals למדד אחד,
    ומחזירה DataFrame עם עמודות ציונים:

        value_score, quality_score, growth_score, composite_score,
        value_* subscores, optional momentum/stability columns.

    שלבים:
    -------
    1. Normalization לתאריכים.
    2. חישוב Value subscores + value_score.
    3. חישוב quality_score + growth_score.
    4. Composite score.
    5. Score momentum + stability (לפי המוגדר ב-ScoreEngineConfig).
    """
    if df is None or df.empty:
        logger.warning("_compute_scores_for_df: empty DF for %s", symbol)
        return df

    df = _ensure_datetime_index(df).copy()

    # ----- Value -----
    value_subs = _compute_value_subscores(
        df,
        cfg=engine_cfg.value,
        log_transform=True,  # נשתמש בלוג עבור P/E וכד'
        winsorize_threshold=5.0,
        robust=False,
    )
    for k, s in value_subs.items():
        df[k] = s

    value_score = _aggregate_value_score(
        value_subs,
        engine_cfg.value,
        allow_negative=False,
    )

    # ----- Quality -----
    quality_score = _compute_quality_score(
        df,
        engine_cfg.quality,
        winsorize_threshold=5.0,
        robust=False,
    )

    # ----- Growth -----
    growth_score = _compute_growth_score(
        df,
        engine_cfg.growth,
        winsorize_threshold=5.0,
        robust=False,
    )

    # ----- Composite -----
    composite_score = _compute_composite_score(
        value_score,
        quality_score,
        growth_score,
        engine_cfg.composite,
        smoothing_span=engine_cfg.score_momentum_window_days // 30
        if engine_cfg.score_momentum_window_days > 30
        else None,
    )

    df["value_score"] = value_score
    df["quality_score"] = quality_score
    df["growth_score"] = growth_score
    df["composite_score"] = composite_score

    # ----- Momentum & Stability -----
    if engine_cfg.score_momentum_window_days > 0:
        df["value_score_momentum"] = _compute_series_momentum(
            df["value_score"],
            window_days=engine_cfg.score_momentum_window_days,
        )
        df["quality_score_momentum"] = _compute_series_momentum(
            df["quality_score"],
            window_days=engine_cfg.score_momentum_window_days,
        )
        df["growth_score_momentum"] = _compute_series_momentum(
            df["growth_score"],
            window_days=engine_cfg.score_momentum_window_days,
        )
        df["composite_score_momentum"] = _compute_series_momentum(
            df["composite_score"],
            window_days=engine_cfg.score_momentum_window_days,
        )

    if engine_cfg.score_stability_window_days > 0:
        df["value_score_stability"] = _compute_series_stability(
            df["value_score"],
            window_days=engine_cfg.score_stability_window_days,
        )
        df["quality_score_stability"] = _compute_series_stability(
            df["quality_score"],
            window_days=engine_cfg.score_stability_window_days,
        )
        df["growth_score_stability"] = _compute_series_stability(
            df["growth_score"],
            window_days=engine_cfg.score_stability_window_days,
        )
        df["composite_score_stability"] = _compute_series_stability(
            df["composite_score"],
            window_days=engine_cfg.score_stability_window_days,
        )

    return df


# Part 2/4 מסתיים כאן.
# בחלק 3/4 נוסיף:
#   - score_index_fundamentals(...) – API פומבי למדד בודד
#   - score_universe_fundamentals(...) – ל-Universe שלם
#   - פונקציות rank/percentile cross-sectional
#   - Multi-horizon percentiles ותמיכה מלאה ל-20 הרעיונות המתקדמים ברמת Universe.
# ============================================================
# Part 3/4 — Public Scoring API & Cross-Sectional Analytics
# ============================================================
"""
בחלק הזה אנחנו בונים את ה־API הפומבי ואת כל שכבת ה־Universe / Cross-Section:

מה נכנס כאן בפועל
------------------
1. score_index_fundamentals(...)
   - טוען fundamentals למדד בודד דרך common.fundamental_loader.
   - מחשב Value / Quality / Growth / Composite + Momentum + Stability.
   - מחזיר IndexScoreSeries (כולל DataFrame מלא עם העמודות החדשות).

2. score_index_fundamentals_df(...)
   - עטיפה נוחה שמחזירה רק DataFrame (לשימוש מהיר).

3. score_universe_fundamentals(...)
   - מחיל את מנוע הציונים על Universe שלם (רשימת מדדים).
   - בונה FundamentalPanel (MultiIndex: (date, symbol)).
   - מחזיר UniverseScoresSummary הכולל:
       * panel – הדאטה המלא.
       * snapshot_date – תאריך snapshot אחרון.
       * rank_columns – טבלאות דירוג cross-sectional.
       * coverage_info – מי הצליח/נכשל, כמה שורות, וכו'.

4. compute_cross_sectional_ranks(...)
   - מחשב rank (1 = הטוב ביותר) cross-sectional לכל עמודה נבחרת בתאריך נתון.

5. compute_cross_sectional_percentiles(...)
   - אחוזונים cross-sectional [0,100] לכל עמודה בתאריך נתון.

6. get_universe_score_snapshot(...)
   - מחזיר DataFrame של snapshot (תאריך אחד) עם scores + ranks + percentiles.

7. select_top_n_by_score(...) / select_bottom_n_by_score(...)
   - בחירת טופ/בוטום N מדדים לפי score מסוים בתאריך נתון.

8. bucket_scores_into_quantiles(...)
   - מחלק מדדים ל־k quantiles (למשל quintiles) לפי ציון (לניתוח Bucket-based).

9. summarize_universe_momentum(...)
   - מסכם מומנטום ממוצע / חציוני ב־Universe.

10. summarize_universe_stability(...)
    - מסכם יציבות (std של score) ברמת Universe.

בנוסף, לפי הבקשה שלך, אנחנו מוסיפים פה **עוד 20 רעיונות/פרמטרים ברמת Universe** –
חלקם ממומשים, חלקם נחשפים כפרמטרים/Hookים ל־Part 4:

20 רעיונות/פרמטרים חדשים (Universe-level)
------------------------------------------
1.  universe_min_rows_threshold     – מינימום שורות כדי שסימול ייכנס ל-Panel.
2.  universe_min_non_nan_ratio      – דרישת יחס לא-NaN לכל score כדי להיכנס לציון.
3.  snapshot_date_policy            – איך לבחור snapshot:
                                       "latest", "common", "custom:<date>".
4.  columns_for_ranking             – אילו עמודות scores להכניס לדירוג cross-sectional.
5.  rank_ascending_flags            – dict {column -> bool} האם עמודה דורשת ascending.
6.  top_n_default                   – N ברירת מחדל לטופ/בוטום.
7.  quantiles_default               – מספר quantiles ברירת מחדל (למשל 5).
8.  drop_symbols_without_scores     – האם להשליך מדדים בלי ציון composite.
9.  include_momentum_in_snapshot    – האם להציג גם momentum ב-snapshot.
10. include_stability_in_snapshot   – האם להציג גם stability ב-snapshot.
11. universe_debug_log              – debug verbosity: 0/1/2.
12. allow_partial_universe_scores   – האם להתעלם ממדדים שהחישוב להם כשל.
13. per_universe_score_normalization – נירמול ציונים ברמת Universe (zscore global).
14. score_clip_low/high             – גבולות clip ברמת Universe.
15. snapshot_join_fields            – אפשרות לצרף עוד שדות snapshot (כגון market_cap).
16. top_bucket_label, bottom_bucket_label – שמות bucket מותאמים ל-UI.
17. cross_sectional_rank_method     – שיטת rank (min, dense, average).
18. percentile_decimals             – מספר ספרות אחרי הנקודה באחוזונים.
19. allow_empty_universe_return     – האם להחזיר DataFrame ריק אם הכל נכשל, או לזרוק שגיאה.
20. universe_profile_name           – שם פרופיל (לוגי) ל-Universe (למשל "global_equities").

חלק מהפרמטרים הללו מיושמים ישירות בפונקציות, וחלקם התשתית ל-Part 4 (Regime-aware,
Risk-adjusted וכו').
"""

# ============================================================
# 9) Public API — single index
# ============================================================

def score_index_fundamentals(
    symbol: str,
    *,
    start: date | None = None,
    end: date | None = None,
    base_fields: Sequence[str] | None = None,
    engine_cfg: ScoreEngineConfig | None = None,
    allow_remote: bool = True,
) -> IndexScoreSeries:
    """
    מחשב את כל הציונים הפנדומנטליים למדד בודד:

    מחזיר:
        IndexScoreSeries:
            .symbol  – שם המדד (Upper).
            .df      – DataFrame עם fundamentals + scores + momentum + stability.
    """
    if engine_cfg is None:
        engine_cfg = ScoreEngineConfig()

    fields_needed = set(ALL_FUNDAMENTAL_FIELDS_DEFAULT)
    if base_fields:
        fields_needed.update(base_fields)

    df_raw = load_index_fundamentals(
        symbol,
        start=start,
        end=end,
        fields=sorted(fields_needed),
        allow_remote=allow_remote,
        force_refresh=False,
        require_fresh_local=False,
        allow_partial=True,
    )
    if df_raw is None or df_raw.empty:
        logger.warning("score_index_fundamentals: empty fundamentals for %s", symbol)
        return IndexScoreSeries(symbol=symbol.upper(), df=pd.DataFrame())

    df_scored = _compute_scores_for_df(df_raw, symbol, engine_cfg)
    return IndexScoreSeries(
        symbol=symbol.upper(),
        df=df_scored,
        value_config=engine_cfg.value,
        quality_config=engine_cfg.quality,
        growth_config=engine_cfg.growth,
        composite_config=engine_cfg.composite,
    )


def score_index_fundamentals_df(
    symbol: str,
    *,
    start: date | None = None,
    end: date | None = None,
    base_fields: Sequence[str] | None = None,
    engine_cfg: ScoreEngineConfig | None = None,
    allow_remote: bool = True,
) -> FundamentalFrame:
    """
    עטיפת נוחות שמחזירה רק DataFrame עם הציונים.
    """
    series = score_index_fundamentals(
        symbol,
        start=start,
        end=end,
        base_fields=base_fields,
        engine_cfg=engine_cfg,
        allow_remote=allow_remote,
    )
    return series.df


# ============================================================
# 10) Public API — universe scoring
# ============================================================

def score_universe_fundamentals(
    symbols: Sequence[str],
    *,
    start: date | None = None,
    end: date | None = None,
    engine_cfg: ScoreEngineConfig | None = None,
    allow_remote: bool = True,
    universe_min_rows_threshold: int = 4,
    universe_min_non_nan_ratio: float = 0.5,
    snapshot_date_policy: str = "latest",  # "latest" | "common" | "custom:YYYY-MM-DD"
    columns_for_ranking: Sequence[str] | None = None,
    rank_ascending_flags: Mapping[str, bool] | None = None,
    drop_symbols_without_scores: bool = True,
    universe_profile_name: str | None = None,
    universe_debug_log: int = 0,
) -> UniverseScoresSummary:
    """
    מחיל את מנוע הציונים על Universe של מדדים ומחזיר UniverseScoresSummary.

    שלבים:
    -------
    1. מחשב scores לכל symbol (ביחד עם fundamentals).
    2. מאחד ל-Panel MultiIndex (date, symbol).
    3. בוחר snapshot_date לפי policy.
    4. מחשב cross-sectional ranks/percentiles עבור columns_for_ranking.
    5. בונה UniverseScoresSummary.

    פרמטרים חשובים:
    ----------------
    universe_min_rows_threshold :
        מינימום שורות (תצפיות בזמן) כדי שמדד ייחשב כשמיש.
    universe_min_non_nan_ratio :
        שיעור מינימום של ערכים לא-NaN ב-scoreים כדי להיחשב.
    snapshot_date_policy :
        "latest" – התאריך המאוחר ביותר שקיים ב-Panel.
        "common" – התאריך המאוחר ביותר שקיים לכל המדדים (intersect).
        "custom:YYYY-MM-DD" – תאריך ספציפי אם קיים.
    columns_for_ranking :
        רשימת עמודות scores שיש למיין/לקבל percentile עבורן.
        אם None → נשתמש בברירת המחדל: value_score, quality_score, growth_score, composite_score.
    rank_ascending_flags :
        dict {column -> bool} האם עמודה צריכה rank ascending (True) או descending (False).
        ברירת מחדל: descending (כלומר score גבוה עדיף).
    """
    if engine_cfg is None:
        engine_cfg = ScoreEngineConfig()

    symbols_list = list(symbols)
    if not symbols_list:
        return UniverseScoresSummary(panel=pd.DataFrame(), snapshot_date=None)

    coverage_info: Dict[str, Any] = {}
    pieces: List[pd.DataFrame] = []

    for sym in symbols_list:
        sym_norm = sym.upper()
        try:
            series = score_index_fundamentals(
                sym,
                start=start,
                end=end,
                engine_cfg=engine_cfg,
                allow_remote=allow_remote,
            )
            df_scored = series.df
            if df_scored is None or df_scored.empty:
                coverage_info[sym_norm] = {"status": "empty"}
                continue

            df_scored = _ensure_datetime_index(df_scored)
            if df_scored.shape[0] < universe_min_rows_threshold:
                coverage_info[sym_norm] = {
                    "status": "too_few_rows",
                    "rows": df_scored.shape[0],
                }
                continue

            # בדיקת יחס non-NaN ב-composite_score
            if "composite_score" in df_scored.columns:
                non_nan_ratio = df_scored["composite_score"].notna().mean()
                if non_nan_ratio < universe_min_non_nan_ratio:
                    coverage_info[sym_norm] = {
                        "status": "too_many_nans",
                        "rows": df_scored.shape[0],
                        "non_nan_ratio": float(non_nan_ratio),
                    }
                    continue
            elif drop_symbols_without_scores:
                coverage_info[sym_norm] = {"status": "no_composite_score"}
                continue

            df_scored = df_scored.copy()
            df_scored["symbol"] = sym_norm
            pieces.append(df_scored)
            coverage_info[sym_norm] = {
                "status": "ok",
                "rows": df_scored.shape[0],
                "cols": df_scored.shape[1],
            }
        except Exception as exc:
            coverage_info[sym_norm] = {"status": "error", "error": str(exc)}
            if universe_debug_log > 0:
                logger.warning("Failed scoring fundamentals for %s: %s", sym_norm, exc)
            continue

    if not pieces:
        if not symbols_list:
            logger.warning("score_universe_fundamentals: empty universe.")
        elif universe_debug_log > 0:
            logger.warning(
                "score_universe_fundamentals: all symbols failed filters; "
                "coverage_info=%s",
                coverage_info,
            )
        return UniverseScoresSummary(panel=pd.DataFrame(), snapshot_date=None, coverage_info=coverage_info)

    panel = pd.concat(pieces)
    panel = panel.reset_index().rename(columns={"index": "date"})
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.set_index(["date", "symbol"]).sort_index()

    # -----------------------------
    # בחירת snapshot_date
    # -----------------------------
    all_dates = panel.index.get_level_values("date").unique().sort_values()
    if snapshot_date_policy.startswith("custom:"):
        date_str = snapshot_date_policy.split(":", 1)[1]
        try:
            snap_date = pd.to_datetime(date_str)
        except Exception:
            snap_date = all_dates.max()
    elif snapshot_date_policy == "common":
        # התאריך המאוחר ביותר המשותף לכולם
        per_symbol_dates = {}
        for sym in panel.index.get_level_values("symbol").unique():
            dates_sym = panel.xs(sym, level="symbol").index.unique()
            per_symbol_dates[sym] = set(dates_sym)
        common_dates = set(all_dates)
        for s in per_symbol_dates.values():
            common_dates &= s
        if common_dates:
            snap_date = max(common_dates)
        else:
            snap_date = all_dates.max()
    else:  # "latest" או כל ערך אחר
        snap_date = all_dates.max()

    snapshot_date = snap_date

    # -----------------------------
    # cross-sectional ranks/percentiles
    # -----------------------------
    if columns_for_ranking is None:
        columns_for_ranking = [
            "value_score",
            "quality_score",
            "growth_score",
            "composite_score",
        ]

    if rank_ascending_flags is None:
        # ברירת מחדל: descending (higher is better)
        rank_ascending_flags = {col: False for col in columns_for_ranking}

    rank_columns: Dict[str, pd.DataFrame] = {}

    try:
        snapshot_df = panel.xs(snapshot_date, level="date")
    except KeyError:
        # אם אין snapshot בדיוק בתאריך הנבחר, ניקח את התאריך הזמין הקרוב ביותר (אחרון לפני)
        subset = panel[panel.index.get_level_values("date") <= snapshot_date]
        if subset.empty:
            snapshot_df = pd.DataFrame()
        else:
            last_date = subset.index.get_level_values("date").max()
            snapshot_df = panel.xs(last_date, level="date")
            snapshot_date = last_date

    if not snapshot_df.empty:
        for col in columns_for_ranking:
            if col not in snapshot_df.columns:
                continue
            asc = bool(rank_ascending_flags.get(col, False))
            vals = pd.to_numeric(snapshot_df[col], errors="coerce")
            # Rank: 1 = הטוב ביותר (לפי ascending/descending)
            r = vals.rank(ascending=asc, method="min")
            pct = vals.rank(ascending=asc, pct=True) * 100.0
            df_rank = pd.DataFrame(
                {
                    "score": vals,
                    "rank": r,
                    "percentile": pct,
                }
            )
            df_rank.index.name = "symbol"
            rank_columns[col] = df_rank.sort_values("rank")

    return UniverseScoresSummary(
        panel=panel,
        snapshot_date=snapshot_date,
        rank_columns=rank_columns,
        coverage_info=coverage_info,
    )


# ============================================================
# 11) Cross-sectional helpers
# ============================================================

def compute_cross_sectional_ranks(
    panel: FundamentalPanel,
    *,
    date_point: pd.Timestamp | None = None,
    columns: Sequence[str] | None = None,
    ascending_flags: Mapping[str, bool] | None = None,
    method: str = "min",
) -> Dict[str, pd.DataFrame]:
    """
    מחשב ranks cross-sectional עבור עמודות נבחרות בתאריך אחד.

    החזר:
        dict[column -> DataFrame(index=symbol, columns=[score, rank])]
    """
    if panel is None or panel.empty:
        return {}

    all_dates = panel.index.get_level_values("date").unique().sort_values()
    if date_point is None:
        date_point = all_dates.max()

    try:
        snapshot_df = panel.xs(date_point, level="date")
    except KeyError:
        subset = panel[panel.index.get_level_values("date") <= date_point]
        if subset.empty:
            return {}
        last_date = subset.index.get_level_values("date").max()
        snapshot_df = panel.xs(last_date, level="date")

    if columns is None:
        columns = [
            c
            for c in ["value_score", "quality_score", "growth_score", "composite_score"]
            if c in snapshot_df.columns
        ]

    if ascending_flags is None:
        ascending_flags = {c: False for c in columns}

    results: Dict[str, pd.DataFrame] = {}
    for col in columns:
        if col not in snapshot_df.columns:
            continue
        asc = bool(ascending_flags.get(col, False))
        vals = pd.to_numeric(snapshot_df[col], errors="coerce")
        r = vals.rank(ascending=asc, method=method)
        df_rank = pd.DataFrame(
            {"score": vals, "rank": r},
            index=vals.index,
        )
        df_rank.index.name = "symbol"
        results[col] = df_rank.sort_values("rank")

    return results


def compute_cross_sectional_percentiles(
    panel: FundamentalPanel,
    *,
    date_point: pd.Timestamp | None = None,
    columns: Sequence[str] | None = None,
    ascending_flags: Mapping[str, bool] | None = None,
    decimals: int = 2,
) -> Dict[str, pd.DataFrame]:
    """
    מחשב percentiles cross-sectional עבור עמודות נבחרות בתאריך אחד.

    החזר:
        dict[column -> DataFrame(index=symbol, columns=[score, percentile])]
    """
    if panel is None or panel.empty:
        return {}

    all_dates = panel.index.get_level_values("date").unique().sort_values()
    if date_point is None:
        date_point = all_dates.max()

    try:
        snapshot_df = panel.xs(date_point, level="date")
    except KeyError:
        subset = panel[panel.index.get_level_values("date") <= date_point]
        if subset.empty:
            return {}
        last_date = subset.index.get_level_values("date").max()
        snapshot_df = panel.xs(last_date, level="date")

    if columns is None:
        columns = [
            c
            for c in ["value_score", "quality_score", "growth_score", "composite_score"]
            if c in snapshot_df.columns
        ]

    if ascending_flags is None:
        ascending_flags = {c: False for c in columns}

    results: Dict[str, pd.DataFrame] = {}
    for col in columns:
        if col not in snapshot_df.columns:
            continue
        asc = bool(ascending_flags.get(col, False))
        vals = pd.to_numeric(snapshot_df[col], errors="coerce")
        pct = vals.rank(ascending=asc, pct=True) * 100.0
        pct = pct.round(decimals)
        df_pct = pd.DataFrame(
            {"score": vals, "percentile": pct},
            index=vals.index,
        )
        df_pct.index.name = "symbol"
        results[col] = df_pct.sort_values("percentile", ascending=False)

    return results


# ============================================================
# 12) Snapshot + Top/Bottom selection + Buckets
# ============================================================

def get_universe_score_snapshot(
    universe_summary: UniverseScoresSummary,
    *,
    include_momentum: bool = True,
    include_stability: bool = True,
    snapshot_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    מחזיר DataFrame של snapshot (תאריך אחד) עם כל scoreים, כולל אופציה למומנטום/סטביליות.
    """
    panel = universe_summary.panel
    if panel is None or panel.empty:
        return pd.DataFrame()

    if snapshot_date is None:
        snapshot_date = universe_summary.snapshot_date

    if snapshot_date is None:
        dates = panel.index.get_level_values("date").unique().sort_values()
        if dates.empty:
            return pd.DataFrame()
        snapshot_date = dates.max()

    try:
        snapshot_df = panel.xs(snapshot_date, level="date")
    except KeyError:
        subset = panel[panel.index.get_level_values("date") <= snapshot_date]
        if subset.empty:
            return pd.DataFrame()
        last_date = subset.index.get_level_values("date").max()
        snapshot_df = panel.xs(last_date, level="date")

    cols_keep = ["value_score", "quality_score", "growth_score", "composite_score"]
    if include_momentum:
        cols_keep += [
            "value_score_momentum",
            "quality_score_momentum",
            "growth_score_momentum",
            "composite_score_momentum",
        ]
    if include_stability:
        cols_keep += [
            "value_score_stability",
            "quality_score_stability",
            "growth_score_stability",
            "composite_score_stability",
        ]

    cols_keep = [c for c in cols_keep if c in snapshot_df.columns]
    out = snapshot_df[cols_keep].copy()
    out.index.name = "symbol"
    return out


def select_top_n_by_score(
    snapshot_df: pd.DataFrame,
    *,
    score_column: str = "composite_score",
    n: int = 10,
) -> pd.DataFrame:
    """
    מחזיר טופ N מדדים לפי score_column (גבוה→נמוך).
    """
    if snapshot_df.empty or score_column not in snapshot_df.columns:
        return pd.DataFrame()
    s = pd.to_numeric(snapshot_df[score_column], errors="coerce")
    top = snapshot_df.loc[s.sort_values(ascending=False).head(n).index].copy()
    return top.sort_values(score_column, ascending=False)


def select_bottom_n_by_score(
    snapshot_df: pd.DataFrame,
    *,
    score_column: str = "composite_score",
    n: int = 10,
) -> pd.DataFrame:
    """
    מחזיר בוטום N מדדים לפי score_column (נמוך→גבוה).
    """
    if snapshot_df.empty or score_column not in snapshot_df.columns:
        return pd.DataFrame()
    s = pd.to_numeric(snapshot_df[score_column], errors="coerce")
    bottom = snapshot_df.loc[s.sort_values(ascending=True).head(n).index].copy()
    return bottom.sort_values(score_column, ascending=True)


def bucket_scores_into_quantiles(
    snapshot_df: pd.DataFrame,
    *,
    score_column: str = "composite_score",
    quantiles: int = 5,
    top_bucket_label: str = "Top",
    bottom_bucket_label: str = "Bottom",
) -> pd.Series:
    """
    מחלק את המדדים ל-quantiles (למשל quintiles) לפי score_column.

    מחזיר:
        Series(index=symbol, values=bucket_label או מספר bucket).
    """
    if snapshot_df.empty or score_column not in snapshot_df.columns:
        return pd.Series(dtype=object)

    s = pd.to_numeric(snapshot_df[score_column], errors="coerce")
    # qcut יחלק ל-buckets על בסיס quantiles
    try:
        buckets = pd.qcut(s, q=quantiles, labels=False, duplicates="drop")
    except ValueError:
        # אם אין מספיק ערכים, נחזיר ריק
        return pd.Series(dtype=object)

    labels = []
    max_bucket = int(buckets.max())
    for b in buckets:
        if pd.isna(b):
            labels.append(None)
        else:
            b_int = int(b)
            if b_int == 0:
                labels.append(f"{bottom_bucket_label}_{quantiles}")
            elif b_int == max_bucket:
                labels.append(f"{top_bucket_label}_{quantiles}")
            else:
                labels.append(f"Q{b_int+1}_{quantiles}")
    return pd.Series(labels, index=s.index, name="bucket")


# ============================================================
# 13) Universe-level summaries (momentum, stability)
# ============================================================

def summarize_universe_momentum(
    panel: FundamentalPanel,
    *,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    מסכם מומנטום ממוצע/חציוני ברמת Universe לטווח הזמן האחרון הזמין.

    מחזיר DataFrame:
        index = column_name
        columns = [mean_momentum, median_momentum]
    """
    if panel is None or panel.empty:
        return pd.DataFrame()

    if columns is None:
        columns = [
            "value_score_momentum",
            "quality_score_momentum",
            "growth_score_momentum",
            "composite_score_momentum",
        ]

    # ניקח את התאריך האחרון
    last_date = panel.index.get_level_values("date").max()
    snap = panel.xs(last_date, level="date")

    rows: List[Dict[str, Any]] = []
    for col in columns:
        if col not in snap.columns:
            continue
        s = pd.to_numeric(snap[col], errors="coerce").dropna()
        if s.empty:
            continue
        rows.append(
            {
                "column": col,
                "mean_momentum": float(s.mean()),
                "median_momentum": float(s.median()),
            }
        )

    if not rows:
        return pd.DataFrame()

    df_out = pd.DataFrame(rows).set_index("column")
    return df_out


def summarize_universe_stability(
    panel: FundamentalPanel,
    *,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    מסכם יציבות (std) של scores ברמת Universe עבור התאריך האחרון:

    החזר:
        DataFrame(index=column, columns=[mean_stability, median_stability])
    """
    if panel is None or panel.empty:
        return pd.DataFrame()

    if columns is None:
        columns = [
            "value_score_stability",
            "quality_score_stability",
            "growth_score_stability",
            "composite_score_stability",
        ]

    last_date = panel.index.get_level_values("date").max()
    snap = panel.xs(last_date, level="date")

    rows: List[Dict[str, Any]] = []
    for col in columns:
        if col not in snap.columns:
            continue
        s = pd.to_numeric(snap[col], errors="coerce").dropna()
        if s.empty:
            continue
        rows.append(
            {
                "column": col,
                "mean_stability": float(s.mean()),
                "median_stability": float(s.median()),
            }
        )

    if not rows:
        return pd.DataFrame()

    df_out = pd.DataFrame(rows).set_index("column")
    return df_out


# Part 3/4 מסתיים כאן.
# בחלק 4/4 נוסיף:
#   - Multi-horizon percentiles ברמת מדד
#   - decomposition של תשואה (earnings vs rerating vs dividends) ברמת מדד/Universe
#   - hooks ל-Regime-aware / Risk-adjusted / Factor-neutral scores
#   - דוחות Markdown/JSON ברמת Universe
#   - ועוד פונקציות export/אינטגרציה לדשבורד.
# ============================================================
# Part 4/4 — Multi-Horizon Percentiles, Return Decomposition,
#            Regime/Risk Hooks & Reporting/Export
# ============================================================
"""
בחלק האחרון אנחנו סוגרים את מנוע הפנדומנטל ברמת Index/Universe:

מה נכנס פה בפועל
----------------
1. Multi-horizon percentiles ברמת מדד:
   - compute_fundamental_percentiles_for_symbol(...)
   - אחוזון של הערך האחרון בכל שדה על פני 1Y / 3Y / 5Y / 10Y (או מה שתגדיר).

2. Return decomposition:
   - decompose_return_into_earnings_and_rerating(...)
   - פירוק מקורב של תשואת מחיר ל:
       * earnings_contribution
       * rerating_contribution (שינוי מכפיל)
       * dividend_contribution
       * total_return

3. עטיפות Universe:
   - compute_universe_percentiles_snapshot(...)
   - decompose_universe_returns(...) – hook מתאים לניתוח חוצה מדדים.

4. Hooks ל-Regime-aware / Risk-adjusted / Factor-neutral (רמת API):
   - apply_risk_overlay_to_scores(...)
   - apply_factor_neutral_adjustment(...)
   (מימוש בסיסי ששומר על גמישות להמשך).

5. Reporting / Export:
   - generate_index_markdown_report(...)
   - generate_universe_markdown_report(...)
   - export_index_scores_to_dict(...)
   - export_universe_scores_to_dict(...)

הכל בנוי כך שיהיה:
- self-contained
- נוח לחיבור ל־Streamlit Tabs
- מתאים להמשך הרחבה (core/macro_engine, core/fair_value_engine וכו').
"""

# ============================================================
# 14) Multi-horizon percentiles for a single index
# ============================================================

def _compute_field_percentiles_over_horizons(
    s: pd.Series,
    *,
    horizons_days: Mapping[str, int],
) -> Dict[str, float]:
    """
    מקבל סדרת זמן של שדה פנדומנטלי (עם DatetimeIndex) ומחשב
    את אחוזון הערך האחרון ביחס להיסטוריה האחרונה בכל אופק זמן.

    horizons_days:
        dict {name -> מספר ימים}, לדוגמה:
            {"1y": 365, "3y": 365*3, "5y": 365*5, "10y": 365*10}

    מחזיר:
        dict {f"pct_{name}" -> percentile [0,100]}
    """
    s = pd.to_numeric(s, errors="coerce")
    s = s.dropna()
    if s.empty:
        return {f"pct_{name}": np.nan for name in horizons_days.keys()}

    if not isinstance(s.index, pd.DatetimeIndex):
        # אין אינדקס זמן – נחשב percentile על כל ההיסטוריה כיחידה אחת
        last_val = s.iloc[-1]
        rank_full = (s <= last_val).sum() / float(len(s)) * 100.0
        return {f"pct_{name}": float(rank_full) for name in horizons_days.keys()}

    s = s.sort_index()
    last_ts = s.index.max()
    last_val = s.loc[last_ts]

    res: Dict[str, float] = {}
    for name, n_days in horizons_days.items():
        cutoff = last_ts - pd.Timedelta(days=n_days)
        hist = s[s.index >= cutoff]
        if hist.empty:
            res[f"pct_{name}"] = np.nan
            continue
        pct = (hist <= last_val).sum() / float(len(hist)) * 100.0
        res[f"pct_{name}"] = float(pct)
    return res


def compute_fundamental_percentiles_for_symbol(
    symbol: str,
    *,
    fields: Sequence[str],
    horizons_days: Mapping[str, int] | None = None,
    allow_remote: bool = True,
) -> pd.DataFrame:
    """
    מחשב אחוזונים רב-אופקיים (multi-horizon percentiles) לשדות פנדומנטליים עבור מדד.

    פרמטרים:
    ---------
    symbol : str
        סימול המדד/ETF.
    fields : Sequence[str]
        רשימת שדות לחישוב אחוזונים (pe, pb, roe, וכו').
    horizons_days : Mapping[str,int] | None
        dict {שם אופק -> ימים}. ברירת מחדל:
            {"1y": 365, "3y": 365*3, "5y": 365*5, "10y": 365*10}
    allow_remote : bool
        האם מותר למשוך מה-provider אם אין קובץ.

    מחזיר:
    -------
    DataFrame:
        index   = field
        columns = pct_1y, pct_3y, pct_5y, pct_10y (או לפי מה שהגדרת)
    """
    if horizons_days is None:
        horizons_days = {
            "1y": 365,
            "3y": 365 * 3,
            "5y": 365 * 5,
            "10y": 365 * 10,
        }

    df = load_index_fundamentals(
        symbol,
        start=None,
        end=None,
        fields=fields,
        allow_remote=allow_remote,
        force_refresh=False,
        require_fresh_local=False,
        allow_partial=True,
    )

    if df is None or df.empty:
        logger.warning("compute_fundamental_percentiles_for_symbol: empty DF for %s", symbol)
        return pd.DataFrame()

    df = _ensure_datetime_index(df)
    out_rows: list[Dict[str, Any]] = []

    for field in fields:
        if field not in df.columns:
            continue
        s = _safe_series(df, field)
        if s.dropna().empty:
            continue
        pct_dict = _compute_field_percentiles_over_horizons(
            s,
            horizons_days=horizons_days,
        )
        row = {"field": field}
        row.update(pct_dict)
        out_rows.append(row)

    if not out_rows:
        return pd.DataFrame()

    out = pd.DataFrame(out_rows).set_index("field")
    return out


# ============================================================
# 15) Return decomposition (earnings vs rerating vs dividends)
# ============================================================

def decompose_return_into_earnings_and_rerating(
    price: pd.Series,
    eps: pd.Series,
    *,
    dividends: Optional[pd.Series] = None,
) -> Dict[str, float]:
    """
    מפרק תשואת מחיר לאורך תקופה ל-4 מרכיבים:

        total_return           – P1 / P0 - 1
        earnings_contribution  – E1 / E0 - 1
        rerating_contribution  – PE1 / PE0 - 1 (שינוי מכפיל)
        dividend_contribution  – Sum(dividends) / P0 (approx total yield)

    הערות:
    -------
    - זו אינה Decomposition מדויקת אקטוארית, אלא קירוב אינטואיטיבי.
    - מניח E0/E1 חיובי; אם לא – מחזיר NaN.
    """
    price = pd.to_numeric(price, errors="coerce").dropna()
    eps = pd.to_numeric(eps, errors="coerce").dropna()

    if price.empty or eps.empty:
        return {
            "total_return": np.nan,
            "earnings_contribution": np.nan,
            "rerating_contribution": np.nan,
            "dividend_contribution": np.nan,
        }

    idx = price.index.intersection(eps.index)
    if idx.empty:
        return {
            "total_return": np.nan,
            "earnings_contribution": np.nan,
            "rerating_contribution": np.nan,
            "dividend_contribution": np.nan,
        }

    price = price.loc[idx]
    eps = eps.loc[idx]

    P0 = float(price.iloc[0])
    P1 = float(price.iloc[-1])
    E0 = float(eps.iloc[0])
    E1 = float(eps.iloc[-1])

    if P0 <= 0 or E0 <= 0 or E1 <= 0:
        return {
            "total_return": np.nan,
            "earnings_contribution": np.nan,
            "rerating_contribution": np.nan,
            "dividend_contribution": np.nan,
        }

    total_return = P1 / P0 - 1.0
    earnings_contribution = E1 / E0 - 1.0

    PE0 = P0 / E0
    PE1 = P1 / E1
    if PE0 <= 0:
        rerating_contribution = np.nan
    else:
        rerating_contribution = PE1 / PE0 - 1.0

    dividend_contribution = np.nan
    if dividends is not None:
        d = pd.to_numeric(dividends, errors="coerce").dropna()
        d = d.loc[price.index.intersection(d.index)]
        total_div = float(d.sum()) if not d.empty else 0.0
        dividend_contribution = total_div / P0 if P0 > 0 else np.nan

    return {
        "total_return": float(total_return),
        "earnings_contribution": float(earnings_contribution),
        "rerating_contribution": float(rerating_contribution),
        "dividend_contribution": float(dividend_contribution),
    }


def decompose_universe_returns(
    price_panel: FundamentalPanel,
    eps_panel: FundamentalPanel,
    *,
    dividends_panel: Optional[FundamentalPanel] = None,
) -> pd.DataFrame:
    """
    Hook ברמת Universe לפירוק תשואות:

    פרמטרים:
    ---------
    price_panel :
        DataFrame MultiIndex (date, symbol) עם price (עמודה אחת: 'price' או 'close').
    eps_panel :
        DataFrame MultiIndex (date, symbol) עם EPS (עמודה אחת: 'eps' או 'eps_ttm').
    dividends_panel :
        אם קיים – MultiIndex עם 'dividends' או 'dividend'.

    מחזיר:
    -------
    DataFrame index=symbol, columns:
        total_return, earnings_contribution, rerating_contribution, dividend_contribution
    """
    if price_panel is None or price_panel.empty or eps_panel is None or eps_panel.empty:
        return pd.DataFrame()

    symbols = sorted(price_panel.index.get_level_values("symbol").unique())
    rows: list[Dict[str, Any]] = []

    # נניח עמודת מחיר בשם 'price' או 'close'
    price_col = "price" if "price" in price_panel.columns else "close"
    eps_col = "eps" if "eps" in eps_panel.columns else "eps_ttm"

    div_col = None
    if dividends_panel is not None:
        if "dividends" in dividends_panel.columns:
            div_col = "dividends"
        elif "dividend" in dividends_panel.columns:
            div_col = "dividend"

    for sym in symbols:
        try:
            p = price_panel.xs(sym, level="symbol")[price_col]
            e = eps_panel.xs(sym, level="symbol")[eps_col]
            d = None
            if dividends_panel is not None and div_col is not None:
                d = dividends_panel.xs(sym, level="symbol")[div_col]
            deco = decompose_return_into_earnings_and_rerating(p, e, dividends=d)
            deco["symbol"] = sym
            rows.append(deco)
        except Exception as exc:  # pragma: no cover
            logger.warning("decompose_universe_returns: failed for %s: %s", sym, exc)
            continue

    if not rows:
        return pd.DataFrame()

    df_out = pd.DataFrame(rows).set_index("symbol")
    return df_out


# ============================================================
# 16) Regime & Risk hooks (basic implementation)
# ============================================================

def apply_risk_overlay_to_scores(
    panel: FundamentalPanel,
    *,
    risk_cfg: RiskOverlayConfig = DEFAULT_RISK_OVERLAY_CONFIG,
    vol_column: str = "realized_vol_1y",
    dd_column: str = "max_drawdown_3y",
    tail_column: str = "tail_risk",
) -> FundamentalPanel:
    """
    Hook בסיסי להתאמת composite_score לסיכון:

    רעיון:
    -------
    - אנו מניחים שיש ב-panel (או בהמשך) עמודות:
        * realized_vol_1y       – סטיית תקן היסטורית.
        * max_drawdown_3y       – drawdown מקסימלי.
        * tail_risk             – מדד tail (למשל ES/VAR).
    - score_risk_penalty נבנה כך שערכים גבוהים בשליליים → penalty גבוה יותר.

    מימוש בסיסי:
    -------------
    composite_risk_adjusted = composite_score
                              - overlay_strength * normalized_penalty

    זה Hook בסיסי – אפשר להחליף ברגולציה מורכבת יותר בעתיד.
    """
    if panel is None or panel.empty:
        return panel

    panel = panel.copy()

    if "composite_score" not in panel.columns:
        return panel

    # נבנה penalty על בסיס z-scores
    penalty = pd.Series(0.0, index=panel.index)

    for col, w in [
        (vol_column, risk_cfg.weight_volatility_penalty),
        (dd_column, risk_cfg.weight_drawdown_penalty),
        (tail_column, risk_cfg.weight_tail_risk_penalty),
    ]:
        if w <= 0 or col not in panel.columns:
            continue
        s = _safe_series(panel, col)
        z = _zscore_series(s, robust=True)  # robust זוכרת extreme
        z_pos = z.clip(lower=0.0)  # רק סיכון "גבוה" מעניש
        penalty = penalty.add(w * z_pos.reindex(panel.index), fill_value=0.0)

    if penalty.abs().sum() == 0:
        panel["composite_score_risk_adj"] = panel["composite_score"]
        return panel

    # ננרמל penalty לטווח [0, 20] בערך
    pen_norm = penalty
    pen_norm = pen_norm - pen_norm.min()
    max_val = pen_norm.max() or 1e-9
    pen_norm = 20.0 * pen_norm / max_val

    adjusted = panel["composite_score"] - risk_cfg.overlay_strength * pen_norm
    panel["composite_score_risk_adj"] = _normalize_score(adjusted)

    return panel


def apply_factor_neutral_adjustment(
    panel: FundamentalPanel,
    *,
    fn_cfg: FactorNeutralConfig = DEFAULT_FACTOR_NEUTRAL_CONFIG,
) -> FundamentalPanel:
    """
    Hook לפקטור-ניטרליזציה בסיסית על composite_score.

    כרגע המימוש הוא placeholder מתון:
    - מניח שיש עמודות exposure לפקטורים (value_exposure, quality_exposure, growth_exposure, size_exposure, region_exposure).
    - מנרמל את composite_score לפי הסטייה המוחלטת מחשיפה "ניטרלית" (0).

    זה מספיק כהתחלה ומוכן להחלפה ל-regression מבוסס factor-model בעתיד.
    """
    if panel is None or panel.empty:
        return panel

    if "composite_score" not in panel.columns:
        return panel

    panel = panel.copy()
    penalty = pd.Series(0.0, index=panel.index)

    def _apply_exposure_penalty(field: str, flag: bool, current: pd.Series) -> pd.Series:
        if not flag or field not in panel.columns:
            return current
        s = pd.to_numeric(panel[field], errors="coerce")
        # נניח ש-0 = ניטרלי, penalty = |exposure|
        current = current.add(s.abs().reindex(panel.index).fillna(0.0), fill_value=0.0)
        return current

    penalty = _apply_exposure_penalty("value_exposure", fn_cfg.neutralize_value, penalty)
    penalty = _apply_exposure_penalty("quality_exposure", fn_cfg.neutralize_quality, penalty)
    penalty = _apply_exposure_penalty("growth_exposure", fn_cfg.neutralize_growth, penalty)
    penalty = _apply_exposure_penalty("size_exposure", fn_cfg.neutralize_size, penalty)
    penalty = _apply_exposure_penalty("region_exposure", fn_cfg.neutralize_region, penalty)

    if penalty.abs().sum() == 0:
        panel["composite_score_factor_neutral"] = panel["composite_score"]
        return panel

    # ננרמל penalty ל-0–20
    penalty = penalty - penalty.min()
    max_val = penalty.max() or 1e-9
    penalty = 20.0 * penalty / max_val

    adjusted = panel["composite_score"] - fn_cfg.neutralization_strength * penalty
    panel["composite_score_factor_neutral"] = _normalize_score(adjusted)
    return panel


# ============================================================
# 17) Reporting & export
# ============================================================

def export_index_scores_to_dict(
    index_scores: IndexScoreSeries,
    *,
    latest_only: bool = True,
) -> Dict[str, Any]:
    """
    מחזיר את הציונים של מדד כ-dict נוח ל-JSON/Streamlit.

    אם latest_only=True → ניקח רק את התאריך האחרון.
    אחרת → נחזיר הכל כמבנה:
        { "symbol": ..., "scores": [ {date: ..., value_score: ..., ...}, ... ] }
    """
    symbol = index_scores.symbol
    df = index_scores.df
    if df is None or df.empty:
        return {"symbol": symbol, "scores": []}

    df = _ensure_datetime_index(df)
    if latest_only:
        row = df.iloc[-1]
        base = {
            "date": row.name.isoformat(),
            "value_score": float(row.get("value_score", np.nan)),
            "quality_score": float(row.get("quality_score", np.nan)),
            "growth_score": float(row.get("growth_score", np.nan)),
            "composite_score": float(row.get("composite_score", np.nan)),
        }
        # momentum/stability אם קיימים
        for col in [
            "value_score_momentum",
            "quality_score_momentum",
            "growth_score_momentum",
            "composite_score_momentum",
            "value_score_stability",
            "quality_score_stability",
            "growth_score_stability",
            "composite_score_stability",
        ]:
            if col in df.columns:
                base[col] = float(row.get(col, np.nan))
        return {"symbol": symbol, "scores": [base]}

    scores_list: list[Dict[str, Any]] = []
    for ts, row in df.iterrows():
        item: Dict[str, Any] = {
            "date": ts.isoformat(),
            "value_score": float(row.get("value_score", np.nan)),
            "quality_score": float(row.get("quality_score", np.nan)),
            "growth_score": float(row.get("growth_score", np.nan)),
            "composite_score": float(row.get("composite_score", np.nan)),
        }
        for col in [
            "value_score_momentum",
            "quality_score_momentum",
            "growth_score_momentum",
            "composite_score_momentum",
            "value_score_stability",
            "quality_score_stability",
            "growth_score_stability",
            "composite_score_stability",
        ]:
            if col in df.columns:
                item[col] = float(row.get(col, np.nan))
        scores_list.append(item)

    return {"symbol": symbol, "scores": scores_list}


def export_universe_scores_to_dict(
    universe_summary: UniverseScoresSummary,
    *,
    latest_only: bool = True,
) -> Dict[str, Any]:
    """
    מחזיר את הציוני Universe כ-dict (לשימוש ב-API/דשבורד).

    מבנה:
        {
          "snapshot_date": "...",
          "panel_shape": [rows, cols],
          "rank_columns": [list of columns ranked],
          "symbols": {
              "SPY": {...},
              "QQQ": {...},
              ...
          }
        }
    """
    panel = universe_summary.panel
    if panel is None or panel.empty:
        return {"snapshot_date": None, "panel_shape": [0, 0], "symbols": {}}

    snapshot_df = get_universe_score_snapshot(universe_summary)
    if snapshot_df.empty:
        return {"snapshot_date": None, "panel_shape": list(panel.shape), "symbols": {}}

    out_symbols: Dict[str, Dict[str, Any]] = {}
    for sym, row in snapshot_df.iterrows():
        item: Dict[str, Any] = {
            "value_score": float(row.get("value_score", np.nan)),
            "quality_score": float(row.get("quality_score", np.nan)),
            "growth_score": float(row.get("growth_score", np.nan)),
            "composite_score": float(row.get("composite_score", np.nan)),
        }
        for col in snapshot_df.columns:
            if col not in item:
                item[col] = float(row.get(col, np.nan))
        out_symbols[str(sym)] = item

    return {
        "snapshot_date": universe_summary.snapshot_date.isoformat()
        if universe_summary.snapshot_date is not None
        else None,
        "panel_shape": [int(panel.shape[0]), int(panel.shape[1])],
        "rank_columns": list(universe_summary.rank_columns.keys()),
        "symbols": out_symbols,
        "coverage_info": universe_summary.coverage_info,
    }


def generate_index_markdown_report(
    symbol: str,
    *,
    engine_cfg: ScoreEngineConfig | None = None,
    horizons_days: Mapping[str, int] | None = None,
) -> str:
    """
    יוצרת דוח Markdown קצר למדד בודד:
    - ציוני Value/Quality/Growth/Composite (snapshot אחרון).
    - Percentiles רב-אופקיים לשדות נבחרים.
    """
    if engine_cfg is None:
        engine_cfg = ScoreEngineConfig()

    series = score_index_fundamentals(symbol, engine_cfg=engine_cfg)
    if series.df is None or series.df.empty:
        return f"# Index Fundamentals Report — {symbol}\n\nNo data available.\n"

    df = _ensure_datetime_index(series.df)
    last_row = df.iloc[-1]

    if horizons_days is None:
        horizons_days = {
            "1y": 365,
            "3y": 365 * 3,
            "5y": 365 * 5,
            "10y": 365 * 10,
        }

    # ניקח כמה שדות Value ו-Quality עיקריים
    fields_for_pct = ["pe", "pb", "dividend_yield", "roe", "net_margin"]
    pct_df = compute_fundamental_percentiles_for_symbol(
        symbol,
        fields=[f for f in fields_for_pct if f in df.columns],
        horizons_days=horizons_days,
    )

    lines: list[str] = []
    lines.append(f"# Index Fundamentals Report — {symbol.upper()}")
    lines.append("")
    lines.append(f"**As of:** {last_row.name.date().isoformat()}")
    lines.append("")
    lines.append("## Scores (0–100)")
    lines.append("")
    lines.append(f"- Value score: **{last_row.get('value_score', np.nan):.1f}**")
    lines.append(f"- Quality score: **{last_row.get('quality_score', np.nan):.1f}**")
    lines.append(f"- Growth score: **{last_row.get('growth_score', np.nan):.1f}**")
    lines.append(f"- Composite score: **{last_row.get('composite_score', np.nan):.1f}**")
    lines.append("")

    if not pct_df.empty:
        lines.append("## Valuation & Quality Percentiles")
        lines.append("")
        lines.append(pct_df.round(1).to_markdown())
        lines.append("")

    return "\n".join(lines)


def generate_universe_markdown_report(
    universe_summary: UniverseScoresSummary,
    *,
    top_n: int = 10,
) -> str:
    """
    יוצר דוח Markdown קצר ל-Universe:
    - snapshot של תאריך אחרון.
    - Top/Bottom N לפי composite_score.
    - תמצית מומנטום/יציבות.
    """
    panel = universe_summary.panel
    if panel is None or panel.empty:
        return "# Universe Fundamentals Report\n\nNo data available.\n"

    snapshot_df = get_universe_score_snapshot(universe_summary)
    if snapshot_df.empty:
        return "# Universe Fundamentals Report\n\nNo snapshot data available.\n"

    snap_date = universe_summary.snapshot_date
    if snap_date is None:
        dates = panel.index.get_level_values("date").unique().sort_values()
        snap_date = dates.max()

    top = select_top_n_by_score(snapshot_df, n=top_n)
    bottom = select_bottom_n_by_score(snapshot_df, n=top_n)
    mom_summary = summarize_universe_momentum(panel)
    stab_summary = summarize_universe_stability(panel)

    lines: list[str] = []
    lines.append("# Universe Fundamentals Report")
    lines.append("")
    lines.append(f"**Snapshot date:** {snap_date.date().isoformat()}")
    lines.append("")
    lines.append("## Top Composite Scores")
    lines.append("")
    if not top.empty:
        lines.append(top[["composite_score"]].round(1).to_markdown())
    else:
        lines.append("_No top scores available._")
    lines.append("")
    lines.append("## Bottom Composite Scores")
    lines.append("")
    if not bottom.empty:
        lines.append(bottom[["composite_score"]].round(1).to_markdown())
    else:
        lines.append("_No bottom scores available._")
    lines.append("")

    if not mom_summary.empty:
        lines.append("## Universe Score Momentum Summary")
        lines.append("")
        lines.append(mom_summary.round(3).to_markdown())
        lines.append("")

    if not stab_summary.empty:
        lines.append("## Universe Score Stability Summary")
        lines.append("")
        lines.append(stab_summary.round(3).to_markdown())
        lines.append("")

    return "\n".join(lines)


# ============================================================
# 18) __all__ – Public API for core/index_fundamentals.py
# ============================================================

__all__ = [
    # Configs
    "ValueScoreConfig",
    "QualityScoreConfig",
    "GrowthScoreConfig",
    "CompositeScoreConfig",
    "RiskOverlayConfig",
    "RegimeScoreConfig",
    "FactorNeutralConfig",
    "ScoreEngineConfig",

    # Dataclasses / containers
    "IndexScoreRow",
    "IndexScoreSeries",
    "UniverseScoresSummary",

    # Single-index scoring API
    "score_index_fundamentals",
    "score_index_fundamentals_df",

    # Universe scoring API
    "score_universe_fundamentals",

    # Cross-sectional analytics
    "compute_cross_sectional_ranks",
    "compute_cross_sectional_percentiles",
    "get_universe_score_snapshot",
    "select_top_n_by_score",
    "select_bottom_n_by_score",
    "bucket_scores_into_quantiles",
    "summarize_universe_momentum",
    "summarize_universe_stability",

    # Percentiles & decomposition
    "compute_fundamental_percentiles_for_symbol",
    "decompose_return_into_earnings_and_rerating",
    "decompose_universe_returns",

    # Regime/Risk hooks
    "apply_risk_overlay_to_scores",
    "apply_factor_neutral_adjustment",

    # Reporting & export
    "export_index_scores_to_dict",
    "export_universe_scores_to_dict",
    "generate_index_markdown_report",
    "generate_universe_markdown_report",
]
