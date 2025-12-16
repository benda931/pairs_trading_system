# -*- coding: utf-8 -*-
"""
analysis_core.py — מנוע ניתוח ודירוג ברמת קרן גידור
====================================================

Part 1/6 — Core models & scoring engine (גרסה מורחבת)
----------------------------------------------------
בחלק הזה אנחנו בונים את:

1. מודלי קונטקסט ו-Regime (שוק, תנודתיות, מזהי ריצה/סריקה).
2. מודלי פרופיל חכמים (Pydantic):
   - ParamProfile עם target_direction, invert, hard caps, lambda_override, grouping.
   - AnalysisProfile עם equal_weights, winsorization, Regime-aware profiles,
     מיפוי סיגמואידי (Sigmoid) ועוד.
3. מנוע דירוג פר פרמטר + ציון כולל לזוג:
   - טווחים [lo, hi]
   - דעיכה אקספוננציאלית ביחידות רוחב-טווח
   - תמיכה ב-inside / above / below + invert
   - משקולות, equal_weights, טיפול ב-NaN.
   - Sigmoid mapping אחרי הממוצע (לנרמול בין יוניברס שונים).
   - Hook לכיול (calibration) לציון→הסתברות רווח (דרך callback אופציונלי).
4. Breakdown מלא (PairScoreBreakdown) שמכיל את כל המידע ל-UI, לוגים, Export.

שאר החלקים (2–6) יבנו על זה:
- metrics (חישובי half-life, corr, z-score וכו’)
- טעינת pair.json / config.json
- Streamlit UI
- Cache, Persist ל-DuckDB/Parquet, Drill-Down, Export, וכו'.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime, date
from enum import Enum
from math import exp, isfinite
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, ConfigDict, field_validator

from core.tab_comparison_matrices import (
    TabProfile,
    MetricMeta,
    TabComparisonConfig,
    build_comparison_bundle,
    explain_similarity_contributions,
    compute_alignment_scores,
    detect_tab_anomalies,
    build_composite_profile,  
)

# ========================= Logging & Constants =========================

logger = logging.getLogger(__name__)

DEFAULT_LAMBDA = 0.5  # ברירת מחדל לחדות הדעיכה האקספוננציאלית
SCORE_MIN = 0.0
SCORE_MAX = 100.0


# ========================= Regime & Context Models =========================

def _extract_symbols_from_pair(pair_obj: Any) -> Tuple[str, str]:
    """
    מקבל אובייקט "זוג" בכל פורמט סביר,
    ומחזיר (sym_x, sym_y) כמחרוזות.

    תומך:
    - dict עם key "symbols": {'symbols': ['XLY', 'XLC']}
    - dict עם keys 'sym_x'/'sym_y'
    - list/tuple באורך 2
    - מחרוזת בפורמטים כמו "XLY/XLC" או "XLY,XLC"
    """
    # dict {'symbols': ['XLY', 'XLC']}
    if isinstance(pair_obj, dict):
        if "symbols" in pair_obj and isinstance(pair_obj["symbols"], (list, tuple)):
            syms = pair_obj["symbols"]
            if len(syms) == 2:
                return str(syms[0]), str(syms[1])
        if "sym_x" in pair_obj and "sym_y" in pair_obj:
            return str(pair_obj["sym_x"]), str(pair_obj["sym_y"])

    # רשימה / טופל
    if isinstance(pair_obj, (list, tuple)) and len(pair_obj) == 2:
        return str(pair_obj[0]), str(pair_obj[1])

    # מחרוזת "XLY/XLC" או "XLY,XLC"
    if isinstance(pair_obj, str):
        if "/" in pair_obj:
            a, b = pair_obj.split("/", 1)
            return a.strip(), b.strip()
        if "," in pair_obj:
            a, b = pair_obj.split(",", 1)
            return a.strip(), b.strip()

    raise ValueError(f"Cannot extract symbols from pair object: {pair_obj!r}")


def _safe_corr(a: pd.Series, b: pd.Series) -> float:
    """קורלציה בטוחה בין שני וקטורים מספריים (לא על Datetime/index)."""
    a_arr = np.asarray(a.values, dtype="float64")
    b_arr = np.asarray(b.values, dtype="float64")

    if a_arr.size < 2 or b_arr.size < 2:
        return float("nan")

    try:
        return float(np.corrcoef(a_arr, b_arr)[0, 1])
    except Exception:
        return float("nan")
    
class VolRegime(str, Enum):
    """סיווגי משטר תנודתיות בסיסיים."""

    UNKNOWN = "unknown"
    LOW_VOL = "low_vol"
    MID_VOL = "mid_vol"
    HIGH_VOL = "high_vol"
    EXTREME_VOL = "extreme_vol"


class MarketBias(str, Enum):
    """סיווג שוק: שורי/דובי/צדדי."""

    NEUTRAL = "neutral"
    BULL = "bull"
    BEAR = "bear"
    RANGE = "range"


class TargetDirection(str, Enum):
    """
    איך לפרש 'טוב' עבור הפרמטר:

    - INSIDE: הכי טוב כשהערך בתוך [lo, hi] (ברירת מחדל).
    - ABOVE:  הכי טוב כשהערך *גדול* מה-hi (למשל win_rate, sharpe).
    - BELOW:  הכי טוב כשהערך *קטן* מה-lo (למשל drawdown, risk).
    """

    INSIDE = "inside"
    ABOVE = "above"
    BELOW = "below"


@dataclass
class AnalysisContext:
    """
    קונטקסט ניתוח זוג — יישלח בין השכבות (metrics, scoring, UI).

    הערה: dataclass קליל, לא תלוי ב-Streamlit.
    """

    sym_x: str
    sym_y: str
    start_date: Optional[date] = None
    end_date: Optional[date] = None

    vol_regime: VolRegime = VolRegime.UNKNOWN
    market_bias: MarketBias = MarketBias.NEUTRAL

    # מזהים לצורך לוגים / Persist / Audit
    run_id: Optional[str] = None
    scan_id: Optional[str] = None
    universe_id: Optional[str] = None

    # שדה חופשי לאינפורמציה נוספת (למשל source: "pair.json", "optuna")
    extra: Dict[str, Any] | None = None


# ========================= Profile & Param Models =========================


class ParamProfile(BaseModel):
    """
    תיאור פרמטר בודד במנוע הדירוג.

    name:   שם לוגי (צריך להתאים למפתח במדדים, למשל "half_life").
    lo/hi:  טווח מועדף / reference.
    weight: משקל בציון הכולל.
    invert: להפוך את ההיגיון (נדיר; בשימוש מתקדם בלבד).
    target_direction:
      - INSIDE (ברירת מחדל) → הכי טוב בתוך [lo, hi]
      - ABOVE  → טוב כשהערך גבוה מה-hi
      - BELOW  → טוב כשהערך נמוך מה-lo
    enabled: אם False → הפרמטר לא משתתף בציון הכולל.
    group:  לקיבוץ לוגי/גרפי (risk/stat/liquidity/...) — עוזר ב-UI.

    הרחבות:
    - lambda_override: λ מותאם לפרמטר (אם None → lambda_default של הפרופיל).
    - hard_cap_min/hard_cap_max: ערכים שמאפסים מיד את הציון (לפני דעיכה).
    """

    model_config = ConfigDict(extra="ignore")

    name: str = Field(..., min_length=1)
    display_name: Optional[str] = Field(
        default=None,
        description="Label ידידותי להצגה ב-UI; אם None משתמשים ב-name",
    )

    lo: Optional[float] = Field(
        default=None,
        description="גבול תחתון של הטווח (יכול להיות None במקרה ABOVE)",
    )
    hi: Optional[float] = Field(
        default=None,
        description="גבול עליון של הטווח (יכול להיות None במקרה BELOW)",
    )

    weight: float = Field(
        default=1.0,
        ge=0.0,
        description="משקל פרמטר בציון הכולל",
    )
    invert: bool = Field(
        default=False,
        description="היפוך הלוגיקה (בדרך כלל אין צורך; legacy/advanced only)",
    )
    target_direction: TargetDirection = Field(
        default=TargetDirection.INSIDE,
        description="כיוון יעד רצוי (within / above / below)",
    )
    enabled: bool = Field(default=True)
    description: Optional[str] = Field(default=None)
    group: Optional[str] = Field(
        default=None,
        description="קבוצה לוגית (risk/stat/liquidity/quality/...)",
    )

    # פרמטרים מתקדמים:
    lambda_override: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="אם מוגדר, ידרוס את λ הכללי עבור פרמטר זה",
    )
    hard_cap_min: Optional[float] = Field(
        default=None,
        description="מתחת לערך הזה הציון קופץ ישירות ל-0 (לפני הדעיכה הרגילה)",
    )
    hard_cap_max: Optional[float] = Field(
        default=None,
        description="מעל ערך זה הציון קופץ ישירות ל-0",
    )

    @field_validator("hi")
    @classmethod
    def validate_range(cls, hi: Optional[float], info: Any) -> Optional[float]:
        """
        ולידציה בסיסית:
        - במצב INSIDE חייבים lo < hi (אם שניהם לא None).
        - במצבים ABOVE/BELOW מותר lo/hi להיות None בהתאם, אבל לא בוחרים טווח "הפוך".
        """
        lo = info.data.get("lo", None)
        target_dir: TargetDirection = info.data.get(
            "target_direction", TargetDirection.INSIDE
        )

        if target_dir == TargetDirection.INSIDE:
            if lo is not None and hi is not None and not (lo < hi):
                raise ValueError(f"ParamProfile: hi must be > lo (got lo={lo}, hi={hi})")
        return hi

    @property
    def label(self) -> str:
        return self.display_name or self.name


class ProfileMeta(BaseModel):
    """מטא-דאטה עבור פרופיל ניתוח."""

    model_config = ConfigDict(extra="ignore")

    name: str = "default"
    version: str = "1.0"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    author: Optional[str] = None
    notes: Optional[str] = None


class AnalysisProfile(BaseModel):
    """
    פרופיל ניתוח שלם — אוסף פרמטרים + הגדרות גלובליות.

    - params: רשימת ParamProfile (ה“לב”).
    - lambda_default: ברירת מחדל λ לכל הפרמטרים.
    - equal_weights: האם להתעלם ממשקולות ולהשתמש במשקל שווה לכל פרמטר.

    הרחבות:

    - winsorization_pct: קיצוץ זנבות (לסטטיסטיקות לפני חישוב metrics; מנוהל בחלק 2).
    - allow_nan_metrics: אם False וכל פרמטר חסר → ציון כולל 0.
    - regimes: מפה של פרופילים חלופיים לשימוש לפי VolRegime (כל פרופיל שלם בפני עצמו).
    - use_sigmoid_mapping + sigmoid_a/sigmoid_b:
        מאפשרים מיפוי סיגמואידי אחרי ממוצע משוקלל, כך שציון 50 הופך לנקודת גזירה,
        ו"קצוות" מתקרבים ל-0/100 בצורה חלקה.
    """

    model_config = ConfigDict(extra="ignore")

    meta: ProfileMeta = Field(default_factory=ProfileMeta)
    params: List[ParamProfile] = Field(default_factory=list)

    # פרמטרי scoring כלליים:
    lambda_default: float = Field(
        default=DEFAULT_LAMBDA,
        ge=0.0,
        description="λ ברירת מחדל לדעיכה האקספוננציאלית",
    )
    equal_weights: bool = Field(
        default=False,
        description="אם True מתעלמים מהמשקולות שבפרמטרים",
    )

    # הגדרות איכות נתונים:
    winsorization_pct: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=0.25,
        description="קאטיל לקיצוץ זנבות (למשל 0.01 ל-1%) — מיושם בשכבת metrics",
    )
    allow_nan_metrics: bool = Field(
        default=True,
        description="אם False, NaN בפרמטר כלשהו גורם לציון הכולל להיות 0",
    )

    # Regime-aware profiles:
    regimes: Optional[Dict[str, "AnalysisProfile"]] = Field(
        default=None,
        description="פרופילים חלופיים לפי Regime (מפתח=שם רג'ים, ערך=AnalysisProfile)",
    )

    # Sigmoid mapping after weighted mean (לנרמול יקומים שונים)
    use_sigmoid_mapping: bool = Field(
        default=False,
        description="אם True, מפעיל מיפוי סיגמואידי לציון הכולל",
    )
    sigmoid_a: float = Field(
        default=4.0,
        description="פרמטר a ללוגיסטית: שולט על חדות",
    )
    sigmoid_b: float = Field(
        default=0.5,
        description="פרמטר b ללוגיסטית: נקודת גזירה (S/100 - b)",
    )

    @field_validator("params")
    @classmethod
    def ensure_non_empty(cls, v: List[ParamProfile]) -> List[ParamProfile]:
        if not v:
            logger.warning("AnalysisProfile with no params defined.")
        return v

    # ------------------------------------------------------------------ #
    # APIs נוחים לשימוש בקוד                                           #
    # ------------------------------------------------------------------ #

    def get_active_params(self) -> List[ParamProfile]:
        """מחזיר רק פרמטרים פעילים (enabled)."""
        return [p for p in self.params if p.enabled]

    def get_param(self, name: str) -> Optional[ParamProfile]:
        name_lower = name.lower()
        for p in self.params:
            if p.name.lower() == name_lower:
                return p
        return None

    def select_for_regime(self, regime: VolRegime) -> "AnalysisProfile":
        """
        מחזיר פרופיל תואם רג'ים, אם קיים ב-self.regimes.
        אחרת יחזיר את self.

        זה מאפשר:
        - פרופיל בסיסי "default"
        - פרופילים משניים: "low_vol", "high_vol", וכו'.
        """
        if not self.regimes:
            return self
        key = regime.value
        if key in self.regimes:
            return self.regimes[key]
        # fallback למפתח גנרי יותר
        if regime == VolRegime.LOW_VOL and "low_vol" in self.regimes:
            return self.regimes["low_vol"]
        if regime == VolRegime.HIGH_VOL and "high_vol" in self.regimes:
            return self.regimes["high_vol"]
        return self

    def compute_hash(self) -> str:
        """
        מחזיר hash קצר של הפרופיל (meta + params + הגדרות),
        לשימוש ב-DuckDB/Parquet/Cache/Persist.
        """
        # נשתמש ב-json סטנדרטי ללא סדר-מפתחות כדי לקבל hash יציב
        data = self.model_dump(mode="json")
        # אנחנו לא רוצים לכלול regimes משניים בתוך החישוב (כדי לא לנפח),
        # אפשר להחליט אחרת לפי הצורך.
        data.pop("regimes", None)
        raw_bytes = repr(data).encode("utf-8")
        return hashlib.sha256(raw_bytes).hexdigest()[:16]


# ========================= Scoring Result Models =========================


class ParamScore(BaseModel):
    """תוצאה מפורטת לציון פרמטר בודד."""

    name: str
    label: str
    value: Optional[float]
    lo: Optional[float]
    hi: Optional[float]
    score: float
    weight: float
    effective_weight: float
    target_direction: TargetDirection
    invert: bool
    d_norm: Optional[float] = Field(
        default=None,
        description="מרחק מנורמל מהטווח (ביחידות width); None אם בתוך הטווח או לא רלוונטי",
    )
    reason: Optional[str] = Field(
        default=None,
        description="הסבר קצר (למשל NaN, מחוץ לטווח, hard cap וכו')",
    )


class PairScoreBreakdown(BaseModel):
    """
    Breakdown מלא לציון זוג:

    - ציון גולמי (raw_total_score) אחרי ממוצע משוקלל.
    - ציון אחרי Sigmoid (postprocessed_score) אם מופעל.
    - ציון מכויל (calibrated_score) אם עבר דרך Calibrator חיצוני.
    - ציון סופי (total_score) — זה מה שיוצג למשתמש.
    - רשימת ParamScore לכל פרמטר.
    - מידע על הפרופיל, Regime, hash וכו'.
    """

    model_config = ConfigDict(extra="ignore")

    sym_x: str
    sym_y: str

    # ציונים
    raw_total_score: float
    postprocessed_score: float
    calibrated_score: Optional[float] = None
    total_score: float

    # פרמטרים מפורטים
    param_scores: List[ParamScore]

    # הגדרות scoring שנעשה בהן שימוש
    lambda_used: float
    equal_weights: bool
    vol_regime: VolRegime = VolRegime.UNKNOWN
    market_bias: MarketBias = MarketBias.NEUTRAL

    # פרופיל
    profile_name: str = "default"
    profile_version: str = "1.0"
    profile_hash: Optional[str] = None

    # אינדיקטורים לאיכות נתונים
    missing_params: List[str] = Field(default_factory=list)

    # מידע לכיול / הסתברות
    calibrator_name: Optional[str] = None
    probability_of_profit: Optional[float] = None


# ========================= Low-level Helpers =========================


def _clamp_score(x: float) -> float:
    """חיתוך ציון לתוך [SCORE_MIN, SCORE_MAX]."""
    if not isfinite(x):
        return SCORE_MIN
    return float(max(SCORE_MIN, min(SCORE_MAX, x)))


def _safe_float(value: Any) -> Optional[float]:
    """ממיר לכל היותר ל-float אם אפשר, אחרת מחזיר None."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if not isfinite(f):
        return None
    return f


def _sigmoid(x: float) -> float:
    """
    לוגיסטית סטנדרטית σ(x) = 1 / (1 + e^-x)

    עם הגנה מפני Overflow.
    """
    try:
        return 1.0 / (1.0 + exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


# ========================= Core Scoring for Single Param =========================


def _compute_inside_score(
    value: float, lo: float, hi: float, lam: float
) -> Tuple[float, Optional[float]]:
    """
    דירוג INSIDE: הכי טוב להיות בין [lo, hi].
    מחוץ לטווח → דעיכה אקספוננציאלית לפי d_norm.

    מחזיר:
      (score, d_norm)
    """
    if lo is None or hi is None or not isfinite(value):
        return SCORE_MIN, None

    width = hi - lo
    if width <= 0:
        return SCORE_MIN, None

    if lo <= value <= hi:
        return SCORE_MAX, 0.0

    d = min(abs(value - lo), abs(value - hi))
    d_norm = d / width
    raw_score = SCORE_MAX * exp(-lam * d_norm)
    return _clamp_score(raw_score), d_norm


def _compute_above_score(
    value: float, hi: float, lam: float
) -> Tuple[float, Optional[float]]:
    """
    דירוג ABOVE: הכי טוב כשהערך גדול מה-hi.
    ככל שהערך נמוך מה-hi הציון יורד.

    הגיון:
      - אם value >= hi → 100
      - אחרת → דעיכה לפי המרחק מטה ביחס לערך בסיס (hi).
    """
    if hi is None or not isfinite(value):
        return SCORE_MIN, None

    base = abs(hi) if hi != 0 else 1.0
    d = max(0.0, hi - value)
    if d == 0:
        return SCORE_MAX, 0.0

    d_norm = d / base
    raw_score = SCORE_MAX * exp(-lam * d_norm)
    return _clamp_score(raw_score), d_norm


def _compute_below_score(
    value: float, lo: float, lam: float
) -> Tuple[float, Optional[float]]:
    """
    דירוג BELOW: הכי טוב כשהערך קטן מה-lo.
    ככל שהערך גבוה מה-lo הציון יורד.

    הגיון:
      - אם value <= lo → 100
      - אחרת → דעיכה לפי המרחק למעלה ביחס לערך בסיס (lo).
    """
    if lo is None or not isfinite(value):
        return SCORE_MIN, None

    base = abs(lo) if lo != 0 else 1.0
    d = max(0.0, value - lo)
    if d == 0:
        return SCORE_MAX, 0.0

    d_norm = d / base
    raw_score = SCORE_MAX * exp(-lam * d_norm)
    return _clamp_score(raw_score), d_norm


def score_param(
    value: Any,
    param: ParamProfile,
    lam_default: float,
) -> ParamScore:
    """
    מחשב ציון לפרמטר בודד (בלי להשתמש במשקולות של אחרים).

    value: ערך המדד (יכול להיות NaN/None/str/מספר).
    param: הגדרת הפרופיל לפרמטר.
    lam_default: λ ברירת מחדל לפרופיל (ניתן לדריסה ב-param.lambda_override).
    """
    # המרה ל-float בטוחה
    v = _safe_float(value)
    if v is None:
        reason = "missing_or_nan"
        logger.debug("Param %s: value is NaN/None/invalid, score=0", param.name)
        return ParamScore(
            name=param.name,
            label=param.label,
            value=None,
            lo=param.lo,
            hi=param.hi,
            score=SCORE_MIN,
            weight=param.weight,
            effective_weight=0.0,
            target_direction=param.target_direction,
            invert=param.invert,
            d_norm=None,
            reason=reason,
        )

    lam = param.lambda_override if param.lambda_override is not None else lam_default
    lam = max(0.0, float(lam))

    # בדיקת hard caps
    if param.hard_cap_min is not None and v < param.hard_cap_min:
        return ParamScore(
            name=param.name,
            label=param.label,
            value=v,
            lo=param.lo,
            hi=param.hi,
            score=SCORE_MIN,
            weight=param.weight,
            effective_weight=param.weight,
            target_direction=param.target_direction,
            invert=param.invert,
            d_norm=None,
            reason=f"value<{param.hard_cap_min}",
        )

    if param.hard_cap_max is not None and v > param.hard_cap_max:
        return ParamScore(
            name=param.name,
            label=param.label,
            value=v,
            lo=param.lo,
            hi=param.hi,
            score=SCORE_MIN,
            weight=param.weight,
            effective_weight=param.weight,
            target_direction=param.target_direction,
            invert=param.invert,
            d_norm=None,
            reason=f"value>{param.hard_cap_max}",
        )

    # לוגיקת scoring לפי target_direction
    if param.target_direction == TargetDirection.INSIDE:
        score, d_norm = _compute_inside_score(v, param.lo, param.hi, lam)
        reason = "inside_range" if d_norm == 0.0 else "outside_range"
    elif param.target_direction == TargetDirection.ABOVE:
        score, d_norm = _compute_above_score(v, param.hi, lam)
        reason = "above_hi" if d_norm == 0.0 else "below_hi"
    else:  # BELOW
        score, d_norm = _compute_below_score(v, param.lo, lam)
        reason = "below_lo" if d_norm == 0.0 else "above_lo"

    # invert במידת הצורך
    if param.invert:
        score = SCORE_MAX - score
        reason = (reason or "") + "|invert"

    return ParamScore(
        name=param.name,
        label=param.label,
        value=v,
        lo=param.lo,
        hi=param.hi,
        score=score,
        weight=param.weight,
        effective_weight=param.weight,  # equal_weights יטופל בשלב האגרגציה
        target_direction=param.target_direction,
        invert=param.invert,
        d_norm=d_norm,
        reason=reason,
    )


# ========================= Pair-level Scoring Engine =========================


def score_pair(
    metrics: Mapping[str, Any],
    profile: AnalysisProfile,
    ctx: Optional[AnalysisContext] = None,
    *,
    calibrator: Optional[
        Callable[[float, Mapping[str, float], AnalysisContext | None], float]
    ] = None,
    calibrator_name: Optional[str] = None,
) -> PairScoreBreakdown:
    """
    מחשב ציון כולל לזוג לפי metrics ו-profile.

    metrics:
        dict {metric_name -> value}
        לדוגמה: {"half_life": 23.4, "corr": 0.91, "z_score": 2.3, ...}

    profile:
        AnalysisProfile עם רשימת ParamProfile ושאר הגדרות.

    ctx:
        AnalysisContext (sym_x/sym_y, regime, bias וכו').

    calibrator:
        callback אופציונלי שמקבל:
            (postprocessed_score, metrics_numeric, ctx)
        ומחזיר ציון *מכויל* (למשל הסתברות רווח או score מותאם Backtest).
        שימושית ל-calibration ברמת מערכת (לפי Backtest history).

    שלבים:
    1. בוחרים פרופיל לפי Regime (אם יש).
    2. מחשבים ציון לכל פרמטר פעיל.
    3. משקללים לציון גולמי (raw_total_score) לפי weight/equal_weights.
    4. מפעילים Sigmoid mapping אם profile.use_sigmoid_mapping=True.
    5. אם יש calibrator → מחשבים calibrated_score.
    6. total_score = calibrated_score (אם קיים) אחרת postprocessed_score.
    7. מחזירים PairScoreBreakdown מלא.
    """
    # שלב 0: קונטקסט
    regime = ctx.vol_regime if ctx is not None else VolRegime.UNKNOWN
    market_bias = ctx.market_bias if ctx is not None else MarketBias.NEUTRAL
    sym_x = ctx.sym_x if ctx else "UNKNOWN_X"
    sym_y = ctx.sym_y if ctx else "UNKNOWN_Y"

    # שלב 1: פרופיל לפי Regime
    selected_profile = profile.select_for_regime(regime)
    active_params = selected_profile.get_active_params()
    if not active_params:
        logger.warning("score_pair called with profile that has no active params.")
        return PairScoreBreakdown(
            sym_x=sym_x,
            sym_y=sym_y,
            raw_total_score=SCORE_MIN,
            postprocessed_score=SCORE_MIN,
            calibrated_score=None,
            total_score=SCORE_MIN,
            param_scores=[],
            lambda_used=selected_profile.lambda_default,
            equal_weights=selected_profile.equal_weights,
            vol_regime=regime,
            market_bias=market_bias,
            profile_name=selected_profile.meta.name,
            profile_version=selected_profile.meta.version,
            profile_hash=selected_profile.compute_hash(),
            missing_params=[],
            calibrator_name=calibrator_name,
            probability_of_profit=None,
        )

    lam_default = selected_profile.lambda_default
    param_scores: List[ParamScore] = []
    missing: List[str] = []

    # שלב 2: חישוב ציון לכל פרמטר
    numeric_metrics: Dict[str, float] = {}
    for p in active_params:
        raw_value = metrics.get(p.name)
        v = _safe_float(raw_value)
        if v is None:
            missing.append(p.name)
        else:
            numeric_metrics[p.name] = v

        ps = score_param(raw_value, p, lam_default)
        param_scores.append(ps)

    # שלב 3: טיפול במשקולות
    if selected_profile.equal_weights:
        for ps in param_scores:
            ps.effective_weight = 1.0
    else:
        total_w = sum(ps.weight for ps in param_scores if ps.weight > 0)
        if total_w <= 0:
            logger.warning(
                "All param weights are zero; falling back to equal_weights for this pair."
            )
            for ps in param_scores:
                ps.effective_weight = 1.0
        else:
            for ps in param_scores:
                ps.effective_weight = ps.weight

    weights = np.array([ps.effective_weight for ps in param_scores], dtype=float)
    scores = np.array([ps.score for ps in param_scores], dtype=float)

    # אם allow_nan_metrics=False ויש Missing → ציון גולמי 0
    if not selected_profile.allow_nan_metrics and missing:
        raw_total = SCORE_MIN
    else:
        w_sum = float(weights.sum())
        if w_sum <= 0:
            raw_total = float(scores.mean()) if len(scores) > 0 else SCORE_MIN
        else:
            raw_total = float((weights * scores).sum() / w_sum)

    raw_total = _clamp_score(raw_total)

    # שלב 4: Sigmoid mapping (אופציונלי)
    if selected_profile.use_sigmoid_mapping:
        # S in [0,100] → x = a*(S/100 - b) → S' = 100*σ(x)
        a = float(selected_profile.sigmoid_a)
        b = float(selected_profile.sigmoid_b)
        x = a * (raw_total / 100.0 - b)
        mapped = 100.0 * _sigmoid(x)
        postprocessed_score = _clamp_score(mapped)
    else:
        postprocessed_score = raw_total

    # שלב 5: Calibration (אופציונלי)
    calibrated_score: Optional[float] = None
    probability_of_profit: Optional[float] = None

    if calibrator is not None:
        try:
            calibrated_score = float(
                calibrator(postprocessed_score, numeric_metrics, ctx)
            )
            # נניח שה-calibrator מחזיר ציון [0,100] או הסתברות [0,1];
            # אפשר להחליט לפי גודל ולהמיר.
            if calibrated_score <= 1.0:
                # מתייחסים לזה כהסתברות ישירה
                probability_of_profit = max(0.0, min(1.0, calibrated_score))
                calibrated_score = 100.0 * probability_of_profit
            else:
                # מתייחסים לציון [0,100]; אפשר לגזור הסתברות נורמלית אם רוצים
                calibrated_score = _clamp_score(calibrated_score)
                probability_of_profit = calibrated_score / 100.0
        except Exception as exc:
            logger.exception("Calibrator failed for %s/%s: %s", sym_x, sym_y, exc)
            calibrated_score = None
            probability_of_profit = None

    # שלב 6: ציון סופי
    final_score = calibrated_score if calibrated_score is not None else postprocessed_score
    final_score = _clamp_score(final_score)

    return PairScoreBreakdown(
        sym_x=sym_x,
        sym_y=sym_y,
        raw_total_score=raw_total,
        postprocessed_score=postprocessed_score,
        calibrated_score=calibrated_score,
        total_score=final_score,
        param_scores=param_scores,
        lambda_used=lam_default,
        equal_weights=selected_profile.equal_weights,
        vol_regime=regime,
        market_bias=market_bias,
        profile_name=selected_profile.meta.name,
        profile_version=selected_profile.meta.version,
        profile_hash=selected_profile.compute_hash(),
        missing_params=missing,
        calibrator_name=calibrator_name,
        probability_of_profit=probability_of_profit,
    )

# ======================================================================
# Part 2/6 — מנוע מדדים לזוג (Pair Metrics Engine)
# ======================================================================
"""
בחלק הזה אנחנו בונים שכבת מדדים מקצועית לזוג אחד:

- PairMetrics — אובייקט Pydantic שמרכז *את כל* המדדים הרלוונטיים לזוג.
- compute_pair_metrics — מקבל מחירי X/Y ומחזיר PairMetrics עם:
    * מתאמים (קורלציה) לטווחים שונים
    * סטטיסטיקות ספרד (mean/std/z/vol_z, skew/kurtosis)
    * מדדי mean reversion (half-life, Hurst, ADF, KPSS)
    * מדדי סיכון/ביצועים (Sharpe, max drawdown, volatility)
- to_metric_dict — מייצר dict שטוח של מדדים → float לשימוש במנוע הציון (score_pair).

⚙ מאפיינים חשובים:
- שימוש פוטנציאלי ב-helpers קיימים מ-common.utils ו-common.advanced_metrics אם זמינים.
- Fallback בטוח אם חבילות כמו statsmodels לא זמינות.
- קיצוץ זנבות (winsorization) לפי AnalysisProfile.winsorization_pct.
- טיפול רגיש במחסור נתונים — לא זורק חריגות, רק מחזיר None ומעדכן לוג.
"""

from typing import Dict, Any, Optional, Tuple

# ===== ניסיונות יבוא מפונקציות קיימות (אם יש) =====
try:
    from common.utils import (  # type: ignore
        calculate_correlation as _calc_corr,
        calculate_half_life as _calc_half_life,
        calculate_zscore as _calc_zscore,
        calculate_volatility_zscore as _calc_vol_z,
        compute_drawdown as _compute_drawdown,
        max_drawdown as _max_drawdown,
        to_log_returns as _to_log_returns,
    )
except Exception:  # pragma: no cover - Fallback אם המודול לא קיים
    _calc_corr = None
    _calc_half_life = None
    _calc_zscore = None
    _calc_vol_z = None
    _compute_drawdown = None
    _max_drawdown = None
    _to_log_returns = None

try:
    from common.advanced_metrics import hurst_exponent as _hurst_exponent  # type: ignore
except Exception:  # pragma: no cover
    _hurst_exponent = None

try:
    from statsmodels.tsa.stattools import adfuller as _adfuller, kpss as _kpss  # type: ignore
except Exception:  # pragma: no cover
    _adfuller = None
    _kpss = None


class PairMetrics(BaseModel):
    """
    כל המדדים הרלוונטיים לזוג אחד, ברמת קרן גידור.

    המדדים כאן הם "raw features" שמוזנים למנוע הציון (score_pair),
    וכל שדה יכול להיות Optional[float] אם לא ניתן היה לחשב אותו.
    """

    model_config = ConfigDict(extra="ignore")

    sym_x: str
    sym_y: str

    # מידע בסיסי על הדאטה
    n_obs: int
    start_date: Optional[date] = None
    end_date: Optional[date] = None

    # מתאם בין לוג מחירים
    corr: Optional[float] = None
    corr_20d: Optional[float] = None
    corr_60d: Optional[float] = None
    corr_120d: Optional[float] = None

    # ספרד (log_x - log_y)
    spread_mean: Optional[float] = None
    spread_std: Optional[float] = None
    spread_skew: Optional[float] = None
    spread_kurtosis: Optional[float] = None
    spread_z_current: Optional[float] = None
    spread_z_max_60d: Optional[float] = None
    spread_z_min_60d: Optional[float] = None
    spread_vol_z: Optional[float] = None  # Z של תנודתיות הספרד

    # מדדי mean reversion
    half_life: Optional[float] = None
    hurst: Optional[float] = None
    adf_stat: Optional[float] = None
    adf_pvalue: Optional[float] = None
    kpss_stat: Optional[float] = None
    kpss_pvalue: Optional[float] = None

    # מדדי סיכון/ביצועים על הספרד
    sharpe_60d: Optional[float] = None
    max_dd_60d: Optional[float] = None
    vol_60d: Optional[float] = None

    # עתידי: מדדי בטא/ratio וכו'
    beta_60d: Optional[float] = None
    beta_120d: Optional[float] = None
    ratio_vol_60d: Optional[float] = None  # תנודתיות של px/py

    # extra metrics (להרחבות עתידיות)
    extra: Dict[str, Any] = Field(default_factory=dict)

    def to_metric_dict(self) -> Dict[str, float]:
        """
        מחזיר dict שטוח של מדדים לשימוש במנוע הציון.

        השמות כאן צריכים להתאים ל-param.name בפרופיל. דוגמאות:
        - "corr", "corr_20d", "corr_60d", "corr_120d"
        - "spread_mean", "spread_std", "spread_skew", "spread_kurtosis"
        - "spread_z_current", "spread_z_max_60d", "spread_z_min_60d", "spread_vol_z"
        - "half_life", "hurst", "adf_pvalue", "kpss_pvalue"
        - "sharpe_60d", "max_dd_60d", "vol_60d"
        - "beta_60d", "beta_120d", "ratio_vol_60d"
        """
        out: Dict[str, float] = {}

        def _add(name: str, value: Optional[float]) -> None:
            v = _safe_float(value)
            if v is not None:
                out[name] = v

        # מתאמים
        _add("corr", self.corr)
        _add("corr_20d", self.corr_20d)
        _add("corr_60d", self.corr_60d)
        _add("corr_120d", self.corr_120d)

        # ספרד
        _add("spread_mean", self.spread_mean)
        _add("spread_std", self.spread_std)
        _add("spread_skew", self.spread_skew)
        _add("spread_kurtosis", self.spread_kurtosis)
        _add("spread_z_current", self.spread_z_current)
        _add("spread_z_max_60d", self.spread_z_max_60d)
        _add("spread_z_min_60d", self.spread_z_min_60d)
        _add("spread_vol_z", self.spread_vol_z)

        # mean reversion
        _add("half_life", self.half_life)
        _add("hurst", self.hurst)
        _add("adf_stat", self.adf_stat)
        _add("adf_pvalue", self.adf_pvalue)
        _add("kpss_stat", self.kpss_stat)
        _add("kpss_pvalue", self.kpss_pvalue)

        # סיכון/ביצועים
        _add("sharpe_60d", self.sharpe_60d)
        _add("max_dd_60d", self.max_dd_60d)
        _add("vol_60d", self.vol_60d)

        # בטאות/ratio
        _add("beta_60d", self.beta_60d)
        _add("beta_120d", self.beta_120d)
        _add("ratio_vol_60d", self.ratio_vol_60d)

        # extra metrics (אם הוספת בתהליך)
        for k, v in self.extra.items():
            vv = _safe_float(v)
            if vv is not None:
                out[k] = vv

        return out


# ----------------------------------------------------------------------
# Helpers פנימיים לחלק 2
# ----------------------------------------------------------------------


def _align_pair_series(
    px: pd.Series, py: pd.Series
) -> Tuple[pd.Series, pd.Series, int, Optional[date], Optional[date]]:
    """
    יישור מחירים של sym_x/sym_y:
    - הורדת NaN.
    - intersection על האינדקס.
    מחזיר את שני הסדרות + n_obs + תאריכי start/end.
    """
    px = px.dropna().astype(float)
    py = py.dropna().astype(float)
    idx = px.index.intersection(py.index)

    px = px.loc[idx]
    py = py.loc[idx]

    n_obs = int(len(idx))
    if n_obs == 0:
        return px, py, 0, None, None

    start_dt = idx[0].date() if hasattr(idx[0], "date") else None
    end_dt = idx[-1].date() if hasattr(idx[-1], "date") else None
    return px, py, n_obs, start_dt, end_dt


def _winsorize_series(s: pd.Series, pct: Optional[float]) -> pd.Series:
    """קיצוץ זנבות (winsorization) אם pct>0, אחרת מחזיר כמות־שהוא."""
    if pct is None or pct <= 0.0:
        return s
    p_low = pct
    p_high = 1.0 - pct
    try:
        lo = s.quantile(p_low)
        hi = s.quantile(p_high)
        return s.clip(lower=lo, upper=hi)
    except Exception:
        return s


def _compute_corr(x: pd.Series, y: pd.Series) -> Optional[float]:
    if _calc_corr is not None:
        try:
            return float(_calc_corr(x, y))
        except Exception:
            logger.exception("calculate_correlation failed")
    try:
        return float(x.corr(y))
    except Exception:
        return None


def _compute_rolling_corr(
    x: pd.Series, y: pd.Series, window: int
) -> Optional[float]:
    try:
        if len(x) < window or len(y) < window:
            return None
        r = x.rolling(window).corr(y)
        r = r.dropna()
        if r.empty:
            return None
        return float(r.iloc[-1])
    except Exception:
        return None


def _compute_beta(
    x: pd.Series, y: pd.Series, window: int
) -> Optional[float]:
    """
    Beta של X על פני Y בחלון נתון (על log returns).
    """
    try:
        if len(x) < window or len(y) < window:
            return None
        rx = x.pct_change().dropna()
        ry = y.pct_change().dropna()
        idx = rx.index.intersection(ry.index)
        rx = rx.loc[idx].iloc[-window:]
        ry = ry.loc[idx].iloc[-window:]
        if len(rx) < 10:
            return None
        cov = float(np.cov(rx, ry)[0, 1])
        var_y = float(np.var(ry))
        if var_y == 0 or not isfinite(var_y):
            return None
        beta = cov / var_y
        return float(beta)
    except Exception:
        return None


def _compute_spread_z(spread: pd.Series) -> Tuple[Optional[float], Optional[float]]:
    """מחזיר (z_current, std_spread)."""
    s = spread.dropna()
    if s.empty:
        return None, None

    if _calc_zscore is not None:
        try:
            z = _calc_zscore(s)  # type: ignore[arg-type]
            if hasattr(z, "iloc"):
                z_current = float(z.iloc[-1])
            else:
                z_current = float(z)
            return z_current, float(s.std())
        except Exception:
            logger.exception("calculate_zscore failed")

    try:
        mean = float(s.mean())
        std = float(s.std())
        if std <= 0 or not isfinite(std):
            return None, std if isfinite(std) else None
        z_current = float((s.iloc[-1] - mean) / std)
        return z_current, std
    except Exception:
        return None, None


def _compute_spread_z_window(spread: pd.Series, window: int) -> Tuple[Optional[float], Optional[float]]:
    """
    מחזיר (z_max, z_min) בחלון נתון על הספרד.
    שימושי לבדיקת "קצוות" אחרונים.
    """
    s = spread.dropna()
    if len(s) < window:
        return None, None

    tail = s.iloc[-window:]
    mean = float(tail.mean())
    std = float(tail.std())
    if std <= 0 or not isfinite(std):
        return None, None

    z_vals = (tail - mean) / std
    try:
        return float(z_vals.max()), float(z_vals.min())
    except Exception:
        return None, None


def _compute_spread_vol_z(spread: pd.Series) -> Optional[float]:
    if _calc_vol_z is not None:
        try:
            return float(_calc_vol_z(spread))
        except Exception:
            logger.exception("calculate_volatility_zscore failed")
    # fallback: z-score של rolling std על פני חלון קצר
    try:
        s = spread.dropna()
        if len(s) < 40:
            return None
        vol = s.rolling(20).std().dropna()
        if vol.empty:
            return None
        mean = float(vol.mean())
        std = float(vol.std())
        if std <= 0 or not isfinite(std):
            return None
        return float((vol.iloc[-1] - mean) / std)
    except Exception:
        return None


def _compute_half_life(spread: pd.Series) -> Optional[float]:
    if _calc_half_life is not None:
        try:
            return float(_calc_half_life(spread))
        except Exception:
            logger.exception("calculate_half_life failed")

    # Fallback: AR(1) רגרסיה פשוטה
    try:
        y = spread.dropna().values
        if len(y) < 20:
            return None
        y_lag = y[:-1]
        y_curr = y[1:]
        x = np.vstack([y_lag, np.ones_like(y_lag)]).T
        beta, alpha = np.linalg.lstsq(x, y_curr, rcond=None)[0]
        if beta >= 1:
            return None
        hl = -np.log(2) / np.log(beta) if beta > 0 else None
        return float(hl) if hl is not None and isfinite(hl) and hl > 0 else None
    except Exception:
        logger.exception("fallback half-life estimation failed")
        return None


def _compute_hurst(spread: pd.Series) -> Optional[float]:
    if _hurst_exponent is not None:
        try:
            return float(_hurst_exponent(spread))
        except Exception:
            logger.exception("hurst_exponent failed")
    # fallback מאוד גס (לא חובה)
    return None


def _compute_adf(spread: pd.Series) -> Tuple[Optional[float], Optional[float]]:
    if _adfuller is None:
        return None, None
    try:
        s = spread.dropna()
        if len(s) < 40:
            return None, None
        res = _adfuller(s, maxlag=1, autolag="AIC")
        stat, pval = res[0], res[1]
        return float(stat), float(pval)
    except Exception:
        logger.exception("adfuller failed")
        return None, None


def _compute_kpss(spread: pd.Series) -> Tuple[Optional[float], Optional[float]]:
    if _kpss is None:
        return None, None
    try:
        s = spread.dropna()
        if len(s) < 40:
            return None, None
        stat, pval, _, _ = _kpss(s, regression="c", nlags="auto")
        return float(stat), float(pval)
    except Exception:
        logger.exception("kpss failed")
        return None, None


def _compute_sharpe_60d(return_series: pd.Series) -> Tuple[Optional[float], Optional[float]]:
    """Sharpe על 60 ימים אחרונים + volatility."""
    try:
        r = return_series.dropna()
        if len(r) < 20:
            return None, None
        r_tail = r.iloc[-60:] if len(r) > 60 else r
        mu = float(r_tail.mean())
        sigma = float(r_tail.std())
        if sigma <= 0 or not isfinite(sigma):
            return None, sigma if isfinite(sigma) else None
        sharpe_daily = mu / sigma
        sharpe_annual = sharpe_daily * np.sqrt(252.0)
        return float(sharpe_annual), sigma
    except Exception:
        logger.exception("sharpe_60d computation failed")
        return None, None


def _compute_max_dd_60d(equity: pd.Series) -> Optional[float]:
    """Max Drawdown על פני 60 נקודות אחרונות של עקומת equity של הספרד."""
    s = equity.dropna()
    if s.empty:
        return None
    s_tail = s.iloc[-60:] if len(s) > 60 else s

    if _compute_drawdown is not None and _max_drawdown is not None:
        try:
            dd = _compute_drawdown(s_tail)
            mdd = _max_drawdown(dd)
            return float(mdd)
        except Exception:
            logger.exception("compute_drawdown/max_drawdown failed")

    # fallback: חישוב drawdown ידני
    try:
        running_max = s_tail.cummax()
        dd = (s_tail - running_max) / running_max
        return float(dd.min())
    except Exception:
        return None


# ----------------------------------------------------------------------
# פונקציית הליבה לחלק 2 — חישוב מדדים לזוג
# ----------------------------------------------------------------------


def compute_pair_metrics(
    sym_x: str,
    sym_y: str,
    prices_x: pd.Series,
    prices_y: pd.Series,
    profile: Optional[AnalysisProfile] = None,
) -> PairMetrics:
    """
    חישוב מדדים לזוג (sym_x, sym_y) מתוך מחירי closing של שני הסימבולים.

    prices_x / prices_y:
        pandas.Series עם אינדקס תאריכים וערכי מחיר.
        הפונקציה תבצע:
        - יישור אינדקסים (intersection).
        - חישוב log-prices.
        - ספרד: log(px) - log(py).
        - חישובי corr, half-life, z-score, ADF, KPSS, Hurst, Sharpe, Max DD ועוד.

    profile.winsorization_pct (אם קיים):
        יחול על ספרד לפני חישובי half-life/Hurst/ADF/KPSS כדי להקטין השפעת outliers.
    """
    # 1) יישור נתונים
    px, py, n_obs, start_dt, end_dt = _align_pair_series(prices_x, prices_y)
    if n_obs == 0:
        logger.warning("compute_pair_metrics: no overlapping data for %s/%s", sym_x, sym_y)
        return PairMetrics(
            sym_x=sym_x,
            sym_y=sym_y,
            n_obs=0,
            start_date=None,
            end_date=None,
        )

    # 2) לוג מחירים, ספרד ו-ratio
    log_x = np.log(px)
    log_y = np.log(py)
    spread = log_x - log_y
    ratio = px / py

    # winsorization אם הפרופיל דורש (על הספרד)
    winsor_pct = profile.winsorization_pct if profile is not None else None
    spread_for_stats = _winsorize_series(spread, winsor_pct)

    # 3) מתאמים
    corr = _compute_corr(log_x, log_y)
    corr_20d = _compute_rolling_corr(log_x, log_y, 20)
    corr_60d = _compute_rolling_corr(log_x, log_y, 60)
    corr_120d = _compute_rolling_corr(log_x, log_y, 120)

    # 4) סטטיסטיקות ספרד + Z + Vol_Z + skew/kurtosis
    s = spread_for_stats.dropna()
    if len(s) > 0:
        spread_mean = float(s.mean())
        spread_std = float(s.std())
        try:
            spread_skew = float(s.skew())
            spread_kurtosis = float(s.kurtosis())
        except Exception:
            spread_skew = None
            spread_kurtosis = None
    else:
        spread_mean = spread_std = spread_skew = spread_kurtosis = None

    spread_z_current, _ = _compute_spread_z(spread_for_stats)
    spread_vol_z = _compute_spread_vol_z(spread_for_stats)
    spread_z_max_60d, spread_z_min_60d = _compute_spread_z_window(spread_for_stats, 60)

    # 5) mean reversion: half-life, Hurst, ADF, KPSS
    half_life = _compute_half_life(spread_for_stats)
    hurst = _compute_hurst(spread_for_stats)
    adf_stat, adf_pvalue = _compute_adf(spread_for_stats)
    kpss_stat, kpss_pvalue = _compute_kpss(spread_for_stats)

    # 6) returns, Sharpe, Max DD
    if _to_log_returns is not None:
        try:
            spread_returns = _to_log_returns(spread)  # type: ignore[arg-type]
        except Exception:
            logger.exception("to_log_returns failed; falling back to diff")
            spread_returns = spread.diff()
    else:
        spread_returns = spread.diff()

    sharpe_60d, vol_60d = _compute_sharpe_60d(spread_returns)

    # Equity curve של הספרד: cumulative sum of returns (פשוט)
    try:
        equity = spread_returns.fillna(0).cumsum()
    except Exception:
        equity = spread_returns.fillna(0)

    max_dd_60d = _compute_max_dd_60d(equity)

    # 7) בטא על פני מחירי close (על returns)
    beta_60d = _compute_beta(px, py, 60)
    beta_120d = _compute_beta(px, py, 120)

    # 8) volatility של ratio (px/py)
    try:
        ratio_ret = ratio.pct_change().dropna()
        ratio_tail = ratio_ret.iloc[-60:] if len(ratio_ret) > 60 else ratio_ret
        ratio_vol_60d = float(ratio_tail.std()) if len(ratio_tail) > 0 else None
    except Exception:
        ratio_vol_60d = None

    return PairMetrics(
        sym_x=sym_x,
        sym_y=sym_y,
        n_obs=n_obs,
        start_date=start_dt,
        end_date=end_dt,
        corr=corr,
        corr_20d=corr_20d,
        corr_60d=corr_60d,
        corr_120d=corr_120d,
        spread_mean=spread_mean,
        spread_std=spread_std,
        spread_skew=spread_skew,
        spread_kurtosis=spread_kurtosis,
        spread_z_current=spread_z_current,
        spread_z_max_60d=spread_z_max_60d,
        spread_z_min_60d=spread_z_min_60d,
        spread_vol_z=spread_vol_z,
        half_life=half_life,
        hurst=hurst,
        adf_stat=adf_stat,
        adf_pvalue=adf_pvalue,
        kpss_stat=kpss_stat,
        kpss_pvalue=kpss_pvalue,
        sharpe_60d=sharpe_60d,
        max_dd_60d=max_dd_60d,
        vol_60d=vol_60d,
        beta_60d=beta_60d,
        beta_120d=beta_120d,
        ratio_vol_60d=ratio_vol_60d,
        extra={},
    )
# ======================================================================
# Part 3/6 — Smart Scan Orchestration & Universe Loading (HF-grade v2)
# ======================================================================
"""
בחלק הזה אנחנו בונים שכבת "הדבק החכם" בין כל הליבה:

- יקום זוגות (pair.json / config.json / מקור חיצוני אחר)
- טעינת מחירים (price_loader)
- שכבת המדדים (compute_pair_metrics)
- מנוע הדירוג (score_pair)

עם שדרוגים מאסיביים:

1. מודלים עשירים ליקום:
   - UniversePair עם helpers לסקטור, asset_class, tags, scoring מראש.
   - UniverseMetadata עם universe_id, pair_count, timestamps.

2. SmartScanConfig מורחב:
   - ספי איכות: min_obs, min_abs_corr, half-life range, p-values וכו'.
   - פילטרים לפי tags, asset_class, sector.
   - אפשרויות ביצועים: max_pairs, batch_size, max_workers (קריאה עתידית ל-parallel).
   - פרמטרים לטעינת דאטה: price_period, bar_size.

3. SmartScanResult משודרג:
   - מיון, Top-N, סטטיסטיקות גלובליות.
   - סטטיסטיקות לפי סקטור/asset_class/tag.
   - המרה ל-DataFrame (ל-Export / Insights).
   - Hooks ל-export ל-CSV/Parquet (אם תרצה להשתמש).

4. smart_scan_pairs:
   - לולאת סריקה חזקה, עם:
       * פילטרים על meta עוד לפני טעינת דאטה.
       * טיפול מפורט בשגיאות price_loader/metrics/scoring.
       * callback להתקדמות + callback אחרי scoring של זוג (on_pair_scored).
       * הכנה להאצה עתידית (parallel) דרך max_workers/batch_size.
"""

from pathlib import Path
import json
import uuid
from datetime import datetime as _dt
from typing import Callable, Sequence, List  # כבר יובאו חלקית בחלק 1, זה בסדר ב-Python


# ========================= Universe Models =========================


class UniversePair(BaseModel):
    """
    ייצוג אחיד של זוג ביקום.

    sym_x / sym_y:
        שמות הסימבולים (tickers).
    meta:
        מידע נוסף (אופציונלי): sector, asset_class, score, tags, מקור, וכו'.
        לדוגמה:
            {
              "sector_x": "Technology",
              "sector_y": "Technology",
              "asset_class": "ETF",
              "score": 0.83,
              "tags": ["etf", "us", "large_cap"]
            }
    """

    model_config = ConfigDict(extra="ignore")

    sym_x: str
    sym_y: str
    meta: Dict[str, Any] = Field(default_factory=dict)

    @property
    def label(self) -> str:
        return f"{self.sym_x} / {self.sym_y}"

    @property
    def tags(self) -> List[str]:
        t = self.meta.get("tags") or []
        if isinstance(t, str):
            return [t]
        return [str(x) for x in t]

    @property
    def sector_x(self) -> Optional[str]:
        return self.meta.get("sector_x") or self.meta.get("sector") or None

    @property
    def sector_y(self) -> Optional[str]:
        return self.meta.get("sector_y") or None

    @property
    def asset_class(self) -> Optional[str]:
        return self.meta.get("asset_class") or None

    @property
    def pre_score(self) -> Optional[float]:
        """ציון "ראשוני" מתוך הקונפיג (אם יש), לפני Smart Scan."""
        v = self.meta.get("score")
        return _safe_float(v)


class UniverseMetadata(BaseModel):
    """
    מטא-דאטה עבור יקום זוגות:
    - universe_id: מזהה ייחודי.
    - name: שם לוגי (למשל "pairs_from_config").
    - source: מקור (path, DB, API).
    - created_at: זמן טעינה.
    - version/author/note: מידע חופשי.
    - pair_count: מספר זוגות ביקום.
    """

    model_config = ConfigDict(extra="ignore")

    universe_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "default_universe"
    source: Optional[str] = None
    created_at: _dt = Field(default_factory=_dt.utcnow)
    version: Optional[str] = None
    author: Optional[str] = None
    note: Optional[str] = None
    pair_count: int = 0


class SmartScanConfig(BaseModel):
    """
    קונפיגורציה לסריקת יקום (Smart Scan):

    quality thresholds:
    -------------------
    min_obs:
        מינימום נקודות זמן (תצפיות) כדי שזוג ייחשב תקין.
    min_abs_corr:
        מינימום |corr| גלובלי על לוג-מחירים (לפני שאר המדדים).
    max_half_life:
        ערך מקסימלי ל-half-life (ימים) כדי לפסול זוגות איטיים מדי.
    max_hurst:
        ערך מקסימלי ל-Hurst (מעליו התנהגות לא מספיק mean-reverting).
    max_adf_pvalue / max_kpss_pvalue:
        ספים ל-stationarity.

    filtering:
    ----------
    max_pairs:
        הגבלת מספר זוגות לסריקה (לניסויים/בדיקות).
    top_n:
        ברירת מחדל לכמה זוגות מוצגים כ-Top N.
    include_tags / exclude_tags:
        סינון לפי tags ב-meta.
    include_asset_classes / exclude_asset_classes:
        סינון לפי asset_class.
    include_sectors / exclude_sectors:
        סינון לפי סקטור.

    data loading:
    -------------
    price_period:
        לדוגמה "6mo", "1y", "3y" — רק לידע לחיצוני (הפונקציה price_loader לא חייבת להשתמש).
    bar_size:
        "1d", "1h" וכו'.

    execution:
    ----------
    allow_failures:
        אם False — כישלון בטעינת מחירים/metrics/score יפסיק את הסריקה.
    batch_size:
        גודל אצווה ל-UI / logging (למשל: הדפסת סטטוס כל 10 זוגות).
    max_workers:
        לעתיד (אם נרצה להשתמש ב-thread pool). כרגע לא מפעיל parallel בפועל.
    """

    model_config = ConfigDict(extra="ignore")

    # quality thresholds
    min_obs: int = 60
    min_abs_corr: Optional[float] = 0.3
    max_half_life: Optional[float] = 200.0
    max_hurst: Optional[float] = 0.80
    max_adf_pvalue: Optional[float] = 0.10
    max_kpss_pvalue: Optional[float] = 0.10

    # filtering
    max_pairs: Optional[int] = None
    top_n: int = 50

    include_tags: Optional[List[str]] = None
    exclude_tags: Optional[List[str]] = None
    include_asset_classes: Optional[List[str]] = None
    exclude_asset_classes: Optional[List[str]] = None
    include_sectors: Optional[List[str]] = None
    exclude_sectors: Optional[List[str]] = None

    # data loading hints (אינפורמטיבי, לא מחייב את price_loader)
    price_period: Optional[str] = None
    bar_size: str = "1d"

    # execution behavior
    allow_failures: bool = True
    batch_size: int = 10
    max_workers: Optional[int] = None  # ל-parallel עתידי

    def _match_tags(self, tags: List[str]) -> bool:
        tag_set = set(tags)
        if self.include_tags:
            if not tag_set.intersection(self.include_tags):
                return False
        if self.exclude_tags:
            if tag_set.intersection(self.exclude_tags):
                return False
        return True

    def _match_asset_class(self, asset_class: Optional[str]) -> bool:
        if asset_class is None:
            return True
        ac = str(asset_class).lower()
        if self.include_asset_classes:
            if ac not in [x.lower() for x in self.include_asset_classes]:
                return False
        if self.exclude_asset_classes:
            if ac in [x.lower() for x in self.exclude_asset_classes]:
                return False
        return True

    def _match_sector(self, sector_x: Optional[str], sector_y: Optional[str]) -> bool:
        # מחליטים על "sector" אחד מועדף (למשל של sym_x) לצורך פילטר
        sector = sector_x or sector_y
        if sector is None:
            return True
        sc = str(sector).lower()
        if self.include_sectors:
            if sc not in [x.lower() for x in self.include_sectors]:
                return False
        if self.exclude_sectors:
            if sc in [x.lower() for x in self.exclude_sectors]:
                return False
        return True

    def should_keep_pair(self, pair: UniversePair) -> bool:
        """
        פילטר בסיסי לפי meta (tags, asset_class, sector)
        עוד לפני שנוגעים בדאטה.
        """
        if not self._match_tags(pair.tags):
            return False
        if not self._match_asset_class(pair.asset_class):
            return False
        if not self._match_sector(pair.sector_x, pair.sector_y):
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        """ייצוג נוח ל-Export / לוגים."""
        return self.model_dump(mode="json")


class SmartScanError(BaseModel):
    """תיאור כישלון עבור זוג במהלך הסריקה."""

    sym_x: str
    sym_y: str
    reason: str


class SmartScanResult(BaseModel):
    """
    תוצאות סריקה:

    - scan_id: מזהה סריקה.
    - created_at: מתי הסריקה בוצעה.
    - universe_meta: מטא-דאטה על היקום.
    - config: SmartScanConfig ששימש.
    - scores: רשימת PairScoreBreakdown עבור זוגות שעברו את הפילטרים.
    - errors: רשימת זוגות שנכשלו (טעינת דאטה / metrics / scoring).
    - stats: סטטיסטיקות Aggregate גלובליות.
    - group_stats: סטטיסטיקות לפי grouping (סקטור/asset_class/tag).
    """

    model_config = ConfigDict(extra="ignore")

    scan_id: str
    created_at: _dt
    universe_meta: UniverseMetadata
    config: SmartScanConfig

    scores: List[PairScoreBreakdown] = Field(default_factory=list)
    errors: List[SmartScanError] = Field(default_factory=list)

    stats: Dict[str, Any] = Field(default_factory=dict)
    group_stats: Dict[str, Any] = Field(default_factory=dict)

    def sort_by_score(self, descending: bool = True) -> None:
        """מיין את הציונים לפי total_score."""
        self.scores.sort(key=lambda s: s.total_score, reverse=descending)

    def top_n_pairs(self, n: Optional[int] = None) -> List[PairScoreBreakdown]:
        n = n or self.config.top_n
        return self.scores[:n]

    # ===== סטטיסטיקות גלובליות =====

    def compute_stats(self) -> None:
        """מחשב סטטיסטיקות בסיסיות על הציונים הכוללים."""
        if not self.scores:
            self.stats = {"count": 0}
            return

        scores = np.array([s.total_score for s in self.scores], dtype=float)
        self.stats = {
            "count": int(len(scores)),
            "mean_score": float(scores.mean()),
            "median_score": float(np.median(scores)),
            "min_score": float(scores.min()),
            "max_score": float(scores.max()),
            "p25": float(np.percentile(scores, 25)),
            "p75": float(np.percentile(scores, 75)),
        }

    # ===== סטטיסטיקות לפי קבוצה (sector/asset_class/tag) =====

    def _compute_group_stats_for(
        self,
        pairs: Sequence[UniversePair],
        key_fn: Callable[[UniversePair], Optional[str]],
        group_name: str,
    ) -> Dict[str, Any]:
        """
        מחשב סטטיסטיקות ממוצע ציון עבור grouping מסוים (לדוגמה לפי סקטור/asset_class).
        pairs:
            רשימת UniversePair תואמת (באותו סדר של scores).
        key_fn:
            פונקציה שמקבלת UniversePair ומחזירה מפתח קבוצה (למשל sector).
        group_name:
            שם הגרופ (למשל "sector", "asset_class").
        """
        group_map: Dict[str, List[float]] = {}
        for pair, score in zip(pairs, self.scores):
            key = key_fn(pair)
            if not key:
                continue
            k = str(key)
            group_map.setdefault(k, []).append(score.total_score)

        stats: Dict[str, Any] = {}
        for k, vals in group_map.items():
            arr = np.array(vals, dtype=float)
            if len(arr) == 0:
                continue
            stats[k] = {
                "count": int(len(arr)),
                "mean_score": float(arr.mean()),
                "median_score": float(np.median(arr)),
                "min_score": float(arr.min()),
                "max_score": float(arr.max()),
            }

        return {
            "group_by": group_name,
            "groups": stats,
        }

    def compute_group_stats(
        self,
        universe_pairs: Sequence[UniversePair],
    ) -> None:
        """
        מחשב group_stats עבור:

        - לפי asset_class
        - לפי sector (של sym_x)
        - לפי tag (tag אחד לפחות בזוג)

        universe_pairs:
            אותה רשימת UniversePair שבה השתמשנו בסריקה (באותו סדר).
        """
        if not self.scores:
            self.group_stats = {}
            return

        # מיפוי sym_x/sym_y → UniversePair כדי להתאים ל-scores
        pair_map: Dict[Tuple[str, str], UniversePair] = {
            (p.sym_x, p.sym_y): p for p in universe_pairs
        }
        aligned_pairs: List[UniversePair] = []
        for s in self.scores:
            key = (s.sym_x, s.sym_y)
            if key in pair_map:
                aligned_pairs.append(pair_map[key])

        if not aligned_pairs:
            self.group_stats = {}
            return

        # לפי asset_class
        by_asset = self._compute_group_stats_for(
            aligned_pairs,
            key_fn=lambda p: p.asset_class,
            group_name="asset_class",
        )

        # לפי sector (ניקח sector_x כברירת מחדל)
        by_sector = self._compute_group_stats_for(
            aligned_pairs,
            key_fn=lambda p: p.sector_x,
            group_name="sector",
        )

        # לפי tag: מרכיבים קבוצה לכל tag אפשרי
        tag_map: Dict[str, List[float]] = {}
        for pair, score in zip(aligned_pairs, self.scores):
            for tag in pair.tags:
                t = str(tag)
                tag_map.setdefault(t, []).append(score.total_score)

        by_tag: Dict[str, Any] = {}
        for t, vals in tag_map.items():
            arr = np.array(vals, dtype=float)
            by_tag[t] = {
                "count": int(len(arr)),
                "mean_score": float(arr.mean()),
                "median_score": float(np.median(arr)),
                "min_score": float(arr.min()),
                "max_score": float(arr.max()),
            }

        self.group_stats = {
            "by_asset_class": by_asset,
            "by_sector": by_sector,
            "by_tag": by_tag,
        }

    # ===== המרה ל-DataFrame / יצוא =====

    def to_dataframe(self) -> pd.DataFrame:
        """
        ממיר את scores ל-DataFrame שטוח, נוח לניתוח/Export:

        עמודות לדוגמה:
        - sym_x, sym_y, total_score, raw_total_score, postprocessed_score, probability_of_profit, ...
        - פרמטרים: ניצור עמודה לכל param: score_{param_name}
        """
        if not self.scores:
            return pd.DataFrame()

        rows: List[Dict[str, Any]] = []
        for s in self.scores:
            row: Dict[str, Any] = {
                "sym_x": s.sym_x,
                "sym_y": s.sym_y,
                "total_score": s.total_score,
                "raw_total_score": s.raw_total_score,
                "postprocessed_score": s.postprocessed_score,
                "calibrated_score": s.calibrated_score,
                "probability_of_profit": s.probability_of_profit,
                "vol_regime": s.vol_regime.value,
                "market_bias": s.market_bias.value,
                "profile_name": s.profile_name,
                "profile_version": s.profile_version,
            }
            for ps in s.param_scores:
                key = f"score_{ps.name}"
                row[key] = ps.score
            rows.append(row)

        return pd.DataFrame(rows)

    def export_csv(self, path: Path) -> None:
        """שומר את תוצאות הסריקה כ-CSV."""
        df = self.to_dataframe()
        df.to_csv(path, index=False)

    def export_parquet(self, path: Path) -> None:
        """שומר את תוצאות הסריקה כ-Parquet (לשימוש ב-DuckDB/פנדס עתידי)."""
        df = self.to_dataframe()
        df.to_parquet(path, index=False)


# ========================= Universe Loading Helpers =========================


def _load_json(path: Path) -> Any:
    """טעינת JSON בטוחה עם לוגים."""
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.error("Failed to load JSON from %s: %s", path, exc)
        raise


def load_universe_from_pair_json(
    path: Path,
    universe_name: str = "pair_json_universe",
) -> Tuple[List[UniversePair], UniverseMetadata]:
    """
    טוען יקום זוגות מקובץ pair.json בפורמט:

    [
      { "symbols": ["XLY", "XLC"] },
      { "symbols": ["XLY", "XLP"], "score": 0.9 },
      ...
    ]

    תומך גם בגרסאות:
      { "SYMBOLS": [...] }
      { "sym_x": "XLY", "sym_y": "XLC" }

    meta נוסף (score, tags וכו') נשמר בתוך UniversePair.meta.
    """
    data = _load_json(path)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list of pairs")

    pairs: List[UniversePair] = []

    for raw in data:
        if not isinstance(raw, dict):
            continue

        symbols = raw.get("symbols") or raw.get("SYMBOLS")
        sym_x = raw.get("sym_x")
        sym_y = raw.get("sym_y")

        if symbols is not None and isinstance(symbols, (list, tuple)) and len(symbols) == 2:
            sx, sy = str(symbols[0]).strip(), str(symbols[1]).strip()
        elif sym_x and sym_y:
            sx, sy = str(sym_x).strip(), str(sym_y).strip()
        else:
            # מבנה לא מזוהה → מדלגים
            continue

        meta = {k: v for k, v in raw.items() if k not in ("symbols", "SYMBOLS", "sym_x", "sym_y")}
        pairs.append(UniversePair(sym_x=sx, sym_y=sy, meta=meta))

    meta = UniverseMetadata(
        name=universe_name,
        source=str(path),
        version=None,
        note="Loaded from pair.json",
        pair_count=len(pairs),
    )
    return pairs, meta


def load_universe_from_config_json(
    path: Path,
    universe_name: str = "config_pairs_universe",
) -> Tuple[List[UniversePair], UniverseMetadata]:
    """
    טוען יקום זוגות מקובץ config.json בפורמט:

    {
      "pairs": [
        { "symbols": ["XLY", "XLC"], "score": 0.812, ... },
        ...
      ],
      "metadata": { ... },
      ...
    }

    meta של כל זוג כולל את score ושדות נוספים שנמצאים באובייקט הזוג.
    metadata גלובלי נשמר ב-UniverseMetadata.
    """
    data = _load_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")

    raw_pairs = data.get("pairs") or []
    if not isinstance(raw_pairs, list):
        raise ValueError(f"{path}['pairs'] must be a list")

    pairs: List[UniversePair] = []

    for raw in raw_pairs:
        if not isinstance(raw, dict):
            continue

        symbols = raw.get("symbols") or raw.get("SYMBOLS")
        sym_x = raw.get("sym_x")
        sym_y = raw.get("sym_y")

        if symbols is not None and isinstance(symbols, (list, tuple)) and len(symbols) == 2:
            sx, sy = str(symbols[0]).strip(), str(symbols[1]).strip()
        elif sym_x and sym_y:
            sx, sy = str(sym_x).strip(), str(sym_y).strip()
        else:
            continue

        meta = {k: v for k, v in raw.items() if k not in ("symbols", "SYMBOLS", "sym_x", "sym_y")}
        pairs.append(UniversePair(sym_x=sx, sym_y=sy, meta=meta))

    raw_meta = data.get("metadata") or {}
    meta = UniverseMetadata(
        name=universe_name,
        source=str(path),
        version=str(raw_meta.get("version") or raw_meta.get("Version") or ""),
        author=raw_meta.get("author"),
        note=str(raw_meta.get("note") or ""),
        pair_count=len(pairs),
    )
    return pairs, meta


# ========================= Smart Scan Core =========================


def smart_scan_pairs(
    pairs: Sequence[UniversePair],
    profile: AnalysisProfile,
    price_loader: Callable[[str, Optional[str], str], pd.Series],
    ctx_base: Optional[AnalysisContext] = None,
    config: Optional[SmartScanConfig] = None,
    on_progress: Optional[Callable[[int, int, UniversePair], None]] = None,
    on_pair_scored: Optional[
        Callable[[PairScoreBreakdown, PairMetrics, UniversePair, int, int], None]
    ] = None,
) -> SmartScanResult:
    """
    מנוע Smart Scan כללי ליקום נתון:

    pairs:
        רשימה של UniversePair.
    profile:
        AnalysisProfile לשימוש במנוע הציון.
    price_loader(symbol, period, bar_size) -> pd.Series:
        פונקציה שמחזירה סדרת מחירים (Close) לסימבול.
        לדוגמה:
            def loader(sym, period, bar_size):
                df = load_price_data(sym, period=period or "1y", bar_size=bar_size)
                return df["close"]

    ctx_base:
        קונטקסט בסיסי (עם vol_regime, market_bias, run_id וכו') שיועתק לכל זוג.
    config:
        SmartScanConfig עם מינימום תצפיות, max_pairs, ספים ועוד.
    on_progress(index, total, pair):
        callback התקדמות (אפשר להשתמש ל-logging/ProgressBar ב-UI).
    on_pair_scored(breakdown, metrics_obj, pair, index, total):
        callback שמופעל אחרי שלזוג מסוים כבר יש scores + metrics.

    מחזיר:
        SmartScanResult עם רשימת PairScoreBreakdown + סטטיסטיקות.
    """
    cfg = config or SmartScanConfig()
    scan_id = str(uuid.uuid4())
    created_at = _dt.utcnow()

    universe_meta = UniverseMetadata(
        name=(ctx_base.universe_id if ctx_base and ctx_base.universe_id else "universe"),
        source=None,
        created_at=created_at,
        version=None,
        author=None,
        note="Smart scan run",
        pair_count=len(pairs),
    )

    result = SmartScanResult(
        scan_id=scan_id,
        created_at=created_at,
        universe_meta=universe_meta,
        config=cfg,
        scores=[],
        errors=[],
        stats={},
        group_stats={},
    )

    all_pairs = list(pairs)
    if cfg.max_pairs is not None and cfg.max_pairs > 0:
        all_pairs = all_pairs[: cfg.max_pairs]

    total = len(all_pairs)
    if total == 0:
        logger.warning("smart_scan_pairs: no pairs to scan.")
        result.compute_stats()
        return result

    for idx, pair in enumerate(all_pairs, start=1):
        # callback התקדמות
        if on_progress is not None:
            try:
                on_progress(idx, total, pair)
            except Exception:
                logger.exception("on_progress callback failed")

        # פילטר לפי tags/asset_class/sector לפני דאטה
        if not cfg.should_keep_pair(pair):
            continue

        # טעינת מחירים
        try:
            px = price_loader(pair.sym_x, cfg.price_period, cfg.bar_size)
            py = price_loader(pair.sym_y, cfg.price_period, cfg.bar_size)
        except Exception as exc:
            msg = f"price_loader failed: {exc}"
            logger.exception("price_loader failed for %s/%s", pair.sym_x, pair.sym_y)
            result.errors.append(
                SmartScanError(sym_x=pair.sym_x, sym_y=pair.sym_y, reason=msg)
            )
            if not cfg.allow_failures:
                break
            continue

        # חישוב metrics
        try:
            metrics_obj = compute_pair_metrics(
                pair.sym_x,
                pair.sym_y,
                px,
                py,
                profile=profile,
            )
        except Exception as exc:
            msg = f"compute_pair_metrics failed: {exc}"
            logger.exception("compute_pair_metrics failed for %s/%s", pair.sym_x, pair.sym_y)
            result.errors.append(
                SmartScanError(sym_x=pair.sym_x, sym_y=pair.sym_y, reason=msg)
            )
            if not cfg.allow_failures:
                break
            continue

        # סף מינימום תצפיות
        if metrics_obj.n_obs < cfg.min_obs:
            result.errors.append(
                SmartScanError(
                    sym_x=pair.sym_x,
                    sym_y=pair.sym_y,
                    reason=f"n_obs<{cfg.min_obs}",
                )
            )
            continue

        metrics_dict = metrics_obj.to_metric_dict()

        # סף מתאם גלובלי
        if cfg.min_abs_corr is not None:
            corr_val = _safe_float(metrics_dict.get("corr"))
            if corr_val is None or abs(corr_val) < cfg.min_abs_corr:
                result.errors.append(
                    SmartScanError(
                        sym_x=pair.sym_x,
                        sym_y=pair.sym_y,
                        reason=f"|corr|<{cfg.min_abs_corr}",
                    )
                )
                continue

        # סף half-life
        if cfg.max_half_life is not None:
            hl_val = _safe_float(metrics_dict.get("half_life"))
            if hl_val is not None and hl_val > cfg.max_half_life:
                result.errors.append(
                    SmartScanError(
                        sym_x=pair.sym_x,
                        sym_y=pair.sym_y,
                        reason=f"half_life>{cfg.max_half_life}",
                    )
                )
                continue

        # סף Hurst
        if cfg.max_hurst is not None:
            hurst_val = _safe_float(metrics_dict.get("hurst"))
            if hurst_val is not None and hurst_val > cfg.max_hurst:
                result.errors.append(
                    SmartScanError(
                        sym_x=pair.sym_x,
                        sym_y=pair.sym_y,
                        reason=f"hurst>{cfg.max_hurst}",
                    )
                )
                continue

        # ספי p-values (ADF/KPSS)
        if cfg.max_adf_pvalue is not None:
            adf_p = _safe_float(metrics_dict.get("adf_pvalue"))
            if adf_p is not None and adf_p > cfg.max_adf_pvalue:
                result.errors.append(
                    SmartScanError(
                        sym_x=pair.sym_x,
                        sym_y=pair.sym_y,
                        reason=f"adf_pvalue>{cfg.max_adf_pvalue}",
                    )
                )
                continue

        if cfg.max_kpss_pvalue is not None:
            kpss_p = _safe_float(metrics_dict.get("kpss_pvalue"))
            if kpss_p is not None and kpss_p > cfg.max_kpss_pvalue:
                result.errors.append(
                    SmartScanError(
                        sym_x=pair.sym_x,
                        sym_y=pair.sym_y,
                        reason=f"kpss_pvalue>{cfg.max_kpss_pvalue}",
                    )
                )
                continue

        # ctx לכל זוג
        ctx = AnalysisContext(
            sym_x=pair.sym_x,
            sym_y=pair.sym_y,
            start_date=metrics_obj.start_date,
            end_date=metrics_obj.end_date,
            vol_regime=(ctx_base.vol_regime if ctx_base else VolRegime.UNKNOWN),
            market_bias=(ctx_base.market_bias if ctx_base else MarketBias.NEUTRAL),
            run_id=(ctx_base.run_id if ctx_base else None),
            scan_id=scan_id,
            universe_id=(ctx_base.universe_id if ctx_base else universe_meta.universe_id),
            extra={"pair_meta": pair.meta},
        )

        # ציונים לזוג
        try:
            breakdown = score_pair(
                metrics=metrics_dict,
                profile=profile,
                ctx=ctx,
            )
        except Exception as exc:
            msg = f"score_pair failed: {exc}"
            logger.exception("score_pair failed for %s/%s", pair.sym_x, pair.sym_y)
            result.errors.append(
                SmartScanError(sym_x=pair.sym_x, sym_y=pair.sym_y, reason=msg)
            )
            if not cfg.allow_failures:
                break
            continue

        result.scores.append(breakdown)

        # callback אחרי scoring לזוג
        if on_pair_scored is not None:
            try:
                on_pair_scored(breakdown, metrics_obj, pair, idx, total)
            except Exception:
                logger.exception("on_pair_scored callback failed")

        # לוג התקדמות כל batch_size
        if cfg.batch_size > 0 and idx % cfg.batch_size == 0:
            logger.info(
                "SmartScan %s: processed %d/%d pairs (current best=%.2f)",
                scan_id,
                idx,
                total,
                result.scores[0].total_score if result.scores else 0.0,
            )

    # אחרי הסריקה: מיון וסטטיסטיקות
    result.sort_by_score(descending=True)
    result.compute_stats()
    result.compute_group_stats(all_pairs)

    return result
# ======================================================================
# Part 4/6 — Streamlit UI לטאב "ניתוח זוג / Smart Scan"
# ======================================================================
"""
בחלק הזה אנחנו בונים את שכבת ה-UI (Streamlit) לטאב ניתוח זוג:

- טענת יקום (pair.json / config.json) מתוך ה-UI.
- שליטה בפרופיל (λ, equal_weights, בחירת רג'ים).
- הרצת Smart Scan על כל ה-Universe (באמצעות smart_scan_pairs).
- הצגת Top-N זוגות: טבלה, KPIs, סינון.
- Drill-Down לזוג נבחר:
    * גרף מחירים לשני הנכסים.
    * גרף ספרד + Z-score.
    * Breakdown של ציוני הפרמטרים.
- חיבור נקי ל-Classים מהחלקים הקודמים:
    * AnalysisProfile, AnalysisContext
    * UniversePair, SmartScanConfig, SmartScanResult
    * compute_pair_metrics, score_pair, smart_scan_pairs
"""

# נסה לייבא Streamlit ו-Plotly — אם לא זמינים, נשמור fallback
try:
    import streamlit as st
except Exception:  # pragma: no cover
    st = None  # type: ignore

try:
    import plotly.graph_objects as go
    import plotly.express as px
except Exception:  # pragma: no cover
    go = None  # type: ignore
    px = None  # type: ignore


def _ensure_streamlit_available() -> None:
    """זריקת שגיאה ברורה אם מנסים לרנדר UI בלי Streamlit."""
    if st is None:
        raise RuntimeError("Streamlit לא זמין בסביבה — לא ניתן להציג את טאב הניתוח.")


# ----------------------------------------------------------------------
# Helpers ל-UI
# ----------------------------------------------------------------------


def _ui_edit_profile(profile: AnalysisProfile) -> AnalysisProfile:
    """
    Editor קטן לפרופיל בתוך ה-UI:
    - שליטה ב-lambda_default, equal_weights, use_sigmoid_mapping.
    - עריכת משקולות / enable לפרמטרים בסיסית (כרגע תצוגת read-only / toggle חלקי).
    החזרת עותק מעודכן של הפרופיל (מקורי לא משתנה).
    """
    _ensure_streamlit_available()

    # נעבוד על copy כדי לא לשנות את האובייקט המקורי בטעות
    p = profile.model_copy(deep=True)

    st.subheader("⚙️ הגדרות דירוג גלובליות", anchor=False)
    col1, col2, col3 = st.columns(3)
    with col1:
        p.lambda_default = st.number_input(
            "Lambda (λ) — חדות דעיכה",
            min_value=0.0,
            max_value=10.0,
            value=float(p.lambda_default),
            step=0.1,
            help="ערך גבוה → עונש חזק יותר על יציאה מהטווח.",
            key="analysis_lambda_default",
        )
    with col2:
        p.equal_weights = st.checkbox(
            "משקולות שוות לכל הפרמטרים",
            value=p.equal_weights,
            help="אם מסומן, מתעלמים מהמשקולות בפראמטרים וכל הפרמטרים נספרים במידה שווה.",
            key="analysis_equal_weights",
        )
    with col3:
        p.use_sigmoid_mapping = st.checkbox(
            "מיפוי Sigmoid לציון הכולל",
            value=p.use_sigmoid_mapping,
            help="מחליק את הציון הכולל כך שקצוות יהיו פחות קיצוניים.",
            key="analysis_use_sigmoid",
        )

    if p.use_sigmoid_mapping:
        col4, col5 = st.columns(2)
        with col4:
            p.sigmoid_a = st.number_input(
                "Sigmoid a (חדות)",
                min_value=0.1,
                max_value=20.0,
                value=float(p.sigmoid_a),
                step=0.5,
                key="analysis_sigmoid_a",
            )
        with col5:
            p.sigmoid_b = st.number_input(
                "Sigmoid b (נקודת גזירה)",
                min_value=0.0,
                max_value=1.0,
                value=float(p.sigmoid_b),
                step=0.05,
                key="analysis_sigmoid_b",
            )

    st.markdown("### 🎚️ פרמטרים פעילים בפרופיל", unsafe_allow_html=True)
    # טבלה קומפקטית לקריאה + אפשרות enable/disable בסיסית
    rows = []
    for i, param in enumerate(p.params):
        rows.append(
            {
                "name": param.name,
                "label": param.label,
                "group": param.group or "",
                "target": param.target_direction.value,
                "lo": param.lo,
                "hi": param.hi,
                "weight": param.weight,
                "enabled": param.enabled,
            }
        )
    df_params = pd.DataFrame(rows)
    edited_df = st.data_editor(
        df_params,
        width="stretch",
        hide_index=True,
        key="analysis_profile_editor",
    )

    # עדכון enabled ו-weight מהעריכה (שאר השדות נשאיר ל-static editing בקובץ)
    name_to_row = {row["name"]: row for _, row in edited_df.iterrows()}
    for param in p.params:
        row = name_to_row.get(param.name)
        if row is None:
            continue
        param.enabled = bool(row["enabled"])
        w = _safe_float(row["weight"])
        if w is not None and w >= 0:
            param.weight = w

    return p


def _ui_show_kpi_cards(scan_result: SmartScanResult) -> None:
    """מציג כרטיסי KPI בסיסיים מתוך SmartScanResult."""
    _ensure_streamlit_available()
    stats = scan_result.stats or {}
    n_pairs = stats.get("count", 0)
    mean_score = stats.get("mean_score", 0.0)
    median_score = stats.get("median_score", 0.0)
    max_score = stats.get("max_score", 0.0)
    min_score = stats.get("min_score", 0.0)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("🔢 מספר זוגות", f"{n_pairs}")
    with c2:
        st.metric("📊 ציון ממוצע", f"{mean_score:0.1f}")
    with c3:
        st.metric("📈 ציון חציון", f"{median_score:0.1f}")
    with c4:
        st.metric("🏆 ציון מקסימלי", f"{max_score:0.1f}")
    with c5:
        st.metric("⚠️ ציון מינימלי", f"{min_score:0.1f}")


def _ui_show_group_stats(scan_result: SmartScanResult) -> None:
    """מציג סטטיסטיקות קבוצתיות (סקטור/asset_class/tag) אם קיימות."""
    _ensure_streamlit_available()
    if not scan_result.group_stats:
        return

    st.markdown("### 🧩 סטטיסטיקות לפי קבוצות", unsafe_allow_html=True)
    tabs = st.tabs(["Asset Class", "Sector", "Tags"])

    # Asset Class
    with tabs[0]:
        by_asset = scan_result.group_stats.get("by_asset_class", {})
        groups = by_asset.get("groups", {})
        if not groups:
            st.write("אין נתונים לפי Asset Class.")
        else:
            rows = []
            for k, v in groups.items():
                rows.append({"asset_class": k, **v})
            st.dataframe(pd.DataFrame(rows), width = "stretch")

    # Sector
    with tabs[1]:
        by_sector = scan_result.group_stats.get("by_sector", {})
        groups = by_sector.get("groups", {})
        if not groups:
            st.write("אין נתונים לפי Sector.")
        else:
            rows = []
            for k, v in groups.items():
                rows.append({"sector": k, **v})
            st.dataframe(pd.DataFrame(rows), width = "stretch")

    # Tags
    with tabs[2]:
        by_tag = scan_result.group_stats.get("by_tag", {})
        if not by_tag:
            st.write("אין נתונים לפי Tags.")
        else:
            rows = []
            for k, v in by_tag.items():
                rows.append({"tag": k, **v})
            st.dataframe(pd.DataFrame(rows), width = "stretch")


def _ui_plot_prices_and_spread(
    sym_x: str,
    sym_y: str,
    prices_x: pd.Series,
    prices_y: pd.Series,
    metrics: PairMetrics,
) -> None:
    """גרפי מחירים + ספרד + Z-score."""
    _ensure_streamlit_available()
    if go is None:
        st.warning("Plotly לא זמין — לא ניתן להציג גרפים.")
        return

    px_ = prices_x.dropna()
    py_ = prices_y.dropna()
    idx = px_.index.intersection(py_.index)
    px_ = px_.loc[idx]
    py_ = py_.loc[idx]
    if len(idx) == 0:
        st.warning("אין חפיפה בזמן בין שני הסימבולים.")
        return

    # גרף מחירים
    fig_prices = go.Figure()
    fig_prices.add_trace(
        go.Scatter(
            x=idx,
            y=px_,
            name=sym_x,
            mode="lines",
        )
    )
    fig_prices.add_trace(
        go.Scatter(
            x=idx,
            y=py_,
            name=sym_y,
            mode="lines",
        )
    )
    fig_prices.update_layout(
        title=f"מחירי {sym_x} ו-{sym_y}",
        xaxis_title="תאריך",
        yaxis_title="מחיר",
        legend_title="סימבול",
        height=350,
    )

    # גרף ספרד + Z-score
    log_x = np.log(px_)
    log_y = np.log(py_)
    spread = log_x - log_y
    spread = spread.dropna()

    if spread.empty:
        st.plotly_chart(fig_prices, width = "stretch")
        st.info("לא ניתן לחשב ספרד לזוג זה.")
        return

    # z-score "גלם" (לגרף) – לא חישוב מדויק כמו metrics, אבל מספיק ויזואלית
    s = spread
    mu = float(s.mean())
    std = float(s.std()) if len(s) > 1 else None
    if std and std > 0:
        z_series = (s - mu) / std
    else:
        z_series = pd.Series(index=s.index, data=0.0)

    fig_spread = go.Figure()
    fig_spread.add_trace(
        go.Scatter(
            x=s.index,
            y=s,
            name="Spread (logX - logY)",
            mode="lines",
        )
    )
    fig_spread.add_hline(y=mu, line_dash="dot", line_color="gray")
    if std and std > 0:
        fig_spread.add_hline(y=mu + std, line_dash="dot", line_color="orange")
        fig_spread.add_hline(y=mu - std, line_dash="dot", line_color="orange")

    fig_spread.update_layout(
        title="ספרד + רמות סטיית תקן",
        xaxis_title="תאריך",
        yaxis_title="Spread",
        height=300,
    )

    fig_z = go.Figure()
    fig_z.add_trace(
        go.Scatter(
            x=z_series.index,
            y=z_series,
            name="Z-Score (approx)",
            mode="lines",
        )
    )
    fig_z.add_hline(y=0.0, line_dash="dot", line_color="gray")
    fig_z.add_hline(y=2.0, line_dash="dot", line_color="red")
    fig_z.add_hline(y=-2.0, line_dash="dot", line_color="red")
    fig_z.update_layout(
        title="Z-Score של הספרד (קירוב)",
        xaxis_title="תאריך",
        yaxis_title="Z",
        height=300,
    )

    st.plotly_chart(fig_prices, width = "stretch")
    st.plotly_chart(fig_spread, width = "stretch")
    st.plotly_chart(fig_z, width = "stretch")


def _ui_show_param_breakdown(breakdown: PairScoreBreakdown) -> None:
    """מציג טבלת Breakdown של ציוני הפרמטרים לזוג נבחר."""
    _ensure_streamlit_available()
    rows = []
    for ps in breakdown.param_scores:
        rows.append(
            {
                "פרמטר": ps.label,
                "שם לוגי": ps.name,
                "ציון": round(ps.score, 2),
                "טווח רצוי [lo, hi]": f"[{ps.lo}, {ps.hi}]",
                "כיוון יעד": ps.target_direction.value,
                "משקל": ps.weight,
                "מרחק מנורמל": ps.d_norm,
                "הערה": ps.reason or "",
            }
        )
    df = pd.DataFrame(rows)
    st.dataframe(df, width = "stretch")

def render_tab_comparison_lab(
    breakdown: "PairScoreBreakdown",
    metrics_obj: "PairMetrics",
    macro_metrics: Optional[Mapping[str, Any]] = None,
    fund_metrics: Optional[Mapping[str, Any]] = None,
    risk_metrics: Optional[Mapping[str, Any]] = None,
) -> None:
    """
    Tab Comparison Lab מחובר ל-Analysis:
    ------------------------------------
    • Stats Tab נבנה אוטומטית מתוך PairMetrics + הציון הכולל של הזוג.
    • Macro / Fundamentals / Risk מוזנים מהמאקרו ומהטאב הפנדומנטלי.
    • Composite Tab — שילוב Stats+Macro+Fund לפי משקלים שאתה בוחר.

    המטריקות המרכזיות:
        - sharpe_60d / max_dd_60d / vol_60d
        - macro_sensitivity
        - valuation_score
        - pair_score
        - risk_exposure / risk_inclusion_ratio / risk_composite
    """
    _ensure_streamlit_available()
    import streamlit as st
    import numpy as np
    import pandas as pd

    st.markdown("### 🔀 Tab Comparison Lab — Stats vs Macro vs Fundamentals + Risk")

    # ---- שליפת מדדי Stats אמיתיים מהזוג ----
    sharpe_60d = float(metrics_obj.sharpe_60d or 0.0)
    max_dd_60d = float(metrics_obj.max_dd_60d or 0.0)
    vol_60d = float(metrics_obj.vol_60d or 0.0)

    if sharpe_60d == 0.0 and metrics_obj.sharpe_60d is None:
        sharpe_60d = np.nan
    if max_dd_60d == 0.0 and metrics_obj.max_dd_60d is None:
        max_dd_60d = np.nan
    if vol_60d == 0.0 and metrics_obj.vol_60d is None:
        vol_60d = np.nan

    pair_score = float(breakdown.total_score)

    # ---- helper קטן לשליפת ערך ממילונים שונים ----
    def _get_metric(
        src: Optional[Mapping[str, Any]],
        *keys: str,
        default: float | None = None,
    ) -> Optional[float]:
        if not src:
            return default
        for k in keys:
            if k in src:
                v = _safe_float(src.get(k))
                if v is not None:
                    return v
        return default

    # ---- ברירות מחדל למאקרו/פנדומנטלי (אם הועברו מבחוץ) ----
    macro_sens_default = _get_metric(
        macro_metrics,
        "macro_sensitivity",
        "macro_regime_score",
        default=0.6,
    )
    macro_dd_default = _get_metric(
        macro_metrics,
        "max_dd_60d",
        "macro_max_dd",
        default=max_dd_60d if np.isfinite(max_dd_60d) else None,
    )
    macro_vol_default = _get_metric(
        macro_metrics,
        "vol_60d",
        "macro_vol_60d",
        default=vol_60d if np.isfinite(vol_60d) else None,
    )
    macro_score_default = _get_metric(
        macro_metrics,
        "macro_score",
        "tab_score",
        default=pair_score * 0.8,
    )

    val_score_default = _get_metric(
        fund_metrics,
        "valuation_score",
        "fundamental_score",
        "value_score",
        default=0.8,
    )
    fund_sharpe_default = _get_metric(
        fund_metrics,
        "sharpe_60d",
        "fund_sharpe_60d",
        default=sharpe_60d * 0.9 if np.isfinite(sharpe_60d) else None,
    )
    fund_dd_default = _get_metric(
        fund_metrics,
        "max_dd_60d",
        "fund_max_dd_60d",
        default=max_dd_60d * 0.9 if np.isfinite(max_dd_60d) else None,
    )
    fund_score_default = _get_metric(
        fund_metrics,
        "fund_score",
        "tab_score",
        default=pair_score * 0.9,
    )

    # ---- ברירות מחדל ל-Risk Tab (אם יש risk_metrics) ----
    risk_exposure_default = _get_metric(
        risk_metrics,
        "risk_exposure",
        "exposure_multiplier",
        default=1.0,
    )
    risk_incl_default = _get_metric(
        risk_metrics,
        "risk_inclusion_ratio",
        default=1.0,
    )
    risk_score_default = _get_metric(
        risk_metrics,
        "risk_score",
        default=pair_score,
    )

    # ---- Normalization & Similarity ----
    with st.expander("⚙️ הגדרות Normalization & Similarity", expanded=False):
        col_norm, col_sim = st.columns(2)
        with col_norm:
            normalization = st.selectbox(
                "שיטת נירמול:",
                options=["zscore", "minmax", "robust"],
                index=0,
                key="tabcmp_norm",
            )
        with col_sim:
            similarity_method = st.selectbox(
                "שיטת דמיון:",
                options=["cosine", "corr", "euclidean"],
                index=0,
                key="tabcmp_sim",
            )

    # ---- משקלים לקומפוזיט (Stats / Macro / Fund) ----
    st.markdown("#### ⚖️ משקלי קומפוזיט (Stats / Macro / Fundamentals)")
    c_w1, c_w2, c_w3 = st.columns(3)
    with c_w1:
        w_stats_raw = st.number_input(
            "משקל Stats",
            min_value=0.0,
            max_value=10.0,
            value=1.0,
            step=0.1,
            key="tabcmp_w_stats",
        )
    with c_w2:
        w_macro_raw = st.number_input(
            "משקל Macro",
            min_value=0.0,
            max_value=10.0,
            value=0.7,
            step=0.1,
            key="tabcmp_w_macro",
        )
    with c_w3:
        w_fund_raw = st.number_input(
            "משקל Fundamentals",
            min_value=0.0,
            max_value=10.0,
            value=0.9,
            step=0.1,
            key="tabcmp_w_fund",
        )

    w_sum = w_stats_raw + w_macro_raw + w_fund_raw
    if w_sum <= 0:
        w_stats, w_macro, w_fund = 0.4, 0.3, 0.3
    else:
        w_stats = w_stats_raw / w_sum
        w_macro = w_macro_raw / w_sum
        w_fund = w_fund_raw / w_sum

    # ---- תצוגת Stats אמיתיים ----
    st.markdown("#### 📊 Stats אמיתיים של הזוג")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Sharpe 60d", f"{sharpe_60d:0.2f}" if np.isfinite(sharpe_60d) else "N/A")
    with c2:
        st.metric("Max DD 60d", f"{max_dd_60d:0.2%}" if np.isfinite(max_dd_60d) else "N/A")
    with c3:
        st.metric("Vol 60d", f"{vol_60d:0.2%}" if np.isfinite(vol_60d) else "N/A")
    with c4:
        st.metric("Pair Score", f"{pair_score:0.1f}")

    st.markdown("#### 🧩 התאמת טאב־פרופילים (אפשר לכוונן ידנית)")

    # ---------- Stats / Macro / Fund UI ----------
    col_stats, col_macro, col_fund = st.columns(3)

    with col_stats:
        st.markdown("##### 📊 Stats Tab")
        ui_sharpe_60d = st.number_input(
            "Sharpe 60d (Stats):",
            value=float(0.0 if np.isnan(sharpe_60d) else sharpe_60d),
            step=0.1,
            key="tabcmp_stats_sharpe_60d",
        )
        ui_max_dd_60d = st.number_input(
            "Max DD 60d (Stats, שלילי):",
            value=float(0.0 if np.isnan(max_dd_60d) else max_dd_60d),
            step=0.01,
            key="tabcmp_stats_maxdd_60d",
        )
        ui_vol_60d = st.number_input(
            "Vol 60d (Stats):",
            value=float(0.0 if np.isnan(vol_60d) else vol_60d),
            step=0.01,
            key="tabcmp_stats_vol_60d",
        )
        ui_pair_score_stats = st.number_input(
            "Pair Score (Stats):",
            value=float(pair_score),
            step=0.1,
            key="tabcmp_stats_pair_score",
        )

    with col_macro:
        st.markdown("##### 🌍 Macro Tab")
        ui_macro_sens = st.number_input(
            "Macro Sensitivity (0..1):",
            value=float(macro_sens_default if macro_sens_default is not None else 0.6),
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            key="tabcmp_macro_sens",
        )
        ui_macro_dd = st.number_input(
            "Max DD (Macro proxy):",
            value=float(
                0.0
                if (macro_dd_default is None or np.isnan(macro_dd_default))
                else macro_dd_default
            ),
            step=0.01,
            key="tabcmp_macro_maxdd",
        )
        ui_macro_vol = st.number_input(
            "Vol 60d (Macro proxy):",
            value=float(
                0.0
                if (macro_vol_default is None or np.isnan(macro_vol_default))
                else macro_vol_default
            ),
            step=0.01,
            key="tabcmp_macro_vol",
        )
        ui_pair_score_macro = st.number_input(
            "Macro Score:",
            value=float(macro_score_default if macro_score_default is not None else pair_score * 0.8),
            step=0.1,
            key="tabcmp_macro_score",
        )

    with col_fund:
        st.markdown("##### 💼 Fundamentals Tab")
        ui_val_score = st.number_input(
            "Valuation Score (0..1):",
            value=float(val_score_default if val_score_default is not None else 0.8),
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            key="tabcmp_fund_val_score",
        )
        ui_fund_sharpe = st.number_input(
            "Sharpe 60d (Fund proxy):",
            value=float(
                0.0
                if (fund_sharpe_default is None or np.isnan(fund_sharpe_default))
                else fund_sharpe_default
            ),
            step=0.1,
            key="tabcmp_fund_sharpe_60d",
        )
        ui_fund_dd = st.number_input(
            "Max DD (Fund proxy):",
            value=float(
                0.0
                if (fund_dd_default is None or np.isnan(fund_dd_default))
                else fund_dd_default
            ),
            step=0.01,
            key="tabcmp_fund_maxdd",
        )
        ui_pair_score_fund = st.number_input(
            "Fundamentals Score:",
            value=float(fund_score_default if fund_score_default is not None else pair_score * 0.9),
            step=0.1,
            key="tabcmp_fund_score",
        )

    # ---------- MetricMeta + Profiles ----------
    metric_meta = {
        "sharpe_60d": MetricMeta(
            name="sharpe_60d",
            direction="higher_better",
            group="risk_adj",
            weight=1.2,
            description="Sharpe ratio (60d) על הספרד",
        ),
        "max_dd_60d": MetricMeta(
            name="max_dd_60d",
            direction="lower_better",
            group="drawdown",
            weight=1.5,
            description="Max Drawdown (60d) על הספרד",
        ),
        "vol_60d": MetricMeta(
            name="vol_60d",
            direction="lower_better",
            group="risk_adj",
            weight=1.0,
            description="Volatility (60d) של הספרד",
        ),
        "macro_sensitivity": MetricMeta(
            name="macro_sensitivity",
            direction="neutral",
            group="macro",
            weight=1.0,
            description="רגישות למאקרו (0..1)",
        ),
        "valuation_score": MetricMeta(
            name="valuation_score",
            direction="higher_better",
            group="fundamental",
            weight=1.3,
            description="ציון פנדומנטלי / Valuation (0..1)",
        ),
        "pair_score": MetricMeta(
            name="pair_score",
            direction="higher_better",
            group="composite",
            weight=1.0,
            description="ציון קומפוזיטי של הזוג/טאב",
        ),
        # Risk
        "risk_exposure": MetricMeta(
            name="risk_exposure",
            direction="lower_better",
            group="risk",
            weight=1.0,
            description="חשיפת תיק גלובלית (exposure_multiplier)",
        ),
        "risk_inclusion_ratio": MetricMeta(
            name="risk_inclusion_ratio",
            direction="higher_better",
            group="risk",
            weight=1.0,
            description="שיעור זוגות כלולים מתוך היקום (0..1)",
        ),
        "risk_composite": MetricMeta(
            name="risk_composite",
            direction="higher_better",
            group="risk",
            weight=1.2,
            description="ציון סיכון/חשיפה קומפוזיטי",
        ),
    }

    metric_keys = list(metric_meta.keys())

    profiles: list[TabProfile] = [
        TabProfile(
            tab_id="stats_tab",
            tab_type="stats",
            label="Stats (Backtest / Matrix)",
            metrics={
                "sharpe_60d": ui_sharpe_60d,
                "max_dd_60d": ui_max_dd_60d,
                "vol_60d": ui_vol_60d,
                "macro_sensitivity": 0.0,
                "valuation_score": 0.0,
                "pair_score": ui_pair_score_stats,
                "risk_exposure": np.nan,
                "risk_inclusion_ratio": np.nan,
                "risk_composite": np.nan,
            },
            weight=1.2,
            tags=["stats", "backtest"],
        ),
        TabProfile(
            tab_id="macro_tab",
            tab_type="macro",
            label="Macro Engine / Regime",
            metrics={
                "sharpe_60d": ui_macro_sens,  # proxy פשוט
                "max_dd_60d": ui_macro_dd,
                "vol_60d": ui_macro_vol,
                "macro_sensitivity": ui_macro_sens,
                "valuation_score": 0.0,
                "pair_score": ui_pair_score_macro,
                "risk_exposure": np.nan,
                "risk_inclusion_ratio": np.nan,
                "risk_composite": np.nan,
            },
            weight=1.0,
            tags=["macro", "regime"],
        ),
        TabProfile(
            tab_id="fund_tab",
            tab_type="fundamental",
            label="Index Fundamentals / Valuation",
            metrics={
                "sharpe_60d": ui_fund_sharpe,
                "max_dd_60d": ui_fund_dd,
                "vol_60d": ui_vol_60d,
                "macro_sensitivity": ui_macro_sens * 0.5,
                "valuation_score": ui_val_score,
                "pair_score": ui_pair_score_fund,
                "risk_exposure": np.nan,
                "risk_inclusion_ratio": np.nan,
                "risk_composite": np.nan,
            },
            weight=0.9,
            tags=["fundamental", "index"],
        ),
    ]

    # ---------- Composite Tab (Stats + Macro + Fundamentals) ----------
    try:
        composite = build_composite_profile(
            composite_id="composite_tab",
            label="Composite (Stats + Macro + Fund)",
            profiles=profiles[:3],
            tab_type="composite",
            weights={
                "stats_tab": w_stats,
                "macro_tab": w_macro,
                "fund_tab": w_fund,
            },
        )
        profiles.append(composite)
    except Exception as exc:
        st.warning(f"לא הצלחנו לבנות Composite Tab (לא קריטי): {exc!r}")

    # ---------- Risk Tab (Risk / Constraints) ----------
    try:
        profiles.append(
            TabProfile(
                tab_id="risk_tab",
                tab_type="risk",
                label="Risk / Constraints",
                metrics={
                    "sharpe_60d": np.nan,
                    "max_dd_60d": np.nan,
                    "vol_60d": np.nan,
                    "macro_sensitivity": 0.0,
                    "valuation_score": 0.0,
                    "pair_score": risk_score_default if risk_score_default is not None else pair_score,
                    "risk_exposure": risk_exposure_default if risk_exposure_default is not None else 1.0,
                    "risk_inclusion_ratio": risk_incl_default if risk_incl_default is not None else 1.0,
                    "risk_composite": risk_score_default if risk_score_default is not None else pair_score,
                },
                weight=1.0,
                tags=["risk", "constraints"],
            )
        )
    except Exception as exc:
        st.warning(f"לא הצלחנו לבנות Risk Tab (לא קריטי): {exc!r}")

    cfg = TabComparisonConfig(
        normalization=normalization,
        similarity_method=similarity_method,
        distance_metric="euclidean",
        metric_meta=metric_meta,
        group_weights={
            "risk_adj": 1.0,
            "drawdown": 1.2,
            "macro": 0.8,
            "fundamental": 1.0,
            "composite": 1.0,
            "risk": 1.0,
        },
    )

    st.markdown("#### 📦 חישוב Bundle של השוואת טאבים")

    if st.button("🚀 הרץ Tab Comparison לזוג הזה", key="tabcmp_run_from_analysis"):
        bundle = build_comparison_bundle(profiles, cfg=cfg, metric_keys=metric_keys)

        sim_df = bundle["similarity"]
        dist_df = bundle["distance"]
        kpi_df = bundle["metric_vs_tab"]
        ranks_df = bundle["ranks"]

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("##### 🔁 Similarity Matrix (Tab vs Tab)")
            st.dataframe(sim_df, width = "stretch")
        with col_b:
            st.markdown("##### 📏 Distance Matrix (Tab vs Tab)")
            st.dataframe(dist_df, width = "stretch")

        st.markdown("##### 📊 Metric vs Tab (normalized)")
        st.dataframe(kpi_df, width = "stretch")

        st.markdown("##### 🏆 Ranks per Metric (1 = הכי טוב)")
        st.dataframe(ranks_df, width = "stretch")

        # Alignment מול Stats Tab
        try:
            alignment = compute_alignment_scores(sim_df, benchmark_tab_id="stats_tab")
            st.markdown("##### 🎯 Alignment vs Stats Tab")
            st.dataframe(alignment.to_frame("alignment_score"), width = "stretch")
        except KeyError:
            st.warning("לא נמצא 'stats_tab' ב-Similarity matrix — בדוק את ה-tab_id-ים.")

        # Alignment מול Composite Tab (אם קיים)
        if "composite_tab" in sim_df.index:
            try:
                align_comp = compute_alignment_scores(sim_df, benchmark_tab_id="composite_tab")
                st.markdown("##### 🎯 Alignment vs Composite Tab")
                st.dataframe(align_comp.to_frame("alignment_vs_composite"), width = "stretch")
            except Exception as exc:
                st.warning(f"Alignment מול Composite נכשל (לא קריטי): {exc!r}")

        # Alignment מול Risk Tab (אם קיים)
        if "risk_tab" in sim_df.index:
            try:
                align_risk = compute_alignment_scores(sim_df, benchmark_tab_id="risk_tab")
                st.markdown("##### 🎯 Alignment vs Risk Tab")
                st.dataframe(align_risk.to_frame("alignment_vs_risk"), width = "stretch")
            except Exception as exc:
                st.warning(f"Alignment מול Risk נכשל (לא קריטי): {exc!r}")

        # Anomalies
        anomalies = detect_tab_anomalies(dist_df)
        if not anomalies.empty:
            st.markdown("##### 🚨 Tab Anomalies (אאוטליירים במרחב הטאבים)")
            st.dataframe(anomalies, width = "stretch")

        # Explainability: Stats vs Macro
        try:
            contrib = explain_similarity_contributions(
                profiles,
                cfg=cfg,
                metric_keys=metric_keys,
                tab_pair=("stats_tab", "macro_tab"),
            )
            st.markdown("##### 🧠 Top metric contributions: Stats vs Macro")
            st.dataframe(contrib.head(10), width = "stretch")
        except Exception as exc:  # pragma: no cover
            st.warning(f"לא הצלחנו לחשב תרומות דמיון: {exc!r}")
    else:
        st.info("התאם את הערכים בטאבים ולחץ על הכפתור כדי לראות מטריצות השוואה.")

# ----------------------------------------------------------------------
# פונקציית ה-UI הראשית לטאב ניתוח זוג
# ----------------------------------------------------------------------


def render_analysis_tab(
    profile: AnalysisProfile,
    price_loader: Callable[[str, Optional[str], str], pd.Series],
    *,
    default_pair_json: Optional[Path] = None,
    default_config_json: Optional[Path] = None,
    ctx_base: Optional[AnalysisContext] = None,
) -> None:
    """
    פונקציית Render לטאב "ניתוח זוג / Smart Scan".

    profile:
        AnalysisProfile בסיסי (ניתן לעריכה בתוך ה-UI דרך _ui_edit_profile).
    price_loader(symbol, period, bar_size):
        פונקציית טעינת מחירים.

        דוגמה ל-Integration עם MarketDataRouter:
        ----------------------------------------------------
        def price_loader(sym: str, period: Optional[str], bar_size: str) -> pd.Series:
            df = router.get_history([sym], period=period or "1y", bar_size=bar_size)
            return df["close"]
        ----------------------------------------------------

    default_pair_json:
        קובץ pair.json ברירת מחדל אם המשתמש לא בוחר קובץ.
    default_config_json:
        קובץ config.json ברירת מחדל אם המשתמש לא בוחר קובץ.
    ctx_base:
        AnalysisContext בסיסי (run_id, vol_regime, market_bias וכו').
    """
    _ensure_streamlit_available()

    st.title("🔍 ניתוח זוג ברמת קרן גידור (Smart Scan)")

    # ======================= Sidebar — בחירת Universe =======================

    st.sidebar.header("🌌 יקום זוגות", anchor=False)

    universe_source = st.sidebar.selectbox(
        "מקור יקום:",
        options=["pair.json", "config.json", "Custom (pair.json)", "Custom (config.json)"],
        index=0,
    )

    pair_json_path: Optional[Path] = None
    config_json_path: Optional[Path] = None

    if universe_source == "pair.json":
        pair_json_path = default_pair_json or (Path.cwd() / "pair.json")
    elif universe_source == "config.json":
        config_json_path = default_config_json or (Path.cwd() / "config.json")
    elif universe_source == "Custom (pair.json)":
        uploaded = st.sidebar.file_uploader("בחר pair.json", type=["json"], key="pair_json_uploader")
        if uploaded is not None:
            pair_json_path = Path("_uploaded_pair.json")
            with pair_json_path.open("wb") as f:
                f.write(uploaded.getbuffer())
    else:  # Custom (config.json)
        uploaded = st.sidebar.file_uploader("בחר config.json", type=["json"], key="config_json_uploader")
        if uploaded is not None:
            config_json_path = Path("_uploaded_config.json")
            with config_json_path.open("wb") as f:
                f.write(uploaded.getbuffer())

    # טעינת היקום
    universe_pairs: List[UniversePair] = []
    universe_meta: Optional[UniverseMetadata] = None

    try:
        if pair_json_path and pair_json_path.exists():
            universe_pairs, universe_meta = load_universe_from_pair_json(pair_json_path)
        elif config_json_path and config_json_path.exists():
            universe_pairs, universe_meta = load_universe_from_config_json(config_json_path)
    except Exception as exc:
        st.sidebar.error(f"שגיאה בטעינת היקום: {exc}")

    if not universe_pairs:
        st.warning("לא נטען שום יקום זוגות (pair.json/config.json).")
        return

    st.sidebar.write(f"✅ נטענו {len(universe_pairs)} זוגות מהיקום.")

    # ======================= Sidebar — Smart Scan Config ====================

    st.sidebar.header("⚙️ הגדרות Smart Scan", anchor=False)

    cfg = SmartScanConfig()
    cfg.min_obs = st.sidebar.number_input(
        "מינימום תצפיות (days)",
        min_value=20,
        max_value=2000,
        value=cfg.min_obs,
        step=10,
    )
    cfg.min_abs_corr = st.sidebar.number_input(
        "מינימום |corr|",
        min_value=0.0,
        max_value=1.0,
        value=float(cfg.min_abs_corr or 0.3),
        step=0.05,
    )
    cfg.max_half_life = st.sidebar.number_input(
        "מקסימום Half-Life (ימים)",
        min_value=1.0,
        max_value=1000.0,
        value=float(cfg.max_half_life or 200.0),
        step=5.0,
    )
    cfg.max_hurst = st.sidebar.number_input(
        "מקסימום Hurst",
        min_value=0.5,
        max_value=1.0,
        value=float(cfg.max_hurst or 0.8),
        step=0.01,
    )
    cfg.max_adf_pvalue = st.sidebar.number_input(
        "מקסימום ADF p-value",
        min_value=0.0,
        max_value=0.5,
        value=float(cfg.max_adf_pvalue or 0.1),
        step=0.01,
    )
    cfg.max_kpss_pvalue = st.sidebar.number_input(
        "מקסימום KPSS p-value",
        min_value=0.0,
        max_value=0.5,
        value=float(cfg.max_kpss_pvalue or 0.1),
        step=0.01,
    )

    cfg.max_pairs = st.sidebar.number_input(
        "מספר זוגות מקסימלי לסריקה",
        min_value=1,
        max_value=len(universe_pairs),
        value=min(len(universe_pairs), 200),
        step=10,
    )
    cfg.top_n = st.sidebar.number_input(
        "Top N להצגה",
        min_value=5,
        max_value=200,
        value=min(50, len(universe_pairs)),
        step=5,
    )
    cfg.price_period = st.sidebar.selectbox(
        "טווח דאטה לניתוח",
        options=["3mo", "6mo", "1y", "2y", "3y"],
        index=2,
    )
    cfg.bar_size = "1d"  # כרגע נעול daily; ניתן להרחיב בעתיד

    cfg.allow_failures = st.sidebar.checkbox(
        "להמשיך גם אם יש כישלונות בזוגות מסוימים",
        value=True,
    )
    cfg.batch_size = 10

    # ======================= פרופיל (Center) ================================

    with st.expander("🧬 עריכת פרופיל דירוג (AnalysisProfile)", expanded=False):
        edited_profile = _ui_edit_profile(profile)
    # נשמור ב-session כדי שחזרה לטאב לא תאפס
    st.session_state["analysis_profile_active"] = edited_profile

    # ======================= Smart Scan Execution ===========================

    if "analysis_scan_result" not in st.session_state:
        st.session_state["analysis_scan_result"] = None  # type: ignore[assignment]

    run_scan = st.button("▶ הרץ סריקה חכמה על היקום", type="primary")

    if run_scan:
        with st.spinner("מריץ Smart Scan על היקום..."):
            def _progress(i: int, total: int, pair: UniversePair) -> None:
                if i % cfg.batch_size == 0:
                    logger.info("SmartScan: %d/%d pairs...", i, total)

            scan_result = smart_scan_pairs(
                universe_pairs,
                edited_profile,
                price_loader=price_loader,
                ctx_base=ctx_base,
                config=cfg,
                on_progress=_progress,
            )
            st.session_state["analysis_scan_result"] = scan_result

    scan_result: Optional[SmartScanResult] = st.session_state.get("analysis_scan_result")  # type: ignore[assignment]

    if not scan_result:
        st.info("הרץ סריקה כדי לראות את תוצאות ה-Analysis.")
        return

    # ======================= תוצאות סריקה — Overview =======================

    st.subheader("📈 תוצאות סריקה חכמה (Smart Scan)", anchor=False)
    _ui_show_kpi_cards(scan_result)
    _ui_show_group_stats(scan_result)

    df_scores = scan_result.to_dataframe()
    if df_scores.empty:
        st.warning("לא נמצאו זוגות שעומדים בספים שהגדרת.")
        return

    st.markdown("### 📊 טבלת Top N זוגות (לפי ציון כולל)", unsafe_allow_html=True)
    top_n = min(cfg.top_n, len(df_scores))
    st.dataframe(df_scores.head(top_n), width = "stretch")

    # ======================= Drill-Down לזוג נבחר ===========================

    st.markdown("### 🔎 ניתוח עומק לזוג נבחר", unsafe_allow_html=True)
    # רשימת אפשרויות לפי sym_x/sym_y
    options = [
        f"{row.sym_x} / {row.sym_y} (score={row.total_score:0.1f})"
        for _, row in df_scores.head(top_n).iterrows()
    ]
    default_idx = 0

    selected_label = st.selectbox(
        "בחר זוג לניתוח עומק:",
        options=options,
        index=default_idx if options else 0,
    )

    if not options:
        st.info("אין זוגות לניתוח.")
        return

    # חילוץ sym_x/sym_y מה-label
    sel_row = df_scores.head(top_n).iloc[options.index(selected_label)]
    sym_x = str(sel_row["sym_x"])
    sym_y = str(sel_row["sym_y"])

    # מצא breakdown המתאים
    breakdown = next(
        (s for s in scan_result.scores if s.sym_x == sym_x and s.sym_y == sym_y),
        None,
    )
    if breakdown is None:
        st.warning("לא נמצא breakdown לזוג שנבחר (תקלה פנימית).")
        return

    # טעינת מחירים מחדש לזוג (כדי להציג גרפים ויזואליים)
    try:
        prices_x = price_loader(sym_x, cfg.price_period, cfg.bar_size)
        prices_y = price_loader(sym_y, cfg.price_period, cfg.bar_size)
        metrics_obj = compute_pair_metrics(sym_x, sym_y, prices_x, prices_y, profile=edited_profile)
    except Exception as exc:
        st.error(f"שגיאה בטעינת מחירים או חישוב מדדים לזוג {sym_x}/{sym_y}: {exc}")
        return

    # נסה לשלוף מטריקות מאקרו/פנדומנטלי/סיכון מה-ctx / session_state (אם יש)
    macro_metrics: Dict[str, Any] = {}
    fund_metrics: Dict[str, Any] = {}
    risk_metrics: Dict[str, Any] = {}

    try:
        ctx_dict = st.session_state.get("ctx", {}) if st is not None else {}
        mm = ctx_dict.get("macro_metrics") or st.session_state.get("macro_metrics", {})
        fm = ctx_dict.get("fundamentals_metrics") or st.session_state.get("fundamentals_metrics", {})
        rm = ctx_dict.get("risk_metrics") or st.session_state.get("risk_metrics", {})

        if isinstance(mm, Mapping):
            macro_metrics = dict(mm)
        if isinstance(fm, Mapping):
            fund_metrics = dict(fm)
        if isinstance(rm, Mapping):
            risk_metrics = dict(rm)
    except Exception:
        macro_metrics = {}
        fund_metrics = {}
        risk_metrics = {}


    # כרטיסוני מידע
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("ציון כולל", f"{breakdown.total_score:0.1f}")
    with c2:
        st.metric("ציון גולמי", f"{breakdown.raw_total_score:0.1f}")
    with c3:
        st.metric("ציון לאחר Sigmoid", f"{breakdown.postprocessed_score:0.1f}")
    with c4:
        prob = breakdown.probability_of_profit
        st.metric("הסתברות רווח (אם קיימת)", f"{prob*100:0.1f}%" if prob is not None else "N/A")

    # גרפים
    st.markdown("#### 📉 מחירים + ספרד + Z-Score", unsafe_allow_html=True)
    _ui_plot_prices_and_spread(sym_x, sym_y, prices_x, prices_y, metrics_obj)

    # Breakdown פרמטרים
    st.markdown("#### 🧮 פירוט ציוני פרמטרים", unsafe_allow_html=True)
    _ui_show_param_breakdown(breakdown)

    # 🔀 Tab Comparison Lab מחובר לזוג שנבחר
    st.markdown("---")
    try:
        render_tab_comparison_lab(
            breakdown=breakdown,
            metrics_obj=metrics_obj,
            macro_metrics=macro_metrics,
            fund_metrics=fund_metrics,
            risk_metrics=risk_metrics,
        )
    except Exception as exc:  # pragma: no cover - UI-only
        st.warning(f"Tab Comparison Lab נכשל: {exc!r}")

    # הצגת metrics raw (למי שאוהב טכני)
    with st.expander("📐 מדדים גולמיים (PairMetrics)", expanded=False):
        st.json(metrics_obj.model_dump(mode="json"))


# ======================================================================
# Part 5/6 — Persistence, Caching & Batch Utilities (HF-grade)
# ======================================================================
"""
בחלק הזה אנחנו מוסיפים שכבה מקצועית של:

1. קונפיגורציית אחסון:
   - ScanStorageConfig — הגדרות איפה ואיך לשמור סריקות.
2. אחסון סריקות:
   - save_scan_to_parquet / load_scan_from_parquet
   - שמירה/טעינה ל-DuckDB (אם duckdb זמין).
3. Hashing & Cache חכם:
   - בניית מפתח Cache לפי:
       * universe_id + pair_count
       * profile_hash
       * SmartScanConfig
   - run_smart_scan_with_cache:
       * בודק אם יש סריקה תואמת בדיסק.
       * אם קיימת → טוען ומחזיר SmartScanResult.
       * אחרת → מריץ smart_scan_pairs, שומר ומחזיר.
4. Batch Utilities:
   - run_batch_scans — ריצה סדרתית על כמה יקומים / פרופילים / קונפיגים.
"""

# נסיון יבוא ל-duckdb (אופציונלי)
try:
    import duckdb as _duckdb  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    _duckdb = None  # type: ignore[assignment]

class ScanStorageConfig(BaseModel):
    """
    הגדרות אחסון ו-cache לסריקות Smart Scan.

    base_dir:
        תיקיית הבסיס לשמירת תוצאות (Parquet/CSV/DB).
    use_parquet:
        אם True, נשמור סריקות כ-Parquet (בהתאם ל-SmartScanResult.export_parquet).
    use_duckdb:
        אם True ו-duckdb זמין, נשמור סריקות גם בתוך DB.
    duckdb_path:
        נתיב לקובץ DuckDB (אם לא מוגדר → בתוך base_dir/scan_store.duckdb).
    """

    model_config = ConfigDict(extra="ignore")

    base_dir: Path = Field(default_factory=lambda: Path.cwd() / "scan_store")
    use_parquet: bool = True
    use_duckdb: bool = False
    duckdb_path: Optional[Path] = None

    def ensure_dirs(self) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)

    @property
    def duckdb_file(self) -> Optional[Path]:
        if not self.use_duckdb:
            return None
        if self.duckdb_path is not None:
            return self.duckdb_path
        return self.base_dir / "scan_store.duckdb"

    def parquet_path_for_key(self, cache_key: str) -> Path:
        return self.base_dir / f"scan_{cache_key}.parquet"

    def meta_path_for_key(self, cache_key: str) -> Path:
        return self.base_dir / f"scan_{cache_key}.meta.json"


# ----------------------------------------------------------------------
# Hash / Cache Key Utilities
# ----------------------------------------------------------------------


def _hash_smart_scan_key(
    universe_meta: UniverseMetadata,
    profile: AnalysisProfile,
    config: SmartScanConfig,
) -> str:
    """
    בונה cache_key יציב מתוך:

    - universe_meta.universe_id + pair_count
    - profile_hash
    - SmartScanConfig (כדי להבדיל ספים שונים)

    חייב להיות יציב לפורמט JSON/Dict ולא תלוי בסדר מפתחות.
    """
    data = {
        "universe_id": universe_meta.universe_id,
        "pair_count": universe_meta.pair_count,
        "profile_hash": profile.compute_hash(),
        "scan_config": config.to_dict(),
    }
    raw = repr(data).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:24]


# ----------------------------------------------------------------------
# Parquet / JSON Meta Persistence
# ----------------------------------------------------------------------


def save_scan_to_parquet_with_meta(
    scan_result: SmartScanResult,
    cache_key: str,
    storage_cfg: ScanStorageConfig,
) -> None:
    """
    שומר SmartScanResult כ-Parquet + קובץ meta JSON קטן.

    - Parquet: תוצאות scores בלבד (flattened DataFrame).
    - meta.json: מידע טקסטואלי (scan_id, universe_meta, config, stats).
    """
    storage_cfg.ensure_dirs()
    parquet_path = storage_cfg.parquet_path_for_key(cache_key)
    meta_path = storage_cfg.meta_path_for_key(cache_key)

    # 1) Parquet (scores בלבד)
    try:
        scan_result.export_parquet(parquet_path)
        logger.info("Saved scan parquet to %s", parquet_path)
    except Exception as exc:
        logger.exception("Failed to save scan parquet to %s: %s", parquet_path, exc)

    # 2) meta.json
    try:
        meta = {
            "scan_id": scan_result.scan_id,
            "created_at": scan_result.created_at.isoformat(),
            "universe_meta": scan_result.universe_meta.model_dump(mode="json"),
            "config": scan_result.config.to_dict(),
            "stats": scan_result.stats,
            "group_stats": scan_result.group_stats,
        }
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        logger.info("Saved scan meta to %s", meta_path)
    except Exception as exc:
        logger.exception("Failed to save scan meta to %s: %s", meta_path, exc)


def load_scan_from_parquet_with_meta(
    cache_key: str,
    storage_cfg: ScanStorageConfig,
) -> Optional[SmartScanResult]:
    """
    מנסה לטעון סריקה קיימת מהדיסק:

    - אם קובץ Parquet + meta.json קיימים → משחזר SmartScanResult חלקי.
    - אם חסר משהו → מחזיר None.

    הערה:
      שיחזור מלא של PairScoreBreakdown param_scores מתוך Parquet בלבד
      הוא מורכב, ולכן כאן אנחנו בונים SmartScanResult "aggregated בלבד"
      לפונקציות שמשתמשות ב-DataFrame של scores ולא ב-breakdown הפרמטרי.
    """
    parquet_path = storage_cfg.parquet_path_for_key(cache_key)
    meta_path = storage_cfg.meta_path_for_key(cache_key)

    if not parquet_path.exists() or not meta_path.exists():
        return None

    try:
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception as exc:
        logger.exception("Failed to load scan meta from %s: %s", meta_path, exc)
        return None

    try:
        df = pd.read_parquet(parquet_path)
    except Exception as exc:
        logger.exception("Failed to load scan parquet from %s: %s", parquet_path, exc)
        return None

    # בונים SmartScanResult מצומצם:
    # - scores: PairScoreBreakdown ללא param_scores (רק info בסיסי).
    scores: List[PairScoreBreakdown] = []
    for _, row in df.iterrows():
        # נשתמש רק במידע שאנחנו יודעים לזהות בבטחה
        breakdown = PairScoreBreakdown(
            sym_x=str(row["sym_x"]),
            sym_y=str(row["sym_y"]),
            raw_total_score=float(row.get("raw_total_score", row.get("total_score", 0.0))),
            postprocessed_score=float(row.get("postprocessed_score", row.get("total_score", 0.0))),
            calibrated_score=_safe_float(row.get("calibrated_score")),
            total_score=float(row.get("total_score", 0.0)),
            param_scores=[],  # אין breakdown מפורט מה-Parquet בלבד
            lambda_used=0.0,
            equal_weights=False,
            vol_regime=VolRegime.UNKNOWN,
            market_bias=MarketBias.NEUTRAL,
            profile_name=str(row.get("profile_name", "")),
            profile_version=str(row.get("profile_version", "")),
            profile_hash=None,
            missing_params=[],
            calibrator_name=None,
            probability_of_profit=_safe_float(row.get("probability_of_profit")),
        )
        scores.append(breakdown)

    # universe_meta & config מתוך meta.json
    universe_meta = UniverseMetadata(
        **meta.get("universe_meta", {}),
    )
    scan_config = SmartScanConfig(**meta.get("config", {}))

    scan_result = SmartScanResult(
        scan_id=meta.get("scan_id", ""),
        created_at=_dt.fromisoformat(meta.get("created_at", _dt.utcnow().isoformat())),
        universe_meta=universe_meta,
        config=scan_config,
        scores=scores,
        errors=[],
        stats=meta.get("stats", {}),
        group_stats=meta.get("group_stats", {}),
    )

    logger.info("Loaded cached scan from parquet/meta for key=%s", cache_key)
    return scan_result


# ----------------------------------------------------------------------
# DuckDB persistence (Optional)
# ----------------------------------------------------------------------


def save_scan_to_duckdb(
    scan_result: SmartScanResult,
    universe_pairs: Sequence[UniversePair],
    cache_key: str,
    storage_cfg: ScanStorageConfig,
) -> None:
    """
    שומר סריקה בתוך DuckDB (אם זמין ומוגדר):

    נבנה טבלה:
      smart_scans (
        cache_key TEXT,
        scan_id TEXT,
        created_at TIMESTAMP,
        universe_id TEXT,
        profile_name TEXT,
        profile_version TEXT,
        sym_x TEXT,
        sym_y TEXT,
        total_score DOUBLE,
        raw_total_score DOUBLE,
        postprocessed_score DOUBLE
      )

    וכן אפשר ליצור טבלת meta אם רוצים בהמשך.
    """
    if not storage_cfg.use_duckdb or _duckdb is None:
        return

    duckdb_file = storage_cfg.duckdb_file
    if duckdb_file is None:
        return

    try:
        con = _duckdb.connect(str(duckdb_file))
    except Exception as exc:
        logger.exception("Failed to connect DuckDB at %s: %s", duckdb_file, exc)
        return

    try:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS smart_scans (
                cache_key TEXT,
                scan_id TEXT,
                created_at TIMESTAMP,
                universe_id TEXT,
                profile_name TEXT,
                profile_version TEXT,
                sym_x TEXT,
                sym_y TEXT,
                total_score DOUBLE,
                raw_total_score DOUBLE,
                postprocessed_score DOUBLE
            )
            """
        )

        rows = []
        for s in scan_result.scores:
            rows.append(
                (
                    cache_key,
                    scan_result.scan_id,
                    scan_result.created_at,
                    scan_result.universe_meta.universe_id,
                    s.profile_name,
                    s.profile_version,
                    s.sym_x,
                    s.sym_y,
                    s.total_score,
                    s.raw_total_score,
                    s.postprocessed_score,
                )
            )
        if rows:
            con.executemany(
                """
                INSERT INTO smart_scans (
                    cache_key, scan_id, created_at, universe_id,
                    profile_name, profile_version,
                    sym_x, sym_y,
                    total_score, raw_total_score, postprocessed_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            logger.info(
                "Saved %d rows to DuckDB smart_scans (key=%s)", len(rows), cache_key
            )
    except Exception as exc:
        logger.exception("Failed to save scan to DuckDB: %s", exc)
    finally:
        try:
            con.close()
        except Exception:
            pass


# ----------------------------------------------------------------------
# Cached Smart Scan Runner
# ----------------------------------------------------------------------


def run_smart_scan_with_cache(
    pairs: Sequence[UniversePair],
    profile: AnalysisProfile,
    price_loader: Callable[[str, Optional[str], str], pd.Series],
    *,
    ctx_base: Optional[AnalysisContext] = None,
    config: Optional[SmartScanConfig] = None,
    storage_cfg: Optional[ScanStorageConfig] = None,
    force_rerun: bool = False,
    on_progress: Optional[Callable[[int, int, UniversePair], None]] = None,
    on_pair_scored: Optional[
        Callable[[PairScoreBreakdown, PairMetrics, UniversePair, int, int], None]
    ] = None,
) -> SmartScanResult:
    """
    מריץ Smart Scan עם שכבת cache ואחסון:

    - מחשב cache_key מתוך:
        * universe_meta (universe_id, pair_count)
        * profile_hash
        * SmartScanConfig
    - אם force_rerun=False:
        * מנסה לטעון תוצאה קיימת מה-Parquet/meta.
        * אם מצא → מחזיר מיד.
    - אחרת:
        * מריץ smart_scan_pairs.
        * שומר את התוצאה ל-Parquet/meta (ואפשר גם ל-DuckDB).
        * מחזיר.

    שימוש טיפוסי:
    -------------
    storage_cfg = ScanStorageConfig(base_dir=Path("data/scan_cache"))
    cfg = SmartScanConfig(...)
    res = run_smart_scan_with_cache(pairs, profile, price_loader,
                                    ctx_base=ctx, config=cfg,
                                    storage_cfg=storage_cfg)
    """
    cfg = config or SmartScanConfig()
    st_cfg = storage_cfg or ScanStorageConfig()
    st_cfg.ensure_dirs()

    # universe_meta בסיסי לצורך key
    universe_meta = UniverseMetadata(
        name=ctx_base.universe_id if ctx_base and ctx_base.universe_id else "universe",
        source=None,
        created_at=_dt.utcnow(),
        version=None,
        author=None,
        note="cache-key-universe-meta",
        pair_count=len(pairs),
    )

    cache_key = _hash_smart_scan_key(universe_meta, profile, cfg)
    logger.info("run_smart_scan_with_cache: cache_key=%s", cache_key)

    # ניסיון טעינה מה-cache
    if not force_rerun and st_cfg.use_parquet:
        cached = load_scan_from_parquet_with_meta(cache_key, st_cfg)
        if cached is not None:
            logger.info("Using cached SmartScanResult for key=%s", cache_key)
            return cached

    # אין cache או forced rerun → מריצים סריקה אמיתית
    logger.info("Running fresh SmartScan for key=%s ...", cache_key)
    scan_result = smart_scan_pairs(
        pairs=pairs,
        profile=profile,
        price_loader=price_loader,
        ctx_base=ctx_base,
        config=cfg,
        on_progress=on_progress,
        on_pair_scored=on_pair_scored,
    )

    # עדכון universe_meta עם pair_count אמיתי
    scan_result.universe_meta.pair_count = len(pairs)

    # שמירה ל-Parquet/meta
    if st_cfg.use_parquet:
        save_scan_to_parquet_with_meta(scan_result, cache_key, st_cfg)

    # שמירה ל-DuckDB (אופציונלי)
    save_scan_to_duckdb(scan_result, pairs, cache_key, st_cfg)

    return scan_result


# ----------------------------------------------------------------------
# Batch Utilities — ריצת מספר סריקות ברצף
# ----------------------------------------------------------------------


class BatchScanSpec(BaseModel):
    """
    הגדרה ל-"Job" אחד בריצת Batch:

    - name: שם לוגי (למשל "ETFs 6M", "Tech 1Y").
    - pairs: רשימת UniversePair.
    - profile: AnalysisProfile לציון.
    - config: SmartScanConfig לתנאי הסריקה.
    """

    model_config = ConfigDict(extra="ignore")

    name: str
    pairs: List[UniversePair]
    profile: AnalysisProfile
    config: SmartScanConfig


def run_batch_scans(
    jobs: Sequence[BatchScanSpec],
    price_loader: Callable[[str, Optional[str], str], pd.Series],
    *,
    ctx_base: Optional[AnalysisContext] = None,
    storage_cfg: Optional[ScanStorageConfig] = None,
    force_rerun: bool = False,
    on_job_start: Optional[Callable[[BatchScanSpec, int, int], None]] = None,
    on_job_done: Optional[Callable[[BatchScanSpec, SmartScanResult, int, int], None]] = None,
) -> Dict[str, SmartScanResult]:
    """
    מריץ מספר Smart Scans ברצף (למשל לפי יקומים / פרופילים שונים).

    jobs:
        רשימה של BatchScanSpec, כל אחד מגדיר יקום+פרופיל+קונפיג אחר.
    price_loader:
        פונקציית טעינת מחירים.

    storage_cfg:
        הגדרת אחסון משותפת לכל ה-jobs.
    force_rerun:
        אם True, מתעלמים מה-cache ורצים מחדש בכל job.

    on_job_start(job, idx, total):
        callback לפני התחלת Job.
    on_job_done(job, scan_result, idx, total):
        callback אחרי סיום Job.

    מחזיר:
        dict {job.name: SmartScanResult}
    """
    st_cfg = storage_cfg or ScanStorageConfig()
    results: Dict[str, SmartScanResult] = {}

    total = len(jobs)
    for idx, job in enumerate(jobs, start=1):
        if on_job_start is not None:
            try:
                on_job_start(job, idx, total)
            except Exception:
                logger.exception("on_job_start callback failed")

        logger.info("BatchScan %d/%d: job=%s, pairs=%d", idx, total, job.name, len(job.pairs))

        res = run_smart_scan_with_cache(
            pairs=job.pairs,
            profile=job.profile,
            price_loader=price_loader,
            ctx_base=ctx_base,
            config=job.config,
            storage_cfg=st_cfg,
            force_rerun=force_rerun,
        )
        results[job.name] = res

        if on_job_done is not None:
            try:
                on_job_done(job, res, idx, total)
            except Exception:
                logger.exception("on_job_done callback failed")

    return results

# ======================================================================
# Part 6/6 — Synthetic Testing, Diagnostics & Sanity Utilities (HF-grade+)
# ======================================================================
"""
בחלק הזה אנחנו מוסיפים שכבת בדיקות וסימולציות ברמת קרן גידור, מעל כל המנוע:

1. יצירת דאטה סינתטי:
   - generate_synthetic_pair:
       * mean_reverting=True → זוג עם ספרד AR(1) (cointegration-style).
       * mean_reverting=False → שני random walks בלתי תלויים.
       * מאפשר לשלוט ב-phi (עוצמת mean-reversion), דריפט, תנודתיות ושוקים.
   - SyntheticMarketEnv:
       * מיפוי סימבול → סדרת מחירים.
       * price_loader תואם ל-smart_scan_pairs (symbol, period, bar_size).

2. יקום סינתטי:
   - generate_synthetic_universe:
       * בונה מספר זוגות MR + מספר זוגות random + meta עשיר (tags, asset_class, sector).
       * מחזיר (universe_pairs, market_env).

3. פרופיל sanity מקצועי:
   - create_sanity_profile:
       * פרופיל קטן אך "חד" הבנוי על PairMetrics:
         corr, half_life, hurst, adf_pvalue, sharpe_60d, max_dd_60d.

4. בדיקות sanity:
   - run_sanity_diagnostics:
       * בודק זוג MR בודד מול זוג random אחד.
   - run_sanity_smart_scan:
       * בונה יקום של MR+Random, מריץ Smart Scan, ובודק שה-MR מדורגים גבוה.
   - assert_sanity_ok:
       * זורק AssertionError אם מנוע הניקוד לא מתנהג כמו שצריך ברמת sanity.

המטרה:
- לתת לך סט "בדיקות עשירות" שהורסים מנוע גרוע ומאשרים מנוע טוב — לפני שיורה על שוק אמיתי.
"""


# ========================= Synthetic Pair Generator =========================


def generate_synthetic_pair(
    n_obs: int = 500,
    *,
    mean_reverting: bool = True,
    phi: float = 0.9,
    drift_x: float = 0.0007,
    drift_y: float = 0.0004,
    vol_x: float = 0.02,
    vol_y: float = 0.02,
    spread_vol: float = 0.02,
    shock_prob: float = 0.0,
    shock_scale: float = 0.10,
    seed: Optional[int] = None,
    sym_x: str = "SYN_X",
    sym_y: str = "SYN_Y",
) -> Tuple[str, str, pd.Series, pd.Series]:
    """
    יוצר זוג סינתטי של מחירי Close:

    mean_reverting=True:
      - sym_y: random walk (log-price) עם drift_y, vol_y.
      - spread: תהליך AR(1) עם |phi|<1 ו-shockים נדירים (אם shock_prob>0).
      - sym_x: מוגדר כך ש-spread = log(px) - log(py) → זוג "כמעט cointegrated".

    mean_reverting=False:
      - sym_x, sym_y: שני random walks בלתי תלויים (drift_x, vol_x) ו-(drift_y, vol_y).
      - spread לא mean-reverting (בממוצע).

    פרמטרים:

    - n_obs         : מספר תצפיות (ימים).
    - phi           : פרמטר AR(1) לספרד (0<phi<1; ככל שקרוב ל-1 → half-life ארוך יותר).
    - drift_x, drift_y : דריפט יומי ללוג-מחיר.
    - vol_x, vol_y     : סטיית תקן יומית ללוג-מחיר.
    - spread_vol    : סטיית תקן של רעש הספרד.
    - shock_prob    : הסתברות לשוק "קפיצה" גדולה בספרד בכל יום (0 → אין שוקים).
    - shock_scale   : גודל סטיית תקן של השוק (ביחידות log spread).
    - seed          : seed לחזרתיות.
    """
    rng = np.random.default_rng(seed)

    def _random_walk(mu: float, sigma: float) -> np.ndarray:
        noise = rng.normal(loc=mu, scale=sigma, size=n_obs)
        log_price = np.cumsum(noise)
        return log_price

    if mean_reverting:
        # sym_y = random walk בסיסי
        log_y = _random_walk(mu=drift_y, sigma=vol_y)

        # AR(1) על spread עם שוקים
        eps = rng.normal(loc=0.0, scale=spread_vol, size=n_obs)
        spread = np.zeros(n_obs)
        for t in range(1, n_obs):
            shock = 0.0
            if shock_prob > 0 and rng.random() < shock_prob:
                shock = rng.normal(loc=0.0, scale=shock_scale)
            spread[t] = phi * spread[t - 1] + eps[t] + shock

        log_x = log_y + spread  # spread = log_x - log_y
        px = np.exp(log_x)
        py = np.exp(log_y)
    else:
        # שני random walks בלתי תלויים
        log_x = _random_walk(mu=drift_x, sigma=vol_x)
        log_y = _random_walk(mu=drift_y, sigma=vol_y)
        px = np.exp(log_x)
        py = np.exp(log_y)

    idx = pd.date_range(end=pd.Timestamp.today().normalize(), periods=n_obs, freq="B")
    prices_x = pd.Series(px, index=idx, name=sym_x)
    prices_y = pd.Series(py, index=idx, name=sym_y)
    return sym_x, sym_y, prices_x, prices_y


# ========================= Synthetic Market Env =========================


@dataclass
class SyntheticMarketEnv:
    """
    סביבת שוק סינתטית פשוטה:

    - symbols_to_prices: dict {symbol -> pd.Series של מחירים}
    - price_loader(sym, period, bar_size):
        תואם לחתימה שה-Smart Scan משתמש בה.

    שימושי:
    - בדיקות Smart Scan ללא גישה לשוק אמיתי.
    - Benchmark לפיתוח פרופילים, מדדים, ועוד.
    """

    symbols_to_prices: Dict[str, pd.Series]

    def price_loader(self, symbol: str, period: Optional[str], bar_size: str) -> pd.Series:
        # period/bar_size לא משנים כאן את הדאטה, רק קיימים לתאימות חתימה
        try:
            s = self.symbols_to_prices[symbol]
        except KeyError:
            raise KeyError(f"Symbol {symbol!r} not found in SyntheticMarketEnv.")
        return s.copy()


# ========================= Synthetic Universe Generator =========================


def generate_synthetic_universe(
    n_mr_pairs: int = 10,
    n_random_pairs: int = 10,
    *,
    n_obs: int = 500,
    seed: int = 123,
) -> Tuple[List[UniversePair], SyntheticMarketEnv]:
    """
    בונה יקום סינתטי של זוגות:

    - n_mr_pairs זוגות עם ספרד mean-reverting (cointegration-style).
    - n_random_pairs זוגות random (לא mean-reverting).
    - לכל סימבול נוצר שם ייחודי + meta עשיר:
        * asset_class: "SYN"
        * sector_x/sector_y: "SyntheticMR" או "SyntheticRnd"
        * tags: ["synthetic", "mr"] / ["synthetic", "random"]

    מחזיר:
        (universe_pairs, synthetic_env)
        שבו synthetic_env.price_loader מתאים ל-smart_scan_pairs.
    """
    rng = np.random.default_rng(seed)

    universe_pairs: List[UniversePair] = []
    prices_map: Dict[str, pd.Series] = {}

    # זוגות mean-reverting
    for i in range(n_mr_pairs):
        sx = f"MR_X{i+1}"
        sy = f"MR_Y{i+1}"
        # phi נע באיזור 0.85–0.95
        phi = float(rng.uniform(0.85, 0.95))
        sym_x, sym_y, px, py = generate_synthetic_pair(
            n_obs=n_obs,
            mean_reverting=True,
            phi=phi,
            drift_x=0.0007,
            drift_y=0.0004,
            spread_vol=0.02,
            shock_prob=0.02,
            shock_scale=0.10,
            seed=seed + i * 2,
            sym_x=sx,
            sym_y=sy,
        )
        prices_map[sym_x] = px
        prices_map[sym_y] = py

        meta = {
            "asset_class": "SYN",
            "sector_x": "SyntheticMR",
            "sector_y": "SyntheticMR",
            "tags": ["synthetic", "mr"],
            "phi": phi,
        }
        universe_pairs.append(UniversePair(sym_x=sym_x, sym_y=sym_y, meta=meta))

    # זוגות random
    for j in range(n_random_pairs):
        sx = f"RND_X{j+1}"
        sy = f"RND_Y{j+1}"
        sym_x, sym_y, px, py = generate_synthetic_pair(
            n_obs=n_obs,
            mean_reverting=False,
            drift_x=0.0007,
            drift_y=0.0004,
            vol_x=0.02,
            vol_y=0.02,
            seed=seed + 1000 + j * 2,
            sym_x=sx,
            sym_y=sy,
        )
        prices_map[sym_x] = px
        prices_map[sym_y] = py

        meta = {
            "asset_class": "SYN",
            "sector_x": "SyntheticRnd",
            "sector_y": "SyntheticRnd",
            "tags": ["synthetic", "random"],
        }
        universe_pairs.append(UniversePair(sym_x=sym_x, sym_y=sym_y, meta=meta))

    env = SyntheticMarketEnv(symbols_to_prices=prices_map)
    return universe_pairs, env


# ========================= Sanity Profile =========================


def create_sanity_profile() -> AnalysisProfile:
    """
    יוצר פרופיל קטן ומקצועי לבדיקות sanity בלבד.

    הפרמטרים:

    - corr:         רוצים |corr| גבוה.
    - half_life:    רוצים half-life קצר (מתחת ~100).
    - hurst:        רוצים Hurst נמוך (למשל <=0.6).
    - adf_pvalue:   רוצים p-value נמוך (<=0.1) → stationarity.
    - sharpe_60d:   רוצים Sharpe חיובי ומתון.
    - max_dd_60d:   רוצים drawdown "סביר" (שלילי אבל לא ענק).

    זה לא פרופיל מסחר "אמיתי", אלא כלי לבדיקות.
    """
    params: List[ParamProfile] = []

    # corr — רוצים בתוך [0.6, 1.0]
    params.append(
        ParamProfile(
            name="corr",
            display_name="קורלציה (כוללת)",
            lo=0.6,
            hi=1.0,
            weight=2.0,
            target_direction=TargetDirection.INSIDE,
            group="correlation",
        )
    )

    # half_life — רוצים בין 1 ל-100 ימים (ככל שקטן יותר → טוב יותר)
    params.append(
        ParamProfile(
            name="half_life",
            display_name="Half-Life ימים",
            lo=1.0,
            hi=100.0,
            weight=2.0,
            target_direction=TargetDirection.BELOW,  # ערך נמוך עדיף
            group="mean_reversion",
        )
    )

    # hurst — רוצים <=0.6 (mean-reverting-ish)
    params.append(
        ParamProfile(
            name="hurst",
            display_name="Hurst",
            lo=0.0,
            hi=0.6,
            weight=1.5,
            target_direction=TargetDirection.BELOW,
            group="mean_reversion",
        )
    )

    # adf_pvalue — רוצים <=0.1
    params.append(
        ParamProfile(
            name="adf_pvalue",
            display_name="ADF p-value",
            lo=0.0,
            hi=0.1,
            weight=1.5,
            target_direction=TargetDirection.BELOW,
            group="stationarity",
        )
    )

    # sharpe_60d — רוצים בין 0.0 ל-3.0 (אם הספרד נותן edge)
    params.append(
        ParamProfile(
            name="sharpe_60d",
            display_name="Sharpe 60d",
            lo=0.0,
            hi=3.0,
            weight=1.0,
            target_direction=TargetDirection.ABOVE,
            group="performance",
        )
    )

    # max_dd_60d — drawdown שלילי אבל לא ענק (למשל בין -0.5 ל-0.0)
    params.append(
        ParamProfile(
            name="max_dd_60d",
            display_name="Max Drawdown 60d",
            lo=-0.5,
            hi=0.0,
            weight=1.0,
            target_direction=TargetDirection.ABOVE,  # ככל שקרוב ל-0 (פחות שלילי) → טוב
            group="risk",
        )
    )

    meta = ProfileMeta(
        name="sanity_profile",
        version="0.1",
        author="analysis_core",
        notes="Profile for synthetic sanity tests (mean-reverting vs random).",
    )

    return AnalysisProfile(
        meta=meta,
        params=params,
        lambda_default=0.8,
        equal_weights=False,
        winsorization_pct=0.01,
        allow_nan_metrics=True,
        use_sigmoid_mapping=True,
        sigmoid_a=4.0,
        sigmoid_b=0.5,
    )


# ========================= Sanity Diagnostics — Single Pair =========================


def run_sanity_diagnostics(
    profile: Optional[AnalysisProfile] = None,
    *,
    n_obs: int = 500,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    מריץ בדיקת sanity בסיסית על המנוע:

    1. מייצר זוג Mean-Reverting סינתטי (cointegration-style).
    2. מייצר זוג Random (שני random walks בלתי תלויים).
    3. מחשב PairMetrics לכל אחד.
    4. מריץ score_pair על כל אחד.
    5. מחזיר dict עם:
        - scores: ציון לכל זוג.
        - metrics_snapshot: ערכים עיקריים (corr, half_life, hurst וכו').
        - verdict: האם mean_reverting קיבל ציון גבוה מהרנדומלי ובכמה.

    שימוש:
    - כבדיקת smoke בתחילת Notebook.
    - בסיס ל-unit-tests (אפשר assert על diff>0).
    """
    prof = profile or create_sanity_profile()

    # 1) זוג mean-reverting
    syn_x1, syn_y1, px1, py1 = generate_synthetic_pair(
        n_obs=n_obs,
        mean_reverting=True,
        seed=seed,
        sym_x="MR_X",
        sym_y="MR_Y",
    )

    # 2) זוג random
    syn_x2, syn_y2, px2, py2 = generate_synthetic_pair(
        n_obs=n_obs,
        mean_reverting=False,
        seed=seed + 1,
        sym_x="RND_X",
        sym_y="RND_Y",
    )

    # קונטקסט בסיסי
    ctx_mr = AnalysisContext(sym_x=syn_x1, sym_y=syn_y1)
    ctx_rnd = AnalysisContext(sym_x=syn_x2, sym_y=syn_y2)

    # PairMetrics
    metrics_mr = compute_pair_metrics(syn_x1, syn_y1, px1, py1, profile=prof)
    metrics_rnd = compute_pair_metrics(syn_x2, syn_y2, px2, py2, profile=prof)

    dict_mr = metrics_mr.to_metric_dict()
    dict_rnd = metrics_rnd.to_metric_dict()

    # ציונים
    score_mr = score_pair(dict_mr, prof, ctx_mr)
    score_rnd = score_pair(dict_rnd, prof, ctx_rnd)

    verdict = {
        "mr_score": score_mr.total_score,
        "rnd_score": score_rnd.total_score,
        "mr_vs_rnd_diff": score_mr.total_score - score_rnd.total_score,
        "mr_should_be_higher": score_mr.total_score > score_rnd.total_score,
    }

    logger.info(
        "Sanity diagnostics: MR score=%.2f, RND score=%.2f, diff=%.2f",
        score_mr.total_score,
        score_rnd.total_score,
        verdict["mr_vs_rnd_diff"],
    )

    # כמה מדדים בסיסיים שנוח לראות
    snapshot_mr = {
        k: dict_mr.get(k)
        for k in ("corr", "half_life", "hurst", "adf_pvalue", "sharpe_60d", "max_dd_60d")
    }
    snapshot_rnd = {
        k: dict_rnd.get(k)
        for k in ("corr", "half_life", "hurst", "adf_pvalue", "sharpe_60d", "max_dd_60d")
    }

    return {
        "profile_meta": prof.meta.model_dump(mode="json"),
        "mr_metrics": snapshot_mr,
        "rnd_metrics": snapshot_rnd,
        "mr_score_breakdown": score_mr.model_dump(mode="json"),
        "rnd_score_breakdown": score_rnd.model_dump(mode="json"),
        "verdict": verdict,
    }


# ========================= Sanity Diagnostics — Smart Scan Universe =========================


def run_sanity_smart_scan(
    profile: Optional[AnalysisProfile] = None,
    *,
    n_mr_pairs: int = 10,
    n_random_pairs: int = 10,
    n_obs: int = 500,
    seed: int = 123,
    min_expected_gap: float = 10.0,
) -> Dict[str, Any]:
    """
    מריץ sanity על כל צינור ה-Smart Scan:

    1. יוצר יקום סינתטי עם n_mr_pairs זוגות mean-reverting ו-n_random_pairs random.
    2. בונה SyntheticMarketEnv עם price_loader תואם ל-smart_scan_pairs.
    3. מריץ Smart Scan (smart_scan_pairs) עם SmartScanConfig "קל".
    4. מפריד את הציונים לפי תגיות ("mr" / "random").
    5. מחזיר:
        - mean_score_mr / mean_score_random
        - diff
        - רשימת ציונים לכל קבוצה
        - verdict: האם mean_reverting גבוה, והאם הפער>min_expected_gap.

    פרמטר min_expected_gap:
    - פער מינימלי צפוי בממוצע בין MR ל-Random בציון הכולל.
    """
    prof = profile or create_sanity_profile()

    # 1) יקום סינתטי + סביבה
    universe_pairs, env = generate_synthetic_universe(
        n_mr_pairs=n_mr_pairs,
        n_random_pairs=n_random_pairs,
        n_obs=n_obs,
        seed=seed,
    )

    # 2) קונפיג סריקה קל
    cfg = SmartScanConfig(
        min_obs=int(n_obs * 0.8),  # לוודא שיש מספיק נתונים
        min_abs_corr=0.2,          # לא קשיח מדי
        max_half_life=300.0,
        max_hurst=0.9,
        max_adf_pvalue=0.5,
        max_kpss_pvalue=0.5,
        max_pairs=len(universe_pairs),
        top_n=len(universe_pairs),
        price_period=None,
        bar_size="1d",
        allow_failures=False,
        batch_size=20,
    )

    # 3) הרצת Smart Scan
    scan_result = smart_scan_pairs(
        pairs=universe_pairs,
        profile=prof,
        price_loader=env.price_loader,
        ctx_base=AnalysisContext(sym_x="", sym_y="", universe_id="synthetic_universe"),
        config=cfg,
    )

    # 4) הפרדת ציונים לפי תגיות
    scores_mr: List[float] = []
    scores_rnd: List[float] = []

    # צורך Map sym_x/sym_y → meta
    pair_map: Dict[Tuple[str, str], UniversePair] = {
        (p.sym_x, p.sym_y): p for p in universe_pairs
    }

    for s in scan_result.scores:
        pair = pair_map.get((s.sym_x, s.sym_y))
        if pair is None:
            continue
        tags = set(pair.tags)
        if "mr" in tags:
            scores_mr.append(s.total_score)
        elif "random" in tags:
            scores_rnd.append(s.total_score)

    mean_mr = float(np.mean(scores_mr)) if scores_mr else 0.0
    mean_rnd = float(np.mean(scores_rnd)) if scores_rnd else 0.0
    diff = mean_mr - mean_rnd

    verdict = {
        "mean_score_mr": mean_mr,
        "mean_score_random": mean_rnd,
        "diff": diff,
        "mr_should_be_higher": mean_mr > mean_rnd,
        "gap_ok": diff > min_expected_gap,
        "min_expected_gap": min_expected_gap,
    }

    logger.info(
        "Sanity SmartScan: MR mean=%.2f, RND mean=%.2f, diff=%.2f (gap_ok=%s)",
        mean_mr,
        mean_rnd,
        diff,
        verdict["gap_ok"],
    )

    return {
        "scan_stats": scan_result.stats,
        "verdict": verdict,
        "scores_mr": scores_mr,
        "scores_random": scores_rnd,
    }


# ========================= Assert-style Helper =========================


def assert_sanity_ok(
    *,
    min_pair_diff: float = 5.0,
    min_universe_gap: float = 10.0,
    n_obs: int = 500,
    seed: int = 42,
) -> None:
    """
    פונקציה בסגנון "assert" שמוודאת:

    1. run_sanity_diagnostics:
        * ציון ה-MR גבוה מציון ה-Random.
        * הפער לפחות min_pair_diff.
    2. run_sanity_smart_scan:
        * mean_score_mr > mean_score_random.
        * diff >= min_universe_gap.

    אם משהו נכשל → זורק AssertionError עם הודעה ברורה.

    שימוש:
    -------
    - בקריאות ידניות:
        assert_sanity_ok()
    - ב-unit-tests:
        פשוט להריץ ולהסתמך על exception.
    """
    single = run_sanity_diagnostics(n_obs=n_obs, seed=seed)
    v1 = single["verdict"]
    diff1 = float(v1["mr_vs_rnd_diff"])
    if not v1["mr_should_be_higher"]:
        raise AssertionError(
            f"Sanity (single pair): MR score={v1['mr_score']:.2f} "
            f"<= Random score={v1['rnd_score']:.2f} (expected MR>Random)"
        )
    if diff1 < min_pair_diff:
        raise AssertionError(
            f"Sanity (single pair): MR-Random diff={diff1:.2f} < min_pair_diff={min_pair_diff}"
        )

    universe = run_sanity_smart_scan(
        n_mr_pairs=8,
        n_random_pairs=8,
        n_obs=n_obs,
        seed=seed + 100,
        min_expected_gap=min_universe_gap,
    )
    v2 = universe["verdict"]
    if not v2["mr_should_be_higher"]:
        raise AssertionError(
            f"Sanity (universe): mean MR score={v2['mean_score_mr']:.2f} "
            f"<= mean Random score={v2['mean_score_random']:.2f}"
        )
    if not v2["gap_ok"]:
        raise AssertionError(
            f"Sanity (universe): diff={v2['diff']:.2f} < min_universe_gap={min_universe_gap}"
        )


# ========================= Debug / Manual CLI Entry (Optional) =========================


if __name__ == "__main__":
    """
    כניסה ידנית (אופציונלית) לצורך בדיקות מהירות:

    למשל:
        python analysis.py

    תריץ:
    - Sanity diagnostics לזוג אחד.
    - Sanity Smart Scan ליקום.
    - assert_sanity_ok (שיזרוק חריגה אם משהו מהותי שבור).
    """
    logging.basicConfig(level=logging.INFO)
    print("Running sanity diagnostics for analysis engine...")
    try:
        diag = run_sanity_diagnostics()
        print("Single-pair verdict:")
        print(json.dumps(diag["verdict"], indent=2, ensure_ascii=False))

        print("\nRunning SmartScan sanity on synthetic universe...")
        uni = run_sanity_smart_scan()
        print("Universe verdict:")
        print(json.dumps(uni["verdict"], indent=2, ensure_ascii=False))

        print("\nRunning assert_sanity_ok()...")
        assert_sanity_ok()
        print("✅ Sanity OK — המנוע מתנהג כמו שצריך (לפחות לפי הבדיקות הסינתטיות).")
    except AssertionError as e:
        print("❌ Sanity FAILED:", e)
    except Exception as e:
        print("❌ Unexpected error during sanity tests:", e)
