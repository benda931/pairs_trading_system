# -*- coding: utf-8 -*-
"""
core/pair_ranking.py — Canonical Pair Scoring Engine (HF-grade v4)
==================================================================

מנוע ציון אחיד לזוגות בכל המערכת.

עקרונות:
---------
1. מקור אמת אחד לציון זוגות:
   - כל מקום במערכת שרוצה pair_score / pair_label משתמש במודול הזה.
2. שקיפות:
   - פירוק הציון לתת-סקורים (risk_return, stats_quality, macro, fundamental, structure, penalty).
   - מתאים לדשבורד, לוגים, דוחות.
3. גמישות:
   - פרופילים שונים (research / live / conservative).
   - קונפיג ניתן לטעינה מ-config.json (או dict אחר).
   - תמיכה בשמות עמודות שונים (sym_x/sym_y או symbol_1/symbol_2 וכו').

שימוש בסיסי:
-------------
    import pandas as pd
    from core.pair_ranking import (
        PairScoreConfig,
        PairScoreProfile,
        rank_pairs_df,
        normalize_universe_columns,
    )

    df = pd.read_csv("pairs_universe.csv")
    df = normalize_universe_columns(df)  # ממפה symbol_1->sym_x וכו' אם צריך

    cfg = PairScoreConfig(profile=PairScoreProfile.LIVE)
    ranked = rank_pairs_df(df, cfg=cfg, top=150)
    ranked.to_csv("pairs_universe_ranked.csv", index=False)

שילוב במערכות אחרות:
---------------------
- scripts/select_top_pairs_from_ranked_csv.py:
    - במקום לחשב בעצמו pair_score, פשוט לקרוא ל-rank_pairs_df().
- scripts/research_rank_pairs_from_dq.py:
    - להשתמש ב-enforce_viability=False למחקר, ולשמור is_viable כעמודה.
- core.signals_engine:
    - לציון איכות לזוגות לפני סינון/המלצה.
- core.pair_recommender:
    - לתעדוף universe של זוגות למסחר בזמן אמת.
- dashboard (טאבים שונים):
    - להראות breakdown של הציון לכל זוג + JSON לתצוגות מתקדמות.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, fields as _dc_fields
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =========================
# Enums & dataclasses
# =========================


class PairScoreProfile(str, Enum):
    """פרופיל ציון (להתאמה ל-use-case)."""

    RESEARCH = "research"        # מוכן לקחת יותר סיכון כדי לגלות דברים
    LIVE = "live"                # מסחר חי: יותר שמרני, מעניש drawdown ומינוסים
    CONSERVATIVE = "conservative"  # קרן סופר שמרנית / לקוח עדין


@dataclass
class PairScoreComponents:
    """פירוק של הציון לתת-סקורים (לשקיפות בדשבורד)."""

    risk_return: float = 0.0       # Sharpe / Sortino / DD / Vol
    stats_quality: float = 0.0     # half-life, cointegration, p-value, quality_*
    macro_score: float = 0.0       # אם יש עמודות macro_* (או מוזן מבחוץ)
    fundamental_score: float = 0.0  # quality_score פנדמנטלי מה-index_fundamentals
    structure_score: float = 0.0   # גיוון: קטגוריה, non-clone, לא “אותו מדד”
    penalty_total: float = 0.0     # סך הענישות (חיובי → הוריד מהציון, אבל נשמר)


@dataclass
class PairScoreResult:
    """תוצאה מלאה של ציון לזוג אחד (נוחה ללוגים / דשבורד / דוחות)."""

    sym_x: str
    sym_y: str
    score: float
    label: str
    components: PairScoreComponents
    corr: float
    half_life: float
    spread_vol: float
    spread_sharpe: float
    spread_sortino: float
    max_dd: float
    n_obs: int
    meta: Dict[str, float]


@dataclass
class PairScoreConfig:
    """
    קונפיג בסיסי לציון זוגות.

    הערכים כאן הם ברירת מחדל "סבירה" – אפשר לעדכן אותם מה-config.json
    או לבחור פרופיל (research / live / conservative) ולהתאים לפי הצורך.
    """

    profile: PairScoreProfile = PairScoreProfile.RESEARCH

    # משקלים לתת-סקורים (לא לציון הגולמי, אלא לפירוק ברמת components)
    w_risk_return: float = 1.0
    w_stats_quality: float = 0.7
    w_macro: float = 0.3
    w_fundamental: float = 0.3
    w_structure: float = 0.4

    # משקלים פנימיים לרכיבי risk/return
    w_sharpe: float = 1.0
    w_sortino: float = 0.7

    # בסיס quality_*
    w_quality_core: float = 0.5  # מה-quality / quality_stat / quality_hf

    # ענישות על גורמי סיכון/מוגבלות
    penalty_high_corr: float = 0.5
    penalty_huge_dd: float = 0.3
    penalty_too_long_hl: float = 0.4
    penalty_too_short_hl: float = 0.2
    penalty_clone_like: float = 1.0
    penalty_low_n_obs: float = 0.5

    # בונוסים לקטגוריות מעניינות / גיוון
    bonus_commodity: float = 0.3
    bonus_crypto: float = 0.3
    bonus_sector: float = 0.15
    bonus_factor: float = 0.15
    bonus_mixed_style: float = 0.1  # לדוגמה: sector מול factor, index מול commodity וכו'

    # ספי קורלציה
    max_allowed_corr: float = 0.995      # מעבר לזה → לא viable
    high_corr_soft_threshold: float = 0.98  # מעל זה → ענישה לינארית
    sweet_corr_min: float = 0.65         # טווח "טעים" לקורלציה
    sweet_corr_max: float = 0.95

    # סף כמות תצפיות
    min_n_obs: int = 250
    soft_min_n_obs: int = 350  # מתחת לזה → ענישה, אבל לא פסילה אוטומטית

    # טווח half-life הגיוני (ימים)
    max_half_life: float = 300.0
    min_half_life: float = 1.0

    # מינימום תנודתיות בספרד (כדי שיהיה מה לסחור)
    min_spread_vol: float = 0.5

    # מינימום Sharpe בסיסי
    min_spread_sharpe: float = -0.2

    # עמודות אפשריות ב-DataFrame (ניתן להתאים לפי ה-universe)
    col_sym_x: str = "sym_x"
    col_sym_y: str = "sym_y"
    col_corr: str = "corr"
    col_p_value: str = "p_value"
    col_half_life: str = "half_life"
    col_spread_sharpe: str = "spread_sharpe"
    col_spread_sortino: str = "spread_sortino"
    col_spread_vol: str = "spread_vol"
    col_spread_max_dd: str = "spread_max_dd"
    col_quality: str = "quality"
    col_quality_stat: str = "quality_stat"
    col_quality_hf: str = "quality_hf"
    col_n_obs: str = "n_obs"
    col_seed_category: str = "seed_category"
    col_is_clone_like: str = "is_clone_like"

    # אם יש לך macro/fundamental scores בטבלאות אחרות שמוזגו ל-CSV
    col_macro_score: str = "macro_score"
    col_fundamental_score: str = "fundamental_score"  # composite_score מה-index_fundamentals

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PairScoreConfig":
        """
        בונה PairScoreConfig מתוך dict (למשל section ב-config.json).

        - מתעלם ממפתחות לא מוכרים.
        - profile יכול להגיע כ-str ("live") או כ-PairScoreProfile.
        """
        if not data:
            return cls()

        field_names = {f.name for f in _dc_fields(cls)}
        kwargs: Dict[str, Any] = {}

        for k, v in data.items():
            if k not in field_names:
                continue
            if k == "profile":
                if isinstance(v, PairScoreProfile):
                    kwargs[k] = v
                else:
                    try:
                        kwargs[k] = PairScoreProfile(str(v))
                    except Exception:
                        logger.warning(
                            "PairScoreConfig.from_dict: invalid profile %r, using default.",
                            v,
                        )
            else:
                kwargs[k] = v

        return cls(**kwargs)


# =========================
# Internal helpers
# =========================


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        if val is None:
            return default
        if isinstance(val, float) and np.isnan(val):
            return default
        return float(val)
    except Exception:
        return default


def _get_first_non_nan(
    row: pd.Series,
    candidates: Sequence[str],
    default: float = 0.0,
) -> float:
    for c in candidates:
        if c in row:
            v = row[c]
            if v is None:
                continue
            if isinstance(v, float) and np.isnan(v):
                continue
            try:
                return float(v)
            except Exception:
                continue
    return default


# =========================
# Column normalization
# =========================


_COLUMN_ALIASES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("sym_x", ("sym_x", "symbol_1", "asset_x", "ticker_x", "lhs", "base")),
    ("sym_y", ("sym_y", "symbol_2", "asset_y", "ticker_y", "rhs", "quote")),
)


def normalize_universe_columns(
    df: pd.DataFrame,
    cfg: Optional[PairScoreConfig] = None,
    inplace: bool = False,
) -> pd.DataFrame:
    """
    ממפה שמות עמודות שונים לעמודות הסטנדרטיות (sym_x, sym_y).

    לדוגמה:
        symbol_1 -> sym_x
        symbol_2 -> sym_y
    """
    if not inplace:
        df = df.copy()

    for target, aliases in _COLUMN_ALIASES:
        if target in df.columns:
            continue
        for alias in aliases:
            if alias in df.columns:
                df[target] = df[alias]
                break

    # התאמת cfg (אם הועבר) לשמות החדשים
    if cfg is not None:
        cfg.col_sym_x = "sym_x"
        cfg.col_sym_y = "sym_y"

    return df


# =========================
# Filters (viability)
# =========================


def is_pair_viable(row: pd.Series, cfg: Optional[PairScoreConfig] = None) -> bool:
    """
    בודק אם הזוג "ראוי" להכנס ליוניברס לפני ציון (פילטר קשה).

    חוקים בסיסיים:
    ---------------
    1. מספיק תצפיות (n_obs).
    2. half-life בטווח הגיוני.
    3. קורלציה לא גבוהה מדי (לא pure clones).
    4. יש מספיק תנודתיות בספרד.
    5. Sharpe לא גרוע מדי.

    למחקר (research) לרוב נרצה להשתמש ב-enforce_viability=False ולראות גם זוגות
    שלא עוברים את הפילטר הזה, אבל עדיין לקבל is_viable כעמודה.
    """
    if cfg is None:
        cfg = PairScoreConfig()

    n_obs = int(_safe_float(row.get(cfg.col_n_obs, 0.0), 0.0))
    if n_obs < cfg.min_n_obs:
        return False

    hl = _safe_float(row.get(cfg.col_half_life, 0.0), 0.0)
    if hl <= 0:
        return False
    if hl < cfg.min_half_life:
        return False
    if hl > cfg.max_half_life:
        return False

    corr = _safe_float(row.get(cfg.col_corr, 0.0), 0.0)
    if corr >= cfg.max_allowed_corr:
        return False

    spread_vol = _safe_float(row.get(cfg.col_spread_vol, 0.0), 0.0)
    if spread_vol <= cfg.min_spread_vol:
        return False

    sharpe = _safe_float(row.get(cfg.col_spread_sharpe, 0.0), 0.0)
    if sharpe < cfg.min_spread_sharpe:
        return False

    return True


# =========================
# Score components + label
# =========================


def label_score(score: float) -> str:
    """
    ממפה ציון כללי ל-label בסגנון קרן גידור (A+/A/A-/B+/...).

    אפשר אחרי זה להשתמש בזה בדשבורד / signals / recommendations.
    """
    if score >= 2.0:
        return "A+"
    if score >= 1.5:
        return "A"
    if score >= 1.2:
        return "A-"
    if score >= 0.9:
        return "B+"
    if score >= 0.6:
        return "B"
    if score >= 0.3:
        return "B-"
    if score >= 0.0:
        return "C"
    return "D"


def compute_pair_score_components(
    row: pd.Series,
    cfg: Optional[PairScoreConfig] = None,
    extra_macro: float = 0.0,
    extra_fundamental: float = 0.0,
) -> PairScoreComponents:
    """
    מחשב את תתי-הסקורים (components) עבור זוג אחד.

    extra_macro / extra_fundamental:
        מאפשרים לך בעתיד להעביר score מחישוב אחר
        (למשל macro_engine או index_fundamentals) במקום/בנוסף למה שיש בעמודות.
    """
    if cfg is None:
        cfg = PairScoreConfig()

    sharpe = _safe_float(row.get(cfg.col_spread_sharpe, 0.0), 0.0)
    sortino = _safe_float(row.get(cfg.col_spread_sortino, 0.0), 0.0)
    max_dd = _safe_float(row.get(cfg.col_spread_max_dd, 0.0), 0.0)
    spread_vol = _safe_float(row.get(cfg.col_spread_vol, 0.0), 0.0)
    corr = _safe_float(row.get(cfg.col_corr, 0.0), 0.0)
    hl = _safe_float(row.get(cfg.col_half_life, 0.0), 0.0)
    n_obs = int(_safe_float(row.get(cfg.col_n_obs, 0.0), 0.0))

    # 1. Risk/Return: שילוב של Sharpe/Sortino + ענישה על DD פשוטה (scaled)
    risk_return = cfg.w_sharpe * sharpe + cfg.w_sortino * sortino
    if max_dd > 0:
        # נעניש על DD גדול – תלוי פרופיל
        dd_penalty_factor = 0.02 if cfg.profile == PairScoreProfile.RESEARCH else 0.03
        risk_return -= dd_penalty_factor * min(max_dd, 100.0)

    # ספרד “מסחרית”: מעט בונוס אם יש vol סבירה, בלי להשתולל
    if spread_vol > cfg.min_spread_vol:
        # נורמליזציה גסה: הרבה מעל 10 זה כבר לא מוסיף
        vol_term = min(spread_vol, 10.0) / 10.0
        risk_return += 0.05 * vol_term

    # 2. Stats quality: מבוסס על quality_* + p_value/hl
    quality_core = _get_first_non_nan(
        row,
        [cfg.col_quality_hf, cfg.col_quality_stat, cfg.col_quality],
        default=0.0,
    )
    stats_quality = cfg.w_quality_core * (quality_core / 100.0)

    p_val = _safe_float(row.get(cfg.col_p_value, 1.0), 1.0)
    # p-value נמוך → יותר טוב
    stats_quality += max(0.0, (0.2 - p_val))  # אם p<0.2, יש תוספת קטנה

    # half-life "מתוק": לא קצר מדי ולא ארוך מדי
    if hl > 0:
        sweet_min = cfg.min_half_life * 3
        sweet_max = cfg.max_half_life * 0.5
        if sweet_min < hl < sweet_max:
            stats_quality += 0.2

    # 3. Macro score: או מה-עמודה, או extra_macro (לפי עדיפות)
    macro_score = extra_macro
    if macro_score == 0.0 and cfg.col_macro_score in row:
        macro_score = _safe_float(row.get(cfg.col_macro_score), 0.0)

    # 4. Fundamental score: composite_score / quality_score מה-index_fundamentals
    fundamental_score = extra_fundamental
    if fundamental_score == 0.0 and cfg.col_fundamental_score in row:
        fundamental_score = _safe_float(row.get(cfg.col_fundamental_score), 0.0) / 100.0

    # 5. Structure / diversification: קטגוריות + non-clone preference
    structure_score = 0.0
    cat_raw = str(row.get(cfg.col_seed_category) or "").strip().lower()
    if cat_raw:
        if any(k in cat_raw for k in ("commodity", "metal", "energy")):
            structure_score += cfg.bonus_commodity
        if any(k in cat_raw for k in ("crypto", "bitcoin", "btc", "eth")):
            structure_score += cfg.bonus_crypto
        if any(k in cat_raw for k in ("sector", "etf")):
            structure_score += cfg.bonus_sector
        if any(k in cat_raw for k in ("factor", "value", "quality", "growth")):
            structure_score += cfg.bonus_factor

    # בונוס קטן לאזור קורלציה "טעים"
    if cfg.sweet_corr_min <= corr <= cfg.sweet_corr_max:
        structure_score += 0.1

    # בונוס לזוג "מעורב סגנונות" (sector מול factor / index מול commodity וכו')
    if cat_raw:
        has_index = "index" in cat_raw
        has_sector = "sector" in cat_raw
        has_commodity = any(k in cat_raw for k in ("commodity", "metal", "energy"))
        style_count = sum([has_index, has_sector, has_commodity])
        if style_count >= 2:
            structure_score += cfg.bonus_mixed_style

    # penalties
    penalty_total = 0.0

    # high correlation penalty (מעדיפים לא pure-clone)
    if corr > cfg.high_corr_soft_threshold:
        over = min(max(corr - cfg.high_corr_soft_threshold, 0.0), 0.03)
        penalty_total += cfg.penalty_high_corr * (over / 0.03)

    # huge DD
    if max_dd > 50:
        penalty_total += cfg.penalty_huge_dd * min((max_dd - 50) / 50.0, 2.0)

    # extreme half-life (מאוד קצר/מאוד ארוך)
    if hl > 0:
        if hl < cfg.min_half_life * 2:
            penalty_total += cfg.penalty_too_short_hl
        elif hl > cfg.max_half_life * 0.7:
            penalty_total += cfg.penalty_too_long_hl

    # clone-like flag
    if bool(row.get(cfg.col_is_clone_like, False)):
        penalty_total += cfg.penalty_clone_like

    # מעט ענישה על n_obs נמוך (אבל לא מתחת למינימום הקשה)
    if n_obs < cfg.soft_min_n_obs:
        penalty_total += cfg.penalty_low_n_obs * max(
            (cfg.soft_min_n_obs - n_obs) / cfg.soft_min_n_obs,
            0.0,
        )

    return PairScoreComponents(
        risk_return=risk_return,
        stats_quality=stats_quality,
        macro_score=macro_score,
        fundamental_score=fundamental_score,
        structure_score=structure_score,
        penalty_total=penalty_total,
    )


def compute_pair_score(
    row: pd.Series,
    cfg: Optional[PairScoreConfig] = None,
    extra_macro: float = 0.0,
    extra_fundamental: float = 0.0,
) -> float:
    """
    מחשב ציון pair_score יחיד לזוג.

    - קודם מחשב components.
    - אחר כך מרכיב אותם לסקור סופי לפי משקלים ופרופיל.
    """
    if cfg is None:
        cfg = PairScoreConfig()

    comps = compute_pair_score_components(
        row,
        cfg=cfg,
        extra_macro=extra_macro,
        extra_fundamental=extra_fundamental,
    )

    score = (
        cfg.w_risk_return * comps.risk_return
        + cfg.w_stats_quality * comps.stats_quality
        + cfg.w_macro * comps.macro_score
        + cfg.w_fundamental * comps.fundamental_score
        + cfg.w_structure * comps.structure_score
    )

    score -= comps.penalty_total

    # ניתן להחמיר לפי פרופיל
    if cfg.profile == PairScoreProfile.CONSERVATIVE:
        score *= 0.85
    elif cfg.profile == PairScoreProfile.LIVE:
        score *= 0.9

    return float(score)


# =========================
# Ranking over DataFrame
# =========================


def rank_pairs_df(
    df: pd.DataFrame,
    cfg: Optional[PairScoreConfig] = None,
    top: Optional[int] = None,
    *,
    enforce_viability: bool = True,
    return_results: bool = False,
    normalize_scores: bool = False,
    score_scale: str = "raw",  # "raw" / "zscore" / "0-100"
) -> pd.DataFrame:
    """
    מקבל DataFrame שמכיל את כל המטריקות (כמו pairs_universe.csv),
    ומחזיר DataFrame מדורג לפי pair_score.

    ה-DataFrame המוחזר כולל:
      - pair_score         — ציון ראשי
      - pair_label         — A+/A/A-/B+/...
      - risk_return_score  — תת-סקור
      - stats_quality_score
      - macro_score
      - fundamental_score
      - structure_score
      - penalty_total
      - is_viable          — האם עבר את פילטר ה-viability

    אם normalize_scores=True:
      - pair_score_raw     — הציון הגולמי
      - pair_score         — הציון המנורמל (z-score / 0–100)

    אם return_results=True:
      - pair_score_result_json — JSON קומפקטי לתצוגה/לוגים.
    """
    if cfg is None:
        cfg = PairScoreConfig()

    if df.empty:
        return df.copy()

    # נוודא שיש לנו sym_x/sym_y
    df = normalize_universe_columns(df, cfg=cfg, inplace=False)

    # 1. פילטר בסיסי (אופציונלי)
    mask_viable = df.apply(lambda r: is_pair_viable(r, cfg), axis=1)
    df["is_viable"] = mask_viable  # נשמור כעמודה לשימוש בדשבורד/דו"ח

    if enforce_viability:
        df = df[mask_viable].copy()
        if df.empty:
            return df
    else:
        df = df.copy()

    # 2. חישוב קומפוננטות + ציון
    risk_scores: List[float] = []
    stats_scores: List[float] = []
    macro_scores: List[float] = []
    fund_scores: List[float] = []
    struct_scores: List[float] = []
    penalties: List[float] = []
    pair_scores: List[float] = []
    labels: List[str] = []
    results_json: Optional[List[Dict[str, Any]]] = [] if return_results else None

    for _, row in df.iterrows():
        comps = compute_pair_score_components(row, cfg=cfg)
        s = (
            cfg.w_risk_return * comps.risk_return
            + cfg.w_stats_quality * comps.stats_quality
            + cfg.w_macro * comps.macro_score
            + cfg.w_fundamental * comps.fundamental_score
            + cfg.w_structure * comps.structure_score
        ) - comps.penalty_total

        if cfg.profile == PairScoreProfile.CONSERVATIVE:
            s *= 0.85
        elif cfg.profile == PairScoreProfile.LIVE:
            s *= 0.9

        lbl = label_score(s)

        risk_scores.append(comps.risk_return)
        stats_scores.append(comps.stats_quality)
        macro_scores.append(comps.macro_score)
        fund_scores.append(comps.fundamental_score)
        struct_scores.append(comps.structure_score)
        penalties.append(comps.penalty_total)
        pair_scores.append(float(s))
        labels.append(lbl)

        if results_json is not None:
            meta = {
                "corr": _safe_float(row.get(cfg.col_corr, 0.0)),
                "half_life": _safe_float(row.get(cfg.col_half_life, 0.0)),
                "spread_vol": _safe_float(row.get(cfg.col_spread_vol, 0.0)),
                "spread_sharpe": _safe_float(row.get(cfg.col_spread_sharpe, 0.0)),
                "spread_sortino": _safe_float(row.get(cfg.col_spread_sortino, 0.0)),
                "max_dd": _safe_float(row.get(cfg.col_spread_max_dd, 0.0)),
                "n_obs": _safe_float(row.get(cfg.col_n_obs, 0.0)),
            }
            res = PairScoreResult(
                sym_x=str(row.get(cfg.col_sym_x)),
                sym_y=str(row.get(cfg.col_sym_y)),
                score=float(s),
                label=lbl,
                components=comps,
                corr=meta["corr"],
                half_life=meta["half_life"],
                spread_vol=meta["spread_vol"],
                spread_sharpe=meta["spread_sharpe"],
                spread_sortino=meta["spread_sortino"],
                max_dd=meta["max_dd"],
                n_obs=int(meta["n_obs"]),
                meta=meta,
            )
            results_json.append(
                {
                    "pair": f"{res.sym_x}-{res.sym_y}",
                    "score": res.score,
                    "label": res.label,
                    "components": asdict(res.components),
                    "meta": res.meta,
                }
            )

    # 3. הצבת העמודות ב-DataFrame
    df["pair_score"] = pair_scores
    df["pair_label"] = labels
    df["risk_return_score"] = risk_scores
    df["stats_quality_score"] = stats_scores
    df["macro_score"] = macro_scores
    df["fundamental_score"] = fund_scores
    df["structure_score"] = struct_scores
    df["penalty_total"] = penalties

    # 3b. נירמול ציון (אופציונלי)
    if normalize_scores:
        scores = pd.Series(pair_scores, index=df.index, dtype=float)
        df["pair_score_raw"] = scores

        if score_scale == "zscore":
            mu = scores.mean()
            sigma = scores.std(ddof=0)
            if sigma <= 0:
                sigma = 1.0
            df["pair_score"] = (scores - mu) / sigma
        elif score_scale in ("0-100", "percentile"):
            ranks = scores.rank(method="average", pct=True)
            df["pair_score"] = ranks * 100.0
        else:
            # raw – אבל נשאיר גם pair_score_raw
            df["pair_score"] = scores

    if results_json is not None:
        df["pair_score_result_json"] = results_json

    # 4. מיון
    df = df.sort_values("pair_score", ascending=False)

    if top is not None and top > 0:
        df = df.head(top)

    df.reset_index(drop=True, inplace=True)
    return df


# =========================
# Convenience helper
# =========================


def build_pair_score_config_from_settings(
    settings: Mapping[str, Any],
    section: str = "pair_scoring",
    profile: Optional[str] = None,
) -> PairScoreConfig:
    """
    Helper נוח לבנייה של PairScoreConfig מתוך settings גלובלי (config.json).

    דוגמה ל-config.json:
    --------------------
    {
      "pair_scoring": {
        "profile": "research",
        "min_spread_vol": 0.2,
        "min_spread_sharpe": -0.5
      }
    }

    או:
    {
      "pair_scoring": {
        "default": { ... },
        "live": { ... },
        "research": { ... }
      }
    }
    """
    section_data = settings.get(section, {}) if settings else {}
    if not isinstance(section_data, Mapping):
        return PairScoreConfig()

    # אם יש פרופילים פנימיים
    if profile:
        prof_key = str(profile)
        if prof_key in section_data and isinstance(section_data[prof_key], Mapping):
            base = dict(section_data.get("default", {}))
            base.update(section_data[prof_key])  # type: ignore[arg-type]
            return PairScoreConfig.from_dict(base)

    # אחרת – מניחים שה-section עצמו הוא המפה
    if isinstance(section_data, Mapping):
        return PairScoreConfig.from_dict(section_data)

    return PairScoreConfig()
