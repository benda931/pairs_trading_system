# -*- coding: utf-8 -*-
"""
smart_scan_tab.py — HF-grade Smart Scan Tab
===========================================

- בנוי לעבוד גם בסביבת Dashboard מלאה (עם DashboardService / DashboardContext),
  וגם כסקריפט עצמאי/דמו (fallback ל-Any אם המודולים לא קיימים).
- משתמש ב-Streamlit כ-UI engine, numpy/pandas כבסיס דאטה, ו-logging ללוגים.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Literal,
    TYPE_CHECKING,
)

import logging

import numpy as np
import pandas as pd
import streamlit as st
from uuid import uuid4

from core.app_context import AppContext
from core.sql_store import SqlStore
from common.macro_adjustments import (
    MacroConfig,
    load_macro_bundle,
    compute_adjustments,
)

# ------------------------------------------------
# Logger מרכזי ל-Smart Scan
# ------------------------------------------------
logger = logging.getLogger("SmartScan")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | SmartScan | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# מצב production: לא להשתמש בדמו. אפשר לשנות ל-True רק בסביבת dev.
ALLOW_DEMO_UNIVERSE: bool = False

# ------------------------------------------------
# טיפוסים בזמן Type-check בלבד (Pylance / mypy)
# ------------------------------------------------
if TYPE_CHECKING:
    from core.dashboard_models import DashboardContext
    from core.dashboard_service import DashboardService
else:
    # בזמן ריצה, אם המודולים לא קיימים, נשתמש ב-Any
    DashboardContext = Any  # type: ignore[assignment]
    DashboardService = Any  # type: ignore[assignment]

# ------------------------------------------------
# יצירת Service + Context מתוך dashboard_service_factory
# ------------------------------------------------
try:
    from root.dashboard_service_factory import (
        create_dashboard_service,
        build_default_dashboard_context,
    )
except Exception:
    # fallback – מאפשר להריץ את הטאב גם בלי Dashboard מלא (למשל בדמו/בדיקות)
    def create_dashboard_service() -> Any:
        logger.warning(
            "create_dashboard_service() fallback — "
            "DashboardService not available; running SmartScan in demo mode."
        )
        return None

    def build_default_dashboard_context() -> Any:
        logger.warning(
            "build_default_dashboard_context() fallback — "
            "DashboardContext not available; using empty context."
        )
        return None

# ============================
# 1.1 טווח אופטימלי לפרמטר
# ============================

@dataclass
class ParamOptimalRange:
    """
    טווח אופטימלי לפרמטר יחיד.

    בתוך [optimal_min, optimal_max] → ציון בסיס 1.
    בין hard_min ל-optimal_min או בין optimal_max ל-hard_max → ציון יורד ל-0.
    מחוץ [hard_min, hard_max] → 0.

    weight:
        משקל הפרמטר בציון הכולל (לא כל פרמטר חשוב באותה מידה).
    shape:
        1.0 = ליניארי, >1 = ירידה חדה (עונש גדול מהר יותר),
        <1 = ירידה רכה (סובלני יותר).
    """

    name: str
    hard_min: float
    optimal_min: float
    optimal_max: float
    hard_max: float
    weight: float = 1.0
    shape: float = 1.0

    def score(self, value: Optional[float]) -> float:
        """
        מחשב ציון 0–1 לערך בודד של הפרמטר.
        """
        if value is None or not np.isfinite(value):
            return 0.0

        x = float(value)
        hmin, omin, omax, hmax = (
            float(self.hard_min),
            float(self.optimal_min),
            float(self.optimal_max),
            float(self.hard_max),
        )

        # טווח לא תקין → ציון 0
        if hmax <= hmin:
            return 0.0

        # בתוך הטווח האופטימלי → ציון מלא
        if omin <= x <= omax:
            base = 1.0
        # מתחת לטווח האופטימלי אבל מעל hard_min
        elif hmin <= x < omin:
            base = (x - hmin) / (omin - hmin)
        # מעל הטווח האופטימלי אבל מתחת ל-hard_max
        elif omax < x <= hmax:
            base = (hmax - x) / (hmax - omax)
        else:
            base = 0.0

        base = max(0.0, min(1.0, base))
        if self.shape != 1.0 and base > 0.0:
            base = base ** float(self.shape)

        return float(base)


# ============================================
# 1.2 ParamOptimismConfig + פרופילים דיפולטיים
# ============================================

@dataclass
class ParamOptimismConfig:
    """
    קונפיגורציה לציון אופטימיות פרמטרים.

    ranges:
        name → ParamOptimalRange
    source:
        מקור קונפיגורציה (לוגית בלבד) — "default_profile", "ml_adapted", וכו'.
    profile:
        "default" / "defensive" / "aggressive" וכו'.
    """

    ranges: Dict[str, ParamOptimalRange]
    source: str = "default_profile"
    profile: str = "default"

    @classmethod
    def from_simple_dict(
        cls,
        data: Dict[str, Dict[str, Any]],
        *,
        profile: str = "custom",
        source: str = "manual",
    ) -> "ParamOptimismConfig":
        """
        בניית קונפיגורציה ממילון פשוט (מ-YAML/JSON/DB).
        """
        ranges: Dict[str, ParamOptimalRange] = {}
        for name, cfg in data.items():
            try:
                ranges[name] = ParamOptimalRange(
                    name=name,
                    hard_min=float(cfg["hard_min"]),
                    optimal_min=float(cfg["optimal_min"]),
                    optimal_max=float(cfg["optimal_max"]),
                    hard_max=float(cfg["hard_max"]),
                    weight=float(cfg.get("weight", 1.0)),
                    shape=float(cfg.get("shape", 1.0)),
                )
            except Exception as e:
                logger.warning("Invalid optimal range config for %s: %s", name, e)
        return cls(ranges=ranges, source=source, profile=profile)

    @classmethod
    def default_for_profile(cls, profile: str = "default") -> "ParamOptimismConfig":
        """
        יוצר ParamOptimismConfig דיפולטי לפי פרופיל:

        profile = "default":
            - z_entry: סביב 2.0–2.5
            - z_exit: סביב 0.3–0.8
            - lookback: סביב 40–80 ימים
            - hl_bars: סביב 20–80 ברים
            - corr_min: >= 0.6

        profile = "defensive":
            - z_entry גבוה יותר (פחות כניסות), hl קצר יותר, corr_min גבוה יותר.
        profile = "aggressive":
            - z_entry נמוך יותר, hl ארוך יותר יחסית, פחות דגש על corr.

        זה לא "מחקר מדעי" אלא טווחים סבירים לפי ספרות פופולרית ופרקטיקה,
        משמשים כברירת מחדל עד שה-ML ילמד טווחים פרטניים מההיסטוריה.
        """
        p = profile.lower().strip()
        if p not in {"default", "defensive", "aggressive"}:
            p = "default"

        if p == "defensive":
            cfg_raw = {
                "z_entry": {
                    "hard_min": 1.5,
                    "optimal_min": 2.2,
                    "optimal_max": 2.8,
                    "hard_max": 4.0,
                    "weight": 2.5,
                    "shape": 1.2,
                },
                "z_exit": {
                    "hard_min": 0.1,
                    "optimal_min": 0.5,
                    "optimal_max": 0.9,
                    "hard_max": 1.5,
                    "weight": 2.0,
                    "shape": 1.0,
                },
                "lookback": {
                    "hard_min": 30,
                    "optimal_min": 50,
                    "optimal_max": 90,
                    "hard_max": 200,
                    "weight": 1.0,
                    "shape": 1.0,
                },
                "hl_bars": {
                    "hard_min": 10,
                    "optimal_min": 20,
                    "optimal_max": 60,
                    "hard_max": 200,
                    "weight": 1.5,
                    "shape": 1.0,
                },
                "corr_min": {
                    "hard_min": 0.3,
                    "optimal_min": 0.7,
                    "optimal_max": 0.9,
                    "hard_max": 1.0,
                    "weight": 2.0,
                    "shape": 2.0,
                },
            }
        elif p == "aggressive":
            cfg_raw = {
                "z_entry": {
                    "hard_min": 1.0,
                    "optimal_min": 1.5,
                    "optimal_max": 2.2,
                    "hard_max": 3.5,
                    "weight": 2.0,
                    "shape": 1.0,
                },
                "z_exit": {
                    "hard_min": 0.1,
                    "optimal_min": 0.3,
                    "optimal_max": 0.7,
                    "hard_max": 1.2,
                    "weight": 1.5,
                    "shape": 1.0,
                },
                "lookback": {
                    "hard_min": 10,
                    "optimal_min": 30,
                    "optimal_max": 60,
                    "hard_max": 160,
                    "weight": 1.0,
                    "shape": 1.0,
                },
                "hl_bars": {
                    "hard_min": 5,
                    "optimal_min": 10,
                    "optimal_max": 80,
                    "hard_max": 250,
                    "weight": 1.0,
                    "shape": 0.8,
                },
                "corr_min": {
                    "hard_min": 0.2,
                    "optimal_min": 0.5,
                    "optimal_max": 0.8,
                    "hard_max": 1.0,
                    "weight": 1.5,
                    "shape": 1.5,
                },
            }
        else:  # default
            cfg_raw = {
                "z_entry": {
                    "hard_min": 1.0,
                    "optimal_min": 2.0,
                    "optimal_max": 2.7,
                    "hard_max": 4.0,
                    "weight": 2.0,
                    "shape": 1.0,
                },
                "z_exit": {
                    "hard_min": 0.1,
                    "optimal_min": 0.4,
                    "optimal_max": 0.8,
                    "hard_max": 1.3,
                    "weight": 1.5,
                    "shape": 1.0,
                },
                "lookback": {
                    "hard_min": 20,
                    "optimal_min": 40,
                    "optimal_max": 80,
                    "hard_max": 200,
                    "weight": 1.0,
                    "shape": 1.0,
                },
                "hl_bars": {
                    "hard_min": 10,
                    "optimal_min": 20,
                    "optimal_max": 80,
                    "hard_max": 220,
                    "weight": 1.2,
                    "shape": 1.0,
                },
                "corr_min": {
                    "hard_min": 0.2,
                    "optimal_min": 0.6,
                    "optimal_max": 0.85,
                    "hard_max": 1.0,
                    "weight": 1.8,
                    "shape": 1.5,
                },
            }

        cfg = cls.from_simple_dict(cfg_raw, profile=p, source="default_profile")
        logger.info("ParamOptimismConfig default_for_profile=%s created.", p)
        return cfg

    def score_vector(self, params: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        """
        מחשב ציון אופטימיות משוקלל (0–1) לוקטור פרמטרים + ציוני per-param.
        """
        if not self.ranges:
            return 0.0, {}

        per_param_scores: Dict[str, float] = {}
        weighted_sum = 0.0
        weight_sum = 0.0

        for name, rng in self.ranges.items():
            v = params.get(name)
            s = rng.score(v)
            per_param_scores[name] = s
            w = float(rng.weight)
            weighted_sum += s * w
            weight_sum += abs(w)

        if weight_sum <= 0:
            total = 0.0
        else:
            total = weighted_sum / weight_sum

        total = float(max(0.0, min(1.0, total)))
        return total, per_param_scores


# ==================================================
# 1.3 פונקציות להכנת ציון מ-Fair Value (מרחק באחוזים)
# ==================================================

def fair_value_pct_score(
    pct_diff: Optional[float],
    *,
    tol_pct: float = 0.02,
    max_pct: float = 0.20,
    shape: float = 1.0,
) -> float:
    """
    ציון לפי מרחק מה-Fair Value באחוזים (0–1).

    pct_diff:
        מרחק מה-FV באחוזים: 0.05 = +5%, -0.03 = -3%.
    tol_pct:
        טווח סבילה סביב 0, שבו הציון ≈ 1. לדוגמה: +/-2%.
    max_pct:
        מעבר למרחק זה → ציון 0 (למשל 20% deviation).
    shape:
        1.0 = ליניארי, >1 = ירידה חדה, <1 = רכה יותר.
    """
    if pct_diff is None or not np.isfinite(pct_diff):
        return 0.0

    d = abs(float(pct_diff))
    tol = float(max(tol_pct, 0.0))
    m = float(max(max_pct, tol + 1e-6))  # לוודא m > tol

    if d <= tol:
        base = 1.0
    elif d >= m:
        base = 0.0
    else:
        # ירידה ליניארית מ-1 ל-0
        base = (m - d) / (m - tol)

    base = max(0.0, min(1.0, base))
    if shape != 1.0 and base > 0.0:
        base = base ** float(shape)

    return float(base)


def build_fv_score_map(
    df_fv: pd.DataFrame,
    *,
    pair_col_candidates: Tuple[str, ...] = ("pair", "pair_label", "symbol_pair"),
    pct_diff_cols: Tuple[str, ...] = ("fv_pct_diff", "fair_value_pct_diff"),
) -> Dict[str, float]:
    """
    מקבל DataFrame עם מידע Fair Value ומחזיר map: pair → fv_score (0–1).

    df_fv צפוי להכיל:
        - עמודת pair (pair / pair_label / symbol_pair)
        - עמודת מרחק באחוזים (fv_pct_diff / fair_value_pct_diff)
    """
    if df_fv is None or df_fv.empty:
        return {}

    df = df_fv.copy()

    # pair column
    pair_col = None
    for cand in pair_col_candidates:
        if cand in df.columns:
            pair_col = cand
            break
    if pair_col is None:
        logger.warning("FV score: no pair column found.")
        return {}

    # pct diff column
    pct_col = None
    for cand in pct_diff_cols:
        if cand in df.columns:
            pct_col = cand
            break
    if pct_col is None:
        logger.warning("FV score: no pct diff column (fv_pct_diff/fair_value_pct_diff) found.")
        return {}

    scores: Dict[str, float] = {}
    vals_pct = pd.to_numeric(df[pct_col], errors="coerce")

    for p, diff in zip(df[pair_col].astype(str), vals_pct):
        s = fair_value_pct_score(diff, tol_pct=0.02, max_pct=0.20, shape=1.0)
        scores[str(p)] = s

    return scores


# =======================================================
# 1.4 שילוב: Base Param Score + Score File + Fair Value
# =======================================================

def apply_param_optimism_scan(
    df: pd.DataFrame,
    cfg: ParamOptimismConfig,
    *,
    param_cols: Optional[List[str]] = None,
    df_score_file: Optional[pd.DataFrame] = None,
    df_fair_value: Optional[pd.DataFrame] = None,
    w_base: float = 0.6,
    w_file: float = 0.2,
    w_fv: float = 0.2,
) -> pd.DataFrame:
    """
    מריץ Param Optimism על DataFrame של פרמטרים, ויוצר:

    param_optimism_score_base  : ציון לפי טווחי הפרמטרים בלבד (0–1).
    param_optimism_score_file  : ציון/בוסט לפי קובץ ציונים חיצוני (0–1).
    param_optimism_score_fv    : ציון לפי מרחק מה-FV באחוזים (0–1).
    param_optimism_score_total : ציון משוקלל כולל (0–1).

    df:
        חייב לכלול עמודת pair (pair / pair_label / symbol_pair) + פרמטרים.

    df_score_file (אופציונלי):
        pair + עמודת score (param_score_file / file_score / score / optimism_score).

    df_fair_value (אופציונלי):
        pair + fv_pct_diff / fair_value_pct_diff.

    המשקלים:
        w_base, w_file, w_fv – משקל היחסי של כל רכיב בציון הכולל.
    """
    if df is None or df.empty:
        return df

    df = df.copy()

    # זיהוי עמודת pair
    pair_col = None
    for cand in df.columns:
        if str(cand).lower() in ("pair", "pair_label", "symbol_pair"):
            pair_col = cand
            break
    if pair_col is None:
        pair_col = "pair"
        df[pair_col] = [f"PAIR_{i}" for i in range(len(df))]

    if param_cols is None:
        param_cols = list(cfg.ranges.keys())

    # ---- Base param score ----
    base_scores: List[float] = []
    per_param_details: List[Dict[str, float]] = []

    for _, row in df.iterrows():
        params = {name: row.get(name) for name in param_cols}
        total, per_param = cfg.score_vector(params)
        base_scores.append(total)
        per_param_details.append(per_param)

    df["param_optimism_score_base"] = base_scores
    df["param_optimism_details"] = per_param_details

    # ---- Score file component ----
    file_score_map: Dict[str, float] = {}
    if df_score_file is not None and not df_score_file.empty:
        sff = df_score_file.copy()
        score_col_file = None
        for cand in ("param_score_file", "file_score", "score", "optimism_score"):
            if cand in sff.columns:
                score_col_file = cand
                break
        if score_col_file is not None:
            pair_col_file = None
            for cand in sff.columns:
                if str(cand).lower() in ("pair", "pair_label", "symbol_pair"):
                    pair_col_file = cand
                    break
            if pair_col_file is not None:
                sff = sff[[pair_col_file, score_col_file]].dropna()
                vals = pd.to_numeric(sff[score_col_file], errors="coerce")
                vmin, vmax = float(vals.min()), float(vals.max())
                if vmax > vmin:
                    norm_vals = (vals - vmin) / (vmax - vmin)
                else:
                    norm_vals = vals * 0.0
                for p, v in zip(sff[pair_col_file].astype(str), norm_vals):
                    file_score_map[str(p)] = float(max(0.0, min(1.0, v)))

    file_scores: List[float] = []
    for _, row in df.iterrows():
        p = str(row[pair_col])
        file_scores.append(file_score_map.get(p, 0.0))
    df["param_optimism_score_file"] = file_scores

    # ---- Fair Value component ----
    fv_score_map: Dict[str, float] = {}
    if df_fair_value is not None and not df_fair_value.empty:
        fv_score_map = build_fv_score_map(df_fair_value)

    fv_scores: List[float] = []
    for _, row in df.iterrows():
        p = str(row[pair_col])
        fv_scores.append(fv_score_map.get(p, 0.0))
    df["param_optimism_score_fv"] = fv_scores

    # ---- Total combined score ----
    w_base = float(max(0.0, w_base))
    w_file = float(max(0.0, w_file))
    w_fv = float(max(0.0, w_fv))
    total_w = w_base + w_file + w_fv
    if total_w <= 0:
        w_base = 1.0
        total_w = 1.0

    base_arr = df["param_optimism_score_base"].astype(float).to_numpy()
    file_arr = df["param_optimism_score_file"].astype(float).to_numpy()
    fv_arr = df["param_optimism_score_fv"].astype(float).to_numpy()

    total_score = (w_base * base_arr + w_file * file_arr + w_fv * fv_arr) / total_w
    df["param_optimism_score_total"] = np.clip(total_score, 0.0, 1.0)

    return df


# ==========================================================
# 1.5 "וו" ל-ML — התאמת טווחים לפי היסטוריית אופטימיזציה (hook)
# ==========================================================

def adapt_param_optimism_from_history(
    cfg: ParamOptimismConfig,
    df_history: pd.DataFrame,
    *,
    score_col: str = "Score",
    top_quantile: float = 0.2,
) -> ParamOptimismConfig:
    """
    Hook עתידי ל-ML/Meta-Optimization:

    מתוך df_history (למשל opt_df) אפשר:
    - לקחת Top-X% ריצות.
    - להסתכל על התפלגות הפרמטרים.
    - לעדכן את הטווחים (optimal_min/max) לפי quantiles.

    כרגע מימוש זה:
    - לוקח את top_quantile לפי Score,
    - לכל פרמטר שקיים ב-cfg.ranges,
      שם optimal_min/max לפי q25/q75, ושומר hard_min/max.

    אפשר להחליף בעתיד ללוגיקה מורכבת יותר (ML אמיתי).
    """
    if df_history is None or df_history.empty:
        return cfg

    if score_col not in df_history.columns:
        return cfg

    try:
        df_sorted = df_history.sort_values(score_col, ascending=False)
    except Exception:
        df_sorted = df_history.copy()

    k = max(10, int(len(df_sorted) * float(top_quantile)))
    df_top = df_sorted.head(k)

    new_ranges: Dict[str, ParamOptimalRange] = {}
    for name, rng in cfg.ranges.items():
        if name not in df_top.columns:
            new_ranges[name] = rng
            continue
        vals = pd.to_numeric(df_top[name], errors="coerce").dropna()
        if vals.empty:
            new_ranges[name] = rng
            continue
        q25 = float(vals.quantile(0.25))
        q75 = float(vals.quantile(0.75))
        # נשמור hard_min/max קיימים אך נעדכן את ה-optimal
        new_ranges[name] = ParamOptimalRange(
            name=name,
            hard_min=rng.hard_min,
            optimal_min=q25,
            optimal_max=q75,
            hard_max=rng.hard_max,
            weight=rng.weight,
            shape=rng.shape,
        )

    logger.info("ParamOptimismConfig adapted from history (top %.0f%%).", top_quantile * 100)
    return ParamOptimismConfig(
        ranges=new_ranges,
        source="ml_adapted",
        profile=cfg.profile,
    )

# =============================================
# Part 2/5 — Fundamental Optimism & Quality
# =============================================

from dataclasses import dataclass
from typing import Literal

FundMode = Literal["range", "higher_better", "lower_better"]


@dataclass
class FundamentalFactorSpec:
    """
    פקטור פנדומנטלי בודד:

    name:
        שם הפקטור, לדוגמה: "pe", "roe", "debt_to_equity", "eps_growth_5y".

    mode:
        "range"          → יש טווח אופטימלי (כמו ParamOptimalRange).
        "higher_better"  → ערך גבוה יותר עד גבול מסוים → ציון גבוה.
        "lower_better"   → ערך נמוך יותר עד גבול מסוים → ציון גבוה.

    optimal_min / optimal_max:
        עבור mode="range":
            - בתוך הטווח → ציון ≈ 1.
            - מחוץ לו → ציון יורד.
        עבור modes אחרים:
            משמש כנקודת רפרנס/סטוריישן (למשל ערך "טוב מאוד").

    hard_min / hard_max:
        גבולות תחתונים/עליונים:
        - מחוץ לטווח הזה → ציון 0.
        - בין hard לבין optimal → ציון יורד חלק.

    weight:
        משקל הפקטור בציון הפנדומנטלי הכולל (לא כל פקטור חשוב באותה מידה).

    shape:
        1.0 = ליניארי, >1 = יותר חדה, <1 = יותר רכה.

    Notes:
    ------
    - ב"range": מתאים למקרים כמו P/E "נורמלי" (לא גבוה מדי, לא נמוך מדי בצורה חשודה).
    - ב"higher_better": מתאים ל-ROE, EPS Growth, Margin.
    - ב"lower_better": מתאים ל-Debt/Equity, Volatility, Payout Ratio (במקרים מסוימים).
    """

    name: str
    mode: FundMode
    hard_min: float
    optimal_min: float
    optimal_max: float
    hard_max: float
    weight: float = 1.0
    shape: float = 1.0

    def score(self, value: Optional[float]) -> float:
        if value is None or not np.isfinite(value):
            return 0.0

        x = float(value)
        hmin = float(self.hard_min)
        omin = float(self.optimal_min)
        omax = float(self.optimal_max)
        hmax = float(self.hard_max)

        if hmax <= hmin:
            return 0.0

        # ערכים מחוץ לגבולות קשיחים → 0
        if x <= hmin or x >= hmax:
            base = 0.0
        elif self.mode == "range":
            # דומה ל-ParamOptimalRange: "פעמון שטוח"
            if omin <= x <= omax:
                base = 1.0
            elif hmin < x < omin:
                base = (x - hmin) / (omin - hmin)
            elif omax < x < hmax:
                base = (hmax - x) / (hmax - omax)
            else:
                base = 0.0
        elif self.mode == "higher_better":
            # hmin → 0, omin→0.5, omax→1, hmax→0
            if x <= hmin:
                base = 0.0
            elif x >= omax:
                base = 1.0
            elif x <= omin:
                # scale up בין hmin→omin
                base = 0.5 * (x - hmin) / (omin - hmin)
            else:  # בין omin ל-omax
                base = 0.5 + 0.5 * (x - omin) / (omax - omin)
        else:  # lower_better
            # hmin→1, omin→1, omax→0.5, hmax→0
            if x >= hmax:
                base = 0.0
            elif x <= omin:
                base = 1.0
            elif x <= omax:
                base = 0.5 + 0.5 * (omax - x) / (omax - omin)
            else:  # בין omax ל-hmax
                base = 0.5 * (hmax - x) / (hmax - omax)

        base = max(0.0, min(1.0, base))
        if self.shape != 1.0 and base > 0.0:
            base = base ** float(self.shape)
        return float(base)


@dataclass
class FundamentalOptimismConfig:
    """
    קונפיגורציה ל-Fundamental Optimism:

    factors:
        name → FundamentalFactorSpec

    profile:
        "default" / "defensive" / "aggressive" (הרעיון זהה לחלק 1).

    source:
        "default_profile", "manual", "ml_adapted", וכו'.
    """

    factors: Dict[str, FundamentalFactorSpec]
    profile: str = "default"
    source: str = "default_profile"

    @classmethod
    def from_simple_dict(
        cls,
        data: Dict[str, Dict[str, Any]],
        *,
        profile: str = "custom",
        source: str = "manual",
    ) -> "FundamentalOptimismConfig":
        factors: Dict[str, FundamentalFactorSpec] = {}
        for name, cfg in data.items():
            try:
                factors[name] = FundamentalFactorSpec(
                    name=name,
                    mode=str(cfg.get("mode", "range")).lower(),  # type: ignore[arg-type]
                    hard_min=float(cfg["hard_min"]),
                    optimal_min=float(cfg["optimal_min"]),
                    optimal_max=float(cfg["optimal_max"]),
                    hard_max=float(cfg["hard_max"]),
                    weight=float(cfg.get("weight", 1.0)),
                    shape=float(cfg.get("shape", 1.0)),
                )
            except Exception as e:
                logger.warning("Invalid fundamental factor config for %s: %s", name, e)
        return cls(factors=factors, profile=profile, source=source)

    @classmethod
    def default_for_profile(cls, profile: str = "default") -> "FundamentalOptimismConfig":
        """
        יוצר קונפיגורציה דיפולטית לפקטורים פנדומנטליים לפי פרופיל.

        הגיון (גבוה-רמה, מבוסס פרקטיקה שלקרן איכות/ערך):

        default:
            - P/E: טווח "בריא" בערך [10,25], לא נמוך מדי (אולי מלכודת ערך),
              ולא גבוה מדי (ציפיות קיצוניות).
            - ROE: higher_better, טווח מועדף נגיד [8%,25%].
            - Debt/Equity: lower_better, נמוך מ-0.5 עדיף, מעל 2 בעייתי.
            - EPS Growth 5Y: higher_better, 0–5% סביר, 5–15% טוב, מעבר לזה סטוריישן.
        defensive:
            יותר דגש על:
            - Debt נמוך,
            - ROE יציב,
            - P/E לא גבוה.

        aggressive:
            יותר דגש על:
            - Growth (EPS, Revenue),
            - ROE גבוה,
            - פחות רגיש ל-P/E גבוה.

        בהמשך:
            - ניתן לעדכן את הטווחים מתוך היסטוריית אופטימיזציה (adapt_fundamentals_from_history).
        """
        p = profile.lower().strip()
        if p not in {"default", "defensive", "aggressive"}:
            p = "default"

        if p == "defensive":
            raw = {
                "pe": {
                    "mode": "range",
                    "hard_min": 5.0,
                    "optimal_min": 10.0,
                    "optimal_max": 20.0,
                    "hard_max": 30.0,
                    "weight": 2.0,
                    "shape": 1.0,
                },
                "roe": {
                    "mode": "higher_better",
                    "hard_min": 0.0,
                    "optimal_min": 0.08,   # 8%
                    "optimal_max": 0.20,   # 20%
                    "hard_max": 0.35,
                    "weight": 2.5,
                    "shape": 1.0,
                },
                "debt_to_equity": {
                    "mode": "lower_better",
                    "hard_min": 0.0,
                    "optimal_min": 0.0,
                    "optimal_max": 0.5,
                    "hard_max": 2.0,
                    "weight": 2.5,
                    "shape": 1.2,
                },
                "eps_growth_5y": {
                    "mode": "higher_better",
                    "hard_min": -0.10,
                    "optimal_min": 0.02,
                    "optimal_max": 0.10,
                    "hard_max": 0.25,
                    "weight": 1.5,
                    "shape": 1.0,
                },
            }
        elif p == "aggressive":
            raw = {
                "pe": {
                    "mode": "range",
                    "hard_min": 5.0,
                    "optimal_min": 15.0,
                    "optimal_max": 35.0,
                    "hard_max": 60.0,
                    "weight": 1.0,
                    "shape": 1.0,
                },
                "roe": {
                    "mode": "higher_better",
                    "hard_min": 0.0,
                    "optimal_min": 0.10,
                    "optimal_max": 0.25,
                    "hard_max": 0.45,
                    "weight": 2.0,
                    "shape": 1.0,
                },
                "debt_to_equity": {
                    "mode": "lower_better",
                    "hard_min": 0.0,
                    "optimal_min": 0.0,
                    "optimal_max": 1.0,
                    "hard_max": 3.0,
                    "weight": 1.5,
                    "shape": 1.0,
                },
                "eps_growth_5y": {
                    "mode": "higher_better",
                    "hard_min": -0.20,
                    "optimal_min": 0.05,
                    "optimal_max": 0.20,
                    "hard_max": 0.40,
                    "weight": 2.5,
                    "shape": 1.0,
                },
            }
        else:  # default
            raw = {
                "pe": {
                    "mode": "range",
                    "hard_min": 5.0,
                    "optimal_min": 10.0,
                    "optimal_max": 25.0,
                    "hard_max": 40.0,
                    "weight": 1.5,
                    "shape": 1.0,
                },
                "roe": {
                    "mode": "higher_better",
                    "hard_min": 0.0,
                    "optimal_min": 0.08,
                    "optimal_max": 0.20,
                    "hard_max": 0.40,
                    "weight": 2.0,
                    "shape": 1.0,
                },
                "debt_to_equity": {
                    "mode": "lower_better",
                    "hard_min": 0.0,
                    "optimal_min": 0.0,
                    "optimal_max": 0.8,
                    "hard_max": 2.5,
                    "weight": 2.0,
                    "shape": 1.0,
                },
                "eps_growth_5y": {
                    "mode": "higher_better",
                    "hard_min": -0.15,
                    "optimal_min": 0.03,
                    "optimal_max": 0.12,
                    "hard_max": 0.30,
                    "weight": 1.8,
                    "shape": 1.0,
                },
            }

        cfg = cls.from_simple_dict(raw, profile=p, source="default_profile")
        logger.info("FundamentalOptimismConfig default_for_profile=%s created.", p)
        return cfg

    def score_vector(self, factors: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        """
        מחשב ציון פנדומנטלי משוקלל (0–1) + ציוני per-factor.
        """
        if not self.factors:
            return 0.0, {}

        details: Dict[str, float] = {}
        ws = 0.0
        ssum = 0.0

        for name, spec in self.factors.items():
            v = factors.get(name)
            s = spec.score(v)
            details[name] = s
            w = float(spec.weight)
            ssum += s * w
            ws += abs(w)

        if ws <= 0:
            total = 0.0
        else:
            total = ssum / ws

        total = float(max(0.0, min(1.0, total)))
        return total, details


def apply_fundamental_optimism_scan(
    df: pd.DataFrame,
    cfg: FundamentalOptimismConfig,
    *,
    factor_cols_map: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    מריץ Fundamental Optimism על DataFrame של זוגות.

    df:
        צפוי לכלול עמודות פקטורים פנדומנטליים ברמת זוג:
        לדוגמה:
            "pe", "roe", "debt_to_equity", "eps_growth_5y"

        אם factor_cols_map לא None:
            מיפוי בין שמות הפקטורים בקונפיג לשמות העמודות ב-df, לדוגמה:
                {"pe": "pair_pe", "roe": "pair_roe"}

    מוסיף:
        - fundamental_score_total   (0–1)
        - fundamental_score_details (dict per-factor)
    """
    if df is None or df.empty:
        return df

    df = df.copy()

    if factor_cols_map is None:
        factor_cols_map = {name: name for name in cfg.factors.keys()}

    scores: List[float] = []
    details_list: List[Dict[str, float]] = []

    for _, row in df.iterrows():
        fvals: Dict[str, Any] = {}
        for name, spec in cfg.factors.items():
            col = factor_cols_map.get(name, name)
            fvals[name] = row.get(col)
        total, d = cfg.score_vector(fvals)
        scores.append(total)
        details_list.append(d)

    df["fundamental_score_total"] = scores
    df["fundamental_score_details"] = details_list
    return df


def adapt_fundamentals_from_history(
    cfg: FundamentalOptimismConfig,
    df_history: pd.DataFrame,
    *,
    score_col: str = "Score",
    top_quantile: float = 0.2,
) -> FundamentalOptimismConfig:
    """
    Hook ל-ML/Meta-Optimization לחלק הפנדומנטלי:

    df_history:
        DataFrame של תוצאות (למשל opt_df) עם עמודת Score
        ועמודות פנדומנטליות רלוונטיות (pe, roe, ...).

    לוגיקה:
        - לוקחים Top top_quantile לפי Score.
        - לכל פקטור שקיים ב-cfg.factors וגם ב-df_history:
            * אם mode="range" → optimal_min/max = q25/q75.
            * אם higher/lower → מעדכנים רק optimal_max/min לפי quantiles, ומשאירים hard_min/max.
    """
    if df_history is None or df_history.empty or score_col not in df_history.columns:
        return cfg

    try:
        df_sorted = df_history.sort_values(score_col, ascending=False)
    except Exception:
        df_sorted = df_history.copy()

    k = max(10, int(len(df_sorted) * float(top_quantile)))
    df_top = df_sorted.head(k)

    new_factors: Dict[str, FundamentalFactorSpec] = {}
    for name, spec in cfg.factors.items():
        if name not in df_top.columns:
            new_factors[name] = spec
            continue

        vals = pd.to_numeric(df_top[name], errors="coerce").dropna()
        if vals.empty:
            new_factors[name] = spec
            continue

        q25 = float(vals.quantile(0.25))
        q75 = float(vals.quantile(0.75))

        if spec.mode == "range":
            new_factors[name] = FundamentalFactorSpec(
                name=name,
                mode=spec.mode,
                hard_min=spec.hard_min,
                optimal_min=q25,
                optimal_max=q75,
                hard_max=spec.hard_max,
                weight=spec.weight,
                shape=spec.shape,
            )
        elif spec.mode == "higher_better":
            # נזיז את optimal_min/max כלפי ה-Top
            new_factors[name] = FundamentalFactorSpec(
                name=name,
                mode=spec.mode,
                hard_min=spec.hard_min,
                optimal_min=q25,
                optimal_max=q75,
                hard_max=spec.hard_max,
                weight=spec.weight,
                shape=spec.shape,
            )
        else:  # lower_better
            # ערכים טובים יותר נמצאים נמוך → נשתמש ב-q25 "למטה"
            new_factors[name] = FundamentalFactorSpec(
                name=name,
                mode=spec.mode,
                hard_min=spec.hard_min,
                optimal_min=spec.optimal_min,
                optimal_max=q75,
                hard_max=spec.hard_max,
                weight=spec.weight,
                shape=spec.shape,
            )

    logger.info("FundamentalOptimismConfig adapted from history (top %.0f%%).", top_quantile * 100)
    return FundamentalOptimismConfig(
        factors=new_factors,
        profile=cfg.profile,
        source="ml_adapted",
    )

# =============================================
# Part 3/5 — Macro Fit Optimism (HF-grade)
# =============================================

from typing import Literal

MacroMode = Literal["range", "higher_better", "lower_better", "enum_regime", "beta_exposure"]


@dataclass
class MacroFactorSpec:
    """
    פקטור מאקרו לתזמון / התאמה ברמת זוג (Pair Macro Fit).

    דוגמאות לפקטורים:
        - global_regime      : 'Risk-On' / 'Risk-Off' / 'Neutral' / 'Crisis'
        - local_regime       : משטר סקטור/מדינה, למשל 'China_slowdown', 'Tech_boom'
        - level_rates        : רמת הריבית (למשל 10Y yield)
        - slope_curve        : שיפוע עקום (10Y-2Y)
        - level_vol          : VIX / Volatility
        - credit_spread      : HY-OAS / IG-OAS
        - beta_equity        : בטא למדד מניות (SPX/World)
        - beta_rates         : בטא לריביות
        - beta_vol           : בטא ל-VIX
        - beta_fx            : בטא ל-FX index

    mode:
        "range"
            → יש טווח אופטימלי [optimal_min, optimal_max].
        "higher_better"
            → ערך גבוה עד גבול מסוים → יותר טוב.
        "lower_better"
            → ערך נמוך עד גבול מסוים → יותר טוב.
        "enum_regime"
            → הערכת משטר לפי מילון enum_scores (מבוסס label string).
        "beta_exposure"
            → בטא מבוקשת; מתגמל בטא "מתאימה" לפרופיל המאקרו.

    enum_scores:
        משמש כאשר mode="enum_regime":
            dict[label_lower → score 0–1], למשל:
                {
                  "risk-on": 0.8,
                  "risk-off": 0.4,
                  "crisis": 0.1,
                  "neutral": 0.6,
                }

    horizon:
        "short" / "medium" / "long" — למיפוי פקטורים שמתאימים לזמני אחזקה שונים.

    style_tags:
        רשימת תוויות שמסבירות לאיזה סוג אסטרטגיה הפקטור רלוונטי:
        לדוגמה: ["mean_reversion", "carry", "trend"].

    pair_side:
        "long", "short", "both" — אם לפקטור יש משמעות שונה לסטרטגיה שמועדפת long/short.

    Remarks:
    --------
    - בשילוב עם macro_context גלובלי, ניתן להכיל גם פקטורים שרלוונטיים לכל המערכת
      (למשל yield_curve_slope, global_regime), וגם פקטורים ברמת זוג (row).
    """

    name: str
    mode: MacroMode
    hard_min: float = 0.0
    optimal_min: float = 0.0
    optimal_max: float = 0.0
    hard_max: float = 0.0
    weight: float = 1.0
    shape: float = 1.0
    enum_scores: Optional[Dict[str, float]] = None
    horizon: str = "medium"
    style_tags: Optional[List[str]] = None
    pair_side: str = "both"  # "long", "short", "both"

    def score(self, value: Any, *, beta_target: Optional[float] = None) -> float:
        """
        מחשב ציון 0–1 לערך של הפקטור.

        עבור mode="beta_exposure":
            value = beta בפועל, beta_target = רמת בטא רצויה (אם ידועה).
        """
        if self.mode == "enum_regime":
            return self._score_enum_regime(value)

        if self.mode == "beta_exposure":
            return self._score_beta_exposure(value, beta_target=beta_target)

        if value is None or not np.isfinite(value):
            return 0.0

        x = float(value)
        hmin = float(self.hard_min)
        omin = float(self.optimal_min)
        omax = float(self.optimal_max)
        hmax = float(self.hard_max)

        if hmax <= hmin:
            return 0.0

        # מחוץ לגבולות קשיחים → 0
        if x <= hmin or x >= hmax:
            base = 0.0
        elif self.mode == "range":
            if omin <= x <= omax:
                base = 1.0
            elif hmin < x < omin:
                base = (x - hmin) / (omin - hmin)
            elif omax < x < hmax:
                base = (hmax - x) / (hmax - omax)
            else:
                base = 0.0
        elif self.mode == "higher_better":
            if x <= hmin:
                base = 0.0
            elif x >= omax:
                base = 1.0
            elif x <= omin:
                base = 0.5 * (x - hmin) / (omin - hmin)
            else:  # בין omin ל-omax
                base = 0.5 + 0.5 * (x - omin) / (omax - omin)
        else:  # lower_better
            if x >= hmax:
                base = 0.0
            elif x <= omin:
                base = 1.0
            elif x <= omax:
                base = 0.5 + 0.5 * (omax - x) / (omax - omin)
            else:
                base = 0.5 * (hmax - x) / (hmax - omax)

        base = max(0.0, min(1.0, base))
        if self.shape != 1.0 and base > 0.0:
            base = base ** float(self.shape)

        return float(base)

    def _score_enum_regime(self, value: Any) -> float:
        """
        mode="enum_regime" — ציון לפי תווית משטר (Risk-On/Off/Neutral/Crisis, וכו').
        """
        if not self.enum_scores:
            return 0.5  # נייטרלי אם לא הוגדר כלום

        try:
            lab = str(value or "").lower().strip()
        except Exception:
            return 0.0

        if not lab:
            return 0.5

        # match לפי substring (למשל 'risk-on', 'risk on', 'risk_on'...)
        best = None
        for key, val in self.enum_scores.items():
            if key in lab:
                best = float(val)
                break

        if best is None:
            return 0.5
        return float(np.clip(best, 0.0, 1.0))

    def _score_beta_exposure(self, value: Any, *, beta_target: Optional[float]) -> float:
        """
        mode="beta_exposure" — ציון לפי מרחק מ-beta_target.

        אם beta_target=None → מניחים יעד 0 (ניטרלי).

        לוגיקה:
            distance = |beta - beta_target|
            distance 0 → score=1
            אם distance >= hard_max → score=0
            אחרת — יורד ליניארית ומתעגל לפי shape.
        """
        if value is None or not np.isfinite(value):
            return 0.0

        beta = float(value)
        if beta_target is None:
            beta_target = 0.0

        d = abs(beta - float(beta_target))
        threshold = float(self.hard_max) if self.hard_max > 0 else 2.0

        if d >= threshold:
            base = 0.0
        else:
            base = 1.0 - d / threshold

        base = max(0.0, min(1.0, base))
        if self.shape != 1.0 and base > 0.0:
            base = base ** float(self.shape)

        return float(base)


@dataclass
class MacroOptimismConfig:
    """
    קונפיגורציה ל-Macro Fit Optimism.

    factors:
        name → MacroFactorSpec
    profile:
        "default" / "defensive" / "aggressive"
    source:
        "default_profile" / "manual" / "ml_adapted"
    """

    factors: Dict[str, MacroFactorSpec]
    profile: str = "default"
    source: str = "default_profile"

    @classmethod
    def from_simple_dict(
        cls,
        data: Dict[str, Dict[str, Any]],
        *,
        profile: str = "custom",
        source: str = "manual",
    ) -> "MacroOptimismConfig":
        factors: Dict[str, MacroFactorSpec] = {}
        for name, cfg in data.items():
            try:
                enum_scores = cfg.get("enum_scores", None)
                factors[name] = MacroFactorSpec(
                    name=name,
                    mode=str(cfg.get("mode", "range")).lower(),  # type: ignore[arg-type]
                    hard_min=float(cfg.get("hard_min", 0.0)),
                    optimal_min=float(cfg.get("optimal_min", 0.0)),
                    optimal_max=float(cfg.get("optimal_max", 0.0)),
                    hard_max=float(cfg.get("hard_max", 0.0)),
                    weight=float(cfg.get("weight", 1.0)),
                    shape=float(cfg.get("shape", 1.0)),
                    enum_scores=enum_scores,
                    horizon=str(cfg.get("horizon", "medium")),
                    style_tags=list(cfg.get("style_tags", []) or []),
                    pair_side=str(cfg.get("pair_side", "both")),
                )
            except Exception as e:
                logger.warning("Invalid macro factor config for %s: %s", name, e)
        return cls(factors=factors, profile=profile, source=source)

    @classmethod
    def default_for_profile(cls, profile: str = "default") -> "MacroOptimismConfig":
        """
        דיפולט מקצועי למאקרו, לפי פרופיל:

        default:
            - global_regime: אוהבים neutral / light risk-on.
            - level_vol   : lower_better, טווח רצוי ~ [10,25] (VIX).
            - credit_spread: lower_better, HY-OAS נמוך.
            - beta_equity  : beta_exposure → יעד 0.5~0.8.
            - beta_vol     : beta_exposure → יעד קרוב ל-0 (לא בטא גבוהה ל-Vol).

        defensive:
            - מעדיף regimes risk-off/defensive.
            - מאוד עונש על vol גבוה / credit spreads גבוהים.
            - beta_equity נמוך, beta_vol נמוך מאוד.

        aggressive:
            - מוטה risk-on, סובל קצת יותר vol/spreads,
            - מעדיף beta_equity גבוה יותר (למשל 0.8–1.2).

        בהמשך ניתן לעדכן את הטווחים מתוך adapt_macro_from_history.
        """
        p = profile.lower().strip()
        if p not in {"default", "defensive", "aggressive"}:
            p = "default"

        if p == "defensive":
            raw = {
                "global_regime": {
                    "mode": "enum_regime",
                    "weight": 2.0,
                    "shape": 1.0,
                    "enum_scores": {
                        "risk-off": 0.9,
                        "defensive": 0.85,
                        "neutral": 0.6,
                        "risk-on": 0.4,
                        "crisis": 0.3,
                    },
                    "horizon": "medium",
                },
                "level_vol": {
                    "mode": "lower_better",
                    "hard_min": 8.0,
                    "optimal_min": 10.0,
                    "optimal_max": 20.0,
                    "hard_max": 35.0,
                    "weight": 2.0,
                    "shape": 1.2,
                    "horizon": "short",
                },
                "credit_spread": {
                    "mode": "lower_better",
                    "hard_min": 0.0,
                    "optimal_min": 0.0,
                    "optimal_max": 0.04,
                    "hard_max": 0.10,
                    "weight": 2.0,
                    "shape": 1.2,
                    "horizon": "medium",
                },
                "beta_equity": {
                    "mode": "beta_exposure",
                    "hard_max": 1.0,   # threshold for distance
                    "weight": 1.5,
                    "shape": 1.0,
                    "horizon": "medium",
                },
                "beta_vol": {
                    "mode": "beta_exposure",
                    "hard_max": 1.0,
                    "weight": 2.0,
                    "shape": 1.5,
                    "horizon": "short",
                },
            }
        elif p == "aggressive":
            raw = {
                "global_regime": {
                    "mode": "enum_regime",
                    "weight": 2.0,
                    "shape": 1.0,
                    "enum_scores": {
                        "risk-on": 0.9,
                        "neutral": 0.7,
                        "defensive": 0.5,
                        "risk-off": 0.4,
                        "crisis": 0.2,
                    },
                    "horizon": "medium",
                },
                "level_vol": {
                    "mode": "range",
                    "hard_min": 8.0,
                    "optimal_min": 12.0,
                    "optimal_max": 25.0,
                    "hard_max": 40.0,
                    "weight": 1.5,
                    "shape": 1.0,
                    "horizon": "short",
                },
                "credit_spread": {
                    "mode": "range",
                    "hard_min": 0.02,
                    "optimal_min": 0.03,
                    "optimal_max": 0.06,
                    "hard_max": 0.12,
                    "weight": 1.0,
                    "shape": 1.0,
                    "horizon": "medium",
                },
                "beta_equity": {
                    "mode": "beta_exposure",
                    "hard_max": 1.5,
                    "weight": 2.0,
                    "shape": 1.0,
                    "horizon": "medium",
                },
                "beta_vol": {
                    "mode": "beta_exposure",
                    "hard_max": 1.5,
                    "weight": 1.5,
                    "shape": 1.0,
                    "horizon": "short",
                },
            }
        else:  # default
            raw = {
                "global_regime": {
                    "mode": "enum_regime",
                    "weight": 1.5,
                    "shape": 1.0,
                    "enum_scores": {
                        "risk-on": 0.75,
                        "neutral": 0.7,
                        "defensive": 0.65,
                        "risk-off": 0.5,
                        "crisis": 0.3,
                    },
                    "horizon": "medium",
                },
                "level_vol": {
                    "mode": "lower_better",
                    "hard_min": 8.0,
                    "optimal_min": 10.0,
                    "optimal_max": 22.0,
                    "hard_max": 40.0,
                    "weight": 1.5,
                    "shape": 1.0,
                    "horizon": "short",
                },
                "credit_spread": {
                    "mode": "lower_better",
                    "hard_min": 0.0,
                    "optimal_min": 0.0,
                    "optimal_max": 0.05,
                    "hard_max": 0.12,
                    "weight": 1.5,
                    "shape": 1.0,
                    "horizon": "medium",
                },
                "beta_equity": {
                    "mode": "beta_exposure",
                    "hard_max": 1.2,
                    "weight": 1.8,
                    "shape": 1.0,
                    "horizon": "medium",
                },
                "beta_vol": {
                    "mode": "beta_exposure",
                    "hard_max": 1.2,
                    "weight": 1.8,
                    "shape": 1.2,
                    "horizon": "short",
                },
            }

        cfg = cls.from_simple_dict(raw, profile=p, source="default_profile")
        logger.info("MacroOptimismConfig default_for_profile=%s created.", p)
        return cfg

    def score_vector(
        self,
        row: pd.Series,
        *,
        factor_cols_map: Optional[Dict[str, str]] = None,
        macro_context: Optional[Dict[str, Any]] = None,
        beta_targets: Optional[Dict[str, float]] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """
        מחשב ציון מאקרו משוקלל (0–1) + פירוט לכל פקטור.

        row:
            שורה אחת של df שמייצגת זוג (יכול להכיל בטאיות, labels וכו').
        factor_cols_map:
            how factor_name → column_name in df, למשל {"global_regime": "macro_regime"}.
        macro_context:
            קונטקסט גלובלי שנשמר ע"י macro_engine (לדוגמה st.session_state["macro_context"]),
            יכול להכיל:
                - "global_regime"
                - "vix_level"
                - "hy_oas"
                וכו' — נעדיף להשתמש בו אם אין בערך בשורה.
        beta_targets:
            מילון factor_name → beta_target; אם לא קיים, משתמשים ב-Dfault 0 או 0.5 וכו'.
        """
        if not self.factors:
            return 0.0, {}

        if factor_cols_map is None:
            factor_cols_map = {name: name for name in self.factors.keys()}
        if macro_context is None:
            macro_context = {}
        if beta_targets is None:
            beta_targets = {}

        details: Dict[str, float] = {}
        ws = 0.0
        ssum = 0.0

        for name, spec in self.factors.items():
            col = factor_cols_map.get(name, name)

            # value יכול להגיע מה-row או מה-macro_context
            if col in row.index and row.get(col) is not None:
                v = row.get(col)
            else:
                v = macro_context.get(name)

            beta_target = beta_targets.get(name)
            s = spec.score(v, beta_target=beta_target)
            details[name] = s
            w = float(spec.weight)
            ssum += s * w
            ws += abs(w)

        if ws <= 0:
            total = 0.0
        else:
            total = ssum / ws

        total = float(max(0.0, min(1.0, total)))
        return total, details


def apply_macro_optimism_scan(
    df: pd.DataFrame,
    cfg: MacroOptimismConfig,
    *,
    factor_cols_map: Optional[Dict[str, str]] = None,
    macro_context: Optional[Dict[str, Any]] = None,
    beta_targets: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    מוסיף לכל שורה:
        - macro_score_total
        - macro_score_details

    משלב מידע:
        - ברמת pair (row), מתוך df.
        - ברמת macro_context גלובלי (למשל st.session_state["macro_context"]).
    """
    if df is None or df.empty:
        return df

    df = df.copy()
    scores: List[float] = []
    details_list: List[Dict[str, float]] = []

    if macro_context is None:
        macro_context = {}
        # אם שמרת macro_context ב-session → נשתמש בו
        try:
            macro_context = st.session_state.get("macro_context", {}) or {}
        except Exception:
            macro_context = {}

    for _, row in df.iterrows():
        total, d = cfg.score_vector(
            row,
            factor_cols_map=factor_cols_map,
            macro_context=macro_context,
            beta_targets=beta_targets,
        )
        scores.append(total)
        details_list.append(d)

    df["macro_score_total"] = scores
    df["macro_score_details"] = details_list
    return df


def adapt_macro_from_history(
    cfg: MacroOptimismConfig,
    df_history: pd.DataFrame,
    *,
    score_col: str = "Score",
    top_quantile: float = 0.2,
) -> MacroOptimismConfig:
    """
    Hook ל-ML/Meta-Optimization לחלק המאקרו:

    df_history:
        DataFrame של ריצות (למשל opt_df) עם Score
        ועמודות מאקרו רלוונטיות (global_regime, level_vol, credit_spread, beta_equity, ...).

    לוגיקה:
        - לוקחים Top-X% לפי Score.
        - עבור factors מספריים:
            * mode="range"/"higher"/"lower"/"beta_exposure" → מעדכנים optimal_min/max לפי q25/q75.
        - עבור enum_regime:
            * בונים enum_scores לפי תדירויות של labels בין הטובים.
    """
    if df_history is None or df_history.empty or score_col not in df_history.columns:
        return cfg

    try:
        df_sorted = df_history.sort_values(score_col, ascending=False)
    except Exception:
        df_sorted = df_history.copy()

    k = max(10, int(len(df_sorted) * float(top_quantile)))
    df_top = df_sorted.head(k)

    new_factors: Dict[str, MacroFactorSpec] = {}

    for name, spec in cfg.factors.items():
        if spec.mode == "enum_regime":
            col = name
            if col not in df_top.columns:
                new_factors[name] = spec
                continue
            labs = df_top[col].astype(str).str.lower()
            counts = labs.value_counts(normalize=True)
            enum_scores: Dict[str, float] = {}
            for lab, freq in counts.items():
                # regimes שהופיעו יותר ב-top יקבלו ציון גבוה יותר
                enum_scores[lab] = float(np.clip(freq * 1.2, 0.1, 1.0))
            new_factors[name] = MacroFactorSpec(
                name=name,
                mode=spec.mode,
                hard_min=spec.hard_min,
                optimal_min=spec.optimal_min,
                optimal_max=spec.optimal_max,
                hard_max=spec.hard_max,
                weight=spec.weight,
                shape=spec.shape,
                enum_scores=enum_scores,
                horizon=spec.horizon,
                style_tags=spec.style_tags,
                pair_side=spec.pair_side,
            )
        else:
            col = name
            if col not in df_top.columns:
                new_factors[name] = spec
                continue
            vals = pd.to_numeric(df_top[col], errors="coerce").dropna()
            if vals.empty:
                new_factors[name] = spec
                continue
            q25 = float(vals.quantile(0.25))
            q75 = float(vals.quantile(0.75))
            new_factors[name] = MacroFactorSpec(
                name=name,
                mode=spec.mode,
                hard_min=spec.hard_min,
                optimal_min=q25,
                optimal_max=q75,
                hard_max=spec.hard_max,
                weight=spec.weight,
                shape=spec.shape,
                enum_scores=spec.enum_scores,
                horizon=spec.horizon,
                style_tags=spec.style_tags,
                pair_side=spec.pair_side,
            )

    logger.info("MacroOptimismConfig adapted from history (top %.0f%%).", top_quantile * 100)
    return MacroOptimismConfig(
        factors=new_factors,
        profile=cfg.profile,
        source="ml_adapted",
    )

# =============================================
# Part 4/5 — Metrics / Performance Optimism (HF-grade)
# =============================================

MetricsMode = Literal["range", "higher_better", "lower_better"]

@dataclass
class MetricsFactorSpec:
    """
    פקטור ביצועים / סיכון אחד, לדוגמה:

    name:
        "Sharpe", "Sortino", "Calmar", "Drawdown", "TailRisk",
        "WinRate", "DSR", "Trades", "Skew", "Kurtosis",
        "MaxConsecLosses", "UlcerIndex", "ExposurePct", "AvgBarsHeld", "Profit".

    mode:
        "range"
            → יש טווח אופטימלי (Sharpe 1–3, WinRate 0.5–0.7 וכו').
        "higher_better"
            → ערך גבוה יותר טוב (Sharpe, Sortino, Calmar, Profit, Trades).
        "lower_better"
            → ערך נמוך יותר טוב (Drawdown, TailRisk, UlcerIndex, MaxConsecLosses).

    hard_min / hard_max:
        גבולות קשיחים; מעבר אליהם → ציון 0.
    optimal_min / optimal_max:
        טווח "Sweet spot" שבו הציון ≈ 1.

    weight:
        משקל הפקטור בציון metrics הכולל.
    shape:
        1.0 = ליניארי, >1 = ירידה/עלייה חדה יותר (פחות סובלני),
        <1  = רכה יותר (סובלני יותר).

    horizon:
        "short" / "medium" / "long" — אופציונלי, מכוון לחיבור עם אסטרטגיה רב-אופקית.
    style_tags:
        למשל: ["risk", "tail", "path", "capacity"] — מאפיינים את סוג הפקטור.
    """

    name: str
    mode: MetricsMode
    hard_min: float
    optimal_min: float
    optimal_max: float
    hard_max: float
    weight: float = 1.0
    shape: float = 1.0
    horizon: str = "medium"
    style_tags: Optional[List[str]] = None

    def score(self, value: Optional[float]) -> float:
        if value is None or not np.isfinite(value):
            return 0.0

        x = float(value)
        hmin = float(self.hard_min)
        omin = float(self.optimal_min)
        omax = float(self.optimal_max)
        hmax = float(self.hard_max)

        if hmax <= hmin:
            return 0.0

        if self.mode == "higher_better":
            if x <= hmin:
                base = 0.0
            elif x >= omax:
                base = 1.0
            elif x <= omin:
                base = 0.5 * (x - hmin) / (omin - hmin)
            else:  # בין omin ל-omax
                base = 0.5 + 0.5 * (x - omin) / (omax - omin)
        elif self.mode == "lower_better":
            if x >= hmax:
                base = 0.0
            elif x <= omin:
                base = 1.0
            elif x <= omax:
                base = 0.5 + 0.5 * (omax - x) / (omax - omin)
            else:  # בין omax ל-hmax
                base = 0.5 * (hmax - x) / (hmax - omax)
        else:  # "range"
            if x <= hmin or x >= hmax:
                base = 0.0
            elif omin <= x <= omax:
                base = 1.0
            elif hmin < x < omin:
                base = (x - hmin) / (omin - hmin)
            elif omax < x < hmax:
                base = (hmax - x) / (hmax - omax)
            else:
                base = 0.0

        base = max(0.0, min(1.0, base))
        if self.shape != 1.0 and base > 0.0:
            base = base ** float(self.shape)

        return float(base)


@dataclass
class MetricsOptimismConfig:
    """
    קונפיגורציה לציון Metrics Optimism.

    factors:
        שם → MetricsFactorSpec (Sharpe/Sortino/DD/WinRate/DSR/...).

    profile:
        "default" / "defensive" / "aggressive"
    source:
        "default_profile" / "manual" / "ml_adapted"
    """

    factors: Dict[str, MetricsFactorSpec]
    profile: str = "default"
    source: str = "default_profile"

    @classmethod
    def from_simple_dict(
        cls,
        data: Dict[str, Dict[str, Any]],
        *,
        profile: str = "custom",
        source: str = "manual",
    ) -> "MetricsOptimismConfig":
        factors: Dict[str, MetricsFactorSpec] = {}
        for name, cfg in data.items():
            try:
                factors[name] = MetricsFactorSpec(
                    name=name,
                    mode=str(cfg.get("mode", "range")).lower(),  # type: ignore[arg-type]
                    hard_min=float(cfg["hard_min"]),
                    optimal_min=float(cfg["optimal_min"]),
                    optimal_max=float(cfg["optimal_max"]),
                    hard_max=float(cfg["hard_max"]),
                    weight=float(cfg.get("weight", 1.0)),
                    shape=float(cfg.get("shape", 1.0)),
                    horizon=str(cfg.get("horizon", "medium")),
                    style_tags=list(cfg.get("style_tags", []) or []),
                )
            except Exception as e:
                logger.warning("Invalid metrics factor config for %s: %s", name, e)
        return cls(factors=factors, profile=profile, source=source)

    @classmethod
    def default_for_profile(cls, profile: str = "default") -> "MetricsOptimismConfig":
        """
        דיפולט מקצועי לציון Metrics לפי פרופיל:

        default:
            - Sharpe          : range [0.7, 2.5]
            - Drawdown        : lower_better [0, 0.20]
            - WinRate         : higher_better [0.48, 0.68]
            - DSR             : higher_better [0.0, 2.5]
            - Trades          : higher_better [15, 300]
            - TailRisk        : lower_better [0, 0.20] (CVaR / ES / Worst-Case)
            - MaxConsecLosses : lower_better [0, 6]
            - UlcerIndex      : lower_better [0, 0.10] (path risk)
            - ExposurePct     : range [0.20, 0.70]
            - Profit          : higher_better (scaled בינארי, נותן בונוס).

        defensive:
            - יותר משקל ל-DD, TailRisk, UlcerIndex.
            - דרישה גבוהה יותר ל-DSR / Sharpe.

        aggressive:
            - יותר משקל ל-Profit, Sharpe, Trades.
            - סובל קצת יותר DD/Tail, אבל עדיין מעניש קצוות.

        שים לב:
        -------
        - אם df לא מכיל חלק מהעמודות — פשוט יקבל 0.0 לפקטור הזה (ולא ישבור).
        - אפשר לעדכן את הטווחים מתוך adapt_metrics_from_history (בהמשך).
        """
        p = profile.lower().strip()
        if p not in {"default", "defensive", "aggressive"}:
            p = "default"

        if p == "defensive":
            raw = {
                "Sharpe": {
                    "mode": "range",
                    "hard_min": 0.0,
                    "optimal_min": 1.0,
                    "optimal_max": 2.5,
                    "hard_max": 4.0,
                    "weight": 2.5,
                    "shape": 1.0,
                    "horizon": "medium",
                },
                "Drawdown": {
                    "mode": "lower_better",
                    "hard_min": 0.0,
                    "optimal_min": 0.0,
                    "optimal_max": 0.15,
                    "hard_max": 0.35,
                    "weight": 3.0,
                    "shape": 1.3,
                    "horizon": "medium",
                    "style_tags": ["risk", "tail"],
                },
                "TailRisk": {
                    "mode": "lower_better",
                    "hard_min": 0.0,
                    "optimal_min": 0.0,
                    "optimal_max": 0.20,
                    "hard_max": 0.50,
                    "weight": 2.5,
                    "shape": 1.3,
                    "horizon": "medium",
                    "style_tags": ["tail"],
                },
                "WinRate": {
                    "mode": "higher_better",
                    "hard_min": 0.40,
                    "optimal_min": 0.50,
                    "optimal_max": 0.72,
                    "hard_max": 0.90,
                    "weight": 1.8,
                    "shape": 1.0,
                    "horizon": "medium",
                },
                "DSR": {
                    "mode": "higher_better",
                    "hard_min": -1.0,
                    "optimal_min": 0.0,
                    "optimal_max": 2.5,
                    "hard_max": 4.0,
                    "weight": 2.5,
                    "shape": 1.0,
                    "horizon": "long",
                    "style_tags": ["quality"],
                },
                "Trades": {
                    "mode": "higher_better",
                    "hard_min": 10,
                    "optimal_min": 30,
                    "optimal_max": 250,
                    "hard_max": 800,
                    "weight": 1.5,
                    "shape": 1.0,
                    "horizon": "long",
                },
                "MaxConsecLosses": {
                    "mode": "lower_better",
                    "hard_min": 0,
                    "optimal_min": 0,
                    "optimal_max": 5,
                    "hard_max": 15,
                    "weight": 1.8,
                    "shape": 1.2,
                    "horizon": "medium",
                    "style_tags": ["path"],
                },
                "UlcerIndex": {
                    "mode": "lower_better",
                    "hard_min": 0.0,
                    "optimal_min": 0.0,
                    "optimal_max": 0.08,
                    "hard_max": 0.25,
                    "weight": 2.0,
                    "shape": 1.2,
                    "horizon": "long",
                    "style_tags": ["path", "risk"],
                },
                "ExposurePct": {
                    "mode": "range",
                    "hard_min": 0.0,
                    "optimal_min": 0.25,
                    "optimal_max": 0.60,
                    "hard_max": 1.0,
                    "weight": 1.0,
                    "shape": 1.0,
                    "horizon": "medium",
                    "style_tags": ["capacity"],
                },
                "Profit": {
                    "mode": "higher_better",
                    "hard_min": -1e6,
                    "optimal_min": 0.0,
                    "optimal_max": 1e5,
                    "hard_max": 5e5,
                    "weight": 1.5,
                    "shape": 1.0,
                    "horizon": "long",
                    "style_tags": ["pnl"],
                },
            }
        elif p == "aggressive":
            raw = {
                "Sharpe": {
                    "mode": "higher_better",
                    "hard_min": 0.0,
                    "optimal_min": 1.0,
                    "optimal_max": 3.0,
                    "hard_max": 5.0,
                    "weight": 2.8,
                    "shape": 1.0,
                    "horizon": "medium",
                },
                "Drawdown": {
                    "mode": "lower_better",
                    "hard_min": 0.0,
                    "optimal_min": 0.0,
                    "optimal_max": 0.25,
                    "hard_max": 0.50,
                    "weight": 2.0,
                    "shape": 1.0,
                    "horizon": "medium",
                    "style_tags": ["risk"],
                },
                "TailRisk": {
                    "mode": "lower_better",
                    "hard_min": 0.0,
                    "optimal_min": 0.0,
                    "optimal_max": 0.25,
                    "hard_max": 0.60,
                    "weight": 1.8,
                    "shape": 1.0,
                    "horizon": "medium",
                    "style_tags": ["tail"],
                },
                "WinRate": {
                    "mode": "range",
                    "hard_min": 0.30,
                    "optimal_min": 0.45,
                    "optimal_max": 0.65,
                    "hard_max": 0.85,
                    "weight": 1.2,
                    "shape": 1.0,
                    "horizon": "medium",
                },
                "DSR": {
                    "mode": "higher_better",
                    "hard_min": -1.0,
                    "optimal_min": 0.5,
                    "optimal_max": 3.0,
                    "hard_max": 4.5,
                    "weight": 2.5,
                    "shape": 1.0,
                    "horizon": "long",
                    "style_tags": ["quality"],
                },
                "Trades": {
                    "mode": "higher_better",
                    "hard_min": 15,
                    "optimal_min": 40,
                    "optimal_max": 450,
                    "hard_max": 1500,
                    "weight": 1.8,
                    "shape": 1.0,
                    "horizon": "long",
                },
                "MaxConsecLosses": {
                    "mode": "lower_better",
                    "hard_min": 0,
                    "optimal_min": 0,
                    "optimal_max": 7,
                    "hard_max": 20,
                    "weight": 1.5,
                    "shape": 1.1,
                    "horizon": "medium",
                    "style_tags": ["path"],
                },
                "UlcerIndex": {
                    "mode": "lower_better",
                    "hard_min": 0.0,
                    "optimal_min": 0.0,
                    "optimal_max": 0.12,
                    "hard_max": 0.30,
                    "weight": 1.5,
                    "shape": 1.1,
                    "horizon": "long",
                    "style_tags": ["path", "risk"],
                },
                "ExposurePct": {
                    "mode": "range",
                    "hard_min": 0.0,
                    "optimal_min": 0.30,
                    "optimal_max": 0.75,
                    "hard_max": 1.0,
                    "weight": 1.2,
                    "shape": 1.0,
                    "horizon": "medium",
                    "style_tags": ["capacity"],
                },
                "Profit": {
                    "mode": "higher_better",
                    "hard_min": -1e6,
                    "optimal_min": 2e4,
                    "optimal_max": 2e5,
                    "hard_max": 1e6,
                    "weight": 2.0,
                    "shape": 1.0,
                    "horizon": "long",
                    "style_tags": ["pnl"],
                },
            }
        else:  # default
            raw = {
                "Sharpe": {
                    "mode": "range",
                    "hard_min": 0.0,
                    "optimal_min": 0.7,
                    "optimal_max": 2.5,
                    "hard_max": 4.0,
                    "weight": 2.1,
                    "shape": 1.0,
                    "horizon": "medium",
                },
                "Drawdown": {
                    "mode": "lower_better",
                    "hard_min": 0.0,
                    "optimal_min": 0.0,
                    "optimal_max": 0.20,
                    "hard_max": 0.40,
                    "weight": 2.3,
                    "shape": 1.1,
                    "horizon": "medium",
                    "style_tags": ["risk", "tail"],
                },
                "TailRisk": {
                    "mode": "lower_better",
                    "hard_min": 0.0,
                    "optimal_min": 0.0,
                    "optimal_max": 0.22,
                    "hard_max": 0.55,
                    "weight": 2.0,
                    "shape": 1.1,
                    "horizon": "medium",
                    "style_tags": ["tail"],
                },
                "WinRate": {
                    "mode": "higher_better",
                    "hard_min": 0.40,
                    "optimal_min": 0.48,
                    "optimal_max": 0.68,
                    "hard_max": 0.90,
                    "weight": 1.6,
                    "shape": 1.0,
                    "horizon": "medium",
                },
                "DSR": {
                    "mode": "higher_better",
                    "hard_min": -1.0,
                    "optimal_min": 0.0,
                    "optimal_max": 2.5,
                    "hard_max": 4.0,
                    "weight": 2.0,
                    "shape": 1.0,
                    "horizon": "long",
                    "style_tags": ["quality"],
                },
                "Trades": {
                    "mode": "higher_better",
                    "hard_min": 8,
                    "optimal_min": 20,
                    "optimal_max": 300,
                    "hard_max": 1200,
                    "weight": 1.4,
                    "shape": 1.0,
                    "horizon": "long",
                },
                "MaxConsecLosses": {
                    "mode": "lower_better",
                    "hard_min": 0,
                    "optimal_min": 0,
                    "optimal_max": 6,
                    "hard_max": 18,
                    "weight": 1.6,
                    "shape": 1.15,
                    "horizon": "medium",
                    "style_tags": ["path"],
                },
                "UlcerIndex": {
                    "mode": "lower_better",
                    "hard_min": 0.0,
                    "optimal_min": 0.0,
                    "optimal_max": 0.10,
                    "hard_max": 0.28,
                    "weight": 1.8,
                    "shape": 1.1,
                    "horizon": "long",
                    "style_tags": ["path", "risk"],
                },
                "ExposurePct": {
                    "mode": "range",
                    "hard_min": 0.0,
                    "optimal_min": 0.20,
                    "optimal_max": 0.70,
                    "hard_max": 1.0,
                    "weight": 1.1,
                    "shape": 1.0,
                    "horizon": "medium",
                    "style_tags": ["capacity"],
                },
                "Profit": {
                    "mode": "higher_better",
                    "hard_min": -1e6,
                    "optimal_min": 1e4,
                    "optimal_max": 1.5e5,
                    "hard_max": 8e5,
                    "weight": 1.7,
                    "shape": 1.0,
                    "horizon": "long",
                    "style_tags": ["pnl"],
                },
            }

        cfg = cls.from_simple_dict(raw, profile=p, source="default_profile")
        logger.info("MetricsOptimismConfig default_for_profile=%s created.", p)
        return cfg

    def score_vector(
        self,
        row: pd.Series,
        factor_cols_map: Optional[Dict[str, str]] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """
        מחשב ציון metrics משוקלל (0–1) + ציונים per-factor.

        row:
            שורה אחת של df שמכילה עמודות Sharpe/Drawdown/WinRate/DSR/Trades וכו'.
        factor_cols_map:
            מיפוי בין שם הפקטור בקונפיג לבין שם העמודה ב-df, לדוגמה:
            {"Sharpe": "Sharpe", "Drawdown": "Drawdown_pct"}.
        """
        if not self.factors:
            return 0.0, {}

        if factor_cols_map is None:
            factor_cols_map = {name: name for name in self.factors.keys()}

        details: Dict[str, float] = {}
        ws = 0.0
        ssum = 0.0

        for name, spec in self.factors.items():
            col = factor_cols_map.get(name, name)
            v = row.get(col)
            s = spec.score(v)
            details[name] = s
            w = float(spec.weight)
            ssum += s * w
            ws += abs(w)

        if ws <= 0:
            total = 0.0
        else:
            total = ssum / ws

        total = float(max(0.0, min(1.0, total)))
        return total, details


def apply_metrics_optimism_scan(
    df: pd.DataFrame,
    cfg: MetricsOptimismConfig,
    *,
    factor_cols_map: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    מוסיף לכל שורה:
        - metrics_score_total
        - metrics_score_details

    df צפוי להכיל:
        Sharpe, Drawdown, TailRisk, WinRate, DSR, Trades, MaxConsecLosses,
        UlcerIndex, ExposurePct, Profit (לפי config).
    """
    if df is None or df.empty:
        return df

    df = df.copy()
    scores: List[float] = []
    details_list: List[Dict[str, float]] = []

    for _, row in df.iterrows():
        total, det = cfg.score_vector(row, factor_cols_map=factor_cols_map)
        scores.append(total)
        details_list.append(det)

    df["metrics_score_total"] = scores
    df["metrics_score_details"] = details_list
    return df


def adapt_metrics_from_history(
    cfg: MetricsOptimismConfig,
    df_history: pd.DataFrame,
    *,
    score_col: str = "Score",
    top_quantile: float = 0.2,
) -> MetricsOptimismConfig:
    """
    Hook ל-ML/Meta-Optimization לחלק metrics:

    df_history:
        DataFrame של opt_df / backtests עם Score
        ועמודות metrics (Sharpe, Drawdown, TailRisk, WinRate, DSR, Trades וכו').

    לוגיקה:
        - לוקחים Top-X% לפי Score.
        - עבור כל factor מספרי שקיים ב-df_history:
            * mode="range","higher_better","lower_better" → מעדכנים optimal_min/max לפי q25/q75.
    """
    if df_history is None or df_history.empty or score_col not in df_history.columns:
        return cfg

    try:
        df_sorted = df_history.sort_values(score_col, ascending=False)
    except Exception:
        df_sorted = df_history.copy()

    k = max(10, int(len(df_sorted) * float(top_quantile)))
    df_top = df_sorted.head(k)

    new_factors: Dict[str, MetricsFactorSpec] = {}

    for name, spec in cfg.factors.items():
        col = name
        if col not in df_top.columns:
            new_factors[name] = spec
            continue

        vals = pd.to_numeric(df_top[col], errors="coerce").dropna()
        if vals.empty:
            new_factors[name] = spec
            continue

        q25 = float(vals.quantile(0.25))
        q75 = float(vals.quantile(0.75))

        new_factors[name] = MetricsFactorSpec(
            name=name,
            mode=spec.mode,
            hard_min=spec.hard_min,
            optimal_min=q25,
            optimal_max=q75,
            hard_max=spec.hard_max,
            weight=spec.weight,
            shape=spec.shape,
            horizon=spec.horizon,
            style_tags=spec.style_tags,
        )

    logger.info("MetricsOptimismConfig adapted from history (top %.0f%%).", top_quantile * 100)
    return MetricsOptimismConfig(
        factors=new_factors,
        profile=cfg.profile,
        source="ml_adapted",
    )
 
 # =============================================
# Part 5/5 — Composite Smart Score + Smart Scan UI (HF-grade)
# =============================================
from datetime import datetime, timezone

def compute_composite_smart_score(
    df: pd.DataFrame,
    *,
    profile: str = "default",
    w_param: float = 0.30,
    w_fund: float = 0.25,
    w_macro: float = 0.15,
    w_metrics: float = 0.30,
) -> pd.DataFrame:
    """
    Composite Smart Score — מחבר 4 שכבות לציון אחד:

        - param_optimism_score_total
        - fundamental_score_total
        - macro_score_total
        - metrics_score_total

    המשקולות יכולות להגיע מה-UI או מברירת מחדל לפי profile.

    השכבות הן תמיד 0–1, ולכן composite גם 0–1.
    """
    if df is None or df.empty:
        return df

    df = df.copy()
    p = profile.lower().strip()
    if p not in {"default", "defensive", "aggressive"}:
        p = "default"

    # אם המשתמש לא שינה משקולות, נדרוס לפי פרופיל
    default_tuple = (0.30, 0.25, 0.15, 0.30)
    if (w_param, w_fund, w_macro, w_metrics) == default_tuple:
        if p == "defensive":
            w_param, w_fund, w_macro, w_metrics = 0.20, 0.30, 0.20, 0.30
        elif p == "aggressive":
            w_param, w_fund, w_macro, w_metrics = 0.35, 0.20, 0.10, 0.35

    def _get(name: str) -> np.ndarray:
        if name not in df.columns:
            return np.zeros(len(df))
        arr = pd.to_numeric(df[name], errors="coerce").fillna(0.0).to_numpy()
        return np.clip(arr, 0.0, 1.0)

    par = _get("param_optimism_score_total")
    fun = _get("fundamental_score_total")
    mac = _get("macro_score_total")
    met = _get("metrics_score_total")

    w_param = max(0.0, float(w_param))
    w_fund = max(0.0, float(w_fund))
    w_macro = max(0.0, float(w_macro))
    w_metrics = max(0.0, float(w_metrics))
    total_w = w_param + w_fund + w_macro + w_metrics
    if total_w <= 0:
        w_param = 1.0
        total_w = 1.0

    comp = (w_param * par + w_fund * fun + w_macro * mac + w_metrics * met) / total_w
    df["smart_score_total"] = np.clip(comp, 0.0, 1.0)

    # נשמור גם את המשקולות לצורך דוחות/אודיט
    df.attrs["smart_layer_weights"] = {
        "param": w_param,
        "fundamental": w_fund,
        "macro": w_macro,
        "metrics": w_metrics,
        "profile": p,
    }
    return df

def _persist_signals_to_sql_store(
    signals_df: pd.DataFrame,
    app_ctx: AppContext,
    *,
    profile_name: str = "default",
) -> None:
    """
    שומר signals_df לטבלאות SQL דרך SqlStore:

    - signals_universe
    - signals_summary

    מיועד לקריאה אחרי שסיימת לבנות signals_df ברמת universe.
    """
    if signals_df is None or signals_df.empty:
        logger.info("_persist_signals_to_sql_store: empty signals_df — nothing to save.")
        return

    try:
        store = SqlStore.from_settings(app_ctx.settings)
    except Exception:
        logger.exception("_persist_signals_to_sql_store: failed to init SqlStore.from_settings")
        return

    run_id: Optional[str] = getattr(app_ctx, "run_id", None)

    try:
        store.save_signals(
            signals_df,
            profile_name=profile_name,
            run_id=run_id,
            env=None,  # ישתמש ב-default_env של SqlStore (dev אצלך)
        )
        store.save_signals_summary(
            signals_df,
            profile_name=profile_name,
            run_id=run_id,
            env=None,
        )
        logger.info(
            "Persisted %d signals rows + summary to SqlStore (profile=%s, run_id=%s)",
            len(signals_df),
            profile_name,
            run_id,
        )
    except Exception:
        logger.exception("_persist_signals_to_sql_store: failed to save signals to SqlStore")

def render_smart_scan_tab(
    app_ctx: Any,
    feature_flags: Any,
    nav_payload: Optional[Dict[str, Any]] = None,
) -> None:
    """
    🔍 Smart Scan Tab — Composite HF-grade Scanner (v3 HF+)
    ======================================================

    מה הטאב הזה עושה:
    ------------------
    1. בונה Universe של זוגות (pairs + parameters) ממקור Service/CSV/Demo.
    2. טוען שכבות דאטה:
       - Fair Value (יצירת "זוג יקר/זול" מול Fair Value).
       - Fundamentals (איכות, צמיחה, מינוף, איכות רווח).
       - Backtest Metrics (Sharpe, Drawdown, WinRate, TailRisk, DSR, Trades...).
       - Param Scores (מתוצאות אופטימיזציה — אופציונלי).
    3. בונה קונפיגורציות לשכבות:
       - ParamOptimismConfig
       - FundamentalOptimismConfig
       - MacroOptimismConfig
       - MetricsOptimismConfig
    4. מגדיר משקלי שכבות ו-Regime Profile (neutral/risk_off/risk_on/...).
    5. **שכבת Macro Overlay אמיתית**:
       - שימוש ב-MacroConfig + MacroBundle + compute_adjustments כדי לתת:
         macro_multiplier / macro_include / macro_score_overlay / macro_cap_hint לכל זוג.
       - שומר overlay ב-session_state לשימוש בטאבים אחרים (Risk, Portfolio, Agents).
    6. (בחלק 2) יריץ את כל השכבות, יחשב smart_score_total, יבדוק Stability, יבנה Shortlist,
       וישמור meta מלא ו-history.

    הערה:
    -----
    • הפונקציה הזו מניחה שכל הקונפיגים והפונקציות (DashboardService, create_dashboard_service,
      DashboardContext, ParamOptimismConfig, FundamentalOptimismConfig, MacroOptimismConfig,
      MetricsOptimismConfig, MacroConfig, load_macro_bundle, compute_adjustments, וכו')
      כבר מיובאים בראש הקובץ כרגיל.
    """

    st.markdown("### 🔍 Smart Scan — Composite HF-grade Scanner")

    # ======================================================
    # 0) הקמת Service + Context + Run ID ו-telemetry בסיסי
    # ======================================================
    try:
        service: DashboardService = create_dashboard_service()
        ctx: DashboardContext = build_default_dashboard_context()
    except Exception as exc:
        st.error("Smart Scan לא הצליח להקים DashboardService/Context.")
        st.caption(str(exc))
        return

    profile = str(st.session_state.get("opt_profile", ctx.profile or "default"))
    env = getattr(ctx, "env", "dev")
    start_date = getattr(ctx, "start_date", None)
    end_date = getattr(ctx, "end_date", None)

    scan_run_id = str(uuid4())
    st.session_state["smart_scan_run_id"] = scan_run_id

    # אפשרות: override מתוך nav_payload (אם הגיע מה-Dashboard).
    if isinstance(nav_payload, dict):
        profile = str(nav_payload.get("profile", profile))
        env = str(nav_payload.get("env", env))

    # ======================================================
    # 1) Header חכם + Regime Profile + חיווי מאקרו גלובלי
    # ======================================================
    col_hdr1, col_hdr2, col_hdr3 = st.columns(3)
    with col_hdr1:
        st.caption(
            f"Env=`{env}` | Profile=`{profile}` | "
            f"Dates=`{start_date}` → `{end_date}` | ScanID=`{scan_run_id[:8]}`"
        )

    with col_hdr2:
        auto_link_key = "smart_scan_auto_link_flag"
        auto_link = bool(st.session_state.get(auto_link_key, True))
        st.checkbox(
            "Auto-select top pair",
            value=auto_link,
            key=auto_link_key,
            help="כשפעיל, הסורק יבחר את הזוג המוביל וישלח אותו לטאבים אחרים (Pair / Backtest / Optimisation).",
        )

    with col_hdr3:
        regime_profile = st.selectbox(
            "Regime Profile",
            [
                "auto",
                "neutral",
                "low_vol_trending",
                "high_vol_choppy",
                "risk_off",
                "risk_on",
                "crisis",
                "reflation",
            ],
            index=0,
            key="smart_scan_regime_profile",
            help=(
                "auto → ינסה לנחש מתוך מצב המאקרו (Macro Tab / Macro Factors).\n"
                "שאר הפרופילים → משנים משקלים בין Param/Fund/Macro/Metrics בהתאם למשטר."
            ),
        )

    # חיווי מקוצר על מצב המאקרו מה-Macro Tab (אם הוא רץ):
    macro_factor_summary = st.session_state.get("macro_factor_summary_text")
    macro_regime_label = st.session_state.get("macro_regime_label")
    macro_risk_alert = bool(st.session_state.get("macro_risk_alert", False))
    macro_risk_budget_hint = st.session_state.get("macro_risk_budget_hint")

    if macro_factor_summary or macro_regime_label or macro_risk_budget_hint is not None:
        with st.expander("🧠 Macro State Snapshot (מהטאב Macro)", expanded=False):
            if macro_regime_label:
                st.markdown(f"**Macro Regime Label:** `{macro_regime_label}`")
            if macro_factor_summary:
                st.markdown(macro_factor_summary)
            if macro_risk_budget_hint is not None:
                st.caption(f"Macro Risk Budget Hint: ~{float(macro_risk_budget_hint):.2f}x רגיל")
            if macro_risk_alert:
                st.warning("⚠️ Macro Risk Alert פעיל (Risk-Off / Crisis) — מומלץ סלקטיביות גבוהה יותר.")

    st.markdown("---")

    # ======================================================
    # 2) Universe מקור — Service / CSV / Demo
    # ======================================================
    st.markdown("#### 1️⃣ Universe מקור (pairs + parameters)")

    universe_options: List[str] = ["Service (SqlStore/Runtime)", "Upload CSV"]
    if ALLOW_DEMO_UNIVERSE:
        universe_options.append("Demo (internal)")

    universe_source = st.selectbox(
        "Universe source",
        universe_options,
        index=0,
        key="smart_scan_universe_source",
        help=(
            "Service → Universe אמיתי מתוך SqlStore/DashboardService.\n"
            "Upload → CSV חיצוני.\n"
            "Demo → דמו פנימי (ל-dev בלבד)."
        ),
    )

    df_universe: Optional[pd.DataFrame] = None

    # 2.1 טעינה מה-Service (DashboardService → SqlStore)
    if universe_source.startswith("Service"):
        if service is None or not hasattr(service, "get_smart_scan_universe"):
            st.error("DashboardService לא זמין או לא מימש get_smart_scan_universe — אין Universe אמיתי.")
        else:
            try:
                df_universe = service.get_smart_scan_universe(ctx)
            except Exception as e:
                st.error("קריאה ל-get_smart_scan_universe נכשלה.")
                st.caption(str(e))
                df_universe = None

    # 2.2 טעינה מ-CSV (Override ידני)
    if universe_source.startswith("Upload"):
        uni_file = st.file_uploader(
            "Upload Universe CSV (חייב לכלול 'pair' + עמודות פרמטרים)",
            type=["csv"],
            key="smart_scan_universe_csv",
        )
        if uni_file is not None:
            try:
                df_universe = pd.read_csv(uni_file)
            except Exception as e:
                st.error(f"Error reading universe CSV: {e}")
                df_universe = None

    # 2.3 DEMO (רק אם ALLOW_DEMO_UNIVERSE=True ונבחר במפורש)
    if df_universe is None or df_universe.empty:
        if universe_source.startswith("Service"):
            if not ALLOW_DEMO_UNIVERSE:
                st.error(
                    "לא נמצא Universe אמיתי מה-Service (SqlStore.load_pair_quality).\n"
                    "וודא שיש טבלאות dq_pairs / pairs_quality או השתמש ב-Upload CSV."
                )
                return
        if universe_source.startswith("Upload"):
            if df_universe is None:
                st.info("לא הועלה קובץ Universe. העלה CSV או בחר מקור Service.")
                return

        if universe_source.startswith("Demo") and ALLOW_DEMO_UNIVERSE:
            n = 30
            df_universe = pd.DataFrame(
                {
                    "pair": [f"PAIR_{i}" for i in range(n)],
                    "z_entry": np.linspace(1.0, 3.5, n),
                    "z_exit": np.linspace(0.1, 1.0, n),
                    "lookback": np.linspace(20, 120, n),
                    "hl_bars": np.linspace(10, 120, n),
                    "corr_min": np.linspace(0.4, 0.9, n),
                }
            )
            st.warning(
                "Universe Demo פעיל (ALLOW_DEMO_UNIVERSE=True). במצב Production מומלץ לכבות דמו."
            )
        elif df_universe is None or df_universe.empty:
            st.error("אין Universe זמין ל-Smart Scan. עצירה.")
            return

    # --- Universe preview + Diagnostics ---
    st.markdown("**Universe preview (עד 50 שורות):**")
    st.dataframe(df_universe.head(50), width="stretch")

    with st.expander("Universe diagnostics", expanded=False):
        try:
            n_pairs = len(df_universe)
            st.write(f"מספר זוגות ב-Universe: `{n_pairs}`")
            diag_rows = []
            for c in ["z_entry", "z_exit", "lookback", "hl_bars", "corr_min"]:
                if c in df_universe.columns:
                    vals = pd.to_numeric(df_universe[c], errors="coerce")
                    diag_rows.append(
                        {
                            "field": c,
                            "mean": float(vals.mean()),
                            "min": float(vals.min()),
                            "median": float(vals.median()),
                            "max": float(vals.max()),
                        }
                    )
            if diag_rows:
                st.dataframe(pd.DataFrame(diag_rows), width="stretch")
        except Exception:
            pass

    # Universe פרטני לזוגות — pairs_df בליבה (נשתמש בו לשכבת Macro Overlay)
    pairs_df = df_universe.copy()
    if "pair_id" not in pairs_df.columns:
        if "pair" in pairs_df.columns:
            pairs_df["pair_id"] = pairs_df["pair"].astype(str)
        elif {"sym_x", "sym_y"} <= set(pairs_df.columns):
            pairs_df["pair_id"] = pairs_df["sym_x"].astype(str) + "-" + pairs_df["sym_y"].astype(str)
        elif {"a", "b"} <= set(pairs_df.columns):
            pairs_df["pair_id"] = pairs_df["a"].astype(str) + "-" + pairs_df["b"].astype(str)

    # ======================================================
    # 3) Data layers: FV / Fundamentals / Metrics / Param Scores
    # ======================================================
    st.markdown("#### 2️⃣ Data layers (Fair Value / Fundamentals / Metrics / Param Scores)")

    col_files1, col_files2 = st.columns(2)

    df_fv = None
    df_fund = None
    df_metrics = None
    df_param_scores = None

    auto_load = st.checkbox(
        "נסה לטעון FV/Fundamentals/Metrics אוטומטית מה-Service (SqlStore)",
        value=True,
        key="smart_scan_auto_load_service",
    )

    if auto_load and service is not None:
        # Fair Value
        if hasattr(service, "get_fair_value_universe"):
            try:
                df_fv = service.get_fair_value_universe(ctx)
            except Exception as e:
                logger.warning("DashboardService.get_fair_value_universe failed: %s", e)
        # Fundamentals
        if hasattr(service, "get_fundamentals_universe"):
            try:
                df_fund = service.get_fundamentals_universe(ctx)
            except Exception as e:
                logger.warning("DashboardService.get_fundamentals_universe failed: %s", e)
        # Metrics
        if hasattr(service, "get_pair_metrics_universe"):
            try:
                df_metrics = service.get_pair_metrics_universe(ctx)
            except Exception as e:
                logger.warning("DashboardService.get_pair_metrics_universe failed: %s", e)

    # ----- העלאה ידנית (Override / Merge) -----
    with col_files1:
        st.caption("Fair Value (Override/השלמה):")
        fv_file = st.file_uploader(
            "Fair Value CSV (pair + fv_pct_diff / fair_value_pct_diff)",
            type=["csv"],
            key="smart_scan_fv_file",
        )
        if fv_file is not None:
            try:
                df_fv = pd.read_csv(fv_file)
                st.caption("✅ FV CSV loaded (override service).")
            except Exception as e:
                st.error(f"Error reading FV file: {e}")

        st.caption("Fundamentals (Override/Merge):")
        fund_file = st.file_uploader(
            "Fundamentals CSV (pair + pe, roe, debt_to_equity, eps_growth_5y, ...)",
            type=["csv"],
            key="smart_scan_fund_file",
        )
        if fund_file is not None:
            try:
                df_fund = pd.read_csv(fund_file)
                st.caption("✅ Fundamental CSV loaded.")
            except Exception as e:
                st.error(f"Error reading fundamental file: {e}")

    with col_files2:
        st.caption("Metrics (Override/Merge):")
        metrics_file = st.file_uploader(
            "Metrics CSV (pair + Sharpe, Drawdown, TailRisk, WinRate, DSR, Trades, ...)",
            type=["csv"],
            key="smart_scan_metrics_file",
        )
        if metrics_file is not None:
            try:
                df_metrics = pd.read_csv(metrics_file)
                st.caption("✅ Metrics CSV loaded.")
            except Exception as e:
                st.error(f"Error reading Metrics file: {e}")

        st.caption("Param Score file (מתוצאות אופטימיזציה, אופציונלי):")
        param_score_file = st.file_uploader(
            "Param Score CSV (pair + score)",
            type=["csv"],
            key="smart_scan_param_score_file",
        )
        if param_score_file is not None:
            try:
                df_param_scores = pd.read_csv(param_score_file)
                st.caption("✅ Param Score CSV loaded.")
            except Exception as e:
                st.error(f"Error reading Param score file: {e}")

    # Diagnostic קטן על coverage של השכבות
    with st.expander("Data layers coverage", expanded=False):
        coverage_rows = []
        if df_fv is not None and not df_fv.empty:
            coverage_rows.append({"layer": "Fair Value", "rows": len(df_fv)})
        if df_fund is not None and not df_fund.empty:
            coverage_rows.append({"layer": "Fundamentals", "rows": len(df_fund)})
        if df_metrics is not None and not df_metrics.empty:
            coverage_rows.append({"layer": "Metrics", "rows": len(df_metrics)})
        if df_param_scores is not None and not df_param_scores.empty:
            coverage_rows.append({"layer": "Param Scores", "rows": len(df_param_scores)})
        if coverage_rows:
            st.dataframe(pd.DataFrame(coverage_rows), width="stretch")

    # ======================================================
    # 4) Layer configs (Param / Fundamental / Macro / Metrics)
    # ======================================================
    st.markdown("#### 3️⃣ Layer configs (Param / Fundamental / Macro / Metrics)")

    cfg_param = ParamOptimismConfig.default_for_profile(profile)
    cfg_fund = FundamentalOptimismConfig.default_for_profile(profile)
    cfg_macro = MacroOptimismConfig.default_for_profile(profile)
    cfg_metrics = MetricsOptimismConfig.default_for_profile(profile)

    with st.expander("Configs (summary)", expanded=False):
        st.caption("ParamOptimismConfig:")
        st.json({k: cfg_param.ranges[k].__dict__ for k in cfg_param.ranges.keys()})

        st.caption("FundamentalOptimismConfig:")
        st.json({k: cfg_fund.factors[k].__dict__ for k in cfg_fund.factors.keys()})

        st.caption("MacroOptimismConfig:")
        st.json({k: cfg_macro.factors[k].__dict__ for k in cfg_macro.factors.keys()})

        st.caption("MetricsOptimismConfig:")
        st.json({k: cfg_metrics.factors[k].__dict__ for k in cfg_metrics.factors.keys()})

    # ======================================================
    # 5) Composite weights & toggles + Regime Profile
    # ======================================================
    st.markdown("#### 4️⃣ Composite weights & layer toggles")

    col_layer1, col_layer2 = st.columns(2)

    with col_layer1:
        use_param = st.checkbox("Use Param layer", value=True, key="smart_scan_use_param")
        use_fund = st.checkbox("Use Fundamental layer", value=True, key="smart_scan_use_fund")
    with col_layer2:
        use_macro = st.checkbox("Use Macro layer", value=True, key="smart_scan_use_macro")
        use_metrics = st.checkbox("Use Metrics layer", value=True, key="smart_scan_use_metrics")

    col_w = st.columns(4)
    with col_w[0]:
        w_param = st.number_input(
            "w_param", 0.0, 1.0, 0.30, 0.05, key="smart_scan_w_param"
        )
    with col_w[1]:
        w_fund = st.number_input(
            "w_fund", 0.0, 1.0, 0.25, 0.05, key="smart_scan_w_fund"
        )
    with col_w[2]:
        w_macro = st.number_input(
            "w_macro", 0.0, 1.0, 0.15, 0.05, key="smart_scan_w_macro"
        )
    with col_w[3]:
        w_metrics = st.number_input(
            "w_metrics", 0.0, 1.0, 0.30, 0.05, key="smart_scan_w_metrics"
        )

    if not use_param:
        w_param = 0.0
    if not use_fund:
        w_fund = 0.0
    if not use_macro:
        w_macro = 0.0
    if not use_metrics:
        w_metrics = 0.0

    def _apply_regime_profile(
        regime: str,
        w_param: float,
        w_fund: float,
        w_macro: float,
        w_metrics: float,
    ) -> Tuple[float, float, float, float, str]:
        """
        מתאים משקלים לפי Regime Profile:

        neutral           → כמו המקור (normalize בלבד).
        low_vol_trending  → יותר משקל ל-Fundamentals + Param, פחות למאקרו.
        high_vol_choppy   → יותר משקל ל-Metrics + Macro, פחות Fundamentals.
        risk_off          → Macro + Metrics דומיננטיים, Param מוחלש.
        risk_on           → Param + Metrics מוטי Growth/Momentum.
        crisis            → Macro דומיננטי, Param כמעט כבוי.
        reflation         → איזון מחדש לטובת Macro + Fundamentals.
        auto              → ינסה לקרוא Regime מתוך macro_factor_snapshot / macro_regime_label.
        """
        regime = (regime or "neutral").lower().strip()

        if regime == "auto":
            macro_state = st.session_state.get("macro_factor_snapshot") or {}
            regime_label = str(st.session_state.get("macro_regime_label", "") or "").lower()
            risk_on_score = 0.0
            if isinstance(macro_state, dict):
                try:
                    risk_on_score = float(macro_state.get("risk_on_score", 0.0))
                except Exception:
                    risk_on_score = 0.0

            if regime_label.startswith("risk_off") or risk_on_score < -0.5:
                regime = "risk_off"
            elif regime_label.startswith("risk_on") or risk_on_score > 0.5:
                regime = "risk_on"
            elif "reflation" in regime_label:
                regime = "reflation"
            else:
                regime = "neutral"

        orig = np.array([w_param, w_fund, w_macro, w_metrics], dtype=float)

        if regime == "low_vol_trending":
            adj = np.array([1.2, 1.2, 0.8, 1.0])
        elif regime == "high_vol_choppy":
            adj = np.array([0.8, 0.8, 1.3, 1.3])
        elif regime == "risk_off":
            adj = np.array([0.6, 1.1, 1.5, 1.0])
        elif regime == "risk_on":
            adj = np.array([1.3, 0.9, 0.9, 1.3])
        elif regime == "crisis":
            adj = np.array([0.3, 0.7, 1.8, 1.0])
        elif regime == "reflation":
            adj = np.array([0.9, 1.2, 1.3, 0.9])
        else:
            adj = np.array([1.0, 1.0, 1.0, 1.0])

        new = orig * adj
        if new.sum() <= 0:
            new = np.array([1.0, 1.0, 1.0, 1.0], dtype=float)
        new = new / new.sum()
        return float(new[0]), float(new[1]), float(new[2]), float(new[3]), regime

    w_param, w_fund, w_macro, w_metrics, regime_resolved = _apply_regime_profile(
        regime_profile, w_param, w_fund, w_macro, w_metrics
    )

    run_btn = st.button("🚀 Run Smart Scan (all layers)", key="smart_scan_run_btn")
    if not run_btn:
        # לא מריצים סריקה חדשה, אבל נותנים לטאב להמשיך לרנדר את ה-UI הקיים
        st.caption("⚙️ בחר פרמטרים ולחץ על \"Run Smart Scan\" כדי להריץ סריקה חדשה.")
        return  # חשוב: return ולא st.stop()


    # ======================================================
    # 🔹 Macro Overlay — Regime → Multipliers/Filters → Scores
    # ======================================================

    # 1) קונפיג מאקרו:
    settings_for_macro = getattr(app_ctx, "settings", None)
    if settings_for_macro is not None and hasattr(settings_for_macro, "macro"):
        macro_cfg: MacroConfig = getattr(settings_for_macro, "macro")  # type: ignore[assignment]
    else:
        macro_cfg = MacroConfig()

    # 2) טעינת MacroBundle פעם אחת (אפשר לשתף עם macro_tab)
    macro_bundle = load_macro_bundle(macro_cfg)

    # 3) חישוב התאמות מאקרו לכל הזוגות ב-Smart Scan
    adj = None
    try:
        adj = compute_adjustments(pairs_df, macro_bundle, macro_cfg)
    except Exception:
        logger.exception("SmartScan: compute_adjustments failed → ממשיכים בלי שכבת מאקרו.")
        adj = None

    if adj is not None:
        pairs_df = pairs_df.copy()
        key_col = "pair_id"

        pairs_df["macro_multiplier"] = pairs_df[key_col].map(adj.pair_adjustments)
        pairs_df["macro_include"] = pairs_df[key_col].map(adj.filters)
        pairs_df["macro_score_overlay"] = pairs_df[key_col].map(adj.pair_scores)
        pairs_df["macro_cap_hint"] = pairs_df[key_col].map(adj.caps_hints)

        if "smart_score_total" in pairs_df.columns:
            pairs_df["smart_score_with_macro"] = (
                pairs_df["smart_score_total"] * pairs_df["macro_multiplier"].fillna(1.0)
            )

        # שמירה ל-session_state לשימוש בטאבים אחרים
        st.session_state["smart_scan_macro_overlay_pairs"] = {
            "run_id": scan_run_id,
            "multipliers": adj.pair_adjustments,
            "filters": adj.filters,
            "scores": adj.pair_scores,
            "caps_hints": adj.caps_hints,
            "regime_label": adj.regime_label,
        }

    # כאן pairs_df כבר כולל (אם אפשר) שכבת מאקרו מעל ה-Universe.
    # בחלק 2 נמשיך: הרצת השכבות (Param/Fund/Macro/Metrics), Composite, Stability, Shortlist, History וכו'.
    # ============================================
    # >>> כאן ממשיכים לחלק 2 (המשך הפונקציה) <<<
    # ============================================
    # ======================================================
    # 6) הרצת שכבות על ה-Universe (Param / Fundamental / Macro / Metrics)
    # ======================================================

    # נתחיל מ-copy של df_universe – זה הבסיס לשכבות השונות
    df_scan = df_universe.copy()

    # Param + FV + Param score file
    df_scan = apply_param_optimism_scan(
        df_scan,
        cfg_param,
        param_cols=[c for c in ["z_entry", "z_exit", "lookback", "hl_bars", "corr_min"] if c in df_scan.columns],
        df_score_file=df_param_scores,
        df_fair_value=df_fv,
        w_base=0.6 if use_param else 0.0,
        w_file=0.2 if use_param else 0.0,
        w_fv=0.2 if use_param else 0.0,
    )

    # Fundamentals
    if df_fund is not None and not df_fund.empty and use_fund:
        try:
            df_scan = df_scan.merge(df_fund, how="left", on="pair")
        except Exception:
            logger.exception("SmartScan: merge df_fund failed, ממשיכים בלי merge.")
        df_scan = apply_fundamental_optimism_scan(df_scan, cfg_fund)
    else:
        df_scan["fundamental_score_total"] = 0.5
        df_scan["fundamental_score_details"] = [{} for _ in range(len(df_scan))]

    # Macro layer (scores מתוך MacroOptimismConfig – שונים מ-macro_overlay)
    if use_macro:
        df_scan = apply_macro_optimism_scan(df_scan, cfg_macro)
    else:
        df_scan["macro_score_total"] = 0.5
        df_scan["macro_score_details"] = [{} for _ in range(len(df_scan))]

    # Metrics
    if df_metrics is not None and not df_metrics.empty and use_metrics:
        try:
            df_scan = df_scan.merge(df_metrics, how="left", on="pair")
        except Exception:
            logger.exception("SmartScan: merge df_metrics failed, ממשיכים בלי merge.")
        df_scan = apply_metrics_optimism_scan(df_scan, cfg_metrics)
    else:
        df_scan["metrics_score_total"] = 0.5
        df_scan["metrics_score_details"] = [{} for _ in range(len(df_scan))]

    # Composite
    df_scan = compute_composite_smart_score(
        df_scan,
        profile=profile,
        w_param=w_param,
        w_fund=w_fund,
        w_macro=w_macro,
        w_metrics=w_metrics,
    )

    # אם שכבת Macro Overlay הצליחה (pairs_df עם macro_*), נשרשר לעמודות df_scan
    try:
        if "pair" in df_scan.columns and "pair_id" in pairs_df.columns:
            overlay_cols = [
                "pair_id",
                "macro_multiplier",
                "macro_include",
                "macro_score_overlay",
                "macro_cap_hint",
                "smart_score_with_macro",
            ]

            # ניקוי: נשתמש רק בעמודות שבאמת קיימות ב-pairs_df
            existing_cols = [c for c in overlay_cols if c in pairs_df.columns]
            if not existing_cols:
                logger.warning(
                    "SmartScan: no macro overlay columns found on pairs_df; "
                    "skipping macro overlay merge."
                )
            else:
                df_overlay = pairs_df[existing_cols].copy()

                # מיזוג לפי pair (ב-scan) מול pair_id (ב-pairs_df)
                df_scan = df_scan.merge(
                    df_overlay,
                    how="left",
                    left_on="pair",
                    right_on="pair_id",
                )

                # אם pair_id הגיע רק בשביל המיזוג – אפשר למחוק אותו מה-DF הסופי
                if "pair_id" in df_scan.columns:
                    df_scan.drop(columns=["pair_id"], inplace=True)

    except Exception:
        logger.exception(
            "SmartScan: merge overlay from pairs_df failed, ממשיכים בלי smart_score_with_macro."
        )

    # ===== Persist signals → SqlStore (signals_universe + signals_summary) =====
    try:
        # אם app_ctx אמיתי – נשמור את הסיגנלים ל-DB
        if isinstance(app_ctx, AppContext):
            _persist_signals_to_sql_store(
                df_scan,
                app_ctx=app_ctx,
                profile_name=profile,
            )
        else:
            logger.info(
                "SmartScan: app_ctx is not an AppContext instance (%r) — skipping SqlStore persist.",
                type(app_ctx),
            )
    except Exception:
        logger.exception("SmartScan: failed to persist signals_df to SqlStore")

    # ======================================================
    # 7) Results & Diagnostics — דירוג + בריאות
    # ======================================================
    st.markdown("#### 5️⃣ Results & Diagnostics")

    # נבחר metric לדירוג – אם יש smart_score_with_macro, נציע גם אותו
    ranking_options = []
    for m in [
        "smart_score_total",
        "smart_score_with_macro",
        "param_optimism_score_total",
        "fundamental_score_total",
        "macro_score_total",
        "metrics_score_total",
    ]:
        if m in df_scan.columns:
            ranking_options.append(m)

    ranking_metric = st.selectbox(
        "Ranking metric",
        ranking_options,
        index=ranking_options.index("smart_score_with_macro") if "smart_score_with_macro" in ranking_options else 0,
        key="smart_scan_ranking_metric",
    )

    df_view = df_scan.sort_values(ranking_metric, ascending=False)

    # Health banner — כיסוי שכבות + קשר למשטר
    with st.expander("Scanner Health / Coverage", expanded=False):
        health_rows = []
        layer_cols = {
            "Param": "param_optimism_score_total",
            "Fundamental": "fundamental_score_total",
            "MacroScore": "macro_score_total",
            "Metrics": "metrics_score_total",
        }
        for lname, col in layer_cols.items():
            if col in df_scan.columns:
                vals = pd.to_numeric(df_scan[col], errors="coerce")
                coverage = float(vals.notna().mean())
                health_rows.append(
                    {
                        "layer": lname,
                        "coverage": coverage,
                        "mean": float(vals.mean()),
                    }
                )
        if health_rows:
            st.dataframe(pd.DataFrame(health_rows), width="stretch")

        # אם macro_coverage נמוך מאוד – אזהרה
        macro_coverage = next((r["coverage"] for r in health_rows if r["layer"] == "MacroScore"), None)
        if macro_coverage is not None and macro_coverage < 0.4:
            st.warning("⚠️ Macro layer coverage נמוך (<40%) — כדאי לא להסתמך יותר מדי על macro_score_total.")

    st.markdown(f"**Top 50 by `{ranking_metric}`**")
    st.dataframe(df_view.head(50), width="stretch")

    # ======================================================
    # 8) Stability Scan (weight perturbations)
    # ======================================================
    with st.expander("🧭 Stability Scan (weight perturbations)", expanded=False):
        st.caption(
            "בודק כמה יציב הדירוג תחת שינויים קטנים במשקלי השכבות (Param/Fund/Macro/Metrics)."
        )
        enable_stab = st.checkbox(
            "Run Stability Scan on top-K pairs",
            value=False,
            key="smart_scan_run_stability",
        )
        if enable_stab:
            max_k_stab = max(1, min(100, len(df_view)))
            min_k_stab = 1 if max_k_stab < 5 else 5
            top_k_stab = st.slider(
                "Top-K to test",
                min_value=min_k_stab,
                max_value=max_k_stab,
                value=min(30, max_k_stab),
                step=1 if max_k_stab < 5 else 5,
                key="smart_scan_topk_stab",
            )
            n_scenarios = st.number_input(
                "Number of perturbation scenarios",
                min_value=20,
                max_value=500,
                value=100,
                step=10,
                key="smart_scan_stab_scenarios",
            )
            perturb_scale = st.number_input(
                "Perturbation scale (σ of log-normal noise)",
                min_value=0.01,
                max_value=0.5,
                value=0.15,
                step=0.01,
                key="smart_scan_stab_scale",
            )

            df_top = df_scan.copy().sort_values("smart_score_total", ascending=False).head(int(top_k_stab))

            base_weights = np.array([w_param, w_fund, w_macro, w_metrics], dtype=float)
            base_weights = base_weights / (base_weights.sum() or 1.0)

            pairs_list = df_top["pair"].astype(str).tolist()
            layer_cols = [
                "param_optimism_score_total",
                "fundamental_score_total",
                "macro_score_total",
                "metrics_score_total",
            ]
            layer_mat = df_top[layer_cols].to_numpy(dtype=float)

            in_top_counts = np.zeros(len(pairs_list), dtype=int)
            rng = np.random.default_rng(1337)

            with st.spinner("מריץ Stability Scan עם פרטורבציות למשקלים..."):
                for _ in range(int(n_scenarios)):
                    noise = rng.normal(loc=0.0, scale=float(perturb_scale), size=4)
                    w_pert = base_weights * np.exp(noise)
                    w_pert = w_pert / (w_pert.sum() or 1.0)
                    scores = (layer_mat * w_pert).sum(axis=1)
                    order = np.argsort(-scores)
                    cutoff = max(1, int(len(pairs_list) * 0.3))  # Top 30% בכל סצנריו
                    in_top_counts[order[:cutoff]] += 1

            stability_ratio = in_top_counts / float(n_scenarios)
            stab_df = pd.DataFrame(
                {
                    "pair": pairs_list,
                    "stability_ratio": stability_ratio,
                }
            ).sort_values("stability_ratio", ascending=False)

            st.caption("Higher stability_ratio → הזוג כמעט תמיד נשאר בצמרת גם כשמשנים קצת משקלים.")
            st.dataframe(stab_df.head(50), width="stretch")

            # נסמן stability_ratio ב-df_scan
            stab_map = dict(zip(stab_df["pair"], stab_df["stability_ratio"]))
            df_scan["stability_ratio"] = df_scan["pair"].map(stab_map).fillna(0.0)

            st.session_state["smart_scan_stability"] = stab_df

    # Layer Diagnostics (ממוצעים לכל שכבה)
    with st.expander("Layer Diagnostics (averages)", expanded=False):
        diag_rows = []
        for col_name, layer_name in [
            ("param_optimism_score_total", "Param"),
            ("fundamental_score_total", "Fundamental"),
            ("macro_score_total", "Macro"),
            ("metrics_score_total", "Metrics"),
            ("smart_score_total", "Composite"),
            ("smart_score_with_macro", "Composite+Macro"),
        ]:
            if col_name in df_scan.columns:
                vals = pd.to_numeric(df_scan[col_name], errors="coerce")
                diag_rows.append(
                    {
                        "layer": layer_name,
                        "mean": float(vals.mean()),
                        "min": float(vals.min()),
                        "max": float(vals.max()),
                    }
                )
        if diag_rows:
            st.dataframe(pd.DataFrame(diag_rows), width="stretch")

    # שמירת df_scan ל-session_state
    st.session_state["smart_scan_results"] = df_scan

    # ======================================================
    # 9) Meta ל-dashboard / טאבים אחרים (History, opt hints, וכו')
    # ======================================================
    try:
        best_score = float(pd.to_numeric(df_view["smart_score_total"], errors="coerce").max())
    except Exception:
        best_score = None
    try:
        avg_sharpe = float(
            pd.to_numeric(df_view.get("Sharpe", pd.Series([0.0] * len(df_view))), errors="coerce").mean()
        )
    except Exception:
        avg_sharpe = None

    # שימוש ב-UTC בצורה בטוחה (בלי timezone אובייקט כדי לא להחזיר את הבאג)
    now_utc = datetime.utcnow()

    smart_meta = {
        "scan_run_id": scan_run_id,
        "env": env,
        "profile": profile,
        "regime_profile": regime_profile,
        "regime_resolved": regime_resolved,
        "n_rows": int(df_view.shape[0]),
        "best_smart_score": best_score,
        "avg_sharpe": avg_sharpe,
        "ranking_metric": ranking_metric,
        "layer_weights": {
            "param": w_param,
            "fundamental": w_fund,
            "macro": w_macro,
            "metrics": w_metrics,
        },
        "timestamp_utc": now_utc.isoformat().replace("+00:00", "Z"),
        "macro_regime_label": macro_regime_label,
    }
    st.session_state["smart_scan_last_meta"] = smart_meta

    # History / Time-machine view
    try:
        history = st.session_state.get("smart_scan_history", [])
        if not isinstance(history, list):
            history = []
        history.append(
            {
                "ts_utc": smart_meta["timestamp_utc"],
                "scan_run_id": scan_run_id,
                "env": env,
                "profile": profile,
                "regime_resolved": regime_resolved,
                "ranking_metric": ranking_metric,
                "n_rows": int(df_view.shape[0]),
                "best_smart_score": best_score,
            }
        )
        st.session_state["smart_scan_history"] = history[-50:]
    except Exception:
        pass

    # Auto-select top pair
    top_pair = None
    try:
        top_pair = str(df_view["pair"].iloc[0])
    except Exception:
        top_pair = None

    if top_pair and bool(st.session_state.get("smart_scan_auto_link_flag", True)):
        st.session_state["selected_pair"] = top_pair
        st.caption(f"📌 selected_pair from Smart Scan: `{top_pair}`")

    # ======================================================
    # 10) Actions — Batch / Shortlist / Optimisation hints
    # ======================================================
    st.markdown("#### 6️⃣ Actions — Batch optimisation / Smart Shortlist / optimisation hints")

    # ---------- Smart Shortlist Builder ----------
    with st.expander("🎯 Smart Shortlist Builder", expanded=False):
        st.caption(
            "בונה רשימת מועמדים איכותית תחת מגבלות גיוון (sector / asset_class / style_tag אם קיימים)."
        )

        max_k = max(1, min(100, len(df_view)))
        min_k = 1 if max_k < 5 else 5
        default_k = min(20, max_k)

        shortlist_size = st.number_input(
            "Shortlist size",
            min_value=min_k,
            max_value=max_k,
            value=default_k,
            step=1 if max_k < 5 else 5,
            key="smart_scan_shortlist_size",
        )

        possible_group_cols = [
            c for c in ["sector", "sector_x", "asset_class", "style_tag"] if c in df_view.columns
        ]
        group_col = st.selectbox(
            "Diversification group column (optional)",
            options=["(none)"] + possible_group_cols,
            index=0,
            key="smart_scan_shortlist_group_col",
        )
        max_per_group = st.number_input(
            "Max pairs per group (אם נבחר group column)",
            min_value=1,
            max_value=int(shortlist_size),
            value=min(3, int(shortlist_size)),
            step=1,
            key="smart_scan_shortlist_max_per_group",
        )

        if st.button("Build Smart Shortlist", key="smart_scan_build_shortlist"):
            df_sorted = df_view.copy()
            picked_rows = []
            group_counts: Dict[str, int] = {}
            for _, row in df_sorted.iterrows():
                if len(picked_rows) >= int(shortlist_size):
                    break
                if group_col != "(none)":
                    grp = str(row.get(group_col, "unknown"))
                    cnt = group_counts.get(grp, 0)
                    if cnt >= int(max_per_group):
                        continue
                    group_counts[grp] = cnt + 1
                picked_rows.append(row)

            if picked_rows:
                shortlist_df = pd.DataFrame(picked_rows).reset_index(drop=True)
                st.session_state["smart_scan_shortlist"] = shortlist_df
                st.success(f"נבנתה רשימת Shortlist של {len(shortlist_df)} זוגות.")
                st.dataframe(shortlist_df, width="stretch")
            else:
                st.info("לא נבחרו זוגות ל-Shortlist (יתכן שמגבלות הגיוון היו מחמירות מדי).")

    # ---------- Batch pairs for optimisation ----------
    top_k_batch = st.slider(
        "Top-K pairs for optimisation batch",
        min_value=5,
        max_value=min(50, len(df_view)),
        value=min(10, len(df_view)),
        step=5,
        key="smart_scan_topk_batch",
    )
    top_pairs_for_batch = df_view.head(int(top_k_batch))["pair"].astype(str).tolist()

    if st.button("📤 Send Top-K pairs to Optimisation batch", key="smart_scan_send_to_opt"):
        st.session_state["opt_batch_pairs"] = top_pairs_for_batch
        st.success(f"Sent {len(top_pairs_for_batch)} pairs to optimisation batch (opt_batch_pairs).")

    # ---------- opt_pair_status hint לטאב האופטימיזציה ----------
    if top_pair:
        if avg_sharpe is not None and avg_sharpe < 0.5:
            opt_profile_hint = "defensive"
            scenario_profile = "Risk-Off"
            tail_weight = 0.5
        elif avg_sharpe is not None and avg_sharpe > 1.5:
            opt_profile_hint = "aggressive"
            scenario_profile = "Risk-On"
            tail_weight = 0.2
        else:
            opt_profile_hint = "default"
            scenario_profile = "Neutral"
            tail_weight = 0.3

        opt_hint = {
            "primary_objective": "Sharpe",
            "profile": opt_profile_hint,
            "scenario_profile": scenario_profile,
            "scenario_tail_weight": tail_weight,
            "wf_use": True,
            "macro_risk_budget_hint": macro_risk_budget_hint,
            "macro_regime_label": macro_regime_label,
        }
        st.session_state["opt_pair_status"] = {
            "pair": top_pair,
            "opt_hint": opt_hint,
            "scan_meta": smart_meta,
        }

    # ---------- History / Time-machine view ----------
    with st.expander("📆 Smart Scan History (this session)", expanded=False):
        hist = st.session_state.get("smart_scan_history", [])
        if not hist:
            st.caption("לא בוצעו עדיין סריקות לשמירה בהיסטוריה.")
        else:
            df_hist = pd.DataFrame(hist)
            st.dataframe(df_hist.sort_values("ts_utc"), width="stretch", height=260)

            if len(df_hist) >= 2:
                c1, c2 = st.columns(2)
                idx_a = c1.number_input(
                    "Index A (0 = oldest)",
                    min_value=0,
                    max_value=len(df_hist) - 1,
                    value=max(0, len(df_hist) - 2),
                    step=1,
                    key="smart_hist_idx_a",
                )
                idx_b = c2.number_input(
                    "Index B (0 = oldest)",
                    min_value=0,
                    max_value=len(df_hist) - 1,
                    value=len(df_hist) - 1,
                    step=1,
                    key="smart_hist_idx_b",
                )
                if 0 <= idx_a < len(df_hist) and 0 <= idx_b < len(df_hist):
                    a = df_hist.iloc[int(idx_a)]
                    b = df_hist.iloc[int(idx_b)]
                    st.markdown("**השוואת שתי סריקות (A vs B):**")
                    st.write(
                        {
                            "A_ts": a["ts_utc"],
                            "B_ts": b["ts_utc"],
                            "Δ best_smart_score": (b.get("best_smart_score") or 0.0)
                            - (a.get("best_smart_score") or 0.0),
                            "A_regime": a.get("regime_resolved"),
                            "B_regime": b.get("regime_resolved"),
                        }
                    )

    st.caption("✅ Smart Scan finished — all results + hints + history stored in session_state.")
