# core/meta_optimizer.py
# -*- coding: utf-8 -*-
"""
Meta-Optimization for Pairs Trading
===================================

מטרת המודול:
------------
לקחת טבלת "קלאסטרים" / סטים של פרמטרים (למשל פלט של cluster_pairs / grid-search),
ולבנות מעליהם שכבת Meta-Scoring ברמת קרן גידור:

- נרמול מטריקות שונות לסקאלה [0, 1].
- שילוב משקולות וסימן (higher/lower is better) לכל מטריקה.
- חישוב Meta-Score כולל לכל שורה.
- בחירת סט פרמטרים "Winner" ברמת Meta.
- חישוב Feature Importance (עד כמה כל פרמטר משפיע על ה-Meta Score).

המודול מחזיר:
    {
        "best_params": dict,
        "feature_importance": pd.DataFrame,
        "all_scores": pd.DataFrame,
    }

כך שטאב האופטימיזציה / הדשבורד יכולים להציג:
- את סט הפרמטרים הטוב ביותר.
- את השקלול המלא לכל הקלאסטרים.
- את החשיבות היחסית של כל פרמטר.
"""

from __future__ import annotations

from typing import Dict, Any, Mapping, Optional, List

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s [%(name)s] [%(levelname)s] %(message)s"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)


# ============================================================
# Helpers — metric spec & normalization
# ============================================================

def _get_metric_spec(config: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    מחלץ spec של המטריקות מתוך ה-config.

    פורמט גמיש:

    1) פורמט חדש (מומלץ):
        config["meta_optimize"]["metrics"] = {
            "sharpe":   {"weight": 0.3, "higher_is_better": True},
            "sortino":  {"weight": 0.15, "higher_is_better": True},
            "calmar":   {"weight": 0.10, "higher_is_better": True},
            "return":   {"weight": 0.10, "higher_is_better": True},
            "win_rate": {"weight": 0.05, "higher_is_better": True},
            "drawdown": {"weight": 0.10, "higher_is_better": False},
            "volatility": {"weight": 0.05, "higher_is_better": False},
            "tail_risk": {"weight": 0.05, "higher_is_better": False},
            "turnover": {"weight": 0.05, "higher_is_better": False},
            "skew":     {"weight": 0.05, "higher_is_better": True},
        }

    2) פורמט ישן (Backwards compatible):
        config["meta_optimize"]["weights"] = {
            "sharpe": 0.5,
            "return": 0.2,
            "drawdown": 0.2,
            "win_rate": 0.1,
        }

    אם אין שום הגדרה, משתמשים בברירת מחדל מורחבת (Sharpe + Sortino + Return + Drawdown + Volatility + Tail Risk וכו')
    אבל בפועל *נלקחות בחשבון רק המטריקות שבאמת קיימות* ב-DataFrame של clusters.
    """
    meta_cfg = config.get("meta_optimize", {}) if isinstance(config, Mapping) else {}

    # ---- פורמט חדש: metrics ----
    metrics = meta_cfg.get("metrics")
    if isinstance(metrics, Mapping) and metrics:
        spec: Dict[str, Dict[str, Any]] = {}
        for name, m in metrics.items():
            if not isinstance(m, Mapping):
                continue
            weight = float(m.get("weight", 0.0))
            if weight == 0.0:
                continue
            higher_is_better = bool(m.get("higher_is_better", True))
            spec[name] = {
                "weight": weight,
                "higher_is_better": higher_is_better,
            }
        if spec:
            return spec

    # ---- פורמט ישן: weights ----
    weights = meta_cfg.get("weights", {}) if isinstance(meta_cfg, Mapping) else {}
    if isinstance(weights, Mapping) and weights:
        spec = {}
        for name, w in weights.items():
            weight = float(w)
            if weight == 0.0:
                continue

            lower_name = name.lower()
            # כברירת מחדל: most performance metrics – higher is better
            # חריגים ברורים: drawdown, volatility, tail_risk, turnover
            if lower_name in {"drawdown", "vol", "volatility", "tail_risk", "turnover"}:
                hib = False
            else:
                hib = True

            spec[name] = {"weight": weight, "higher_is_better": hib}
        if spec:
            return spec

    # ---- ברירת מחדל מורחבת (אם לא הוגדר כלום בקונפיג) ----
    logger.info("meta_optimizer: using extended default metric spec.")
    return {
        # איכות תשואה
        "sharpe":     {"weight": 0.30, "higher_is_better": True},
        "sortino":    {"weight": 0.15, "higher_is_better": True},
        "calmar":     {"weight": 0.10, "higher_is_better": True},
        "return":     {"weight": 0.10, "higher_is_better": True},
        "win_rate":   {"weight": 0.05, "higher_is_better": True},

        # סיכון / זנבות / תנודתיות
        "drawdown":   {"weight": 0.10, "higher_is_better": False},
        "volatility": {"weight": 0.05, "higher_is_better": False},
        "tail_risk":  {"weight": 0.05, "higher_is_better": False},

        # מבנה / סחירות / יציבות
        "turnover":   {"weight": 0.05, "higher_is_better": False},  # פחות turnover = עדיף
        "skew":       {"weight": 0.05, "higher_is_better": True},   # skew חיובי עדיף בד"כ
    }

def _normalize_series(
    series: pd.Series,
    higher_is_better: bool,
) -> pd.Series:
    """
    נרמול מטריקה לסקאלה [0, 1] בצורה יציבה:

    - אם higher_is_better=True:
        norm = (x - min) / (max - min)
    - אם higher_is_better=False:
        norm = (max - x) / (max - min)

    טיפול במקרים מיוחדים:
    - אם max == min (כל הערכים שווים) => מחזירים 0.5 לכל השורות.
    - מתעלמים מ-Nan בבניית המינימום/מקסימום.
    """
    s = series.astype(float)
    mask_valid = s.notna()
    if not mask_valid.any():
        # הכל NaN -> מחזירים 0.5
        return pd.Series(0.5, index=series.index)

    s_valid = s[mask_valid]
    v_min = float(s_valid.min())
    v_max = float(s_valid.max())

    if np.isclose(v_max, v_min):
        # אין שונות – אין מידע; נותנים 0.5 ניטרלי
        norm = pd.Series(0.5, index=series.index)
        return norm

    denom = v_max - v_min

    if higher_is_better:
        norm_values = (s - v_min) / denom
    else:
        norm_values = (v_max - s) / denom

    norm = norm_values.clip(0.0, 1.0)
    norm[~mask_valid] = 0.5  # NaN מקבל ערך ניטרלי
    return norm


def _build_meta_scores(
    clusters: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """
    מקבל DataFrame של קלאסטרים/סטי פרמטרים, ומוסיף לו:

      - col_norm לכל Metric.
      - meta_score – סכום משוקלל של כל המטריקות.
      - meta_rank – דירוג יורד של meta_score (1=הטוב ביותר).

    מחזיר DataFrame חדש עם כל העמודות המקוריות + עמודות meta_*.
    """
    df = clusters.copy()
    metric_spec = _get_metric_spec(config)

    if df.empty:
        logger.info("meta_optimizer: received empty clusters DataFrame.")
        return df

    # נרמול והוספת עמודות *_norm
    meta_components: List[pd.Series] = []
    weights_used: List[float] = []

    for metric_name, spec in metric_spec.items():
        col = metric_name
        if col not in df.columns:
            logger.warning(
                "meta_optimizer: metric '%s' not found in clusters columns, skipping.",
                metric_name,
            )
            continue

        higher_is_better = bool(spec.get("higher_is_better", True))
        weight = float(spec.get("weight", 0.0))
        if weight == 0.0:
            continue

        norm_col_name = f"{col}_norm"
        df[norm_col_name] = _normalize_series(df[col], higher_is_better)
        meta_components.append(df[norm_col_name] * weight)
        weights_used.append(weight)

    if not meta_components:
        logger.warning(
            "meta_optimizer: no valid metrics for scoring – returning DataFrame without meta_score."
        )
        df["meta_score"] = np.nan
        df["meta_rank"] = np.nan
        return df

    total_weight = float(sum(weights_used))
    if total_weight == 0.0:
        total_weight = 1.0  # הגנה מיותרת

    # meta_score = Σ (weight * normalized_metric) / Σ weights
    df["meta_score"] = sum(meta_components) / total_weight

    # Ranking – גדול יותר = טוב יותר
    df["meta_rank"] = df["meta_score"].rank(method="min", ascending=False)

    return df


# ============================================================
# Feature Importance
# ============================================================

def _compute_feature_importance(
    scored_df: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """
    מחשב feature importance פשוט על בסיס:
        |corr(feature, meta_score)|

    תומך רק בעמודות מספריות ומחזיר DataFrame:
        feature, importance

    הערה:
      - מוריד NaN לפני החישוב.
      - אם אין שונות ב-meta_score => importance=0 לכל הפיצ'רים.
    """
    df = scored_df.copy()
    if "meta_score" not in df.columns:
        return pd.DataFrame(columns=["feature", "importance"])

    meta = pd.to_numeric(df["meta_score"], errors="coerce")
    if meta.notna().nunique() <= 1:
        # אין שונות ב-meta_score => אין מה ללמוד
        return pd.DataFrame(columns=["feature", "importance"])

    # אילו עמודות נחשבות "פרמטרים"?
    ranges_cfg = config.get("ranges", {}) if isinstance(config, Mapping) else {}
    param_names = set(ranges_cfg.keys())

    feat_cols: List[str] = []
    for col in df.columns:
        if col in param_names:
            feat_cols.append(col)

    importance_rows: List[Dict[str, Any]] = []
    for f in feat_cols:
        s = pd.to_numeric(df[f], errors="coerce")
        # אם אין שונות – corr לא מוגדר
        if s.notna().nunique() <= 1:
            importance_rows.append({"feature": f, "importance": 0.0})
            continue

        corr = s.corr(meta)
        if pd.isna(corr):
            imp = 0.0
        else:
            imp = float(abs(corr))

        importance_rows.append({"feature": f, "importance": imp})

    if not importance_rows:
        return pd.DataFrame(columns=["feature", "importance"])

    feature_importance = pd.DataFrame(importance_rows)
    feature_importance = feature_importance.sort_values("importance", ascending=False)
    feature_importance.reset_index(drop=True, inplace=True)
    return feature_importance


# ============================================================
# Public API
# ============================================================

def meta_optimize(
    clusters: pd.DataFrame,
    config: dict,
) -> dict:
    """
    Meta-optimizes across clusters to find the best blend of parameters.

    הקלט:
    -----
    clusters:
        DataFrame שמגיע, למשל, מ-cluster_pairs או מפייפליין אופטימיזציה.
        אמור להכיל:
            - עמודות מטריקות (sharpe, return, drawdown, win_rate וכו').
            - עמודות פרמטרים (לפי config["ranges"].keys()).

    config:
        dict של הגדרות. שימושים רלוונטיים:
            - config["meta_optimize"]["metrics"]   (פורמט חדש, מומלץ)
            - config["meta_optimize"]["weights"]   (פורמט ישן, נתמך)
            - config["ranges"]                     (מיפוי פרמטרים לשם העמודה).

    הפלט:
    -----
    {
        "best_params": dict  – סט הפרמטרים עם meta_score הטוב ביותר.
        "feature_importance": pd.DataFrame(feature, importance),
        "all_scores": pd.DataFrame – טבלת הקלאסטרים כולל meta_score/meta_rank.
    }

    אם clusters ריק – מחזיר {}.
    """
    if clusters is None or clusters.empty:
        logger.info("meta_optimizer.meta_optimize: empty clusters, nothing to do.")
        return {}

    # 1) חישוב meta_score + meta_rank
    scored_df = _build_meta_scores(clusters, config)

    if "meta_score" not in scored_df.columns or scored_df["meta_score"].isna().all():
        logger.warning("meta_optimizer: meta_score could not be computed – returning minimal result.")
        return {
            "best_params": {},
            "feature_importance": pd.DataFrame(columns=["feature", "importance"]),
            "all_scores": scored_df,
        }

    # 2) בחירת השורה הטובה ביותר
    best_idx = scored_df["meta_score"].idxmax()
    best_row = scored_df.loc[best_idx]

    # הפרמטרים הרלוונטיים – לפי config["ranges"].keys()
    ranges_cfg = config.get("ranges", {}) if isinstance(config, Mapping) else {}
    param_names = set(ranges_cfg.keys())

    best_params: Dict[str, Any] = {}
    for p_name in param_names:
        if p_name in best_row.index:
            best_params[p_name] = best_row[p_name]

    # 3) Feature Importance
    feature_importance = _compute_feature_importance(scored_df, config)

    logger.info(
        "meta_optimizer: selected best params (meta_score=%.6f): %s",
        float(best_row["meta_score"]),
        best_params,
    )

    return {
        "best_params": best_params,
        "feature_importance": feature_importance,
        "all_scores": scored_df,
    }
