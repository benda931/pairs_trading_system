# -*- coding: utf-8 -*-
"""
core/feature_selection.py
=========================

Feature Selection for Pairs Trading Optimization (HF-grade)
-----------------------------------------------------------

תפקיד המודול
------------
לבחור *סט פרמטרים / Features* "בריא" ורלוונטי מתוך תוצאות אופטימיזציה (`opt_results`),
בצורה שמתאימה למערכת מסחר ברמת קרן:

- מתחשב **במדד איכות מרכזי** (Sharpe / score / hf_score / וכו').
- מנקה עמודות לא שמישות (NaN, קבועות, דלילות).
- מסנן **קורלציה גבוהה** בין פרמטרים, ושומר את היותר "מחובר לתוצאה".
- מאפשר קונפיגורציה של:
    * metric יעד
    * correlation threshold
    * min / top_n
- מחזיר DataFrame מסודר של *trials נבחרים* + עמודות פרמטרים שעברו פילטר.

המודול כתוב כך שיהיה:
- דפנסיבי (לא מפיל את המערכת על דאטה בעייתי).
- ברירת מחדל "safe" כאשר חסר מידע בקונפיג.
- ניתן להרחבה (SHAP, Meta-Selection וכו') בלי לשבור API קיים.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import logging

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


# -------------------------
# Helpers: metric & columns
# -------------------------


_METRIC_CANDIDATES_ORDER: Sequence[str] = (
    # HF-grade metrics
    "score",
    "hf_score",
    "classic_score",
    # legacy / simpler metrics
    "sharpe",
    "Sharpe",
    "return",
    "Profit",
    "total_return",
)


_DEFAULT_EXCLUDE_COLS: Sequence[str] = (
    "trial_id",
    "study_name",
    "run_id",
    "pair",
    "pairs",
    "pair_key",
    "timestamp",
    "created_at",
    "updated_at",
    "profile",
    "env",
    "mode",
    "preset",
    "opt_mode",
    "opt_preset",
    "opt_objective",
    "source",
    "_source",
)


def _resolve_metric_column(
    df: pd.DataFrame,
    fs_cfg: Mapping[str, Any],
) -> Optional[str]:
    """
    בוחר עמודת Metric מרכזית מתוך opt_results לפי:
      1. feature_selection.metric בקונפיג (אם קיים).
      2. סדר עדיפויות מובנה (score, hf_score, sharpe, ...).
    """
    # מהקונפיג
    cfg_metric = fs_cfg.get("metric")
    if isinstance(cfg_metric, str) and cfg_metric in df.columns:
        return cfg_metric

    # לפי רשימה קבועה
    for cand in _METRIC_CANDIDATES_ORDER:
        if cand in df.columns:
            return cand

    logger.warning(
        "feature_selection: לא נמצא metric מתאים ב-DataFrame. "
        "עמודות קיימות: %s",
        list(df.columns),
    )
    return None


def _get_param_columns(
    df: pd.DataFrame,
    metric_col: Optional[str],
    extra_exclude: Optional[Iterable[str]] = None,
) -> List[str]:
    """
    מגדיר רשימת עמודות "פרמטרים" (features) מתוך opt_results:
    - מסיר עמודות metric/score ועמודות metadata נפוצות.
    - שומר רק עמודות מספריות (float/int).
    """
    exclude: List[str] = list(_DEFAULT_EXCLUDE_COLS)
    if metric_col:
        exclude.append(metric_col)

    if extra_exclude:
        for c in extra_exclude:
            if isinstance(c, str):
                exclude.append(c)

    exclude_set = set(exclude)

    param_cols: List[str] = []
    for col in df.columns:
        if col in exclude_set:
            continue
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            param_cols.append(col)

    return param_cols


def _basic_column_cleanup(
    df: pd.DataFrame,
    param_cols: List[str],
    *,
    max_nan_frac: float = 0.5,
) -> List[str]:
    """
    שלב ניקוי ראשוני:
    - מסיר עמודות שכל הערכים בהן NaN.
    - מסיר עמודות עם אחוז NaN גבוה מדי.
    - מסיר עמודות קבועות (variance=0).
    """
    kept: List[str] = []
    for col in param_cols:
        s = df[col]
        # NaN-only
        if s.isna().all():
            logger.debug("feature_selection: dropping column %s (all NaN)", col)
            continue

        # יותר מדי NaN
        nan_frac = float(s.isna().mean())
        if nan_frac > max_nan_frac:
            logger.debug(
                "feature_selection: dropping column %s (nan_frac=%.2f)",
                col,
                nan_frac,
            )
            continue

        # constant / near-constant
        s_clean = s.dropna()
        if len(s_clean) <= 1:
            continue
        if float(s_clean.std(ddof=0)) == 0.0:
            logger.debug("feature_selection: dropping column %s (constant)", col)
            continue

        kept.append(col)

    return kept


# ---------------------------------------------------
# Main HF-grade feature selection (correlation filter)
# ---------------------------------------------------


def _correlation_filter(
    df: pd.DataFrame,
    param_cols: List[str],
    metric_col: Optional[str],
    *,
    corr_threshold: float = 0.95,
) -> List[str]:
    """
    מסנן עמודות בעלות מתאם גבוה ביניהן.
    כאשר יש שני פרמטרים עם קורלציה גבוהה:
      - נשמור את זה שיש לו קורלציה *חזקה יותר* עם ה-Metric.
      - אם אין metric, נשמור סכמתית את העמודה הראשונה.

    מחזיר רשימת עמודות שנשארו.
    """
    if not param_cols:
        return []

    sub = df[param_cols].copy()
    # החלפת אינפים / NaN
    sub = sub.astype("float64").replace([np.inf, -np.inf], np.nan)
    sub = sub.fillna(sub.median(numeric_only=True))

    corr = sub.corr().abs()
    if corr.empty:
        return param_cols

    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    cols_to_keep = set(param_cols)
    cols_to_drop = set()

    if metric_col and metric_col in df.columns:
        metric_series = df[metric_col].astype("float64")
        metric_series = metric_series.replace([np.inf, -np.inf], np.nan)
        metric_series = metric_series.fillna(metric_series.median())
        # קשר "איכות" – נשתמש במתאם Spearman (רגיש ליחס מונוטוני)
        metric_corr = sub.apply(
            lambda c: metric_series.corr(c, method="spearman")
        ).abs()
    else:
        metric_corr = None

    for col in upper.columns:
        if col not in cols_to_keep:
            continue
        # כל העמודות שמעל הסף ב-abs corr
        high_corr_partners = [
            row_col
            for row_col, val in upper[col].items()
            if (not np.isnan(val)) and (val >= corr_threshold)
        ]
        for partner in high_corr_partners:
            if partner not in cols_to_keep:
                continue

            # בוחרים איזה משניים להשאיר
            if metric_corr is not None:
                c1 = float(metric_corr.get(col, 0.0))
                c2 = float(metric_corr.get(partner, 0.0))
                # נשאיר את זה עם abs corr גבוה יותר ל-metric
                if c1 >= c2:
                    drop_col = partner
                    keep_col = col
                else:
                    drop_col = col
                    keep_col = partner
            else:
                # בלי metric – נשמור את הראשון (אינדיקציה גסה)
                drop_col = partner
                keep_col = col

            if drop_col in cols_to_keep:
                cols_to_keep.remove(drop_col)
                cols_to_drop.add(drop_col)
                logger.debug(
                    "feature_selection: dropping %s (high corr with %s)",
                    drop_col,
                    keep_col,
                )

    logger.info(
        "feature_selection: correlation filter → kept %d params, dropped %d",
        len(cols_to_keep),
        len(cols_to_drop),
    )
    return sorted(cols_to_keep)


# -----------------
# Main entry point
# -----------------


def select_features(opt_results: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Select optimal features/parameters using statistics and (future) SHAP importance.

    Args
    ----
    opt_results:
        DataFrame, output of run_optimization (עם כל הפרמטרים/מטריקות).
        צפוי להכיל לפחות:
            - אחד מ: ["score", "hf_score", "classic_score", "sharpe", ...]
            - עמודות פרמטרים מוזנים.

    config:
        dict עם בלוק feature_selection אופציונלי, למשל:

        feature_selection:
            metric: "hf_score"         # metric יעד
            corr_threshold: 0.90       # סף מתאם בין פרמטרים
            max_nan_frac: 0.5          # אחוז NaN מקסימלי לעמודה
            top_n: 30                  # כמה trials לבחור
            min_trials: 10             # מינימום רשומות כדי לעשות בחירה מורכבת
            extra_exclude: ["_source"] # עמודות להחרגה ידנית

    Returns
    -------
    DataFrame
        - מכיל רק את העמודות שנשארו (metrics + features שנבחרו).
        - מסודר לפי metric (יורד).
        - כולל רק top_n רשומות.
        אם opt_results ריק → DataFrame ריק.
    """

    if opt_results is None or opt_results.empty:
        logger.warning("feature_selection: opt_results is empty – returning empty DataFrame.")
        return pd.DataFrame()

    fs_cfg: Dict[str, Any] = (config or {}).get("feature_selection", {}) or {}

    # ---- 1. resolve metric column ----
    metric_col = _resolve_metric_column(opt_results, fs_cfg)
    if metric_col is None:
        # fallback: אם אין metric ברור – נחזיר את כל הדאטה כפי שהוא (או top_n לפי Sharpe אם יש)
        logger.warning(
            "feature_selection: לא נמצא metric, מחזיר top_n (אם מוגדר) בלי Feature filtering."
        )
        top_n = int(fs_cfg.get("top_n", 20))
        if "sharpe" in opt_results.columns:
            return (
                opt_results.sort_values("sharpe", ascending=False)
                .head(top_n)
                .reset_index(drop=True)
            )
        # לא מצאנו שום metric סביר – נחזיר פשוט את הראשונים
        return opt_results.head(top_n).reset_index(drop=True)

    # ---- 2. מינימום רשומות כדי להריץ pipeline מלא ----
    min_trials = int(fs_cfg.get("min_trials", 10))
    if len(opt_results) < min_trials:
        logger.info(
            "feature_selection: פחות מ-%d רשומות (%d) – מדלג על feature selection מתקדם.",
            min_trials,
            len(opt_results),
        )
        top_n = int(fs_cfg.get("top_n", 20))
        return (
            opt_results.sort_values(metric_col, ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )

    # ---- 3. קביעת רשימת עמודות פרמטרים ----
    extra_exclude = fs_cfg.get("extra_exclude", []) or []
    param_cols = _get_param_columns(opt_results, metric_col, extra_exclude=extra_exclude)

    if not param_cols:
        logger.warning("feature_selection: לא נמצאו עמודות פרמטרים – מחזיר לפי metric בלבד.")
        top_n = int(fs_cfg.get("top_n", 20))
        return (
            opt_results.sort_values(metric_col, ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )

    # ---- 4. ניקוי בסיסי של עמודות (NaN, constant) ----
    max_nan_frac = float(fs_cfg.get("max_nan_frac", 0.5))
    cleaned_param_cols = _basic_column_cleanup(opt_results, param_cols, max_nan_frac=max_nan_frac)

    if not cleaned_param_cols:
        logger.warning("feature_selection: כל הפרמטרים נפסלו בניקוי בסיסי – נחזור ל-param_cols המקורי.")
        cleaned_param_cols = param_cols

    # ---- 5. מסנן קורלציה בין פרמטרים (HF-grade) ----
    corr_threshold = float(fs_cfg.get("corr_threshold", 0.95))
    filtered_param_cols = _correlation_filter(
        opt_results,
        cleaned_param_cols,
        metric_col,
        corr_threshold=corr_threshold,
    )

    # במקרה קצה – אם מסנן הקורלציה "אכל הכל", נחזור ל-cleaned
    if not filtered_param_cols:
        logger.warning(
            "feature_selection: correlation filter removed all params – reverting to cleaned_param_cols."
        )
        filtered_param_cols = cleaned_param_cols

    # ---- 6. דירוג trials לפי metric + אופציונלית מטריקות נוספות ----
    sort_by: List[Tuple[str, bool]] = [(metric_col, False)]  # (column, ascending?)

    # אפשרויות: אם יש 'Drawdown' / 'drawdown' / 'ES_95' – לתת להם משקל משני בסידור
    if "Drawdown" in opt_results.columns:
        sort_by.append(("Drawdown", True))
    elif "drawdown" in opt_results.columns:
        sort_by.append(("drawdown", True))

    if "ES_95" in opt_results.columns:
        sort_by.append(("ES_95", True))

    # בונים רשימה מתאימה ל-sort_values
    sort_cols = [c for c, _ in sort_by if c in opt_results.columns]
    ascending_flags = [asc for c, asc in sort_by if c in opt_results.columns]

    df_sorted = (
        opt_results.sort_values(by=sort_cols, ascending=ascending_flags)
        if sort_cols
        else opt_results.copy()
    )

    top_n = int(fs_cfg.get("top_n", 20))
    df_top = df_sorted.head(top_n).reset_index(drop=True)

    # ---- 7. בניית DataFrame סופי עם פרמטרים שנבחרו + מטריקות רלוונטיות ----
    # נשאיר:
    #   - metric_col
    #   - כל עמודת מטריקה נוספת (Sharpe/score/drawdown) אם יש
    #   - הפרמטרים שנבחרו
    metric_like_cols = [
        c
        for c in opt_results.columns
        if c in ("score", "hf_score", "classic_score", "sharpe", "Sharpe", "return", "drawdown", "Drawdown", "ES_95")
    ]
    cols_to_keep = sorted(set([metric_col] + metric_like_cols + filtered_param_cols))

    df_out = df_top[cols_to_keep].copy()

    # ---- 8. אפשרות לעיטור: rank ו-zscore למטריק ----
    try:
        metric_values = df_out[metric_col].astype("float64")
        df_out["metric_rank"] = metric_values.rank(ascending=False, method="min").astype(int)
        if metric_values.std(ddof=0) > 0:
            df_out["metric_zscore"] = (metric_values - metric_values.mean()) / metric_values.std(ddof=0)
        else:
            df_out["metric_zscore"] = 0.0
    except Exception as exc:
        logger.debug("feature_selection: failed to compute rank/zscore for metric: %s", exc)

    # ---- 9. TODO: SHAP importance fallback (להמשך) ----
    # כאן אפשר בעתיד להוסיף:
    # - אם יש SHAP values עבור trials (למשל בדאטה חיצוני/אובייקט), לשלב אותם לצורך דירוג features.
    # - או לחבר ל-core/automl_tools כדי לקבל Feature importance ממודל ML.
    # כרגע זה רק מקום מוגדר להרחבה.

    logger.info(
        "feature_selection: selected %d trials with %d parameters (metric=%s, top_n=%d)",
        len(df_out),
        len(filtered_param_cols),
        metric_col,
        top_n,
    )
    return df_out
