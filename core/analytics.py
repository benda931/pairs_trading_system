# core/analytics.py
# -*- coding: utf-8 -*-
"""
Analytics & Reporting for Pairs Trading Optimization
====================================================

מודול אנליטיקה ברמת קרן גידור מעל תוצאות אופטימיזציה ו-meta-optimization.

מטרות עיקריות
-------------
1. לקבל את פלט ה-meta-optimizer (meta_optimize) ולבנות ממנו:
   - סיכום גלובלי של המערכת.
   - סיכום לפי זוג (pair) / cluster key.
   - דירוגים (global_rank, rank_in_pair).
   - טבלת ניהול ברורה ל-Streamlit / יצוא ל-CSV.

2. לזהות באופן אוטומטי:
   - עמודת meta_score / Score.
   - עמודות Sharpe / Sortino / Calmar / Profit / Return / Drawdown / WinRate.
   - עמודת pair (pair_id / Pair / pair).

3. לספק פונקציות עזר:
   - summarize_results(meta, full=True)  → DataFrame סיכום מקצועי.
   - summarize_by_pair(df)              → סיכום לפי זוג.
   - summarize_by_param(df, params=..)  → סיכום לפי פרמטר (sensitivity-style).
   - compute_metric_matrix(df)          → מטריצת קורלציה בין מטריקות.

המודול לא תלוי ב-Streamlit – מתאים גם לשימוש מתוך CLI/סקריפט/סוכן AI.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)


# =====================================================================
# Helpers: column inference (pair / metrics / params)
# =====================================================================

# מילון קנוניזציה – מזהה שמות שונים של אותה מטריקה
_METRIC_ALIAS_MAP: Dict[str, str] = {
    # meta score
    "meta_score": "meta_score",
    "metascore": "meta_score",
    "score": "meta_score",
    "score_agg": "meta_score",

    # sharpe
    "sharpe": "sharpe",
    "sharpe_ratio": "sharpe",

    # sortino
    "sortino": "sortino",
    "sortino_ratio": "sortino",

    # calmar
    "calmar": "calmar",
    "calmar_ratio": "calmar",

    # profit / return
    "profit": "profit",
    "pnl": "profit",
    "pnl_usd": "profit",
    "ret": "return",
    "return": "return",
    "cagr": "return",
    "total_return": "return",

    # drawdown
    "drawdown": "drawdown",
    "max_drawdown": "drawdown",
    "maxdd": "drawdown",
    "dd": "drawdown",

    # win rate
    "win_rate": "win_rate",
    "winrate": "win_rate",
    "hit_rate": "win_rate",
    "hitrate": "win_rate",
}

# סדר עדיפויות לקולון pair
_PAIR_CANDIDATES = ["pair_id", "pair", "Pair", "PAIR"]


def _infer_pair_column(df: pd.DataFrame) -> Optional[str]:
    """
    מזהה עמודת pair (אם קיימת).

    לוגיקה:
    - מנסה pair_id / pair / Pair לפי סדר עדיפויות.
    - אם אין, אבל יש symbol_a + symbol_b → יוצר pair_label = symA_symB.
    - אחרת מחזיר None.
    """
    for col in _PAIR_CANDIDATES:
        if col in df.columns:
            return col

    if {"symbol_a", "symbol_b"}.issubset(df.columns):
        # נבנה עמודת pair חדשה
        df["pair"] = (
            df["symbol_a"].astype(str).str.strip()
            + "_"
            + df["symbol_b"].astype(str).str.strip()
        )
        return "pair"

    return None


def _infer_metric_columns(df: pd.DataFrame) -> Dict[str, str]:
    """
    מאתר עמודות מטריקות ומחזיר mapping:

        {"meta_score": "meta_score_colname", "sharpe": "Sharpe", ...}

    רק מטריקות שקיימות בפועל ב-DataFrame נכללות.
    """
    metrics: Dict[str, str] = {}
    col_map: Dict[str, str] = {c.lower(): c for c in df.columns}

    for key_lower, canon in _METRIC_ALIAS_MAP.items():
        if key_lower in col_map:
            metrics[canon] = col_map[key_lower]

    return metrics


def _infer_param_columns(
    df: pd.DataFrame,
    metric_cols: Dict[str, str],
    pair_col: Optional[str] = None,
    extra_exclude: Optional[List[str]] = None,
) -> List[str]:
    """
    מזהה עמודות פרמטרים: כל עמודה מספרית שאינה:
    - pair / symbol_a / symbol_b / study_id / meta_score / Sharpe / Profit/Drawdown וכו'.
    """
    exclude: List[str] = []
    if pair_col:
        exclude.append(pair_col)
    exclude += ["symbol_a", "symbol_b", "study_id", "Pair", "pair"]
    # exclude metric columns
    exclude += list(metric_cols.values())
    if extra_exclude:
        exclude += extra_exclude

    exclude_set = {e for e in exclude if e in df.columns}

    param_cols: List[str] = []
    for c in df.columns:
        if c in exclude_set:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            param_cols.append(c)

    return param_cols


# =====================================================================
# Core summary functions
# =====================================================================

def summarize_results(meta: dict, full: bool = True) -> pd.DataFrame:
    """
    Summarize all scores and clusters from meta-optimizer.

    שימוש טיפוסי:
    -------------
        meta = meta_optimize(clusters, config)
        summary_df = summarize_results(meta, full=True)

    Args
    ----
    meta:
        dict מפלט meta_optimize, חייב לכלול meta['all_scores'] (DataFrame).
    full:
        אם True → מחזיר summary ברמת זוג (ולמעלה Global row).
        אם False → מחזיר top-50 שורות מפורטות לפי meta_score.

    Returns
    -------
    pd.DataFrame
        כאשר full=True:
            טבלת סיכום ברמת pair + שורה GLOBAL, עם העמודות למשל:
                - level          ("GLOBAL" / "PAIR")
                - pair           (רק ברמת pair)
                - n_candidates
                - meta_score_mean / max / min / std
                - sharpe_mean    / max / min
                - profit_mean    / max / min
                - drawdown_mean  / min / max
                - win_rate_mean  / max / min
                - meta_score_q05 / meta_score_q50 / meta_score_q95

        כאשר full=False:
            מחזיר את df המפורט מ-meta['all_scores'], ממוין לפי meta_score
            (או Score), עם דירוג global_rank ו-rank_in_pair (אם pair קיים).
    """
    if not meta or "all_scores" not in meta or meta["all_scores"] is None:
        return pd.DataFrame()

    df = meta["all_scores"].copy()
    if df.empty:
        return df

    # --- איתור pair / metrics ---
    pair_col = _infer_pair_column(df)
    metric_cols = _infer_metric_columns(df)

    # meta_score column
    meta_col = metric_cols.get("meta_score")
    if meta_col is None:
        # fallback: אם יש 'meta_score' בעמודות נשתמש בו, אחרת 'Score' אם קיים
        if "meta_score" in df.columns:
            meta_col = "meta_score"
        elif "Score" in df.columns:
            meta_col = "Score"
        else:
            # אם אין שום דבר, נזרום עם NaN
            df["meta_score"] = np.nan
            meta_col = "meta_score"
    # Sharpe/Profit/Drawdown/WinRate עוזרים לדוחות
    sharpe_col = metric_cols.get("sharpe")
    profit_col = metric_cols.get("profit") or metric_cols.get("return")
    dd_col = metric_cols.get("drawdown")
    wr_col = metric_cols.get("win_rate")

    # --- דירוגים גלובליים + per-pair ---
    df[meta_col] = pd.to_numeric(df[meta_col], errors="coerce")
    df["global_rank"] = df[meta_col].rank(method="min", ascending=False)

    if pair_col:
        # rank בתוך pair
        df["pair"] = df[pair_col].astype(str)
        df["rank_in_pair"] = df.groupby("pair")[meta_col].rank(
            method="min", ascending=False
        )
    else:
        df["pair"] = "(none)"
        df["rank_in_pair"] = np.nan

    # אפשר להחזיר את המפורט עם הדירוגים אם full=False
    if not full:
        return df.sort_values(meta_col, ascending=False).reset_index(drop=True)

    # --- סיכום per-pair + שורה גלובלית ---
    # עזר לחילוץ "מספרי" מ-Series
    def _safe_stats(s: pd.Series) -> Dict[str, float]:
        s = pd.to_numeric(s, errors="coerce").dropna()
        if s.empty:
            return {
                "mean": np.nan,
                "std": np.nan,
                "min": np.nan,
                "max": np.nan,
                "q05": np.nan,
                "q50": np.nan,
                "q95": np.nan,
            }
        return {
            "mean": float(s.mean()),
            "std": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
            "min": float(s.min()),
            "max": float(s.max()),
            "q05": float(s.quantile(0.05)),
            "q50": float(s.quantile(0.50)),
            "q95": float(s.quantile(0.95)),
        }

    rows: List[Dict[str, Any]] = []

    def _row_for_group(name: str, g: pd.DataFrame, level: str) -> Dict[str, Any]:
        r: Dict[str, Any] = {"level": level, "pair": name, "n_candidates": int(len(g))}
        # meta_score stats
        ms_stats = _safe_stats(g[meta_col])
        r.update(
            {
                "meta_score_mean": ms_stats["mean"],
                "meta_score_std": ms_stats["std"],
                "meta_score_min": ms_stats["min"],
                "meta_score_max": ms_stats["max"],
                "meta_score_q05": ms_stats["q05"],
                "meta_score_q50": ms_stats["q50"],
                "meta_score_q95": ms_stats["q95"],
            }
        )
        # Sharpe
        if sharpe_col and sharpe_col in g.columns:
            sh_stats = _safe_stats(g[sharpe_col])
            r.update(
                {
                    "sharpe_mean": sh_stats["mean"],
                    "sharpe_min": sh_stats["min"],
                    "sharpe_max": sh_stats["max"],
                }
            )
        # Profit/Return
        if profit_col and profit_col in g.columns:
            pr_stats = _safe_stats(g[profit_col])
            r.update(
                {
                    "profit_mean": pr_stats["mean"],
                    "profit_min": pr_stats["min"],
                    "profit_max": pr_stats["max"],
                }
            )
        # Drawdown
        if dd_col and dd_col in g.columns:
            dd_stats = _safe_stats(g[dd_col])
            r.update(
                {
                    "drawdown_mean": dd_stats["mean"],
                    "drawdown_min": dd_stats["min"],
                    "drawdown_max": dd_stats["max"],
                }
            )
        # WinRate
        if wr_col and wr_col in g.columns:
            wr_stats = _safe_stats(g[wr_col])
            r.update(
                {
                    "win_rate_mean": wr_stats["mean"],
                    "win_rate_min": wr_stats["min"],
                    "win_rate_max": wr_stats["max"],
                }
            )

        return r

    # per-pair
    for pair_name, g in df.groupby("pair"):
        rows.append(_row_for_group(str(pair_name), g, level="PAIR"))

    # global row
    rows.insert(0, _row_for_group("GLOBAL", df, level="GLOBAL"))

    summary_df = pd.DataFrame(rows)
    return summary_df


# =====================================================================
# Additional helpers: by-pair / by-parameter / correlation matrix
# =====================================================================

def summarize_by_pair(df: pd.DataFrame) -> pd.DataFrame:
    """
    סיכום מהיר לפי pair על DataFrame מפורט (לא בהכרח meta).

    שימושי גם כאשר אין meta_optimize; לדוגמה:
        df_opt  → תוצאות אופטימיזציה לזוג אחד/מספר זוגות.

    מחזיר DataFrame עם שורות ברמת pair:
        Pair, n_rows, Sharpe_mean/max, Profit_mean/max, Drawdown_mean/min, Score_best
    """
    if df is None or df.empty:
        return pd.DataFrame()

    pair_col = _infer_pair_column(df) or "Pair"
    if pair_col not in df.columns:
        df = df.copy()
        df[pair_col] = "(none)"

    metric_cols = _infer_metric_columns(df)
    sharpe_col = metric_cols.get("sharpe")
    profit_col = metric_cols.get("profit") or metric_cols.get("return")
    dd_col = metric_cols.get("drawdown")
    meta_col = metric_cols.get("meta_score") or ("Score" if "Score" in df.columns else None)

    rows: List[Dict[str, Any]] = []
    for pair_name, g in df.groupby(pair_col):
        r: Dict[str, Any] = {"Pair": str(pair_name), "n_rows": int(len(g))}
        if sharpe_col and sharpe_col in g.columns:
            s = pd.to_numeric(g[sharpe_col], errors="coerce")
            r["Sharpe_mean"] = float(s.mean())
            r["Sharpe_max"] = float(s.max())
        if profit_col and profit_col in g.columns:
            p = pd.to_numeric(g[profit_col], errors="coerce")
            r["Profit_mean"] = float(p.mean())
            r["Profit_max"] = float(p.max())
        if dd_col and dd_col in g.columns:
            d = pd.to_numeric(g[dd_col], errors="coerce")
            r["Drawdown_mean"] = float(d.mean())
            r["Drawdown_min"] = float(d.min())
        if meta_col and meta_col in g.columns:
            m = pd.to_numeric(g[meta_col], errors="coerce")
            r["Score_best"] = float(m.max())
        rows.append(r)

    return pd.DataFrame(rows).sort_values("Score_best", ascending=False, na_position="last") if rows else pd.DataFrame()


def summarize_by_param(
    df: pd.DataFrame,
    params: Optional[List[str]] = None,
    *,
    metric_col: str = "Score",
    bins: int = 10,
) -> pd.DataFrame:
    """
    סיכום sensitivity לפי פרמטרים:

    - מחלק כל פרמטר לבינים (quantile bins).
    - לכל bin מחשב mean/median/max של metric_col.
    - מחזיר DataFrame שניתן להצגה כטבלת heatmap/line.

    Args:
        df: DataFrame עם תוצאות אופטימיזציה.
        params: רשימת שמות פרמטרים (אם None → מזהה לבד).
        metric_col: שם המטריקה למעקב (Score/meta_score וכו').
        bins: מספר quantile bins (למשל 10).

    Return:
        DataFrame עם עמודות:
            param, bin_id, bin_low, bin_high, n, metric_mean, metric_median, metric_max
    """
    if df is None or df.empty:
        return pd.DataFrame()

    if metric_col not in df.columns:
        metric_col = "Score" if "Score" in df.columns else metric_col
        if metric_col not in df.columns:
            return pd.DataFrame()

    metric_series = pd.to_numeric(df[metric_col], errors="coerce")
    df = df.copy()
    df[metric_col] = metric_series

    metric_cols_map = _infer_metric_columns(df)
    pair_col = _infer_pair_column(df)
    param_cols = _infer_param_columns(df, metric_cols_map, pair_col=pair_col)

    if params is None:
        params = param_cols
    else:
        params = [p for p in params if p in df.columns]

    if not params:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for p in params:
        s = pd.to_numeric(df[p], errors="coerce").dropna()
        if s.empty:
            continue
        try:
            q = np.linspace(0.0, 1.0, num=bins + 1)
            edges = s.quantile(q).to_numpy()
            edges[0] = float(s.min())
            edges[-1] = float(s.max())
            edges = np.unique(edges)
            if len(edges) < 2:
                continue
            labels = [f"[{edges[i]:.4g}, {edges[i+1]:.4g})" for i in range(len(edges) - 1)]
            cut = pd.cut(df[p], bins=edges, labels=range(len(edges) - 1), include_lowest=True)
        except Exception:
            continue

        for bin_id in range(len(edges) - 1):
            mask = cut == bin_id
            g = df.loc[mask]
            if g.empty:
                continue
            m = pd.to_numeric(g[metric_col], errors="coerce").dropna()
            if m.empty:
                continue
            rows.append(
                {
                    "param": p,
                    "bin_id": int(bin_id),
                    "bin_label": labels[bin_id],
                    "bin_low": float(edges[bin_id]),
                    "bin_high": float(edges[bin_id + 1]),
                    "n": int(len(g)),
                    "metric_col": metric_col,
                    "metric_mean": float(m.mean()),
                    "metric_median": float(m.median()),
                    "metric_max": float(m.max()),
                }
            )

    return pd.DataFrame(rows)


def compute_metric_matrix(
    df: pd.DataFrame,
    include_meta: bool = True,
) -> pd.DataFrame:
    """
    מחשב מטריצת קורלציה בין המטריקות השונות (Sharpe, Profit, Drawdown וכו').

    Args:
        df: DataFrame עם תוצאות אופטימיזציה / meta_scores.
        include_meta: אם True → כולל גם meta_score/Score במטריצה.

    Returns:
        DataFrame קורלציה (Pearson) בין המטריקות שנמצאו בפועל.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    metric_cols = _infer_metric_columns(df)
    cols: List[str] = []

    # סדר לוגי: meta, Sharpe, Sortino, Calmar, Profit/Return, Drawdown, WinRate
    if include_meta and "meta_score" in metric_cols:
        cols.append(metric_cols["meta_score"])
    for key in ("sharpe", "sortino", "calmar", "profit", "return", "drawdown", "win_rate"):
        col = metric_cols.get(key)
        if col and col not in cols:
            cols.append(col)

    if not cols:
        # אם אין שום עמודת מטריקה מוכרת, ננסה פשוט את כל העמודות המספריות
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            return pd.DataFrame()
        cols = num_cols

    mdf = df[cols].apply(pd.to_numeric, errors="coerce")
    if mdf.empty:
        return pd.DataFrame()

    return mdf.corr()
