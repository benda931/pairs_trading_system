# -*- coding: utf-8 -*-
"""
core/tab_comparison_matrices.py — Tab Comparison Matrices (HF-grade, v3)
========================================================================

מודול ליבה להשוואה חכמה בין טאבים במערכת:

    • טאבים סטטיסטיים (Matrix, Pair Analysis, Backtest KPIs)
    • טאבים מאקרו־כלכליים (Macro Engine / Macro Data)
    • טאבים פנדומנטליים (Index Fundamentals / Earnings / Valuation)
    • טאבי Risk / Portfolio / Optimization ועוד

המודול מתמקד **בלוגיקה בלבד**:
--------------------------------
- אין תלות ב-Streamlit או Qt.
- הצד של ה-UI רק בונה TabProfile-ים ומקבל DataFrames מוכנים לתצוגה.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Iterable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    # מומלץ: יש לך כבר common.typing_compat במערכת
    from common.typing_compat import Sequence, MutableMapping  # type: ignore
except Exception:  # fallback בטוח
    from collections.abc import Sequence, MutableMapping  # type: ignore

try:
    # שמירה JSON-safe אם זמין
    from common.json_safe import make_json_safe  # type: ignore
except Exception:  # pragma: no cover
    def make_json_safe(obj: Any) -> Any:  # type: ignore
        return obj

try:
    # אופציונלי: SciPy ל-clustering היררכי
    from scipy.cluster.hierarchy import linkage, fcluster  # type: ignore
    from scipy.spatial.distance import squareform  # type: ignore
except Exception:  # pragma: no cover
    linkage = None  # type: ignore
    fcluster = None  # type: ignore
    squareform = None  # type: ignore


__all__ = [
    "MetricMeta",
    "TabProfile",
    "TabComparisonConfig",
    "build_tab_similarity_matrix",
    "build_tab_distance_matrix",
    "build_metric_vs_tab_matrix",
    "summarize_by_tab_type",
    "build_tab_type_similarity_matrix",
    "rank_tabs_per_metric",
    "compute_metric_correlation",
    "build_comparison_bundle",
    "build_time_slice_similarity_overview",
    "summarize_distributions_to_metrics",
    "build_profiles_from_backtest_df",
    "build_profiles_from_context",
    "build_group_based_metric_weights",
    "build_tab_similarity_with_multiindex",
    "explain_similarity_contributions",
    "compute_alignment_scores",
    "detect_tab_anomalies",
    "save_comparison_bundle_to_sql",
    "build_metaopt_candidate_groups",
    "hierarchical_cluster_tabs",
    "stack_similarity_history",
    "inject_risk_budget_metric",
    "build_composite_profile",
]


# =============================================================================
# Dataclasses
# =============================================================================


@dataclass
class MetricMeta:
    """
    תיאור של מטריקה אחת (meta-data), כדי לאפשר:
        - כיוון (Higher is better / Lower is better)
        - משקל (חשיבות יחסית)
        - קיבוץ (group) — למשל 'risk', 'macro', 'valuation'
        - תיאור חופשי

    direction:
        'higher_better' | 'lower_better' | 'neutral'
        אם lower_better → נהפוך סימן לפני נירמול (כדי שה"גבוה יותר" תמיד טוב יותר).
    """

    name: str
    direction: str = "higher_better"
    weight: float = 1.0
    group: Optional[str] = None
    description: Optional[str] = None


@dataclass
class TabProfile:
    """
    TabProfile — ייצוג מספרי של טאב אחד במערכת.

    דוגמאות ל-tab_type:
        - 'stats'        → טאבים סטטיסטיים (Matrix, Backtest, Pair Analysis)
        - 'macro'        → טאבים מאקרו (Macro Engine, Regimes, Risk-on/off)
        - 'fundamental'  → Fundamentals / Index / Earnings
        - 'risk'         → Risk Engine / Exposure / Kill Switch
        - 'portfolio'    → Fund / Portfolio View
        - 'optimization' → Optimization / Meta-Opt

    weight:
        משקל כללי של הטאב בחישובי דמיון / מרחק (ברירת מחדל 1.0).

    tags:
        רשימת טאגים חופשית, למשל: ['live', 'overview', 'advanced'].

    metadata:
        כל מידע נוסף שנרצה לשמור על הטאב (לא משמש לחישובים, אבל נשמר ל-JSON).
    """

    tab_id: str
    tab_type: str
    label: str
    metrics: Dict[str, float]
    weight: float = 1.0
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TabComparisonConfig:
    """
    קונפיג מרכזי למודול השוואת טאבים.

    normalization:
        'zscore'  → (x - mean) / std
        'minmax'  → (x - min) / (max - min)
        'robust'  → (x - median) / IQR

    similarity_method:
        'cosine'    → דמיון קוסינוסי (mapped ל-[0,1])
        'corr'      → קורלציה (mapped ל-[0,1])
        'euclidean' → מרחק אוקלידי → דמיון כ-1 / (1 + distance)

    distance_metric:
        כרגע תומך ב-'euclidean' בלבד, אבל נשאיר מקום להרחבה.

    metric_weights:
        dict[metric_name, weight]. אם None → משקל 1 לכל המטריקות.

    metric_meta:
        dict[metric_name, MetricMeta]. אם קיים direction='lower_better',
        נהפוך סימן לפני נירמול כדי ש"הגבוה יותר" תמיד יהיה טוב יותר.

    group_weights:
        dict[group_name, weight] — משקל ברמת group (risk/macro/valuation...).

    fill_value:
        ערך ברירת מחדל למטריקות חסרות.
    """

    normalization: str = "zscore"
    similarity_method: str = "cosine"
    distance_metric: str = "euclidean"
    metric_weights: Optional[Dict[str, float]] = None
    metric_meta: Optional[Dict[str, MetricMeta]] = None
    group_weights: Optional[Dict[str, float]] = None
    fill_value: float = np.nan
    normalize_for_similarity: bool = True
    normalize_for_distance: bool = True


# =============================================================================
# Metric & profile helpers
# =============================================================================


def _collect_all_metric_keys(profiles: Sequence[TabProfile]) -> List[str]:
    """
    מחזיר את כל שמות המטריקות שמופיעות לפחות בפרופיל אחד.
    """
    keys: set[str] = set()
    for p in profiles:
        keys.update(p.metrics.keys())
    return sorted(keys)


def _build_metric_meta_map(
    metric_meta: Optional[Mapping[str, MetricMeta]],
    metric_keys: Sequence[str],
) -> Dict[str, MetricMeta]:
    """
    בונה מפה מלאה של MetricMeta לכל metric_key.
    אם חסר meta → יוצרים אובייקט ברירת מחדל.
    """
    meta_map: Dict[str, MetricMeta] = {}
    metric_meta = metric_meta or {}
    for k in metric_keys:
        if k in metric_meta:
            meta_map[k] = metric_meta[k]
        else:
            meta_map[k] = MetricMeta(name=k)
    return meta_map


def _apply_metric_direction(
    df: pd.DataFrame,
    meta_map: Mapping[str, MetricMeta],
) -> pd.DataFrame:
    """
    מתקן את כיוון המטריקות:
        - אם direction == 'lower_better' → הופכים סימן (x → -x).
        - 'higher_better' או 'neutral' → משאירים.
    """
    out = df.copy()
    for col in out.columns:
        meta = meta_map.get(col)
        if meta is not None and meta.direction == "lower_better":
            out[col] = -out[col]
    return out


def build_group_based_metric_weights(
    meta_map: Mapping[str, MetricMeta],
    group_weights: Mapping[str, float],
) -> Dict[str, float]:
    """
    יוצר dict של משקלי מטריקות לפי משקל group:

        effective_weight(metric) = meta.weight * group_weight[group]

    אם group לא קיים ב-group_weights → group_weight=1.0.
    """
    weights: Dict[str, float] = {}
    for name, meta in meta_map.items():
        g = meta.group
        gw = 1.0
        if g is not None and g in group_weights:
            gw = float(group_weights[g])
        weights[name] = float(meta.weight) * gw
    return weights


def _apply_metric_weights(
    df: pd.DataFrame,
    metric_weights: Optional[Mapping[str, float]],
    meta_map: Mapping[str, MetricMeta],
    group_weights: Optional[Mapping[str, float]] = None,
) -> pd.DataFrame:
    """
    מכפיל כל עמודה במשקל מטריקה:
        - אם metric_weights לא None → משתמש בהם.
        - אחרת → משתמש ב-meta_map[col].weight.
        - אם group_weights לא None → מכפיל לפי משקל קבוצה (group).
    """
    out = df.copy()

    effective: Dict[str, float] = {}
    if group_weights:
        effective = build_group_based_metric_weights(meta_map, group_weights)
    else:
        for col, meta in meta_map.items():
            effective[col] = float(meta.weight)

    for col in out.columns:
        w: float
        if metric_weights and col in metric_weights:
            w = float(metric_weights[col])
        else:
            w = effective.get(col, 1.0)

        if not np.isclose(w, 1.0):
            out[col] = out[col] * w

    return out


def _apply_tab_weights(
    df: pd.DataFrame,
    profiles: Sequence[TabProfile],
) -> pd.DataFrame:
    """
    מכפיל כל שורה במשקל הטאב (TabProfile.weight).
    """
    out = df.copy()
    weights = np.array([float(p.weight) for p in profiles], dtype=float)
    out = out.mul(weights.reshape(-1, 1))
    return out


def _profiles_to_matrix(
    profiles: Sequence[TabProfile],
    metric_keys: Optional[Sequence[str]] = None,
    fill_value: float = np.nan,
) -> Tuple[pd.DataFrame, List[TabProfile]]:
    """
    ממפה רשימת TabProfile למטריצה (DataFrame):

        index  → tab_id
        cols   → metric_keys
        values → metric values (float)

    אם metric_keys=None → משתמש בכל המפתחות שמופיעים.
    """
    if metric_keys is None:
        metric_keys = _collect_all_metric_keys(profiles)

    data: List[List[float]] = []
    index: List[str] = []

    for p in profiles:
        row: List[float] = []
        for k in metric_keys:
            val = p.metrics.get(k, fill_value)
            try:
                row.append(float(val))
            except Exception:
                row.append(float(fill_value))
        data.append(row)
        index.append(p.tab_id)

    df = pd.DataFrame(data=data, index=index, columns=list(metric_keys))
    return df, list(profiles)


# =============================================================================
# Normalization helpers
# =============================================================================


def _normalize_df(df: pd.DataFrame, method: str) -> pd.DataFrame:
    """
    מנרמל DataFrame לפי שיטת normalization:

        'zscore'  → (x - mean) / std
        'minmax'  → (x - min) / (max - min)
        'robust'  → (x - median) / IQR
    """
    if df.empty:
        return df

    out = df.copy()

    for col in out.columns:
        series = out[col]
        if series.isna().all():
            continue

        if method == "zscore":
            mean = series.mean(skipna=True)
            std = series.std(skipna=True)
            if std and not np.isclose(std, 0.0):
                out[col] = (series - mean) / std
            else:
                out[col] = series - mean

        elif method == "minmax":
            min_val = series.min(skipna=True)
            max_val = series.max(skipna=True)
            denom = max_val - min_val
            if denom and not np.isclose(denom, 0.0):
                out[col] = (series - min_val) / denom
            else:
                out[col] = 0.0

        elif method == "robust":
            median = series.median(skipna=True)
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            if iqr and not np.isclose(iqr, 0.0):
                out[col] = (series - median) / iqr
            else:
                out[col] = series - median

        else:
            raise ValueError(f"Unknown normalization method: {method!r}")

    return out


# =============================================================================
# Internal feature prep helper (משותף לדמיון / הסברים)
# =============================================================================


def _prepare_feature_matrix(
    profiles: Sequence[TabProfile],
    metric_keys: Optional[Sequence[str]],
    cfg: TabComparisonConfig,
    for_distance: bool = False,
) -> Tuple[pd.DataFrame, List[TabProfile], Dict[str, MetricMeta]]:
    """
    מכין את מטריצת הפיצ'רים אחרי:
        - מילוי missing
        - תיקון כיוון (lower_better)
        - נירמול
        - חלפת NaN ל-0
        - משקלי מטריקות
        - משקלי טאבים

    מחזיר:
        df_features, ordered_profiles, meta_map
    """
    df_raw, ordered_profiles = _profiles_to_matrix(
        profiles,
        metric_keys=metric_keys,
        fill_value=cfg.fill_value,
    )
    metric_keys_final = list(df_raw.columns)
    meta_map = _build_metric_meta_map(cfg.metric_meta, metric_keys_final)

    # כיוון מטריקות
    df = _apply_metric_direction(df_raw, meta_map)

    # נירמול
    do_norm = cfg.normalize_for_distance if for_distance else cfg.normalize_for_similarity
    if do_norm:
        df = _normalize_df(df, cfg.normalization)

    # NaN→0
    df = df.fillna(0.0)

    # משקלי מטריקות + groups
    df = _apply_metric_weights(df, cfg.metric_weights, meta_map, cfg.group_weights)

    # משקלי טאבים
    df = _apply_tab_weights(df, ordered_profiles)

    return df, ordered_profiles, meta_map


# =============================================================================
# 1) Tab Similarity Matrix
# =============================================================================


def build_tab_similarity_matrix(
    profiles: Sequence[TabProfile],
    metric_keys: Optional[Sequence[str]] = None,
    cfg: Optional[TabComparisonConfig] = None,
) -> pd.DataFrame:
    """
    בונה מטריצת דמיון בין טאבים (Tab vs Tab).

    cfg:
        אם None → TabComparisonConfig() ברירת מחדל.

    הערות:
        - מטפל בכיוון המטריקות (lower_better).
        - תומך במשקלי מטריקות ו-Tab-level.
        - מחזיר ערכים בטווח [0,1] — 1 = הכי דומה.
    """
    if len(profiles) == 0:
        return pd.DataFrame()

    cfg = cfg or TabComparisonConfig()
    df, ordered_profiles, _ = _prepare_feature_matrix(
        profiles, metric_keys=metric_keys, cfg=cfg, for_distance=False
    )

    X = df.to_numpy(dtype=float)
    n = X.shape[0]

    if n == 1:
        idx = list(df.index)
        return pd.DataFrame([[1.0]], index=idx, columns=idx)

    sim = np.zeros((n, n), dtype=float)
    method = cfg.similarity_method

    if method == "cosine":
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        Xn = X / norms
        sim = Xn @ Xn.T
        sim = np.clip(sim, -1.0, 1.0)
        sim = (sim + 1.0) / 2.0  # [-1,1] -> [0,1]

    elif method == "corr":
        sim = np.corrcoef(X)
        sim = np.nan_to_num(sim, nan=0.0)
        sim = (sim + 1.0) / 2.0

    elif method == "euclidean":
        for i in range(n):
            for j in range(n):
                d = np.linalg.norm(X[i] - X[j])
                sim[i, j] = 1.0 / (1.0 + d)

    else:
        raise ValueError(f"Unknown similarity_method: {method!r}")

    index = [p.tab_id for p in ordered_profiles]
    return pd.DataFrame(sim, index=index, columns=index)


# =============================================================================
# 2) Tab Distance Matrix
# =============================================================================


def build_tab_distance_matrix(
    profiles: Sequence[TabProfile],
    metric_keys: Optional[Sequence[str]] = None,
    cfg: Optional[TabComparisonConfig] = None,
) -> pd.DataFrame:
    """
    בונה מטריצת מרחקים בין טאבים (Tab vs Tab, מרחק אוקלידי).

    מרחק גבוה → הטאבים שונים מאוד בוקטור המטריקות שלהם.
    """
    if len(profiles) == 0:
        return pd.DataFrame()

    cfg = cfg or TabComparisonConfig()
    df, ordered_profiles, _ = _prepare_feature_matrix(
        profiles, metric_keys=metric_keys, cfg=cfg, for_distance=True
    )

    X = df.to_numpy(dtype=float)
    n = X.shape[0]
    dist = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(n):
            dist[i, j] = float(np.linalg.norm(X[i] - X[j]))

    index = [p.tab_id for p in ordered_profiles]
    return pd.DataFrame(dist, index=index, columns=index)


# =============================================================================
# 3) KPI Matrix (Metric vs Tab)
# =============================================================================


def build_metric_vs_tab_matrix(
    profiles: Sequence[TabProfile],
    metric_keys: Optional[Sequence[str]] = None,
    cfg: Optional[TabComparisonConfig] = None,
    include_tab_type: bool = True,
) -> pd.DataFrame:
    """
    בונה מטריצה שבה:
        index   → metric name
        columns → tab label (או TABTYPE: label אם include_tab_type=True)

    שימושי כדי:
        - לראות איך כל טאב נראה יחסית לאחרים עבור כל מטריקה.
        - לחבר בין סטטיסטיקה / מאקרו / פנדומנטלי במבט אחד.

    cfg.normalization:
        מיושם ברמת שורה (per-metric) כדי להדגיש הבדלים יחסיים.
    """
    if len(profiles) == 0:
        return pd.DataFrame()

    cfg = cfg or TabComparisonConfig()

    if metric_keys is None:
        metric_keys = _collect_all_metric_keys(profiles)

    meta_map = _build_metric_meta_map(cfg.metric_meta, metric_keys)

    data: Dict[str, List[float]] = {}
    for p in profiles:
        col_label = p.label
        if include_tab_type:
            col_label = f"{p.tab_type.upper()}: {p.label}"

        col_values: List[float] = []
        for k in metric_keys:
            v = p.metrics.get(k, np.nan)
            try:
                col_values.append(float(v))
            except Exception:
                col_values.append(np.nan)

        data[col_label] = col_values

    df = pd.DataFrame(data=data, index=list(metric_keys))

    # כיוון מטריקה
    df = _apply_metric_direction(df.T, meta_map).T

    # נירמול per-metric
    if cfg.normalize_for_similarity:
        normalized_rows: Dict[str, pd.Series] = {}
        for idx in df.index:
            row = df.loc[idx]
            if row.isna().all():
                normalized_rows[idx] = row
                continue
            if cfg.normalization == "zscore":
                mean = row.mean(skipna=True)
                std = row.std(skipna=True)
                normalized_rows[idx] = (row - mean) / std if std and not np.isclose(std, 0.0) else (row - mean)
            elif cfg.normalization == "minmax":
                min_val = row.min(skipna=True)
                max_val = row.max(skipna=True)
                denom = max_val - min_val
                if denom and not np.isclose(denom, 0.0):
                    normalized_rows[idx] = (row - min_val) / denom
                else:
                    normalized_rows[idx] = 0.0
            elif cfg.normalization == "robust":
                median = row.median(skipna=True)
                q1 = row.quantile(0.25)
                q3 = row.quantile(0.75)
                iqr = q3 - q1
                if iqr and not np.isclose(iqr, 0.0):
                    normalized_rows[idx] = (row - median) / iqr
                else:
                    normalized_rows[idx] = row - median
            else:
                raise ValueError(f"Unknown normalization method: {cfg.normalization!r}")
        df = pd.DataFrame(normalized_rows).T

    return df


# =============================================================================
# 4) Grouping by Tab-Type
# =============================================================================


def summarize_by_tab_type(
    profiles: Sequence[TabProfile],
    agg: str = "mean",
    metric_keys: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    מסכם את המטריקות לפי סוג טאב (tab_type).

    returns:
        index   → tab_type
        columns → metric_name
        values  → aggregated value (mean/median/...)
    """
    if len(profiles) == 0:
        return pd.DataFrame()

    if metric_keys is None:
        metric_keys = _collect_all_metric_keys(profiles)

    records: List[Dict[str, Any]] = []
    for p in profiles:
        row: Dict[str, Any] = {"tab_type": p.tab_type}
        for k in metric_keys:
            row[k] = p.metrics.get(k, np.nan)
        records.append(row)

    df = pd.DataFrame.from_records(records)

    if agg == "mean":
        grouped = df.groupby("tab_type", dropna=True).mean(numeric_only=True)
    elif agg == "median":
        grouped = df.groupby("tab_type", dropna=True).median(numeric_only=True)
    else:
        raise ValueError(f"Unsupported agg={agg!r}. Use 'mean' or 'median'.")

    return grouped


def build_tab_type_similarity_matrix(
    profiles: Sequence[TabProfile],
    cfg: Optional[TabComparisonConfig] = None,
) -> pd.DataFrame:
    """
    בונה מטריצת דמיון ברמת סוג טאב (tab_type vs tab_type).
    """
    if len(profiles) == 0:
        return pd.DataFrame()

    cfg = cfg or TabComparisonConfig()
    metric_keys = _collect_all_metric_keys(profiles)

    summary = summarize_by_tab_type(profiles, agg="mean", metric_keys=metric_keys)
    if summary.empty:
        return pd.DataFrame()

    type_profiles: List[TabProfile] = []
    for tab_type, row in summary.iterrows():
        metrics = {k: float(row.get(k, np.nan)) for k in metric_keys}
        type_profiles.append(
            TabProfile(
                tab_id=str(tab_type),
                tab_type=str(tab_type),
                label=f"Type: {tab_type}",
                metrics=metrics,
            )
        )

    return build_tab_similarity_matrix(type_profiles, metric_keys=metric_keys, cfg=cfg)


# =============================================================================
# 5) Ranks & Metric Correlations
# =============================================================================


def rank_tabs_per_metric(
    metric_vs_tab_df: pd.DataFrame,
    meta_map: Optional[Mapping[str, MetricMeta]] = None,
) -> pd.DataFrame:
    """
    בונה מטריצת ראנקים: לכל מטריקה (שורה) נותן דירוג לטאבים (1=הכי טוב).

    לוגיקה:
        - אם metric_meta.direction == 'lower_better' → ascending=True.
        - אחרת → ascending=False (ככל שהערך גבוה יותר, הרנק קטן יותר).
    """
    if metric_vs_tab_df.empty:
        return pd.DataFrame()

    out = pd.DataFrame(index=metric_vs_tab_df.index, columns=metric_vs_tab_df.columns, dtype=float)
    metric_keys = list(metric_vs_tab_df.index)
    meta_map_full = _build_metric_meta_map(meta_map, metric_keys)

    for metric in metric_vs_tab_df.index:
        row = metric_vs_tab_df.loc[metric]
        meta = meta_map_full.get(metric)
        ascending = meta.direction == "lower_better" if meta is not None else False
        ranks = row.rank(method="min", ascending=ascending, na_option="bottom")
        out.loc[metric] = ranks

    return out


def compute_metric_correlation(
    metric_vs_tab_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    מחשב קורלציה בין המטריקות עצמם (Metric vs Metric).
    """
    if metric_vs_tab_df.empty:
        return pd.DataFrame()

    df = metric_vs_tab_df.T
    corr = df.corr()
    return corr


# =============================================================================
# 6) Time-slice comparison (Idea #1)
# =============================================================================


def build_time_slice_similarity_overview(
    time_slices: Mapping[str, Sequence[TabProfile]],
    cfg: Optional[TabComparisonConfig] = None,
    metric_keys: Optional[Sequence[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    בונה overview של מטריצות דמיון לפי חלונות זמן.

    parameters
    ----------
    time_slices:
        dict[time_window, list[TabProfile]]
        לדוגמה: {
            "YTD": profiles_ytd,
            "1Y": profiles_1y,
            "3Y": profiles_3y,
        }

    returns
    -------
    dict[str, DataFrame]:
        מיפוי בין שם חלון הזמן לבין מטריצת similarity (Tab vs Tab).

    הערה:
        ה-UI יכול להשתמש בזה כדי לצייר כמה heatmaps לפי חלון זמן.
    """
    cfg = cfg or TabComparisonConfig()
    out: Dict[str, pd.DataFrame] = {}
    for window, profs in time_slices.items():
        out[window] = build_tab_similarity_matrix(profs, metric_keys=metric_keys, cfg=cfg)
    return out


# =============================================================================
# 7) Distributions → metrics (Idea #2)
# =============================================================================


def summarize_distributions_to_metrics(
    distributions: Mapping[str, Iterable[float]],
    prefix: str = "",
    include_moments: bool = True,
    include_quantiles: bool = True,
) -> Dict[str, float]:
    """
    ממיר התפלגויות (samples) לפיצ'רים סקאלריים לשימוש בתוך metrics של TabProfile.

    parameters
    ----------
    distributions:
        dict[metric_name, Iterable[float]]
    prefix:
        מחרוזת להוספה בתחילת שם הפיצ'ר (למשל "ret_" או "pnl_").
    include_moments:
        אם True → מוסיף mean, std, skew, kurtosis.
    include_quantiles:
        אם True → מוסיף q05, q25, q50, q75, q95.

    returns
    -------
    dict[str, float]:
        feature_name → value
    """
    def _safe_array(xs: Iterable[float]) -> np.ndarray:
        arr = np.asarray(list(xs), dtype=float)
        return arr[~np.isnan(arr)]

    features: Dict[str, float] = {}

    for name, samples in distributions.items():
        arr = _safe_array(samples)
        if arr.size == 0:
            continue

        base = f"{prefix}{name}"

        if include_moments:
            mean = float(arr.mean())
            std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
            if std and not np.isclose(std, 0.0):
                skew = float(((arr - mean) ** 3).mean() / (std ** 3))
                kurt = float(((arr - mean) ** 4).mean() / (std ** 4)) - 3.0
            else:
                skew = 0.0
                kurt = 0.0

            features[f"{base}_mean"] = mean
            features[f"{base}_std"] = std
            features[f"{base}_skew"] = skew
            features[f"{base}_kurt"] = kurt

        if include_quantiles:
            for q in (0.05, 0.25, 0.5, 0.75, 0.95):
                features[f"{base}_q{int(q*100):02d}"] = float(np.quantile(arr, q))

        # simple downside deviation (למשל לציון tail-risk)
        downside = arr[arr < 0]
        if downside.size > 0:
            features[f"{base}_downside_std"] = float(downside.std(ddof=1))
        else:
            features[f"{base}_downside_std"] = 0.0

    return features


# =============================================================================
# 8) Builders from existing system objects (Ideas #3, #4)
# =============================================================================


def build_profiles_from_backtest_df(
    df: pd.DataFrame,
    *,
    tab_type: str = "stats",
    id_col: str = "profile_id",
    label_col: Optional[str] = None,
    metric_cols: Optional[Sequence[str]] = None,
) -> List[TabProfile]:
    """
    בונה TabProfile-ים מ-DataFrame של תוצאות Backtest / Risk.

    df:
        צפוי להכיל עמודת מזהה (id_col) ועוד עמודות מטריקות.
        לדוגמה: sharpe, sortino, max_dd, macro_sensitivity, valuation_score וכו'.

    id_col:
        שם העמודה שמשמש כ-tab_id.

    label_col:
        אם קיים → ישמש כ-label. אחרת label=id.

    metric_cols:
        רשימת עמודות שנכנסות ל-metrics. אם None → כל העמודות המספריות
        למעט id_col ו-label_col.
    """
    if df.empty:
        return []

    if id_col not in df.columns:
        raise ValueError(f"id_col={id_col!r} not found in DataFrame")

    if metric_cols is None:
        exclude = {id_col}
        if label_col:
            exclude.add(label_col)
        metric_cols = [c for c in df.columns if c not in exclude and np.issubdtype(df[c].dtype, np.number)]

    profiles: List[TabProfile] = []
    for _, row in df.iterrows():
        tab_id = str(row[id_col])
        label = str(row[label_col]) if label_col and label_col in df.columns else tab_id
        metrics = {col: float(row[col]) for col in metric_cols}
        profiles.append(
            TabProfile(
                tab_id=tab_id,
                tab_type=tab_type,
                label=label,
                metrics=metrics,
            )
        )
    return profiles


def build_profiles_from_context(
    ctx: Mapping[str, Any],
    spec: Mapping[str, Mapping[str, Any]],
) -> List[TabProfile]:
    """
    בונה TabProfile-ים מתוך AppContext / DashboardContext כללי.

    ctx:
        אובייקט קונטקסט (dict-like). למשל st.session_state["ctx"] או app_ctx.ctx_dict.

    spec:
        תיאור איך לכרות את המטריקות מכל טאב. לדוגמה:

        spec = {
            "stats_backtest": {
                "tab_type": "stats",
                "label": "Backtest KPIs",
                "path": ["tabs", "backtest", "metrics"],  # ctx['tabs']['backtest']['metrics']
            },
            "macro_engine": {
                "tab_type": "macro",
                "label": "Macro Engine",
                "path": ["tabs", "macro", "scores"],
            },
        }

    כל path אמור להוביל ל-dict[str,float] שמייצג מטריקות לטאב.
    """
    def _get_from_path(root: Mapping[str, Any], path: Sequence[str]) -> Any:
        cur: Any = root
        for key in path:
            if not isinstance(cur, Mapping) or key not in cur:
                raise KeyError(f"path {path!r} not found in ctx (missing key={key!r})")
            cur = cur[key]
        return cur

    profiles: List[TabProfile] = []

    for tab_id, conf in spec.items():
        tab_type = conf.get("tab_type", "stats")
        label = conf.get("label", tab_id)
        path = conf.get("path")
        if not path:
            raise ValueError(f"spec for tab_id={tab_id!r} must contain 'path'")

        metrics_obj = _get_from_path(ctx, path)
        if not isinstance(metrics_obj, Mapping):
            raise TypeError(f"ctx path for tab_id={tab_id!r} must yield Mapping[str,float]")

        metrics: Dict[str, float] = {}
        for k, v in metrics_obj.items():
            try:
                metrics[str(k)] = float(v)
            except Exception:
                continue

        profiles.append(
            TabProfile(
                tab_id=tab_id,
                tab_type=str(tab_type),
                label=str(label),
                metrics=metrics,
            )
        )

    return profiles


# =============================================================================
# 9) MultiIndex similarity (Idea #6)
# =============================================================================


def build_tab_similarity_with_multiindex(
    profiles: Sequence[TabProfile],
    metric_keys: Optional[Sequence[str]] = None,
    cfg: Optional[TabComparisonConfig] = None,
) -> pd.DataFrame:
    """
    עטיפה ל-build_tab_similarity_matrix שמחזירה MultiIndex ל-index/columns:

        index  → (tab_type, tab_label)
        columns → (tab_type, tab_label)

    שימושי ל-heatmaps ו-pivot ב-UI.
    """
    sim = build_tab_similarity_matrix(profiles, metric_keys=metric_keys, cfg=cfg)
    if sim.empty:
        return sim

    id_to_profile = {p.tab_id: p for p in profiles}
    tuples_index = []
    for tab_id in sim.index:
        p = id_to_profile.get(tab_id)
        if p is None:
            tuples_index.append(("unknown", tab_id))
        else:
            tuples_index.append((p.tab_type, p.label))

    index = pd.MultiIndex.from_tuples(tuples_index, names=["tab_type", "tab_label"])
    sim.index = index
    sim.columns = index
    return sim


# =============================================================================
# 10) Explainability of similarity (Idea #7)
# =============================================================================


def explain_similarity_contributions(
    profiles: Sequence[TabProfile],
    cfg: Optional[TabComparisonConfig] = None,
    metric_keys: Optional[Sequence[str]] = None,
    tab_pair: Optional[Tuple[str, str]] = None,
) -> pd.DataFrame:
    """
    מסביר עבור זוג טאבים אילו מטריקות תרמו הכי הרבה לדמיון/מרחק.

    מתבסס על מטריצת הפיצ'רים אחרי משקלים ונירמול.

    parameters
    ----------
    tab_pair:
        (tab_id_1, tab_id_2). אם None → מחזיר תרומה לכל זוג כ-MultiIndex:

            index → (pair, metric_name)
            columns → ['abs_diff', 'weighted_diff', 'contribution']
    """
    if len(profiles) < 2:
        return pd.DataFrame()

    cfg = cfg or TabComparisonConfig()
    df, ordered_profiles, meta_map = _prepare_feature_matrix(
        profiles, metric_keys=metric_keys, cfg=cfg, for_distance=False
    )

    metric_names = list(df.columns)
    id_to_idx = {p.tab_id: i for i, p in enumerate(ordered_profiles)}
    X = df.to_numpy(dtype=float)

    def _pair_contrib(i: int, j: int) -> pd.DataFrame:
        v1 = X[i]
        v2 = X[j]
        diff = np.abs(v1 - v2)
        # משקל מטריקה אפקטיבי (כולל groups וכו')
        eff_weights = np.ones_like(diff)
        for k, name in enumerate(metric_names):
            meta = meta_map.get(name)
            eff_weights[k] = float(meta.weight) if meta is not None else 1.0

        weighted = diff * eff_weights
        total = weighted.sum()
        if total <= 0 or np.isnan(total):
            contrib = np.zeros_like(weighted)
        else:
            contrib = weighted / total

        return pd.DataFrame(
            {
                "metric": metric_names,
                "abs_diff": diff,
                "weighted_diff": weighted,
                "contribution": contrib,
            }
        ).set_index("metric").sort_values("contribution", ascending=False)

    if tab_pair is not None:
        i = id_to_idx.get(tab_pair[0])
        j = id_to_idx.get(tab_pair[1])
        if i is None or j is None:
            raise KeyError(f"tab_pair {tab_pair!r} contains unknown tab_id")
        return _pair_contrib(i, j)

    # otherwise: build MultiIndex for all unordered pairs
    records: List[pd.Series] = []
    idx_tuples: List[Tuple[str, str]] = []
    for i, pi in enumerate(ordered_profiles):
        for j, pj in enumerate(ordered_profiles):
            if j <= i:
                continue
            df_pair = _pair_contrib(i, j)
            for metric_name, row in df_pair.iterrows():
                idx_tuples.append((f"{pi.tab_id}|{pj.tab_id}", metric_name))
                records.append(row)

    if not records:
        return pd.DataFrame()

    contrib_df = pd.DataFrame(records)
    index = pd.MultiIndex.from_tuples(idx_tuples, names=["pair", "metric"])
    contrib_df.index = index
    return contrib_df


# =============================================================================
# 11) Alignment scoring vs benchmark (Idea #8)
# =============================================================================


def compute_alignment_scores(
    similarity_df: pd.DataFrame,
    benchmark_tab_id: str,
) -> pd.Series:
    """
    בונה Alignment Score ביחס לטאב Benchmark יחיד.

    similarity_df:
        מטריצת similarity (Tab vs Tab) כפי שמוחזרת מ-build_tab_similarity_matrix.

    benchmark_tab_id:
        הטאב המרכזי אליו מודדים קרבה (למשל 'stats_backtest' או 'portfolio_main').

    returns
    -------
    pd.Series:
        index → tab_id
        value → similarity score מול benchmark (benchmark עצמו=1.0)
    """
    if similarity_df.empty:
        return pd.Series(dtype=float)

    if benchmark_tab_id not in similarity_df.index:
        raise KeyError(f"benchmark_tab_id={benchmark_tab_id!r} not found in similarity_df")

    scores = similarity_df.loc[benchmark_tab_id].copy()
    scores.name = "alignment_score"
    return scores


# =============================================================================
# 12) Anomaly detection (Idea #9)
# =============================================================================


def detect_tab_anomalies(
    distance_df: pd.DataFrame,
    zscore_threshold: float = 2.5,
    min_peers: int = 2,
) -> pd.DataFrame:
    """
    מזהה טאבים "אאוטליירים" לפי ממוצע מרחקים גבוה מהשאר.

    distance_df:
        מטריצת מרחקים (Tab vs Tab) כפי שמוחזרת מ-build_tab_distance_matrix.

    returns
    -------
    DataFrame עם עמודות:
        - avg_distance
        - zscore
        - is_anomaly (bool)
    """
    if distance_df.empty:
        return pd.DataFrame()

    df = distance_df.copy()

    # מתעלמים מהאלכסון (0) ומחשבים ממוצע על השאר
    mask = ~np.eye(len(df), dtype=bool)
    vals = df.where(mask)
    avg_dist = vals.mean(axis=1, skipna=True)

    if avg_dist.count() < min_peers:
        # לא מספיק טאבים להשוואה
        return pd.DataFrame(
            {
                "avg_distance": avg_dist,
                "zscore": np.nan,
                "is_anomaly": False,
            }
        )

    mean = avg_dist.mean()
    std = avg_dist.std()
    if std and not np.isclose(std, 0.0):
        zscore = (avg_dist - mean) / std
    else:
        zscore = (avg_dist - mean)

    is_anomaly = zscore > zscore_threshold

    return pd.DataFrame(
        {
            "avg_distance": avg_dist,
            "zscore": zscore,
            "is_anomaly": is_anomaly,
        }
    )


# =============================================================================
# 13) SQL integration (Idea #10)
# =============================================================================


def save_comparison_bundle_to_sql(
    store: Any,
    bundle: Mapping[str, Any],
    *,
    run_id: Optional[str] = None,
    section: str = "tab_comparison",
    schema: Optional[str] = None,
    if_exists: str = "append",
) -> None:
    """
    שומר Comparison Bundle ל-SQL דרך SqlStore או אובייקט דומה.

    החוזה המינימלי:
        - אם ל-store יש מתודה בשם save_df(df, table_name, extra=None) → נשתמש בה.
        - אחרת, אם יש לו .engine התואם ל-SQLAlchemy → df.to_sql(...).

    לכל טבלה נוסיף עמודות מטאדאטה:
        - run_id (אם קיים)
        - section
        - kind (similarity/distance/metric_vs_tab/...)
    """
    tables: Dict[str, pd.DataFrame] = {}
    for kind in ("similarity", "distance", "metric_vs_tab", "tab_type_summary", "tab_type_similarity", "ranks", "metric_corr"):
        df = bundle.get(kind)
        if isinstance(df, pd.DataFrame) and not df.empty:
            tables[kind] = df.copy()

    meta = bundle.get("meta", {})
    meta_json = make_json_safe(meta)

    for kind, df in tables.items():
        df_to_save = df.copy()
        df_to_save["__section"] = section
        df_to_save["__kind"] = kind
        df_to_save["__run_id"] = run_id
        df_to_save["__meta"] = str(meta_json)

        table_name = f"{section}_{kind}"

        if hasattr(store, "save_df"):
            store.save_df(df_to_save, table_name)  # type: ignore[arg-type]
        elif hasattr(store, "engine"):
            # fallback גנרי ל-SQLAlchemy
            df_to_save.to_sql(table_name, store.engine, if_exists=if_exists, schema=schema)  # type: ignore[attr-defined]
        else:
            raise TypeError(
                "store must provide either .save_df(df, table_name, extra=None) "
                "or .engine compatible with pandas.to_sql"
            )


# =============================================================================
# 14) Meta-optimization helper (Idea #11)
# =============================================================================


def build_metaopt_candidate_groups(
    similarity_df: pd.DataFrame,
    threshold: float = 0.7,
) -> List[List[str]]:
    """
    בונה קבוצות טאבים "קשורים" לפי similarity >= threshold.

    הלוגיקה:
        - מסתכלים על גרף שבו צומת=tab_id, קשת קיימת אם similarity>=threshold.
        - מחזירים רשימת connected components.
    """
    if similarity_df.empty:
        return []

    tabs = list(similarity_df.index)
    n = len(tabs)
    visited = set()
    groups: List[List[str]] = []

    for i in range(n):
        if tabs[i] in visited:
            continue
        # BFS
        queue = [tabs[i]]
        component = []
        visited.add(tabs[i])

        while queue:
            t = queue.pop(0)
            component.append(t)
            sims = similarity_df.loc[t]
            neighbors = sims[sims >= threshold].index.tolist()
            for nb in neighbors:
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)

        groups.append(component)

    return groups


# =============================================================================
# 15) Hierarchical clustering (Idea #12)
# =============================================================================


def hierarchical_cluster_tabs(
    distance_df: pd.DataFrame,
    *,
    method: str = "ward",
    max_clusters: Optional[int] = None,
) -> Tuple[pd.Series, Optional[np.ndarray]]:
    """
    מבצע clustering היררכי לטאבים על בסיס distance_df.

    דורש SciPy. אם אין SciPy בסביבה → מעלה RuntimeError.

    returns
    -------
    labels: pd.Series
        index → tab_id
        value → cluster_id (1..K)

    linkage_matrix: np.ndarray | None
        מטריצת linkage של SciPy (לשימוש ב-dendrogram בצד ה-UI).
    """
    if distance_df.empty:
        return pd.Series(dtype=int), None

    if linkage is None or squareform is None or fcluster is None:
        raise RuntimeError("SciPy is required for hierarchical_cluster_tabs but is not installed.")

    # SciPy מצפה ל-condensed distance matrix (vector)
    condensed = squareform(distance_df.values, checks=False)
    Z = linkage(condensed, method=method)

    if max_clusters is None:
        # ברירת מחדל: מספר קלאסטרים ≈ sqrt(N)
        max_clusters = max(2, int(len(distance_df) ** 0.5))

    cluster_labels = fcluster(Z, max_clusters, criterion="maxclust")
    labels = pd.Series(cluster_labels, index=distance_df.index, name="cluster")
    return labels, Z


# =============================================================================
# 16) Snapshot history (Idea #13)
# =============================================================================


def stack_similarity_history(
    history: Mapping[Any, pd.DataFrame],
) -> pd.DataFrame:
    """
    יוצר היסטוריה של similarity לאורך זמן באמצעות MultiIndex.

    history:
        dict[timestamp, similarity_df]

    returns
    -------
    DataFrame:
        index   → MultiIndex (timestamp, tab_id)
        columns → tab_id
        values  → similarity score

    הערות:
        - מניח שכל similarity_df הוא ריבועי (אותם index & columns).
        - אם יש טיפה חוסר התאמה → ניישר לפי union ונמלא NaN.
    """
    if not history:
        return pd.DataFrame()

    # union של כל הטאבים
    all_tabs: set = set()
    for df in history.values():
        all_tabs.update(df.index.tolist())
        all_tabs.update(df.columns.tolist())
    all_tabs = sorted(all_tabs)

    frames: List[pd.DataFrame] = []
    idx_tuples: List[Tuple[Any, str]] = []

    for ts, df in history.items():
        aligned = df.reindex(index=all_tabs, columns=all_tabs)
        frames.append(aligned)
        for t in aligned.index:
            idx_tuples.append((ts, t))

    stacked = pd.concat(frames, axis=0)
    index = pd.MultiIndex.from_tuples(idx_tuples, names=["timestamp", "tab_id"])
    stacked.index = index
    return stacked


# =============================================================================
# 17) Risk budget metric helper (Idea #14)
# =============================================================================


def inject_risk_budget_metric(
    profiles: Sequence[TabProfile],
    risk_budget: Mapping[str, float],
    metric_name: str = "risk_budget",
) -> List[TabProfile]:
    """
    מוסיף מטריקת 'risk_budget' (או כל שם אחר) לכל TabProfile לפי מיפוי.

    risk_budget:
        dict[tab_id, risk_budget_value] — למשל חלק יחסי מתקציב הסיכון (0..1).
    """
    out: List[TabProfile] = []
    for p in profiles:
        metrics = dict(p.metrics)
        if p.tab_id in risk_budget:
            metrics[metric_name] = float(risk_budget[p.tab_id])
        else:
            metrics.setdefault(metric_name, np.nan)
        out.append(
            TabProfile(
                tab_id=p.tab_id,
                tab_type=p.tab_type,
                label=p.label,
                metrics=metrics,
                weight=p.weight,
                tags=list(p.tags),
                metadata=dict(p.metadata),
            )
        )
    return out


# =============================================================================
# 18) Composite profiles (Idea #15)
# =============================================================================


def build_composite_profile(
    composite_id: str,
    label: str,
    profiles: Sequence[TabProfile],
    *,
    tab_type: str = "composite",
    weights: Optional[Mapping[str, float]] = None,
) -> TabProfile:
    """
    בונה TabProfile "מורכב" שממוצע כמה טאבים.

    לדוגמה:
        - Composite "Macro" מתוך כל ה-Macro Tabs.
        - Composite "Stats+Macro" מכמה טאבים.

    weights:
        dict[tab_id, weight]. אם None → ממוצע שווה.

    הלוגיקה:
        - מאחדים את רשימת המטריקות הכוללת מכולם.
        - לכל מטריקה מחשבים ממוצע משוקלל על פני הטאבים שבהם היא קיימת.
    """
    if not profiles:
        raise ValueError("profiles must be non-empty for build_composite_profile")

    metric_keys = _collect_all_metric_keys(profiles)

    if weights is None:
        weights_arr = {p.tab_id: 1.0 for p in profiles}
    else:
        weights_arr = {p.tab_id: float(weights.get(p.tab_id, 0.0)) for p in profiles}

    # נוודא שאף טאב לא מקבל משקל שלילי/NaN
    for k, v in list(weights_arr.items()):
        if not np.isfinite(v) or v < 0:
            weights_arr[k] = 0.0

    if all(v == 0 for v in weights_arr.values()):
        raise ValueError("All composite weights are zero or invalid")

    composite_metrics: Dict[str, float] = {}
    for m in metric_keys:
        num = 0.0
        den = 0.0
        for p in profiles:
            if m in p.metrics:
                w = weights_arr.get(p.tab_id, 0.0)
                if w <= 0:
                    continue
                num += w * float(p.metrics[m])
                den += w
        if den > 0:
            composite_metrics[m] = num / den

    composite_tags: List[str] = sorted({tag for p in profiles for tag in p.tags})
    composite_meta: Dict[str, Any] = {
        "composite_of": [p.tab_id for p in profiles],
        "weights": weights_arr,
    }

    return TabProfile(
        tab_id=composite_id,
        tab_type=tab_type,
        label=label,
        metrics=composite_metrics,
        weight=1.0,
        tags=composite_tags,
        metadata=composite_meta,
    )


# =============================================================================
# 19) Comparison bundle (central API)
# =============================================================================


def build_comparison_bundle(
    profiles: Sequence[TabProfile],
    cfg: Optional[TabComparisonConfig] = None,
    metric_keys: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    בונה "חבילה" אחת עם כל הנתונים שמעניינים את ה-UI:

        {
            "similarity":          Tab vs Tab similarity matrix (DataFrame)
            "distance":            Tab vs Tab distance matrix (DataFrame)
            "metric_vs_tab":       Metric vs Tab normalized matrix (DataFrame)
            "tab_type_summary":    Summary by tab_type (DataFrame)
            "tab_type_similarity": TabType vs TabType similarity (DataFrame)
            "ranks":               Rank matrix (metric vs tab, DataFrame)
            "metric_corr":         Metric vs Metric correlation (DataFrame)
            "meta":                JSON-safe metadata על הפרופילים והקונפיג
        }
    """
    cfg = cfg or TabComparisonConfig()

    sim = build_tab_similarity_matrix(profiles, metric_keys=metric_keys, cfg=cfg)
    dist = build_tab_distance_matrix(profiles, metric_keys=metric_keys, cfg=cfg)
    kpi = build_metric_vs_tab_matrix(profiles, metric_keys=metric_keys, cfg=cfg)
    summary = summarize_by_tab_type(profiles, metric_keys=metric_keys)
    type_sim = build_tab_type_similarity_matrix(profiles, cfg=cfg)

    metric_keys_final = list(kpi.index) if not kpi.empty else list(metric_keys or [])
    meta_map = _build_metric_meta_map(cfg.metric_meta, metric_keys_final)
    ranks = rank_tabs_per_metric(kpi, meta_map=meta_map)
    metric_corr = compute_metric_correlation(kpi)

    meta_payload = {
        "profiles": [
            {
                "tab_id": p.tab_id,
                "tab_type": p.tab_type,
                "label": p.label,
                "weight": p.weight,
                "tags": list(p.tags),
                "metadata": p.metadata,
                "metrics_keys": sorted(p.metrics.keys()),
            }
            for p in profiles
        ],
        "config": {
            "normalization": cfg.normalization,
            "similarity_method": cfg.similarity_method,
            "distance_metric": cfg.distance_metric,
            "metric_weights": cfg.metric_weights,
            "group_weights": cfg.group_weights,
            "fill_value": cfg.fill_value,
        },
    }

    return {
        "similarity": sim,
        "distance": dist,
        "metric_vs_tab": kpi,
        "tab_type_summary": summary,
        "tab_type_similarity": type_sim,
        "ranks": ranks,
        "metric_corr": metric_corr,
        "meta": make_json_safe(meta_payload),
    }


# =============================================================================
# 20) Example / Smoke Test
# =============================================================================


def _example_smoke() -> None:
    """
    פונקציה קטנה לבדיקה ידנית (לא רצה בפרוד).

    שימוש:
        from core.tab_comparison_matrices import _example_smoke
        _example_smoke()
    """
    profiles: List[TabProfile] = [
        TabProfile(
            tab_id="stats_backtest",
            tab_type="stats",
            label="Backtest KPIs",
            weight=1.2,
            metrics={
                "sharpe": 1.5,
                "sortino": 2.3,
                "max_dd": -0.20,
                "macro_sensitivity": 0.4,
                "valuation_score": np.nan,
            },
            tags=["stats", "backtest", "universe"],
        ),
        TabProfile(
            tab_id="macro_engine",
            tab_type="macro",
            label="Macro Engine",
            weight=1.0,
            metrics={
                "sharpe": 0.9,
                "sortino": 1.4,
                "max_dd": -0.12,
                "macro_sensitivity": 0.9,
                "valuation_score": np.nan,
            },
            tags=["macro", "regime"],
        ),
        TabProfile(
            tab_id="fundamentals_index",
            tab_type="fundamental",
            label="Index Fundamentals",
            weight=0.9,
            metrics={
                "sharpe": 1.1,
                "sortino": 1.7,
                "max_dd": -0.15,
                "macro_sensitivity": 0.5,
                "valuation_score": 0.8,
            },
            tags=["fundamental", "index"],
        ),
    ]

    metric_meta = {
        "sharpe": MetricMeta(name="sharpe", direction="higher_better", group="risk_adj", weight=1.2),
        "sortino": MetricMeta(name="sortino", direction="higher_better", group="risk_adj", weight=1.0),
        "max_dd": MetricMeta(name="max_dd", direction="lower_better", group="drawdown", weight=1.5),
        "macro_sensitivity": MetricMeta(name="macro_sensitivity", direction="neutral", group="macro", weight=1.0),
        "valuation_score": MetricMeta(name="valuation_score", direction="higher_better", group="fundamental", weight=1.3),
    }

    cfg = TabComparisonConfig(
        normalization="zscore",
        similarity_method="cosine",
        distance_metric="euclidean",
        metric_meta=metric_meta,
        group_weights={"risk_adj": 1.0, "drawdown": 1.2, "macro": 0.8, "fundamental": 1.0},
    )

    bundle = build_comparison_bundle(profiles, cfg=cfg)

    print("=== Similarity (Tab vs Tab) ===")
    print(bundle["similarity"])
    print("\n=== Distance (Tab vs Tab) ===")
    print(bundle["distance"])
    print("\n=== Metric vs Tab (normalized) ===")
    print(bundle["metric_vs_tab"])
    print("\n=== Summary by tab_type ===")
    print(bundle["tab_type_summary"])
    print("\n=== TabType similarity ===")
    print(bundle["tab_type_similarity"])
    print("\n=== Ranks (metric vs tab) ===")
    print(bundle["ranks"])
    print("\n=== Metric correlation ===")
    print(bundle["metric_corr"])

    # Explainability example
    print("\n=== Similarity contributions (stats_backtest vs macro_engine) ===")
    contrib = explain_similarity_contributions(
        profiles,
        cfg=cfg,
        tab_pair=("stats_backtest", "macro_engine"),
    )
    print(contrib.head())

    # Metaopt groups
    groups = build_metaopt_candidate_groups(bundle["similarity"], threshold=0.7)
    print("\n=== Meta-opt candidate groups ===")
    print(groups)

    # Anomalies
    anomalies = detect_tab_anomalies(bundle["distance"])
    print("\n=== Anomaly detection ===")
    print(anomalies)

    # Composite example
    composite = build_composite_profile("macro_composite", "Macro Composite", [profiles[1], profiles[2]])
    print("\n=== Composite profile metrics ===")
    print(composite.tab_id, composite.metrics)


if __name__ == "__main__":
    _example_smoke()
