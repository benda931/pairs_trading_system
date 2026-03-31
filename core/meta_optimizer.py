# -*- coding: utf-8 -*-
"""
core/meta_optimizer.py — Meta-Optimization for Pairs Trading
============================================================

Role
----
This module sits *above* the single-pair backtester and clustering logic and
provides a **meta-layer**:

- Takes a DataFrame of candidate parameter sets / clusters (each row is a
  parameter configuration with attached performance metrics).
- Normalizes multiple metrics (Sharpe, Sortino, Calmar, Return, Drawdown,
  Volatility, Tail-Risk, Turnover, etc.) into a common [0, 1] scale.
- Applies a configurable weighted meta-score combining all metrics.
- Selects the “best” parameter set and ranks all candidates.
- Computes a simple feature-importance proxy via |corr(parameter, meta_score)|.

Input expectation
-----------------
`clusters: pd.DataFrame` should typically contain:
- One row per parameter set / cluster.
- Columns for performance metrics: e.g. 'sharpe', 'sortino', 'calmar',
  'return', 'drawdown', 'volatility', 'tail_risk', 'turnover', 'win_rate',
  'skew', etc. (only those available will be used).
- Columns for parameters: keys defined in `config["ranges"]`.

Config expectation
------------------
`config: dict` can provide:

1. New/extended format (recommended):

    config["meta_optimize"]["metrics"] = {
        "sharpe":      {"weight": 0.25, "higher_is_better": True},
        "sortino":     {"weight": 0.15, "higher_is_better": True},
        "calmar":      {"weight": 0.10, "higher_is_better": True},
        "return":      {"weight": 0.10, "higher_is_better": True},
        "win_rate":    {"weight": 0.05, "higher_is_better": True},
        "drawdown":    {"weight": 0.10, "higher_is_better": False},
        "volatility":  {"weight": 0.05, "higher_is_better": False},
        "tail_risk":   {"weight": 0.05, "higher_is_better": False},
        "turnover":    {"weight": 0.05, "higher_is_better": False},
        "skew":        {"weight": 0.05, "higher_is_better": True},
    }

2. Legacy format (backwards compatible):

    config["meta_optimize"]["weights"] = {
        "sharpe": 0.5,
        "return": 0.2,
        "drawdown": 0.2,
        "win_rate": 0.1
    }

In both cases:
- Any metric not present as a column in `clusters` is silently skipped.
- Any metric with zero weight is ignored.

Output
------
meta_optimize(...) returns:

    {
        "best_params": dict,                 # parameter set with max meta_score
        "feature_importance": pd.DataFrame,  # columns: feature, importance
        "all_scores": pd.DataFrame,          # clusters + meta_score/meta_rank/...
        "top_candidates": pd.DataFrame,      # (optional) top-N rows by meta_score
    }

The extra "top_candidates" key is additive and does not break existing callers
that only expect best_params / feature_importance / all_scores.
"""

from __future__ import annotations

from typing import Dict, Any, Mapping, Optional, List

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    )
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)


MetricSpec = Dict[str, Dict[str, Any]]  # {"metric_name": {"weight": float, "higher_is_better": bool}}


# ======================================================================
# Metric spec extraction & normalization helpers
# ======================================================================

def _get_metric_spec(config: Mapping[str, Any]) -> MetricSpec:
    """
    מחלץ spec של המטריקות מתוך ה-config.

    Priority:
      1) New “metrics” format (recommended).
      2) Legacy “weights” format.
      3) Extended default spec if nothing is specified.

    New format (recommended)
    ------------------------
        config["meta_optimize"]["metrics"] = {
            "sharpe":      {"weight": 0.25, "higher_is_better": True},
            "sortino":     {"weight": 0.15, "higher_is_better": True},
            "calmar":      {"weight": 0.10, "higher_is_better": True},
            "return":      {"weight": 0.10, "higher_is_better": True},
            "win_rate":    {"weight": 0.05, "higher_is_better": True},
            "drawdown":    {"weight": 0.10, "higher_is_better": False},
            "volatility":  {"weight": 0.05, "higher_is_better": False},
            "tail_risk":   {"weight": 0.05, "higher_is_better": False},
            "turnover":    {"weight": 0.05, "higher_is_better": False},
            "skew":        {"weight": 0.05, "higher_is_better": True},
        }

    Legacy format
    -------------
        config["meta_optimize"]["weights"] = {
            "sharpe": 0.5,
            "return": 0.2,
            "drawdown": 0.2,
            "win_rate": 0.1
        }

        In this case, we infer higher_is_better heuristically:
        - drawdown, vol, volatility, tail_risk, turnover => lower is better.
        - everything else => higher is better.

    Extended default
    ----------------
    If nothing is provided in config, we use an extended default set that covers
    typical hedge-fund grade metrics (only those present in the DataFrame will
    actually be used).
    """
    meta_cfg = config.get("meta_optimize", {}) if isinstance(config, Mapping) else {}

    # ---- 1) New "metrics" format ----
    metrics = meta_cfg.get("metrics")
    if isinstance(metrics, Mapping) and metrics:
        spec: MetricSpec = {}
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

    # ---- 2) Legacy "weights" format ----
    weights = meta_cfg.get("weights", {}) if isinstance(meta_cfg, Mapping) else {}
    if isinstance(weights, Mapping) and weights:
        spec: MetricSpec = {}
        for name, w in weights.items():
            weight = float(w)
            if weight == 0.0:
                continue
            lname = name.lower()
            # drawdown/vol/tail_risk/turnover – נמוך עדיף; אחרת גבוה עדיף
            if lname in {"drawdown", "dd", "max_drawdown", "vol", "volatility", "tail_risk", "turnover"}:
                hib = False
            else:
                hib = True
            spec[name] = {"weight": weight, "higher_is_better": hib}
        if spec:
            return spec

    # ---- 3) Extended default spec ----
    logger.info("meta_optimizer: using extended default metric spec.")
    return {
        # Return-quality ratios
        "sharpe":      {"weight": 0.25, "higher_is_better": True},
        "sortino":     {"weight": 0.15, "higher_is_better": True},
        "calmar":      {"weight": 0.10, "higher_is_better": True},

        # Return & hit stats
        "return":      {"weight": 0.10, "higher_is_better": True},   # CAGR / total return
        "win_rate":    {"weight": 0.05, "higher_is_better": True},

        # Risk & tail measures
        "drawdown":    {"weight": 0.10, "higher_is_better": False},
        "max_drawdown": {"weight": 0.05, "higher_is_better": False},
        "volatility":  {"weight": 0.05, "higher_is_better": False},
        "tail_risk":   {"weight": 0.05, "higher_is_better": False},

        # Structure / trading style
        "turnover":    {"weight": 0.05, "higher_is_better": False},
        "skew":        {"weight": 0.05, "higher_is_better": True},
    }


def _normalize_series(
    series: pd.Series,
    higher_is_better: bool,
) -> pd.Series:
    """
    Normalize a metric to [0, 1] robustly.

    Logic:
      - Ignore NaN for min/max computation.
      - If range collapses (max == min) => return 0.5 for all rows
        (no information in that metric).
      - If higher_is_better = True:
          norm = (x - min) / (max - min)
        else:
          norm = (max - x) / (max - min)
      - Any NaN are set to 0.5 (neutral).

    This keeps the meta-score stable even in edge cases.
    """
    s = series.astype(float)
    mask_valid = s.notna()
    if not mask_valid.any():
        return pd.Series(0.5, index=series.index)

    s_valid = s[mask_valid]
    v_min = float(s_valid.min())
    v_max = float(s_valid.max())

    if np.isclose(v_max, v_min):
        # No variation => no information
        return pd.Series(0.5, index=series.index)

    denom = v_max - v_min
    if higher_is_better:
        norm_values = (s - v_min) / denom
    else:
        norm_values = (v_max - s) / denom

    norm = norm_values.clip(0.0, 1.0)
    norm[~mask_valid] = 0.5
    return norm


def _build_meta_scores(
    clusters: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """
    Builds meta-score and ranking on top of the clusters DataFrame.

    Adds the following columns:

      - <metric>_norm: normalized version of each metric in [0, 1].
      - meta_score:    weighted average of normalized metrics.
      - meta_rank:     rank of meta_score (1 = best).
      - meta_zscore:   optional standardized version of meta_score.

    Metrics that are missing from `clusters` are silently skipped.
    Metrics with zero weight are ignored.
    """
    df = clusters.copy()
    metric_spec = _get_metric_spec(config)

    if df.empty:
        logger.info("meta_optimizer: received empty clusters DataFrame.")
        return df

    meta_components: List[pd.Series] = []
    weights_used: List[float] = []

    for metric_name, spec in metric_spec.items():
        col = metric_name
        if col not in df.columns:
            logger.debug(
                "meta_optimizer: metric '%s' not found in clusters columns, skipping.",
                metric_name,
            )
            continue

        weight = float(spec.get("weight", 0.0))
        if weight == 0.0:
            continue

        higher_is_better = bool(spec.get("higher_is_better", True))
        norm_col = f"{col}_norm"

        df[norm_col] = _normalize_series(df[col], higher_is_better)
        meta_components.append(df[norm_col] * weight)
        weights_used.append(weight)

    if not meta_components:
        logger.warning(
            "meta_optimizer: no valid metrics for scoring – meta_score will be NaN."
        )
        df["meta_score"] = np.nan
        df["meta_rank"] = np.nan
        df["meta_zscore"] = np.nan
        return df

    total_weight = float(sum(weights_used)) or 1.0
    df["meta_score"] = sum(meta_components) / total_weight

    # Rank: highest is best
    df["meta_rank"] = df["meta_score"].rank(method="min", ascending=False)

    # Z-score (useful to see outliers)
    ms = pd.to_numeric(df["meta_score"], errors="coerce")
    if ms.notna().nunique() > 1:
        mean = float(ms.mean())
        std = float(ms.std(ddof=1)) or 1.0
        df["meta_zscore"] = (ms - mean) / std
    else:
        df["meta_zscore"] = 0.0

    return df


# ======================================================================
# Feature importance — simple |corr(param, meta_score)|
# ======================================================================

def _compute_feature_importance(
    scored_df: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """
    Computes a simple feature importance proxy:

        importance(feature) = |corr(feature, meta_score)|

    Only parameters defined in `config["ranges"]` are considered as features.

    Returns
    -------
    DataFrame with columns:
        - feature
        - importance   (0..1, where 1 = perfect linear correlation in absolute value)
    """
    if "meta_score" not in scored_df.columns:
        return pd.DataFrame(columns=["feature", "importance"])

    df = scored_df.copy()
    meta = pd.to_numeric(df["meta_score"], errors="coerce")
    if meta.notna().nunique() <= 1:
        return pd.DataFrame(columns=["feature", "importance"])

    ranges_cfg = config.get("ranges", {}) if isinstance(config, Mapping) else {}
    param_names = set(ranges_cfg.keys())

    importance_rows: List[Dict[str, Any]] = []

    for name in param_names:
        if name not in df.columns:
            continue
        s = pd.to_numeric(df[name], errors="coerce")
        if s.notna().nunique() <= 1:
            importance_rows.append({"feature": name, "importance": 0.0})
            continue

        corr = s.corr(meta)
        imp = float(abs(corr)) if not pd.isna(corr) else 0.0
        importance_rows.append({"feature": name, "importance": imp})

    if not importance_rows:
        return pd.DataFrame(columns=["feature", "importance"])

    fi = pd.DataFrame(importance_rows)
    fi = fi.sort_values("importance", ascending=False).reset_index(drop=True)
    return fi


# ======================================================================
# Public API
# ======================================================================

def meta_optimize(
    clusters: pd.DataFrame,
    config: dict,
) -> dict:
    """
    Meta-optimizes across clusters to find the best blend of parameters.

    Parameters
    ----------
    clusters:
        DataFrame from cluster_pairs / optimization pipeline.
        Each row represents a parameter set (cluster) with:
            - Performance metrics columns (sharpe, sortino, return, drawdown, ...)
            - Parameter columns (names defined in config["ranges"].keys()).

    config:
        dict with at least:
            - "ranges": mapping of parameter names to their ranges (for identifying
              which columns are parameters when computing feature importance).
            - "meta_optimize": optional dict with:
                * "metrics"  (new format)  OR
                * "weights"  (legacy format)
              plus possibly:
                * "top_n": int – how many top candidates to expose.

    Returns
    -------
    dict with:
        - "best_params": dict
            Parameter set of the row with maximum meta_score.
        - "feature_importance": pd.DataFrame(feature, importance)
        - "all_scores": pd.DataFrame
            Original clusters + meta_score, meta_rank, meta_zscore, *_norm columns.
        - "top_candidates": pd.DataFrame
            Top-N rows by meta_score (N can be configured).
    """
    if clusters is None or clusters.empty:
        logger.info("meta_optimizer.meta_optimize: empty clusters, nothing to do.")
        return {}

    cfg_mapping: Mapping[str, Any] = config if isinstance(config, Mapping) else {}

    # Build meta-scores
    scored_df = _build_meta_scores(clusters, cfg_mapping)

    if "meta_score" not in scored_df.columns or scored_df["meta_score"].isna().all():
        logger.warning(
            "meta_optimizer: meta_score could not be computed – returning minimal result."
        )
        return {
            "best_params": {},
            "feature_importance": pd.DataFrame(columns=["feature", "importance"]),
            "all_scores": scored_df,
            "top_candidates": scored_df.head(0),
        }

    # Top-N logic (for dashboard display)
    meta_cfg = cfg_mapping.get("meta_optimize", {})
    top_n = int(meta_cfg.get("top_n", 5) or 5)

    # Best row
    best_idx = scored_df["meta_score"].idxmax()
    best_row = scored_df.loc[best_idx]

    ranges_cfg = cfg_mapping.get("ranges", {}) if isinstance(cfg_mapping, Mapping) else {}
    param_names = set(ranges_cfg.keys())

    best_params: Dict[str, Any] = {}
    for p_name in param_names:
        if p_name in best_row.index:
            best_params[p_name] = best_row[p_name]

    # Feature importance
    feature_importance = _compute_feature_importance(scored_df, cfg_mapping)

    # Top candidates
    top_candidates = (
        scored_df.sort_values("meta_score", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    logger.info(
        "meta_optimizer: selected best params (meta_score=%.6f): %s",
        float(best_row["meta_score"]),
        best_params,
    )

    return {
        "best_params": best_params,
        "feature_importance": feature_importance,
        "all_scores": scored_df,
        "top_candidates": top_candidates,
    }

# ======================================================================
# Optional helper: build clusters DataFrame from backtests
# ======================================================================

def build_clusters_dataframe(
    pairs: list[tuple[str, str]],
    param_grid: list[dict[str, Any]],
    *,
    static_params: Optional[dict[str, Any]] = None,
    max_fail_ratio: float = 0.3,
) -> pd.DataFrame:
    """
    Run a grid of backtests across multiple pairs and aggregate results
    into a single `clusters` DataFrame suitable for `meta_optimize`.

    Parameters
    ----------
    pairs:
        List of (symbol_a, symbol_b) tuples. Example:
            [("XLY", "XLP"), ("QQQ", "SOXX")]

    param_grid:
        List of parameter dictionaries. Each dict represents a *strategy
        parameter set* (lookback, z_open, z_close, filters, etc.).
        These params will be merged with `static_params` and passed to
        `core.optimization_backtester.run_backtest_pair`.

        Example:
            param_grid = [
                {"lookback": 60, "z_open": 2.0, "z_close": 0.5},
                {"lookback": 90, "z_open": 2.5, "z_close": 0.7},
                ...
            ]

    static_params:
        Optional dictionary of parameters that are shared across all
        runs (dates, capital, commission, risk settings, etc.).
        Values in `param_grid` override `static_params` if keys overlap.

        Example:
            static_params = {
                "start": date(2015, 1, 1),
                "end": date(2025, 1, 1),
                "initial_capital": 100_000,
                "commission_bps": 0.5,
                ...
            }

    max_fail_ratio:
        If more than this fraction of backtests fail (raise exceptions),
        the function will log a warning. Successful runs are still returned.

    Returns
    -------
    pd.DataFrame
        Each row corresponds to (pair, parameter set) with columns:

          - symbol_a, symbol_b, pair_id
          - all keys from the chosen parameter set
          - performance metrics from the backtester (Sharpe, Profit, Drawdown, ...)
            normalized to lower-case column names where possible:
                "Sharpe"   -> "sharpe"
                "Profit"   -> "profit"
                "Drawdown" -> "drawdown"
                "WinRate"  -> "win_rate"
                etc.

        This DataFrame can be fed directly into `meta_optimize(...)`.
    """
    # local import to avoid circular dependency at module import time
    from core.optimization_backtester import run_backtest_pair  # type: ignore

    if not pairs or not param_grid:
        logger.info("build_clusters_dataframe: empty pairs/param_grid – returning empty DataFrame.")
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    static_params = dict(static_params or {})

    total_runs = len(pairs) * len(param_grid)
    fail_count = 0

    # mapping from common perf keys to canonical column names
    key_map = {
        "sharpe": "sharpe",
        "sortino": "sortino",
        "calmar": "calmar",
        "profit": "profit",
        "pnl": "profit",
        "return": "return",
        "cagr": "return",
        "drawdown": "drawdown",
        "max_drawdown": "drawdown",
        "winrate": "win_rate",
        "win_rate": "win_rate",
        "trades": "n_trades",
        "n_trades": "n_trades",
        "avgtradepnl": "avg_trade_pnl",
        "avg_trade_pnl": "avg_trade_pnl",
        "exposure": "exposure",
        "turnover": "turnover",
        "vol": "volatility",
        "volatility": "volatility",
        "tail_risk": "tail_risk",
        "skew": "skew",
    }

    run_idx = 0
    for sym_a, sym_b in pairs:
        pair_id = f"{sym_a}_{sym_b}"
        for param_set in param_grid:
            run_idx += 1
            params = static_params.copy()
            params.update(param_set)

            try:
                perf_raw = run_backtest_pair(sym_a, sym_b, **params)
            except Exception as exc:
                fail_count += 1
                logger.warning(
                    "build_clusters_dataframe: backtest failed for pair %s with params %s: %s",
                    pair_id,
                    param_set,
                    exc,
                )
                continue

            if not isinstance(perf_raw, dict):
                logger.warning(
                    "build_clusters_dataframe: run_backtest_pair returned non-dict (%r), skipping.",
                    type(perf_raw),
                )
                continue

            row: dict[str, Any] = {
                "symbol_a": sym_a,
                "symbol_b": sym_b,
                "pair_id": pair_id,
            }

            # attach parameters
            for pk, pv in param_set.items():
                row[pk] = pv

            # attach performance metrics, normalize key names where possible
            for k, v in perf_raw.items():
                if not isinstance(k, str):
                    continue
                key_norm = k.strip()
                lower = key_norm.lower()
                col_name = key_map.get(lower, lower)
                row[col_name] = v

            rows.append(row)

    if fail_count > 0:
        frac = fail_count / max(total_runs, 1)
        if frac > max_fail_ratio:
            logger.warning(
                "build_clusters_dataframe: %.1f%% (%d/%d) of backtests failed.",
                frac * 100.0,
                fail_count,
                total_runs,
            )
        else:
            logger.info(
                "build_clusters_dataframe: %d/%d backtests failed (%.1f%%).",
                fail_count,
                total_runs,
                frac * 100.0,
            )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return df
