# core/ranges.py
"""Advanced parameter range utilities for optimization & research.

This module is the **single source of truth** for all parameter ranges used by
Omri's pairs‑trading system:

- Centralizes numeric ranges, groups, and sampling distributions.
- Provides helpers to derive ranges from historical runs & macro regimes.
- Exposes sensitivity tools for quant research (low/mid/high, priors, etc.).
- Offers a RangeManager facade that higher‑level code (UI / optimization)
  can use instead of dealing with raw dicts.

The public API keeps the same function names as the previous version:
- DEFAULT_PARAM_RANGES, PARAM_GROUPS, PARAM_DISTRIBUTIONS
- load_ranges_config, filter_param_ranges, compute_data_driven_ranges,
  compute_macro_driven_ranges, enforce_conditional_ranges, zoom_in_ranges,
  save_range_profile, load_range_profile, validate_ranges
- analyze_range_sensitivity, visualize_range_histograms,
  adjust_steps_dynamically, generate_bayesian_priors, RangeManager
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import Any, Callable, Dict, List, Tuple, TypedDict
import argparse
import json
import logging
import os

from pydantic import BaseModel, Field, ConfigDict

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)
if not logger.handlers:
    # Do not spam root logger if the app already configured logging.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

RangeTuple = Tuple[float, float, float]
RangeDict = Dict[str, RangeTuple]


class DistributionSpec(TypedDict, total=False):
    """Sampling distribution spec consumed by optimization samplers.

    Attributes
    ----------
    type: str
        'uniform', 'log_uniform' or 'categorical'.
    low, high: float
        Bounds for numeric distributions.
    step: float
        Optional granularity for grid‑like sampling.
    choices: List[Any]
        Discrete support for categorical parameters.
    """

    type: str
    low: float
    high: float
    step: float
    choices: List[Any]


# ---------------------------------------------------------------------------
# Pydantic models (config layer)
# ---------------------------------------------------------------------------


class ParamRangeModel(BaseModel):
    """Strongly‑typed representation of a single parameter range."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    low: float = Field(..., description="Lower bound of the parameter")
    high: float = Field(..., description="Upper bound of the parameter")
    step: float = Field(..., gt=0, description="Step size for sampling within the range")


class RangesConfig(BaseModel):
    """Container for default ranges & parameter groups.

    UI / API layers should consume this instead of raw dicts when possible.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    default_ranges: Dict[str, ParamRangeModel]
    param_groups: Dict[str, List[str]]


# ---------------------------------------------------------------------------
# Default configuration – single source of truth
# ---------------------------------------------------------------------------

# Any parameter that appears in PARAM_GROUPS **must** exist here.

DEFAULT_PARAM_RANGES: RangeDict = {
    # Core Z‑Score thresholds
    "z_entry": (0.5, 3.0, 0.1),
    "z_exit": (0.1, 1.5, 0.1),

    # Lookback / timing
    "lookback": (20.0, 252.0, 5.0),
    "entry_decay": (0.1, 1.0, 0.05),
    "exit_decay": (0.1, 1.5, 0.05),
    "min_trade_distance": (1.0, 20.0, 1.0),
    "max_trade_duration": (5.0, 60.0, 1.0),

    # Cointegration & hedging
    "cointegration_deviation": (0.0, 3.0, 0.1),
    "ADF_pval": (0.0, 0.3, 0.01),
    "hedge_ratio": (-3.0, 3.0, 0.05),
    "ECM_coeff": (-1.0, 0.0, 0.02),
    "kalman_beta": (-3.0, 3.0, 0.05),

    # Spread statistics & mean‑reversion
    "spread_mean": (-5.0, 5.0, 0.1),
    "spread_std": (0.01, 5.0, 0.05),
    "half_life": (1.0, 120.0, 1.0),
    "mean_reversion_speed": (0.0, 1.0, 0.02),
    "hurst": (0.0, 1.0, 0.02),
    "rolling_skewness": (-2.0, 2.0, 0.05),
    "rolling_kurtosis": (0.0, 10.0, 0.1),
    "percentile_95": (0.0, 3.0, 0.05),
    "percentile_05": (-3.0, 0.0, 0.05),
    "time_in_band": (0.0, 1.0, 0.05),
    "spread_momentum": (-3.0, 3.0, 0.1),
    "z_momentum": (-3.0, 3.0, 0.1),
    "ema_spread": (-5.0, 5.0, 0.1),
    "cusum_spread": (-5.0, 5.0, 0.1),
    "spread_drawdown": (-5.0, 0.0, 0.1),
    "vol_adj_spread": (-5.0, 5.0, 0.1),

    # ML & meta‑models
    "ml_pred": (-1.0, 1.0, 0.01),
    "meta_score": (0.0, 1.0, 0.01),
    "shap_importance": (0.0, 1.0, 0.01),
    "pca_1": (-3.0, 3.0, 0.1),
    "pca_2": (-3.0, 3.0, 0.1),
    "pca_3": (-3.0, 3.0, 0.1),
    "cluster_label": (0.0, 10.0, 1.0),
    "copula_tail": (0.0, 1.0, 0.02),

    # Regime sensitivities
    "volatility_regime": (0.1, 2.0, 0.1),
    "trend_strength": (0.0, 1.0, 0.05),
    "vix_level": (10.0, 50.0, 1.0),

    # Advanced risk metrics
    "var_confidence": (0.90, 0.99, 0.01),
    "cvar_confidence": (0.90, 0.99, 0.01),
    "drawdown_recovery": (5.0, 60.0, 5.0),

    # Macro / external factors
    "interest_rate_diff": (0.0, 0.05, 0.001),
    "fx_rate_volatility": (0.01, 0.2, 0.01),

    # Position & leverage
    "max_leverage": (1.0, 10.0, 0.5),
    "position_sizing_pct": (0.01, 0.2, 0.01),

    # Additional performance metrics
    "sortino_ratio": (0.0, 3.0, 0.1),
    "omega_ratio": (0.5, 2.0, 0.1),

    # Quality of trades
    "avg_hold_time": (1.0, 30.0, 1.0),
    "std_hold_time": (1.0, 30.0, 1.0),
    "trade_frequency": (1.0, 20.0, 1.0),

    # Advanced ML diagnostics
    "concept_drift_score": (0.0, 1.0, 0.01),
    "meta_model_confidence": (0.5, 1.0, 0.01),

    # Execution costs
    "slippage_pct": (0.0, 0.005, 0.0001),
    "commission_pct": (0.0, 0.002, 0.0001),
    "bid_ask_spread": (0.0, 0.01, 0.0001),

    # Liquidity metrics
    "avg_daily_volume": (10_000.0, 10_000_000.0, 10_000.0),
    "market_depth": (1_000.0, 1_000_000.0, 1_000.0),

    # Seasonality / time‑of‑day
    "time_of_day_factor": (0.5, 1.5, 0.1),
    "dow_bias": (0.9, 1.1, 0.01),

    # Regime detection
    "regime_cluster_count": (2.0, 5.0, 1.0),
    "regime_switch_penalty": (0.0, 0.5, 0.01),

    # Dynamic risk management
    "stopout_level": (0.05, 0.5, 0.01),
    "dynamic_sl_multiplier": (0.5, 2.0, 0.1),

    # Feature drift
    "drift_window": (50.0, 500.0, 50.0),
    "drift_sensitivity": (0.1, 1.0, 0.1),

    # Super‑professional advanced metrics
    "volatility_skew": (-2.0, 2.0, 0.1),            # volatility distribution asymmetry
    "liquidity_ratio": (0.1, 5.0, 0.1),              # depth / volume ratio
    "momentum_decay": (0.5, 1.5, 0.05),              # decay factor for momentum signals
    "execution_risk": (0.0, 0.02, 0.0005),           # execution uncertainty premium
    "carry_trade_score": (-1.0, 1.0, 0.01),          # normalized carry indicator
}


PARAM_GROUPS: Dict[str, List[str]] = {
    # Z‑Score logic
    "Z-Score Basics": ["z_entry", "z_exit", "lookback"],

    # Decay & timing controls
    "Decay & Timing": ["entry_decay", "exit_decay", "min_trade_distance", "max_trade_duration"],

    # Cointegration & hedging
    "Cointegration & Hedging": [
        "cointegration_deviation",
        "ADF_pval",
        "hedge_ratio",
        "ECM_coeff",
        "kalman_beta",
    ],

    # Spread statistics
    "Spread Statistics": [
        "spread_mean",
        "spread_std",
        "half_life",
        "mean_reversion_speed",
        "hurst",
        "rolling_skewness",
        "rolling_kurtosis",
        "percentile_95",
        "percentile_05",
        "time_in_band",
        "spread_momentum",
        "z_momentum",
        "ema_spread",
        "cusum_spread",
        "spread_drawdown",
        "vol_adj_spread",
    ],

    # ML & cointegration
    "ML & Cointegration": [
        "ml_pred",
        "meta_score",
        "shap_importance",
        "pca_1",
        "pca_2",
        "pca_3",
        "cluster_label",
        "copula_tail",
    ],

    # Regime sensitivities
    "Regime Sensitivities": ["volatility_regime", "trend_strength", "vix_level"],

    # Advanced risk metrics
    "Advanced Risk Metrics": ["var_confidence", "cvar_confidence", "drawdown_recovery"],

    # Macro / external factors
    "Macro / External Factors": ["interest_rate_diff", "fx_rate_volatility"],

    # Position & leverage
    "Position & Leverage": ["max_leverage", "position_sizing_pct"],

    # Additional performance metrics
    "Additional Perf. Metrics": ["sortino_ratio", "omega_ratio"],

    # Quality of trades
    "Quality of Trades": ["avg_hold_time", "std_hold_time", "trade_frequency"],

    # Advanced ML metrics
    "Advanced ML Metrics": ["concept_drift_score", "meta_model_confidence"],

    # Execution costs
    "Execution Costs": ["slippage_pct", "commission_pct", "bid_ask_spread"],

    # Liquidity metrics
    "Liquidity Metrics": ["avg_daily_volume", "market_depth"],

    # Seasonality & time‑of‑day
    "Seasonality & Time-of-Day": ["time_of_day_factor", "dow_bias"],

    # Regime detection
    "Regime Detection": ["regime_cluster_count", "regime_switch_penalty"],

    # Dynamic risk management
    "Dynamic Risk Management": ["stopout_level", "dynamic_sl_multiplier"],

    # Feature drift
    "Feature Drift": ["drift_window", "drift_sensitivity"],
}


# Directory to persist range profiles (can be overridden by env var)
RANGE_PROFILE_DIR = os.environ.get("RANGE_PROFILE_DIR", "range_profiles")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _coerce_numeric(value: Any) -> float:
    """Best‑effort cast to float with a clear error on failure."""

    try:
        return float(value)
    except Exception as exc:  # pragma: no cover - defensive
        raise TypeError(f"Cannot coerce value {value!r} to float") from exc


try:  # Prefer shared helper if available to avoid duplication
    from common.helpers import make_json_safe  # type: ignore
except Exception:  # pragma: no cover - fallback

    def make_json_safe(obj: Any) -> Any:
        """Minimal JSON‑safe converter used by CLI and profile persistence.

        - tuples -> lists
        - numpy / pandas scalars -> Python scalars
        - datetime / Timestamp -> ISO strings
        """

        try:  # numpy is optional
            import numpy as _np  # type: ignore
        except Exception:  # pragma: no cover
            _np = None
        try:  # pandas is optional
            import pandas as _pd  # type: ignore
        except Exception:  # pragma: no cover
            _pd = None

        if isinstance(obj, dict):
            return {str(k): make_json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, set)):
            return [make_json_safe(v) for v in obj]
        if _np is not None and isinstance(obj, (_np.generic,)):
            return obj.item()
        if _pd is not None and isinstance(getattr(_pd, "Timestamp", object)) and isinstance(
            obj, getattr(_pd, "Timestamp", ())
        ):
            return obj.isoformat()
        return obj


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def load_ranges_config() -> RangesConfig:
    """Return a strongly‑typed config object for UI & API layers."""

    default = {
        name: ParamRangeModel(low=v[0], high=v[1], step=v[2])
        for name, v in DEFAULT_PARAM_RANGES.items()
    }
    logger.debug("Loaded ranges config via Pydantic models (%d params)", len(default))
    return RangesConfig(default_ranges=default, param_groups=PARAM_GROUPS)


def filter_param_ranges(
    base_ranges: RangeDict,
    use_ecm: bool = True,
    use_kalman: bool = True,
    use_ml: bool = True,
    use_meta: bool = True,
) -> RangeDict:
    """Filter out unused parameter ranges according to feature toggles.

    Notes
    -----
    - Returned dict is a *copy* and safe to mutate downstream.
    """

    if not isinstance(base_ranges, dict):  # pragma: no cover - defensive
        raise TypeError("base_ranges must be a dict")

    logger.debug(
        "filter_param_ranges: ecm=%s, kalman=%s, ml=%s, meta=%s",
        use_ecm,
        use_kalman,
        use_ml,
        use_meta,
    )

    ranges: RangeDict = dict(base_ranges)

    if not use_ecm:
        ranges.pop("ECM_coeff", None)
    if not use_kalman:
        ranges.pop("kalman_beta", None)
    if not use_ml:
        for key in ("ml_pred", "shap_importance"):
            ranges.pop(key, None)
    if not use_meta:
        ranges.pop("meta_score", None)

    logger.debug("filter_param_ranges: remaining %d params", len(ranges))
    return ranges


def compute_data_driven_ranges(
    history_df: Any,
    sel_params: List[str],
    multiplier: float = 1.0,
) -> RangeDict:
    """Derive dynamic ranges from historical optimization results.

    For each parameter ``p`` in ``sel_params`` that exists as a column in
    ``history_df``:
    - compute mean and standard deviation
    - shrink the search region to ``mu ± multiplier * sigma``
    - clamp to the global DEFAULT_PARAM_RANGES bounds
    """

    import pandas as pd  # local import to keep core import light

    if not hasattr(history_df, "columns"):
        raise TypeError("history_df must be a pandas DataFrame‑like object with .columns")
    if multiplier < 0:
        raise ValueError("multiplier must be non‑negative")

    df = history_df.copy()
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    logger.debug("compute_data_driven_ranges: params=%s, mul=%s", sel_params, multiplier)

    dyn: RangeDict = {}
    for p in sel_params:
        if p not in df.columns or p not in DEFAULT_PARAM_RANGES:
            continue
        series = pd.to_numeric(df[p], errors="coerce").dropna()
        if series.empty:
            continue
        mu = _coerce_numeric(series.mean())
        sigma = _coerce_numeric(series.std() or 0.0)
        low_default, high_default, step_default = DEFAULT_PARAM_RANGES[p]
        if sigma == 0:
            low = low_default
            high = high_default
        else:
            low = max(low_default, mu - multiplier * sigma)
            high = min(high_default, mu + multiplier * sigma)
        dyn[p] = (low, high, step_default)
        logger.debug("data_range[%s] -> (%.4f, %.4f, %.4f)", p, low, high, step_default)

    return dyn


def compute_macro_driven_ranges(
    history_df: Any,
    macro_df: Any,
    sel_params: List[str],
    multiplier: float = 1.0,
    macro_factor: float = 0.1,
) -> RangeDict:
    """Compute parameter ranges influenced by macroeconomic indicators.

    Flow
    ----
    1. Start from :func:`compute_data_driven_ranges`.
    2. For each parameter, look for a macro column named ``"{param}_macro"``.
    3. Expand/shrink the range proportionally to the *average* macro value.

    This gives a simple, explainable link between macro regimes and the
    optimization search space (e.g. wider ranges in high‑vol regimes).
    """

    import pandas as pd

    base_ranges = compute_data_driven_ranges(history_df, sel_params, multiplier)

    if not hasattr(macro_df, "columns"):
        raise TypeError("macro_df must be a pandas DataFrame‑like object with .columns")

    mdf = macro_df.copy()
    if not isinstance(mdf, pd.DataFrame):
        mdf = pd.DataFrame(mdf)

    adjusted: RangeDict = {}
    for p in sel_params:
        low, high, step = base_ranges.get(p, DEFAULT_PARAM_RANGES.get(p, (0.0, 0.0, 0.0)))
        macro_col = f"{p}_macro"
        if macro_col in mdf.columns:
            series = pd.to_numeric(mdf[macro_col], errors="coerce").dropna()
            if not series.empty:
                mean_macro = _coerce_numeric(series.mean())
                span = max(0.0, high - low)
                adj = span * macro_factor * mean_macro
                default_low, default_high, _ = DEFAULT_PARAM_RANGES.get(p, (low, high, step))
                low_adj = max(default_low, low - adj)
                high_adj = min(default_high, high + adj)
                adjusted[p] = (low_adj, high_adj, step)
                logger.debug(
                    "macro_range[%s] adjusted by macro=%.4f -> (%.4f, %.4f)",
                    p,
                    mean_macro,
                    low_adj,
                    high_adj,
                )
                continue
        adjusted[p] = (low, high, step)
    return adjusted


# ---------------------------------------------------------------------------
# Distributions for samplers (Optuna etc.)
# ---------------------------------------------------------------------------

PARAM_DISTRIBUTIONS: Dict[str, DistributionSpec] = {
    "lookback": {"type": "categorical", "choices": [30, 60, 90, 120, 252]},
    "entry_decay": {"type": "log_uniform", "low": 0.1, "high": 1.0},
}


@lru_cache(maxsize=None)
def get_param_distribution(param: str) -> DistributionSpec:
    """Return distribution spec for Optuna‑style sampling.

    If a custom distribution is not defined for ``param``, fall back to a
    simple uniform distribution over the default range.
    """

    if not isinstance(param, str):  # pragma: no cover - defensive
        raise TypeError("param must be a string")

    logger.debug("get_param_distribution: %s", param)

    spec = PARAM_DISTRIBUTIONS.get(param)
    if spec is not None:
        return spec

    if param not in DEFAULT_PARAM_RANGES:
        raise KeyError(f"Unknown parameter for distribution: {param}")

    low, high, step = DEFAULT_PARAM_RANGES[param]
    return {"type": "uniform", "low": low, "high": high, "step": step}


# ---------------------------------------------------------------------------
# Range logic helpers
# ---------------------------------------------------------------------------


def enforce_conditional_ranges(ranges: RangeDict) -> RangeDict:
    """Enforce logical relationships between parameters.

    Currently enforced:
    - ``entry_decay < exit_decay``
    - ``min_trade_distance < max_trade_duration``
    """

    if not isinstance(ranges, dict):  # pragma: no cover - defensive
        raise TypeError("ranges must be a dict")

    logger.debug("enforce_conditional_ranges starting")

    out = dict(ranges)
    if "entry_decay" in out and "exit_decay" in out:
        ed_low, _, ed_step = out["entry_decay"]
        _, ex_high, ex_step = out["exit_decay"]
        out["exit_decay"] = (ed_low + ed_step, ex_high, ex_step)

    if "min_trade_distance" in out and "max_trade_duration" in out:
        mtd_low, _, mtd_step = out["min_trade_distance"]
        _, md_high, md_step = out["max_trade_duration"]
        out["max_trade_duration"] = (mtd_low + mtd_step, md_high, md_step)

    return out


def zoom_in_ranges(
    base_ranges: RangeDict,
    best_params: Dict[str, float],
    factor: float = 0.1,
) -> RangeDict:
    """Shrink ranges around best values by a given fraction of the span.

    Parameters
    ----------
    base_ranges:
        Parameter -> (low, high, step).
    best_params:
        Parameter -> best value found so far.
    factor:
        Fraction of the full span to use as half‑width around ``center``.
    """

    if factor < 0 or factor > 1:
        raise ValueError("factor must be between 0 and 1")

    logger.debug("zoom_in_ranges: factor=%s", factor)

    out: RangeDict = {}
    for p, (low, high, step) in base_ranges.items():
        if p in best_params:
            center = _coerce_numeric(best_params[p])
            span = max(0.0, high - low)
            half = span * factor
            new_low = max(low, center - half)
            new_high = min(high, center + half)
            out[p] = (new_low, new_high, step)
        else:
            out[p] = (low, high, step)
    return out


def save_range_profile(name: str, ranges: RangeDict) -> None:
    """Persist a named range profile to disk (JSON)."""

    logger.debug("save_range_profile: %s", name)
    os.makedirs(RANGE_PROFILE_DIR, exist_ok=True)
    path = os.path.join(RANGE_PROFILE_DIR, f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(make_json_safe(ranges), f, ensure_ascii=False, indent=4)


@lru_cache(maxsize=32)
def load_range_profile(name: str) -> RangeDict:
    """Load a range profile created by :func:`save_range_profile`."""

    logger.debug("load_range_profile: %s", name)
    path = os.path.join(RANGE_PROFILE_DIR, f"{name}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Profile not found: {name}")
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    # Convert lists back to tuples for internal consistency
    return {k: tuple(v) for k, v in raw.items()}


def validate_ranges(ranges: RangeDict) -> None:
    """Validate that each range is well‑formed.

    Conditions
    ----------
    - ``low < high``
    - ``step > 0``
    - ``(high - low) >= step``
    """

    if not isinstance(ranges, dict):  # pragma: no cover - defensive
        raise TypeError("ranges must be a dict")

    logger.debug("validate_ranges starting (%d params)", len(ranges))

    errs: List[str] = []
    for p, (low, high, step) in ranges.items():
        if low >= high:
            errs.append(f"{p}: low >= high ({low} >= {high})")
        if step <= 0:
            errs.append(f"{p}: step <= 0 ({step})")
        if (high - low) < step:
            errs.append(f"{p}: span < step ({high - low} < {step})")

    if errs:
        raise ValueError("Invalid ranges: " + "; ".join(errs))


# ---------------------------------------------------------------------------
# Sensitivity & visualization
# ---------------------------------------------------------------------------


def analyze_range_sensitivity(
    simulate_fn: Callable[[Dict[str, float], Tuple[str, str]], Dict[str, Any]],
    symbols: Tuple[str, str],
    base_params: Dict[str, float],
    param_ranges: RangeDict,
    sel_params: List[str],
    max_workers: int | None = None,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Run a low/mid/high sensitivity analysis per parameter.

    Returns
    -------
    mapping
        ``param -> { 'low': perf, 'mid': perf, 'high': perf }``
    """

    logger.debug("analyze_range_sensitivity: %s", sel_params)
    sensitivity: Dict[str, Dict[str, Dict[str, Any]]] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for p in sel_params:
            low, high, _ = param_ranges[p]
            mid = (low + high) / 2.0
            for key, val in ("low", low), ("mid", mid), ("high", high):
                params = dict(base_params)
                params[p] = float(val)
                future = executor.submit(simulate_fn, params, symbols)
                futures[future] = (p, key)

        for future in as_completed(futures):
            p, key = futures[future]
            try:
                perf = future.result()
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("Error in sensitivity simulation for %s @%s: %s", p, key, exc)
                perf = {}
            sensitivity.setdefault(p, {})[key] = perf

    logger.debug("analyze_range_sensitivity completed")
    return sensitivity


def visualize_range_histograms(ranges: RangeDict, sel_params: List[str] | None = None) -> None:
    """Quick matplotlib visualization of parameter supports.

    Intended for ad‑hoc research, not for production dashboards.
    """

    import numpy as np
    import matplotlib.pyplot as plt

    logger.debug("visualize_range_histograms")

    params = sel_params or list(ranges.keys())
    for p in params:
        low, high, step = ranges[p]
        vals = np.arange(low, high + step, step)
        plt.figure()
        plt.hist(vals, bins=min(len(vals), 50))
        plt.title(p)
        plt.tight_layout()
        plt.show()


def adjust_steps_dynamically(ranges: RangeDict, max_steps: int = 100) -> RangeDict:
    """Adjust step sizes so that each parameter has ~max_steps support points.

    Useful when moving from a coarse global grid to a finer local search while
    keeping the number of evaluations bounded.
    """

    logger.debug("adjust_steps_dynamically: max_steps=%s", max_steps)

    out: RangeDict = {}
    for p, (low, high, old_step) in ranges.items():
        span = max(0.0, high - low)
        if span <= 0:
            step = old_step
        else:
            # keep at least 10% of the previous resolution to avoid over‑tightening
            step = max(span / max(max_steps, 1), old_step * 0.1)
        out[p] = (low, high, step)
    return out


def generate_bayesian_priors(ranges: RangeDict, factor: float = 0.1) -> Dict[str, Dict[str, float]]:
    """Generate simple Gaussian priors centered at mid‑range.

    ``sigma = (high - low) * factor``
    """

    logger.debug("generate_bayesian_priors: factor=%s", factor)
    priors: Dict[str, Dict[str, float]] = {}
    for p, (low, high, _) in ranges.items():
        span = max(0.0, high - low)
        mu = (low + high) / 2.0
        sigma = span * factor
        priors[p] = {"type": "normal", "mu": mu, "sigma": sigma}
    return priors


# ---------------------------------------------------------------------------
# Super‑professional extensions
# ---------------------------------------------------------------------------


def synchronize_ranges(ranges_list: List[RangeDict]) -> RangeDict:
    """Ensemble multiple range sets by averaging bounds.

    Parameters
    ----------
    ranges_list:
        List of parameter‑range dicts.
    """

    if not ranges_list:
        return {}

    merged: Dict[str, List[RangeTuple]] = {}
    for ranges in ranges_list:
        for p, val in ranges.items():
            merged.setdefault(p, []).append(val)

    out: RangeDict = {}
    for p, vals in merged.items():
        lows, highs, steps = zip(*vals)
        # simple arithmetic mean aggregation
        low = float(sum(lows) / len(lows))
        high = float(sum(highs) / len(highs))
        step = float(sum(steps) / len(steps))
        out[p] = (low, high, step)
    return out


def intersect_ranges(a: RangeDict, b: RangeDict) -> RangeDict:
    """Intersection of two range dicts (per‑parameter overlap).

    Only parameters present in **both** dicts are returned. If the overlap is
    empty (high <= low) for a parameter, it is dropped.
    """

    out: RangeDict = {}
    for p, (a_low, a_high, a_step) in a.items():
        if p not in b:
            continue
        b_low, b_high, b_step = b[p]
        low = max(a_low, b_low)
        high = min(a_high, b_high)
        if high <= low:
            continue
        step = min(a_step, b_step)
        out[p] = (low, high, step)
    return out


def merge_with_defaults(overrides: RangeDict) -> RangeDict:
    """Merge user overrides on top of DEFAULT_PARAM_RANGES.

    Any parameter not present in ``overrides`` keeps its default range.
    """

    base = dict(DEFAULT_PARAM_RANGES)
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Manager facade (object‑oriented convenience wrapper)
# ---------------------------------------------------------------------------


class RangeManager:
    """High‑level facade used by optimization & UI code.

    Typical lifecycle in the optimization tab:

    1. Construct from defaults:

       ``rm = RangeManager.from_defaults()``

    2. Apply feature toggles (ECM / Kalman / ML / Meta):

       ``rm = rm.with_feature_filters(use_ecm, use_kalman, use_ml, use_meta)``

    3. Optionally shrink ranges around historical winners / macro regimes.
    4. Optionally zoom‑in around best trial.
    5. Validate and hand ``rm.ranges`` to the optimizer.
    """

    def __init__(self, ranges: RangeDict | None = None) -> None:
        self._ranges: RangeDict = dict(ranges or DEFAULT_PARAM_RANGES)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_defaults(cls) -> "RangeManager":
        return cls(DEFAULT_PARAM_RANGES)

    @classmethod
    def from_profile(cls, name: str) -> "RangeManager":
        return cls(load_range_profile(name))

    # ------------------------------------------------------------------
    # Core properties
    # ------------------------------------------------------------------

    @property
    def ranges(self) -> RangeDict:
        return dict(self._ranges)

    def keys(self) -> List[str]:
        return list(self._ranges.keys())

    # ------------------------------------------------------------------
    # Transformations (functional style, always returning *new* manager)
    # ------------------------------------------------------------------

    def with_feature_filters(
        self,
        *,
        use_ecm: bool = True,
        use_kalman: bool = True,
        use_ml: bool = True,
        use_meta: bool = True,
    ) -> "RangeManager":
        filtered = filter_param_ranges(
            self._ranges,
            use_ecm=use_ecm,
            use_kalman=use_kalman,
            use_ml=use_ml,
            use_meta=use_meta,
        )
        return RangeManager(filtered)

    def with_zoom(self, best_params: Dict[str, float], factor: float = 0.1) -> "RangeManager":
        zoomed = zoom_in_ranges(self._ranges, best_params, factor=factor)
        return RangeManager(zoomed)

    def with_data_driven(
        self,
        history_df: Any,
        sel_params: List[str],
        multiplier: float = 1.0,
        merge_mode: str = "shrink",
    ) -> "RangeManager":
        """Incorporate history‑based ranges.

        merge_mode
        ----------
        - "shrink": intersection of current ranges and data‑driven ranges.
        - "replace": use data‑driven ranges for ``sel_params`` and defaults
          for others.
        """

        data_ranges = compute_data_driven_ranges(history_df, sel_params, multiplier)
        if merge_mode == "replace":
            merged = merge_with_defaults(data_ranges)
        elif merge_mode == "shrink":
            merged = intersect_ranges(self._ranges, merge_with_defaults(data_ranges))
        else:
            raise ValueError("merge_mode must be 'shrink' or 'replace'")
        return RangeManager(merged)

    def with_macro_driven(
        self,
        history_df: Any,
        macro_df: Any,
        sel_params: List[str],
        multiplier: float = 1.0,
        macro_factor: float = 0.1,
        merge_mode: str = "shrink",
    ) -> "RangeManager":
        macro_ranges = compute_macro_driven_ranges(
            history_df,
            macro_df,
            sel_params,
            multiplier=multiplier,
            macro_factor=macro_factor,
        )
        if merge_mode == "replace":
            merged = merge_with_defaults(macro_ranges)
        elif merge_mode == "shrink":
            merged = intersect_ranges(self._ranges, merge_with_defaults(macro_ranges))
        else:
            raise ValueError("merge_mode must be 'shrink' or 'replace'")
        return RangeManager(merged)

    def with_enforced_conditions(self) -> "RangeManager":
        return RangeManager(enforce_conditional_ranges(self._ranges))

    def with_adjusted_steps(self, max_steps: int = 100) -> "RangeManager":
        return RangeManager(adjust_steps_dynamically(self._ranges, max_steps=max_steps))

    def with_custom_overrides(self, overrides: RangeDict) -> "RangeManager":
        return RangeManager(merge_with_defaults(overrides))

    # ------------------------------------------------------------------
    # Validation & export
    # ------------------------------------------------------------------

    def validate(self) -> None:
        validate_ranges(self._ranges)

    def to_profile(self, name: str) -> None:
        save_range_profile(name, self._ranges)

    def to_config(self) -> RangesConfig:
        return load_ranges_config()


# ---------------------------------------------------------------------------
# CLI entry point (research / debugging only)
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Range utilities for Omri's pairs‑trading system")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("dump-defaults", help="Print DEFAULT_PARAM_RANGES as JSON")

    p_profile = sub.add_parser("profile", help="Save or load range profiles")
    p_profile.add_argument("action", choices=["save", "load"], help="save or load profile")
    p_profile.add_argument("name", help="Profile name (without .json)")

    p_hist = sub.add_parser("hist", help="Quick histogram visualization for selected params")
    p_hist.add_argument("params", nargs="+", help="Parameter names to visualize")

    return parser


def main(argv: List[str] | None = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.command == "dump-defaults":
        print(json.dumps(make_json_safe(DEFAULT_PARAM_RANGES), ensure_ascii=False, indent=2))
        return

    if args.command == "profile":
        if args.action == "save":
            save_range_profile(args.name, DEFAULT_PARAM_RANGES)
            print(f"Saved profile '{args.name}' to {RANGE_PROFILE_DIR}")
        else:
            loaded = load_range_profile(args.name)
            print(json.dumps(make_json_safe(loaded), ensure_ascii=False, indent=2))
        return

    if args.command == "hist":
        visualize_range_histograms(DEFAULT_PARAM_RANGES, sel_params=list(args.params))
        return


if __name__ == "__main__":  # pragma: no cover
    main()
