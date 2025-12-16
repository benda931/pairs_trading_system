# -*- coding: utf-8 -*-
"""
params.py — Central parameter specification & Fair-Value configuration
=====================================================================

This is a **single, clean source of truth** for:

1. ParamSpec & PARAM_SPECS
   ------------------------
   - Typed, immutable description of each tunable parameter:
       * name, bounds [lo, hi], step, log, choices, tags
   - Helpers for:
       * Optuna distributions & suggest()
       * Filtering by tags (signal / volatility / liquidity / macro / etc.)
       * Freezing / clamping / random sampling

2. Fair Value Engine configuration
   --------------------------------
   - `FairValueConfig` dataclass that aggregates all FAIR_VALUE_* flags
     into a single structured object.
   - `.from_params(params_dict)` / `.to_flat_params()` helpers used by
     the Fair-Value engine to read/write configuration from simple dicts.
   - All legacy FAIR_VALUE_* module-level constants are still provided
     for backward-compatibility, but `FairValueConfig` is the recommended
     interface going forward.

Design goals
============
- Hedge-fund–grade clarity and extensibility.
- Single source of truth for parameter bounds & defaults.
- Backwards-compatible with existing code that imports FAIR_VALUE_* constants
  or uses PARAM_SPECS + simulate().
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace, asdict
from typing import Dict, Iterable, List, Tuple, Union, Any, Optional
import hashlib
import math
import random
import warnings

import pandas as pd  # לחקר/תיעוד Spec-ים בצורה נוחה

# Optional deps
try:
    import optuna  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("optuna is required for ParamSpec.distribution/suggest") from e

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # YAML helpers will raise if used

Number = Union[int, float]
NumOrStr = Union[Number, str]


# =====================================================================
# ParamSpec — core param description
# =====================================================================


@dataclass(frozen=True, slots=True)
class ParamSpec:
    """
    Specification of a tunable parameter.

    Fields
    ------
    name : str
        Name of the parameter as used in Optuna / config dicts.
    lo, hi : Number | None
        Numeric bounds of the parameter (inclusive). If None for
        categorical parameters, the bounds are implied by `choices`.
    step : Number | None
        Step size for grid-like discretization. For ints, this is the
        grid step; for floats, Optuna will treat it as step size if given.
    log : bool
        If True, use log-uniform sampling for floats/ints where applicable.
    choices : Tuple[NumOrStr, ...] | None
        If not None, the parameter is treated as categorical.
    tags : Tuple[str, ...]
        Lightweight tagging system (e.g. ("signal","volatility")) used
        for filtering / grouping parameters in research.

    Notes
    -----
    - The class provides helper methods for building Optuna distributions
      and suggesting values on a trial.
    - It is **immutable** (frozen=True, slots=True) for safety and speed.
    """

    name: str
    lo: Number | None = None
    hi: Number | None = None
    step: Number | None = None
    log: bool = False
    choices: Tuple[NumOrStr, ...] | None = None
    tags: Tuple[str, ...] = field(default_factory=tuple)

    # ---- type helpers ----
    @property
    def is_categorical(self) -> bool:
        return self.choices is not None

    @property
    def is_int(self) -> bool:
        return (
            not self.is_categorical
            and self.step is not None
            and self.lo is not None
            and self.hi is not None
            and all(float(x).is_integer() for x in (self.lo, self.hi, self.step))
        )

    # ---- Optuna primitives ----
    def _float_kwargs(self) -> Dict[str, Any]:
        if self.log:
            return dict(log=True)
        return dict(step=float(self.step)) if self.step else {}

    def _int_kwargs(self) -> Dict[str, Any]:
        if self.log:
            return dict(log=True)
        return dict(step=int(self.step)) if self.step else {}

    def distribution(self):  # optuna.distributions.BaseDistribution
        """Build matching Optuna distribution (auto-fix hi vs step)."""
        if self.is_categorical:
            return optuna.distributions.CategoricalDistribution(self.choices)  # type: ignore[attr-defined]

        if self.is_int:
            assert self.lo is not None and self.hi is not None and self.step is not None
            lo_i, hi_i, st_i = int(self.lo), int(self.hi), int(self.step)
            # Ensure (hi - lo) is divisible by step
            adj_hi = hi_i - ((hi_i - lo_i) % st_i)
            return optuna.distributions.IntDistribution(lo_i, adj_hi, **self._int_kwargs())  # type: ignore[attr-defined]

        # Float
        assert self.lo is not None and self.hi is not None
        if self.log and self.lo <= 0:
            raise ValueError(f"{self.name}: lo must be >0 for log distribution")
        if not self.log and self.step:
            span = (self.hi - self.lo) / self.step
            if span != int(span):
                # Adjust hi so that (hi - lo) % step == 0
                adj_hi = self.lo + int(span) * self.step
                warnings.warn(
                    f"{self.name}: hi adjusted from {self.hi} to {adj_hi} for step divisibility"
                )
                return optuna.distributions.FloatDistribution(float(self.lo), float(adj_hi), **self._float_kwargs())  # type: ignore[attr-defined]
        return optuna.distributions.FloatDistribution(float(self.lo), float(self.hi), **self._float_kwargs())  # type: ignore[attr-defined]

    def suggest(self, trial):  # optuna.trial.Trial -> NumOrStr
        """Suggest a value for this parameter on an Optuna trial."""
        if self.is_categorical:
            return trial.suggest_categorical(self.name, self.choices)
        if self.is_int:
            assert self.lo is not None and self.hi is not None
            return trial.suggest_int(self.name, int(self.lo), int(self.hi), **self._int_kwargs())
        assert self.lo is not None and self.hi is not None
        return trial.suggest_float(self.name, float(self.lo), float(self.hi), **self._float_kwargs())


# =====================================================================
# PARAM_SPECS — master list of all parameters
# =====================================================================

PARAM_SPECS: List[ParamSpec] = [
    # --- Core signal / mean-reversion parameters ---
    ParamSpec("z_entry",               0.5,  3.0, 0.1, tags=("signal",)),
    ParamSpec("z_exit",                0.1,  1.5, 0.1, tags=("signal",)),
    ParamSpec("lookback",              30,   250, 5,   tags=("window",)),
    ParamSpec("entry_decay",           0.1,  1.0, 0.1, tags=("signal",)),
    ParamSpec("exit_decay",            0.1,  1.0, 0.1, tags=("signal",)),
    ParamSpec("min_trade_distance",       1,  30, 1,   tags=("risk", "execution")),
    ParamSpec("max_trade_duration",       1, 100, 1,   tags=("risk", "execution")),
    ParamSpec("cointegration_deviation",  0.0, 1.0, 0.05, tags=("stat", "cointegration")),
    ParamSpec("spread_mean",          -1.0, 1.0, 0.1, tags=("spread",)),
    ParamSpec("spread_std",            0.01, 5.0, 0.01, log=True, tags=("volatility", "spread")),
    ParamSpec("half_life",                1, 100, 1,      tags=("mean_reversion",)),
    ParamSpec("mean_reversion_speed", 0.01, 1.0, 0.01,   tags=("mean_reversion",)),
    ParamSpec("hurst",                 0.1, 1.0, 0.01,   tags=("fractal", "mean_reversion")),
    ParamSpec("rolling_skewness",     -2.0, 2.0, 0.1,   tags=("spread", "tail")),
    ParamSpec("rolling_kurtosis",      1.0,10.0, 0.1,   tags=("spread", "tail")),
    ParamSpec("percentile_95",         0.9, 1.0, 0.01,  tags=("spread", "percentile")),
    ParamSpec("percentile_05",         0.0, 0.1, 0.01,  tags=("spread", "percentile")),
    ParamSpec("z_momentum",           -1.0, 1.0, 0.01,  tags=("momentum", "signal")),
    ParamSpec("time_in_band",          0.0, 1.0, 0.01,  tags=("behaviour",)),
    ParamSpec("ema_spread",           -1.0, 1.0, 0.01,  tags=("spread",)),
    ParamSpec("cusum_spread",        -10.0,10.0, 0.1,   tags=("spread", "regime")),
    ParamSpec("spread_drawdown",       0.0, 1.0, 0.01,  tags=("risk", "drawdown")),
    ParamSpec("vol_adj_spread",       -1.0, 1.0, 0.01,  tags=("spread", "volatility")),
    ParamSpec("ADF_pval",              0.0, 0.1, 0.01,  tags=("stat", "stationarity")),
    ParamSpec("hedge_ratio",          -2.0, 2.0, 0.1,   tags=("hedge", "correlation")),
    ParamSpec("ECM_coeff",             0.0, 1.0, 0.1,   tags=("cointegration",)),
    ParamSpec("kalman_beta",          -2.0, 2.0, 0.1,   tags=("hedge", "dynamic")),

    ParamSpec("cluster_label", choices=("bull", "bear", "sideways"), tags=("regime",)),
    ParamSpec("copula_tail",           0.0, 1.0, 0.01, tags=("tail", "copula")),
    ParamSpec("ml_pred",               0.0, 1.0, 0.01, tags=("ml", "signal")),
    ParamSpec("meta_score",            0.0, 1.0, 0.01, tags=("meta", "score")),
    ParamSpec("shap_importance",       0.0, 1.0, 0.01, tags=("ml", "explain")),
    ParamSpec("pca_1",                -3.0, 3.0, 0.1,  tags=("pca",)),
    ParamSpec("pca_2",                -3.0, 3.0, 0.1,  tags=("pca",)),
    ParamSpec("pca_3",                -3.0, 3.0, 0.1,  tags=("pca",)),

    # --- 10 PRO additions ---
    ParamSpec("rolling_corr",         -1.0, 1.0, 0.01, tags=("correlation",)),
    ParamSpec("beta_OLS",             -2.0, 2.0, 0.05, tags=("correlation", "hedge")),
    ParamSpec("price_ratio",           0.1, 10.0, 0.01, log=True, tags=("valuation",)),
    ParamSpec("z_score_trend",        -0.5, 0.5, 0.01, tags=("momentum",)),
    ParamSpec("realized_vol_30d",     0.01, 2.0, 0.01, log=True, tags=("volatility",)),
    ParamSpec("implied_realized_spread", -1.0, 1.0, 0.01, tags=("volatility",)),
    ParamSpec("corr_half_life",          1, 100, 1, tags=("correlation", "mean_reversion")),
    ParamSpec("vol_regime", choices=("low", "medium", "high"), tags=("volatility", "regime")),
    ParamSpec("pair_spread_percentile", 0.0, 1.0, 0.01, tags=("relative", "spread")),
    ParamSpec("cross_section_rank",      1, 100, 1, tags=("relative", "rank")),

    # --- 5 "Crazy" ---
    ParamSpec("z_spread_skew",        -3.0, 3.0, 0.05, tags=("tail", "spread")),
    ParamSpec("roll_ic",              -1.0, 1.0, 0.01, tags=("alpha", "correlation")),
    ParamSpec("beta_kalman_vol",       0.01, 2.0, 0.02, log=True, tags=("hedge", "dynamic")),
    ParamSpec("adf_tstat",           -10.0, 0.0, 0.1, tags=("stationarity",)),
    ParamSpec("spread_entropy",        0.0, 1.0, 0.01, tags=("entropy",)),

    # --- 5 "Ultra" + microstructure / macro ---
    ParamSpec("wavelet_energy",        0.0, 1.0, 0.01, tags=("frequency", "volatility")),
    ParamSpec("fractal_dimension",     1.0, 2.0, 0.01, tags=("fractal",)),
    ParamSpec("cointegration_t50",       1, 200, 1, tags=("cointegration",)),
    ParamSpec("vol_of_vol",            0.01, 1.0, 0.01, log=True, tags=("volatility", "second_order")),
    ParamSpec("kalman_q_ratio",        0.01, 10.0, 0.01, log=True, tags=("filter", "dynamic")),
    ParamSpec("order_flow_imbalance",  -5.0, 5.0, 0.1, tags=("flow", "micro")),
    ParamSpec("amihud_illiq",          1e-6, 1e-2, log=True, tags=("liquidity",)),
    ParamSpec("bid_ask_spread_pct",     0.01, 1.0, 0.01, tags=("liquidity", "micro")),
    ParamSpec("intraday_vol_ratio",     0.2, 3.0, 0.05, tags=("volatility", "intraday")),
    ParamSpec("autocorr_lag1",         -1.0, 1.0, 0.05, tags=("signal", "autocorr")),
    ParamSpec("drawdown_half_life",        1, 60, 1, tags=("risk", "drawdown")),
    ParamSpec("pca_residual_vol",       0.01, 1.0, log=True, tags=("volatility", "pca")),
    ParamSpec("beta_market_dynamic",   -2.0, 2.0, 0.05, tags=("hedge", "market")),
    ParamSpec("news_sentiment_score",  -1.0, 1.0, 0.01, tags=("sentiment", "nlp")),
    ParamSpec("macro_regime", choices=("growth", "inflation", "stagflation", "recession"), tags=("macro", "regime")),
    
     # --- Backtest wiring params (single-pair backtest knobs) ---
    # חלון ATR לבניית סטיות תקן / גודל פוזיציה
    ParamSpec("atr_window",              5,     60,   1,    tags=("window", "atr")),

    # Gates איכות / פילטרים לזוג
    ParamSpec("edge_min",                0.0,   2.0,  0.05, tags=("signal", "gate")),
    ParamSpec("atr_max",                 0.0,  10.0,  0.10, tags=("volatility", "gate")),
    ParamSpec("corr_min",               -1.0,   1.0,  0.01, tags=("correlation", "gate")),

    # טווח לבטא – אחר כך נהפוך ל-beta_range=(beta_lo, beta_hi)
    ParamSpec("beta_lo",                -2.0,   2.0,  0.10, tags=("hedge", "gate")),
    ParamSpec("beta_hi",                -2.0,   2.0,  0.10, tags=("hedge", "gate")),

    # Cointegration / Half-life gates
    ParamSpec("coint_pmax",              0.0,   0.20, 0.01, tags=("cointegration", "gate")),
    ParamSpec("half_life_max",           1.0, 100.0, 1.00,  tags=("mean_reversion", "gate")),

    # Execution & costs (משפיעים על הביצוע)
    ParamSpec("notional",             5_000, 200_000, 1_000, log=True, tags=("sizing", "risk")),
    ParamSpec("slippage_bps",            0.0,   5.0,  0.10, tags=("execution", "costs")),
    ParamSpec("slippage_mode", choices=("bps", "atr_frac"), tags=("execution", "costs")),
    ParamSpec("slippage_atr_frac",       0.0,   0.20, 0.005, tags=("execution", "costs", "atr")),
    ParamSpec("transaction_cost_per_trade", 0.0, 10.0, 0.10, tags=("execution", "costs")),
    ParamSpec("bar_lag",                 0,      5,   1,    tags=("execution", "lag")),
    ParamSpec("max_bars_held",           5,    100,   1,    tags=("risk", "stops")),

    # סטופים על Z / Drawdown
    ParamSpec("z_stop",                  0.0,   5.0,  0.10, tags=("risk", "stops")),
    ParamSpec("run_dd_stop_pct",         0.0,   0.50, 0.01, tags=("risk", "stops")),

]


# --- Advanced residual diagnostics & structural break protection ---
PARAM_SPECS += [
    ParamSpec("resid_lb_pval_max",      0.00, 0.25, 0.01, tags=("diagnostics","stationarity")),
    ParamSpec("resid_arch_pval_min",    0.00, 0.25, 0.01, tags=("diagnostics","heteroskedasticity")),
    ParamSpec("chow_pval_min",          0.00, 0.25, 0.01, tags=("stability","breaks")),
    ParamSpec("cusum_break_penalty",    0.00, 2.00, 0.05, tags=("stability","breaks")),
    ParamSpec("johansen_rank_min",         0,    2,   1,   tags=("cointegration","rank")),
]

# --- Execution / costs / impact & portfolio heat ---
PARAM_SPECS += [
    ParamSpec("slippage_model", choices=("fixed","sqrt","prop"), tags=("execution","costs")),
    ParamSpec("slippage_bps",           0.00, 10.00, 0.10, tags=("execution","costs")),
    ParamSpec("impact_k",               0.00, 10.00, 0.10, tags=("execution","impact")),
    ParamSpec("fill_ratio_floor",       0.50,  1.00, 0.01, tags=("execution","liquidity")),
    ParamSpec("heat_cap_pct",           0.01,  0.10, 0.005, tags=("risk","portfolio")),
    ParamSpec("max_consec_losers",         1,    10,   1,   tags=("risk","safety")),
]

# --- Macro gating & global conditions ---
PARAM_SPECS += [
    ParamSpec("gate_vix_q",             0.20, 0.95, 0.01, tags=("macro","gate")),
    ParamSpec("gate_move_q",            0.20, 0.95, 0.01, tags=("macro","gate")),
    ParamSpec("gate_credit_q",          0.20, 0.95, 0.01, tags=("macro","gate")),
]

# --- Risk & validation guards (anti-overfit / robustness) ---
PARAM_SPECS += [
    ParamSpec("stop_band_sigma",        0.25,  3.00, 0.05, tags=("risk","stops")),
    ParamSpec("time_stop_days",            1,    20,   1,  tags=("risk","stops")),
    ParamSpec("anti_cluster_penalty",   0.00,  1.00, 0.05, tags=("portfolio","diversification")),
    ParamSpec("psr_min",                0.00,  0.50, 0.01, tags=("validation","psr")),
    ParamSpec("oos_window_frac",        0.10,  0.60, 0.05, tags=("validation","cv")),
    ParamSpec("cv_kfolds",                 2,     8,   1,  tags=("validation","cv")),
    ParamSpec("bagging_k",                 1,    10,   1,  tags=("validation","ensemble")),
    ParamSpec("bagging_alpha",          0.00,  1.00, 0.05, tags=("validation","ensemble")),
]

# --- HF-grade entry/exit, regime, risk & robustness parameters (new block) ---
PARAM_SPECS += [
    ParamSpec(
        "z_entry_regime_mult",
        0.50,
        2.00,
        0.05,
        tags=("signal", "entry", "regime", "risk"),
    ),
    ParamSpec(
        "z_exit_trailing_frac",
        0.10,
        1.00,
        0.05,
        tags=("signal", "exit", "trailing", "behaviour"),
    ),
    ParamSpec(
        "pnl_stop_loss_sigma",
        0.50,
        5.00,
        0.10,
        tags=("risk", "exit", "stops", "pnl"),
    ),
    ParamSpec(
        "vol_target_smoothing_lambda",
        0.50,
        0.99,
        0.01,
        tags=("risk", "sizing", "smooth", "volatility"),
    ),
    ParamSpec(
        "order_book_depth_pct",
        0.05,
        0.50,
        0.05,
        tags=("liquidity", "execution", "microstructure"),
    ),
    ParamSpec(
        "regime_min_confidence",
        0.00,
        1.00,
        0.05,
        tags=("regime", "validation", "macro", "risk"),
    ),
    ParamSpec(
        "correlation_regime_threshold",
        0.20,
        0.90,
        0.05,
        tags=("correlation", "regime", "risk", "pair_quality"),
    ),
    ParamSpec(
        "min_shap_importance_pct",
        0.00,
        0.50,
        0.01,
        tags=("ml", "explain", "robustness", "feature_selection"),
    ),
    # עוד כמה פרמטרי HF-level כלליים:
    ParamSpec(
        "max_intraday_trades_per_pair",
        0,
        50,
        1,
        tags=("execution", "intraday", "risk"),
    ),
    ParamSpec(
        "max_concurrent_pairs",
        1,
        200,
        1,
        tags=("portfolio", "risk", "heat"),
    ),
    ParamSpec(
        "min_backtest_years",
        0.5,
        10.0,
        0.5,
        tags=("validation", "backtest", "robustness"),
    ),
    ParamSpec(
        "min_oos_years",
        0.5,
        5.0,
        0.5,
        tags=("validation", "oos", "robustness"),
    ),
]

PARAM_INDEX: Dict[str, ParamSpec] = {p.name: p for p in PARAM_SPECS}

# =====================================================================
# Importance meta — colors & weights for ALL params
# =====================================================================

# רמות חשיבות לוגיות
IMPORTANCE_LEVELS = ("high", "medium", "low")

# מיפוי רמה → אימוג'י צבעוני (לתצוגה בדאשבורד / דוחות)
IMPORTANCE_COLOR_BY_LEVEL: Dict[str, str] = {
    "high": "🟥",
    "medium": "🟧",
    "low": "🟨",
}

# משקל בסיס לכל רמה (כאן אתה משחק כשבא לך)
DEFAULT_LEVEL_WEIGHTS: Dict[str, float] = {
    "high": 1.00,   # פרמטרים קריטיים
    "medium": 0.60, # פרמטרים חשובים
    "low": 0.30,    # nice-to-have / מחקר
}

# סדר עדיפויות של רמות (כדי לבחור את "הכי גבוהה" מבין כמה תגיות)
_LEVEL_PRIORITY: Dict[str, int] = {
    "high": 3,
    "medium": 2,
    "low": 1,
}

# מיפוי tag → רמת חשיבות
# כאן אתה שולט בברזים ברמת סל (signal/risk/macro/...)
IMPORTANCE_LEVEL_BY_TAG: Dict[str, str] = {
    # ליבה של סיגנל / ריסק / קורלציה
    "signal": "high",
    "spread": "high",
    "mean_reversion": "high",
    "correlation": "high",
    "cointegration": "high",
    "hedge": "high",
    "risk": "high",
    "stops": "high",
    "portfolio": "high",
    "execution": "high",
    "validation": "high",
    "backtest": "high",
    "oos": "high",
    "gate": "high",  # גייטים קריטיים

    # דברים חשובים אבל לא core של הכניסה/יציאה
    "volatility": "medium",
    "tail": "medium",
    "drawdown": "medium",
    "regime": "medium",
    "macro": "medium",
    "liquidity": "medium",
    "micro": "medium",
    "diagnostics": "medium",
    "stability": "medium",
    "pca": "medium",
    "alpha": "medium",
    "ensemble": "medium",
    "cv": "medium",
    "frequency": "medium",
    "fractal": "medium",
    "entropy": "medium",

    # דברים יותר מחקריים / future alpha
    "ml": "low",
    "explain": "low",
    "nlp": "low",
    "sentiment": "low",
    "_untagged": "low",
}

# אם תרצה לכפות צבע מסוים לפרמטרים בודדים (מעל החוקים של tags)
# למשל: {"z_entry": "high", "news_sentiment_score": "low"}
PARAM_IMPORTANCE_OVERRIDE: Dict[str, str] = {}


def get_param_importance_level(name: str) -> str:
    """
    מחזיר 'high' / 'medium' / 'low' עבור פרמטר אחד.

    לוגיקה:
    -------
    1. אם יש override ב-PARAM_IMPORTANCE_OVERRIDE → משתמש בזה.
    2. אחרת:
       - לוקח את כל ה-tags של הפרמטר.
       - עבור כל tag שיש לו חשיבות במפה IMPORTANCE_LEVEL_BY_TAG,
         בוחר את הרמה הכי גבוהה לפי _LEVEL_PRIORITY.
    3. אם לא נמצא כלום → "low".
    """
    # Override מפורש – מנצח תמיד
    override = PARAM_IMPORTANCE_OVERRIDE.get(name)
    if override in IMPORTANCE_LEVELS:
        return override

    spec = PARAM_INDEX.get(name)
    if spec is None:
        return "low"

    best_level = "low"
    best_priority = _LEVEL_PRIORITY["low"]

    tags = spec.tags or ()
    if not tags:
        # תיוג לא קיים → מתייחסים אליו כ-_untagged אם הוגדר
        level = IMPORTANCE_LEVEL_BY_TAG.get("_untagged")
        if level in IMPORTANCE_LEVELS:
            return level
        return "low"

    for t in tags:
        level = IMPORTANCE_LEVEL_BY_TAG.get(t)
        if level not in IMPORTANCE_LEVELS:
            continue
        prio = _LEVEL_PRIORITY[level]
        if prio > best_priority:
            best_level = level
            best_priority = prio

    return best_level


def get_param_color(name: str) -> str:
    """
    מחזיר אימוג'י צבעוני (🟥/🟧/🟨) לפי רמת החשיבות.
    """
    level = get_param_importance_level(name)
    return IMPORTANCE_COLOR_BY_LEVEL.get(level, "🟨")


def get_param_weight(
    name: str,
    level_weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    מחזיר משקל מספרי לפרמטר לפי רמת החשיבות.

    אפשר להעביר level_weights מותאם אישית, למשל:
        {"high": 1.2, "medium": 0.7, "low": 0.4}
    """
    level = get_param_importance_level(name)
    table = level_weights or DEFAULT_LEVEL_WEIGHTS
    return float(table.get(level, 0.0))


def build_param_importance_table(
    specs: Iterable[ParamSpec] = PARAM_SPECS,
    level_weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    מחזיר DataFrame עם כל הפרמטרים:
        name, tags, importance_level, color, weight

    זה ה-"playground" שלך:
    - לראות את כולם.
    - לשנות IMPORTANCE_LEVEL_BY_TAG / PARAM_IMPORTANCE_OVERRIDE / DEFAULT_LEVEL_WEIGHTS
      ואז לקרוא שוב לפונקציה.
    """
    rows: List[Dict[str, Any]] = []
    for p in specs:
        level = get_param_importance_level(p.name)
        color = IMPORTANCE_COLOR_BY_LEVEL.get(level, "🟨")
        weight = get_param_weight(p.name, level_weights=level_weights)
        rows.append(
            {
                "name": p.name,
                "tags": ",".join(p.tags),
                "importance_level": level,
                "color": color,
                "weight": weight,
            }
        )
    return pd.DataFrame(rows)

# =====================================================================
# Scoring helpers — turn params + importance into a numeric score
# =====================================================================

def _normalize_param_value(name: str, value: NumOrStr) -> float:
    """
    מנרמל ערך פרמטר ל-[0,1] לפי ParamSpec:

    • Numeric עם lo/hi → min-max normalization.
    • Categorical → אינדקס / (n-1) (כלומר בחירה ראשונה=0, אחרונה=1).
    • אם אין spec / אין lo/hi → מחזיר 0.5 (אמצע).

    זה נועד לסקורינג / מחקר, לא ככלי פיננסי קדוש.
    """
    spec = PARAM_INDEX.get(name)
    if spec is None:
        return 0.5

    # קטגוריאלי
    if spec.is_categorical and spec.choices:
        try:
            idx = spec.choices.index(value)  # type: ignore[arg-type]
        except Exception:
            # לא נמצא ב-choices → אמצע
            return 0.5
        n = len(spec.choices)
        if n <= 1:
            return 1.0
        return idx / float(n - 1)

    # נומרי עם lo/hi
    if spec.lo is not None and spec.hi is not None:
        try:
            v = float(value)
        except Exception:
            return 0.5
        lo, hi = float(spec.lo), float(spec.hi)
        if hi <= lo:
            return 0.5
        # Clamp ואז נרמול
        v = max(lo, min(hi, v))
        return (v - lo) / (hi - lo)

    # אין גבולות → אמצע
    return 0.5


def compute_param_contribution(
    name: str,
    value: NumOrStr,
    level_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    מחשב את התרומה של פרמטר יחיד לציון הכולל.

    מחזיר dict עם:
        {
            "name": ...,
            "value": ...,
            "normalized": 0..1,
            "importance_level": "high/medium/low",
            "color": "🟥/🟧/🟨",
            "weight": float,
            "contribution": weight * normalized,
        }
    """
    level = get_param_importance_level(name)
    color = get_param_color(name)
    weight = get_param_weight(name, level_weights=level_weights)
    norm_val = _normalize_param_value(name, value)
    contrib = weight * norm_val

    return {
        "name": name,
        "value": value,
        "normalized": norm_val,
        "importance_level": level,
        "color": color,
        "weight": weight,
        "contribution": contrib,
    }


def score_params_dict(
    params: Dict[str, NumOrStr],
    level_weights: Optional[Dict[str, float]] = None,
    include_only_known: bool = True,
) -> Dict[str, Any]:
    """
    מחשב ציון כולל לסט params לפי:

        score = sum(weight_i * normalized_i) / sum(weight_i)

    Args
    ----
    params : dict
        מילון פרמטרים (למשל זה שמגיע מ-Optuna או מה-optimizer).
    level_weights : אופציונלי
        מיפוי רמה→משקל, אם רוצים לעקוף את DEFAULT_LEVEL_WEIGHTS.
    include_only_known : bool
        אם True → מתחשב רק בפרמטרים שיש להם ParamSpec.
        אם False → נותן 0.5 normalized + low weight למי שלא מוגדר.

    Returns
    -------
    dict:
        {
            "total_score": float,           # ציון ממוצע משוכלל (0..1)
            "raw_score": float,             # סכום contributions (לא מנורמל)
            "total_weight": float,          # סכום משקולות
            "n_params": int,                # כמה פרמטרים נספרו
            "details": List[dict],          # תרומה פר פרמטר (לניתוח/דאשבורד)
        }
    """
    details: List[Dict[str, Any]] = []
    total_contrib = 0.0
    total_weight = 0.0
    n_params = 0

    for name, value in params.items():
        if include_only_known and name not in PARAM_INDEX:
            continue

        info = compute_param_contribution(name, value, level_weights=level_weights)
        w = float(info["weight"])
        if w <= 0.0:
            # פרמטר עם משקל 0 → לא נכנס לציון
            continue

        total_contrib += float(info["contribution"])
        total_weight += w
        n_params += 1
        details.append(info)

    if total_weight <= 0.0:
        total_score = 0.0
    else:
        total_score = total_contrib / total_weight

    return {
        "total_score": float(total_score),
        "raw_score": float(total_contrib),
        "total_weight": float(total_weight),
        "n_params": int(n_params),
        "details": details,
    }

# =====================================================================
# Helper API for research / Optuna
# =====================================================================

def get_param_spec(name: str) -> ParamSpec:
    """Return ParamSpec by name, or raise KeyError if not found."""
    return PARAM_INDEX[name]


def build_distributions(specs: Iterable[ParamSpec] = PARAM_SPECS) -> Dict[str, Any]:
    """Build a dict of Optuna distributions keyed by param name."""
    return {p.name: p.distribution() for p in specs}


def suggest(trial, specs: Iterable[ParamSpec] = PARAM_SPECS) -> Dict[str, NumOrStr]:
    """Suggest a full param dict on an Optuna trial from `specs`."""
    return {p.name: p.suggest(trial) for p in specs}


def filter_by_tags(
    specs: Iterable[ParamSpec] = PARAM_SPECS,
    include: Iterable[str] | None = None,
    exclude: Iterable[str] | None = None,
) -> List[ParamSpec]:
    """Filter ParamSpecs by tags."""
    inc, exc = set(include or ()), set(exclude or ())

    out: List[ParamSpec] = []
    for p in specs:
        tags = set(p.tags)
        if inc and not (inc & tags):
            continue
        if exc and (exc & tags):
            continue
        out.append(p)
    return out


def freeze(specs: Iterable[ParamSpec], **fixed: NumOrStr) -> List[ParamSpec]:
    """
    Freeze some ParamSpecs to exact values.

    Example:
        frozen_specs = freeze(PARAM_SPECS, z_entry=2.0, lookback=120)
    """
    out: List[ParamSpec] = []
    for p in specs:
        if p.name in fixed:
            val = fixed[p.name]
            if p.is_categorical:
                out.append(replace(p, choices=(val,)))
            else:
                out.append(ParamSpec(p.name, val, val, step=0, tags=p.tags))
        else:
            out.append(p)
    return out


def clamp_params_to_specs(params: Dict[str, NumOrStr]) -> Dict[str, NumOrStr]:
    """
    Clamp a params dict so all numeric values fall within [lo, hi] of the spec.

    Non-numeric and categorical parameters are left unchanged.
    """
    out = dict(params)
    for name, val in params.items():
        spec = PARAM_INDEX.get(name)
        if spec is None or spec.is_categorical:
            continue
        try:
            v = float(val)
        except Exception:
            continue
        if spec.lo is not None:
            v = max(v, float(spec.lo))
        if spec.hi is not None:
            v = min(v, float(spec.hi))
        out[name] = type(val)(v) if not isinstance(val, str) else str(v)
    return out


def random_sample_from_specs(
    specs: Iterable[ParamSpec] = PARAM_SPECS,
    rng: random.Random | None = None,
) -> Dict[str, NumOrStr]:
    """Draw a random param dict using basic uniform/log-uniform sampling."""
    rng = rng or random.Random()
    params: Dict[str, NumOrStr] = {}
    for p in specs:
        if p.is_categorical and p.choices:
            params[p.name] = rng.choice(p.choices)
        elif p.is_int and p.lo is not None and p.hi is not None:
            params[p.name] = rng.randrange(int(p.lo), int(p.hi) + 1)
        elif p.lo is not None and p.hi is not None:
            if p.log:
                lo, hi = float(p.lo), float(p.hi)
                lv = rng.uniform(math.log(lo), math.log(hi))
                params[p.name] = math.exp(lv)
            else:
                params[p.name] = rng.uniform(float(p.lo), float(p.hi))
    return params


def dump_to_yaml(specs: Iterable[ParamSpec], path: str) -> None:
    """Dump ParamSpecs to YAML file."""
    if yaml is None:
        raise RuntimeError("pyyaml not installed; can't dump YAML")
    with open(path, "w", encoding="utf8") as f:
        yaml.safe_dump([asdict(p) for p in specs], f, sort_keys=False, allow_unicode=True)


def load_from_yaml(path: str) -> List[ParamSpec]:
    """Load ParamSpecs from YAML file."""
    if yaml is None:
        raise RuntimeError("pyyaml not installed; can't load YAML")
    with open(path, "r", encoding="utf8") as f:
        raw = yaml.safe_load(f) or []
    return [ParamSpec(**d) for d in raw]


def extend_specs(user_specs: Iterable[Dict[str, Any]]) -> None:
    """Append custom ParamSpec dicts at runtime into PARAM_SPECS and PARAM_INDEX."""
    global PARAM_SPECS, PARAM_INDEX
    new_specs = [ParamSpec(**d) for d in user_specs]
    PARAM_SPECS.extend(new_specs)
    PARAM_INDEX.update({p.name: p for p in new_specs})


# =====================================================================
# Integration helpers — regime-aware entry/exit, vol-smoothing & SHAP
# =====================================================================


def compute_regime_adjusted_z_entry(
    base_z_entry: float,
    params: Dict[str, NumOrStr],
    macro_regime_confidence: float,
    pair_corr: float,
) -> float:
    """
    מחשב z_entry מותאם-רג'ים:

    • אם confidence של רג'ים < regime_min_confidence → לא משנים את z_entry.
    • אם הקורלציה של הזוג < correlation_regime_threshold → pair נחשב "stressed"
      ואז מכפילים את z_entry ב-z_entry_regime_mult.
    • אחרת → מחזירים base_z_entry.

    Usage:
        z_entry_eff = compute_regime_adjusted_z_entry(
            params["z_entry"], params, macro_conf, rolling_corr
        )
    """
    regime_min_conf = float(params.get("regime_min_confidence", 0.0) or 0.0)
    mult = float(params.get("z_entry_regime_mult", 1.0) or 1.0)
    corr_thr = float(params.get("correlation_regime_threshold", 0.5) or 0.5)

    if macro_regime_confidence < regime_min_conf:
        return float(base_z_entry)

    if pair_corr < corr_thr:
        return float(base_z_entry) * mult

    return float(base_z_entry)


def compute_exit_and_stop_levels(
    params: Dict[str, NumOrStr],
    spread_std: float,
    base_z_exit: Optional[float] = None,
) -> Dict[str, float]:
    """
    מחשב levels ל-exit ול-stop-loss לפי הפרמטרים:

    • z_exit_trailing_frac — שליטה עד כמה קל/קשה לצאת כשהספרד מתכנס.
    • pnl_stop_loss_sigma — stop-loss בסטיות תקן של spread/PnL.

    Returns:
    --------
        {
            "z_exit_effective": float,   # ערך Z ל-exit אחרי trailing
            "pnl_stop_loss": float,      # spread units (לא NAV)
        }
    """
    if base_z_exit is None:
        base_z_exit = float(params.get("z_exit", 1.0) or 1.0)

    trailing_frac = float(params.get("z_exit_trailing_frac", 0.5) or 0.5)
    stop_sigma = float(params.get("pnl_stop_loss_sigma", 3.0) or 3.0)

    z_exit_eff = abs(base_z_exit) * trailing_frac
    stop_loss_level = stop_sigma * float(spread_std)

    return {
        "z_exit_effective": z_exit_eff,
        "pnl_stop_loss": stop_loss_level,
    }


def smooth_vol_target(
    prev_target_vol: float,
    current_estimated_vol: float,
    params: Dict[str, NumOrStr],
) -> float:
    """
    החלקה על Vol-target / sizing:

        new_target = λ * prev_target + (1-λ) * current_estimated_vol

    • λ = vol_target_smoothing_lambda (0.5–0.99)
    """
    lam = float(params.get("vol_target_smoothing_lambda", 0.9) or 0.9)
    lam = max(0.0, min(0.999, lam))
    return lam * float(prev_target_vol) + (1.0 - lam) * float(current_estimated_vol)


def is_regime_confident_enough(
    macro_regime_confidence: float,
    params: Dict[str, NumOrStr],
) -> bool:
    """
    בודק אם כדאי בכלל להשתמש ב-Regime gating לפי:
    • regime_min_confidence
    """
    regime_min_conf = float(params.get("regime_min_confidence", 0.0) or 0.0)
    return macro_regime_confidence >= regime_min_conf


def filter_features_by_shap_importance(
    shap_values: pd.Series,
    params: Dict[str, NumOrStr],
) -> pd.Series:
    """
    חותך SHAP values לפי min_shap_importance_pct.

    Args:
    -----
    shap_values : pd.Series עם index=feature_name, values=SHAP
    params      : dict עם min_shap_importance_pct

    Returns:
    --------
    pd.Series של רק הפיצ'רים שעברו את הסף.
    """
    if shap_values.empty:
        return shap_values

    min_pct = float(params.get("min_shap_importance_pct", 0.0) or 0.0)
    if min_pct <= 0:
        return shap_values

    abs_vals = shap_values.abs()
    max_val = abs_vals.max()
    if max_val <= 0:
        return shap_values

    threshold = min_pct * max_val
    return shap_values[abs_vals >= threshold]


# =====================================================================
# Extra research helpers — grouping, grids, validation (10 שדרוגים)
# =====================================================================

def group_specs_by_tag(
    specs: Iterable[ParamSpec] = PARAM_SPECS,
) -> Dict[str, List[ParamSpec]]:
    """
    מחזיר dict: tag → רשימת ParamSpecs עם התגית הזו.

    שימושי ל:
    • תיעוד פנימי.
    • בניית search spaces לפי קבוצות (signal/risk/liquidity/...).
    """
    out: Dict[str, List[ParamSpec]] = {}
    for p in specs:
        if not p.tags:
            out.setdefault("_untagged", []).append(p)
        else:
            for t in p.tags:
                out.setdefault(t, []).append(p)
    return out


def specs_to_dataframe(
    specs: Iterable[ParamSpec] = PARAM_SPECS,
) -> pd.DataFrame:
    """
    ממיר רשימת ParamSpecs ל-DataFrame לצורך בדיקה/תיעוד.

    עמודות:
        name, lo, hi, step, log, is_categorical, choices, tags
    """
    rows: List[Dict[str, Any]] = []
    for p in specs:
        rows.append(
            {
                "name": p.name,
                "lo": p.lo,
                "hi": p.hi,
                "step": p.step,
                "log": p.log,
                "is_categorical": p.is_categorical,
                "choices": p.choices,
                "tags": ",".join(p.tags),
            }
        )
    return pd.DataFrame(rows)


def validate_params_dict_against_specs(
    params: Dict[str, NumOrStr],
    specs: Iterable[ParamSpec] = PARAM_SPECS,
) -> List[str]:
    """
    בודק params dict מול ה-ParamSpecs:

    מחזיר רשימת warnings (אם יש):
    • ערכים מחוץ לטווח [lo, hi].
    • פרמטרים לא מוכרים (שאין להם ParamSpec).
    """
    idx = {p.name: p for p in specs}
    warnings_list: List[str] = []

    for name, val in params.items():
        spec = idx.get(name)
        if spec is None:
            warnings_list.append(f"{name}: no ParamSpec defined for this parameter")
            continue
        if spec.is_categorical:
            if spec.choices and val not in spec.choices:
                warnings_list.append(f"{name}: value {val} not in choices {spec.choices}")
        else:
            try:
                v = float(val)
            except Exception:
                warnings_list.append(f"{name}: value {val} not numeric for numeric spec")
                continue
            if spec.lo is not None and v < spec.lo:
                warnings_list.append(f"{name}: {v} < lo={spec.lo}")
            if spec.hi is not None and v > spec.hi:
                warnings_list.append(f"{name}: {v} > hi={spec.hi}")

    return warnings_list


def make_param_grid(
    specs: Iterable[ParamSpec],
    grid_steps_override: Dict[str, int] | None = None,
) -> Dict[str, List[NumOrStr]]:
    """
    בונה "Grid" דיסקרטי קטן לכל פרמטר לצורך grid search / debug:

    • עבור פרמטרים נומריים:
        - אם יש step → משתמשים ב-step.
        - אחרת טווח מחולק לפי grid_steps_override.get(name, 5).
    • עבור categorical → פשוט מחזירים את choices.

    הערה: זה מיועד בעיקר לרשתות קטנות / ניסויים, לא ל-search אמיתי בייצור.
    """
    grid: Dict[str, List[NumOrStr]] = {}
    steps_map = grid_steps_override or {}

    for p in specs:
        if p.is_categorical and p.choices:
            grid[p.name] = list(p.choices)
            continue

        if p.lo is None or p.hi is None:
            continue

        lo, hi = float(p.lo), float(p.hi)
        if p.step is not None:
            step = float(p.step)
            n = int((hi - lo) / step) + 1
            grid[p.name] = [type(lo)(lo + i * step) for i in range(n)]
        else:
            n = steps_map.get(p.name, 5)
            if n <= 1:
                grid[p.name] = [lo]
            else:
                step = (hi - lo) / (n - 1)
                grid[p.name] = [lo + i * step for i in range(n)]

    return grid


def sample_params_with_tag_filter(
    rng: random.Random,
    include_tags: Iterable[str],
    specs: Iterable[ParamSpec] = PARAM_SPECS,
) -> Dict[str, NumOrStr]:
    """
    דגימת params רק מתוך פרמטרים שיש להם לפחות אחת מהתגיות ב-include_tags.

    לדוגמה:
        sample_params_with_tag_filter(rng, ["signal","entry"])
    """
    inc = set(include_tags)
    filtered_specs = [p for p in specs if inc & set(p.tags)]
    return random_sample_from_specs(filtered_specs, rng=rng)


# =====================================================================
# Demo simulator (toy Sharpe proxy)
# =====================================================================

def _stable_seed(payload: Dict[str, Any]) -> int:
    s = repr(sorted(payload.items())).encode("utf8")
    return int.from_bytes(hashlib.sha256(s).digest()[:8], "big")


def simulate(params: Dict[str, NumOrStr]) -> float:
    """
    Toy Sharpe proxy — replace with real backtest.

    Deterministic across runs for the same params.
    """
    random.seed(_stable_seed(params))
    base = random.gauss(1.2, 0.3)
    bonus = math.log1p(float(params.get("lookback", 30))) * 0.05
    penalty = (
        abs(float(params.get("rolling_corr", 0))) * 0.2
        + abs(float(params.get("z_spread_skew", 0))) * 0.05
        + float(params.get("vol_of_vol", 0)) * 0.1
    )
    entropy = float(params.get("spread_entropy", 0)) * 0.1
    return round(base + bonus + entropy - penalty, 4)


# =====================================================================
# Fair Value Defaults — legacy constants
# =====================================================================

FAIR_VALUE_WINDOW = 252
FAIR_VALUE_MIN_OVERLAP = 60
FAIR_VALUE_SECONDARY_WINDOWS = (63, 126)
FAIR_VALUE_LOG_MODE = True

FAIR_VALUE_WINSOR = True
FAIR_VALUE_WINSOR_P = 0.01
FAIR_VALUE_Z_CLIP = (-6.0, 6.0)
FAIR_VALUE_VOL_ADJ = True

FAIR_VALUE_USE_RET_FOR_CORR = True
FAIR_VALUE_USE_RET_FOR_DCOR = True

FAIR_VALUE_ENSEMBLE_MODE = "weighted"        # "none" | "weighted"
FAIR_VALUE_ENSEMBLE_TARGET = "mispricing"    # or "zscore"

FAIR_VALUE_MR_PVALUE = 0.05

# Costs & hysteresis
FAIR_VALUE_COSTS_BPS = 2.0
FAIR_VALUE_SLIPPAGE_BPS = 2.0
FAIR_VALUE_BORROW_BPS = 0.0
FAIR_VALUE_Z_IN = 1.25
FAIR_VALUE_Z_OUT = 0.75

# Sizing & vol targeting
FAIR_VALUE_TARGET_VOL = 0.12
FAIR_VALUE_KELLY_FRAC = 0.5
FAIR_VALUE_MAX_LEV = 5.0

# Evaluation
FAIR_VALUE_PSR_SR_STAR = 0.0
FAIR_VALUE_DSR_TRIALS = 30

# Beta / Kalman
FAIR_VALUE_BETA_MODE = "kalman"      # "static" | "kalman"
FAIR_VALUE_KALMAN_Q = 1e-6
FAIR_VALUE_KALMAN_R = 1e-3

# Portfolio
FAIR_VALUE_RP_METHOD = "hrp"         # "invvol" | "erc" | "hrp"
FAIR_VALUE_COV_METHOD = "ridge"      # "sample" | "ridge" | "lw"
FAIR_VALUE_COV_SHRINK_LAMBDA = 0.2

# Robust volatility & Z models
FAIR_VALUE_VOL_EST = "ewma"        # "std" | "ewma" | "parkinson"
FAIR_VALUE_EWMA_LAMBDA = 0.94      # if VOL_EST == "ewma"
FAIR_VALUE_Z_MODE = "std"          # "std" | "mad" | "huber"
FAIR_VALUE_HUBER_C = 1.345         # only if z_mode == "huber"

# Residual-ADF options
FAIR_VALUE_ADF_MAXLAG = "auto"      # or integer 0..10
FAIR_VALUE_ADF_REG = "c"            # trend: "c" (const) | "ct" (const+trend)
FAIR_VALUE_RESIDUAL_DOMAIN = "auto" # "auto" | "log" | "linear"

# Dynamic thresholds & regime gates
FAIR_VALUE_Z_IN_Q = 0.85            # if set, entry uses |Z| quantile in-window
FAIR_VALUE_Z_OUT_Q = 0.60           # exit quantile; ideally < Z_IN_Q
FAIR_VALUE_VOL_GATE = (0.05, 0.35)  # annualized spread-vol bounds

# Liquidity / execution filters
FAIR_VALUE_MIN_ADV_USD = None       # e.g., 2_000_000 to require min ADV
FAIR_VALUE_MAX_SPREAD_BPS = None    # max bid-ask spread in bps
FAIR_VALUE_REQUIRE_BORROW = False   # require borrow for short legs

# Portfolio constraints & neutrality
FAIR_VALUE_RP_MAX_WEIGHT = 0.10     # per-pair cap (post-normalization)
FAIR_VALUE_SECTOR_CAP = None        # e.g., 0.25 if sector mapping exists
FAIR_VALUE_BETA_NEUTRAL = False     # try to neutralize market beta (future hook)

# Kalman fine-tuning (dynamic alpha/beta)
FAIR_VALUE_KALMAN_ALPHA_BETA_RATIO = 1.0
FAIR_VALUE_KALMAN_P0 = 1e3
FAIR_VALUE_KALMAN_R_DECAY = 0.99

# Ensemble weights override (data-driven)
FAIR_VALUE_ENSEMBLE_WEIGHTS = {"adf": 0.5, "hl": 0.3, "corr": 0.2, "dcor": 0.0}

# Risk limits, logging, reproducibility
FAIR_VALUE_MAX_DD_PAIR = None       # e.g., 0.15 (15%) — per pair (future hook)
FAIR_VALUE_MAX_DD_PORT = None       # e.g., 0.10 (10%) — portfolio (future hook)
FAIR_VALUE_STOP_MODE = "band"       # "band" | "atr" | "time" (future hook)
FAIR_VALUE_LOG_LEVEL = "INFO"       # "INFO" | "DEBUG"
FAIR_VALUE_RNG_SEED = 42            # optional global seed

# --- Advanced residual diagnostics & break detection (hooks) ---
FAIR_VALUE_BREAK_DETECT_MODE   = "none"   # "none"|"cusum"|"chow"|"bai_perron"
FAIR_VALUE_BREAK_PVAL          = 0.05
FAIR_VALUE_LB_PVAL_MAX         = 0.25
FAIR_VALUE_ARCH_PVAL_MIN       = 0.05

# --- Macro gating (quantiles; None = disabled) ---
FAIR_VALUE_GATE_VIX_Q          = None
FAIR_VALUE_GATE_MOVE_Q         = None
FAIR_VALUE_GATE_CREDIT_Q       = None

# --- Execution & heat management ---
FAIR_VALUE_IMPACT_MODEL        = "sqrt"   # "fixed"|"sqrt"|"prop"
FAIR_VALUE_IMPACT_K_BPS        = 2.0
FAIR_VALUE_FILL_RATIO_FLOOR    = 0.80
FAIR_VALUE_HEAT_CAP_PCT        = 0.05
FAIR_VALUE_MAX_CONSEC_LOSERS   = 5


# =====================================================================
# FairValueConfig — structured view over FAIR_VALUE_* flags
# =====================================================================


@dataclass
class FairValueConfig:
    """Structured configuration for the Fair-Value engine."""

    window: int = FAIR_VALUE_WINDOW
    min_overlap: int = FAIR_VALUE_MIN_OVERLAP
    secondary_windows: Tuple[int, int] = FAIR_VALUE_SECONDARY_WINDOWS
    log_mode: bool = FAIR_VALUE_LOG_MODE

    winsor: bool = FAIR_VALUE_WINSOR
    winsor_p: float = FAIR_VALUE_WINSOR_P
    z_clip: Tuple[float, float] = FAIR_VALUE_Z_CLIP
    vol_adj: bool = FAIR_VALUE_VOL_ADJ

    use_ret_for_corr: bool = FAIR_VALUE_USE_RET_FOR_CORR
    use_ret_for_dcor: bool = FAIR_VALUE_USE_RET_FOR_DCOR

    ensemble_mode: str = FAIR_VALUE_ENSEMBLE_MODE
    ensemble_target: str = FAIR_VALUE_ENSEMBLE_TARGET
    mr_pvalue: float = FAIR_VALUE_MR_PVALUE

    # Costs & hysteresis
    costs_bps: float = FAIR_VALUE_COSTS_BPS
    slippage_bps: float = FAIR_VALUE_SLIPPAGE_BPS
    borrow_bps: float = FAIR_VALUE_BORROW_BPS
    z_in: float = FAIR_VALUE_Z_IN
    z_out: float = FAIR_VALUE_Z_OUT

    # Sizing & vol targeting
    target_vol: float = FAIR_VALUE_TARGET_VOL
    kelly_frac: float = FAIR_VALUE_KELLY_FRAC
    max_lev: float = FAIR_VALUE_MAX_LEV

    # Evaluation
    psr_sr_star: float = FAIR_VALUE_PSR_SR_STAR
    dsr_trials: int = FAIR_VALUE_DSR_TRIALS

    # Beta / Kalman
    beta_mode: str = FAIR_VALUE_BETA_MODE
    kalman_q: float = FAIR_VALUE_KALMAN_Q
    kalman_r: float = FAIR_VALUE_KALMAN_R

    # Portfolio
    rp_method: str = FAIR_VALUE_RP_METHOD
    cov_method: str = FAIR_VALUE_COV_METHOD
    cov_shrink_lambda: float = FAIR_VALUE_COV_SHRINK_LAMBDA

    # Robust volatility & Z models
    vol_est: str = FAIR_VALUE_VOL_EST
    ewma_lambda: float = FAIR_VALUE_EWMA_LAMBDA
    z_mode: str = FAIR_VALUE_Z_MODE
    huber_c: float = FAIR_VALUE_HUBER_C

    # Residual-ADF options
    adf_maxlag: Union[str, int] = FAIR_VALUE_ADF_MAXLAG
    adf_reg: str = FAIR_VALUE_ADF_REG
    residual_domain: str = FAIR_VALUE_RESIDUAL_DOMAIN

    # Dynamic thresholds & regime gates
    z_in_q: Optional[float] = FAIR_VALUE_Z_IN_Q
    z_out_q: Optional[float] = FAIR_VALUE_Z_OUT_Q
    vol_gate: Tuple[float, float] = FAIR_VALUE_VOL_GATE

    # Liquidity / execution filters
    min_adv_usd: Optional[float] = FAIR_VALUE_MIN_ADV_USD
    max_spread_bps: Optional[float] = FAIR_VALUE_MAX_SPREAD_BPS
    require_borrow: bool = FAIR_VALUE_REQUIRE_BORROW

    # Portfolio constraints & neutrality
    rp_max_weight: float = FAIR_VALUE_RP_MAX_WEIGHT
    sector_cap: Optional[float] = FAIR_VALUE_SECTOR_CAP
    beta_neutral: bool = FAIR_VALUE_BETA_NEUTRAL

    # Kalman fine-tuning
    kalman_alpha_beta_ratio: float = FAIR_VALUE_KALMAN_ALPHA_BETA_RATIO
    kalman_p0: float = FAIR_VALUE_KALMAN_P0
    kalman_r_decay: float = FAIR_VALUE_KALMAN_R_DECAY

    # Ensemble weights override
    ensemble_weights: Dict[str, float] = field(default_factory=lambda: dict(FAIR_VALUE_ENSEMBLE_WEIGHTS))

    # Risk limits, logging, reproducibility
    max_dd_pair: Optional[float] = FAIR_VALUE_MAX_DD_PAIR
    max_dd_port: Optional[float] = FAIR_VALUE_MAX_DD_PORT
    stop_mode: str = FAIR_VALUE_STOP_MODE
    log_level: str = FAIR_VALUE_LOG_LEVEL
    rng_seed: int = FAIR_VALUE_RNG_SEED

    # Advanced residual diagnostics & break detection
    break_detect_mode: str = FAIR_VALUE_BREAK_DETECT_MODE
    break_pval: float = FAIR_VALUE_BREAK_PVAL
    lb_pval_max: float = FAIR_VALUE_LB_PVAL_MAX
    arch_pval_min: float = FAIR_VALUE_ARCH_PVAL_MIN

    # Macro gating
    gate_vix_q: Optional[float] = FAIR_VALUE_GATE_VIX_Q
    gate_move_q: Optional[float] = FAIR_VALUE_GATE_MOVE_Q
    gate_credit_q: Optional[float] = FAIR_VALUE_GATE_CREDIT_Q

    # Execution & heat management
    impact_model: str = FAIR_VALUE_IMPACT_MODEL
    impact_k_bps: float = FAIR_VALUE_IMPACT_K_BPS
    fill_ratio_floor: float = FAIR_VALUE_FILL_RATIO_FLOOR
    heat_cap_pct: float = FAIR_VALUE_HEAT_CAP_PCT
    max_consec_losers: int = FAIR_VALUE_MAX_CONSEC_LOSERS

    @classmethod
    def from_params(cls, params: Dict[str, Any]) -> "FairValueConfig":
        """Build FairValueConfig from a flat params dict."""
        fields = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        kwargs = {k: v for k, v in params.items() if k in fields}
        return cls(**kwargs)

    def to_flat_params(self, prefix: str | None = None) -> Dict[str, Any]:
        """
        Export config to flat dict, optionally namespaced with prefix.
        """
        base = asdict(self)
        out: Dict[str, Any] = {}
        for k, v in base.items():
            key = k
            if prefix:
                key = f"{prefix}_{k}"
            out[key] = v
        return out


# =====================================================================
# Public exports
# =====================================================================

__all__ = [
    # ParamSpec + helpers
    "ParamSpec",
    "PARAM_SPECS",
    "PARAM_INDEX",
    "get_param_spec",
    "build_distributions",
    "suggest",
    "freeze",
    "filter_by_tags",
    "clamp_params_to_specs",
    "random_sample_from_specs",
    "dump_to_yaml",
    "load_from_yaml",
    "extend_specs",
    "simulate",
    "compute_regime_adjusted_z_entry",
    "compute_exit_and_stop_levels",
    "smooth_vol_target",
    "is_regime_confident_enough",
    "filter_features_by_shap_importance",
    "group_specs_by_tag",
    "specs_to_dataframe",
    "validate_params_dict_against_specs",
    "make_param_grid",
    "sample_params_with_tag_filter",
    # FairValueConfig + legacy constants
    "FairValueConfig",
    "FAIR_VALUE_WINDOW",
    "FAIR_VALUE_MIN_OVERLAP",
    "FAIR_VALUE_SECONDARY_WINDOWS",
    "FAIR_VALUE_LOG_MODE",
    "FAIR_VALUE_WINSOR",
    "FAIR_VALUE_WINSOR_P",
    "FAIR_VALUE_Z_CLIP",
    "FAIR_VALUE_VOL_ADJ",
    "FAIR_VALUE_USE_RET_FOR_CORR",
    "FAIR_VALUE_USE_RET_FOR_DCOR",
    "FAIR_VALUE_ENSEMBLE_MODE",
    "FAIR_VALUE_ENSEMBLE_TARGET",
    "FAIR_VALUE_MR_PVALUE",
    "FAIR_VALUE_COSTS_BPS",
    "FAIR_VALUE_SLIPPAGE_BPS",
    "FAIR_VALUE_BORROW_BPS",
    "FAIR_VALUE_Z_IN",
    "FAIR_VALUE_Z_OUT",
    "FAIR_VALUE_TARGET_VOL",
    "FAIR_VALUE_KELLY_FRAC",
    "FAIR_VALUE_MAX_LEV",
    "FAIR_VALUE_PSR_SR_STAR",
    "FAIR_VALUE_DSR_TRIALS",
    "FAIR_VALUE_BETA_MODE",
    "FAIR_VALUE_KALMAN_Q",
    "FAIR_VALUE_KALMAN_R",
    "FAIR_VALUE_RP_METHOD",
    "FAIR_VALUE_COV_METHOD",
    "FAIR_VALUE_COV_SHRINK_LAMBDA",
    "FAIR_VALUE_VOL_EST",
    "FAIR_VALUE_EWMA_LAMBDA",
    "FAIR_VALUE_Z_MODE",
    "FAIR_VALUE_HUBER_C",
    "FAIR_VALUE_ADF_MAXLAG",
    "FAIR_VALUE_ADF_REG",
    "FAIR_VALUE_RESIDUAL_DOMAIN",
    "FAIR_VALUE_Z_IN_Q",
    "FAIR_VALUE_Z_OUT_Q",
    "FAIR_VALUE_VOL_GATE",
    "FAIR_VALUE_MIN_ADV_USD",
    "FAIR_VALUE_MAX_SPREAD_BPS",
    "FAIR_VALUE_REQUIRE_BORROW",
    "FAIR_VALUE_RP_MAX_WEIGHT",
    "FAIR_VALUE_SECTOR_CAP",
    "FAIR_VALUE_BETA_NEUTRAL",
    "FAIR_VALUE_KALMAN_ALPHA_BETA_RATIO",
    "FAIR_VALUE_KALMAN_P0",
    "FAIR_VALUE_KALMAN_R_DECAY",
    "FAIR_VALUE_ENSEMBLE_WEIGHTS",
    "FAIR_VALUE_MAX_DD_PAIR",
    "FAIR_VALUE_MAX_DD_PORT",
    "FAIR_VALUE_STOP_MODE",
    "FAIR_VALUE_LOG_LEVEL",
    "FAIR_VALUE_RNG_SEED",
    "FAIR_VALUE_BREAK_DETECT_MODE",
    "FAIR_VALUE_BREAK_PVAL",
    "FAIR_VALUE_LB_PVAL_MAX",
    "FAIR_VALUE_ARCH_PVAL_MIN",
    "FAIR_VALUE_GATE_VIX_Q",
    "FAIR_VALUE_GATE_MOVE_Q",
    "FAIR_VALUE_GATE_CREDIT_Q",
    "FAIR_VALUE_IMPACT_MODEL",
    "FAIR_VALUE_IMPACT_K_BPS",
    "FAIR_VALUE_FILL_RATIO_FLOOR",
    "FAIR_VALUE_HEAT_CAP_PCT",
    "FAIR_VALUE_MAX_CONSEC_LOSERS",
]
