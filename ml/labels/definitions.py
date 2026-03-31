# -*- coding: utf-8 -*-
"""
ml/labels/definitions.py — Label Definition Registry
======================================================

Central registry for ALL labels used across the ML platform.
Every label used in training or evaluation must be registered here.
No ad-hoc label computation without a definition.

Organised by LabelFamily:
  A. Reversion success labels
  B. Meta-labeling targets
  C. Relationship persistence labels
  D. Break/instability labels
  E. Holding-time labels
  F. Threshold labels
  G. Portfolio utility labels
  H. Execution viability labels
"""

from __future__ import annotations

from typing import Dict

from ml.contracts import (
    EntityScope,
    LabelDefinition,
    LabelFamily,
)


# ---------------------------------------------------------------------------
# A. Reversion success labels
# ---------------------------------------------------------------------------

REVERSION_5D = LabelDefinition(
    name="reversion_5d",
    family=LabelFamily.REVERSION_SUCCESS,
    description="Did spread revert to 0.5 * |entry_z| within 5 days?",
    horizon_days=5,
    entity_scope=EntityScope.PAIR,
    output_type="binary",
    path_dependent=True,
    censoring_logic="NaN if fewer than 5 days of future data remain",
    cost_treatment="none",
    class_balance_note="Expect ~40-60% positive; varies with entry threshold",
    leakage_risks="Label looks forward 5 days; features must not include any data after decision time",
    valid_use_cases=("meta_labeling", "reversion_signal_quality", "threshold_calibration"),
    version="1.0",
)

REVERSION_10D = LabelDefinition(
    name="reversion_10d",
    family=LabelFamily.REVERSION_SUCCESS,
    description="Did spread revert to 0.5 * |entry_z| within 10 days?",
    horizon_days=10,
    entity_scope=EntityScope.PAIR,
    output_type="binary",
    path_dependent=True,
    censoring_logic="NaN if fewer than 10 days of future data remain",
    cost_treatment="none",
    class_balance_note="Typically higher positive rate than reversion_5d",
    leakage_risks="Label looks forward 10 days; feature cutoff must precede label window",
    valid_use_cases=("meta_labeling", "reversion_signal_quality", "holding_time_estimation"),
    version="1.0",
)

REVERSION_20D = LabelDefinition(
    name="reversion_20d",
    family=LabelFamily.REVERSION_SUCCESS,
    description="Did spread revert to 0.5 * |entry_z| within 20 days?",
    horizon_days=20,
    entity_scope=EntityScope.PAIR,
    output_type="binary",
    path_dependent=True,
    censoring_logic="NaN if fewer than 20 days of future data remain",
    cost_treatment="none",
    class_balance_note="High positive rate for cointegrated pairs; ~65-75% expected",
    leakage_risks="Label looks forward 20 days; embargo >= 20 days required for safety",
    valid_use_cases=("meta_labeling", "pair_selection", "portfolio_ranking"),
    version="1.0",
)

PROFIT_TARGET_5D = LabelDefinition(
    name="profit_target_5d",
    family=LabelFamily.REVERSION_SUCCESS,
    description="Did trade hit 80% of z-score target within 5 days?",
    horizon_days=5,
    entity_scope=EntityScope.PAIR,
    output_type="binary",
    path_dependent=True,
    censoring_logic="NaN if fewer than 5 days of future data remain",
    cost_treatment="none",
    class_balance_note="Stricter than reversion_5d; expect ~30-50% positive",
    leakage_risks="Forward-looking 5 days; must not share normalization with test set",
    valid_use_cases=("meta_labeling", "entry_timing", "threshold_calibration"),
    version="1.0",
)

PROFITABLE_NET_10D = LabelDefinition(
    name="profitable_net_10d",
    family=LabelFamily.REVERSION_SUCCESS,
    description="Was net P&L > 0 after transaction costs within 10 days?",
    horizon_days=10,
    entity_scope=EntityScope.PAIR,
    output_type="binary",
    path_dependent=False,
    censoring_logic="NaN if fewer than 10 days of future data remain",
    cost_treatment="fixed_bps",
    class_balance_note="Cost-adjusted; positive rate lower than gross reversion rate",
    leakage_risks="Includes cost model; cost assumptions must be fixed at train time",
    valid_use_cases=("meta_labeling", "execution_viability", "sizing_assistance"),
    version="1.0",
)

PATH_QUALITY_10D = LabelDefinition(
    name="path_quality_10d",
    family=LabelFamily.REVERSION_SUCCESS,
    description="Was max adverse excursion < 1.5 sigma within 10 days?",
    horizon_days=10,
    entity_scope=EntityScope.PAIR,
    output_type="binary",
    path_dependent=True,
    censoring_logic="NaN if fewer than 10 days of future data remain",
    cost_treatment="none",
    class_balance_note="MAE-based; strongly correlated with regime and entry timing",
    leakage_risks="Intra-horizon path access; do not use for feature normalization",
    valid_use_cases=("meta_labeling", "sizing_assistance", "risk_filtering"),
    version="1.0",
)


# ---------------------------------------------------------------------------
# B. Meta-labeling targets
# ---------------------------------------------------------------------------

META_TAKE_5D = LabelDefinition(
    name="meta_take_5d",
    family=LabelFamily.META_LABEL,
    description="Should the rule-based signal be taken? Binary (1=take, 0=skip) based on reversion_5d.",
    horizon_days=5,
    entity_scope=EntityScope.SIGNAL,
    output_type="binary",
    path_dependent=True,
    censoring_logic="NaN if reversion_5d is NaN",
    cost_treatment="none",
    class_balance_note="Mirrors reversion_5d class balance; oversample minority class",
    leakage_risks="Derived from reversion_5d; same forward-looking risks apply",
    valid_use_cases=("meta_labeling",),
    version="1.0",
)

META_TAKE_10D = LabelDefinition(
    name="meta_take_10d",
    family=LabelFamily.META_LABEL,
    description="Should the rule-based signal be taken? Binary based on reversion_10d.",
    horizon_days=10,
    entity_scope=EntityScope.SIGNAL,
    output_type="binary",
    path_dependent=True,
    censoring_logic="NaN if reversion_10d is NaN",
    cost_treatment="none",
    class_balance_note="Mirrors reversion_10d class balance",
    leakage_risks="Derived from reversion_10d; same forward-looking risks apply",
    valid_use_cases=("meta_labeling",),
    version="1.0",
)

META_SIZE_FULL_10D = LabelDefinition(
    name="meta_size_full_10d",
    family=LabelFamily.META_LABEL,
    description="Was the position worth full size? Binary: profitable_net_10d AND path_quality_10d.",
    horizon_days=10,
    entity_scope=EntityScope.SIGNAL,
    output_type="binary",
    path_dependent=True,
    censoring_logic="NaN if either component is NaN",
    cost_treatment="fixed_bps",
    class_balance_note="AND condition reduces positive rate significantly; expect ~20-40%",
    leakage_risks="Intersection of two forward labels; both have same contamination risks",
    valid_use_cases=("sizing_assistance", "meta_labeling"),
    version="1.0",
)

META_DELAY_RECOMMENDED = LabelDefinition(
    name="meta_delay_recommended",
    family=LabelFamily.META_LABEL,
    description="Should entry be delayed for confirmation? Entry improved by waiting 2 bars.",
    horizon_days=12,
    entity_scope=EntityScope.SIGNAL,
    output_type="binary",
    path_dependent=True,
    censoring_logic="NaN if insufficient future data for 2-bar-delayed comparison",
    cost_treatment="none",
    class_balance_note="Typically low positive rate; class imbalance expected",
    leakage_risks="Compares immediate vs 2-bar-delayed entry; requires 12-day forward window",
    valid_use_cases=("meta_labeling", "entry_timing"),
    version="1.0",
)


# ---------------------------------------------------------------------------
# C. Relationship persistence labels
# ---------------------------------------------------------------------------

PERSISTENCE_30D = LabelDefinition(
    name="persistence_30d",
    family=LabelFamily.RELATIONSHIP_PERSISTENCE,
    description="Did ADF p-value remain < 0.10 over next 30 days?",
    horizon_days=30,
    entity_scope=EntityScope.PAIR,
    output_type="binary",
    path_dependent=True,
    censoring_logic="NaN if fewer than 30 days of future data remain",
    cost_treatment="none",
    class_balance_note="Strongly correlated with current cointegration strength",
    leakage_risks="Requires rolling ADF computation 30 days forward; do not reuse test ADF values",
    valid_use_cases=("pair_selection", "candidate_ranking", "regime_classification"),
    version="1.0",
)

PERSISTENCE_60D = LabelDefinition(
    name="persistence_60d",
    family=LabelFamily.RELATIONSHIP_PERSISTENCE,
    description="Did ADF p-value remain < 0.10 over next 60 days?",
    horizon_days=60,
    entity_scope=EntityScope.PAIR,
    output_type="binary",
    path_dependent=True,
    censoring_logic="NaN if fewer than 60 days of future data remain",
    cost_treatment="none",
    class_balance_note="Lower positive rate than persistence_30d; use with long lookback features",
    leakage_risks="60-day forward window; train_end must have at least 60 days of buffer",
    valid_use_cases=("pair_selection", "portfolio_ranking"),
    version="1.0",
)

HALF_LIFE_STABLE_30D = LabelDefinition(
    name="half_life_stable_30d",
    family=LabelFamily.RELATIONSHIP_PERSISTENCE,
    description="Did half-life stay within 2x of current value for 30 days?",
    horizon_days=30,
    entity_scope=EntityScope.PAIR,
    output_type="binary",
    path_dependent=True,
    censoring_logic="NaN if fewer than 30 days of future data remain",
    cost_treatment="none",
    class_balance_note="Expect ~50-70% positive for well-validated pairs",
    leakage_risks="Forward half-life computation; current half-life must be feature, not label",
    valid_use_cases=("pair_selection", "threshold_calibration", "holding_time_estimation"),
    version="1.0",
)

COINT_STRENGTH_30D = LabelDefinition(
    name="coint_strength_30d",
    family=LabelFamily.RELATIONSHIP_PERSISTENCE,
    description="Continuous — mean cointegration strength over next 30 days.",
    horizon_days=30,
    entity_scope=EntityScope.PAIR,
    output_type="continuous",
    path_dependent=False,
    censoring_logic="NaN if fewer than 30 days of future data remain",
    cost_treatment="none",
    class_balance_note="Continuous output; use regression or ranking models",
    leakage_risks="Mean of forward rolling cointegration scores; test normalization separately",
    valid_use_cases=("pair_selection", "candidate_ranking", "portfolio_ranking"),
    version="1.0",
)


# ---------------------------------------------------------------------------
# D. Break/instability labels
# ---------------------------------------------------------------------------

BREAK_5D = LabelDefinition(
    name="break_5d",
    family=LabelFamily.BREAK_INSTABILITY,
    description="Did a structural break occur within 5 days? (CUSUM-based)",
    horizon_days=5,
    entity_scope=EntityScope.PAIR,
    output_type="binary",
    path_dependent=True,
    censoring_logic="NaN if fewer than 5 days of future data remain",
    cost_treatment="none",
    class_balance_note="Rare event; expect <10% positive — use class weighting",
    leakage_risks="CUSUM computed on forward spread; requires future spread data",
    valid_use_cases=("break_detection", "risk_filtering", "regime_classification"),
    version="1.0",
)

BREAK_20D = LabelDefinition(
    name="break_20d",
    family=LabelFamily.BREAK_INSTABILITY,
    description="Did a structural break occur within 20 days? (CUSUM-based)",
    horizon_days=20,
    entity_scope=EntityScope.PAIR,
    output_type="binary",
    path_dependent=True,
    censoring_logic="NaN if fewer than 20 days of future data remain",
    cost_treatment="none",
    class_balance_note="Less rare than break_5d; still imbalanced — use precision/recall over accuracy",
    leakage_risks="CUSUM computed on 20-day forward spread window",
    valid_use_cases=("break_detection", "risk_filtering"),
    version="1.0",
)

HR_INSTABILITY_10D = LabelDefinition(
    name="hr_instability_10d",
    family=LabelFamily.BREAK_INSTABILITY,
    description="Did hedge ratio change by >30% within 10 days?",
    horizon_days=10,
    entity_scope=EntityScope.PAIR,
    output_type="binary",
    path_dependent=False,
    censoring_logic="NaN if fewer than 10 days of future data remain",
    cost_treatment="none",
    class_balance_note="Frequency varies with lookback; rolling OLS more volatile than Kalman",
    leakage_risks="Forward hedge ratio requires future price data; share no normalizer with test",
    valid_use_cases=("break_detection", "risk_filtering", "regime_classification"),
    version="1.0",
)

RESIDUAL_VAR_SPIKE_10D = LabelDefinition(
    name="residual_var_spike_10d",
    family=LabelFamily.BREAK_INSTABILITY,
    description="Did residual variance increase >2x within 10 days?",
    horizon_days=10,
    entity_scope=EntityScope.PAIR,
    output_type="binary",
    path_dependent=False,
    censoring_logic="NaN if fewer than 10 days of future data remain",
    cost_treatment="none",
    class_balance_note="Spiky; correlated with market stress events and regime transitions",
    leakage_risks="Forward variance computation; variance normalizer must be fit on train only",
    valid_use_cases=("break_detection", "anomaly_detection", "regime_classification"),
    version="1.0",
)


# ---------------------------------------------------------------------------
# E. Holding-time labels
# ---------------------------------------------------------------------------

TIME_TO_MEAN_REVERSION = LabelDefinition(
    name="time_to_mean_reversion",
    family=LabelFamily.HOLDING_TIME,
    description="Days until z crosses 0.5 (continuous, censored at 60d).",
    horizon_days=60,
    entity_scope=EntityScope.PAIR,
    output_type="continuous",
    path_dependent=True,
    censoring_logic="Censored at 60 days if no crossing observed; use survival models",
    cost_treatment="none",
    class_balance_note="Right-censored continuous; use AFT or Cox models for training",
    leakage_risks="Scans forward up to 60 days; train_end must have 60-day buffer",
    valid_use_cases=("holding_time_estimation", "exit_timing", "position_sizing"),
    version="1.0",
)

TIME_TO_STOP = LabelDefinition(
    name="time_to_stop",
    family=LabelFamily.HOLDING_TIME,
    description="Days until z hits stop level (continuous, censored at 60d).",
    horizon_days=60,
    entity_scope=EntityScope.PAIR,
    output_type="continuous",
    path_dependent=True,
    censoring_logic="Censored at 60 days if stop not hit; stop level is a runtime parameter",
    cost_treatment="none",
    class_balance_note="Right-censored continuous; low values indicate fragile trades",
    leakage_risks="Forward path scan for stop crossing; do not use in feature pipelines",
    valid_use_cases=("holding_time_estimation", "risk_filtering", "sizing_assistance"),
    version="1.0",
)

HOLDING_TIME_ACTUAL = LabelDefinition(
    name="holding_time_actual",
    family=LabelFamily.HOLDING_TIME,
    description="Actual holding days (realized, continuous).",
    horizon_days=60,
    entity_scope=EntityScope.PAIR,
    output_type="continuous",
    path_dependent=True,
    censoring_logic="For open positions at dataset end: use max_horizon as censored value",
    cost_treatment="none",
    class_balance_note="Continuous; skewed right; log-transform before regression",
    leakage_risks="Realized holding time requires trade outcome; available only in backtests",
    valid_use_cases=("holding_time_estimation", "execution_viability"),
    version="1.0",
)


# ---------------------------------------------------------------------------
# F. Threshold labels
# ---------------------------------------------------------------------------

OPTIMAL_ENTRY_Z_BUCKET = LabelDefinition(
    name="optimal_entry_z_bucket",
    family=LabelFamily.THRESHOLD_BAND,
    description="Which entry z-band performed best? Ordinal: 0=2.0-2.5, 1=2.5-3.0, 2=3.0+.",
    horizon_days=20,
    entity_scope=EntityScope.PAIR,
    output_type="ordinal",
    path_dependent=True,
    censoring_logic="NaN if fewer than 20 days of future data remain",
    cost_treatment="fixed_bps",
    class_balance_note="Ordinal with 3 classes; class 0 typically most frequent",
    leakage_risks="Performance comparison across z-bands uses forward returns",
    valid_use_cases=("threshold_recommendation",),
    version="1.0",
)


# ---------------------------------------------------------------------------
# G. Portfolio utility labels
# ---------------------------------------------------------------------------

PORTFOLIO_ADDITIVE_10D = LabelDefinition(
    name="portfolio_additive_10d",
    family=LabelFamily.PORTFOLIO_UTILITY,
    description="Did this trade improve portfolio Sharpe? Binary.",
    horizon_days=10,
    entity_scope=EntityScope.PORTFOLIO,
    output_type="binary",
    path_dependent=False,
    censoring_logic="NaN if portfolio snapshot unavailable for the period",
    cost_treatment="fixed_bps",
    class_balance_note="Portfolio-context label; requires portfolio state at decision time",
    leakage_risks="Portfolio Sharpe computation uses forward returns of all portfolio legs",
    valid_use_cases=("portfolio_ranking", "sizing_assistance"),
    version="1.0",
)

CAPITAL_EFFICIENCY_10D = LabelDefinition(
    name="capital_efficiency_10d",
    family=LabelFamily.PORTFOLIO_UTILITY,
    description="Return per unit of margin used (continuous).",
    horizon_days=10,
    entity_scope=EntityScope.PAIR,
    output_type="continuous",
    path_dependent=False,
    censoring_logic="NaN if fewer than 10 days of future data remain",
    cost_treatment="fixed_bps",
    class_balance_note="Continuous; can be negative; winsorize at [1st, 99th] percentile",
    leakage_risks="Margin requirement must be computed at train time; forward returns are labels",
    valid_use_cases=("portfolio_ranking", "sizing_assistance"),
    version="1.0",
)


# ---------------------------------------------------------------------------
# H. Execution viability labels
# ---------------------------------------------------------------------------

SURVIVES_COSTS_10D = LabelDefinition(
    name="survives_costs_10d",
    family=LabelFamily.EXECUTION_VIABILITY,
    description="Was gross P&L > 30bps transaction costs? Binary.",
    horizon_days=10,
    entity_scope=EntityScope.EXECUTION,
    output_type="binary",
    path_dependent=False,
    censoring_logic="NaN if fewer than 10 days of future data remain",
    cost_treatment="fixed_bps",
    class_balance_note="Cost threshold is fixed at 30bps; adjust for different cost regimes",
    leakage_risks="Gross P&L is forward-looking; cost assumption must be fixed pre-train",
    valid_use_cases=("execution_viability", "meta_labeling"),
    version="1.0",
)

FILL_QUALITY_PROXY = LabelDefinition(
    name="fill_quality_proxy",
    family=LabelFamily.EXECUTION_VIABILITY,
    description="Did entry z deviate from signal z by <0.3? Binary.",
    horizon_days=1,
    entity_scope=EntityScope.EXECUTION,
    output_type="binary",
    path_dependent=False,
    censoring_logic="NaN if fill data unavailable",
    cost_treatment="none",
    class_balance_note="Depends heavily on liquidity; expect high positive rate in normal markets",
    leakage_risks="Minimal leakage risk; 1-day horizon; intraday fill vs EOD signal comparison",
    valid_use_cases=("execution_viability",),
    version="1.0",
)


# ---------------------------------------------------------------------------
# Aggregate registry
# ---------------------------------------------------------------------------

LABEL_REGISTRY: Dict[str, LabelDefinition] = {
    defn.name: defn
    for defn in [
        # A. Reversion success
        REVERSION_5D,
        REVERSION_10D,
        REVERSION_20D,
        PROFIT_TARGET_5D,
        PROFITABLE_NET_10D,
        PATH_QUALITY_10D,
        # B. Meta-labeling
        META_TAKE_5D,
        META_TAKE_10D,
        META_SIZE_FULL_10D,
        META_DELAY_RECOMMENDED,
        # C. Relationship persistence
        PERSISTENCE_30D,
        PERSISTENCE_60D,
        HALF_LIFE_STABLE_30D,
        COINT_STRENGTH_30D,
        # D. Break/instability
        BREAK_5D,
        BREAK_20D,
        HR_INSTABILITY_10D,
        RESIDUAL_VAR_SPIKE_10D,
        # E. Holding-time
        TIME_TO_MEAN_REVERSION,
        TIME_TO_STOP,
        HOLDING_TIME_ACTUAL,
        # F. Threshold
        OPTIMAL_ENTRY_Z_BUCKET,
        # G. Portfolio utility
        PORTFOLIO_ADDITIVE_10D,
        CAPITAL_EFFICIENCY_10D,
        # H. Execution viability
        SURVIVES_COSTS_10D,
        FILL_QUALITY_PROXY,
    ]
}
