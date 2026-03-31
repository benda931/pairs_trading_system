# -*- coding: utf-8 -*-
"""
ml/features/definitions.py — Feature Definition Registry
=========================================================

Central registry for ALL features used across the ML platform.
Every feature used in training, inference, or model evaluation must
be registered here.  No ad-hoc feature computation without a definition.

Organised by EntityScope:
  A. Instrument-level features
  B. Pair-level features
  C. Regime-level features
  D. Signal-context features
  E. Portfolio-context features
  F. Execution-context features

After all individual definitions, FEATURE_REGISTRY aggregates them by
name and FEATURE_GROUPS maps logical use-case groups to sets of names.
"""

from __future__ import annotations

from typing import Dict

from ml.contracts import (
    EntityScope,
    FeatureCategory,
    FeatureDefinition,
    FeatureGroup,
)

# ---------------------------------------------------------------------------
# A. Instrument-level features
# ---------------------------------------------------------------------------

INST_RET_1D = FeatureDefinition(
    name="inst_ret_1d",
    description="1-day log return for the instrument",
    entity_scope=EntityScope.INSTRUMENT,
    category=FeatureCategory.RETURN,
    required_inputs=("px",),
    lookback_days=2,
    version="1.0",
    notes="log(px[t] / px[t-1])",
)

INST_RET_5D = FeatureDefinition(
    name="inst_ret_5d",
    description="5-day cumulative log return for the instrument",
    entity_scope=EntityScope.INSTRUMENT,
    category=FeatureCategory.RETURN,
    required_inputs=("px",),
    lookback_days=6,
    version="1.0",
    notes="log(px[t] / px[t-5])",
)

INST_RET_20D = FeatureDefinition(
    name="inst_ret_20d",
    description="20-day cumulative log return for the instrument",
    entity_scope=EntityScope.INSTRUMENT,
    category=FeatureCategory.RETURN,
    required_inputs=("px",),
    lookback_days=21,
    version="1.0",
    notes="log(px[t] / px[t-20])",
)

INST_VOL_20D = FeatureDefinition(
    name="inst_vol_20d",
    description="20-day realized volatility (annualized, std of log returns × sqrt(252))",
    entity_scope=EntityScope.INSTRUMENT,
    category=FeatureCategory.VOLATILITY,
    required_inputs=("px",),
    lookback_days=21,
    version="1.0",
    notes="std(log_returns[-20:]) * sqrt(252)",
)

INST_VOL_60D = FeatureDefinition(
    name="inst_vol_60d",
    description="60-day realized volatility (annualized)",
    entity_scope=EntityScope.INSTRUMENT,
    category=FeatureCategory.VOLATILITY,
    required_inputs=("px",),
    lookback_days=61,
    version="1.0",
    notes="std(log_returns[-60:]) * sqrt(252)",
)

INST_VOL_RATIO = FeatureDefinition(
    name="inst_vol_ratio",
    description="Ratio of 20-day vol to 60-day vol — detects vol regime transitions",
    entity_scope=EntityScope.INSTRUMENT,
    category=FeatureCategory.VOLATILITY,
    required_inputs=("px",),
    lookback_days=61,
    version="1.0",
    notes="inst_vol_20d / inst_vol_60d; > 1 means recent vol elevated",
)

INST_MOM_20D = FeatureDefinition(
    name="inst_mom_20d",
    description="20-day price momentum (normalised by 20-day vol)",
    entity_scope=EntityScope.INSTRUMENT,
    category=FeatureCategory.MOMENTUM,
    required_inputs=("px",),
    lookback_days=21,
    version="1.0",
    notes="inst_ret_20d / inst_vol_20d; vol-normalised so comparable across instruments",
)

INST_MOM_60D = FeatureDefinition(
    name="inst_mom_60d",
    description="60-day price momentum (normalised by 60-day vol)",
    entity_scope=EntityScope.INSTRUMENT,
    category=FeatureCategory.MOMENTUM,
    required_inputs=("px",),
    lookback_days=61,
    version="1.0",
    notes="inst_ret_60d / inst_vol_60d",
)

INST_GAP_5D = FeatureDefinition(
    name="inst_gap_5d",
    description="5-day high/low price range normalised by 5-day average price",
    entity_scope=EntityScope.INSTRUMENT,
    category=FeatureCategory.VOLATILITY,
    required_inputs=("px",),
    lookback_days=6,
    version="1.0",
    notes="(max(px[-5:]) - min(px[-5:])) / mean(px[-5:]); proxy for intra-week range",
)

INST_VOLUME_RATIO = FeatureDefinition(
    name="inst_volume_ratio",
    description="Recent 5-day average volume divided by 20-day average volume",
    entity_scope=EntityScope.INSTRUMENT,
    category=FeatureCategory.LIQUIDITY,
    required_inputs=("volume",),
    lookback_days=21,
    version="1.0",
    notes="mean(volume[-5:]) / mean(volume[-20:]); > 1 means elevated recent activity",
)

# ---------------------------------------------------------------------------
# B. Pair-level features
# ---------------------------------------------------------------------------

PAIR_Z = FeatureDefinition(
    name="pair_z",
    description="Current z-score of the spread (signed)",
    entity_scope=EntityScope.PAIR,
    category=FeatureCategory.Z_SCORE,
    required_inputs=("z",),
    lookback_days=1,
    version="1.0",
    notes="Raw z-score value at as_of; positive = spread above mean",
)

PAIR_Z_ABS = FeatureDefinition(
    name="pair_z_abs",
    description="Absolute value of current z-score",
    entity_scope=EntityScope.PAIR,
    category=FeatureCategory.Z_SCORE,
    required_inputs=("z",),
    lookback_days=1,
    version="1.0",
    notes="|z[t]|; direction-agnostic measure of spread dislocation",
)

PAIR_Z_MEAN_5D = FeatureDefinition(
    name="pair_z_mean_5d",
    description="5-day rolling mean of z-score",
    entity_scope=EntityScope.PAIR,
    category=FeatureCategory.Z_SCORE,
    required_inputs=("z",),
    lookback_days=5,
    version="1.0",
    notes="mean(z[-5:]); sustained dislocation vs single-bar spike",
)

PAIR_Z_MEAN_20D = FeatureDefinition(
    name="pair_z_mean_20d",
    description="20-day rolling mean of z-score",
    entity_scope=EntityScope.PAIR,
    category=FeatureCategory.Z_SCORE,
    required_inputs=("z",),
    lookback_days=20,
    version="1.0",
    notes="mean(z[-20:]); medium-term z-score level",
)

PAIR_Z_STD_20D = FeatureDefinition(
    name="pair_z_std_20d",
    description="20-day rolling standard deviation of z-score",
    entity_scope=EntityScope.PAIR,
    category=FeatureCategory.Z_SCORE,
    required_inputs=("z",),
    lookback_days=20,
    version="1.0",
    notes="std(z[-20:]); low std = well-behaved mean reversion",
)

PAIR_Z_MOM_5D = FeatureDefinition(
    name="pair_z_mom_5d",
    description="5-day z-score momentum (z[t] - z[t-5])",
    entity_scope=EntityScope.PAIR,
    category=FeatureCategory.Z_SCORE,
    required_inputs=("z",),
    lookback_days=6,
    version="1.0",
    notes="Positive = diverging, negative = converging over 5 days",
)

PAIR_Z_MOM_20D = FeatureDefinition(
    name="pair_z_mom_20d",
    description="20-day z-score momentum (z[t] - z[t-20])",
    entity_scope=EntityScope.PAIR,
    category=FeatureCategory.Z_SCORE,
    required_inputs=("z",),
    lookback_days=21,
    version="1.0",
    notes="Positive = slow divergence trend; negative = recovering",
)

PAIR_Z_AR1 = FeatureDefinition(
    name="pair_z_ar1",
    description="AR(1) coefficient of z-score estimated over 20-day window",
    entity_scope=EntityScope.PAIR,
    category=FeatureCategory.Z_SCORE,
    required_inputs=("z",),
    lookback_days=22,
    version="1.0",
    notes="OLS of z[t] ~ z[t-1] over [-20:]; coefficient close to 0 = fast mean reversion",
)

PAIR_Z_CROSS_ZERO_20D = FeatureDefinition(
    name="pair_z_cross_zero_20d",
    description="Zero-crossing frequency of z-score over 20-day window",
    entity_scope=EntityScope.PAIR,
    category=FeatureCategory.Z_SCORE,
    required_inputs=("z",),
    lookback_days=20,
    version="1.0",
    notes="count(sign changes in z[-20:]) / 20; high value = active mean reversion",
)

PAIR_CORR_20D = FeatureDefinition(
    name="pair_corr_20d",
    description="20-day rolling Pearson correlation of log returns",
    entity_scope=EntityScope.PAIR,
    category=FeatureCategory.CORRELATION,
    required_inputs=("px", "py"),
    lookback_days=21,
    version="1.0",
    notes="corr(log_ret_x[-20:], log_ret_y[-20:])",
)

PAIR_CORR_60D = FeatureDefinition(
    name="pair_corr_60d",
    description="60-day rolling Pearson correlation of log returns",
    entity_scope=EntityScope.PAIR,
    category=FeatureCategory.CORRELATION,
    required_inputs=("px", "py"),
    lookback_days=61,
    version="1.0",
    notes="corr(log_ret_x[-60:], log_ret_y[-60:])",
)

PAIR_CORR_TREND = FeatureDefinition(
    name="pair_corr_trend",
    description="Correlation trend: corr_20d minus corr_60d",
    entity_scope=EntityScope.PAIR,
    category=FeatureCategory.CORRELATION,
    required_inputs=("px", "py"),
    lookback_days=61,
    version="1.0",
    notes="Negative = recent correlation degrading vs long-term baseline; potential break signal",
)

PAIR_HEDGE_RATIO = FeatureDefinition(
    name="pair_hedge_ratio",
    description="Current OLS hedge ratio (beta of py on px)",
    entity_scope=EntityScope.PAIR,
    category=FeatureCategory.STABILITY,
    required_inputs=("px", "py"),
    lookback_days=60,
    version="1.0",
    notes="beta from OLS(py ~ px) over lookback window; should be stable for tradable pair",
)

PAIR_HR_STABILITY = FeatureDefinition(
    name="pair_hr_stability",
    description="Rolling hedge ratio stability: std(hr) / |mean(hr)| over 60d",
    entity_scope=EntityScope.PAIR,
    category=FeatureCategory.STABILITY,
    required_inputs=("px", "py"),
    lookback_days=60,
    version="1.0",
    notes="Lower = more stable hedge ratio; > 0.30 is a soft warning per ValidationThresholds",
)

PAIR_HALF_LIFE = FeatureDefinition(
    name="pair_half_life",
    description="Current half-life estimate of spread mean reversion (days)",
    entity_scope=EntityScope.PAIR,
    category=FeatureCategory.STABILITY,
    required_inputs=("z",),
    lookback_days=60,
    version="1.0",
    notes="Estimated via AR(1): hl = -log(2) / log(|ar1_coef|); valid range 2–120 days",
)

PAIR_HALF_LIFE_RATIO = FeatureDefinition(
    name="pair_half_life_ratio",
    description="Half-life normalised by 30 days",
    entity_scope=EntityScope.PAIR,
    category=FeatureCategory.STABILITY,
    required_inputs=("z",),
    lookback_days=60,
    version="1.0",
    notes="half_life / 30; < 1 = fast reverter, > 1 = slow reverter",
)

PAIR_SPREAD_VOL_20D = FeatureDefinition(
    name="pair_spread_vol_20d",
    description="20-day volatility of spread residuals (std of daily z changes)",
    entity_scope=EntityScope.PAIR,
    category=FeatureCategory.SPREAD,
    required_inputs=("z",),
    lookback_days=21,
    version="1.0",
    notes="std(diff(z[-20:])); measures how noisy the spread is",
)

PAIR_SPREAD_VOL_RATIO = FeatureDefinition(
    name="pair_spread_vol_ratio",
    description="Recent spread vol (20d) divided by long-term spread vol (60d)",
    entity_scope=EntityScope.PAIR,
    category=FeatureCategory.SPREAD,
    required_inputs=("z",),
    lookback_days=61,
    version="1.0",
    notes="pair_spread_vol_20d / std(diff(z[-60:])); > 1 = noisier than usual",
)

PAIR_RESIDUAL_SKEW = FeatureDefinition(
    name="pair_residual_skew",
    description="20-day skewness of the z-score distribution",
    entity_scope=EntityScope.PAIR,
    category=FeatureCategory.SPREAD,
    required_inputs=("z",),
    lookback_days=20,
    version="1.0",
    notes="skew(z[-20:]); non-zero skew implies asymmetric mean reversion risk",
)

PAIR_RESIDUAL_KURT = FeatureDefinition(
    name="pair_residual_kurt",
    description="20-day excess kurtosis of the z-score distribution",
    entity_scope=EntityScope.PAIR,
    category=FeatureCategory.SPREAD,
    required_inputs=("z",),
    lookback_days=20,
    version="1.0",
    notes="kurt(z[-20:]); high kurtosis = fat tails, larger stop-loss risk",
)

PAIR_BB_POS_20D = FeatureDefinition(
    name="pair_bb_pos_20d",
    description="Bollinger Band position of z-score (20-day window)",
    entity_scope=EntityScope.PAIR,
    category=FeatureCategory.Z_SCORE,
    required_inputs=("z",),
    lookback_days=20,
    version="1.0",
    notes="(z[t] - mean(z[-20:])) / std(z[-20:]); equivalent to z-of-z",
)

PAIR_BB_POS_60D = FeatureDefinition(
    name="pair_bb_pos_60d",
    description="Bollinger Band position of z-score (60-day window)",
    entity_scope=EntityScope.PAIR,
    category=FeatureCategory.Z_SCORE,
    required_inputs=("z",),
    lookback_days=60,
    version="1.0",
    notes="(z[t] - mean(z[-60:])) / std(z[-60:]); longer-window normalisation",
)

PAIR_DIV_SPEED = FeatureDefinition(
    name="pair_div_speed",
    description="Rate of z-score divergence (first difference of z, last 3 bars avg)",
    entity_scope=EntityScope.PAIR,
    category=FeatureCategory.Z_SCORE,
    required_inputs=("z",),
    lookback_days=5,
    version="1.0",
    notes="mean(diff(z[-3:])); positive = spread still widening; negative = converging",
)

PAIR_ENTRY_ATTEMPTS_5D = FeatureDefinition(
    name="pair_entry_attempts_5d",
    description="Number of times |z| crossed entry threshold (2.0) in prior 5 days",
    entity_scope=EntityScope.PAIR,
    category=FeatureCategory.Z_SCORE,
    required_inputs=("z",),
    lookback_days=6,
    version="1.0",
    notes="Counts upward crossings of |z| >= 2.0 in last 5 bars; many crossings = choppy spread",
)

PAIR_REL_MOMENTUM = FeatureDefinition(
    name="pair_rel_momentum",
    description="20-day relative momentum of the spread (spread ret vs own vol)",
    entity_scope=EntityScope.PAIR,
    category=FeatureCategory.MOMENTUM,
    required_inputs=("z",),
    lookback_days=21,
    version="1.0",
    notes="(z[t] - z[t-20]) / pair_spread_vol_20d; vol-normalised spread momentum",
)

# ---------------------------------------------------------------------------
# C. Regime-level features
# ---------------------------------------------------------------------------

REG_VOL_REGIME = FeatureDefinition(
    name="reg_vol_regime",
    description="Volatility regime indicator: recent vol / long-run vol",
    entity_scope=EntityScope.REGIME,
    category=FeatureCategory.REGIME,
    required_inputs=("px", "py"),
    lookback_days=120,
    version="1.0",
    notes="mean([vol_20d_x, vol_20d_y]) / mean([vol_120d_x, vol_120d_y]); > 1.5 = stressed",
)

REG_VOL_OF_VOL = FeatureDefinition(
    name="reg_vol_of_vol",
    description="Volatility-of-volatility: rolling std of 20-day vol estimates",
    entity_scope=EntityScope.REGIME,
    category=FeatureCategory.REGIME,
    required_inputs=("px", "py"),
    lookback_days=60,
    version="1.0",
    notes="std of rolling 20d vol window over 60-bar lookback; high = unstable vol environment",
)

REG_TREND_SLOPE = FeatureDefinition(
    name="reg_trend_slope",
    description="20-day normalised trend slope of spread (OLS slope / spread vol)",
    entity_scope=EntityScope.REGIME,
    category=FeatureCategory.REGIME,
    required_inputs=("z",),
    lookback_days=21,
    version="1.0",
    notes="OLS(z[-20:] ~ time) slope normalised by std(z[-20:]); near zero = no trend",
)

REG_MEAN_REVERSION_QUALITY = FeatureDefinition(
    name="reg_mean_reversion_quality",
    description="AR(1) coefficient of spread — close to 0 (or negative) = mean-reverting",
    entity_scope=EntityScope.REGIME,
    category=FeatureCategory.REGIME,
    required_inputs=("z",),
    lookback_days=60,
    version="1.0",
    notes="OLS of z[t] ~ z[t-1] over 60d; > 0.9 implies near random walk",
)

REG_SPREAD_PERSISTENCE = FeatureDefinition(
    name="reg_spread_persistence",
    description="ACF at lag 1 for the spread series (60-day window)",
    entity_scope=EntityScope.REGIME,
    category=FeatureCategory.REGIME,
    required_inputs=("z",),
    lookback_days=62,
    version="1.0",
    notes="Pearson corr(z[-60:], z[-60:].shift(1)); close to 0 = low persistence = good reverter",
)

REG_BREAK_INDICATOR = FeatureDefinition(
    name="reg_break_indicator",
    description="CUSUM-based structural break indicator (rolling)",
    entity_scope=EntityScope.REGIME,
    category=FeatureCategory.REGIME,
    required_inputs=("z",),
    lookback_days=60,
    version="1.0",
    notes="Normalised CUSUM statistic on z[-60:]; > 1.0 suggests potential structural break",
)

REG_MARKET_STRESS = FeatureDefinition(
    name="reg_market_stress",
    description="Cross-sectional volatility dispersion proxy (rolling)",
    entity_scope=EntityScope.REGIME,
    category=FeatureCategory.REGIME,
    required_inputs=("px", "py"),
    lookback_days=21,
    version="1.0",
    notes="std([vol_20d_x, vol_20d_y]); high dispersion = instruments decoupling",
)

REG_LIQUIDITY_PROXY = FeatureDefinition(
    name="reg_liquidity_proxy",
    description="Relative volume vs 60-day average (both legs combined)",
    entity_scope=EntityScope.REGIME,
    category=FeatureCategory.LIQUIDITY,
    required_inputs=("volume_x", "volume_y"),
    lookback_days=61,
    version="1.0",
    notes="mean(volume_x[-5:] + volume_y[-5:]) / mean(volume_x[-60:] + volume_y[-60:])",
)

# ---------------------------------------------------------------------------
# D. Signal-context features
# ---------------------------------------------------------------------------

SIG_Z_AT_ENTRY = FeatureDefinition(
    name="sig_z_at_entry",
    description="Z-score value at the moment the entry threshold was breached",
    entity_scope=EntityScope.SIGNAL,
    category=FeatureCategory.SIGNAL_CONTEXT,
    required_inputs=("z", "entry_timestamp"),
    lookback_days=1,
    version="1.0",
    notes="z[entry_timestamp]; captures how extreme the entry dislocation was",
)

SIG_TIME_SINCE_BREACH = FeatureDefinition(
    name="sig_time_since_breach",
    description="Days elapsed since the entry threshold was first breached",
    entity_scope=EntityScope.SIGNAL,
    category=FeatureCategory.SIGNAL_CONTEXT,
    required_inputs=("entry_timestamp",),
    lookback_days=1,
    version="1.0",
    notes="(as_of - entry_timestamp).days; long time since breach = stale signal",
)

SIG_TIME_SINCE_PRIOR_CROSS = FeatureDefinition(
    name="sig_time_since_prior_cross",
    description="Days since the last z-score zero crossing before the current signal",
    entity_scope=EntityScope.SIGNAL,
    category=FeatureCategory.SIGNAL_CONTEXT,
    required_inputs=("z",),
    lookback_days=60,
    version="1.0",
    notes="Short time = recently mean reverting; long time = possibly trending",
)

SIG_CONFIRMATION_BARS = FeatureDefinition(
    name="sig_confirmation_bars",
    description="Number of consecutive bars where |z| >= entry threshold",
    entity_scope=EntityScope.SIGNAL,
    category=FeatureCategory.SIGNAL_CONTEXT,
    required_inputs=("z", "entry_timestamp"),
    lookback_days=10,
    version="1.0",
    notes="More bars = stronger persistence of dislocation before signal taken",
)

SIG_PRIOR_FAILED_5D = FeatureDefinition(
    name="sig_prior_failed_5d",
    description="Binary: any failed/stopped trades on this pair in the prior 5 days",
    entity_scope=EntityScope.SIGNAL,
    category=FeatureCategory.SIGNAL_CONTEXT,
    required_inputs=("trade_outcomes",),
    lookback_days=5,
    version="1.0",
    notes="1 if any loss/stop in prior 5d; recent failures suggest adverse regime",
)

SIG_ENTRY_Z_PERCENTILE = FeatureDefinition(
    name="sig_entry_z_percentile",
    description="Percentile of entry z-score vs 60-day z-score distribution",
    entity_scope=EntityScope.SIGNAL,
    category=FeatureCategory.SIGNAL_CONTEXT,
    required_inputs=("z", "entry_timestamp"),
    lookback_days=61,
    version="1.0",
    notes="percentile_rank(z_at_entry, z[-60:]); 0.95+ = extreme historical dislocation",
)

SIG_SPREAD_ACCEL = FeatureDefinition(
    name="sig_spread_accel",
    description="Spread acceleration at entry: rate of change of z (second difference)",
    entity_scope=EntityScope.SIGNAL,
    category=FeatureCategory.SIGNAL_CONTEXT,
    required_inputs=("z", "entry_timestamp"),
    lookback_days=5,
    version="1.0",
    notes="diff(diff(z[-5:])) near entry; positive = spread still accelerating away",
)

# ---------------------------------------------------------------------------
# E. Portfolio-context features
# ---------------------------------------------------------------------------

PORT_ACTIVE_PAIRS = FeatureDefinition(
    name="port_active_pairs",
    description="Count of currently active (open) pair positions in the portfolio",
    entity_scope=EntityScope.PORTFOLIO,
    category=FeatureCategory.PORTFOLIO_CONTEXT,
    required_inputs=("active_positions",),
    lookback_days=1,
    version="1.0",
    notes="Integer count; used to assess portfolio congestion",
)

PORT_CLUSTER_CROWDING = FeatureDefinition(
    name="port_cluster_crowding",
    description="Count of active pairs in the same cluster as the candidate pair",
    entity_scope=EntityScope.PORTFOLIO,
    category=FeatureCategory.PORTFOLIO_CONTEXT,
    required_inputs=("active_positions", "cluster_map"),
    lookback_days=1,
    version="1.0",
    notes="0 = no cluster crowding; high count = diversification penalty applies",
)

PORT_SHARED_LEG_COUNT = FeatureDefinition(
    name="port_shared_leg_count",
    description="Number of currently active pairs sharing at least one instrument leg",
    entity_scope=EntityScope.PORTFOLIO,
    category=FeatureCategory.PORTFOLIO_CONTEXT,
    required_inputs=("active_positions", "pair_id"),
    lookback_days=1,
    version="1.0",
    notes="Shared legs create hidden correlation; high count = exposure management needed",
)

PORT_PORTFOLIO_HEAT = FeatureDefinition(
    name="port_portfolio_heat",
    description="Current portfolio drawdown heat level on a 0–5 scale",
    entity_scope=EntityScope.PORTFOLIO,
    category=FeatureCategory.PORTFOLIO_CONTEXT,
    required_inputs=("drawdown_heat",),
    lookback_days=1,
    normalise=False,
    version="1.0",
    notes="0 = cool, 5 = kill switch territory; feeds directly into size scalar",
)

PORT_CAPITAL_UTILIZATION = FeatureDefinition(
    name="port_capital_utilization",
    description="Fraction of total capital currently deployed across all positions",
    entity_scope=EntityScope.PORTFOLIO,
    category=FeatureCategory.PORTFOLIO_CONTEXT,
    required_inputs=("deployed_capital", "total_capital"),
    lookback_days=1,
    version="1.0",
    notes="[0, 1]; high utilisation limits new position funding",
)

PORT_OPPORTUNITY_DENSITY = FeatureDefinition(
    name="port_opportunity_density",
    description="Ratio of available signals to remaining capital capacity",
    entity_scope=EntityScope.PORTFOLIO,
    category=FeatureCategory.PORTFOLIO_CONTEXT,
    required_inputs=("signal_count", "capital_capacity"),
    lookback_days=1,
    version="1.0",
    notes="signal_count / max(capital_capacity, 1); high density = capital constraint active",
)

PORT_AVG_ACTIVE_HALF_LIFE = FeatureDefinition(
    name="port_avg_active_half_life",
    description="Average half-life (days) of currently active positions",
    entity_scope=EntityScope.PORTFOLIO,
    category=FeatureCategory.PORTFOLIO_CONTEXT,
    required_inputs=("active_positions", "half_life_map"),
    lookback_days=1,
    version="1.0",
    notes="Portfolio-level mean; high avg HL = capital tied up in slow reverters",
)

# ---------------------------------------------------------------------------
# F. Execution-context features
# ---------------------------------------------------------------------------

EXEC_SIZE_VS_ADV = FeatureDefinition(
    name="exec_size_vs_adv",
    description="Planned position size relative to average daily volume (both legs combined)",
    entity_scope=EntityScope.EXECUTION,
    category=FeatureCategory.EXECUTION_CONTEXT,
    required_inputs=("position_size", "volume_x", "volume_y"),
    lookback_days=21,
    version="1.0",
    notes="position_notional / (adv_x + adv_y); > 0.05 = meaningful market impact expected",
)

EXEC_SPREAD_PROXY = FeatureDefinition(
    name="exec_spread_proxy",
    description="Estimated bid-ask spread proxy using Garman-Klass range estimator",
    entity_scope=EntityScope.EXECUTION,
    category=FeatureCategory.EXECUTION_CONTEXT,
    required_inputs=("px", "py"),
    lookback_days=5,
    version="1.0",
    notes="(high - low) / close averaged over 5d for each leg; combined proxy for transaction costs",
)

EXEC_FILL_COMPLEXITY = FeatureDefinition(
    name="exec_fill_complexity",
    description="Execution complexity proxy: number of legs × total position size",
    entity_scope=EntityScope.EXECUTION,
    category=FeatureCategory.EXECUTION_CONTEXT,
    required_inputs=("position_size", "n_legs"),
    lookback_days=1,
    version="1.0",
    notes="Always 2 for standard pair; higher if multi-leg construct; × size captures drag",
)

EXEC_SESSION_QUALITY = FeatureDefinition(
    name="exec_session_quality",
    description="Binary: 1 if signal is in primary trading session hours, 0 otherwise",
    entity_scope=EntityScope.EXECUTION,
    category=FeatureCategory.EXECUTION_CONTEXT,
    required_inputs=("timestamp",),
    lookback_days=1,
    normalise=False,
    version="1.0",
    notes="Primary session = NYSE 09:30–16:00 ET; off-hours = wider spreads, lower liquidity",
)

# ---------------------------------------------------------------------------
# FEATURE_REGISTRY — aggregated dict of all feature definitions
# ---------------------------------------------------------------------------

FEATURE_REGISTRY: Dict[str, FeatureDefinition] = {
    # Instrument-level
    "inst_ret_1d": INST_RET_1D,
    "inst_ret_5d": INST_RET_5D,
    "inst_ret_20d": INST_RET_20D,
    "inst_vol_20d": INST_VOL_20D,
    "inst_vol_60d": INST_VOL_60D,
    "inst_vol_ratio": INST_VOL_RATIO,
    "inst_mom_20d": INST_MOM_20D,
    "inst_mom_60d": INST_MOM_60D,
    "inst_gap_5d": INST_GAP_5D,
    "inst_volume_ratio": INST_VOLUME_RATIO,
    # Pair-level
    "pair_z": PAIR_Z,
    "pair_z_abs": PAIR_Z_ABS,
    "pair_z_mean_5d": PAIR_Z_MEAN_5D,
    "pair_z_mean_20d": PAIR_Z_MEAN_20D,
    "pair_z_std_20d": PAIR_Z_STD_20D,
    "pair_z_mom_5d": PAIR_Z_MOM_5D,
    "pair_z_mom_20d": PAIR_Z_MOM_20D,
    "pair_z_ar1": PAIR_Z_AR1,
    "pair_z_cross_zero_20d": PAIR_Z_CROSS_ZERO_20D,
    "pair_corr_20d": PAIR_CORR_20D,
    "pair_corr_60d": PAIR_CORR_60D,
    "pair_corr_trend": PAIR_CORR_TREND,
    "pair_hedge_ratio": PAIR_HEDGE_RATIO,
    "pair_hr_stability": PAIR_HR_STABILITY,
    "pair_half_life": PAIR_HALF_LIFE,
    "pair_half_life_ratio": PAIR_HALF_LIFE_RATIO,
    "pair_spread_vol_20d": PAIR_SPREAD_VOL_20D,
    "pair_spread_vol_ratio": PAIR_SPREAD_VOL_RATIO,
    "pair_residual_skew": PAIR_RESIDUAL_SKEW,
    "pair_residual_kurt": PAIR_RESIDUAL_KURT,
    "pair_bb_pos_20d": PAIR_BB_POS_20D,
    "pair_bb_pos_60d": PAIR_BB_POS_60D,
    "pair_div_speed": PAIR_DIV_SPEED,
    "pair_entry_attempts_5d": PAIR_ENTRY_ATTEMPTS_5D,
    "pair_rel_momentum": PAIR_REL_MOMENTUM,
    # Regime-level
    "reg_vol_regime": REG_VOL_REGIME,
    "reg_vol_of_vol": REG_VOL_OF_VOL,
    "reg_trend_slope": REG_TREND_SLOPE,
    "reg_mean_reversion_quality": REG_MEAN_REVERSION_QUALITY,
    "reg_spread_persistence": REG_SPREAD_PERSISTENCE,
    "reg_break_indicator": REG_BREAK_INDICATOR,
    "reg_market_stress": REG_MARKET_STRESS,
    "reg_liquidity_proxy": REG_LIQUIDITY_PROXY,
    # Signal-context
    "sig_z_at_entry": SIG_Z_AT_ENTRY,
    "sig_time_since_breach": SIG_TIME_SINCE_BREACH,
    "sig_time_since_prior_cross": SIG_TIME_SINCE_PRIOR_CROSS,
    "sig_confirmation_bars": SIG_CONFIRMATION_BARS,
    "sig_prior_failed_5d": SIG_PRIOR_FAILED_5D,
    "sig_entry_z_percentile": SIG_ENTRY_Z_PERCENTILE,
    "sig_spread_accel": SIG_SPREAD_ACCEL,
    # Portfolio-context
    "port_active_pairs": PORT_ACTIVE_PAIRS,
    "port_cluster_crowding": PORT_CLUSTER_CROWDING,
    "port_shared_leg_count": PORT_SHARED_LEG_COUNT,
    "port_portfolio_heat": PORT_PORTFOLIO_HEAT,
    "port_capital_utilization": PORT_CAPITAL_UTILIZATION,
    "port_opportunity_density": PORT_OPPORTUNITY_DENSITY,
    "port_avg_active_half_life": PORT_AVG_ACTIVE_HALF_LIFE,
    # Execution-context
    "exec_size_vs_adv": EXEC_SIZE_VS_ADV,
    "exec_spread_proxy": EXEC_SPREAD_PROXY,
    "exec_fill_complexity": EXEC_FILL_COMPLEXITY,
    "exec_session_quality": EXEC_SESSION_QUALITY,
}

# ---------------------------------------------------------------------------
# FEATURE_GROUPS — logical groupings for ML task wiring
# ---------------------------------------------------------------------------

FEATURE_GROUPS: Dict[str, FeatureGroup] = {
    "pair_zscore_features": FeatureGroup(
        name="pair_zscore_features",
        description="All z-score derived features for a pair (level, momentum, distribution)",
        feature_names=(
            "pair_z",
            "pair_z_abs",
            "pair_z_mean_5d",
            "pair_z_mean_20d",
            "pair_z_std_20d",
            "pair_z_mom_5d",
            "pair_z_mom_20d",
            "pair_z_ar1",
            "pair_z_cross_zero_20d",
            "pair_bb_pos_20d",
            "pair_bb_pos_60d",
            "pair_div_speed",
            "pair_entry_attempts_5d",
            "pair_rel_momentum",
            "pair_residual_skew",
            "pair_residual_kurt",
        ),
        entity_scope=EntityScope.PAIR,
        version="1.0",
    ),
    "pair_correlation_features": FeatureGroup(
        name="pair_correlation_features",
        description="Return correlation features for a pair over multiple horizons",
        feature_names=(
            "pair_corr_20d",
            "pair_corr_60d",
            "pair_corr_trend",
        ),
        entity_scope=EntityScope.PAIR,
        version="1.0",
    ),
    "pair_stability_features": FeatureGroup(
        name="pair_stability_features",
        description="Hedge ratio stability, half-life, and spread volatility features",
        feature_names=(
            "pair_hedge_ratio",
            "pair_hr_stability",
            "pair_half_life",
            "pair_half_life_ratio",
            "pair_spread_vol_20d",
            "pair_spread_vol_ratio",
        ),
        entity_scope=EntityScope.PAIR,
        version="1.0",
    ),
    "instrument_return_features": FeatureGroup(
        name="instrument_return_features",
        description="Single-instrument return, volatility, momentum, and liquidity features",
        feature_names=(
            "inst_ret_1d",
            "inst_ret_5d",
            "inst_ret_20d",
            "inst_vol_20d",
            "inst_vol_60d",
            "inst_vol_ratio",
            "inst_mom_20d",
            "inst_mom_60d",
            "inst_gap_5d",
            "inst_volume_ratio",
        ),
        entity_scope=EntityScope.INSTRUMENT,
        version="1.0",
    ),
    "regime_features": FeatureGroup(
        name="regime_features",
        description="All regime-level features for market environment classification",
        feature_names=(
            "reg_vol_regime",
            "reg_vol_of_vol",
            "reg_trend_slope",
            "reg_mean_reversion_quality",
            "reg_spread_persistence",
            "reg_break_indicator",
            "reg_market_stress",
            "reg_liquidity_proxy",
        ),
        entity_scope=EntityScope.REGIME,
        version="1.0",
    ),
    "signal_context_features": FeatureGroup(
        name="signal_context_features",
        description="Signal entry context features for meta-labeling and entry quality assessment",
        feature_names=(
            "sig_z_at_entry",
            "sig_time_since_breach",
            "sig_time_since_prior_cross",
            "sig_confirmation_bars",
            "sig_prior_failed_5d",
            "sig_entry_z_percentile",
            "sig_spread_accel",
        ),
        entity_scope=EntityScope.SIGNAL,
        version="1.0",
    ),
    "portfolio_context_features": FeatureGroup(
        name="portfolio_context_features",
        description="Portfolio-level context features for position sizing and risk management",
        feature_names=(
            "port_active_pairs",
            "port_cluster_crowding",
            "port_shared_leg_count",
            "port_portfolio_heat",
            "port_capital_utilization",
            "port_opportunity_density",
            "port_avg_active_half_life",
        ),
        entity_scope=EntityScope.PORTFOLIO,
        version="1.0",
    ),
    "execution_context_features": FeatureGroup(
        name="execution_context_features",
        description="Execution quality and market impact features for viability assessment",
        feature_names=(
            "exec_size_vs_adv",
            "exec_spread_proxy",
            "exec_fill_complexity",
            "exec_session_quality",
        ),
        entity_scope=EntityScope.EXECUTION,
        version="1.0",
    ),
    "meta_label_features": FeatureGroup(
        name="meta_label_features",
        description=(
            "Combined feature set for meta-labeling: pair z-score features + "
            "signal context; used to predict whether a primary signal will succeed"
        ),
        feature_names=(
            # z-score features
            "pair_z",
            "pair_z_abs",
            "pair_z_mean_5d",
            "pair_z_mean_20d",
            "pair_z_std_20d",
            "pair_z_mom_5d",
            "pair_z_ar1",
            "pair_z_cross_zero_20d",
            "pair_bb_pos_20d",
            "pair_div_speed",
            "pair_residual_skew",
            "pair_residual_kurt",
            # signal context
            "sig_z_at_entry",
            "sig_time_since_breach",
            "sig_time_since_prior_cross",
            "sig_confirmation_bars",
            "sig_prior_failed_5d",
            "sig_entry_z_percentile",
            "sig_spread_accel",
        ),
        entity_scope=EntityScope.SIGNAL,
        version="1.0",
    ),
    "regime_classification_features": FeatureGroup(
        name="regime_classification_features",
        description=(
            "Combined feature set for regime classification: pair z-score "
            "+ regime + correlation features"
        ),
        feature_names=(
            # z-score
            "pair_z_ar1",
            "pair_z_cross_zero_20d",
            "pair_z_std_20d",
            "pair_bb_pos_20d",
            "pair_bb_pos_60d",
            "pair_rel_momentum",
            # regime
            "reg_vol_regime",
            "reg_vol_of_vol",
            "reg_trend_slope",
            "reg_mean_reversion_quality",
            "reg_spread_persistence",
            "reg_market_stress",
            # correlation
            "pair_corr_20d",
            "pair_corr_60d",
            "pair_corr_trend",
        ),
        entity_scope=EntityScope.REGIME,
        version="1.0",
    ),
    "break_detection_features": FeatureGroup(
        name="break_detection_features",
        description=(
            "Features for structural break detection: stability + regime + spread vol"
        ),
        feature_names=(
            # stability
            "pair_hedge_ratio",
            "pair_hr_stability",
            "pair_half_life",
            "pair_half_life_ratio",
            "pair_spread_vol_20d",
            "pair_spread_vol_ratio",
            # regime
            "reg_break_indicator",
            "reg_vol_regime",
            "reg_trend_slope",
            "reg_mean_reversion_quality",
            "reg_spread_persistence",
            # spread vol / distribution
            "pair_residual_skew",
            "pair_residual_kurt",
            "pair_z_std_20d",
        ),
        entity_scope=EntityScope.REGIME,
        version="1.0",
    ),
    "candidate_ranking_features": FeatureGroup(
        name="candidate_ranking_features",
        description=(
            "Feature set for pair candidate ranking: pair-level features + "
            "per-instrument return/vol features"
        ),
        feature_names=(
            # pair
            "pair_z_ar1",
            "pair_z_cross_zero_20d",
            "pair_corr_20d",
            "pair_corr_60d",
            "pair_corr_trend",
            "pair_hedge_ratio",
            "pair_hr_stability",
            "pair_half_life",
            "pair_spread_vol_20d",
            "pair_spread_vol_ratio",
            # instrument
            "inst_vol_20d",
            "inst_vol_60d",
            "inst_vol_ratio",
            "inst_mom_20d",
            "inst_mom_60d",
            "inst_volume_ratio",
        ),
        entity_scope=EntityScope.PAIR,
        version="1.0",
    ),
}
