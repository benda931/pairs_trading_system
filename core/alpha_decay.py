# -*- coding: utf-8 -*-
"""
core/alpha_decay.py — Pair Strategy Health Monitor & Edge Decay Tracker
========================================================================

Monitors the health of each active pair strategy across 5 dimensions:
1. Alpha Integrity - Is the edge still present? (Sharpe trend, z-score quality)
2. Cointegration Integrity - Is the pair still cointegrated? (ADF, Johansen)
3. Regime Sensitivity - Is performance regime-dependent?
4. Execution Integrity - Are costs eroding the edge? (slippage, turnover)
5. Stability - Is performance stable over time? (rolling Sharpe variance)

Health States:
- HEALTHY: All dimensions within bounds
- WATCH: One dimension degrading
- EARLY_DECAY: Edge showing signs of erosion
- REGIME_SUPPRESSED: Strategy OK but regime unfavorable
- STRUCTURAL_DECAY: Fundamental relationship breaking down
- RETIREMENT_CANDIDATE: Multiple failures, recommend removal

Adapted from srv_quant's alpha_decay agent for pair-specific use cases.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================================
# Health States
# ============================================================================

HEALTH_STATES = [
    "HEALTHY", "WATCH", "EARLY_DECAY", "REGIME_SUPPRESSED",
    "STRUCTURAL_DECAY", "RETIREMENT_CANDIDATE",
]

HEALTH_COLORS = {
    "HEALTHY": "green",
    "WATCH": "yellow",
    "EARLY_DECAY": "orange",
    "REGIME_SUPPRESSED": "blue",
    "STRUCTURAL_DECAY": "red",
    "RETIREMENT_CANDIDATE": "darkred",
}


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class DimensionScore:
    """Score for a single health dimension."""
    name: str
    score: float  # [0, 1] — higher is healthier
    weight: float
    status: str  # "ok" / "warning" / "critical"
    details: str = ""


@dataclass
class PairHealthReport:
    """Complete health assessment for one pair strategy."""
    pair_label: str
    sym_x: str
    sym_y: str
    assessed_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Dimension scores
    alpha_integrity: float = 0.0
    cointegration_integrity: float = 0.0
    regime_sensitivity: float = 0.0
    execution_integrity: float = 0.0
    stability: float = 0.0

    # Overall
    health_score: float = 0.0
    health_state: str = "HEALTHY"
    root_causes: List[str] = field(default_factory=list)
    recommended_action: str = "HOLD"  # HOLD / REDUCE / PAUSE / RETIRE

    # Dimension details
    dimensions: List[DimensionScore] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pair_label": self.pair_label,
            "sym_x": self.sym_x,
            "sym_y": self.sym_y,
            "assessed_at": self.assessed_at,
            "health_score": round(self.health_score, 3),
            "health_state": self.health_state,
            "root_causes": self.root_causes,
            "recommended_action": self.recommended_action,
            "alpha_integrity": round(self.alpha_integrity, 3),
            "cointegration_integrity": round(self.cointegration_integrity, 3),
            "regime_sensitivity": round(self.regime_sensitivity, 3),
            "execution_integrity": round(self.execution_integrity, 3),
            "stability": round(self.stability, 3),
        }


# ============================================================================
# Dimension Assessors
# ============================================================================

def _assess_alpha_integrity(
    returns: pd.Series,
    lookback: int = 60,
) -> DimensionScore:
    """Is the edge still present? Check rolling Sharpe trend."""
    if len(returns) < lookback:
        return DimensionScore("alpha_integrity", 0.5, 0.30, "warning", "insufficient data")

    recent = returns.iloc[-lookback:]
    sharpe = float(recent.mean() / max(recent.std(), 1e-10) * np.sqrt(252))

    if sharpe > 0.5:
        score, status = 1.0, "ok"
    elif sharpe > 0.0:
        score, status = 0.5 + sharpe, "ok"
    elif sharpe > -0.5:
        score, status = max(0.1, 0.5 + sharpe), "warning"
    else:
        score, status = 0.0, "critical"

    return DimensionScore(
        "alpha_integrity", score, 0.30, status,
        f"rolling_sharpe={sharpe:.2f}",
    )


def _assess_cointegration(
    spread: pd.Series,
    lookback: int = 252,
) -> DimensionScore:
    """Is the pair still cointegrated?"""
    recent = spread.dropna().iloc[-lookback:]
    if len(recent) < 60:
        return DimensionScore("cointegration_integrity", 0.5, 0.25, "warning", "insufficient data")

    try:
        from statsmodels.tsa.stattools import adfuller
        result = adfuller(recent.values, maxlag=12, autolag="AIC")
        p_value = float(result[1])
    except Exception:
        p_value = 0.5

    if p_value < 0.05:
        score, status = 1.0, "ok"
    elif p_value < 0.10:
        score, status = 0.6, "ok"
    elif p_value < 0.20:
        score, status = 0.3, "warning"
    else:
        score, status = 0.1, "critical"

    return DimensionScore(
        "cointegration_integrity", score, 0.25, status,
        f"adf_pvalue={p_value:.3f}",
    )


def _assess_regime_sensitivity(
    returns: pd.Series,
    vix_series: Optional[pd.Series] = None,
) -> DimensionScore:
    """Does performance depend heavily on regime?"""
    if vix_series is None or len(returns) < 60:
        return DimensionScore("regime_sensitivity", 0.7, 0.15, "ok", "no_vix_data")

    # Align
    common = returns.index.intersection(vix_series.index)
    if len(common) < 60:
        return DimensionScore("regime_sensitivity", 0.7, 0.15, "ok", "insufficient_overlap")

    ret = returns.loc[common]
    vix = vix_series.loc[common]

    # Split into high/low VIX regimes
    vix_median = vix.median()
    high_vix_ret = ret[vix > vix_median]
    low_vix_ret = ret[vix <= vix_median]

    sharpe_high = float(high_vix_ret.mean() / max(high_vix_ret.std(), 1e-10) * np.sqrt(252))
    sharpe_low = float(low_vix_ret.mean() / max(low_vix_ret.std(), 1e-10) * np.sqrt(252))

    # Big divergence = regime-sensitive
    diff = abs(sharpe_high - sharpe_low)
    if diff < 0.5:
        score, status = 1.0, "ok"
    elif diff < 1.0:
        score, status = 0.6, "ok"
    elif diff < 2.0:
        score, status = 0.3, "warning"
    else:
        score, status = 0.1, "critical"

    return DimensionScore(
        "regime_sensitivity", score, 0.15, status,
        f"sharpe_high_vix={sharpe_high:.2f}, sharpe_low_vix={sharpe_low:.2f}",
    )


def _assess_execution_integrity(
    n_trades: int,
    avg_cost_bps: float = 5.0,
    avg_pnl_bps: float = 20.0,
) -> DimensionScore:
    """Are execution costs eroding the edge?"""
    if n_trades == 0:
        return DimensionScore("execution_integrity", 0.5, 0.10, "warning", "no trades")

    cost_ratio = avg_cost_bps / max(avg_pnl_bps, 0.1) if avg_pnl_bps > 0 else 1.0

    if cost_ratio < 0.15:
        score, status = 1.0, "ok"
    elif cost_ratio < 0.30:
        score, status = 0.7, "ok"
    elif cost_ratio < 0.50:
        score, status = 0.4, "warning"
    else:
        score, status = 0.1, "critical"

    return DimensionScore(
        "execution_integrity", score, 0.10, status,
        f"cost_ratio={cost_ratio:.1%}, avg_cost={avg_cost_bps:.1f}bps",
    )


def _assess_stability(
    returns: pd.Series,
    window: int = 63,
) -> DimensionScore:
    """Is performance stable over time? Check rolling Sharpe variance."""
    if len(returns) < window * 3:
        return DimensionScore("stability", 0.5, 0.20, "warning", "insufficient data")

    rolling_sharpe = (
        returns.rolling(window).mean() /
        returns.rolling(window).std().replace(0, np.nan)
    ) * np.sqrt(252)

    sharpe_std = float(rolling_sharpe.dropna().std())

    if sharpe_std < 0.5:
        score, status = 1.0, "ok"
    elif sharpe_std < 1.0:
        score, status = 0.6, "ok"
    elif sharpe_std < 2.0:
        score, status = 0.3, "warning"
    else:
        score, status = 0.1, "critical"

    return DimensionScore(
        "stability", score, 0.20, status,
        f"rolling_sharpe_std={sharpe_std:.2f}",
    )


# ============================================================================
# Main Assessment
# ============================================================================

def assess_pair_health(
    pair_label: str,
    sym_x: str,
    sym_y: str,
    spread: pd.Series,
    pair_returns: pd.Series,
    *,
    n_trades: int = 0,
    avg_cost_bps: float = 5.0,
    avg_pnl_bps: float = 20.0,
    vix_series: Optional[pd.Series] = None,
) -> PairHealthReport:
    """
    Full health assessment for a pair strategy.

    Returns PairHealthReport with health_score, state, and recommended action.
    """
    dimensions = [
        _assess_alpha_integrity(pair_returns),
        _assess_cointegration(spread),
        _assess_regime_sensitivity(pair_returns, vix_series),
        _assess_execution_integrity(n_trades, avg_cost_bps, avg_pnl_bps),
        _assess_stability(pair_returns),
    ]

    # Weighted health score
    health_score = sum(d.score * d.weight for d in dimensions)

    # Identify root causes
    root_causes = []
    for d in dimensions:
        if d.status == "critical":
            root_causes.append(f"{d.name}_critical")
        elif d.status == "warning":
            root_causes.append(f"{d.name}_degrading")

    # Determine health state
    critical_count = sum(1 for d in dimensions if d.status == "critical")
    warning_count = sum(1 for d in dimensions if d.status == "warning")

    if critical_count >= 2:
        health_state = "RETIREMENT_CANDIDATE"
        action = "RETIRE"
    elif critical_count == 1 and "cointegration" in str(root_causes):
        health_state = "STRUCTURAL_DECAY"
        action = "PAUSE"
    elif critical_count == 1:
        health_state = "EARLY_DECAY"
        action = "REDUCE"
    elif warning_count >= 2:
        health_state = "WATCH"
        action = "HOLD"
    elif "regime_sensitivity_degrading" in root_causes:
        health_state = "REGIME_SUPPRESSED"
        action = "HOLD"
    else:
        health_state = "HEALTHY"
        action = "HOLD"

    report = PairHealthReport(
        pair_label=pair_label,
        sym_x=sym_x,
        sym_y=sym_y,
        alpha_integrity=dimensions[0].score,
        cointegration_integrity=dimensions[1].score,
        regime_sensitivity=dimensions[2].score,
        execution_integrity=dimensions[3].score,
        stability=dimensions[4].score,
        health_score=health_score,
        health_state=health_state,
        root_causes=root_causes,
        recommended_action=action,
        dimensions=dimensions,
    )

    logger.info(
        "Pair %s health: %.2f (%s) → %s",
        pair_label, health_score, health_state, action,
    )
    return report
