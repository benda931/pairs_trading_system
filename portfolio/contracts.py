# -*- coding: utf-8 -*-
"""
portfolio/contracts.py — Portfolio Construction Domain Objects
==============================================================

All typed, inspectable, serializable domain objects for the portfolio
construction and risk operating model.  These are the API boundary
between all portfolio sub-layers.

Design rules:
  - Every critical portfolio/risk output is a typed dataclass
  - No loose dicts for allocation, sizing, risk, or constraint outputs
  - All objects support to_dict() for audit/logging
  - Enum values are str-serializable (str, enum.Enum)
  - Optional ML fields are present but never required
"""
from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import numpy as np

from core.contracts import (
    ExitReason,
    PairId,
    RegimeLabel,
    SignalDirection,
    SignalQualityGrade,
)


# ══════════════════════════════════════════════════════════════════
# PORTFOLIO HEAT LEVEL (State Machine)
# ══════════════════════════════════════════════════════════════════

class PortfolioHeatLevel(str, enum.Enum):
    """Ordinal portfolio operating states driven by drawdown/heat."""
    NORMAL        = "NORMAL"         # Full capacity; all sleeves open
    CAUTIOUS      = "CAUTIOUS"       # Mild restriction; tighter ranking threshold
    THROTTLED     = "THROTTLED"      # Reduced new-entry capacity; size scaled down
    DEFENSIVE     = "DEFENSIVE"      # Only exits/reduces; minimal new entries
    RECOVERY_ONLY = "RECOVERY_ONLY"  # No new entries; only reductions and exits
    HALTED        = "HALTED"         # No new exposure; only risk reduction


# ══════════════════════════════════════════════════════════════════
# KILL-SWITCH STATE
# ══════════════════════════════════════════════════════════════════

class KillSwitchMode(str, enum.Enum):
    OFF    = "OFF"       # No kill-switch active
    SOFT   = "SOFT"      # Warn; reduce new entries; alert operators
    REDUCE = "REDUCE"    # Begin reducing existing positions
    HARD   = "HARD"      # Halt all new activity; emergency reduce


@dataclass
class KillSwitchState:
    """Current kill-switch status and rationale."""
    mode: KillSwitchMode = KillSwitchMode.OFF
    triggered: bool = False
    triggered_at: Optional[datetime] = None
    reason: str = ""
    triggered_rules: list[str] = field(default_factory=list)
    recommended_actions: list[str] = field(default_factory=list)
    severity_score: float = 0.0          # 0=none, 1=critical
    scaling_factor: float = 1.0          # 0=full halt, 1=normal
    acknowledged: bool = False

    def is_blocking_new_entries(self) -> bool:
        return self.mode in (KillSwitchMode.HARD, KillSwitchMode.REDUCE)

    def to_dict(self) -> dict:
        return {
            "mode": self.mode.value,
            "triggered": self.triggered,
            "triggered_at": self.triggered_at.isoformat() if self.triggered_at else None,
            "reason": self.reason,
            "rules": self.triggered_rules,
            "severity": round(self.severity_score, 4),
            "scaling_factor": round(self.scaling_factor, 4),
        }


# ══════════════════════════════════════════════════════════════════
# DRAWDOWN STATE
# ══════════════════════════════════════════════════════════════════

@dataclass
class DrawdownState:
    """Real-time drawdown tracking and throttle factor."""
    current_dd_pct: float = 0.0      # Current drawdown as fraction [0,1]
    peak_value: float = 1.0          # Peak portfolio value (normalised)
    current_value: float = 1.0       # Current portfolio value (normalised)
    rolling_dd_7d: float = 0.0       # 7-day rolling drawdown
    rolling_dd_30d: float = 0.0      # 30-day rolling drawdown
    heat_level: PortfolioHeatLevel = PortfolioHeatLevel.NORMAL
    throttle_factor: float = 1.0     # New-entry size multiplier [0,1]
    max_new_positions: int = 20      # Maximum new positions in this state
    updated_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def is_stressed(self) -> bool:
        return self.heat_level not in (PortfolioHeatLevel.NORMAL, PortfolioHeatLevel.CAUTIOUS)

    def to_dict(self) -> dict:
        return {
            "current_dd_pct": round(self.current_dd_pct, 4),
            "peak_value": round(self.peak_value, 4),
            "current_value": round(self.current_value, 4),
            "heat_level": self.heat_level.value,
            "throttle_factor": round(self.throttle_factor, 4),
            "max_new_positions": self.max_new_positions,
        }


# ══════════════════════════════════════════════════════════════════
# THROTTLE STATE
# ══════════════════════════════════════════════════════════════════

@dataclass
class ThrottleState:
    """Capital throttling decisions for the current heat level."""
    heat_level: PortfolioHeatLevel = PortfolioHeatLevel.NORMAL
    # Size multiplier applied to all new entries
    size_multiplier: float = 1.0
    # Maximum simultaneous open positions
    max_positions: int = 30
    # Maximum new entries per rebalance cycle
    max_new_entries_per_cycle: int = 5
    # Minimum ranking score to receive any funding
    min_ranking_score_to_fund: float = 0.0
    # Sleeves available for new entries (empty = all)
    allowed_sleeves: list[str] = field(default_factory=list)
    # Risk budget fraction available for new entries
    available_risk_budget_fraction: float = 1.0

    def to_dict(self) -> dict:
        return {
            "heat_level": self.heat_level.value,
            "size_multiplier": round(self.size_multiplier, 4),
            "max_positions": self.max_positions,
            "max_new_entries_per_cycle": self.max_new_entries_per_cycle,
            "min_ranking_score": round(self.min_ranking_score_to_fund, 4),
            "available_risk_budget_fraction": round(self.available_risk_budget_fraction, 4),
        }


# ══════════════════════════════════════════════════════════════════
# HEAT STATE
# ══════════════════════════════════════════════════════════════════

@dataclass
class HeatState:
    """Portfolio 'heat' — aggregate risk consumption relative to budget."""
    total_risk_budget: float = 1.0    # Total allocated risk budget
    consumed_risk: float = 0.0        # Currently consumed
    reserved_risk: float = 0.0        # Reserved for pending orders
    free_risk: float = 1.0            # Available for new positions
    heat_pct: float = 0.0             # Consumed / total [0,1]
    n_active_positions: int = 0
    n_pending_orders: int = 0
    gross_exposure: float = 0.0
    net_exposure: float = 0.0
    gross_leverage: float = 0.0
    net_leverage: float = 0.0

    def to_dict(self) -> dict:
        return {
            "heat_pct": round(self.heat_pct, 4),
            "consumed_risk": round(self.consumed_risk, 4),
            "free_risk": round(self.free_risk, 4),
            "n_active_positions": self.n_active_positions,
            "gross_leverage": round(self.gross_leverage, 4),
        }


# ══════════════════════════════════════════════════════════════════
# CAPITAL POOL & SLEEVE
# ══════════════════════════════════════════════════════════════════

@dataclass
class CapitalPool:
    """Tracks capital usage across the portfolio."""
    total_capital: float = 1_000_000.0
    allocated_capital: float = 0.0    # In active positions
    reserved_capital: float = 0.0     # Pending orders / reserved for fills
    pending_recycle: float = 0.0      # Released from exits; awaiting redeployment

    @property
    def free_capital(self) -> float:
        return max(0.0, self.total_capital - self.allocated_capital - self.reserved_capital)

    @property
    def utilisation_pct(self) -> float:
        if self.total_capital <= 0:
            return 0.0
        return (self.allocated_capital + self.reserved_capital) / self.total_capital

    def can_allocate(self, amount: float) -> bool:
        return self.free_capital >= amount

    def to_dict(self) -> dict:
        return {
            "total": round(self.total_capital, 2),
            "allocated": round(self.allocated_capital, 2),
            "reserved": round(self.reserved_capital, 2),
            "free": round(self.free_capital, 2),
            "utilisation_pct": round(self.utilisation_pct, 4),
        }


@dataclass
class SleeveDef:
    """Definition of a strategy sleeve / capital sub-bucket."""
    name: str
    description: str = ""
    max_capital_fraction: float = 1.0     # Fraction of total capital
    max_positions: int = 50
    allowed_regimes: list[str] = field(default_factory=list)   # Empty = all
    blocked_regimes: list[str] = field(default_factory=list)
    min_quality_grade: str = "D"          # Minimum SignalQualityGrade
    enabled: bool = True
    priority: int = 0                     # Lower = higher priority in capital competition

    def is_regime_allowed(self, regime: RegimeLabel) -> bool:
        if self.blocked_regimes and regime.value in self.blocked_regimes:
            return False
        if self.allowed_regimes and regime.value not in self.allowed_regimes:
            return False
        return True

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "max_capital_fraction": self.max_capital_fraction,
            "max_positions": self.max_positions,
            "enabled": self.enabled,
        }


@dataclass
class CapitalBudget:
    """Current capital budget state for one sleeve."""
    sleeve: SleeveDef
    allocated: float = 0.0
    reserved: float = 0.0
    n_positions: int = 0

    @property
    def max_capital(self) -> float:
        # Computed externally; stored here for convenience
        return getattr(self, "_max_capital", float("inf"))

    @property
    def free(self) -> float:
        return max(0.0, self.max_capital - self.allocated - self.reserved)

    def to_dict(self) -> dict:
        return {
            "sleeve": self.sleeve.name,
            "allocated": round(self.allocated, 2),
            "reserved": round(self.reserved, 2),
            "n_positions": self.n_positions,
            "free": round(self.free, 2),
        }


# ══════════════════════════════════════════════════════════════════
# OPPORTUNITY RANKING
# ══════════════════════════════════════════════════════════════════

@dataclass
class RankedOpportunity:
    """
    A single approved trade intent that has been ranked for capital competition.

    Score decomposition provides full auditability of why an opportunity
    ranked where it did.
    """
    pair_id: PairId
    raw_pair_label: str = ""

    # ── Ranking scores (all [0, 1]) ─────────────────────────────
    signal_strength_score: float = 0.0      # z-score / threshold attractiveness
    signal_quality_score: float = 0.0       # Quality grade translated to score
    regime_suitability_score: float = 0.0   # How favourable current regime is
    reversion_probability: float = 0.0      # ML or rule-based success probability
    diversification_value: float = 1.0      # Marginal diversification contribution
    stability_score: float = 0.0            # Rolling spread stability
    capacity_score: float = 1.0             # How close to capacity limits
    freshness_score: float = 1.0            # Model/signal freshness penalty

    # ── Composite ───────────────────────────────────────────────
    composite_score: float = 0.0            # Weighted combination
    rank: int = 0                           # Lower = more attractive

    # ── Context ─────────────────────────────────────────────────
    quality_grade: str = "B"
    regime: str = "UNKNOWN"
    conviction: float = 0.0
    z_score: float = 0.0
    half_life_days: float = float("nan")
    recommended_sleeve: str = "default"

    # ── Penalties and blocks ─────────────────────────────────────
    overlap_penalty: float = 0.0            # Penalty for shared-leg / cluster overlap
    blockers: list[str] = field(default_factory=list)    # Hard blocks
    strengths: list[str] = field(default_factory=list)   # Key positives
    penalties: list[str] = field(default_factory=list)   # Soft penalties applied

    # ── Extended ranking scores ──────────────────────────────────
    capital_efficiency_score: float = 0.5   # Expected P&L per unit capital × time
    liquidity_score: float = 0.7            # Trade-size executability vs ADV
    edge_quality_score: float = 0.5         # OOS walk-forward edge quality

    # ── ML overlay ───────────────────────────────────────────────
    ml_ranking_score: Optional[float] = None
    ml_model_id: Optional[str] = None

    generated_at: datetime = field(default_factory=datetime.utcnow)

    def is_fundable(self) -> bool:
        return len(self.blockers) == 0 and self.composite_score > 0

    def to_dict(self) -> dict:
        return {
            "pair": self.pair_id.label,
            "rank": self.rank,
            "composite_score": round(self.composite_score, 4),
            "signal_strength": round(self.signal_strength_score, 4),
            "signal_quality": round(self.signal_quality_score, 4),
            "regime_suitability": round(self.regime_suitability_score, 4),
            "reversion_probability": round(self.reversion_probability, 4),
            "diversification_value": round(self.diversification_value, 4),
            "stability": round(self.stability_score, 4),
            "capital_efficiency": round(self.capital_efficiency_score, 4),
            "liquidity": round(self.liquidity_score, 4),
            "edge_quality": round(self.edge_quality_score, 4),
            "overlap_penalty": round(self.overlap_penalty, 4),
            "regime": self.regime,
            "quality_grade": self.quality_grade,
            "conviction": round(self.conviction, 4),
            "z_score": round(self.z_score, 4),
            "sleeve": self.recommended_sleeve,
            "blockers": self.blockers,
            "strengths": self.strengths,
            "penalties": self.penalties,
        }


@dataclass
class OpportunitySet:
    """Ranked collection of trade opportunities competing for capital."""
    opportunities: list[RankedOpportunity] = field(default_factory=list)
    generated_at: datetime = field(default_factory=datetime.utcnow)
    n_input_intents: int = 0
    n_blocked: int = 0
    n_fundable: int = 0

    def sorted_by_rank(self) -> list[RankedOpportunity]:
        return sorted(self.opportunities, key=lambda o: o.rank)

    def fundable_only(self) -> list[RankedOpportunity]:
        return [o for o in self.sorted_by_rank() if o.is_fundable()]

    def to_dict(self) -> dict:
        return {
            "n_input_intents": self.n_input_intents,
            "n_blocked": self.n_blocked,
            "n_fundable": self.n_fundable,
            "top_5": [o.to_dict() for o in self.sorted_by_rank()[:5]],
        }


# ══════════════════════════════════════════════════════════════════
# SIZING DECISION
# ══════════════════════════════════════════════════════════════════

@dataclass
class SizingDecision:
    """
    Complete sizing output for one approved pair trade.

    Distinguishes gross notional from risk notional and records
    all sizing inputs for audit / debugging.
    """
    pair_id: PairId
    sleeve: str = "default"

    # ── Sizes ────────────────────────────────────────────────────
    gross_notional: float = 0.0        # Total gross exposure (both legs)
    risk_notional: float = 0.0         # Risk-equivalent notional
    capital_usage: float = 0.0         # Capital consumed (margin-adjusted)
    weight_of_portfolio: float = 0.0   # Fraction of total capital
    risk_contribution: float = 0.0     # Fraction of total portfolio risk

    # Per-leg sizes (for execution routing)
    leg_x_notional: float = 0.0
    leg_y_notional: float = 0.0
    hedge_ratio: float = 1.0           # long X / short Y (or vice versa)
    direction: str = "LONG_SPREAD"     # "LONG_SPREAD" or "SHORT_SPREAD"

    # ── Sizing inputs (for audit) ────────────────────────────────
    base_weight: float = 0.0           # Pre-scaling weight
    conviction_scalar: float = 1.0     # Conviction multiplier applied
    vol_target_scalar: float = 1.0     # Vol-targeting multiplier
    drawdown_scalar: float = 1.0       # Drawdown throttle multiplier
    quality_scalar: float = 1.0        # Quality grade multiplier
    regime_scalar: float = 1.0         # Regime multiplier

    # ── Sizing method used ───────────────────────────────────────
    sizing_method: str = "conviction"  # "kelly" or "conviction"

    # ── Constraints applied ──────────────────────────────────────
    was_capped: bool = False
    cap_reason: str = ""
    min_executable_size: float = 0.0   # Below this: not executable
    is_executable: bool = True

    computed_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "pair": self.pair_id.label,
            "sleeve": self.sleeve,
            "gross_notional": round(self.gross_notional, 2),
            "risk_notional": round(self.risk_notional, 2),
            "capital_usage": round(self.capital_usage, 2),
            "weight": round(self.weight_of_portfolio, 4),
            "risk_contribution": round(self.risk_contribution, 4),
            "leg_x": round(self.leg_x_notional, 2),
            "leg_y": round(self.leg_y_notional, 2),
            "direction": self.direction,
            "conviction_scalar": round(self.conviction_scalar, 4),
            "vol_target_scalar": round(self.vol_target_scalar, 4),
            "drawdown_scalar": round(self.drawdown_scalar, 4),
            "sizing_method": self.sizing_method,
            "was_capped": self.was_capped,
            "is_executable": self.is_executable,
        }


# ══════════════════════════════════════════════════════════════════
# CONSTRAINT VIOLATION
# ══════════════════════════════════════════════════════════════════

class ConstraintType(str, enum.Enum):
    MAX_POSITIONS         = "MAX_POSITIONS"
    MAX_GROSS_EXPOSURE    = "MAX_GROSS_EXPOSURE"
    MAX_NET_EXPOSURE      = "MAX_NET_EXPOSURE"
    MAX_SECTOR_EXPOSURE   = "MAX_SECTOR_EXPOSURE"
    MAX_CLUSTER_EXPOSURE  = "MAX_CLUSTER_EXPOSURE"
    MAX_SHARED_LEG        = "MAX_SHARED_LEG"
    MAX_SINGLE_PAIR       = "MAX_SINGLE_PAIR"
    MAX_LEVERAGE          = "MAX_LEVERAGE"
    MIN_QUALITY_GRADE     = "MIN_QUALITY_GRADE"
    REGIME_BLOCKED        = "REGIME_BLOCKED"
    KILL_SWITCH           = "KILL_SWITCH"
    DRAWDOWN_THROTTLE     = "DRAWDOWN_THROTTLE"
    CAPITAL_BUDGET        = "CAPITAL_BUDGET"
    PENDING_EXPOSURE      = "PENDING_EXPOSURE"
    STALE_MODEL           = "STALE_MODEL"
    SLEEVE_BLOCKED        = "SLEEVE_BLOCKED"


@dataclass
class ConstraintViolation:
    """A single constraint that was violated during allocation."""
    constraint_type: ConstraintType
    pair_id: Optional[PairId] = None
    description: str = ""
    current_value: float = float("nan")
    limit_value: float = float("nan")
    severity: str = "HARD"     # "HARD" (blocks) or "SOFT" (warns)
    action_taken: str = ""

    @property
    def is_hard(self) -> bool:
        return self.severity == "HARD"

    def to_dict(self) -> dict:
        return {
            "type": self.constraint_type.value,
            "pair": self.pair_id.label if self.pair_id else None,
            "description": self.description,
            "current": round(self.current_value, 4) if not math.isnan(self.current_value) else None,
            "limit": round(self.limit_value, 4) if not math.isnan(self.limit_value) else None,
            "severity": self.severity,
            "action": self.action_taken,
        }


@dataclass
class RiskConstraintResult:
    """Aggregate result of running risk constraints against an allocation proposal."""
    violations: list[ConstraintViolation] = field(default_factory=list)
    hard_violations: list[ConstraintViolation] = field(default_factory=list)
    soft_violations: list[ConstraintViolation] = field(default_factory=list)
    approved: bool = True
    rejection_reason: str = ""
    checked_at: datetime = field(default_factory=datetime.utcnow)

    def add(self, v: ConstraintViolation) -> None:
        self.violations.append(v)
        if v.is_hard:
            self.hard_violations.append(v)
            self.approved = False
            self.rejection_reason = v.description
        else:
            self.soft_violations.append(v)

    def to_dict(self) -> dict:
        return {
            "approved": self.approved,
            "hard_violations": len(self.hard_violations),
            "soft_violations": len(self.soft_violations),
            "violations": [v.to_dict() for v in self.violations],
        }


# ══════════════════════════════════════════════════════════════════
# ALLOCATION RATIONALE
# ══════════════════════════════════════════════════════════════════

class AllocationOutcome(str, enum.Enum):
    FUNDED          = "FUNDED"
    PARTIAL_FUNDED  = "PARTIAL_FUNDED"
    UNFUNDED        = "UNFUNDED"
    BLOCKED_SIGNAL  = "BLOCKED_SIGNAL"
    BLOCKED_RISK    = "BLOCKED_RISK"
    BLOCKED_CAPITAL = "BLOCKED_CAPITAL"
    BLOCKED_OVERLAP = "BLOCKED_OVERLAP"
    BLOCKED_REGIME  = "BLOCKED_REGIME"
    QUEUED          = "QUEUED"


@dataclass
class AllocationRationale:
    """Structured explanation of why an allocation was made or denied."""
    pair_id: PairId
    outcome: AllocationOutcome = AllocationOutcome.UNFUNDED
    rank: int = 0
    composite_score: float = 0.0
    capital_requested: float = 0.0
    capital_granted: float = 0.0
    sleeve: str = ""
    sizing_factors: dict[str, float] = field(default_factory=dict)
    constraint_violations: list[str] = field(default_factory=list)
    overlap_details: list[str] = field(default_factory=list)
    decision_notes: list[str] = field(default_factory=list)
    decided_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def funding_fraction(self) -> float:
        if self.capital_requested <= 0:
            return 0.0
        return min(1.0, self.capital_granted / self.capital_requested)

    def to_dict(self) -> dict:
        return {
            "pair": self.pair_id.label,
            "outcome": self.outcome.value,
            "rank": self.rank,
            "composite_score": round(self.composite_score, 4),
            "capital_granted": round(self.capital_granted, 2),
            "funding_fraction": round(self.funding_fraction, 4),
            "sleeve": self.sleeve,
            "violations": self.constraint_violations,
            "overlap": self.overlap_details,
            "notes": self.decision_notes,
        }


# ══════════════════════════════════════════════════════════════════
# ALLOCATION PROPOSAL & DECISION
# ══════════════════════════════════════════════════════════════════

@dataclass
class AllocationProposal:
    """
    Proposed allocation for a single ranked opportunity.
    This is a pre-approval proposal; it still must pass risk constraints.
    """
    pair_id: PairId
    ranked_opportunity: RankedOpportunity
    sizing: SizingDecision
    sleeve: str = "default"
    proposed_capital: float = 0.0
    risk_budget_consumed: float = 0.0
    is_add_to_existing: bool = False
    is_rebalance: bool = False
    proposed_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "pair": self.pair_id.label,
            "sleeve": self.sleeve,
            "proposed_capital": round(self.proposed_capital, 2),
            "risk_budget_consumed": round(self.risk_budget_consumed, 4),
            "is_add": self.is_add_to_existing,
            "is_rebalance": self.is_rebalance,
        }


@dataclass
class AllocationDecision:
    """
    Final allocation decision after risk constraint enforcement.
    This is the canonical output of the portfolio construction engine.
    """
    pair_id: PairId
    proposal: AllocationProposal
    rationale: AllocationRationale
    constraint_result: RiskConstraintResult
    sizing: SizingDecision

    # Final approved values (may differ from proposal after constraints)
    approved_capital: float = 0.0
    approved_weight: float = 0.0
    approved: bool = True
    scaling_applied: float = 1.0   # Fraction of proposal approved

    decided_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "pair": self.pair_id.label,
            "approved": self.approved,
            "approved_capital": round(self.approved_capital, 2),
            "approved_weight": round(self.approved_weight, 4),
            "scaling_applied": round(self.scaling_applied, 4),
            "sizing": self.sizing.to_dict(),
            "rationale": self.rationale.to_dict(),
            "constraints": self.constraint_result.to_dict(),
        }


# ══════════════════════════════════════════════════════════════════
# EXPOSURE SUMMARY
# ══════════════════════════════════════════════════════════════════

@dataclass
class ExposureContribution:
    """One pair's contribution to a portfolio exposure bucket."""
    pair_id: PairId
    notional: float = 0.0
    weight: float = 0.0
    direction: str = ""
    sector: str = "UNKNOWN"
    cluster_id: str = "UNKNOWN"
    leg_x: str = ""
    leg_y: str = ""

    def to_dict(self) -> dict:
        return {
            "pair": self.pair_id.label,
            "notional": round(self.notional, 2),
            "weight": round(self.weight, 4),
            "sector": self.sector,
            "cluster": self.cluster_id,
        }


@dataclass
class SharedLegSummary:
    """Summary of shared-leg concentration for one instrument."""
    instrument: str
    n_pairs_using: int = 0
    total_notional: float = 0.0
    net_notional: float = 0.0      # Long - Short
    pairs: list[str] = field(default_factory=list)
    is_dominant: bool = False      # True if instrument is heavily concentrated
    concentration_score: float = 0.0  # [0,1] — how concentrated

    def to_dict(self) -> dict:
        return {
            "instrument": self.instrument,
            "n_pairs": self.n_pairs_using,
            "total_notional": round(self.total_notional, 2),
            "net_notional": round(self.net_notional, 2),
            "is_dominant": self.is_dominant,
            "concentration_score": round(self.concentration_score, 4),
        }


@dataclass
class ClusterExposureSummary:
    """Exposure concentration in one correlation cluster."""
    cluster_id: str
    n_pairs: int = 0
    total_notional: float = 0.0
    fraction_of_portfolio: float = 0.0
    pairs: list[str] = field(default_factory=list)
    is_overcrowded: bool = False
    max_allowed_fraction: float = 0.25

    def to_dict(self) -> dict:
        return {
            "cluster_id": self.cluster_id,
            "n_pairs": self.n_pairs,
            "fraction_of_portfolio": round(self.fraction_of_portfolio, 4),
            "is_overcrowded": self.is_overcrowded,
        }


@dataclass
class ExposureSummary:
    """Aggregate portfolio exposure across all dimensions."""
    total_capital: float = 0.0
    gross_exposure: float = 0.0
    net_exposure: float = 0.0
    gross_leverage: float = 0.0
    net_leverage: float = 0.0

    # By dimension
    by_sector: dict[str, float] = field(default_factory=dict)
    by_cluster: dict[str, float] = field(default_factory=dict)
    by_regime: dict[str, float] = field(default_factory=dict)
    by_sleeve: dict[str, float] = field(default_factory=dict)

    # Shared-leg analysis
    shared_legs: list[SharedLegSummary] = field(default_factory=list)
    cluster_exposures: list[ClusterExposureSummary] = field(default_factory=list)

    # Factor exposures (optional; populated when factor model is available)
    factor_exposures: dict[str, float] = field(default_factory=dict)

    # Concentration flags
    max_sector_concentration: float = 0.0
    max_cluster_concentration: float = 0.0
    dominant_legs: list[str] = field(default_factory=list)

    computed_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "gross_leverage": round(self.gross_leverage, 4),
            "net_leverage": round(self.net_leverage, 4),
            "by_sector": {k: round(v, 4) for k, v in self.by_sector.items()},
            "by_cluster": {k: round(v, 4) for k, v in self.by_cluster.items()},
            "max_sector_concentration": round(self.max_sector_concentration, 4),
            "max_cluster_concentration": round(self.max_cluster_concentration, 4),
            "n_shared_leg_alerts": sum(1 for s in self.shared_legs if s.is_dominant),
            "n_overcrowded_clusters": sum(1 for c in self.cluster_exposures if c.is_overcrowded),
        }


# ══════════════════════════════════════════════════════════════════
# PORTFOLIO SNAPSHOT
# ══════════════════════════════════════════════════════════════════

@dataclass
class PortfolioSnapshot:
    """
    Complete point-in-time state of the portfolio.
    The canonical hand-off object between portfolio and downstream layers.
    """
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Capital state
    capital_pool: CapitalPool = field(default_factory=CapitalPool)

    # Heat and drawdown
    heat_state: HeatState = field(default_factory=HeatState)
    drawdown_state: DrawdownState = field(default_factory=DrawdownState)
    throttle_state: ThrottleState = field(default_factory=ThrottleState)

    # Kill-switch
    kill_switch: KillSwitchState = field(default_factory=KillSwitchState)

    # Exposure summary
    exposure: ExposureSummary = field(default_factory=ExposureSummary)

    # Active allocation decisions
    active_allocations: list[AllocationDecision] = field(default_factory=list)

    # Pending (unfilled) allocations
    pending_allocations: list[AllocationDecision] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "capital": self.capital_pool.to_dict(),
            "heat": self.heat_state.to_dict(),
            "drawdown": self.drawdown_state.to_dict(),
            "throttle": self.throttle_state.to_dict(),
            "kill_switch": self.kill_switch.to_dict(),
            "exposure": self.exposure.to_dict(),
            "n_active": len(self.active_allocations),
            "n_pending": len(self.pending_allocations),
        }


# ══════════════════════════════════════════════════════════════════
# PORTFOLIO MANAGEMENT DECISIONS
# ══════════════════════════════════════════════════════════════════

@dataclass
class RebalanceDecision:
    """Decision to adjust the size of an existing position."""
    pair_id: PairId
    current_weight: float = 0.0
    target_weight: float = 0.0
    delta_weight: float = 0.0       # target - current
    delta_notional: float = 0.0
    rationale: str = ""
    urgency: str = "NORMAL"         # "URGENT", "NORMAL", "OPPORTUNISTIC"
    triggered_by: str = ""          # "CONCENTRATION", "DRIFT", "DRAWDOWN", etc.

    def is_increase(self) -> bool:
        return self.delta_weight > 0

    def to_dict(self) -> dict:
        return {
            "pair": self.pair_id.label,
            "current_weight": round(self.current_weight, 4),
            "target_weight": round(self.target_weight, 4),
            "delta_notional": round(self.delta_notional, 2),
            "rationale": self.rationale,
            "urgency": self.urgency,
        }


@dataclass
class CapitalRecycleDecision:
    """Decision to recycle capital from one pair to another (or to reserve)."""
    source_pair: PairId
    target_pair: Optional[PairId] = None   # None = return to free capital
    amount: float = 0.0
    reason: str = ""
    opportunity_cost_score: float = 0.0    # Score of unfunded opportunity missed
    triggered_by: str = ""
    decided_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "from": self.source_pair.label,
            "to": self.target_pair.label if self.target_pair else "FREE",
            "amount": round(self.amount, 2),
            "reason": self.reason,
            "opp_cost_score": round(self.opportunity_cost_score, 4),
        }


@dataclass
class DeRiskingDecision:
    """Decision to reduce or exit positions due to risk/drawdown state."""
    heat_level: PortfolioHeatLevel = PortfolioHeatLevel.NORMAL
    pairs_to_exit: list[PairId] = field(default_factory=list)
    pairs_to_reduce: list[PairId] = field(default_factory=list)
    reduction_fractions: dict[str, float] = field(default_factory=dict)  # pair_label → fraction to reduce
    reason: str = ""
    urgency: str = "NORMAL"
    decided_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "heat_level": self.heat_level.value,
            "n_exits": len(self.pairs_to_exit),
            "n_reduces": len(self.pairs_to_reduce),
            "reason": self.reason,
            "urgency": self.urgency,
        }


# ══════════════════════════════════════════════════════════════════
# PORTFOLIO DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════

@dataclass
class PortfolioDiagnostics:
    """Comprehensive diagnostics for one portfolio construction cycle."""
    cycle_id: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Opportunity funnel
    n_intents_received: int = 0
    n_qualified: int = 0
    n_ranked: int = 0
    n_funded: int = 0
    n_blocked_signal: int = 0
    n_blocked_risk: int = 0
    n_blocked_capital: int = 0
    n_blocked_overlap: int = 0

    # Capital usage
    capital_allocated: float = 0.0
    capital_reserved: float = 0.0
    capital_free: float = 0.0
    utilisation_pct: float = 0.0

    # Risk metrics
    portfolio_heat: float = 0.0
    gross_leverage: float = 0.0
    net_leverage: float = 0.0
    max_sector_concentration: float = 0.0
    max_cluster_concentration: float = 0.0
    n_shared_leg_alerts: int = 0

    # State
    heat_level: str = "NORMAL"
    kill_switch_mode: str = "OFF"

    # Constraint activity
    n_hard_violations: int = 0
    n_soft_violations: int = 0
    binding_constraints: list[str] = field(default_factory=list)

    # Top opportunities funded/unfunded
    top_funded: list[str] = field(default_factory=list)
    top_unfunded_reasons: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "cycle_id": self.cycle_id,
            "n_funded": self.n_funded,
            "n_blocked": self.n_blocked_signal + self.n_blocked_risk + self.n_blocked_capital,
            "utilisation_pct": round(self.utilisation_pct, 4),
            "heat_level": self.heat_level,
            "gross_leverage": round(self.gross_leverage, 4),
            "n_hard_violations": self.n_hard_violations,
            "binding_constraints": self.binding_constraints,
        }


@dataclass
class PortfolioAuditRecord:
    """Master audit record for one complete portfolio construction cycle."""
    cycle_id: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    snapshot_before: Optional[PortfolioSnapshot] = None
    snapshot_after: Optional[PortfolioSnapshot] = None
    diagnostics: Optional[PortfolioDiagnostics] = None
    allocation_decisions: list[AllocationDecision] = field(default_factory=list)
    rebalance_decisions: list[RebalanceDecision] = field(default_factory=list)
    de_risking_decisions: list[DeRiskingDecision] = field(default_factory=list)
    recycle_decisions: list[CapitalRecycleDecision] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def duration_seconds(self) -> Optional[float]:
        if self.completed_at and self.started_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    def to_dict(self) -> dict:
        return {
            "cycle_id": self.cycle_id,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "n_allocations": len(self.allocation_decisions),
            "n_rebalances": len(self.rebalance_decisions),
            "diagnostics": self.diagnostics.to_dict() if self.diagnostics else None,
        }


__all__ = [
    "AllocationDecision",
    "AllocationOutcome",
    "AllocationProposal",
    "AllocationRationale",
    "CapitalBudget",
    "CapitalPool",
    "CapitalRecycleDecision",
    "ClusterExposureSummary",
    "ConstraintType",
    "ConstraintViolation",
    "DeRiskingDecision",
    "DrawdownState",
    "ExposureContribution",
    "ExposureSummary",
    "HeatState",
    "KillSwitchMode",
    "KillSwitchState",
    "OpportunitySet",
    "PortfolioAuditRecord",
    "PortfolioDiagnostics",
    "PortfolioHeatLevel",
    "PortfolioSnapshot",
    "RankedOpportunity",
    "RebalanceDecision",
    "RiskConstraintResult",
    "SharedLegSummary",
    "SizingDecision",
    "SleeveDef",
    "ThrottleState",
]
