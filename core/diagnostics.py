# -*- coding: utf-8 -*-
"""
core/diagnostics.py — Signal, Spread, and Lifecycle Diagnostics
================================================================

Provides structured diagnostic objects for observability across the
signal engine, regime engine, and lifecycle state machine.

Objects:
  SpreadStateSnapshot  — Point-in-time spread measurement
  SignalFeatures       — Computed feature set consumed by signal engines
  SignalDiagnostics    — Detailed audit record for one signal computation
  RegimeDiagnostics    — Regime classification audit record
  LifecycleDiagnostics — State machine history and audit trail
  ExitDiagnostics      — Post-exit analysis record
  SignalAuditRecord    — Top-level audit record linking all diagnostics

Design principles:
  - Every diagnostic object is serializable to dict
  - All timestamps are UTC
  - Feature values are always float or NaN (never missing silently)
  - Diagnostic objects are append-only records; never mutate after creation
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np

from core.contracts import (
    ExitReason,
    PairId,
    RegimeLabel,
    SignalDirection,
    SignalQualityGrade,
    TradeLifecycleState,
)


# ══════════════════════════════════════════════════════════════════
# SPREAD STATE SNAPSHOT
# ══════════════════════════════════════════════════════════════════

@dataclass
class SpreadStateSnapshot:
    """Point-in-time measurement of a spread's current state.

    Captures all observables needed to make a signal decision.
    All values should be computed from data up to `as_of` only.
    """
    pair_id: PairId
    as_of: datetime

    # Core spread measurements
    z_score: float = np.nan          # Current z-score
    spread_level: float = np.nan     # Raw spread value (before z-scoring)
    spread_mean: float = np.nan      # Training mean
    spread_std: float = np.nan       # Training std
    spread_vol_20d: float = np.nan   # Rolling 20-day spread volatility
    spread_vol_baseline: float = np.nan  # Long-window baseline vol

    # Percentile and range
    z_percentile_252d: float = np.nan  # Percentile of current |z| over past year
    spread_range_20d: float = np.nan   # High-low range of spread over 20 days

    # Statistical health
    half_life_days: float = np.nan
    correlation: float = np.nan
    correlation_baseline: float = np.nan  # Long-window baseline correlation
    beta: float = np.nan
    beta_cv: float = np.nan           # Coefficient of variation of rolling beta

    # Signal momentum
    z_velocity: float = np.nan        # dz/dt — how fast z is moving
    z_acceleration: float = np.nan    # d²z/dt² — acceleration of divergence
    time_since_threshold_breach_days: float = np.nan
    time_since_last_mean_cross_days: float = np.nan
    consecutive_bars_above_threshold: int = 0

    # ADF / stationarity
    adf_pvalue: float = np.nan
    hurst_exponent: float = np.nan

    # Regime context
    regime_label: RegimeLabel = RegimeLabel.UNKNOWN
    regime_confidence: float = 0.0

    def to_dict(self) -> dict:
        return {
            "pair": self.pair_id.label,
            "as_of": self.as_of.isoformat(),
            "z_score": _safe(self.z_score),
            "spread_vol_20d": _safe(self.spread_vol_20d),
            "z_percentile_252d": _safe(self.z_percentile_252d),
            "half_life_days": _safe(self.half_life_days),
            "correlation": _safe(self.correlation),
            "beta_cv": _safe(self.beta_cv),
            "z_velocity": _safe(self.z_velocity),
            "z_acceleration": _safe(self.z_acceleration),
            "adf_pvalue": _safe(self.adf_pvalue),
            "hurst_exponent": _safe(self.hurst_exponent),
            "regime": self.regime_label.value,
            "regime_confidence": round(self.regime_confidence, 4),
        }


# ══════════════════════════════════════════════════════════════════
# SIGNAL FEATURES
# ══════════════════════════════════════════════════════════════════

@dataclass
class SignalFeatures:
    """Flat feature set for signal computation and ML overlays.

    All features are scalars in known ranges.
    NaN is allowed where the feature cannot be computed.
    """
    pair_id: PairId
    as_of: datetime

    # ── Spread-level features ─────────────────────────────────────
    z_score: float = np.nan
    z_robust: float = np.nan          # Median-IQR z-score
    z_percentile: float = np.nan      # Percentile rank of |z| over lookback
    z_velocity: float = np.nan        # Change in z over last 5 bars
    z_acceleration: float = np.nan    # Change in z_velocity over last 5 bars
    time_above_threshold_days: float = np.nan
    time_since_mean_cross_days: float = np.nan
    residual_slope: float = np.nan    # OLS slope of z over last 20 bars
    spread_vol: float = np.nan        # Current spread volatility (20d)
    spread_atr: float = np.nan        # ATR-style spread range

    # ── Relationship features ─────────────────────────────────────
    corr_20d: float = np.nan
    corr_63d: float = np.nan
    corr_252d: float = np.nan
    corr_drift: float = np.nan        # corr_20d - corr_252d
    coint_score: float = np.nan       # 1 - ADF p-value quality
    half_life_days: float = np.nan
    half_life_change_pct: float = np.nan  # Change in HL vs 60d prior
    beta_cv: float = np.nan
    residual_var_stability: float = np.nan  # Stability of spread variance

    # ── Regime / environment features ────────────────────────────
    spread_vol_regime: float = np.nan  # spread_vol / baseline_vol
    vol_of_vol: float = np.nan         # Vol of rolling spread volatility
    mean_reversion_quality: float = np.nan  # From signal_stack Layer 3
    regime_safety: float = np.nan      # From signal_stack Layer 4
    z_persistence: float = np.nan      # AR(1) of recent z-scores
    break_risk: float = np.nan         # Probability of structural break

    # ── Lifecycle features ────────────────────────────────────────
    position_age_days: int = 0
    entry_age_days: int = 0
    time_in_state_days: float = 0.0
    unrealized_excursion: float = np.nan  # Max adverse z-score since entry
    failed_reversion_attempts: int = 0    # Consecutive failed reversions
    cooldown_days_remaining: float = 0.0

    def to_dict(self) -> dict:
        return {k: _safe(v) if isinstance(v, float) else v
                for k, v in self.__dict__.items()
                if k not in ("pair_id", "as_of")} | {
            "pair": self.pair_id.label,
            "as_of": self.as_of.isoformat(),
        }


# ══════════════════════════════════════════════════════════════════
# SIGNAL DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════

@dataclass
class SignalDiagnostics:
    """Full audit record for one signal computation event.

    Captures every intermediate result so decisions can be replayed.
    """
    pair_id: PairId
    run_id: str                          # Links to a specific signal run
    computed_at: datetime = field(default_factory=datetime.utcnow)

    # Input state
    snapshot: Optional[SpreadStateSnapshot] = None
    features: Optional[SignalFeatures] = None

    # Signal stack layer scores
    distortion_score: float = np.nan    # Layer 1
    dislocation_score: float = np.nan   # Layer 2
    mr_quality_score: float = np.nan    # Layer 3
    regime_safety_score: float = np.nan # Layer 4
    conviction: float = np.nan          # Product of all layers

    # Quality assessment
    quality_grade: SignalQualityGrade = SignalQualityGrade.F
    quality_score: float = np.nan
    quality_reasons: list[str] = field(default_factory=list)
    skip_recommended: bool = False

    # Threshold applied
    entry_z_used: float = np.nan
    exit_z_used: float = np.nan
    stop_z_used: float = np.nan
    threshold_mode: str = ""           # e.g. "VOLATILITY_SCALED"
    threshold_modifiers: dict[str, float] = field(default_factory=dict)

    # Outcome
    direction_recommended: SignalDirection = SignalDirection.FLAT
    action_taken: str = ""             # IntentAction value
    block_reasons: list[str] = field(default_factory=list)
    rationale: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "pair": self.pair_id.label,
            "run_id": self.run_id,
            "computed_at": self.computed_at.isoformat(),
            "distortion_score": _safe(self.distortion_score),
            "dislocation_score": _safe(self.dislocation_score),
            "mr_quality_score": _safe(self.mr_quality_score),
            "regime_safety_score": _safe(self.regime_safety_score),
            "conviction": _safe(self.conviction),
            "quality_grade": self.quality_grade.value,
            "quality_score": _safe(self.quality_score),
            "skip_recommended": self.skip_recommended,
            "entry_z_used": _safe(self.entry_z_used),
            "threshold_mode": self.threshold_mode,
            "direction": self.direction_recommended.value,
            "action": self.action_taken,
            "block_reasons": self.block_reasons,
            "rationale": self.rationale,
        }


# ══════════════════════════════════════════════════════════════════
# REGIME DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════

@dataclass
class RegimeDiagnostics:
    """Audit record for a regime classification event."""
    pair_id: PairId
    classified_at: datetime = field(default_factory=datetime.utcnow)
    regime_label: RegimeLabel = RegimeLabel.UNKNOWN
    regime_confidence: float = 0.0
    regime_probabilities: dict[str, float] = field(default_factory=dict)
    # Rule-based signals that drove classification
    is_trending: bool = False
    is_high_vol: bool = False
    is_mean_shifted: bool = False
    has_break_risk: bool = False
    spread_vol_ratio: float = np.nan   # current / baseline
    z_persistence: float = np.nan      # AR(1) of recent z-scores
    break_confidence: float = np.nan
    # Tradability
    is_tradable: bool = True
    entry_z_multiplier: float = 1.0
    exit_z_multiplier: float = 1.0
    size_modifier: float = 1.0
    restrictions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "pair": self.pair_id.label,
            "classified_at": self.classified_at.isoformat(),
            "regime": self.regime_label.value,
            "confidence": round(self.regime_confidence, 4),
            "is_tradable": self.is_tradable,
            "entry_z_multiplier": round(self.entry_z_multiplier, 3),
            "size_modifier": round(self.size_modifier, 3),
            "restrictions": self.restrictions,
        }


# ══════════════════════════════════════════════════════════════════
# LIFECYCLE DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════

@dataclass
class LifecycleTransitionRecord:
    """Single state transition event."""
    from_state: TradeLifecycleState
    to_state: TradeLifecycleState
    trigger: str
    timestamp: datetime
    rationale: str = ""

    def to_dict(self) -> dict:
        return {
            "from": self.from_state.value,
            "to": self.to_state.value,
            "trigger": self.trigger,
            "timestamp": self.timestamp.isoformat(),
            "rationale": self.rationale,
        }


@dataclass
class LifecycleDiagnostics:
    """Complete lifecycle history for a spread."""
    pair_id: PairId
    current_state: TradeLifecycleState = TradeLifecycleState.WATCHLIST
    state_entered_at: datetime = field(default_factory=datetime.utcnow)
    transitions: list[LifecycleTransitionRecord] = field(default_factory=list)
    cooldown_until: Optional[datetime] = None
    suspension_count: int = 0
    total_trades: int = 0
    failed_entry_attempts: int = 0

    @property
    def time_in_current_state_days(self) -> float:
        delta = datetime.utcnow() - self.state_entered_at
        return delta.total_seconds() / 86400.0

    @property
    def is_in_cooldown(self) -> bool:
        if self.cooldown_until is None:
            return False
        return datetime.utcnow() < self.cooldown_until

    @property
    def cooldown_days_remaining(self) -> float:
        if not self.is_in_cooldown:
            return 0.0
        delta = self.cooldown_until - datetime.utcnow()
        return max(0.0, delta.total_seconds() / 86400.0)

    def to_dict(self) -> dict:
        return {
            "pair": self.pair_id.label,
            "current_state": self.current_state.value,
            "time_in_state_days": round(self.time_in_current_state_days, 2),
            "cooldown_days_remaining": round(self.cooldown_days_remaining, 2),
            "suspension_count": self.suspension_count,
            "total_trades": self.total_trades,
            "failed_entry_attempts": self.failed_entry_attempts,
            "n_transitions": len(self.transitions),
        }


# ══════════════════════════════════════════════════════════════════
# EXIT DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════

@dataclass
class ExitDiagnostics:
    """Post-exit analysis record for a closed trade."""
    pair_id: PairId
    trade_id: str
    entry_z: float = np.nan
    exit_z: float = np.nan
    entry_date: Optional[datetime] = None
    exit_date: Optional[datetime] = None
    holding_days: int = 0
    realized_pnl_pct: float = np.nan
    max_favorable_excursion: float = np.nan   # Best z vs entry
    max_adverse_excursion: float = np.nan     # Worst z vs entry
    exit_reasons: list[ExitReason] = field(default_factory=list)
    regime_at_entry: RegimeLabel = RegimeLabel.UNKNOWN
    regime_at_exit: RegimeLabel = RegimeLabel.UNKNOWN
    half_life_at_entry: float = np.nan
    half_life_at_exit: float = np.nan
    entry_quality_grade: SignalQualityGrade = SignalQualityGrade.F
    notes: str = ""

    @property
    def converged(self) -> bool:
        """Did the spread actually mean-revert?"""
        return ExitReason.MEAN_REVERSION_COMPLETE in self.exit_reasons or \
               ExitReason.TARGET_REACHED in self.exit_reasons

    @property
    def was_stopped(self) -> bool:
        return ExitReason.ADVERSE_EXCURSION_STOP in self.exit_reasons or \
               ExitReason.SPREAD_STOP in self.exit_reasons

    def to_dict(self) -> dict:
        return {
            "pair": self.pair_id.label,
            "trade_id": self.trade_id,
            "entry_z": _safe(self.entry_z),
            "exit_z": _safe(self.exit_z),
            "holding_days": self.holding_days,
            "realized_pnl_pct": _safe(self.realized_pnl_pct),
            "mae": _safe(self.max_adverse_excursion),
            "mfe": _safe(self.max_favorable_excursion),
            "exit_reasons": [r.value for r in self.exit_reasons],
            "converged": self.converged,
            "was_stopped": self.was_stopped,
            "regime_at_entry": self.regime_at_entry.value,
            "regime_at_exit": self.regime_at_exit.value,
        }


# ══════════════════════════════════════════════════════════════════
# SIGNAL AUDIT RECORD — TOP-LEVEL RECORD
# ══════════════════════════════════════════════════════════════════

@dataclass
class SignalAuditRecord:
    """Master audit record linking all diagnostics for one signal event.

    Stored per-pair per-timestamp. Used for replay, debugging, and analytics.
    """
    audit_id: str
    pair_id: PairId
    timestamp: datetime = field(default_factory=datetime.utcnow)

    signal_diagnostics: Optional[SignalDiagnostics] = None
    regime_diagnostics: Optional[RegimeDiagnostics] = None
    lifecycle_diagnostics: Optional[LifecycleDiagnostics] = None
    spread_snapshot: Optional[SpreadStateSnapshot] = None

    # Final decision summary
    final_action: str = ""
    final_conviction: float = np.nan
    final_quality_grade: str = ""
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "audit_id": self.audit_id,
            "pair": self.pair_id.label,
            "timestamp": self.timestamp.isoformat(),
            "final_action": self.final_action,
            "final_conviction": _safe(self.final_conviction),
            "final_quality_grade": self.final_quality_grade,
            "has_signal_diag": self.signal_diagnostics is not None,
            "has_regime_diag": self.regime_diagnostics is not None,
            "has_lifecycle_diag": self.lifecycle_diagnostics is not None,
            "notes": self.notes,
        }


# ══════════════════════════════════════════════════════════════════
# UTILITY
# ══════════════════════════════════════════════════════════════════

def _safe(v: float) -> float | None:
    """Return None for NaN/Inf, else rounded float."""
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return None
    return round(float(v), 6)


__all__ = [
    "SpreadStateSnapshot",
    "SignalFeatures",
    "SignalDiagnostics",
    "RegimeDiagnostics",
    "LifecycleTransitionRecord",
    "LifecycleDiagnostics",
    "ExitDiagnostics",
    "SignalAuditRecord",
]
