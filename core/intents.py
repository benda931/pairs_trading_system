# -*- coding: utf-8 -*-
"""
core/intents.py — Structured Trade Intent Objects
==================================================

Signal intents are the OUTPUT of the signal engine and the INPUT to the
portfolio / risk / execution layers. They represent PROPOSALS, not automatic
actions. The signal engine proposes; portfolio/risk/execution decide and act.

Intent hierarchy:
    BaseIntent
    ├── WatchIntent     — monitor, no action
    ├── EntryIntent     — open a new position
    ├── AddIntent       — scale into an existing position
    ├── HoldIntent      — maintain current position unchanged
    ├── ReduceIntent    — partial exit
    ├── ExitIntent      — full exit
    ├── SuspendIntent   — temporarily pause trading this spread
    └── RetireIntent    — permanently deactivate until revalidated

SignalDecision wraps an intent with full context:
  - regime label + confidence
  - signal quality grade
  - lifecycle state
  - block reasons (if action is WATCH/SUSPEND/RETIRE due to blocking)
  - rationale list (human/machine-readable reasons)
  - timestamp and generated_at

Design rules:
  1. No intent may directly trigger an order. It is always consumed downstream.
  2. BlockReason lists are always populated when action != ENTER.
  3. ExitReason lists always contain at least one reason on ExitIntent / ReduceIntent.
  4. Confidence is in [0, 1]; 1 = highest conviction.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from core.contracts import (
    BlockReason,
    ExitReason,
    IntentAction,
    PairId,
    RegimeLabel,
    SignalDirection,
    SignalQualityGrade,
    TradeLifecycleState,
)


# ══════════════════════════════════════════════════════════════════
# SUPPORTING ENUMERATIONS
# ══════════════════════════════════════════════════════════════════

class EntryMode(str, enum.Enum):
    """How the entry should be executed."""
    THRESHOLD_CROSS   = "THRESHOLD_CROSS"    # Simple: enter when |z| > entry_z
    CONFIRMED         = "CONFIRMED"          # Wait for N consecutive bars above threshold
    REVERSAL_CONFIRM  = "REVERSAL_CONFIRM"   # Enter after z-score starts retracing
    STAGED            = "STAGED"             # Scale in over multiple bars
    CONSERVATIVE      = "CONSERVATIVE"       # Only enter in calm, high-conviction setups


class ExitMode(str, enum.Enum):
    """How urgently to execute the exit."""
    IMMEDIATE      = "IMMEDIATE"   # Market order; use for stops
    PATIENT        = "PATIENT"     # Work the order over 1-2 bars
    TWAP           = "TWAP"        # Spread execution over longer window
    AGGRESSIVE     = "AGGRESSIVE"  # Lean against the spread; pay up


class SuspendReason(str, enum.Enum):
    """Why trading on this spread is being suspended."""
    REGIME_DETERIORATION   = "REGIME_DETERIORATION"
    DATA_QUALITY_CONCERN   = "DATA_QUALITY_CONCERN"
    STRUCTURAL_BREAK_RISK  = "STRUCTURAL_BREAK_RISK"
    RISK_LIMIT_APPROACH    = "RISK_LIMIT_APPROACH"
    EVENT_CONTAMINATION    = "EVENT_CONTAMINATION"
    MANUAL_OVERRIDE        = "MANUAL_OVERRIDE"
    VOLATILITY_SPIKE       = "VOLATILITY_SPIKE"


class RetireReason(str, enum.Enum):
    """Why this spread is being permanently retired."""
    CONFIRMED_STRUCTURAL_BREAK = "CONFIRMED_STRUCTURAL_BREAK"
    PERSISTENT_REGIME_FAILURE  = "PERSISTENT_REGIME_FAILURE"
    VALIDATION_EXPIRED         = "VALIDATION_EXPIRED"
    LIQUIDITY_DETERIORATION    = "LIQUIDITY_DETERIORATION"
    MANUAL_RETIREMENT          = "MANUAL_RETIREMENT"


# ══════════════════════════════════════════════════════════════════
# BASE INTENT
# ══════════════════════════════════════════════════════════════════

@dataclass
class BaseIntent:
    """Base class for all trade intents.

    Never instantiate directly — use a concrete subclass.
    """
    pair_id: PairId
    action: IntentAction
    confidence: float = 0.0           # [0, 1] signal-engine conviction
    rationale: list[str] = field(default_factory=list)
    block_reasons: list[BlockReason] = field(default_factory=list)
    generated_at: datetime = field(default_factory=datetime.utcnow)
    correlation_id: str = ""          # Link to originating signal run

    @property
    def is_blocked(self) -> bool:
        return len(self.block_reasons) > 0

    @property
    def label(self) -> str:
        return self.pair_id.label

    def to_dict(self) -> dict:
        return {
            "pair": self.label,
            "action": self.action.value,
            "confidence": round(self.confidence, 4),
            "rationale": self.rationale,
            "block_reasons": [r.value for r in self.block_reasons],
            "generated_at": self.generated_at.isoformat(),
            "correlation_id": self.correlation_id,
        }


# ══════════════════════════════════════════════════════════════════
# CONCRETE INTENT TYPES
# ══════════════════════════════════════════════════════════════════

@dataclass
class WatchIntent(BaseIntent):
    """Monitor the spread; no entry action recommended.

    Emitted when the relationship is valid but:
    - signal strength is insufficient
    - regime is unsuitable
    - spread is in cooldown
    - signal quality is too low
    """
    action: IntentAction = IntentAction.WATCH
    watch_reason: str = ""
    next_review_hint: Optional[datetime] = None

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update({"watch_reason": self.watch_reason})
        return d


@dataclass
class EntryIntent(BaseIntent):
    """Propose opening a new position on this spread.

    The downstream layers (portfolio, risk, execution) decide whether
    and how to act on this intent.
    """
    action: IntentAction = IntentAction.ENTER
    direction: SignalDirection = SignalDirection.LONG_SPREAD
    z_score: float = 0.0              # Current z-score that triggered signal
    entry_z_threshold: float = 2.0   # Threshold that was crossed
    exit_z_target: float = 0.5       # Target for exit (mean reversion)
    stop_z: float = 4.0              # Hard stop z-score
    expected_half_life_days: float = 20.0  # Expected holding period
    entry_mode: EntryMode = EntryMode.THRESHOLD_CROSS
    scale_fraction: float = 1.0      # Fraction of target notional (1.0 = full)
    # Signal characteristics
    divergence_velocity: float = 0.0  # How fast z-score is moving away from mean
    is_reversal_confirmed: bool = False  # z-score is turning back toward mean

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update({
            "direction": self.direction.value,
            "z_score": round(self.z_score, 4),
            "entry_z_threshold": self.entry_z_threshold,
            "exit_z_target": self.exit_z_target,
            "stop_z": self.stop_z,
            "expected_half_life_days": self.expected_half_life_days,
            "entry_mode": self.entry_mode.value,
            "scale_fraction": self.scale_fraction,
        })
        return d


@dataclass
class AddIntent(BaseIntent):
    """Propose scaling into an existing position.

    Only valid when lifecycle state is ACTIVE or SCALING_IN.
    """
    action: IntentAction = IntentAction.ADD
    add_fraction: float = 0.25        # Fraction of original notional to add
    add_rationale: str = ""           # e.g. "z-score widened after initial entry"
    new_avg_entry_z: float = 0.0      # Expected avg entry z after scale-in
    max_total_scale: float = 2.0      # Maximum total scale vs original notional

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update({
            "add_fraction": self.add_fraction,
            "add_rationale": self.add_rationale,
            "max_total_scale": self.max_total_scale,
        })
        return d


@dataclass
class HoldIntent(BaseIntent):
    """Recommend maintaining current position without change.

    Emitted during normal position monitoring when:
    - z-score has not yet hit exit target
    - regime remains supportive
    - no risk/time triggers have fired
    """
    action: IntentAction = IntentAction.HOLD
    hold_reason: str = ""
    z_score: float = 0.0
    days_held: int = 0
    max_hold_days: int = 30
    next_review_at: Optional[datetime] = None

    @property
    def days_remaining(self) -> int:
        return max(0, self.max_hold_days - self.days_held)

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update({
            "hold_reason": self.hold_reason,
            "z_score": round(self.z_score, 4),
            "days_held": self.days_held,
            "days_remaining": self.days_remaining,
        })
        return d


@dataclass
class ReduceIntent(BaseIntent):
    """Propose a partial exit (de-risking).

    Used for staged profit-taking, regime weakening, or volatility spikes.
    Multiple exit_reasons may apply.
    """
    action: IntentAction = IntentAction.REDUCE
    reduce_fraction: float = 0.5      # Fraction of current position to close
    exit_reasons: list[ExitReason] = field(default_factory=list)
    exit_mode: ExitMode = ExitMode.PATIENT
    z_score: float = 0.0

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update({
            "reduce_fraction": self.reduce_fraction,
            "exit_reasons": [r.value for r in self.exit_reasons],
            "exit_mode": self.exit_mode.value,
            "z_score": round(self.z_score, 4),
        })
        return d


@dataclass
class ExitIntent(BaseIntent):
    """Propose full exit of a position.

    Always populated with at least one exit_reason.
    exit_mode controls urgency.
    """
    action: IntentAction = IntentAction.EXIT
    exit_reasons: list[ExitReason] = field(default_factory=list)
    exit_mode: ExitMode = ExitMode.PATIENT
    z_score: float = 0.0
    days_held: int = 0
    unrealized_pnl_pct: float = 0.0

    @property
    def is_stop(self) -> bool:
        """True if this is a risk-based stop exit."""
        return ExitReason.ADVERSE_EXCURSION_STOP in self.exit_reasons or \
               ExitReason.SPREAD_STOP in self.exit_reasons or \
               ExitReason.KILL_SWITCH in self.exit_reasons

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update({
            "exit_reasons": [r.value for r in self.exit_reasons],
            "exit_mode": self.exit_mode.value,
            "z_score": round(self.z_score, 4),
            "days_held": self.days_held,
            "unrealized_pnl_pct": round(self.unrealized_pnl_pct, 4),
            "is_stop": self.is_stop,
        })
        return d


@dataclass
class SuspendIntent(BaseIntent):
    """Propose temporarily suspending trading on this spread.

    Suspension is reversible; retirement is permanent.
    """
    action: IntentAction = IntentAction.SUSPEND
    suspend_reason: SuspendReason = SuspendReason.REGIME_DETERIORATION
    suspend_description: str = ""
    duration_days: int = 5            # Expected suspension duration (advisory)
    retry_after: Optional[datetime] = None

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update({
            "suspend_reason": self.suspend_reason.value,
            "suspend_description": self.suspend_description,
            "duration_days": self.duration_days,
            "retry_after": self.retry_after.isoformat() if self.retry_after else None,
        })
        return d


@dataclass
class RetireIntent(BaseIntent):
    """Propose permanent retirement of this spread relationship.

    After retirement, the spread must be re-validated from scratch.
    """
    action: IntentAction = IntentAction.RETIRE
    retire_reason: RetireReason = RetireReason.CONFIRMED_STRUCTURAL_BREAK
    retire_description: str = ""
    revalidation_suggested_after_days: int = 90

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update({
            "retire_reason": self.retire_reason.value,
            "retire_description": self.retire_description,
            "revalidation_suggested_after_days": self.revalidation_suggested_after_days,
        })
        return d


# ══════════════════════════════════════════════════════════════════
# SIGNAL DECISION — WRAPS AN INTENT WITH FULL CONTEXT
# ══════════════════════════════════════════════════════════════════

@dataclass
class SignalDecision:
    """
    Complete output of the signal engine for one pair at one moment.

    Wraps the recommended intent with full observability context:
    regime state, quality grade, lifecycle state, and rationale.

    This is the canonical contract between the signal layer and the
    portfolio / risk / execution layers.
    """
    pair_id: PairId
    intent: BaseIntent
    regime_label: RegimeLabel = RegimeLabel.UNKNOWN
    regime_confidence: float = 0.0
    quality_grade: SignalQualityGrade = SignalQualityGrade.F
    quality_score: float = 0.0       # [0, 1] from signal quality engine
    lifecycle_state: TradeLifecycleState = TradeLifecycleState.WATCHLIST
    conviction: float = 0.0          # [0, 1] combined conviction from signal stack
    z_score: float = 0.0
    half_life_days: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    diagnostics_ref: str = ""         # ID of linked SignalDiagnostics record
    notes: str = ""

    @property
    def action(self) -> IntentAction:
        return self.intent.action

    @property
    def is_entry_proposal(self) -> bool:
        return self.intent.action in (IntentAction.ENTER, IntentAction.ADD)

    @property
    def is_exit_proposal(self) -> bool:
        return self.intent.action in (IntentAction.EXIT, IntentAction.REDUCE)

    @property
    def is_blocked(self) -> bool:
        return self.intent.is_blocked

    def to_dict(self) -> dict:
        return {
            "pair": self.pair_id.label,
            "action": self.intent.action.value,
            "regime": self.regime_label.value,
            "regime_confidence": round(self.regime_confidence, 4),
            "quality_grade": self.quality_grade.value,
            "quality_score": round(self.quality_score, 4),
            "lifecycle_state": self.lifecycle_state.value,
            "conviction": round(self.conviction, 4),
            "z_score": round(self.z_score, 4),
            "half_life_days": round(self.half_life_days, 2),
            "block_reasons": [r.value for r in self.intent.block_reasons],
            "rationale": self.intent.rationale,
            "intent": self.intent.to_dict(),
            "timestamp": self.timestamp.isoformat(),
            "notes": self.notes,
        }


# Convenience type union
AnyIntent = (
    WatchIntent | EntryIntent | AddIntent | HoldIntent |
    ReduceIntent | ExitIntent | SuspendIntent | RetireIntent
)


__all__ = [
    # Supporting enums
    "EntryMode", "ExitMode", "SuspendReason", "RetireReason",
    # Intent types
    "BaseIntent", "WatchIntent", "EntryIntent", "AddIntent",
    "HoldIntent", "ReduceIntent", "ExitIntent", "SuspendIntent", "RetireIntent",
    # Decision wrapper
    "SignalDecision",
]
