# -*- coding: utf-8 -*-
"""
core/lifecycle.py — Trade Lifecycle State Machine
=================================================

Explicit, validated state machine for spread trading.

Rules:
  1. All state transitions are validated; invalid transitions raise ValueError.
  2. Every transition is logged with a trigger and rationale.
  3. Cooldown state is enforced: no re-entry until cooldown expires.
  4. Each state defines allowed actions, forbidden actions, and optional timeout.
  5. Suspension and retirement are differentiated (reversible vs. permanent).

State flow (typical):
    WATCHLIST → SETUP_FORMING → ENTRY_READY → PENDING_ENTRY → ACTIVE
                                                                ↓       ↓
                                                         SCALING_IN  REDUCING
                                                                ↓
                         COOLDOWN ←──────── EXIT_READY ←── ACTIVE
                            ↓
                     WATCHLIST (normal re-entry)
                     SUSPENDED (regime/risk issue)
                     RETIRED   (permanent)

Special paths:
    Any state → SUSPENDED (regime deterioration, manual block)
    SUSPENDED → WATCHLIST (after suspension resolves)
    Any state → RETIRED   (confirmed structural break)
    NOT_ELIGIBLE → WATCHLIST (after re-validation)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from core.contracts import (
    BlockReason,
    ExitReason,
    IntentAction,
    PairId,
    TradeLifecycleState,
)
from core.diagnostics import LifecycleDiagnostics, LifecycleTransitionRecord
from core.intents import (
    BaseIntent,
    EntryIntent,
    ExitIntent,
    HoldIntent,
    ReduceIntent,
    RetireIntent,
    RetireReason,
    SuspendIntent,
    SuspendReason,
    WatchIntent,
)

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════
# STATE POLICY — DEFINES WHAT'S ALLOWED IN EACH STATE
# ══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class StatePolicy:
    """Policy for a lifecycle state.

    allowed_actions: set of IntentActions the state machine will accept.
    forbidden_actions: explicitly disallowed (for clarity in audit logs).
    max_time_days: optional timeout; triggers on_timeout_action if exceeded.
    on_timeout_action: state to transition to if max_time_days exceeded.
    """
    state: TradeLifecycleState
    allowed_actions: frozenset
    forbidden_actions: frozenset
    max_time_days: Optional[float] = None
    on_timeout_action: Optional[TradeLifecycleState] = None


_POLICIES: dict[TradeLifecycleState, StatePolicy] = {
    TradeLifecycleState.NOT_ELIGIBLE: StatePolicy(
        state=TradeLifecycleState.NOT_ELIGIBLE,
        allowed_actions=frozenset(),
        forbidden_actions=frozenset({IntentAction.ENTER, IntentAction.ADD}),
        max_time_days=None,
    ),
    TradeLifecycleState.WATCHLIST: StatePolicy(
        state=TradeLifecycleState.WATCHLIST,
        allowed_actions=frozenset({IntentAction.WATCH, IntentAction.SUSPEND, IntentAction.RETIRE}),
        forbidden_actions=frozenset({IntentAction.ENTER, IntentAction.ADD, IntentAction.HOLD,
                                     IntentAction.REDUCE, IntentAction.EXIT}),
        max_time_days=None,
    ),
    TradeLifecycleState.SETUP_FORMING: StatePolicy(
        state=TradeLifecycleState.SETUP_FORMING,
        allowed_actions=frozenset({IntentAction.WATCH, IntentAction.SUSPEND, IntentAction.RETIRE}),
        forbidden_actions=frozenset({IntentAction.ADD, IntentAction.REDUCE, IntentAction.EXIT}),
        max_time_days=10.0,
        on_timeout_action=TradeLifecycleState.WATCHLIST,
    ),
    TradeLifecycleState.ENTRY_READY: StatePolicy(
        state=TradeLifecycleState.ENTRY_READY,
        allowed_actions=frozenset({IntentAction.ENTER, IntentAction.WATCH,
                                    IntentAction.SUSPEND, IntentAction.RETIRE}),
        forbidden_actions=frozenset({IntentAction.ADD, IntentAction.REDUCE,
                                     IntentAction.HOLD, IntentAction.EXIT}),
        max_time_days=3.0,   # Entry-ready signal decays quickly
        on_timeout_action=TradeLifecycleState.WATCHLIST,
    ),
    TradeLifecycleState.PENDING_ENTRY: StatePolicy(
        state=TradeLifecycleState.PENDING_ENTRY,
        allowed_actions=frozenset({IntentAction.SUSPEND, IntentAction.RETIRE}),
        forbidden_actions=frozenset({IntentAction.ENTER, IntentAction.ADD,
                                     IntentAction.REDUCE, IntentAction.EXIT}),
        max_time_days=1.0,   # Pending execution should resolve within 1 day
        on_timeout_action=TradeLifecycleState.WATCHLIST,
    ),
    TradeLifecycleState.ACTIVE: StatePolicy(
        state=TradeLifecycleState.ACTIVE,
        allowed_actions=frozenset({IntentAction.HOLD, IntentAction.ADD, IntentAction.REDUCE,
                                    IntentAction.EXIT, IntentAction.SUSPEND, IntentAction.RETIRE}),
        forbidden_actions=frozenset({IntentAction.ENTER, IntentAction.WATCH}),
        max_time_days=None,
    ),
    TradeLifecycleState.SCALING_IN: StatePolicy(
        state=TradeLifecycleState.SCALING_IN,
        allowed_actions=frozenset({IntentAction.HOLD, IntentAction.REDUCE,
                                    IntentAction.EXIT, IntentAction.SUSPEND}),
        forbidden_actions=frozenset({IntentAction.ENTER, IntentAction.WATCH, IntentAction.ADD}),
        max_time_days=2.0,
        on_timeout_action=TradeLifecycleState.ACTIVE,
    ),
    TradeLifecycleState.REDUCING: StatePolicy(
        state=TradeLifecycleState.REDUCING,
        allowed_actions=frozenset({IntentAction.EXIT, IntentAction.HOLD, IntentAction.SUSPEND}),
        forbidden_actions=frozenset({IntentAction.ENTER, IntentAction.ADD, IntentAction.WATCH}),
        max_time_days=2.0,
        on_timeout_action=TradeLifecycleState.ACTIVE,
    ),
    TradeLifecycleState.EXIT_READY: StatePolicy(
        state=TradeLifecycleState.EXIT_READY,
        allowed_actions=frozenset({IntentAction.EXIT, IntentAction.SUSPEND}),
        forbidden_actions=frozenset({IntentAction.ENTER, IntentAction.ADD,
                                     IntentAction.HOLD, IntentAction.WATCH}),
        max_time_days=1.0,
        on_timeout_action=TradeLifecycleState.EXIT_READY,  # stays until executed
    ),
    TradeLifecycleState.PENDING_EXIT: StatePolicy(
        state=TradeLifecycleState.PENDING_EXIT,
        allowed_actions=frozenset(),
        forbidden_actions=frozenset({IntentAction.ENTER, IntentAction.ADD}),
        max_time_days=1.0,
        on_timeout_action=TradeLifecycleState.COOLDOWN,
    ),
    TradeLifecycleState.COOLDOWN: StatePolicy(
        state=TradeLifecycleState.COOLDOWN,
        allowed_actions=frozenset({IntentAction.WATCH, IntentAction.SUSPEND, IntentAction.RETIRE}),
        forbidden_actions=frozenset({IntentAction.ENTER, IntentAction.ADD}),
        max_time_days=None,   # Cooldown duration controlled by CooldownPolicy
    ),
    TradeLifecycleState.SUSPENDED: StatePolicy(
        state=TradeLifecycleState.SUSPENDED,
        allowed_actions=frozenset({IntentAction.WATCH, IntentAction.RETIRE}),
        forbidden_actions=frozenset({IntentAction.ENTER, IntentAction.ADD,
                                     IntentAction.HOLD, IntentAction.EXIT}),
        max_time_days=30.0,
        on_timeout_action=TradeLifecycleState.WATCHLIST,  # Auto-resume if not retired
    ),
    TradeLifecycleState.RETIRED: StatePolicy(
        state=TradeLifecycleState.RETIRED,
        allowed_actions=frozenset(),
        forbidden_actions=frozenset({IntentAction.ENTER, IntentAction.ADD, IntentAction.EXIT}),
        max_time_days=None,
    ),
}


# ══════════════════════════════════════════════════════════════════
# VALID TRANSITION TABLE
# ══════════════════════════════════════════════════════════════════

# Maps (from_state, trigger) → to_state
# Trigger is a short string label for the event that caused the transition.
# Use transition_trigger constants below.

TRIGGER_SIGNAL_FORMING     = "signal_forming"
TRIGGER_ENTRY_READY        = "entry_ready"
TRIGGER_ENTRY_SUBMITTED    = "entry_submitted"
TRIGGER_ENTRY_FILLED       = "entry_filled"
TRIGGER_ENTRY_FAILED       = "entry_failed"
TRIGGER_SCALE_IN           = "scale_in"
TRIGGER_SCALE_COMPLETE     = "scale_complete"
TRIGGER_REDUCE             = "reduce"
TRIGGER_REDUCE_COMPLETE    = "reduce_complete"
TRIGGER_EXIT_SIGNAL        = "exit_signal"
TRIGGER_EXIT_SUBMITTED     = "exit_submitted"
TRIGGER_EXIT_FILLED        = "exit_filled"
TRIGGER_COOLDOWN_EXPIRED   = "cooldown_expired"
TRIGGER_SUSPEND            = "suspend"
TRIGGER_RESUME             = "resume"
TRIGGER_RETIRE             = "retire"
TRIGGER_REVALIDATE         = "revalidate"
TRIGGER_TIMEOUT            = "timeout"
TRIGGER_SIGNAL_DECAY       = "signal_decay"

_VALID_TRANSITIONS: dict[tuple, TradeLifecycleState] = {
    # ── Discovery / setup ─────────────────────────────────────────
    (TradeLifecycleState.NOT_ELIGIBLE,    TRIGGER_REVALIDATE):       TradeLifecycleState.WATCHLIST,
    (TradeLifecycleState.WATCHLIST,       TRIGGER_SIGNAL_FORMING):   TradeLifecycleState.SETUP_FORMING,
    (TradeLifecycleState.SETUP_FORMING,   TRIGGER_ENTRY_READY):      TradeLifecycleState.ENTRY_READY,
    (TradeLifecycleState.SETUP_FORMING,   TRIGGER_SIGNAL_DECAY):     TradeLifecycleState.WATCHLIST,
    (TradeLifecycleState.SETUP_FORMING,   TRIGGER_TIMEOUT):          TradeLifecycleState.WATCHLIST,
    # ── Entry ─────────────────────────────────────────────────────
    (TradeLifecycleState.ENTRY_READY,     TRIGGER_ENTRY_SUBMITTED):  TradeLifecycleState.PENDING_ENTRY,
    (TradeLifecycleState.ENTRY_READY,     TRIGGER_SIGNAL_DECAY):     TradeLifecycleState.WATCHLIST,
    (TradeLifecycleState.ENTRY_READY,     TRIGGER_TIMEOUT):          TradeLifecycleState.WATCHLIST,
    (TradeLifecycleState.PENDING_ENTRY,   TRIGGER_ENTRY_FILLED):     TradeLifecycleState.ACTIVE,
    (TradeLifecycleState.PENDING_ENTRY,   TRIGGER_ENTRY_FAILED):     TradeLifecycleState.WATCHLIST,
    (TradeLifecycleState.PENDING_ENTRY,   TRIGGER_TIMEOUT):          TradeLifecycleState.WATCHLIST,
    # ── Active management ─────────────────────────────────────────
    (TradeLifecycleState.ACTIVE,          TRIGGER_SCALE_IN):         TradeLifecycleState.SCALING_IN,
    (TradeLifecycleState.ACTIVE,          TRIGGER_REDUCE):           TradeLifecycleState.REDUCING,
    (TradeLifecycleState.ACTIVE,          TRIGGER_EXIT_SIGNAL):      TradeLifecycleState.EXIT_READY,
    (TradeLifecycleState.SCALING_IN,      TRIGGER_SCALE_COMPLETE):   TradeLifecycleState.ACTIVE,
    (TradeLifecycleState.SCALING_IN,      TRIGGER_EXIT_SIGNAL):      TradeLifecycleState.EXIT_READY,
    (TradeLifecycleState.SCALING_IN,      TRIGGER_TIMEOUT):          TradeLifecycleState.ACTIVE,
    (TradeLifecycleState.REDUCING,        TRIGGER_REDUCE_COMPLETE):  TradeLifecycleState.ACTIVE,
    (TradeLifecycleState.REDUCING,        TRIGGER_EXIT_SIGNAL):      TradeLifecycleState.EXIT_READY,
    (TradeLifecycleState.REDUCING,        TRIGGER_TIMEOUT):          TradeLifecycleState.ACTIVE,
    # ── Exit ──────────────────────────────────────────────────────
    (TradeLifecycleState.EXIT_READY,      TRIGGER_EXIT_SUBMITTED):   TradeLifecycleState.PENDING_EXIT,
    (TradeLifecycleState.EXIT_READY,      TRIGGER_TIMEOUT):          TradeLifecycleState.EXIT_READY,
    (TradeLifecycleState.PENDING_EXIT,    TRIGGER_EXIT_FILLED):      TradeLifecycleState.COOLDOWN,
    (TradeLifecycleState.PENDING_EXIT,    TRIGGER_TIMEOUT):          TradeLifecycleState.COOLDOWN,
    # ── Cooldown / suspension ─────────────────────────────────────
    (TradeLifecycleState.COOLDOWN,        TRIGGER_COOLDOWN_EXPIRED): TradeLifecycleState.WATCHLIST,
    (TradeLifecycleState.SUSPENDED,       TRIGGER_RESUME):           TradeLifecycleState.WATCHLIST,
    (TradeLifecycleState.SUSPENDED,       TRIGGER_TIMEOUT):          TradeLifecycleState.WATCHLIST,
    # ── Universal suspend paths ───────────────────────────────────
    **{(s, TRIGGER_SUSPEND): TradeLifecycleState.SUSPENDED
       for s in (TradeLifecycleState.WATCHLIST, TradeLifecycleState.SETUP_FORMING,
                 TradeLifecycleState.ENTRY_READY, TradeLifecycleState.ACTIVE,
                 TradeLifecycleState.SCALING_IN, TradeLifecycleState.REDUCING,
                 TradeLifecycleState.EXIT_READY, TradeLifecycleState.COOLDOWN)},
    # ── Universal retire paths ────────────────────────────────────
    **{(s, TRIGGER_RETIRE): TradeLifecycleState.RETIRED
       for s in (TradeLifecycleState.WATCHLIST, TradeLifecycleState.SETUP_FORMING,
                 TradeLifecycleState.ENTRY_READY, TradeLifecycleState.ACTIVE,
                 TradeLifecycleState.SUSPENDED, TradeLifecycleState.COOLDOWN,
                 TradeLifecycleState.NOT_ELIGIBLE)},
}


# ══════════════════════════════════════════════════════════════════
# COOLDOWN POLICY
# ══════════════════════════════════════════════════════════════════

@dataclass
class CooldownPolicy:
    """Controls how long cooldown lasts after an exit."""
    default_days: int = 3
    after_stop_days: int = 7       # Longer cooldown after adverse stop
    after_regime_exit_days: int = 5
    after_break_exit_days: int = 14  # Very long after structural break
    max_days: int = 30

    def get_cooldown_days(self, exit_reasons: list[ExitReason]) -> int:
        if not exit_reasons:
            return self.default_days
        if ExitReason.STRUCTURAL_BREAK in exit_reasons or ExitReason.BREAK_RISK in exit_reasons:
            return min(self.after_break_exit_days, self.max_days)
        if ExitReason.ADVERSE_EXCURSION_STOP in exit_reasons or ExitReason.SPREAD_STOP in exit_reasons:
            return min(self.after_stop_days, self.max_days)
        if ExitReason.REGIME_FLIP in exit_reasons or ExitReason.REGIME_WEAKENING in exit_reasons:
            return min(self.after_regime_exit_days, self.max_days)
        return self.default_days


# ══════════════════════════════════════════════════════════════════
# LIFECYCLE STATE MACHINE
# ══════════════════════════════════════════════════════════════════

class TradeLifecycleStateMachine:
    """
    Explicit state machine for a single pair spread.

    Responsibilities:
    - Validate and execute state transitions
    - Enforce cooldown intervals
    - Log all transitions with trigger and rationale
    - Provide policy queries (what actions are allowed, etc.)

    One state machine per pair. Create, persist, and restore.
    """

    def __init__(
        self,
        pair_id: PairId,
        initial_state: TradeLifecycleState = TradeLifecycleState.WATCHLIST,
        cooldown_policy: Optional[CooldownPolicy] = None,
    ) -> None:
        self.pair_id = pair_id
        self._state = initial_state
        self._state_entered_at = datetime.utcnow()
        self._cooldown_policy = cooldown_policy or CooldownPolicy()
        self._diagnostics = LifecycleDiagnostics(
            pair_id=pair_id,
            current_state=initial_state,
            state_entered_at=self._state_entered_at,
        )
        self._logger = logging.getLogger(f"lifecycle.{pair_id.label}")

    # ── State access ──────────────────────────────────────────────

    @property
    def state(self) -> TradeLifecycleState:
        return self._state

    @property
    def policy(self) -> StatePolicy:
        return _POLICIES[self._state]

    @property
    def diagnostics(self) -> LifecycleDiagnostics:
        return self._diagnostics

    def is_action_allowed(self, action: IntentAction) -> bool:
        return action in self.policy.allowed_actions

    def is_in_cooldown(self) -> bool:
        return self._diagnostics.is_in_cooldown

    def cooldown_days_remaining(self) -> float:
        return self._diagnostics.cooldown_days_remaining

    # ── Timeout check ─────────────────────────────────────────────

    def check_timeout(self) -> Optional[TradeLifecycleState]:
        """If the state has a timeout and it's exceeded, return the next state."""
        pol = self.policy
        if pol.max_time_days is None or pol.on_timeout_action is None:
            return None
        days_in_state = self._diagnostics.time_in_current_state_days
        if days_in_state >= pol.max_time_days:
            return pol.on_timeout_action
        return None

    def apply_timeout_if_needed(self) -> bool:
        """Apply timeout transition if exceeded. Returns True if transitioned."""
        next_state = self.check_timeout()
        if next_state is not None and next_state != self._state:
            self._do_transition(next_state, TRIGGER_TIMEOUT,
                                f"State timeout after {self.policy.max_time_days} days")
            return True
        return False

    # ── Primary transition interface ──────────────────────────────

    def transition(
        self,
        trigger: str,
        rationale: str = "",
        *,
        exit_reasons: Optional[list[ExitReason]] = None,
    ) -> TradeLifecycleState:
        """
        Execute a validated state transition.

        Parameters
        ----------
        trigger : str
            One of the TRIGGER_* constants defined in this module.
        rationale : str
            Human-readable explanation for audit log.
        exit_reasons : optional list[ExitReason]
            Provided when trigger == TRIGGER_EXIT_FILLED to set cooldown length.

        Returns
        -------
        New state after transition.

        Raises
        ------
        ValueError
            If the transition is not in the valid transition table.
        """
        key = (self._state, trigger)
        next_state = _VALID_TRANSITIONS.get(key)

        if next_state is None:
            raise ValueError(
                f"Invalid lifecycle transition for {self.pair_id.label}: "
                f"{self._state.value} → trigger='{trigger}'. "
                f"Valid triggers from this state: "
                f"{[t for (s, t) in _VALID_TRANSITIONS if s == self._state]}"
            )

        # Set cooldown duration when entering COOLDOWN state
        if next_state == TradeLifecycleState.COOLDOWN:
            days = self._cooldown_policy.get_cooldown_days(exit_reasons or [])
            self._diagnostics.cooldown_until = datetime.utcnow() + timedelta(days=days)
            self._logger.info(
                "%s: entering COOLDOWN for %d days (reasons: %s)",
                self.pair_id.label, days, [r.value for r in (exit_reasons or [])]
            )

        self._do_transition(next_state, trigger, rationale)
        return self._state

    def _do_transition(
        self,
        next_state: TradeLifecycleState,
        trigger: str,
        rationale: str,
    ) -> None:
        old_state = self._state
        now = datetime.utcnow()
        record = LifecycleTransitionRecord(
            from_state=old_state,
            to_state=next_state,
            trigger=trigger,
            timestamp=now,
            rationale=rationale,
        )
        self._state = next_state
        self._state_entered_at = now
        self._diagnostics.current_state = next_state
        self._diagnostics.state_entered_at = now
        self._diagnostics.transitions.append(record)

        if next_state == TradeLifecycleState.SUSPENDED:
            self._diagnostics.suspension_count += 1
        if old_state == TradeLifecycleState.PENDING_ENTRY and next_state == TradeLifecycleState.ACTIVE:
            self._diagnostics.total_trades += 1

        self._logger.info(
            "%s: %s → %s (trigger=%s) %s",
            self.pair_id.label, old_state.value, next_state.value,
            trigger, f"| {rationale}" if rationale else "",
        )

    # ── Convenience transition helpers ────────────────────────────

    def on_signal_forming(self, rationale: str = "") -> TradeLifecycleState:
        return self.transition(TRIGGER_SIGNAL_FORMING, rationale)

    def on_entry_ready(self, rationale: str = "") -> TradeLifecycleState:
        return self.transition(TRIGGER_ENTRY_READY, rationale)

    def on_entry_submitted(self, rationale: str = "") -> TradeLifecycleState:
        return self.transition(TRIGGER_ENTRY_SUBMITTED, rationale)

    def on_entry_filled(self, rationale: str = "") -> TradeLifecycleState:
        return self.transition(TRIGGER_ENTRY_FILLED, rationale)

    def on_entry_failed(self, rationale: str = "") -> TradeLifecycleState:
        self._diagnostics.failed_entry_attempts += 1
        return self.transition(TRIGGER_ENTRY_FAILED, rationale)

    def on_exit_signal(self, rationale: str = "") -> TradeLifecycleState:
        return self.transition(TRIGGER_EXIT_SIGNAL, rationale)

    def on_exit_filled(
        self, exit_reasons: Optional[list[ExitReason]] = None, rationale: str = ""
    ) -> TradeLifecycleState:
        return self.transition(TRIGGER_EXIT_FILLED, rationale, exit_reasons=exit_reasons)

    def on_suspend(
        self,
        reason: SuspendReason = SuspendReason.REGIME_DETERIORATION,
        rationale: str = "",
    ) -> TradeLifecycleState:
        return self.transition(TRIGGER_SUSPEND, rationale or reason.value)

    def on_retire(self, rationale: str = "") -> TradeLifecycleState:
        return self.transition(TRIGGER_RETIRE, rationale)

    def on_resume(self, rationale: str = "") -> TradeLifecycleState:
        return self.transition(TRIGGER_RESUME, rationale)

    def on_cooldown_expired(self) -> TradeLifecycleState:
        if self._state != TradeLifecycleState.COOLDOWN:
            raise ValueError(f"Cannot expire cooldown from state {self._state.value}")
        if self.is_in_cooldown():
            raise ValueError(
                f"Cooldown has not expired yet ({self.cooldown_days_remaining():.1f} days remaining)"
            )
        return self.transition(TRIGGER_COOLDOWN_EXPIRED, "Cooldown period expired")

    def on_revalidated(self, rationale: str = "") -> TradeLifecycleState:
        """Called when a NOT_ELIGIBLE spread is re-validated successfully."""
        return self.transition(TRIGGER_REVALIDATE, rationale)

    def on_signal_decay(self, rationale: str = "") -> TradeLifecycleState:
        return self.transition(TRIGGER_SIGNAL_DECAY, rationale)

    # ── State queries ─────────────────────────────────────────────

    def can_enter(self) -> tuple[bool, list[BlockReason]]:
        """Can a new position be entered? Returns (allowed, reasons_if_blocked)."""
        reasons: list[BlockReason] = []
        if self._state == TradeLifecycleState.RETIRED:
            reasons.append(BlockReason.RETIRED)
        elif self._state == TradeLifecycleState.SUSPENDED:
            reasons.append(BlockReason.SUSPENDED)
        elif self._state == TradeLifecycleState.COOLDOWN:
            if self.is_in_cooldown():
                reasons.append(BlockReason.COOLDOWN_ACTIVE)
        elif self._state == TradeLifecycleState.PENDING_ENTRY:
            reasons.append(BlockReason.PENDING_EXECUTION)
        elif self._state not in (TradeLifecycleState.ENTRY_READY, TradeLifecycleState.WATCHLIST):
            if self._state == TradeLifecycleState.NOT_ELIGIBLE:
                reasons.append(BlockReason.NOT_VALIDATED)

        return len(reasons) == 0, reasons

    def can_add(self) -> tuple[bool, list[BlockReason]]:
        allowed = self._state in (TradeLifecycleState.ACTIVE, TradeLifecycleState.SCALING_IN)
        return allowed, [] if allowed else [BlockReason.PENDING_EXECUTION]

    def is_position_active(self) -> bool:
        return self._state in (
            TradeLifecycleState.ACTIVE,
            TradeLifecycleState.SCALING_IN,
            TradeLifecycleState.REDUCING,
            TradeLifecycleState.EXIT_READY,
            TradeLifecycleState.PENDING_EXIT,
        )

    def __repr__(self) -> str:
        return (
            f"TradeLifecycleStateMachine("
            f"pair={self.pair_id.label}, "
            f"state={self._state.value}, "
            f"cooldown={self.is_in_cooldown()})"
        )


# ══════════════════════════════════════════════════════════════════
# LIFECYCLE REGISTRY — MANAGES STATE MACHINES FOR A UNIVERSE
# ══════════════════════════════════════════════════════════════════

class LifecycleRegistry:
    """Manages lifecycle state machines for all active pairs."""

    def __init__(self, cooldown_policy: Optional[CooldownPolicy] = None) -> None:
        self._machines: dict[str, TradeLifecycleStateMachine] = {}
        self._policy = cooldown_policy or CooldownPolicy()

    def get_or_create(self, pair_id: PairId) -> TradeLifecycleStateMachine:
        label = pair_id.label
        if label not in self._machines:
            self._machines[label] = TradeLifecycleStateMachine(
                pair_id, cooldown_policy=self._policy
            )
        return self._machines[label]

    def get(self, pair_id: PairId) -> Optional[TradeLifecycleStateMachine]:
        return self._machines.get(pair_id.label)

    def get_by_state(self, state: TradeLifecycleState) -> list[TradeLifecycleStateMachine]:
        return [m for m in self._machines.values() if m.state == state]

    def active_pairs(self) -> list[PairId]:
        return [m.pair_id for m in self._machines.values() if m.is_position_active()]

    def apply_all_timeouts(self) -> list[tuple[PairId, TradeLifecycleState]]:
        """Check all machines for timeout and apply if needed. Returns list of (pair, new_state)."""
        transitioned = []
        for machine in self._machines.values():
            if machine.apply_timeout_if_needed():
                transitioned.append((machine.pair_id, machine.state))
        return transitioned

    def state_summary(self) -> dict[str, int]:
        """Count of pairs in each state."""
        counts: dict[str, int] = {}
        for m in self._machines.values():
            key = m.state.value
            counts[key] = counts.get(key, 0) + 1
        return counts


__all__ = [
    "StatePolicy",
    "CooldownPolicy",
    "TradeLifecycleStateMachine",
    "LifecycleRegistry",
    # Trigger constants
    "TRIGGER_SIGNAL_FORMING", "TRIGGER_ENTRY_READY", "TRIGGER_ENTRY_SUBMITTED",
    "TRIGGER_ENTRY_FILLED", "TRIGGER_ENTRY_FAILED", "TRIGGER_SCALE_IN",
    "TRIGGER_SCALE_COMPLETE", "TRIGGER_REDUCE", "TRIGGER_REDUCE_COMPLETE",
    "TRIGGER_EXIT_SIGNAL", "TRIGGER_EXIT_SUBMITTED", "TRIGGER_EXIT_FILLED",
    "TRIGGER_COOLDOWN_EXPIRED", "TRIGGER_SUSPEND", "TRIGGER_RESUME",
    "TRIGGER_RETIRE", "TRIGGER_REVALIDATE", "TRIGGER_TIMEOUT", "TRIGGER_SIGNAL_DECAY",
]
