# -*- coding: utf-8 -*-
"""
control_plane/contracts.py — Control Plane Domain Contracts
=============================================================

All typed domain objects for the operator control surface.

Design principles:
  - Frozen dataclasses for immutable value objects (actions, records, state snapshots).
  - Enums inherit from str for JSON round-trip compatibility.
  - ThrottleState.sizing_multiplier is derived from ThrottleLevel at construction.
  - stdlib only — no external dependencies.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Optional

from runtime.contracts import ServiceState, ThrottleLevel


# ══════════════════════════════════════════════════════════════════
# 1. ENUMERATIONS
# ══════════════════════════════════════════════════════════════════


class ControlPlaneActionType(str, enum.Enum):
    """All actions that can be dispatched through the control plane engine.

    Each value maps to a dedicated handler method on ControlPlaneEngine.
    """

    ENABLE_STRATEGY = "enable_strategy"
    DISABLE_STRATEGY = "disable_strategy"
    PAUSE_NEW_ENTRIES = "pause_new_entries"
    ALLOW_EXITS_ONLY = "allow_exits_only"
    THROTTLE_SIZING = "throttle_sizing"
    FREEZE_UNIVERSE = "freeze_universe"
    DISABLE_MODEL = "disable_model"
    ENABLE_FALLBACK = "enable_fallback"
    DISABLE_AGENT = "disable_agent"
    APPLY_TEMP_OVERRIDE = "apply_temp_override"
    CLEAR_OVERRIDE = "clear_override"
    FORCE_RECONCILE = "force_reconcile"
    TRIGGER_DRAIN = "trigger_drain"
    TRIGGER_HALT = "trigger_halt"
    RELEASE_HALT = "release_halt"
    INSPECT_STATE = "inspect_state"
    ACTIVATE_RUNTIME = "activate_runtime"
    DEACTIVATE_RUNTIME = "deactivate_runtime"


# ══════════════════════════════════════════════════════════════════
# 2. CORE ACTION / RECORD TYPES
# ══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ControlPlaneAction:
    """Immutable descriptor for a single operator control action.

    Created by callers and passed to ControlPlaneEngine.execute_action().
    Every field is captured verbatim into the resulting ControlPlaneActionRecord
    for full audit traceability.
    """

    action_id: str
    """Unique identifier for this action, e.g. "act_{uuid4().hex[:12]}"."""

    action_type: ControlPlaneActionType
    """What the action does."""

    scope: str
    """Scope of effect: "global" | "strategy:{id}" | "model:{id}" |
    "agent:{name}" | "universe:{id}" | "sleeve:{id}"."""

    value: Optional[str]
    """New value where applicable (throttle level, bool as string, etc.)."""

    reason: str
    """Human-readable justification — required for audit."""

    requested_by: str
    """Operator identifier (username, service account, etc.)."""

    requested_at: str
    """ISO-8601 timestamp when the action was requested."""

    approval_id: Optional[str]
    """Reference to a prior approval record, where required."""

    environment: str
    """Target environment name, e.g. "paper" | "live"."""

    expiry: Optional[str]
    """ISO-8601 expiry for temporary actions.  None = permanent."""

    notes: str = ""
    """Free-text supplementary notes."""


@dataclass
class ControlPlaneActionRecord:
    """Mutable execution record for a ControlPlaneAction.

    Created immediately when the engine begins executing an action.
    Updated with outcome, new_state, and error fields once execution
    completes.  The audit_trail list accumulates step-by-step notes.
    """

    record_id: str
    """Unique execution record ID."""

    action: dict
    """ControlPlaneAction serialised as a plain dict for persistence."""

    executed_at: str
    """ISO-8601 timestamp when execution started."""

    succeeded: bool
    """True if the action completed without error."""

    previous_state: str
    """Serialised representation of the relevant state before the action."""

    new_state: str
    """Serialised representation of the relevant state after the action."""

    error: Optional[str]
    """Error message if succeeded=False."""

    audit_trail: list
    """Ordered list of audit notes appended during execution."""


# ══════════════════════════════════════════════════════════════════
# 3. OPERATIONAL STATE SNAPSHOTS
# ══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class KillSwitchState:
    """Immutable snapshot of the kill-switch state.

    The kill switch is a hard, global stop on all new capital activity.
    It is engaged automatically by risk systems or manually by operators.
    Releasing it requires explicit approval.
    """

    active: bool
    reason: Optional[str]
    triggered_at: Optional[str]     # ISO-8601
    triggered_by: Optional[str]
    scope: str                      # "global" | "env:{name}" | "strategy:{id}"
    release_criteria: str
    approval_required_to_release: bool
    auto_engage_triggers: tuple     # tuple[str, ...]


@dataclass(frozen=True)
class ThrottleState:
    """Immutable snapshot of a sizing throttle applied to a scope.

    The sizing_multiplier is derived deterministically from the ThrottleLevel
    so callers never need to implement the mapping themselves.
    """

    level: ThrottleLevel
    scope: str
    reason: str
    applied_at: str                 # ISO-8601
    applied_by: str
    expires_at: Optional[str]       # ISO-8601; None = permanent
    sizing_multiplier: float
    """Derived from level: NONE=1.0, LIGHT=0.75, MODERATE=0.5,
    HEAVY=0.25, EXITS_ONLY=0.0, HALTED=0.0."""

    @staticmethod
    def multiplier_for(level: ThrottleLevel) -> float:
        """Return the sizing multiplier for a ThrottleLevel."""
        _MAP = {
            ThrottleLevel.NONE: 1.0,
            ThrottleLevel.LIGHT: 0.75,
            ThrottleLevel.MODERATE: 0.5,
            ThrottleLevel.HEAVY: 0.25,
            ThrottleLevel.EXITS_ONLY: 0.0,
            ThrottleLevel.HALTED: 0.0,
        }
        return _MAP.get(level, 0.0)

    @classmethod
    def build(
        cls,
        level: ThrottleLevel,
        scope: str,
        reason: str,
        applied_at: str,
        applied_by: str,
        expires_at: Optional[str] = None,
    ) -> "ThrottleState":
        """Factory that derives sizing_multiplier automatically."""
        return cls(
            level=level,
            scope=scope,
            reason=reason,
            applied_at=applied_at,
            applied_by=applied_by,
            expires_at=expires_at,
            sizing_multiplier=cls.multiplier_for(level),
        )


# ══════════════════════════════════════════════════════════════════
# 4. AUDIT RECORDS
# ══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class OperatorActionRecord:
    """Immutable audit record of a completed operator action.

    Written to the in-memory audit log and persisted (when a store is
    available) after every control-plane action completes.
    """

    record_id: str
    operator: str
    action_type: str
    description: str
    environment: str
    timestamp: str                  # ISO-8601
    outcome: str
    """One of: "succeeded" | "failed" | "blocked" | "pending_approval"."""

    policy_check_id: Optional[str]
    approval_id: Optional[str]
    affected_components: tuple      # tuple[str, ...]
    notes: str = ""


@dataclass(frozen=True)
class HeartbeatRecord:
    """Immutable heartbeat emitted by a named component.

    Used by the monitoring layer to detect stale or missing components.
    Sequence numbers are monotonically increasing per component so
    missed heartbeats can be detected by gaps.
    """

    component: str
    env: str
    heartbeat_at: str               # ISO-8601
    sequence: int
    state: ServiceState
    metadata: dict
    """Component-specific health data (latency, queue depth, error count, etc.)."""
