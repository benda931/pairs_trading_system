# -*- coding: utf-8 -*-
"""
runtime/contracts.py — Runtime State Domain Contracts
=======================================================

All typed domain objects for runtime state management.

Design principles:
  - Frozen dataclasses are immutable value objects; use for configs, decisions, records.
  - Mutable state objects (StrategyActivationRecord, RuntimeState, etc.) are regular
    dataclasses so the state manager can update them in-place.
  - Enums inherit from str for JSON round-trip compatibility.
  - stdlib only — no external dependencies.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Optional


# ══════════════════════════════════════════════════════════════════
# 1. ENUMERATIONS
# ══════════════════════════════════════════════════════════════════


class RuntimeMode(str, enum.Enum):
    """Deployment / execution mode for the trading system.

    Determines which operations are permitted, what data sources are
    reachable, and whether real capital can be committed.
    """

    RESEARCH = "research"
    BACKTEST = "backtest"
    PAPER = "paper"
    SHADOW = "shadow"
    STAGING = "staging"
    LIVE = "live"


class ServiceState(str, enum.Enum):
    """Lifecycle state of an individual system service / component.

    The state machine is not strictly linear; components can move between
    states as the operator or automated health checks trigger transitions.
    """

    INITIALIZING = "initializing"
    WARMING = "warming"
    READY = "ready"
    DEGRADED = "degraded"
    PAUSED = "paused"
    DRAINING = "draining"
    HALTED = "halted"
    RECOVERING = "recovering"
    RECONCILING = "reconciling"
    UNHEALTHY = "unhealthy"
    DISABLED = "disabled"


class ActivationStatus(str, enum.Enum):
    """Lifecycle status of a strategy, model, or agent activation."""

    INACTIVE = "inactive"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    ACTIVATING = "activating"
    ACTIVE = "active"
    PAUSED = "paused"
    DEACTIVATING = "deactivating"
    DISABLED = "disabled"
    FAILED = "failed"


class ThrottleLevel(str, enum.Enum):
    """Sizing throttle levels applied globally or per-scope.

    Translates to a sizing multiplier applied by the portfolio layer.
    EXITS_ONLY and HALTED both result in a 0.0 multiplier for new entries;
    EXITS_ONLY still allows exit orders while HALTED blocks everything.
    """

    NONE = "none"
    LIGHT = "light"           # 75 % normal sizing
    MODERATE = "moderate"     # 50 % normal sizing
    HEAVY = "heavy"           # 25 % normal sizing
    EXITS_ONLY = "exits_only"
    HALTED = "halted"


class EnvironmentRestrictionType(str, enum.Enum):
    """How an action is treated in a given environment."""

    ALLOWED = "allowed"
    BLOCKED = "blocked"
    REQUIRES_APPROVAL = "requires_approval"
    REQUIRES_DUAL_APPROVAL = "requires_dual_approval"


# ══════════════════════════════════════════════════════════════════
# 2. ENVIRONMENT SPEC
# ══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class EnvironmentSpec:
    """Static specification for a named runtime environment.

    Captures all permissions, restrictions, and configuration for one
    environment (e.g. "paper", "live").  Immutable after construction.
    """

    env_name: str
    """Human-readable environment name: "research" | "paper" | "shadow" | "staging" | "live"."""

    runtime_mode: RuntimeMode
    """Canonical RuntimeMode that this environment maps to."""

    allow_live_capital: bool
    """True only for the "live" environment."""

    allow_broker_orders: bool
    """True for paper + live (paper broker orders are permitted in "paper")."""

    allowed_data_sources: tuple
    """Tuple of allowed data-source identifiers, e.g. ("sql", "fmp", "ibkr_paper")."""

    allowed_agent_classes: tuple
    """Tuple of allowed agent-class strings."""

    max_risk_class: str
    """Maximum RiskClass string permitted in this environment."""

    requires_approval_for: tuple
    """Action-type strings that require prior approval in this environment."""

    requires_human_review_above_risk: str
    """Any action at or above this RiskClass requires human review."""

    audit_level: str
    """Verbosity of audit logging: "minimal" | "standard" | "full"."""

    log_level: str
    """Python log level: "DEBUG" | "INFO" | "WARNING" | "ERROR"."""

    max_position_size_usd: float
    """Hard cap on a single position's notional exposure (0.0 = no positions)."""

    kill_switch_auto_engage: bool
    """Whether the kill switch engages automatically on breach conditions."""

    notes: str = ""
    """Optional free-text notes for documentation."""


# ══════════════════════════════════════════════════════════════════
# 3. TRANSITION AND ACTIVATION RECORDS
# ══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class RuntimeTransition:
    """Immutable record of a requested or completed mode transition.

    A transition moves the system from one RuntimeMode to another.
    Policy checks and (optionally) human approval are required before
    the transition completes.
    """

    transition_id: str
    from_mode: RuntimeMode
    to_mode: RuntimeMode
    requested_by: str
    requested_at: str           # ISO-8601
    approved_by: Optional[str]
    approved_at: Optional[str]
    reason: str
    approval_required: bool
    policy_check_passed: bool
    completed: bool = False
    blocked_reason: Optional[str] = None


@dataclass(frozen=True)
class ActivationRequest:
    """Immutable request to activate a named subject (strategy, model, agent, etc.)
    in a specific runtime environment.
    """

    request_id: str
    subject_type: str           # "strategy" | "model" | "agent" | "universe" | "sleeve"
    subject_id: str
    subject_name: str
    target_mode: RuntimeMode
    target_env: str
    requested_by: str
    requested_at: str           # ISO-8601
    policy_check_ids: tuple
    evidence_ids: tuple
    config_version: str
    model_version: Optional[str]
    notes: str = ""


@dataclass(frozen=True)
class ActivationDecision:
    """Immutable decision record for an ActivationRequest.

    Captures whether the request was approved, any conditions attached
    to the approval, and an optional expiry for time-limited activations.
    """

    decision_id: str
    request_id: str
    decided_at: str             # ISO-8601
    decided_by: str
    approved: bool
    conditions: tuple           # tuple[str, ...]
    expiry: Optional[str]       # ISO-8601 — activation auto-expires if set
    notes: str = ""


# ══════════════════════════════════════════════════════════════════
# 4. ACTIVATION RECORDS (mutable, maintained in RuntimeState)
# ══════════════════════════════════════════════════════════════════


@dataclass
class StrategyActivationRecord:
    """Mutable activation record for a trading strategy.

    Maintained in RuntimeState.active_strategies keyed by strategy_id.
    The state manager updates this in-place as the strategy lifecycle
    progresses.
    """

    record_id: str
    strategy_id: str
    strategy_name: str
    env: str
    mode: RuntimeMode
    status: ActivationStatus
    activated_at: Optional[str]
    deactivated_at: Optional[str]
    activation_decision_id: Optional[str]
    config_version: str
    policy_version: str
    throttle_level: ThrottleLevel = ThrottleLevel.NONE
    pause_reason: Optional[str] = None
    notes: str = ""


@dataclass
class ModelActivationRecord:
    """Mutable activation record for an ML model.

    Tracks whether the model is feature-compatible, whether drift checks
    have passed, and whether the fallback inference path is active.
    """

    record_id: str
    model_id: str
    model_name: str
    model_version: str
    env: str
    status: ActivationStatus
    approved_at: Optional[str]
    expires_at: Optional[str]       # stale model auto-expires
    feature_compatibility_ok: bool = True
    drift_ok: bool = True
    fallback_active: bool = False
    notes: str = ""


@dataclass
class AgentActivationRecord:
    """Mutable activation record for a named agent.

    Captures the permission envelope granted at activation time and which
    task types are currently enabled or disabled.
    """

    record_id: str
    agent_name: str
    env: str
    status: ActivationStatus
    permission_envelope: str        # JSON-serialized AgentPermissionSet summary
    enabled_task_types: list
    disabled_task_types: list
    activated_at: Optional[str]
    notes: str = ""


# ══════════════════════════════════════════════════════════════════
# 5. OVERRIDES
# ══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class RuntimeOverride:
    """Immutable descriptor for a single runtime override.

    Overrides are applied by operators (or automated risk systems) to
    temporarily alter the behaviour of a scope.  Every override has an
    audit trail and may carry an expiry after which it is auto-cleared.
    """

    override_id: str
    override_type: str
    """One of: "throttle" | "pause" | "halt" | "freeze_universe" | "disable_model" | "exits_only"."""

    scope: str
    """Scope of effect: "global" | "strategy:{id}" | "sleeve:{id}" | "universe:{id}" | "model:{id}"."""

    value: str
    """New value (throttle level, bool as string, etc.)."""

    reason: str
    applied_by: str
    applied_at: str             # ISO-8601
    expires_at: Optional[str]  # None = permanent until explicitly cleared
    approval_id: Optional[str]
    is_emergency: bool = False


@dataclass(frozen=True)
class OverrideApprovalRecord:
    """Immutable record that an override was explicitly approved.

    Linked to a RuntimeOverride via override_id.  The risk_accepted
    field captures the free-text statement from the approver about which
    risks were reviewed and accepted.
    """

    approval_id: str
    override_id: str
    approved_by: str
    approved_at: str            # ISO-8601
    risk_accepted: str
    conditions: tuple           # tuple[str, ...]


# ══════════════════════════════════════════════════════════════════
# 6. RUNTIME STATE SNAPSHOTS
# ══════════════════════════════════════════════════════════════════


@dataclass
class RuntimeState:
    """The complete observable runtime state.  Single source of truth.

    This object is the "what is actually running right now" view.
    It is compared to DesiredRuntimeState to detect drift.

    All sub-records are stored as plain dicts (serialised from their
    dataclass form) to allow JSON persistence without a custom encoder.
    """

    snapshot_id: str
    """Unique ID for this snapshot, e.g. "snap_{uuid4()}"."""

    captured_at: str
    """ISO-8601 timestamp when the snapshot was taken."""

    env: str
    mode: RuntimeMode
    overall_service_state: ServiceState
    global_throttle: ThrottleLevel
    kill_switch_active: bool
    kill_switch_reason: Optional[str]
    exits_only_mode: bool

    active_strategies: dict
    """strategy_id -> StrategyActivationRecord serialised as dict."""

    active_models: dict
    """model_id -> ModelActivationRecord serialised as dict."""

    active_agents: dict
    """agent_name -> AgentActivationRecord serialised as dict."""

    active_overrides: list
    """list of RuntimeOverride serialised as dicts."""

    component_states: dict
    """component_name -> ServiceState.value string."""

    pending_transitions: list
    """list of RuntimeTransition serialised as dicts."""

    last_updated: str
    """ISO-8601 of the most recent state mutation."""


@dataclass(frozen=True)
class DesiredRuntimeState:
    """Operator's declared intent for what should be running.

    Compared to RuntimeState to detect configuration drift.  The operator
    submits a DesiredRuntimeState; the control plane reconciles any gaps.
    """

    snapshot_id: str
    created_at: str             # ISO-8601
    created_by: str
    env: str
    mode: RuntimeMode
    global_throttle: ThrottleLevel
    enabled_strategies: tuple   # tuple[str, ...]
    enabled_models: tuple       # tuple[str, ...]
    enabled_agents: tuple       # tuple[str, ...]
    config_version: str
    notes: str = ""


# ══════════════════════════════════════════════════════════════════
# 7. PAUSE / DRAIN STATE
# ══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class PauseState:
    """Immutable snapshot of a pause event for a given scope.

    A paused scope stops accepting new entry signals; existing positions
    remain open until the scope is resumed or drained.
    """

    paused: bool
    scope: str
    reason: str
    paused_at: Optional[str]    # ISO-8601
    paused_by: Optional[str]
    resume_criteria: str
    auto_resume_at: Optional[str]


@dataclass(frozen=True)
class DrainState:
    """Immutable snapshot of an active drain operation.

    During a drain, no new entries are allowed; existing positions are
    exited gracefully as opportunities arise.
    """

    draining: bool
    started_at: Optional[str]       # ISO-8601
    initiated_by: Optional[str]
    reason: str
    allow_new_entries: bool = False  # always False during drain
    positions_remaining: int = 0
    estimated_completion: Optional[str] = None


# ══════════════════════════════════════════════════════════════════
# 8. CONFIGURATION SNAPSHOT
# ══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class RuntimeConfigSnapshot:
    """Point-in-time snapshot of all versioned configuration components.

    Used for audit, drift detection, and rollback.  The hash field is a
    SHA-256 digest of the serialised snapshot for integrity verification.
    """

    snapshot_id: str
    captured_at: str            # ISO-8601
    env: str
    config_version: str
    policy_version: str
    model_versions: dict        # component_name -> version string
    feature_set_versions: dict  # feature_group -> version string
    schema_version: str
    hash: str                   # SHA-256 of serialised snapshot
    notes: str = ""


# ══════════════════════════════════════════════════════════════════
# 9. READINESS AND PREFLIGHT REPORTS
# ══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class LiveTradingReadinessReport:
    """Full readiness assessment before activating a target RuntimeMode.

    Generated by ControlPlaneEngine.run_preflight_checks().  The
    recommendation field encodes the engine's overall verdict.
    """

    report_id: str
    generated_at: str           # ISO-8601
    env: str
    mode: RuntimeMode
    overall_ready: bool
    blocking_issues: tuple      # tuple[str, ...]
    warnings: tuple             # tuple[str, ...]
    checks_passed: tuple        # tuple[str, ...]
    checks_failed: tuple        # tuple[str, ...]
    broker_ready: bool
    data_ready: bool
    risk_ready: bool
    models_ready: bool
    config_valid: bool
    reconciliation_clean: bool
    required_approvals_present: bool
    recommendation: str         # "proceed" | "proceed_with_caution" | "halt"


@dataclass(frozen=True)
class PreflightCheckReport:
    """Result of a single preflight check.

    Aggregated into LiveTradingReadinessReport by the engine.
    """

    report_id: str
    generated_at: str           # ISO-8601
    check_name: str
    passed: bool
    details: str
    severity: str               # "critical" | "warning" | "info"
    remediation: str = ""
