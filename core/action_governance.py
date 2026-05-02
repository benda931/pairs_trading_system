# -*- coding: utf-8 -*-
"""
core/action_governance.py — Agent Action Governance Framework
==============================================================

Canonical governance registry for every agent-generated action type.

Every action that `AgentFeedbackLoop` can produce maps to exactly one
`ActionGovernanceProfile` per `TradingEnvironment`. The profile specifies:
- Which governance tier determines the execution pathway
- What evidence fields are required (and how fresh they must be)
- Which `ApprovalMode` to use when routing through the ApprovalEngine
- Cooldown, rollback TTL, incident severity thresholds
- Precision demotion threshold (auto-tighten if agent is wrong too often)

The `GOVERNANCE_REGISTRY` dict is a module-level frozen constant. It is
not configurable at runtime. Any deviation from this institutional standard
must be captured in an `OverrideRecord` via the GovernanceRouter — so the
governance matrix itself is auditable.

Design principles:
- Every action type × every environment has exactly one profile.
- Missing profiles are a startup error (GovernanceRouter validates on init).
- The registry is read-only at runtime.
- `GovernedActionRecord` is the immutable audit artifact for every routing
  decision, whether executed, suppressed, vetoed, or advisory.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Tuple


# ══════════════════════════════════════════════════════════════════
# 1. ENVIRONMENT + TIER ENUMS
# ══════════════════════════════════════════════════════════════════


class TradingEnvironment(str, enum.Enum):
    """Deployment environment in which an action is being evaluated."""

    RESEARCH = "research"
    PAPER = "paper"
    LIVE = "live"


class ActionGovernanceTier(str, enum.Enum):
    """Execution pathway for a governed action.

    ADVISORY_ONLY   — Never executed automatically. Written to audit ledger
                      as a recommendation; surfaced in dashboard/PM brief.
    AUTO_EXECUTABLE — Executes without approval if all evidence and cooldown
                      checks pass. Rollback available within TTL.
    POLICY_GATED    — Routed through ApprovalEngine with mode=POLICY_GATED.
                      Auto-approves if policy function passes; escalates to
                      HUMAN_REQUIRED if policy function fails or throws.
    HUMAN_REQUIRED  — Always creates a HumanReviewTicket. Never auto-executes.
                      SLA breach opens a P2 incident automatically.
    EMERGENCY_ONLY  — Bypasses normal approval only when a circuit-breaker
                      co-signal is independently confirmed AND two distinct
                      agents agree. Always opens a P0 incident. Requires
                      postmortem within 24 hours.
    """

    ADVISORY_ONLY = "advisory_only"
    AUTO_EXECUTABLE = "auto_executable"
    POLICY_GATED = "policy_gated"
    HUMAN_REQUIRED = "human_required"
    EMERGENCY_ONLY = "emergency_only"


# ══════════════════════════════════════════════════════════════════
# 2. EVIDENCE REQUIREMENT
# ══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class EvidenceRequirement:
    """Declares a required or optional evidence field on a FeedbackAction.

    Fields
    ------
    field_name : str
        Key expected in `FeedbackAction.parameters`.
    description : str
        Human-readable explanation of what this field represents.
    required : bool
        If True, absence of this field fails evidence validation and demotes
        the action to ADVISORY_ONLY for this execution.
    max_age_seconds : Optional[int]
        Maximum age of the evidence (checked against the action's
        `evidence_timestamp` if provided). None = no staleness check.
    min_confidence : Optional[float]
        Minimum value if the field is a float confidence score. None = not checked.
    """

    field_name: str
    description: str
    required: bool
    max_age_seconds: Optional[int] = None
    min_confidence: Optional[float] = None


# ══════════════════════════════════════════════════════════════════
# 3. ACTION GOVERNANCE PROFILE
# ══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ActionGovernanceProfile:
    """Complete governance specification for one (action_type, environment) pair.

    Fields
    ------
    action_type : str
        Matches `FeedbackAction.action_type`.
    environment : TradingEnvironment
        The deployment context this profile applies to.
    tier : ActionGovernanceTier
        Execution pathway (see ActionGovernanceTier docstring).
    required_evidence : tuple[EvidenceRequirement, ...]
        Evidence fields that must be present and valid. If any required field
        is absent or stale, the tier is demoted to ADVISORY_ONLY.
    approval_mode_str : str
        ApprovalMode value string to use when routing through ApprovalEngine.
        String rather than enum to avoid a circular import at module load.
    veto_hierarchy : tuple[str, ...]
        Ordered list of veto authorities (most authoritative first).
        Used for documentation and audit; enforcement is in GovernanceRouter.
    cooldown_seconds : int
        Minimum seconds between successive executions of this action type
        on the same target. 0 = no cooldown.
    rollback_available : bool
        Whether a state snapshot is captured before execution to enable rollback.
    rollback_ttl_seconds : int
        Seconds after execution during which rollback is valid. Ignored if
        rollback_available is False.
    incident_severity_on_execution : Optional[str]
        IncidentSeverity value string. If set, the incident manager opens an
        incident of this severity whenever the action executes (live only).
    incident_severity_on_failure : Optional[str]
        IncidentSeverity value string opened on execution failure.
    max_concurrent_pending : int
        Maximum number of this action type that may be pending approval
        simultaneously. Additional requests are suppressed until one resolves.
    recommendation_expiry_seconds : int
        Seconds after which an unexecuted recommendation is considered stale
        and suppressed without execution.
    precision_demotion_threshold : float
        If the agent's rolling 30-day precision on this action type falls
        below this value, the tier is automatically promoted one level
        (e.g. AUTO_EXECUTABLE → POLICY_GATED) by PrecisionDemotionEngine.
    requires_dual_agent_confirmation : bool
        If True, two distinct source_agent names must independently recommend
        this action before it can execute. Used for EMERGENCY_ONLY tier.
    """

    action_type: str
    environment: TradingEnvironment
    tier: ActionGovernanceTier
    required_evidence: Tuple[EvidenceRequirement, ...]
    approval_mode_str: str
    veto_hierarchy: Tuple[str, ...]
    cooldown_seconds: int
    rollback_available: bool
    rollback_ttl_seconds: int
    incident_severity_on_execution: Optional[str]
    incident_severity_on_failure: Optional[str]
    max_concurrent_pending: int
    recommendation_expiry_seconds: int
    precision_demotion_threshold: float
    requires_dual_agent_confirmation: bool = False


# ══════════════════════════════════════════════════════════════════
# 4. GOVERNED ACTION RECORD
# ══════════════════════════════════════════════════════════════════


@dataclass
class GovernedActionRecord:
    """Mutable audit artifact for a single governed action routing decision.

    Written to the AuditLedger BEFORE execution (executed=False), then
    updated in-place after the execution attempt. This ensures the audit
    trail is never incomplete even if the process crashes mid-execution.

    The record is the central observability artifact. Every field that
    matters for compliance, postmortem analysis, or precision tracking
    is captured here.

    Fields
    ------
    action_id : str
        UUID copied from the originating FeedbackAction.
    action_type : str
        Action type string (e.g. "KILL_SWITCH").
    source_agent : str
        Name of the agent that produced this action.
    confirming_agent : Optional[str]
        Second agent name for dual-confirmation actions. None otherwise.
    environment : TradingEnvironment
        Environment in which this action was routed.
    governance_tier : ActionGovernanceTier
        The tier that was applied (may differ from profile default if demoted).
    tier_demoted : bool
        True if the tier was demoted from the profile default (e.g. evidence
        validation failed, precision threshold crossed).
    demotion_reason : Optional[str]
        Explanation for tier demotion. None if not demoted.
    evidence_snapshot : dict
        Full copy of FeedbackAction.parameters at routing time. Immutable
        after construction (GovernanceRouter makes a deepcopy).
    evidence_valid : bool
        True if all required evidence fields passed validation.
    evidence_validation_errors : tuple[str, ...]
        Validation error messages. Empty if evidence_valid is True.
    evidence_age_seconds : int
        Age of the evidence in seconds at routing time. -1 if not determinable.
    is_stale : bool
        True if evidence exceeded max_age_seconds.
    suppressed : bool
        True if the action was not routed to execution or approval.
    suppression_reason : Optional[str]
        "STALE_EVIDENCE" | "DUPLICATE" | "CONFLICT_DEFERRED" | "MAX_CONCURRENT" | None.
    conflict_with_action_id : Optional[str]
        action_id of the conflicting action that caused this one to be deferred.
    approval_request_id : Optional[str]
        ID of the ApprovalRequest submitted for POLICY_GATED / HUMAN_REQUIRED.
    approval_status : Optional[str]
        ApprovalStatus value at last check. None if no approval was submitted.
    override_record_id : Optional[str]
        ID of OverrideRecord if an emergency bypass was used.
    executed : bool
        True if the action executor was called.
    execution_timestamp : Optional[datetime]
        UTC timestamp of execution. None if not executed.
    execution_result : str
        Result string from the executor. Empty if not executed.
    execution_error : Optional[str]
        Exception message if execution threw. None on clean execution.
    rollback_handle_id : Optional[str]
        ID in the RollbackRegistry for this execution's state snapshot.
    rollback_deadline : Optional[datetime]
        UTC timestamp after which rollback is no longer available.
    incident_id : Optional[str]
        ID of the IncidentRecord opened for this action, if any.
    outcome_observed : bool
        True once the ActionObserver has recorded an outcome.
    outcome_observation_deadline : Optional[datetime]
        UTC timestamp by which outcome should be observed.
    outcome_correct : Optional[bool]
        True if the action's outcome was evaluated as correct. None = not yet.
    regret_score : Optional[float]
        0.0 = no regret, 1.0 = maximum regret. None = not yet evaluated.
    created_at : datetime
        UTC timestamp when this record was created.
    expires_at : datetime
        UTC timestamp after which the recommendation is considered stale.
    """

    action_id: str
    action_type: str
    source_agent: str
    confirming_agent: Optional[str]
    environment: TradingEnvironment
    governance_tier: ActionGovernanceTier
    tier_demoted: bool
    demotion_reason: Optional[str]
    evidence_snapshot: dict
    evidence_valid: bool
    evidence_validation_errors: Tuple[str, ...]
    evidence_age_seconds: int
    is_stale: bool
    suppressed: bool
    suppression_reason: Optional[str]
    conflict_with_action_id: Optional[str]
    approval_request_id: Optional[str]
    approval_status: Optional[str]
    override_record_id: Optional[str]
    executed: bool
    execution_timestamp: Optional[datetime]
    execution_result: str
    execution_error: Optional[str]
    rollback_handle_id: Optional[str]
    rollback_deadline: Optional[datetime]
    incident_id: Optional[str]
    outcome_observed: bool
    outcome_observation_deadline: Optional[datetime]
    outcome_correct: Optional[bool]
    regret_score: Optional[float]
    created_at: datetime
    expires_at: datetime


# ══════════════════════════════════════════════════════════════════
# 5. GOVERNANCE REGISTRY
# ══════════════════════════════════════════════════════════════════

# Shorthand evidence requirement constructors
def _ev(field: str, desc: str, required: bool = True,
        max_age: Optional[int] = None, min_conf: Optional[float] = None
        ) -> EvidenceRequirement:
    return EvidenceRequirement(
        field_name=field,
        description=desc,
        required=required,
        max_age_seconds=max_age,
        min_confidence=min_conf,
    )


# ── Common evidence bundles ────────────────────────────────────────

_REGIME_EVIDENCE = (
    _ev("regime", "Current regime label (e.g. CRISIS, BROKEN, NORMAL)", max_age=300),
)

_DRAWDOWN_EVIDENCE = (
    _ev("drawdown", "Current portfolio drawdown as a negative float (e.g. -0.12)", max_age=1200),
)

_PAIR_EVIDENCE = (
    _ev("pair_id", "Target pair identifier", max_age=None),
    _ev("quality_grade", "Signal quality grade A-F", max_age=1800),
    _ev("z_score", "Current z-score of the spread", max_age=1800),
)

_LEVERAGE_EVIDENCE = (
    _ev("current_leverage", "Current gross portfolio leverage", max_age=1200),
    _ev("target_leverage_multiplier", "Target deleverage multiplier 0-1", max_age=None),
)

_MODEL_EVIDENCE = (
    _ev("model_id", "Model identifier being evaluated for retraining", max_age=None),
    _ev("champion_oos_ic", "Champion model OOS information coefficient", max_age=604800),  # 7d
    _ev("challenger_oos_ic", "Challenger model OOS information coefficient", max_age=604800),
    _ev("purged_cv_score", "Purged cross-validation stability score 0-1", max_age=604800),
    _ev("days_since_last_retrain", "Days since the previous retrain", max_age=None),
)

_THRESHOLD_EVIDENCE = (
    _ev("threshold_name", "Name of the threshold parameter being adjusted", max_age=None),
    _ev("current_value", "Current threshold value", max_age=None),
    _ev("proposed_value", "Proposed new threshold value", max_age=None),
    _ev("ic_rolling_30d", "30-day rolling IC supporting the change", max_age=172800),  # 48h
)

_CONFIG_EVIDENCE = (
    _ev("config_key", "Configuration key being modified", max_age=None),
    _ev("old_value", "Current value of the config key", max_age=None),
    _ev("new_value", "Proposed new value", max_age=None),
    _ev("impact_class", "Impact classification: TRADING | OPERATIONAL | DISPLAY", max_age=None),
)

_PAUSE_EVIDENCE = (
    _ev("trigger_type", "Pause trigger: DATA_QUALITY | SYSTEM_HEALTH | MARKET_CONDITION", max_age=300),
    _ev("affected_components", "List of components that will be paused", required=False, max_age=None),
)


# ── Registry: (action_type, environment) → ActionGovernanceProfile ─

def _p(
    action_type: str,
    env: TradingEnvironment,
    tier: ActionGovernanceTier,
    evidence: Tuple[EvidenceRequirement, ...],
    approval_mode: str,
    veto: Tuple[str, ...],
    cooldown: int,
    rollback: bool,
    rollback_ttl: int,
    inc_exec: Optional[str],
    inc_fail: Optional[str],
    max_pend: int,
    expiry: int,
    precision_floor: float,
    dual: bool = False,
) -> ActionGovernanceProfile:
    return ActionGovernanceProfile(
        action_type=action_type,
        environment=env,
        tier=tier,
        required_evidence=evidence,
        approval_mode_str=approval_mode,
        veto_hierarchy=veto,
        cooldown_seconds=cooldown,
        rollback_available=rollback,
        rollback_ttl_seconds=rollback_ttl,
        incident_severity_on_execution=inc_exec,
        incident_severity_on_failure=inc_fail,
        max_concurrent_pending=max_pend,
        recommendation_expiry_seconds=expiry,
        precision_demotion_threshold=precision_floor,
        requires_dual_agent_confirmation=dual,
    )


R = TradingEnvironment.RESEARCH
P = TradingEnvironment.PAPER
L = TradingEnvironment.LIVE

T = ActionGovernanceTier

GOVERNANCE_REGISTRY: dict[tuple[str, TradingEnvironment], ActionGovernanceProfile] = {

    # ── KILL_SWITCH ────────────────────────────────────────────────
    ("KILL_SWITCH", R): _p(
        "KILL_SWITCH", R, T.AUTO_EXECUTABLE,
        evidence=_DRAWDOWN_EVIDENCE + _REGIME_EVIDENCE,
        approval_mode="AUTOMATIC",
        veto=("pm",),
        cooldown=86400, rollback=False, rollback_ttl=0,
        inc_exec=None, inc_fail="P2",
        max_pend=1, expiry=600, precision_floor=0.60,
    ),
    ("KILL_SWITCH", P): _p(
        "KILL_SWITCH", P, T.AUTO_EXECUTABLE,
        evidence=_DRAWDOWN_EVIDENCE + _REGIME_EVIDENCE,
        approval_mode="AUTOMATIC",
        veto=("pm",),
        cooldown=86400, rollback=False, rollback_ttl=0,
        inc_exec=None, inc_fail="P2",
        max_pend=1, expiry=600, precision_floor=0.60,
    ),
    ("KILL_SWITCH", L): _p(
        "KILL_SWITCH", L, T.EMERGENCY_ONLY,
        evidence=_DRAWDOWN_EVIDENCE + _REGIME_EVIDENCE + (
            _ev("portfolio_snapshot", "Full portfolio state at decision time", max_age=300),
            _ev("estimated_loss_if_not_acted", "Estimated P&L loss (float) if action not taken",
                required=False, max_age=300),
        ),
        approval_mode="HUMAN_REQUIRED",
        veto=("pm", "risk_committee", "system"),
        cooldown=86400, rollback=False, rollback_ttl=0,
        inc_exec="P0", inc_fail="P1",
        max_pend=1, expiry=600, precision_floor=0.70,
        dual=True,
    ),

    # ── FORCE_EXIT ─────────────────────────────────────────────────
    ("FORCE_EXIT", R): _p(
        "FORCE_EXIT", R, T.AUTO_EXECUTABLE,
        evidence=_PAIR_EVIDENCE,
        approval_mode="AUTOMATIC",
        veto=("pm",),
        cooldown=7200, rollback=False, rollback_ttl=0,
        inc_exec=None, inc_fail="P3",
        max_pend=10, expiry=1800, precision_floor=0.60,
    ),
    ("FORCE_EXIT", P): _p(
        "FORCE_EXIT", P, T.POLICY_GATED,
        evidence=_PAIR_EVIDENCE + (
            _ev("stop_loss_trigger", "Type of stop-loss trigger: HARD_STOP | REGIME | SIGNAL_DECAY",
                max_age=900),
        ),
        approval_mode="POLICY_GATED",
        veto=("pm",),
        cooldown=7200, rollback=False, rollback_ttl=0,
        inc_exec=None, inc_fail="P3",
        max_pend=10, expiry=1800, precision_floor=0.60,
    ),
    ("FORCE_EXIT", L): _p(
        "FORCE_EXIT", L, T.HUMAN_REQUIRED,
        evidence=_PAIR_EVIDENCE + (
            _ev("current_pnl", "Current unrealised P&L on position (float)", max_age=900),
            _ev("stop_loss_trigger", "Type of stop-loss trigger", max_age=900),
            _ev("liquidity_score", "Spread liquidity score 0-1", required=False, max_age=3600),
        ),
        approval_mode="HUMAN_REQUIRED",
        veto=("pm", "risk_committee"),
        cooldown=7200, rollback=False, rollback_ttl=0,
        inc_exec="P2", inc_fail="P1",
        max_pend=5, expiry=1800, precision_floor=0.65,
    ),

    # ── BLOCK_ENTRY ────────────────────────────────────────────────
    ("BLOCK_ENTRY", R): _p(
        "BLOCK_ENTRY", R, T.AUTO_EXECUTABLE,
        evidence=_PAIR_EVIDENCE,
        approval_mode="AUTOMATIC",
        veto=("risk_guardian",),
        cooldown=3600, rollback=True, rollback_ttl=14400,
        inc_exec=None, inc_fail="P4",
        max_pend=20, expiry=7200, precision_floor=0.55,
    ),
    ("BLOCK_ENTRY", P): _p(
        "BLOCK_ENTRY", P, T.AUTO_EXECUTABLE,
        evidence=_PAIR_EVIDENCE,
        approval_mode="AUTOMATIC",
        veto=("risk_guardian",),
        cooldown=3600, rollback=True, rollback_ttl=14400,
        inc_exec=None, inc_fail="P4",
        max_pend=20, expiry=7200, precision_floor=0.55,
    ),
    ("BLOCK_ENTRY", L): _p(
        "BLOCK_ENTRY", L, T.POLICY_GATED,
        evidence=_PAIR_EVIDENCE + _REGIME_EVIDENCE,
        approval_mode="POLICY_GATED",
        veto=("risk_guardian", "signal_quality_gate"),
        cooldown=3600, rollback=True, rollback_ttl=14400,
        inc_exec=None, inc_fail="P3",
        max_pend=20, expiry=7200, precision_floor=0.65,
    ),

    # ── DELEVERAGE ─────────────────────────────────────────────────
    ("DELEVERAGE", R): _p(
        "DELEVERAGE", R, T.AUTO_EXECUTABLE,
        evidence=_LEVERAGE_EVIDENCE + _DRAWDOWN_EVIDENCE,
        approval_mode="AUTOMATIC",
        veto=("portfolio_construction",),
        cooldown=21600, rollback=True, rollback_ttl=172800,
        inc_exec=None, inc_fail="P3",
        max_pend=1, expiry=2700, precision_floor=0.60,
    ),
    ("DELEVERAGE", P): _p(
        "DELEVERAGE", P, T.AUTO_EXECUTABLE,
        evidence=_LEVERAGE_EVIDENCE + _DRAWDOWN_EVIDENCE,
        approval_mode="AUTOMATIC",
        veto=("portfolio_construction",),
        cooldown=21600, rollback=True, rollback_ttl=172800,
        inc_exec=None, inc_fail="P3",
        max_pend=1, expiry=2700, precision_floor=0.60,
    ),
    ("DELEVERAGE", L): _p(
        "DELEVERAGE", L, T.POLICY_GATED,   # <20% delev; router upgrades to HUMAN_REQUIRED if >20%
        evidence=_LEVERAGE_EVIDENCE + _DRAWDOWN_EVIDENCE + _REGIME_EVIDENCE,
        approval_mode="POLICY_GATED",
        veto=("portfolio_construction", "risk_guardian", "pm"),
        cooldown=21600, rollback=True, rollback_ttl=172800,
        inc_exec=None, inc_fail="P2",
        max_pend=1, expiry=2700, precision_floor=0.65,
    ),

    # ── ADJUST_THRESHOLD ───────────────────────────────────────────
    ("ADJUST_THRESHOLD", R): _p(
        "ADJUST_THRESHOLD", R, T.AUTO_EXECUTABLE,
        evidence=_THRESHOLD_EVIDENCE,
        approval_mode="AUTOMATIC",
        veto=("auto_improve_agent",),
        cooldown=86400, rollback=True, rollback_ttl=604800,
        inc_exec=None, inc_fail="P4",
        max_pend=3, expiry=259200, precision_floor=0.55,
    ),
    ("ADJUST_THRESHOLD", P): _p(
        "ADJUST_THRESHOLD", P, T.AUTO_EXECUTABLE,
        evidence=_THRESHOLD_EVIDENCE,
        approval_mode="AUTOMATIC",
        veto=("auto_improve_agent",),
        cooldown=86400, rollback=True, rollback_ttl=604800,
        inc_exec=None, inc_fail="P4",
        max_pend=3, expiry=259200, precision_floor=0.55,
    ),
    ("ADJUST_THRESHOLD", L): _p(
        "ADJUST_THRESHOLD", L, T.POLICY_GATED,  # ≤±15%; router → ADVISORY if >±15%
        evidence=_THRESHOLD_EVIDENCE,
        approval_mode="POLICY_GATED",
        veto=("auto_improve_agent",),
        cooldown=86400, rollback=True, rollback_ttl=604800,
        inc_exec=None, inc_fail="P3",
        max_pend=3, expiry=259200, precision_floor=0.65,
    ),

    # ── RETRAIN_MODEL ──────────────────────────────────────────────
    ("RETRAIN_MODEL", R): _p(
        "RETRAIN_MODEL", R, T.AUTO_EXECUTABLE,
        evidence=_MODEL_EVIDENCE,
        approval_mode="AUTOMATIC",
        veto=("quality_gate",),
        cooldown=604800, rollback=True, rollback_ttl=7776000,  # 90d snapshot
        inc_exec=None, inc_fail="P3",
        max_pend=1, expiry=1209600, precision_floor=0.55,      # 14d expiry
    ),
    ("RETRAIN_MODEL", P): _p(
        "RETRAIN_MODEL", P, T.POLICY_GATED,
        evidence=_MODEL_EVIDENCE,
        approval_mode="POLICY_GATED",
        veto=("quality_gate", "stability_check"),
        cooldown=604800, rollback=True, rollback_ttl=7776000,
        inc_exec=None, inc_fail="P2",
        max_pend=1, expiry=1209600, precision_floor=0.60,
    ),
    ("RETRAIN_MODEL", L): _p(
        "RETRAIN_MODEL", L, T.HUMAN_REQUIRED,
        evidence=_MODEL_EVIDENCE + (
            _ev("training_data_days", "Number of training days used", max_age=None),
            _ev("feature_set_hash", "Hash of the feature set for reproducibility", max_age=None),
        ),
        approval_mode="HUMAN_REQUIRED",
        veto=("quality_gate", "stability_check", "pm"),
        cooldown=2592000, rollback=True, rollback_ttl=7776000,  # 30d cooldown live
        inc_exec=None, inc_fail="P2",
        max_pend=1, expiry=1209600, precision_floor=0.65,
    ),

    # ── OPTIMIZE_PARAMS ────────────────────────────────────────────
    ("OPTIMIZE_PARAMS", R): _p(
        "OPTIMIZE_PARAMS", R, T.AUTO_EXECUTABLE,
        evidence=(
            _ev("parameter_set", "Name of the parameter set being optimised", max_age=None),
            _ev("convergence_score", "Bayesian optimisation convergence metric", max_age=604800),
        ),
        approval_mode="AUTOMATIC",
        veto=("methodology_agent",),
        cooldown=172800, rollback=True, rollback_ttl=2592000,
        inc_exec=None, inc_fail="P4",
        max_pend=2, expiry=604800, precision_floor=0.55,
    ),
    ("OPTIMIZE_PARAMS", P): _p(
        "OPTIMIZE_PARAMS", P, T.POLICY_GATED,
        evidence=(
            _ev("parameter_set", "Name of the parameter set", max_age=None),
            _ev("convergence_score", "Optimisation convergence metric", max_age=604800),
            _ev("oos_validation_score", "OOS validation result", max_age=604800),
        ),
        approval_mode="POLICY_GATED",
        veto=("methodology_agent",),
        cooldown=172800, rollback=True, rollback_ttl=2592000,
        inc_exec=None, inc_fail="P3",
        max_pend=2, expiry=604800, precision_floor=0.60,
    ),
    ("OPTIMIZE_PARAMS", L): _p(
        "OPTIMIZE_PARAMS", L, T.ADVISORY_ONLY,  # major; router → POLICY_GATED if ≤5% delta
        evidence=(
            _ev("parameter_set", "Name of the parameter set", max_age=None),
            _ev("convergence_score", "Optimisation convergence metric", max_age=604800),
            _ev("oos_validation_score", "OOS validation result", max_age=604800),
            _ev("param_delta_pct", "Maximum percentage change across all params", max_age=None),
        ),
        approval_mode="POLICY_GATED",
        veto=("methodology_agent", "pm"),
        cooldown=172800, rollback=True, rollback_ttl=2592000,
        inc_exec=None, inc_fail="P3",
        max_pend=2, expiry=604800, precision_floor=0.65,
    ),

    # ── UPDATE_CONFIG ──────────────────────────────────────────────
    ("UPDATE_CONFIG", R): _p(
        "UPDATE_CONFIG", R, T.AUTO_EXECUTABLE,
        evidence=_CONFIG_EVIDENCE,
        approval_mode="AUTOMATIC",
        veto=("pm",),
        cooldown=86400, rollback=True, rollback_ttl=2592000,
        inc_exec=None, inc_fail="P3",
        max_pend=5, expiry=172800, precision_floor=0.55,
    ),
    ("UPDATE_CONFIG", P): _p(
        "UPDATE_CONFIG", P, T.POLICY_GATED,
        evidence=_CONFIG_EVIDENCE,
        approval_mode="POLICY_GATED",
        veto=("risk_guardian", "pm"),
        cooldown=86400, rollback=True, rollback_ttl=2592000,
        inc_exec=None, inc_fail="P2",
        max_pend=5, expiry=172800, precision_floor=0.60,
    ),
    ("UPDATE_CONFIG", L): _p(
        # TRADING config → HUMAN_REQUIRED; OPERATIONAL config → POLICY_GATED.
        # GovernanceRouter reads impact_class from evidence and upgrades tier when needed.
        "UPDATE_CONFIG", L, T.POLICY_GATED,
        evidence=_CONFIG_EVIDENCE,
        approval_mode="POLICY_GATED",
        veto=("risk_guardian", "pm"),
        cooldown=86400, rollback=True, rollback_ttl=2592000,
        inc_exec=None, inc_fail="P2",
        max_pend=5, expiry=172800, precision_floor=0.65,
    ),

    # ── PAUSE_PIPELINE ─────────────────────────────────────────────
    ("PAUSE_PIPELINE", R): _p(
        "PAUSE_PIPELINE", R, T.AUTO_EXECUTABLE,
        evidence=_PAUSE_EVIDENCE,
        approval_mode="AUTOMATIC",
        veto=("data_scout",),
        cooldown=0, rollback=False, rollback_ttl=0,
        inc_exec=None, inc_fail="P3",
        max_pend=1, expiry=900, precision_floor=0.50,
    ),
    ("PAUSE_PIPELINE", P): _p(
        "PAUSE_PIPELINE", P, T.AUTO_EXECUTABLE,
        evidence=_PAUSE_EVIDENCE,
        approval_mode="AUTOMATIC",
        veto=("data_scout",),
        cooldown=0, rollback=False, rollback_ttl=0,
        inc_exec=None, inc_fail="P3",
        max_pend=1, expiry=900, precision_floor=0.50,
    ),
    ("PAUSE_PIPELINE", L): _p(
        # <4h → POLICY_GATED; ≥4h or all-strategy → HUMAN_REQUIRED.
        # GovernanceRouter reads estimated_duration_hours from evidence to upgrade.
        "PAUSE_PIPELINE", L, T.POLICY_GATED,
        evidence=_PAUSE_EVIDENCE + (
            _ev("estimated_duration_hours", "Estimated pause duration in hours",
                required=False, max_age=None),
        ),
        approval_mode="POLICY_GATED",
        veto=("data_scout", "operations", "pm"),
        cooldown=0, rollback=False, rollback_ttl=0,
        inc_exec=None, inc_fail="P2",
        max_pend=1, expiry=900, precision_floor=0.55,
    ),
}


# ══════════════════════════════════════════════════════════════════
# 6. REGISTRY HELPERS
# ══════════════════════════════════════════════════════════════════


def get_profile(
    action_type: str,
    environment: TradingEnvironment,
) -> ActionGovernanceProfile:
    """Return the governance profile for (action_type, environment).

    Raises
    ------
    KeyError
        If no profile is registered for the given combination.
    """
    key = (action_type, environment)
    profile = GOVERNANCE_REGISTRY.get(key)
    if profile is None:
        raise KeyError(
            "No governance profile registered for action_type='{}' "
            "environment='{}'. Register it in GOVERNANCE_REGISTRY.".format(
                action_type, environment.value
            )
        )
    return profile


def validate_registry_completeness() -> list[str]:
    """Check that all expected action types have profiles for all three environments.

    Returns a list of missing (action_type, environment) strings.
    Call at application startup to catch configuration gaps early.
    """
    expected_actions = [
        "KILL_SWITCH", "FORCE_EXIT", "BLOCK_ENTRY", "DELEVERAGE",
        "ADJUST_THRESHOLD", "RETRAIN_MODEL", "OPTIMIZE_PARAMS",
        "UPDATE_CONFIG", "PAUSE_PIPELINE",
    ]
    missing = []
    for action in expected_actions:
        for env in TradingEnvironment:
            if (action, env) not in GOVERNANCE_REGISTRY:
                missing.append("({}, {})".format(action, env.value))
    return missing
