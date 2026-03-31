# -*- coding: utf-8 -*-
"""
governance/contracts.py — Governance Domain Contracts
=======================================================

Frozen dataclasses and enumerations for the governance layer:
policy check results, guardrail violations, policy versions,
artifact references, change impact reports, and promotion reviews.

Design principles:
- All types are frozen dataclasses (immutable after construction).
- Enums inherit from str for JSON-serialization compatibility.
- No circular imports — this module depends only on stdlib.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Optional, Tuple


# ══════════════════════════════════════════════════════════════════
# 1. ENUMERATIONS
# ══════════════════════════════════════════════════════════════════


class PolicyStatus(str, enum.Enum):
    """Lifecycle status of a governance policy version."""

    ACTIVE = "ACTIVE"
    DEPRECATED = "DEPRECATED"
    DRAFT = "DRAFT"
    SUSPENDED = "SUSPENDED"


class PolicyViolationSeverity(str, enum.Enum):
    """Severity level of a policy check result or guardrail violation.

    Ordered from least to most severe:
        INFO < WARNING < VIOLATION < CRITICAL < EMERGENCY
    """

    INFO = "INFO"
    WARNING = "WARNING"
    VIOLATION = "VIOLATION"
    CRITICAL = "CRITICAL"
    EMERGENCY = "EMERGENCY"

    def __lt__(self, other: "PolicyViolationSeverity") -> bool:
        _order = ["INFO", "WARNING", "VIOLATION", "CRITICAL", "EMERGENCY"]
        return _order.index(self.value) < _order.index(other.value)

    def __le__(self, other: "PolicyViolationSeverity") -> bool:
        return self == other or self < other

    def __gt__(self, other: "PolicyViolationSeverity") -> bool:
        return not self <= other

    def __ge__(self, other: "PolicyViolationSeverity") -> bool:
        return not self < other


# ══════════════════════════════════════════════════════════════════
# 2. POLICY CHECK RESULT
# ══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class PolicyCheckResult:
    """Result of evaluating a single governance policy against an agent action.

    Fields
    ------
    check_id : str
        UUID string uniquely identifying this check event.
    policy_name : str
        Human-readable name of the policy evaluated.
    policy_version : str
        Version string of the policy (e.g. "1.0", "2.1").
    agent_name : str
        Name of the agent whose action was checked.
    task_id : str
        ID of the AgentTask involved.
    action_type : str
        Semantic type of the action being checked.
    passed : bool
        True if the action passes the policy; False if it violates it.
    severity : PolicyViolationSeverity
        Severity of the result (relevant when passed=False).
    message : str
        Human-readable description of the check outcome.
    details : tuple[str, ...]
        Ordered list of detail strings (rule evaluations, data points checked).
    timestamp : str
        ISO 8601 timestamp of the check.
    remediation_hint : str
        Optional suggestion for how to resolve a failed check.
    """

    check_id: str
    policy_name: str
    policy_version: str
    agent_name: str
    task_id: str
    action_type: str
    passed: bool
    severity: PolicyViolationSeverity
    message: str
    details: Tuple[str, ...]
    timestamp: str
    remediation_hint: str = ""


# ══════════════════════════════════════════════════════════════════
# 3. GUARDRAIL VIOLATION
# ══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class GuardrailViolation:
    """A recorded violation of a hard or soft guardrail policy.

    Violations are append-only records — they are never deleted.
    ``blocked=True`` indicates the action was halted; ``blocked=False``
    means a warning was raised but execution continued.

    Fields
    ------
    violation_id : str
        UUID string uniquely identifying this violation.
    policy_name : str
        Name of the policy that was violated.
    agent_name : str
        Agent whose action triggered the violation.
    task_id : str
        ID of the associated AgentTask.
    workflow_run_id : Optional[str]
        Workflow run involved, if applicable.
    description : str
        Description of what was violated.
    severity : PolicyViolationSeverity
        Severity of the violation.
    blocked : bool
        Whether the action was blocked (True) or merely warned (False).
    timestamp : str
        ISO 8601 timestamp of the violation.
    remediation : str
        Recommended remediation action.
    """

    violation_id: str
    policy_name: str
    agent_name: str
    task_id: str
    workflow_run_id: Optional[str]
    description: str
    severity: PolicyViolationSeverity
    blocked: bool
    timestamp: str
    remediation: str


# ══════════════════════════════════════════════════════════════════
# 4. GOVERNANCE POLICY VERSION
# ══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class GovernancePolicyVersion:
    """A versioned, authored governance policy with approval metadata.

    Policy versions are immutable once published. To revise a policy,
    create a new version and deprecate the old one.

    Fields
    ------
    policy_id : str
        Unique stable identifier for this policy (e.g. "NO_RISK_LIMIT_OVERRIDE").
    policy_name : str
        Human-readable name.
    version : str
        Semantic version string (e.g. "1.0", "2.1.3").
    description : str
        Full description of the policy's intent and scope.
    status : PolicyStatus
        Lifecycle status.
    effective_from : str
        ISO date string when the policy takes effect.
    effective_until : Optional[str]
        ISO date string when the policy expires, if bounded.
    author : str
        Identity of the policy author.
    approved_by : Optional[str]
        Identity of the approver, if formally approved.
    rules : tuple[str, ...]
        Machine-readable rule identifiers or human-readable rule statements.
    risk_class : str
        Risk class this policy governs (e.g. "ALL", "HIGH", "SENSITIVE").
    environment_scope : tuple[str, ...]
        Environments where this policy is active (e.g. ("production",)).
    last_reviewed : str
        ISO date string of last review.
    next_review : Optional[str]
        ISO date string of next scheduled review.
    notes : str
        Optional free-text notes.
    """

    policy_id: str
    policy_name: str
    version: str
    description: str
    status: PolicyStatus
    effective_from: str
    effective_until: Optional[str]
    author: str
    approved_by: Optional[str]
    rules: Tuple[str, ...]
    risk_class: str
    environment_scope: Tuple[str, ...]
    last_reviewed: str
    next_review: Optional[str]
    notes: str = ""


# ══════════════════════════════════════════════════════════════════
# 5. ARTIFACT REFERENCE
# ══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ArtifactReference:
    """A lightweight reference to a produced artifact with provenance metadata.

    Used by governance workflows to track evidence bundles, model outputs,
    recommendations, and other artifacts without embedding full payloads.

    Fields
    ------
    ref_id : str
        UUID string for this reference.
    artifact_type : str
        Semantic type (e.g. "EVIDENCE_BUNDLE", "MODEL_OUTPUT", "RECOMMENDATION").
    artifact_id : str
        ID of the referenced artifact in its own store.
    producing_agent : str
        Name of the agent that produced the artifact.
    produced_at : str
        ISO 8601 timestamp.
    schema_version : str
        Schema version of the artifact (for forward compatibility).
    is_stale : bool
        Whether the artifact is considered stale relative to current system state.
    staleness_reason : Optional[str]
        Explanation of why the artifact is stale, if applicable.
    access_class : str
        Access classification (e.g. "PUBLIC", "INTERNAL", "RESTRICTED").
    """

    ref_id: str
    artifact_type: str
    artifact_id: str
    producing_agent: str
    produced_at: str
    schema_version: str
    is_stale: bool
    staleness_reason: Optional[str]
    access_class: str


# ══════════════════════════════════════════════════════════════════
# 6. CHANGE IMPACT REPORT
# ══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ChangeImpactReport:
    """Structured impact assessment for a proposed system change.

    Change impact reports are produced before executing any change that
    affects multiple components, workflows, or agents, and are attached
    to ApprovalRequests when approval is required.

    Fields
    ------
    report_id : str
        UUID string for this report.
    change_description : str
        Full description of the proposed change.
    proposed_by : str
        Identity of the proposer.
    proposed_at : str
        ISO 8601 timestamp.
    affected_components : tuple[str, ...]
        Identifiers of system components affected.
    affected_workflows : tuple[str, ...]
        Identifiers of workflow types affected.
    affected_agents : tuple[str, ...]
        Names of agents affected.
    risk_class : str
        Overall risk classification of this change.
    estimated_impact : str
        Human-readable assessment of expected impact.
    reversibility : str
        "REVERSIBLE" | "PARTIALLY_REVERSIBLE" | "IRREVERSIBLE".
    requires_approval : bool
        Whether formal approval is required before executing.
    rollback_plan : str
        Description of the rollback procedure if the change must be undone.
    notes : str
        Optional additional notes.
    """

    report_id: str
    change_description: str
    proposed_by: str
    proposed_at: str
    affected_components: Tuple[str, ...]
    affected_workflows: Tuple[str, ...]
    affected_agents: Tuple[str, ...]
    risk_class: str
    estimated_impact: str
    reversibility: str
    requires_approval: bool
    rollback_plan: str
    notes: str = ""


# ══════════════════════════════════════════════════════════════════
# 7. PROMOTION REVIEW RECORD
# ══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class PromotionReviewRecord:
    """Record of a governance review for promoting a subject (e.g. model, policy).

    Links the subject under review, the evaluating reviewer, the criteria
    evaluated, and the resulting decision with conditions.

    Fields
    ------
    review_id : str
        UUID string for this review.
    subject_type : str
        Type of subject being reviewed (e.g. "MODEL", "POLICY", "AGENT").
    subject_id : str
        Unique identifier of the subject.
    subject_name : str
        Human-readable name of the subject.
    reviewed_by : str
        Identity of the reviewer.
    reviewed_at : str
        ISO 8601 timestamp.
    current_status : str
        Current status of the subject before review (e.g. "CANDIDATE").
    proposed_status : str
        Proposed status if promotion is approved (e.g. "CHAMPION").
    evidence_bundle_ids : tuple[str, ...]
        Evidence bundles consulted during the review.
    criteria_passed : tuple[str, ...]
        Promotion criteria that were met.
    criteria_failed : tuple[str, ...]
        Promotion criteria that were not met.
    decision : str
        "APPROVED" | "REJECTED" | "DEFERRED".
    rationale : str
        Explanation for the decision.
    conditions : tuple[str, ...]
        Conditions attached to an APPROVED decision.
    approval_request_id : Optional[str]
        Linked ApprovalRequest, if formal approval was requested.
    """

    review_id: str
    subject_type: str
    subject_id: str
    subject_name: str
    reviewed_by: str
    reviewed_at: str
    current_status: str
    proposed_status: str
    evidence_bundle_ids: Tuple[str, ...]
    criteria_passed: Tuple[str, ...]
    criteria_failed: Tuple[str, ...]
    decision: str
    rationale: str
    conditions: Tuple[str, ...]
    approval_request_id: Optional[str]
