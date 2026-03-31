# -*- coding: utf-8 -*-
"""
approvals/contracts.py — Approval Domain Contracts
====================================================

Frozen dataclasses and enums representing the full approval lifecycle:
requests, decisions, overrides, human review tickets, and escalations.

All types are immutable frozen dataclasses to prevent accidental mutation
across thread boundaries. The ApprovalEngine operates on copies or
reconstructed objects when state must change.

Design principles:
- Every action requiring human or policy gate produces an ApprovalRequest.
- Every completed review produces an ApprovalDecision.
- Any override of a decision is captured in an OverrideRecord.
- Human-required reviews produce a HumanReviewTicket for the review queue.
- Escalations link back to the originating request and optionally an incident.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Optional, Tuple


# ══════════════════════════════════════════════════════════════════
# 1. ENUMERATIONS
# ══════════════════════════════════════════════════════════════════


class ApprovalStatus(str, enum.Enum):
    """Lifecycle status of an approval request or decision."""

    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    ESCALATED = "ESCALATED"
    EXPIRED = "EXPIRED"
    AUTO_APPROVED = "AUTO_APPROVED"


class ReviewPriority(str, enum.Enum):
    """Priority level for a human review ticket or escalation."""

    ROUTINE = "ROUTINE"
    ELEVATED = "ELEVATED"
    URGENT = "URGENT"
    CRITICAL = "CRITICAL"


class ApprovalMode(str, enum.Enum):
    """Controls how an approval request is adjudicated.

    AUTOMATIC      — No human or policy check needed; auto-approve.
    POLICY_GATED   — Policy engine decides; no human required unless policy escalates.
    HUMAN_REQUIRED — Must route to a human reviewer; cannot be auto-approved.
    DUAL_APPROVAL  — Two separate human approvers required.
    BLOCKED        — Action is categorically forbidden; never approve.
    """

    AUTOMATIC = "AUTOMATIC"
    POLICY_GATED = "POLICY_GATED"
    HUMAN_REQUIRED = "HUMAN_REQUIRED"
    DUAL_APPROVAL = "DUAL_APPROVAL"
    BLOCKED = "BLOCKED"


# ══════════════════════════════════════════════════════════════════
# 2. APPROVAL REQUEST
# ══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ApprovalRequest:
    """A formal request for approval before an agent action is executed.

    Fields
    ------
    request_id : str
        UUID string uniquely identifying this request.
    workflow_run_id : Optional[str]
        ID of the workflow run that triggered this request, if any.
    agent_name : str
        Name of the agent submitting the request.
    task_id : str
        ID of the AgentTask associated with this action.
    action_type : str
        Semantic action type (e.g. "MODEL_PROMOTE", "POSITION_SIZE_OVERRIDE").
    action_description : str
        Human-readable description of the action being requested.
    risk_class : str
        Risk classification of the action (e.g. "LOW", "MEDIUM", "HIGH", "SENSITIVE").
    environment : str
        Target environment (e.g. "research", "staging", "production").
    evidence_bundle_ids : tuple[str, ...]
        IDs of evidence bundles supporting this request.
    recommendation_id : Optional[str]
        ID of a recommendation artifact, if the action follows a recommendation.
    requested_by : str
        Identity of the requesting agent or user.
    requested_at : str
        ISO 8601 timestamp when the request was submitted.
    expires_at : Optional[str]
        ISO 8601 timestamp after which the request is considered expired.
    approval_mode : ApprovalMode
        How this request should be adjudicated.
    required_approvers : tuple[str, ...]
        Identities required to approve (relevant for HUMAN_REQUIRED / DUAL_APPROVAL).
    context_summary : str
        Brief summary of the business context.
    policy_check_results : tuple[str, ...]
        IDs of PolicyCheckResult objects from a prior policy evaluation.
    notes : str
        Optional free-text notes.
    """

    request_id: str
    workflow_run_id: Optional[str]
    agent_name: str
    task_id: str
    action_type: str
    action_description: str
    risk_class: str
    environment: str
    evidence_bundle_ids: Tuple[str, ...]
    recommendation_id: Optional[str]
    requested_by: str
    requested_at: str
    expires_at: Optional[str]
    approval_mode: ApprovalMode
    required_approvers: Tuple[str, ...]
    context_summary: str
    policy_check_results: Tuple[str, ...]
    notes: str = ""


# ══════════════════════════════════════════════════════════════════
# 3. APPROVAL DECISION
# ══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ApprovalDecision:
    """The outcome of an adjudicated ApprovalRequest.

    Fields
    ------
    decision_id : str
        UUID string uniquely identifying this decision.
    request_id : str
        ID of the ApprovalRequest this decision resolves.
    decided_at : str
        ISO 8601 timestamp when the decision was made.
    decided_by : str
        Identity of the human reviewer or system that made the decision.
    status : ApprovalStatus
        Outcome status (APPROVED, REJECTED, PENDING, etc.).
    rationale : str
        Explanation for the decision.
    conditions : tuple[str, ...]
        Any conditions attached to an APPROVED decision.
    evidence_reviewed : tuple[str, ...]
        IDs of evidence bundles explicitly reviewed.
    override_used : bool
        Whether an override was used to bypass normal policy.
    notes : str
        Optional free-text notes from the reviewer.
    """

    decision_id: str
    request_id: str
    decided_at: str
    decided_by: str
    status: ApprovalStatus
    rationale: str
    conditions: Tuple[str, ...]
    evidence_reviewed: Tuple[str, ...]
    override_used: bool = False
    notes: str = ""


# ══════════════════════════════════════════════════════════════════
# 4. OVERRIDE RECORD
# ══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class OverrideRecord:
    """Audit trail for any decision that bypassed normal policy adjudication.

    Override records are immutable and append-only. They exist to provide a
    full audit trail when a human or privileged process forces an approval
    that would otherwise be rejected or require additional review.

    Fields
    ------
    override_id : str
        UUID string uniquely identifying this override event.
    original_request_id : str
        ID of the ApprovalRequest that was overridden.
    decision_id : str
        ID of the ApprovalDecision that recorded the override outcome.
    overridden_by : str
        Identity of the person or system that applied the override.
    override_justification : str
        Required explanation for why the override was necessary.
    risk_accepted : str
        Description of the risk explicitly accepted by the overrider.
    timestamp : str
        ISO 8601 timestamp of the override event.
    audit_trail : tuple[str, ...]
        Ordered sequence of action descriptions forming the audit trail.
    """

    override_id: str
    original_request_id: str
    decision_id: str
    overridden_by: str
    override_justification: str
    risk_accepted: str
    timestamp: str
    audit_trail: Tuple[str, ...]


# ══════════════════════════════════════════════════════════════════
# 5. HUMAN REVIEW TICKET
# ══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class HumanReviewTicket:
    """A ticket placed in the human review queue for manual adjudication.

    Created when ApprovalMode is HUMAN_REQUIRED or DUAL_APPROVAL.
    The ticket tracks assignment, due date, and resolution.

    Fields
    ------
    ticket_id : str
        UUID string uniquely identifying this ticket.
    created_at : str
        ISO 8601 timestamp when the ticket was created.
    priority : ReviewPriority
        Priority level driving review SLA.
    title : str
        Short title summarizing what needs to be reviewed.
    description : str
        Full description including context, risk, and request details.
    agent_name : str
        Name of the agent that triggered this ticket.
    workflow_run_id : Optional[str]
        Workflow run associated with this ticket, if any.
    approval_request_id : Optional[str]
        The ApprovalRequest that generated this ticket.
    evidence_bundle_ids : tuple[str, ...]
        Evidence bundles the reviewer should examine.
    due_by : Optional[str]
        ISO 8601 deadline for review completion.
    assigned_to : Optional[str]
        Reviewer assigned to this ticket, if already assigned.
    status : str
        Ticket lifecycle state: "open" | "in_review" | "resolved" | "closed".
    resolution : Optional[str]
        Resolution notes written by the reviewer.
    resolved_at : Optional[str]
        ISO 8601 timestamp when the ticket was resolved.
    """

    ticket_id: str
    created_at: str
    priority: ReviewPriority
    title: str
    description: str
    agent_name: str
    workflow_run_id: Optional[str]
    approval_request_id: Optional[str]
    evidence_bundle_ids: Tuple[str, ...]
    due_by: Optional[str]
    assigned_to: Optional[str]
    status: str  # "open" | "in_review" | "resolved" | "closed"
    resolution: Optional[str] = None
    resolved_at: Optional[str] = None


# ══════════════════════════════════════════════════════════════════
# 6. ESCALATION RECORD
# ══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class EscalationRecord:
    """Records an escalation event from one party or tier to another.

    Escalations occur when a request cannot be resolved at the current
    level and must be routed to a higher authority or broader team.

    Fields
    ------
    escalation_id : str
        UUID string uniquely identifying this escalation.
    escalated_at : str
        ISO 8601 timestamp of escalation.
    escalated_from : str
        Identity or role of the party escalating.
    escalated_to : str
        Identity or role of the party receiving the escalation.
    reason : str
        Why the escalation was necessary.
    priority : ReviewPriority
        Priority assigned to this escalation.
    workflow_run_id : Optional[str]
        Workflow run involved, if applicable.
    approval_request_id : Optional[str]
        Linked ApprovalRequest, if applicable.
    incident_id : Optional[str]
        Linked IncidentRecord, if applicable.
    resolved : bool
        Whether the escalation has been resolved.
    resolved_at : Optional[str]
        ISO 8601 timestamp of resolution, if resolved.
    resolution_notes : str
        Notes describing how the escalation was resolved.
    """

    escalation_id: str
    escalated_at: str
    escalated_from: str
    escalated_to: str
    reason: str
    priority: ReviewPriority
    workflow_run_id: Optional[str]
    approval_request_id: Optional[str]
    incident_id: Optional[str]
    resolved: bool = False
    resolved_at: Optional[str] = None
    resolution_notes: str = ""
