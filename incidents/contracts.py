# -*- coding: utf-8 -*-
"""
incidents/contracts.py — Incident Management Domain Contracts
==============================================================

Domain types for incident management, remediation planning, postmortem
artifacts, and the append-only audit record ledger.

Design principles:
- IncidentRecord is mutable (normal dataclass) to support timeline updates.
- RemediationPlan is mutable to support step-status updates.
- All other types are frozen (immutable after construction).
- Enums inherit from str for JSON-serialization compatibility.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ══════════════════════════════════════════════════════════════════
# 1. ENUMERATIONS
# ══════════════════════════════════════════════════════════════════


class IncidentSeverity(str, enum.Enum):
    """Severity tiers aligned to standard incident management practice.

    P0_CRITICAL : Complete system or trading halt; immediate response required.
    P1_HIGH     : Major degradation; high-priority response.
    P2_MEDIUM   : Partial degradation; response within business hours.
    P3_LOW      : Minor issue; response within SLA window.
    P4_INFO     : Informational; no response urgency.
    """

    P0_CRITICAL = "P0"
    P1_HIGH = "P1"
    P2_MEDIUM = "P2"
    P3_LOW = "P3"
    P4_INFO = "P4"


class IncidentStatus(str, enum.Enum):
    """Lifecycle status of an IncidentRecord."""

    OPEN = "OPEN"
    IN_PROGRESS = "IN_PROGRESS"
    MONITORING = "MONITORING"
    RESOLVED = "RESOLVED"
    CLOSED = "CLOSED"
    POSTMORTEM = "POSTMORTEM"


class AuditRecordType(str, enum.Enum):
    """Categorises the source event that produced an AuditRecord."""

    AGENT_EXECUTION = "AGENT_EXECUTION"
    POLICY_CHECK = "POLICY_CHECK"
    APPROVAL_DECISION = "APPROVAL_DECISION"
    WORKFLOW_TRANSITION = "WORKFLOW_TRANSITION"
    DELEGATION = "DELEGATION"
    ESCALATION = "ESCALATION"
    EMERGENCY_ACTION = "EMERGENCY_ACTION"
    INCIDENT_CREATED = "INCIDENT_CREATED"
    INCIDENT_UPDATED = "INCIDENT_UPDATED"
    OVERRIDE = "OVERRIDE"
    # ── Governance layer additions ─────────────────────────────────
    GOVERNED_ACTION_EXECUTED = "GOVERNED_ACTION_EXECUTED"
    GOVERNED_ACTION_SUPPRESSED = "GOVERNED_ACTION_SUPPRESSED"
    GOVERNED_ACTION_VETOED = "GOVERNED_ACTION_VETOED"
    GOVERNED_ACTION_ADVISORY = "GOVERNED_ACTION_ADVISORY"
    GOVERNED_ACTION_PENDING = "GOVERNED_ACTION_PENDING"
    ROLLBACK_EXECUTED = "ROLLBACK_EXECUTED"
    PRECISION_DEMOTION = "PRECISION_DEMOTION"
    EMERGENCY_BYPASS_ACTIVATED = "EMERGENCY_BYPASS_ACTIVATED"
    PATTERN_DETECTED = "PATTERN_DETECTED"
    SLA_BREACH = "SLA_BREACH"


class IncidentTriggerSource(str, enum.Enum):
    """Records what triggered an IncidentRecord to be opened.

    Enables filtering and postmortem analysis by trigger type.
    """

    AGENT_ACTION = "agent_action"
    CIRCUIT_BREAKER = "circuit_breaker"
    APPROVAL_SLA_BREACH = "approval_sla_breach"
    PRECISION_DEGRADATION = "precision_degradation"
    PATTERN_DETECTION = "pattern_detection"
    EXECUTION_FAILURE = "execution_failure"
    EMERGENCY_BYPASS = "emergency_bypass"
    MANUAL = "manual"


# ══════════════════════════════════════════════════════════════════
# 2. RUNBOOK REFERENCE
# ══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class RunbookReference:
    """A reference to an operational runbook for incident response.

    Fields
    ------
    runbook_id : str
        Unique identifier for the runbook.
    title : str
        Short human-readable title.
    description : str
        Description of when this runbook applies.
    applicable_severity : tuple[str, ...]
        Severity values (IncidentSeverity.value) this runbook applies to.
    applicable_components : tuple[str, ...]
        System component identifiers this runbook covers.
    steps : tuple[str, ...]
        Ordered list of procedural steps.
    url : Optional[str]
        Link to the full runbook document, if hosted externally.
    version : str
        Version string of the runbook.
    last_reviewed : str
        ISO date string of last review.
    """

    runbook_id: str
    title: str
    description: str
    applicable_severity: Tuple[str, ...]
    applicable_components: Tuple[str, ...]
    steps: Tuple[str, ...]
    url: Optional[str]
    version: str
    last_reviewed: str


# ══════════════════════════════════════════════════════════════════
# 3. INCIDENT RECORD (MUTABLE)
# ══════════════════════════════════════════════════════════════════


@dataclass
class IncidentRecord:
    """Mutable record tracking an active or historical incident.

    The ``timeline`` field is an append-only list of event dicts with
    keys: ``ts`` (ISO timestamp), ``actor`` (str), ``action`` (str),
    ``notes`` (str).

    Mutation is intentional — incident status changes, timeline events,
    and attachments need to be applied without creating full copies.
    All mutations are protected by the IncidentManager's lock.

    Fields
    ------
    incident_id : str
        UUID string.
    title : str
        Short title.
    description : str
        Full description of the incident.
    severity : IncidentSeverity
        Severity tier.
    status : IncidentStatus
        Current lifecycle status.
    detected_at : str
        ISO 8601 timestamp of initial detection.
    detected_by : str
        Identity of the detecting agent, alert, or person.
    affected_components : list[str]
        System components involved.
    affected_agents : list[str]
        Agent names involved.
    affected_workflows : list[str]
        Workflow IDs or types affected.
    evidence_bundle_ids : list[str]
        Evidence bundle IDs attached to this incident.
    related_alert_ids : list[str]
        Alert IDs that triggered or relate to this incident.
    runbook_refs : list[str]
        Runbook IDs linked to this incident.
    timeline : list[dict]
        Ordered sequence of timeline events.
    assigned_to : Optional[str]
        Current assignee.
    resolved_at : Optional[str]
        ISO timestamp of resolution.
    resolution_summary : Optional[str]
        Summary of how the incident was resolved.
    postmortem_id : Optional[str]
        ID of the linked PostmortemArtifact, if created.
    tags : list[str]
        Free-form tags for filtering.
    """

    incident_id: str
    title: str
    description: str
    severity: IncidentSeverity
    status: IncidentStatus
    detected_at: str
    detected_by: str
    affected_components: List[str]
    affected_agents: List[str]
    affected_workflows: List[str]
    evidence_bundle_ids: List[str]
    related_alert_ids: List[str]
    runbook_refs: List[str]
    timeline: List[Dict]
    assigned_to: Optional[str] = None
    resolved_at: Optional[str] = None
    resolution_summary: Optional[str] = None
    postmortem_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    # Governance layer additions
    trigger_source: Optional[str] = None   # IncidentTriggerSource.value
    governed_action_id: Optional[str] = None  # GovernedActionRecord.action_id
    requires_postmortem: bool = False      # True for P0/EMERGENCY_BYPASS incidents
    postmortem_deadline: Optional[str] = None  # ISO timestamp (24h for P0)


# ══════════════════════════════════════════════════════════════════
# 4. REMEDIATION STEP (FROZEN)
# ══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class RemediationStep:
    """A single step in a RemediationPlan.

    Fields
    ------
    step_id : str
        UUID string for this step.
    order : int
        Execution order (1-based, lower = earlier).
    description : str
        Description of the action to take.
    responsible : str
        Identity or role responsible for this step.
    automated : bool
        Whether this step can be executed automatically.
    estimated_duration_minutes : int
        Estimated time to complete.
    verification_criteria : str
        How to verify the step was completed successfully.
    status : str
        Current status: "pending" | "in_progress" | "completed" | "skipped".
    """

    step_id: str
    order: int
    description: str
    responsible: str
    automated: bool
    estimated_duration_minutes: int
    verification_criteria: str
    status: str = "pending"


# ══════════════════════════════════════════════════════════════════
# 5. REMEDIATION PLAN (MUTABLE)
# ══════════════════════════════════════════════════════════════════


@dataclass
class RemediationPlan:
    """Mutable remediation plan attached to an IncidentRecord.

    Mutable to allow step status updates as remediation progresses
    without reconstructing the full plan.

    Fields
    ------
    plan_id : str
        UUID string.
    incident_id : str
        ID of the IncidentRecord this plan addresses.
    created_at : str
        ISO 8601 timestamp.
    created_by : str
        Identity of the creator.
    steps : list[RemediationStep]
        Ordered list of remediation steps.
    estimated_resolution_minutes : int
        Estimated total resolution time.
    rollback_available : bool
        Whether a rollback plan is available.
    rollback_steps : list[str]
        Ordered list of rollback step descriptions.
    status : str
        Plan status: "draft" | "approved" | "in_progress" | "completed" | "abandoned".
    approved_by : Optional[str]
        Identity of the plan approver.
    completed_at : Optional[str]
        ISO timestamp of plan completion.
    """

    plan_id: str
    incident_id: str
    created_at: str
    created_by: str
    steps: List[RemediationStep]
    estimated_resolution_minutes: int
    rollback_available: bool
    rollback_steps: List[str]
    status: str = "draft"
    approved_by: Optional[str] = None
    completed_at: Optional[str] = None


# ══════════════════════════════════════════════════════════════════
# 6. POSTMORTEM ARTIFACT (FROZEN)
# ══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class PostmortemArtifact:
    """Immutable postmortem document produced after incident closure.

    Postmortems are blameless and focus on systemic improvements.
    They are linked back to the originating IncidentRecord.

    Fields
    ------
    postmortem_id : str
        UUID string.
    incident_id : str
        ID of the incident this postmortem covers.
    authored_by : str
        Identity of the primary author.
    authored_at : str
        ISO 8601 timestamp.
    incident_summary : str
        1-3 sentence summary of what happened.
    timeline_summary : str
        Narrative summary of key events.
    root_cause : str
        Root cause analysis.
    contributing_factors : tuple[str, ...]
        Factors that contributed to the incident.
    what_went_well : tuple[str, ...]
        Things that worked well during response.
    what_went_wrong : tuple[str, ...]
        Things that did not work well.
    action_items : tuple[str, ...]
        Concrete follow-up action items with owners.
    prevention_recommendations : tuple[str, ...]
        Recommendations to prevent recurrence.
    related_incident_ids : tuple[str, ...]
        IDs of related past incidents.
    schema_version : str
        Schema version for forward compatibility.
    """

    postmortem_id: str
    incident_id: str
    authored_by: str
    authored_at: str
    incident_summary: str
    timeline_summary: str
    root_cause: str
    contributing_factors: Tuple[str, ...]
    what_went_well: Tuple[str, ...]
    what_went_wrong: Tuple[str, ...]
    action_items: Tuple[str, ...]
    prevention_recommendations: Tuple[str, ...]
    related_incident_ids: Tuple[str, ...]
    schema_version: str = "1.0"


# ══════════════════════════════════════════════════════════════════
# 7. AUDIT RECORD (FROZEN)
# ══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class AuditRecord:
    """Immutable, append-only record of a material system event.

    AuditRecords form the primary audit ledger for the platform. They are
    never modified or deleted — only appended. The IncidentManager enforces
    a rolling cap of 50,000 records in memory.

    Fields
    ------
    record_id : str
        UUID string.
    record_type : AuditRecordType
        Category of the event.
    actor : str
        Identity of the agent, user, or system that performed the action.
    action : str
        Short action description (e.g. "POSITION_OPENED", "MODEL_PROMOTED").
    subject : str
        Target of the action (e.g. pair_id, model_id, agent_name).
    outcome : str
        Outcome: "SUCCESS" | "FAILURE" | "PARTIAL" | "SKIPPED".
    timestamp : str
        ISO 8601 timestamp.
    workflow_run_id : Optional[str]
        Workflow run context, if applicable.
    task_id : Optional[str]
        AgentTask ID, if applicable.
    evidence_ids : tuple[str, ...]
        Evidence bundle IDs relevant to this event.
    policy_check_ids : tuple[str, ...]
        PolicyCheckResult IDs from prior evaluation.
    details : str
        Free-text details.
    schema_version : str
        Schema version for forward compatibility.
    """

    record_id: str
    record_type: AuditRecordType
    actor: str
    action: str
    subject: str
    outcome: str
    timestamp: str
    workflow_run_id: Optional[str]
    task_id: Optional[str]
    evidence_ids: Tuple[str, ...]
    policy_check_ids: Tuple[str, ...]
    details: str = ""
    schema_version: str = "1.0"
