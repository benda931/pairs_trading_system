# -*- coding: utf-8 -*-
"""
orchestration/contracts.py — Orchestration Domain Types
=========================================================

Single source of truth for all workflow, agent-architecture, and orchestration
types used by the pairs trading system's orchestration layer.

Design principles:
  - All frozen dataclasses are JSON-serializable (str, int, float, bool, None,
    tuple of primitives/dicts).  Use ``dataclasses.asdict()`` for serialisation.
  - Mutable workflow state objects (WorkflowRun, WorkflowStepRun) are regular
    (non-frozen) dataclasses so the engine can update them in-place.
  - Enums inherit from ``str`` so their values survive JSON round-trips.
  - No Pydantic, no external dependencies beyond stdlib.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Optional


# ══════════════════════════════════════════════════════════════════
# 1. ENUMERATIONS
# ══════════════════════════════════════════════════════════════════


class WorkflowStatus(str, enum.Enum):
    """Top-level lifecycle status of a workflow run."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    PAUSED = "PAUSED"


class WorkflowStepStatus(str, enum.Enum):
    """Lifecycle status of a single step within a workflow run."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    RETRYING = "RETRYING"


class RiskClass(str, enum.Enum):
    """
    Risk classification for agent operations and workflows.

    Determines whether human approval is required and what safety checks apply.
    Order is meaningful: higher ordinal = higher risk.
    """

    INFORMATIONAL = "INFORMATIONAL"          # Read-only, no side effects
    BOUNDED_SAFE = "BOUNDED_SAFE"            # Writes research artefacts only
    MEDIUM_RISK = "MEDIUM_RISK"              # Modifies non-critical state
    HIGH_RISK = "HIGH_RISK"                  # Modifies critical state / config
    SENSITIVE = "SENSITIVE"                  # Order submission, capital allocation


class EnvironmentClass(str, enum.Enum):
    """
    Deployment environment.

    Governs which operations are permitted and whether auto-approval is allowed.
    RESEARCH is the most permissive; PRODUCTION the most restrictive.
    """

    RESEARCH = "RESEARCH"
    STAGING = "STAGING"
    PAPER = "PAPER"
    PRODUCTION = "PRODUCTION"


class ActionBoundary(str, enum.Enum):
    """
    The class of external action an agent or step is allowed to take.

    Used for audit and approval-gate routing decisions.
    """

    INFORMATIONAL = "INFORMATIONAL"                  # Observe, read, analyse
    RECOMMENDATION = "RECOMMENDATION"                # Propose, suggest
    CONTROLLED_OPERATIONAL = "CONTROLLED_OPERATIONAL"  # Controlled write / state change
    SENSITIVE = "SENSITIVE"                          # Order, transfer, critical mutation


class FailureClass(str, enum.Enum):
    """
    Canonical classification of failure causes for structured incident handling.

    Used in FailureRecord to drive retry / escalation policy.
    """

    INVALID_CONTEXT = "INVALID_CONTEXT"
    MISSING_ARTIFACTS = "MISSING_ARTIFACTS"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    POLICY_VIOLATION = "POLICY_VIOLATION"
    MALFORMED_OUTPUT = "MALFORMED_OUTPUT"
    TIMEOUT = "TIMEOUT"
    DEPENDENCY_FAILURE = "DEPENDENCY_FAILURE"
    STALE_DATA = "STALE_DATA"
    SCHEMA_MISMATCH = "SCHEMA_MISMATCH"
    CONFLICTING_EVIDENCE = "CONFLICTING_EVIDENCE"
    APPROVAL_REJECTED = "APPROVAL_REJECTED"
    ORCHESTRATION_DEAD_END = "ORCHESTRATION_DEAD_END"
    RETRY_EXHAUSTED = "RETRY_EXHAUSTED"


class DelegationDepth(int, enum.Enum):
    """
    Maximum number of agent-to-agent delegation hops permitted.

    ZERO means the agent cannot sub-delegate at all.
    """

    ZERO = 0
    ONE = 1
    TWO = 2
    THREE = 3


# ══════════════════════════════════════════════════════════════════
# 2. AGENT ARCHITECTURE TYPES
# ══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class AgentCapability:
    """
    Declares one specific capability that an agent possesses.

    Capabilities are referenced by name in AgentPermissionSet.  They encode
    the action boundary and risk class so that the orchestration engine can
    enforce safety constraints without reading agent source code.
    """

    name: str
    """Unique capability identifier, e.g. ``"read_prices"`` or ``"emit_order"``."""

    description: str
    """Human-readable explanation of what the capability does."""

    action_boundary: ActionBoundary
    """Highest action boundary this capability reaches."""

    risk_class: RiskClass
    """Risk classification of exercising this capability."""

    requires_approval: bool = False
    """If True, any workflow step using this capability needs an approval gate."""

    max_autonomy_level: int = 1
    """
    1 = informational output only
    2 = recommendation (no side effects)
    3 = controlled operational (bounded write / state change)
    """

    notes: str = ""
    """Optional free-text context for human reviewers."""


@dataclass(frozen=True)
class AgentPermissionSet:
    """
    Declared permission boundary for a specific agent.

    The orchestration engine checks these at dispatch time.  An agent that
    attempts an action outside its permission set causes a PERMISSION_DENIED
    FailureRecord.
    """

    agent_name: str
    """The agent this permission set applies to."""

    allowed_capabilities: tuple[str, ...]
    """Capability names the agent is permitted to exercise."""

    forbidden_capabilities: tuple[str, ...]
    """Capability names explicitly denied (overrides ``allowed_capabilities``)."""

    allowed_environments: tuple[EnvironmentClass, ...]
    """Environments in which this agent may execute."""

    max_delegation_depth: DelegationDepth = DelegationDepth.ONE
    """Maximum depth to which this agent may sub-delegate tasks."""

    requires_human_review_above_risk: RiskClass = RiskClass.HIGH_RISK
    """Any action at or above this risk level requires a human approval gate."""

    emergency_disable: bool = False
    """If True, the agent is suspended and all tasks are immediately rejected."""

    notes: str = ""


@dataclass(frozen=True)
class AgentSpec:
    """
    Complete static specification for an agent.

    Combines identity, mission, allowed task types, capabilities, permissions,
    and failure behaviour into one immutable record that can be stored in a
    registry or serialised to JSON.
    """

    name: str
    """Unique agent name matching ``BaseAgent.NAME``."""

    version: str
    """Semantic version string, e.g. ``"1.0.0"``."""

    description: str
    """One-sentence description of the agent's role."""

    mission: str
    """Detailed statement of the agent's mandate (what it does and does not do)."""

    allowed_task_types: tuple[str, ...]
    """Task type strings this agent will accept."""

    capabilities: tuple[AgentCapability, ...]
    """All capabilities the agent may exercise."""

    permission_set: AgentPermissionSet
    """Declared permission boundary."""

    failure_behavior: str
    """
    Strategy on task failure.

    One of: ``"fail_fast"`` | ``"retry"`` | ``"escalate"`` | ``"degrade"``
    """

    retry_max: int = 3
    """Maximum retry attempts before reporting RETRY_EXHAUSTED."""

    timeout_seconds: float = 60.0
    """Per-task execution timeout."""

    owner: str = "system"
    """Team or individual responsible for this agent."""

    notes: str = ""


@dataclass(frozen=True)
class AgentRunRecord:
    """
    Immutable audit record capturing the outcome of a single agent task execution.

    Created by the orchestration engine after each task completes (success or
    failure) and appended to the durable audit log.
    """

    run_id: str
    """UUID for this specific run."""

    agent_name: str
    task_id: str
    task_type: str
    environment: EnvironmentClass
    started_at: str
    """ISO-8601 UTC timestamp."""

    completed_at: Optional[str]
    """ISO-8601 UTC timestamp; None if still running."""

    status: str
    """Value of ``AgentStatus``."""

    duration_ms: float

    output_keys: tuple[str, ...]
    """Keys present in the agent's output dict (values are not stored here)."""

    error: Optional[str]
    """Error message if status is FAILED."""

    audit_entries_count: int
    """Number of entries in the agent's audit trail."""

    policy_checks_passed: int
    policy_checks_failed: int

    approval_required: bool
    approval_status: Optional[str]
    """``"PENDING"`` | ``"APPROVED"`` | ``"REJECTED"`` | ``None``"""


@dataclass(frozen=True)
class AgentHealthStatus:
    """
    Summarised health metrics for one agent, computed by the monitoring layer.

    Used to surface degraded agents before they impact a live workflow.
    """

    agent_name: str
    checked_at: str
    """ISO-8601 UTC timestamp when the health check ran."""

    is_healthy: bool

    total_runs: int
    success_rate: float
    """Fraction [0, 1] of runs that completed successfully."""

    avg_duration_ms: float

    recent_failures: int
    """Failure count in the most recent observation window (typically 1 hour)."""

    recent_timeout_count: int

    last_failure_reason: Optional[str]

    policy_violation_count: int
    """Cumulative number of permission or policy violations."""

    stale_context_count: int
    """Number of runs where a freshness warning was emitted."""

    recommendation: str
    """
    One of: ``"healthy"`` | ``"monitor"`` | ``"investigate"`` | ``"disable"``
    """


# ══════════════════════════════════════════════════════════════════
# 3. CONTEXT AND MEMORY
# ══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ContextReference:
    """
    A pointer to one item of context that an agent has been given.

    Agents never store raw context values in memory — only references.  The
    actual values are fetched through the artefact store.
    """

    ref_id: str
    """UUID for this reference."""

    source: str
    """
    Origin type.

    One of: ``"model_card"`` | ``"config"`` | ``"experiment"`` |
    ``"policy"`` | ``"report"`` | ``"artifact"``
    """

    key: str
    """Human-readable key identifying the context item."""

    version: Optional[str]
    """Artefact version or commit hash; None if not versioned."""

    timestamp: Optional[str]
    """ISO-8601 creation or last-modified time of the referenced item."""

    freshness_note: str
    """Human-readable description of how fresh this context is."""

    sensitivity: str
    """
    Data sensitivity class.

    One of: ``"public"`` | ``"internal"`` | ``"restricted"``
    """


@dataclass(frozen=True)
class AgentContextPackage:
    """
    The assembled context delivered to an agent before task execution.

    The package is assembled by the orchestration engine from registered
    sources.  It records what was included, what was excluded, and any
    freshness concerns so the agent can make an informed decision about
    how much to trust its context.
    """

    package_id: str
    """UUID for this context package assembly."""

    assembled_for_agent: str
    assembled_for_task_type: str
    assembled_at: str
    """ISO-8601 UTC timestamp."""

    environment: EnvironmentClass

    context_items: tuple[ContextReference, ...]
    """All context items included in this package."""

    excluded_items: tuple[str, ...]
    """Keys that were considered but excluded (e.g. too large, restricted)."""

    total_items: int
    size_budget_used: int
    """Approximate token count consumed by this package."""

    freshness_warnings: tuple[str, ...]
    """Any context items that are stale or near expiry."""

    notes: str = ""


@dataclass(frozen=True)
class AgentMemoryRecord:
    """
    One entry in an agent's durable memory store.

    Agents must not store raw values here — only human-readable summaries and
    external artefact references.  This prevents unbounded memory growth and
    keeps the memory log auditable.
    """

    record_id: str
    """UUID."""

    memory_type: str
    """
    Classification of what this memory entry represents.

    One of: ``"run_local"`` | ``"workflow_state"`` | ``"artifact_ref"`` |
    ``"policy_snapshot"`` | ``"incident_ref"`` | ``"experiment_ref"``
    """

    agent_name: str

    workflow_run_id: Optional[str]
    """Workflow run this memory is scoped to; None for agent-global memory."""

    key: str
    """Unique memory key within its scope."""

    value_summary: str
    """Human-readable summary of the stored value (NOT the value itself)."""

    artifact_reference: Optional[str]
    """ID of the external artefact if the full value is stored elsewhere."""

    created_at: str
    """ISO-8601 UTC timestamp."""

    expires_at: Optional[str]
    """ISO-8601 UTC timestamp after which this record is considered stale."""

    source_provenance: str
    """Describes how this memory was created (e.g. agent name + task type)."""

    access_class: str
    """
    Visibility scope.

    One of: ``"agent_local"`` | ``"workflow_shared"`` | ``"registry"``
    """

    is_expired: bool = False


# ══════════════════════════════════════════════════════════════════
# 4. EVIDENCE AND RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class AgentWarning:
    """
    A structured warning emitted by an agent during task execution.

    Warnings do not cause task failure but must be surfaced to human reviewers
    and downstream agents.
    """

    warning_id: str
    """UUID."""

    agent_name: str
    task_id: str

    category: str
    """
    One of: ``"stale_data"`` | ``"low_confidence"`` | ``"policy_near_limit"``
    | ``"schema_mismatch"`` | ``"missing_context"``
    """

    message: str
    severity: str
    """One of: ``"low"`` | ``"medium"`` | ``"high"``"""

    timestamp: str
    """ISO-8601 UTC."""


@dataclass(frozen=True)
class AgentEvidenceBundle:
    """
    A structured collection of evidence produced by an agent.

    Evidence bundles are the primary output artefact of analytical agents.
    They feed into AgentRecommendation objects and are stored durably so
    that any downstream decision can be traced back to its evidence.

    Each ``item`` in ``items`` is a dict with keys:
      - ``"key"`` (str): identifier for this evidence item
      - ``"value"`` (Any): the evidence value (must be JSON-serialisable)
      - ``"source"`` (str): where this value came from
      - ``"confidence"`` (float): [0, 1] confidence in this value
    """

    bundle_id: str
    """UUID."""

    producing_agent: str
    task_id: str
    workflow_run_id: Optional[str]

    evidence_type: str
    """
    One of: ``"signal_analysis"`` | ``"regime_scan"`` |
    ``"portfolio_analysis"`` | ``"risk_review"`` |
    ``"model_comparison"`` | ``"validation_report"``
    """

    timestamp: str
    """ISO-8601 UTC."""

    items: tuple[dict, ...]
    """Evidence items — see class docstring for schema."""

    summary: str
    """One-paragraph human-readable summary of the evidence."""

    warnings: tuple[AgentWarning, ...]
    """Any warnings generated while producing this bundle."""

    confidence_overall: float
    """Aggregate confidence [0, 1] across all evidence items."""

    context_package_id: Optional[str]
    """ID of the AgentContextPackage used to produce this bundle."""

    schema_version: str = "1.0"


@dataclass(frozen=True)
class AgentRecommendation:
    """
    A typed, auditable recommendation produced by an agent.

    Recommendations are proposals — they never execute themselves.  The
    portfolio layer or a human operator must explicitly accept or reject them.

    Accepted recommendations feed into ``WorkflowRun.recommendation_ids``.
    """

    recommendation_id: str
    """UUID."""

    producing_agent: str
    task_id: str
    workflow_run_id: Optional[str]

    recommendation_type: str
    """
    One of: ``"promote_model"`` | ``"adjust_risk_limit"`` | ``"retrain"`` |
    ``"de_risk"`` | ``"investigate"`` | ``"escalate"`` | ``"approve"`` |
    ``"reject"`` | ``"freeze"``
    """

    action_boundary: ActionBoundary
    risk_class: RiskClass

    description: str
    """Short human-readable description of the recommended action."""

    rationale: str
    """Detailed reasoning explaining why this recommendation is being made."""

    evidence_bundle_ids: tuple[str, ...]
    """IDs of AgentEvidenceBundles that support this recommendation."""

    confidence: float
    """Agent's confidence [0, 1] in this recommendation."""

    urgency: str
    """One of: ``"routine"`` | ``"elevated"`` | ``"urgent"`` | ``"critical"``"""

    requires_approval: bool
    """If True, a human must approve before acting on this recommendation."""

    suggested_reviewer: Optional[str]
    """Role or name of the suggested human reviewer."""

    timestamp: str
    """ISO-8601 UTC."""

    expiry: Optional[str]
    """ISO-8601 UTC after which this recommendation should be considered stale."""

    accepted: Optional[bool] = None
    """None = pending review; True = accepted; False = rejected."""

    rejection_reason: Optional[str] = None
    """Populated if ``accepted is False``."""


# ══════════════════════════════════════════════════════════════════
# 5. WORKFLOW OBJECTS
# ══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class WorkflowStep:
    """
    Immutable definition of one step within a WorkflowDefinition.

    Steps are executed by the orchestration engine in dependency order.
    The engine creates a WorkflowStepRun for each step as it executes.
    """

    step_id: str
    """Unique identifier within the workflow definition."""

    name: str
    """Human-readable step name."""

    agent_name: str
    """Name of the agent that will execute this step."""

    task_type: str
    """Task type string forwarded to the agent."""

    depends_on: tuple[str, ...]
    """step_ids that must complete before this step can start."""

    timeout_seconds: float
    """Per-step execution timeout."""

    retry_max: int
    """Maximum retry attempts for this step."""

    risk_class: RiskClass
    """Risk classification used for approval-gate routing."""

    requires_approval_before: bool
    """If True, an approval gate must pass before the agent is invoked."""

    requires_approval_after: bool
    """If True, an approval gate must pass after the agent completes."""

    on_failure: str
    """
    Failure disposition.

    One of: ``"fail_workflow"`` | ``"skip"`` | ``"escalate"`` | ``"retry"``
    """

    notes: str = ""


@dataclass(frozen=True)
class WorkflowTransition:
    """
    A directed edge in the workflow step graph.

    Transitions encode conditional routing: the engine evaluates the
    ``condition`` after a step completes to decide which step to run next.
    A ``to_step_id`` of None indicates a terminal transition.
    """

    from_step_id: str
    to_step_id: Optional[str]
    """None if this transition leads to workflow termination."""

    condition: str
    """
    Routing condition evaluated after ``from_step_id`` completes.

    One of: ``"always"`` | ``"on_success"`` | ``"on_failure"`` |
    ``"on_approval"`` | ``"on_rejection"``
    """

    notes: str = ""


@dataclass(frozen=True)
class WorkflowDefinition:
    """
    Immutable blueprint describing a complete workflow.

    A WorkflowDefinition is a template.  Each time it is executed, the
    engine creates a WorkflowRun.  Multiple runs of the same definition
    can co-exist simultaneously.
    """

    workflow_id: str
    """Unique identifier for this workflow definition."""

    name: str
    description: str
    version: str
    """Semantic version string."""

    steps: tuple[WorkflowStep, ...]
    """All steps in this workflow."""

    transitions: tuple[WorkflowTransition, ...]
    """All directed edges between steps."""

    entry_condition: str
    """
    Human-readable description of the conditions that trigger this workflow,
    e.g. ``"Scheduled daily at 06:00 UTC"`` or ``"Drift alert received"``.
    """

    termination_condition: str
    """
    Human-readable description of the terminal condition,
    e.g. ``"All steps completed or first hard failure"``
    """

    environment_class: EnvironmentClass
    """Environment this definition is designed for."""

    risk_class: RiskClass
    """Highest risk class present in any step."""

    max_duration_seconds: float
    """Hard wall-clock timeout for the entire workflow run."""

    idempotent: bool
    """
    If True, re-running the same workflow with the same payload produces the
    same artefacts (useful for scheduled refresh workflows).
    """

    replayable: bool
    """
    If True, a failed run can be replayed from the last successful step,
    rather than restarting from scratch.
    """

    owner: str
    tags: tuple[str, ...]
    """Free-form classification tags."""

    notes: str = ""


@dataclass
class WorkflowStepRun:
    """
    Mutable runtime state for one step within a WorkflowRun.

    Created by the engine when a step starts.  The engine updates this object
    in place as the step progresses.  On completion it is frozen into the
    WorkflowRun's ``step_runs`` list.
    """

    step_run_id: str
    """UUID."""

    step_id: str
    """References WorkflowStep.step_id."""

    workflow_run_id: str

    status: WorkflowStepStatus

    started_at: Optional[str]
    """ISO-8601 UTC; None until the step actually starts."""

    completed_at: Optional[str]
    """ISO-8601 UTC; None until the step finishes."""

    agent_task_id: Optional[str]
    """UUID of the AgentTask dispatched for this step; None if not yet dispatched."""

    agent_result_status: Optional[str]
    """Value of AgentStatus from the result; None if not yet completed."""

    retry_count: int = 0
    """Number of retries attempted so far."""

    error: Optional[str] = None
    """Error message from the most recent attempt; None on success."""

    output_summary: str = ""
    """Human-readable one-line summary of the step's output."""

    artifact_ids: list[str] = field(default_factory=list)
    """IDs of artefacts produced by this step."""


@dataclass
class WorkflowRun:
    """
    Mutable runtime state for one execution of a WorkflowDefinition.

    Created by the engine when ``WorkflowEngine.run()`` is called.
    The engine updates this object in place throughout execution.
    On completion it is stored in ``WorkflowEngine._runs``.
    """

    run_id: str
    """UUID."""

    workflow_id: str
    """References WorkflowDefinition.workflow_id."""

    workflow_name: str
    status: WorkflowStatus
    environment: EnvironmentClass

    triggered_by: str
    """
    How this run was initiated.

    One of: ``"schedule"`` | ``"event"`` | ``"manual"`` |
    ``"agent"`` | ``"incident"``
    """

    trigger_payload: dict
    """Input payload provided at trigger time."""

    started_at: str
    """ISO-8601 UTC."""

    completed_at: Optional[str] = None
    """ISO-8601 UTC; None while running."""

    step_runs: list[WorkflowStepRun] = field(default_factory=list)
    artifact_ids: list[str] = field(default_factory=list)
    recommendation_ids: list[str] = field(default_factory=list)
    approval_request_ids: list[str] = field(default_factory=list)

    error: Optional[str] = None
    """Top-level error message if the workflow failed."""

    replay_of_run_id: Optional[str] = None
    """If this run is a replay, the original run_id; otherwise None."""

    notes: str = ""


@dataclass(frozen=True)
class WorkflowOutcome:
    """
    Immutable summary produced when a WorkflowRun reaches a terminal state.

    Intended for dashboards, alerting, and audit trails.
    """

    run_id: str
    workflow_id: str
    status: WorkflowStatus
    completed_at: str
    """ISO-8601 UTC."""

    steps_completed: int
    steps_failed: int
    steps_skipped: int
    total_duration_ms: float

    artifact_count: int
    recommendation_count: int
    approval_count: int
    escalation_count: int

    notes: str = ""


# ══════════════════════════════════════════════════════════════════
# 6. DELEGATION AND FAILURE RECORDS
# ══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class DelegationRecord:
    """
    Audit record capturing one agent-to-agent task delegation.

    Created whenever a coordinator agent sub-delegates a task to another agent.
    Used to detect and cap delegation depth, preventing infinite delegation
    chains.
    """

    delegation_id: str
    """UUID."""

    coordinator_agent: str
    """Agent that is delegating the task."""

    delegated_to_agent: str
    """Agent receiving the delegated task."""

    task_id: str
    """The original task being delegated."""

    workflow_run_id: Optional[str]

    delegation_depth: int
    """Depth of this delegation (0 = direct from engine, 1 = first hop, etc.)."""

    reason: str
    """Human-readable explanation of why delegation was necessary."""

    delegated_at: str
    """ISO-8601 UTC."""

    result_task_id: Optional[str] = None
    """Task ID returned by the delegated agent; None until complete."""

    result_status: Optional[str] = None
    """AgentStatus value of the delegated result; None until complete."""


@dataclass(frozen=True)
class FailureRecord:
    """
    Structured record of a task or step failure.

    Created by the engine whenever a step enters a failed terminal state.
    ``failure_class`` drives the escalation and retry policy.
    """

    failure_id: str
    """UUID."""

    agent_name: str
    task_id: str
    workflow_run_id: Optional[str]

    failure_class: FailureClass
    """Canonical classification of the failure root cause."""

    message: str
    """Human-readable description of the failure."""

    is_retryable: bool
    """Whether this failure class supports automatic retry."""

    retry_count: int
    """Number of retries already attempted before this record was created."""

    escalated: bool
    """True if this failure has been escalated to a human reviewer."""

    timestamp: str
    """ISO-8601 UTC."""

    resolution: Optional[str] = None
    """Human-provided resolution description; None until resolved."""
