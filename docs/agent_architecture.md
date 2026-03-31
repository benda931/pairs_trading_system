# Agent, Orchestration, and Governance Architecture — Pairs Trading System

**Version:** 1.0
**Last updated:** 2026-03-31
**Scope:** Agent platform, orchestration engine, governance policy layer, approval
system, incident management, and audit trail

> **Integration Status: SCAFFOLD**
> Agents are registered and individually tested (91 tests pass). As of 2026-03-31,
> **no operational workflow dispatches agent tasks.** Zero agents are called from
> backtesting, signal generation, or portfolio allocation paths.
> See: `docs/INTEGRATION_STATUS.md`, `docs/remediation/remediation_ledger.md:P1-AGENTS`

---

## Table of Contents

1. [Overview and Philosophy](#1-overview-and-philosophy)
2. [Agent Taxonomy](#2-agent-taxonomy)
3. [Core Contract Model](#3-core-contract-model)
4. [Orchestration Model](#4-orchestration-model)
5. [Governance and Approval Model](#5-governance-and-approval-model)
6. [Incident and Audit Model](#6-incident-and-audit-model)
7. [Multi-Agent Coordination Patterns](#7-multi-agent-coordination-patterns)
8. [Action Boundaries](#8-action-boundaries)
9. [Safety Doctrine](#9-safety-doctrine)
10. [Integration with Core Platform Layers](#10-integration-with-core-platform-layers)
11. [Known Limitations and Roadmap](#11-known-limitations-and-roadmap)

---

## 1. Overview and Philosophy

### Purpose

The agent and orchestration layer is a **disciplined, auditable, safe multi-agent
platform** for research automation, continuous monitoring, and controlled decision
support in an institutional statistical arbitrage system.

Agents are not conversational assistants and not autonomous actors with broad
discretion. They are narrow-mandate analytical workers: each accepts a typed task,
produces a typed result, logs every significant decision to an audit trail, and
returns — without side effects — a structured output that a human or a higher-level
orchestration engine decides how to act upon.

### What This System Is NOT

- Not a free-form autonomous LLM reasoning chain with unbounded tool access.
- Not a chat-based agent that discovers its own goals.
- Not a broker-facing execution system (no agent submits orders directly).
- Not a system where agents modify shared state as a side effect of running.
- Not a black box: every recommendation is backed by a typed evidence bundle with
  full provenance, and every decision gate produces a permanent audit record.

### Three Guiding Principles

**1. Narrow Mandate.**
Each agent does exactly one job. Its `NAME` constant, `ALLOWED_TASK_TYPES` list,
and `AgentSpec.mission` string are the single contract. An agent that can do
everything is an agent that can be exploited to do anything — including things the
system was not designed to permit. Narrow mandate is the primary safety mechanism.

**2. Structured Evidence.**
Agents produce typed artifacts: `AgentEvidenceBundle`, `AgentRecommendation`,
`ChangeImpactReport`, `MonitoringSummary`. They never return raw dicts with implicit
semantics. Every artifact carries `producing_agent`, `task_id`, `timestamp`,
`confidence_overall`, and `warnings`. Downstream consumers always know where a
piece of information came from and how confident the producer was.

**3. Explicit Governance.**
No action above `ActionBoundary.RECOMMENDATION` executes without passing the
`GovernancePolicyEngine`. Sensitive actions (`RiskClass.HIGH_RISK` or
`RiskClass.SENSITIVE`) always require a human approval gate. The `ApprovalEngine`
maintains the full `ApprovalRequest → ApprovalDecision` chain. Emergency disable
controls are available for any agent pending investigation and are logged permanently.

---

## 2. Agent Taxonomy

The platform defines 37 agents across five classes. Each entry shows the module,
the `NAME` constant expected from `BaseAgent`, the accepted task types, the action
boundary ceiling, and the risk class.

### A. Research and Discovery Agents

| Agent | Module | NAME | Key Task Types | Action Boundary | Risk Class |
|-------|--------|------|---------------|-----------------|------------|
| UniverseCuratorAgent | agents/research_agents.py | `universe_curator` | `curate_universe`, `check_eligibility` | INFORMATIONAL | INFORMATIONAL |
| CandidateDiscoveryAgent | agents/research_agents.py | `candidate_discovery` | `discover_candidates`, `run_correlation_sweep` | INFORMATIONAL | BOUNDED_SAFE |
| RelationshipValidationAgent | agents/research_agents.py | `relationship_validation` | `validate_pair`, `batch_validate` | INFORMATIONAL | BOUNDED_SAFE |
| SpreadSpecificationAgent | agents/research_agents.py | `spread_specification` | `fit_spread`, `select_spread_model` | INFORMATIONAL | BOUNDED_SAFE |
| RegimeResearchAgent | agents/research_agents.py | `regime_research` | `analyse_regime_history`, `profile_regime_transitions` | INFORMATIONAL | BOUNDED_SAFE |
| SignalResearchAgent | agents/research_agents.py | `signal_research` | `backtest_signal_family`, `parameter_sensitivity` | INFORMATIONAL | BOUNDED_SAFE |
| ExperimentCoordinatorAgent | agents/research_agents.py | `experiment_coordinator` | `schedule_experiment`, `collect_results` | RECOMMENDATION | MEDIUM_RISK |
| ResearchSummarizationAgent | agents/research_agents.py | `research_summarization` | `summarize_discovery_run`, `draft_research_note` | INFORMATIONAL | INFORMATIONAL |

**Boundary notes:** All discovery agents are read-only analysts. They write to the
research artifact store (which is `BOUNDED_SAFE`) but never modify live system state.
`ExperimentCoordinatorAgent` reaches `RECOMMENDATION` because it may schedule
compute jobs, which requires a policy gate.

### B. ML and Model Agents

| Agent | Module | NAME | Key Task Types | Action Boundary | Risk Class |
|-------|--------|------|---------------|-----------------|------------|
| FeatureStewardAgent | agents/ml_agents.py | `feature_steward` | `audit_features`, `check_leakage`, `flag_drift` | INFORMATIONAL | INFORMATIONAL |
| LabelGovernanceAgent | agents/ml_agents.py | `label_governance` | `audit_labels`, `validate_horizon_alignment` | INFORMATIONAL | INFORMATIONAL |
| ModelResearchAgent | agents/ml_agents.py | `model_research` | `train_candidate`, `evaluate_oot` | RECOMMENDATION | BOUNDED_SAFE |
| MetaLabelingAgent | agents/ml_agents.py | `meta_labeling` | `run_meta_label_filter`, `score_signals` | RECOMMENDATION | MEDIUM_RISK |
| RegimeModelingAgent | agents/ml_agents.py | `regime_modeling` | `train_regime_classifier`, `compare_classifiers` | RECOMMENDATION | MEDIUM_RISK |
| ModelRiskAgent | agents/ml_agents.py | `model_risk` | `assess_model_risk`, `review_drift_report` | INFORMATIONAL | INFORMATIONAL |
| PromotionReviewAgent | agents/ml_agents.py | `promotion_review` | `review_promotion_request`, `check_governance_criteria` | RECOMMENDATION | HIGH_RISK |

**Boundary notes:** `PromotionReviewAgent` reaches `HIGH_RISK` because its
recommendation output feeds directly into the `PromotionGateAgent` approval chain.
The agent itself only recommends; the `ApprovalEngine` and a human approver must act.
No ML agent may call `registry.promote()` directly — this would be a PERMISSION_DENIED
violation.

### C. Signal and Portfolio Agents (existing layer)

| Agent | Module | NAME | Key Task Types | Action Boundary | Risk Class |
|-------|--------|------|---------------|-----------------|------------|
| SignalAnalystAgent | agents/signal_agents.py | `signal_analyst` | `analyse_signal`, `score_entry_intent` | INFORMATIONAL | INFORMATIONAL |
| RegimeSurveillanceAgent | agents/signal_agents.py | `regime_surveillance` | `classify_regime`, `monitor_regime_transition` | INFORMATIONAL | INFORMATIONAL |
| TradeLifecycleAgent | agents/signal_agents.py | `trade_lifecycle` | `check_lifecycle_state`, `recommend_transition` | RECOMMENDATION | MEDIUM_RISK |
| ExitOversightAgent | agents/signal_agents.py | `exit_oversight` | `assess_exit_signal`, `score_exit_intent` | RECOMMENDATION | MEDIUM_RISK |
| PortfolioConstructionAgent | agents/portfolio_agents.py | `portfolio_construction` | `rank_opportunities`, `check_constraints` | RECOMMENDATION | HIGH_RISK |
| CapitalBudgetAgent | agents/portfolio_agents.py | `capital_budget` | `review_sleeve_usage`, `flag_budget_breach` | INFORMATIONAL | MEDIUM_RISK |
| ExposureMonitorAgent | agents/portfolio_agents.py | `exposure_monitor` | `check_sector_exposure`, `check_cluster_concentration` | INFORMATIONAL | MEDIUM_RISK |
| DrawdownMonitorAgent | agents/portfolio_agents.py | `drawdown_monitor` | `update_heat_level`, `recommend_de_risk` | RECOMMENDATION | HIGH_RISK |
| KillSwitchAgent | agents/portfolio_agents.py | `kill_switch` | `evaluate_kill_conditions`, `recommend_halt` | CONTROLLED_OPERATIONAL | HIGH_RISK |
| DeRiskingAgent | agents/portfolio_agents.py | `de_risking` | `plan_de_risk`, `execute_de_risk_recommendation` | RECOMMENDATION | SENSITIVE |

**Boundary notes:** `KillSwitchAgent` reaches `CONTROLLED_OPERATIONAL` because
triggering a kill-switch recommendation creates a formal incident and may schedule
automated de-risking. `DeRiskingAgent` is `SENSITIVE` at the recommendation level
because its output may directly influence position-sizing decisions; it still only
recommends — execution is owned by the portfolio allocator cycle, not the agent.

### D. Monitoring and Incident Agents

| Agent | Module | NAME | Key Task Types | Action Boundary | Risk Class |
|-------|--------|------|---------------|-----------------|------------|
| SystemHealthAgent | agents/monitoring_agents.py | `system_health` | `check_agent_health`, `produce_monitoring_summary` | INFORMATIONAL | INFORMATIONAL |
| DriftMonitoringAgent | agents/monitoring_agents.py | `drift_monitoring` | `check_feature_drift`, `check_model_drift` | INFORMATIONAL | BOUNDED_SAFE |
| DataIntegrityAgent | agents/monitoring_agents.py | `data_integrity` | `validate_price_data`, `check_stale_feeds` | INFORMATIONAL | BOUNDED_SAFE |
| OrchestrationReliabilityAgent | agents/monitoring_agents.py | `orchestration_reliability` | `check_workflow_health`, `flag_stuck_runs` | CONTROLLED_OPERATIONAL | MEDIUM_RISK |
| IncidentTriageAgent | agents/monitoring_agents.py | `incident_triage` | `triage_alert`, `classify_severity` | CONTROLLED_OPERATIONAL | MEDIUM_RISK |
| PostmortemDraftingAgent | agents/monitoring_agents.py | `postmortem_drafting` | `draft_postmortem`, `extract_timeline` | INFORMATIONAL | INFORMATIONAL |
| AlertAggregationAgent | agents/monitoring_agents.py | `alert_aggregation` | `aggregate_alerts`, `deduplicate_alerts` | INFORMATIONAL | BOUNDED_SAFE |

**Boundary notes:** `OrchestrationReliabilityAgent` and `IncidentTriageAgent` reach
`CONTROLLED_OPERATIONAL` because they may create `IncidentRecord` objects and open
formal tickets in the incident manager — bounded write actions that require a policy
check but not human approval for routine severity levels (P3/P4).

### E. Governance and Review Agents

| Agent | Module | NAME | Key Task Types | Action Boundary | Risk Class |
|-------|--------|------|---------------|-----------------|------------|
| PolicyReviewAgent | agents/governance_agents.py | `policy_review` | `review_policy_version`, `flag_expiring_policies` | INFORMATIONAL | INFORMATIONAL |
| ApprovalRecommendationAgent | agents/governance_agents.py | `approval_recommendation` | `assess_approval_request`, `recommend_approval_mode` | RECOMMENDATION | MEDIUM_RISK |
| ChangeImpactAgent | agents/governance_agents.py | `change_impact` | `assess_change_impact`, `produce_change_report` | INFORMATIONAL | MEDIUM_RISK |
| PromotionGateAgent | agents/governance_agents.py | `promotion_gate` | `gate_model_promotion`, `check_promotion_criteria` | CONTROLLED_OPERATIONAL | HIGH_RISK |
| AuditTrailValidationAgent | agents/governance_agents.py | `audit_trail_validation` | `validate_audit_completeness`, `flag_gaps` | INFORMATIONAL | INFORMATIONAL |

**Boundary notes:** `PromotionGateAgent` reaches `CONTROLLED_OPERATIONAL` because
it writes `PromotionReviewRecord` objects and, upon successful gate check, forwards
an approval request to the `ApprovalEngine`. The actual `registry.promote()` call
only happens after human sign-off via `ApprovalDecision`.

---

## 3. Core Contract Model

### AgentTask and AgentResult

The fundamental unit of agent interaction is `AgentTask → AgentResult`, defined in
`core/contracts.py`.

```
AgentTask
  task_id          UUID for this invocation
  agent_name       Target agent (must match BaseAgent.NAME)
  task_type        Accepted by ALLOWED_TASK_TYPES
  payload          dict — validated against REQUIRED_PAYLOAD_KEYS by BaseAgent
  context          Optional dict for supplementary context
  created_at       ISO-8601 UTC timestamp
  priority         "low" | "normal" | "high" | "critical"
  timeout_seconds  Per-task execution budget

AgentResult
  task_id          Mirrors the request task_id
  agent_name       Producing agent name
  status           AgentStatus: SUCCESS | FAILED | SKIPPED | PARTIAL
  output           dict — always non-None; empty dict on failure
  error            Optional error message (populated when status=FAILED)
  warnings         list[str] — non-blocking issues observed during execution
  created_at       ISO-8601 UTC timestamp
  duration_ms      Wall-clock time for this execution
```

`BaseAgent.execute()` is the only public entry point. It validates payload keys,
calls `_execute()` inside a try/except, and guarantees that an `AgentResult` is
always returned — it never raises. Consumers must always check `result.status`
before using `result.output`.

### AgentSpec, AgentCapability, AgentPermissionSet

`AgentSpec` (in `orchestration/contracts.py`) is the complete static specification
for an agent stored in the registry. It captures identity, mission, capabilities,
permissions, and failure behaviour in a single immutable frozen dataclass that is
serializable to JSON.

```
AgentCapability
  name                  e.g. "read_prices", "write_research_artifact"
  description           Human-readable explanation
  action_boundary       Highest ActionBoundary this capability reaches
  risk_class            Risk class of exercising this capability
  requires_approval     Whether any step using this capability needs a gate
  max_autonomy_level    1 = informational, 2 = recommendation, 3 = controlled

AgentPermissionSet
  agent_name                        Target agent
  allowed_capabilities              Capability names permitted
  forbidden_capabilities            Names explicitly denied (overrides allowed)
  allowed_environments              EnvironmentClass values where agent may run
  max_delegation_depth              DelegationDepth enum (max = THREE)
  requires_human_review_above_risk  RiskClass threshold for automatic human gate
  emergency_disable                 If True, all tasks immediately rejected
```

The orchestration engine checks `AgentPermissionSet` at dispatch time. A capability
not listed in `allowed_capabilities`, or listed in `forbidden_capabilities`, causes
a `FailureClass.PERMISSION_DENIED` record. An agent with `emergency_disable=True`
causes immediate rejection of all tasks until re-enabled through the governance
engine.

### ActionBoundary Enum

`ActionBoundary` (in `orchestration/contracts.py`) is the four-level classification
of what an agent or workflow step is allowed to do. It drives approval-gate routing.

```
INFORMATIONAL            Observe, read, analyse — no external side effects
RECOMMENDATION           Propose, suggest, rank — no state mutation
CONTROLLED_OPERATIONAL   Bounded write or state change, requires policy check
SENSITIVE                Order submission, critical mutation, capital allocation
```

### RiskClass Enum

`RiskClass` (in `orchestration/contracts.py`) is the five-level risk classification
used by workflow steps, agent capabilities, and governance policies. Ordinal is
meaningful: higher ordinal = higher risk.

```
INFORMATIONAL   Read-only, no side effects — never needs approval
BOUNDED_SAFE    Writes research artifacts only — auto-approvable in RESEARCH env
MEDIUM_RISK     Modifies non-critical state — policy-gated
HIGH_RISK       Modifies critical state or config — human approval in PRODUCTION
SENSITIVE       Order submission, capital allocation — always human-required
```

### EnvironmentClass Enum

```
RESEARCH     Most permissive: auto-approval below HIGH_RISK
STAGING      Intermediate: policy-gated for MEDIUM_RISK and above
PAPER        Like STAGING but allows full signal/portfolio cycle
PRODUCTION   Most restrictive: human approval at HIGH_RISK, dual at SENSITIVE
```

---

## 4. Orchestration Model

### WorkflowDefinition, WorkflowRun, WorkflowStepRun

The orchestration layer separates **definition** (template) from **execution**
(runtime state).

```
WorkflowDefinition   (frozen)
  workflow_id          Stable identifier for this template
  name / description
  version              Semantic version string
  steps                tuple[WorkflowStep, ...] — ordered step definitions
  transitions          tuple[WorkflowTransition, ...] — conditional routing edges
  entry_condition      Human-readable trigger description
  termination_condition
  environment_class    Which environment this is designed for
  risk_class           Highest risk class in any step
  max_duration_seconds Hard wall-clock timeout
  idempotent           Whether re-running with same payload is safe
  replayable           Whether a failed run can continue from last success
```

```
WorkflowStep   (frozen)
  step_id                  Unique within the workflow definition
  agent_name               Target agent
  task_type                Task type string forwarded to the agent
  depends_on               tuple[str, ...] — step_ids that must complete first
  timeout_seconds
  retry_max
  risk_class
  requires_approval_before  Gate must pass before agent is invoked
  requires_approval_after   Gate must pass after agent completes
  on_failure               "fail_workflow" | "skip" | "escalate" | "retry"
```

```
WorkflowRun   (mutable)
  run_id
  workflow_id / workflow_name
  status                   WorkflowStatus: PENDING | RUNNING | COMPLETED | ...
  environment
  triggered_by             "schedule" | "event" | "manual" | "agent" | "incident"
  trigger_payload          Input dict at trigger time
  step_runs                list[WorkflowStepRun]
  artifact_ids             Artifacts produced across all steps
  recommendation_ids       Recommendations generated during run
  approval_request_ids     Approval requests opened during run
```

```
WorkflowStepRun   (mutable)
  step_run_id
  step_id / workflow_run_id
  status               WorkflowStepStatus: PENDING | RUNNING | COMPLETED | ...
  started_at / completed_at
  agent_task_id        UUID of the dispatched AgentTask
  agent_result_status  Value from AgentResult.status
  retry_count
  error
  output_summary       One-line description of what this step produced
  artifact_ids         Artifacts produced by this step
```

### WorkflowEngine

The `WorkflowEngine` (in `orchestration/engine.py`) is the central coordinator:

1. **Dependency resolution.** Steps with `depends_on` are held in `PENDING` until
   all upstream steps have status `COMPLETED`. Steps with no dependencies run
   immediately.

2. **Approval gate integration.** Before dispatching any step with
   `requires_approval_before=True`, the engine submits an `ApprovalRequest` and
   waits. Steps with `requires_approval_after=True` are similarly held for post-run
   review. If the approval is rejected, the step transitions to `SKIPPED` (or
   `FAILED` depending on `on_failure`).

3. **Retry semantics.** Steps with `on_failure="retry"` are re-dispatched up to
   `retry_max` times. The `WorkflowStepRun.retry_count` is incremented on each
   attempt. After exhaustion, the step fails with `FailureClass.RETRY_EXHAUSTED`.

4. **Timeout enforcement.** Both per-step `timeout_seconds` and workflow-level
   `max_duration_seconds` are enforced. Timed-out steps produce
   `FailureClass.TIMEOUT`.

5. **Cancellation.** `WorkflowEngine.cancel(run_id)` sets the run to `CANCELLED`
   and skips all pending steps. Running steps are allowed to complete their current
   operation before termination.

6. **Replay semantics.** For `WorkflowDefinition.replayable=True`, a failed run
   can be resumed from the last `COMPLETED` step. Steps already in `COMPLETED`
   state are not re-executed; their prior output is passed to the next step in
   dependency order. This is safe only for idempotent agent operations.

### Pre-Built Workflows

Three canonical workflows ship with the platform:

**`research_discovery` workflow**

```
Steps:
  1. curate_universe        (UniverseCuratorAgent)       BOUNDED_SAFE
  2. discover_candidates    (CandidateDiscoveryAgent)    BOUNDED_SAFE, depends_on=[1]
  3. validate_pairs         (RelationshipValidationAgent) BOUNDED_SAFE, depends_on=[2]
  4. fit_spreads            (SpreadSpecificationAgent)   BOUNDED_SAFE, depends_on=[3]
  5. analyse_regime         (RegimeResearchAgent)        BOUNDED_SAFE, depends_on=[4]
  6. summarize_discovery    (ResearchSummarizationAgent) INFORMATIONAL, depends_on=[5]

Entry condition:    Scheduled weekly, or triggered by universe change event
Termination:        All steps completed, or step 1 or 2 hard-fail
Environment:        RESEARCH
Risk class:         BOUNDED_SAFE
Replayable:         True
Idempotent:         True
```

**`model_promotion` workflow**

```
Steps:
  1. assess_model_risk      (ModelRiskAgent)             INFORMATIONAL
  2. review_promotion       (PromotionReviewAgent)       HIGH_RISK, requires_approval_before=True
  3. gate_promotion         (PromotionGateAgent)         HIGH_RISK, depends_on=[2]
  4. summarize_promotion    (ResearchSummarizationAgent) INFORMATIONAL, depends_on=[3]

Entry condition:    Triggered by operator or PromotionReviewAgent recommendation
Termination:        Step 3 completed (approved) or approval rejected
Environment:        PRODUCTION
Risk class:         HIGH_RISK
Replayable:         False  (promotion actions are not idempotent)
Idempotent:         False
```

**`drift_alert` workflow**

```
Steps:
  1. check_drift            (DriftMonitoringAgent)       BOUNDED_SAFE
  2. aggregate_alerts       (AlertAggregationAgent)      BOUNDED_SAFE, depends_on=[1]
  3. triage_alert           (IncidentTriageAgent)        MEDIUM_RISK, depends_on=[2]
  4. route_to_incident      (IncidentTriageAgent)        MEDIUM_RISK, depends_on=[3]

Transitions:
  3 → 4: on_success (drift confirmed as material)
  3 → END: on_success (alert was noise; aggregated, no incident)

Entry condition:    Drift monitor fires threshold breach
Termination:        Triage complete and incident opened, or alert resolved as noise
Environment:        PAPER / PRODUCTION
Risk class:         MEDIUM_RISK
Replayable:         True
Idempotent:         False
```

---

## 5. Governance and Approval Model

### GovernancePolicyEngine

`GovernancePolicyEngine` (in `governance/engine.py`) is the single authority for
policy evaluation. It is thread-safe (single `threading.Lock`) and accessible as a
singleton via `get_governance_engine()`.

Five default policies are registered at startup:

| Policy ID | Name | Scope | What It Enforces |
|-----------|------|-------|-----------------|
| `NO_RISK_LIMIT_OVERRIDE` | No Risk Limit Override | ALL environments | Agents may never override hard risk limits set by `DrawdownManager` or `KillSwitchManager` — not even with explicit task payload instructions |
| `MAX_DELEGATION_DEPTH_3` | Max Delegation Depth | ALL environments | Agent-to-agent delegation depth is capped at `DelegationDepth.THREE`; a depth of 4 or more causes `FailureClass.POLICY_VIOLATION` |
| `PRODUCTION_HIGH_RISK_APPROVAL` | Production High-Risk Approval | PRODUCTION only | Any action classified `HIGH_RISK` or `SENSITIVE` in production requires a passing `ApprovalDecision` before execution |
| `NO_SILENT_MODEL_RETRAIN` | No Silent Model Retrain | PAPER, PRODUCTION | Retraining a model without an open `ApprovalRequest` in the governance log is forbidden; `FeatureStewardAgent` and `ModelResearchAgent` check this on every training task |
| `AUDIT_TRAIL_REQUIRED` | Audit Trail Required | ALL environments | Every material action (any `ActionBoundary` above `INFORMATIONAL`) must produce at least one `AuditRecord`; steps with zero audit entries after completion are flagged as `POLICY_VIOLATION` |

Each policy is stored as a `GovernancePolicyVersion` (frozen dataclass) with
`policy_id`, `version`, `status`, `effective_from`, `rules`, `risk_class`,
`environment_scope`, and `next_review`. Policies are versioned and immutable —
to update a policy, deprecate the old version and register a new one.

### ApprovalEngine

`ApprovalEngine` (in `approvals/engine.py`) manages the full lifecycle of approval
requests. It is thread-safe and accessible as a singleton via `get_approval_engine()`.

**ApprovalMode values and when each applies:**

| Mode | When Used | Auto-Resolves? |
|------|-----------|----------------|
| `AUTOMATIC` | `INFORMATIONAL` tasks in any environment; `BOUNDED_SAFE` in RESEARCH | Yes — immediately approved |
| `POLICY_GATED` | `MEDIUM_RISK` actions; `BOUNDED_SAFE` in STAGING/PAPER | Yes — if all policy checks pass |
| `HUMAN_REQUIRED` | `HIGH_RISK` actions in PRODUCTION; any `SENSITIVE` action | No — requires human `ApprovalDecision` |
| `DUAL_APPROVAL` | `SENSITIVE` actions in PRODUCTION with capital implications | No — requires two separate human approvers |
| `BLOCKED` | Actions categorically forbidden (e.g., emergency-disabled agents) | Permanent rejection |

**ApprovalRequest → ApprovalDecision chain:**

```
1. Agent or WorkflowEngine creates ApprovalRequest:
     request_id, agent_name, task_id, action_type,
     risk_class, environment, evidence_bundle_ids,
     approval_mode, required_approvers, expires_at

2. ApprovalEngine routes based on approval_mode:
     AUTOMATIC       → create AUTO_APPROVED ApprovalDecision immediately
     POLICY_GATED    → call GovernancePolicyEngine.check_policy()
                         pass → AUTO_APPROVED decision
                         fail → create HumanReviewTicket, route to reviewer queue
     HUMAN_REQUIRED  → create HumanReviewTicket with priority from urgency field
     DUAL_APPROVAL   → create two HumanReviewTickets for two distinct approvers

3. Human reviewer acts:
     ApprovalDecision:
       decision_id, request_id, decided_by, status (APPROVED | REJECTED),
       rationale, conditions, evidence_bundle_ids, decided_at

4. Workflow resumes (APPROVED) or step is skipped/failed (REJECTED)
```

**HumanReviewTicket lifecycle:**
`OPEN → IN_REVIEW → DECIDED → CLOSED`
Tickets with `expires_at` that pass without action transition to
`ApprovalStatus.EXPIRED` and the associated workflow step is failed with
`FailureClass.APPROVAL_REJECTED`.

### Emergency Disable Controls

Any agent can be suspended immediately via:

```python
governance_engine.emergency_disable("agent_name", reason="Under investigation: P1-042")
```

This sets `AgentPermissionSet.emergency_disable=True` for the named agent. All
subsequent `AgentRegistry.dispatch()` calls for that agent return an `AgentResult`
with `status=FAILED` and `error="Agent emergency_disabled"`. The suspension is
logged as an `AuditRecord` with `AuditRecordType.EMERGENCY_ACTION`.

Re-enabling requires:
```python
governance_engine.reenable_agent("agent_name", authorized_by="ops_lead")
```
This also produces a permanent `AuditRecord`. Emergency disable history is never
deleted.

---

## 6. Incident and Audit Model

### IncidentRecord and Severity Tiers

`IncidentRecord` (mutable dataclass, `incidents/contracts.py`) is the central
tracking object for any operational event that requires investigation or response.

| Severity | Value | Meaning | Response |
|----------|-------|---------|----------|
| P0_CRITICAL | `"P0"` | Complete system or trading halt | Immediate response; all active workflows paused |
| P1_HIGH | `"P1"` | Major degradation; significant alpha impact | High-priority response within 30 minutes |
| P2_MEDIUM | `"P2"` | Partial degradation; non-critical path affected | Response within business hours |
| P3_LOW | `"P3"` | Minor issue; easily recoverable | Response within SLA window |
| P4_INFO | `"P4"` | Informational; no response urgency | Logged for pattern analysis |

**IncidentRecord fields:**
```
incident_id     UUID
title           Short human-readable title
description     Full description of the issue
severity        IncidentSeverity
status          OPEN | IN_PROGRESS | MONITORING | RESOLVED | CLOSED | POSTMORTEM
created_at / updated_at / resolved_at
created_by      Identity of reporter (agent name or human)
assigned_to     Optional current owner
affected_agents        list[str] — agent names involved
affected_workflows     list[str] — workflow run IDs involved
affected_components    list[str] — system component identifiers
timeline        list[dict] — ordered event log (timestamp, event, author)
runbook_ids     list[str] — linked RunbookReference identifiers
tags            list[str]
```

### IncidentManager Lifecycle

`IncidentManager` (in `incidents/manager.py`) is the entry point for all incident
operations. Singleton access via `get_incident_manager()`.

```
create_incident(title, severity, description, created_by, ...) → IncidentRecord
  - Assigns severity-appropriate runbook references automatically by keyword match
  - Writes AuditRecord(AuditRecordType.INCIDENT_CREATED)
  - For P0/P1: immediately broadcasts to all registered watchers

update_incident(incident_id, status=None, assigned_to=None, timeline_event=None)
  - Appends to timeline; writes AuditRecord(AuditRecordType.INCIDENT_UPDATED)
  - Status MONITORING means: incident known, being watched, no active remediation

close_incident(incident_id, resolution_notes) → None
  - Sets status=CLOSED; writes final AuditRecord
  - For P0/P1: triggers create_postmortem() automatically

create_postmortem(incident_id, drafted_by) → PostmortemArtifact
  - Assembled from incident timeline, affected agents, runbook references
  - status=POSTMORTEM on the incident
  - Stored as ArtifactType.POSTMORTEM
```

### RunbookReference

`RunbookReference` (frozen dataclass) links an incident to a standard operating
procedure. Fields: `runbook_id`, `title`, `description`, `applicable_severity`,
`applicable_components`, `steps` (ordered procedure), `url`, `version`, `last_reviewed`.

The `IncidentManager` pre-registers runbooks at startup covering:
- Kill-switch activation response
- Model drift escalation
- Data feed failure
- Agent permission violation
- Workflow deadlock recovery

### AuditRecord and AuditRecordType

`AuditRecord` (frozen dataclass, `incidents/contracts.py`) is the fundamental unit
of the append-only audit ledger.

```
AuditRecordType values:
  AGENT_EXECUTION       — Every BaseAgent.execute() completion
  POLICY_CHECK          — Every GovernancePolicyEngine.check_policy() evaluation
  APPROVAL_DECISION     — Every ApprovalDecision created
  WORKFLOW_TRANSITION   — Every WorkflowStepRun status change
  DELEGATION            — Every agent sub-delegation
  ESCALATION            — Every escalation event (approval, incident)
  EMERGENCY_ACTION      — Emergency disable/reenable of an agent
  INCIDENT_CREATED      — Incident creation
  INCIDENT_UPDATED      — Incident timeline update
  OVERRIDE              — Any manual override of a governance decision
```

Audit records are never deleted and never modified. The audit log is the safety
proof that the system operated within its declared boundaries.

### PostmortemArtifact Schema

```
PostmortemArtifact   (frozen)
  postmortem_id        UUID
  incident_id          Links to IncidentRecord
  artifact_type        ArtifactType.POSTMORTEM
  schema_version       "1.0"
  incident_title
  incident_severity
  timeline_summary     Ordered list of key timeline events
  affected_agents
  affected_workflows
  root_cause           Assessed root cause (human-drafted)
  contributing_factors list[str]
  impact_summary       Business and operational impact
  remediation_steps    What was done to resolve
  prevention_items     Proposed actions to prevent recurrence
  runbook_updates_needed  Which runbooks require revision
  drafted_by
  drafted_at
  reviewed_by          Optional reviewer identity
  reviewed_at
  open_action_items    list[dict] — owner, description, due_date
```

---

## 7. Multi-Agent Coordination Patterns

### Pattern 1: Fan-Out / Fan-In (Discovery Sweep)

Used when parallel, independent analyses must be collected before a synthesis step.

```
Orchestrator (WorkflowEngine)
  ├── CandidateDiscoveryAgent [CORRELATION family]  ──┐
  ├── CandidateDiscoveryAgent [CLUSTER family]      ──┤
  ├── CandidateDiscoveryAgent [DISTANCE family]     ──┼──> ResearchSummarizationAgent
  └── CandidateDiscoveryAgent [COINTEGRATION family]──┘

All discovery steps have depends_on=[] (parallel).
Summarization step has depends_on=[all four discovery step_ids].
The engine waits for all four COMPLETED statuses before dispatching summarization.
```

**Constraint:** The fan-out depth is bounded by `max_delegation_depth=3`. A
discovery agent cannot itself spawn a second level of parallel sub-agents. If
additional depth is needed, it must be modelled as additional workflow steps in the
`WorkflowDefinition`, not as agent-to-agent delegation.

### Pattern 2: Proposal → Review → Approval → Act

The canonical pattern for any action above `RECOMMENDATION` boundary.

```
1. PromotionReviewAgent produces AgentRecommendation(recommendation_type="promote_model")
2. WorkflowEngine sees requires_approval_before=True on next step
3. ApprovalEngine creates ApprovalRequest, routes to HUMAN_REQUIRED queue
4. Human reviewer creates ApprovalDecision(status=APPROVED, rationale=...)
5. WorkflowEngine resumes: PromotionGateAgent executes
6. PromotionGateAgent calls GovernancePolicyEngine.check_policy()
7. If policy passes: PromotionGateAgent writes PromotionReviewRecord
8. Governance engine logs AuditRecord(APPROVAL_DECISION)
9. ML registry.promote() is called — only now, not before
```

At step 9 the actual state mutation occurs. Before that, every object produced
(recommendation, approval request, approval decision, policy check result) is a
read-only structured record. This makes every promotion fully replayable from the
audit log.

### Pattern 3: Watchdog (Governance Agent Wrapping Research Agent)

Used when a research or ML agent's output must always be reviewed before it feeds
downstream.

```
ModelResearchAgent._execute() → output dict
  ↓  (WorkflowEngine step transition)
ModelRiskAgent._execute(context=prior_output) → risk assessment
  ↓  (if risk assessment flags HIGH_RISK)
WorkflowTransition: on_success → ApprovalRequest created
  ↓  (if risk assessment is INFORMATIONAL)
WorkflowTransition: on_success → next research step directly
```

The watchdog agent (ModelRiskAgent) does not modify the prior agent's output — it
appends its own risk assessment as a separate `AgentEvidenceBundle`. The
orchestration engine routes based on the watchdog's result.

### Pattern 4: Challenger vs Champion Comparison

Used to evaluate whether a new model version should replace the current champion.

```
1. ModelResearchAgent produces CandidateModel artifact (training complete)
2. ModelRiskAgent compares Candidate vs current Champion on OOT holdout
3. AgentRecommendation: "challenger_better" | "champion_retained" | "inconclusive"
4. If "challenger_better": trigger model_promotion workflow
5. If "champion_retained" or "inconclusive": log result, no further action
```

The comparison step always runs in `RESEARCH` or `STAGING` environment. The
subsequent promotion workflow runs in `PRODUCTION`. These are two separate
`WorkflowDefinition` objects — not one workflow spanning environments.

### Pattern 5: Bounded Delegation Depth

`DelegationDepth.THREE` is the hard ceiling. The engine enforces this at dispatch
time by tracking the current delegation depth in the `AgentTask` payload. Any
attempt to delegate beyond depth 3 results in `FailureClass.POLICY_VIOLATION`.

In practice, the system rarely uses depth beyond 2:
- Depth 0: Human or scheduler triggers workflow
- Depth 1: WorkflowEngine dispatches to a primary agent (e.g., ExperimentCoordinatorAgent)
- Depth 2: ExperimentCoordinatorAgent dispatches a sub-task to ModelResearchAgent
- Depth 3: ModelResearchAgent may invoke FeatureStewardAgent for a leakage check

Depth 3 is reserved for inline validation checks only. No agent at depth 3 may
return a recommendation that triggers another delegation.

---

## 8. Action Boundaries

The `ActionBoundary` enum governs what each class of agent activity is permitted to
do and what approval requirements apply.

| Boundary | Operations Permitted | Examples | Approval Required? |
|----------|---------------------|----------|-------------------|
| `INFORMATIONAL` | Read, analyse, summarize, diagnose, compute metrics | Classify regime, score signal quality, check data freshness, produce monitoring summary | Never |
| `RECOMMENDATION` | Propose, rank, flag, suggest, produce evidence bundle | Recommend model promotion, propose de-risking plan, rank opportunities | May require policy check depending on `RiskClass`; no human approval in RESEARCH |
| `CONTROLLED_OPERATIONAL` | Open incident ticket, schedule compute job, create approval request, flag agent for review | Triage alert → create incident, orchestration reliability check → cancel stuck workflow, promotion gate → write PromotionReviewRecord | Always policy-gated; human approval required in PRODUCTION for `HIGH_RISK` |
| `SENSITIVE` | Modify risk limits, enable/disable kill-switch, alter capital allocation parameters | (None currently implemented; reserved for future live-operations agents) | Always `HUMAN_REQUIRED` or `DUAL_APPROVAL`; never auto-approved |

**Decision matrix by environment and risk class:**

| Environment | INFORMATIONAL | BOUNDED_SAFE | MEDIUM_RISK | HIGH_RISK | SENSITIVE |
|-------------|---------------|--------------|-------------|-----------|-----------|
| RESEARCH | Auto | Auto | Policy-gated | Policy-gated | Human |
| STAGING | Auto | Auto | Policy-gated | Human | Human |
| PAPER | Auto | Auto | Policy-gated | Human | Dual |
| PRODUCTION | Auto | Auto | Policy-gated | Human | Dual |

---

## 9. Safety Doctrine

The eleven principles that govern every design decision in the agent layer.

**1. Agents are specialists, not sovereigns.**
An agent's mandate is precisely its `AgentSpec.mission` string. Agents do not
infer goals, expand scope, or decide that a related task is worth doing. If a task
arrives that is not in `ALLOWED_TASK_TYPES`, it is rejected — not redirected.

**2. Every result is typed and auditable.**
`AgentResult.output` is always a dict with known keys declared in the agent's
docstring. Untyped dicts with opaque semantics are never acceptable. The producing
agent's `NAME` is always traceable from any downstream artifact.

**3. No side effects inside `_execute()`.**
Agents compute and return. They do not mutate shared module-level state, do not
write to disk, do not call external services directly (all I/O is abstracted through
injected dependencies), and do not trigger other agents. The only exception is
logging to the `AgentAuditLogger` passed as an argument.

**4. Failures are first-class results, not exceptions.**
`BaseAgent.execute()` catches all exceptions inside `_execute()` and converts them
to `AgentResult(status=FAILED)`. The workflow engine receives a typed failure result,
not a Python exception that could crash the orchestration loop.

**5. The governance engine is not optional.**
For any action above `ActionBoundary.INFORMATIONAL`, the workflow step must be
configured with `risk_class` and `requires_approval_before/after` fields. The
engine enforces these automatically. There is no code path in the orchestration
layer that dispatches a `CONTROLLED_OPERATIONAL` step without a policy check.

**6. Emergency disable is always available.**
If an agent behaves unexpectedly, it can be suspended in a single call to
`governance_engine.emergency_disable()` without restarting any process. The
suspension is instant, permanent in the log, and requires an explicit re-enable.

**7. Delegation depth is bounded.**
`max_delegation_depth=3` is enforced by the engine. This prevents unbounded
recursive agent spawning that could exhaust resources or obscure the audit trail.
The depth limit is checked at dispatch time, not at task creation time.

**8. Approval history is immutable.**
`ApprovalRequest` and `ApprovalDecision` are frozen dataclasses. Once written to
the `ApprovalEngine`, they are never modified. Overrides are modelled as new
`OverrideRecord` objects that reference the original decision, not as mutations.

**9. The audit trail IS the safety proof.**
The AuditRecord ledger is append-only. Any material action that cannot be traced
back to an `AuditRecord` is a `POLICY_VIOLATION` under the `AUDIT_TRAIL_REQUIRED`
policy. The audit trail is the only trustworthy record of what the system did and
why. It must never be bypassed for performance.

**10. ML never overrides hard risk rules.**
This principle from the ML layer applies equally here: no agent, regardless of its
confidence score or recommendation urgency, may suggest an action that would bypass
`DrawdownManager.heat_level`, `KillSwitchManager.mode`, or
`TradeLifecycleStateMachine` hard vetoes. These are inviolable. An agent's
`AgentRecommendation` with `recommendation_type="adjust_risk_limit"` is always
`SENSITIVE` and always requires dual human approval.

**11. Agents propose; humans and the portfolio cycle dispose.**
The system is designed for controlled decision support, not autonomous execution.
The portfolio allocator cycle, operating under explicit risk budgets, makes
allocation decisions. The kill-switch and drawdown manager enforce hard limits.
Agents surface information and proposals into this cycle — they do not replace it.

---

## 10. Integration with Core Platform Layers

The table below shows which agents interact with which platform layer and what they
are and are not permitted to do within that layer.

| Agent Class | Platform Layer | May Read | May Write/Trigger | Must NOT |
|-------------|---------------|----------|-------------------|----------|
| Research/Discovery | `research/` | All: `universe.py`, `candidate_generator.py`, `pair_validator.py`, `spread_constructor.py`, `stability_analysis.py` | Write research artifact store only | Call `DiscoveryPipeline.run()` with `train_end=None` in backtest; modify `ValidationThresholds` |
| ML/Model | `ml/` | All: features, labels, datasets, registry, inference, monitoring, governance | Write experiment artifacts; call `registry.register()` for CANDIDATE | Call `registry.promote()` directly; call `model.predict()` directly (use `ModelScorer.score()`) |
| Signal/Portfolio | `core/signals_engine.py`, `core/lifecycle.py`, `core/threshold_engine.py`, `core/signal_quality.py`, `portfolio/` | All signal and portfolio state | Recommend via `AgentRecommendation` | Mutate `TradeLifecycleStateMachine` state; modify `ThresholdConfig` in place; call `PortfolioAllocator.run_cycle()` |
| Monitoring/Incident | `orchestration/contracts.py`, `incidents/manager.py`, `agents/registry.py` | All health metrics; `AgentHealthStatus`; workflow run states | Create `IncidentRecord`; update incident timeline; create `AlertBundle` | Close or resolve incidents without human confirmation (P0/P1); modify agent `AgentPermissionSet` |
| Governance/Review | `governance/engine.py`, `approvals/engine.py`, `incidents/contracts.py` | All policy versions, approval history, audit records | Write `PromotionReviewRecord`; submit `ApprovalRequest`; log `AuditRecord` | Register new policies without operator authorization; auto-approve `HIGH_RISK` actions in PRODUCTION; delete audit records |

**Key cross-layer constraints:**

- Research agents must always pass `train_end` to `DiscoveryPipeline.run()`. The
  leakage contract from the ML layer extends to the agent layer.

- Signal and portfolio agents receive `EntryIntent` and `ExitIntent` objects as
  read inputs. They must never construct these objects themselves — intents are
  produced by `core/signals_engine.py` and `core/threshold_engine.py`.

- Monitoring agents have `READ_ONLY` permission in `AgentRegistry`. They may query
  health status but never call `dispatch()` themselves (which would create a
  feedback loop in health monitoring).

- Governance agents have `RESEARCH` permission for artifact writes and `ADMIN`
  permission for policy registration — but admin permission is only active for the
  `GovernancePolicyEngine.register_policy()` call, not for `emergency_disable()`,
  which is reserved for human operators.

---

## 11. Known Limitations and Roadmap

### Current Limitations

**In-memory workflow store.**
`WorkflowEngine._runs` is a Python dict. There is no persistent disk-based workflow
store. If the process restarts mid-workflow, all in-progress runs are lost. The
`replayable=True` flag is currently advisory — it documents intent but cannot be
actioned without a durable run store.

**Single-process task queue.**
Agent dispatch is synchronous and single-threaded per workflow run. There is no
distributed task queue (Celery, Ray, or similar). High-volume fan-out workflows
(e.g., large discovery sweeps over 500+ pairs) will be sequentially slow. Parallel
steps within a workflow run today only if the engine is extended with a thread pool
executor.

**No LLM reasoning layer.**
All agents are rule-based analyzers. They compute metrics, apply thresholds, and
produce structured outputs. There is no LLM inference hook for narrative reasoning,
anomaly explanation, or adaptive threshold suggestion. The agent architecture is
designed to accommodate LLM reasoning hooks (via `AgentContextPackage` and
`ContextReference`) but none are wired yet.

**No broker-facing execution agents.**
`DeRiskingAgent` and `KillSwitchAgent` recommend actions but cannot submit orders
to IBKR or any other broker. There are no execution agents. All order submission
remains a manual or a separate out-of-band process.

**In-memory audit log bounded.**
`AgentRegistry._audit_log` is bounded at 10,000 entries (configurable via
`max_audit_log` in `AgentRegistry.__init__`). Entries beyond this limit are
silently dropped. There is no durable audit database.

**No versioned artifact store.**
Research artifacts, evidence bundles, and recommendations are produced as Python
objects and may be stored in in-memory dicts. There is no versioned, queryable
artifact database. Artifact references (`ArtifactReference`) encode provenance
metadata but point to objects that may not survive process restart.

### Roadmap

1. **Durable workflow store.** Add SQLite or PostgreSQL-backed `WorkflowRunStore`
   so that workflow state survives restart and `replayable=True` workflows can be
   resumed correctly.

2. **Async task queue.** Integrate an async dispatch layer (e.g., `asyncio`-native
   or lightweight broker) to allow true parallel execution of fan-out steps within
   a workflow run.

3. **Durable audit database.** Replace the bounded in-memory list with a SQLite
   append-only table. Expose a query interface for compliance review.

4. **LLM reasoning hooks.** Define a `ReasoningHookProtocol` analogous to
   `RegimeClassifierHookProtocol` in `core/regime_engine.py`. Research agents that
   produce `AgentEvidenceBundle` objects could optionally pass those to an LLM
   hook for narrative synthesis before writing the `ResearchSummarizationAgent`
   output.

5. **Live operations agents.** Design `OrderRoutingAgent` and
   `PositionReconciliationAgent` as `SENSITIVE`-boundary agents that interface with
   the IBKR connector. These will require dual human approval in production and a
   full broker-adapter abstraction layer.

6. **Versioned artifact store.** Implement a content-addressed artifact store where
   each artifact is keyed by `(producing_agent, task_id, schema_version)` and
   stored durably with a retrieval API.

7. **Policy hot-reload.** Allow `GovernancePolicyEngine` to reload policy versions
   from a config file without process restart, with a grace period during which
   in-flight tasks complete under the old policy.
