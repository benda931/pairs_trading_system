# -*- coding: utf-8 -*-
"""
agent_artifacts/contracts.py — Agent Artifact Domain Contracts
================================================================

Typed artifact types used across the agent and orchestration layer:
monitoring summaries, alert bundles, and experiment summaries.

Also provides the ``ArtifactType`` enumeration used throughout the platform
to tag produced artifacts with a semantic category.

Design principles:
- All types are frozen dataclasses (immutable after construction).
- ``dict`` fields (parameters, results, etc.) are used for open-ended payloads
  where a closed schema is impractical (experiment parameters vary by type).
- ``tuple[dict, ...]`` is used for ordered homogeneous collections.
- Enums inherit from str for JSON-serialization compatibility.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


# ══════════════════════════════════════════════════════════════════
# 1. ARTIFACT TYPE ENUMERATION
# ══════════════════════════════════════════════════════════════════


class ArtifactType(str, enum.Enum):
    """Semantic type tags for all produced artifacts on the platform.

    Used to label artifact references in governance, audit, and evidence
    bundles so consumers can filter and route by type.
    """

    AGENT_TASK_REQUEST = "AGENT_TASK_REQUEST"
    AGENT_TASK_RESULT = "AGENT_TASK_RESULT"
    EVIDENCE_BUNDLE = "EVIDENCE_BUNDLE"
    RECOMMENDATION = "RECOMMENDATION"
    POLICY_CHECK = "POLICY_CHECK"
    APPROVAL_REQUEST = "APPROVAL_REQUEST"
    APPROVAL_DECISION = "APPROVAL_DECISION"
    WORKFLOW_STATE = "WORKFLOW_STATE"
    INCIDENT = "INCIDENT"
    REMEDIATION = "REMEDIATION"
    AUDIT = "AUDIT"
    CHANGE_IMPACT = "CHANGE_IMPACT"
    MODEL_REVIEW = "MODEL_REVIEW"
    EXPERIMENT_SUMMARY = "EXPERIMENT_SUMMARY"
    MONITORING_SUMMARY = "MONITORING_SUMMARY"
    ALERT_BUNDLE = "ALERT_BUNDLE"
    POSTMORTEM = "POSTMORTEM"


# ══════════════════════════════════════════════════════════════════
# 2. MONITORING SUMMARY
# ══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class MonitoringSummary:
    """Periodic operational health summary produced by a monitoring agent.

    Covers agent health, workflow outcomes, incident and approval counts,
    policy violations, drift alerts, and top-failing components.

    Fields
    ------
    summary_id : str
        UUID string.
    produced_by : str
        Name of the agent or process that produced this summary.
    period_start : str
        ISO 8601 timestamp for the start of the monitoring period.
    period_end : str
        ISO 8601 timestamp for the end of the monitoring period.
    agent_health : tuple[dict, ...]
        Per-agent health entries. Each dict should contain at minimum:
        ``agent_name`` (str), ``status`` (str), ``task_count`` (int),
        ``error_rate`` (float).
    workflow_outcomes : tuple[dict, ...]
        Per-workflow outcome entries. Each dict should contain at minimum:
        ``workflow_type`` (str), ``run_count`` (int), ``success_rate`` (float),
        ``avg_duration_seconds`` (float).
    incident_count : int
        Number of incidents opened during the period.
    approval_count : int
        Number of approval requests processed during the period.
    approval_acceptance_rate : float
        Fraction of approval requests that were approved [0.0, 1.0].
    policy_violation_count : int
        Number of governance policy violations during the period.
    drift_alerts : int
        Number of feature drift alerts fired during the period.
    escalation_count : int
        Number of escalations during the period.
    top_failing_agents : tuple[str, ...]
        Agent names with the highest error rates, ordered descending.
    top_failing_workflows : tuple[str, ...]
        Workflow types with the highest failure rates, ordered descending.
    recommendations : tuple[str, ...]
        Actionable recommendations derived from the monitoring data.
    timestamp : str
        ISO 8601 timestamp when this summary was produced.
    """

    summary_id: str
    produced_by: str
    period_start: str
    period_end: str
    agent_health: Tuple[Dict, ...]
    workflow_outcomes: Tuple[Dict, ...]
    incident_count: int
    approval_count: int
    approval_acceptance_rate: float
    policy_violation_count: int
    drift_alerts: int
    escalation_count: int
    top_failing_agents: Tuple[str, ...]
    top_failing_workflows: Tuple[str, ...]
    recommendations: Tuple[str, ...]
    timestamp: str


# ══════════════════════════════════════════════════════════════════
# 3. ALERT BUNDLE
# ══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class AlertBundle:
    """A batched collection of alerts produced in a monitoring cycle.

    AlertBundles are produced by monitoring agents and consumed by the
    incident management and escalation layers. When
    ``requires_immediate_action=True``, consumers must process
    ``action_items`` before the next monitoring cycle.

    Fields
    ------
    bundle_id : str
        UUID string.
    produced_by : str
        Agent or process that produced this bundle.
    timestamp : str
        ISO 8601 timestamp.
    alert_count : int
        Total number of alerts in this bundle.
    alerts : tuple[dict, ...]
        Individual alert entries. Each dict should contain at minimum:
        ``alert_id`` (str), ``source`` (str), ``severity`` (str),
        ``message`` (str), ``timestamp`` (str), ``component`` (str).
    severity_breakdown : dict
        Counts per severity level (e.g. {"P0": 0, "P1": 2, "P2": 5}).
    source_breakdown : dict
        Counts per source component (e.g. {"drift_monitor": 3, "risk_engine": 2}).
    requires_immediate_action : bool
        True if any alert in this bundle requires immediate operator response.
    action_items : tuple[str, ...]
        Ordered list of immediate action items for the on-call operator.
    """

    bundle_id: str
    produced_by: str
    timestamp: str
    alert_count: int
    alerts: Tuple[Dict, ...]
    severity_breakdown: Dict
    source_breakdown: Dict
    requires_immediate_action: bool
    action_items: Tuple[str, ...]


# ══════════════════════════════════════════════════════════════════
# 4. EXPERIMENT SUMMARY
# ══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ExperimentSummary:
    """Summary of a completed research or optimization experiment.

    Produced by research agents, optimization agents, or walk-forward
    harnesses. Captures parameters, results, and comparison to baseline.

    Fields
    ------
    summary_id : str
        UUID string.
    experiment_name : str
        Human-readable name for the experiment.
    experiment_type : str
        Category (e.g. "WALK_FORWARD_OPTIMIZATION", "REGIME_BACKTEST",
        "HYPERPARAMETER_SEARCH", "UNIVERSE_SWEEP").
    run_by : str
        Identity of the agent or user that ran the experiment.
    started_at : str
        ISO 8601 timestamp.
    completed_at : str
        ISO 8601 timestamp.
    parameters : dict
        Experiment configuration parameters.
    results : dict
        Full results dict. Structure is experiment-type-specific.
    best_configuration : dict
        The best-performing parameter configuration found.
    comparison_baseline : Optional[dict]
        Baseline configuration for comparison, if applicable.
    improvement_over_baseline : Optional[dict]
        Metric improvements over baseline (e.g. {"sharpe": 0.15, "max_dd": -0.03}).
    recommendations : tuple[str, ...]
        Actionable recommendations derived from experiment results.
    artifact_ids : tuple[str, ...]
        IDs of supporting artifacts (e.g. walk-forward fold results).
    workflow_run_id : Optional[str]
        Workflow run that executed this experiment, if applicable.
    notes : str
        Optional free-text notes.
    """

    summary_id: str
    experiment_name: str
    experiment_type: str
    run_by: str
    started_at: str
    completed_at: str
    parameters: Dict
    results: Dict
    best_configuration: Dict
    comparison_baseline: Optional[Dict]
    improvement_over_baseline: Optional[Dict]
    recommendations: Tuple[str, ...]
    artifact_ids: Tuple[str, ...]
    workflow_run_id: Optional[str]
    notes: str = ""
