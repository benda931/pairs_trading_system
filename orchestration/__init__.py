# -*- coding: utf-8 -*-
"""
orchestration — Workflow orchestration layer for the pairs trading system.

Exposes all public types from ``orchestration.contracts`` and the
``WorkflowEngine`` plus pre-built workflow factories from
``orchestration.engine``.

Usage
-----
    from orchestration import (
        WorkflowEngine,
        WorkflowDefinition,
        WorkflowRun,
        WorkflowStatus,
        EnvironmentClass,
        build_research_discovery_workflow,
    )

    engine = WorkflowEngine(registry=get_default_registry())
    defn   = build_research_discovery_workflow()
    run    = engine.run(defn, trigger_payload={"symbols": [...]})
"""

from orchestration.contracts import (
    ActionBoundary,
    AgentCapability,
    AgentContextPackage,
    AgentEvidenceBundle,
    AgentHealthStatus,
    AgentMemoryRecord,
    AgentPermissionSet,
    AgentRecommendation,
    AgentRunRecord,
    AgentSpec,
    AgentWarning,
    ContextReference,
    DelegationDepth,
    DelegationRecord,
    EnvironmentClass,
    FailureClass,
    FailureRecord,
    RiskClass,
    WorkflowDefinition,
    WorkflowOutcome,
    WorkflowRun,
    WorkflowStatus,
    WorkflowStep,
    WorkflowStepRun,
    WorkflowStepStatus,
    WorkflowTransition,
)
from orchestration.engine import (
    WorkflowEngine,
    build_drift_alert_workflow,
    build_model_promotion_workflow,
    build_research_discovery_workflow,
)

__all__ = [
    # ── Enums ──────────────────────────────────────────────────────
    "ActionBoundary",
    "DelegationDepth",
    "EnvironmentClass",
    "FailureClass",
    "RiskClass",
    "WorkflowStatus",
    "WorkflowStepStatus",
    # ── Agent architecture types ───────────────────────────────────
    "AgentCapability",
    "AgentPermissionSet",
    "AgentSpec",
    "AgentRunRecord",
    "AgentHealthStatus",
    # ── Context and memory ─────────────────────────────────────────
    "ContextReference",
    "AgentContextPackage",
    "AgentMemoryRecord",
    # ── Evidence and recommendations ───────────────────────────────
    "AgentWarning",
    "AgentEvidenceBundle",
    "AgentRecommendation",
    # ── Workflow objects ────────────────────────────────────────────
    "WorkflowDefinition",
    "WorkflowStep",
    "WorkflowTransition",
    "WorkflowRun",
    "WorkflowStepRun",
    "WorkflowOutcome",
    # ── Delegation and failure records ─────────────────────────────
    "DelegationRecord",
    "FailureRecord",
    # ── Engine ─────────────────────────────────────────────────────
    "WorkflowEngine",
    # ── Pre-built workflow factories ───────────────────────────────
    "build_research_discovery_workflow",
    "build_model_promotion_workflow",
    "build_drift_alert_workflow",
]
