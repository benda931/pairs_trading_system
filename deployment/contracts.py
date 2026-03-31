# -*- coding: utf-8 -*-
"""
deployment/contracts.py — Deployment Domain Contracts
======================================================

All typed domain objects for the deployment lifecycle subsystem.

Design principles:
  - DeploymentArtifact, RolloutPlan, RollbackDecision, and
    ConfigVersionRecord are frozen (immutable after construction).
  - ReleaseRecord is mutable so the engine can update stage,
    timestamps, and approval references as the release progresses.
  - Enums inherit from str for JSON round-trip compatibility.
  - stdlib only — no external dependencies.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ══════════════════════════════════════════════════════════════════
# 1. ENUMERATIONS
# ══════════════════════════════════════════════════════════════════


class DeploymentStage(str, enum.Enum):
    """Stage in the deployment lifecycle.

    The canonical progression is:
    BUILT → TESTED → PACKAGED → APPROVED → DEPLOYED → VERIFIED
    → ACTIVATED → (CANARIED → EXPANDED) | ROLLED_BACK → RETIRED

    ``deployed != activated`` is a hard invariant: a release can sit
    in DEPLOYED / VERIFIED state indefinitely without being ACTIVATED.
    Activation requires an explicit operator action + approval.
    """

    BUILT = "built"
    TESTED = "tested"
    PACKAGED = "packaged"
    APPROVED = "approved"
    DEPLOYED = "deployed"
    VERIFIED = "verified"
    ACTIVATED = "activated"
    CANARIED = "canaried"
    EXPANDED = "expanded"
    ROLLED_BACK = "rolled_back"
    RETIRED = "retired"


class RollbackReason(str, enum.Enum):
    """Reason category for a rollback decision."""

    HEALTH_CHECK_FAILED = "health_check_failed"
    PERFORMANCE_REGRESSION = "performance_regression"
    RECONCILIATION_FAILURE = "reconciliation_failure"
    OPERATOR_REQUEST = "operator_request"
    POLICY_VIOLATION = "policy_violation"
    INCIDENT_TRIGGERED = "incident_triggered"


# ══════════════════════════════════════════════════════════════════
# 2. ARTIFACTS AND RELEASES
# ══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class DeploymentArtifact:
    """An immutable deployment artifact (code, model, config, policy, schema).

    Fields
    ------
    artifact_id : str
        UUID for this artifact.
    artifact_type : str
        Category: "code" | "model" | "config" | "policy" | "schema".
    name : str
        Human-readable artifact name.
    version : str
        Semantic version string.
    built_at : str
        ISO-8601 build timestamp.
    built_by : str
        CI system or operator that produced this artifact.
    git_sha : Optional[str]
        Git commit SHA if applicable.
    checksum : str
        SHA-256 hex digest for integrity verification.
    dependencies : dict
        Dependency versions: {name: version_string}.
    target_envs : tuple[str, ...]
        Environments this artifact is authorised for.
    compatible_schema_versions : tuple[str, ...]
        Database schema versions this artifact is compatible with.
    notes : str
        Free-text notes.
    """

    artifact_id: str
    artifact_type: str
    name: str
    version: str
    built_at: str
    built_by: str
    git_sha: Optional[str]
    checksum: str
    dependencies: Dict[str, str]
    target_envs: Tuple[str, ...]
    compatible_schema_versions: Tuple[str, ...]
    notes: str = ""


@dataclass
class ReleaseRecord:
    """Mutable record tracking a release through its deployment lifecycle.

    Mutation is intentional — the engine updates stage, timestamps, and
    approval references in-place. All mutations are protected by the
    DeploymentEngine's lock.

    Fields
    ------
    release_id : str
        UUID for this release.
    release_name : str
        Human-readable release name (e.g. "v2.4.1-hotfix").
    version : str
        Semantic version string.
    stage : DeploymentStage
        Current lifecycle stage.
    artifacts : list[str]
        Artifact IDs included in this release.
    created_at : str
        ISO-8601 creation timestamp.
    created_by : str
        Operator or CI system that created the release.
    deployed_at : Optional[str]
        ISO-8601 timestamp of deployment.
    deployed_to : Optional[str]
        Target environment identifier.
    activated_at : Optional[str]
        ISO-8601 timestamp of activation.
    activated_by : Optional[str]
        Operator who activated the release.
    rolled_back_at : Optional[str]
        ISO-8601 timestamp of rollback.
    rollback_reason : Optional[RollbackReason]
        Reason for rollback.
    approval_ids : list[str]
        Approval record identifiers authorising this release.
    preflight_report_id : Optional[str]
        ID of the pre-deploy check report.
    notes : str
        Free-text notes.
    """

    release_id: str
    release_name: str
    version: str
    stage: DeploymentStage
    artifacts: List[str]
    created_at: str
    created_by: str
    deployed_at: Optional[str]
    deployed_to: Optional[str]
    activated_at: Optional[str]
    activated_by: Optional[str]
    rolled_back_at: Optional[str]
    rollback_reason: Optional[RollbackReason]
    approval_ids: List[str]
    preflight_report_id: Optional[str]
    notes: str = ""


# ══════════════════════════════════════════════════════════════════
# 3. ROLLOUT AND ROLLBACK PLANS
# ══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class RolloutPlan:
    """Immutable rollout execution plan for a release.

    Fields
    ------
    plan_id : str
        UUID for this plan.
    release_id : str
        Release this plan applies to.
    strategy : str
        Rollout strategy: "immediate" | "canary" | "staged" | "blue_green".
    target_env : str
        Deployment target environment.
    steps : tuple[str, ...]
        Ordered step descriptions.
    canary_fraction : float
        Fraction of traffic/instances for canary (0.0–1.0).
    observation_window_minutes : float
        How long to observe a canary before expanding.
    rollback_on_error : bool
        Whether to auto-rollback on health check failure.
    requires_approval : bool
        Whether final activation requires an additional approval gate.
    created_at : str
        ISO-8601 creation timestamp.
    created_by : str
        Operator who created the plan.
    """

    plan_id: str
    release_id: str
    strategy: str
    target_env: str
    steps: Tuple[str, ...]
    canary_fraction: float
    observation_window_minutes: float
    rollback_on_error: bool
    requires_approval: bool
    created_at: str
    created_by: str


@dataclass(frozen=True)
class RollbackDecision:
    """Immutable record of a rollback decision.

    Fields
    ------
    decision_id : str
        UUID for this decision record.
    release_id : str
        Release being rolled back.
    reason : RollbackReason
        Categorised reason for the rollback.
    decided_at : str
        ISO-8601 timestamp.
    decided_by : str
        Operator or automated system that triggered the rollback.
    previous_version : str
        Version being rolled back from.
    rollback_to_version : str
        Version being restored.
    approval_id : Optional[str]
        Approval record for manual rollbacks, if required.
    automated : bool
        True if triggered by an automated health check.
    notes : str
        Free-text notes.
    """

    decision_id: str
    release_id: str
    reason: RollbackReason
    decided_at: str
    decided_by: str
    previous_version: str
    rollback_to_version: str
    approval_id: Optional[str]
    automated: bool
    notes: str = ""


# ══════════════════════════════════════════════════════════════════
# 4. CONFIGURATION VERSION TRACKING
# ══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ConfigVersionRecord:
    """Immutable record of a configuration version application.

    Fields
    ------
    record_id : str
        UUID for this record.
    config_key : str
        Configuration domain key (e.g. "strategy_params", "risk_limits").
    version : str
        Version string for this configuration snapshot.
    value_hash : str
        SHA-256 hex digest of the configuration value.
    applied_at : str
        ISO-8601 timestamp when this version was applied.
    applied_by : str
        Operator or deployment system that applied the config.
    env : str
        Deployment environment.
    approved : bool
        Whether this config version has been approved.
    approval_id : Optional[str]
        Approval record identifier.
    notes : str
        Free-text notes.
    """

    record_id: str
    config_key: str
    version: str
    value_hash: str
    applied_at: str
    applied_by: str
    env: str
    approved: bool
    approval_id: Optional[str]
    notes: str = ""
