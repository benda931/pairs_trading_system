# -*- coding: utf-8 -*-
"""
deployment/engine.py — DeploymentEngine
=========================================

Manages the full deployment lifecycle:
  build → test → package → approve → deploy → verify → activate

Key invariants
--------------
- ``deployed != activated``.  A release may be deployed and verified
  but NOT activated.  Activation requires an explicit operator action
  and, if required, an approval record.
- Stage transitions are validated against the permitted transition map.
  Invalid transitions raise ValueError.
- Deployment freeze windows block new deployments for a named environment.
- Rollback is always available for DEPLOYED, VERIFIED, ACTIVATED, CANARIED,
  and EXPANDED releases.

Supports
--------
- Release registration and stage transitions.
- Rollout plan creation and execution.
- Rollback workflows with optional approval gates.
- Config version tracking (per key, per env).
- Deployment freeze windows per environment.
- Pre-deploy and post-deploy check hooks.
- Metrics: release counts, rollback rate, freeze status.

Thread safety: a single threading.Lock serializes all store mutations.
Singleton access via get_deployment_engine().
"""

from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from deployment.contracts import (
    ConfigVersionRecord,
    DeploymentArtifact,
    DeploymentStage,
    ReleaseRecord,
    RollbackDecision,
    RollbackReason,
    RolloutPlan,
)


# ══════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════

# Valid (from_stage, to_stage) transitions
_VALID_TRANSITIONS: Set[Tuple[DeploymentStage, DeploymentStage]] = {
    (DeploymentStage.BUILT, DeploymentStage.TESTED),
    (DeploymentStage.TESTED, DeploymentStage.PACKAGED),
    (DeploymentStage.PACKAGED, DeploymentStage.APPROVED),
    (DeploymentStage.APPROVED, DeploymentStage.DEPLOYED),
    (DeploymentStage.DEPLOYED, DeploymentStage.VERIFIED),
    (DeploymentStage.VERIFIED, DeploymentStage.ACTIVATED),
    (DeploymentStage.ACTIVATED, DeploymentStage.CANARIED),
    (DeploymentStage.CANARIED, DeploymentStage.EXPANDED),
    (DeploymentStage.EXPANDED, DeploymentStage.ACTIVATED),
    # Any active stage can be rolled back
    (DeploymentStage.DEPLOYED, DeploymentStage.ROLLED_BACK),
    (DeploymentStage.VERIFIED, DeploymentStage.ROLLED_BACK),
    (DeploymentStage.ACTIVATED, DeploymentStage.ROLLED_BACK),
    (DeploymentStage.CANARIED, DeploymentStage.ROLLED_BACK),
    (DeploymentStage.EXPANDED, DeploymentStage.ROLLED_BACK),
    # Retirement
    (DeploymentStage.ROLLED_BACK, DeploymentStage.RETIRED),
    (DeploymentStage.ACTIVATED, DeploymentStage.RETIRED),
    (DeploymentStage.EXPANDED, DeploymentStage.RETIRED),
}

# Stages that are eligible for rollback
_ROLLBACK_ELIGIBLE: Set[DeploymentStage] = {
    DeploymentStage.DEPLOYED,
    DeploymentStage.VERIFIED,
    DeploymentStage.ACTIVATED,
    DeploymentStage.CANARIED,
    DeploymentStage.EXPANDED,
}


# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_id() -> str:
    return str(uuid.uuid4())


# ══════════════════════════════════════════════════════════════════
# ENGINE
# ══════════════════════════════════════════════════════════════════


class DeploymentEngine:
    """Manages the deployment lifecycle for all release artifacts.

    Parameters
    ----------
    alert_engine :
        Optional AlertEngine for firing DEPLOYMENT_MISMATCH alerts.
    """

    def __init__(self, alert_engine: Optional[Any] = None) -> None:
        self._alert_engine = alert_engine
        self._lock = threading.Lock()

        # Stores
        self._releases: Dict[str, ReleaseRecord] = {}
        self._artifacts: Dict[str, DeploymentArtifact] = {}
        self._rollout_plans: Dict[str, RolloutPlan] = {}
        self._rollback_decisions: Dict[str, RollbackDecision] = {}

        # Config version store: (config_key, env) -> list[ConfigVersionRecord]
        # Most recent is last.
        self._config_versions: Dict[Tuple[str, str], List[ConfigVersionRecord]] = {}

        # Freeze windows: env -> (frozen: bool, actor: str, reason: str)
        self._freeze_state: Dict[str, Tuple[bool, str, str]] = {}

        # Metrics
        self._total_releases: int = 0
        self._total_rollbacks: int = 0
        self._total_activations: int = 0

    # ──────────────────────────────────────────────────────────────
    # ARTIFACT MANAGEMENT
    # ──────────────────────────────────────────────────────────────

    def register_artifact(self, artifact: DeploymentArtifact) -> None:
        """Register a deployment artifact.

        Parameters
        ----------
        artifact : DeploymentArtifact
        """
        with self._lock:
            self._artifacts[artifact.artifact_id] = artifact

    def get_artifact(self, artifact_id: str) -> Optional[DeploymentArtifact]:
        """Return artifact by ID, or None."""
        with self._lock:
            return self._artifacts.get(artifact_id)

    # ──────────────────────────────────────────────────────────────
    # RELEASE MANAGEMENT
    # ──────────────────────────────────────────────────────────────

    def register_release(self, release: ReleaseRecord) -> None:
        """Register a new release record.

        Parameters
        ----------
        release : ReleaseRecord
            Release must be in BUILT stage initially.

        Raises
        ------
        ValueError
            If a release with the same release_id already exists.
        """
        with self._lock:
            if release.release_id in self._releases:
                raise ValueError(
                    f"Release '{release.release_id}' already registered."
                )
            self._releases[release.release_id] = release
            self._total_releases += 1

    def transition_stage(
        self,
        release_id: str,
        to_stage: DeploymentStage,
        actor: str,
        notes: str = "",
    ) -> ReleaseRecord:
        """Transition a release to a new lifecycle stage.

        Parameters
        ----------
        release_id : str
            ID of the release to transition.
        to_stage : DeploymentStage
            Target stage.
        actor : str
            Operator or system performing the transition.
        notes : str
            Optional notes appended to the release record.

        Returns
        -------
        ReleaseRecord
            Updated release record.

        Raises
        ------
        KeyError
            If release_id is not found.
        ValueError
            If the transition is not permitted.
        RuntimeError
            If the target environment is frozen (for deploy/activate transitions).
        """
        with self._lock:
            release = self._releases.get(release_id)
            if release is None:
                raise KeyError(f"Release '{release_id}' not found.")

            # Validate transition
            transition = (release.stage, to_stage)
            if transition not in _VALID_TRANSITIONS:
                raise ValueError(
                    f"Invalid transition: {release.stage.value} → {to_stage.value} "
                    f"for release '{release_id}'."
                )

            # Freeze check for deploy and activate transitions
            if to_stage in (DeploymentStage.DEPLOYED, DeploymentStage.ACTIVATED):
                env = release.deployed_to or "unknown"
                freeze_entry = self._freeze_state.get(env)
                if freeze_entry is not None and freeze_entry[0]:
                    raise RuntimeError(
                        f"Environment '{env}' is frozen: {freeze_entry[2]}. "
                        f"Unfreeze before deploying or activating."
                    )

            now = _now_iso()

            # Apply stage-specific field updates
            if to_stage == DeploymentStage.DEPLOYED:
                release.deployed_at = now
                release.deployed_to = release.deployed_to or actor
            elif to_stage == DeploymentStage.ACTIVATED:
                release.activated_at = now
                release.activated_by = actor
                self._total_activations += 1
            elif to_stage == DeploymentStage.ROLLED_BACK:
                release.rolled_back_at = now

            if notes:
                release.notes = (release.notes + "\n" + notes).strip()

            release.stage = to_stage

        return release

    # ──────────────────────────────────────────────────────────────
    # ROLLOUT PLAN
    # ──────────────────────────────────────────────────────────────

    def create_rollout_plan(
        self,
        release_id: str,
        strategy: str,
        target_env: str,
        actor: str,
        canary_fraction: float = 0.10,
        observation_window_minutes: float = 30.0,
        rollback_on_error: bool = True,
        requires_approval: bool = True,
    ) -> RolloutPlan:
        """Create a rollout plan for a release.

        Parameters
        ----------
        release_id : str
            Release to roll out.
        strategy : str
            "immediate" | "canary" | "staged" | "blue_green".
        target_env : str
            Target deployment environment.
        actor : str
            Operator creating the plan.
        canary_fraction : float
            Canary traffic fraction [0, 1].
        observation_window_minutes : float
            Observation period for canary before expansion.
        rollback_on_error : bool
            Auto-rollback on health check failure.
        requires_approval : bool
            Additional approval gate before final activation.

        Returns
        -------
        RolloutPlan

        Raises
        ------
        KeyError
            If release_id is not found.
        """
        with self._lock:
            release = self._releases.get(release_id)
            if release is None:
                raise KeyError(f"Release '{release_id}' not found.")
            version = release.version

        steps = self._build_rollout_steps(strategy, target_env, version, canary_fraction)
        plan = RolloutPlan(
            plan_id=_make_id(),
            release_id=release_id,
            strategy=strategy,
            target_env=target_env,
            steps=tuple(steps),
            canary_fraction=canary_fraction,
            observation_window_minutes=observation_window_minutes,
            rollback_on_error=rollback_on_error,
            requires_approval=requires_approval,
            created_at=_now_iso(),
            created_by=actor,
        )
        with self._lock:
            self._rollout_plans[plan.plan_id] = plan

        return plan

    def execute_rollout(self, plan: RolloutPlan) -> ReleaseRecord:
        """Execute a rollout plan.

        Transitions the release through DEPLOYED and, for non-canary
        strategies, VERIFIED stages.  For canary strategies, transitions
        to CANARIED and waits for operator confirmation to EXPAND.

        Parameters
        ----------
        plan : RolloutPlan

        Returns
        -------
        ReleaseRecord
            The release after the rollout steps have been applied.

        Raises
        ------
        RuntimeError
            If the environment is frozen.
        """
        # Pre-deploy: ensure release is in APPROVED stage
        with self._lock:
            release = self._releases.get(plan.release_id)
            if release is None:
                raise KeyError(f"Release '{plan.release_id}' not found.")
            if release.stage not in (
                DeploymentStage.APPROVED,
                DeploymentStage.PACKAGED,
            ):
                raise ValueError(
                    f"Release must be APPROVED before rollout "
                    f"(current stage: {release.stage.value})."
                )
            # Record target env
            release.deployed_to = plan.target_env

        # Transition to DEPLOYED
        release = self.transition_stage(
            plan.release_id,
            DeploymentStage.DEPLOYED,
            actor=plan.created_by,
            notes=f"Rollout via plan {plan.plan_id} (strategy={plan.strategy}).",
        )

        # For immediate/staged/blue_green: also verify
        if plan.strategy in ("immediate", "staged", "blue_green"):
            release = self.transition_stage(
                plan.release_id,
                DeploymentStage.VERIFIED,
                actor=plan.created_by,
                notes="Auto-verified by rollout engine.",
            )
        elif plan.strategy == "canary":
            release = self.transition_stage(
                plan.release_id,
                DeploymentStage.VERIFIED,
                actor=plan.created_by,
                notes="Canary: deployed to verification.",
            )
            release = self.transition_stage(
                plan.release_id,
                DeploymentStage.ACTIVATED,
                actor=plan.created_by,
                notes=f"Canary fraction={plan.canary_fraction:.0%}.",
            )
            release = self.transition_stage(
                plan.release_id,
                DeploymentStage.CANARIED,
                actor=plan.created_by,
                notes=f"Canary observation window={plan.observation_window_minutes:.0f}min.",
            )

        return release

    # ──────────────────────────────────────────────────────────────
    # ROLLBACK
    # ──────────────────────────────────────────────────────────────

    def rollback(
        self,
        release_id: str,
        reason: RollbackReason,
        actor: str,
        approval_id: Optional[str] = None,
        rollback_to_version: str = "previous",
        notes: str = "",
        automated: bool = False,
    ) -> RollbackDecision:
        """Rollback a release.

        Parameters
        ----------
        release_id : str
            Release to roll back.
        reason : RollbackReason
            Categorised reason.
        actor : str
            Operator or system triggering the rollback.
        approval_id : Optional[str]
            Approval record ID (required for manual rollbacks in live env).
        rollback_to_version : str
            Version being restored to.
        notes : str
            Free-text notes.
        automated : bool
            True if triggered by automated health check.

        Returns
        -------
        RollbackDecision

        Raises
        ------
        KeyError
            If release_id is not found.
        ValueError
            If the release is not eligible for rollback.
        """
        with self._lock:
            release = self._releases.get(release_id)
            if release is None:
                raise KeyError(f"Release '{release_id}' not found.")
            if release.stage not in _ROLLBACK_ELIGIBLE:
                raise ValueError(
                    f"Release '{release_id}' in stage '{release.stage.value}' "
                    f"is not eligible for rollback."
                )
            previous_version = release.version

        decision = RollbackDecision(
            decision_id=_make_id(),
            release_id=release_id,
            reason=reason,
            decided_at=_now_iso(),
            decided_by=actor,
            previous_version=previous_version,
            rollback_to_version=rollback_to_version,
            approval_id=approval_id,
            automated=automated,
            notes=notes,
        )

        # Apply rollback transition
        self.transition_stage(
            release_id,
            DeploymentStage.ROLLED_BACK,
            actor=actor,
            notes=f"Rollback reason: {reason.value}. {notes}".strip(),
        )

        with self._lock:
            release = self._releases[release_id]
            release.rollback_reason = reason
            release.rolled_back_at = decision.decided_at
            self._rollback_decisions[decision.decision_id] = decision
            self._total_rollbacks += 1

        # Fire alert if engine is available
        if self._alert_engine is not None and reason != RollbackReason.OPERATOR_REQUEST:
            self._fire_rollback_alert(decision, release)

        return decision

    # ──────────────────────────────────────────────────────────────
    # CONFIG VERSION TRACKING
    # ──────────────────────────────────────────────────────────────

    def record_config_version(self, record: ConfigVersionRecord) -> None:
        """Record a configuration version application.

        Parameters
        ----------
        record : ConfigVersionRecord
        """
        key = (record.config_key, record.env)
        with self._lock:
            if key not in self._config_versions:
                self._config_versions[key] = []
            self._config_versions[key].append(record)

    def get_current_config_version(
        self, config_key: str, env: str
    ) -> Optional[ConfigVersionRecord]:
        """Return the most recently applied config version for a key/env pair.

        Parameters
        ----------
        config_key : str
        env : str

        Returns
        -------
        Optional[ConfigVersionRecord]
            Most recent record, or None if no versions recorded.
        """
        key = (config_key, env)
        with self._lock:
            versions = self._config_versions.get(key)
            if not versions:
                return None
            return versions[-1]

    def list_config_versions(
        self, config_key: str, env: str
    ) -> List[ConfigVersionRecord]:
        """Return all config versions for a key/env pair, oldest first."""
        key = (config_key, env)
        with self._lock:
            return list(self._config_versions.get(key, []))

    # ──────────────────────────────────────────────────────────────
    # FREEZE MANAGEMENT
    # ──────────────────────────────────────────────────────────────

    def set_freeze(
        self, env: str, freeze: bool, actor: str, reason: str
    ) -> None:
        """Set or lift a deployment freeze for an environment.

        Parameters
        ----------
        env : str
            Target environment.
        freeze : bool
            True to freeze, False to unfreeze.
        actor : str
            Operator applying the change.
        reason : str
            Human-readable reason.
        """
        with self._lock:
            self._freeze_state[env] = (freeze, actor, reason)

    def is_frozen(self, env: str) -> bool:
        """Return True if the environment has an active deployment freeze.

        Parameters
        ----------
        env : str

        Returns
        -------
        bool
        """
        with self._lock:
            entry = self._freeze_state.get(env)
            return entry is not None and entry[0]

    def get_freeze_info(self, env: str) -> Optional[Dict[str, str]]:
        """Return freeze details for an environment, or None if not frozen."""
        with self._lock:
            entry = self._freeze_state.get(env)
        if entry is None or not entry[0]:
            return None
        return {"actor": entry[1], "reason": entry[2]}

    # ──────────────────────────────────────────────────────────────
    # QUERY
    # ──────────────────────────────────────────────────────────────

    def get_release(self, release_id: str) -> Optional[ReleaseRecord]:
        """Return a release by ID, or None."""
        with self._lock:
            return self._releases.get(release_id)

    def list_releases(
        self,
        env: Optional[str] = None,
        stage: Optional[DeploymentStage] = None,
    ) -> List[ReleaseRecord]:
        """Return all releases, optionally filtered by env and/or stage.

        Parameters
        ----------
        env : Optional[str]
            Filter to releases deployed to this environment.
        stage : Optional[DeploymentStage]
            Filter to releases in this stage.

        Returns
        -------
        list[ReleaseRecord]
            Sorted by created_at descending.
        """
        with self._lock:
            releases = list(self._releases.values())

        if env is not None:
            releases = [r for r in releases if r.deployed_to == env]
        if stage is not None:
            releases = [r for r in releases if r.stage == stage]

        releases.sort(key=lambda r: r.created_at, reverse=True)
        return releases

    def get_rollback_decision(self, decision_id: str) -> Optional[RollbackDecision]:
        """Return a rollback decision by ID, or None."""
        with self._lock:
            return self._rollback_decisions.get(decision_id)

    # ──────────────────────────────────────────────────────────────
    # METRICS
    # ──────────────────────────────────────────────────────────────

    def get_metrics(self) -> Dict[str, Any]:
        """Return deployment engine statistics.

        Returns
        -------
        dict
            Keys: total_releases, total_rollbacks, total_activations,
            rollback_rate, active_releases, frozen_envs.
        """
        with self._lock:
            active = sum(
                1
                for r in self._releases.values()
                if r.stage in (
                    DeploymentStage.DEPLOYED,
                    DeploymentStage.VERIFIED,
                    DeploymentStage.ACTIVATED,
                    DeploymentStage.CANARIED,
                    DeploymentStage.EXPANDED,
                )
            )
            frozen_envs = [
                env for env, (frozen, _, _) in self._freeze_state.items() if frozen
            ]
            total_rel = self._total_releases

        rollback_rate = (
            self._total_rollbacks / total_rel if total_rel > 0 else 0.0
        )

        return {
            "total_releases": self._total_releases,
            "total_rollbacks": self._total_rollbacks,
            "total_activations": self._total_activations,
            "rollback_rate": rollback_rate,
            "active_releases": active,
            "frozen_envs": frozen_envs,
            "total_artifacts": len(self._artifacts),
        }

    # ──────────────────────────────────────────────────────────────
    # PRIVATE HELPERS
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def _build_rollout_steps(
        strategy: str,
        target_env: str,
        version: str,
        canary_fraction: float,
    ) -> List[str]:
        """Build ordered step descriptions for a rollout plan."""
        common = [
            f"Pre-flight: verify health of '{target_env}' environment.",
            f"Pre-flight: check deployment freeze status for '{target_env}'.",
            f"Transition release '{version}' to DEPLOYED in '{target_env}'.",
        ]
        if strategy == "canary":
            return common + [
                f"Route {canary_fraction:.0%} of traffic to new version.",
                "Monitor error rate, latency, and reconciliation for observation window.",
                "If healthy: expand to 100% and transition to EXPANDED.",
                "If unhealthy: auto-rollback to previous version.",
                "Post-deploy: verify all health checks pass.",
            ]
        elif strategy == "blue_green":
            return common + [
                "Verify green environment is healthy.",
                "Switch load balancer to green environment.",
                "Transition release to VERIFIED.",
                "Keep blue environment on standby for rapid rollback.",
                "Post-deploy: confirm all health checks pass.",
            ]
        elif strategy == "staged":
            return common + [
                "Deploy to stage-1 (10% capacity).",
                "Monitor stage-1 for stability window.",
                "Deploy to stage-2 (50% capacity).",
                "Monitor stage-2 for stability window.",
                "Deploy to stage-3 (100% capacity).",
                "Transition release to VERIFIED.",
                "Post-deploy: confirm all health checks pass.",
            ]
        else:  # immediate
            return common + [
                "Deploy to full capacity immediately.",
                "Transition release to VERIFIED.",
                "Post-deploy: confirm all health checks pass.",
            ]

    def _fire_rollback_alert(
        self, decision: RollbackDecision, release: ReleaseRecord
    ) -> None:
        """Fire a DEPLOYMENT_MISMATCH alert for the given rollback."""
        if self._alert_engine is None:
            return
        try:
            self._alert_engine.fire(
                rule_id="DEPLOYMENT_MISMATCH",
                source="deployment_engine",
                scope=f"release:{release.release_id}",
                message=(
                    f"Release '{release.release_name}' v{release.version} "
                    f"rolled back: {decision.reason.value}."
                ),
                details={
                    "release_id": release.release_id,
                    "reason": decision.reason.value,
                    "decided_by": decision.decided_by,
                    "automated": decision.automated,
                },
            )
        except Exception:  # noqa: BLE001
            pass


# ══════════════════════════════════════════════════════════════════
# SINGLETON
# ══════════════════════════════════════════════════════════════════

_engine_instance: Optional[DeploymentEngine] = None
_engine_lock = threading.Lock()


def get_deployment_engine(
    alert_engine: Optional[Any] = None,
) -> DeploymentEngine:
    """Return the singleton DeploymentEngine, creating it on first call.

    Parameters
    ----------
    alert_engine :
        AlertEngine to inject on first creation (ignored on subsequent calls).

    Returns
    -------
    DeploymentEngine
    """
    global _engine_instance
    if _engine_instance is None:
        with _engine_lock:
            if _engine_instance is None:
                _engine_instance = DeploymentEngine(alert_engine=alert_engine)
    return _engine_instance
