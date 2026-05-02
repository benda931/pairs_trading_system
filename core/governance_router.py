# -*- coding: utf-8 -*-
"""
core/governance_router.py — ML Governance Entry Point
======================================================

GovernanceRouter is the ONLY sanctioned path from production decision code
to ML inference. All calls to ML models MUST go through this router — never
directly to ModelScorer, never through pickle files.

Enforces at every call:
  1. Model status check (only CHAMPION may serve production inference)
  2. Blend cap (max 30% influence on any composite score)
  3. Grade upgrade limit (at most 1 grade step per ML call)
  4. Audit trail with model_id in every output
  5. Fallback to rule-based scoring when no governed model is available

Usage
-----
    from core.governance_router import GovernanceRouter

    router = GovernanceRouter()

    # For signal grade overlay:
    result = router.score_signal(features, current_grade="D")
    if result.allowed:
        apply_grade_upgrade(result.upgraded_grade)

    # For portfolio ranking blend:
    result = router.score_opportunity(opportunity)
    if result.allowed:
        blend_into_composite(result.score, blend_fraction=result.effective_blend)
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger("core.governance_router")

# Grade ordering for upgrade-limit enforcement
_GRADE_ORDER = ["A+", "A", "B", "C", "D", "F"]
_GRADE_NUMERIC = {g: i for i, g in enumerate(_GRADE_ORDER)}


# ---------------------------------------------------------------------------
# Result objects
# ---------------------------------------------------------------------------

@dataclass
class GovernanceRouterResult:
    """
    Result of a governance-gated ML inference call.

    allowed=True  : ML score is valid, governed, and may influence decisions.
    allowed=False : ML inference blocked; callers must use rule-based path only.
    """
    allowed: bool
    score: float = float("nan")                  # ML score [0, 1], or NaN if not allowed
    model_id: str = ""                           # Model that produced the score (for audit)
    model_status: str = ""                       # Status of the model at inference time
    fallback_used: bool = False                  # True if ML unavailable, rule-based used
    fallback_reason: str = ""                    # Why ML was bypassed
    upgraded_grade: Optional[str] = None         # Grade after applying ML overlay (if applicable)
    effective_blend: float = 0.0                 # Blend fraction actually applied (capped)
    block_reason: str = ""                       # Reason consumer was blocked (empty if not blocked)
    warnings: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# GovernanceRouter
# ---------------------------------------------------------------------------

class GovernanceRouter:
    """
    Single sanctioned entry point for all production ML inference.

    This class is the governance enforcement layer between production decision
    code and the ML platform. It enforces all governance rules so that
    individual callers (signal_quality, portfolio/ranking, etc.) do not need
    to re-implement them.

    Parameters
    ----------
    registry : optional
        ML model registry. If None, loaded from ml.registry.get_ml_registry().
    max_blend_fraction : float
        Hard cap on ML blend into any composite score (default 0.30).
    max_grade_upgrade_steps : int
        Maximum number of grade levels ML may upgrade in one call (default 1).
    """

    MAX_BLEND_FRACTION: float = 0.30
    MAX_GRADE_UPGRADE_STEPS: int = 1

    def __init__(
        self,
        registry=None,
        max_blend_fraction: float = MAX_BLEND_FRACTION,
        max_grade_upgrade_steps: int = MAX_GRADE_UPGRADE_STEPS,
        consumer_allowlist: Optional[List[str]] = None,
        consumer_blocklist: Optional[List[str]] = None,
    ):
        self._registry = registry
        self._max_blend = min(max_blend_fraction, self.MAX_BLEND_FRACTION)
        self._max_grade_steps = max_grade_upgrade_steps
        self._scorer = None  # Lazy-loaded
        # Consumer allow/block lists.  Blocklist takes precedence over allowlist.
        # If allowlist is None (not set), all consumers are permitted (unless blocked).
        # If allowlist is an empty list, no consumers are permitted.
        self.consumer_allowlist: Optional[List[str]] = consumer_allowlist
        self.consumer_blocklist: List[str] = consumer_blocklist or []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def score_signal(
        self,
        features: Dict[str, float],
        current_grade: str,
        as_of: str = "",
        entity_id: str = "",
    ) -> GovernanceRouterResult:
        """
        Score a signal for meta-labeling / grade overlay.

        Enforces:
        - CHAMPION status required
        - Grade upgrade at most MAX_GRADE_UPGRADE_STEPS steps
        - Grade F is inviolable (never upgraded by ML)
        - model_id in result for audit trail

        Parameters
        ----------
        features : Dict[str, float]
            Signal feature vector.
        current_grade : str
            Current rule-based quality grade ("A+", "A", "B", "C", "D", "F").
        as_of : str
            ISO-8601 timestamp for lookahead-bias guard.
        entity_id : str
            Pair/instrument identifier (for logging and audit).

        Returns
        -------
        GovernanceRouterResult with upgraded_grade set if ML allowed upgrade.
        """
        result = self._get_governed_score(
            task_family_name="META_LABELING",
            features=features,
            as_of=as_of,
            entity_id=entity_id,
        )

        if not result.allowed:
            return result

        # Grade F is inviolable — ML cannot upgrade a grade-F signal
        if current_grade == "F":
            result.allowed = False
            result.fallback_used = True
            result.fallback_reason = "grade_F_inviolable"
            result.score = float("nan")
            return result

        # Compute upgrade
        current_idx = _GRADE_NUMERIC.get(current_grade)
        if current_idx is None:
            result.warnings.append(f"Unknown grade {current_grade!r} — no upgrade applied")
            result.upgraded_grade = current_grade
            return result

        # A score > 0.5 suggests upgrade; enforce max steps
        if result.score > 0.5 and math.isfinite(result.score):
            steps = 1  # Always upgrade by exactly 1 step max
            new_idx = max(0, current_idx - steps)  # Lower index = better grade
            result.upgraded_grade = _GRADE_ORDER[new_idx]
        else:
            result.upgraded_grade = current_grade

        return result

    def score_opportunity(
        self,
        opportunity,
        requested_blend: float = 0.20,
        as_of: str = "",
        entity_id: str = "",
    ) -> GovernanceRouterResult:
        """
        Score a ranked opportunity for portfolio ranking blend.

        Enforces:
        - CHAMPION status required
        - Blend fraction capped at MAX_BLEND_FRACTION
        - Neutral-score (0.5) fallbacks excluded from blending
        - model_id in result

        Parameters
        ----------
        opportunity : RankedOpportunity
            The opportunity object (passed to ml_hook.score()).
        requested_blend : float
            Desired blend fraction (will be capped).
        """
        from ml.contracts import MLTaskFamily
        result = self._get_governed_score(
            task_family_name="PORTFOLIO_RANKING",
            features={},   # Opportunity scoring uses the object directly
            as_of=as_of,
            entity_id=entity_id,
            opportunity=opportunity,
        )

        if not result.allowed:
            return result

        # Neutral fallback detection — do not blend 0.5 into composites
        if abs(result.score - 0.5) < 1e-9:
            result.allowed = False
            result.fallback_reason = "neutral_fallback_excluded"
            result.effective_blend = 0.0
            return result

        # Cap blend
        result.effective_blend = min(requested_blend, self._max_blend)
        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_consumer_permission(self, consumer_id: str) -> tuple:
        """
        Check whether a named consumer is permitted to receive ML inference.

        Blocklist takes precedence over allowlist.  If the allowlist is None
        (not configured), any consumer that is not explicitly blocked is
        considered permitted.

        Parameters
        ----------
        consumer_id : str
            Identifier for the calling consumer (e.g. task family name).

        Returns
        -------
        tuple[bool, str]
            (allowed, reason) — reason is empty when allowed is True.
        """
        if consumer_id in self.consumer_blocklist:
            return False, f"consumer '{consumer_id}' is explicitly blocked"
        if self.consumer_allowlist is not None and consumer_id not in self.consumer_allowlist:
            return False, f"consumer '{consumer_id}' not in allowlist"
        return True, "permitted"

    def _get_governed_score(
        self,
        task_family_name: str,
        features: Dict[str, float],
        as_of: str = "",
        entity_id: str = "",
        opportunity=None,
    ) -> GovernanceRouterResult:
        """Load scorer, enforce governance rules, return score."""
        try:
            # Router-level consumer allow/block check (evaluated before any ML I/O)
            consumer_id = task_family_name.lower()
            allowed, reason = self._check_consumer_permission(consumer_id)
            if not allowed:
                return GovernanceRouterResult(
                    allowed=False,
                    score=0.5,   # Neutral fallback
                    model_id="",
                    model_status=None,
                    upgraded_grade=None,
                    effective_blend=0.0,
                    block_reason=reason,
                )

            from ml.contracts import InferenceRequest, MLTaskFamily, ModelStatus
            scorer = self._get_scorer()
            if scorer is None:
                return GovernanceRouterResult(
                    allowed=False,
                    fallback_used=True,
                    fallback_reason="ml_registry_unavailable",
                )

            task_family = MLTaskFamily(task_family_name.lower())

            # Check that a CHAMPION model exists for this task
            model = scorer.get_model_for_task(task_family)
            if model is None:
                return GovernanceRouterResult(
                    allowed=False,
                    fallback_used=True,
                    fallback_reason=f"no_champion_for_{task_family_name}",
                )

            # Status check
            model_id = getattr(model, "model_id", "") or getattr(
                getattr(model, "metadata", None), "model_id", ""
            )
            metadata = getattr(model, "metadata", None)
            if metadata is not None:
                status = getattr(metadata, "status", None)
                if status != ModelStatus.CHAMPION:
                    return GovernanceRouterResult(
                        allowed=False,
                        model_id=model_id,
                        model_status=str(status),
                        fallback_used=True,
                        fallback_reason=f"model_status_{status}_not_CHAMPION",
                    )

            # Usage contract check: verify this consumer is allowed to use the model
            if metadata is not None:
                allowed_consumers = getattr(metadata, "allowed_consumers", [])
                forbidden_consumers = getattr(metadata, "forbidden_consumers", [])
                consumer = task_family_name.lower()

                if forbidden_consumers and consumer in forbidden_consumers:
                    return GovernanceRouterResult(
                        allowed=False,
                        model_id=model_id,
                        fallback_used=True,
                        fallback_reason=f"consumer_{consumer}_is_forbidden",
                    )

                if allowed_consumers and consumer not in allowed_consumers:
                    return GovernanceRouterResult(
                        allowed=False,
                        model_id=model_id,
                        fallback_used=True,
                        fallback_reason=f"consumer_{consumer}_not_in_allowed_list",
                    )

            # Build inference request
            if opportunity is not None:
                # For opportunity scoring, delegate to the scorer's opportunity interface
                try:
                    raw_score = float(model.score(opportunity))
                except Exception as exc:
                    return GovernanceRouterResult(
                        allowed=False,
                        model_id=model_id,
                        fallback_used=True,
                        fallback_reason=f"score_error: {exc}",
                    )
            else:
                req = InferenceRequest(
                    entity_id=entity_id,
                    task_family=task_family,
                    as_of=as_of,
                    features=features,
                    allow_fallback=False,  # GovernanceRouter never silently falls back
                    strict_feature_check=False,
                )
                inference_result = scorer.score(req)
                if inference_result.fallback_used:
                    return GovernanceRouterResult(
                        allowed=False,
                        model_id=model_id,
                        fallback_used=True,
                        fallback_reason=inference_result.fallback_reason,
                    )
                raw_score = inference_result.score

            if not math.isfinite(raw_score):
                return GovernanceRouterResult(
                    allowed=False,
                    model_id=model_id,
                    fallback_used=True,
                    fallback_reason="non_finite_score",
                )

            score = max(0.0, min(1.0, raw_score))
            model_status_str = str(getattr(getattr(model, "metadata", None), "status", "unknown"))

            return GovernanceRouterResult(
                allowed=True,
                score=score,
                model_id=model_id,
                model_status=model_status_str,
                fallback_used=False,
            )

        except Exception as exc:
            logger.warning("GovernanceRouter._get_governed_score failed: %s", exc)
            return GovernanceRouterResult(
                allowed=False,
                fallback_used=True,
                fallback_reason=f"router_error: {exc}",
            )

    def _get_scorer(self):
        """Lazy-load ModelScorer. Returns None if unavailable."""
        if self._scorer is not None:
            return self._scorer
        try:
            if self._registry is None:
                from ml.registry import get_ml_registry
                self._registry = get_ml_registry()
            from ml.inference.scorer import ModelScorer
            self._scorer = ModelScorer(registry=self._registry)
            return self._scorer
        except Exception as exc:
            logger.debug("GovernanceRouter: ModelScorer unavailable: %s", exc)
            return None


@dataclass
class GovernancePrecisionMetric:
    """Compatibility metric payload for agent feedback governance reporting."""

    action_type: str
    environment: Any
    current_tier: Any
    rolling_precision_30d: float
    rolling_precision_90d: float
    total_executed: int
    false_positive_rate: float
    tier_demoted_at: Optional[datetime] = None
    tier_demotion_reason: Optional[str] = None


class FeedbackGovernanceAdapter:
    """
    Lightweight action-governance adapter for AgentFeedbackLoop compatibility.

    This preserves the existing ML GovernanceRouter class above while exposing
    the `route(...)` / `get_precision_metrics()` API that feedback execution
    expects during dashboard and daemon runs.
    """

    def __init__(
        self,
        *,
        environment: Any = "research",
        executor_registry: Optional[Dict[str, Any]] = None,
        validate_on_init: bool = False,
    ) -> None:
        from core.action_governance import ActionGovernanceTier, TradingEnvironment

        self._ActionGovernanceTier = ActionGovernanceTier
        self._TradingEnvironment = TradingEnvironment
        self.environment = self._coerce_environment(environment)
        self.executor_registry = dict(executor_registry or {})
        self.validate_on_init = bool(validate_on_init)
        self._records: List[Any] = []

    def _coerce_environment(self, environment: Any) -> Any:
        if isinstance(environment, self._TradingEnvironment):
            return environment
        raw = str(getattr(environment, "value", environment) or "research").lower()
        try:
            return self._TradingEnvironment(raw)
        except Exception:
            return self._TradingEnvironment.RESEARCH

    def route(
        self,
        *,
        action_type: str,
        action_id: str,
        source_agent: str,
        target: str,
        parameters: Optional[Dict[str, Any]] = None,
        severity: str = "INFO",
        created_at: Optional[datetime] = None,
        confirming_agent: Optional[str] = None,
    ) -> Any:
        from core.action_governance import GovernedActionRecord

        params = dict(parameters or {})
        executor = self.executor_registry.get(action_type)
        created_dt = created_at or datetime.now(timezone.utc)
        executed = False
        execution_result = ""
        execution_error = None

        if callable(executor):
            try:
                try:
                    execution_result = str(
                        executor(
                            action_type=action_type,
                            action_id=action_id,
                            source_agent=source_agent,
                            target=target,
                            parameters=params,
                            severity=severity,
                            confirming_agent=confirming_agent,
                        )
                    )
                except TypeError:
                    execution_result = str(executor(target, params))
                executed = True
            except Exception as exc:
                execution_error = str(exc)

        tier = (
            self._ActionGovernanceTier.AUTO_EXECUTABLE
            if callable(executor)
            else self._ActionGovernanceTier.ADVISORY_ONLY
        )

        record = GovernedActionRecord(
            action_id=str(action_id),
            action_type=str(action_type),
            source_agent=str(source_agent),
            confirming_agent=confirming_agent,
            environment=self.environment,
            governance_tier=tier,
            tier_demoted=False,
            demotion_reason=None,
            evidence_snapshot=params,
            evidence_valid=True,
            evidence_validation_errors=(),
            evidence_age_seconds=-1,
            is_stale=False,
            suppressed=False,
            suppression_reason=None,
            conflict_with_action_id=None,
            approval_request_id=None,
            approval_status=None,
            override_record_id=None,
            executed=executed,
            execution_timestamp=datetime.now(timezone.utc) if executed else None,
            execution_result=execution_result,
            execution_error=execution_error,
            rollback_handle_id=None,
            rollback_deadline=None,
            incident_id=None,
            outcome_observed=False,
            outcome_observation_deadline=None,
            outcome_correct=None,
            regret_score=None,
            created_at=created_dt,
            expires_at=created_dt,
        )
        self._records.append(record)
        return record

    def get_precision_metrics(self) -> List[GovernancePrecisionMetric]:
        metrics: List[GovernancePrecisionMetric] = []
        by_action: Dict[str, List[Any]] = {}
        for record in self._records:
            by_action.setdefault(str(record.action_type), []).append(record)

        for action_type, records in sorted(by_action.items()):
            total_executed = sum(1 for record in records if getattr(record, "executed", False))
            metrics.append(
                GovernancePrecisionMetric(
                    action_type=action_type,
                    environment=self.environment,
                    current_tier=(
                        self._ActionGovernanceTier.AUTO_EXECUTABLE
                        if total_executed > 0
                        else self._ActionGovernanceTier.ADVISORY_ONLY
                    ),
                    rolling_precision_30d=1.0 if total_executed > 0 else 0.0,
                    rolling_precision_90d=1.0 if total_executed > 0 else 0.0,
                    total_executed=total_executed,
                    false_positive_rate=0.0,
                )
            )
        return metrics


_FEEDBACK_ROUTER_SINGLETONS: Dict[str, FeedbackGovernanceAdapter] = {}


def get_governance_router(
    *,
    environment: Any = "research",
    executor_registry: Optional[Dict[str, Any]] = None,
    validate_on_init: bool = False,
) -> FeedbackGovernanceAdapter:
    """Return a singleton feedback-governance adapter keyed by environment."""

    key = str(getattr(environment, "value", environment) or "research").lower()
    router = _FEEDBACK_ROUTER_SINGLETONS.get(key)
    if router is None:
        router = FeedbackGovernanceAdapter(
            environment=environment,
            executor_registry=executor_registry,
            validate_on_init=validate_on_init,
        )
        _FEEDBACK_ROUTER_SINGLETONS[key] = router
        return router

    if executor_registry:
        router.executor_registry.update(dict(executor_registry))
    return router
